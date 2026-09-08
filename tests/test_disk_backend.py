#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Disk backend contract, failure recovery and actual cross-process persistence."""

import asyncio
import hashlib
import json
import sys
import threading
from pathlib import Path

import pytest

import pipecat_tts_cache.backends.disk as disk_module
from pipecat_tts_cache import DiskCacheBackend, generate_cache_key
from pipecat_tts_cache.models import CachedAudioChunk, CachedTTSResponse, CachedWordTimestamp


def _response(audio=b"\x00\x01" * 16):
    return CachedTTSResponse(
        audio_chunks=[CachedAudioChunk(audio, 16000, 1, 123), CachedAudioChunk(b"", 16000, 1)],
        sample_rate=16000,
        num_channels=1,
        word_timestamps=[CachedWordTimestamp("你好", 0.0), CachedWordTimestamp("world", 0.2)],
        total_duration_s=0.4,
        created_at=1000.5,
        metadata={"text": "你好 world", "nested": {"values": [None, True, 3, 1.25]}},
    )


@pytest.fixture
def backend(tmp_path):
    return DiskCacheBackend(tmp_path / "cache")


async def test_full_response_round_trip_and_new_instance(backend):
    response = _response()
    assert await backend.get("absent") is None
    assert await backend.set("k", response) is True
    fetched = await backend.get("k")
    assert fetched == response
    assert fetched is not response
    assert await DiskCacheBackend(backend._cache_dir).get("k") == response


@pytest.mark.parametrize("timestamps", [None, []])
async def test_audio_only_response(backend, timestamps):
    response = _response()
    response.word_timestamps = timestamps
    assert await backend.set("k", response)
    assert await backend.get("k") == response


async def test_ttl_survives_new_instance_and_expires(backend, monkeypatch):
    monkeypatch.setattr(disk_module.time, "time", lambda: 1000.0)
    assert await backend.set("k", _response(), ttl=10)
    reopened = DiskCacheBackend(backend._cache_dir)
    monkeypatch.setattr(disk_module.time, "time", lambda: 1009.0)
    assert await reopened.exists("k")
    monkeypatch.setattr(disk_module.time, "time", lambda: 1011.0)
    assert await reopened.get("k") is None
    assert await reopened.exists("k") is False


@pytest.mark.parametrize("ttl", [None, 0, -10])
async def test_non_positive_ttl_never_expires(backend, monkeypatch, ttl):
    monkeypatch.setattr(disk_module.time, "time", lambda: 1000.0)
    assert await backend.set("k", _response(), ttl=ttl)
    monkeypatch.setattr(disk_module.time, "time", lambda: 1e12)
    assert await backend.get("k") == _response()


async def test_overwrite_resets_ttl(backend, monkeypatch):
    monkeypatch.setattr(disk_module.time, "time", lambda: 1000.0)
    assert await backend.set("k", _response(), ttl=1)
    replacement = _response(b"new")
    assert await backend.set("k", replacement)
    monkeypatch.setattr(disk_module.time, "time", lambda: 2000.0)
    assert await backend.get("k") == replacement


async def test_delete_and_clear_with_literal_namespace(backend):
    for key in ["a:k", "a:sub:k", "ab:k", "b:k", "[x]*:k", "x:k"]:
        assert await backend.set(key, _response())
    assert await backend.clear("a") == 2
    assert await backend.exists("ab:k")
    assert await backend.clear("[x]*") == 1
    assert await backend.exists("x:k")
    assert await backend.delete("b:k") is True
    assert await backend.delete("b:k") is False
    assert await backend.clear() == 2
    assert await backend.clear() == 0


async def test_close_and_context_manager_preserve_data(backend):
    async with backend:
        assert await backend.set("k", _response())
    await backend.close()
    assert await DiskCacheBackend(backend._cache_dir).get("k") == _response()


async def test_stats_and_exists_do_not_count_as_hits(backend):
    assert await backend.set("k", _response())
    assert await backend.exists("k")
    assert not await backend.exists("absent")
    await backend.get("k")
    await backend.get("absent")
    stats = await backend.get_stats()
    assert stats["type"] == "disk"
    assert stats["size"] == 1
    assert stats["backend_hits"] == stats["backend_misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert (await DiskCacheBackend(backend._cache_dir).get_stats())["backend_hits"] == 0


@pytest.mark.parametrize("content", [b"", b"partial", b"0\n{}", b"\xff\xfe"])
async def test_corrupt_or_truncated_file_is_a_miss_and_can_be_replaced(backend, content):
    assert await backend.set("k", _response())
    backend._path("k").write_bytes(content)
    assert await backend.get("k") is None
    assert not await backend.exists("k")
    assert await backend.set("k", _response())
    assert await backend.get("k") == _response()


@pytest.mark.parametrize("field", ["version", "key", "audio", "sample_rate", "metadata", "expiry"])
async def test_invalid_schema_is_a_safe_miss_even_with_valid_checksum(backend, field):
    assert await backend.set("k", _response())
    entry = backend._read(backend._path("k"))
    if field == "version":
        entry["version"] = 999
    elif field == "key":
        entry["key"] = "another-key"
    elif field == "audio":
        entry["response"]["audio_chunks"][0]["audio"] = "not base64!"
    elif field == "sample_rate":
        entry["response"]["sample_rate"] = 0
    elif field == "metadata":
        entry["response"]["metadata"] = []
    else:
        entry["expires_at"] = "tomorrow"
    payload = json.dumps(entry).encode()
    backend._path("k").write_bytes(hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload)
    assert await backend.get("k") is None
    assert await backend.clear() == 1


async def test_clear_leaves_unrelated_files_and_in_progress_temps(backend):
    assert await backend.set("k", _response())
    backend._path("k").write_bytes(b"broken")
    unrelated = backend._cache_dir / "notes.tts-cache"
    unrelated.write_text("keep")
    temporary = backend._cache_dir / ".tts-cache-in-progress.tmp"
    temporary.write_bytes(b"partial")
    assert await backend.clear("unknown") == 0
    assert await backend.clear() == 1
    assert unrelated.read_text() == "keep"
    assert temporary.read_bytes() == b"partial"


async def test_keys_cannot_traverse_directories(backend):
    key = "../../outside: /你好\\file"
    assert await backend.set(key, _response())
    assert await backend.get(key) == _response()
    assert list(backend._cache_dir.iterdir()) == [backend._path(key)]
    assert backend._path(key).parent == backend._cache_dir


async def test_generated_keys_exclude_secrets_and_session_settings(backend):
    kwargs = dict(text="hello", voice_id="v", model="m", sample_rate=16000, namespace="demo")
    key = generate_cache_key(**kwargs, settings={"api_key": "secret-one", "session_id": "old"})
    new_key = generate_cache_key(**kwargs, settings={"api_key": "secret-two", "session_id": "new"})
    assert key == new_key
    assert await backend.set(key, _response())
    assert await DiskCacheBackend(backend._cache_dir).get(new_key) == _response()
    contents = backend._path(key).read_bytes()
    assert b"secret-one" not in contents
    assert b"session_id" not in contents


async def test_unavailable_directory_is_fail_safe(tmp_path):
    occupied = tmp_path / "file"
    occupied.write_text("not a directory")
    backend = DiskCacheBackend(occupied)
    assert await backend.set("k", _response()) is False
    assert await backend.get("k") is None
    assert await backend.exists("k") is False
    assert await backend.delete("k") is False
    assert await backend.clear() == 0
    assert (await backend.get_stats())["type"] == "disk"


@pytest.mark.parametrize("failure", ["fsync", "replace"])
async def test_failed_write_preserves_old_entry_and_cleans_temp(backend, monkeypatch, failure):
    assert await backend.set("k", _response())

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(disk_module.os, failure, fail)
    assert await backend.set("k", _response(b"new")) is False
    assert await backend.get("k") == _response()
    assert list(backend._cache_dir.iterdir()) == [backend._path("k")]


async def test_non_json_metadata_does_not_destroy_previous_entry(backend):
    assert await backend.set("k", _response())
    unsupported = _response()
    unsupported.metadata["opaque"] = object()
    assert await backend.set("k", unsupported) is False
    assert await backend.get("k") == _response()


async def test_atomic_replacement_and_io_does_not_block_event_loop(backend, monkeypatch):
    assert await backend.set("k", _response())
    started = threading.Event()
    release = threading.Event()
    replace = disk_module.os.replace

    def paused_replace(source, target):
        started.set()
        assert release.wait(timeout=10)
        replace(source, target)

    monkeypatch.setattr(disk_module.os, "replace", paused_replace)
    writing = asyncio.create_task(backend.set("k", _response(b"new")))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        reader = DiskCacheBackend(backend._cache_dir)
        assert await reader.get("k") == _response()
        # This read finishes while the writer is paused in a worker thread.
    finally:
        release.set()
        assert await writing
    assert await reader.get("k") == _response(b"new")


async def test_concurrent_instances_read_complete_entries(backend):
    responses = [_response(bytes([i]) * 8192) for i in range(8)]
    assert await backend.set("k", responses[0])

    async def writer(response):
        instance = DiskCacheBackend(backend._cache_dir)
        for _ in range(8):
            assert await instance.set("k", response)
            assert await instance.get("k") in responses

    await asyncio.gather(*(writer(response) for response in responses))
    assert await backend.get("k") in responses
    assert list(backend._cache_dir.iterdir()) == [backend._path("k")]


_PROCESS_SCRIPT = """
import asyncio
import sys
from pipecat_tts_cache import DiskCacheBackend, CachedAudioChunk, CachedTTSResponse

async def main():
    backend = DiskCacheBackend(sys.argv[1])
    mode = sys.argv[2]
    response = CachedTTSResponse([CachedAudioChunk(b'\\x00\\x01' * 8192, 16000, 1)], 16000, 1)
    if mode == 'write':
        assert await backend.get('k') is None
        assert await backend.set('k', response, ttl=3600)
        print('miss -> stored')
        # Intentionally no close(): persistence cannot depend on graceful shutdown.
    elif mode == 'read':
        cached = await backend.get('k')
        assert cached.audio_chunks == response.audio_chunks
        print('hit')
    else:
        for _ in range(30):
            assert await backend.set('k', response)
            cached = await backend.get('k')
            assert cached.audio_chunks == response.audio_chunks
        print('concurrent ok')

asyncio.run(main())
"""


async def _process(directory, mode):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _PROCESS_SCRIPT,
        str(directory),
        mode,
        cwd=Path(__file__).resolve().parents[1],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, stderr.decode()
    return stdout.decode()


async def test_real_process_exit_and_restart(tmp_path):
    assert "miss -> stored" in await _process(tmp_path, "write")
    assert "hit" in await _process(tmp_path, "read")
    assert "hit" in await _process(tmp_path, "read")


async def test_concurrent_process_writers(tmp_path):
    results = await asyncio.gather(*(_process(tmp_path, "concurrent") for _ in range(3)))
    assert all("concurrent ok" in result for result in results)
    assert len(list(tmp_path.iterdir())) == 1
