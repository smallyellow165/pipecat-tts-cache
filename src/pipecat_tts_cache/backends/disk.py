#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Persistent local-file cache, using the same response model as Memory and Redis."""

import asyncio
import base64
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from pipecat_tts_cache.backends.base import CacheBackend
from pipecat_tts_cache.models import CachedAudioChunk, CachedTTSResponse, CachedWordTimestamp


class DiskCacheBackend(CacheBackend):
    """Persistent cache for a local filesystem, with no additional dependencies.

    Each entry is a SHA-256 checksum line followed by versioned UTF-8 JSON. Audio
    bytes are Base64; metadata must be JSON-compatible. Files contain response data,
    never Pipecat frames or service objects. A dedicated directory is recommended.

    I/O runs in worker threads. Atomic replacement allows concurrent processes to
    read complete old/new entries; the last replacement wins. Clear is best-effort
    during concurrent writes, like Redis SCAN. Expired/corrupt files remain until
    overwritten, deleted or cleared: a stale reader must not unlink a fresh write.
    There is no size eviction or background sweeper. Stats counters are per instance.
    """

    def __init__(self, cache_dir: str | os.PathLike[str]):
        """Initialize the backend; create the directory lazily on the first write.

        Args:
            cache_dir: Persistent local directory, reused across server restarts.
                Relative paths are resolved when the backend is constructed.
        """
        self._cache_dir = Path(cache_dir).expanduser().absolute()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _path(self, key: str) -> Path:
        # Arbitrary keys/namespaces cannot escape the cache directory.
        return self._cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.tts-cache"

    @staticmethod
    def _number(value: Any, *, positive_int: bool = False) -> None:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Invalid numeric cache field")
        if positive_int and (type(value) is not int or value <= 0):
            raise ValueError("Invalid audio format")

    @classmethod
    def _decode_response(cls, data: dict) -> CachedTTSResponse:
        chunks = []
        for chunk in data["audio_chunks"]:
            chunk = dict(chunk)
            chunk["audio"] = base64.b64decode(chunk["audio"], validate=True)
            cls._number(chunk["sample_rate"], positive_int=True)
            cls._number(chunk["num_channels"], positive_int=True)
            if chunk["pts"] is not None and type(chunk["pts"]) is not int:
                raise ValueError("Invalid audio timestamp")
            chunks.append(CachedAudioChunk(**chunk))
        timestamps = data["word_timestamps"]
        if timestamps is not None:
            timestamps = [CachedWordTimestamp(**item) for item in timestamps]
            for item in timestamps:
                if not isinstance(item.word, str):
                    raise ValueError("Invalid word")
                cls._number(item.timestamp)
        cls._number(data["sample_rate"], positive_int=True)
        cls._number(data["num_channels"], positive_int=True)
        cls._number(data["total_duration_s"])
        cls._number(data["created_at"])
        if not isinstance(data["metadata"], dict):
            raise ValueError("Invalid metadata")
        return CachedTTSResponse(**{**data, "audio_chunks": chunks, "word_timestamps": timestamps})

    def _read(self, path: Path) -> dict:
        checksum, payload = path.read_bytes().split(b"\n", 1)
        if checksum != hashlib.sha256(payload).hexdigest().encode("ascii"):
            raise ValueError("Cache checksum mismatch")
        entry = json.loads(payload)
        if type(entry["version"]) is not int or entry["version"] != 1:
            raise ValueError("Unsupported disk cache version")
        if self._path(entry["key"]) != path:
            raise ValueError("Cache key mismatch")
        self._number(entry["expires_at"])
        return entry

    def _get(self, key: str, count: bool) -> Optional[CachedTTSResponse]:
        with self._lock:
            response = None
            try:
                entry = self._read(self._path(key))
                expiry = entry["expires_at"]
                if expiry == 0 or time.time() <= expiry:
                    response = self._decode_response(entry["response"])
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Disk cache read failed: {type(e).__name__}")
            if count:
                self._hits += response is not None
                self._misses += response is None
            return response

    async def get(self, key: str) -> Optional[CachedTTSResponse]:
        """Retrieve cached response, or None if missing, expired or unreadable."""
        return await asyncio.to_thread(self._get, key, True)

    def _set(self, key: str, response: CachedTTSResponse, ttl: Optional[int]) -> bool:
        with self._lock:
            temporary = None
            try:
                data = asdict(response)
                for chunk in data["audio_chunks"]:
                    chunk["audio"] = base64.b64encode(chunk["audio"]).decode("ascii")
                self._decode_response(data)
                entry = {
                    "version": 1,
                    "key": key,
                    "expires_at": time.time() + ttl if ttl and ttl > 0 else 0.0,
                    "response": data,
                }
                payload = json.dumps(entry, ensure_ascii=True, allow_nan=False).encode("utf-8")
                checksum = hashlib.sha256(payload).hexdigest().encode("ascii")
                self._cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                # Same-directory temp + replace keeps readers away from partial writes.
                # NamedTemporaryFile creates owner-only files and closes before replace
                # (also required on Windows). A failed write preserves the old entry.
                with tempfile.NamedTemporaryFile(
                    dir=self._cache_dir, prefix=".tts-cache-", suffix=".tmp", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(checksum + b"\n" + payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self._path(key))
                return True
            except Exception as e:
                logger.warning(f"Disk cache write failed: {type(e).__name__}")
                return False
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass

    async def set(self, key: str, response: CachedTTSResponse, ttl: Optional[int] = None) -> bool:
        """Atomically store a response; None or non-positive TTL means no expiry."""
        return await asyncio.to_thread(self._set, key, response, ttl)

    @staticmethod
    def _unlink(path: Path) -> bool:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.warning(f"Disk cache delete failed: {type(e).__name__}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete an entry, including expired/corrupt data; return whether deleted."""
        return await asyncio.to_thread(self._unlink, self._path(key))

    def _entries(self) -> list[Path]:
        return [
            path
            for path in self._cache_dir.glob("*.tts-cache")
            if len(path.stem) == 64 and all(c in "0123456789abcdef" for c in path.stem)
        ]

    def _clear(self, namespace: Optional[str]) -> int:
        count = 0
        try:
            for path in self._entries():
                if namespace is not None:
                    try:
                        if not self._read(path)["key"].startswith(f"{namespace}:"):
                            continue
                    except Exception:
                        # Cannot attribute a corrupt entry to a namespace safely.
                        continue
                count += self._unlink(path)
        except OSError as e:
            logger.warning(f"Disk cache clear failed: {type(e).__name__}")
        return count

    async def clear(self, namespace: Optional[str] = None) -> int:
        """Remove all entries or a literal namespace prefix; return deleted count."""
        return await asyncio.to_thread(self._clear, namespace)

    async def exists(self, key: str) -> bool:
        """Check for a readable, unexpired entry without incrementing hit counters."""
        return await asyncio.to_thread(self._get, key, False) is not None

    def _stats(self) -> Dict[str, Any]:
        with self._lock:
            try:
                total = self._hits + self._misses
                return {
                    "type": "disk",
                    "cache_dir": str(self._cache_dir),
                    "size": len(self._entries()),
                    "backend_hits": self._hits,
                    "backend_misses": self._misses,
                    "hit_rate": self._hits / total if total else 0.0,
                }
            except OSError as e:
                return {"type": "disk", "error": type(e).__name__}

    async def get_stats(self) -> Dict[str, Any]:
        """Report file count (including expired/corrupt entries) and local counters."""
        return await asyncio.to_thread(self._stats)

    async def close(self) -> None:
        """Keep persistent files; no connections or background workers are owned."""
