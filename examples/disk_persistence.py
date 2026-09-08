"""Run twice with the same directory to verify persistence without API credentials.

This exercises storage with synthetic PCM, not TTS synthesis or playback. For a
voice pipeline, set DISK_CACHE_DIR and use basic_caching.py.
"""

import argparse
import asyncio

from pipecat_tts_cache import (
    CachedAudioChunk,
    CachedTTSResponse,
    DiskCacheBackend,
    generate_cache_key,
)


async def main(cache_dir: str):
    key = generate_cache_key("disk demo", "demo", "demo", 16000, namespace="disk-demo")
    async with DiskCacheBackend(cache_dir) as backend:
        for _ in range(2):
            response = await backend.get(key)
            if response is None:
                print("miss")
                response = CachedTTSResponse(
                    audio_chunks=[CachedAudioChunk(b"\x00\x00" * 160, 16000, 1)],
                    sample_rate=16000,
                    num_channels=1,
                )
                if not await backend.set(key, response, ttl=3600):
                    raise RuntimeError("Cache write failed; check the directory")
                print("stored")
            else:
                print("hit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir")
    asyncio.run(main(parser.parse_args().cache_dir))
