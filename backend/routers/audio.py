"""
audio.py — proxy endpoint for audio streaming (currently using HuggingFace).

Why a proxy instead of direct HF URLs:
  - HF token doesn't get exposed to browser
  - LRU cache allows for revisited tracks to be streamed from cache rather than re-download
  - Easy one-place swap if we want to change audio source in future (i.e., local files)

Used with: GET /api/audio/stream/{track_id}
Returns audio bytes with appropriate Content-Type for playing audio in browser.

To use, you must set HF_TOKEN in your .env file.
"""

import os
import asyncio
from collections import OrderedDict

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from backend.services.embedding_service import get_embedding_service

router = APIRouter()

# Audio source — swap these if serving locally hosted files
AUDIO_SOURCE = "huggingface"  # "huggingface" or "local"

LOCAL_AUDIO_PATH = None  # e.g. Path("/path/to/audio/files") if hosting locally

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_AUDIO_BASE = "https://huggingface.co/datasets/vectors2vibes/vectors2vibes-discogs-audio/resolve/main/{file_path}"

# ── LRU audio cache ──────────────────────────────────────────────
# Stores raw audio bytes in memory. Capped at MAX_CACHED_TRACKS entries.
# Least recently used tracks are dropped when cap is reached.
# Each .ogg is ~2-5MB, so 100 tracks ≈ 200-500MB RAM.
# Lower MAX_CACHED_TRACKS if lag issue continues.
MAX_CACHED_TRACKS = 100
_cache = OrderedDict()
_pending = {} # prevents duplicate simultaneous fetches


def _cache_get(track_id):
    """Return cached audio bytes for a track ID, or None if not cached. Marks as recently used."""
    if track_id in _cache:
        _cache.move_to_end(track_id) # mark as recently used
        return _cache[track_id]
    return None


def _cache_set(track_id, data):
    """Store audio bytes in the LRU cache, dropping the oldest entry if the cache is full."""
    if track_id in _cache:
        _cache.move_to_end(track_id)
    else:
        if len(_cache) >= MAX_CACHED_TRACKS:
            _cache.popitem(last=False) # drop oldest
        _cache[track_id] = data

# ── Stream endpoint ──────────────────────────────────────────────

@router.get("/stream/{track_id}")
async def stream_audio(track_id: str):
    """Proxy audio with LRU caching. Returns audio bytes for browser playback."""
    # Check cache first
    cached = _cache_get(track_id)
    if cached:
        return Response(
            content=cached,
            media_type="audio/ogg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
    
    # Get file path from embedding service
    emb_svc = get_embedding_service()
    idx = emb_svc.get_idx(track_id)
    file_path = emb_svc.file_paths[idx] if idx is not None else None

    if AUDIO_SOURCE == "local":
        if not LOCAL_AUDIO_PATH:
            raise HTTPException(status_code=503, detail="LOCAL_AUDIO_PATH not set")
        audio_bytes = (LOCAL_AUDIO_PATH / file_path).read_bytes()
    else:
        url = HF_AUDIO_BASE.format(file_path=file_path)
        if not HF_TOKEN:
            raise HTTPException(
                status_code=503,
                detail="HF_TOKEN not set — add it to your .env file to enable audio streaming"
            )
        # Avoid fetching the same track twice simultaneously
        if track_id in _pending:
            await _pending[track_id].wait()
            cached = _cache_get(track_id)
            if cached:
                return Response(content=cached, media_type="audio/ogg")
            raise HTTPException(status_code=502, detail="Audio fetch failed")
        
        event = asyncio.Event()
        _pending[track_id] = event
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                    follow_redirects=True,
                )
                if resp.status_code == 401:
                    raise HTTPException(status_code=401, detail="HF token invalid or expired")
                if resp.status_code == 404:
                    raise HTTPException(status_code=404, detail=f"Audio file not found on HF: {file_path}")
                if resp.status_code != 200:
                    raise HTTPException(status_code=502, detail=f"HF returned {resp.status_code}")
                audio_bytes = resp.content
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="HF request timed out")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Audio fetch error: {str(e)}")
        finally:
            event.set()
            _pending.pop(track_id, None)

    _cache_set(track_id, audio_bytes)
    return Response(
        content=audio_bytes,
        media_type="audio/ogg",
        headers={"Cache-Control": "public, max-age=3600"}
    )

@router.get("/cache/status")
def cache_status():
    """Dev utility — shows how many tracks are cached and total size."""
    total_bytes = sum(len(v) for v in _cache.values())
    return {
        "cached_tracks": len(_cache),
        "max_tracks":    MAX_CACHED_TRACKS,
        "total_mb":      round(total_bytes / 1_048_576, 2),
    }
