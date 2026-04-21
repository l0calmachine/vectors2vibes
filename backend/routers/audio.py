"""
audio.py — audio serving endpoint.

Priority order (first matching env var wins):

  1. LOCAL_AUDIO_DIR — serve from local disk via FileResponse.
     Range requests handled natively; instant playback start.
     Set after running scripts/download_assets.py.

  2. PI_BASE_URL — redirect to Raspberry Pi static file server (302).
     Browser streams directly from the Pi; range requests handled by Caddy.
     No credentials exposed. Set PI_BASE_URL=http://<pi-ip>:8080 in .env.

  3. HuggingFace proxy (default) — downloads full file before responding.
     Adds latency on first play; LRU cache helps on repeat plays.
     Requires HF_TOKEN in .env. Note: proxying keeps the token server-side.
     Do NOT use a redirect to HF — that would expose HF_TOKEN in the browser.

Used with: GET /api/audio/stream/{track_id}
"""

import os
import asyncio
from collections import OrderedDict
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, FileResponse, RedirectResponse
from backend.services.embedding_service import get_embedding_service

router = APIRouter()

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_AUDIO_BASE = "https://huggingface.co/datasets/vectors2vibes/vectors2vibes-discogs-audio/resolve/main/{file_path}"

LOCAL_AUDIO_DIR = os.environ.get("LOCAL_AUDIO_DIR", "")
PI_BASE_URL     = os.environ.get("PI_BASE_URL", "").rstrip("/")

# ── LRU audio cache (HF proxy mode only) ─────────────────────────────────────
MAX_CACHED_TRACKS = 100
_cache   = OrderedDict()
_pending = {}


def _cache_get(track_id):
    if track_id in _cache:
        _cache.move_to_end(track_id)
        return _cache[track_id]
    return None


def _cache_set(track_id, data):
    if track_id in _cache:
        _cache.move_to_end(track_id)
    else:
        if len(_cache) >= MAX_CACHED_TRACKS:
            _cache.popitem(last=False)
        _cache[track_id] = data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_file_path(track_id):
    emb_svc = get_embedding_service()
    idx = emb_svc.get_idx(track_id)
    if idx is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return emb_svc.file_paths[idx]


# ── Stream endpoint ───────────────────────────────────────────────────────────

@router.get("/stream/{track_id}")
async def stream_audio(track_id: str):
    file_path = _get_file_path(track_id)

    # ── Local mode: FileResponse handles Range requests and chunked sending
    if LOCAL_AUDIO_DIR:
        local_path = Path(LOCAL_AUDIO_DIR) / file_path
        if not local_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Audio file not found locally: {file_path}. "
                       f"Run scripts/download_assets.py to populate LOCAL_AUDIO_DIR."
            )
        return FileResponse(
            path=str(local_path),
            media_type="audio/ogg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # ── Pi redirect mode: browser streams directly, range requests handled by Caddy
    if PI_BASE_URL:
        return RedirectResponse(
            url=f"{PI_BASE_URL}/audio/{file_path}",
            status_code=302,
        )

    # ── HuggingFace proxy mode ────────────────────────────────────────────────
    cached = _cache_get(track_id)
    if cached:
        return Response(
            content=cached,
            media_type="audio/ogg",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    if not HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Neither LOCAL_AUDIO_DIR nor HF_TOKEN is set. "
                   "Run scripts/download_assets.py or add HF_TOKEN to .env."
        )

    if track_id in _pending:
        await _pending[track_id].wait()
        cached = _cache_get(track_id)
        if cached:
            return Response(content=cached, media_type="audio/ogg")
        raise HTTPException(status_code=502, detail="Audio fetch failed")

    event = asyncio.Event()
    _pending[track_id] = event
    try:
        url = HF_AUDIO_BASE.format(file_path=file_path)
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
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/cache/status")
def cache_status():
    """Dev utility — shows how many tracks are cached (HF proxy mode only)."""
    total_bytes = sum(len(v) for v in _cache.values())
    return {
        "mode":          "local" if LOCAL_AUDIO_DIR else "pi-redirect" if PI_BASE_URL else "hf-proxy",
        "local_dir":     LOCAL_AUDIO_DIR or None,
        "pi_base_url":   PI_BASE_URL or None,
        "cached_tracks": len(_cache),
        "max_tracks":    MAX_CACHED_TRACKS,
        "total_mb":      round(total_bytes / 1_048_576, 2),
    }
