"""
thumbnails.py — thumbnail serving endpoint.

Serves from local disk when LOCAL_THUMB_DIR is set (recommended).
Falls back to proxying from HuggingFace when it isn't.

Local path layout mirrors the HF dataset: {LOCAL_THUMB_DIR}/{shard}/{track_id}.webp
where shard = track_id[:2].

Used with: GET /api/thumb/{track_id}
"""

import os
from collections import OrderedDict
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, FileResponse

router = APIRouter()

HF_THUMB_BASE  = "https://huggingface.co/datasets/vectors2vibes/vectors2vibes-preprocessed-thumbnails/resolve/main/{shard}/{track_id}.webp"
LOCAL_THUMB_DIR = os.environ.get("LOCAL_THUMB_DIR", "")

# ── LRU thumbnail cache (HF proxy mode only) ──────────────────────────────────
MAX_CACHED = 2000
_cache   = OrderedDict()
_missing = set()


def _cache_get(vid):
    if vid in _cache:
        _cache.move_to_end(vid)
        return _cache[vid]
    return None


def _cache_set(vid, data, content_type):
    if len(_cache) >= MAX_CACHED:
        _cache.popitem(last=False)
    _cache[vid] = (data, content_type)


# ── Thumbnail endpoint ────────────────────────────────────────────────────────

@router.get("/{video_id}")
async def get_thumbnail(video_id: str):
    shard = video_id[:2]

    # ── Local mode
    if LOCAL_THUMB_DIR:
        local_path = Path(LOCAL_THUMB_DIR) / shard / f"{video_id}.webp"
        if not local_path.exists():
            raise HTTPException(status_code=404, detail=f"Thumbnail not found locally: {local_path}")
        return FileResponse(
            path=str(local_path),
            media_type="image/webp",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # ── HuggingFace proxy mode ────────────────────────────────────────────────
    if video_id in _missing:
        raise HTTPException(status_code=404, detail="No thumbnail")

    cached = _cache_get(video_id)
    if cached:
        data, ct = cached
        return Response(
            content=data,
            media_type=ct,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    url = HF_THUMB_BASE.format(shard=shard, track_id=video_id)
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            r = await client.get(url)
            if r.status_code == 200 and len(r.content) > 500:
                ct = r.headers.get("content-type", "image/webp")
                _cache_set(video_id, r.content, ct)
                return Response(
                    content=r.content,
                    media_type=ct,
                    headers={"Cache-Control": "public, max-age=86400"},
                )
        except Exception as e:
            print(f"[thumb] fetch error: {e}")

    _missing.add(video_id)
    raise HTTPException(status_code=404, detail=f"No thumbnail found for {video_id}")
