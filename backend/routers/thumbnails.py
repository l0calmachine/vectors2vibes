"""
thumbnails.py — proxy endpoint for track thumbnail images.

Serves preprocessed album art sourced from Discogs and YouTube.
Hosted on HuggingFace (vectors2vibes-preprocessed-thumbnails).
Caches aggressively.

Use with: GET /api/thumb/{video_id}
"""

import httpx
from collections import OrderedDict
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter()

HF_THUMB_BASE = "https://huggingface.co/datasets/vectors2vibes/vectors2vibes-preprocessed-thumbnails/resolve/main/{shard}/{track_id}.webp"

# Caches in memory
MAX_CACHED = 2000
_cache = OrderedDict()


def _cache_get(vid):
    """Return cached thumbnail bytes and content type, or None if not cached. Marks as recently used."""
    if vid in _cache:
        _cache.move_to_end(vid)
        return _cache[vid]
    return None


def _cache_set(vid, data, content_type):
    """Store thumbnail bytes in the cache, dropping the oldest if the cache is full."""
    if len(_cache) >= MAX_CACHED:
        _cache.popitem(last=False)
    _cache[vid] = (data, content_type)

_missing = set()  # track IDs with no thumbnail

@router.get("/{video_id}")
async def get_thumbnail(video_id):
    """Fetch preprocessed thumbnail from HuggingFace with caching."""
    if video_id in _missing:
        raise HTTPException(status_code=404, detail="No thumbnail")
    
    cached = _cache_get(video_id)
    if cached:
        data, ct = cached
        return Response(content=data, media_type=ct,
                        headers={"Cache-Control": "public, max-age=86400"})

    url = HF_THUMB_BASE.format(shard=video_id[:2], track_id=video_id)

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            r = await client.get(url)
            if r.status_code == 200 and len(r.content) > 500:
                ct = r.headers.get("content-type", "image/webp")
                _cache_set(video_id, r.content, ct)
                return Response(content=r.content, media_type=ct,
                                headers={"Cache-Control": "public, max-age=86400"})
        except Exception as e:
            print(f"[thumb] fetch error: {e}")

    _missing.add(video_id)
    raise HTTPException(status_code=404, detail=f"No thumbnail found for {video_id}")
