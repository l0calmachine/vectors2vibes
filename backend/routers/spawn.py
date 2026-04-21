"""
spawn.py — API endpoint for spawning.
Calls spawn_service.py which contains spawn function from Greg's Navigation_v2.ipynb.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.spawn_service import SpawnService
from backend.services.embedding_service import get_embedding_service

router = APIRouter()
spawn_service = SpawnService()

class YearRequest(BaseModel):
    year: int = 1990

def _require_loaded():
    """Raise 503 if EmbeddingService hasn't finished loading — prevents spawning into an empty world."""
    if not get_embedding_service().is_loaded():
        raise HTTPException(status_code=503, detail="Embedding service not ready")

@router.post("/year")
def spawn_year(req: YearRequest):
    """Spawn at the centroid of the given year."""
    _require_loaded()
    return spawn_service.spawn_year(req.year)

