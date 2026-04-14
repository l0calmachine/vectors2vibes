"""
world.py — endpoints for serving track metadata, embedding coordinates, and layer positions to the frontend.
"""

from fastapi import APIRouter, Query
from backend.services.world_service import WorldService

router = APIRouter()
world_service = WorldService()


@router.get("/embeddings")
def get_embeddings(
    page: int = Query(0, ge=0),
    page_size: int = Query(200, le=500)
):
    """Return full metadata for a single track by ID."""
    return world_service.get_embeddings_page(page, page_size)


@router.get("/embedding/{embedding_id}")
def get_embedding(embedding_id: str):
    """Return a paginated slice of all track metadata and coordinates."""
    return world_service.get_embedding_detail(embedding_id)


@router.get("/layers")
def get_layer_positions(layer: str = Query("audio")):
    """Return {track_id: {pos_x, pos_z}} for all tracks in the requested layer.
    Accepts: audio, lyrical, year"""
    return world_service.get_layer_positions(layer)
