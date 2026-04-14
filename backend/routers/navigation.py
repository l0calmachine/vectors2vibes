"""
navigation.py — endpoints for navigation styles (derive, detourn, frolic).
uses the embedding matrix for layer aware requests.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.navigation_service import NavigationService

router = APIRouter()
nav_service = NavigationService()


class NavRequest(BaseModel):
    current_ids: list[str]
    weights:     dict  = None
    layer:       str   = 'audio'


class DeriveRequest(BaseModel):
    current_ids: list[str]
    target_year: float = 1990.0
    weights:     dict  = None
    layer:       str   = 'audio'


@router.post("/derive")
def derive(req: DeriveRequest):
    """Drift toward the centroid of a target year's audio embeddings."""
    return nav_service.derive(
        req.current_ids, req.target_year,
        weights=req.weights, layer=req.layer,
    )


@router.post("/detourn")
def detourn(req: NavRequest):
    """Jump to the track most dissimilar to the current position."""
    return nav_service.detourn(req.current_ids, req.weights, req.layer)


@router.post("/frolic")
def frolic(req: NavRequest):
    """Jump to a random track away from the current position."""
    return nav_service.frolic(req.current_ids, req.layer)
