"""
navigation.py — endpoints for navigation styles (derive, detourn, stroll).
uses the embedding matrix for layer aware requests.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.navigation_service import NavigationService

router = APIRouter()
nav_service = NavigationService()


class NavRequest(BaseModel):
    current_ids: list[str]
    layer:       str   = 'audio'


class DeriveRequest(BaseModel):
    current_ids:      list[str]
    similarity_input: float = 0.5
    layer:            str   = 'audio'


class DetournRequest(BaseModel):
    current_ids: list[str]
    target_year: int   = 1990
    layer:       str   = 'audio'


@router.post("/derive")
def derive(req: DeriveRequest):
    """Drift toward tracks at a target cosine similarity from current position."""
    return nav_service.derive(
        req.current_ids, req.similarity_input, layer=req.layer,
    )


@router.post("/detourn")
def detourn(req: DetournRequest):
    """Jump to the centroid of a target year's songs."""
    return nav_service.detourn(
        req.current_ids, req.target_year, layer=req.layer,
    )


@router.post("/stroll")
def stroll(req: NavRequest):
    """Stroll to a random track not too similar to the current position."""
    return nav_service.stroll(req.current_ids, layer=req.layer)

