"""
listener.py — endpoints for user behavior tracking and weight adjustment (session only).
              drives user centroid "ghost".
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.listener_service import ListenerService

router = APIRouter()
listener_service = ListenerService()


class ListenEvent(BaseModel):
    session_id: str
    embedding_id: str
    duration_ms: int
    pos_x: float = 0.0
    pos_z: float = 0.0
    year: int = 0


class LayerEvent(BaseModel):
    session_id: str
    layer: str
    duration_ms: int # how long user spent on this layer


class NavEvent(BaseModel):
    session_id:       str
    nav_style:        str            # derive/detourn/stroll
    similarity_input: Optional[float] = None  # derive only


@router.post("/listen")
def record_listen(event: ListenEvent):
    """
    Records how long a user listened to an embedding.
    Adjusts the weight of dwell time in centroid calculation.
    """
    return listener_service.record_listen(
        event.session_id, event.embedding_id, event.duration_ms,
        event.pos_x, event.pos_z, event.year
    )


@router.post("/layer")
def record_layer(event: LayerEvent):
    """
    Records which layers the user spends the most time using.
    Adjusts layer preference weights in centroid calculation.
    """
    return listener_service.record_layer(event.session_id, event.layer, event.duration_ms)


@router.post("/nav")
def record_nav(event: NavEvent):
    """
    Records which navigational style the user prefers.
    Adjusts nav weights in centroid calculation.
    """
    return listener_service.record_nav(event.session_id, event.nav_style, event.similarity_input)
