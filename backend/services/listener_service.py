"""
listener_service.py — tracks user behavior per session.

This service builds a personal centroid in embedding space as a weighted
average of the tracks a user has dwelled on, further augmented by navigational style
and visual layer interaction. This centroid is returned to the frontend and presented
as a "ghost," haunting the user throughout the world plane.

The centroid only "haunts" embedding planes, defaulting to the audio embedding plane
until lyrical_similarity exceeds audio_similarity, when it switches to haunting the
lyrical embedding plane. 

Behavior is stored per session. A new SESSION_ID is generated on page refresh.

"""

import random
from collections import defaultdict

from backend.services.embedding_service import get_embedding_service


DEFAULT_WEIGHTS = {
    "audio_similarity": 1.0,
    "lyrical_similarity": 1.0,
    "year_proximity": 1.0,
    "nav_derive": 1.0,
    "nav_detourn": 1.0,
    "nav_frolic": 1.0,
}

LEARNING_RATE = 0.05


class ListenerService:
    def __init__(self):
        """Initialise empty session store."""
        self._sessions = {}
        self._emb_svc = get_embedding_service()

    def _get_session(self, session_id):
        """Return the session dict for a given session ID, creating it if it doesn't exist."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "weights":          {**DEFAULT_WEIGHTS},
                "listen_log":       [], # [(emb_id, duration_ms, pos_x, pos_z, year)]
                "layer_log":        [], # [(layer, duration_ms)]
                "nav_log":          [], # [nav_style]
                "nav_counts":       defaultdict(int), # nav_style → count
                "layer_counts":     defaultdict(float), # layer → total ms spent on a visual layer
                "total_listen":     defaultdict(float), # track_id → total ms listened to an embedding
                "visit_counts":     defaultdict(int),   # track_id → number of separate visits to an embedding
                "centroid":         None,               # weighted centroid updated on each listen event
            }
        return self._sessions[session_id]

    def _recompute_centroid(self, s):
        """Recompute the weighted centroid based on user dwell, navigation, and visual layer behavior."""
        if not s["listen_log"]:
            return None

        emb_svc = self._emb_svc
        weights = s["weights"]

        # Determine dominant layer (audio or lyrical)
        audio_w   = weights["audio_similarity"]
        lyrical_w = weights["lyrical_similarity"]
        dominant  = 'lyrical' if lyrical_w > audio_w else 'audio'

        track_data = {}
        for emb_id, duration_ms, pos_x, pos_z, year in s["listen_log"]:
            if emb_id not in track_data:
                # Use layer aware coordinates
                idx = emb_svc.get_idx(emb_id)
                if idx is not None and dominant == 'lyrical':
                    lx, lz = emb_svc.get_world_pos(idx, layer='lyrical')
                else:
                    lx, lz = pos_x, pos_z  # audio coords

                track_data[emb_id] = {
                    'pos_x':  lx,
                    'pos_z':  lz,
                    'year':   year,
                    'weight': 0.0,
                }

            visits = s["visit_counts"][emb_id]
            return_bonus = 1.0 + (visits - 1) * 0.5
            track_data[emb_id]["weight"] += duration_ms * return_bonus

        total_weight = sum(t["weight"] for t in track_data.values())
        if total_weight == 0:
            return None

        # Base centroid position
        # cy = weighted average release year for the ghost centroid
        cx = sum(t["pos_x"] * t["weight"] for t in track_data.values()) / total_weight
        cz = sum(t["pos_z"] * t["weight"] for t in track_data.values()) / total_weight
        cy = sum(t["year"]  * t["weight"] for t in track_data.values()) / total_weight

        # nav_detourn — push centroid away from its current cluster
        # by inverting a fraction of the position proportional to the weight
        detourn_strength = weights["nav_detourn"] - 1.0  # 0 until detourn is used
        if detourn_strength > 0:
            cx = cx - (cx * detourn_strength * 0.1)
            cz = cz - (cz * detourn_strength * 0.1)

        # nav_frolic — add slight randomness proportional to frolic weight
        frolic_strength = weights["nav_frolic"] - 1.0
        if frolic_strength > 0:
            cx += random.gauss(0, frolic_strength * 10)
            cz += random.gauss(0, frolic_strength * 10)

        return {
            "pos_x":  round(cx, 2),
            "pos_z":  round(cz, 2),
            "year":   round(cy, 1),
        }

    def record_listen(self, session_id, embedding_id, duration_ms, pos_x=0, pos_z=0, year=0):
        """Record a listen event and update the session centroid and weights."""
        s = self._get_session(session_id)
        s["listen_log"].append((embedding_id, duration_ms, pos_x, pos_z, year))
        s["total_listen"][embedding_id] += duration_ms
        s["visit_counts"][embedding_id] += 1

        new_centroid = self._recompute_centroid(s)
        if new_centroid:
            s["centroid"] = new_centroid

        if duration_ms > 30_000:
            s["weights"]["audio_similarity"] = min(
                3.0, s["weights"]["audio_similarity"] + LEARNING_RATE
            )
        if s["visit_counts"][embedding_id] > 1:
            s["weights"]["year_proximity"] = min(
                3.0, s["weights"]["year_proximity"] + LEARNING_RATE * 0.3
            )

        return {
            "status":   "ok",
            "weights":  s["weights"],
            "centroid": s["centroid"],
        }

    def record_layer(self, session_id, layer, duration_ms):
        """Record a layer switch event and adjust weights toward the active layer."""
        s = self._get_session(session_id)
        s["layer_log"].append((layer, duration_ms))
        s["layer_counts"][layer] += duration_ms
        weight_map = {
            "audio": "audio_similarity",
            "lyrical": "lyrical_similarity",
            "year": "year_proximity",
        }
        if layer in weight_map:
            key = weight_map[layer]
            s["weights"][key] = min(3.0, s["weights"][key] + LEARNING_RATE * 0.5)
        return {"status": "ok", "weights": s["weights"]}

    def record_nav(self, session_id, nav_style):
        """Record a navigation event and adjust weights based on navigation style."""
        s = self._get_session(session_id)
        s["nav_log"].append(nav_style)
        s["nav_counts"][nav_style] += 1
        adjustments = {
            "detourn": ("nav_detourn", +LEARNING_RATE),
            "frolic":  ("nav_frolic",  +LEARNING_RATE),
            "derive":  ("nav_derive",  +LEARNING_RATE),
        }
        if nav_style in adjustments:
            key, delta = adjustments[nav_style]
            s["weights"][key] = max(0.1, min(3.0, s["weights"][key] + delta))
        return {"status": "ok", "weights": s["weights"]}
