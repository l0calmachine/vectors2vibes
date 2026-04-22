"""
listener_service.py — tracks user behavior per session.

This service builds a personal centroid in embedding space as a weighted
average of the tracks a user has dwelled on, further augmented by navigational style
and visual layer interaction. This centroid is returned to the frontend and presented
as a "ghost," haunting the user throughout the world plane.

The centroid is computed in embedding space (not raw world coordinates), then
projected to world coordinates via the dominant layer. Nav and layer weights
determine which embedding space to use and how the centroid is pulled:

  derive (high sim input)  → pull centroid toward similar-sounding neighbors
  derive (low sim input)   → pull centroid toward dissimilar-sounding neighbors
  detourn                  → pull centroid toward the year centroid of dwelled tracks
  stroll                   → add slight randomness to the centroid position
  audio layer              → audio embedding space dominates
  lyrical layer            → lyrical embedding space dominates
  combined layer           → audio and lyrical weighted equally

Behavior is stored per session. A new SESSION_ID is generated on page refresh.

"""

from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.services.embedding_service import get_embedding_service


DEFAULT_WEIGHTS = {
    "audio_similarity":  1.0,
    "lyrical_similarity": 1.0,
    "year_proximity":    1.0,
    "nav_derive":        1.0,
    "nav_detourn":       1.0,
    "nav_stroll":        1.0,
}

LEARNING_RATE = 0.05


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class ListenerService:
    def __init__(self):
        """Initialise empty session store."""
        self._sessions = {}
        self._emb_svc = get_embedding_service()

    def _get_session(self, session_id):
        """Return the session dict for a given session ID, creating it if it doesn't exist."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "weights":      {**DEFAULT_WEIGHTS},
                "listen_log":   [],              # [(emb_id, duration_ms, pos_x, pos_z, year)]
                "layer_log":    [],              # [(layer, duration_ms)]
                "nav_log":      [],              # [nav_style]
                "nav_counts":   defaultdict(int),
                "layer_counts": defaultdict(float),
                "total_listen": defaultdict(float),
                "visit_counts": defaultdict(int),
                "derive_inputs": [],             # similarity_input floats from derive calls
                "centroid":     None,
            }
        return self._sessions[session_id]

    def _recompute_centroid(self, s):
        """
        Recompute the weighted centroid in embedding space.

        1. Build per-track dwell weights (duration * return bonus).
        2. Compute weighted mean embedding in the dominant layer space.
        3. Apply derive effect: pull mean toward/away from similar tracks.
        4. Apply detourn effect: blend in year centroid of dwelled tracks.
        5. Apply stroll noise.
        6. Find nearest track to final embedding; return its world position.
        """
        if not s["listen_log"]:
            return None

        emb_svc = self._emb_svc
        w = s["weights"]

        # ── Step 1: per-track dwell weights ─────────────────────────────────
        track_data = {}
        for emb_id, duration_ms, pos_x, pos_z, year in s["listen_log"]:
            idx = emb_svc.get_idx(emb_id)
            if idx is None:
                continue
            if emb_id not in track_data:
                track_data[emb_id] = {'idx': idx, 'weight': 0.0, 'year': year}
            visits = s["visit_counts"][emb_id]
            return_bonus = 1.0 + (visits - 1) * 0.5
            track_data[emb_id]['weight'] += duration_ms * return_bonus

        if not track_data:
            return None

        total_dwell = sum(t['weight'] for t in track_data.values())
        if total_dwell == 0:
            return None

        # ── Step 2: dominant layer + weighted mean embedding ─────────────────
        audio_w   = w["audio_similarity"]
        lyrical_w = w["lyrical_similarity"]

        if audio_w > lyrical_w:
            layer      = 'audio'
            emb_matrix = emb_svc.audio_embs
        elif lyrical_w > audio_w:
            layer      = 'lyrical'
            emb_matrix = emb_svc.lyric_embs
        else:
            layer      = 'combined'
            emb_matrix = emb_svc.combined_embs

        mean_emb  = np.zeros(emb_matrix.shape[1], dtype=np.float32)
        mean_year = 0.0
        for data in track_data.values():
            frac      = data['weight'] / total_dwell
            mean_emb += emb_matrix[data['idx']] * frac
            mean_year += data['year'] * frac

        mean_emb = _norm(mean_emb)

        # ── Step 3: derive effect ────────────────────────────────────────────
        # Pull toward (high input) or away from (low input) similar tracks.
        derive_strength = (w["nav_derive"] - 1.0) * (audio_w / max(audio_w + lyrical_w, 1))
        if derive_strength > 0 and s["derive_inputs"]:
            mean_sim_input = float(np.mean(s["derive_inputs"]))
            sims       = cosine_similarity([mean_emb], emb_matrix)[0]
            tol        = 0.1
            candidates = np.where(np.abs(sims - mean_sim_input) < tol)[0]
            if len(candidates) == 0:
                candidates = np.array([int(np.argmin(np.abs(sims - mean_sim_input)))])
            derive_emb = _norm(np.mean(emb_matrix[candidates], axis=0))
            blend      = min(derive_strength * 0.3, 0.5)
            mean_emb   = _norm(mean_emb * (1 - blend) + derive_emb * blend)

        # ── Step 4: detourn effect ───────────────────────────────────────────
        # Blend in the year centroid of the user's weighted-average dwell year.
        year_w          = w["year_proximity"]
        detourn_strength = (w["nav_detourn"] - 1.0) * (year_w / max(audio_w + lyrical_w + year_w, 1))
        if detourn_strength > 0:
            target_year  = int(round(mean_year))
            year_indices = [i for i, y in enumerate(emb_svc.years) if y == target_year]
            if not year_indices:
                year_indices = [i for i, y in enumerate(emb_svc.years) if abs(y - target_year) <= 2]
            if year_indices:
                year_centroid_emb = _norm(np.mean(emb_matrix[year_indices], axis=0))
                blend    = min(detourn_strength * 0.3, 0.5)
                mean_emb = _norm(mean_emb * (1 - blend) + year_centroid_emb * blend)

        # ── Step 5: stroll noise ─────────────────────────────────────────────
        stroll_strength = w["nav_stroll"] - 1.0
        if stroll_strength > 0:
            noise    = np.random.randn(len(mean_emb)).astype(np.float32) * stroll_strength * 0.02
            mean_emb = _norm(mean_emb + noise)

        # ── Step 6: nearest track → world position ───────────────────────────
        sims        = cosine_similarity([mean_emb], emb_matrix)[0]
        nearest_idx = int(np.argmax(sims))
        cx, cz      = emb_svc.get_world_pos(nearest_idx, layer=layer)

        return {
            "pos_x": round(cx, 2),
            "pos_z": round(cz, 2),
            "year":  round(mean_year, 1),
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

        delta = LEARNING_RATE * 0.5
        if layer == "audio":
            s["weights"]["audio_similarity"] = min(3.0, s["weights"]["audio_similarity"] + delta)
        elif layer == "lyrical":
            s["weights"]["lyrical_similarity"] = min(3.0, s["weights"]["lyrical_similarity"] + delta)
        elif layer == "combined":
            s["weights"]["audio_similarity"]   = min(3.0, s["weights"]["audio_similarity"]   + delta)
            s["weights"]["lyrical_similarity"] = min(3.0, s["weights"]["lyrical_similarity"] + delta)

        return {"status": "ok", "weights": s["weights"]}

    def record_nav(self, session_id, nav_style, similarity_input=None):
        """Record a navigation event and adjust weights based on navigation style."""
        s = self._get_session(session_id)
        s["nav_log"].append(nav_style)
        s["nav_counts"][nav_style] += 1

        # Always bump the nav dominance weight for centroid pull logic
        nav_key_map = {
            "derive":  "nav_derive",
            "detourn": "nav_detourn",
            "stroll":  "nav_stroll",
        }
        if nav_style in nav_key_map:
            key = nav_key_map[nav_style]
            s["weights"][key] = max(0.1, min(3.0, s["weights"][key] + LEARNING_RATE))

        if nav_style == "derive" and similarity_input is not None:
            s["derive_inputs"].append(float(similarity_input))
            # High input (→1) = seeking similar = increase audio_similarity
            # Low input (→0)  = seeking dissimilar = decrease audio_similarity
            direction = (float(similarity_input) - 0.5) * 2  # maps [0,1] → [-1,+1]
            s["weights"]["audio_similarity"] = max(
                0.1, min(3.0, s["weights"]["audio_similarity"] + direction * LEARNING_RATE)
            )

        elif nav_style == "detourn":
            s["weights"]["year_proximity"] = max(
                0.1, min(3.0, s["weights"]["year_proximity"] + LEARNING_RATE)
            )

        return {"status": "ok", "weights": s["weights"]}
