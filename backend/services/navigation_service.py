"""
navigation_service.py — navigation functions written from Greg's Navigation_v2.ipynb, adapted for the UI.

HOW THIS DIFFERS FROM THE NOTEBOOK
------------------------------------
The notebook operated on a pandas DataFrame (df) and a stacked numpy embeddings array
passed as function arguments. In the service layer (~/vectors2vibes/backend/services)
these are replaced by EmbeddingService (defined in embedding_service.py),
which loads the parquet data once at startup, caches it into memory, and
holds its attributes (i.e., audio_embs, years, genres, etc.) so that any service
can access them directly via the function get_embedding_service().
Specific changes (changes not specified here are commented inline):

  drift()                 → derive()
  jump()                  → detourn()
  df["year"]              → emb_svc.years
  embeddings              → emb_svc.audio_embs
  position (numpy array)  → current_ids: centroid of k nearest track embeddings
  (from getCurrentCentroidIds, defined in index.html) resolved as the nearest track ID.
  get_year_centroid()     → layer aware: now uses audio_embs or lyric_embs to match the
  visual layer (audio/lyrical) by calling get_embeddings_matrix(layer) (defined in embedding_service.py).
  The year visual layer defaults to audio_embs.
            
_format_result() was created as a helper function. It returns the track ID dicts, which the
service layer uses to populate metadata and coordinates to the frontend.

"""

import random

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.services.world_service import WorldService
from backend.services.embedding_service import get_embedding_service


class NavigationService:
    def __init__(self):
        self.world   = WorldService()
        self.emb_svc = get_embedding_service()

    # ── Support functions ─────────────────────────────────────────────────

    def get_year_centroid(self, target_year, layer='audio'):
        """Get the mean embedding vector for all songs from a given year."""
        emb_svc = self.emb_svc
        indices = [i for i, y in enumerate(emb_svc.years) if y == target_year]
        # Added to handle user input years without an exact dataset match in derive()
        if not indices:
            return None

        # Added for layer aware functionality
        matrix = emb_svc.get_embeddings_matrix(layer)
        
        return np.mean(matrix[indices], axis=0)
    

    def _format_result(self, t):
        """
        Returns the track fields needed for the frontend.

        pos_x/pos_z are included so warpTo() has a fallback position
        when an embedding neighborhood hasn't been loaded yet (i.e., if it's too far away).

        """
        return {
            'id':            t['id'],
            'title':         t['title'],
            'artist':        t['artist'],
            'year':          t['year'],
            'pos_x':         t['pos_x'],
            'pos_z':         t['pos_z'],
            'thumbnail_url': t['thumbnail_url'],
        }


    # ── Navigation functions ─────────────────────────────────────────────────

    # Added user behavior weights and layer aware arguments
    def derive(self, current_ids, target_year, step_size=0.1, weights=None, layer='audio'):
        """Drift position toward a target year's centroid."""
        emb_svc = self.emb_svc

        # Added to establish centroid as position
        indices = [emb_svc.get_idx(id) for id in current_ids if emb_svc.get_idx(id) is not None]
        matrix = emb_svc.get_embeddings_matrix(layer)
        position = np.mean(matrix[indices], axis=0)

        target = self.get_year_centroid(target_year, layer)
        # Added fallback in case user input year doesn't exist in dataset
        if target is None:
            available = sorted(set(emb_svc.years), key=lambda y: abs(y - target_year))
            target = self.get_year_centroid(available[0], layer)

        direction = target - position
        direction /= np.linalg.norm(direction)
        new_position = position + step_size * direction

        # Added to resolve new_position to the closest track ID
        sims = cosine_similarity([new_position], emb_svc.get_embeddings_matrix(layer))[0]
        dest_id = emb_svc.get_id_at(int(np.argmax(sims)))
        dest = self.world.get_by_id(dest_id) 

        return {
            'mode':        'derive',
            'target_year': target_year,
            'destination': self._format_result(dest),
        }

    # Added user behavior weights and layer aware arguments
    def detourn(self, current_ids, weights=None, layer='audio'):
        """Jump to the song with the lowest cosine similarity to the current position."""
        # Print for debug
        print(f"[detourn] layer={layer}, current_ids={current_ids[:3]}")
        
        emb_svc = self.emb_svc

        # Added to establish centroid as position
        indices = [emb_svc.get_idx(id) for id in current_ids if emb_svc.get_idx(id) is not None]
        matrix = emb_svc.get_embeddings_matrix(layer)
        position = np.mean(matrix[indices], axis=0)

        sims = cosine_similarity([position], matrix)[0]
        furthest_idx = np.argmin(sims)

        # Added to resolve new_position to the closest track ID 
        dest_id = emb_svc.get_id_at(int(furthest_idx))
        dest = self.world.get_by_id(dest_id)
        
        return {'mode': 'detourn', 'destination': self._format_result(dest)}

    # Added user behavior weights and layer aware arguments
    def frolic(self, current_ids, weights=None, layer='audio'):
        """Frolic to a random song that is not too similar to the current position."""
        emb_svc = self.emb_svc

        # Added to establish centroid as position
        indices = [emb_svc.get_idx(id) for id in current_ids if emb_svc.get_idx(id) is not None]
        matrix = emb_svc.get_embeddings_matrix(layer)
        position = np.mean(matrix[indices], axis=0)

        random_idx = np.random.randint(len(emb_svc.ids))
        sim = cosine_similarity([position], [matrix[random_idx]])[0][0]

        while np.array_equal(matrix[random_idx], position) or sim > 0.9:
            random_idx = np.random.randint(len(emb_svc.ids))
            sim = cosine_similarity([position], [matrix[random_idx]])[0][0]

        # Added to resolve new_position to the closest track ID 
        dest_id = emb_svc.get_id_at(int(random_idx))
        dest = self.world.get_by_id(dest_id)
        
        return {'mode': 'frolic', 'destination': self._format_result(dest)}


