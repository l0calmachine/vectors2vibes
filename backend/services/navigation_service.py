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
  get_year_centroid()     → layer aware: now uses audio_embs, lyric_embs, or combined_embs to match the
  visual layer by calling get_embeddings_matrix(layer) (defined in embedding_service.py).

_format_result() was created as a helper function. It returns the track ID dicts, which the
service layer uses to populate metadata and coordinates to the frontend.

"""

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


    def _centroid_meta(self, position, similarity, layer):
        """Return centroid_umap and similarity for a nav response."""
        try:
            cx, cz = self.emb_svc.project_to_world(position, layer)
            centroid_umap = {'pos_x': round(cx, 1), 'pos_z': round(cz, 1)}
        except Exception:
            centroid_umap = None
        return {
            'centroid_umap': centroid_umap,
            'similarity':    round(float(similarity), 4),
        }

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
    def derive(self, current_ids, similarity_input, layer='audio', tolerance=0.05):
        """
        Drift away from the current position based on a target cosine similarity.
        similarity_input: float between -1 and 1
                          positive values drift toward similar songs
                          negative values drift toward dissimilar songs
        """
        if not (-1 <= similarity_input <= 1):
            raise ValueError("similarity_input must be between -1 and 1")

        emb_svc = self.emb_svc

        # Establish centroid as position from current track IDs
        indices = [emb_svc.get_idx(id) for id in current_ids if emb_svc.get_idx(id) is not None]
        matrix = emb_svc.get_embeddings_matrix(layer)
        position = np.mean(matrix[indices], axis=0)

        sims = cosine_similarity([position], matrix)[0]

        # Find candidates within tolerance range of the target similarity
        lower = similarity_input - tolerance
        upper = similarity_input + tolerance
        candidate_indices = [i for i, s in enumerate(sims) if lower <= s <= upper]

        if len(candidate_indices) > 5:
            candidate_indices = np.random.choice(candidate_indices, size=5, replace=False).tolist()

        if candidate_indices:
            # Randomly select one candidate from within the range
            chosen_idx = int(np.random.choice(candidate_indices))
        else:
            # Fallback: find the song whose similarity is closest to the input
            chosen_idx = int(np.argmin(np.abs(sims - similarity_input)))

        dest_id = emb_svc.get_id_at(chosen_idx)
        dest = self.world.get_by_id(dest_id)

        return {
            'mode':             'derive',
            'similarity_input': similarity_input,
            'sim_min':          round(float(sims.min()), 4),
            'sim_max':          round(float(sims.max()), 4),
            'destination':      self._format_result(dest),
            **self._centroid_meta(position, sims[chosen_idx], layer),
        }

    def detourn(self, current_ids, target_year, layer='audio'):
        """
        Transport the user to the centroid of a target year's songs.
        """
        emb_svc = self.emb_svc
        matrix = emb_svc.get_embeddings_matrix(layer)

        # Find closest available year if exact year not in dataset
        available_years = list(set(emb_svc.years))
        if target_year in available_years:
            year = target_year
        else:
            year = min(available_years, key=lambda y: abs(y - target_year))

        new_position = self.get_year_centroid(year, layer)

        # Resolve centroid to nearest track
        sims = cosine_similarity([new_position], matrix)[0]
        dest_idx = int(np.argmax(sims))
        dest_id = emb_svc.get_id_at(dest_idx)
        dest = self.world.get_by_id(dest_id)

        return {
            'mode':        'detourn',
            'target_year': target_year,
            'destination': self._format_result(dest),
            **self._centroid_meta(new_position, sims[dest_idx], layer),
        }

    def stroll(self, current_ids, layer='audio'):
        """
        Stroll to a random song that is not too similar to the current position.
        """
        emb_svc = self.emb_svc

        # Establish centroid as position from current track IDs
        indices = [emb_svc.get_idx(id) for id in current_ids if emb_svc.get_idx(id) is not None]
        matrix = emb_svc.get_embeddings_matrix(layer)
        position = np.mean(matrix[indices], axis=0)

        random_idx = np.random.randint(len(emb_svc.ids))
        sim = cosine_similarity([position], [matrix[random_idx]])[0][0]

        attempts = 0
        while np.array_equal(matrix[random_idx], position) or sim > 0.9:
            if attempts >= 100:
                raise ValueError("Could not find a dissimilar enough song — try raising the similarity threshold.")
            random_idx = np.random.randint(len(emb_svc.ids))
            sim = cosine_similarity([position], [matrix[random_idx]])[0][0]
            attempts += 1

        dest_id = emb_svc.get_id_at(int(random_idx))
        dest = self.world.get_by_id(dest_id)

        return {
            'mode': 'stroll',
            'destination': self._format_result(dest),
            **self._centroid_meta(position, sim, layer),
        }
