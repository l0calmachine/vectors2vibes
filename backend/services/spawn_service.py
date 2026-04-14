"""
spawn_service.py — spawning functions written from Greg's Navigation_v2.ipynb, adapted for the UI.

HOW THIS DIFFERS FROM THE NOTEBOOK
------------------------------------
The notebook operated on a pandas DataFrame (df) and a stacked numpy embeddings array
passed as function arguments. In the service layer (~/vectors2vibes/backend/services)
these are replaced by EmbeddingService (defined in embedding_service.py),
which loads the parquet data once at startup, caches it into memory, and
holds its attributes (i.e., audio_embs, years, genres, etc.) so that any service
can access them directly via the function get_embedding_service().
Specific changes (changes not specified here are commented inline):

  df["year"]              → emb_svc.years
  embeddings              → emb_svc.audio_embs
  user_params["birth_year"]       → birth_year argument
  return position (embedding vec) → return closest track dict via _format()

neighbourhood_songs() in the notebook returned a DataFrame slice for inspection.
Now it returns a list of track IDs, which is what the service layer operates off of.

spawn_nostalgia() and spawn_nostalgia_genre() had a bug in the OG notebook
which picked a random place (0-11) from the entire dataset (embeddings):
  position_idx = np.random.randint(0, len(neighborhood_songs(...)))
  position = embeddings[position_idx]
The fix uses rnd.choice(neighbours) to pick one ID (0-11) directly
from the list of neighboring (12) track IDs.

_format() was created as a helper function. It returns the track ID dicts, which the
service layer uses to populate metadata and coordinates to the frontend.

Omitted get_genre_centroid() support function and spawn_nostalgia_genre() and spawn_random()
functions as we narrowed down to one spawning option (spawn_nostalgia()).

"""

import random as rnd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.services.embedding_service import get_embedding_service
from backend.services.world_service import WorldService


class SpawnService:
    def __init__(self):
        self.emb_svc = get_embedding_service()
        self.world   = WorldService()

    # ── Support functions ─────────────────────────────────────────────────

    def get_year_centroid(self, target_year):
        """Get the mean embedding vector for all songs from a given year."""
        emb_svc = self.emb_svc
        indices = [i for i, y in enumerate(emb_svc.years) if y == target_year]
        return np.mean(emb_svc.audio_embs[indices], axis=0)

    def neighbourhood_songs(self, position, k=10):
        """Return the k nearest songs to a position."""
        # Returns track IDs instead of a DataFrame slice — service layer doesn't use df
        sims = cosine_similarity([position], self.emb_svc.audio_embs)[0]
        top_k = np.argsort(sims)[::-1][:k]
        return [self.emb_svc.get_id_at(int(i)) for i in top_k]

    def _format(self, track_id):
        t = self.world.get_by_id(track_id)
        if not t:
            return {'error': 'not found'}
        return {
            'id':            t['id'],
            'title':         t['title'],
            'artist':        t['artist'],
            'year':          t['year'],
            'pos_x':         t['pos_x'],
            'pos_z':         t['pos_z'],
            'thumbnail_url': t['thumbnail_url'],
            'audio_url':     f"/api/audio/stream/{t['id']}",
        }

    # ── Spawn options ─────────────────────────────────────────────────────

    def spawn_nostalgia(self, birth_year):
        """Spawn at the centroid of the user's nostalgia year."""
        emb_svc = self.emb_svc

        nostalgia_year = birth_year + 15
        year_window    = 2
        # Notebook used a DataFrame and pandas methods to find the nostalgia_year.
        # Converted to a Python operation, compatible with the service layer.
        available      = set(emb_svc.years)
        mask = any(abs(y - nostalgia_year) <= year_window for y in available)
        if mask:
            year = nostalgia_year
        else:
            year = min(available, key=lambda y: abs(y - nostalgia_year))

        # Notebook picked a random index into len(neighbours), not neighbours itself.
        position  = self.get_year_centroid(year)
        neighbours = self.neighbourhood_songs(position, k=12)
        dest_id   = rnd.choice(neighbours)

        # Added print for debugging
        print(f"[spawn] birth_year={birth_year}, nostalgia_year={nostalgia_year}, selected_year={year}, dest_id={dest_id}")
        return {
            'spawn_mode':     'nostalgia',
            'nostalgia_year': int(year),
            'destination':    self._format(dest_id),
        }
