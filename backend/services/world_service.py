"""
world_service.py — provides frontend with track metadata and world coordinates.

Gets all embedding and coordinate data from EmbeddingService using WorldService emb_svc.get_world_pos().
Coordinates (x,z) are UMAP-derived by EmbeddingService from audio/lyrical embeddings.

Visual layers:
  'audio'   → UMAP on audio_embedding  (1024-dim)
  'lyrical' → UMAP on lyric_embedding  (384-dim)
  'year'    → chronological spiral (metadata-based, no embedding)

"""

import ast
import math
import pandas as pd
from pathlib import Path

from backend.services.embedding_service import get_embedding_service

PARQUET_PATH = Path(__file__).parent.parent / "data" / "master_dataset.parquet"

class WorldService:
    def __init__(self):
        self._emb_svc = get_embedding_service()
        self._tracks = {}
        self._track_list = []
        self._build_tracks()

    # ── Build track metadata ───────────────────────────────────────────────

    def _build_tracks(self):
        """Build track metadata from parquet via EmbeddingService."""
        self._build_from_parquet(self._emb_svc)

    def _build_from_parquet(self, emb_svc):
        """Build track list from parquet, resolving world coordinates from EmbeddingService."""
        if not PARQUET_PATH.exists():
            print("[WorldService] Parquet cache not found.")
            return

        df = pd.read_parquet(PARQUET_PATH)
        df = df.drop_duplicates(subset='id', keep='first').reset_index(drop=True)

        # earliest_release NaN value warning (remove after final dataset is foolproof)
        n_before = len(df)
        df = df.dropna(subset=['earliest_release'])
        n_after = len(df)
        if n_before != n_after:
            print(f"[WorldService] Dropped {n_before - n_after} rows with NaN earliest_release — fix upstream")

        def parse_list_str(val, fallback='Unknown'):
            """Parse plain strings or lists into a comma-separated string."""
            if isinstance(val, list):
                return ', '.join(str(v) for v in val)
            try:
                result = ast.literal_eval(str(val))
                if isinstance(result, list):
                    return ', '.join(str(v) for v in result)
                return str(result)
            except Exception:
                return str(val).strip("[]'\"") or fallback

        for i, row in df.iterrows():
            track_id = str(row.get('id', ''))
            if not track_id:
                continue

            idx = emb_svc.get_idx(track_id)
            if idx is None:
                continue
            pos_x, pos_z = emb_svc.get_world_pos(idx, layer='audio')

            artist = parse_list_str(row.get('release_artist_names', ''), 'Unknown')

            genre_raw = row.get('release_genres', '')
            try:
                genre_list = ast.literal_eval(str(genre_raw))
                genre = genre_list[0] if isinstance(genre_list, list) and genre_list else str(genre_raw)
            except Exception:
                genre = str(genre_raw).strip("[]'\"")

            year = int(float(row.get('earliest_release', 0) or 0))
            duration = float(row.get('duration', 0) or 0)

            track = {
                'id':            track_id,
                'title':         str(row.get('track_title', f'Track {i}')),
                'artist':        artist,
                'year':          year,
                'lyrics':        str(row.get('lyrics', '')),
                'duration':      duration,
                'youtube_url':   str(row.get('webpage_url', f'https://www.youtube.com/watch?v={track_id}')),
                'thumbnail_url': f'/api/thumb/{track_id}',
                'file_path':     str(row.get('file_path', '')),
                'genre':         genre,
                'audio_coords':  [round(float(pos_x), 1), round(float(pos_z), 1)],
                'lyric_coords':  [round(float(emb_svc.lyric_coords[idx, 0]), 1), round(float(emb_svc.lyric_coords[idx, 1]), 1)],
                'pos_x':         pos_x,
                'pos_z':         pos_z,
            }
            self._tracks[track_id] = track
            self._track_list.append(track)

        print(f"[WorldService] Built {len(self._track_list)} tracks from parquet.")

    # ── Layer positions ────────────────────────────────────────────────────

    def get_layer_positions(self, layer):
        """Return {track_id: {pos_x, pos_z}} for all tracks in the requested layer."""
        emb_svc = self._emb_svc

        if layer == 'audio' and emb_svc.is_loaded():
            return emb_svc.get_all_world_positions(layer='audio')

        if layer == 'lyrical' and emb_svc.is_loaded():
            return emb_svc.get_all_world_positions(layer='lyrical')

        if layer == 'year':
            return self._year_spiral_positions()

        return {t['id']: {'pos_x': t['pos_x'], 'pos_z': t['pos_z']}
                for t in self._track_list}

    def _year_spiral_positions(self):
        """Return coordinates for the year layer — tracks arranged chronologically in a spiral."""
        sorted_tracks = sorted(self._track_list, key=lambda t: t['year'])
        n = len(sorted_tracks)
        print(f"[year spiral] {n} tracks, max radius: {1000}")
        positions = {}
        for i, t in enumerate(sorted_tracks):
            angle  = i * 0.3
            radius = 50 + (i / n) * 950 # scaled
            if i < 3: # print for debug
                print(f"[year spiral] i={i}, radius={radius}, pos_x={round(math.cos(angle) * radius, 2)}")
            positions[t['id']] = {
                'pos_x': round(math.cos(angle) * radius, 2),
                'pos_z': round(math.sin(angle) * radius, 2),
            }
        return positions

    # ── Public API ─────────────────────────────────────────────────────────

    def get_embeddings_page(self, page, page_size):
        """Return a paginated slice of the track list with all fields the frontend needs."""
        start  = page * page_size
        slice_ = self._track_list[start:start + page_size]
        return {
            'total': len(self._track_list),
            'page':  page,
            'items': [
                {
                    'id':             t['id'],
                    'title':          t['title'],
                    'artist':         t['artist'],
                    'year':           t['year'],
                    'duration':       t['duration'],
                    'thumbnail_url':  t['thumbnail_url'],
                    'audio_url':      self._resolve_audio(t),
                    'pos_x':          t['pos_x'],
                    'pos_z':          t['pos_z'],
                    'audio_coords':   t['audio_coords'],
                    'lyric_coords':   t['lyric_coords'],
                    'lyrics':         t.get('lyrics'),
                }
                for t in slice_
            ]
        }

    def get_embedding_detail(self, track_id):
        """Return the full track dict for a single track ID."""
        t = self._tracks.get(track_id)
        if not t:
            return {'error': 'not found'}
        return {**t, 'audio_url': self._resolve_audio(t)}

    def get_all(self):
        """Return the full track list."""
        return self._track_list

    def get_by_id(self, track_id):
        """Return a single track dict for a given ID, or returns None if not found."""
        return self._tracks.get(track_id)

    def _resolve_audio(self, track):
        """Return the audio stream URL for a track."""
        return f"/api/audio/stream/{track['id']}"