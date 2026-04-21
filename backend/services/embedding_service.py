"""
embedding_service.py — central embedding store for vectors2vibes.

What this service does, in order:
  1. Downloads master_dataset.parquet from HF repo: vectors2vibes/vectors2vibes-discogs-metadata (token needed)
  2. Deduplicates on _id, keeping first occurrence
  3. Stacks audio_embedding (1024-dim), lyric_embeddings (384-dim), and combined_embedding (512-dim) into numpy arrays
  4. Runs UMAP separately on each to produce 2D world coordinates.
     World is scaled [-1000,1000] with WORLD_SCALE (prioritizing usability over raw UMAP representation).
  5. Caches both UMAP results to disk so restarts are fast

UMAP parameters explained:
  n_neighbors=10-20  — local neighbourhood size (calculated based on trustworthiness/continuity scores derived from tune_umap_neighbors.py)
  min_dist=0.1    — how tightly clusters pack (lower = denser clusters)
  metric='cosine' — matches the cosine similarity used in navigation functions.
                    spatial layout and navigation should use the same metric.
  n_components=2  — output 2D coordinates for the world x/z plane.
  random_state    — none; removed to avoid deterministic UMAP output.
                    geography changes when cache is cleared.

Startup time:
  First run  — ~3–5 minutes (UMAP fit)
  Subsequent — ~1–2 seconds (load from cache pickle)

get_embedding_service() returns a singleton instance shared amongst the service layer.

"""

import os
import io
import pickle
import ast
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET_URL = "https://huggingface.co/datasets/vectors2vibes/vectors2vibes-discogs-metadata/resolve/main/data/master_dataset.parquet"

CACHE_PATH    = Path(__file__).parent.parent / "data" / "umap_cache.pkl"
PARQUET_CACHE = Path(__file__).parent.parent / "data" / "master_dataset.parquet"

WORLD_SCALE = 1000.0

# n_neighbors defines the local neighbourhood scale for each embedding type.
# Matched by the JS LAYER_K constant so navigation and display use the same k.
UMAP_N_NEIGHBORS_AUDIO    = 10
UMAP_N_NEIGHBORS_LYRIC    = 20
UMAP_N_NEIGHBORS_COMBINED = 10


# ── UMAP coordinate scaling ───────────────────────────────────────────────────

def _scale_to_world(coords):
    """Normalise UMAP output to [-WORLD_SCALE, WORLD_SCALE] on both axes."""
    result = coords.copy().astype(np.float32)
    for axis in range(2):
        mn = result[:, axis].min()
        mx = result[:, axis].max()
        rng = mx - mn if mx != mn else 1.0
        result[:, axis] = (result[:, axis] - mn) / rng * 2 * WORLD_SCALE - WORLD_SCALE
    return result


def _apply_scale(raw_point, raw_min, raw_max):
    """Apply the same linear transform used by _scale_to_world to a single 2D point."""
    result = np.zeros(2, dtype=np.float32)
    for axis in range(2):
        rng = raw_max[axis] - raw_min[axis] if raw_max[axis] != raw_min[axis] else 1.0
        result[axis] = (raw_point[axis] - raw_min[axis]) / rng * 2 * WORLD_SCALE - WORLD_SCALE
    return result


# ── Main service class ────────────────────────────────────────────────────────

class EmbeddingService:
    def __init__(self):
        """Initialise empty attribute arrays and trigger parquet load and UMAP computation."""
        # Parallel arrays — all indexed by integer position i
        self.ids = [] # track ID (YouTube ID)
        self.years = [] # earliest release year
        self.genres = [] # genre list
        self.file_paths = [] # HF file path

        self.audio_embs    = np.array([]) # (N, 1024)
        self.lyric_embs    = np.array([]) # (N, 384)
        self.combined_embs = np.array([]) # (N, 512)

        # UMAP-derived 2D positions, shape (N, 2), scaled to world units
        self.audio_coords    = np.array([])
        self.lyric_coords    = np.array([])
        self.combined_coords = np.array([])

        # Fitted UMAP reducers and raw output ranges — used by project_to_world()
        self._reducer_audio      = None
        self._reducer_lyric      = None
        self._reducer_combined   = None
        self._audio_raw_range    = None
        self._lyric_raw_range    = None
        self._combined_raw_range = None

        self._id_to_idx = {} # track ID maps to index lookup
        self._loaded = False

        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        """Load parquet, stack embeddings, run UMAP, and store all data as class attributes."""
        print("[EmbeddingService] Starting load...")

        df = self._fetch_parquet()
        if df is None or len(df) == 0:
            print("[EmbeddingService] Failed to load parquet.")
            return

        # ── Step 1: Use 'id' column as primary lookup key
        df['_id'] = df['id'].astype(str)

        # ── Step 2: Deduplicate on 'id' (parquet should already be clean, this is a safety net)
        n_before = len(df)
        df = df.drop_duplicates(subset='_id', keep='first').reset_index(drop=True)
        n_after = len(df)
        if n_before != n_after:
            print(f"[EmbeddingService] Deduplicated: {n_before} → {n_after} rows")

        # ── Step 3: Stack embedding columns into parallel numpy matrices
        print(f"[EmbeddingService] Stacking {n_after} embeddings...")
        try:
            audio_embs    = np.stack(df['audio_embedding'].tolist()).astype(np.float32)
            lyric_embs    = np.stack(df['lyric_embeddings'].tolist()).astype(np.float32)
            combined_embs = np.stack(df['combined_embedding'].tolist()).astype(np.float32)
        except Exception as e:
            print(f"[EmbeddingService] Failed to stack embeddings: {e}")
            return

        # ── Step 4: Parse metadata
        ids = df['_id'].tolist()
        file_paths = df['file_path'].tolist()

        # Year uses 'earliest_release' (enriched from MusicBrainz; more reliable than Discogs release year)
        year_col = 'earliest_release' if 'earliest_release' in df.columns else 'year'
        years    = df[year_col].fillna(0).astype(float).tolist()

        def parse_genres(val):
            """Parse a genre value into a Python list: "Jazz" → ["Jazz"]"""
            if isinstance(val, list):
                return val
            if not val or str(val) in ('nan', 'None', ''):
                return []
            try:
                result = ast.literal_eval(str(val))
                return result if isinstance(result, list) else [str(result)]
            except Exception:
                return [str(val)]

        genre_col = 'release_genres' if 'release_genres' in df.columns else \
                    'genre' if 'genre' in df.columns else None
        genres = df[genre_col].apply(parse_genres).tolist() if genre_col else \
                 [[] for _ in range(n_after)]

        # ── Step 5: Run UMAP or load from cache
        audio_coords, lyric_coords, combined_coords = self._get_umap_coords(audio_embs, lyric_embs, combined_embs, n_after)

        # ── Step 6: Store everything as parallel arrays indexed by integer position
        self.ids             = ids
        self.years           = years
        self.genres          = genres
        self.file_paths      = file_paths
        self.audio_embs      = audio_embs
        self.lyric_embs      = lyric_embs
        self.combined_embs   = combined_embs
        self.audio_coords    = audio_coords
        self.lyric_coords    = lyric_coords
        self.combined_coords = combined_coords
        self._id_to_idx   = {eid: i for i, eid in enumerate(ids)}
        self._loaded      = True

        print(f"[EmbeddingService] Ready. {n_after} tracks loaded.")
        print(f"  Audio embeddings: {audio_embs.shape}")
        print(f"  Lyric embeddings: {lyric_embs.shape}")
        print(f"  Combined embeddings: {combined_embs.shape}")
        print(f"  World coords range X: [{audio_coords[:,0].min():.0f}, {audio_coords[:,0].max():.0f}]")

    def _fetch_parquet(self):
        """Load parquet from local cache if available, otherwise download from HuggingFace."""
        # Try local cache first
        if PARQUET_CACHE.exists():
            print(f"[EmbeddingService] Loading parquet from local cache...")
            try:
                return pd.read_parquet(PARQUET_CACHE)
            except Exception as e:
                print(f"[EmbeddingService] Local cache read failed: {e}, re-downloading...")

        # Download from HuggingFace
        print(f"[EmbeddingService] Downloading parquet from HuggingFace...")
        try:
            import httpx
            PARQUET_CACHE.parent.mkdir(parents=True, exist_ok=True)

            with httpx.Client(timeout=300.0, follow_redirects=True) as client:
                headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"}
                r = client.get(PARQUET_URL, headers=headers)
                r.raise_for_status()
                PARQUET_CACHE.write_bytes(r.content)
                print(f"[EmbeddingService] Parquet saved to {PARQUET_CACHE}")
                return pd.read_parquet(io.BytesIO(r.content))
        except Exception as e:
            print(f"[EmbeddingService] Download failed: {e}")
            return None

    def _get_umap_coords(self, audio_embs, lyric_embs, combined_embs, n):
        """Run UMAP on audio, lyrical, and combined embeddings and return scaled 2D world coordinates. Loads from cache if available."""
        # Cache key encodes n_neighbors so changing them forces a recompute
        cache_key = f"n={n}_audio_nn={UMAP_N_NEIGHBORS_AUDIO}_lyric_nn={UMAP_N_NEIGHBORS_LYRIC}_combined_nn={UMAP_N_NEIGHBORS_COMBINED}"

        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('key') == cache_key and 'reducer_combined' in cached:
                    print("[EmbeddingService] UMAP coords loaded from cache.")
                    self._reducer_audio      = cached['reducer_audio']
                    self._reducer_lyric      = cached['reducer_lyric']
                    self._reducer_combined   = cached['reducer_combined']
                    self._audio_raw_range    = cached['audio_raw_range']
                    self._lyric_raw_range    = cached['lyric_raw_range']
                    self._combined_raw_range = cached['combined_raw_range']
                    return cached['audio_coords'], cached['lyric_coords'], cached['combined_coords']
                else:
                    print(f"[EmbeddingService] Cache miss ({cached.get('key')}), recomputing...")
            except Exception as e:
                print(f"[EmbeddingService] Cache load failed: {e}, recomputing...")

        # ── Compute UMAP ──────────────────────────────────────────────────
        try:
            import umap as umap_lib
        except ImportError:
            print("[EmbeddingService] umap-learn not installed. Run: pip install umap-learn")
            return np.array([]), np.array([])

        # ── Audio UMAP
        print("[EmbeddingService] Computing UMAP for audio embeddings (1024-dim)...")
        print("  This takes ~3–5 minutes on first run. Subsequent restarts use cache.")
        reducer_audio = umap_lib.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS_AUDIO,
            min_dist=0.1,
            metric='cosine',
            low_memory=True, # reduces RAM usage at slight speed cost to address lag
        )
        raw_audio = reducer_audio.fit_transform(audio_embs)
        audio_raw_range = {'min': raw_audio.min(axis=0), 'max': raw_audio.max(axis=0)}
        audio_coords = _scale_to_world(raw_audio)
        print(f"  Audio UMAP done. Range: x=[{raw_audio[:,0].min():.2f},{raw_audio[:,0].max():.2f}]")

        # ── Lyrical UMAP
        print("[EmbeddingService] Computing UMAP for lyrical embeddings (384-dim)...")
        reducer_lyric = umap_lib.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS_LYRIC,
            min_dist=0.1,
            metric='cosine',
            low_memory=True,
        )
        raw_lyric = reducer_lyric.fit_transform(lyric_embs)
        lyric_raw_range = {'min': raw_lyric.min(axis=0), 'max': raw_lyric.max(axis=0)}
        lyric_coords = _scale_to_world(raw_lyric)
        print(f"  Lyric UMAP done. Range: x=[{raw_lyric[:,0].min():.2f},{raw_lyric[:,0].max():.2f}]")

        # ── Combined UMAP
        print("[EmbeddingService] Computing UMAP for combined embeddings (512-dim)...")
        reducer_combined = umap_lib.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS_COMBINED,
            min_dist=0.1,
            metric='cosine',
            low_memory=True,
        )
        raw_combined = reducer_combined.fit_transform(combined_embs)
        combined_raw_range = {'min': raw_combined.min(axis=0), 'max': raw_combined.max(axis=0)}
        combined_coords = _scale_to_world(raw_combined)
        print(f"  Combined UMAP done. Range: x=[{raw_combined[:,0].min():.2f},{raw_combined[:,0].max():.2f}]")

        # ── Save reducers and scale params alongside coords
        self._reducer_audio      = reducer_audio
        self._reducer_lyric      = reducer_lyric
        self._reducer_combined   = reducer_combined
        self._audio_raw_range    = audio_raw_range
        self._lyric_raw_range    = lyric_raw_range
        self._combined_raw_range = combined_raw_range

        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, 'wb') as f:
                pickle.dump({
                    'key':               cache_key,
                    'audio_coords':      audio_coords,
                    'lyric_coords':      lyric_coords,
                    'combined_coords':   combined_coords,
                    'reducer_audio':     reducer_audio,
                    'reducer_lyric':     reducer_lyric,
                    'reducer_combined':  reducer_combined,
                    'audio_raw_range':   audio_raw_range,
                    'lyric_raw_range':   lyric_raw_range,
                    'combined_raw_range': combined_raw_range,
                }, f)
            print(f"[EmbeddingService] UMAP cache saved to {CACHE_PATH}")
        except Exception as e:
            print(f"[EmbeddingService] Cache save failed (non-fatal): {e}")

        return audio_coords, lyric_coords, combined_coords

    # ── Public API ────────────────────────────────────────────────────────────

    def is_loaded(self):
        """Return True if embeddings have been successfully loaded."""
        return self._loaded

    def get_idx(self, track_id):
        """Return the integer index for a track ID, or None if not found."""
        return self._id_to_idx.get(track_id)

    def get_id_at(self, idx):
        """Return the track ID at a given integer index."""
        return self.ids[idx]

    def get_embedding(self, track_id, layer='audio'):
        """Return the embedding vector for a track, for the given layer.
        Layer 'audio'    → audio_embs
        Layer 'lyrical'  → lyric_embs
        Layer 'combined' → combined_embs
        """
        idx = self.get_idx(track_id)
        if idx is None:
            return None
        if layer == 'lyrical':
            return self.lyric_embs[idx]
        if layer == 'combined':
            return self.combined_embs[idx]
        return self.audio_embs[idx]

    def get_embeddings_matrix(self, layer='audio'):
        """Return the full embedding matrix for the given layer."""
        if layer == 'lyrical':
            return self.lyric_embs
        if layer == 'combined':
            return self.combined_embs
        return self.audio_embs

    def get_world_pos(self, idx, layer='audio'):
        """Return (pos_x, pos_z) world coordinates for a track index and layer."""
        if layer == 'lyrical':
            coords = self.lyric_coords
        elif layer == 'combined':
            coords = self.combined_coords
        else:
            coords = self.audio_coords
        return float(coords[idx, 0]), float(coords[idx, 1])

    def project_to_world(self, emb_vector, layer='audio'):
        """Project a single embedding vector to world coordinates via the fitted UMAP model.
        Returns (pos_x, pos_z) scaled to world units, matching the existing track coordinates.
        Used to get the precise UMAP position of a mean embedding (e.g. neighbourhood centroid).
        """
        if layer == 'lyrical':
            reducer, raw_range = self._reducer_lyric, self._lyric_raw_range
        elif layer == 'combined':
            reducer, raw_range = self._reducer_combined, self._combined_raw_range
        else:
            reducer, raw_range = self._reducer_audio, self._audio_raw_range
        if reducer is None or raw_range is None:
            return 0.0, 0.0
        raw = reducer.transform(emb_vector.reshape(1, -1))[0]
        scaled = _apply_scale(raw, raw_range['min'], raw_range['max'])
        return float(scaled[0]), float(scaled[1])

    def get_all_world_positions(self, layer='audio'):
        """Return {track_id: {pos_x, pos_z}} for all tracks in a given layer."""
        if layer == 'lyrical':
            coords = self.lyric_coords
        elif layer == 'combined':
            coords = self.combined_coords
        else:
            coords = self.audio_coords
        return {
            self.ids[i]: {
                'pos_x': float(coords[i, 0]),
                'pos_z': float(coords[i, 1]),
            }
            for i in range(len(self.ids))
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance = None

def get_embedding_service():
    """Return the shared EmbeddingService singleton, creating it on first call."""
    global _instance
    if _instance is None:
        _instance = EmbeddingService()
    return _instance
