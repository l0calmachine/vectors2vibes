# vectors2vibes

An immersive 3D world allowing users to explore the audio and lyrical embedding space of ~24k music tracks, made navigable as a 2D ground plane with UMAP reduction. Users are encouraged to explore the space unusually: dériving, détourning, and frolicing amongst cacophonical neighborhoods. What do these sonic neighborhoods reveal (or obscure) about machinic listening?

Built with **Three.js** (frontend) + **Python FastAPI** (backend).

---

## Quick Start

```bash
# 1. Clone the repo and install dependencies
pip install -r requirements.txt

# 2. Download assets (~82GB)
python scripts/download_assets.py

# 3. Add your HF_TOKEN and local directory paths to a .env file in the project root
cat > .env << 'EOF'
HF_TOKEN=your_token_here
LOCAL_AUDIO_DIR=/path/to/vectors2vibes/backend/data/audio
LOCAL_THUMB_DIR=/path/to/vectors2vibes/backend/data/thumbnails
EOF

# 4. Run the server
uvicorn server:app --reload

# 5. Open in browser
# http://localhost:8000
```

On first run, the server runs UMAP reduction (~45 minutes). Subsequent starts load from cache (~1–2 seconds).

**Asset serving modes** (set in `.env`, first match wins):

| Mode | Env var | Notes |
|------|---------|-------|
| Local disk | `LOCAL_AUDIO_DIR`, `LOCAL_THUMB_DIR` | Fastest; run `scripts/download_assets.py` first |
| Raspberry Pi | `PI_BASE_URL=http://<pi-ip>:8080` | Browser streams directly from Pi via 302 redirect |
| HuggingFace proxy | `HF_TOKEN` | Default fallback; server proxies from HF, token stays server-side |

---

## Project Structure

```
vectors2vibes/
├── server.py                        # Server entry point: FastAPI app + uvicorn
├── requirements.txt
├── .env                             # Add your HF_TOKEN and local asset paths here
│
├── scripts/
│   ├── download_assets.py           # Download audio + thumbnails from HF
│   └── tune_umap_neighbors.py       # UMAP n_neighbors hyperparameter search
|
├── static/
│   └── index.html                   # Three.js frontend
│
└── backend/
    ├── data/
    |   ├── audio/                   # Audio directory created with download_assets.py 
    |       └── 1f/                  # Sharded audio directories (downloaded from HF with download_assets.py)
    |           └── 1f8T3c_oPfA.ogg
    |   ├── thumbnails/              # Thumbnail directory created with download_assets.py 
    |       └── 1f/                  # Sharded thumbnail directories (downloaded from HF with download_assets.py)
    |           └── 1f8T3c_oPfA.webp
    │   ├── master_dataset.parquet   # Dataset (downloaded from HF on first server run)
    │   └── umap_cache.pkl           # UMAP output cache (delete to recompute UMAP)
    │
    ├── routers/
    │   ├── world.py                 # /api/world/*     — track metadata + visual layer positions
    │   ├── navigation.py            # /api/nav/*       — derive, detourn, frolic navigational styles
    │   ├── listener.py              # /api/listener/*  — behavior tracking; drives ghost centroid
    │   ├── audio.py                 # /api/audio/*     — local / Pi / HF proxy serving
    │   ├── thumbnails.py            # /api/thumb/*     — local / Pi / HF proxy serving
    │   └── spawn.py                 # /api/spawn/*     — nostalgia spawn
    │
    └── services/
        ├── embedding_service.py     # Loads parquet, runs UMAP, holds all embedding data
        ├── world_service.py         # Builds track metadata store, serves coordinates
        ├── navigation_service.py    # Navigation logic (derive, detourn, frolic)
        ├── spawn_service.py         # Spawn logic (nostalgia)
        └── listener_service.py      # Session behavior tracking + ghost centroid
```

---

## How the Server Works

The server runs entirely in Python. **Uvicorn** listens for incoming HTTP requests on port 8000 and passes them to **FastAPI**, which sends each request to the appropriate router and returns the response back to Uvicorn for serving to the browser.

Python was chosen rather than Node.js because the backend uses Python heavily (i.e., UMAP dimensionality reduction, cosine similarity calculations, numpy embedding matrices, parquet file parsing). Running the server in Python means the data processing and the API can live in the same runtime without having to translate between languages.

---

## Visual Layers

The same tracks can be viewed in three spatial layouts, switchable via the visual layer buttons in the UI:

| Layer   | Layout                                                          |
|---------|-----------------------------------------------------------------|
| audio    | UMAP on 1024-dim audio embeddings — tracks grouped by sound              |
| lyrical  | UMAP on 384-dim lyric embeddings — tracks grouped by semantics           |
| combined | UMAP on 512-dim combined embeddings — blend of audio + lyrical structure |

Switching layers animates all planes from their old positions to their new ones.

---

## Navigation Modes

All intra-layer navigation operates on the **centroid of the 12 nearest tracks** to the user rather than a single track position. This gives navigation a more "vibey" sense of where a user is in the embedding space, rather than focusing on individual tracks.

| Mode      | Key | What it does                                                                           |
|-----------|-----|----------------------------------------------------------------------------------------|
| dérive    | [1] | Opens a year input panel. Drifts toward the audio centroid of that era by one step.    |
| détourn   | [2] | Reroutes to the track most dissimilar to the user's current position.                  |
| frolic    | [3] | Jumps to a random track sufficiently far enough away from the user's current position. |

Navigation functions are adapted from `Navigation_v2.ipynb` — see `navigation_service.py` for a full description of what changed from the notebook originals.

---

## Spawning

On load, the user enters their birth year. The server calculates `birth_year + 15` (the "nostalgia year" based on reminiscence bump) and finds the 12 nearest tracks to that era's audio embedding centroid, then picks one at random as the spawn destination.

---

## Audio

Tracks within `AUDIO_R` world units of the user start playing, with volume increasing or decreasing by distance. Up to `MAX_AUDIO_ELS` tracks can play simultaneously — when the cap is reached, the quietest track is dropped.

Audio and thumbnails are served in one of three modes (see Quick Start for configuration). In HF proxy mode, both are cached server-side (LRU) to reduce repeat fetch latency.

---

## Ghost Centroid

As the user navigates around the embedding space, the server builds a **ghost centroid:** a weighted average of the user's listening, navigational, and visual layer behavior. This centroid is visualized as a billboard node that drifts through the world, displaying the user's weighted average coordinates and the average year of the user's listening history.

The centroid's coordinates shift between audio and lyrical embedding space depending on which layer the user spends more time on.

---

## Dataset

- **Source:** `vectors2vibes/vectors2vibes-discogs-metadata` (HuggingFace, public)
- **Tracks:** ~24k unique tracks sourced from Discogs dumps
- **Audio embeddings:** 1024-dim (from audio files)
- **Lyric embeddings:** 384-dim (from transcribed lyrics)
- **Combined embeddings:** 512-dim (audio + lyrical blend)
- **Audio files:** `vectors2vibes/vectors2vibes-discogs-audio` (HuggingFace, private)
- **Thumbnails:** `vectors2vibes/vectors2vibes-preprocessed-thumbnails` (HuggingFace, public)

---

## Controls

| Input              | Action                        |
|--------------------|-------------------------------|
| WASD / Arrow keys  | Move through the world        |
| Mouse (click+drag) | Look around (pointer lock)    |
| [1]                | dérive                        |
| [2]                | détourn                       |
| [3]                | frolic                        |
| Layer buttons      | Switch between audio/lyrical/combined layouts |

---

## Known Issues / Pending Work

- **Pi external access**: Caddy static server runs on the Pi but may require firewall/router configuration to be reachable from outside the local network.
- **Audio rsync to Pi**: ~82GB transfer — run `rsync -avz --progress backend/data/audio/ pi:/mnt/hdd/vectors2vibes/audio/` when ready.
