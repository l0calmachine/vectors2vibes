# vectors2vibes

An immersive 3D world allowing users to explore the audio and lyrical embedding space of ~22k music tracks, made navigable as a 2D ground plane with UMAP reduction. Users are encouraged to explore the space unusually: dériving, détourning, and frolicing amongst cacophonical neighborhoods. What do these sonic neighborhoods reveal (or obscure) about machinic listening?

Built with **Three.js** (frontend) + **Python FastAPI** (backend).

---

## Quick Start

```bash
# 1. Clone the repo and install dependencies
pip install -r requirements.txt

# 2. Add your HF_TOKEN to a .env file in the project root
echo "HF_TOKEN=your_token_here" > .env

# 3. Run the server
uvicorn server:app --reload

# 4. Open in browser
# http://localhost:8000
```

On first run, the server downloads the parquet dataset from HuggingFace and runs UMAP (~3–5 minutes). Subsequent starts load from cache (~1–2 seconds).

---

## Project Structure

```
vectors2vibes/
├── server.py                        # Server entry point: FastAPI app + uvicorn
├── requirements.txt
├── .env                             # Add your HF_TOKEN here
│
├── static/
│   └── index.html                   # Three.js frontend
│
└── backend/
    ├── data/
    │   ├── 28-3-2026-both-emeddings.parquet   # Dataset (downloaded from HF on first run)
    │   └── umap_cache.pkl                      # UMAP output cache (delete to recompute UMAP)
    │
    ├── routers/
    │   ├── world.py                 # /api/world/*     — track metadata + visual layer positions
    │   ├── navigation.py            # /api/nav/*       — derive, detourn, frolic navigational styles
    │   ├── listener.py              # /api/listener/*  — behavior tracking; drives ghost centroid
    │   ├── audio.py                 # /api/audio/*     — HF audio proxy + LRU cache
    │   ├── thumbnails.py            # /api/thumb/*     — album thumbnail proxy + cache
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
| audio   | UMAP on 1024-dim audio embeddings — tracks grouped by sound     |
| lyrical | UMAP on 384-dim lyric embeddings — tracks grouped by semantics  |
| year    | Chronological — tracks arranged by release year                 |

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

Audio streams from a private HuggingFace repo via a server proxy (`/api/audio/stream/{track_id}`), which keeps the HF token secure. Tracks within `AUDIO_R` world units of the user start playing, with volume increasing or decreasing by distance. Up to `MAX_AUDIO_ELS` tracks can play simultaneously — when the cap is reached, the quietest track is dropped to avoid overloading the server with too many simultaneous streams.

Thumbnails are streamed from a public HuggingFace repo, also via a server proxy. Both audio and thumbnails are cached (LRU cache) to avoid re-fetching and incurring additional buffer time.

---

## Ghost Centroid

As the user navigates around the embedding space, the server builds a **ghost centroid:** a weighted average of the user's listening, navigational, and visual layer behavior. This centroid is visualized as a billboard node that drifts through the world, displaying the user's weighted average coordinates and the average year of the user's listening history.

The centroid's coordinates shift between audio and lyrical embedding space depending on which layer the user spends more time on.

---

## Dataset

- **Source:** `vectors2vibes/28-3-26-both-emeddings` (HuggingFace, public)
- **Tracks:** ~22k unique tracks sourced from Discogs dumps
- **Audio embeddings:** 1024-dim (from audio files)
- **Lyric embeddings:** 384-dim (from lyrical transcripts)
- **Audio files:** `vectors2vibes/vectors2vibes-discogs-audio` (HuggingFace, private)

---

## Controls

| Input              | Action                        |
|--------------------|-------------------------------|
| WASD / Arrow keys  | Move through the world        |
| Mouse (click+drag) | Look around (pointer lock)    |
| [1]                | dérive                        |
| [2]                | détourn                       |
| [3]                | frolic                        |
| Layer buttons      | Switch between audio/lyrical/year layouts |

---

## Known Issues / Pending Work

- **Lag** — primary causes are too many audio streams and per-frame GLSL shader processing. Tune `AUDIO_R`, `MAX_AUDIO_ELS`, `SPAWN_R`, and `MAX_PLANES` in `index.html` to reduce load. Long-term fix: preprocess thumbnails (removing real-time shader), and implement instanced rendering.
- **Minimap** — shows all 12,620 tracks as a static offscreen canvas, redrawn on layer switch. Could be improved with a server-generated image for better performance.
- **Thumbnails** — Plan: preprocess and host on HuggingFace as pre-filtered WebP images to remove real-time shader and reduce lag.
