"""
vectors2vibes — server entry point
Run with: uvicorn server:app --reload

HOW THE SERVER WORKS
---------------------
Uvicorn is the network layer that listens for incoming HTTP requests on port 8000.
It passes them to FastAPI, which matches each request to the appropriate router,
runs it, and sends the response back to Uvicorn for serving to the browser.

WHY THIS SETUP INSTEAD OF JS
-----------------------------
Uvicorn and FastAPI are both Python libraries, meaning they are compatible with
backend processes (i.e., UMAP dimensionality reduction, cosine similarity
calculations, numpy embedding matrices, parquet file parsing) which also depend
on Python libraries (umap-learn, sklearn, numpy, pandas). Running the server in
Python means the data processing and the API can live in the same runtime without
needing translation between Python and JS.
"""

# Load .env first
from dotenv import load_dotenv
load_dotenv()

# Find backend package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.routers import world, navigation, listener, audio, thumbnails, spawn

app = FastAPI(title="vectors2vibes")

# ── Routers ────────────
app.include_router(world.router,       prefix="/api/world")
app.include_router(navigation.router,  prefix="/api/nav")
app.include_router(listener.router,    prefix="/api/listener")
app.include_router(audio.router,       prefix="/api/audio")
app.include_router(thumbnails.router,  prefix="/api/thumb")
app.include_router(spawn.router,       prefix="/api/spawn")

# ── Serve frontend ────────────
# Serve static files (fonts, etc.) from /static
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve index.html at the root URL
@app.get("/")
def root():
    return FileResponse("static/index.html")
