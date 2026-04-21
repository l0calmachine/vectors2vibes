"""
download_assets.py — download audio and thumbnail HuggingFace datasets for local serving.

Run once before starting the server:
    python scripts/download_assets.py

Requires HF_TOKEN in your .env or environment.
Downloads to:
    backend/data/audio/       (vectors2vibes-discogs-audio dataset)
    backend/data/thumbnails/  (vectors2vibes-preprocessed-thumbnails dataset)

Disk space: ~81.3GB audio + ~613MB thumbnails

Re-running is safe — huggingface_hub skips files that are already complete.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set. Add it to your .env file.")
    sys.exit(1)

if os.environ.get("LOCAL_AUDIO_DIR"):
    AUDIO_DIR = Path(os.environ["LOCAL_AUDIO_DIR"]) / "data/audio"
else:
    Path(__file__).parent.parent / "backend" / "data" / "audio"

if os.environ.get("LOCAL_THUMB_DIR"):
    THUMB_DIR = Path(os.environ["LOCAL_THUMB_DIR"]) / "data/thumb"
else:
    Path(__file__).parent.parent / "backend" / "data" / "thumb"

AUDIO_REPO = "vectors2vibes/vectors2vibes-discogs-audio"
THUMB_REPO  = "vectors2vibes/vectors2vibes-preprocessed-thumbnails"


def download(repo_id, local_dir, description):
    print(f"\n[download] {description}")
    print(f"  repo : {repo_id}")
    print(f"  dest : {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=HF_TOKEN,
        ignore_patterns=["*.md", "*.gitattributes", ".gitattributes"],
    )
    print(f"[download] Done: {local_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download vectors2vibes assets from HuggingFace.")
    parser.add_argument("--only", choices=["audio", "thumbnails"], help="Download only one asset type (default: both)")
    args = parser.parse_args()

    if args.only != "thumbnails":
        download(AUDIO_REPO, AUDIO_DIR, "audio files (~81.3GB, takes a while)")
    if args.only != "audio":
        download(THUMB_REPO, THUMB_DIR, "thumbnail images (~613MB)")

    print("\nDone.")
    print("Set these env vars before starting the server:")
    print(f"  LOCAL_AUDIO_DIR={AUDIO_DIR}")
    print(f"  LOCAL_THUMB_DIR={THUMB_DIR}")
