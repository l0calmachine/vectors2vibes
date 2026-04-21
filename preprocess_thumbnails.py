"""
preprocess_thumbnails.py — thumbnail gather and filter pipeline

STEP 1 — GATHER
    Fetches album art for every track from the Discogs API (master_id → release_id),
    falling back to the YouTube thumbnail URL stored in the dataset.
    Fully resumable — tracks with an image already saved are skipped.

    Run with:
        python preprocess_thumbnails.py --step gather

    Source:
        parquet  — master_dataset.parquet (~24,691 unique tracks)

    Output:
        ~/vectors2vibes/raw_thumbnails/{shard}/{id}.webp
        e.g. raw_thumbnails/sZ/sZSpQwL2nks.webp

STEP 2 — FILTER (apply_filter)
    Applies a computer vision filter pipeline to each raw thumbnail.
    See apply_filter() docstring for full pipeline description.

NOTES
    Required .env keys: DISCOGS_TOKEN, HF_TOKEN
"""

import argparse
import ast
import os
import time
import cv2
import hashlib
import numpy as np
from PIL import Image
import io
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / '.env')

DISCOGS_TOKEN = os.environ.get('DISCOGS_TOKEN', '')
HF_TOKEN      = os.environ.get('HF_TOKEN', '')

PARQUET_PATH  = Path(__file__).parent / 'backend' / 'data' / 'master_dataset.parquet'
OUT_DIR       = Path(__file__).parent / 'raw_thumbnails'

DISCOGS_RATE  = 55        # requests per minute (Discogs allows 60 authenticated)
SLEEP_BETWEEN = 60 / DISCOGS_RATE  # seconds between Discogs requests

DISCOGS_MASTER_URL  = "https://api.discogs.com/masters/{master_id}"
DISCOGS_RELEASE_URL = "https://api.discogs.com/releases/{release_id}"
DISCOGS_HEADERS     = {
    "Authorization": f"Discogs token={DISCOGS_TOKEN}",
    "User-Agent": "vectors2vibes/1.0 +https://github.com/vectors2vibes",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def shard(track_id):
    """Return the two-character shard prefix for a track ID."""
    return track_id[:2]


def output_path(track_id):
    """Return the expected output path for a track's thumbnail."""
    return OUT_DIR / shard(track_id) / f"{track_id}.webp"


def already_downloaded(track_id):
    """Return True if a thumbnail has already been saved for this track."""
    s = shard(track_id)
    return (OUT_DIR / s / f"{track_id}.webp").exists() or (OUT_DIR / s / f"{track_id}.jpg").exists()


def save_image(track_id, data):
    """Write image bytes to the correct shard directory. Returns True on success."""
    path = output_path(track_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def parse_id(val) -> str:
    """Safely parse master_id or release_id — may be int, float, or string."""
    try:
        v = str(val).strip()
        if v in ('', 'nan', 'None'):
            return ''
        return str(int(float(v)))
    except Exception:
        return ''



# ── STEP 1: Gather thumbnail images from Discogs API ──────────────────────────

def fetch_discogs_art_url(master_id: str, release_id: str, client: httpx.Client) -> str | None:
    """
    Try master_id first, fall back to release_id.
    Returns the primary image URL or None if not found.
    Rate-limited by caller.
    """
    for url_template, id_val in [
        (DISCOGS_MASTER_URL, master_id),
        (DISCOGS_RELEASE_URL, release_id),
    ]:
        if not id_val:
            continue
        url = url_template.format(master_id=id_val, release_id=id_val)
        try:
            r = client.get(url, headers=DISCOGS_HEADERS, timeout=15)
            if r.status_code == 429:
                # Rate limited — wait and retry once
                print("  [rate limit] sleeping 60s...")
                time.sleep(60)
                r = client.get(url, headers=DISCOGS_HEADERS, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json()
            images = data.get('images', [])
            if images:
                # Prefer primary image type
                for img in images:
                    if img.get('type') == 'primary':
                        return img.get('uri') or img.get('uri150')
                # Fall back to first image
                return images[0].get('uri') or images[0].get('uri150')
        except Exception as e:
            print(f"  [discogs error] {e}")
        finally:
            time.sleep(SLEEP_BETWEEN)  # rate limit between each request

    return None


def fetch_image_bytes(url: str, client: httpx.Client) -> bytes | None:
    """Download raw image bytes from any URL."""
    if not url:
        return None
    try:
        r = client.get(url, timeout=20, follow_redirects=True)
        if r.status_code == 200 and len(r.content) > 500:
            return r.content
    except Exception as e:
        print(f"  [download error] {e}")
    return None


# ── Load sources ──────────────────────────────────────────────────────────────

def load_parquet_tracks() -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        print(f"[ERROR] Parquet not found at {PARQUET_PATH}")
        return pd.DataFrame()
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[parquet] Loaded {len(df)} rows")
    return df




# ── Main gather step ──────────────────────────────────────────────────────────

def gather():
    """
    For each track:
      1. Skip if already downloaded
      2. Try Discogs API (master_id → release_id)
      3. Fall back to YouTube thumbnail URL from dataset
      4. Save to raw_thumbnails/{shard}/{id}.webp
    """
    if not DISCOGS_TOKEN:
        print("[WARNING] No DISCOGS_TOKEN found in .env — will skip Discogs and use YouTube only")

    df = load_parquet_tracks()
    if df.empty:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total       = len(df)
    skipped     = 0
    discogs_ok  = 0
    youtube_ok  = 0
    failed      = 0

    with httpx.Client(timeout=20, follow_redirects=True) as client:
        for i, row in df.iterrows():
            track_id   = str(row.get('id', '')).strip()
            master_id  = parse_id(row.get('master_id', ''))
            release_id = parse_id(row.get('release_id', ''))
            yt_url     = str(row.get('thumbnail', '')).strip()

            if not track_id:
                continue

            # ── Already downloaded ────────────────────────────────────────
            if already_downloaded(track_id):
                skipped += 1
                continue

            print(f"[{i+1}/{total}] {track_id}", end=' ')

            image_bytes = None
            source_used = None

            # ── Try Discogs ───────────────────────────────────────────────
            if DISCOGS_TOKEN and image_bytes is None:
                art_url = fetch_discogs_art_url(master_id, release_id, client)
                if art_url:
                    image_bytes = fetch_image_bytes(art_url, client)
                    if image_bytes:
                        source_used = 'discogs'

            # ── Fall back to YouTube thumbnail ────────────────────────────
            if image_bytes is None and yt_url and yt_url.startswith('http'):
                image_bytes = fetch_image_bytes(yt_url, client)
                if image_bytes:
                    source_used = 'youtube'

            # ── Try hqdefault if maxresdefault failed ─────────────────────
            if image_bytes is None and track_id:
                hq_url = f"https://img.youtube.com/vi/{track_id}/hqdefault.jpg"
                image_bytes = fetch_image_bytes(hq_url, client)
                if image_bytes:
                    source_used = 'youtube_hq'

            # ── Save or record failure ────────────────────────────────────
            if image_bytes:
                save_image(track_id, image_bytes)
                if source_used == 'discogs':
                    discogs_ok += 1
                    print(f"✓ discogs")
                else:
                    youtube_ok += 1
                    print(f"✓ {source_used}")
            else:
                failed += 1
                print(f"✗ no image found")

    # ── Summary ───────────────────────────────────────────────────────────────

    print(f"""
    ── Gather complete ──────────────────────────
    Total tracks:       {total}
    Already downloaded: {skipped}
    From Discogs:       {discogs_ok}
    From YouTube:       {youtube_ok}
    Failed (no image):  {failed}
    Saved to:           {OUT_DIR}
    ──────────────────────────────────────────────""")


# ── STEP 2: Apply filter ──────────────────────────
def apply_filter(img_bytes, track_id):
    """
    Apply a computer vision filter pipeline to raw thumbnails.

    The filter makes visible the contours that a machine learning model
    would extract from an image, overlaid on a spatially and
    frequency-compressed version of the original thumbnail.

    Pipeline:
      1. Canny edge detection — preserves readable structure
         even when the image becomes too compressed to interpret visually.
      2. Block quantization — samples one pixel per grid cell and fills
        the cell with that colour. Block size is seeded per track for variation.
         Blended 65/35 with the original so album art remains partially legible.
      3. JPEG DCT compression at quality=8 — introduces frequency-domain block
         artifacts. Standard lossy compression.
      4. Contour overlay — edge contours overlaid onto the filtered image.
      5. Noise grain — adds stochastic pixel variance.
      6. Output as WebP at source resolution.
    """
    # Decode image bytes to numpy array
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return img_bytes

    # Step 1: Contour detection on original full-resolution image.
    # Run before any compression so edges reflect actual image structure.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    # Step 2: Block quantization — each track gets a unique block size derived from its ID.
    h, w = img.shape[:2]
    seed = int(hashlib.md5((track_id or '').encode()).hexdigest()[:8], 16) % 1000 / 1000.0
    block_size = max(1, int(w / (4 + seed * 4)))
    blocked = img.copy()
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blocked[y:y+block_size, x:x+block_size] = img[min(y, h-1), min(x, w-1)]
    img = cv2.addWeighted(img, 0.65, blocked, 0.35, 0)

    # Step 4: JPEG DCT compression — frequency-domain blocking artifacts.
    _, compressed = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 8])
    base = cv2.imdecode(compressed, cv2.IMREAD_COLOR)

    # Step 5: Contour overlay — only if edges were detected
    ink = np.array([32, 37, 42], dtype=np.uint8)
    mask = edges > 0
    if mask.any():
        base[mask] = cv2.addWeighted(base[mask], 0.5, np.full_like(base[mask], ink), 0.5, 0)

    # Step 6: Noise grain
    noise = np.random.randint(-15, 15, base.shape, dtype=np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Step 7: Output as WebP
    pil_out = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    webp_buf = io.BytesIO()
    pil_out.save(webp_buf, format='WEBP', quality=60)
    return webp_buf.getvalue()

# ── Parser argument --filter function ──────────────────────────

def filter_all(force=False):
    """Apply the computer vision filter to all gathered thumbnails.
    force=True reprocesses files that already exist in the output directory.
    """
    raw_dir      = Path(__file__).parent / 'raw_thumbnails'
    filtered_dir = Path(__file__).parent / 'filtered_thumbnails'
    filtered_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(raw_dir.rglob('*.jpg')) + list(raw_dir.rglob('*.webp'))
    total   = len(all_images)
    skipped = 0
    done    = 0
    failed  = 0

    for i, src_path in enumerate(all_images):
        track_id = src_path.stem
        out_path = filtered_dir / shard(track_id) / f"{track_id}.webp"

        if out_path.exists() and not force:
            skipped += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = apply_filter(src_path.read_bytes(), track_id=track_id)
            out_path.write_bytes(result)
            done += 1
            print(f"[{i+1}/{total}] {track_id} ✓")
        except Exception as e:
            failed += 1
            print(f"[{i+1}/{total}] {track_id} ✗ {e}")

    print(f"""
── Filter complete ──────────────────────────
  Total:   {total}
  Done:    {done}
  Skipped: {skipped}
  Failed:  {failed}
  Saved to: {filtered_dir}
──────────────────────────────────────────────""")
    

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thumbnail pipeline — gather step')
    parser.add_argument(
        '--step', required=True, choices=['gather', 'filter'],
        help='Pipeline step to run'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Reprocess files that already exist in the output directory'
    )
    args = parser.parse_args()

    if args.step == 'gather':
        gather()
    elif args.step == 'filter':
        filter_all(force=args.force)