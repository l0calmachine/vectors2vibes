"""
trim_audio.py — trim .ogg audio files in-place to the first N seconds.

Usage:
    python scripts/trim_audio.py --seconds 30
    python scripts/trim_audio.py --seconds 60 --dir path/to/audio

Defaults to backend/data/audio/ if --dir is not specified.

Requires:
    pip install pydub
    ffmpeg on your PATH (https://ffmpeg.org/download.html)
"""

import sys
from pathlib import Path


def trim_audio_files(audio_dir: Path, seconds: int):
    try:
        from pydub import AudioSegment
    except ImportError:
        print("ERROR: pydub not installed. Run: pip install pydub")
        print("       Also ensure ffmpeg is available on your PATH.")
        sys.exit(1)

    ogg_files = sorted(audio_dir.rglob("*.ogg"))
    if not ogg_files:
        print(f"[trim] No .ogg files found in {audio_dir} — nothing to do.")
        return

    target_ms = seconds * 1000
    trimmed = skipped = deleted = errors = 0

    print(f"[trim] Found {len(ogg_files)} .ogg files in {audio_dir}")
    print(f"[trim] Trimming to first {seconds}s, then deleting originals ...\n")

    for i, path in enumerate(ogg_files, 1):
        trimmed_path = path.with_stem(path.stem + "_trimmed")
        try:
            audio = AudioSegment.from_ogg(path)

            if len(audio) <= target_ms:
                skipped += 1
            else:
                clipped = audio[:target_ms]
                clipped.export(trimmed_path, format="ogg", codec="libvorbis")
                path.unlink()
                trimmed_path.rename(path)
                trimmed += 1
                deleted += 1

        except Exception as exc:
            errors += 1
            print(f"  [warn] Could not process {path.name}: {exc}")
            # Clean up partial trimmed file if it exists
            if trimmed_path.exists():
                trimmed_path.unlink()

        if i % 500 == 0 or i == len(ogg_files):
            print(f"  {i}/{len(ogg_files)}  trimmed={trimmed}  skipped={skipped}  deleted={deleted}  errors={errors}")

    print(f"\n[trim] Done — trimmed: {trimmed}, skipped (already short): {skipped}, errors: {errors}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trim .ogg audio files to the first N seconds.")
    parser.add_argument(
        "--seconds",
        type=int,
        required=True,
        metavar="N",
        help="Trim each .ogg to its first N seconds (e.g. --seconds 30)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).parent.parent / "backend" / "data" / "audio",
        metavar="PATH",
        help="Directory to scan for .ogg files (default: backend/data/audio/)",
    )
    args = parser.parse_args()

    if args.seconds <= 0:
        print("ERROR: --seconds must be a positive integer.")
        sys.exit(1)

    if not args.dir.exists():
        print(f"ERROR: Directory not found: {args.dir}")
        sys.exit(1)

    trim_audio_files(args.dir, args.seconds)