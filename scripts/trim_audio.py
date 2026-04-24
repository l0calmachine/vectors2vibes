"""
trim_audio.py — trim .ogg audio files in-place to the first N seconds,
re-encoding at a lower bitrate to reduce file size.

Usage:
    python scripts/trim_audio.py --seconds 30
    python scripts/trim_audio.py --seconds 30 --quality 2
    python scripts/trim_audio.py --seconds 60 --dir path/to/audio

Defaults to backend/data/audio/ if --dir is not specified.

Quality scale (libvorbis VBR):
    -1  ~45 kbps      (low)
     0  ~64 kbps
     2  ~80 kbps  ←── default, good for previews
     4  ~128 kbps
     6  ~192 kbps
     10 ~500 kbps     (overkill for previews)

Requires:
    ffmpeg on your PATH (https://ffmpeg.org/download.html)
"""

import sys
import shutil
import subprocess
from pathlib import Path


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install from https://ffmpeg.org/download.html")
        sys.exit(1)


def trim_and_reencode(
    path: Path,
    trimmed_path: Path,
    seconds: int,
    quality: float,
) -> str:
    """
    Run ffmpeg to trim + re-encode a single file.
    Returns "trimmed", "reencoded" (short file, only re-encoded), or raises.
    """
    # Probe duration first so we know whether to trim or just re-encode
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = None  # unknown — let ffmpeg decide

    will_trim = duration is None or duration > seconds

    cmd = [
        "ffmpeg",
        "-y",                        # overwrite output without asking
        "-i", str(path),
        "-c:a", "libvorbis",
        "-q:a", str(quality),
    ]
    if will_trim:
        cmd += ["-t", str(seconds)]  # only pass -t if actually trimming
    cmd.append(str(trimmed_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-300:])  # last 300 chars of ffmpeg stderr

    return "trimmed" if will_trim else "reencoded"


def trim_audio_files(audio_dir: Path, seconds: int, quality: float):
    check_ffmpeg()

    ogg_files = sorted(audio_dir.rglob("*.ogg"))
    if not ogg_files:
        print(f"[trim] No .ogg files found in {audio_dir} — nothing to do.")
        return

    trimmed = reencoded = errors = 0

    print(f"[trim] Found {len(ogg_files)} .ogg files in {audio_dir}")
    print(f"[trim] Trimming to first {seconds}s, re-encoding at libvorbis quality {quality} ...\n")

    for i, path in enumerate(ogg_files, 1):
        trimmed_path = path.with_stem(path.stem + "_trimmed")
        try:
            action = trim_and_reencode(path, trimmed_path, seconds, quality)
            path.unlink()
            trimmed_path.rename(path)
            if action == "trimmed":
                trimmed += 1
            else:
                reencoded += 1

        except Exception as exc:
            errors += 1
            print(f"  [warn] Could not process {path.name}: {exc}")
            if trimmed_path.exists():
                trimmed_path.unlink()

        if i % 500 == 0 or i == len(ogg_files):
            print(
                f"  {i}/{len(ogg_files)}  "
                f"trimmed={trimmed}  reencoded-only={reencoded}  errors={errors}"
            )

    print(
        f"\n[trim] Done — trimmed+reencoded: {trimmed}, "
        f"reencoded-only (already short): {reencoded}, errors: {errors}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trim .ogg files to N seconds and re-encode at a lower bitrate."
    )
    parser.add_argument(
        "--seconds",
        type=int,
        required=True,
        metavar="N",
        help="Trim each .ogg to its first N seconds (e.g. --seconds 30)",
    )
    parser.add_argument(
        "--quality",
        type=float,
        default=2.0,
        metavar="Q",
        help="libvorbis VBR quality: -1 (45kbps) to 10 (500kbps). Default: 2 (~80kbps)",
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

    trim_audio_files(args.dir, args.seconds, args.quality)
