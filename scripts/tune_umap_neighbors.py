"""
tune_umap_neighbors.py — evaluate UMAP n_neighbors values for audio and lyrical embeddings.

Measures two complementary metrics for each n_neighbors value:

  Trustworthiness — of each point's k nearest neighbors in 2D, what fraction were also
    among its k nearest neighbors in high-D space? Detects false positives: points that
    appear close in 2D but were not actually close in embedding space.

  Continuity — of each point's k nearest neighbors in high-D space, what fraction appear
    among its k nearest neighbors in 2D? Detects false negatives: true high-D neighbors
    that the projection lost or pushed far away.

Both scores are in [0, 1]; higher is better. Together they describe local structure
preservation from both directions. Neither captures global topology.

Usage:
    python scripts/tune_umap_neighbors.py
    python scripts/tune_umap_neighbors.py --layer audio
    python scripts/tune_umap_neighbors.py --layer lyrical
    python scripts/tune_umap_neighbors.py --n 5 10 15 20 30

Output:
    scripts/umap_tuning/results_audio.csv
    scripts/umap_tuning/results_lyrical.csv
    scripts/umap_tuning/scatter_audio_n{k}.png
    scripts/umap_tuning/scatter_lyrical_n{k}.png

Runtime: ~3–5 min UMAP + ~2–4 min continuity per (layer × n) pair on CPU.
Results are saved incrementally — safe to interrupt and resume.
Existing rows missing continuity are filled without re-running UMAP.
"""

import argparse
import ast
import csv
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

PARQUET_PATH = Path(__file__).parent.parent / "backend" / "data" / "master_dataset.parquet"
OUT_DIR      = Path(__file__).parent / "umap_tuning"

DEFAULT_N_VALUES = [5, 10, 15, 20, 30, 50]

BATCH_SIZE = 256  # rows processed at once during continuity computation


# ── Data loading ──────────────────────────────────────────────────────────────

def load_embeddings():
    """Load audio and lyric embedding matrices plus metadata for scatter plots."""
    print(f"[load] Reading {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    df = df.drop_duplicates(subset='id', keep='first').reset_index(drop=True)
    print(f"[load] {len(df)} tracks")

    audio_embs    = np.stack(df['audio_embedding'].tolist()).astype(np.float32)
    lyric_embs    = np.stack(df['lyric_embeddings'].tolist()).astype(np.float32)
    combined_embs = np.stack(df['combined_embedding'].tolist()).astype(np.float32)

    year_col = 'earliest_release' if 'earliest_release' in df.columns else 'year'
    years = df[year_col].fillna(0).astype(float).values

    def first_genre(val):
        if isinstance(val, list):
            return val[0] if val else 'Unknown'
        try:
            parsed = ast.literal_eval(str(val))
            return parsed[0] if isinstance(parsed, list) and parsed else str(val)
        except Exception:
            return str(val).strip("[]'\"") or 'Unknown'

    genre_col = next((c for c in ('release_genres', 'genre') if c in df.columns), None)
    genres = df[genre_col].apply(first_genre).tolist() if genre_col else ['Unknown'] * len(df)

    return audio_embs, lyric_embs, combined_embs, years, genres


# ── UMAP ──────────────────────────────────────────────────────────────────────

def run_umap(embs, n_neighbors):
    """Fit 2D UMAP with fixed seed for comparable runs. Same params as embedding_service.py."""
    import umap as umap_lib
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='cosine',
        low_memory=True,
        random_state=42,  # fixed so runs are comparable across n_neighbors values
        n_jobs=1,         # explicit — random_state forces this, silences the warning
    )
    return reducer.fit_transform(embs)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_trustworthiness(X_hd, X_2d, k):
    """Fraction of 2D neighbors that were also high-D neighbors (false-positive detector)."""
    from sklearn.manifold import trustworthiness
    return trustworthiness(X_hd, X_2d, n_neighbors=k, metric='cosine')


def compute_continuity(X_hd, X_2d, k):
    """Fraction of high-D neighbors that appear in 2D neighbors (false-negative detector).

    Implemented from Venna & Kaski (2006). Symmetric counterpart to trustworthiness:
    penalises high-D neighbors that were pushed away in the 2D projection.
    Distances in 2D are euclidean; distances in high-D are cosine (matching UMAP's metric).
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import cdist

    n = len(X_hd)

    # Ground-truth: k nearest neighbors per point in high-D cosine space
    ind_hd = NearestNeighbors(
        n_neighbors=k + 1, metric='cosine', algorithm='brute', n_jobs=1
    ).fit(X_hd).kneighbors(return_distance=False)[:, 1:]   # (n, k), excludes self

    # Projection: k nearest neighbors per point in 2D euclidean space
    ind_2d = NearestNeighbors(
        n_neighbors=k + 1, metric='euclidean', algorithm='brute', n_jobs=1
    ).fit(X_2d).kneighbors(return_distance=False)[:, 1:]   # (n, k), excludes self

    # For each point, find high-D neighbors missing from the 2D projection and
    # penalise by their 2D rank. Computed in batches to avoid an n×n distance matrix.
    total_penalty = 0
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        # Euclidean distances from batch points to all points: shape (batch, n)
        d_batch = cdist(X_2d[start:end], X_2d, metric='euclidean')

        for local_i, i in enumerate(range(start, end)):
            missing = set(ind_hd[i]) - set(ind_2d[i])
            if not missing:
                continue
            d_row = d_batch[local_i].copy()
            d_row[i] = np.inf  # exclude self from ranking
            for j in missing:
                # 1-indexed rank of j among all other points sorted by 2D distance from i
                rank_j = int(np.sum(d_row < d_row[j])) + 1
                total_penalty += rank_j - k

    return 1.0 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * total_penalty


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_results(csv_path):
    """Return {n: {'trustworthiness': float|None, 'continuity': float|None}}."""
    results = {}
    if not csv_path.exists():
        return results
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                n = int(row['n_neighbors'])
                results[n] = {
                    'trustworthiness': float(row['trustworthiness']) if row.get('trustworthiness') else None,
                    'continuity':      float(row['continuity'])      if row.get('continuity')      else None,
                }
            except (KeyError, ValueError):
                pass
    return results


def save_results(csv_path, results):
    """Write full results dict to CSV."""
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_neighbors', 'trustworthiness', 'continuity'])
        for n in sorted(results.keys()):
            r = results[n]
            t = f"{r['trustworthiness']:.6f}" if r.get('trustworthiness') is not None else ''
            c = f"{r['continuity']:.6f}"      if r.get('continuity')      is not None else ''
            w.writerow([n, t, c])


# ── Scatter plot ──────────────────────────────────────────────────────────────

def save_scatter(coords_2d, years, layer, n_neighbors):
    """Save scatter plot coloured by decade to umap_tuning/."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        decades = ((years // 10) * 10).astype(int)
        unique_decades = sorted(set(decades[decades > 0]))
        cmap = cm.get_cmap('plasma', len(unique_decades))
        decade_to_color = {d: cmap(i) for i, d in enumerate(unique_decades)}
        colors = [decade_to_color.get(d, (0.5, 0.5, 0.5, 0.3)) for d in decades]

        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, s=1.5, alpha=0.6, linewidths=0)
        ax.set_title(f'UMAP {layer}  n_neighbors={n_neighbors}', fontsize=12)
        ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
        ax.set_facecolor('#111')
        fig.patch.set_facecolor('#111')
        ax.tick_params(colors='#aaa')
        ax.title.set_color('#eee')
        ax.xaxis.label.set_color('#aaa')
        ax.yaxis.label.set_color('#aaa')

        handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=cmap(i), markersize=5, label=str(d))
                   for i, d in enumerate(unique_decades)]
        ax.legend(handles=handles, title='Decade', fontsize=6, title_fontsize=7,
                  loc='lower right', framealpha=0.3, labelcolor='#ccc')

        out = OUT_DIR / f"scatter_{layer}_n{n_neighbors:02d}.png"
        fig.savefig(out, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [scatter] saved → {out.name}")
    except ImportError:
        print("  [scatter] matplotlib not installed — skipping plots")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(layers, n_values):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_embs, lyric_embs, combined_embs, years, genres = load_embeddings()

    layer_map = {
        'audio':    (audio_embs,    OUT_DIR / 'results_audio.csv'),
        'lyrical':  (lyric_embs,    OUT_DIR / 'results_lyrical.csv'),
        'combined': (combined_embs, OUT_DIR / 'results_combined.csv'),
    }

    for layer in layers:
        embs, csv_path = layer_map[layer]
        results = load_results(csv_path)
        print(f"\n── {layer} embeddings {embs.shape} ─────────────────────────")

        for n in n_values:
            row = results.get(n, {})
            need_trust = row.get('trustworthiness') is None
            need_cont  = row.get('continuity') is None

            if not need_trust and not need_cont:
                print(f"  n={n:2d}  [complete, skipping]")
                continue

            # Only re-run UMAP if trustworthiness is missing (continuity alone can reuse coords)
            if need_trust:
                print(f"  n={n:2d}  fitting UMAP...", flush=True)
                coords = run_umap(embs, n_neighbors=n)

                print(f"  n={n:2d}  trustworthiness (k={n})...", flush=True)
                trust = compute_trustworthiness(embs, coords, k=n)
                print(f"  n={n:2d}  trustworthiness = {trust:.4f}")
                if n not in results:
                    results[n] = {}
                results[n]['trustworthiness'] = trust
                save_results(csv_path, results)
                save_scatter(coords, years, layer, n)
            else:
                print(f"  n={n:2d}  trustworthiness already done, recomputing 2D coords for continuity...", flush=True)
                coords = run_umap(embs, n_neighbors=n)

            if need_cont:
                print(f"  n={n:2d}  continuity (k={n})...", flush=True)
                cont = compute_continuity(embs, coords, k=n)
                print(f"  n={n:2d}  continuity      = {cont:.4f}")
                results[n]['continuity'] = cont
                save_results(csv_path, results)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Results ──────────────────────────────────────────────────────────")
    for layer in layers:
        _, csv_path = layer_map[layer]
        if not csv_path.exists():
            continue
        rows = load_results(csv_path)
        print(f"\n  {layer}")
        print(f"  {'n_neighbors':>12}  {'trustworthiness':>16}  {'continuity':>12}")
        for n in sorted(rows.keys()):
            r = rows[n]
            t = f"{r['trustworthiness']:.4f}" if r.get('trustworthiness') is not None else '    —'
            c = f"{r['continuity']:.4f}"      if r.get('continuity')      is not None else '    —'
            current = {'audio': 10, 'lyrical': 20, 'combined': 15}
            marker = '  ← current' if n == current.get(layer) else ''
            print(f"  {n:>12}  {t:>16}  {c:>12}{marker}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune UMAP n_neighbors for vectors2vibes.')
    parser.add_argument('--layer', choices=['audio', 'lyrical', 'combined', 'both'], default='both')
    parser.add_argument('--n', type=int, nargs='+', default=DEFAULT_N_VALUES, metavar='N')
    args = parser.parse_args()

    layers = ['audio', 'lyrical', 'combined'] if args.layer == 'both' else [args.layer]
    run(layers, sorted(set(args.n)))
