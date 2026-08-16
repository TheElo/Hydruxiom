"""Benchmark UMAP CPU core count for the 3D tag map pipeline.

Sweeps n_jobs (CPU cores) for UMAP on synthetic sparse TF-IDF-like data
that mimics the real pipeline (150k files x ~30k tag dims, low density).

For each core count it measures:
  - NN phase time  (pynndescent graph construction, the dominant cost)
  - Opt phase time (SGD optimization)
  - Total time
  - Peak Python memory (tracemalloc, MB)
  - Process RSS delta (psutil, MB) if psutil is installed

Usage:
  python benchmarks/benchmark_umap_cores.py            # full sweep (50k samples)
  python benchmarks/benchmark_umap_cores.py --quick    # fast smoke test (10k samples)
  python benchmarks/benchmark_umap_cores.py --n 150000 # real-scale test

Results are printed as a table and saved to benchmarks/umap_cores_report.csv.
"""

import argparse
import csv
import gc
import os
import sys
import time
import tracemalloc

import numpy as np

# Make project root importable (for nothing critical here, but keeps parity)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional: psutil for process-level RSS measurement
try:
    import psutil
    _PROCESS = psutil.Process()
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


# ---------------------------------------------------------------------------
# Data generation (mimics the real TF-IDF sparse matrix)
# ---------------------------------------------------------------------------
def make_sparse_data(n_samples, n_features, density, seed=42):
    """Create a sparse binary-ish TF-IDF-like matrix.

    Real data: each file has ~30-80 tags out of ~20k-50k vocabulary,
    so density is roughly 0.001-0.005. We use a slightly higher density
    so the benchmark is representative without being trivially empty.
    """
    from scipy.sparse import random as sparse_random
    rng = np.random.RandomState(seed)
    X = sparse_random(
        n_samples, n_features, density=density,
        format='csr', random_state=rng, data_rvs=lambda n: rng.rand(n) + 0.1,
    )
    return X


# ---------------------------------------------------------------------------
# Phase timing via NNDescent monkeypatch
# ---------------------------------------------------------------------------
class _NNPhaseTimer:
    """Records time spent inside pynndescent NNDescent (fit + query)."""
    def __init__(self):
        self.nn_time = 0.0
        self.calls = 0


def run_umap_timed(X, n_jobs, timer, n_neighbors=15, min_dist=0.0,
                   n_epochs=None, metric='cosine'):
    """Run UMAP with NN-phase timing. Returns (embedding, total_time)."""
    import umap
    import umap.umap_ as umap_mod

    nn_timer = timer

    # Patch NNDescent in umap's namespace to time the NN phase.
    # NOTE: in umap-learn 0.5.x the NN graph is built inside the NNDescent
    # CONSTRUCTOR (self.neighbor_graph is populated in __init__), so we
    # must wrap __init__, not fit/query.
    orig_NNDescent = umap_mod.NNDescent

    class TimedNNDescent(orig_NNDescent):
        def __init__(self, *a, **kw):
            t0 = time.perf_counter()
            super().__init__(*a, **kw)
            nn_timer.nn_time += time.perf_counter() - t0
            nn_timer.calls += 1

    umap_mod.NNDescent = TimedNNDescent
    try:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs,
            metric=metric,
            n_jobs=n_jobs,
            verbose=False,
        )
        t0 = time.perf_counter()
        emb = reducer.fit_transform(X)
        total = time.perf_counter() - t0
    finally:
        umap_mod.NNDescent = orig_NNDescent

    return emb, total


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------
def _rss_mb():
    """Current process RSS in MB, or None if psutil unavailable."""
    if not HAVE_PSUTIL:
        return None
    return _PROCESS.memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep(n_samples, n_features, density, cores_list,
          n_neighbors=15, min_dist=0.0, n_epochs=None):
    """Run the core-count sweep and return results list."""
    print(f"\n{'='*70}")
    print(f"Data: {n_samples} samples x {n_features} features, density={density}")
    print(f"Cores to test: {cores_list}")
    print(f"UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}, "
          f"n_epochs={n_epochs or 'auto'}, metric=cosine")
    print(f"Memory tracking: tracemalloc=on, psutil={'on' if HAVE_PSUTIL else 'OFF (pip install psutil)'}")
    print(f"{'='*70}")

    X = make_sparse_data(n_samples, n_features, density)
    print(f"Generated sparse matrix: {X.shape}, nnz={X.nnz}")

    results = []

    for n_jobs in cores_list:
        timer = _NNPhaseTimer()

        # --- Memory setup: clean slate for this run ---
        gc.collect()
        tracemalloc.start()
        rss_before = _rss_mb()

        t0 = time.perf_counter()
        emb, total = run_umap_timed(
            X, n_jobs=n_jobs, timer=timer,
            n_neighbors=n_neighbors, min_dist=min_dist,
            n_epochs=n_epochs,
        )
        wall = time.perf_counter() - t0

        # --- Memory teardown: capture peak, then release ---
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = _rss_mb()

        peak_mb = peak / (1024 * 1024)
        rss_delta_mb = (rss_after - rss_before) if (rss_before is not None and rss_after is not None) else None

        nn_t = timer.nn_time
        opt_t = total - nn_t
        results.append({
            "n_jobs": n_jobs,
            "nn_phase_s": round(nn_t, 2),
            "opt_phase_s": round(opt_t, 2),
            "total_s": round(total, 2),
            "wall_s": round(wall, 2),
            "peak_mem_mb": round(peak_mb, 1),
            "rss_delta_mb": round(rss_delta_mb, 1) if rss_delta_mb is not None else "",
        })

        rss_str = f"  RSS+{rss_delta_mb:7.1f}MB" if rss_delta_mb is not None else ""
        print(f"  n_jobs={n_jobs:>2}: NN={nn_t:7.2f}s  OPT={opt_t:7.2f}s  "
              f"TOTAL={total:7.2f}s  PEAK={peak_mb:8.1f}MB{rss_str}  "
              f"(NN patch calls: {timer.calls})")

        # Release per-run artifacts before the next iteration
        del emb
        gc.collect()

    return results


def save_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_jobs", "nn_phase_s", "opt_phase_s", "total_s", "wall_s",
                    "peak_mem_mb", "rss_delta_mb"])
        for r in results:
            w.writerow([r["n_jobs"], r["nn_phase_s"],
                        r["opt_phase_s"], r["total_s"], r["wall_s"],
                        r["peak_mem_mb"], r["rss_delta_mb"]])
    print(f"\nResults saved to {path}")


def print_summary(results):
    print(f"\n{'='*70}")
    print("SUMMARY (total UMAP time + peak memory per core count)")
    print(f"{'='*70}")
    best = min(results, key=lambda r: r["total_s"])
    worst = max(results, key=lambda r: r["total_s"])
    for r in results:
        marker = " <-- BEST" if r is best else ""
        rss = f"  RSS+{r['rss_delta_mb']:>7.1f}MB" if r["rss_delta_mb"] != "" else ""
        print(f"  n_jobs={r['n_jobs']:>2}: NN={r['nn_phase_s']:8.2f}s  "
              f"OPT={r['opt_phase_s']:8.2f}s  TOTAL={r['total_s']:8.2f}s  "
              f"PEAK={r['peak_mem_mb']:8.1f}MB{rss}{marker}")
    speedup = worst["total_s"] / best["total_s"]
    print(f"\n  Speedup best vs worst: {speedup:.2f}x")
    print(f"  Recommended n_jobs: {best['n_jobs']}")
    if any(r["rss_delta_mb"] != "" for r in results):
        max_rss = max((r for r in results if r["rss_delta_mb"] != ""),
                      key=lambda r: r["rss_delta_mb"])
        print(f"  Largest RSS growth: n_jobs={max_rss['n_jobs']} "
              f"(+{max_rss['rss_delta_mb']}MB)")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="UMAP CPU core sweep benchmark")
    parser.add_argument("--n", type=int, default=50000,
                        help="Number of samples (default: 50000)")
    parser.add_argument("--features", type=int, default=30000,
                        help="Number of features (default: 30000)")
    parser.add_argument("--density", type=float, default=0.002,
                        help="Sparsity density (default: 0.002)")
    parser.add_argument("--cores", type=str, default="1,2,4,8,12,16,20",
                        help="Comma-separated core counts to test")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (10k samples, fewer cores)")
    args = parser.parse_args()

    if args.quick:
        n_samples = 10000
        cores_list = [1, 4, 8, 20]
    else:
        n_samples = args.n
        cores_list = [int(c) for c in args.cores.split(",")]

    # Cap cores at physical count
    max_cores = os.cpu_count() or 4
    cores_list = [c for c in cores_list if c <= max_cores]
    print(f"System has {max_cores} logical cores; testing: {cores_list}")

    results = sweep(
        n_samples=n_samples,
        n_features=args.features,
        density=args.density,
        cores_list=cores_list,
    )

    print_summary(results)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "umap_cores_report.csv")
    save_csv(results, out_path)


if __name__ == "__main__":
    main()
