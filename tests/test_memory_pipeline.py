"""Memory test for the 3D tag-space pipeline.

Reproduces the ArrayMemoryError crash scenario by measuring peak memory
(tracemalloc) of the vectorize -> reduce -> cluster -> scene-build pipeline
in two variants:

  * BEFORE (no fix): the fitted UMAP model (red.model) and the sparse matrix
    stay alive through clustering + scene-graph building.
  * AFTER  (fix):    red.model is released and the sparse matrix deleted
    immediately after fit_transform, then gc.collect() before returning.

The test loads REAL data from the first configured client (direct-DB mode,
same path the app uses) so the numbers reflect actual usage.

Run (no pytest required):
    .venv\\Scripts\\python.exe tests\\test_memory_pipeline.py
    .venv\\Scripts\\python.exe tests\\test_memory_pipeline.py --max-files 40000

Exit code 0 = fix reduces peak memory (or no client available, skipped).
Exit code 1 = fix did NOT reduce peak memory (regression).
"""

import argparse
import gc
import os
import sys
import tracemalloc

# Make the project root importable regardless of CWD.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_real_tag_data(max_files):
    """Load real tag data from the first configured client (direct-DB mode).

    Returns (tag_data, client_name) or (None, None) if no client is available.
    """
    from src.data.clients import client_ids, get_client_config
    from src.data.loader import DataLoader
    from src.utils.utility_functions import ConnectToClient

    ids = client_ids()
    if not ids:
        return None, None
    client_name = ids[0]
    cfg = get_client_config(client_name)
    if not cfg:
        return None, None

    use_direct_db = bool((cfg.get("db_dir") or "").strip())
    client = ConnectToClient(client_name)
    loader = DataLoader(
        client,
        chunk_size=2048,
        client_name=client_name,
        use_direct_db=use_direct_db,
    )
    loader.load_in_chunks(
        tag_service="local",
        max_files=max_files,
        search_tags=["system:archive"],
    )
    tag_data = loader.get_tag_data()
    if not tag_data:
        return None, None
    return tag_data, client_name


def _run_pipeline(tag_data, release_model):
    """Run vectorize -> reduce -> cluster -> scene-build.

    Args:
        tag_data: dict file_id -> list of tags
        release_model: if True, apply the fix (release red.model + sparse
            matrix after fit_transform, gc.collect before return).

    Returns:
        (scene, peak_bytes) where peak_bytes is the tracemalloc peak for the
        reduce+cluster+scene phase (the phase where the leak matters).
    """
    from src.pipeline.vectorizer import Vectorizer
    from src.pipeline.reducer import Reducer
    from src.pipeline.clusterer import Clusterer
    from src.core.models import SceneGraph

    vec = Vectorizer(min_doc_freq=3)
    sparse_matrix, file_ids = vec.create_vectors(tag_data)

    red = Reducer(
        algorithm="umap",
        n_neighbors=10,
        min_dist=0.0,
        n_epochs=500,
        learning_rate=0.2,
        low_memory=False,
        metric="cosine",
        n_jobs=os.cpu_count() or 4,
    )

    # Start measuring right before the reduce phase (the leak window).
    gc.collect()
    tracemalloc.start()

    positions = red.fit_transform(sparse_matrix)

    if release_model:
        # THE FIX: release the fitted model + sparse matrix now.
        red.model = None
        del red
        del sparse_matrix
    else:
        # BEFORE: keep references alive (simulates the old behaviour).
        _keepalive = (red, sparse_matrix)  # noqa: F841

    clust = Clusterer(eps=0.22, min_samples=2)
    cluster_labels = clust.fit_predict(positions)

    scene = SceneGraph()
    scene.build_from_data(
        file_ids, positions, tag_data, cluster_labels,
        min_size=0.3, max_size=0.8,
    )

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if release_model:
        gc.collect()
    return scene, peak


def main():
    parser = argparse.ArgumentParser(description="Pipeline memory test")
    parser.add_argument("--max-files", type=int, default=20000,
                        help="Max files to load from the client (default 20000)")
    args = parser.parse_args()

    print(f"Loading real data (max_files={args.max_files})...")
    tag_data, client_name = _load_real_tag_data(args.max_files)
    if tag_data is None:
        print("SKIP: no configured client / no data available. "
              "Add a client to clients.json to run this test.")
        return 0
    print(f"Loaded {len(tag_data)} files from client '{client_name}'.")

    print("\n--- BEFORE (no fix): keep red.model + sparse_matrix alive ---")
    _, peak_before = _run_pipeline(tag_data, release_model=False)
    print(f"Peak memory (reduce+cluster+scene): {peak_before / 1e6:.1f} MB")

    print("\n--- AFTER (fix): release red.model + sparse_matrix ---")
    _, peak_after = _run_pipeline(tag_data, release_model=True)
    print(f"Peak memory (reduce+cluster+scene): {peak_after / 1e6:.1f} MB")

    saved = peak_before - peak_after
    print(f"\nMemory saved by fix: {saved / 1e6:.1f} MB "
          f"({100.0 * saved / peak_before:.1f}% of before-peak)")

    if saved > 0:
        print("PASS: fix reduces peak memory.")
        return 0
    print("FAIL: fix did not reduce peak memory (regression).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
