"""Headless unit test for the SoA SceneGraph (src/core/models.py).

Verifies, without Qt or Hydrus:
  * build_from_data populates flat arrays with correct shapes/values
  * cluster node_indices point at the right members
  * dominant tags resolve correctly (string + tokenized modes)
  * without_cluster() remaps surviving clusters' indices correctly
  * to_json() emits a well-formed structure

Run:
    .venv\\Scripts\\python.exe tests\\test_soa_scene_graph.py

Exit code 0 = all checks pass, 1 = failure.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np


def _make_data():
    # 8 files across 2 real clusters + noise. file_ids are large ints (like hash_ids).
    file_ids = [1001, 1002, 1003, 1004, 2001, 2002, 3001, 3002]
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.1, 0.1, 0.0],   # cluster 0
        [5.0, 5.0, 5.0],
        [5.1, 5.0, 5.0],   # cluster 1
        [9.0, 9.0, 9.0],   # noise (-1)
        [9.2, 8.9, 9.1],   # noise (-1)
    ])
    cluster_labels = np.array([0, 0, 0, 0, 1, 1, -1, -1])
    tag_data = {
        1001: ["a", "b"],
        1002: ["a", "c"],
        1003: ["a", "b"],
        1004: ["a"],
        2001: ["x", "y"],
        2002: ["x", "z"],
        3001: ["n1"],
        3002: ["n2"],
    }
    return file_ids, positions, cluster_labels, tag_data


def test_build_string_mode():
    from src.core.models import SceneGraph
    file_ids, positions, labels, tag_data = _make_data()

    scene = SceneGraph()
    scene.build_from_data(file_ids, positions, tag_data, labels, top_n_dominant_tags=3)

    assert len(scene) == 8
    assert scene.positions.shape == (8, 3)
    assert scene.cluster_ids.dtype == np.int32
    assert scene.colors.shape == (8, 3) and scene.colors.dtype == np.uint8
    assert scene.sizes.shape == (8,)

    # file_id_to_index maps each id to its array position
    assert scene.index_of(1001) == 0
    assert scene.index_of(2002) == 5
    assert scene.index_of(9999) is None

    # Cluster membership via node_indices
    c0 = scene.clusters[0]
    assert set(c0.node_indices.tolist()) == {0, 1, 2, 3}
    assert np.allclose(c0.centroid, [0.05, 0.05, 0.0])
    # dominant tags for cluster 0: 'a' x4, then b/c
    assert c0.dominant_tags[0] == "a"

    c1 = scene.clusters[1]
    assert set(c1.node_indices.tolist()) == {4, 5}
    assert c1.dominant_tags[0] == "x"

    noise = scene.clusters[-1]
    assert set(noise.node_indices.tolist()) == {6, 7}
    assert noise.label == "Noise"

    # colors: cluster members share a color, noise is NOISE_COLOR
    assert np.array_equal(scene.colors[0], scene.colors[3])
    assert np.array_equal(scene.colors[6], SceneGraph.NOISE_COLOR)


def test_without_cluster_remap():
    from src.core.models import SceneGraph
    file_ids, positions, labels, tag_data = _make_data()

    scene = SceneGraph()
    scene.build_from_data(file_ids, positions, tag_data, labels)

    # Pop cluster 0 (indices 0..3). Survivors: old [4,5,6,7] -> new [0,1,2,3].
    popped = scene.without_cluster(0)

    assert len(popped) == 4
    assert popped.file_ids == [2001, 2002, 3001, 3002]
    # cluster 0 must be gone; clusters 1 and -1 remain
    assert 0 not in popped.clusters
    assert set(popped.clusters[1].node_indices.tolist()) == {0, 1}   # remapped from {4,5}
    assert set(popped.clusters[-1].node_indices.tolist()) == {2, 3}  # remapped from {6,7}
    # positions carried over correctly for the survivors
    assert np.allclose(popped.positions[0], [5.0, 5.0, 5.0])


def test_tokenized_mode():
    from src.core.models import SceneGraph
    file_ids, positions, labels, _ = _make_data()

    # Tokenize: a->0, b->1, c->2, x->3, y->4, z->5, n1->6, n2->7
    vocab = ["a", "b", "c", "x", "y", "z", "n1", "n2"]
    idx_of = {t: i for i, t in enumerate(vocab)}
    tokenized_data = {
        1001: [idx_of["a"], idx_of["b"]],
        1002: [idx_of["a"], idx_of["c"]],
        1003: [idx_of["a"], idx_of["b"]],
        1004: [idx_of["a"]],
        2001: [idx_of["x"], idx_of["y"]],
        2002: [idx_of["x"], idx_of["z"]],
        3001: [idx_of["n1"]],
        3002: [idx_of["n2"]],
    }

    scene = SceneGraph()
    scene.build_from_data(file_ids, positions, tokenized_data, labels,
                          tokenized=True, reverse_vocab=vocab)

    # dominant tags must be resolved back to strings
    assert scene.clusters[0].dominant_tags[0] == "a"
    assert scene.clusters[1].dominant_tags[0] == "x"


def test_to_json():
    from src.core.models import SceneGraph
    file_ids, positions, labels, tag_data = _make_data()

    scene = SceneGraph()
    scene.build_from_data(file_ids, positions, tag_data, labels)

    j = scene.to_json()
    assert len(j["nodes"]) == 8
    assert len(j["clusters"]) == 3
    node0 = j["nodes"][0]
    assert node0["id"] == 1001
    assert node0["cluster_id"] == 0
    assert "a" in node0["tags"]
    assert set(j["camera"].keys()) == {"position", "target"}


def main():
    tests = [test_build_string_mode, test_without_cluster_remap,
             test_tokenized_mode, test_to_json]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    # Confirm headless importability (no Qt/Hydrus pulled in)
    assert "PySide6" not in sys.modules and "hydrus_api" not in sys.modules, \
        "SoA scene graph test must run without Qt/Hydrus"
    print("ALL PASS")


if __name__ == "__main__":
    main()
