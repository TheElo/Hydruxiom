"""Scene Graph for 3D Tag Space Visualization (Structure-of-Arrays).

Stores all node data in flat numpy arrays instead of one Python object per
node. At 2M files this cuts the scene graph from ~1.2-1.4 GB (a `TagNode`
object + dict entry each) to a few hundred MB of contiguous arrays, and turns
the O(n) Python loops that used to rebuild positions/colors/cluster_ids on
every render into direct array access.

Design decisions (see docs/soa_scene_graph_study.md):
- D1: No compatibility shim. `TagNode` is gone; all consumers use the arrays.
- D2: Per-node tag lists are NOT stored here. The UI keeps ``tag_data``
  (file_id -> list of tags) as the source of truth; we hold a reference to it
  only so cluster dominant-tags and JSON export can resolve strings.
- D3: ``file_id_to_index`` is a dict (Hydrus hash_ids are large, non-dense ints).
- D4: Clusters store ``node_indices`` (indices into the arrays), not node objects.

This module is pure Python + numpy (no Qt, no Hydrus) so it stays importable in
headless benchmarks and unit tests.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np


@dataclass
class Cluster:
    """A cluster of related files.

    ``node_indices`` are positions into the SceneGraph arrays (file_ids,
    positions, colors, ...), not node objects — this avoids duplicating any
    per-node data and lets every cluster op be a vectorized array slice.
    """
    cluster_id: int
    centroid: np.ndarray  # Center of cluster (3,)
    node_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    dominant_tags: List[str] = field(default_factory=list)
    color: Tuple[int, int, int] = (255, 255, 255)
    label: str = ""
    density: float = 0.0

    @property
    def size(self):
        return int(self.node_indices.shape[0])


class SceneGraph:
    """Manages all nodes and clusters in the 3D scene as flat arrays."""

    # Distinct pastel colors for clusters (soft but visible)
    CLUSTER_COLORS = [
        (255, 107, 107),  # Soft Red
        (78, 205, 196),   # Soft Teal
        (170, 111, 255),  # Soft Purple
        (255, 196, 88),   # Soft Orange
        (129, 236, 236),  # Soft Cyan
        (255, 159, 241),  # Soft Pink
        (162, 255, 129),  # Soft Lime
        (255, 152, 152),  # Soft Coral
        (129, 140, 255),  # Soft Indigo
        (255, 234, 129),  # Soft Yellow
        (199, 129, 255),  # Soft Violet
        (129, 255, 190),  # Soft Mint
        (255, 178, 107),  # Soft Peach
        (107, 255, 205),  # Soft Aqua
        (255, 129, 178),  # Soft Rose
    ]

    NOISE_COLOR = (169, 169, 169)  # Dark Gray for noise points

    def __init__(self):
        """Initialize an empty scene graph."""
        self.file_ids: List[Any] = []                 # original file ids (order == array index)
        self.positions: np.ndarray = np.zeros((0, 3))  # (n, 3)
        self.cluster_ids: np.ndarray = np.zeros(0, dtype=np.int32)  # (n,) -1 = noise
        self.colors: np.ndarray = np.zeros((0, 3), dtype=np.uint8)   # (n, 3) RGB
        self.sizes: np.ndarray = np.zeros(0, dtype=np.float32)       # (n,)
        self.scores: np.ndarray = np.zeros(0, dtype=np.float32)      # (n,)

        self.file_id_to_index: Dict[Any, int] = {}   # {file_id: array index}
        self.clusters: Dict[int, Cluster] = {}

        # Tag source of truth is owned by the UI; we keep a reference only.
        self.tag_data: Optional[Dict[Any, list]] = None
        self.tokenized = False
        self.reverse_vocab = None

        self.camera_position = np.array([0.0, 0.0, 100.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.filters: Dict[str, Any] = {}

    # ------------------------------------------------------------------ build

    def build_from_data(self, file_ids, positions, tag_data, cluster_labels,
                        top_n_dominant_tags=5, node_size=0.02,
                        tokenized=False, reverse_vocab=None):
        """Build the complete scene graph from processed data (no per-node objects).

        Args:
            file_ids: Sequence of file IDs (order defines array index 0..n-1)
            positions: np.ndarray of shape (n_files, 3)
            tag_data: dict mapping file_id -> list of tags
                (strings, or integer indices when tokenized=True)
            cluster_labels: np.ndarray of cluster labels (int per file)
            top_n_dominant_tags: Number of dominant tags per cluster
            node_size: Uniform node size for all nodes
            tokenized: Whether tag_data values are integer indices (default: False)
            reverse_vocab: List mapping index -> tag string, used only when
                tokenized=True (e.g. from a TagInterner)
        """
        self.tokenized = tokenized
        self.reverse_vocab = reverse_vocab
        self.tag_data = tag_data

        n = len(file_ids)
        self.file_ids = list(file_ids)
        self.file_id_to_index = {fid: i for i, fid in enumerate(self.file_ids)}
        self.positions = np.asarray(positions, dtype=np.float64).reshape(n, 3)
        self.cluster_ids = np.asarray(cluster_labels, dtype=np.int32).reshape(n)

        # Uniform node size
        self.sizes = np.full(n, float(node_size), dtype=np.float32)

        # Colors by cluster (vectorized via the label array)
        colors = np.empty((n, 3), dtype=np.uint8)
        noise_mask = self.cluster_ids == -1
        colors[noise_mask] = self.NOISE_COLOR
        for cid in np.unique(self.cluster_ids):
            if cid == -1:
                continue
            m = self.cluster_ids == cid
            colors[m] = self.CLUSTER_COLORS[int(cid) % len(self.CLUSTER_COLORS)]
        self.colors = colors

        # Scores from external tag weights (one-time O(total tags))
        self.scores = self._compute_scores()

        # Build cluster objects (indices + centroid + dominant tags + density)
        self._build_clusters(top_n_dominant_tags)

        print(f"Scene graph built with {n} nodes and {len(self.clusters)} clusters")

    def _resolve_tag(self, tag):
        """Return the display string for a tag (resolving tokenized indices)."""
        if self.tokenized and self.reverse_vocab is not None:
            if 0 <= tag < len(self.reverse_vocab):
                return self.reverse_vocab[tag]
            return None
        return tag

    def _compute_scores(self) -> np.ndarray:
        """Per-node score = sum of ExternalTagScores over its tags."""
        from src.core.tag_scores import ExternalTagScores
        n = len(self.file_ids)
        scores = np.zeros(n, dtype=np.float32)
        if not ExternalTagScores or self.tag_data is None:
            return scores
        for i, fid in enumerate(self.file_ids):
            tags = self.tag_data.get(fid)
            if not tags:
                continue
            total = 0.0
            for tag in tags:
                s = ExternalTagScores.get(tag)
                if s is None and self.tokenized and self.reverse_vocab is not None \
                        and 0 <= tag < len(self.reverse_vocab):
                    s = ExternalTagScores.get(self.reverse_vocab[tag])
                if s:
                    total += s
            scores[i] = total
        return scores

    def _build_clusters(self, top_n):
        """Build Cluster objects (node_indices + centroid + dominant tags)."""
        self.clusters = {}
        for cid in np.unique(self.cluster_ids):
            idx = np.where(self.cluster_ids == cid)[0]  # member array indices
            if len(idx) == 0:
                continue

            centroid = self.positions[idx].mean(axis=0)

            if int(cid) == -1:
                cluster = Cluster(
                    cluster_id=-1,
                    centroid=centroid,
                    node_indices=idx,
                    color=self.NOISE_COLOR,
                    label="Noise",
                )
            else:
                # Dominant tags from the UI's tag_data over this cluster's members
                dominant_tags = self._dominant_tags(idx, top_n)

                # Density (points per unit volume)
                distances = np.linalg.norm(self.positions[idx] - centroid, axis=1)
                avg_distance = float(distances.mean()) if len(distances) else 0.0
                density = len(idx) / (avg_distance ** 3 + 1e-6) if avg_distance > 0 else float(len(idx))

                cluster = Cluster(
                    cluster_id=int(cid),
                    centroid=centroid,
                    node_indices=idx,
                    dominant_tags=dominant_tags,
                    color=self.CLUSTER_COLORS[int(cid) % len(self.CLUSTER_COLORS)],
                    label=f"Cluster {int(cid)}",
                    density=density,
                )
            self.clusters[cluster.cluster_id] = cluster

    def _dominant_tags(self, member_indices: np.ndarray, top_n: int) -> List[str]:
        """Top-N most common tags among a cluster's members (resolved to strings)."""
        if self.tag_data is None:
            return []
        counts = Counter()
        for i in member_indices:
            tags = self.tag_data.get(self.file_ids[i])
            if tags:
                counts.update(tags)
        result = []
        for tag, _count in counts.most_common(top_n):
            s = self._resolve_tag(tag)
            if s is not None:
                result.append(s)
        return result

    # ------------------------------------------------------------- accessors

    def __len__(self):
        return len(self.file_ids)

    @property
    def n(self):
        return len(self.file_ids)

    def get_node_positions(self) -> np.ndarray:
        """All node positions, shape (n, 3)."""
        return self.positions

    def get_node_colors(self) -> np.ndarray:
        """All node colors as float in [0,1], shape (n, 3)."""
        if len(self.file_ids) == 0:
            return np.zeros((0, 3))
        return self.colors.astype(np.float64) / 255.0

    def get_node_sizes(self) -> np.ndarray:
        """All node sizes, shape (n,)."""
        return self.sizes

    def get_file_ids(self) -> List[Any]:
        """All file IDs in array order."""
        return self.file_ids

    def index_of(self, file_id) -> Optional[int]:
        """Array index for a file id, or None if absent."""
        return self.file_id_to_index.get(file_id)

    def indices_for_cluster(self, cluster_id) -> np.ndarray:
        """Boolean-free member indices for a cluster (empty array if unknown)."""
        c = self.clusters.get(cluster_id)
        if c is not None:
            return c.node_indices
        return np.where(self.cluster_ids == cluster_id)[0]

    # ------------------------------------------------------------- mutation

    def without_cluster(self, cluster_id):
        """Return a new SceneGraph with all nodes of `cluster_id` removed.

        Fast path for "pop cohort": slices the arrays once and remaps each
        surviving cluster's node_indices to the new index space. No per-node
        objects are created or copied.
        """
        keep = self.cluster_ids != cluster_id
        survivors_old = np.where(keep)[0]  # old index of each survivor, in order

        new = SceneGraph()
        new.tokenized = self.tokenized
        new.reverse_vocab = self.reverse_vocab
        new.tag_data = self.tag_data
        new.camera_position = self.camera_position.copy()
        new.camera_target = self.camera_target.copy()

        new.file_ids = [self.file_ids[i] for i in survivors_old]
        new.positions = self.positions[survivors_old].copy()
        new.cluster_ids = self.cluster_ids[survivors_old].copy()
        new.colors = self.colors[survivors_old].copy()
        new.sizes = self.sizes[survivors_old].copy()
        new.scores = self.scores[survivors_old].copy()
        new.file_id_to_index = {fid: i for i, fid in enumerate(new.file_ids)}

        # Remap surviving clusters' node_indices from old -> new index space.
        for cid, cluster in self.clusters.items():
            if cid == cluster_id:
                continue
            new_idx = np.searchsorted(survivors_old, cluster.node_indices)
            new_cluster = Cluster(
                cluster_id=cluster.cluster_id,
                centroid=cluster.centroid.copy(),
                node_indices=new_idx,
                dominant_tags=list(cluster.dominant_tags),
                color=cluster.color,
                label=cluster.label,
                density=cluster.density,
            )
            new.clusters[cid] = new_cluster

        return new

    # ------------------------------------------------------------- export

    def to_json(self):
        """Serialize scene graph to a JSON-compatible dict."""
        def _resolve_tags(tags):
            out = []
            for t in tags:
                s = self._resolve_tag(t)
                if s is not None:
                    out.append(s)
            return out

        nodes = []
        for i, fid in enumerate(self.file_ids):
            tags = (self.tag_data or {}).get(fid, [])
            nodes.append({
                "id": fid,
                "pos": self.positions[i].tolist(),
                "color": [int(c) for c in self.colors[i]],
                "size": float(self.sizes[i]),
                "tags": _resolve_tags(tags)[:10],  # Limit for performance
                "cluster_id": int(self.cluster_ids[i]),
            })

        clusters = []
        for c in self.clusters.values():
            clusters.append({
                "id": c.cluster_id,
                "centroid": c.centroid.tolist(),
                "size": c.size,
                "dominant_tags": list(c.dominant_tags)[:5],
                "color": [int(x) for x in c.color],
                "label": c.label,
            })

        return {
            "nodes": nodes,
            "clusters": clusters,
            "camera": {
                "position": self.camera_position.tolist(),
                "target": self.camera_target.tolist(),
            },
        }
