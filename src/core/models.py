"""Scene Graph for 3D Tag Space Visualization.

Manages TagNodes, Clusters, and the overall scene structure.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class TagNode:
    """Represents a file node in the 3D tag space."""
    file_id: str
    position: np.ndarray  # (x, y, z)
    tags: List[str]
    score: float
    cluster_id: int  # -1 for noise
    color: Tuple[int, int, int]  # RGB
    size: float  # Based on score or tag count
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Cluster:
    """Represents a cluster of related files."""
    cluster_id: int
    centroid: np.ndarray  # Center of cluster
    nodes: List[TagNode] = field(default_factory=list)
    dominant_tags: List[str] = field(default_factory=list)
    color: Tuple[int, int, int] = (255, 255, 255)
    label: str = ""
    density: float = 0.0


class SceneGraph:
    """Manages all nodes and clusters in the 3D scene."""

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
        """Initialize the scene graph."""
        self.nodes: Dict[str, TagNode] = {}
        self.clusters: Dict[int, Cluster] = {}
        self.camera_position = np.array([0.0, 0.0, 100.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.filters: Dict[str, Any] = {}
        self.tokenized = False
        self.reverse_vocab = None

    def add_nodes(self, nodes: List[TagNode]):
        """Add nodes to the scene.

        Args:
            nodes: List of TagNode objects
        """
        self.nodes.update({n.file_id: n for n in nodes})

    def add_cluster(self, cluster: Cluster):
        """Add a cluster to the scene.

        Args:
            cluster: Cluster object
        """
        self.clusters[cluster.cluster_id] = cluster

    def build_from_data(self, file_ids, positions, tag_data, cluster_labels,
                        top_n_dominant_tags=5, node_size=0.02,
                        tokenized=False, reverse_vocab=None):
        """Build the complete scene graph from processed data.

        Args:
            file_ids: List of file IDs
            positions: np.ndarray of shape (n_files, 3)
            tag_data: Dictionary mapping file_id to list of tags
                (strings, or integer indices when tokenized=True)
            cluster_labels: np.ndarray of cluster labels
            top_n_dominant_tags: Number of dominant tags per cluster
            node_size: Uniform node size for all nodes
            tokenized: Whether tag_data values are integer indices (default: False)
            reverse_vocab: List mapping index -> tag string, used only when
                tokenized=True (e.g. from a TagInterner)
        """
        self.tokenized = tokenized
        self.reverse_vocab = reverse_vocab

        # Create nodes
        nodes = []
        for i, file_id in enumerate(file_ids):
            pos = positions[i]
            tags = tag_data.get(file_id, [])
            cluster_id = cluster_labels[i]
            
            # Calculate score based on tag count and external scores
            from src.utils.query_comperator import ExternalTagScores
            if tokenized and reverse_vocab:
                score = sum(ExternalTagScores.get(reverse_vocab[tag], 0) for tag in tags if tag < len(reverse_vocab))
            else:
                score = sum(ExternalTagScores.get(tag, 0) for tag in tags)
            
            # Uniform node size
            size = node_size
            
            # Color based on cluster
            if cluster_id == -1:
                color = self.NOISE_COLOR
            else:
                color = self.CLUSTER_COLORS[cluster_id % len(self.CLUSTER_COLORS)]
            
            node = TagNode(
                file_id=file_id,
                position=pos,
                tags=tags,
                score=score,
                cluster_id=cluster_id,
                color=color,
                size=size
            )
            nodes.append(node)
        
        self.add_nodes(nodes)
        
        # Build clusters
        self._build_clusters(file_ids, tag_data, cluster_labels, top_n_dominant_tags)
        
        print(f"Scene graph built with {len(self.nodes)} nodes and {len(self.clusters)} clusters")

    def _build_clusters(self, file_ids, tag_data, cluster_labels, top_n):
        """Build cluster objects from data.

        Args:
            file_ids: List of file IDs
            tag_data: Dictionary mapping file_id to list of tags
                (strings, or integer indices when tokenized=True)
            cluster_labels: np.ndarray of cluster labels
            top_n: Number of dominant tags per cluster
        """
        from collections import Counter
        
        # Group nodes by cluster
        cluster_nodes = {}
        for node in self.nodes.values():
            cid = node.cluster_id
            if cid not in cluster_nodes:
                cluster_nodes[cid] = []
            cluster_nodes[cid].append(node)
        
        # Create cluster objects
        for cluster_id, nodes in cluster_nodes.items():
            if cluster_id == -1:
                # Noise cluster
                centroid = np.mean([n.position for n in nodes], axis=0)
                cluster = Cluster(
                    cluster_id=-1,
                    centroid=centroid,
                    nodes=nodes,
                    color=self.NOISE_COLOR,
                    label="Noise"
                )
            else:
                # Regular cluster
                centroid = np.mean([n.position for n in nodes], axis=0)
                
                # Calculate dominant tags
                tag_counts = Counter()
                for node in nodes:
                    tag_counts.update(node.tags)
                if self.tokenized and self.reverse_vocab:
                    dominant_tags = [
                        self.reverse_vocab[tag] for tag, count in tag_counts.most_common(top_n)
                        if tag < len(self.reverse_vocab)
                    ]
                else:
                    dominant_tags = [tag for tag, count in tag_counts.most_common(top_n)]
                
                # Calculate density (points per unit volume)
                distances = [np.linalg.norm(n.position - centroid) for n in nodes]
                avg_distance = np.mean(distances) if distances else 0
                density = len(nodes) / (avg_distance ** 3 + 1e-6) if avg_distance > 0 else len(nodes)
                
                color = self.CLUSTER_COLORS[cluster_id % len(self.CLUSTER_COLORS)]
                
                cluster = Cluster(
                    cluster_id=cluster_id,
                    centroid=centroid,
                    nodes=nodes,
                    dominant_tags=dominant_tags,
                    color=color,
                    label=f"Cluster {cluster_id}",
                    density=density
                )
            
            self.add_cluster(cluster)

    def get_nodes_by_cluster(self, cluster_id):
        """Get all nodes in a specific cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            List[TagNode]: Nodes in the cluster
        """
        return [node for node in self.nodes.values() if node.cluster_id == cluster_id]

    def without_cluster(self, cluster_id):
        """Return a new SceneGraph with all nodes of `cluster_id` removed.

        Fast path for "pop cohort": reuses the existing TagNode and Cluster
        objects directly (no recreation, no tag-list copying, no cluster
        recomputation). Removing one cohort does not affect any other cohort's
        positions, colors, dominant tags, or density, so those are carried over
        as-is.

        Args:
            cluster_id: The cluster to remove.

        Returns:
            SceneGraph: A new scene graph without the given cluster's nodes.
        """
        new = SceneGraph()
        new.tokenized = self.tokenized
        new.reverse_vocab = self.reverse_vocab
        new.camera_position = self.camera_position.copy()
        new.camera_target = self.camera_target.copy()

        # Reuse node objects (skip only the popped cluster's nodes)
        for fid, node in self.nodes.items():
            if node.cluster_id != cluster_id:
                new.nodes[fid] = node

        # Reuse cluster objects (they reference the same surviving nodes)
        for cid, cluster in self.clusters.items():
            if cid != cluster_id:
                new.clusters[cid] = cluster

        return new

    def get_node_positions(self):
        """Get all node positions as a numpy array.

        Returns:
            np.ndarray: Array of shape (n_nodes, 3)
        """
        if not self.nodes:
            return np.array([]).reshape(0, 3)
        
        positions = np.array([node.position for node in self.nodes.values()])
        return positions

    def get_node_colors(self):
        """Get all node colors as a numpy array.

        Returns:
            np.ndarray: Array of shape (n_nodes, 3)
        """
        if not self.nodes:
            return np.array([]).reshape(0, 3)
        
        colors = np.array([node.color for node in self.nodes.values()])
        return colors

    def get_node_sizes(self):
        """Get all node sizes as a numpy array.

        Returns:
            np.ndarray: Array of shape (n_nodes,)
        """
        if not self.nodes:
            return np.array([])
        
        sizes = np.array([node.size for node in self.nodes.values()])
        return sizes

    def get_file_ids(self):
        """Get all file IDs in the scene.

        Returns:
            List[str]: List of file IDs
        """
        return list(self.nodes.keys())

    def to_json(self):
        """Serialize scene graph to JSON-compatible format.

        Returns:
            dict: JSON-serializable representation
        """
        def _resolve_tags(tags):
            if self.tokenized and self.reverse_vocab:
                return [self.reverse_vocab[t] for t in tags if t < len(self.reverse_vocab)]
            return tags

        return {
            "nodes": [{
                "id": n.file_id,
                "pos": n.position.tolist(),
                "color": list(n.color),
                "size": n.size,
                "tags": _resolve_tags(n.tags)[:10],  # Limit for performance
                "cluster_id": n.cluster_id
            } for n in self.nodes.values()],
            "clusters": [{
                "id": c.cluster_id,
                "centroid": c.centroid.tolist(),
                "size": len(c.nodes),
                "dominant_tags": c.dominant_tags[:5],
                "color": list(c.color),
                "label": c.label
            } for c in self.clusters.values()],
            "camera": {
                "position": self.camera_position.tolist(),
                "target": self.camera_target.tolist()
            }
        }
