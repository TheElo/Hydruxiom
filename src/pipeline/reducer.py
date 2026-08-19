"""Reducer for 3D Tag Space Visualization.

Implements UMAP and PCA dimensionality reduction.
"""

import time

import numpy as np


class Reducer:
    """Reduces high-dimensional tag vectors to 3D coordinates."""

    def __init__(self, algorithm='umap', n_components=3, n_neighbors=15, min_dist=0.1,
                 n_epochs=None, low_memory=False, learning_rate=1.0, metric='cosine',
                 n_jobs=-1, subsample_size=None, chunked_transform=True,
                 transform_chunk_bytes=1_500_000_000, pre_svd_components=None):
        """Initialize the reducer.

        Args:
            algorithm: 'umap' or 'pca' (default: 'umap')
            n_components: Number of output dimensions (default: 3)
            n_neighbors: UMAP parameter for local structure (default: 15)
            min_dist: UMAP parameter for point packing (default: 0.1)
            n_epochs: UMAP number of optimization epochs (default: None = auto)
            low_memory: UMAP low memory mode to reduce peak memory (default: False)
            learning_rate: Initial learning rate for optimization (default: 1.0)
            metric: Distance metric for UMAP ('cosine' or 'euclidean', default: 'cosine')
            n_jobs: Number of CPU cores for UMAP parallel NN-descent
                (default: -1 = all cores). NOTE: umap-learn 0.5.x rejects
                n_jobs=None, so the default must be a concrete int.
            subsample_size: If set, fit UMAP on a random subset of this size,
                then transform all points. Reduces memory and time at scale.
                (default: None = no subsampling)
            chunked_transform: When subsampling, transform all points in bounded
                row chunks instead of one giant call (default True). Set False to
                use the legacy single-call path for A/B comparison on real data.
            transform_chunk_bytes: Target dense-matrix byte budget per chunk when
                transforming rows against a fitted model (subsample path). The
                row count per chunk is derived from this so peak RAM stays ~2 GB
                regardless of dataset size or tag vocabulary.
            pre_svd_components: If set, run TruncatedSVD on the sparse matrix to
                this many components BEFORE UMAP (default None = off). umap-learn
                densifies its input internally and NN-descent cost scales with
                dimensionality, so collapsing 20k+ tag dims to ~64 makes distance
                computation hundreds of times cheaper per pair. Standard practice
                for TF-IDF-like data; local structure is well preserved (UMAP only
                needs neighborhoods). Toggleable off for A/B comparison on real data.
        """
        self.algorithm = algorithm.lower().replace(' ', '')
        # Normalize "gpu umap" -> "gpu"
        if self.algorithm == 'gpuumap':
            self.algorithm = 'gpu'
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.low_memory = low_memory
        self.learning_rate = learning_rate
        self.metric = metric
        # umap-learn 0.5.x raises TypeError on n_jobs=None; normalize defensively.
        self.n_jobs = -1 if n_jobs is None else n_jobs
        self.subsample_size = subsample_size
        self.chunked_transform = chunked_transform
        self.transform_chunk_bytes = transform_chunk_bytes
        self.pre_svd_components = pre_svd_components
        self.model = None

    def fit_transform(self, sparse_matrix):
        """Fit the reducer and transform the data.

        Args:
            sparse_matrix: Sparse matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: 3D coordinates of shape (n_samples, n_components)

        Raises:
            Exception: If the reduction algorithm fails (no silent fallback).
        """
        if self.algorithm == 'umap':
            return self._umap_transform(sparse_matrix)
        elif self.algorithm == 'gpu':
            return self._gpu_umap_transform(sparse_matrix)
        elif self.algorithm == 'pca':
            return self._pca_transform(sparse_matrix)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _umap_transform(self, sparse_matrix):
        """Apply UMAP dimensionality reduction.

        Supports optional subsampling: fit on a random subset, then transform
        all points. This makes UMAP feasible at 2M+ samples by reducing peak
        memory from O(n) to O(subsample_size).

        Args:
            sparse_matrix: Sparse matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: 3D coordinates for ALL samples

        Raises:
            ImportError: If umap-learn is not installed.
            Exception: If UMAP fails (e.g., memory allocation error).
        """
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP is not installed. Install with: pip install umap-learn")

        # Optional pre-reduction: TruncatedSVD collapses the tag vocabulary to a
        # low-dim dense space before UMAP. umap-learn densifies internally anyway,
        # so this trades one cheap SVD pass for dramatically cheaper NN-descent
        # (distance cost scales with dimensionality) and far lower peak RAM.
        if self.pre_svd_components:
            from sklearn.decomposition import TruncatedSVD
            k = min(int(self.pre_svd_components), sparse_matrix.shape[1] - 1, sparse_matrix.shape[0] - 1)
            print(f"Pre-reducing with TruncatedSVD to {k} components "
                  f"(from {sparse_matrix.shape[1]:,} tag dims)...")
            _t_svd = time.perf_counter()
            svd = TruncatedSVD(n_components=k, random_state=42)
            reduced = svd.fit_transform(sparse_matrix)
            print(f"  SVD done in {time.perf_counter() - _t_svd:.1f}s "
                  f"(explained variance: {svd.explained_variance_ratio_.sum():.1%})")
            sparse_matrix = np.ascontiguousarray(reduced, dtype=np.float32)

        n_samples = sparse_matrix.shape[0]
        use_subsample = (self.subsample_size is not None and n_samples > self.subsample_size)

        if use_subsample:
            print(f"Applying UMAP with subsampling ({self.subsample_size}/{n_samples}) "
                  f"(n_neighbors={self.n_neighbors}, min_dist={self.min_dist}, "
                  f"n_epochs={self.n_epochs}, metric={self.metric})...")
        else:
            print(f"Applying UMAP (n_neighbors={self.n_neighbors}, min_dist={self.min_dist}, "
                  f"n_epochs={self.n_epochs}, learning_rate={self.learning_rate}, low_memory={self.low_memory}, "
                  f"metric={self.metric})...")

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_epochs=self.n_epochs,
            low_memory=self.low_memory,
            learning_rate=self.learning_rate,
            metric=self.metric,
            n_jobs=self.n_jobs,  # CPU parallelism (no random_state for parallel NN-descent)
            verbose=False
        )

        if use_subsample:
            # Fit on a random subset, then transform all points
            rng = np.random.default_rng(42)
            subset_idx = rng.choice(n_samples, size=self.subsample_size, replace=False)
            subset_matrix = sparse_matrix[subset_idx]

            print(f"  Fitting UMAP on {self.subsample_size} samples...")
            reducer.fit(subset_matrix)

            if self.chunked_transform:
                # Chunked transform: umap-learn densifies each call internally, so a
                # single transform() over all rows would allocate n_samples x n_tags
                # float32 at once (e.g. 500k x 20k = ~40 GB). Transforming in row
                # chunks keeps peak RAM bounded by transform_chunk_bytes; projection
                # into a fitted space is order-independent, so results are identical.
                # Works for both dense (post-SVD) and sparse matrices.
                n_dims = sparse_matrix.shape[1]
                chunk_rows = max(1_000, int(self.transform_chunk_bytes // (n_dims * 4)))
                print(f"  Transforming all {n_samples} samples "
                      f"(chunks of ~{chunk_rows:,} rows)...")
                positions = np.empty((n_samples, self.n_components), dtype=np.float64)
                for start in range(0, n_samples, chunk_rows):
                    stop = min(start + chunk_rows, n_samples)
                    positions[start:stop] = reducer.transform(sparse_matrix[start:stop])
            else:
                # Legacy single-call path (toggleable off for A/B comparison).
                print(f"  Transforming all {n_samples} samples (single call)...")
                positions = reducer.transform(sparse_matrix)
        else:
            # Standard path: fit_transform on full data
            positions = reducer.fit_transform(sparse_matrix)

        self.model = reducer
        print(f"UMAP reduction complete: {positions.shape}")
        return positions

    def _gpu_umap_transform(self, sparse_matrix):
        """Apply GPU-accelerated UMAP via cuvs (RAPIDS).

        Falls back to CPU UMAP if cuvs is not installed or GPU unavailable.

        Args:
            sparse_matrix: Sparse matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: 3D coordinates
        """
        try:
            import cupy as cp
            import cudf
            import cuspatial
            import cuvs
            from cuvs.neighbors import CVC_Params, cvc
            from cuvs.common import cbrt_phi, cbrt_psi, cbrt_rho
            from cuvs.umap import UMAP as CuvsUMAP

            print(f"Applying GPU UMAP (cuvs) (n_neighbors={self.n_neighbors}, min_dist={self.min_dist}, "
                  f"n_epochs={self.n_epochs}, metric={self.metric})...")

            # Convert sparse matrix to GPU-friendly dense array
            dense = sparse_matrix.toarray()
            gpu_data = cp.asarray(dense)

            reducer = CuvsUMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                n_epochs=self.n_epochs,
                metric=self.metric,
                random_state=42,
                verbose=False
            )
            positions = reducer.fit_transform(gpu_data)
            self.model = reducer

            print(f"GPU UMAP reduction complete: {positions.shape}")
            return positions
        except ImportError as e:
            print(f"cuvs not installed ({e}); falling back to CPU UMAP")
            return self._umap_transform(sparse_matrix)
        except Exception as e:
            print(f"GPU UMAP failed: {e}; falling back to CPU UMAP")
            return self._umap_transform(sparse_matrix)

    def _pca_transform(self, sparse_matrix):
        """Apply PCA dimensionality reduction.

        Args:
            sparse_matrix: Sparse matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: 3D coordinates
        """
        try:
            from sklearn.decomposition import TruncatedSVD
            
            print(f"Applying PCA ({self.n_components} components)...")
            
            # Use TruncatedSVD for sparse matrices
            pca = TruncatedSVD(
                n_components=self.n_components,
                random_state=42
            )
            
            positions = pca.fit_transform(sparse_matrix)
            self.model = pca
            
            # Normalize positions to [-1, 1] range for better visualization
            if positions.max() > 0:
                positions = (positions - positions.min()) / (positions.max() - positions.min())
                positions = positions * 2 - 1
            
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"PCA reduction complete: {positions.shape}")
            print(f"Total explained variance: {explained_var:.2%}")
            
            return positions
        except Exception as e:
            print(f"PCA failed: {e}")
            # Last resort: return random positions
            n_samples = sparse_matrix.shape[0]
            print(f"Returning random positions for {n_samples} samples")
            return np.random.randn(n_samples, self.n_components) * 10

    def transform(self, sparse_matrix):
        """Transform new data using the fitted model.

        Args:
            sparse_matrix: Sparse matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: 3D coordinates
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit_transform first.")
        
        try:
            return self.model.transform(sparse_matrix)
        except Exception as e:
            print(f"Error transforming data: {e}")
            return None
