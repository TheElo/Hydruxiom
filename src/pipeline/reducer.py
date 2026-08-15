"""Reducer for 3D Tag Space Visualization.

Implements UMAP and PCA dimensionality reduction.
"""

import numpy as np


class Reducer:
    """Reduces high-dimensional tag vectors to 3D coordinates."""

    def __init__(self, algorithm='umap', n_components=3, n_neighbors=15, min_dist=0.1,
                 n_epochs=None, low_memory=False, learning_rate=1.0, metric='cosine',
                 n_jobs=None):
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
            n_jobs: Number of CPU cores for UMAP parallel NN-descent (default: None = auto)
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
        self.n_jobs = n_jobs
        self.model = None

    def fit_transform(self, sparse_matrix):
        """Fit the reducer and transform the data.

        Args:
            sparse_matrix: Sparse matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: 3D coordinates of shape (n_samples, n_components)
        """
        try:
            if self.algorithm == 'umap':
                return self._umap_transform(sparse_matrix)
            elif self.algorithm == 'gpu':
                return self._gpu_umap_transform(sparse_matrix)
            elif self.algorithm == 'pca':
                return self._pca_transform(sparse_matrix)
            else:
                print(f"Unknown algorithm {self.algorithm}, falling back to PCA")
                return self._pca_transform(sparse_matrix)
        except Exception as e:
            print(f"Error in {self.algorithm}: {e}")
            print("Falling back to PCA...")
            return self._pca_transform(sparse_matrix)

    def _umap_transform(self, sparse_matrix):
        """Apply UMAP dimensionality reduction.

        Args:
            sparse_matrix: Sparse matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: 3D coordinates
        """
        try:
            import umap
            
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
                verbose=True
            )
            
            # UMAP can work directly with sparse matrices
            positions = reducer.fit_transform(sparse_matrix)
            self.model = reducer
            
            print(f"UMAP reduction complete: {positions.shape}")
            return positions
        except ImportError:
            print("UMAP not installed, falling back to PCA")
            return self._pca_transform(sparse_matrix)
        except Exception as e:
            print(f"UMAP failed: {e}")
            return self._pca_transform(sparse_matrix)

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
                verbose=True
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
