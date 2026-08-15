"""Clusterer for 3D Tag Space Visualization.

Implements DBSCAN clustering on 3D positions.
"""

import numpy as np


class Clusterer:
    """Clusters 3D positions using DBSCAN."""

    def __init__(self, eps=0.5, min_samples=10):
        """Initialize the clusterer.

        Args:
            eps: Maximum distance between two samples in same cluster (default: 0.5)
            min_samples: Minimum samples to form a cluster (default: 10)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.n_clusters = 0

    def fit_predict(self, positions):
        """Cluster the 3D positions.

        Args:
            positions: np.ndarray of shape (n_samples, 3)

        Returns:
            np.ndarray: Cluster labels (-1 for noise)
        """
        try:
            from sklearn.cluster import DBSCAN
            
            print(f"Applying DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
            
            clustering = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric='euclidean'
            )
            
            self.labels = clustering.fit_predict(positions)
            
            # Count clusters (exclude noise label -1)
            unique_labels = set(self.labels)
            self.n_clusters = len(unique_labels - {-1})
            
            noise_count = np.sum(self.labels == -1)
            print(f"DBSCAN clustering complete:")
            print(f"  - Found {self.n_clusters} clusters")
            print(f"  - Noise points: {noise_count}")
            
            return self.labels
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
            # Return all as noise if clustering fails
            n_samples = positions.shape[0]
            self.labels = np.full(n_samples, -1, dtype=int)
            self.n_clusters = 0
            return self.labels

    def evaluate(self, positions, max_cohort_size, max_noise_ratio):
        """Evaluate a DBSCAN parameter combination against optimization goals.

        Args:
            positions: np.ndarray of shape (n_samples, 3)
            max_cohort_size: Target maximum cohort size (larger = should split)
            max_noise_ratio: Target maximum noise ratio (0-100, as %)

        Returns:
            dict: Evaluation metrics for the current eps/min_samples:
                - noise_count: number of non-cohorted (noise) nodes
                - noise_ratio: noise_count / total (0-1)
                - max_cohort_size: size of the largest cohort
                - oversized_cohorts: number of cohorts exceeding max_cohort_size
                - n_clusters: number of clusters found
                - score: composite score (lower is better)
        """
        import numpy as np

        labels = self.fit_predict(positions)
        total = len(labels)
        if total == 0:
            return {
                "noise_count": 0, "noise_ratio": 0.0,
                "max_cohort_size": 0, "oversized_cohorts": 0,
                "n_clusters": 0, "score": float('inf'),
            }

        noise_count = int(np.sum(labels == -1))
        noise_ratio = noise_count / total

        # Compute cohort sizes (exclude noise)
        cohort_sizes = []
        for cluster_id in range(self.n_clusters):
            size = int(np.sum(labels == cluster_id))
            cohort_sizes.append(size)
        max_cohort = max(cohort_sizes) if cohort_sizes else 0
        oversized = sum(1 for s in cohort_sizes if s > max_cohort_size)

        # Composite score: penalize noise and oversized cohorts.
        # Normalize noise ratio to 0-1 against the target; penalize oversize linearly.
        noise_penalty = max(0.0, noise_ratio - (max_noise_ratio / 100.0))
        oversize_penalty = oversized
        # Small bonus for having more clusters (more granular) but not too many.
        cluster_bonus = min(self.n_clusters, 50) * 0.01

        score = noise_penalty * 100.0 + oversize_penalty * 10.0 - cluster_bonus

        return {
            "noise_count": noise_count,
            "noise_ratio": noise_ratio,
            "max_cohort_size": max_cohort,
            "oversized_cohorts": oversized,
            "n_clusters": self.n_clusters,
            "score": score,
        }

    def _normalize_positions(self, positions):
        """Normalize positions to a consistent scale for DBSCAN.

        DBSCAN's eps is an absolute distance in coordinate units. UMAP/PCA
        outputs have arbitrary scales that vary with file count and reducer
        settings, so a fixed eps fraction behaves differently across datasets.
        Normalizing to a unit scale (e.g., [-1, 1] or std-based) makes eps
        comparable regardless of the underlying coordinate magnitude.

        Args:
            positions: np.ndarray of shape (n_samples, 3)

        Returns:
            np.ndarray: Normalized positions (same shape)
        """
        import numpy as np

        if positions.size == 0:
            return positions

        # Center and scale by standard deviation to a consistent magnitude.
        # This makes eps a relative measure of local density rather than an
        # absolute coordinate distance.
        mean = positions.mean(axis=0)
        centered = positions - mean
        std = centered.std(axis=0)
        std[std == 0] = 1.0  # avoid divide-by-zero for degenerate axes
        return centered / std

    def optimize(self, positions, max_cohort_size, max_noise_ratio,
                 eps_min, eps_max, min_samples_min, min_samples_max,
                 max_attempts=60, progress_callback=None):
        """Search for the ideal eps/min_samples combination using iterative 5x5 grid refinement.

        Strategy:
        1. Phase 1: 5 evenly-spaced eps x 5 evenly-spaced min_samples = 25 runs (coarse map)
        2. Phase 2+: Narrow range to best +/- 50% of current range, pick 5 new points
           (center = best value, avoid re-testing), run another 5x5 grid
        3. Repeat until max_attempts exhausted or no improvement

        Prints a score landscape map to the console after each phase.

        Args:
            positions: np.ndarray of shape (n_samples, 3)
            max_cohort_size: Target maximum cohort size
            max_noise_ratio: Target maximum noise ratio (0-100, as %)
            eps_min, eps_max: EPS search range (as fractions, e.g. 0.05-1.0)
            min_samples_min, min_samples_max: Min Samples search range
            max_attempts: Maximum number of DBSCAN runs
            progress_callback: Optional callable(attempt, total, message)

        Returns:
            dict: Best parameters and evaluation:
                - eps: best eps value
                - min_samples: best min_samples value
                - evaluation: metrics for the best combination
                - attempts: number of runs performed
        """
        import numpy as np

        GRID = 5  # points per axis per phase
        tested = {}  # (eps_rounded, min_samples) -> score
        best = None
        best_score = float('inf')
        attempts = 0
        phase = 0

        # Current search ranges (narrowed each phase)
        cur_eps_lo, cur_eps_hi = eps_min, eps_max
        cur_ms_lo, cur_ms_hi = min_samples_min, min_samples_max

        while attempts < max_attempts:
            phase += 1
            runs_this_phase = min(GRID * GRID, max_attempts - attempts)
            if runs_this_phase <= 0:
                break

            # Pick 5 evenly-spaced values for each axis within current range
            eps_candidates = self._pick_grid_values(cur_eps_lo, cur_eps_hi, GRID, tested, is_int=False)
            ms_candidates = self._pick_grid_values(cur_ms_lo, cur_ms_hi, GRID, tested, is_int=True)

            # Build the 5x5 grid for this phase
            grid_scores = {}  # (eps, ms) -> score
            phase_best = None
            phase_best_score = float('inf')

            for eps in eps_candidates:
                for ms in ms_candidates:
                    if attempts >= max_attempts:
                        break
                    key = (round(eps, 4), int(ms))
                    if key in tested:
                        # Already tested - reuse score
                        grid_scores[(eps, ms)] = tested[key]
                        continue

                    attempts += 1
                    if progress_callback:
                        progress_callback(attempts, max_attempts,
                                          f"P{phase} eps={eps:.3f}, ms={ms}")

                    self.eps = float(eps)
                    self.min_samples = int(ms)
                    eval_result = self.evaluate(positions, max_cohort_size, max_noise_ratio)
                    score = eval_result["score"]
                    tested[key] = score
                    grid_scores[(eps, ms)] = score

                    if score < phase_best_score:
                        phase_best_score = score
                        phase_best = (float(eps), int(ms), eval_result)

                if attempts >= max_attempts:
                    break

            # Print the score landscape map
            self._print_score_map(phase, eps_candidates, ms_candidates, grid_scores,
                                  best_eps=best["eps"] if best else None,
                                  best_ms=best["min_samples"] if best else None)

            # Update global best
            if phase_best and phase_best_score < best_score:
                best_score = phase_best_score
                best = {
                    "eps": phase_best[0],
                    "min_samples": phase_best[1],
                    "evaluation": phase_best[2],
                }

            # Check for improvement - if no new best, stop
            if phase_best is None or phase_best_score >= best_score:
                print(f"[Optimize] Phase {phase}: no improvement, stopping.")
                break

            # Narrow range: best +/- 50% of current range width
            if best is None:
                break
            eps_width = cur_eps_hi - cur_eps_lo
            ms_width = cur_ms_hi - cur_ms_lo
            if eps_width <= 0 or ms_width <= 0:
                break

            best_eps_val = best["eps"]
            best_ms_val = best["min_samples"]
            new_eps_lo = max(eps_min, best_eps_val - eps_width * 0.5)
            new_eps_hi = min(eps_max, best_eps_val + eps_width * 0.5)
            new_ms_lo = max(min_samples_min, int(best_ms_val - ms_width * 0.5))
            new_ms_hi = min(min_samples_max, int(best_ms_val + ms_width * 0.5))

            # Ensure at least some spread
            if new_eps_hi - new_eps_lo < eps_width * 0.1:
                new_eps_lo = max(eps_min, best_eps_val - eps_width * 0.25)
                new_eps_hi = min(eps_max, best_eps_val + eps_width * 0.25)
            if new_ms_hi - new_ms_lo < max(1, ms_width * 0.1):
                new_ms_lo = max(min_samples_min, int(best_ms_val - ms_width * 0.25))
                new_ms_hi = min(min_samples_max, int(best_ms_val + ms_width * 0.25))

            cur_eps_lo, cur_eps_hi = new_eps_lo, new_eps_hi
            cur_ms_lo, cur_ms_hi = new_ms_lo, new_ms_hi

            print(f"[Optimize] Phase {phase} done. Narrowing to "
                  f"eps=[{cur_eps_lo:.3f}, {cur_eps_hi:.3f}], "
                  f"ms=[{cur_ms_lo}, {cur_ms_hi}]")

        if best is None:
            # Fall back to current settings
            best = {
                "eps": self.eps,
                "min_samples": self.min_samples,
                "evaluation": self.evaluate(positions, max_cohort_size, max_noise_ratio),
            }

        best["attempts"] = attempts
        return best

    def _pick_grid_values(self, lo, hi, n, tested, is_int=False):
        """Pick n evenly-spaced values in [lo, hi], avoiding already-tested values.

        If a candidate was already tested, nudge it to the nearest untested value.
        For integer parameters, ensures unique values.

        Args:
            lo: Lower bound
            hi: Upper bound
            n: Number of values to pick
            tested: dict of tested keys (for dedup)
            is_int: Whether values should be integers

        Returns:
            list of n values
        """
        if hi <= lo:
            return [lo] * n if not is_int else [int(lo)] * n

        # Evenly spaced candidates
        if is_int:
            lo_i, hi_i = int(lo), int(hi)
            if hi_i - lo_i < n - 1:
                # Range too small for n unique values - use all available
                return list(range(lo_i, hi_i + 1))
            candidates = [lo_i + i * (hi_i - lo_i) // (n - 1) for i in range(n)]
        else:
            candidates = [lo + i * (hi - lo) / (n - 1) for i in range(n)]

        # Deduplicate and nudge tested values
        result = []
        for c in candidates:
            val = int(c) if is_int else round(c, 4)
            # If already tested, find nearest untested
            if is_int:
                if val in [int(v) for v in result]:
                    # Try neighbors
                    for offset in range(1, int(hi) - int(lo) + 1):
                        for candidate in [val + offset, val - offset]:
                            if int(lo) <= candidate <= int(hi) and candidate not in [int(v) for v in result]:
                                val = candidate
                                break
                        if val != c:
                            break
                val = int(val)
            else:
                if val in [round(v, 4) for v in result]:
                    step = (hi - lo) / max(n * 10, 1)
                    val = round(val + step, 4)
                    if val > hi:
                        val = round(val - 2 * step, 4)
                val = float(val)
            result.append(val)

        return result

    def _print_score_map(self, phase, eps_values, ms_values, grid_scores, best_eps=None, best_ms=None):
        """Print a formatted score landscape map to the console.

        Args:
            phase: Current phase number
            eps_values: List of eps values (rows)
            ms_values: List of min_samples values (columns)
            grid_scores: dict (eps, ms) -> score
            best_eps: Best eps found so far (for highlighting)
            best_ms: Best min_samples found so far (for highlighting)
        """
        print(f"\n{'='*60}")
        print(f"  DBSCAN Optimizer - Phase {phase} Score Map")
        print(f"{'='*60}")

        # Column headers
        col_width = 8
        header = "eps\\ms  " + "".join(f"{int(m):>{col_width}}" for m in ms_values)
        print(header)
        print("-" * len(header))

        for eps in eps_values:
            row = f"{eps:<8.3f}"
            for ms in ms_values:
                score = grid_scores.get((eps, ms), None)
                if score is None:
                    cell = f"{'---':>{col_width}}"
                elif score == float('inf'):
                    cell = f"{'inf':>{col_width}}"
                else:
                    # Highlight best
                    if (best_eps is not None and best_ms is not None
                            and abs(eps - best_eps) < 1e-6 and int(ms) == int(best_ms)):
                        cell = f"{score:>{col_width-1}*}"
                    else:
                        cell = f"{score:>{col_width}.1f}"
                row += cell
            print(row)

        print(f"{'='*60}")
        print(f"  (* = current best)  Lower score = better")
        print(f"{'='*60}\n")

    def get_cluster_centers(self, positions):
        """Calculate centroid for each cluster.

        Args:
            positions: np.ndarray of shape (n_samples, 3)

        Returns:
            dict: cluster_id -> centroid (np.ndarray)
        """
        if self.labels is None:
            raise RuntimeError("Cluster labels not computed. Call fit_predict first.")
        
        centers = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels == cluster_id
            cluster_points = positions[cluster_mask]
            if len(cluster_points) > 0:
                centers[cluster_id] = np.mean(cluster_points, axis=0)
        
        return centers

    def get_cluster_sizes(self):
        """Get size of each cluster.

        Returns:
            dict: cluster_id -> number of points
        """
        if self.labels is None:
            raise RuntimeError("Cluster labels not computed. Call fit_predict first.")
        
        sizes = {}
        for cluster_id in range(self.n_clusters):
            sizes[cluster_id] = np.sum(self.labels == cluster_id)
        
        # Add noise count
        noise_count = np.sum(self.labels == -1)
        if noise_count > 0:
            sizes[-1] = noise_count
        
        return sizes

    def get_dominant_tags(self, cluster_id, file_ids, tag_data, top_n=5):
        """Get dominant tags for a cluster.

        Args:
            cluster_id: Cluster ID to analyze
            file_ids: List of file IDs in same order as labels
            tag_data: Dictionary mapping file_id to list of tags
            top_n: Number of top tags to return

        Returns:
            list: List of (tag, count) tuples
        """
        from collections import Counter
        
        if self.labels is None:
            raise RuntimeError("Cluster labels not computed. Call fit_predict first.")
        
        # Get file IDs in this cluster
        cluster_file_ids = [
            fid for i, fid in enumerate(file_ids)
            if self.labels[i] == cluster_id
        ]
        
        # Count tags
        tag_counts = Counter()
        for fid in cluster_file_ids:
            tags = tag_data.get(fid, [])
            tag_counts.update(tags)
        
        return tag_counts.most_common(top_n)
