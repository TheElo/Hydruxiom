"""Cohort operations for the 3D tag map tab.

Mixin: methods operate on the ``TagMap3DTab`` instance (``self``). Covers cut,
pop, recluster, and DBSCAN-optimize worker operations. Moved here from
``tag_map_3d_tab.py`` to reduce its size without changing behavior.
"""
import time

import numpy as np

from src.ui.workers import WorkerThread


class CohortOpsMixin:
    def _cut_selected_cohort(self):
        """Cut out the selected cohort - keep only its nodes, remove everything else.

        Takes the nodes in the currently selected cohort, re-runs the
        vectorize -> reduce -> cluster pipeline on just those files, and
        renders the resulting sub-cohorts as a new scene (removing all other nodes).
        """
        if self.selected_cluster_id is None or not hasattr(self, 'node_list') or not self.node_list:
            self.status_label.setText("Select a cohort first to cut it out.")
            return
        if self._is_worker_busy():
            self.status_label.setText("Please wait - a process is already running.")
            return

        # Collect member indices of the selected cohort (vectorized mask)
        import numpy as np
        idx = np.where(self.scene_graph.cluster_ids == self.selected_cluster_id)[0]
        if len(idx) < 2:
            self.status_label.setText("Cohort too small to cut out (need 2+ files).")
            return

        # Build sub tag_data from the cohort's members (tags live in self.tag_data)
        sub_tag_data = {}
        for i in idx:
            fid = self.scene_graph.file_ids[i]
            tags = (self.tag_data or {}).get(fid, [])
            sub_tag_data[fid] = list(tags)

        self.status_label.setText(f"Cutting out cohort {self.selected_cluster_id} ({len(idx)} files)...")
        self._set_cohort_action_buttons(False)

        def worker_func():
            return self._split_compute(sub_tag_data)

        self.worker = WorkerThread(worker_func)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.error.connect(self.on_loading_error)
        self.worker.start()

    def _split_compute(self, sub_tag_data):
        """Run the algorithm pipeline on a subset of files to create sub-cohorts."""
        from src.pipeline.vectorizer import Vectorizer
        from src.pipeline.reducer import Reducer
        from src.pipeline.clusterer import Clusterer
        from src.core.models import SceneGraph

        algorithm = self.algorithm_combo.currentText().lower()
        n_neighbors = self.n_neighbors_spin.value()
        min_dist = self.min_dist_spin.value() / 100.0
        n_epochs = self.n_epochs_spin.value() if hasattr(self, 'n_epochs_spin') else None
        n_epochs = n_epochs if n_epochs and n_epochs > 0 else None
        learning_rate = self.learning_rate_spin.value() if hasattr(self, 'learning_rate_spin') else 1.0
        low_memory = self.low_memory
        n_jobs = self.n_jobs
        metric = self.metric_combo.currentText() if hasattr(self, 'metric_combo') else 'cosine'
        eps = self.eps_spin.value() / 100.0
        min_samples = self.min_samples_spin.value()
        node_size = float(self.min_size_spin.value()) / 10.0
        min_doc_freq = self.min_doc_freq_spin.value() if hasattr(self, 'min_doc_freq_spin') else 3
        drop_universal = getattr(self, 'drop_universal', True)

        # Vectorize the subset
        self.worker.progress.emit(10, "Vectorizing sub-cohort...")
        reverse_vocab = self.tag_interner.index_to_tag if self.tag_interner else None
        vec = Vectorizer(min_doc_freq=min_doc_freq, tokenized=bool(self.tag_interner), reverse_vocab=reverse_vocab, drop_universal_tags=drop_universal)
        _t_vec = time.perf_counter()
        sparse_matrix, file_ids = vec.create_vectors(sub_tag_data)
        print(f"[Timing] Vectorizing sub-cohort took {time.perf_counter() - _t_vec:.2f}s")

        # Reduce dimensionality
        self.worker.progress.emit(40, f"Applying {algorithm.upper()}...")
        subsample_size = self.subsample_size_spin.value() if self.subsample_checkbox.isChecked() else None
        red = Reducer(
            algorithm=algorithm,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            low_memory=low_memory,
            metric=metric,
            n_jobs=n_jobs,
            subsample_size=subsample_size
        )
        _t_red = time.perf_counter()
        positions = red.fit_transform(sparse_matrix)
        print(f"[Timing] {algorithm.upper()} reduction took {time.perf_counter() - _t_red:.2f}s")

        # Release the fitted reducer and the sparse matrix now that positions
        # are computed (see _load_and_compute). Lowers peak memory during
        # clustering + scene-graph building.
        red.model = None
        del red
        del sparse_matrix

        # Cosmetic status: positions are ready, before clustering
        self.worker.progress.emit(55, "Post-processing positions...")

        # Cluster the subset
        self.worker.progress.emit(70, "Clustering sub-cohort...")
        clust = Clusterer(eps=eps, min_samples=min_samples)
        _t_clust = time.perf_counter()
        cluster_positions = self._maybe_normalize_positions(positions)
        cluster_labels = clust.fit_predict(cluster_positions)
        print(f"[Timing] Clustering sub-cohort took {time.perf_counter() - _t_clust:.2f}s")

        # Build scene graph for sub-cohorts
        self.worker.progress.emit(85, "Building sub-scene...")
        scene = SceneGraph()
        _t_scene = time.perf_counter()
        scene.build_from_data(file_ids, positions, sub_tag_data, cluster_labels,
                              node_size=node_size,
                              tokenized=bool(self.tag_interner), reverse_vocab=reverse_vocab)
        print(f"[Timing] Building sub-scene took {time.perf_counter() - _t_scene:.2f}s")

        self.worker.progress.emit(100, "Split complete!")
        import gc
        gc.collect()
        return scene, sub_tag_data

    def _pop_selected_cohort(self):
        """Pop the selected cohort - remove it from the view, keep everything else.

        Inverse of Cut out: the selected cohort's nodes are removed and the
        scene is rebuilt from the remaining nodes. Positions, colors, and
        cluster labels of the remaining nodes stay unchanged.
        """
        if self.selected_cluster_id is None or not hasattr(self, 'node_list') or not self.node_list:
            self.status_label.setText("Select a cohort first to pop it.")
            return
        if self._is_worker_busy():
            self.status_label.setText("Please wait - a process is already running.")
            return

        cluster_id = self.selected_cluster_id
        # Count removed/remaining via the label array (no per-node iteration)
        import numpy as np
        removed_count = int(np.sum(self.scene_graph.cluster_ids == cluster_id))
        total = len(self.scene_graph.file_ids)
        if total - removed_count <= 0:
            self.status_label.setText("Cannot pop: this is the only cohort in the view.")
            return

        # Clear the selection (the cohort is about to disappear)
        self.clear_selection()

        self.status_label.setText(f"Popping cohort {cluster_id} ({removed_count} files)...")
        self._set_cohort_action_buttons(False)

        def worker_func():
            return self._pop_compute(cluster_id, removed_count)

        self.worker = WorkerThread(worker_func)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.error.connect(self.on_loading_error)
        self.worker.start()

    def _pop_compute(self, cluster_id, removed_count):
        """Remove the popped cohort from the scene (positions unchanged).

        Fast path: slices the SoA arrays once via SceneGraph.without_cluster()
        (no per-node object creation) and remaps surviving cluster indices.
        Only the popped cohort's file IDs are dropped from
        tag_data.
        """
        self.worker.progress.emit(40, "Removing cohort...")

        # Reuse existing scene; drop only the popped cluster's nodes/clusters.
        scene = self.scene_graph.without_cluster(cluster_id)

        # Filter tag_data: drop only the popped cohort's file IDs (no list copies).
        if self.tag_data is not None:
            import numpy as np
            idx = np.where(self.scene_graph.cluster_ids == cluster_id)[0]
            removed_ids = {self.scene_graph.file_ids[i] for i in idx}
            self.tag_data = {fid: tags for fid, tags in self.tag_data.items()
                             if fid not in removed_ids}

        self.worker.progress.emit(100, f"Cohort popped! ({removed_count} files removed)")
        return scene, self.tag_data

    def _recluster_selection(self):
        """Re-cluster the selected cohort using existing positions.

        Applies DBSCAN on the selected nodes' current positions to identify
        smaller sub-cohorts within the selection. Positions stay unchanged;
        only coloring and cohort label grouping change.
        """
        if self.selected_cluster_id is None or not hasattr(self, 'node_list') or not self.node_list:
            self.status_label.setText("Select a cohort first to re-cluster it.")
            return
        if self._is_worker_busy():
            self.status_label.setText("Please wait - a process is already running.")
            return

        import numpy as np
        cluster_nodes = np.where(self.scene_graph.cluster_ids == self.selected_cluster_id)[0]
        if len(cluster_nodes) < 2:
            self.status_label.setText("Cohort too small to re-cluster (need 2+ files).")
            return

        self.status_label.setText(f"Re-clustering cohort {self.selected_cluster_id} ({len(cluster_nodes)} files)...")
        self._pending_recluster = True
        self._set_cohort_action_buttons(False)

        def worker_func():
            return self._recluster_compute(cluster_nodes)

        self.worker = WorkerThread(worker_func)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.error.connect(self.on_loading_error)
        self.worker.start()

    def _recluster_compute(self, cluster_nodes):
        """Run DBSCAN on the selected nodes' existing positions.

        Returns a scene with ALL nodes preserved; only the selected cohort's
        nodes are re-assigned into smaller sub-cohort labels/colors.
        """
        import numpy as np
        from src.pipeline.clusterer import Clusterer
        from src.core.models import SceneGraph

        eps = self.eps_spin.value() / 100.0
        min_samples = self.min_samples_spin.value()
        node_size = float(self.min_size_spin.value()) / 10.0

        # Use existing positions (no re-reduce); cluster_nodes are array indices
        spread = float(self.spread_spin.value())
        positions = self.scene_graph.positions[cluster_nodes] * spread

        # Use SEPARATE sub-clustering settings (independent from global eps/min).
        # The selected cohort is already a dense sub-region; the global eps would
        # treat it as one cluster and produce no split. The user controls these
        # independently via Sub EPS / Sub Min Samples.
        sub_eps = self.sub_eps_spin.value() / 100.0
        sub_min_samples = self.sub_min_samples_spin.value()
        self.worker.progress.emit(40, "Clustering selection (sub-cohorts)...")
        clust = Clusterer(eps=sub_eps, min_samples=sub_min_samples)
        cluster_positions = self._maybe_normalize_positions(positions)
        cluster_labels = np.asarray(clust.fit_predict(cluster_positions))

        # Build sub tag_data from the selected members (tags live in self.tag_data)
        file_ids_arr = self.scene_graph.file_ids
        sub_tag_data = {}
        for i in cluster_nodes:
            fid = file_ids_arr[i]
            tags = (self.tag_data or {}).get(fid, [])
            sub_tag_data[fid] = list(tags)

        # Build a FULL scene: keep all nodes, only re-label the selected cohort.
        # Non-selected nodes keep their original positions and cluster labels.
        n = len(file_ids_arr)
        all_file_ids = list(file_ids_arr)
        all_positions = self.scene_graph.positions * spread
        all_tag_data = {fid: list((self.tag_data or {}).get(fid, [])) for fid in file_ids_arr}

        selected_ids = set(int(file_ids_arr[i]) for i in cluster_nodes)
        # Map selected node -> new sub-cohort label (by index into the selection)
        selected_label_map = {}
        for k, i in enumerate(cluster_nodes):
            selected_label_map[int(file_ids_arr[i])] = int(cluster_labels[k])

        # Remap sub-cohort labels to UNIQUE IDs that don't collide with existing
        # cluster IDs of non-selected nodes. DBSCAN returns 0-based labels which
        # may overlap with existing cluster IDs, causing color/group confusion.
        existing_ids = set(np.unique(self.scene_graph.cluster_ids[self.scene_graph.cluster_ids != -1]).tolist())
        max_existing = max(existing_ids) if existing_ids else -1
        sub_label_remap = {}
        next_id = max_existing + 1
        for label in cluster_labels:
            lab = int(label)
            if lab == -1:
                sub_label_remap[lab] = -1
            elif lab not in sub_label_remap:
                sub_label_remap[lab] = next_id
                next_id += 1

        # New labels: selected members get remapped sub-cohort ids, others keep theirs.
        all_labels = self.scene_graph.cluster_ids.astype(np.int32).copy()
        for k, i in enumerate(cluster_nodes):
            all_labels[i] = sub_label_remap[int(cluster_labels[k])]

        self.worker.progress.emit(90, "Building sub-scene...")
        reverse_vocab = self.tag_interner.index_to_tag if self.tag_interner else None
        scene = SceneGraph()
        scene.build_from_data(
            all_file_ids,
            all_positions,
            all_tag_data,
            all_labels,
            node_size=node_size,
            tokenized=bool(self.tag_interner),
            reverse_vocab=reverse_vocab,
        )

        # Preserve original colors for non-selected nodes so their appearance
        # stays exactly the same; only the split sub-cohorts get new colors.
        if self.scene_graph is not None:
            old = self.scene_graph
            for i in range(n):
                fid = all_file_ids[i]
                if int(fid) not in selected_ids:
                    oi = old.file_id_to_index.get(fid)
                    if oi is not None:
                        scene.colors[i] = old.colors[oi]

        self.worker.progress.emit(100, "Re-cluster complete!")
        return scene, all_tag_data

    def _on_normalize_toggled(self, state):
        """Handle the normalize positions toggle change.

        When toggled, re-run DBSCAN on current positions so the effect of
        normalization is immediately visible (no re-reduce needed).
        """
        self.normalize_positions = self.normalize_checkbox.isChecked()
        self.save_settings()
        if self.tag_data is not None and hasattr(self, 'node_list') and self.node_list:
            self.start_recluster()

    def start_recluster(self):
        """Start re-applying DBSCAN on all current positions (no re-reduce)."""
        if self.tag_data is None or not hasattr(self, 'node_list') or not self.node_list:
            self.status_label.setText("Error: No data loaded. Load data first.")
            return

        self.load_button.setEnabled(False)
        self.recompute_button.setEnabled(False)
        self.recluster_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.load_session_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Re-applying DBSCAN...")
        self._pending_recluster = True

        def worker_func():
            return self._recluster_all()

        self.worker = WorkerThread(worker_func)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.error.connect(self.on_loading_error)
        self.worker.start()

    def _recluster_all(self):
        """Re-run DBSCAN on ALL existing node positions.

        Positions stay unchanged (no UMAP/PCA re-run); only cluster labels,
        colors, and cohort grouping change. Uses current eps/min_samples.
        """
        import numpy as np
        from src.pipeline.clusterer import Clusterer
        from src.core.models import SceneGraph

        eps = self.eps_spin.value() / 100.0
        min_samples = self.min_samples_spin.value()
        node_size = float(self.min_size_spin.value()) / 10.0
        spread = float(self.spread_spin.value())

        # Use existing positions for ALL nodes (no re-reduce) — direct array access
        all_file_ids = list(self.scene_graph.file_ids)
        all_positions = self.scene_graph.positions * spread
        all_tag_data = {fid: list((self.tag_data or {}).get(fid, [])) for fid in all_file_ids}

        self.worker.progress.emit(40, "Clustering all positions...")
        _t_clust = time.perf_counter()
        clust = Clusterer(eps=eps, min_samples=min_samples)
        cluster_positions = self._maybe_normalize_positions(all_positions)
        cluster_labels = clust.fit_predict(cluster_positions)
        print(f"[Timing] DBSCAN re-cluster took {time.perf_counter() - _t_clust:.2f}s")

        self.worker.progress.emit(90, "Building scene graph...")
        reverse_vocab = self.tag_interner.index_to_tag if self.tag_interner else None
        scene = SceneGraph()
        scene.build_from_data(
            all_file_ids,
            all_positions,
            all_tag_data,
            cluster_labels,
            node_size=node_size,
            tokenized=bool(self.tag_interner),
            reverse_vocab=reverse_vocab,
        )

        self.worker.progress.emit(100, "DBSCAN re-applied!")
        return scene, all_tag_data

    def _maybe_normalize_positions(self, positions):
        """Normalize positions before DBSCAN if the toggle is enabled.

        When enabled, positions are centered and std-scaled so eps behaves
        consistently across datasets with different file counts / reducer
        scales. Returns the (possibly normalized) positions array.
        """
        if not getattr(self, 'normalize_positions', True):
            return positions
        import numpy as np
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            return positions
        mean = positions.mean(axis=0)
        centered = positions - mean
        std = centered.std(axis=0)
        std[std == 0] = 1.0
        return centered / std

    def start_deorphan(self):
        """Assign each noise (-1) node to the cohort of its nearest non-noise node."""
        if self.tag_data is None or not hasattr(self, 'node_list') or not self.node_list:
            self.status_label.setText("Error: No data loaded. Load data first.")
            return

        self.load_button.setEnabled(False)
        self.recompute_button.setEnabled(False)
        self.recluster_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.deorphan_button.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.load_session_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Deorphaning: assigning orphans to nearest cohort...")
        # Deorphan only re-labels orphans (positions unchanged), so the lighter
        # in-place color refresh path is used on completion (like _recluster).
        self._pending_recluster = True
        self._pending_deorphan = True

        def worker_func():
            return self._deorphan()

        self.worker = WorkerThread(worker_func)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.error.connect(self.on_loading_error)
        self.worker.start()

    def _deorphan(self):
        """Re-label noise nodes to the cluster of their nearest non-noise neighbor.

        Uses scipy.spatial.cKDTree for O(n log n) nearest-neighbor queries.
        Positions are unchanged; only cluster labels are updated.
        Returns (scene, tag_data) in the same format as _recluster_all.
        """
        import numpy as np
        from scipy.spatial import cKDTree
        from src.core.models import SceneGraph

        scene = self.scene_graph
        positions = np.asarray(scene.positions, dtype=float)
        cluster_ids = np.asarray(scene.cluster_ids, dtype=int)
        file_ids = list(scene.file_ids)
        node_size = float(self.min_size_spin.value()) / 10.0

        noise_mask = cluster_ids == -1
        n_orphans = int(noise_mask.sum())
        if n_orphans == 0:
            self.worker.progress.emit(100, "No orphans to assign.")
            return scene, {fid: list((self.tag_data or {}).get(fid, [])) for fid in file_ids}

        # Build KD-tree on non-noise positions only.
        anchored_positions = positions[~noise_mask]
        anchored_labels = cluster_ids[~noise_mask]
        tree = cKDTree(anchored_positions)

        # Query nearest anchor for each orphan.
        orphan_positions = positions[noise_mask]
        _, nearest_idx = tree.query(orphan_positions, k=1)
        new_labels_for_orphans = anchored_labels[nearest_idx]

        # Apply: replace -1 with the assigned cluster id.
        updated_labels = cluster_ids.copy()
        updated_labels[noise_mask] = new_labels_for_orphans

        self.worker.progress.emit(70, f"Assigned {n_orphans} orphans to nearest cohorts.")

        # Rebuild scene graph with updated labels (positions unchanged).
        all_tag_data = {fid: list((self.tag_data or {}).get(fid, [])) for fid in file_ids}
        reverse_vocab = self.tag_interner.index_to_tag if self.tag_interner else None
        new_scene = SceneGraph()
        new_scene.build_from_data(
            file_ids,
            positions,
            all_tag_data,
            updated_labels,
            node_size=node_size,
            tokenized=bool(self.tag_interner),
            reverse_vocab=reverse_vocab,
        )

        self.worker.progress.emit(100, f"Deorphaned {n_orphans} nodes.")
        return new_scene, all_tag_data

    def start_optimize(self):
        """Start the DBSCAN optimizer to find ideal eps/min_samples."""
        if self.tag_data is None or not hasattr(self, 'node_list') or not self.node_list:
            self.status_label.setText("Error: No data loaded. Load data first.")
            return

        self.load_button.setEnabled(False)
        self.recompute_button.setEnabled(False)
        self.recluster_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.load_session_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Optimizing DBSCAN parameters...")
        self._pending_recluster = True

        def worker_func():
            return self._optimize_dbscan()

        self.worker = WorkerThread(worker_func)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.error.connect(self.on_loading_error)
        self.worker.start()

    def _optimize_dbscan(self):
        """Search for the ideal eps/min_samples combination.

        Runs DBSCAN multiple times across the configured search ranges and
        applies the best-found settings to the current positions (no re-reduce).
        Goal: reduce non-cohorted (noise) nodes and split disproportionately
        large cohorts.
        """
        import numpy as np
        from src.pipeline.clusterer import Clusterer
        from src.core.models import SceneGraph

        # Optimizer parameters (from settings dialog)
        max_cohort_size = getattr(self, 'opt_max_cohort_size', 500)
        max_noise_ratio = getattr(self, 'opt_max_noise_ratio', 10)
        max_attempts = getattr(self, 'opt_max_attempts', 60)
        eps_min = getattr(self, 'opt_eps_min', 5) / 100.0
        eps_max = getattr(self, 'opt_eps_max', 100) / 100.0
        min_samples_min = getattr(self, 'opt_min_samples_min', 2)
        min_samples_max = getattr(self, 'opt_min_samples_max', 30)

        node_size = float(self.min_size_spin.value()) / 10.0
        spread = float(self.spread_spin.value())

        # Use existing positions for ALL nodes (no re-reduce) — direct array access
        all_file_ids = list(self.scene_graph.file_ids)
        all_positions = self.scene_graph.positions * spread
        all_tag_data = {fid: list((self.tag_data or {}).get(fid, [])) for fid in all_file_ids}

        self.worker.progress.emit(30, "Searching DBSCAN parameters...")

        def progress_callback(attempt, total, message):
            pct = 30 + int(60 * attempt / max(total, 1))
            self.worker.progress.emit(pct, f"Optimizing ({attempt}/{total}) {message}")

        clust = Clusterer(eps=eps_min, min_samples=min_samples_min)
        best = clust.optimize(
            all_positions,
            max_cohort_size=max_cohort_size,
            max_noise_ratio=max_noise_ratio,
            eps_min=eps_min,
            eps_max=eps_max,
            min_samples_min=min_samples_min,
            min_samples_max=min_samples_max,
            max_attempts=max_attempts,
            progress_callback=progress_callback,
        )

        best_eps = best["eps"]
        best_min_samples = best["min_samples"]
        eval_result = best["evaluation"]

        # Apply the best settings to the UI
        self.eps_spin.setValue(int(round(best_eps * 100.0)))
        self.min_samples_spin.setValue(best_min_samples)

        # Re-run DBSCAN with the best settings on current positions
        self.worker.progress.emit(90, "Applying best DBSCAN settings...")
        _t_clust = time.perf_counter()
        clust = Clusterer(eps=best_eps, min_samples=best_min_samples)
        cluster_positions = self._maybe_normalize_positions(all_positions)
        cluster_labels = clust.fit_predict(cluster_positions)
        print(f"[Timing] DBSCAN optimize re-cluster took {time.perf_counter() - _t_clust:.2f}s")

        self.worker.progress.emit(95, "Building scene graph...")
        reverse_vocab = self.tag_interner.index_to_tag if self.tag_interner else None
        scene = SceneGraph()
        scene.build_from_data(
            all_file_ids,
            all_positions,
            all_tag_data,
            cluster_labels,
            node_size=node_size,
            tokenized=bool(self.tag_interner),
            reverse_vocab=reverse_vocab,
        )

        # Report the optimization result
        print(f"[Optimize] Best eps={best_eps:.3f}, min_samples={best_min_samples} "
              f"({best['attempts']} attempts)")
        print(f"[Optimize] Noise: {eval_result['noise_count']} "
              f"({eval_result['noise_ratio']:.1%}), "
              f"Max cohort: {eval_result['max_cohort_size']}, "
              f"Oversized: {eval_result['oversized_cohorts']}")

        self.worker.progress.emit(100, "DBSCAN optimized!")
        return scene, all_tag_data
