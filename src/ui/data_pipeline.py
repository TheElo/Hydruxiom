"""Data loading & computation pipeline for the 3D tag map tab.

Mixin: methods operate on the ``TagMap3DTab`` instance (``self``). Covers data
loading, UMAP/DBSCAN computation, and recomputation workers. Moved here from
``tag_map_3d_tab.py`` to reduce its size without changing behavior.
"""
import time

import numpy as np

from src.ui.workers import WorkerThread
from src.ui.tag_map_utils import compile_tag_patterns


class DataPipelineMixin:
    @staticmethod
    def _pre_svd(tab):
        """Read the Pre-SVD toggle; return component count or None (off)."""
        cb = getattr(tab, 'pre_svd_checkbox', None)
        if cb is not None and cb.isChecked():
            spin = getattr(tab, 'pre_svd_components_spin', None)
            return int(spin.value()) if spin is not None else 64
        return None

    # ------------------------------------------------------------------
    # Pre-flight RAM check (formulas in benchmarks/ram_simulator.py)
    # ------------------------------------------------------------------
    def _preflight_ram_check(self, n_files_estimate, n_tags_override=None):
        """Estimate UMAP peak RAM before spawning the worker and warn if it
        will likely OOM. Non-blocking: console print + status bar message only,
        execution always continues. GUI-thread (status label access).

        Tag vocabulary size is taken from the previous session's interner when
        available (good proxy); otherwise 20k is assumed and flagged as such.
        Pass n_tags_override for exact counts (recompute path).
        """
        try:
            from benchmarks.ram_simulator import estimate_pipeline
        except ImportError:
            return  # never block a load on an optional check

        algorithm = self.algorithm_combo.currentText().lower()
        if not algorithm.startswith("umap"):
            return  # PCA is cheap; GPU path handled separately (Linux-only)

        if n_tags_override:
            n_tags, assumed = int(n_tags_override), "exact"
        else:
            interner = getattr(self, 'tag_interner', None)
            if interner is not None and len(getattr(interner, 'index_to_tag', [])) > 0:
                n_tags, assumed = len(interner.index_to_tag), "prev session"
            else:
                n_tags, assumed = 20_000, "assumed (no previous load)"

        subsample_size = self.subsample_size_spin.value() if (hasattr(self, 'subsample_checkbox') and self.subsample_checkbox.isChecked()) else None
        est = estimate_pipeline(n_files=n_files_estimate, n_tags=n_tags, algorithm="umap", subsample_size=subsample_size)
        full_est = estimate_pipeline(n_files=n_files_estimate, n_tags=n_tags, algorithm="umap")
        budget = est.budget_gib or float("inf")

        print(f"[RAM check] ~{n_files_estimate:,} files x {n_tags:,} tags ({assumed}): "
              f"current settings peak ~{est.peak_cpu_gib:.1f} GiB, full fit ~{full_est.peak_cpu_gib:.1f} GiB, budget ~{budget:.0f} GiB")

        # Non-blocking by design: warn via console + status bar, never stop the run.
        if subsample_size is None and est.peak_cpu_gib > budget:
            print(f"[RAM check] WARNING: likely OOM (~{est.peak_cpu_gib:.0f} GiB needed vs "
                  f"~{budget:.0f} GiB safe). Enable Subsample to cap memory.")
            self.status_label.setText(
                f"RAM warning: ~{est.peak_cpu_gib:.0f} GiB expected, only ~{budget:.0f} GiB safe - "
                f"consider enabling Subsample (continuing anyway)...")
        elif subsample_size is not None and full_est.peak_cpu_gib <= budget:
            print("[RAM check] Note: plain UMAP would fit in RAM; Subsample mode is slower "
                  "(transform cost scales with rows x subset size). Uncheck it if you want speed.")

    def start_loading(self):
        """Start the data loading and computation process."""
        if self._is_worker_busy():
            self.status_label.setText("Please wait - a process is already running.")
            return
        # Capture previous session's tag count BEFORE _release_session_data()
        # clears the interner; used as a proxy for the RAM pre-flight estimate.
        prev_n_tags = len(getattr(self.tag_interner, 'index_to_tag', [])) if self.tag_interner else None
        # Free the previous session's resources BEFORE starting the query so
        # old GPU buffers / node objects don't pile up on top of new data (OOM).
        self._release_session_data()
        self.load_button.setEnabled(False)
        self.recompute_button.setEnabled(False)
        self.recluster_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.load_session_button.setEnabled(False)
        self._set_cohort_action_buttons(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        # Tag the operation for per-op auto-deorphan (Settings -> DBSCAN Optimizer).
        self._last_op = "load"
        # Fresh full load produces new positions -> re-fit camera on next render
        self._camera_initialized = False

        # Smart Scale: apply the size-appropriate profile to the setting widgets
        # BEFORE spawning the worker, so UMAP/DBSCAN/visual params read below
        # reflect it. Uses max_files as the size estimate (exact count is applied
        # again in on_loading_finished once known). No-op when disabled.
        self._smart_scale_apply_for_load(self.max_files_spin.value())

        # Pre-flight RAM estimate (warns / offers Subsample before the heavy run)
        self._preflight_ram_check(self.max_files_spin.value(), n_tags_override=prev_n_tags)

        # Create worker
        def worker_func():
            return self._load_and_compute()
        
        self.worker = WorkerThread(worker_func)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.error.connect(self.on_loading_error)
        self.worker.start()

    @staticmethod
    def _compile_tag_patterns(tag_list):
        """Split a tag list into exact-match set and compiled wildcard patterns.

        Thin wrapper around :func:`src.ui.tag_map_utils.compile_tag_patterns`.
        """
        return compile_tag_patterns(tag_list)

    def _load_and_compute(self):
        """Load data and compute 3D positions."""
        from src.utils.utility_functions import ConnectToClient
        from src.data.loader import DataLoader
        from src.pipeline.vectorizer import Vectorizer
        from src.pipeline.reducer import Reducer
        from src.pipeline.clusterer import Clusterer
        from src.core.models import SceneGraph
        from src.core.tag_interner import TagInterner

        client_name = self.client_combo.currentText()
        # Chunk Size is a plain int attribute (edited in Settings -> Clients).
        # Path-specific load tuning (benchmarks/benchmark_api_io.py). The legacy
        # single self.chunk_size is the fallback for both paths.
        try:
            api_chunk_size = int(getattr(self, 'api_chunk_size', getattr(self, 'chunk_size', 8192)))
        except (TypeError, ValueError):
            api_chunk_size = 8192
        try:
            direct_chunk_size = int(getattr(self, 'direct_chunk_size', getattr(self, 'chunk_size', 4096)))
        except (TypeError, ValueError):
            direct_chunk_size = 4096
        max_files = self.max_files_spin.value()
        tag_service = self.tag_service_combo.currentText()
        algorithm = self.algorithm_combo.currentText().lower()
        n_neighbors = self.n_neighbors_spin.value()
        min_dist = self.min_dist_spin.value() / 100.0
        n_epochs = self.n_epochs_spin.value() if hasattr(self, 'n_epochs_spin') else None
        n_epochs = n_epochs if n_epochs > 0 else None  # 0 means auto
        learning_rate = self.learning_rate_spin.value() if hasattr(self, 'learning_rate_spin') else 1.0
        low_memory = self.low_memory
        n_jobs = self.n_jobs
        metric = self.metric_combo.currentText() if hasattr(self, 'metric_combo') else 'cosine'
        eps = self.eps_spin.value() / 100.0
        min_samples = self.min_samples_spin.value()
        node_size = float(self.min_size_spin.value()) / 10.0
        spread = float(self.spread_spin.value())
        whitelist = [t.strip() for t in self.whitelist_edit.text().split(',') if t.strip()]
        blacklist = [t.strip() for t in self.blacklist_edit.text().split(',') if t.strip()]
        drop_empty = getattr(self, 'drop_empty_files', False)
        query = self.query_edit.text().strip()
        # Min Tag Frequency is unit-aware (n / %). The '%' threshold needs the final
        # document count, which is only known after loading + filtering, so it is
        # resolved right before Vectorizer construction below.
        drop_universal = getattr(self, 'drop_universal', True)

        # Connect to client
        self.worker.progress.emit(5, f"Connecting to {client_name}...")
        client = ConnectToClient(client_name)

        # Build search tags (convert OR bracket groups to nested lists for API)
        search_tags = None
        if query:
            search_tags = self._query_to_api_tags(query)

        # Load data
        self.worker.progress.emit(10, "Loading file data...")
        use_direct_db = self.use_direct_db
        # Parallel loading (sweet spots from the benchmark): ~4 concurrent API
        # requests (~1.8x), ~2 direct-DB connections (~1.7x). 1 = legacy sequential.
        loader = DataLoader(
            client, chunk_size=api_chunk_size, client_name=client_name, use_direct_db=use_direct_db,
            api_chunk_size=api_chunk_size, direct_chunk_size=direct_chunk_size,
            api_max_workers=int(getattr(self, 'api_load_threads', 4)),
            direct_max_workers=int(getattr(self, 'direct_load_threads', 2)),
        )

        def progress_callback(chunk, tags, total):
            pct = int(10 + 40 * total / max(len(loader.all_file_ids), 1))
            self.worker.progress.emit(pct, f"Loaded {total} files...")

        # Pre-compile filter patterns (exact + wildcard) for tag-level filters.
        # Tags from Hydrus are already lowercase, and the exact sets /
        # wildcard patterns are compiled lowercase, so matching is done
        # directly (no per-tag .lower() needed).
        wl_exact, wl_patterns = self._compile_tag_patterns(whitelist)
        bl_exact, bl_patterns = self._compile_tag_patterns(blacklist)

        # Tokenize tags (optional): convert strings to integer indices once,
        # carried through the pipeline as ints to reduce RAM and string hashing.
        # Build as a local first; only swap into self.tag_interner after the
        # load succeeds so a failed load (e.g. 0 files) doesn't orphan the
        # old scene graph's indices against an empty interner.
        tokenize = getattr(self, 'tokenize', True)
        new_interner = TagInterner() if tokenize else None

        # Per-chunk transform (string phase): applied DURING load so each file
        # is stored exactly once in its final form (no separate filter/tokenize
        # passes over the full dataset).
        # - Whitelist (exact + wildcard) runs here: it is a KEEP-set, so it
        #   must be fully resolved before tokenization (non-matching tags are
        #   discarded and never reach the interner).
        # - Blacklist wildcards run here (regex needs strings).
        # - Blacklist EXACT is deferred to the int pass below when tokenize is
        #   ON (removals commute; int set checks are cheaper than string ones).
        #   When tokenize is OFF there is no int pass, so it runs here.
        def _transform(fid, tags):
            # Whitelist: keep ONLY tags that match (limits attribute pool)
            if wl_exact or wl_patterns:
                kept_tags = []
                for tag in tags:
                    if tag in wl_exact:
                        kept_tags.append(tag)
                        continue
                    for compiled in wl_patterns:
                        if compiled.match(tag):
                            kept_tags.append(tag)
                            break
                tags = kept_tags

            # Blacklist: remove matching tags from the file's tag list
            if bl_patterns:
                cleaned_tags = []
                for tag in tags:
                    should_remove = False
                    for compiled in bl_patterns:
                        if compiled.match(tag):
                            should_remove = True
                            break
                    if not should_remove:
                        cleaned_tags.append(tag)
                tags = cleaned_tags
            elif bl_exact and new_interner is None:
                # Tokenize OFF: no int pass, apply exact blacklist here too.
                tags = [tag for tag in tags if tag not in bl_exact]

            # Keep files with no tags remaining unless the user opted to
            # drop them (they render as untagged nodes at the origin).
            if not tags and drop_empty:
                return None

            if new_interner is not None:
                tags = new_interner.tokenize_list(tags)
            return tags

        loader.load_in_chunks(callback=progress_callback, tag_service=tag_service, max_files=max_files, search_tags=search_tags, transform=_transform)
        tag_data = loader.get_tag_data()

        # Post-load int pass: remove deferred blacklist EXACT tags by integer
        # index (only when tokenize is ON; the string phase skipped them so
        # they are present in the vocabulary).
        if new_interner is not None and bl_exact:
            self.worker.progress.emit(50, "Filtering tags (blacklist, int pass)...")
            # Non-creating lookup: a blacklist tag that appears in no file has
            # no index and is a no-op (nothing to remove).
            bl_exact_indices = {new_interner.tag_to_index[t] for t in bl_exact if t in new_interner.tag_to_index}
            if bl_exact_indices:
                for fid in list(tag_data.keys()):
                    tags = [i for i in tag_data[fid] if i not in bl_exact_indices]
                    # The int pass can empty a file that survived the string
                    # phase (all its tags were blacklisted) -> re-apply toggle.
                    if not tags and drop_empty:
                        del tag_data[fid]
                    else:
                        tag_data[fid] = tags

        if new_interner is not None:
            print(f"[Tokenize] Interned {len(new_interner.index_to_tag)} unique tags")

        if not tag_data:
            # No files matched the query/filters. Keep the old interner so any
            # existing scene graph (whose node indices were built from it) stays
            # consistent, and return a sentinel so the UI shows a friendly
            # status message instead of an error.
            print("[DataLoader] No files matched the current query/filters.")
            return None, None

        # Swap in the new interner only now that the load succeeded.
        self.tag_interner = new_interner

        # Vectorize
        self.worker.progress.emit(50, "Vectorizing tags...")
        reverse_vocab = self.tag_interner.index_to_tag if self.tag_interner else None
        min_doc_freq = (self._resolve_min_doc_freq(len(tag_data))
                        if hasattr(self, '_resolve_min_doc_freq')
                        else (self.min_doc_freq_spin.value() if hasattr(self, 'min_doc_freq_spin') else 3))
        vec = Vectorizer(min_doc_freq=min_doc_freq, tokenized=bool(self.tag_interner), reverse_vocab=reverse_vocab, drop_universal_tags=drop_universal)
        _t_vec = time.perf_counter()
        sparse_matrix, file_ids = vec.create_vectors(tag_data)
        print(f"[Timing] Vectorizing took {time.perf_counter() - _t_vec:.2f}s")

        # Reduce dimensionality
        self.worker.progress.emit(60, f"Applying {algorithm.upper()}...")
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
            subsample_size=subsample_size,
            chunked_transform=self.chunked_transform_checkbox.isChecked(),
            pre_svd_components=self._pre_svd(self)
        )
        _t_red = time.perf_counter()
        positions = red.fit_transform(sparse_matrix)
        print(f"[Timing] {algorithm.upper()} reduction took {time.perf_counter() - _t_red:.2f}s")

        # Release the fitted reducer and the sparse matrix now that positions
        # are computed. UMAP retains the fuzzy simplicial set + embedding,
        # which would otherwise stay alive through clustering + scene-graph
        # building and inflate peak memory (Reducer.transform() is never used).
        red.model = None
        del red
        del sparse_matrix

        # Cosmetic status: positions are ready, before clustering
        self.worker.progress.emit(70, "Post-processing positions...")

        # Cluster
        self.worker.progress.emit(80, "Clustering...")
        clust = Clusterer(eps=eps, min_samples=min_samples)
        _t_clust = time.perf_counter()
        cluster_positions = self._maybe_normalize_positions(positions)
        cluster_labels = clust.fit_predict(cluster_positions)
        print(f"[Timing] Clustering took {time.perf_counter() - _t_clust:.2f}s")

        # Build scene graph
        self.worker.progress.emit(90, "Building scene graph...")
        scene = SceneGraph()
        _t_scene = time.perf_counter()
        scene.build_from_data(file_ids, positions, tag_data, cluster_labels,
                             node_size=node_size,
                             tokenized=bool(self.tag_interner), reverse_vocab=reverse_vocab)
        print(f"[Timing] Building scene graph took {time.perf_counter() - _t_scene:.2f}s")

        self.worker.progress.emit(100, "Complete!")
        # Force reclamation of pipeline temporaries (UMAP internals may form
        # reference cycles) before the UI renders the new scene.
        import gc
        gc.collect()
        return scene, tag_data

    def start_recompute(self):
        """Start the recomputation process using existing tag_data."""
        if self.tag_data is None:
            self.status_label.setText("Error: No data loaded. Load data first.")
            return
        
        self.load_button.setEnabled(False)
        self.recompute_button.setEnabled(False)
        self.recluster_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.load_session_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Recomputing...")
        # Recompute re-runs UMAP/PCA on the same files; tag it as "load" so the
        # Load & Compute auto-deorphan checkbox covers both.
        self._last_op = "load"
        # Recompute leaves every node unclustered (noise) by design — skip the
        # fresh auto-split cycle that would otherwise log "No cohorts to split".
        self._auto_split_allowed = False

        # Pre-flight RAM estimate with exact counts (tag_data + interner are known here)
        n_tags_exact = len(getattr(self.tag_interner, 'index_to_tag', [])) if self.tag_interner else None
        self._preflight_ram_check(len(self.tag_data), n_tags_override=n_tags_exact)

        def worker_func():
            return self._recompute()

        self.worker = WorkerThread(worker_func)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.error.connect(self.on_loading_error)
        self.worker.start()

    def _recompute(self):
        """Recompute 3D positions using existing tag_data with new algorithm/cluster settings."""
        from src.pipeline.vectorizer import Vectorizer
        from src.pipeline.reducer import Reducer
        from src.pipeline.clusterer import Clusterer
        from src.core.models import SceneGraph

        tag_data = self.tag_data
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
        # Unit-aware Min Tag Frequency (n / %); tag_data is already loaded here.
        min_doc_freq = (self._resolve_min_doc_freq(len(tag_data))
                        if hasattr(self, '_resolve_min_doc_freq')
                        else (self.min_doc_freq_spin.value() if hasattr(self, 'min_doc_freq_spin') else 3))
        drop_universal = getattr(self, 'drop_universal', True)

        if not tag_data:
            raise RuntimeError("No tag data available")

        # Vectorize
        self.worker.progress.emit(10, "Vectorizing tags...")
        reverse_vocab = self.tag_interner.index_to_tag if self.tag_interner else None
        vec = Vectorizer(min_doc_freq=min_doc_freq, tokenized=bool(self.tag_interner), reverse_vocab=reverse_vocab, drop_universal_tags=drop_universal)
        _t_vec = time.perf_counter()
        sparse_matrix, file_ids = vec.create_vectors(tag_data)
        print(f"[Timing] Vectorizing took {time.perf_counter() - _t_vec:.2f}s")

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
            subsample_size=subsample_size,
            chunked_transform=self.chunked_transform_checkbox.isChecked(),
            pre_svd_components=self._pre_svd(self)
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

        # Recompute is UMAP-only: skip DBSCAN so the user can tune clustering
        # separately via "Regroup". Mark all nodes as unclustered (noise).
        self.worker.progress.emit(70, "Positions ready (use Regroup to cluster)...")
        cluster_labels = np.full(len(file_ids), -1, dtype=int)

        # Build scene graph
        self.worker.progress.emit(85, "Building scene graph...")
        scene = SceneGraph()
        _t_scene = time.perf_counter()
        scene.build_from_data(file_ids, positions, tag_data, cluster_labels,
                             node_size=node_size,
                             tokenized=bool(self.tag_interner), reverse_vocab=reverse_vocab)
        print(f"[Timing] Building scene graph took {time.perf_counter() - _t_scene:.2f}s")

        self.worker.progress.emit(100, "Recompute complete!")
        import gc
        gc.collect()
        return scene, tag_data
