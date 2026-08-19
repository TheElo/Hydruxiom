"""Settings persistence for the 3D tag map tab.

Mixin: methods operate on the ``TagMap3DTab`` instance (``self``). Moved here
from ``tag_map_3d_tab.py`` to reduce its size without changing behavior.
"""
import json
import os
import tempfile

from PySide6.QtCore import QTimer

from src.ui.tag_map_utils import SETTINGS_FILE


class SettingsPersistenceMixin:
    def load_settings(self):
        """Load settings from JSON file."""
        if not os.path.exists(SETTINGS_FILE):
            return

        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)

            # Client settings
            client_idx = self.client_combo.findText(settings.get("client", ""))
            if client_idx >= 0:
                self.client_combo.setCurrentIndex(client_idx)
            # Chunk Size is now a plain int attribute (its spinbox lives in the
            # Settings window -> Clients tab, not on this widget tree).
            # Path-specific chunk sizes. Back-compat: pre-split settings only have a
            # single "chunk_size" — use it for the API path and default direct-DB to 4096.
            try:
                _legacy_chunk = int(settings.get("chunk_size", 8192))
            except (TypeError, ValueError):
                _legacy_chunk = 8192
            self.chunk_size = _legacy_chunk
            try:
                self.api_chunk_size = int(settings.get("api_chunk_size", _legacy_chunk))
            except (TypeError, ValueError):
                self.api_chunk_size = _legacy_chunk
            try:
                self.direct_chunk_size = int(settings.get("direct_chunk_size", 4096))
            except (TypeError, ValueError):
                self.direct_chunk_size = 4096
            self.max_files_spin.setValue(settings.get("max_files", 20000))
            # Populate tag services dynamically for the selected client, then
            # restore the saved tag service selection.
            self._populate_tag_services(self.client_combo.currentText())
            tag_service_idx = self.tag_service_combo.findText(settings.get("tag_service", "auto2"))
            if tag_service_idx >= 0:
                self.tag_service_combo.setCurrentIndex(tag_service_idx)

            # Direct DB mode toggle
            self.use_direct_db = settings.get("use_direct_db", False)

            # Auto-load last data setting
            if hasattr(self, 'auto_load_checkbox'):
                self.auto_load_checkbox.setChecked(settings.get("auto_load_last_data", True))

            # Algorithm settings
            algo_idx = self.algorithm_combo.findText(settings.get("algorithm", "UMAP"))
            if algo_idx >= 0:
                self.algorithm_combo.setCurrentIndex(algo_idx)
            self.n_neighbors_spin.setValue(settings.get("n_neighbors", 12))
            self.min_dist_spin.setValue(settings.get("min_dist", 1))
            self.n_epochs_spin.setValue(settings.get("n_epochs", 64))
            self.learning_rate_spin.setValue(settings.get("learning_rate", 2.0))
            metric_idx = self.metric_combo.findText(settings.get("metric", "cosine"))
            if metric_idx >= 0:
                self.metric_combo.setCurrentIndex(metric_idx)
            self.subsample_checkbox.setChecked(settings.get("subsample_enabled", False))
            self.subsample_size_spin.setValue(settings.get("subsample_size", 70000))

            # Advanced settings (low RAM, CPU cores)
            self.low_memory = settings.get("low_memory", True)
            self.n_jobs = settings.get("n_jobs", os.cpu_count() or 4)
            # Reserved parallel-load thread counts (applied once the threaded loader lands).
            self.api_load_threads = int(settings.get("api_load_threads", 4))
            self.direct_load_threads = int(settings.get("direct_load_threads", 2))
            # Chunked transform toggle (subsample path); sync the checkbox if present.
            self.chunked_transform = bool(settings.get("chunked_transform", True))
            _ct_cb = getattr(self, 'chunked_transform_checkbox', None)
            if _ct_cb is not None:
                _ct_cb.setChecked(self.chunked_transform)
            # Pre-SVD toggle (performance); sync the checkbox if present.
            self.pre_svd_enabled = bool(settings.get("pre_svd_enabled", False))
            self.pre_svd_components = int(settings.get("pre_svd_components", 64))
            _ps_cb = getattr(self, 'pre_svd_checkbox', None)
            if _ps_cb is not None:
                _ps_cb.setChecked(self.pre_svd_enabled)
            _ps_spin = getattr(self, 'pre_svd_components_spin', None)
            if _ps_spin is not None:
                _ps_spin.setValue(self.pre_svd_components)

            # Cluster settings
            self.eps_spin.setValue(settings.get("eps", 12))
            self.min_samples_spin.setValue(settings.get("min_samples", 8))
            # DBSCAN optimizer settings
            self.opt_max_cohort_size = settings.get("opt_max_cohort_size", 500)
            self.opt_max_noise_ratio = settings.get("opt_max_noise_ratio", 10)
            self.opt_max_attempts = settings.get("opt_max_attempts", 60)
            self.opt_eps_min = settings.get("opt_eps_min", 5)
            self.opt_eps_max = settings.get("opt_eps_max", 100)
            self.opt_min_samples_min = settings.get("opt_min_samples_min", 2)
            self.opt_min_samples_max = settings.get("opt_min_samples_max", 30)
            self.normalize_positions = settings.get("normalize_positions", True)
            # Sync the inline checkbox so the UI reflects the saved value
            self.normalize_checkbox.setChecked(self.normalize_positions)
            # Auto-Deorphan per-operation flags. Migrate the legacy single-mode
            # string ("Never" / "After Load and Compute" / "After Regroup") so
            # existing settings files keep working unchanged.
            if isinstance(settings.get("auto_deorphan_ops"), dict):
                _ad = settings["auto_deorphan_ops"]
                self.auto_deorphan_ops = {k: bool(_ad.get(k)) for k in ("load", "regroup", "split")}
            else:
                _legacy = settings.get("auto_deorphan", "Never")
                if _legacy not in ("Never", "After Load and Compute", "After Regroup"):
                    _legacy = "Never"
                self.auto_deorphan_ops = {
                    "load": _legacy == "After Load and Compute",
                    # Old "After Regroup" also fired after Optimize (full re-cluster).
                    "regroup": _legacy == "After Regroup",
                    "split": False,
                }
            # Auto-split oversized cohorts after Load & Compute
            self.auto_split_enabled = bool(settings.get("auto_split_enabled", True))
            self.auto_split_threshold = int(settings.get("auto_split_threshold", 5000))
            self.auto_split_max_cycles = int(settings.get("auto_split_max_cycles", 3))
            # Session auto-save delay (seconds; 0 = save immediately)
            self.session_save_delay = int(settings.get("session_save_delay", 60))
            # Node blending mode for the 3D scatter (normal alpha = default)
            _nb = settings.get("node_blending", "Normal Alpha")
            self.node_blending = _nb if _nb in ("Additive", "Normal Alpha", "Simple") else "Normal Alpha"
            # Tag query grid layout (columns x rows; max tags shown = cols * rows)
            self.tag_query_columns = int(settings.get("tag_query_columns", 3))
            self.tag_query_rows = int(settings.get("tag_query_rows", 14))
            # Explore (helicopter orbit) parameters
            _em = settings.get("explore_mode", "Random")
            if _em in ("Random", "Linear Path", "Contrast", "Size"):
                self.explore_mode = _em
            self.explore_show_path = bool(settings.get("explore_show_path", False))
            self.explore_accel = float(settings.get("explore_accel", 0.05))
            self.explore_decel = float(settings.get("explore_decel", 0.05))
            self.explore_orbit_radius_base = float(settings.get("explore_orbit_radius_base", 0.0))
            self.explore_orbit_size_factor = float(settings.get("explore_orbit_size_factor", 0.2))
            self.explore_orbit_speed = float(settings.get("explore_orbit_speed", 12.0))
            self.explore_cycles = int(settings.get("explore_cycles", 1))
            self.explore_max_orbit_time = float(settings.get("explore_max_orbit_time", 15.0))
            self.explore_elevation = float(settings.get("explore_elevation", 35.0))
            # Smart Scale: master toggle + node-count-based profiles.
            self.smart_scale_enabled = bool(settings.get("smart_scale_enabled", False))
            _ssp = settings.get("smart_scale_profiles")
            if isinstance(_ssp, list) and _ssp:
                self.smart_scale_profiles = _ssp
            # Optional tag-score DB path
            self.score_db_path = settings.get("score_db_path", "")
            # UI scale (percent); applied at startup, restart required to change
            try:
                self.ui_scale = int(settings.get("ui_scale", 100))
            except (TypeError, ValueError):
                self.ui_scale = 100
            # Sub-clustering settings
            self.sub_eps_spin.setValue(settings.get("sub_eps", 12))
            self.sub_min_samples_spin.setValue(settings.get("sub_min_samples", 10))

            # Filter settings
            self.query_edit.setText(settings.get("query", ""))
            self.whitelist_edit.setText(settings.get("whitelist", ""))
            self.blacklist_edit.setText(settings.get("blacklist", ""))
            self.tokenize = settings.get("tokenize", True)
            self.drop_empty_files = settings.get("drop_empty_files", False)
            self.right_click_select_cohort = bool(settings.get("right_click_select_cohort", False))
            self.auto_center_on_selection = bool(settings.get("auto_center_on_selection", True))
            self.wasd_paths_enabled = bool(settings.get("wasd_paths_enabled", True))

            # WASD navigation preview appearance + persistent-labels toggle.
            _wlc = settings.get("wasd_line_color")
            if isinstance(_wlc, (list, tuple)) and len(_wlc) == 3:
                self.wasd_line_color = tuple(int(c) for c in _wlc)
            _wcc = settings.get("wasd_letter_color")
            if isinstance(_wcc, (list, tuple)) and len(_wcc) == 3:
                self.wasd_letter_color = tuple(int(c) for c in _wcc)
            try:
                self.wasd_label_size = int(settings.get("wasd_label_size", 36))
            except (TypeError, ValueError):
                self.wasd_label_size = 36
            self.wasd_persistent_labels = bool(settings.get("wasd_persistent_labels", False))
            # Sync the widgets built in setup_ui (before this load).
            if hasattr(self, 'wasd_line_btn'):
                r, g, b = self.wasd_line_color
                self.wasd_line_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #4050a0;")
            if hasattr(self, 'wasd_letter_btn'):
                r, g, b = self.wasd_letter_color
                self.wasd_letter_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #4050a0;")
            if hasattr(self, 'wasd_label_spin'):
                self.wasd_label_spin.blockSignals(True)
                self.wasd_label_spin.setValue(max(8, min(120, self.wasd_label_size)))
                self.wasd_label_spin.blockSignals(False)
            if hasattr(self, 'wasd_persistent_checkbox'):
                self.wasd_persistent_checkbox.setChecked(self.wasd_persistent_labels)
            self.smooth_center_transition = bool(settings.get("smooth_center_transition", False))
            self.smooth_center_speed = float(settings.get("smooth_center_speed", 1.0))
            # Unit-aware Min Tag Frequency (n / %). Restore both stored values + unit.
            if hasattr(self, '_apply_min_doc_freq_state'):
                self._apply_min_doc_freq_state({
                    "min_doc_freq": settings.get("min_doc_freq", 5),
                    "min_doc_freq_pct": settings.get("min_doc_freq_pct", 1),
                    "min_doc_freq_unit": settings.get("min_doc_freq_unit", "n"),
                })
            else:
                self.min_doc_freq_spin.setValue(settings.get("min_doc_freq", 5))
            self.drop_universal = settings.get("drop_universal_tags", True)

            # Visualization settings (node_size is stored as actual value, displayed x10)
            # Backward compat: fall back to old "min_size" key if "node_size" not present
            node_size_actual = settings.get("node_size", settings.get("min_size", 0.02))
            self.min_size_spin.setValue(node_size_actual * 10.0)
            self.spread_spin.setValue(settings.get("spread", 1.0))
            self.orbit_speed_spin.setValue(settings.get("orbit_speed", 0.2))
            self.transparency_spin.setValue(settings.get("transparency", 0.9))

            # Node blending mode (combo built in setup_ui, before this load).
            # Also sync the transparency/dim-alpha enable state for Simple mode.
            if hasattr(self, 'node_blending_combo'):
                _nb_idx = self.node_blending_combo.findText(self.node_blending)
                if _nb_idx >= 0:
                    self.node_blending_combo.blockSignals(True)
                    self.node_blending_combo.setCurrentIndex(_nb_idx)
                    self.node_blending_combo.blockSignals(False)
                if hasattr(self, '_sync_blending_controls'):
                    self._sync_blending_controls()

            # Anti-noise / quality settings
            self.supersample_checkbox.setChecked(settings.get("supersample", False))
            self.supersample_fps_spin.setValue(settings.get("supersample_fps", 10))

            # Dim non-selected nodes settings
            self.dim_non_selected_checkbox.setChecked(settings.get("dim_non_selected", True))
            self.dim_alpha_spin.setValue(settings.get("dim_alpha", 0.15))
            if "highlight_color" in settings:
                self.highlight_color = tuple(settings["highlight_color"])
                r, g, b = int(self.highlight_color[0]*255), int(self.highlight_color[1]*255), int(self.highlight_color[2]*255)
                self.highlight_color_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: black; font-weight: bold;")

            # 3D view background color (default black). Applied live if the GL
            # view already exists, otherwise create_3d_view reads it on startup.
            _bg = settings.get("bg_color", None)
            if isinstance(_bg, (list, tuple)) and len(_bg) == 3:
                self.bg_color = tuple(float(c) for c in _bg)
                r, g, b = int(self.bg_color[0]*255), int(self.bg_color[1]*255), int(self.bg_color[2]*255)
                if hasattr(self, 'bg_color_btn'):
                    self.bg_color_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #4050a0;")
                if getattr(self, 'gl_view', None) is not None:
                    try:
                        self.gl_view.setBackgroundColor((r, g, b, 255))
                    except Exception:
                        pass

            # Star twinkle settings (set values first, then toggle to avoid premature spawn)
            self.twinkle_count_spin.setValue(settings.get("twinkle_count", 4000))
            self.twinkle_lifespan_min_spin.setValue(settings.get("twinkle_lifespan_min", 1.0))
            self.twinkle_lifespan_max_spin.setValue(settings.get("twinkle_lifespan_max", 6.0))
            self.twinkle_freq_spin.setValue(settings.get("twinkle_freq", 0.5))
            self.twinkle_brightness_spin.setValue(settings.get("twinkle_brightness", 1.5))
            # Toggle last: _on_twinkle_toggle will no-op if no scene loaded yet
            self.twinkle_checkbox.setChecked(settings.get("twinkle_enabled", False))

            # V5: Color scheme setting (+ user-generated custom schemes + palette size).
            _custom_schemes = settings.get("custom_color_schemes") or {}
            if isinstance(_custom_schemes, dict):
                self.custom_color_schemes = {k: [list(c) for c in v] for k, v in _custom_schemes.items()}
            if hasattr(self, 'palette_colors_spin'):
                self.palette_colors_spin.setValue(int(settings.get("palette_colors", 19)))
            # Rebuild the dropdown with stored schemes included, then select active.
            if hasattr(self, '_rebuild_color_scheme_combo'):
                _cs = settings.get("color_scheme", "Pastel")
                # A previously-active ephemeral "Generated" scheme isn't persisted; fall back to Pastel.
                self._rebuild_color_scheme_combo(select=_cs if _cs != "Generated" else "Pastel")
            else:
                color_scheme_idx = self.color_scheme_combo.findText(settings.get("color_scheme", "Pastel"))
                if color_scheme_idx >= 0:
                    self.color_scheme_combo.setCurrentIndex(color_scheme_idx)

            # Node sizing mode (Distance is the legacy default). Set directly so we
            # don't trigger a scatter rebuild during load (no scene exists yet).
            _ns = settings.get("node_sizing_mode", "Distance")
            if hasattr(self, 'node_sizing_combo'):
                self.node_sizing_mode = _ns if _ns in (
                    "Distance", "Screen-constant", "Uniform single size",
                    "Auto-scale to view distance"
                ) else "Distance"
                _ns_idx = self.node_sizing_combo.findText(self.node_sizing_mode)
                if _ns_idx >= 0:
                    self.node_sizing_combo.blockSignals(True)
                    self.node_sizing_combo.setCurrentIndex(_ns_idx)
                    self.node_sizing_combo.blockSignals(False)

            # Cohort label settings
            self.cohort_threshold_spin.setValue(settings.get("cohort_threshold", 0.9))
            self.show_cohort_labels_checkbox.setChecked(settings.get("show_cohort_labels", True))
            self.cohort_label_size_spin.setValue(settings.get("cohort_label_size", 15))
            self.dynamic_label_size_checkbox.setChecked(settings.get("dynamic_label_size", False))
            color = settings.get("cohort_label_color", [255, 255, 255])
            self._cohort_label_color = tuple(color)
            self._update_label_color_button()
            color2 = settings.get("cohort_label_color2", [255, 200, 0])
            self._cohort_label_color2 = tuple(color2)
            self._update_label_color_button2()
            # Outline color + width
            outline_color = settings.get("cohort_label_outline_color", [0, 0, 0])
            self._cohort_label_outline_color = tuple(outline_color)
            self._update_outline_color_button()
            self.cohort_label_outline_width_spin.setValue(settings.get("cohort_label_outline_width", 3))
            # Label mode + N
            mode_idx = self.cohort_label_mode_combo.findText(settings.get("cohort_label_mode", "Selected & N neighbors"))
            if mode_idx >= 0:
                self.cohort_label_mode_combo.setCurrentIndex(mode_idx)
            self.cohort_label_n_spin.setValue(settings.get("cohort_label_n", 7))
            self.cohort_label_max_tags_spin.setValue(settings.get("cohort_label_max_tags", 2))
            # Fade label transitions (selection switching)
            self.label_fade_enabled = bool(settings.get("label_fade_enabled", True))
            if hasattr(self, 'label_fade_checkbox'):
                self.label_fade_checkbox.setChecked(self.label_fade_enabled)
            if hasattr(self, 'label_fade_duration_spin'):
                self.label_fade_duration_spin.setValue(int(settings.get("label_fade_duration_ms", 2000)))
            # Smart labels settings (merged into mode combo; "Raw" = disabled)
            smart_mode_idx = self.smart_label_mode_combo.findText(
                settings.get("smart_label_mode", "Absolute Unique")
            )
            if smart_mode_idx >= 0:
                self.smart_label_mode_combo.setCurrentIndex(smart_mode_idx)

            # Label Space (overlap handling): mode + gap. Set the attributes first so
            # _apply_label_space() can read them, then sync the widgets without firing
            # a redundant re-apply during load.
            _lsm = settings.get("label_space_mode", "Fade")
            self.label_space_mode = _lsm if _lsm in ("None", "Fade", "Move") else "Fade"
            try:
                self.label_space_gap = int(settings.get("label_space_gap", 25))
            except (TypeError, ValueError):
                self.label_space_gap = 25
            if hasattr(self, 'label_space_combo'):
                _ls_idx = self.label_space_combo.findText(self.label_space_mode)
                if _ls_idx >= 0:
                    self.label_space_combo.blockSignals(True)
                    self.label_space_combo.setCurrentIndex(_ls_idx)
                    self.label_space_combo.blockSignals(False)
            if hasattr(self, 'label_gap_spin'):
                self.label_gap_spin.blockSignals(True)
                self.label_gap_spin.setValue(max(0, min(200, self.label_space_gap)))
                self.label_gap_spin.blockSignals(False)

            # Split window settings (image preview)
            if hasattr(self, 'split_window') and self.split_window:
                self.split_window.columns_spin.setValue(settings.get("split_columns", 4))
                self.split_window.max_files_spin.setValue(settings.get("split_max_files", 28))
                self.split_window.image_size_spin.setValue(settings.get("split_image_size", 400))
            
            # Camera wobble settings
            self.wobble_enabled_checkbox.setChecked(settings.get("wobble_enabled", False))
            self.wobble_speed_spin.setValue(settings.get("wobble_speed", 1.0))
            self.wobble_azim_range_spin.setValue(settings.get("wobble_azim_range", 15.0))
            self.wobble_elev_range_spin.setValue(settings.get("wobble_elev_range", 10.0))

            # Send to tab settings
            if hasattr(self, 'tab_name_edit'):
                self.tab_name_edit.setText(settings.get("tab_name", ""))

            # Media viewer open/closed state (applied after the window is shown)
            self._restore_media_viewer_open = bool(settings.get("media_viewer_open", False))

            # Window geometry (main window + splitter sizes)
            main_window = getattr(self, 'main_window', None)
            if main_window is not None:
                self._restore_window_geometry(settings, "window_geometry", main_window)
            if hasattr(self, 'main_splitter'):
                try:
                    raw = settings.get("splitter_sizes")
                    if raw:
                        from PySide6.QtCore import QByteArray
                        self.main_splitter.restoreState(QByteArray(bytes.fromhex(raw)))
                except Exception:
                    pass
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Use defaults if settings file is corrupted

    def _schedule_settings_save(self):
        """Schedule a debounced settings save (2s after last widget change).

        This ensures settings are persisted to disk even if the app crashes
        before closeEvent() fires.
        """
        if not hasattr(self, '_settings_save_timer'):
            self._settings_save_timer = QTimer(self)
            self._settings_save_timer.setSingleShot(True)
            self._settings_save_timer.setInterval(2000)
            self._settings_save_timer.timeout.connect(self.save_settings)
        self._settings_save_timer.start()

    def _connect_settings_signals(self):
        """Connect all settings widget change signals to the debounced auto-save.

        This ensures any change to a spinbox, checkbox, combo, or line edit
        triggers a save 2 seconds later, so settings survive crashes.
        """
        # Map of attribute name -> signal name
        signal_map = {
            # QSpinBox / QDoubleSpinBox -> valueChanged
            'max_files_spin': 'valueChanged',
            'n_neighbors_spin': 'valueChanged',
            'min_dist_spin': 'valueChanged',
            'n_epochs_spin': 'valueChanged',
            'learning_rate_spin': 'valueChanged',
            'eps_spin': 'valueChanged',
            'min_samples_spin': 'valueChanged',
            'sub_eps_spin': 'valueChanged',
            'sub_min_samples_spin': 'valueChanged',
            'min_doc_freq_spin': 'valueChanged',
            'min_doc_freq_unit_combo': 'currentTextChanged',
            'min_size_spin': 'valueChanged',
            'spread_spin': 'valueChanged',
            'orbit_speed_spin': 'valueChanged',
            'transparency_spin': 'valueChanged',
            'supersample_fps_spin': 'valueChanged',
            'dim_alpha_spin': 'valueChanged',
            'cohort_threshold_spin': 'valueChanged',
            'cohort_label_size_spin': 'valueChanged',
            'cohort_label_outline_width_spin': 'valueChanged',
            'cohort_label_n_spin': 'valueChanged',
            'cohort_label_max_tags_spin': 'valueChanged',
            'label_fade_duration_spin': 'valueChanged',
            'wobble_speed_spin': 'valueChanged',
            'wobble_azim_range_spin': 'valueChanged',
            'wobble_elev_range_spin': 'valueChanged',
            # QCheckBox -> stateChanged
            'auto_load_checkbox': 'stateChanged',
            'normalize_checkbox': 'stateChanged',
            'supersample_checkbox': 'stateChanged',
            'dim_non_selected_checkbox': 'stateChanged',
            'show_cohort_labels_checkbox': 'stateChanged',
            'label_fade_checkbox': 'stateChanged',
            'dynamic_label_size_checkbox': 'stateChanged',
            'wobble_enabled_checkbox': 'stateChanged',
            # QComboBox -> currentTextChanged
            'client_combo': 'currentTextChanged',
            'tag_service_combo': 'currentTextChanged',
            'algorithm_combo': 'currentTextChanged',
            'metric_combo': 'currentTextChanged',
            'color_scheme_combo': 'currentTextChanged',
            'node_sizing_combo': 'currentTextChanged',
            'cohort_label_mode_combo': 'currentTextChanged',
            'smart_label_mode_combo': 'currentTextChanged',
            # QLineEdit -> textChanged
            'whitelist_edit': 'textChanged',
            'blacklist_edit': 'textChanged',
            'query_edit': 'textChanged',
            'tab_name_edit': 'textChanged',
        }
        for attr, signal_name in signal_map.items():
            widget = getattr(self, attr, None)
            if widget is not None:
                signal = getattr(widget, signal_name, None)
                if signal is not None:
                    signal.connect(self._schedule_settings_save)

    def save_settings(self):
        """Save current settings to JSON file (atomic write)."""
        try:
            # Start from the EXISTING file contents so keys written by other
            # code paths survive (e.g. "split_window_geometry" is persisted by
            # _persist_split_window_geometry() on media-viewer close, and
            # "settings_dialog_geometry" by the settings window). Rebuilding a
            # fresh dict here used to silently drop them on every auto-save.
            existing = {}
            if os.path.exists(SETTINGS_FILE):
                try:
                    with open(SETTINGS_FILE, 'r') as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        existing = loaded
                except (json.JSONDecodeError, OSError):
                    pass

            settings = dict(existing)
            settings.update({
                "client": self.client_combo.currentText(),
                # Path-specific chunk sizes (legacy single key kept in sync for back-compat).
                "api_chunk_size": int(getattr(self, 'api_chunk_size', getattr(self, 'chunk_size', 8192))),
                "direct_chunk_size": int(getattr(self, 'direct_chunk_size', 4096)),
                "chunk_size": int(getattr(self, 'api_chunk_size', getattr(self, 'chunk_size', 8192))),
                "max_files": self.max_files_spin.value(),
                "tag_service": self.tag_service_combo.currentText(),
                "use_direct_db": self.use_direct_db,
                "low_memory": self.low_memory,
                "n_jobs": self.n_jobs,
                # Reserved parallel-load thread counts (see benchmarks/benchmark_api_io.py).
                "api_load_threads": int(getattr(self, 'api_load_threads', 4)),
                "direct_load_threads": int(getattr(self, 'direct_load_threads', 2)),
                # Read live widget state (attributes can be stale if the user
                # toggled the checkbox without opening the Settings dialog).
                "chunked_transform": bool(
                    self.chunked_transform_checkbox.isChecked()
                    if hasattr(self, 'chunked_transform_checkbox') else getattr(self, 'chunked_transform', True)
                ),
                "pre_svd_enabled": bool(
                    self.pre_svd_checkbox.isChecked()
                    if hasattr(self, 'pre_svd_checkbox') else getattr(self, 'pre_svd_enabled', False)
                ),
                "pre_svd_components": int(
                    self.pre_svd_components_spin.value()
                    if hasattr(self, 'pre_svd_components_spin') else getattr(self, 'pre_svd_components', 64)
                ),
                "algorithm": self.algorithm_combo.currentText(),
                "n_neighbors": self.n_neighbors_spin.value(),
                "min_dist": self.min_dist_spin.value(),
                "n_epochs": self.n_epochs_spin.value(),
                "learning_rate": self.learning_rate_spin.value(),
                "metric": self.metric_combo.currentText(),
                "subsample_enabled": self.subsample_checkbox.isChecked(),
                "subsample_size": self.subsample_size_spin.value(),
                "eps": self.eps_spin.value(),
                "min_samples": self.min_samples_spin.value(),
                "sub_eps": self.sub_eps_spin.value(),
                "sub_min_samples": self.sub_min_samples_spin.value(),
                # DBSCAN optimizer settings
                "opt_max_cohort_size": getattr(self, 'opt_max_cohort_size', 500),
                "opt_max_noise_ratio": getattr(self, 'opt_max_noise_ratio', 10),
                "opt_max_attempts": getattr(self, 'opt_max_attempts', 60),
                "opt_eps_min": getattr(self, 'opt_eps_min', 5),
                "opt_eps_max": getattr(self, 'opt_eps_max', 100),
                "opt_min_samples_min": getattr(self, 'opt_min_samples_min', 2),
                "opt_min_samples_max": getattr(self, 'opt_min_samples_max', 30),
                "normalize_positions": getattr(self, 'normalize_positions', True),
                "auto_deorphan_ops": {k: bool(v) for k, v in (getattr(self, 'auto_deorphan_ops', None) or {}).items()},
                "auto_split_enabled": bool(getattr(self, 'auto_split_enabled', True)),
                "auto_split_threshold": int(getattr(self, 'auto_split_threshold', 5000)),
                "auto_split_max_cycles": int(getattr(self, 'auto_split_max_cycles', 3)),
                "session_save_delay": int(getattr(self, 'session_save_delay', 60)),
                # Explore (helicopter orbit) parameters
                "explore_mode": getattr(self, 'explore_mode', "Random"),
                "explore_show_path": bool(getattr(self, 'explore_show_path', False)),
                "explore_accel": float(getattr(self, 'explore_accel', 0.6)),
                "explore_decel": float(getattr(self, 'explore_decel', 0.6)),
                "explore_orbit_radius_base": float(getattr(self, 'explore_orbit_radius_base', 8.0)),
                "explore_orbit_size_factor": float(getattr(self, 'explore_orbit_size_factor', 2.0)),
                "explore_orbit_speed": float(getattr(self, 'explore_orbit_speed', 12.0)),
                "explore_cycles": int(getattr(self, 'explore_cycles', 3)),
                "explore_max_orbit_time": float(getattr(self, 'explore_max_orbit_time', 30.0)),
                "explore_elevation": float(getattr(self, 'explore_elevation', 40.0)),
                # Smart Scale: master toggle + node-count-based profiles.
                "smart_scale_enabled": bool(getattr(self, 'smart_scale_enabled', False)),
                "smart_scale_profiles": getattr(self, 'smart_scale_profiles', []),
                "score_db_path": getattr(self, 'score_db_path', ''),
                # UI scale (percent); applied at startup via QT_SCALE_FACTOR
                "ui_scale": getattr(self, 'ui_scale', 100),
                "query": self.query_edit.text(),
                "whitelist": self.whitelist_edit.text(),
                "blacklist": self.blacklist_edit.text(),
                "tokenize": getattr(self, 'tokenize', True),
                "drop_empty_files": getattr(self, 'drop_empty_files', False),
                "right_click_select_cohort": bool(getattr(self, 'right_click_select_cohort', False)),
                "auto_center_on_selection": bool(getattr(self, 'auto_center_on_selection', True)),
                "wasd_paths_enabled": bool(getattr(self, 'wasd_paths_enabled', True)),
                # WASD navigation preview appearance + persistent-labels toggle.
                "wasd_line_color": list(getattr(self, 'wasd_line_color', (80, 255, 140))),
                "wasd_letter_color": list(getattr(self, 'wasd_letter_color', (80, 255, 140))),
                "wasd_label_size": int(getattr(self, 'wasd_label_size', 36)),
                "wasd_persistent_labels": bool(getattr(self, 'wasd_persistent_labels', False)),
                "smooth_center_transition": bool(getattr(self, 'smooth_center_transition', False)),
                "smooth_center_speed": float(getattr(self, 'smooth_center_speed', 1.0)),
                # Unit-aware Min Tag Frequency (n / %): store both values + active unit.
                **(self._min_doc_freq_state() if hasattr(self, '_min_doc_freq_state')
                   else {"min_doc_freq": self.min_doc_freq_spin.value()}),
                # Tag query grid layout (columns x rows; max tags = cols * rows)
                "tag_query_columns": int(getattr(self, 'tag_query_columns', 3)),
                "tag_query_rows": int(getattr(self, 'tag_query_rows', 14)),
                "drop_universal_tags": self.drop_universal,
                "node_size": self.min_size_spin.value() / 10.0,
                "spread": self.spread_spin.value(),
                "orbit_speed": self.orbit_speed_spin.value(),
                "transparency": self.transparency_spin.value(),
                # Node blending mode (normal alpha is the default)
                "node_blending": getattr(self, 'node_blending', "Normal Alpha"),
                # Node sizing mode (distance = legacy perspective scaling)
                "node_sizing_mode": getattr(self, 'node_sizing_mode', "Distance"),
                # Anti-noise / quality settings
                "supersample": self.supersample_checkbox.isChecked(),
                "supersample_fps": self.supersample_fps_spin.value(),
                # Dim non-selected nodes settings
                "dim_non_selected": self.dim_non_selected_checkbox.isChecked(),
                "dim_alpha": self.dim_alpha_spin.value(),
                "highlight_color": list(self.highlight_color),
                # 3D view background color (0-1 floats)
                "bg_color": list(getattr(self, 'bg_color', (0.0, 0.0, 0.0))),
                # Star twinkle settings
                "twinkle_enabled": self.twinkle_checkbox.isChecked(),
                "twinkle_count": self.twinkle_count_spin.value(),
                "twinkle_lifespan_min": self.twinkle_lifespan_min_spin.value(),
                "twinkle_lifespan_max": self.twinkle_lifespan_max_spin.value(),
                "twinkle_freq": self.twinkle_freq_spin.value(),
                "twinkle_brightness": self.twinkle_brightness_spin.value(),
                # Color scheme (+ user-generated custom schemes + palette size).
                "color_scheme": self.color_scheme_combo.currentText(),
                "custom_color_schemes": getattr(self, 'custom_color_schemes', {}),
                "palette_colors": int(getattr(self, 'palette_colors_spin').value()) if hasattr(self, 'palette_colors_spin') else 19,
                # Cohort label settings
                "cohort_threshold": self.cohort_threshold_spin.value(),
                "show_cohort_labels": self.show_cohort_labels_checkbox.isChecked(),
                "cohort_label_size": self.cohort_label_size_spin.value(),
                "dynamic_label_size": self.dynamic_label_size_checkbox.isChecked(),
                "cohort_label_color": list(self._cohort_label_color),
                "cohort_label_color2": list(self._cohort_label_color2),
                "cohort_label_outline_color": list(getattr(self, '_cohort_label_outline_color', (0, 0, 0))),
                "cohort_label_outline_width": getattr(self, 'cohort_label_outline_width_spin', None).value() if hasattr(self, 'cohort_label_outline_width_spin') else 3,
                "cohort_label_mode": self.cohort_label_mode_combo.currentText(),
                "cohort_label_n": self.cohort_label_n_spin.value(),
                "cohort_label_max_tags": self.cohort_label_max_tags_spin.value(),
                # Fade label transitions (selection switching)
                "label_fade_enabled": bool(getattr(self, 'label_fade_enabled', True)),
                "label_fade_duration_ms": int(self.label_fade_duration_spin.value()) if hasattr(self, 'label_fade_duration_spin') else 2000,
                # Smart labels settings (merged into mode combo; "Raw" = disabled)
                "smart_label_mode": self.smart_label_mode_combo.currentText(),
                # Label Space (overlap handling): mode + gap
                "label_space_mode": getattr(self, 'label_space_mode', "None"),
                "label_space_gap": int(getattr(self, 'label_space_gap', 8)),
                # Media viewer open/closed state (restored on startup)
                "media_viewer_open": bool(getattr(self, 'split_window', None)),
                # Split window settings (image preview)
                # Fall back to previously saved values when the split window is closed
                "split_columns": self.split_window.columns_spin.value() if hasattr(self, 'split_window') and self.split_window else existing.get("split_columns", 4),
                "split_max_files": self.split_window.max_files_spin.value() if hasattr(self, 'split_window') and self.split_window else existing.get("split_max_files", 28),
                "split_image_size": self.split_window.image_size_spin.value() if hasattr(self, 'split_window') and self.split_window else existing.get("split_image_size", 400),
                # Camera wobble settings
                "wobble_enabled": self.wobble_enabled_checkbox.isChecked(),
                "wobble_speed": self.wobble_speed_spin.value(),
                "wobble_azim_range": self.wobble_azim_range_spin.value(),
                "wobble_elev_range": self.wobble_elev_range_spin.value(),
                # Send to tab settings
                "tab_name": self.tab_name_edit.text() if hasattr(self, 'tab_name_edit') else "",
                # Auto-load last data setting
                "auto_load_last_data": self.auto_load_checkbox.isChecked() if hasattr(self, 'auto_load_checkbox') else True,
            })

            # Window geometry (main window + splitter sizes)
            main_window = getattr(self, 'main_window', None)
            if main_window is not None:
                self._save_window_geometry(settings, "window_geometry", main_window)
            if hasattr(self, 'main_splitter'):
                try:
                    # PySide6: saveState() takes NO argument and returns the QByteArray.
                    ba = self.main_splitter.saveState()
                    if len(ba) > 0:
                        settings["splitter_sizes"] = bytes(ba).hex()
                except Exception:
                    pass
            # Atomic write: write to temp file, then replace.
            # This prevents corruption if the app crashes mid-write.
            settings_dir = os.path.dirname(SETTINGS_FILE) or '.'
            tmp_fd, tmp_path = tempfile.mkstemp(dir=settings_dir, suffix='.tmp')
            try:
                with os.fdopen(tmp_fd, 'w') as f:
                    json.dump(settings, f, indent=2)
                os.replace(tmp_path, SETTINGS_FILE)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError:
            pass  # Silently fail if we can't write settings

    @staticmethod
    def _save_window_geometry(settings, key, widget):
        """Serialize a window's geometry (pos + size) into the settings dict.

        Stored as base64 of QByteArray(saveGeometry()) so it round-trips
        through JSON. No-op if the widget is missing or not yet shown.
        """
        try:
            # PySide6: saveGeometry() takes NO argument and returns the
            # QByteArray (unlike PyQt5's in/out buffer form).
            ba = widget.saveGeometry()
            if len(ba) > 0:
                settings[key] = bytes(ba).hex()
        except Exception:
            pass

    @staticmethod
    def _frame_margins(widget):
        """Return (left, top, right, bottom) window-frame margins in logical px.

        ``geometry()`` is the CLIENT area; on Windows the WM draws the title bar
        ABOVE it and borders around it. We need those margins so a clamped window
        keeps its caption visible instead of getting "stuck at the top". When the
        widget isn't decorated yet (not shown) Qt reports zero frame, so fall back
        to an estimate for the caption height — better slightly too much than none.
        """
        try:
            fg = widget.frameGeometry()
            g = widget.geometry()
            left = max(0, g.x() - fg.x())
            top = max(0, g.y() - fg.y())
            right = max(0, (fg.x() + fg.width()) - (g.x() + g.width()))
            bottom = max(0, (fg.y() + fg.height()) - (g.y() + g.height()))
        except Exception:
            left = top = right = bottom = 0
        if not widget.isVisible():
            # Not decorated yet -> frame margins are unreliable; estimate caption.
            top = max(top, 32)
            left = max(left, 8)
            right = max(right, 8)
            bottom = max(bottom, 8)
        return left, top, right, bottom

    @staticmethod
    def _clamp_widget_to_screens(widget):
        """Keep a window's size + position within the available screen area.

        A geometry saved under one UI Scale / resolution can come back LARGER
        than the current screen (UI scale multiplies logical sizes at startup),
        leaving e.g. the bottom status bar off-screen and the window feeling
        "unresizable". The size is capped to the available area of the screen it
        sits on, RESERVING room for the window frame so the title bar stays
        visible (not pushed above the top edge). If it intersects no screen at all
        (e.g. a monitor was unplugged) it is moved onto the primary one.
        """
        from PySide6.QtGui import QGuiApplication
        # NOTE: screens() is a static of QGuiApplication (QCoreApplication has no such method).
        screens = [s for s in (QGuiApplication.screens() or []) if s.availableGeometry().isValid()]
        if not screens:
            return
        geo = widget.geometry()
        host = next((s for s in screens if s.availableGeometry().intersects(geo)), None)
        # QScreen has no isPrimary(); use the app's primary screen instead.
        primary = QGuiApplication.primaryScreen() or screens[0]
        avail = (host or primary).availableGeometry()

        left, top, right, bottom = SettingsPersistenceMixin._frame_margins(widget)
        max_w = max(1, avail.width() - left - right)
        max_h = max(1, avail.height() - top - bottom)

        w, h = geo.width(), geo.height()
        if w > max_w:
            w = max_w
        if h > max_h:
            h = max_h
        # Client-area origin; the caption sits `top` px above it, so keep x/y at
        # least one frame margin inside the available area.
        x = min(max(geo.x(), avail.left() + left), avail.right() - right - w)
        y = min(max(geo.y(), avail.top() + top), avail.bottom() - bottom - h)
        widget.setGeometry(x, y, w, h)

    @staticmethod
    def _restore_window_geometry(settings, key, widget):
        """Restore a window's geometry from the settings dict (best effort).

        The restored size/position is clamped to the available screen area so a
        saved state that no longer fits (UI scale change, resolution change) can't
        push parts of the window off-screen. If it still doesn't intersect any
        monitor (e.g. a second screen was unplugged), re-center on primary.
        """
        try:
            from PySide6.QtCore import QByteArray
            raw = settings.get(key)
            if not raw:
                return False
            ba = QByteArray(bytes.fromhex(raw))
            ok = widget.restoreGeometry(ba)
            # Clamp to the available screen area (size + position). This also pulls a
            # window saved on an unplugged monitor back onto the primary screen.
            SettingsPersistenceMixin._clamp_widget_to_screens(widget)
            return bool(ok)
        except Exception:
            return False

    def _persist_split_window_geometry(self):
        """Persist the split window's geometry to the settings file (atomic).

        Called from TagMap3DSplitWindow.closeEvent(). Reads the existing JSON,
        updates only the geometry key and writes it back, so no other saved
        values are touched.
        """
        if getattr(self, 'split_window', None) is None:
            return
        try:
            settings = {}
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    settings = loaded
            self._save_window_geometry(settings, "split_window_geometry", self.split_window)
            # The viewer is closing -> record the closed state so a restart
            # does not re-open it (this path runs before the tab's reference
            # is cleared and no debounced save_settings() may follow).
            settings["media_viewer_open"] = False
            settings_dir = os.path.dirname(SETTINGS_FILE) or '.'
            tmp_fd, tmp_path = tempfile.mkstemp(dir=settings_dir, suffix='.tmp')
            try:
                with os.fdopen(tmp_fd, 'w', encoding="utf-8") as f:
                    json.dump(settings, f, indent=2)
                os.replace(tmp_path, SETTINGS_FILE)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            pass

    def _load_client_db_path(self):
        """Load all client DB paths into the settings dialog state."""
        from src.data.direct_db import get_client_db_path
        from src.data.clients import client_ids
        paths = {}
        for client_id in (client_ids() or []):
            path = get_client_db_path(client_id)
            if path:
                paths[client_id] = path
        self.client_db_paths = paths

    def _save_client_db_path(self):
        """Save all client DB paths to ClientSettings."""
        from src.data.direct_db import set_client_db_path
        for client_id, path in self.client_db_paths.items():
            if path:
                set_client_db_path(client_id, path)
