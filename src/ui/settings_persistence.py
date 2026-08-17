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
            self.chunk_size_spin.setValue(settings.get("chunk_size", 8192))
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
            self.n_neighbors_spin.setValue(settings.get("n_neighbors", 15))
            self.min_dist_spin.setValue(settings.get("min_dist", 10))
            self.n_epochs_spin.setValue(settings.get("n_epochs", 0))
            self.learning_rate_spin.setValue(settings.get("learning_rate", 1.0))
            metric_idx = self.metric_combo.findText(settings.get("metric", "cosine"))
            if metric_idx >= 0:
                self.metric_combo.setCurrentIndex(metric_idx)
            self.subsample_checkbox.setChecked(settings.get("subsample_enabled", False))
            self.subsample_size_spin.setValue(settings.get("subsample_size", 70000))

            # Advanced settings (low RAM, CPU cores)
            self.low_memory = settings.get("low_memory", False)
            self.n_jobs = settings.get("n_jobs", os.cpu_count() or 4)

            # Cluster settings
            self.eps_spin.setValue(settings.get("eps", 50))
            self.min_samples_spin.setValue(settings.get("min_samples", 10))
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
            # Auto-Deorphan behavior (validated against known values below)
            _ad = settings.get("auto_deorphan", "Never")
            if _ad in ("Never", "After Load and Compute", "After Regroup"):
                self.auto_deorphan = _ad
            # Optional tag-score DB path
            self.score_db_path = settings.get("score_db_path", "")
            # UI scale (percent); applied at startup, restart required to change
            try:
                self.ui_scale = int(settings.get("ui_scale", 100))
            except (TypeError, ValueError):
                self.ui_scale = 100
            # Sub-clustering settings
            self.sub_eps_spin.setValue(settings.get("sub_eps", 20))
            self.sub_min_samples_spin.setValue(settings.get("sub_min_samples", 4))

            # Filter settings
            self.query_edit.setText(settings.get("query", ""))
            self.whitelist_edit.setText(settings.get("whitelist", ""))
            self.blacklist_edit.setText(settings.get("blacklist", ""))
            self.tokenize = settings.get("tokenize", True)
            self.drop_empty_files = settings.get("drop_empty_files", False)
            self.right_click_select_cohort = bool(settings.get("right_click_select_cohort", False))
            self.min_doc_freq_spin.setValue(settings.get("min_doc_freq", 3))
            self.drop_universal = settings.get("drop_universal_tags", True)

            # Visualization settings (node_size is stored as actual value, displayed x10)
            # Backward compat: fall back to old "min_size" key if "node_size" not present
            node_size_actual = settings.get("node_size", settings.get("min_size", 0.02))
            self.min_size_spin.setValue(node_size_actual * 10.0)
            self.spread_spin.setValue(settings.get("spread", 1.0))
            self.orbit_speed_spin.setValue(settings.get("orbit_speed", 0.2))
            self.transparency_spin.setValue(settings.get("transparency", 0.8))

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

            # Star twinkle settings (set values first, then toggle to avoid premature spawn)
            self.twinkle_count_spin.setValue(settings.get("twinkle_count", 2000))
            self.twinkle_lifespan_min_spin.setValue(settings.get("twinkle_lifespan_min", 1.0))
            self.twinkle_lifespan_max_spin.setValue(settings.get("twinkle_lifespan_max", 6.0))
            self.twinkle_freq_spin.setValue(settings.get("twinkle_freq", 2.0))
            self.twinkle_brightness_spin.setValue(settings.get("twinkle_brightness", 1.5))
            # Toggle last: _on_twinkle_toggle will no-op if no scene loaded yet
            self.twinkle_checkbox.setChecked(settings.get("twinkle_enabled", False))

            # V5: Color scheme setting
            color_scheme_idx = self.color_scheme_combo.findText(settings.get("color_scheme", "Pastel"))
            if color_scheme_idx >= 0:
                self.color_scheme_combo.setCurrentIndex(color_scheme_idx)
            
            # Cohort label settings
            self.cohort_threshold_spin.setValue(settings.get("cohort_threshold", 0.9))
            self.show_cohort_labels_checkbox.setChecked(settings.get("show_cohort_labels", False))
            self.cohort_label_size_spin.setValue(settings.get("cohort_label_size", 18))
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
            self.cohort_label_n_spin.setValue(settings.get("cohort_label_n", 5))
            self.cohort_label_max_tags_spin.setValue(settings.get("cohort_label_max_tags", 5))
            # Smart labels settings (merged into mode combo; "Raw" = disabled)
            smart_mode_idx = self.smart_label_mode_combo.findText(
                settings.get("smart_label_mode", "Absolute Unique")
            )
            if smart_mode_idx >= 0:
                self.smart_label_mode_combo.setCurrentIndex(smart_mode_idx)

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
            'chunk_size_spin': 'valueChanged',
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
            'wobble_speed_spin': 'valueChanged',
            'wobble_azim_range_spin': 'valueChanged',
            'wobble_elev_range_spin': 'valueChanged',
            # QCheckBox -> stateChanged
            'auto_load_checkbox': 'stateChanged',
            'normalize_checkbox': 'stateChanged',
            'supersample_checkbox': 'stateChanged',
            'dim_non_selected_checkbox': 'stateChanged',
            'show_cohort_labels_checkbox': 'stateChanged',
            'dynamic_label_size_checkbox': 'stateChanged',
            'wobble_enabled_checkbox': 'stateChanged',
            # QComboBox -> currentTextChanged
            'client_combo': 'currentTextChanged',
            'tag_service_combo': 'currentTextChanged',
            'algorithm_combo': 'currentTextChanged',
            'metric_combo': 'currentTextChanged',
            'color_scheme_combo': 'currentTextChanged',
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
                "chunk_size": self.chunk_size_spin.value(),
                "max_files": self.max_files_spin.value(),
                "tag_service": self.tag_service_combo.currentText(),
                "use_direct_db": self.use_direct_db,
                "low_memory": self.low_memory,
                "n_jobs": self.n_jobs,
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
                "auto_deorphan": getattr(self, 'auto_deorphan', "Never"),
                "score_db_path": getattr(self, 'score_db_path', ''),
                # UI scale (percent); applied at startup via QT_SCALE_FACTOR
                "ui_scale": getattr(self, 'ui_scale', 100),
                "query": self.query_edit.text(),
                "whitelist": self.whitelist_edit.text(),
                "blacklist": self.blacklist_edit.text(),
                "tokenize": getattr(self, 'tokenize', True),
                "drop_empty_files": getattr(self, 'drop_empty_files', False),
                "right_click_select_cohort": bool(getattr(self, 'right_click_select_cohort', False)),
                "min_doc_freq": self.min_doc_freq_spin.value(),
                "drop_universal_tags": self.drop_universal,
                "node_size": self.min_size_spin.value() / 10.0,
                "spread": self.spread_spin.value(),
                "orbit_speed": self.orbit_speed_spin.value(),
                "transparency": self.transparency_spin.value(),
                # Anti-noise / quality settings
                "supersample": self.supersample_checkbox.isChecked(),
                "supersample_fps": self.supersample_fps_spin.value(),
                # Dim non-selected nodes settings
                "dim_non_selected": self.dim_non_selected_checkbox.isChecked(),
                "dim_alpha": self.dim_alpha_spin.value(),
                "highlight_color": list(self.highlight_color),
                # Star twinkle settings
                "twinkle_enabled": self.twinkle_checkbox.isChecked(),
                "twinkle_count": self.twinkle_count_spin.value(),
                "twinkle_lifespan_min": self.twinkle_lifespan_min_spin.value(),
                "twinkle_lifespan_max": self.twinkle_lifespan_max_spin.value(),
                "twinkle_freq": self.twinkle_freq_spin.value(),
                "twinkle_brightness": self.twinkle_brightness_spin.value(),
                # Color scheme
                "color_scheme": self.color_scheme_combo.currentText(),
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
                # Smart labels settings (merged into mode combo; "Raw" = disabled)
                "smart_label_mode": self.smart_label_mode_combo.currentText(),
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
    def _restore_window_geometry(settings, key, widget):
        """Restore a window's geometry from the settings dict (best effort).

        If the saved position lies on a monitor that no longer exists (e.g. a
        second screen was unplugged), re-center the window on the primary
        screen instead of leaving it stranded off-screen.
        """
        try:
            from PySide6.QtCore import QByteArray, QCoreApplication
            raw = settings.get(key)
            if not raw:
                return False
            ba = QByteArray(bytes.fromhex(raw))
            ok = widget.restoreGeometry(ba)
            # Guard against a saved position on a monitor that no longer exists.
            geo = widget.geometry()
            screens = QCoreApplication.screens()
            if ok and screens and not any(s.availableGeometry().intersects(geo) for s in screens):
                primary = QCoreApplication.primaryScreen()
                if primary is not None:
                    c = primary.availableGeometry().center()
                    widget.move(c.x() - widget.width() // 2, c.y() - widget.height() // 2)
                return False
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
