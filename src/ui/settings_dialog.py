"""Settings dialog for the 3D Tag Space Visualization tab.

Holds the "advanced" settings that were moved out of the main control panel:
- Low RAM toggle (UMAP low_memory)
- CPU core count (UMAP n_jobs)
- Direct DB mode toggle + per-client DB path fields

The dialog reads/writes the owning tab's plain-value attributes directly
(self.low_memory, self.n_jobs, self.use_direct_db, self.client_db_paths)
and calls tab.save_settings() on OK so the JSON settings file stays in sync.
"""

import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QSpinBox, QLineEdit, QGroupBox, QFormLayout,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QInputDialog,
    QComboBox, QTableWidget, QTableWidgetItem, QAbstractItemView,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont


class _ClientTestWorker(QThread):
    """Runs a client connection test off the UI thread."""
    done = Signal(bool, str)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func

    def run(self):
        try:
            ok, msg = self.func()
        except Exception as e:  # noqa: BLE001 - report any failure to the UI
            ok, msg = False, f"Failed: {e}"
        self.done.emit(ok, msg)


class TagMap3DSettingsDialog(QDialog):
    """Settings window for the 3D tag map tab."""

    def __init__(self, tab, parent=None):
        super().__init__(parent)
        self.tab = tab
        self.setWindowTitle("3D Tag Map Settings")
        self.setMinimumSize(520, 420)

        # Load current values from the tab
        self.low_memory = getattr(self.tab, 'low_memory', False)
        self.n_jobs = getattr(self.tab, 'n_jobs', os.cpu_count() or 4)
        self.use_direct_db = getattr(self.tab, 'use_direct_db', False)
        self.client_db_paths = dict(getattr(self.tab, 'client_db_paths', {}))
        self.tokenize = getattr(self.tab, 'tokenize', True)
        self.drop_universal = getattr(self.tab, 'drop_universal', True)
        self.drop_empty_files = getattr(self.tab, 'drop_empty_files', False)

        # Client management state (working copy of clients.json; saved on OK)
        from src.data.clients import load_clients
        self._clients = dict(load_clients())

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # --- Clients group (full CRUD over clients.json) ---
        clients_group = QGroupBox("Clients")
        clients_group.setToolTip("Manage Hydrus clients. Changes are saved to clients.json on OK.")
        cg_layout = QVBoxLayout()

        self.client_list = QListWidget()
        self.client_list.setMaximumHeight(90)
        self.client_list.currentRowChanged.connect(self._on_client_selected)
        cg_layout.addWidget(self.client_list)

        # Field form for the selected client
        cform = QFormLayout()
        self.c_label_edit = QLineEdit()
        self.c_api_url_edit = QLineEdit()
        self.c_api_key_edit = QLineEdit()
        self.c_db_dir_edit = QLineEdit()
        self.c_files_dir_edit = QLineEdit()
        self.c_thumbs_dir_edit = QLineEdit()

        cform.addRow("Label:", self.c_label_edit)
        cform.addRow("API URL:", self.c_api_url_edit)
        cform.addRow("API Key:", self.c_api_key_edit)
        cform.addRow("DB Dir:", self._browse_row(self.c_db_dir_edit))
        cform.addRow("Files Dir:", self._browse_row(self.c_files_dir_edit))
        cform.addRow("Thumbs Dir:", self._browse_row(self.c_thumbs_dir_edit))
        cg_layout.addLayout(cform)

        # Action buttons
        btns = QHBoxLayout()
        self.client_add_btn = QPushButton("+ Add")
        self.client_add_btn.clicked.connect(self._client_add)
        self.client_rename_btn = QPushButton("Rename ID…")
        self.client_rename_btn.clicked.connect(self._client_rename)
        self.client_remove_btn = QPushButton("Remove")
        self.client_remove_btn.clicked.connect(self._client_remove)
        self.client_test_btn = QPushButton("Test Connection")
        self.client_test_btn.clicked.connect(self._client_test)
        for b in (self.client_add_btn, self.client_rename_btn, self.client_remove_btn, self.client_test_btn):
            btns.addWidget(b)
        cg_layout.addLayout(btns)

        self.client_status = QLabel("")
        self.client_status.setWordWrap(True)
        cg_layout.addWidget(self.client_status)

        clients_group.setLayout(cg_layout)
        main_layout.addWidget(clients_group)

        # Populate the client list from the working copy.
        for cid in self._clients.keys():
            self.client_list.addItem(cid)
        if self.client_list.count() > 0:
            self.client_list.setCurrentRow(0)
            self._on_client_selected(0)

        # --- Performance group ---
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout()

        self.low_memory_checkbox = QCheckBox("Low RAM (UMAP low-memory)")
        self.low_memory_checkbox.setChecked(self.low_memory)
        self.low_memory_checkbox.setToolTip(
            "Use the UMAP low-memory algorithm.\nReduces peak memory usage but may be slower.\nEnable if you run out of memory."
        )
        perf_layout.addRow("Low RAM:", self.low_memory_checkbox)

        self.n_jobs_spin = QSpinBox()
        self.n_jobs_spin.setRange(1, 64)
        self.n_jobs_spin.setValue(self.n_jobs)
        self.n_jobs_spin.setToolTip(
            "Number of CPU cores to use for UMAP parallel NN-descent.\nHigher = faster but uses more CPU.\nDefault: all cores."
        )
        perf_layout.addRow("CPU Cores:", self.n_jobs_spin)

        self.tokenize_checkbox = QCheckBox("Tokenize tags")
        self.tokenize_checkbox.setChecked(self.tokenize)
        self.tokenize_checkbox.setToolTip(
            "When enabled, tags are converted to integer indices once at load\n"
            "time and carried through the pipeline as integers. This reduces RAM\n"
            "(no per-node string copies) and replaces repeated string hashing with\n"
            "integer lookups. Strings are only materialised for display.\n"
            "Default: ON."
        )
        perf_layout.addRow(self.tokenize_checkbox)

        self.drop_universal_checkbox = QCheckBox("Drop Universal Tags")
        self.drop_universal_checkbox.setChecked(self.drop_universal)
        self.drop_universal_checkbox.setToolTip(
            "Exclude tags that appear in EVERY loaded file from the vocabulary.\n"
            "These tags provide zero discriminative power (they are usually\n"
            "already visible in your query field) and only add noise dimensions.\n"
            "Useful for large AND queries where all files share the same tags.\n"
            "Default: ON."
        )
        perf_layout.addRow(self.drop_universal_checkbox)

        self.drop_empty_files_checkbox = QCheckBox("Drop empty files")
        self.drop_empty_files_checkbox.setChecked(self.drop_empty_files)
        self.drop_empty_files_checkbox.setToolTip(
            "When enabled, files with no tags remaining after the\n"
            "whitelist/blacklist filters are excluded from the map.\n"
            "When disabled (default), they are kept and appear as untagged\n"
            "nodes at the origin.\n"
            "Only applies when a whitelist or blacklist is set."
        )
        perf_layout.addRow(self.drop_empty_files_checkbox)

        self.right_click_select_cohort_checkbox = QCheckBox("Right-click also selects cohort")
        self.right_click_select_cohort_checkbox.setChecked(
            getattr(self.tab, 'right_click_select_cohort', False)
        )
        self.right_click_select_cohort_checkbox.setToolTip(
            "When enabled, right-clicking the 3D view both re-centers the camera\n"
            "AND selects the cohort under the cursor (navigate + inspect in one\n"
            "gesture). When disabled (default), right-click only moves the camera."
        )
        perf_layout.addRow(self.right_click_select_cohort_checkbox)

        perf_group.setLayout(perf_layout)
        main_layout.addWidget(perf_group)

        # --- UI Scale group (applied at startup via QT_SCALE_FACTOR) ---
        scale_group = QGroupBox("UI Scale")
        scale_layout = QFormLayout()
        self.ui_scale_combo = QComboBox()
        for pct in (100, 125, 150, 200):
            self.ui_scale_combo.addItem(f"{pct}%", userData=pct)
        try:
            current_scale = int(getattr(self.tab, 'ui_scale', 100))
        except (TypeError, ValueError):
            current_scale = 100
        idx = self.ui_scale_combo.findData(current_scale)
        if idx < 0:
            # Non-standard saved value: show it as a custom entry so the UI
            # reflects reality instead of silently resetting to 100%.
            self.ui_scale_combo.addItem(f"{current_scale}% (custom)", userData=current_scale)
            idx = self.ui_scale_combo.count() - 1
        self.ui_scale_combo.setCurrentIndex(idx)
        self.ui_scale_combo.setToolTip(
            "Uniformly scales the whole UI (fonts + widgets) for high-DPI displays.\n"
            "Applied at app startup — restart Hydruxiom after changing it.\n"
            "Note: this multiplies on top of Windows display scaling."
        )
        scale_layout.addRow("Scale:", self.ui_scale_combo)
        scale_group.setLayout(scale_layout)
        main_layout.addWidget(scale_group)

        # --- DBSCAN Optimizer group ---
        opt_group = QGroupBox("DBSCAN Optimizer")
        opt_layout = QFormLayout()

        self.normalize_checkbox = QCheckBox("Normalize positions before DBSCAN")
        self.normalize_checkbox.setChecked(getattr(self.tab, 'normalize_positions', True))
        self.normalize_checkbox.setToolTip(
            "When enabled, positions are normalized (centered + std-scaled) before\n"
            "DBSCAN clustering. This makes eps a relative measure of local density\n"
            "rather than an absolute coordinate distance, so parameters behave\n"
            "consistently across datasets with different file counts / reducer scales.\n"
            "Applies to global clustering, re-cluster, and sub-cohort splitting."
        )
        opt_layout.addRow(self.normalize_checkbox)

        self.opt_max_cohort_size_spin = QSpinBox()
        self.opt_max_cohort_size_spin.setRange(2, 100000)
        self.opt_max_cohort_size_spin.setValue(getattr(self.tab, 'opt_max_cohort_size', 500))
        self.opt_max_cohort_size_spin.setToolTip(
            "Target maximum cohort size. Cohorts larger than this are considered\n"
            "disproportionately large and should be split into smaller sub-cohorts."
        )
        opt_layout.addRow("Max Cohort Size:", self.opt_max_cohort_size_spin)

        self.opt_max_noise_ratio_spin = QSpinBox()
        self.opt_max_noise_ratio_spin.setRange(0, 100)
        self.opt_max_noise_ratio_spin.setValue(getattr(self.tab, 'opt_max_noise_ratio', 10))
        self.opt_max_noise_ratio_spin.setSuffix("%")
        self.opt_max_noise_ratio_spin.setToolTip(
            "Target maximum noise ratio (non-cohorted nodes as % of total).\n"
            "Lower = fewer unclustered nodes. The optimizer tries to stay at or below this."
        )
        opt_layout.addRow("Max Noise Ratio:", self.opt_max_noise_ratio_spin)

        self.opt_max_attempts_spin = QSpinBox()
        self.opt_max_attempts_spin.setRange(1, 500)
        self.opt_max_attempts_spin.setValue(getattr(self.tab, 'opt_max_attempts', 60))
        self.opt_max_attempts_spin.setToolTip(
            "Maximum number of DBSCAN runs the optimizer may attempt\n"
            "to find the ideal eps/min_samples combination."
        )
        opt_layout.addRow("Max Attempts:", self.opt_max_attempts_spin)

        self.opt_eps_min_spin = QSpinBox()
        self.opt_eps_min_spin.setRange(1, 200)
        self.opt_eps_min_spin.setValue(getattr(self.tab, 'opt_eps_min', 5))
        self.opt_eps_min_spin.setToolTip("Lower bound of the EPS (%) search range.")
        opt_layout.addRow("EPS Min (%):", self.opt_eps_min_spin)

        self.opt_eps_max_spin = QSpinBox()
        self.opt_eps_max_spin.setRange(1, 200)
        self.opt_eps_max_spin.setValue(getattr(self.tab, 'opt_eps_max', 100))
        self.opt_eps_max_spin.setToolTip("Upper bound of the EPS (%) search range.")
        opt_layout.addRow("EPS Max (%):", self.opt_eps_max_spin)

        self.opt_min_samples_min_spin = QSpinBox()
        self.opt_min_samples_min_spin.setRange(2, 100)
        self.opt_min_samples_min_spin.setValue(getattr(self.tab, 'opt_min_samples_min', 2))
        self.opt_min_samples_min_spin.setToolTip("Lower bound of the Min Samples search range.")
        opt_layout.addRow("Min Samples Min:", self.opt_min_samples_min_spin)

        self.opt_min_samples_max_spin = QSpinBox()
        self.opt_min_samples_max_spin.setRange(2, 100)
        self.opt_min_samples_max_spin.setValue(getattr(self.tab, 'opt_min_samples_max', 30))
        self.opt_min_samples_max_spin.setToolTip("Upper bound of the Min Samples search range.")
        opt_layout.addRow("Min Samples Max:", self.opt_min_samples_max_spin)

        opt_group.setLayout(opt_layout)
        main_layout.addWidget(opt_group)

        # --- Direct DB group (per-client DB paths are now in the Clients section) ---
        db_group = QGroupBox("Direct DB")
        db_layout = QFormLayout()

        self.direct_db_checkbox = QCheckBox("Use Direct DB (tag loading)")
        self.direct_db_checkbox.setChecked(self.use_direct_db)
        self.direct_db_checkbox.setToolTip(
            "When enabled, load tags directly from the Hydrus client DB\ninstead of the API. Much faster at scale (~99%).\n"
            "Requires a valid DB Dir set for the client (see Clients section). Falls back to API if no path set."
        )
        db_layout.addRow(self.direct_db_checkbox)

        db_group.setLayout(db_layout)
        main_layout.addWidget(db_group)

        # --- Tag Score Weighting group (optional) ---
        score_group = QGroupBox("Tag Score Weighting (optional)")
        score_layout = QFormLayout()
        self.score_db_edit = QLineEdit()
        self.score_db_edit.setPlaceholderText("Path to SQLite with ExternalTagScores table (empty = off)")
        self.score_db_edit.setToolTip(
            "Optional: a SQLite file containing an ExternalTagScores (tag, score) table.\n"
            "When set, each tag's TF-IDF weight is multiplied by (1.0 + 0.1 * score),\n"
            "so high-scored tags pull the embedding more strongly.\n"
            "Leave empty to disable score weighting."
        )
        self.score_db_edit.setText(getattr(self.tab, 'score_db_path', '') or '')
        score_layout.addRow("Score DB Path:", self.score_db_edit)
        score_group.setLayout(score_layout)
        main_layout.addWidget(score_group)

        # --- Shortcuts group (read-only reference; editable in the future) ---
        shortcuts_group = QGroupBox("Shortcuts")
        sc_layout = QVBoxLayout()
        self.shortcuts_table = QTableWidget(0, 2)
        _shortcut_rows = [
            ("F3", "Open settings window"),
            ("F4", "Toggle media viewer"),
            ("F5", "Load & Compute"),
            ("F6", "Recompute (UMAP only)"),
            ("F7", "Regroup (DBSCAN only)"),
            ("F12", "4x snapshot screenshot (saved to screenshots/)"),
            ("Ctrl+X", "Clear session (free memory)"),
            ("Ctrl+S", "Split group (re-cluster selection)"),
            ("Ctrl+E", "Cut out selected cohort"),
            ("Ctrl+T", "Send selection to tab"),
        ]
        self.shortcuts_table.setRowCount(len(_shortcut_rows))
        for r, (key, desc) in enumerate(_shortcut_rows):
            key_item = QTableWidgetItem(key)
            key_item.setFont(QFont("Consolas", 10, QFont.Bold))
            key_item.setFlags(Qt.ItemFlag.NoItemFlags)  # read-only
            self.shortcuts_table.setItem(r, 0, key_item)
            desc_item = QTableWidgetItem(desc)
            desc_item.setFlags(Qt.ItemFlag.NoItemFlags)  # read-only
            self.shortcuts_table.setItem(r, 1, desc_item)
        self.shortcuts_table.horizontalHeader().setVisible(False)
        self.shortcuts_table.verticalHeader().setVisible(False)
        self.shortcuts_table.setColumnWidth(0, 90)
        self.shortcuts_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        sc_layout.addWidget(self.shortcuts_table)
        shortcuts_group.setLayout(sc_layout)
        main_layout.addWidget(shortcuts_group)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.clicked.connect(self.apply_settings)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.apply_dark_theme()

    def apply_settings(self):
        """Apply dialog values back to the tab and save settings."""
        tab = self.tab

        tab.low_memory = self.low_memory_checkbox.isChecked()
        tab.n_jobs = self.n_jobs_spin.value()
        tab.use_direct_db = self.direct_db_checkbox.isChecked()
        tab.tokenize = self.tokenize_checkbox.isChecked()
        tab.drop_universal = self.drop_universal_checkbox.isChecked()
        tab.drop_empty_files = self.drop_empty_files_checkbox.isChecked()
        tab.right_click_select_cohort = self.right_click_select_cohort_checkbox.isChecked()

        # DBSCAN optimizer parameters
        tab.normalize_positions = self.normalize_checkbox.isChecked()
        # Sync the tab's inline normalize checkbox (if present)
        if hasattr(tab, 'normalize_checkbox'):
            tab.normalize_checkbox.setChecked(self.normalize_checkbox.isChecked())
        tab.opt_max_cohort_size = self.opt_max_cohort_size_spin.value()
        tab.opt_max_noise_ratio = self.opt_max_noise_ratio_spin.value()
        tab.opt_max_attempts = self.opt_max_attempts_spin.value()
        tab.opt_eps_min = self.opt_eps_min_spin.value()
        tab.opt_eps_max = self.opt_eps_max_spin.value()
        tab.opt_min_samples_min = self.opt_min_samples_min_spin.value()
        tab.opt_min_samples_max = self.opt_min_samples_max_spin.value()

        # Persist the full clients working copy (Clients section is authoritative).
        self._sync_selected_client_to_dict()
        try:
            from src.data.clients import save_clients
            ok = save_clients(self._clients)
            if not ok:
                QMessageBox.warning(self, "Clients", "Failed to save clients.json.")
        except Exception as e:
            print(f"Error saving clients: {e}")

        # Refresh the tab's client combo + per-client DB path cache.
        paths = {}
        for cid, cfg in self._clients.items():
            db_dir = (cfg.get("db_dir") or "").strip()
            if db_dir:
                paths[cid] = db_dir
        tab.client_db_paths = paths
        if hasattr(tab, "_refresh_client_combo"):
            tab._refresh_client_combo()

        # Optional tag-score weighting
        tab.score_db_path = self.score_db_edit.text().strip()
        try:
            from src.core.tag_scores import reload_external_tag_scores
            reload_external_tag_scores(tab.score_db_path or None)
        except Exception as e:
            print(f"Error reloading tag scores: {e}")

        # UI scale (applied at startup; restart required to take effect)
        new_scale = self.ui_scale_combo.currentData() or 100
        scale_changed = int(getattr(tab, 'ui_scale', 100)) != int(new_scale)
        tab.ui_scale = int(new_scale)

        # Save the rest of the settings JSON
        if hasattr(tab, 'save_settings'):
            tab.save_settings()

        if scale_changed:
            QMessageBox.information(
                self, "UI Scale",
                f"UI scale set to {new_scale}%.\n\nRestart Hydruxiom for it to take effect.",
            )

    # ── Client management helpers ─────────────────────────────────────────

    def _browse_row(self, edit):
        """Return a horizontal layout of [edit][Browse…] for a folder path field."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        btn = QPushButton("…")
        btn.setFixedWidth(32)
        btn.setToolTip("Browse for a folder")
        btn.clicked.connect(lambda _=False, e=edit: self._browse_folder(e))
        row.addWidget(btn)
        return row

    def _browse_folder(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        if d:
            edit.setText(d)

    def _current_client_id(self):
        item = self.client_list.currentItem()
        return item.text() if item else None

    def _on_client_selected(self, row):
        """Load the selected client's fields into the form."""
        cid = self._current_client_id()
        if not cid:
            return
        cfg = self._clients.get(cid, {})
        self.c_label_edit.setText(cfg.get("label", cid))
        self.c_api_url_edit.setText(cfg.get("api_url", ""))
        self.c_api_key_edit.setText(cfg.get("api_key", ""))
        self.c_db_dir_edit.setText(cfg.get("db_dir", ""))
        self.c_files_dir_edit.setText(cfg.get("files_dir", ""))
        self.c_thumbs_dir_edit.setText(cfg.get("thumbs_dir", ""))

    def _sync_selected_client_to_dict(self):
        """Write the current form values back into the working dict for the selected client."""
        cid = self._current_client_id()
        if not cid:
            return
        cfg = self._clients.setdefault(cid, {})
        cfg["label"] = self.c_label_edit.text().strip() or cid
        cfg["api_url"] = self.c_api_url_edit.text().strip()
        cfg["api_key"] = self.c_api_key_edit.text().strip()
        cfg["db_dir"] = self.c_db_dir_edit.text().strip()
        cfg["files_dir"] = self.c_files_dir_edit.text().strip()
        cfg["thumbs_dir"] = self.c_thumbs_dir_edit.text().strip()

    def _client_add(self):
        new_id, ok = QInputDialog.getText(self, "Add Client", "New client ID (short, e.g. HE):")
        if not ok or not new_id.strip():
            return
        new_id = new_id.strip()
        if new_id in self._clients:
            QMessageBox.warning(self, "Clients", f"Client '{new_id}' already exists.")
            return
        self._sync_selected_client_to_dict()  # save current edits first
        self._clients[new_id] = {
            "label": new_id, "api_url": "", "api_key": "",
            "db_dir": "", "files_dir": "", "thumbs_dir": "",
        }
        idx = self.client_list.count()
        self.client_list.insertItem(idx, new_id)
        self.client_list.setCurrentRow(idx)

    def _client_rename(self):
        old_id = self._current_client_id()
        if not old_id:
            return
        new_id, ok = QInputDialog.getText(self, "Rename Client", f"New ID for '{old_id}':", text=old_id)
        if not ok or not new_id.strip():
            return
        new_id = new_id.strip()
        if new_id == old_id:
            return
        if new_id in self._clients:
            QMessageBox.warning(self, "Clients", f"Client '{new_id}' already exists.")
            return
        self._sync_selected_client_to_dict()  # save current edits under old id first
        cfg = self._clients.pop(old_id)
        cfg["label"] = new_id
        self._clients[new_id] = cfg
        row = self.client_list.row(self.client_list.currentItem())
        self.client_list.item(row).setText(new_id)

    def _client_remove(self):
        cid = self._current_client_id()
        if not cid:
            return
        reply = QMessageBox.question(
            self, "Remove Client", f"Remove client '{cid}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        del self._clients[cid]
        row = self.client_list.row(self.client_list.currentItem())
        self.client_list.takeItem(row)
        # Select the first remaining client (if any).
        if self.client_list.count() > 0:
            self.client_list.setCurrentRow(0)

    def _client_test(self):
        """Test the connection using the current form values (in a worker thread)."""
        cid = self._current_client_id() or "(new)"
        api_url = self.c_api_url_edit.text().strip()
        api_key = self.c_api_key_edit.text().strip()
        if not api_url or not api_key:
            self.client_status.setText("Enter API URL and API Key first.")
            return
        self.client_status.setText(f"Testing {cid}…")

        def _run():
            try:
                import hydrus_api
                client = hydrus_api.Client(access_key=api_key, api_url=api_url)
                # Light call to verify connectivity.
                client.get_services()
                return True, "Connected ✓"
            except Exception as e:
                return False, f"Failed: {e}"

        self._test_worker = _ClientTestWorker(_run)
        self._test_worker.done.connect(lambda ok, msg: self.client_status.setText(msg))
        self._test_worker.start()

    def apply_dark_theme(self):
        """Apply a dark theme matching the rest of the app."""
        try:
            from src.ui.ui_utils import apply_dark_theme
            apply_dark_theme(self)
        except Exception:
            pass

        style = (
            "QSpinBox { background-color: rgb(33, 37, 43); color: rgb(255, 255, 255);"
            " border-radius: 5px; padding: 3px; }"
            "QSpinBox::up-button { subcontrol-origin: padding; subcontrol-position: top right;"
            " width: 16px; height: 16px; }"
            "QSpinBox::down-button { subcontrol-origin: padding; subcontrol-position: bottom right;"
            " width: 16px; height: 16px; }"
        )
        self.n_jobs_spin.setStyleSheet(style)
        # Style the DBSCAN optimizer spinboxes
        for spin in [
            self.opt_max_cohort_size_spin,
            self.opt_max_noise_ratio_spin,
            self.opt_max_attempts_spin,
            self.opt_eps_min_spin,
            self.opt_eps_max_spin,
            self.opt_min_samples_min_spin,
            self.opt_min_samples_max_spin,
        ]:
            spin.setStyleSheet(style)

        line_style = (
            "QLineEdit { background-color: rgb(33, 37, 43); border-radius: 5px;"
            " border: 2px solid rgb(33, 37, 43); padding: 8px; }"
            "QLineEdit:hover { border: 2px solid rgb(64, 71, 88); }"
            "QLineEdit:focus { border: 2px solid rgb(91, 101, 124); }"
        )
        for edit in (self.c_label_edit, self.c_api_url_edit, self.c_api_key_edit,
                     self.c_db_dir_edit, self.c_files_dir_edit, self.c_thumbs_dir_edit):
            edit.setStyleSheet(line_style)

        # Client list widget dark styling
        self.client_list.setStyleSheet(
            "QListWidget { background-color: rgb(33, 37, 43); color: rgb(221, 221, 221);"
            " border: 1px solid rgb(44, 49, 58); }"
            "QListWidget::item:selected { background-color: rgb(60, 80, 180); color: white; }"
        )

        # Shortcuts table dark styling (read-only reference)
        self.shortcuts_table.setStyleSheet(
            "QTableWidget { background-color: rgb(33, 37, 43); color: rgb(221, 221, 221);"
            " border: 1px solid rgb(44, 49, 58); gridline-color: rgb(44, 49, 58); }"
        )
