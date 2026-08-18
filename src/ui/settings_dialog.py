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
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QGroupBox, QFormLayout,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QInputDialog,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QTabWidget,
    QScrollArea,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QIntValidator

from src.ui.smart_scale import (
    SMART_SCALE_SETTINGS, SMART_SCALE_KEYS, default_profiles, read_current_values,
)


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

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(12)

        # The dialog is tabbed: General | UI | Clients | Shortcuts | Smart Scale.
        # Each tab's content is wrapped in a scroll area (see _scroll_tab) so
        # settings never get squished on small screens — the vertical scrollbar
        # only appears when the content doesn't fit.
        self.settings_tabs = QTabWidget()

        general_tab = QWidget()
        main_layout = QVBoxLayout(general_tab)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        # Containers for the other tabs; their groups are built further down in
        # this method and added to these layouts.
        ui_tab = QWidget()
        ui_tab_layout = QVBoxLayout(ui_tab)
        ui_tab_layout.setContentsMargins(8, 8, 8, 8)
        ui_tab_layout.setSpacing(12)

        clients_tab = QWidget()
        clients_tab_layout = QVBoxLayout(clients_tab)
        clients_tab_layout.setContentsMargins(8, 8, 8, 8)
        clients_tab_layout.setSpacing(12)

        shortcuts_tab = QWidget()
        shortcuts_tab_layout = QVBoxLayout(shortcuts_tab)
        shortcuts_tab_layout.setContentsMargins(8, 8, 8, 8)
        shortcuts_tab_layout.setSpacing(12)

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

        # Chunk Size (moved here from the left panel). Stored on the tab as a
        # plain int attribute; written back in apply_settings().
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(50, 100000000)
        try:
            self.chunk_size_spin.setValue(int(getattr(self.tab, 'chunk_size', 8192)))
        except (TypeError, ValueError):
            self.chunk_size_spin.setValue(8192)
        self.chunk_size_spin.setToolTip("Number of files to fetch per API request.\nLarger = faster but may timeout.\nSmaller = more requests but more reliable.\nDefault: 8192")
        cform.addRow("Chunk Size:", self.chunk_size_spin)

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
        clients_tab_layout.addWidget(clients_group)

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

        # --- Tag Query group (clickable tag grid layout in the left panel) ---
        tq_group = QGroupBox("Tag Query")
        tq_layout = QFormLayout()

        self.tag_query_columns_spin = QSpinBox()
        self.tag_query_columns_spin.setRange(1, 20)
        self.tag_query_columns_spin.setValue(getattr(self.tab, 'tag_query_columns', 3))
        self.tag_query_columns_spin.setToolTip(
            "Number of columns in the clickable tag grid (Filter Settings).\n"
            "More columns = wider rows, fewer lines. Combined with Rows it\n"
            "determines how many tags are shown at once."
        )
        tq_layout.addRow("Columns:", self.tag_query_columns_spin)

        self.tag_query_rows_spin = QSpinBox()
        self.tag_query_rows_spin.setRange(1, 200)
        self.tag_query_rows_spin.setValue(getattr(self.tab, 'tag_query_rows', 14))
        self.tag_query_rows_spin.setToolTip(
            "Number of rows in the clickable tag grid.\n"
            "Max tags shown = Columns x Rows (the rest are collapsed into a\n"
            "'... and N more tags' label). Increase for files with many tags."
        )
        tq_layout.addRow("Rows:", self.tag_query_rows_spin)

        self.tag_query_max_label = QLabel("")
        self.tag_query_max_label.setStyleSheet("color: #888; font-size: 10px;")
        self._update_tag_query_max_label()
        self.tag_query_columns_spin.valueChanged.connect(self._update_tag_query_max_label)
        self.tag_query_rows_spin.valueChanged.connect(self._update_tag_query_max_label)
        tq_layout.addRow("Max tags shown:", self.tag_query_max_label)

        tq_group.setLayout(tq_layout)
        main_layout.addWidget(tq_group)

        # --- UI group (scale + camera glide behavior) ---
        scale_group = QGroupBox("UI")
        scale_layout = QFormLayout()

        self.auto_center_on_selection_checkbox = QCheckBox("Auto-center on selection")
        self.auto_center_on_selection_checkbox.setChecked(
            getattr(self.tab, 'auto_center_on_selection', False)
        )
        self.auto_center_on_selection_checkbox.setToolTip(
            "When enabled, selecting a cohort (e.g. via WASD navigation) also moves\n"
            "the camera center to that cohort's centroid. Uses the smooth-center glide\n"
            "if that option is on."
        )
        scale_layout.addRow(self.auto_center_on_selection_checkbox)

        self.smooth_center_transition_checkbox = QCheckBox("Smooth center transition")
        self.smooth_center_transition_checkbox.setChecked(
            getattr(self.tab, 'smooth_center_transition', False)
        )
        self.smooth_center_transition_checkbox.setToolTip(
            "When enabled, right-clicking a node glides the camera to it instead\n"
            "of teleporting. Duration scales with distance and the speed below.\n"
            "Grabbing the camera mid-glide cancels the animation."
        )
        scale_layout.addRow(self.smooth_center_transition_checkbox)

        self.smooth_center_speed_spin = QDoubleSpinBox()
        self.smooth_center_speed_spin.setRange(0.1, 50.0)
        self.smooth_center_speed_spin.setValue(float(getattr(self.tab, 'smooth_center_speed', 1.0)))
        self.smooth_center_speed_spin.setSingleStep(0.1)
        self.smooth_center_speed_spin.setDecimals(1)
        self.smooth_center_speed_spin.setSuffix(" u/s")
        self.smooth_center_speed_spin.setToolTip(
            "Glide speed for the smooth center transition (scene units per second).\n"
            "Higher = faster. Duration is distance / speed, clamped to 0.25–15 s.\n"
            "Default 1 u/s gives a slow, deliberate glide."
        )
        scale_layout.addRow("Center Glide Speed:", self.smooth_center_speed_spin)

        self.wasd_paths_checkbox = QCheckBox("Show WASD navigation paths")
        self.wasd_paths_checkbox.setChecked(getattr(self.tab, 'wasd_paths_enabled', True))
        self.wasd_paths_checkbox.setToolTip(
            "When enabled, pressing W/S/A/D shows preview arrows to the nearest cohort\n"
            "in each screen direction. The arrows fade out over ~2 s when you stop\n"
            "navigating (e.g. click elsewhere). Uncheck to hide them entirely."
        )
        scale_layout.addRow(self.wasd_paths_checkbox)

        self.ui_scale_combo = QComboBox()
        self.ui_scale_combo.setEditable(True)
        for pct in (50, 75, 100, 125, 150, 200):
            self.ui_scale_combo.addItem(f"{pct}%", userData=pct)
        # Allow typing any integer percentage between 25 and 250.
        self.ui_scale_combo.setValidator(QIntValidator(25, 250))
        try:
            current_scale = int(getattr(self.tab, 'ui_scale', 100))
        except (TypeError, ValueError):
            current_scale = 100
        current_scale = max(25, min(250, current_scale))
        idx = self.ui_scale_combo.findData(current_scale)
        if idx < 0:
            # Non-standard saved value: show it as a custom entry so the UI
            # reflects reality instead of silently resetting to 100%.
            self.ui_scale_combo.addItem(f"{current_scale}%", userData=current_scale)
            idx = self.ui_scale_combo.count() - 1
        self.ui_scale_combo.setCurrentIndex(idx)
        self.ui_scale_combo.setToolTip(
            "Uniformly scales the whole UI (fonts + widgets).\n"
            "Use values below 100% on low-resolution screens where the UI\n"
            "doesn't fit; above 100% for high-DPI displays.\n"
            "Any integer from 25 to 250 can be typed in.\n"
            "Applied at app startup — restart Hydruxiom after changing it.\n"
            "Note: this multiplies on top of Windows display scaling."
        )
        scale_layout.addRow("Scale:", self.ui_scale_combo)
        scale_group.setLayout(scale_layout)
        ui_tab_layout.addWidget(scale_group)

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

        # Auto-Deorphan: when to automatically assign noise (-1) nodes to their
        # nearest cohort after a clustering operation.
        self.auto_deorphan_combo = QComboBox()
        for label in ("Never", "After Load and Compute", "After Regroup"):
            self.auto_deorphan_combo.addItem(label)
        _ad_idx = self.auto_deorphan_combo.findText(
            getattr(self.tab, 'auto_deorphan', "Never")
        )
        if _ad_idx >= 0:
            self.auto_deorphan_combo.setCurrentIndex(_ad_idx)
        else:
            self.auto_deorphan_combo.setCurrentIndex(0)
        self.auto_deorphan_combo.setToolTip(
            "Automatically run Deorphan (assign every noise/-1 node to the cohort\n"
            "of its nearest non-noise node) after the chosen operation.\n"
            "- Never: only when you click the Deorphan button manually.\n"
            "- After Load and Compute: deorphan once a fresh load finishes.\n"
            "- After Regroup: deorphan every time DBSCAN re-clustering runs."
        )
        opt_layout.addRow("Auto-Deorphan:", self.auto_deorphan_combo)

        # Auto-split: after Load & Compute, repeatedly select + split the largest
        # cohort while it exceeds the threshold (up to max cycles). The master
        # enable toggle is here; Max Cycles 0 also disables as a fallback.
        self.auto_split_enabled_checkbox = QCheckBox("Auto-Split oversized cohorts")
        self.auto_split_enabled_checkbox.setChecked(
            getattr(self.tab, 'auto_split_enabled', True)
        )
        self.auto_split_enabled_checkbox.setToolTip(
            "Master switch for auto-splitting. When enabled, after Load & Compute\n"
            "any cohort larger than the threshold is selected and split (like the\n"
            "Split group button), repeating until none exceed it or max cycles hit.\n"
            "Uncheck to turn the feature off entirely."
        )
        opt_layout.addRow(self.auto_split_enabled_checkbox)

        self.auto_split_threshold_spin = QSpinBox()
        self.auto_split_threshold_spin.setRange(0, 1000000)
        self.auto_split_threshold_spin.setValue(getattr(self.tab, 'auto_split_threshold', 5000))
        self.auto_split_threshold_spin.setToolTip(
            "Cohort size threshold for auto-splitting. After Load & Compute, any\n"
            "cohort larger than this is selected and split (like the Split group\n"
            "button), then re-checked. Set to 0 to disable."
        )
        opt_layout.addRow("Auto-Split Threshold:", self.auto_split_threshold_spin)

        self.auto_split_max_cycles_spin = QSpinBox()
        self.auto_split_max_cycles_spin.setRange(0, 50)
        self.auto_split_max_cycles_spin.setValue(getattr(self.tab, 'auto_split_max_cycles', 3))
        self.auto_split_max_cycles_spin.setToolTip(
            "Maximum number of auto-split cycles. The loop stops early when no\n"
            "cohort exceeds the threshold or a split fails to shrink its target.\n"
            "Set to 0 to disable auto-split entirely."
        )
        opt_layout.addRow("Auto-Split Max Cycles:", self.auto_split_max_cycles_spin)

        opt_group.setLayout(opt_layout)
        main_layout.addWidget(opt_group)

        # --- Explore (helicopter orbit) group ---
        explore_group = QGroupBox("Explore")
        explore_layout = QFormLayout()

        self.explore_mode_combo = QComboBox()
        self.explore_mode_combo.addItems(["Random", "Linear Path", "Contrast", "Size"])
        _em_idx = self.explore_mode_combo.findText(getattr(self.tab, 'explore_mode', "Random"))
        if _em_idx >= 0:
            self.explore_mode_combo.setCurrentIndex(_em_idx)
        else:
            self.explore_mode_combo.setCurrentIndex(0)
        self.explore_mode_combo.setToolTip(
            "How Explore picks which cohorts to visit and in what order:\n"
            "- Random: shuffled, every cohort once.\n"
            "- Linear Path: starts at one spatial extreme and hops to the nearest\n"
            "  unvisited cohort — a short-step sweep that ends near the other extreme.\n"
            "- Contrast: greedy farthest-point sampling; each next cohort is the most\n"
            "  distant from all previously visited, for the most varied tour.\n"
            "- Size: visits cohorts biggest to smallest, then loops back to repeat."
        )
        explore_layout.addRow("Mode:", self.explore_mode_combo)

        self.explore_show_path_checkbox = QCheckBox("Show path preview")
        self.explore_show_path_checkbox.setChecked(getattr(self.tab, 'explore_show_path', False))
        self.explore_show_path_checkbox.setToolTip(
            "When enabled, Explore draws the planned route before flying it:\n"
            "- orange lines = approach / travel between targets\n"
            "- blue rings = orbit around each target\n"
            "- numbered labels mark each stop in visit order."
        )
        explore_layout.addRow(self.explore_show_path_checkbox)

        self.explore_accel_spin = QDoubleSpinBox()
        self.explore_accel_spin.setRange(0.00001, 5.0)
        self.explore_accel_spin.setValue(float(getattr(self.tab, 'explore_accel', 0.6)))
        self.explore_accel_spin.setSingleStep(0.0001)
        self.explore_accel_spin.setDecimals(5)
        self.explore_accel_spin.setToolTip("Approach speed factor for Explore (higher = faster fly-in).\nCan be very small (e.g. 0.00001) for a slow, deliberate approach.")
        explore_layout.addRow("Accel:", self.explore_accel_spin)

        self.explore_decel_spin = QDoubleSpinBox()
        self.explore_decel_spin.setRange(0.00001, 5.0)
        self.explore_decel_spin.setValue(float(getattr(self.tab, 'explore_decel', 0.6)))
        self.explore_decel_spin.setSingleStep(0.0001)
        self.explore_decel_spin.setDecimals(5)
        self.explore_decel_spin.setToolTip("Deceleration factor as Explore settles into orbit (higher = snappier).\nCan be very small (e.g. 0.00001) for a gentle settle.")
        explore_layout.addRow("Decel:", self.explore_decel_spin)

        self.explore_orbit_radius_base_spin = QDoubleSpinBox()
        self.explore_orbit_radius_base_spin.setRange(0.0, 100.0)
        self.explore_orbit_radius_base_spin.setValue(float(getattr(self.tab, 'explore_orbit_radius_base', 8.0)))
        self.explore_orbit_radius_base_spin.setSingleStep(0.5)
        self.explore_orbit_radius_base_spin.setDecimals(3)
        self.explore_orbit_radius_base_spin.setToolTip("Base orbit distance from a cohort's centroid in Explore.\nCan be very small (e.g. 0.001) for a tight close-up orbit.")
        explore_layout.addRow("Orbit Radius Base:", self.explore_orbit_radius_base_spin)

        self.explore_orbit_size_factor_spin = QDoubleSpinBox()
        self.explore_orbit_size_factor_spin.setRange(0.0, 20.0)
        self.explore_orbit_size_factor_spin.setValue(float(getattr(self.tab, 'explore_orbit_size_factor', 2.0)))
        self.explore_orbit_size_factor_spin.setSingleStep(0.5)
        self.explore_orbit_size_factor_spin.setToolTip("Adds this × √(cohort size) to the orbit radius so big cohorts get more room.")
        explore_layout.addRow("Orbit Size Factor:", self.explore_orbit_size_factor_spin)

        self.explore_orbit_speed_spin = QDoubleSpinBox()
        self.explore_orbit_speed_spin.setRange(0.5, 120.0)
        self.explore_orbit_speed_spin.setValue(float(getattr(self.tab, 'explore_orbit_speed', 12.0)))
        self.explore_orbit_speed_spin.setSingleStep(1.0)
        self.explore_orbit_speed_spin.setSuffix(" °/s")
        self.explore_orbit_speed_spin.setToolTip(
            "Angular speed while orbiting a cohort (degrees per second).\n"
            "Higher = faster circling. The approach banks into the orbit at this\n"
            "same speed, so it also affects how quickly you settle in."
        )
        explore_layout.addRow("Orbit Speed:", self.explore_orbit_speed_spin)

        self.explore_cycles_spin = QSpinBox()
        self.explore_cycles_spin.setRange(1, 20)
        self.explore_cycles_spin.setValue(int(getattr(self.tab, 'explore_cycles', 3)))
        self.explore_cycles_spin.setToolTip("How many full orbits to make around each cohort before flying to the next.")
        explore_layout.addRow("Orbit Cycles:", self.explore_cycles_spin)

        self.explore_max_orbit_time_spin = QDoubleSpinBox()
        self.explore_max_orbit_time_spin.setRange(0.0, 600.0)
        self.explore_max_orbit_time_spin.setValue(float(getattr(self.tab, 'explore_max_orbit_time', 30.0)))
        self.explore_max_orbit_time_spin.setSingleStep(5.0)
        self.explore_max_orbit_time_spin.setSuffix(" s")
        self.explore_max_orbit_time_spin.setToolTip(
            "Maximum time to spend orbiting a cohort before flying to the next.\n"
            "Whichever is reached first — Orbit Cycles or this time limit — advances\n"
            "the tour. Set to 0 to disable the time limit (cycles only)."
        )
        explore_layout.addRow("Max Orbit Time:", self.explore_max_orbit_time_spin)

        self.explore_elevation_spin = QDoubleSpinBox()
        self.explore_elevation_spin.setRange(10.0, 80.0)
        self.explore_elevation_spin.setValue(float(getattr(self.tab, 'explore_elevation', 40.0)))
        self.explore_elevation_spin.setSingleStep(5.0)
        self.explore_elevation_spin.setSuffix("°")
        self.explore_elevation_spin.setToolTip("Camera elevation (height above the plane) while orbiting — the helicopter look.")
        explore_layout.addRow("Orbit Elevation:", self.explore_elevation_spin)

        explore_group.setLayout(explore_layout)
        ui_tab_layout.addWidget(explore_group)

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

        # --- Session auto-save group ---
        session_group = QGroupBox("Session Auto-Save")
        session_layout = QFormLayout()
        self.session_save_delay_spin = QSpinBox()
        self.session_save_delay_spin.setRange(0, 3600)
        self.session_save_delay_spin.setValue(getattr(self.tab, 'session_save_delay', 60))
        self.session_save_delay_spin.setSuffix(" s")
        self.session_save_delay_spin.setToolTip(
            "Delay before the session file (sessions/latest.npz) is written after\n"
            "the last operation that changes session data. Each new change resets\n"
            "the timer, so iterating settings or chaining operations only triggers\n"
            "a single write once you pause.\n"
            "The session is always saved immediately when the window closes.\n"
            "Set to 0 to save right after every operation (legacy behavior)."
        )
        session_layout.addRow("Save Delay:", self.session_save_delay_spin)
        session_group.setLayout(session_layout)
        main_layout.addWidget(session_group)

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
            ("F5", "Load and Compute"),
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
        shortcuts_tab_layout.addWidget(shortcuts_group)

        # --- Smart Scale tab (node-count-based automatic settings) ---
        smart_tab = QWidget()
        self._build_smart_scale_tab(smart_tab)

        # Assemble the tabs (each scrollable) in the requested order.
        self.settings_tabs.addTab(self._scroll_tab(general_tab), "General")
        self.settings_tabs.addTab(self._scroll_tab(ui_tab), "UI")
        self.settings_tabs.addTab(self._scroll_tab(clients_tab), "Clients")
        self.settings_tabs.addTab(self._scroll_tab(shortcuts_tab), "Shortcuts")
        self.settings_tabs.addTab(self._scroll_tab(smart_tab), "Smart Scale")
        outer_layout.addWidget(self.settings_tabs)

        # --- Buttons (shared by all tabs; sit below the tab widget) ---
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.clicked.connect(self.apply_settings)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        outer_layout.addLayout(button_layout)

        self.setLayout(outer_layout)
        self.apply_dark_theme()

    @staticmethod
    def _scroll_tab(content):
        """Wrap a tab page in a QScrollArea that scrolls only when needed.

        Vertical scrollbar policy is AsNeeded, so the bar (and its space)
        appears only when the content doesn't fit; horizontal scrolling is
        disabled and widgets stretch to the tab width instead.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(content)
        return scroll

    # ------------------------------------------------------------------
    # Smart Scale tab
    # ------------------------------------------------------------------

    def _build_smart_scale_tab(self, container):
        """Build the Smart Scale tab: size-based profiles with per-row parameters.

        A profile = an ENDPOINT (file count) + a user-chosen set of settings to
        override for that size range. The left list shows all profiles; selecting
        one opens its editor below where you add/remove individual parameters and
        set their values. Profiles are sorted by endpoint: the highest endpoint at
        or below your file count wins (topmost covers above, bottom covers below).
        """
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        intro = QLabel(
            "Smart Scale automatically picks a settings profile based on how many files\n"
            "are loaded. Each profile has an ENDPOINT (a file count) and the specific\n"
            "settings you choose to override for that size range.\n\n"
            "Profiles are sorted by endpoint: the highest endpoint at or below your file\n"
            "count wins; the topmost also covers everything above it, the bottom covers\n"
            "everything below its endpoint. Select a profile on the left, then add/remove\n"
            "the parameters you want it to control and set their values.\n\n"
            "Enable Smart Scale in the main window to activate it."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # --- Profile list (left) + actions (right) ---
        top = QHBoxLayout()

        self.smart_profile_list = QListWidget()
        self.smart_profile_list.setMinimumWidth(260)
        self.smart_profile_list.currentRowChanged.connect(self._smart_on_profile_selected)
        top.addWidget(self.smart_profile_list, 1)

        actions = QVBoxLayout()
        add_btn = QPushButton("+ Add profile")
        add_btn.clicked.connect(self._smart_add_profile)
        dup_btn = QPushButton("Duplicate")
        dup_btn.clicked.connect(self._smart_duplicate_profile)
        rem_btn = QPushButton("Remove")
        rem_btn.clicked.connect(self._smart_remove_profile)
        up_btn = QPushButton("Move Up")
        up_btn.clicked.connect(lambda: self._smart_move_profile(-1))
        down_btn = QPushButton("Move Down")
        down_btn.clicked.connect(lambda: self._smart_move_profile(1))
        for b in (add_btn, dup_btn, rem_btn, up_btn, down_btn):
            actions.addWidget(b)
        actions.addStretch()
        top.addLayout(actions)
        layout.addLayout(top)

        # --- Selected profile editor (bottom) ---
        self.smart_editor = QWidget()
        ed_layout = QVBoxLayout(self.smart_editor)
        ed_layout.setContentsMargins(0, 6, 0, 0)
        ed_layout.setSpacing(8)

        ep_row = QHBoxLayout()
        ep_label = QLabel("Endpoint (files):")
        self.smart_endpoint_spin = QSpinBox()
        self.smart_endpoint_spin.setRange(1, 100000000)
        self.smart_endpoint_spin.valueChanged.connect(self._smart_on_endpoint_changed)
        ep_row.addWidget(ep_label)
        ep_row.addWidget(self.smart_endpoint_spin)
        ep_row.addStretch()
        ed_layout.addLayout(ep_row)

        # Parameter table: column 0 = setting (combo), column 1 = value.
        self.smart_param_table = QTableWidget(0, 2)
        headers = self.smart_param_table.horizontalHeader()
        headers.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        headers.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.smart_param_table.verticalHeader().setVisible(False)
        self.smart_param_table.setMinimumHeight(200)
        ed_layout.addWidget(self.smart_param_table)

        param_btns = QHBoxLayout()
        addp_btn = QPushButton("+ Add parameter")
        addp_btn.clicked.connect(self._smart_add_param)
        remp_btn = QPushButton("Remove selected parameter")
        remp_btn.clicked.connect(self._smart_remove_param)
        param_btns.addWidget(addp_btn)
        param_btns.addWidget(remp_btn)
        param_btns.addStretch()
        ed_layout.addLayout(param_btns)

        layout.addWidget(self.smart_editor)

        # Load existing profiles. An empty saved list stays empty (the user adds
        # rows they want); no starter profiles are injected by default.
        _saved = getattr(self.tab, 'smart_scale_profiles', None)
        self._smart_profiles = list(_saved) if isinstance(_saved, list) else []
        self._smart_refresh_profile_list(select_first=True)

    def _smart_sorted(self):
        """Profiles sorted by endpoint ascending."""
        return sorted(self._smart_profiles, key=lambda p: int(p.get("endpoint", 0)))

    def _smart_refresh_profile_list(self, select_first=False):
        """Rebuild the profile list widget from self._smart_profiles."""
        self.smart_profile_list.blockSignals(True)
        self.smart_profile_list.clear()
        for prof in self._smart_sorted():
            settings = prof.get("settings", {})
            names = ", ".join(
                dict((k, l) for k, l, _t in SMART_SCALE_SETTINGS).get(k, k)
                for k in settings.keys()
            ) or "(no parameters)"
            self.smart_profile_list.addItem(f"{int(prof.get('endpoint', 0)):,} files — {names}")
        self.smart_profile_list.blockSignals(False)

        if select_first and self.smart_profile_list.count() > 0:
            self.smart_profile_list.setCurrentRow(0)
        elif self.smart_profile_list.currentRow() < 0 and self.smart_profile_list.count() > 0:
            self.smart_profile_list.setCurrentRow(0)
        else:
            # Keep the editor in sync with whatever is currently selected.
            row = self.smart_profile_list.currentRow()
            if row >= 0:
                self._smart_load_editor(row)

    def _smart_current_index(self):
        """Index into the sorted profile list for the current selection (-1 if none)."""
        return self.smart_profile_list.currentRow()

    def _smart_load_editor(self, row):
        """Populate the endpoint spin + parameter table from profiles[row]."""
        profiles = self._smart_sorted()
        if not (0 <= row < len(profiles)):
            self.smart_endpoint_spin.blockSignals(True)
            self.smart_endpoint_spin.setValue(1)
            self.smart_endpoint_spin.blockSignals(False)
            self.smart_param_table.setRowCount(0)
            return
        prof = profiles[row]
        self.smart_endpoint_spin.blockSignals(True)
        self.smart_endpoint_spin.setValue(int(prof.get("endpoint", 1)))
        self.smart_endpoint_spin.blockSignals(False)

        settings = prof.get("settings", {})
        # Preserve a stable display order: canonical key order, then any extras.
        ordered_keys = [k for k in SMART_SCALE_KEYS if k in settings] + \
                       [k for k in settings.keys() if k not in SMART_SCALE_KEYS]
        self.smart_param_table.setRowCount(len(ordered_keys))
        for r, key in enumerate(ordered_keys):
            combo = QComboBox()
            combo.addItems([label for _k, label, _t in SMART_SCALE_SETTINGS])
            idx = next((i for i, (k, _l, _t) in enumerate(SMART_SCALE_SETTINGS) if k == key), 0)
            combo.setCurrentIndex(idx)
            self.smart_param_table.setCellWidget(r, 0, combo)

            val_item = QTableWidgetItem()
            val = settings.get(key)
            if val is not None:
                fv = float(val)
                val_item.setText(str(int(fv)) if fv.is_integer() else str(fv))
            self.smart_param_table.setItem(r, 1, val_item)

    def _smart_on_profile_selected(self, row):
        """A different profile was selected in the list -> load its editor."""
        if row >= 0:
            self._smart_load_editor(row)

    def _smart_commit_endpoint(self):
        """Write the endpoint spin value back into the currently selected profile."""
        row = self._smart_current_index()
        profiles = self._smart_sorted()
        if not (0 <= row < len(profiles)):
            return
        profiles[row]["endpoint"] = int(self.smart_endpoint_spin.value())

    def _smart_on_endpoint_changed(self, _value):
        """Endpoint edited -> persist into the selected profile + refresh list label."""
        self._smart_commit_endpoint()
        # Rebuild labels without disturbing selection.
        current_row = self._smart_current_index()
        self.smart_profile_list.blockSignals(True)
        self.smart_profile_list.clear()
        for prof in self._smart_sorted():
            settings = prof.get("settings", {})
            names = ", ".join(
                dict((k, l) for k, l, _t in SMART_SCALE_SETTINGS).get(k, k)
                for k in settings.keys()
            ) or "(no parameters)"
            self.smart_profile_list.addItem(f"{int(prof.get('endpoint', 0)):,} files — {names}")
        if 0 <= current_row < self.smart_profile_list.count():
            self.smart_profile_list.setCurrentRow(current_row)
        self.smart_profile_list.blockSignals(False)

    def _smart_add_param(self):
        """Add a new (setting, value) row to the selected profile's parameter table."""
        row = self._smart_current_index()
        profiles = self._smart_sorted()
        if not (0 <= row < len(profiles)):
            return
        settings = profiles[row].setdefault("settings", {})

        # Pre-fill with a setting not already present, using the current widget value.
        existing = set(settings.keys())
        default_key = next((k for k in SMART_SCALE_KEYS if k not in existing), None)
        r = self.smart_param_table.rowCount()
        self.smart_param_table.insertRow(r)

        combo = QComboBox()
        combo.addItems([label for _k, label, _t in SMART_SCALE_SETTINGS])
        idx = next((i for i, (k, _l, _t) in enumerate(SMART_SCALE_SETTINGS) if k == default_key), 0)
        combo.setCurrentIndex(idx)
        self.smart_param_table.setCellWidget(r, 0, combo)

        val_item = QTableWidgetItem()
        if default_key is not None:
            current = read_current_values(self.tab).get(default_key)
            if current is not None:
                fv = float(current)
                val_item.setText(str(int(fv)) if fv.is_integer() else str(fv))
        self.smart_param_table.setItem(r, 1, val_item)

    def _smart_remove_param(self):
        """Remove the selected parameter row from the table."""
        r = self.smart_param_table.currentRow()
        if r >= 0:
            self.smart_param_table.removeRow(r)

    def _smart_collect_profiles(self):
        """Read the editor (endpoint + param table) back into the selected profile.

        Called before any list mutation and on OK so in-progress edits are saved.
        Returns the sorted profile list.
        """
        row = self._smart_current_index()
        profiles = self._smart_sorted()
        if 0 <= row < len(profiles):
            prof = profiles[row]
            prof["endpoint"] = int(self.smart_endpoint_spin.value())
            settings = {}
            for r in range(self.smart_param_table.rowCount()):
                combo = self.smart_param_table.cellWidget(r, 0)
                val_item = self.smart_param_table.item(r, 1)
                if combo is None:
                    continue
                key = SMART_SCALE_KEYS[combo.currentIndex()]
                text = (val_item.text().strip() if val_item else "") or ""
                if not text:
                    continue
                try:
                    settings[key] = float(text)
                except ValueError:
                    continue
            prof["settings"] = settings
        self._smart_profiles = profiles
        return profiles

    def _smart_add_profile(self):
        """Add a new profile pre-filled with the current widget values."""
        self._smart_collect_profiles()
        current = read_current_values(self.tab)
        endpoints = [int(p.get("endpoint", 0)) for p in self._smart_profiles]
        new_endpoint = (max(endpoints) + 1000) if endpoints else 1000
        # Start with a couple of common params so the user sees the pattern.
        starter = {k: current[k] for k in ("node_size", "transparency") if k in current}
        self._smart_profiles.append({"endpoint": new_endpoint, "settings": starter})
        self._smart_refresh_profile_list()
        # Select the newly added profile (it's sorted by endpoint).
        target = next(i for i, p in enumerate(self._smart_sorted())
                      if int(p.get("endpoint", 0)) == new_endpoint)
        self.smart_profile_list.setCurrentRow(target)

    def _smart_duplicate_profile(self):
        """Duplicate the selected profile (same params, endpoint nudged up)."""
        row = self._smart_current_index()
        profiles = self._smart_sorted()
        if not (0 <= row < len(profiles)):
            return
        src = profiles[row]
        import copy as _copy
        dup = _copy.deepcopy(src)
        dup["endpoint"] = int(dup.get("endpoint", 0)) + 1
        self._smart_profiles.append(dup)
        self._smart_refresh_profile_list()
        # Select the duplicate (highest endpoint now).
        self.smart_profile_list.setCurrentRow(self.smart_profile_list.count() - 1)

    def _smart_remove_profile(self):
        """Remove the selected profile."""
        row = self._smart_current_index()
        profiles = self._smart_sorted()
        if not (0 <= row < len(profiles)):
            return
        del profiles[row]
        self._smart_profiles = profiles
        self._smart_refresh_profile_list()

    def _smart_move_profile(self, delta):
        """Move the selected profile up (-1) or down (+1)."""
        row = self._smart_current_index()
        profiles = self._smart_sorted()
        if not (0 <= row < len(profiles)):
            return
        target = row + delta
        if not (0 <= target < len(profiles)):
            return
        profiles[row], profiles[target] = profiles[target], profiles[row]
        self._smart_profiles = profiles
        self._smart_refresh_profile_list()
        self.smart_profile_list.setCurrentRow(target)

    def _update_tag_query_max_label(self, *_args):
        """Keep the 'Max tags shown' hint in sync with columns x rows."""
        cols = self.tag_query_columns_spin.value()
        rows = self.tag_query_rows_spin.value()
        self.tag_query_max_label.setText(f"{cols} × {rows} = {cols * rows} tags")

    def _read_ui_scale(self):
        """Read the UI scale (percent) from the editable combo, clamped to 25–250.

        A typed value has no matching item userData, so fall back to parsing
        the line-edit text; unparseable input falls back to 100%.
        """
        data = self.ui_scale_combo.currentData()
        if isinstance(data, int):
            return max(25, min(250, data))
        text = self.ui_scale_combo.currentText().strip()
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            val = int(float(text))
        except ValueError:
            return 100
        return max(25, min(250, val))

    def apply_settings(self):
        """Apply dialog values back to the tab and save settings."""
        tab = self.tab

        # Chunk Size (edited in the Clients group; plain int on the tab)
        try:
            tab.chunk_size = int(self.chunk_size_spin.value())
        except Exception:
            pass

        tab.low_memory = self.low_memory_checkbox.isChecked()
        tab.n_jobs = self.n_jobs_spin.value()
        tab.use_direct_db = self.direct_db_checkbox.isChecked()
        tab.tokenize = self.tokenize_checkbox.isChecked()
        tab.drop_universal = self.drop_universal_checkbox.isChecked()
        tab.drop_empty_files = self.drop_empty_files_checkbox.isChecked()
        tab.right_click_select_cohort = self.right_click_select_cohort_checkbox.isChecked()
        tab.auto_center_on_selection = self.auto_center_on_selection_checkbox.isChecked()
        tab.smooth_center_transition = self.smooth_center_transition_checkbox.isChecked()
        tab.smooth_center_speed = float(self.smooth_center_speed_spin.value())
        # WASD navigation paths: apply immediately (hide any showing if turned off).
        tab.wasd_paths_enabled = self.wasd_paths_checkbox.isChecked()
        if not tab.wasd_paths_enabled and getattr(tab, '_wasd_items', []):
            tab._wasd_mode = False
            tab._wasd_clear_paths(fade=False)

        # Auto-split parameters (master enable + threshold + max cycles)
        tab.auto_split_enabled = self.auto_split_enabled_checkbox.isChecked()
        tab.auto_split_threshold = self.auto_split_threshold_spin.value()
        tab.auto_split_max_cycles = self.auto_split_max_cycles_spin.value()

        # Session auto-save delay (seconds; 0 = save immediately)
        tab.session_save_delay = self.session_save_delay_spin.value()

        # Tag query grid layout (columns x rows; max tags shown = cols * rows).
        # Re-render the live selection so the change is visible immediately.
        old_cols = int(getattr(tab, 'tag_query_columns', 3))
        old_rows = int(getattr(tab, 'tag_query_rows', 14))
        tab.tag_query_columns = self.tag_query_columns_spin.value()
        tab.tag_query_rows = self.tag_query_rows_spin.value()
        if (old_cols, old_rows) != (tab.tag_query_columns, tab.tag_query_rows):
            try:
                tab._rebuild_tag_query_grid()
            except Exception as e:
                print(f"Error rebuilding tag query grid: {e}")

        # Smart Scale profiles (node-count-based automatic settings). The master
        # enable toggle lives in the main window, not this dialog.
        if hasattr(self, 'smart_profile_list'):
            # Commit any in-progress editor edits, then hand over the profiles.
            tab.smart_scale_profiles = self._smart_collect_profiles()

        # Explore (helicopter orbit) parameters
        tab.explore_mode = self.explore_mode_combo.currentText()
        tab.explore_show_path = self.explore_show_path_checkbox.isChecked()
        tab.explore_accel = float(self.explore_accel_spin.value())
        tab.explore_decel = float(self.explore_decel_spin.value())
        tab.explore_orbit_radius_base = float(self.explore_orbit_radius_base_spin.value())
        tab.explore_orbit_size_factor = float(self.explore_orbit_size_factor_spin.value())
        tab.explore_orbit_speed = float(self.explore_orbit_speed_spin.value())
        tab.explore_cycles = int(self.explore_cycles_spin.value())
        tab.explore_max_orbit_time = float(self.explore_max_orbit_time_spin.value())
        tab.explore_elevation = float(self.explore_elevation_spin.value())

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

        # Auto-Deorphan behavior
        tab.auto_deorphan = self.auto_deorphan_combo.currentText()

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

        # UI scale (applied at startup; restart required to take effect).
        # The combo is editable, so a typed value has no userData — parse text.
        new_scale = self._read_ui_scale()
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
