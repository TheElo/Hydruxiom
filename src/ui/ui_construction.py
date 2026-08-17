"""UI construction for the 3D tag map tab.

Mixin: methods operate on the ``TagMap3DTab`` instance (``self``). Moved here
from ``tag_map_3d_tab.py`` to reduce its size without changing behavior. The
top-level imports below mirror those of the original module so every global
name referenced by these methods resolves correctly.
"""
import json
import os
import tempfile

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QComboBox, QProgressBar, QGroupBox, QFormLayout,
    QTextEdit, QSplitter, QScrollArea, QLineEdit, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout, QFileDialog, QTabWidget
)
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QCloseEvent, QMouseEvent, QVector3D, QFont
import pyqtgraph.opengl as gl

from src.ui.styles import (
    GRAY_40, GRAY_33, RED_A, BLUE_60,
    TAB_BACKGROUND, TAB_TEXT, TAB_SELECTED, TAB_BORDER,
)
from src.ui.media_viewer import TagMap3DSplitWindow, SplitWindowLoader, SingleFileLoader
from src.ui.workers import WorkerThread
from src.ui.panels.tag_query import (
    ClickableTag, split_query_preserving_brackets,
    query_to_api_tags, parse_query_tag_states,
)
from src.ui.tag_map_utils import compile_tag_patterns, ease_in_out, SETTINGS_FILE
from src.ui.gl_text_items import get_multiline_text_item_class as _get_multiline_text_item_class


class UIConstructionMixin:
    def setup_ui(self):
        """Create the user interface."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Create main horizontal splitter: left panel | center (3D + info) | right sidebar
        main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = main_splitter  # kept for geometry persistence

        # Left panel - Controls (toggleable)
        self.left_sidebar = self.create_control_panel()
        main_splitter.addWidget(self.left_sidebar)

        # Center panel - 3D View with toggle button overlay
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # 3D View Widget (PyQtGraph) - wrapped in a container for supersample overlay
        self.view_3d = self.create_3d_view()
        self.view_container = QWidget()
        self.view_container_layout = QVBoxLayout(self.view_container)
        self.view_container_layout.setContentsMargins(0, 0, 0, 0)
        self.view_container_layout.addWidget(self.view_3d, stretch=1)
        # Supersample overlay label (hidden by default)
        self.supersample_label = QLabel()
        self.supersample_label.setAlignment(Qt.AlignCenter)
        self.supersample_label.setStyleSheet("background-color: black;")
        self.supersample_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.supersample_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.supersample_label.hide()
        self.view_container_layout.addWidget(self.supersample_label, 0, Qt.AlignCenter)
        center_layout.addWidget(self.view_container, stretch=1)

        main_splitter.addWidget(center_panel)

        # Right sidebar - Actions panel (toggleable)
        self.right_sidebar = self.create_right_sidebar()
        main_splitter.addWidget(self.right_sidebar)

        # Reorganize: right sidebar into tabs, move wobble/status/tag-grid.
        self._reorganize_sidebars()

        main_splitter.setStretchFactor(0, 0)  # Left panel - fixed
        main_splitter.setStretchFactor(1, 1)  # Center - stretches
        main_splitter.setStretchFactor(2, 0)  # Right sidebar - fixed

        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

        # Store visibility state for both sidebars
        self.left_sidebar_visible = True
        self.right_sidebar_visible = True

        # Create floating toggle button for right sidebar (positioned on right edge)
        self.sidebar_toggle_btn = QPushButton(">")
        self.sidebar_toggle_btn.setObjectName("sidebar_toggle")
        self.sidebar_toggle_btn.setStyleSheet("""
            #sidebar_toggle {
                background-color: rgba(40, 40, 40, 200);
                color: #ff6b6b;
                border: 1px solid #4050a0;
                border-right: none;
                border-top-left-radius: 5px;
                border-bottom-left-radius: 5px;
                padding: 15px 5px;
                font-size: 14px;
                font-weight: bold;
            }
            #sidebar_toggle:hover {
                background-color: rgba(64, 80, 160, 255);
            }
        """)
        self.sidebar_toggle_btn.setFixedSize(25, 50)
        self.sidebar_toggle_btn.setToolTip("Toggle right sidebar")
        self.sidebar_toggle_btn.clicked.connect(self.toggle_sidebar)
        self.sidebar_toggle_btn.setParent(self)
        self.sidebar_toggle_btn.setVisible(True)

        # Create floating toggle button for left sidebar (positioned on left edge)
        self.left_toggle_btn = QPushButton("<")
        self.left_toggle_btn.setObjectName("left_toggle")
        self.left_toggle_btn.setStyleSheet("""
            #left_toggle {
                background-color: rgba(40, 40, 40, 200);
                color: #ff6b6b;
                border: 1px solid #4050a0;
                border-left: none;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
                padding: 15px 5px;
                font-size: 14px;
                font-weight: bold;
            }
            #left_toggle:hover {
                background-color: rgba(64, 80, 160, 255);
            }
        """)
        self.left_toggle_btn.setFixedSize(25, 50)
        self.left_toggle_btn.setToolTip("Toggle left sidebar")
        self.left_toggle_btn.clicked.connect(self.toggle_left_sidebar)
        self.left_toggle_btn.setParent(self)
        self.left_toggle_btn.setVisible(True)

    def create_control_panel(self):
        """Create the control panel widget."""
        panel = QWidget()
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(350)
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Title
        title = QLabel("3D Tag Space Visualization")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {RED_A};")
        layout.addWidget(title)

        # Settings button (opens the advanced settings window)
        self.settings_button = QPushButton("Settings")
        self.settings_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {BLUE_60};
                color: {RED_A};
                padding: 8px;
                font-size: 12px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(80, 100, 200);
            }}
        """)
        self.settings_button.setToolTip("Open the 3D tag map settings window (low RAM, CPU cores, direct DB).")
        self.settings_button.clicked.connect(self.open_settings_dialog)
        layout.addWidget(self.settings_button)

        # Client Selection Group
        client_group = QGroupBox("Client Settings")
        client_group.setToolTip("Settings for connecting to your Hydrus/ManuelNightmare instance.")
        client_layout = QFormLayout()

        self.client_combo = QComboBox()
        from src.data.clients import client_ids
        self.client_combo.addItems(client_ids() or [])
        self.client_combo.setToolTip("Select which Hydrus client to connect to.\nClients are defined in clients.json at the project root.")
        self.client_combo.currentTextChanged.connect(self._load_client_db_path)
        self.client_combo.currentTextChanged.connect(self._populate_tag_services)
        client_layout.addRow("Client:", self.client_combo)

        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(50, 100000000)
        self.chunk_size_spin.setValue(8192)
        self.chunk_size_spin.setToolTip("Number of files to fetch per API request.\nLarger = faster but may timeout.\nSmaller = more requests but more reliable.\nDefault: 8192")
        client_layout.addRow("Chunk Size:", self.chunk_size_spin)

        self.max_files_spin = QSpinBox()
        self.max_files_spin.setRange(1, 100000000)
        self.max_files_spin.setValue(20000)
        self.max_files_spin.setSingleStep(512)
        self.max_files_spin.setToolTip("Maximum number of files to load and analyze.\nLower = faster processing, less memory.\nHigher = more complete visualization.\nDefault: 20000")
        max_files_row = QHBoxLayout()
        max_files_row.addWidget(self.max_files_spin)
        max_files_hint = QLabel("(default)")
        max_files_hint.setStyleSheet("color: #888; font-size: 10px;")
        max_files_row.addWidget(max_files_hint)
        max_files_row.addStretch()
        client_layout.addRow("Max Files:", max_files_row)

        self.tag_service_combo = QComboBox()
        self.tag_service_combo.setToolTip("Which tag service to use for fetching tags.\nPopulated dynamically per client.")
        client_layout.addRow("Tag Service:", self.tag_service_combo)

        # Auto-load session checkbox (loads last generated session on startup)
        self.auto_load_checkbox = QCheckBox("Auto load session")
        self.auto_load_checkbox.setChecked(True)
        self.auto_load_checkbox.setToolTip("When enabled, automatically load the last generated\nsession on next launch (skips query, UMAP, DBSCAN).\nA session is saved automatically after every successful load.")
        client_layout.addRow(self.auto_load_checkbox)

        client_group.setLayout(client_layout)
        layout.addWidget(client_group)

        # Algorithm Group
        algo_group = QGroupBox("Algorithm Settings")
        algo_group.setToolTip("Dimensionality reduction algorithm for mapping tags to 3D space.")
        algo_layout = QFormLayout()

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["UMAP", "GPU UMAP", "PCA"])
        self.algorithm_combo.setToolTip("Algorithm for reducing tag space to 3D.\nUMAP = CPU UMAP, preserves local structure and clusters (recommended, slower).\nGPU UMAP = GPU-accelerated via cuvs (RAPIDS), much faster at scale; falls back to CPU UMAP if unavailable.\nPCA = Linear reduction, faster but less detailed clustering.")
        algo_layout.addRow("Algorithm:", self.algorithm_combo)

        self.n_neighbors_spin = QSpinBox()
        self.n_neighbors_spin.setRange(5, 50)
        self.n_neighbors_spin.setValue(15)
        self.n_neighbors_spin.setToolTip("UMAP parameter: Number of neighboring points to consider.\nLower (5-10) = tighter, more clusters.\nHigher (20-50) = smoother, fewer clusters.\nDefault: 15")
        algo_layout.addRow("N Neighbors:", self.n_neighbors_spin)

        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(0, 100)
        self.min_dist_spin.setValue(10)
        self.min_dist_spin.setSuffix("%")
        self.min_dist_spin.setToolTip("UMAP parameter: Minimum distance between points (0-100%).\nLower (0-20%) = points packed tightly together.\nHigher (50-100%) = more spread out, easier to see individual points.\nDefault: 10%")
        algo_layout.addRow("Min Dist:", self.min_dist_spin)

        self.n_epochs_spin = QSpinBox()
        self.n_epochs_spin.setRange(0, 5000)
        self.n_epochs_spin.setValue(0)
        self.n_epochs_spin.setToolTip("UMAP parameter: Number of training epochs (0 = auto).\nAuto typically gives 500-1000 epochs based on data size.\nHigher = more accurate but slower.\nDefault: 0 (auto)")
        algo_layout.addRow("Epochs:", self.n_epochs_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.01, 10.0)
        self.learning_rate_spin.setValue(1.0)
        self.learning_rate_spin.setDecimals(2)
        self.learning_rate_spin.setSingleStep(0.1)
        self.learning_rate_spin.setToolTip("UMAP parameter: Initial learning rate for optimization.\nHigher = faster convergence but may overshoot.\nLower = slower but more stable.\nDefault: 1.0")
        algo_layout.addRow("Learning Rate:", self.learning_rate_spin)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["cosine", "euclidean"])
        self.metric_combo.setToolTip("UMAP distance metric.\ncosine = Angle-based similarity (default, good for TF-IDF).\neuclidean = Straight-line distance (faster, different clustering).")
        algo_layout.addRow("Metric:", self.metric_combo)

        # UMAP Subsampling (for large datasets)
        self.subsample_checkbox = QCheckBox("Subsample")
        self.subsample_checkbox.setChecked(False)
        self.subsample_checkbox.setToolTip("Fit UMAP on a random subset, then transform all points.\n"
                                           "Essential for 1M+ files to avoid memory allocation failures.\n"
                                           "The subset size controls accuracy vs. speed tradeoff.")
        algo_layout.addRow("Subsample:", self.subsample_checkbox)

        self.subsample_size_spin = QSpinBox()
        self.subsample_size_spin.setRange(10000, 1000000)
        self.subsample_size_spin.setValue(70000)
        self.subsample_size_spin.setSingleStep(10000)
        self.subsample_size_spin.setToolTip("Number of samples to use for UMAP fitting.\n"
                                            "All points are still transformed into 3D space.\n"
                                            "Higher = more accurate layout, more memory.\n"
                                            "Default: 70000")
        algo_layout.addRow("Subset Size:", self.subsample_size_spin)

        algo_group.setLayout(algo_layout)

        # Cluster Settings Group
        cluster_group = QGroupBox("Cluster Settings")
        cluster_group.setToolTip("DBSCAN clustering parameters for grouping similar files.")
        cluster_layout = QFormLayout()

        self.eps_spin = QSpinBox()
        self.eps_spin.setRange(1, 200)
        self.eps_spin.setValue(50)
        self.eps_spin.setSingleStep(1)
        self.eps_spin.setToolTip("DBSCAN parameter: Maximum distance between points in same cluster (as % of data spread).\nLower (10-30%) = many small, tight clusters.\nHigher (100-200%) = few large clusters.\nDefault: 50%")
        cluster_layout.addRow("EPS (%):", self.eps_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(2, 100)
        self.min_samples_spin.setValue(10)
        self.min_samples_spin.setToolTip("DBSCAN parameter: Minimum points to form a cluster.\nLower (2-5) = more clusters, even small groups count.\nHigher (20-50) = only large dense groups become clusters.\nDefault: 10")
        cluster_layout.addRow("Min Samples:", self.min_samples_spin)

        # Normalize positions before DBSCAN toggle (global + split clustering)
        self.normalize_checkbox = QCheckBox("Normalize positions before DBSCAN")
        self.normalize_checkbox.setChecked(getattr(self, 'normalize_positions', True))
        self.normalize_checkbox.setToolTip(
            "When enabled, positions are normalized (centered + std-scaled) before\n"
            "DBSCAN clustering. Makes eps a relative measure of local density rather\n"
            "than an absolute coordinate distance, so parameters behave consistently\n"
            "across datasets with different file counts / reducer scales.\n"
            "Applies to global clustering, re-cluster, and sub-cohort splitting."
        )
        self.normalize_checkbox.stateChanged.connect(self._on_normalize_toggled)
        cluster_layout.addRow(self.normalize_checkbox)

        # Sub-clustering settings (used for re-cluster selection splitting).
        # These are SEPARATE from the global eps/min_samples because sub-cohorts
        # within a selected cohort need finer parameters that differ not just by
        # a factor but independently.
        self.sub_eps_spin = QSpinBox()
        self.sub_eps_spin.setRange(1, 200)
        self.sub_eps_spin.setValue(20)
        self.sub_eps_spin.setSingleStep(5)
        self.sub_eps_spin.setToolTip("Sub-cluster EPS (%): DBSCAN distance for splitting a selected cohort into sub-cohorts.\nLower = finer/smaller sub-cohorts. Independent from global EPS.")
        cluster_layout.addRow("Sub EPS (%):", self.sub_eps_spin)

        self.sub_min_samples_spin = QSpinBox()
        self.sub_min_samples_spin.setRange(2, 100)
        self.sub_min_samples_spin.setValue(4)
        self.sub_min_samples_spin.setToolTip("Sub-cluster Min Samples: minimum points for a sub-cohort.\nIndependent from global Min Samples.")
        cluster_layout.addRow("Sub Min Samples:", self.sub_min_samples_spin)

        cluster_group.setLayout(cluster_layout)

        # Filter Settings Group (the tag-query grid is appended at the end)
        filter_group = QGroupBox("Filter Settings")
        filter_group.setToolTip("Filter which files are included in the visualization.")
        filter_layout = QFormLayout()

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("tag1, tag2, -tag3")
        self.query_edit.setToolTip("Hydrus search query to select which files to load.\nComma-separated tags (AND logic - files must have ALL tags).\nUse -tag to exclude files with that tag.\nExample: 'system:archive, playlist:favs, -rating:pending'")
        filter_layout.addRow("Query:", self.query_edit)

        self.whitelist_edit = QLineEdit()
        self.whitelist_edit.setPlaceholderText("character:*, creator:*")
        self.whitelist_edit.setToolTip("Tag-level filter: files keep ONLY tags matching this list.\nAll other tags are removed before positioning.\nUse to limit the attribute pool (e.g. only character/artist tags).\nComma-separated. Supports * wildcard.\nLeave empty to keep all tags.")
        filter_layout.addRow("Tag Whitelist:", self.whitelist_edit)

        self.blacklist_edit = QLineEdit()
        self.blacklist_edit.setPlaceholderText("rating, duration, source:*")
        self.blacklist_edit.setToolTip("Tag-level filter: removes matching tags from all files\nbefore position calculation and visualization.\nComma-separated. Supports * wildcard.\nLeave empty to remove nothing.")
        filter_layout.addRow("Tag Blacklist:", self.blacklist_edit)

        # "Drop empty files" moved to the Settings window (Performance group),
        # backed by the self.drop_empty_files attribute.

        self.min_doc_freq_spin = QSpinBox()
        self.min_doc_freq_spin.setRange(1, 100)
        self.min_doc_freq_spin.setValue(3)
        self.min_doc_freq_spin.setToolTip("Vectorizer: Minimum documents a tag must appear in\nto be included in the vocabulary.\nHigher = fewer rare tags, faster UMAP.\nLower = more tags, slower but more detailed.\nDefault: 3")
        filter_layout.addRow("Min Doc Freq:", self.min_doc_freq_spin)

        # Tag query builder is reparented here from the right sidebar after both
        # panels are built (see setup_ui). Store the layout for that step.
        self._filter_layout = filter_layout

        filter_group.setLayout(filter_layout)
        # Order: Client -> Filter -> Algorithm -> Cluster (Filter sits below Client).
        layout.addWidget(filter_group)
        layout.addWidget(algo_group)
        layout.addWidget(cluster_group)

        # Camera Wobble Group (for depth perception) — moved to right sidebar "Visuals" tab
        wobble_group = QGroupBox("Camera Wobble (Depth Effect)")
        wobble_group.setToolTip("Continuous camera movement to create depth perception through parallax.")
        wobble_layout = QFormLayout()

        self.wobble_enabled_checkbox = QCheckBox()
        self.wobble_enabled_checkbox.setChecked(False)
        self.wobble_enabled_checkbox.setToolTip("Enable/disable continuous camera wobble movement.")
        self.wobble_enabled_checkbox.stateChanged.connect(self._on_wobble_toggle)
        wobble_layout.addRow("Enable:", self.wobble_enabled_checkbox)

        self.wobble_continuous_checkbox = QCheckBox()
        self.wobble_continuous_checkbox.setChecked(False)
        self.wobble_continuous_checkbox.setToolTip(
            "Spin the camera continuously (azimuth rotates in one direction and "
            "wraps around, elevation bobs gently) instead of oscillating back and forth.\n"
            "The Azim/Elev Range values become the spin speed (degrees per second).\n"
            "Default: OFF (classic sine wobble)."
        )
        self.wobble_continuous_checkbox.stateChanged.connect(self._on_wobble_continuous_toggle)
        wobble_layout.addRow("Continuous Spin:", self.wobble_continuous_checkbox)

        self.wobble_speed_spin = QDoubleSpinBox()
        self.wobble_speed_spin.setRange(0.1, 10.0)
        self.wobble_speed_spin.setValue(1.0)
        self.wobble_speed_spin.setDecimals(2)
        self.wobble_speed_spin.setSingleStep(0.1)
        self.wobble_speed_spin.setToolTip("Speed of the wobble oscillation.\nHigher = faster movement.\nDefault: 1.0")
        wobble_layout.addRow("Speed:", self.wobble_speed_spin)

        # Azimuth range
        self.wobble_azim_range_spin = QDoubleSpinBox()
        self.wobble_azim_range_spin.setRange(0.0, 180.0)
        self.wobble_azim_range_spin.setValue(15.0)
        self.wobble_azim_range_spin.setDecimals(2)
        self.wobble_azim_range_spin.setSingleStep(0.5)
        self.wobble_azim_range_spin.setToolTip("Azimuth rotation range (horizontal rotation).\n0 = off.\nDefault: 15.0 degrees")
        wobble_layout.addRow("Azim Range:", self.wobble_azim_range_spin)

        # Elevation range
        self.wobble_elev_range_spin = QDoubleSpinBox()
        self.wobble_elev_range_spin.setRange(0.0, 90.0)
        self.wobble_elev_range_spin.setValue(10.0)
        self.wobble_elev_range_spin.setDecimals(2)
        self.wobble_elev_range_spin.setSingleStep(0.5)
        self.wobble_elev_range_spin.setToolTip("Elevation rotation range (vertical rotation).\n0 = off.\nDefault: 10.0 degrees")
        wobble_layout.addRow("Elev Range:", self.wobble_elev_range_spin)

        wobble_group.setLayout(wobble_layout)
        # Stored (not added to left panel) — placed in the right sidebar "Visuals" tab.
        self.wobble_group = wobble_group

        # Initialize wobble timer and state
        self.wobble_timer = QTimer()
        self.wobble_timer.timeout.connect(self._update_wobble)
        self.wobble_timer.setInterval(16)  # ~60fps
        self.wobble_time = 0.0
        self.wobble_base_azim = 45.0   # base azimuth (captured on enable)
        self.wobble_base_elev = 30.0   # base elevation (captured on enable)
        self.wobble_spin_azim = 0.0    # continuous-spin azimuth accumulator
        self.wobble_user_interacting = False  # True while user drags (pauses wobble)

        # Load Button and Progress
        self.load_button = QPushButton("Load & Compute")
        self.load_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {BLUE_60};
                color: {RED_A};
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(80, 100, 200);
            }}
            QPushButton:disabled {{
                background-color: {GRAY_40};
            }}
        """)
        self.load_button.setToolTip("Load files from Hydrus and compute the full 3D map\n(query -> tags -> UMAP/PCA -> DBSCAN).\nShortcut: F5")
        self.load_button.clicked.connect(self.start_loading)
        layout.addWidget(self.load_button)

        self.recompute_button = QPushButton("Recompute")
        self.recompute_button.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb(60, 80, 160);
                color: {RED_A};
                padding: 8px;
                font-size: 12px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(80, 100, 200);
            }}
            QPushButton:disabled {{
                background-color: {GRAY_40};
            }}
        """)
        self.recompute_button.setEnabled(False)
        self.recompute_button.clicked.connect(self.start_recompute)
        self.recompute_button.setToolTip("Re-run UMAP/PCA only with new settings using currently loaded data.\nClustering is separate (use Regroup).\nShortcut: F6")
        layout.addWidget(self.recompute_button)

        # Regroup + Optimize row: Regroup takes ~3/4 width, Optimize is a small icon (~1/4)
        regroup_row = QHBoxLayout()
        regroup_row.setSpacing(4)

        self.recluster_button = QPushButton("Regroup")
        self.recluster_button.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb(60, 80, 160);
                color: {RED_A};
                padding: 8px;
                font-size: 12px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(80, 100, 200);
            }}
            QPushButton:disabled {{
                background-color: {GRAY_40};
            }}
        """)
        self.recluster_button.setEnabled(False)
        self.recluster_button.clicked.connect(self.start_recluster)
        self.recluster_button.setToolTip("Re-run DBSCAN only on current positions (no UMAP/PCA re-run). Use after changing eps/min_samples.\nShortcut: F7")
        regroup_row.addWidget(self.recluster_button, 3)

        # Optimize: small icon button next to Regroup
        self.optimize_button = QPushButton("⚙")
        self.optimize_button.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb(60, 80, 160);
                color: {RED_A};
                padding: 4px;
                font-size: 14px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(80, 100, 200);
            }}
            QPushButton:disabled {{
                background-color: {GRAY_40};
            }}
        """)
        self.optimize_button.setEnabled(False)
        self.optimize_button.clicked.connect(self.start_optimize)
        self.optimize_button.setToolTip(
            "Optimize DBSCAN: automatically search for the ideal eps/min_samples combination.\n"
            "Goal: reduce non-cohorted (noise) nodes and split disproportionately\n"
            "large cohorts. Runs DBSCAN multiple times to find the best settings."
        )
        regroup_row.addWidget(self.optimize_button, 1)

        layout.addLayout(regroup_row)

        # Clear button - drop the current session data and free resources
        self.clear_button = QPushButton("Clear")
        self.clear_button.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb(120, 50, 60);
                color: {RED_A};
                padding: 8px;
                font-size: 12px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(160, 70, 80);
            }}
            QPushButton:disabled {{
                background-color: {GRAY_40};
            }}
        """)
        self.clear_button.setToolTip("Drop the current session data (scene graph, node list,\nGPU buffers) and clear the view.\nFrees memory before starting a new load.\nShortcut: Ctrl+X")
        self.clear_button.clicked.connect(self.clear_session)
        layout.addWidget(self.clear_button)

        # Session buttons are now automatic (hidden). Objects kept for setEnabled compat.
        self.save_session_button = QPushButton()  # hidden, auto-saved after each load
        self.load_session_button = QPushButton()  # hidden, auto-loaded on startup

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Phase label - shows which pipeline stage is currently running
        self.phase_label = QLabel("Phase: Idle")
        self.phase_label.setStyleSheet(f"color: {BLUE_60}; font-size: 11px; font-weight: bold;")
        layout.addWidget(self.phase_label)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {RED_A}; font-size: 12px;")
        layout.addWidget(self.status_label)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_3d_view(self):
        """Create the 3D view widget with right-click support."""
        try:
            import pyqtgraph.opengl as gl

            # Create a custom GLViewWidget class that inherits from GLViewWidget
            class RightClickGLView(gl.GLViewWidget):
                """GLViewWidget with right-click node inspection and custom orbit speed."""
                
                def __init__(self, parent_tab, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.parent_tab = parent_tab
                    self.selection_start = None
                    self.is_selecting = False
                    # Enable keyboard focus for F11 fullscreen toggle
                    self.setFocusPolicy(Qt.StrongFocus)

                def mouseReleaseEvent(self, event: QMouseEvent):
                    """Resume wobble from the user's new camera position."""
                    if hasattr(self.parent_tab, 'wobble_user_interacting'):
                        self.parent_tab.wobble_user_interacting = False
                        # Re-base so wobble continues smoothly from where the user left it.
                        if hasattr(self.parent_tab, '_capture_wobble_base'):
                            self.parent_tab._capture_wobble_base()
                    super().mouseReleaseEvent(event)

                def evalKeyState(self):
                    """Override to use custom orbit speed from settings."""
                    from PySide6.QtCore import Qt
                    speed = self.parent_tab.orbit_speed_spin.value() if hasattr(self.parent_tab, 'orbit_speed_spin') else 2.0
                    if len(self.keysPressed) > 0:
                        for key in self.keysPressed:
                            if key == Qt.Key_Right:
                                self.orbit(azim=-speed, elev=0)
                            elif key == Qt.Key_Left:
                                self.orbit(azim=speed, elev=0)
                            elif key == Qt.Key_Up:
                                self.orbit(azim=0, elev=-speed)
                            elif key == Qt.Key_Down:
                                self.orbit(azim=0, elev=speed)
                            elif key == Qt.Key_PageUp:
                                pass
                            elif key == Qt.Key_PageDown:
                                pass
                            self.keyTimer.start(16)
                    else:
                        self.keyTimer.stop()
                
                def mousePressEvent(self, event: QMouseEvent):
                    """Handle mouse press events.

                    Shortcuts:
                    - Left click: select cohort under cursor
                    - Ctrl+Left click: select node under cursor
                    - Right click: move camera center to node under cursor
                    - Ctrl+Right click: move camera center to node under cursor
                    """
                    # Pause wobble while the user interacts so manual orbit works.
                    if hasattr(self.parent_tab, 'wobble_user_interacting'):
                        self.parent_tab.wobble_user_interacting = True

                    # Handle left-click for cohort selection
                    if event.button() == Qt.MouseButton.LeftButton:
                        # Ctrl+Left click selects a node
                        if event.modifiers() & Qt.ControlModifier:
                            self.handle_right_click(event)
                        else:
                            self.handle_cluster_selection(event)
                    # Handle right-click for camera movement
                    elif event.button() == Qt.MouseButton.RightButton:
                        self.handle_set_camera_center(event)
                    
                    # Call parent implementation for normal 3D navigation
                    super().mousePressEvent(event)
                
                def handle_right_click(self, event: QMouseEvent):
                    """Handle right-click by picking the node under cursor (vectorized)."""
                    try:
                        closest_idx = self.parent_tab.pick_nearest_point(self, event.pos())
                        if closest_idx is not None:
                            self.parent_tab.show_node_info(closest_idx)
                    except Exception as e:
                        print(f"Error in right-click handling: {e}")
                        import traceback
                        traceback.print_exc()
                
                def handle_cluster_selection(self, event: QMouseEvent):
                    """Handle left-click to select all nodes in the same cluster (vectorized)."""
                    try:
                        closest_idx = self.parent_tab.pick_nearest_point(self, event.pos())
                        if closest_idx is not None:
                            self.parent_tab.show_cluster_info(closest_idx)
                    except Exception as e:
                        print(f"Error in cluster selection: {e}")
                        import traceback
                        traceback.print_exc()
                
                def handle_set_camera_center(self, event: QMouseEvent):
                    """Handle right-click to set the camera center to the clicked node (vectorized).

                    If "Right-click also selects cohort" is enabled in settings, this
                    ALSO selects the cohort under the cursor so you can navigate +
                    inspect in one gesture.
                    """
                    try:
                        import numpy as np
                        from PySide6.QtGui import QVector3D
                        closest_idx = self.parent_tab.pick_nearest_point(self, event.pos())
                        if closest_idx is not None:
                            scatter = self.parent_tab.gl_scatter
                            positions = scatter.pos if isinstance(scatter.pos, np.ndarray) else np.array(scatter.pos)
                            center = positions[closest_idx]
                            self.opts['center'] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
                            self.update()
                            print(f"Camera center set to ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
                            # Optional: also select the cohort under the cursor.
                            if getattr(self.parent_tab, 'right_click_select_cohort', False):
                                self.parent_tab.show_cluster_info(closest_idx)
                    except Exception as e:
                        print(f"Error in set camera center: {e}")
                        import traceback
                        traceback.print_exc()
                
                def keyPressEvent(self, event):
                    """Handle key press events for fullscreen and sidebar toggles."""
                    from PySide6.QtGui import QKeyEvent
                    if event.key() == Qt.Key_F11:
                        # Toggle fullscreen on the parent's main window
                        main_window = self.parent_tab.main_window
                        if main_window.isFullScreen():
                            main_window.showNormal()
                            print("Exited fullscreen")
                        else:
                            main_window.showFullScreen()
                            print("Entered fullscreen")
                        event.accept()
                        return
                    elif event.key() == Qt.Key_F1:
                        # Toggle left sidebar
                        self.parent_tab.toggle_left_sidebar()
                        event.accept()
                        return
                    elif event.key() == Qt.Key_F2:
                        # Toggle right sidebar
                        self.parent_tab.toggle_sidebar()
                        event.accept()
                        return
                    elif event.key() == Qt.Key_R and event.modifiers() & Qt.ControlModifier:
                        # Pop selected cohort
                        self.parent_tab._pop_selected_cohort()
                        event.accept()
                        return
                    super().keyPressEvent(event)
            
            # Create the custom view widget
            view = RightClickGLView(self)
            view.setBackgroundColor((0, 0, 0, 255))  # Complete black
            view.setCameraPosition(distance=200, elevation=30, azimuth=45)
            
            # Add grid (hidden)
            grid = gl.GLGridItem()
            grid.setSize(x=200, y=200)
            grid.setSpacing(x=20, y=20)
            grid.setVisible(False)  # Hide grid
            view.addItem(grid)
            
            # Store reference for later
            self.gl_view = view
            self.gl_scatter = None
            
            return view
        except ImportError:
            # Fallback widget if PyQtGraph not installed
            fallback = QWidget()
            fallback_layout = QVBoxLayout(fallback)
            label = QLabel("PyQtGraph not installed.\nInstall with: pip install pyqtgraph")
            label.setStyleSheet(f"color: {RED_A}; font-size: 16px; padding: 50px;")
            label.setAlignment(Qt.AlignCenter)
            fallback_layout.addWidget(label)
            return fallback

    def create_info_panel(self):
        """Create the info panel for selected file details."""
        panel = QGroupBox("Selected File Info")
        panel.setStyleSheet(f"QGroupBox {{ color: {RED_A}; }}")
        layout = QVBoxLayout()

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        self.info_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {GRAY_40};
                color: {RED_A};
                border: 1px solid {BLUE_60};
                padding: 5px;
            }}
        """)
        self.info_text.setPlaceholderText("Left: cohort | Ctrl+Left: node | Right: move camera")
        layout.addWidget(self.info_text)

        panel.setLayout(layout)
        return panel

    def create_right_sidebar(self):
        """Create the right sidebar with actions panel."""
        sidebar = QWidget()
        sidebar.setMinimumWidth(200)
        sidebar.setMaximumWidth(350)
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(15)

        # Selected File Info (moved from bottom panel)
        info_group = QGroupBox("Selected File Info")
        info_group.setStyleSheet(f"QGroupBox {{ color: {RED_A}; }}")
        info_layout = QVBoxLayout()

        # Container widget for file info content
        self.info_container = QWidget()
        self.info_layout = QVBoxLayout(self.info_container)
        self.info_layout.setContentsMargins(5, 5, 5, 5)
        self.info_layout.setSpacing(5)
        info_layout.addWidget(self.info_container)

        # Text area for non-tag info (file id, cluster, score, position)
        self.info_text = QTextEdit()
        self.info_text.setMinimumHeight(60)
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(80)
        self.info_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {GRAY_40};
                color: {RED_A};
                border: 1px solid {BLUE_60};
                padding: 5px;
                font-size: 11px;
            }}
        """)
        self.info_text.setPlaceholderText("Left: cohort | Ctrl+Left: node | Right: move camera")
        self.info_layout.addWidget(self.info_text)

        # Clickable tags container - direct vertical list (no scroll area, more reliable)
        self.tag_container = QWidget()
        self.tag_container.setMinimumHeight(50)
        
        # Use a QGridLayout for tags so they wrap nicely
        self.tag_grid = QGridLayout()
        self.tag_grid.setContentsMargins(2, 2, 2, 2)
        self.tag_grid.setSpacing(3)
        self.tag_container.setLayout(self.tag_grid)
        
        self.info_layout.addWidget(self.tag_container)

        info_group.setLayout(info_layout)
        self._info_group = info_group  # reorganized into tabs in setup_ui
        layout.addWidget(info_group)

        # Selection Cohort Tags Section
        selection_tags_group = QGroupBox("Cohort Tag Data")
        selection_tags_group.setToolTip("Most common tags in the selected cohort (top 20).")
        selection_tags_layout = QVBoxLayout()
        selection_tags_layout.setSpacing(8)

        self.selection_tags_text = QTextEdit()
        self.selection_tags_text.setReadOnly(True)
        self.selection_tags_text.setMaximumHeight(200)
        self.selection_tags_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {GRAY_40};
                color: {RED_A};
                border: 1px solid {BLUE_60};
                padding: 5px;
                font-size: 11px;
                font-family: Consolas, monospace;
            }}
        """)
        self.selection_tags_text.setPlaceholderText("Top 20 tags for the selected cohort will appear here...")
        selection_tags_layout.addWidget(self.selection_tags_text)

        selection_tags_group.setLayout(selection_tags_layout)
        self._selection_tags_group = selection_tags_group  # reorganized into tabs in setup_ui
        layout.addWidget(selection_tags_group)

        # Send to Tab Section
        send_group = QGroupBox("Send to Tab")
        send_group.setToolTip("Send selected file(s) to a Hydrus tab.")
        send_layout = QVBoxLayout()
        send_layout.setSpacing(8)

        # Tab name input
        tab_label = QLabel("Tab Name:")
        tab_label.setStyleSheet(f"color: {RED_A};")
        send_layout.addWidget(tab_label)

        self.tab_name_edit = QLineEdit()
        self.tab_name_edit.setPlaceholderText("Enter Hydrus tab name...")
        self.tab_name_edit.setToolTip("Name of the Hydrus tab to send files to.")
        self.tab_name_edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: {GRAY_40};
                color: {RED_A};
                border: 1px solid {BLUE_60};
                padding: 5px;
                border-radius: 3px;
            }}
        """)
        send_layout.addWidget(self.tab_name_edit)

        # Send button
        self.send_to_tab_btn = QPushButton("Send to Tab")
        self.send_to_tab_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BLUE_60};
                color: {RED_A};
                padding: 10px;
                font-size: 13px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(80, 100, 200);
            }}
            QPushButton:disabled {{
                background-color: {GRAY_40};
            }}
        """)
        self.send_to_tab_btn.setToolTip("Send selected file(s) to the specified Hydrus tab.")
        self.send_to_tab_btn.clicked.connect(self.send_selected_to_tab)
        self.send_to_tab_btn.setEnabled(False)  # Disabled until nodes are loaded
        send_layout.addWidget(self.send_to_tab_btn)

        # Status label for send operations
        self.send_status_label = QLabel("")
        self.send_status_label.setStyleSheet(f"color: {RED_A}; font-size: 10px;")
        self.send_status_label.setWordWrap(True)
        send_layout.addWidget(self.send_status_label)

        # Explore button
        self.time_travel_button = QPushButton("Explore")
        self.time_travel_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {BLUE_60};
                color: {RED_A};
                padding: 8px;
                font-size: 12px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(80, 100, 200);
            }}
            QPushButton:disabled {{
                background-color: {GRAY_40};
            }}
        """)
        self.time_travel_button.setToolTip("Fly the camera through the largest cluster centroids to explore the map.")
        self.time_travel_button.clicked.connect(self._toggle_time_travel)
        self.time_travel_button.setEnabled(False)  # Enabled when data is loaded
        send_layout.addWidget(self.time_travel_button)

        # Cut out button - select the cohort nodes and remove everything else
        self.cut_button = QPushButton("Cut out")
        self.cut_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {BLUE_60};
                color: {RED_A};
                padding: 8px;
                font-size: 12px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(80, 100, 200);
            }}
            QPushButton:disabled {{
                background-color: {GRAY_40};
            }}
        """)
        self.cut_button.setToolTip("Cut out the selected cohort - keep only its nodes\nand remove everything else from the view.\nShortcut: Ctrl+E")
        self.cut_button.clicked.connect(self._cut_selected_cohort)
        self.cut_button.setEnabled(False)  # Enabled when a cohort is selected
        send_layout.addWidget(self.cut_button)

        # Pop button - remove the selected cohort from the view (inverse of Cut out)
        self.pop_button = QPushButton("Pop cohort")
        self.pop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {BLUE_60};
                color: {RED_A};
                padding: 8px;
                font-size: 12px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(80, 100, 200);
            }}
            QPushButton:disabled {{
                background-color: {GRAY_40};
            }}
        """)
        self.pop_button.setToolTip("Remove the selected cohort from the view\n(keep everything else). Positions of remaining nodes stay unchanged.\nShortcut: Ctrl+R")
        self.pop_button.clicked.connect(self._pop_selected_cohort)
        self.pop_button.setEnabled(False)  # Enabled when a cohort is selected
        send_layout.addWidget(self.pop_button)

        # Split group button - apply cluster algo on selection, keep positions
        self.cluster_button = QPushButton("Split group")
        self.cluster_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {BLUE_60};
                color: {RED_A};
                padding: 8px;
                font-size: 12px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: rgb(80, 100, 200);
            }}
            QPushButton:disabled {{
                background-color: {GRAY_40};
            }}
        """)
        self.cluster_button.setToolTip("Split the selected cohort into sub-groups using the cluster\nalgorithm on its existing positions.\nPositions stay unchanged; only coloring/labels change.\nShortcut: Ctrl+S")
        self.cluster_button.clicked.connect(self._recluster_selection)
        self.cluster_button.setEnabled(False)  # Enabled when a cohort is selected
        send_layout.addWidget(self.cluster_button)

        send_group.setLayout(send_layout)
        self._send_group = send_group  # reorganized into tabs in setup_ui
        layout.addWidget(send_group)

        # Visualization Settings Group (moved from left sidebar)
        vis_group = QGroupBox("Visualization Settings")
        vis_group.setToolTip("Control the appearance of nodes in the 3D view.")
        vis_layout = QFormLayout()

        self.min_size_spin = QDoubleSpinBox()
        self.min_size_spin.setRange(0.01, 0.5)
        self.min_size_spin.setValue(0.2)
        self.min_size_spin.setSingleStep(0.01)
        self.min_size_spin.setDecimals(2)
        self.min_size_spin.valueChanged.connect(self._on_size_changed)
        self.min_size_spin.setToolTip("Node size in the 3D view.\nDisplay value is x10 of the actual size.\nDefault: 0.2 (actual: 0.02)")
        vis_layout.addRow("Node Size:", self.min_size_spin)

        self.spread_spin = QDoubleSpinBox()
        self.spread_spin.setRange(0.1, 10.0)
        self.spread_spin.setValue(1.0)
        self.spread_spin.setSingleStep(0.01)
        self.spread_spin.setDecimals(2)
        self.spread_spin.valueChanged.connect(self._on_spread_changed)
        self.spread_spin.setToolTip("Scale factor for spreading nodes apart in 3D space.\n1.0 = original algorithm output.\nHigher = nodes spread further apart.\nUseful for seeing clusters more clearly.\nDefault: 1.0")
        vis_layout.addRow("Spread:", self.spread_spin)

        self.orbit_speed_spin = QDoubleSpinBox()
        self.orbit_speed_spin.setRange(0.1, 50.0)
        self.orbit_speed_spin.setValue(0.2)
        self.orbit_speed_spin.setDecimals(2)
        self.orbit_speed_spin.setSingleStep(0.1)
        self.orbit_speed_spin.setToolTip("Speed of camera orbit when using arrow keys.\nLower = slower, more precise movement.\nHigher = faster camera rotation.\nDefault: 0.2")
        vis_layout.addRow("Orbit Speed:", self.orbit_speed_spin)

        self.transparency_spin = QDoubleSpinBox()
        self.transparency_spin.setRange(0.0, 1.0)
        self.transparency_spin.setValue(0.8)
        self.transparency_spin.setDecimals(2)
        self.transparency_spin.setSingleStep(0.05)
        self.transparency_spin.valueChanged.connect(self._on_transparency_changed)
        self.transparency_spin.setToolTip("Alpha (transparency) of nodes in the 3D view.\n0.0 = fully transparent, 1.0 = fully opaque.\nDefault: 0.8")
        vis_layout.addRow("Transparency:", self.transparency_spin)

        # Dim non-selected nodes when a selection is active
        self.dim_non_selected_checkbox = QCheckBox("Dim Non-Selected on Selection")
        self.dim_non_selected_checkbox.setChecked(True)
        self.dim_non_selected_checkbox.stateChanged.connect(self._on_dim_toggle_changed)
        self.dim_non_selected_checkbox.setToolTip("When a cohort or node is selected, increase transparency\nof all non-selected nodes to make the selection stand out.\nDefault: ON.")
        vis_layout.addRow(self.dim_non_selected_checkbox)

        self.dim_alpha_spin = QDoubleSpinBox()
        self.dim_alpha_spin.setRange(0.0, 1.0)
        self.dim_alpha_spin.setValue(0.15)
        self.dim_alpha_spin.setDecimals(2)
        self.dim_alpha_spin.setSingleStep(0.05)
        self.dim_alpha_spin.valueChanged.connect(self._on_dim_alpha_changed)
        self.dim_alpha_spin.setToolTip("Alpha (transparency) applied to non-selected nodes\nwhen a selection is active.\n0.0 = fully transparent, 1.0 = fully opaque.\nDefault: 0.15")
        vis_layout.addRow("Dim Alpha:", self.dim_alpha_spin)

        # Highlight color for selected cluster/node
        self.highlight_color = (0.0, 1.0, 1.0)  # Default: cyan
        self.highlight_color_btn = QPushButton("Cyan")
        self.highlight_color_btn.setFixedWidth(80)
        self.highlight_color_btn.setStyleSheet(f"background-color: rgb(0, 255, 255); color: black; font-weight: bold;")
        self.highlight_color_btn.clicked.connect(self._pick_highlight_color)
        self.highlight_color_btn.setToolTip("Color used to highlight the selected cluster or node.\nClick to choose a custom color.")
        vis_layout.addRow("Highlight Color:", self.highlight_color_btn)

        # Star twinkle effect (random nodes blink in their own color)
        self.twinkle_checkbox = QCheckBox("Star Twinkle")
        self.twinkle_checkbox.setChecked(False)
        self.twinkle_checkbox.stateChanged.connect(self._on_twinkle_toggle)
        self.twinkle_checkbox.setToolTip("Animate a random set of nodes blinking in their own color\nlike stars twinkling. Dead nodes are periodically replaced with new random ones.")
        vis_layout.addRow(self.twinkle_checkbox)

        self.twinkle_count_spin = QSpinBox()
        self.twinkle_count_spin.setRange(10, 20000)
        self.twinkle_count_spin.setValue(2000)
        self.twinkle_count_spin.setSingleStep(50)
        self.twinkle_count_spin.valueChanged.connect(self._on_twinkle_param_changed)
        self.twinkle_count_spin.setToolTip("Number of nodes twinkling at any given time.\nDefault: 2000")
        vis_layout.addRow("Twinkle Count:", self.twinkle_count_spin)

        self.twinkle_lifespan_min_spin = QDoubleSpinBox()
        self.twinkle_lifespan_min_spin.setRange(0.1, 60.0)
        self.twinkle_lifespan_min_spin.setValue(1.0)
        self.twinkle_lifespan_min_spin.setDecimals(1)
        self.twinkle_lifespan_min_spin.valueChanged.connect(self._on_twinkle_param_changed)
        self.twinkle_lifespan_min_spin.setToolTip("Minimum lifespan (seconds) before a twinkling node is replaced.\nDefault: 1.0")
        vis_layout.addRow("Lifespan Min (s):", self.twinkle_lifespan_min_spin)

        self.twinkle_lifespan_max_spin = QDoubleSpinBox()
        self.twinkle_lifespan_max_spin.setRange(0.1, 120.0)
        self.twinkle_lifespan_max_spin.setValue(6.0)
        self.twinkle_lifespan_max_spin.setDecimals(1)
        self.twinkle_lifespan_max_spin.valueChanged.connect(self._on_twinkle_param_changed)
        self.twinkle_lifespan_max_spin.setToolTip("Maximum lifespan (seconds) before a twinkling node is replaced.\nEach node gets a random lifespan in [min, max].\nDefault: 6.0")
        vis_layout.addRow("Lifespan Max (s):", self.twinkle_lifespan_max_spin)

        self.twinkle_freq_spin = QDoubleSpinBox()
        self.twinkle_freq_spin.setRange(0.1, 20.0)
        self.twinkle_freq_spin.setValue(2.0)
        self.twinkle_freq_spin.setDecimals(1)
        self.twinkle_freq_spin.valueChanged.connect(self._on_twinkle_param_changed)
        self.twinkle_freq_spin.setToolTip("Blink frequency (Hz): how fast each node pulses.\nDefault: 2.0")
        vis_layout.addRow("Blink Freq (Hz):", self.twinkle_freq_spin)

        self.twinkle_brightness_spin = QDoubleSpinBox()
        self.twinkle_brightness_spin.setRange(0.1, 3.0)
        self.twinkle_brightness_spin.setValue(1.5)
        self.twinkle_brightness_spin.setDecimals(2)
        self.twinkle_brightness_spin.valueChanged.connect(self._on_twinkle_param_changed)
        self.twinkle_brightness_spin.setToolTip("Peak brightness multiplier when a node is at full blink.\n1.0 = no extra brightness, 3.0 = very bright.\nDefault: 1.5")
        vis_layout.addRow("Brightness:", self.twinkle_brightness_spin)

        # V5: Color Scheme dropdown
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["Pastel", "Viridis", "Plasma", "Inferno", "Coolwarm"])
        self.color_scheme_combo.setCurrentText("Pastel")
        self.color_scheme_combo.currentTextChanged.connect(self._on_color_scheme_changed)
        self.color_scheme_combo.setToolTip("Color scheme for cluster nodes.\nPastel = Default soft colors.\nViridis/Plasma/Inferno/Coolwarm = Matplotlib colormaps.")
        vis_layout.addRow("Color Scheme:", self.color_scheme_combo)

        # Anti-noise / quality settings
        self.supersample_checkbox = QCheckBox("4x Snapshot")
        self.supersample_checkbox.setChecked(False)
        self.supersample_checkbox.setToolTip("Render a static 4x-resolution snapshot of the view (reduces aliasing/noise).\nWhile enabled, the image is frozen and you cannot orbit or select.\nDisable to return to normal interactive viewing.\nDefault: OFF.")
        self.supersample_checkbox.stateChanged.connect(self._on_supersample_toggle)
        vis_layout.addRow(self.supersample_checkbox)

        self.supersample_fps_spin = QSpinBox()
        self.supersample_fps_spin.setRange(0, 60)
        self.supersample_fps_spin.setValue(10)
        self.supersample_fps_spin.setToolTip("FPS for the 4x supersample render.\n0 = disable the FPS limiter (render as fast as possible).\nHigher = smoother but heavier (more GPU/CPU).\nDefault: 10")
        self.supersample_fps_spin.valueChanged.connect(self._on_supersample_fps_changed)
        # Hidden: not exposed in the UI, but kept for settings persistence.

        vis_group.setLayout(vis_layout)
        self._vis_group = vis_group  # reorganized into tabs in setup_ui
        layout.addWidget(vis_group)

        # Cohort Selections Section
        cohort_group = QGroupBox("Cohort Selections")
        cohort_group.setToolTip("Top cohorts by file count with dominant tags.")
        cohort_layout = QVBoxLayout()
        cohort_layout.setSpacing(8)

        # Threshold slider/label row
        threshold_row = QHBoxLayout()
        threshold_label = QLabel("Tag Threshold:")
        threshold_label.setStyleSheet(f"color: {RED_A};")
        threshold_row.addWidget(threshold_label)

        self.cohort_threshold_spin = QDoubleSpinBox()
        self.cohort_threshold_spin.setRange(0.0, 1.0)
        self.cohort_threshold_spin.setValue(0.9)
        self.cohort_threshold_spin.setDecimals(2)
        self.cohort_threshold_spin.setSingleStep(0.05)
        self.cohort_threshold_spin.setToolTip("Minimum percentage of files in cohort that must have a tag for it to be listed.")
        self.cohort_threshold_spin.valueChanged.connect(self._on_cohort_threshold_changed)
        threshold_row.addWidget(self.cohort_threshold_spin)
        threshold_row.addStretch()
        cohort_layout.addLayout(threshold_row)

        # Cohort label toggle
        self.show_cohort_labels_checkbox = QCheckBox("Show Cohort Labels")
        self.show_cohort_labels_checkbox.setChecked(False)
        self.show_cohort_labels_checkbox.setToolTip("Display dominant-tag labels centered on each cohort in the 3D view.")
        self.show_cohort_labels_checkbox.stateChanged.connect(self._on_show_cohort_labels_toggled)
        cohort_layout.addWidget(self.show_cohort_labels_checkbox)

        # Label mode dropdown (controls how many labels are shown to avoid noise)
        label_mode_row = QHBoxLayout()
        label_mode_label = QLabel("Label Mode:")
        label_mode_label.setStyleSheet(f"color: {RED_A};")
        label_mode_row.addWidget(label_mode_label)
        self.cohort_label_mode_combo = QComboBox()
        # Ordered by performance impact (lightest first)
        self.cohort_label_mode_combo.addItems([
            "Selected Only",
            "Selected & N neighbors",
            "Top N largest",
            "Above size threshold",
            "All cohorts",
        ])
        self.cohort_label_mode_combo.setToolTip(
            "Controls which cohort labels are shown to avoid overlap noise.\n"
            "Selected Only = only the currently selected cohort's label.\n"
            "Selected & N neighbors = selected cohort + its N nearest neighbors (by centroid distance).\n"
            "Top N largest = only the N biggest cohorts (biggest to smallest).\n"
            "Above size threshold = only cohorts with at least N files.\n"
            "All cohorts = show every cohort label (can be noisy)."
        )
        self.cohort_label_mode_combo.currentTextChanged.connect(self._on_cohort_label_mode_changed)
        # Default to "Selected & N neighbors" (index 1) on a fresh launch.
        self.cohort_label_mode_combo.setCurrentIndex(1)
        label_mode_row.addWidget(self.cohort_label_mode_combo)
        label_mode_row.addStretch()
        cohort_layout.addLayout(label_mode_row)

        # N parameter (used by Top N / Above threshold / Selected & N neighbors modes)
        n_row = QHBoxLayout()
        n_label = QLabel("N:")
        n_label.setStyleSheet(f"color: {RED_A};")
        n_row.addWidget(n_label)
        self.cohort_label_n_spin = QSpinBox()
        self.cohort_label_n_spin.setRange(1, 500)
        self.cohort_label_n_spin.setValue(5)
        self.cohort_label_n_spin.setToolTip("Number of cohort labels to show (Top N largest), minimum cohort size (Above size threshold), or number of neighboring cohorts (Selected & N neighbors).")
        self.cohort_label_n_spin.valueChanged.connect(self._on_cohort_label_n_changed)
        n_row.addWidget(self.cohort_label_n_spin)
        n_row.addStretch()
        cohort_layout.addLayout(n_row)

        # Label text size row
        label_size_row = QHBoxLayout()
        label_size_label = QLabel("Label Size:")
        label_size_label.setStyleSheet(f"color: {RED_A};")
        label_size_row.addWidget(label_size_label)
        self.cohort_label_size_spin = QSpinBox()
        self.cohort_label_size_spin.setRange(4, 40)
        self.cohort_label_size_spin.setValue(18)
        self.cohort_label_size_spin.setToolTip("Fixed text size for cohort labels.")
        self.cohort_label_size_spin.valueChanged.connect(self._on_cohort_label_size_changed)
        label_size_row.addWidget(self.cohort_label_size_spin)
        label_size_row.addStretch()
        cohort_layout.addLayout(label_size_row)

        # Max tags per label row
        max_tags_row = QHBoxLayout()
        max_tags_label = QLabel("Max Tags:")
        max_tags_label.setStyleSheet(f"color: {RED_A};")
        max_tags_row.addWidget(max_tags_label)
        self.cohort_label_max_tags_spin = QSpinBox()
        self.cohort_label_max_tags_spin.setRange(1, 20)
        self.cohort_label_max_tags_spin.setValue(5)
        self.cohort_label_max_tags_spin.setToolTip("Maximum number of dominant tags to show per cohort label.")
        self.cohort_label_max_tags_spin.valueChanged.connect(self._on_cohort_label_max_tags_changed)
        max_tags_row.addWidget(self.cohort_label_max_tags_spin)
        max_tags_row.addStretch()
        cohort_layout.addLayout(max_tags_row)

        # Smart Label Mode (merged toggle + mode into one dropdown)
        smart_mode_row = QHBoxLayout()
        smart_mode_label = QLabel("Label Mode:")
        smart_mode_label.setStyleSheet(f"color: {RED_A};")
        smart_mode_row.addWidget(smart_mode_label)
        self.smart_label_mode_combo = QComboBox()
        self.smart_label_mode_combo.addItems(["Raw", "All Unique", "Overlap", "Absolute Unique"])
        self.smart_label_mode_combo.setCurrentText("Absolute Unique")
        self.smart_label_mode_combo.setToolTip(
            "How to resolve duplicate labels across cohorts.\n"
            "Raw: no deduplication, each cohort shows its own top tags (may overlap).\n"
            "All Unique (proximity): smaller nearby cohort gets completely different tags (e.g. TagC, TagD).\n"
            "Overlap (proximity): smaller nearby cohort keeps the first shared tag, replaces the rest (e.g. TagA, TagC).\n"
            "Absolute Unique (global): no tag is repeated in ANY cohort label. Cohorts are processed\n"
            "biggest-first; each greedily reserves its top dominant tags, skipping already-taken ones.\n"
            "May run out of tags if there are many cohorts (acceptable)."
        )
        self.smart_label_mode_combo.currentTextChanged.connect(self._on_smart_label_mode_changed)
        smart_mode_row.addWidget(self.smart_label_mode_combo)
        smart_mode_row.addStretch()
        cohort_layout.addLayout(smart_mode_row)

        # Dynamic size toggle
        self.dynamic_label_size_checkbox = QCheckBox("Dynamic Label Size")
        self.dynamic_label_size_checkbox.setChecked(False)
        self.dynamic_label_size_checkbox.setToolTip("Scale label size based on cohort size (file count). Larger cohorts get larger labels.")
        self.dynamic_label_size_checkbox.stateChanged.connect(self._on_dynamic_label_size_toggled)
        cohort_layout.addWidget(self.dynamic_label_size_checkbox)

        # Label color row (two colors: the selected cohort's label blinks
        # between Color 1 and Color 2)
        label_color_row = QHBoxLayout()
        label_color_label = QLabel("Label Color:")
        label_color_label.setStyleSheet(f"color: {RED_A};")
        label_color_row.addWidget(label_color_label)
        self.cohort_label_color_btn = QPushButton("")
        self.cohort_label_color_btn.setFixedSize(40, 24)
        self.cohort_label_color_btn.setToolTip("Primary color for cohort labels. Click to change.")
        self.cohort_label_color_btn.clicked.connect(self._pick_cohort_label_color)
        self._cohort_label_color = (255, 255, 255)  # default white
        self._update_label_color_button()
        label_color_row.addWidget(self.cohort_label_color_btn)
        # Second blink color for the selected cohort's label
        label_color2_label = QLabel("Blink Color:")
        label_color2_label.setStyleSheet(f"color: {RED_A};")
        label_color_row.addWidget(label_color2_label)
        self.cohort_label_color2_btn = QPushButton("")
        self.cohort_label_color2_btn.setFixedSize(40, 24)
        self.cohort_label_color2_btn.setToolTip("Secondary color the selected cohort's label blinks to. Click to change.")
        self.cohort_label_color2_btn.clicked.connect(self._pick_cohort_label_color2)
        self._cohort_label_color2 = (255, 200, 0)  # default amber
        self._update_label_color_button2()
        label_color_row.addWidget(self.cohort_label_color2_btn)

        # Outline color + width row
        outline_row = QHBoxLayout()
        outline_label = QLabel("Outline:")
        outline_label.setStyleSheet(f"color: {RED_A};")
        outline_row.addWidget(outline_label)
        self.cohort_label_outline_color_btn = QPushButton("")
        self.cohort_label_outline_color_btn.setFixedSize(40, 24)
        self.cohort_label_outline_color_btn.setToolTip("Outline color for cohort labels. Click to change.")
        self.cohort_label_outline_color_btn.clicked.connect(self._pick_cohort_label_outline_color)
        self._cohort_label_outline_color = (0, 0, 0)  # default black
        self._update_outline_color_button()
        outline_row.addWidget(self.cohort_label_outline_color_btn)
        self.cohort_label_outline_width_spin = QSpinBox()
        self.cohort_label_outline_width_spin.setRange(0, 4)
        self.cohort_label_outline_width_spin.setValue(3)
        self.cohort_label_outline_width_spin.setToolTip("Outline thickness in pixels. 0 disables the outline.")
        self.cohort_label_outline_width_spin.valueChanged.connect(self._on_cohort_label_outline_width_changed)
        outline_row.addWidget(self.cohort_label_outline_width_spin)
        outline_row.addStretch()
        cohort_layout.addLayout(outline_row)

        label_color_row.addStretch()
        cohort_layout.addLayout(label_color_row)

        cohort_group.setLayout(cohort_layout)
        self._cohort_group = cohort_group  # reorganized into tabs in setup_ui
        layout.addWidget(cohort_group)

        # Tag Importance Section
        importance_group = QGroupBox("Tag Importance")
        importance_group.setToolTip("Tags ranked by contribution to cluster separation (chi-square statistic).")
        importance_layout = QVBoxLayout()
        importance_layout.setSpacing(8)

        self.tag_importance_text = QTextEdit()
        self.tag_importance_text.setReadOnly(True)
        self.tag_importance_text.setMaximumHeight(200)
        self.tag_importance_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {GRAY_40};
                color: {RED_A};
                border: 1px solid {BLUE_60};
                padding: 5px;
                font-size: 11px;
                font-family: Consolas, monospace;
            }}
        """)
        self.tag_importance_text.setPlaceholderText("Tag importance will appear here after rendering...")
        importance_layout.addWidget(self.tag_importance_text)

        importance_group.setLayout(importance_layout)
        self._importance_group = importance_group  # reorganized into tabs in setup_ui
        layout.addWidget(importance_group)

        layout.addStretch()
        sidebar.setLayout(layout)
        return sidebar

    def _reorganize_sidebars(self):
        """Reorganize the right sidebar into tabs and move shared widgets.

        Called from setup_ui after both sidebars are built:
        - Right sidebar gets a tab bar (Actions | Visuals).
        - Camera Wobble group moves from left panel to the Visuals tab.
        - Tag query grid moves from "Selected File Info" into Filter Settings (left).
        - Status label + progress bar are pinned to the left sidebar bottom.
        - "Send to Tab" is pinned to the right sidebar bottom (below the tabs).
        """
        # --- Right sidebar: wrap existing groups in a QTabWidget ---
        right_layout = self.right_sidebar.layout()

        visuals_tab = QWidget()
        visuals_lay = QVBoxLayout(visuals_tab)
        visuals_lay.setContentsMargins(0, 0, 0, 0)
        visuals_lay.addWidget(self._vis_group)
        visuals_lay.addWidget(self.wobble_group)      # moved from left panel
        visuals_lay.addWidget(self._cohort_group)

        actions_tab = QWidget()
        actions_lay = QVBoxLayout(actions_tab)
        actions_lay.setContentsMargins(0, 0, 0, 0)
        actions_lay.addWidget(self._info_group)
        actions_lay.addWidget(self._selection_tags_group)
        # Tag importance sits directly under Cohort Tag Data.
        actions_lay.addWidget(self._importance_group)

        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {TAB_BORDER};
                background-color: {TAB_BACKGROUND};
            }}
            QTabBar::tab {{
                background-color: {TAB_BACKGROUND};
                color: {TAB_TEXT};
                padding: 6px;
                font-size: 13px;
                border: none;
            }}
            QTabBar::tab:selected {{
                background-color: {TAB_SELECTED};
            }}
            QTabBar::tab:hover {{
                background-color: rgb(50, 70, 170);
            }}
        """)
        self.right_tabs.addTab(actions_tab, "Actions")
        self.right_tabs.addTab(visuals_tab, "Visuals")

        # Insert the tab widget at the top of the right sidebar (before the stretch).
        right_layout.insertWidget(0, self.right_tabs)

        # --- Pin "Send to Tab" to the bottom of the right sidebar (below tabs) ---
        # The sidebar layout ends with a stretch; adding after it keeps the group
        # at the very bottom regardless of tab content height.
        right_layout.addWidget(self._send_group)

        # --- Move status label + progress bar to the left sidebar bottom ---
        # The left panel layout ends with a stretch, so widgets added after it
        # sit at the very bottom of the left sidebar.
        left_layout = self.left_sidebar.layout()
        for w in (self.progress_bar, self.phase_label, self.status_label):
            if w is not None:
                w.setParent(self.left_sidebar)
                left_layout.addWidget(w)

        # --- Move tag query grid into Filter Settings (left panel) ---
        if hasattr(self, '_filter_layout') and getattr(self, 'tag_container', None) is not None:
            self.tag_container.setParent(None)  # detach from info layout
            tq_label = QLabel("Tag Query:")
            tq_label.setStyleSheet(f"color: {RED_A};")
            self._filter_layout.addRow(tq_label)
            self._filter_layout.addRow(self.tag_container)

    def toggle_sidebar(self):
        """Toggle visibility of the right sidebar."""
        if self.right_sidebar_visible:
            self.right_sidebar.hide()
            self.right_sidebar_visible = False
            self.sidebar_toggle_btn.setText("<")
            self.sidebar_toggle_btn.setToolTip("Show right sidebar")
        else:
            self.right_sidebar.show()
            self.right_sidebar_visible = True
            self.sidebar_toggle_btn.setText(">")
            self.sidebar_toggle_btn.setToolTip("Hide right sidebar")
        
        # Reposition toggle buttons
        self._position_toggle_buttons()
    
    def toggle_left_sidebar(self):
        """Toggle visibility of the left sidebar."""
        if self.left_sidebar_visible:
            self.left_sidebar.hide()
            self.left_sidebar_visible = False
            self.left_toggle_btn.setText(">")
            self.left_toggle_btn.setToolTip("Show left sidebar")
        else:
            self.left_sidebar.show()
            self.left_sidebar_visible = True
            self.left_toggle_btn.setText("<")
            self.left_toggle_btn.setToolTip("Hide left sidebar")
        
        # Reposition toggle buttons
        self._position_toggle_buttons()
    
    def _position_toggle_buttons(self):
        """Position both toggle buttons on the edges of the widget."""
        y = self.height() // 2 - 25  # Vertically centered
        
        # Left toggle button - positioned at left edge
        self.left_toggle_btn.move(0, y)
        
        # Right toggle button - positioned at right edge
        x = self.width() - 25  # Right edge
        self.sidebar_toggle_btn.move(x, y)
