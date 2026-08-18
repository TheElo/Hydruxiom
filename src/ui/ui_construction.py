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
    QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout, QFileDialog, QTabWidget,
    QStyledItemDelegate, QStyleOptionViewItem, QApplication
)
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import (
    QCloseEvent, QMouseEvent, QVector3D, QFont, QPainter, QColor, QFontMetrics,
)
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


def _scheme_preview_colors(name):
    """Return an ordered list of distinct (r,g,b) colors for a color scheme.

    Used to render each letter of the Color Scheme dropdown in one of that
    scheme's own colors, so users can see what's inside before selecting it.
    Discrete schemes use their full palette; matplotlib colormaps are sampled at
    evenly spaced points (no duplicates).
    """
    from src.core.models import SceneGraph
    if name == "Pastel":
        return list(SceneGraph.CLUSTER_COLORS)
    if name == "Nature":
        return list(SceneGraph.NATURE_COLORS)
    if name == "Sci-Fi":
        return list(SceneGraph.SCIFI_COLORS)
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(name.lower())
        n = 12
        out, seen = [], set()
        for i in range(n):
            t = (i + 0.5) / n
            c = tuple(int(round(v * 255)) for v in cmap(t)[:3])
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out or [(170, 170, 170)]
    except Exception:
        return list(SceneGraph.CLUSTER_COLORS[:8])


class _ColorSchemeDelegate(QStyledItemDelegate):
    """Renders each character of a Color Scheme item in one of that scheme's colors.

    Letters cycle through the palette (no repeats until exhausted) so every color is
    visible without duplication. Applied to both the popup list and the closed combo
    display, so the currently selected scheme also shows its own colors. Falls back
    to plain text if no preview data exists for a name.
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        from PySide6.QtWidgets import QStyle
        from PySide6.QtGui import QPalette

        name = str(index.data(Qt.ItemDataRole.DisplayRole))
        colors = _scheme_preview_colors(name) if name else []
        selected = bool(option.state & QStyle.StateFlag.State_Selected)

        # Draw the background/selection highlight OURSELVES (no super().paint() and no
        # drawControl(CE_ItemViewItem), both of which would also render Qt's default white
        # item text — that ghosted/misaligned layer is what we must avoid). Filling from
        # the palette keeps it theme-correct.
        pal = option.palette
        bg_color = (pal.color(QPalette.ColorRole.Highlight) if selected
                    else pal.color(QPalette.ColorRole.Base))
        painter.fillRect(option.rect, bg_color)

        if not colors or not name:
            return

        # Draw each character in its own scheme color. Letters cycle through the palette
        # (no repeats until exhausted). Vertically centered using font metrics.
        fm = QFontMetrics(option.font)
        y_center = (option.rect.top() + option.rect.bottom()) // 2
        baseline = y_center + (fm.ascent() - fm.descent()) // 2
        x = option.rect.left() + 6

        painter.save()
        for i, ch in enumerate(name):
            r, g, b = colors[i % len(colors)]
            if selected:
                # Keep light letters readable on the blue selection row.
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                col = QColor(255, 255, 255) if lum > 160 else QColor(r, g, b)
            else:
                col = QColor(r, g, b)
            painter.setPen(col)
            painter.drawText(x, baseline, ch)
            x += fm.horizontalAdvance(ch)
        painter.restore()


class UIConstructionMixin:
    @staticmethod
    def _make_scrollable(content, min_width=None, max_width=None):
        """Wrap a content widget in a QScrollArea that scrolls only when needed.

        The vertical scrollbar policy is AsNeeded, so the bar (and its space)
        appears only when the content doesn't fit — keeping sidebars compact on
        large screens while staying usable on low-resolution ones. Horizontal
        scrolling is disabled; widgets stretch to the sidebar width instead.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        if min_width is not None:
            scroll.setMinimumWidth(min_width)
        if max_width is not None:
            scroll.setMaximumWidth(max_width)
        scroll.setWidget(content)
        return scroll

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
        # The left sidebar content is scrollable so it scrolls instead of being
        # squished on low-resolution screens (scrollbar only when needed). The
        # progress bar / phase / status labels stay pinned below, OUTSIDE the
        # scroll area. NOTE: the content widget must not be added to any other
        # layout directly — that would reparent it out of the QScrollArea and
        # silently disable scrolling.
        panel = QWidget()
        panel.setMinimumWidth(250)
        panel.setMaximumWidth(350)
        content = QWidget()
        self._left_content = content  # used by _reorganize_sidebars (cohort row)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        # Title
        title = QLabel("3D Tag Space Visualization")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {RED_A};")
        layout.addWidget(title)

        # Settings + Help buttons side by side
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.settings_button = QPushButton("\u2699")  # cog (settings)
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
        self.settings_button.setToolTip("Open the 3D tag map settings window.\nShortcut: F3")
        self.settings_button.clicked.connect(self.open_settings_dialog)
        btn_row.addWidget(self.settings_button)

        # Help button (opens the manual window)
        self.help_button = QPushButton("\U0001F4D6")  # book (manual)
        self.help_button.setStyleSheet(f"""
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
        self.help_button.setToolTip("Open the Hydruxiom manual (controls and interactions).")
        self.help_button.clicked.connect(self.open_manual_dialog)
        btn_row.addWidget(self.help_button)

        layout.addLayout(btn_row)

        # Smart Scale master toggle: when on, after data loads the app picks a
        # profile by file count and overwrites UMAP/DBSCAN/visualization settings.
        # Profiles are managed in Settings -> Smart Scale tab.
        self.smart_scale_checkbox = QCheckBox("Smart Scale")
        self.smart_scale_checkbox.setChecked(getattr(self, 'smart_scale_enabled', False))
        self.smart_scale_checkbox.setToolTip(
            "When enabled, after Load & Compute the app selects a settings profile\n"
            "based on the number of files and overwrites UMAP / DBSCAN / node size /\n"
            "transparency / spread with that profile's values.\n"
            "Configure the profiles (size ranges + their settings) in Settings -> Smart Scale."
        )
        self.smart_scale_checkbox.stateChanged.connect(self._on_smart_scale_toggled)
        layout.addWidget(self.smart_scale_checkbox)

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

        # (Chunk Size moved to the Settings window -> Clients tab. The spinbox is
        # created there and stored on this tab as self.chunk_size_spin, so all
        # existing references keep working.)

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
        self.n_neighbors_spin.setValue(12)
        self.n_neighbors_spin.setToolTip("UMAP parameter: Number of neighboring points to consider.\nLower (5-10) = tighter, more clusters.\nHigher (20-50) = smoother, fewer clusters.\nDefault: 15")
        algo_layout.addRow("N Neighbors:", self.n_neighbors_spin)

        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(0, 100)
        self.min_dist_spin.setValue(1)
        self.min_dist_spin.setSuffix("%")
        self.min_dist_spin.setToolTip("UMAP parameter: Minimum distance between points (0-100%).\nLower (0-20%) = points packed tightly together.\nHigher (50-100%) = more spread out, easier to see individual points.\nDefault: 10%")
        algo_layout.addRow("Min Dist:", self.min_dist_spin)

        self.n_epochs_spin = QSpinBox()
        self.n_epochs_spin.setRange(0, 5000)
        self.n_epochs_spin.setValue(64)
        self.n_epochs_spin.setToolTip("UMAP parameter: Number of training epochs (0 = auto).\nAuto typically gives 500-1000 epochs based on data size.\nHigher = more accurate but slower.\nDefault: 0 (auto)")
        algo_layout.addRow("Epochs:", self.n_epochs_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.01, 10.0)
        self.learning_rate_spin.setValue(2.0)
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
        self.eps_spin.setValue(12)
        self.eps_spin.setSingleStep(1)
        self.eps_spin.setToolTip("DBSCAN parameter: Maximum distance between points in same cluster (as % of data spread).\nLower (10-30%) = many small, tight clusters.\nHigher (100-200%) = few large clusters.\nDefault: 50%")
        cluster_layout.addRow("EPS (%):", self.eps_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(2, 100)
        self.min_samples_spin.setValue(8)
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
        self.sub_eps_spin.setValue(12)
        self.sub_eps_spin.setSingleStep(5)
        self.sub_eps_spin.setToolTip("Sub-cluster EPS (%): DBSCAN distance for splitting a selected cohort into sub-cohorts.\nLower = finer/smaller sub-cohorts. Independent from global EPS.")
        cluster_layout.addRow("Sub EPS (%):", self.sub_eps_spin)

        self.sub_min_samples_spin = QSpinBox()
        self.sub_min_samples_spin.setRange(2, 100)
        self.sub_min_samples_spin.setValue(10)
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
        self.min_doc_freq_spin.setRange(0, 100)
        self.min_doc_freq_spin.setValue(5)
        self.min_doc_freq_spin.setToolTip("Vectorizer: Minimum documents a tag must appear in\nto be included in the vocabulary.\nHigher = fewer rare tags, faster UMAP.\nLower = more tags, slower but more detailed.\n0 = disabled (keep every tag).\nDefault: 5")
        filter_layout.addRow("Min Doc Freq:", self.min_doc_freq_spin)

        # Tag query builder is reparented here from the right sidebar after both
        # panels are built (see setup_ui). Store the layout for that step.
        self._filter_layout = filter_layout

        filter_group.setLayout(filter_layout)
        # Left panel order: Client -> Filter. The Algorithm + Cluster groups are
        # moved to the right sidebar "Algorithm" tab by _reorganize_sidebars().
        layout.addWidget(filter_group)

        # WASD Navigation group: appearance of the W/S/A/D preview arrows + a toggle
        # to keep them visible (anchored on the largest cohort when nothing selected).
        wasd_group = QGroupBox("WASD Navigation")
        wasd_group.setToolTip("Appearance and behavior of the W/S/A/D navigation preview arrows.")
        wl_layout = QFormLayout()

        self.wasd_persistent_checkbox = QCheckBox("Keep WASD labels visible")
        self.wasd_persistent_checkbox.setChecked(getattr(self, 'wasd_persistent_labels', False))
        self.wasd_persistent_checkbox.stateChanged.connect(self._on_wasd_persistent_toggled)
        self.wasd_persistent_checkbox.setToolTip(
            "When enabled the W/S/A/D direction arrows stay on screen instead of\n"
            "fading out after you stop navigating. With no cohort selected they anchor\n"
            "on the largest cohort, so you can always see where each direction leads."
        )
        wl_layout.addRow(self.wasd_persistent_checkbox)

        self.wasd_line_btn = QPushButton("")
        self.wasd_line_btn.setFixedWidth(80)
        _wlc = getattr(self, 'wasd_line_color', (80, 255, 140))
        self.wasd_line_btn.setStyleSheet(f"background-color: rgb({int(_wlc[0])},{int(_wlc[1])},{int(_wlc[2])}); border: 1px solid #4050a0;")
        self.wasd_line_btn.clicked.connect(self._pick_wasd_line_color)
        self.wasd_line_btn.setToolTip("Color of the W/S/A/D path lines. Click to change.")
        wl_layout.addRow("Line Color:", self.wasd_line_btn)

        self.wasd_letter_btn = QPushButton("")
        self.wasd_letter_btn.setFixedWidth(80)
        _wcc = getattr(self, 'wasd_letter_color', (80, 255, 140))
        self.wasd_letter_btn.setStyleSheet(f"background-color: rgb({int(_wcc[0])},{int(_wcc[1])},{int(_wcc[2])}); border: 1px solid #4050a0;")
        self.wasd_letter_btn.clicked.connect(self._pick_wasd_letter_color)
        self.wasd_letter_btn.setToolTip("Color of the W/S/A/D letter labels. Click to change.")
        wl_layout.addRow("Letter Color:", self.wasd_letter_btn)

        self.wasd_label_spin = QSpinBox()
        self.wasd_label_spin.setRange(8, 120)
        try:
            self.wasd_label_spin.setValue(int(getattr(self, 'wasd_label_size', 36)))
        except (TypeError, ValueError):
            self.wasd_label_spin.setValue(36)
        self.wasd_label_spin.valueChanged.connect(self._on_wasd_label_size_changed)
        self.wasd_label_spin.setToolTip("Font size of the W/S/A/D letter labels.")
        wl_layout.addRow("Label Size:", self.wasd_label_spin)

        wasd_group.setLayout(wl_layout)
        layout.addWidget(wasd_group)

        self._algo_group = algo_group
        self._cluster_group = cluster_group

        # Camera Settings Group: orbit speed + wobble (depth effect). Placed in
        # the right sidebar "Visuals" tab by _reorganize_sidebars(). Orbit Speed
        # lives here (not in Visualization Settings) because it controls the
        # camera, not node appearance.
        wobble_group = QGroupBox("Camera Settings")
        wobble_group.setToolTip("Camera behavior: arrow-key orbit speed and continuous wobble for depth perception.")
        wobble_layout = QFormLayout()

        self.orbit_speed_spin = QDoubleSpinBox()
        self.orbit_speed_spin.setRange(0.1, 50.0)
        self.orbit_speed_spin.setValue(0.2)
        self.orbit_speed_spin.setDecimals(2)
        self.orbit_speed_spin.setSingleStep(0.1)
        self.orbit_speed_spin.setToolTip("Speed of camera orbit when using arrow keys.\nLower = slower, more precise movement.\nHigher = faster camera rotation.\nDefault: 0.2")
        wobble_layout.addRow("Orbit Speed:", self.orbit_speed_spin)

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

        # --- Action buttons: pinned to the BOTTOM of the sidebar (outside the
        # scroll area) so they stay reachable while the settings groups above
        # scroll. The Split | Pop | Cut row is inserted into this container by
        # _reorganize_sidebars().
        actions = QWidget()
        self._left_actions_layout = QVBoxLayout(actions)
        self._left_actions_layout.setContentsMargins(0, 0, 0, 0)
        self._left_actions_layout.setSpacing(6)

        # Load Button and Progress
        self.load_button = QPushButton("Load and Compute")
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
        self._left_actions_layout.addWidget(self.load_button)

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
        self._left_actions_layout.addWidget(self.recompute_button)

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

        # Optimize: small icon button next to Regroup (rocket = automation/speed)
        self.optimize_button = QPushButton("\U0001F680")  # rocket
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

        self._left_actions_layout.addLayout(regroup_row)

        # Deorphan: assign each noise (-1) node to its nearest non-noise cohort.
        self.deorphan_button = QPushButton("Deorphan")
        self.deorphan_button.setStyleSheet(f"""
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
        self.deorphan_button.setEnabled(False)
        self.deorphan_button.clicked.connect(self.start_deorphan)
        self.deorphan_button.setToolTip(
            "Assign every orphan (noise, cluster -1) node to the cohort of its\n"
            "nearest non-orphan node. No re-clustering — just re-labels orphans.\n"
            "Use after Regroup when you want fewer unassigned nodes."
        )
        self._left_actions_layout.addWidget(self.deorphan_button)

        # (Split | Pop | Cut row is inserted into this actions container by
        # _reorganize_sidebars(), because those buttons are created later in
        # create_right_sidebar.)

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
        self._left_actions_layout.addWidget(self.clear_button)

        # Trailing stretch: keeps all content top-aligned when the scroll area's
        # page is stretched to fill the viewport (setWidgetResizable). Without it,
        # the group boxes above would expand and look "stretched over the area".
        layout.addStretch()

        # Session buttons are now automatic (hidden). Objects kept for setEnabled compat.
        self.save_session_button = QPushButton()  # hidden, auto-saved after each load
        self.load_session_button = QPushButton()  # hidden, auto-loaded on startup

        # Progress bar + phase/status labels are pinned to the BOTTOM of the
        # sidebar (outside the scroll area) so they stay visible while scrolling.
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Phase label - shows which pipeline stage is currently running
        self.phase_label = QLabel("Phase: Idle")
        self.phase_label.setStyleSheet(f"color: {BLUE_60}; font-size: 11px; font-weight: bold;")

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {RED_A}; font-size: 12px;")

        scroll = self._make_scrollable(content)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addWidget(scroll, stretch=1)
        outer.addWidget(actions)  # pinned action buttons (bottom, above status)
        for w in (self.progress_bar, self.phase_label, self.status_label):
            outer.addWidget(w)
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
                        # Camera moved -> refresh WASD preview paths (screen-space).
                        if getattr(self.parent_tab, '_wasd_mode', False):
                            try:
                                self.parent_tab._on_camera_moved()
                            except Exception:
                                pass
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

                    # Any manual camera grab interrupts an in-flight smooth glide.
                    if event.button() != Qt.MouseButton.RightButton:
                        self.parent_tab._cancel_center_animation()

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

                def mouseDoubleClickEvent(self, event: QMouseEvent):
                    """Double-click on empty space clears the current selection.

                    Double-clicking a node/cluster keeps its selection (only an
                    empty-space double-click unselects), so this is a safe way to
                    dismiss a cohort without touching Clear.
                    """
                    if event.button() == Qt.MouseButton.LeftButton:
                        try:
                            closest_idx = self.parent_tab.pick_nearest_point(self, event.pos())
                            if closest_idx is None:
                                self.parent_tab.clear_selection()
                        except Exception as e:
                            print(f"Error in double-click handling: {e}")
                    super().mouseDoubleClickEvent(event)

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

                    If "Smooth center transition" is enabled in settings, the camera
                    glides to the new center instead of teleporting.

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
                            # Smooth glide vs. instant teleport (settings toggle).
                            if getattr(self.parent_tab, 'smooth_center_transition', False):
                                self.parent_tab._start_center_animation(center)
                            else:
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
                    else:
                        # WASD/QE: screen-space cohort navigation (no modifiers held).
                        # Handled here because the GL view holds keyboard focus while
                        # orbiting with arrow keys; paths refresh as the camera moves.
                        wasd = {Qt.Key_W: 'W', Qt.Key_S: 'S', Qt.Key_A: 'A', Qt.Key_D: 'D',
                                Qt.Key_Q: 'Q', Qt.Key_E: 'E'}.get(event.key())
                        if wasd is not None and not (event.modifiers() & Qt.ControlModifier) \
                                and not (event.modifiers() & Qt.ShiftModifier):
                            self.parent_tab._wasd_mode = True
                            self.parent_tab._wasd_handle_key(wasd)
                            event.accept()
                            return
                    super().keyPressEvent(event)
            
            # Create the custom view widget
            view = RightClickGLView(self)
            # Background from settings (default black; applied again after load_settings)
            _bg = getattr(self, 'bg_color', (0.0, 0.0, 0.0))
            view.setBackgroundColor((int(_bg[0]*255), int(_bg[1]*255), int(_bg[2]*255), 255))
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

    def create_right_sidebar(self):
        """Create the right sidebar with actions panel.

        The tab pages (Stats | Visuals, built in _reorganize_sidebars) are each
        wrapped in a scroll area so their content scrolls instead of being
        squished on low-resolution screens; "Send to Tab" stays pinned at the
        bottom outside the tabs.
        """
        sidebar = QWidget()
        sidebar.setMinimumWidth(200)
        sidebar.setMaximumWidth(350)
        self._right_outer_layout = QVBoxLayout(sidebar)
        self._right_outer_layout.setContentsMargins(0, 0, 0, 0)
        self._right_outer_layout.setSpacing(15)

        # Placeholder layout; the real tab pages are assembled in
        # _reorganize_sidebars() from the groups created below.
        layout = QVBoxLayout()
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

        # Text area for non-tag info (file id, cluster, score, position).
        # No fixed height and no inner border — the groupbox already frames it,
        # so an empty selection doesn't leave a big dead box at the top.
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                color: {RED_A};
                border: none;
                padding: 0px;
                font-size: 12px;
            }}
        """)
        self.info_text.setPlaceholderText("Left-click a node or cohort to inspect it.")
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

        # Rendered as a styled HTML table (see render_cohort_tags_html); no
        # inner border — the groupbox frames it and rows carry their own bg.
        self.selection_tags_text = QTextEdit()
        self.selection_tags_text.setReadOnly(True)
        self.selection_tags_text.setMinimumHeight(80)
        self.selection_tags_text.setMaximumHeight(260)
        self.selection_tags_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {GRAY_33};
                color: {RED_A};
                border: none;
                padding: 4px;
                font-size: 12px;
            }}
        """)
        self.selection_tags_text.setPlaceholderText("Select a cohort to see its most common tags...")
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

        # Explore button (play symbol — starts the camera tour)
        self.time_travel_button = QPushButton("\u25B6")  # play triangle
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
        self.time_travel_button.setToolTip("Explore: fly the camera through the largest cluster centroids.\nShortcut-free tour — click again to stop.")
        self.time_travel_button.clicked.connect(self._toggle_time_travel)
        self.time_travel_button.setEnabled(False)  # Enabled when data is loaded
        send_layout.addWidget(self.time_travel_button)

        # Cohort action buttons (Split / Pop / Cut) are created here but placed
        # in the LEFT panel between Deorphan and Clear by _reorganize_sidebars().
        # Cut - select the cohort nodes and remove everything else
        self.cut_button = QPushButton("Cut")
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

        # Pop button - remove the selected cohort from the view (inverse of Cut out)
        self.pop_button = QPushButton("Pop")
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

        # Split group button - apply cluster algo on selection, keep positions
        self.cluster_button = QPushButton("Split")
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

        # Node Sizing mode — how node size relates to camera distance / screen.
        self.node_sizing_combo = QComboBox()
        self.node_sizing_combo.addItems([
            "Distance",                 # legacy: perspective scaling (default)
            "Screen-constant",          # relative: constant pixel size on screen
            "Uniform single size",      # perf probe: one minimal fixed size
            "Auto-scale to view distance",  # far -> larger, near -> smaller
        ])
        _ns_idx = self.node_sizing_combo.findText(getattr(self, 'node_sizing_mode', "Distance"))
        if _ns_idx >= 0:
            self.node_sizing_combo.setCurrentIndex(_ns_idx)
        else:
            self.node_sizing_combo.setCurrentIndex(0)
        self.node_sizing_combo.currentTextChanged.connect(self._on_node_sizing_changed)
        self.node_sizing_combo.setToolTip(
            "How node size behaves relative to the camera:\n"
            "- Distance (default): nodes scale with distance — bigger when close\n"
            "  (classic perspective look).\n"
            "- Screen-constant: nodes keep a fixed pixel size on screen regardless of\n"
            "  how far the camera is.\n"
            "- Uniform single size: every node uses one minimal fixed size — no scaling\n"
            "  at all. A performance probe (disabling per-node size variation).\n"
            "- Auto-scale to view distance: nodes grow when you zoom out and shrink\n"
            "  when you zoom in, so they stay visible without overdraw."
        )
        vis_layout.addRow("Node Sizing:", self.node_sizing_combo)

        self.spread_spin = QDoubleSpinBox()
        self.spread_spin.setRange(0.1, 10.0)
        self.spread_spin.setValue(1.0)
        self.spread_spin.setSingleStep(0.01)
        self.spread_spin.setDecimals(2)
        self.spread_spin.valueChanged.connect(self._on_spread_changed)
        self.spread_spin.setToolTip("Scale factor for spreading nodes apart in 3D space.\n1.0 = original algorithm output.\nHigher = nodes spread further apart.\nUseful for seeing clusters more clearly.\nDefault: 1.0")
        vis_layout.addRow("Spread:", self.spread_spin)

        # (Orbit Speed moved to the Camera Settings group — it controls the
        # camera, not node appearance.)
        self.transparency_spin = QDoubleSpinBox()
        self.transparency_spin.setRange(0.0, 1.0)
        self.transparency_spin.setValue(0.8)
        self.transparency_spin.setDecimals(2)
        self.transparency_spin.setSingleStep(0.05)
        self.transparency_spin.valueChanged.connect(self._on_transparency_changed)
        self.transparency_spin.setToolTip("Alpha (transparency) of nodes in the 3D view.\n0.0 = fully transparent, 1.0 = fully opaque.\nDefault: 0.8")
        vis_layout.addRow("Transparency:", self.transparency_spin)

        # Node blending mode — controls how overlapping nodes combine color.
        self.node_blending_combo = QComboBox()
        self.node_blending_combo.addItems(["Additive", "Normal Alpha", "Simple"])
        _nb_idx = self.node_blending_combo.findText(getattr(self, 'node_blending', "Normal Alpha"))
        if _nb_idx >= 0:
            self.node_blending_combo.setCurrentIndex(_nb_idx)
        else:
            self.node_blending_combo.setCurrentIndex(0)
        self.node_blending_combo.currentTextChanged.connect(self._on_node_blending_changed)
        self.node_blending_combo.setToolTip(
            "How overlapping nodes combine color in the 3D view:\n"
            "- Normal Alpha (default): standard alpha blending; same-hue overlaps\n"
            "  converge to that hue instead of blowing out. Density still visible.\n"
            "- Additive: colors accumulate where nodes overlap and saturate to\n"
            "  white at high density. Shows density, but loses hue.\n"
            "- Simple: no blending — the nearest node wins per pixel, so overlaps\n"
            "  keep their exact shape and color (no density cue).\n"
            "All three cost ~the same; applies live."
        )
        vis_layout.addRow("Node Blending:", self.node_blending_combo)

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

        # 3D view background color (default: black, matching legacy behavior)
        self.bg_color = getattr(self, 'bg_color', (0.0, 0.0, 0.0))
        self.bg_color_btn = QPushButton("")
        self.bg_color_btn.setFixedWidth(80)
        r, g, b = int(self.bg_color[0]*255), int(self.bg_color[1]*255), int(self.bg_color[2]*255)
        self.bg_color_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #4050a0;")
        self.bg_color_btn.clicked.connect(self._pick_bg_color)
        self.bg_color_btn.setToolTip("Background color of the 3D view.\nClick to choose a custom color.")
        vis_layout.addRow("View Background:", self.bg_color_btn)

        # Reset the persistent WASD travel trail (translucent turquoise line of
        # every cohort visited via W/S/A/D navigation).
        self.reset_trail_btn = QPushButton("Reset Travel Trail")
        self.reset_trail_btn.clicked.connect(self._reset_wasd_trail)
        self.reset_trail_btn.setToolTip(
            "Clear the persistent WASD travel trail — the translucent turquoise line\n"
            "connecting every cohort you've visited with W/S/A/D navigation.\n"
            "The trail is also cleared automatically when a new scene loads."
        )
        vis_layout.addRow(self.reset_trail_btn)

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
        self.twinkle_freq_spin.setValue(0.5)
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
        # Render each option's letters in its own scheme colors (popup + closed display).
        _cs_delegate = _ColorSchemeDelegate(self)
        self.color_scheme_combo.setItemDelegate(_cs_delegate)
        self.color_scheme_combo.view().setItemDelegate(_cs_delegate)
        self.color_scheme_combo.addItems(["Pastel", "Nature", "Sci-Fi", "Viridis", "Plasma", "Inferno", "Coolwarm"])
        _cs_idx = self.color_scheme_combo.findText(getattr(self, 'color_scheme', "Pastel"))
        if _cs_idx >= 0:
            self.color_scheme_combo.setCurrentIndex(_cs_idx)
        else:
            self.color_scheme_combo.setCurrentText("Pastel")
        self.color_scheme_combo.currentTextChanged.connect(self._on_color_scheme_changed)
        self.color_scheme_combo.setToolTip(
            "Color scheme for cluster nodes:\n"
            "- Pastel (default): soft, distinct colors.\n"
            "- Nature: green / camouflage tones.\n"
            "- Sci-Fi: cool clean blues, cyans and purples.\n"
            "- Viridis/Plasma/Inferno/Coolwarm: Matplotlib colormaps."
        )
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
        self.show_cohort_labels_checkbox.setChecked(True)
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
        self.cohort_label_n_spin.setValue(7)
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
        self.cohort_label_size_spin.setValue(15)
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
        self.cohort_label_max_tags_spin.setValue(2)
        self.cohort_label_max_tags_spin.setToolTip("Maximum number of dominant tags to show per cohort label.")
        self.cohort_label_max_tags_spin.valueChanged.connect(self._on_cohort_label_max_tags_changed)
        max_tags_row.addWidget(self.cohort_label_max_tags_spin)
        max_tags_row.addStretch()
        cohort_layout.addLayout(max_tags_row)

        # Fade label transitions: when switching selection in "Selected & N
        # neighbors" mode, labels that drop out fade to transparent over the set
        # duration instead of vanishing instantly (less disorienting).
        self.label_fade_checkbox = QCheckBox("Fade label transitions")
        self.label_fade_checkbox.setChecked(True)
        self.label_fade_checkbox.setToolTip(
            "When enabled, in 'Selected & N neighbors' mode the labels that drop out\n"
            "as you switch selection fade to transparent over the duration below instead\n"
            "of disappearing instantly."
        )
        self.label_fade_checkbox.stateChanged.connect(self._on_label_fade_toggled)
        cohort_layout.addWidget(self.label_fade_checkbox)

        fade_dur_row = QHBoxLayout()
        fade_dur_label = QLabel("Fade Duration:")
        fade_dur_label.setStyleSheet(f"color: {RED_A};")
        fade_dur_row.addWidget(fade_dur_label)
        self.label_fade_duration_spin = QSpinBox()
        self.label_fade_duration_spin.setRange(50, 60000)
        self.label_fade_duration_spin.setValue(2000)
        self.label_fade_duration_spin.setSingleStep(100)
        self.label_fade_duration_spin.setSuffix(" ms")
        self.label_fade_duration_spin.setToolTip("How long the fade-out takes when a label drops out (milliseconds).")
        fade_dur_row.addWidget(self.label_fade_duration_spin)
        fade_dur_row.addStretch()
        cohort_layout.addLayout(fade_dur_row)

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

        # Rendered as a styled HTML table (see render_importance_html); no inner
        # border — the groupbox frames it and rows carry their own bg.
        self.tag_importance_text = QTextEdit()
        self.tag_importance_text.setReadOnly(True)
        self.tag_importance_text.setMinimumHeight(80)
        self.tag_importance_text.setMaximumHeight(260)
        self.tag_importance_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {GRAY_33};
                color: {RED_A};
                border: none;
                padding: 4px;
                font-size: 12px;
            }}
        """)
        self.tag_importance_text.setPlaceholderText("Tag importance will appear here after rendering...")
        importance_layout.addWidget(self.tag_importance_text)

        importance_group.setLayout(importance_layout)
        self._importance_group = importance_group  # reorganized into tabs in setup_ui
        layout.addWidget(importance_group)

        # No stretch / setLayout here: _reorganize_sidebars() wraps the groups
        # in scrollable tab pages and assembles them into self._right_outer_layout.
        return sidebar

    def _reorganize_sidebars(self):
        """Reorganize the right sidebar into tabs and move shared widgets.

        Called from setup_ui after both sidebars are built:
        - Right sidebar gets a tab bar (Stats | Visuals); each tab page is
          scrollable so content scrolls instead of squishing on small screens.
        - Camera Settings group (orbit speed + wobble) moves to the Visuals tab.
        - Tag query grid moves from "Selected File Info" into Filter Settings (left).
        - Status label + progress bar are pinned to the left sidebar bottom.
        - "Send to Tab" is pinned to the right sidebar bottom (below the tabs).
        """
        # --- Right sidebar: wrap existing groups in scrollable tab pages ---
        def _tab_page(*groups):
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(15)
            for g in groups:
                lay.addWidget(g)
            # Trailing stretch keeps content top-aligned when the page is
            # stretched to fill the viewport (setWidgetResizable).
            lay.addStretch()
            return self._make_scrollable(page)

        visuals_tab = _tab_page(self._vis_group, self.wobble_group, self._cohort_group)
        actions_tab = _tab_page(self._info_group, self._selection_tags_group,
                                self._importance_group)  # importance under cohort tags
        # Algorithm + Cluster groups moved here from the left sidebar.
        algo_tab = _tab_page(self._algo_group, self._cluster_group)

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
        self.right_tabs.addTab(actions_tab, "Stats")
        self.right_tabs.addTab(visuals_tab, "Visuals")
        self.right_tabs.addTab(algo_tab, "Algorithm")

        # Tabs fill the right sidebar; "Send to Tab" stays pinned below them.
        self._right_outer_layout.addWidget(self.right_tabs, stretch=1)
        self._right_outer_layout.addWidget(self._send_group)

        # --- Move Split | Pop | Cut row to the left panel actions area, between Deorphan and Clear ---
        # The buttons are created in create_right_sidebar; adding them to a layout
        # reparents them automatically.
        cohort_row = QHBoxLayout()
        cohort_row.setSpacing(6)
        for b in (self.cluster_button, self.pop_button, self.cut_button):
            cohort_row.addWidget(b, 1)
        _clear_idx = self._left_actions_layout.indexOf(self.clear_button)
        if _clear_idx >= 0:
            self._left_actions_layout.insertLayout(_clear_idx, cohort_row)

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
