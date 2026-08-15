"""3D Tag Space Visualization Tab Widget.

PyQt-based 3D visualization tab for exploring tag relationships.
"""

import json
import os
import tempfile
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QComboBox, QProgressBar, QGroupBox, QFormLayout,
    QTextEdit, QSplitter, QScrollArea, QLineEdit, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout, QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QEvent
from PySide6.QtGui import QCloseEvent, QMouseEvent, QVector3D, QFont
from src.ui.styles import GRAY_40, GRAY_33, RED_A, BLUE_60


class ClickableTag(QLabel):
    """Clickable tag label that cycles through four visual states.
    
    States:
    - 0 (neutral): White text
    - 1 (included): Green text (added to query)
    - 2 (excluded): Red text with "-" prefix (excluded from query)
    - 3 (OR): Bright blue text (added to OR bracket group)
    """
    
    stateChanged = Signal(str, int)  # tag_name, new_state
    
    def __init__(self, tag_name, parent=None):
        if tag_name is None:
            tag_name = ""
        super().__init__(tag_name, parent)
        self.tag_name = tag_name
        self.state = 0  # 0=neutral, 1=included, 2=excluded, 3=OR
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            ClickableTag {{
                background-color: transparent;
                color: {RED_A};
                padding: 2px 6px;
                margin: 1px;
                border-radius: 3px;
                font-size: 11px;
            }}
            ClickableTag:hover {{
                background-color: {BLUE_60};
            }}
        """)
        self.setAlignment(Qt.AlignCenter)
    
    def mousePressEvent(self, event):
        """Handle click to cycle through states."""
        if event.button() == Qt.LeftButton:
            self.state = (self.state + 1) % 4
            self._update_appearance()
            self.stateChanged.emit(self.tag_name, self.state)
            event.accept()
        super().mousePressEvent(event)
    
    def _update_appearance(self):
        """Update label text and color based on current state."""
        if self.state == 0:
            # Neutral - white text
            self.setText(self.tag_name)
            self.setStyleSheet(f"""
                ClickableTag {{
                    background-color: transparent;
                    color: {RED_A};
                    padding: 2px 6px;
                    margin: 1px;
                    border-radius: 3px;
                    font-size: 11px;
                }}
                ClickableTag:hover {{
                    background-color: {BLUE_60};
                }}
            """)
        elif self.state == 1:
            # Included - green text
            self.setText(self.tag_name)
            self.setStyleSheet(f"""
                ClickableTag {{
                    background-color: transparent;
                    color: #44ff44;
                    padding: 2px 6px;
                    margin: 1px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }}
                ClickableTag:hover {{
                    background-color: {BLUE_60};
                }}
            """)
        elif self.state == 2:
            # Excluded - red text with "-" prefix and strikethrough
            self.setText(f"-{self.tag_name}")
            self.setStyleSheet(f"""
                ClickableTag {{
                    background-color: transparent;
                    color: #ff4444;
                    padding: 2px 6px;
                    margin: 1px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                    text-decoration: line-through;
                }}
                ClickableTag:hover {{
                    background-color: {BLUE_60};
                }}
            """)
        else:
            # OR - bright blue text
            self.setText(self.tag_name)
            self.setStyleSheet(f"""
                ClickableTag {{
                    background-color: transparent;
                    color: #44aaff;
                    padding: 2px 6px;
                    margin: 1px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }}
                ClickableTag:hover {{
                    background-color: {BLUE_60};
                }}
            """)

class TagMap3DSplitWindow(QWidget):
    """Separate window syncing image previews to the 3D tag map.

    Display modes based on selection state:
    - Single file selected: shows that file
    - Cohort selected: shows a grid of thumbnails for those files
    - Nothing selected: shows one representative image per existing cohort
    """

    def __init__(self, parent_tab):
        super().__init__()
        self.parent_tab = parent_tab
        self.setWindowTitle("3D Tag Map - Image Preview")
        self.resize(900, 700)
        self.setMinimumSize(600, 400)

        # Layout: title bar + control bar + scrollable image grid
        outer = QVBoxLayout()
        outer.setContentsMargins(5, 5, 5, 5)

        self.title_label = QLabel("No Selection")
        self.title_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; padding: 5px;")
        outer.addWidget(self.title_label)

        # Control bar for grid settings
        from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QHBoxLayout
        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("Columns:"))
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 20)
        self.columns_spin.setValue(4)
        self.columns_spin.setToolTip("Number of columns in the image grid.")
        controls.addWidget(self.columns_spin)

        controls.addWidget(QLabel("Max Files:"))
        self.max_files_spin = QSpinBox()
        self.max_files_spin.setRange(1, 500)
        self.max_files_spin.setValue(60)
        self.max_files_spin.setToolTip("Maximum number of thumbnails to pull.")
        controls.addWidget(self.max_files_spin)

        controls.addWidget(QLabel("Image Size:"))
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(50, 800)
        self.image_size_spin.setValue(200)
        self.image_size_spin.setToolTip("Size of each thumbnail in pixels.")
        controls.addWidget(self.image_size_spin)

        controls.addStretch()
        outer.addLayout(controls)

        # Load saved split window settings
        self._load_settings()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(6)
        self.grid_container.setLayout(self.grid_layout)
        self.scroll_area.setWidget(self.grid_container)
        outer.addWidget(self.scroll_area)

        # Single-file view (full-res, scaled to fill the window)
        self.single_file_label = QLabel()
        self.single_file_label.setAlignment(Qt.AlignCenter)
        self.single_file_label.setStyleSheet("background-color: black;")
        self.single_file_label.hide()
        self._single_file_pixmap = None
        outer.addWidget(self.single_file_label, stretch=1)

        self.setLayout(outer)

        # Store clickable cohort tiles for future interaction
        self.cohort_tiles = []  # list of (cluster_id, QLabel)

    def clear_grid(self):
        """Remove all widgets from the grid layout."""
        while self.grid_layout.count() > 0:
            item = self.grid_layout.takeAt(0)
            if item and hasattr(item, 'widget'):
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()
        self.cohort_tiles = []
        # Reset single-file view
        self._single_file_pixmap = None
        self.single_file_label.clear()
        self.single_file_label.hide()
        self.scroll_area.show()

    def show_single_image(self, pixmap, tooltip=""):
        """Show a single full-res image scaled to fill the window."""
        self._single_file_pixmap = pixmap
        self.single_file_label.setToolTip(tooltip)
        self.scroll_area.hide()
        self.single_file_label.show()
        self._scale_single_image()

    def _scale_single_image(self):
        """Scale the full-res pixmap to fit the label (keep aspect ratio)."""
        if self._single_file_pixmap is None or self._single_file_pixmap.isNull():
            return
        avail_w = self.single_file_label.width()
        avail_h = self.single_file_label.height()
        if avail_w <= 0 or avail_h <= 0:
            return
        scaled = self._single_file_pixmap.scaled(
            avail_w, avail_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.single_file_label.setPixmap(scaled)

    def resizeEvent(self, event):
        """Re-scale the single-file image when the window is resized."""
        super().resizeEvent(event)
        if self.single_file_label.isVisible():
            self._scale_single_image()

    def _load_settings(self):
        """Load split window settings from the 3D tag map settings file."""
        import json, os
        settings_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "3d_tag_map_settings.json",
        )
        try:
            if os.path.exists(settings_file):
                with open(settings_file, "r") as f:
                    settings = json.load(f)
                self.columns_spin.setValue(settings.get("split_columns", 4))
                self.max_files_spin.setValue(settings.get("split_max_files", 60))
                self.image_size_spin.setValue(settings.get("split_image_size", 200))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def _save_settings(self):
        """Save split window settings to the 3D tag map settings file."""
        import json, os
        settings_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "3d_tag_map_settings.json",
        )
        try:
            settings = {}
            if os.path.exists(settings_file):
                with open(settings_file, "r") as f:
                    settings = json.load(f)
            settings["split_columns"] = self.columns_spin.value()
            settings["split_max_files"] = self.max_files_spin.value()
            settings["split_image_size"] = self.image_size_spin.value()
            with open(settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except (OSError, TypeError):
            pass

    def closeEvent(self, event):
        """Save settings when the split window closes."""
        self._save_settings()
        super().closeEvent(event)

    def set_title(self, text):
        """Update the title bar text."""
        self.title_label.setText(text)

    def add_image(self, pixmap, tooltip=""):
        """Add a single image label to the grid using configured columns/size."""
        from PySide6.QtWidgets import QLabel
        size = self.image_size_spin.value() if hasattr(self, 'image_size_spin') else 200
        label = QLabel()
        label.setPixmap(pixmap)
        label.setFixedSize(size, size)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("border: 1px solid #4050a0;")
        if tooltip:
            label.setToolTip(tooltip)
        cols = self.columns_spin.value() if hasattr(self, 'columns_spin') else 4
        index = self.grid_layout.count()
        row = index // cols
        col = index % cols
        self.grid_layout.addWidget(label, row, col)
        return label

    def add_cohort_tile(self, cluster_id, pixmap, tooltip=""):
        """Add a clickable cohort representative tile."""
        label = self.add_image(pixmap, tooltip)
        label.setProperty("cluster_id", cluster_id)
        label.setCursor(Qt.PointingHandCursor)
        self.cohort_tiles.append((cluster_id, label))
        return label

    def mousePressEvent(self, event):
        """Handle clicks on cohort tiles to move camera to that cohort."""
        if event.button() == Qt.LeftButton:
            widget = self.childAt(event.position().toPoint())
            if widget is not None:
                cluster_id = widget.property("cluster_id")
                if cluster_id is not None and cluster_id != -1:
                    self.parent_tab._move_camera_to_cluster(cluster_id)
                    event.accept()
                    return
        super().mousePressEvent(event)

# Settings file path (relative to project root)
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "3d_tag_map_settings.json")


# Multi-line GLTextItem (camera-stable stacked labels)
_MULTILINE_TEXT_ITEM_CLASS = None


def _get_multiline_text_item_class():
    """Lazily create (and cache) a GLTextItem subclass that renders
    multi-line text stacked in SCREEN space.

    The base pyqtgraph GLTextItem renders its text via
    QPainter.drawText(QPointF, str), which treats the whole string as a
    SINGLE line -- embedded newlines are NOT rendered as line breaks (the
    lines end up concatenated, e.g. "Tag1Tag2Tag3"). To get a true stacked
    label that stays locked to the camera we override paint() to draw each
    line separately, offset in SCREEN space from the single projected world
    anchor. Because all lines share one world anchor and are offset in
    screen space, the stack does not drift as the camera moves (unlike
    offsetting each line in world space).
    """
    global _MULTILINE_TEXT_ITEM_CLASS
    if _MULTILINE_TEXT_ITEM_CLASS is not None:
        return _MULTILINE_TEXT_ITEM_CLASS

    import pyqtgraph.opengl as gl
    from PySide6.QtGui import QFontMetrics, QVector3D, QPainter
    from PySide6.QtCore import QPointF, Qt as _Qt

    class _MultiLineGLTextItem(gl.GLTextItem):
        def paint(self):
            if len(self.text) < 1:
                return
            self.setupGLState()
            project = self.compute_projection()
            anchor = project.map(QVector3D(*self.pos)).toPointF()

            painter = QPainter(self.view())
            painter.setPen(self.color)
            painter.setFont(self.font)
            painter.setRenderHints(
                QPainter.RenderHint.Antialiasing
                | QPainter.RenderHint.TextAntialiasing
            )

            fm = QFontMetrics(self.font)
            line_height = fm.lineSpacing()
            lines = self.text.split("\n")
            align = self.alignment
            n = len(lines)

            for i, line in enumerate(lines):
                if not line:
                    continue
                # Horizontal alignment (center each line on the anchor)
                dx = 0.0
                if align & _Qt.AlignmentFlag.AlignHCenter:
                    dx = fm.horizontalAdvance(line) / 2.0
                elif align & _Qt.AlignmentFlag.AlignRight:
                    dx = fm.horizontalAdvance(line)
                # Vertical offset for this line (screen space)
                dy = i * line_height
                if align & _Qt.AlignmentFlag.AlignVCenter:
                    dy -= (n - 1) * line_height / 2.0
                elif align & _Qt.AlignmentFlag.AlignTop:
                    dy -= (n - 1) * line_height
                painter.drawText(QPointF(anchor.x() - dx, anchor.y() + dy), line)
            painter.end()

    _MULTILINE_TEXT_ITEM_CLASS = _MultiLineGLTextItem
    return _MULTILINE_TEXT_ITEM_CLASS


class WorkerThread(QThread):
    """Background worker thread for data processing."""
    progress = Signal(int, str)  # percentage, message
    finished = Signal(object)  # result
    error = Signal(str)  # error message

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class SplitWindowLoader(QThread):
    """Background worker for loading thumbnails into the split window."""
    pixmap_ready = Signal(object, str)  # pixmap, tooltip
    finished = Signal()

    def __init__(self, client_name, file_ids, image_size, parent=None):
        super().__init__(parent)
        self.client_name = client_name
        self.file_ids = file_ids
        self.image_size = image_size

    def run(self):
        try:
            from src.utils.utility_functions import ConnectToClient
            from src.utils.image_loader import load_pixmap_with_lanczos
            client = ConnectToClient(self.client_name)
            for file_id in self.file_ids:
                try:
                    response = client.get_thumbnail(file_id=file_id)
                    if response and hasattr(response, 'content'):
                        pixmap = load_pixmap_with_lanczos(response.content, max_size=self.image_size)
                        if pixmap:
                            self.pixmap_ready.emit(pixmap, f"File {file_id}")
                except Exception as e:
                    print(f"Error loading file {file_id}: {e}")
        except Exception as e:
            print(f"Error loading thumbnails: {e}")
        self.finished.emit()


class SingleFileLoader(QThread):
    """Background worker for loading a single full-res file from Hydrus.

    Uses client.get_file() (full resolution) instead of get_thumbnail().
    The in-memory pixmap is capped at 4096px on the longest side to keep
    memory bounded; the split window scales it to fit for display.
    """
    pixmap_ready = Signal(object, str)  # pixmap, tooltip
    finished = Signal()

    MAX_PIXELS = 4096

    def __init__(self, client_name, file_id, parent=None):
        super().__init__(parent)
        self.client_name = client_name
        self.file_id = file_id

    def run(self):
        try:
            from PySide6.QtGui import QPixmap
            from src.utils.utility_functions import ConnectToClient
            client = ConnectToClient(self.client_name)
            response = client.get_file(file_id=self.file_id)
            if response and hasattr(response, 'content'):
                pixmap = QPixmap()
                if pixmap.loadFromData(response.content):
                    # Cap in-memory size (longest side) to bound memory
                    if max(pixmap.width(), pixmap.height()) > self.MAX_PIXELS:
                        pixmap = pixmap.scaled(
                            self.MAX_PIXELS, self.MAX_PIXELS,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )
                    self.pixmap_ready.emit(pixmap, f"File {self.file_id}")
        except Exception as e:
            print(f"Error loading full-res file {self.file_id}: {e}")
        self.finished.emit()


class TagMap3DTab(QWidget):
    """Main tab widget for 3D tag space visualization."""

    def __init__(self, main_window):
        """Initialize the 3D tag map tab.

        Args:
            main_window: The main window instance
        """
        super().__init__(main_window)
        self.main_window = main_window
        self.scene_graph = None
        self.client = None
        self.tag_data = None  # Store tag data for recomputation

        # Advanced settings (managed by the settings dialog)
        self.low_memory = False
        self.n_jobs = os.cpu_count() or 4
        self.use_direct_db = False
        self.client_db_paths = {}

        # DBSCAN optimizer settings (managed by the settings dialog)
        self.opt_max_cohort_size = 500
        self.opt_max_noise_ratio = 10
        self.opt_max_attempts = 60
        self.opt_eps_min = 5
        self.opt_eps_max = 100
        self.opt_min_samples_min = 2
        self.opt_min_samples_max = 30
        # Normalize positions before DBSCAN (global + split clustering)
        self.normalize_positions = False

        # Optional external tag-score DB path (empty = scoring disabled)
        self.score_db_path = ""

        self.setup_ui()
        self.load_settings()
        self._load_client_db_path()
        self._connect_settings_signals()
        
        # Selection tracking for visual indicator
        self.selected_node_index = None
        self.selected_cluster_id = None
        self.selection_timer = QTimer(self)  # Give parent for event loop
        self.selection_timer.timeout.connect(self._toggle_selection_highlight)
        self.selection_visible = False
        self.original_colors = None  # Store original colors for restoration

        # Cohort label blink timer
        self.cohort_label_blink_timer = QTimer(self)
        self.cohort_label_blink_timer.timeout.connect(self._toggle_cohort_label_blink)
        self.cohort_label_blink_visible = True

        # Cluster hull meshes
        self.cluster_hull_meshes = []  # List of GLMeshItem for cluster boundaries

        # Cohort labels
        self.cohort_label_items = []  # List of GLTextItem for cohort labels
        self.cohort_label_map = {}  # cluster_id -> GLTextItem (for targeted blink)

        # V7: Relationship lines between related clusters
        self.relationship_line_items = []  # List of GLLinePlotItem
        self.relationship_label_items = []  # List of GLTextItem
        self.relationship_pairs = []  # List of (cid1, cid2, score, shared_tags)

        # Auto-load last data
        self.auto_load_timer = QTimer(self)
        self.auto_load_timer.setSingleShot(True)
        self.auto_load_timer.timeout.connect(self._auto_load_last_data)
        # Start the timer so auto-load actually fires after the window is shown.
        # The handler (_auto_load_last_data) is a no-op unless the checkbox is
        # enabled and no data is loaded yet, so this is safe to always start.
        self.auto_load_timer.start(1000)

        # Time Travel animation
        self.time_travel_timer = QTimer(self)
        self.time_travel_timer.timeout.connect(self._update_time_travel)
        self.time_travel_active = False
        self.time_travel_waypoints = []  # List of (position, cluster_name) tuples
        self.time_travel_current_index = 0
        self.time_travel_t = 0.0
        self.time_travel_segment_duration = 120  # frames per segment (2s at 60fps)
        self.time_travel_dwell_duration = 120  # frames to dwell at each waypoint
        self.time_travel_frames = 0
        self.time_travel_mode = "dwell"  # "dwell", "travel", or "orbit"
        # Orbit parameters (spaceship circling a cluster)
        self.time_travel_orbit_radius = 8.0  # distance from cluster centroid
        self.time_travel_orbit_speed = 0.02  # radians per frame
        self.time_travel_orbit_angle = 0.0
        self.time_travel_orbit_duration = 180  # frames to orbit (~3s at 60fps)
        self.time_travel_orbit_center = None  # current orbit center
        # Travel curve parameters (spaceship banking turn)
        self.time_travel_arc_height = 6.0  # how high the arc rises between clusters

        # Tag query state tracking for clickable tags
        self.tag_query_states = {}  # tag_name -> 0 (neutral), 1 (included), 2 (excluded)
        self._tag_widgets = []  # Store references to prevent garbage collection

        # Split window for image preview sync
        self.split_window = None

        # Last-used DBSCAN params for recluster grey-out logic
        self._last_used_eps = None
        self._last_used_min_samples = None
        # Flag: next on_loading_finished is a re-cluster (positions unchanged)
        self._pending_recluster = False

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
            self.chunk_size_spin.setValue(settings.get("chunk_size", 500))
            self.max_files_spin.setValue(settings.get("max_files", 4096))
            tag_service_idx = self.tag_service_combo.findText(settings.get("tag_service", "auto2"))
            if tag_service_idx >= 0:
                self.tag_service_combo.setCurrentIndex(tag_service_idx)

            # Direct DB mode toggle
            self.use_direct_db = settings.get("use_direct_db", False)

            # Auto-load last data setting
            if hasattr(self, 'auto_load_checkbox'):
                self.auto_load_checkbox.setChecked(settings.get("auto_load_last_data", False))

            # Algorithm settings
            algo_idx = self.algorithm_combo.findText(settings.get("algorithm", "UMAP"))
            if algo_idx >= 0:
                self.algorithm_combo.setCurrentIndex(algo_idx)
            self.n_neighbors_spin.setValue(settings.get("n_neighbors", 15))
            self.min_dist_spin.setValue(settings.get("min_dist", 50))
            self.n_epochs_spin.setValue(settings.get("n_epochs", 0))
            self.learning_rate_spin.setValue(settings.get("learning_rate", 1.0))
            metric_idx = self.metric_combo.findText(settings.get("metric", "cosine"))
            if metric_idx >= 0:
                self.metric_combo.setCurrentIndex(metric_idx)

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
            self.normalize_positions = settings.get("normalize_positions", False)
            # Sync the inline checkbox so the UI reflects the saved value
            self.normalize_checkbox.setChecked(self.normalize_positions)
            # Optional tag-score DB path
            self.score_db_path = settings.get("score_db_path", "")
            # Sub-clustering settings
            self.sub_eps_spin.setValue(settings.get("sub_eps", 20))
            self.sub_min_samples_spin.setValue(settings.get("sub_min_samples", 4))

            # Filter settings
            self.query_edit.setText(settings.get("query", ""))
            self.whitelist_edit.setText(settings.get("whitelist", ""))
            self.blacklist_edit.setText(settings.get("blacklist", ""))
            self.tokenize_checkbox.setChecked(settings.get("tokenize", False))
            self.drop_empty_checkbox.setChecked(settings.get("drop_empty_files", False))
            self.min_doc_freq_spin.setValue(settings.get("min_doc_freq", 3))
            self.drop_universal_checkbox.setChecked(settings.get("drop_universal_tags", False))

            # Visualization settings
            self.min_size_spin.setValue(settings.get("min_size", 0.03))
            self.max_size_spin.setValue(settings.get("max_size", 0.08))
            self.spread_spin.setValue(settings.get("spread", 1.0))
            self.orbit_speed_spin.setValue(settings.get("orbit_speed", 0.2))
            self.transparency_spin.setValue(settings.get("transparency", 0.8))

            # Anti-noise / quality settings
            self.msaa_checkbox.setChecked(settings.get("msaa", True))
            self.point_smooth_checkbox.setChecked(settings.get("point_smooth", True))
            self.supersample_checkbox.setChecked(settings.get("supersample", False))
            self.supersample_fps_spin.setValue(settings.get("supersample_fps", 10))

            # Dim non-selected nodes settings
            self.dim_non_selected_checkbox.setChecked(settings.get("dim_non_selected", True))
            self.dim_alpha_spin.setValue(settings.get("dim_alpha", 0.15))

            # V5: Color scheme setting
            color_scheme_idx = self.color_scheme_combo.findText(settings.get("color_scheme", "Pastel"))
            if color_scheme_idx >= 0:
                self.color_scheme_combo.setCurrentIndex(color_scheme_idx)
            
            # V3: Cluster boundaries setting
            self.show_boundaries_checkbox.setChecked(settings.get("show_cluster_boundaries", False))
            self.boundary_alpha_spin.setValue(settings.get("boundary_alpha", 0.15))
            
            # V7: Relationship lines setting
            # (backward compat: old "show_particles" key maps to the new toggle)
            self.show_relationships_checkbox.setChecked(
                settings.get("show_relationships", settings.get("show_particles", False))
            )
            metric_idx = self.relationship_metric_combo.findText(
                settings.get("relationship_metric", "IDF Cosine")
            )
            if metric_idx >= 0:
                self.relationship_metric_combo.setCurrentIndex(metric_idx)
            self.relationship_min_sim_spin.setValue(settings.get("relationship_min_sim", 0.1))
            self.relationship_max_spin.setValue(settings.get("relationship_max", 30))
            self.relationship_alpha_spin.setValue(settings.get("relationship_alpha", 0.6))
            self.relationship_width_spin.setValue(settings.get("relationship_width", 2))
            rel_color = settings.get("relationship_color", [120, 160, 255])
            self._relationship_color = tuple(rel_color)
            self._update_relationship_color_button()
            self.show_relationship_labels_checkbox.setChecked(
                settings.get("show_relationship_labels", False)
            )

            # Cohort label settings
            self.cohort_threshold_spin.setValue(settings.get("cohort_threshold", 0.9))
            self.show_cohort_labels_checkbox.setChecked(settings.get("show_cohort_labels", False))
            self.cohort_label_size_spin.setValue(settings.get("cohort_label_size", 14))
            self.dynamic_label_size_checkbox.setChecked(settings.get("dynamic_label_size", False))
            self.invert_label_color_checkbox.setChecked(settings.get("invert_label_color", False))
            color = settings.get("cohort_label_color", [255, 255, 255])
            self._cohort_label_color = tuple(color)
            self._update_label_color_button()
            color2 = settings.get("cohort_label_color2", [255, 200, 0])
            self._cohort_label_color2 = tuple(color2)
            self._update_label_color_button2()
            # Label mode + N
            mode_idx = self.cohort_label_mode_combo.findText(settings.get("cohort_label_mode", "Top N largest"))
            if mode_idx >= 0:
                self.cohort_label_mode_combo.setCurrentIndex(mode_idx)
            self.cohort_label_n_spin.setValue(settings.get("cohort_label_n", 10))
            self.cohort_label_max_tags_spin.setValue(settings.get("cohort_label_max_tags", 5))
            # Smart labels settings
            self.smart_labels_checkbox.setChecked(settings.get("smart_labels", False))
            smart_mode_idx = self.smart_label_mode_combo.findText(
                settings.get("smart_label_mode", "All Unique")
            )
            if smart_mode_idx >= 0:
                self.smart_label_mode_combo.setCurrentIndex(smart_mode_idx)

            # Split window settings (image preview)
            if hasattr(self, 'split_window') and self.split_window:
                self.split_window.columns_spin.setValue(settings.get("split_columns", 4))
                self.split_window.max_files_spin.setValue(settings.get("split_max_files", 60))
                self.split_window.image_size_spin.setValue(settings.get("split_image_size", 200))
            
            # Camera wobble settings
            self.wobble_enabled_checkbox.setChecked(settings.get("wobble_enabled", False))
            self.wobble_speed_spin.setValue(settings.get("wobble_speed", 1.0))
            self.wobble_x_range_spin.setValue(settings.get("wobble_x_range", 10.0))
            self.wobble_y_range_spin.setValue(settings.get("wobble_y_range", 10.0))
            self.wobble_z_range_spin.setValue(settings.get("wobble_z_range", 20.0))
            self.wobble_azim_range_spin.setValue(settings.get("wobble_azim_range", 15.0))
            self.wobble_elev_range_spin.setValue(settings.get("wobble_elev_range", 10.0))

            # Trail settings
            self.trail_enabled_checkbox.setChecked(settings.get("trail_enabled", False))
            self.trail_length_spin.setValue(settings.get("trail_length", 12))
            self.trail_decay_spin.setValue(settings.get("trail_decay", 0.85))
            self.trail_max_nodes_spin.setValue(settings.get("trail_max_nodes", 3000))
            
            # Send to tab settings
            if hasattr(self, 'tab_name_edit'):
                self.tab_name_edit.setText(settings.get("tab_name", ""))
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
            'drop_universal_checkbox': 'stateChanged',
            'min_size_spin': 'valueChanged',
            'max_size_spin': 'valueChanged',
            'spread_spin': 'valueChanged',
            'orbit_speed_spin': 'valueChanged',
            'transparency_spin': 'valueChanged',
            'supersample_fps_spin': 'valueChanged',
            'dim_alpha_spin': 'valueChanged',
            'boundary_alpha_spin': 'valueChanged',
            'relationship_min_sim_spin': 'valueChanged',
            'relationship_max_spin': 'valueChanged',
            'relationship_alpha_spin': 'valueChanged',
            'relationship_width_spin': 'valueChanged',
            'cohort_threshold_spin': 'valueChanged',
            'cohort_label_size_spin': 'valueChanged',
            'cohort_label_n_spin': 'valueChanged',
            'cohort_label_max_tags_spin': 'valueChanged',
            'wobble_speed_spin': 'valueChanged',
            'wobble_x_range_spin': 'valueChanged',
            'wobble_y_range_spin': 'valueChanged',
            'wobble_z_range_spin': 'valueChanged',
            'wobble_azim_range_spin': 'valueChanged',
            'wobble_elev_range_spin': 'valueChanged',
            'trail_length_spin': 'valueChanged',
            'trail_decay_spin': 'valueChanged',
            'trail_max_nodes_spin': 'valueChanged',
            # QCheckBox -> stateChanged
            'auto_load_checkbox': 'stateChanged',
            'normalize_checkbox': 'stateChanged',
            'tokenize_checkbox': 'stateChanged',
            'drop_empty_checkbox': 'stateChanged',
            'msaa_checkbox': 'stateChanged',
            'point_smooth_checkbox': 'stateChanged',
            'supersample_checkbox': 'stateChanged',
            'dim_non_selected_checkbox': 'stateChanged',
            'show_boundaries_checkbox': 'stateChanged',
            'show_relationships_checkbox': 'stateChanged',
            'show_relationship_labels_checkbox': 'stateChanged',
            'show_cohort_labels_checkbox': 'stateChanged',
            'dynamic_label_size_checkbox': 'stateChanged',
            'invert_label_color_checkbox': 'stateChanged',
            'smart_labels_checkbox': 'stateChanged',
            'wobble_enabled_checkbox': 'stateChanged',
            'trail_enabled_checkbox': 'stateChanged',
            # QComboBox -> currentTextChanged
            'client_combo': 'currentTextChanged',
            'tag_service_combo': 'currentTextChanged',
            'algorithm_combo': 'currentTextChanged',
            'metric_combo': 'currentTextChanged',
            'color_scheme_combo': 'currentTextChanged',
            'relationship_metric_combo': 'currentTextChanged',
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
            # Read existing settings to preserve values for widgets that may not exist
            # (e.g., split window closed -> self.split_window is None)
            existing = {}
            if os.path.exists(SETTINGS_FILE):
                try:
                    with open(SETTINGS_FILE, 'r') as f:
                        existing = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            settings = {
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
                "normalize_positions": getattr(self, 'normalize_positions', False),
                "score_db_path": getattr(self, 'score_db_path', ''),
                "query": self.query_edit.text(),
                "whitelist": self.whitelist_edit.text(),
                "blacklist": self.blacklist_edit.text(),
                "tokenize": self.tokenize_checkbox.isChecked(),
                "drop_empty_files": self.drop_empty_checkbox.isChecked(),
                "min_doc_freq": self.min_doc_freq_spin.value(),
                "drop_universal_tags": self.drop_universal_checkbox.isChecked(),
                "min_size": self.min_size_spin.value(),
                "max_size": self.max_size_spin.value(),
                "spread": self.spread_spin.value(),
                "orbit_speed": self.orbit_speed_spin.value(),
                "transparency": self.transparency_spin.value(),
                # Anti-noise / quality settings
                "msaa": self.msaa_checkbox.isChecked(),
                "point_smooth": self.point_smooth_checkbox.isChecked(),
                "supersample": self.supersample_checkbox.isChecked(),
                "supersample_fps": self.supersample_fps_spin.value(),
                # Dim non-selected nodes settings
                "dim_non_selected": self.dim_non_selected_checkbox.isChecked(),
                "dim_alpha": self.dim_alpha_spin.value(),
                # V5: Color scheme
                "color_scheme": self.color_scheme_combo.currentText(),
                # V3: Cluster boundaries
                "show_cluster_boundaries": self.show_boundaries_checkbox.isChecked(),
                "boundary_alpha": self.boundary_alpha_spin.value(),
                # V7: Relationship lines
                "show_relationships": self.show_relationships_checkbox.isChecked(),
                "relationship_metric": self.relationship_metric_combo.currentText(),
                "relationship_min_sim": self.relationship_min_sim_spin.value(),
                "relationship_max": self.relationship_max_spin.value(),
                "relationship_alpha": self.relationship_alpha_spin.value(),
                "relationship_width": self.relationship_width_spin.value(),
                "relationship_color": list(self._relationship_color),
                "show_relationship_labels": self.show_relationship_labels_checkbox.isChecked(),
                # Cohort label settings
                "cohort_threshold": self.cohort_threshold_spin.value(),
                "show_cohort_labels": self.show_cohort_labels_checkbox.isChecked(),
                "cohort_label_size": self.cohort_label_size_spin.value(),
                "dynamic_label_size": self.dynamic_label_size_checkbox.isChecked(),
                "invert_label_color": self.invert_label_color_checkbox.isChecked(),
                "cohort_label_color": list(self._cohort_label_color),
                "cohort_label_color2": list(self._cohort_label_color2),
                "cohort_label_mode": self.cohort_label_mode_combo.currentText(),
                "cohort_label_n": self.cohort_label_n_spin.value(),
                "cohort_label_max_tags": self.cohort_label_max_tags_spin.value(),
                # Smart labels settings
                "smart_labels": self.smart_labels_checkbox.isChecked(),
                "smart_label_mode": self.smart_label_mode_combo.currentText(),
                # Split window settings (image preview)
                # Fall back to previously saved values when the split window is closed
                "split_columns": self.split_window.columns_spin.value() if hasattr(self, 'split_window') and self.split_window else existing.get("split_columns", 4),
                "split_max_files": self.split_window.max_files_spin.value() if hasattr(self, 'split_window') and self.split_window else existing.get("split_max_files", 60),
                "split_image_size": self.split_window.image_size_spin.value() if hasattr(self, 'split_window') and self.split_window else existing.get("split_image_size", 200),
                # Camera wobble settings
                "wobble_enabled": self.wobble_enabled_checkbox.isChecked(),
                "wobble_speed": self.wobble_speed_spin.value(),
                "wobble_x_range": self.wobble_x_range_spin.value(),
                "wobble_y_range": self.wobble_y_range_spin.value(),
                "wobble_z_range": self.wobble_z_range_spin.value(),
                "wobble_azim_range": self.wobble_azim_range_spin.value(),
                "wobble_elev_range": self.wobble_elev_range_spin.value(),
                # Trail settings
                "trail_enabled": self.trail_enabled_checkbox.isChecked(),
                "trail_length": self.trail_length_spin.value(),
                "trail_decay": self.trail_decay_spin.value(),
                "trail_max_nodes": self.trail_max_nodes_spin.value(),
                # Send to tab settings
                "tab_name": self.tab_name_edit.text() if hasattr(self, 'tab_name_edit') else "",
                # Auto-load last data setting
                "auto_load_last_data": self.auto_load_checkbox.isChecked() if hasattr(self, 'auto_load_checkbox') else False,
            }
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

    def open_settings_dialog(self):
        """Open the advanced settings window for the 3D tag map tab."""
        from src.ui.settings_dialog import TagMap3DSettingsDialog
        dialog = TagMap3DSettingsDialog(self)
        dialog.exec()

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event to save settings."""
        self._save_client_db_path()
        self.save_settings()
        super().closeEvent(event)

    def setup_ui(self):
        """Create the user interface."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Create main horizontal splitter: left panel | center (3D + info) | right sidebar
        main_splitter = QSplitter(Qt.Horizontal)

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
        client_layout.addRow("Client:", self.client_combo)

        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(50, 100000000)
        self.chunk_size_spin.setValue(500)
        self.chunk_size_spin.setToolTip("Number of files to fetch per API request.\nLarger = faster but may timeout.\nSmaller = more requests but more reliable.\nDefault: 500")
        client_layout.addRow("Chunk Size:", self.chunk_size_spin)

        self.max_files_spin = QSpinBox()
        self.max_files_spin.setRange(1, 100000000)
        self.max_files_spin.setValue(4096)
        self.max_files_spin.setSingleStep(512)
        self.max_files_spin.setToolTip("Maximum number of files to load and analyze.\nLower = faster processing, less memory.\nHigher = more complete visualization.\nDefault: 4096")
        client_layout.addRow("Max Files:", self.max_files_spin)

        self.tag_service_combo = QComboBox()
        self.tag_service_combo.addItems(["auto2", "local", "all known tags"])
        self.tag_service_combo.setToolTip("Which tag service to use for fetching tags.\nauto2 = Automatic tag detection (recommended).\nlocal = Only locally stored tags.\nall known tags = Include all tags from Hydrus tag database.")
        client_layout.addRow("Tag Service:", self.tag_service_combo)

        # Auto-load last data checkbox
        self.auto_load_checkbox = QCheckBox("Auto-load last data")
        self.auto_load_checkbox.setChecked(False)
        self.auto_load_checkbox.setToolTip("When enabled, automatically load the last used\ndata on next launch with saved parameters.")
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
        self.min_dist_spin.setValue(50)
        self.min_dist_spin.setSuffix("%")
        self.min_dist_spin.setToolTip("UMAP parameter: Minimum distance between points (0-100%).\nLower (0-20%) = points packed tightly together.\nHigher (50-100%) = more spread out, easier to see individual points.\nDefault: 50%")
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

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # Cluster Settings Group
        cluster_group = QGroupBox("Cluster Settings")
        cluster_group.setToolTip("DBSCAN clustering parameters for grouping similar files.")
        cluster_layout = QFormLayout()

        self.eps_spin = QSpinBox()
        self.eps_spin.setRange(1, 200)
        self.eps_spin.setValue(50)
        self.eps_spin.setSingleStep(5)
        self.eps_spin.setToolTip("DBSCAN parameter: Maximum distance between points in same cluster (as % of data spread).\nLower (10-30%) = many small, tight clusters.\nHigher (100-200%) = few large clusters.\nDefault: 50%")
        cluster_layout.addRow("EPS (%):", self.eps_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(2, 100)
        self.min_samples_spin.setValue(10)
        self.min_samples_spin.setToolTip("DBSCAN parameter: Minimum points to form a cluster.\nLower (2-5) = more clusters, even small groups count.\nHigher (20-50) = only large dense groups become clusters.\nDefault: 10")
        cluster_layout.addRow("Min Samples:", self.min_samples_spin)

        # Normalize positions before DBSCAN toggle (global + split clustering)
        self.normalize_checkbox = QCheckBox("Normalize positions before DBSCAN")
        self.normalize_checkbox.setChecked(getattr(self, 'normalize_positions', False))
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

        # Recluster button grey-out: unlock only when target params differ from last-used
        self.eps_spin.valueChanged.connect(self._update_recluster_button_state)
        self.min_samples_spin.valueChanged.connect(self._update_recluster_button_state)

        cluster_group.setLayout(cluster_layout)
        layout.addWidget(cluster_group)

        # Filter Settings Group
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

        self.tokenize_checkbox = QCheckBox("Tokenize tags")
        self.tokenize_checkbox.setChecked(False)
        self.tokenize_checkbox.setToolTip("When enabled, tags are converted to integer indices once at load\ntime and carried through the pipeline as integers. This reduces RAM\n(no per-node string copies) and replaces repeated string hashing with\ninteger lookups. Strings are only materialised for display.\nDefault: OFF (proven string path).")
        filter_layout.addRow("Tokenize:", self.tokenize_checkbox)

        self.drop_empty_checkbox = QCheckBox("Drop empty files")
        self.drop_empty_checkbox.setChecked(False)
        self.drop_empty_checkbox.setToolTip("When enabled, files with no tags remaining after the\nwhitelist/blacklist filters are excluded from the map.\nWhen disabled (default), they are kept and appear as untagged\nnodes at the origin.\nOnly applies when a whitelist or blacklist is set.")
        filter_layout.addRow(self.drop_empty_checkbox)

        self.min_doc_freq_spin = QSpinBox()
        self.min_doc_freq_spin.setRange(1, 100)
        self.min_doc_freq_spin.setValue(3)
        self.min_doc_freq_spin.setToolTip("Vectorizer: Minimum documents a tag must appear in\nto be included in the vocabulary.\nHigher = fewer rare tags, faster UMAP.\nLower = more tags, slower but more detailed.\nDefault: 3")
        filter_layout.addRow("Min Doc Freq:", self.min_doc_freq_spin)

        self.drop_universal_checkbox = QCheckBox("Drop Universal Tags")
        self.drop_universal_checkbox.setChecked(False)
        self.drop_universal_checkbox.setToolTip("Exclude tags that appear in EVERY loaded file from the vocabulary.\nThese tags provide zero discriminative power (they are usually\nalready visible in your query field) and only add noise dimensions.\nUseful for large AND queries where all files share the same tags.\nDefault: OFF")
        filter_layout.addRow(self.drop_universal_checkbox)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Camera Wobble Group (for depth perception)
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
            "Spin the camera continuously (azimuth + elevation rotate in one "
            "direction and wrap around) instead of oscillating back and forth.\n"
            "The Azim/Elev Range values become the spin speed (degrees per second).\n"
            "Default: OFF (classic sine wobble)."
        )
        wobble_layout.addRow("Continuous Spin:", self.wobble_continuous_checkbox)

        self.wobble_speed_spin = QDoubleSpinBox()
        self.wobble_speed_spin.setRange(0.1, 10.0)
        self.wobble_speed_spin.setValue(1.0)
        self.wobble_speed_spin.setDecimals(2)
        self.wobble_speed_spin.setSingleStep(0.1)
        self.wobble_speed_spin.setToolTip("Speed of the wobble oscillation.\nHigher = faster movement.\nDefault: 1.0")
        wobble_layout.addRow("Speed:", self.wobble_speed_spin)

        # X pan range
        self.wobble_x_range_spin = QDoubleSpinBox()
        self.wobble_x_range_spin.setRange(0.0, 100.0)
        self.wobble_x_range_spin.setValue(10.0)
        self.wobble_x_range_spin.setDecimals(1)
        self.wobble_x_range_spin.setSingleStep(1.0)
        self.wobble_x_range_spin.setToolTip("X-axis pan range (left-right movement).\nDefault: 10.0")
        wobble_layout.addRow("X Range:", self.wobble_x_range_spin)

        # Y pan range
        self.wobble_y_range_spin = QDoubleSpinBox()
        self.wobble_y_range_spin.setRange(0.0, 100.0)
        self.wobble_y_range_spin.setValue(10.0)
        self.wobble_y_range_spin.setDecimals(1)
        self.wobble_y_range_spin.setSingleStep(1.0)
        self.wobble_y_range_spin.setToolTip("Y-axis pan range (up-down movement).\nDefault: 10.0")
        wobble_layout.addRow("Y Range:", self.wobble_y_range_spin)

        # Z distance range
        self.wobble_z_range_spin = QDoubleSpinBox()
        self.wobble_z_range_spin.setRange(0.0, 200.0)
        self.wobble_z_range_spin.setValue(20.0)
        self.wobble_z_range_spin.setDecimals(1)
        self.wobble_z_range_spin.setSingleStep(5.0)
        self.wobble_z_range_spin.setToolTip("Z-axis (distance) range (zoom in-out movement).\nDefault: 20.0")
        wobble_layout.addRow("Z Range:", self.wobble_z_range_spin)

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

        # Trail settings (visual trails behind nodes during wobble)
        self.trail_enabled_checkbox = QCheckBox()
        self.trail_enabled_checkbox.setChecked(False)
        self.trail_enabled_checkbox.setToolTip("Draw fading trails behind nodes during camera wobble.\nDefault: OFF (large datasets are expensive).")
        self.trail_enabled_checkbox.stateChanged.connect(self._on_trail_toggle)
        wobble_layout.addRow("Trails:", self.trail_enabled_checkbox)

        self.trail_length_spin = QSpinBox()
        self.trail_length_spin.setRange(2, 60)
        self.trail_length_spin.setValue(12)
        self.trail_length_spin.setToolTip("Trail length: number of recent positions kept per node.\nHigher = longer trails, more memory.\nDefault: 12")
        self.trail_length_spin.setEnabled(False)
        wobble_layout.addRow("Trail Length:", self.trail_length_spin)

        self.trail_decay_spin = QDoubleSpinBox()
        self.trail_decay_spin.setRange(0.0, 1.0)
        self.trail_decay_spin.setValue(0.85)
        self.trail_decay_spin.setDecimals(2)
        self.trail_decay_spin.setSingleStep(0.05)
        self.trail_decay_spin.setToolTip("Trail decay: how quickly trail points fade (0 = instant, 1 = long-lasting).\nDefault: 0.85")
        self.trail_decay_spin.setEnabled(False)
        wobble_layout.addRow("Trail Decay:", self.trail_decay_spin)

        self.trail_max_nodes_spin = QSpinBox()
        self.trail_max_nodes_spin.setRange(100, 50000)
        self.trail_max_nodes_spin.setValue(3000)
        self.trail_max_nodes_spin.setSingleStep(500)
        self.trail_max_nodes_spin.setToolTip("Max nodes that render trails (performance guard).\nLower = faster. Default: 3000")
        self.trail_max_nodes_spin.setEnabled(False)
        wobble_layout.addRow("Trail Max Nodes:", self.trail_max_nodes_spin)

        wobble_group.setLayout(wobble_layout)
        layout.addWidget(wobble_group)

        # Initialize wobble timer and state
        self.wobble_timer = QTimer()
        self.wobble_timer.timeout.connect(self._update_wobble)
        self.wobble_timer.setInterval(16)  # ~60fps
        self.wobble_time = 0.0

        # Trail state (visual trails behind nodes during wobble)
        self.trail_scatter = None  # GLScatterPlotItem for trails
        self.trail_history = []  # list of recent position arrays (ring buffer)
        self.trail_active = False

        # Load Button and Progress
        self.load_button = QPushButton("Load Data and Compute")
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
        self.load_button.clicked.connect(self.start_loading)
        layout.addWidget(self.load_button)

        # Load Last Data button (loads with saved params)
        self.load_last_button = QPushButton("Load Last Data")
        self.load_last_button.setStyleSheet(f"""
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
        self.load_last_button.setToolTip("Load data using the last saved parameters.")
        self.load_last_button.clicked.connect(self._load_last_data)
        layout.addWidget(self.load_last_button)

        self.recompute_button = QPushButton("Recompute (Use Current Data)")
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
        self.recompute_button.setToolTip("Re-run UMAP/PCA and DBSCAN with new settings using currently loaded data.")
        layout.addWidget(self.recompute_button)

        # Reapply DBSCAN button (re-cluster only, no re-reduce)
        self.recluster_button = QPushButton("Reapply DBSCAN")
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
        self.recluster_button.setToolTip("Re-run DBSCAN only on current positions (no UMAP/PCA re-run). Use after changing eps/min_samples.")
        layout.addWidget(self.recluster_button)

        # Optimize DBSCAN button (auto-search ideal eps/min_samples)
        self.optimize_button = QPushButton("Optimize DBSCAN")
        self.optimize_button.setStyleSheet(f"""
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
        self.optimize_button.setEnabled(False)
        self.optimize_button.clicked.connect(self.start_optimize)
        self.optimize_button.setToolTip(
            "Automatically search for the ideal eps/min_samples combination.\n"
            "Goal: reduce non-cohorted (noise) nodes and split disproportionately\n"
            "large cohorts. Runs DBSCAN multiple times to find the best settings."
        )
        layout.addWidget(self.optimize_button)

        # Save Session button
        self.save_session_button = QPushButton("Save Session")
        self.save_session_button.setStyleSheet(f"""
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
        self.save_session_button.setEnabled(False)
        self.save_session_button.clicked.connect(self._save_session)
        self.save_session_button.setToolTip("Save the current rendered session (positions, clusters, tags) to skip reprocessing on next load.")
        layout.addWidget(self.save_session_button)

        # Load Session button
        self.load_session_button = QPushButton("Load Session")
        self.load_session_button.setStyleSheet(f"""
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
        self.load_session_button.setEnabled(True)  # Available at startup to load a session
        self.load_session_button.clicked.connect(self._load_session)
        self.load_session_button.setToolTip("Load a saved session directly (skips query, metadata, UMAP, DBSCAN).")
        layout.addWidget(self.load_session_button)

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

            # Apply MSAA antialiasing if enabled (must be set before GL context creation)
            if self.msaa_checkbox.isChecked() if hasattr(self, 'msaa_checkbox') else True:
                try:
                    from PySide6.QtGui import QSurfaceFormat
                    fmt = QSurfaceFormat.defaultFormat()
                    fmt.setSamples(8)  # 8x multisample antialiasing
                    QSurfaceFormat.setDefaultFormat(fmt)
                except Exception as e:
                    print(f"MSAA setup failed: {e}")

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

                def paintGL(self, *args, **kwargs):
                    """Enable point smoothing for softer dots before rendering."""
                    try:
                        import OpenGL.GL as gl
                        if self.parent_tab.point_smooth_checkbox.isChecked() if hasattr(self.parent_tab, 'point_smooth_checkbox') else True:
                            gl.glEnable(gl.GL_POINT_SMOOTH)
                        else:
                            gl.glDisable(gl.GL_POINT_SMOOTH)
                    except Exception:
                        pass
                    return super().paintGL(*args, **kwargs)

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
                    """Handle right-click by picking the node under cursor."""
                    scatter = self.parent_tab.gl_scatter
                    if scatter is None:
                        return
                    
                    try:
                        import numpy as np
                        from PySide6.QtGui import QMatrix4x4, QVector4D
                        
                        click_pos = event.pos()
                        
                        if scatter.pos is None or len(scatter.pos) == 0:
                            return
                        
                        positions = scatter.pos if isinstance(scatter.pos, np.ndarray) else np.array(scatter.pos)
                        
                        view_matrix = self.viewMatrix()
                        proj_matrix = self.currentProjection()
                        mvp_matrix = proj_matrix * view_matrix
                        
                        width = self.width()
                        height = self.height()
                        
                        closest_idx = None
                        closest_dist = float('inf')
                        pick_threshold = 20.0
                        
                        for i in range(len(positions)):
                            pos = positions[i]
                            vec = QVector4D(pos[0], pos[1], pos[2], 1.0)
                            clip_pos = mvp_matrix.map(vec)
                            
                            if clip_pos.w() == 0:
                                continue
                            
                            ndc_x = clip_pos.x() / clip_pos.w()
                            ndc_y = clip_pos.y() / clip_pos.w()
                            ndc_z = clip_pos.z() / clip_pos.w()
                            if ndc_z < 0 or ndc_z > 1:
                                continue
                            
                            screen_x = (ndc_x * 0.5 + 0.5) * width
                            screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * height
                            
                            dx = screen_x - click_pos.x()
                            dy = screen_y - click_pos.y()
                            dist = (dx*dx + dy*dy) ** 0.5
                            
                            if isinstance(scatter.size, np.ndarray) and i < len(scatter.size):
                                node_size = scatter.size[i]
                            else:
                                node_size = scatter.size if isinstance(scatter.size, (int, float)) else 10
                            
                            effective_threshold = pick_threshold + node_size
                            
                            if dist < effective_threshold and dist < closest_dist:
                                closest_dist = dist
                                closest_idx = i
                        
                        if closest_idx is not None:
                            self.parent_tab.show_node_info(closest_idx)
                            
                    except Exception as e:
                        print(f"Error in right-click handling: {e}")
                        import traceback
                        traceback.print_exc()
                
                def handle_cluster_selection(self, event: QMouseEvent):
                    """Handle left-click to select all nodes in the same cluster."""
                    scatter = self.parent_tab.gl_scatter
                    if scatter is None:
                        return
                    
                    try:
                        import numpy as np
                        from PySide6.QtGui import QMatrix4x4, QVector4D
                        
                        click_pos = event.pos()
                        
                        if scatter.pos is None or len(scatter.pos) == 0:
                            return
                        
                        positions = scatter.pos if isinstance(scatter.pos, np.ndarray) else np.array(scatter.pos)
                        
                        view_matrix = self.viewMatrix()
                        proj_matrix = self.currentProjection()
                        mvp_matrix = proj_matrix * view_matrix
                        
                        width = self.width()
                        height = self.height()
                        
                        closest_idx = None
                        closest_dist = float('inf')
                        pick_threshold = 20.0
                        
                        for i in range(len(positions)):
                            pos = positions[i]
                            vec = QVector4D(pos[0], pos[1], pos[2], 1.0)
                            clip_pos = mvp_matrix.map(vec)
                            
                            if clip_pos.w() == 0:
                                continue
                            
                            ndc_x = clip_pos.x() / clip_pos.w()
                            ndc_y = clip_pos.y() / clip_pos.w()
                            ndc_z = clip_pos.z() / clip_pos.w()
                            if ndc_z < 0 or ndc_z > 1:
                                continue
                            
                            screen_x = (ndc_x * 0.5 + 0.5) * width
                            screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * height
                            
                            dx = screen_x - click_pos.x()
                            dy = screen_y - click_pos.y()
                            dist = (dx*dx + dy*dy) ** 0.5
                            
                            if dist < pick_threshold and dist < closest_dist:
                                closest_dist = dist
                                closest_idx = i
                        
                        if closest_idx is not None:
                            self.parent_tab.show_cluster_info(closest_idx)
                            
                    except Exception as e:
                        print(f"Error in cluster selection: {e}")
                        import traceback
                        traceback.print_exc()
                
                def handle_set_camera_center(self, event: QMouseEvent):
                    """Handle right-click to set the camera center to the clicked node."""
                    scatter = self.parent_tab.gl_scatter
                    if scatter is None:
                        return
                    
                    try:
                        import numpy as np
                        from PySide6.QtGui import QMatrix4x4, QVector4D, QVector3D
                        
                        click_pos = event.pos()
                        
                        if scatter.pos is None or len(scatter.pos) == 0:
                            return
                        
                        positions = scatter.pos if isinstance(scatter.pos, np.ndarray) else np.array(scatter.pos)
                        
                        view_matrix = self.viewMatrix()
                        proj_matrix = self.currentProjection()
                        mvp_matrix = proj_matrix * view_matrix
                        
                        width = self.width()
                        height = self.height()
                        
                        closest_idx = None
                        closest_dist = float('inf')
                        pick_threshold = 20.0
                        
                        for i in range(len(positions)):
                            pos = positions[i]
                            vec = QVector4D(pos[0], pos[1], pos[2], 1.0)
                            clip_pos = mvp_matrix.map(vec)
                            
                            if clip_pos.w() == 0:
                                continue
                            
                            ndc_x = clip_pos.x() / clip_pos.w()
                            ndc_y = clip_pos.y() / clip_pos.w()
                            ndc_z = clip_pos.z() / clip_pos.w()
                            if ndc_z < 0 or ndc_z > 1:
                                continue
                            
                            screen_x = (ndc_x * 0.5 + 0.5) * width
                            screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * height
                            
                            dx = screen_x - click_pos.x()
                            dy = screen_y - click_pos.y()
                            dist = (dx*dx + dy*dy) ** 0.5
                            
                            if dist < pick_threshold and dist < closest_dist:
                                closest_dist = dist
                                closest_idx = i
                        
                        if closest_idx is not None:
                            # Set camera center to this node's position
                            center = positions[closest_idx]
                            self.opts['center'] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
                            self.update()
                            print(f"Camera center set to ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
                            
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
                    elif event.key() == Qt.Key_F4:
                        # Toggle split window
                        self.parent_tab.toggle_split_window()
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

        # Time Travel button
        self.time_travel_button = QPushButton("Time Travel")
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
        self.time_travel_button.setToolTip("Animate camera flying through cluster centroids.")
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
        self.cut_button.setToolTip("Cut out the selected cohort - keep only its nodes\nand remove everything else from the view.")
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
        self.pop_button.setToolTip("Remove the selected cohort from the view\n(keep everything else). Positions of remaining nodes stay unchanged.")
        self.pop_button.clicked.connect(self._pop_selected_cohort)
        self.pop_button.setEnabled(False)  # Enabled when a cohort is selected
        send_layout.addWidget(self.pop_button)

        # Re-cluster selection button - apply cluster algo on selection, keep positions
        self.cluster_button = QPushButton("Re-cluster selection")
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
        self.cluster_button.setToolTip("Apply the cluster algorithm on the selected cohort\nusing its existing positions to identify smaller sub-cohorts.\nPositions stay unchanged; only coloring/labels change.")
        self.cluster_button.clicked.connect(self._recluster_selection)
        self.cluster_button.setEnabled(False)  # Enabled when a cohort is selected
        send_layout.addWidget(self.cluster_button)

        send_group.setLayout(send_layout)
        layout.addWidget(send_group)

        # Visualization Settings Group (moved from left sidebar)
        vis_group = QGroupBox("Visualization Settings")
        vis_group.setToolTip("Control the appearance of nodes in the 3D view.")
        vis_layout = QFormLayout()

        self.min_size_spin = QDoubleSpinBox()
        self.min_size_spin.setRange(0.001, 5.0)
        self.min_size_spin.setValue(0.03)
        self.min_size_spin.setSingleStep(0.01)
        self.min_size_spin.setDecimals(3)
        self.min_size_spin.valueChanged.connect(self._on_size_changed)
        self.min_size_spin.setToolTip("Minimum size of a node (sphere) in the 3D view.\nFiles with few tags will be this size.\nLower = tiny dots, Higher = more visible small nodes.\nDefault: 0.03")
        vis_layout.addRow("Min Size:", self.min_size_spin)

        self.max_size_spin = QDoubleSpinBox()
        self.max_size_spin.setRange(0.001, 5.0)
        self.max_size_spin.setValue(0.08)
        self.max_size_spin.setSingleStep(0.01)
        self.max_size_spin.setDecimals(3)
        self.max_size_spin.valueChanged.connect(self._on_size_changed)
        self.max_size_spin.setToolTip("Maximum size of a node (sphere) in the 3D view.\nFiles with many tags will approach this size.\nDefault: 0.08")
        vis_layout.addRow("Max Size:", self.max_size_spin)

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

        # V5: Color Scheme dropdown
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["Pastel", "Viridis", "Plasma", "Inferno", "Coolwarm"])
        self.color_scheme_combo.setCurrentText("Pastel")
        self.color_scheme_combo.currentTextChanged.connect(self._on_color_scheme_changed)
        self.color_scheme_combo.setToolTip("Color scheme for cluster nodes.\nPastel = Default soft colors.\nViridis/Plasma/Inferno/Coolwarm = Matplotlib colormaps.")
        vis_layout.addRow("Color Scheme:", self.color_scheme_combo)

        # Anti-noise / quality settings
        self.msaa_checkbox = QCheckBox("MSAA Antialiasing")
        self.msaa_checkbox.setChecked(True)
        self.msaa_checkbox.setToolTip("Enable multisample antialiasing (MSAA) on the 3D view.\nReduces jagged/aliased edges on points and lines at high resolution.\nDefault: ON.")
        vis_layout.addRow(self.msaa_checkbox)

        self.point_smooth_checkbox = QCheckBox("Smooth Points")
        self.point_smooth_checkbox.setChecked(True)
        self.point_smooth_checkbox.setToolTip("Enable GL_POINT_SMOOTH for softer, rounder point rendering.\nReduces the 'noisy dot' look at high resolution.\nDefault: ON.")
        vis_layout.addRow(self.point_smooth_checkbox)

        self.supersample_checkbox = QCheckBox("Supersample (4x)")
        self.supersample_checkbox.setChecked(False)
        self.supersample_checkbox.setToolTip("Render the view at 4x resolution then downsample.\nGreatly reduces aliasing/noise but uses more GPU memory.\nDefault: OFF.")
        self.supersample_checkbox.stateChanged.connect(self._on_supersample_toggle)
        vis_layout.addRow(self.supersample_checkbox)

        self.supersample_fps_spin = QSpinBox()
        self.supersample_fps_spin.setRange(0, 60)
        self.supersample_fps_spin.setValue(10)
        self.supersample_fps_spin.setToolTip("FPS for the 4x supersample render.\n0 = disable the FPS limiter (render as fast as possible).\nHigher = smoother but heavier (more GPU/CPU).\nDefault: 10")
        self.supersample_fps_spin.valueChanged.connect(self._on_supersample_fps_changed)
        vis_layout.addRow("Supersample FPS:", self.supersample_fps_spin)

        # Show Cluster Boundaries checkbox
        self.show_boundaries_checkbox = QCheckBox("Show Cluster Boundaries")
        self.show_boundaries_checkbox.setChecked(False)
        self.show_boundaries_checkbox.stateChanged.connect(self._on_boundaries_toggle)
        self.show_boundaries_checkbox.setToolTip("Render semi-transparent convex hull meshes\naround clusters (10+ nodes). Default: OFF.")
        vis_layout.addRow(self.show_boundaries_checkbox)

        # Boundary Opacity spinner
        self.boundary_alpha_spin = QDoubleSpinBox()
        self.boundary_alpha_spin.setRange(0.0, 1.0)
        self.boundary_alpha_spin.setValue(0.15)
        self.boundary_alpha_spin.setDecimals(2)
        self.boundary_alpha_spin.setSingleStep(0.05)
        self.boundary_alpha_spin.valueChanged.connect(self._on_boundary_alpha_changed)
        self.boundary_alpha_spin.setToolTip("Opacity of cluster boundary hulls.\nHigher = more visible.\nDefault: 0.15")
        vis_layout.addRow("Boundary Opacity:", self.boundary_alpha_spin)

        # Show Relationship Lines checkbox
        self.show_relationships_checkbox = QCheckBox("Show Relationship Lines")
        self.show_relationships_checkbox.setChecked(False)
        self.show_relationships_checkbox.stateChanged.connect(self._on_relationships_toggle)
        self.show_relationships_checkbox.setToolTip("Draw lines between related clusters.\nLine opacity encodes relationship strength.\nDefault: OFF.")
        vis_layout.addRow(self.show_relationships_checkbox)

        # Relationship metric combo (enabled only when relationships checked)
        self.relationship_metric_combo = QComboBox()
        self.relationship_metric_combo.addItems(["IDF Cosine", "Shared Tag Count"])
        self.relationship_metric_combo.setCurrentIndex(0)
        self.relationship_metric_combo.currentIndexChanged.connect(self._on_relationship_metric_changed)
        self.relationship_metric_combo.setToolTip("How cluster relatedness is measured.\nIDF Cosine: IDF-weighted cosine similarity between cluster tag vectors\n(rare shared tags count more than common ones).\nShared Tag Count: raw number of shared tags (legacy behavior).\nDefault: IDF Cosine")
        self.relationship_metric_combo.setEnabled(False)
        vis_layout.addRow("Metric:", self.relationship_metric_combo)

        # Min similarity threshold (enabled only when relationships checked)
        self.relationship_min_sim_spin = QDoubleSpinBox()
        self.relationship_min_sim_spin.setRange(0.0, 1.0)
        self.relationship_min_sim_spin.setValue(0.1)
        self.relationship_min_sim_spin.setDecimals(2)
        self.relationship_min_sim_spin.setSingleStep(0.05)
        self.relationship_min_sim_spin.setToolTip("Minimum similarity score for a line to be drawn.\n0.0 = show all related pairs.\nDefault: 0.10")
        self.relationship_min_sim_spin.setEnabled(False)
        self.relationship_min_sim_spin.valueChanged.connect(self._on_relationship_params_changed)
        vis_layout.addRow("Min Similarity:", self.relationship_min_sim_spin)

        # Max relationships (enabled only when relationships checked)
        self.relationship_max_spin = QSpinBox()
        self.relationship_max_spin.setRange(1, 200)
        self.relationship_max_spin.setValue(30)
        self.relationship_max_spin.setToolTip("Maximum number of relationship lines to draw\n(top N strongest pairs).\nDefault: 30")
        self.relationship_max_spin.setEnabled(False)
        self.relationship_max_spin.valueChanged.connect(self._on_relationship_params_changed)
        vis_layout.addRow("Max Lines:", self.relationship_max_spin)

        # Line opacity (enabled only when relationships checked)
        self.relationship_alpha_spin = QDoubleSpinBox()
        self.relationship_alpha_spin.setRange(0.05, 1.0)
        self.relationship_alpha_spin.setValue(0.6)
        self.relationship_alpha_spin.setDecimals(2)
        self.relationship_alpha_spin.setSingleStep(0.05)
        self.relationship_alpha_spin.setToolTip("Base opacity of relationship lines.\nStronger relationships are drawn more opaque.\nDefault: 0.60")
        self.relationship_alpha_spin.setEnabled(False)
        self.relationship_alpha_spin.valueChanged.connect(self._on_relationship_params_changed)
        vis_layout.addRow("Line Opacity:", self.relationship_alpha_spin)

        # Line width (thickness) of relationship lines
        self.relationship_width_spin = QSpinBox()
        self.relationship_width_spin.setRange(1, 20)
        self.relationship_width_spin.setValue(2)
        self.relationship_width_spin.setToolTip("Thickness (width in pixels) of relationship lines.\nHigher = thicker, more visible lines.\nDefault: 2")
        self.relationship_width_spin.setEnabled(False)
        self.relationship_width_spin.valueChanged.connect(self._on_relationship_params_changed)
        vis_layout.addRow("Line Width:", self.relationship_width_spin)

        # Line color for relationship lines
        rel_color_row = QHBoxLayout()
        rel_color_label = QLabel("Line Color:")
        rel_color_label.setStyleSheet(f"color: {RED_A};")
        rel_color_row.addWidget(rel_color_label)
        self.relationship_color_btn = QPushButton("")
        self.relationship_color_btn.setFixedSize(40, 24)
        self.relationship_color_btn.setToolTip("Color of relationship lines. Click to change.")
        self.relationship_color_btn.clicked.connect(self._pick_relationship_color)
        self._relationship_color = (120, 160, 255)  # default soft blue
        self._update_relationship_color_button()
        rel_color_row.addWidget(self.relationship_color_btn)
        rel_color_row.addStretch()
        vis_layout.addRow(rel_color_row)

        # Show Relationship Labels checkbox (separate toggle)
        self.show_relationship_labels_checkbox = QCheckBox("Show Relationship Labels")
        self.show_relationship_labels_checkbox.setChecked(False)
        self.show_relationship_labels_checkbox.stateChanged.connect(self._on_relationship_labels_toggle)
        self.show_relationship_labels_checkbox.setToolTip("Show a label at the midpoint of each relationship line\nlisting the top shared tags between the two clusters.\nOnly applies when Show Relationship Lines is ON.\nDefault: OFF.")
        vis_layout.addRow(self.show_relationship_labels_checkbox)

        vis_group.setLayout(vis_layout)
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
        self.cohort_label_mode_combo.addItems([
            "Top N largest",
            "Above size threshold",
            "All cohorts",
        ])
        self.cohort_label_mode_combo.setToolTip(
            "Controls which cohort labels are shown to avoid overlap noise.\n"
            "Top N largest = only the N biggest cohorts (biggest to smallest).\n"
            "Above size threshold = only cohorts with at least N files.\n"
            "All cohorts = show every cohort label (can be noisy)."
        )
        self.cohort_label_mode_combo.currentTextChanged.connect(self._on_cohort_label_mode_changed)
        label_mode_row.addWidget(self.cohort_label_mode_combo)
        label_mode_row.addStretch()
        cohort_layout.addLayout(label_mode_row)

        # N parameter (used by Top N largest / Above size threshold modes)
        n_row = QHBoxLayout()
        n_label = QLabel("N:")
        n_label.setStyleSheet(f"color: {RED_A};")
        n_row.addWidget(n_label)
        self.cohort_label_n_spin = QSpinBox()
        self.cohort_label_n_spin.setRange(1, 500)
        self.cohort_label_n_spin.setValue(10)
        self.cohort_label_n_spin.setToolTip("Number of cohort labels to show (Top N largest) or minimum cohort size (Above size threshold).")
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
        self.cohort_label_size_spin.setValue(14)
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

        # Smart Labels toggle
        self.smart_labels_checkbox = QCheckBox("Smart Labels")
        self.smart_labels_checkbox.setChecked(False)
        self.smart_labels_checkbox.setToolTip(
            "Resolve duplicate labels on nearby cohorts.\n"
            "When two close cohorts would get the same label, the larger cohort keeps it\n"
            "and the smaller cohort gets the next-in-line dominant tag(s).\n"
            "Only applies to cohorts within the top 5 closest by centroid distance."
        )
        self.smart_labels_checkbox.stateChanged.connect(self._on_smart_labels_toggled)
        cohort_layout.addWidget(self.smart_labels_checkbox)

        # Smart Label Mode (controls behavior when max_tags >= 2)
        smart_mode_row = QHBoxLayout()
        smart_mode_label = QLabel("Smart Mode:")
        smart_mode_label.setStyleSheet(f"color: {RED_A};")
        smart_mode_row.addWidget(smart_mode_label)
        self.smart_label_mode_combo = QComboBox()
        self.smart_label_mode_combo.addItems(["All Unique", "Overlap", "Absolute Unique"])
        self.smart_label_mode_combo.setToolTip(
            "How to resolve duplicate labels.\n"
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
        label_color_row.addStretch()
        cohort_layout.addLayout(label_color_row)

        # Invert color toggle
        self.invert_label_color_checkbox = QCheckBox("Invert Label Color")
        self.invert_label_color_checkbox.setChecked(False)
        self.invert_label_color_checkbox.setToolTip("Invert the label color to contrast with the background for readability.")
        self.invert_label_color_checkbox.stateChanged.connect(self._on_invert_label_color_toggled)
        cohort_layout.addWidget(self.invert_label_color_checkbox)

        # Cohort table
        self.cohort_table = QTableWidget()
        self.cohort_table.setColumnCount(3)
        self.cohort_table.setHorizontalHeaderLabels(["Cohort", "Count", "Dominant Tags"])
        self.cohort_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.cohort_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.cohort_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.cohort_table.setColumnWidth(0, 70)
        self.cohort_table.setColumnWidth(1, 60)
        self.cohort_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cohort_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cohort_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {GRAY_40};
                color: {RED_A};
                border: 1px solid {BLUE_60};
                gridline-color: {BLUE_60};
                font-size: 11px;
            }}
            QTableWidget::item {{
                padding: 3px;
            }}
            QTableWidget::item:selected {{
                background-color: {BLUE_60};
            }}
            QHeaderView::section {{
                background-color: {BLUE_60};
                color: {RED_A};
                padding: 4px;
                border: 1px solid {GRAY_40};
                font-weight: bold;
            }}
        """)
        self.cohort_table.setToolTip("Click a row to select all files in that cohort.")
        self.cohort_table.cellClicked.connect(self._on_cohort_selected)
        cohort_layout.addWidget(self.cohort_table)

        cohort_group.setLayout(cohort_layout)
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
        layout.addWidget(importance_group)

        layout.addStretch()
        sidebar.setLayout(layout)
        return sidebar

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
    
    def resizeEvent(self, event):
        """Reposition toggle buttons on resize."""
        super().resizeEvent(event)
        if hasattr(self, 'sidebar_toggle_btn'):
            self._position_toggle_buttons()

    def _update_cohort_table(self):
        """Update the cohort table with top 20 cohorts by file count."""
        if not hasattr(self, 'node_list') or not self.node_list:
            return

        from collections import defaultdict, Counter

        # Group nodes by cluster_id (exclude noise cluster -1)
        cluster_nodes = defaultdict(list)
        for node in self.node_list:
            if node.cluster_id != -1:
                cluster_nodes[node.cluster_id].append(node)

        if not cluster_nodes:
            self.cohort_table.setRowCount(0)
            return

        # Build cohort info: (cluster_id, count, tag_counter)
        cohort_info = []
        for cluster_id, nodes in cluster_nodes.items():
            count = len(nodes)
            tag_counter = Counter()
            for node in nodes:
                if self.tag_interner:
                    tag_counter.update(self.tag_interner.strings_to_list(node.tags))
                else:
                    tag_counter.update(node.tags)
            cohort_info.append((cluster_id, count, tag_counter))

        # Sort by count descending, take top 20
        cohort_info.sort(key=lambda x: x[1], reverse=True)
        top_cohorts = cohort_info[:20]

        # Get threshold percentage
        threshold = self.cohort_threshold_spin.value()

        # Populate table
        self.cohort_table.setRowCount(len(top_cohorts))
        for row, (cluster_id, count, tag_counter) in enumerate(top_cohorts):
            # Cohort ID
            cohort_item = QTableWidgetItem(f"{cluster_id}")
            cohort_item.setTextAlignment(Qt.AlignCenter)
            self.cohort_table.setItem(row, 0, cohort_item)

            # Count
            count_item = QTableWidgetItem(f"{count}")
            count_item.setTextAlignment(Qt.AlignCenter)
            self.cohort_table.setItem(row, 1, count_item)

            # Dominant tags (tags present in >= threshold % of files)
            dominant_tags = []
            for tag, tag_count in tag_counter.most_common():
                percentage = tag_count / count
                if percentage >= threshold:
                    dominant_tags.append(f"{tag} ({percentage:.0%})")

            # If no tags meet the threshold, fall back to showing the top tags anyway
            if not dominant_tags:
                for tag, tag_count in tag_counter.most_common(5):
                    percentage = tag_count / count
                    dominant_tags.append(f"{tag} ({percentage:.0%})")

            tags_text = ", ".join(dominant_tags[:5])
            if len(dominant_tags) > 5:
                tags_text += " ..."
            tags_item = QTableWidgetItem(tags_text if tags_text else "-")
            self.cohort_table.setItem(row, 2, tags_item)

    def _on_cohort_threshold_changed(self, value):
        """Update cohort table and labels when threshold changes."""
        self._update_cohort_table()
        self._update_cohort_labels()

    def _on_show_cohort_labels_toggled(self, state):
        """Toggle cohort labels in the 3D view."""
        self._update_cohort_labels()
        # Start/stop the label blink timer
        if hasattr(self, 'cohort_label_blink_timer'):
            if self.show_cohort_labels_checkbox.isChecked():
                self.cohort_label_blink_visible = True
                self.cohort_label_blink_timer.start(500)  # ~2Hz blink
            else:
                self.cohort_label_blink_timer.stop()

    def _toggle_cohort_label_blink(self):
        """Blink the selected cohort's label between the two label colors."""
        self.cohort_label_blink_visible = not self.cohort_label_blink_visible
        # Determine the selected cohort's cluster_id
        selected_cid = self.selected_cluster_id
        if selected_cid is None and self.selected_node_index is not None:
            if hasattr(self, 'node_list') and 0 <= self.selected_node_index < len(self.node_list):
                selected_cid = self.node_list[self.selected_node_index].cluster_id
        # Swap only the selected cohort's label color; keep all others visible.
        # The custom paint() reads self.color (a QColor), so set that and
        # request a repaint via update().
        from PySide6.QtGui import QColor
        rgba = self._get_selected_label_rgba()
        for cid, item in self.cohort_label_map.items():
            try:
                if cid == selected_cid:
                    item.color = QColor(rgba[0], rgba[1], rgba[2], rgba[3])
                    item.update()
                else:
                    item.setVisible(True)
            except Exception:
                pass

    def _on_cohort_label_mode_changed(self, value):
        """Update cohort labels when the label mode changes."""
        self._update_cohort_labels()

    def _on_cohort_label_n_changed(self, value):
        """Update cohort labels when the N parameter changes."""
        self._update_cohort_labels()

    def _on_cohort_label_size_changed(self, value):
        """Update cohort label size."""
        self._update_cohort_labels()

    def _on_cohort_label_max_tags_changed(self, value):
        """Update cohort labels when the max-tags-per-label parameter changes."""
        self._update_cohort_labels()

    def _on_smart_labels_toggled(self, state):
        """Update cohort labels when the smart labels toggle changes."""
        self._update_cohort_labels()

    def _on_smart_label_mode_changed(self, value):
        """Update cohort labels when the smart label mode changes."""
        self._update_cohort_labels()

    def _on_dynamic_label_size_toggled(self, state):
        """Toggle dynamic label sizing."""
        self._update_cohort_labels()

    def _on_invert_label_color_toggled(self, state):
        """Toggle label color inversion."""
        self._update_cohort_labels()

    def _pick_cohort_label_color(self):
        """Open color picker for cohort labels."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        color = QColorDialog.getColor(
            QColor(*self._cohort_label_color),
            self,
            "Select Cohort Label Color",
        )
        if color.isValid():
            self._cohort_label_color = (color.red(), color.green(), color.blue())
            self._update_label_color_button()
            self._update_cohort_labels()

    def _update_label_color_button(self):
        """Update the label color button background."""
        r, g, b = self._cohort_label_color
        self.cohort_label_color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                border: 1px solid {BLUE_60};
                border-radius: 3px;
            }}
        """)

    def _pick_cohort_label_color2(self):
        """Open color picker for the selected cohort's blink (secondary) label color."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        color = QColorDialog.getColor(
            QColor(*self._cohort_label_color2),
            self,
            "Select Blink Label Color",
        )
        if color.isValid():
            self._cohort_label_color2 = (color.red(), color.green(), color.blue())
            self._update_label_color_button2()
            self._update_cohort_labels()

    def _update_label_color_button2(self):
        """Update the blink (secondary) label color button background."""
        r, g, b = self._cohort_label_color2
        self.cohort_label_color2_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                border: 1px solid {BLUE_60};
                border-radius: 3px;
            }}
        """)

    def _get_selected_label_rgba(self):
        """Return the RGBA tuple for the selected cohort's label based on blink state.

        The selected cohort's label blinks between the primary label color
        (when blink_visible) and the secondary blink color (when not).
        """
        if self.cohort_label_blink_visible:
            r, g, b = self._cohort_label_color
        else:
            r, g, b = self._cohort_label_color2
        if self.invert_label_color_checkbox.isChecked():
            r, g, b = 255 - r, 255 - g, 255 - b
        return (r, g, b, 255)

    def _get_query_shared_tags(self):
        """Get tags that are AND-shared across all files (from the query).

        Returns:
            set: Tags that appear as plain (non-negative, non-OR) query terms.
        """
        query = self.query_edit.text().strip()
        shared = set()
        if not query:
            return shared
        for part in self._split_query_preserving_brackets(query):
            part = part.strip()
            if not part:
                continue
            # Skip OR brackets and negative tags
            if part.startswith('[') and part.endswith(']'):
                continue
            if part.startswith('-'):
                continue
            shared.add(part)
        return shared

    def _compute_cohort_label_text(self, cluster_id, nodes):
        """Compute the dominant-tag label text for a cohort.

        Filters out tags that are AND-shared across all files (from the query),
        since those tags are present in every cohort and provide no distinction.
        Tags are newline-separated (no percentages) to keep labels short and
        readable in the 3D view.
        """
        from collections import Counter
        count = len(nodes)
        tag_counter = Counter()
        for node in nodes:
            if self.tag_interner:
                tag_counter.update(self.tag_interner.strings_to_list(node.tags))
            else:
                tag_counter.update(node.tags)

        shared_tags = self._get_query_shared_tags()
        threshold = self.cohort_threshold_spin.value()
        max_tags = self.cohort_label_max_tags_spin.value()

        dominant_tags = []
        for tag, tag_count in tag_counter.most_common():
            if tag in shared_tags:
                continue  # Skip AND-shared tags (present in all cohorts)
            percentage = tag_count / count
            if percentage >= threshold:
                dominant_tags.append(tag)

        # If no tags meet the threshold, fall back to top tags (still filtered)
        if not dominant_tags:
            for tag, tag_count in tag_counter.most_common(max_tags):
                if tag in shared_tags:
                    continue
                dominant_tags.append(tag)

        return dominant_tags[:max_tags]

    def _get_cohort_dominant_tags_full(self, nodes):
        """Get the FULL ranked dominant-tag list for a cohort (not truncated).

        Same filtering rules as _compute_cohort_label_text (skips AND-shared
        query tags, applies the threshold with top-tag fallback) but returns
        the complete ranked list so smart-labels resolution can pick
        next-in-line tags when the top ones are taken.

        Returns:
            list: All qualifying tags in dominance order (tag, count) tuples.
        """
        from collections import Counter
        count = len(nodes)
        tag_counter = Counter()
        for node in nodes:
            if self.tag_interner:
                tag_counter.update(self.tag_interner.strings_to_list(node.tags))
            else:
                tag_counter.update(node.tags)

        shared_tags = self._get_query_shared_tags()
        threshold = self.cohort_threshold_spin.value()

        ranked = []
        for tag, tag_count in tag_counter.most_common():
            if tag in shared_tags:
                continue
            percentage = tag_count / count
            if percentage >= threshold:
                ranked.append((tag, tag_count))

        # If no tags meet the threshold, fall back to top tags (still filtered)
        if not ranked:
            for tag, tag_count in tag_counter.most_common():
                if tag in shared_tags:
                    continue
                ranked.append((tag, tag_count))

        return ranked

    def _apply_smart_labels(self, cluster_nodes):
        """Resolve duplicate labels across cohorts (smart labels).

        Modes:
        - "All Unique" / "Overlap" (proximity-based):
            For each cohort, find the top 5 closest cohorts by centroid
            distance. If a cohort's label tags collide with a closer-or-
            equally-large neighbor's labels, the LARGER cohort keeps its
            tags and the smaller cohort substitutes the next-in-line
            dominant tags.
            - All Unique: replace ALL colliding tags with free ones.
            - Overlap: keep the FIRST tag, replace the rest with free ones.
        - "Absolute Unique" (global):
            No tag may appear in more than one cohort label. Cohorts are
            processed biggest-first; each greedily reserves its top
            dominant tags, skipping already-taken ones. May run out of
            tags if there are many cohorts (acceptable).

        Args:
            cluster_nodes: dict of cluster_id -> list of nodes (already
                filtered by label mode).

        Returns:
            dict: cluster_id -> list of tag strings (final labels).
        """
        import numpy as np

        max_tags = self.cohort_label_max_tags_spin.value()
        mode = self.smart_label_mode_combo.currentText()

        # Precompute per-cohort data: centroid, size, full ranked tag list
        cohort_info = {}
        for cid, nodes in cluster_nodes.items():
            positions = np.array([node.position for node in nodes])
            centroid = positions.mean(axis=0)
            ranked = self._get_cohort_dominant_tags_full(nodes)
            cohort_info[cid] = {
                "centroid": centroid,
                "size": len(nodes),
                "ranked": ranked,
            }

        if mode == "Absolute Unique":
            # Global greedy reservation, biggest cohort first
            taken = set()
            result = {}
            for cid in sorted(cohort_info, key=lambda c: cohort_info[c]["size"], reverse=True):
                info = cohort_info[cid]
                chosen = []
                for tag, _count in info["ranked"]:
                    if tag in taken:
                        continue
                    chosen.append(tag)
                    taken.add(tag)
                    if len(chosen) >= max_tags:
                        break
                result[cid] = chosen
            return result

        # Proximity-based modes (All Unique / Overlap)
        cids = list(cohort_info.keys())
        n = len(cids)
        if n < 2:
            return {cid: [t for t, _ in info["ranked"]][:max_tags]
                    for cid, info in cohort_info.items()}

        # Centroid matrix for distance computation
        centroids = np.array([cohort_info[c]["centroid"] for c in cids])
        # Pairwise distance matrix (n x n)
        diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        # For each cohort, find top 5 closest other cohorts
        closest = {}
        for i, cid in enumerate(cids):
            row = dist_matrix[i].copy()
            row[i] = np.inf
            k = min(5, n - 1)
            idx = np.argsort(row)[:k]
            closest[cid] = [cids[j] for j in idx]

        # Initial labels: top max_tags dominant tags
        labels = {
            cid: [t for t, _ in info["ranked"]][:max_tags]
            for cid, info in cohort_info.items()
        }

        # Resolve collisions. Process cohorts smallest-first so the larger
        # cohort always keeps its tags (the smaller one yields).
        for cid in sorted(cids, key=lambda c: cohort_info[c]["size"]):
            my_neighbors = closest[cid]
            # Collect tags held by neighbors that are LARGER than us
            # (we yield to larger neighbors; equal size -> lower cid yields)
            neighbor_tags = set()
            for nb in my_neighbors:
                nb_info = cohort_info[nb]
                if nb_info["size"] > cohort_info[cid]["size"]:
                    neighbor_tags.update(labels[nb])
                elif nb_info["size"] == cohort_info[cid]["size"] and nb < cid:
                    neighbor_tags.update(labels[nb])
            if not neighbor_tags:
                continue

            my_tags = labels[cid]
            colliding = [t for t in my_tags if t in neighbor_tags]
            if not colliding:
                continue

            if mode == "Overlap" and len(my_tags) >= 2:
                # Keep the first tag, replace the rest that collide
                keep = [my_tags[0]]
                to_replace = [t for t in my_tags[1:] if t in neighbor_tags]
                if not to_replace:
                    continue
                # Fill with next-in-line free tags
                for tag, _count in cohort_info[cid]["ranked"]:
                    if len(keep) >= max_tags:
                        break
                    if tag in keep or tag in neighbor_tags:
                        continue
                    keep.append(tag)
                labels[cid] = keep[:max_tags]
            else:
                # All Unique: replace ALL colliding tags with free ones
                kept = [t for t in my_tags if t not in neighbor_tags]
                for tag, _count in cohort_info[cid]["ranked"]:
                    if len(kept) >= max_tags:
                        break
                    if tag in kept or tag in neighbor_tags:
                        continue
                    kept.append(tag)
                labels[cid] = kept[:max_tags]

        return labels

    def _update_cohort_labels(self):
        """Render dominant-tag labels centered on each cohort in the 3D view."""
        try:
            import pyqtgraph.opengl as gl
            import numpy as np

            if not hasattr(self, 'gl_view') or self.gl_view is None:
                return

            # Remove existing labels
            self._remove_cohort_labels()
            self.cohort_label_map = {}

            if not self.show_cohort_labels_checkbox.isChecked():
                return
            if not hasattr(self, 'node_list') or not self.node_list:
                return

            # Group nodes by cluster
            cluster_nodes = {}
            for node in self.node_list:
                cid = node.cluster_id
                if cid == -1:
                    continue  # Skip noise
                cluster_nodes.setdefault(cid, []).append(node)

            if not cluster_nodes:
                return

            # Determine label color (with inversion)
            r, g, b = self._cohort_label_color
            if self.invert_label_color_checkbox.isChecked():
                r, g, b = 255 - r, 255 - g, 255 - b

            # Compute max cohort size for dynamic scaling
            max_count = max(len(nodes) for nodes in cluster_nodes.values()) if cluster_nodes else 1
            base_size = self.cohort_label_size_spin.value()

            # Apply label mode filter to avoid overlap noise
            mode = self.cohort_label_mode_combo.currentText()
            n = self.cohort_label_n_spin.value()
            if mode == "Top N largest":
                # Sort cohorts by size descending, keep top N
                sorted_cohorts = sorted(cluster_nodes.items(), key=lambda kv: len(kv[1]), reverse=True)
                cluster_nodes = dict(sorted_cohorts[:n])
            elif mode == "Above size threshold":
                # Keep only cohorts with at least N files
                cluster_nodes = {cid: nodes for cid, nodes in cluster_nodes.items() if len(nodes) >= n}
            # "All cohorts" -> no filtering

            # Always include the selected cohort so it gets a label even when
            # the label mode filter excluded it (e.g. outside top N or below
            # the size threshold). A single-node selection counts as its cohort.
            selected_cid = self.selected_cluster_id
            if selected_cid is None and self.selected_node_index is not None:
                if 0 <= self.selected_node_index < len(self.node_list):
                    selected_cid = self.node_list[self.selected_node_index].cluster_id
            if selected_cid is not None and selected_cid not in cluster_nodes:
                selected_nodes = [node for node in self.node_list if node.cluster_id == selected_cid]
                if selected_nodes:
                    cluster_nodes[selected_cid] = selected_nodes

            # Smart labels: resolve duplicate labels across cohorts
            smart_labels_enabled = self.smart_labels_checkbox.isChecked()
            smart_label_map = None
            if smart_labels_enabled:
                try:
                    smart_label_map = self._apply_smart_labels(cluster_nodes)
                except Exception as e:
                    print(f"Smart labels failed, falling back to normal labels: {e}")
                    import traceback
                    traceback.print_exc()
                    smart_label_map = None

            self.cohort_label_items = []
            for cid, nodes in cluster_nodes.items():
                # Compute centroid (center of cohort)
                positions = np.array([node.position for node in nodes])
                centroid = positions.mean(axis=0)
                spread = float(self.spread_spin.value())
                centroid = centroid * spread

                # Compute dominant tags (filtered list)
                if smart_label_map is not None:
                    tags = smart_label_map.get(cid, [])
                else:
                    tags = self._compute_cohort_label_text(cid, nodes)
                # Fallback for selected cohort: ensure a label is always shown
                # even when no dominant tags qualify (e.g. all tags are shared
                # query tags or none meet the threshold).
                if not tags and cid == selected_cid:
                    tags = [f"Cohort {cid}"]
                if not tags:
                    continue

                # Determine label size
                if self.dynamic_label_size_checkbox.isChecked():
                    size = base_size + int((len(nodes) / max_count) * (base_size * 2))
                else:
                    size = base_size

                # Render the whole label as a SINGLE multi-line GLTextItem.
                # The base pyqtgraph GLTextItem draws its text via
                # QPainter.drawText(QPointF, str), which treats the whole
                # string as ONE line -- embedded newlines are NOT rendered as
                # line breaks (lines concatenate, e.g. "Tag1Tag2Tag3"). So we
                # use a GLTextItem subclass that overrides paint() to draw each
                # line separately, offset in SCREEN space from the single
                # projected world anchor. All lines share one world anchor and
                # are offset in screen space, so the stack stays locked to the
                # camera (no world-space drift as the camera rotates).
                try:
                    from PySide6.QtGui import QFont
                    font = QFont("Helvetica", size)
                    label_text = "\n".join(tags)
                    MultiLineTextItem = _get_multiline_text_item_class()
                    if cid == selected_cid:
                        # Selected cohort: color follows blink state (primary
                        # vs secondary blink color), always fully opaque.
                        label_rgba = self._get_selected_label_rgba()
                    else:
                        # Other cohorts: primary color, dimmed when a selection
                        # is active and "Dim Non-Selected" is on (mirrors the
                        # node dimming so the selection stands out).
                        label_rgba = (r, g, b, 255)
                        if selected_cid is not None and self.dim_non_selected_checkbox.isChecked():
                            label_rgba = (r, g, b, int(self.dim_alpha_spin.value() * 255))
                    label_item = MultiLineTextItem(
                        pos=centroid,
                        text=label_text,
                        color=label_rgba,
                        font=font,
                        alignment=Qt.AlignHCenter,
                    )
                    self.gl_view.addItem(label_item)
                    label_item.setVisible(True)
                    self.cohort_label_items.append(label_item)
                    self.cohort_label_map[cid] = label_item
                except Exception as e:
                    print(f"Error creating cohort label: {e}")

        except Exception as e:
            import traceback
            print(f"Error updating cohort labels: {e}")
            traceback.print_exc()

    def _remove_cohort_labels(self):
        """Remove all cohort label items from the 3D view."""
        if not hasattr(self, 'cohort_label_items'):
            self.cohort_label_items = []
            return
        for item in self.cohort_label_items:
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self.cohort_label_items = []
        self.cohort_label_map = {}

    def _on_cohort_selected(self, row, column):
        """Handle cohort row selection - highlight all nodes in that cohort."""
        if not hasattr(self, 'node_list') or not self.node_list:
            return

        # Get cluster_id from first column
        cohort_item = self.cohort_table.item(row, 0)
        if not cohort_item:
            return

        try:
            cluster_id = int(cohort_item.text())
        except ValueError:
            return

        # Clear previous selection
        self.clear_selection()

        # Clear previous tag widgets and query states
        self.tag_query_states.clear()
        self._clear_tag_widgets()

        # Find all nodes in this cluster
        cluster_nodes = [n for n in self.node_list if n.cluster_id == cluster_id]
        if not cluster_nodes:
            return

        # Gather info
        file_ids = [n.file_id for n in cluster_nodes]

        # Count tag occurrences
        tag_counts = {}
        for n in cluster_nodes:
            for tag in n.tags:
                if self.tag_interner and isinstance(tag, int):
                    tag = self.tag_interner.index_to_string(tag)
                if tag is None:
                    continue
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))

        # DEBUG: Log what's happening
        print(f"[DEBUG] _on_cohort_selected: cluster_id={cluster_id}, nodes={len(cluster_nodes)}, tags={len(tag_counts)}")
        print(f"[DEBUG] sorted_tags[:5]={sorted_tags[:5]}")
        print(f"[DEBUG] tag_grid exists: {hasattr(self, 'tag_grid')}")
        print(f"[DEBUG] tag_grid count before: {self.tag_grid.count() if hasattr(self, 'tag_grid') else 'N/A'}")

        # Update text info with cohort summary and top 20 tags with counts
        info_lines = [
            f"Cohort {cluster_id} - {len(cluster_nodes)} files",
            "",
            f"Top Tags ({min(len(tag_counts), 20)} shown):",
        ]
        for rank, (tag, count) in enumerate(sorted_tags[:20], 1):
            pct = count / len(cluster_nodes) * 100
            info_lines.append(f"{rank}. {tag} ({count} files, {pct:.0f}%)")
        self.info_text.setText("\n".join(info_lines))

        # Also create clickable tag widgets for top tags (kept alongside text display)
        tags_to_show = sorted_tags[:20]  # Limit to 20 tags for performance
        included_query, excluded_query, or_query = self._get_query_tag_states()
        for idx, (tag, count) in enumerate(tags_to_show):
            tag_widget = ClickableTag(tag)
            tag_widget.setToolTip(f"Appears in {count} files ({count/len(cluster_nodes)*100:.0f}%)")
            tag_widget.stateChanged.connect(self._on_tag_state_changed)
            # Initialize state from the current query so already-queried tags show correctly
            if tag in excluded_query:
                tag_widget.state = 2
            elif tag in or_query:
                tag_widget.state = 3
            elif tag in included_query:
                tag_widget.state = 1
            else:
                tag_widget.state = 0
            tag_widget._update_appearance()
            self.tag_query_states[tag] = tag_widget.state
            self._tag_widgets.append(tag_widget)
            self.tag_grid.addWidget(tag_widget, idx // 3, idx % 3)  # 3 columns

        # Force layout refresh so the tags render immediately
        self.tag_container.update()
        self.tag_grid.update()

        # Set up cluster selection highlight
        self.selected_cluster_id = cluster_id
        self.selection_timer.start(500)

        # Enable cut/pop/re-cluster buttons for the selected cohort
        if hasattr(self, 'cut_button'):
            self.cut_button.setEnabled(True)
        if hasattr(self, 'pop_button'):
            self.pop_button.setEnabled(True)
        if hasattr(self, 'cluster_button'):
            self.cluster_button.setEnabled(True)

        # Update the cohort tag data panel
        self._update_selection_tags(cluster_nodes)

        # Ensure the selected cohort gets a label (even if filtered out by label mode)
        self._update_cohort_labels()

        # Sync split window if open
        self._sync_split_window()

    def _update_selection_tags(self, cluster_nodes):
        """Populate the Cohort Tag Data panel with top 20 tags for the selection.

        Args:
            cluster_nodes: List of nodes in the selected cohort/cluster.
        """
        if not hasattr(self, 'selection_tags_text'):
            return

        if not cluster_nodes:
            self.selection_tags_text.setText("No files in selection.")
            return

        # Count tag occurrences across the selected files
        tag_counts = {}
        for n in cluster_nodes:
            for tag in n.tags:
                # Resolve tokenized indices back to strings for display
                if self.tag_interner and isinstance(tag, int):
                    tag = self.tag_interner.index_to_string(tag)
                if tag is None:
                    continue
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Sort by count (descending), then alphabetically for ties
        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))

        total_files = len(cluster_nodes)
        lines = [
            f"Selection: {total_files} files",
            "",
            f"Most Common Tags (top {min(len(sorted_tags), 20)}):",
        ]
        for rank, (tag, count) in enumerate(sorted_tags[:20], 1):
            pct = count / total_files * 100
            lines.append(f"{rank}. {tag} ({count} files, {pct:.0f}%)")
        self.selection_tags_text.setText("\n".join(lines))

    def send_selected_to_tab(self):
        """Send selected file(s) to the specified Hydrus tab."""
        tab_name = self.tab_name_edit.text().strip()
        if not tab_name:
            self.send_status_label.setText("❌ Please enter a tab name.")
            return

        # Get selected file IDs
        file_ids = self._get_selected_file_ids()
        if not file_ids:
            self.send_status_label.setText("❌ No file selected. Right-click a node to select.")
            return

        # Get client from current settings
        client_name = self.client_combo.currentText()
        try:
            from src.utils.utility_functions import ConnectToClient
            client = ConnectToClient(client_name)

            # Get page list and find target tab
            page_list = client.get_page_list()
            target_page = self._find_page_by_name(page_list, tab_name)

            if not target_page:
                self.send_status_label.setText(f"❌ Tab '{tab_name}' not found.")
                return

            page_key = target_page.get('page_key')
            if not page_key:
                self.send_status_label.setText(f"❌ Could not get page key for tab '{tab_name}'.")
                return

            # Send files to tab
            client.add_files_to_page(page_key=page_key, file_ids=file_ids)
            self.send_status_label.setText(f"✅ Sent {len(file_ids)} file(s) to '{tab_name}'.")

        except Exception as e:
            self.send_status_label.setText(f"❌ Error: {str(e)}")
    
    def _find_page_by_name(self, pages_list, tab_name):
        """Recursively search for a page by name (case-insensitive).
        
        Args:
            pages_list: List of page dictionaries from get_page_list()
            tab_name: Name of the tab to find
            
        Returns:
            Page dictionary if found, None otherwise
        """
        for page_info in pages_list:
            if not isinstance(page_info, dict):
                continue
            
            name = page_info.get('name', '')
            title = page_info.get('title', '')
            
            if (tab_name.lower() == name.lower()) or (tab_name.lower() == title.lower()):
                return page_info
            
            # Recurse into nested pages
            if 'pages' in page_info:
                nested_page = self._find_page_by_name(page_info['pages'], tab_name)
                if nested_page:
                    return nested_page
        
        return None

    def _get_selected_file_ids(self):
        """Get file IDs from current selection."""
        file_ids = []
        # Single node selection
        if self.selected_node_index is not None and hasattr(self, 'file_ids'):
            if self.selected_node_index < len(self.file_ids):
                file_ids.append(self.file_ids[self.selected_node_index])
        # Cluster selection
        elif self.selected_cluster_id is not None and hasattr(self, 'node_list'):
            for node in self.node_list:
                if getattr(node, 'cluster_id', None) == self.selected_cluster_id:
                    file_ids.append(node.file_id)  # TagNode has file_id (singular)
        return file_ids

    def _is_worker_busy(self):
        """Return True if a processing worker thread is currently running.

        Used to prevent starting a second pipeline (e.g. re-clustering a
        cohort) while a first one (e.g. load & compute) is still in flight.
        Concurrent workers share self.node_list / self.scene_graph /
        self.tag_interner and would corrupt each other and crash the UI.
        """
        worker = getattr(self, 'worker', None)
        return worker is not None and worker.isRunning()

    def _set_cohort_action_buttons(self, enabled):
        """Enable/disable the cohort action buttons (cut/pop/re-cluster).

        These buttons start worker threads, so they must be locked while any
        pipeline is running to avoid concurrent workers.
        """
        for btn in (self.cut_button, self.pop_button, self.cluster_button):
            btn.setEnabled(enabled)

    def start_loading(self):
        """Start the data loading and computation process."""
        if self._is_worker_busy():
            self.status_label.setText("Please wait - a process is already running.")
            return
        self.load_button.setEnabled(False)
        self.recompute_button.setEnabled(False)
        self.recluster_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.save_session_button.setEnabled(False)
        self.load_session_button.setEnabled(False)
        self._set_cohort_action_buttons(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        # Fresh full load produces new positions -> re-fit camera on next render
        self._camera_initialized = False

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

        Args:
            tag_list: List of tag strings (may contain wildcards like 'system:*')

        Returns:
            tuple: (exact_set, compiled_patterns)
                - exact_set: set of lowercase exact tag names (no wildcards)
                - compiled_patterns: list of compiled regex for wildcard patterns
        """
        import fnmatch
        import re
        exact = set()
        patterns = []
        for pattern in tag_list:
            if '*' in pattern or '?' in pattern or '[' in pattern:
                patterns.append(re.compile(fnmatch.translate(pattern.lower())))
            else:
                exact.add(pattern.lower())
        return exact, patterns

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
        chunk_size = self.chunk_size_spin.value()
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
        min_size = float(self.min_size_spin.value())
        max_size = float(self.max_size_spin.value())
        spread = float(self.spread_spin.value())
        whitelist = [t.strip() for t in self.whitelist_edit.text().split(',') if t.strip()]
        blacklist = [t.strip() for t in self.blacklist_edit.text().split(',') if t.strip()]
        drop_empty = self.drop_empty_checkbox.isChecked() if hasattr(self, 'drop_empty_checkbox') else False
        query = self.query_edit.text().strip()
        min_doc_freq = self.min_doc_freq_spin.value() if hasattr(self, 'min_doc_freq_spin') else 3
        drop_universal = self.drop_universal_checkbox.isChecked() if hasattr(self, 'drop_universal_checkbox') else False

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
        loader = DataLoader(client, chunk_size=chunk_size, client_name=client_name, use_direct_db=use_direct_db)

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
        tokenize = self.tokenize_checkbox.isChecked() if hasattr(self, 'tokenize_checkbox') else False
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
        vec = Vectorizer(min_doc_freq=min_doc_freq, tokenized=bool(self.tag_interner), reverse_vocab=reverse_vocab, drop_universal_tags=drop_universal)
        _t_vec = time.perf_counter()
        sparse_matrix, file_ids = vec.create_vectors(tag_data)
        print(f"[Timing] Vectorizing took {time.perf_counter() - _t_vec:.2f}s")

        # Reduce dimensionality
        self.worker.progress.emit(60, f"Applying {algorithm.upper()}...")
        red = Reducer(
            algorithm=algorithm,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            low_memory=low_memory,
            metric=metric,
            n_jobs=n_jobs
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
                             min_size=min_size, max_size=max_size,
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
        min_size = float(self.min_size_spin.value())
        max_size = float(self.max_size_spin.value())
        min_doc_freq = self.min_doc_freq_spin.value() if hasattr(self, 'min_doc_freq_spin') else 3
        drop_universal = self.drop_universal_checkbox.isChecked() if hasattr(self, 'drop_universal_checkbox') else False

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
        red = Reducer(
            algorithm=algorithm,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            low_memory=low_memory,
            metric=metric,
            n_jobs=n_jobs
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

        # Cluster
        self.worker.progress.emit(70, "Clustering...")
        clust = Clusterer(eps=eps, min_samples=min_samples)
        _t_clust = time.perf_counter()
        cluster_positions = self._maybe_normalize_positions(positions)
        cluster_labels = clust.fit_predict(cluster_positions)
        print(f"[Timing] Clustering took {time.perf_counter() - _t_clust:.2f}s")

        # Build scene graph
        self.worker.progress.emit(85, "Building scene graph...")
        scene = SceneGraph()
        _t_scene = time.perf_counter()
        scene.build_from_data(file_ids, positions, tag_data, cluster_labels,
                             min_size=min_size, max_size=max_size,
                             tokenized=bool(self.tag_interner), reverse_vocab=reverse_vocab)
        print(f"[Timing] Building scene graph took {time.perf_counter() - _t_scene:.2f}s")

        self.worker.progress.emit(100, "Recompute complete!")
        import gc
        gc.collect()
        return scene, tag_data

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

        # Collect nodes in the selected cohort
        cluster_nodes = [n for n in self.node_list if n.cluster_id == self.selected_cluster_id]
        if len(cluster_nodes) < 2:
            self.status_label.setText("Cohort too small to cut out (need 2+ files).")
            return

        # Build sub tag_data from the cohort's nodes
        sub_tag_data = {}
        for node in cluster_nodes:
            sub_tag_data[node.file_id] = list(node.tags)

        self.status_label.setText(f"Cutting out cohort {self.selected_cluster_id} ({len(cluster_nodes)} files)...")
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
        min_size = float(self.min_size_spin.value())
        max_size = float(self.max_size_spin.value())
        min_doc_freq = self.min_doc_freq_spin.value() if hasattr(self, 'min_doc_freq_spin') else 3
        drop_universal = self.drop_universal_checkbox.isChecked() if hasattr(self, 'drop_universal_checkbox') else False

        # Vectorize the subset
        self.worker.progress.emit(10, "Vectorizing sub-cohort...")
        reverse_vocab = self.tag_interner.index_to_tag if self.tag_interner else None
        vec = Vectorizer(min_doc_freq=min_doc_freq, tokenized=bool(self.tag_interner), reverse_vocab=reverse_vocab, drop_universal_tags=drop_universal)
        _t_vec = time.perf_counter()
        sparse_matrix, file_ids = vec.create_vectors(sub_tag_data)
        print(f"[Timing] Vectorizing sub-cohort took {time.perf_counter() - _t_vec:.2f}s")

        # Reduce dimensionality
        self.worker.progress.emit(40, f"Applying {algorithm.upper()}...")
        red = Reducer(
            algorithm=algorithm,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            low_memory=low_memory,
            metric=metric,
            n_jobs=n_jobs
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
                              min_size=min_size, max_size=max_size,
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

        removed = [n for n in self.node_list if n.cluster_id == self.selected_cluster_id]
        remaining = [n for n in self.node_list if n.cluster_id != self.selected_cluster_id]
        if not remaining:
            self.status_label.setText("Cannot pop: this is the only cohort in the view.")
            return

        # Clear the selection (the cohort is about to disappear)
        self.clear_selection()

        self.status_label.setText(f"Popping cohort {self.selected_cluster_id} ({len(removed)} files)...")
        self._set_cohort_action_buttons(False)

        def worker_func():
            return self._pop_compute(remaining)

        self.worker = WorkerThread(worker_func)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.error.connect(self.on_loading_error)
        self.worker.start()

    def _pop_compute(self, remaining_nodes):
        """Rebuild the scene without the popped cohort (positions unchanged)."""
        import numpy as np
        from src.core.models import SceneGraph

        min_size = float(self.min_size_spin.value())
        max_size = float(self.max_size_spin.value())

        file_ids = [n.file_id for n in remaining_nodes]
        positions = np.array([n.position for n in remaining_nodes])
        cluster_labels = np.array([n.cluster_id for n in remaining_nodes])
        tag_data = {}
        for n in remaining_nodes:
            tag_data[n.file_id] = list(n.tags)

        self.worker.progress.emit(40, "Rebuilding scene...")
        reverse_vocab = self.tag_interner.index_to_tag if self.tag_interner else None
        scene = SceneGraph()
        scene.build_from_data(
            file_ids,
            positions,
            tag_data,
            cluster_labels,
            min_size=min_size,
            max_size=max_size,
            tokenized=bool(self.tag_interner),
            reverse_vocab=reverse_vocab,
        )

        # Preserve original colors so the remaining cohorts look unchanged
        old_nodes = self.scene_graph.nodes if self.scene_graph is not None else {}
        for node in scene.nodes.values():
            old = old_nodes.get(node.file_id)
            if old is not None:
                node.color = old.color

        # Filter tag_data so recompute works on the remaining files
        self.tag_data = tag_data

        self.worker.progress.emit(100, "Cohort popped!")
        return scene, tag_data

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

        cluster_nodes = [n for n in self.node_list if n.cluster_id == self.selected_cluster_id]
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
        min_size = float(self.min_size_spin.value())
        max_size = float(self.max_size_spin.value())

        # Use existing positions (no re-reduce)
        positions = np.array([node.position for node in cluster_nodes])
        spread = float(self.spread_spin.value())
        positions = positions * spread

        # Use SEPARATE sub-clustering settings (independent from global eps/min).
        # The selected cohort is already a dense sub-region; the global eps would
        # treat it as one cluster and produce no split. The user controls these
        # independently via Sub EPS / Sub Min Samples.
        sub_eps = self.sub_eps_spin.value() / 100.0
        sub_min_samples = self.sub_min_samples_spin.value()
        self.worker.progress.emit(40, "Clustering selection (sub-cohorts)...")
        clust = Clusterer(eps=sub_eps, min_samples=sub_min_samples)
        cluster_positions = self._maybe_normalize_positions(positions)
        cluster_labels = clust.fit_predict(cluster_positions)

        # Build sub tag_data from the selected nodes
        sub_tag_data = {}
        for node in cluster_nodes:
            sub_tag_data[node.file_id] = list(node.tags)

        # Build a FULL scene: keep all nodes, only re-label the selected cohort.
        # Non-selected nodes keep their original positions and cluster labels.
        all_file_ids = []
        all_positions = []
        all_labels = []
        all_tag_data = {}

        selected_ids = {n.file_id for n in cluster_nodes}
        # Map selected node -> new sub-cohort label
        selected_label_map = {}
        for node, new_label in zip(cluster_nodes, cluster_labels):
            selected_label_map[node.file_id] = new_label

        # Remap sub-cohort labels to UNIQUE IDs that don't collide with existing
        # cluster IDs of non-selected nodes. DBSCAN returns 0-based labels which
        # may overlap with existing cluster IDs, causing color/group confusion.
        existing_ids = {n.cluster_id for n in self.node_list if n.cluster_id != -1}
        max_existing = max(existing_ids) if existing_ids else -1
        sub_label_remap = {}
        next_id = max_existing + 1
        for label in cluster_labels:
            if label == -1:
                sub_label_remap[label] = -1
            elif label not in sub_label_remap:
                sub_label_remap[label] = next_id
                next_id += 1

        for node in self.node_list:
            all_file_ids.append(node.file_id)
            all_positions.append(node.position)
            all_tag_data[node.file_id] = list(node.tags)
            if node.file_id in selected_ids:
                # Re-labeled sub-cohort (new unique label from DBSCAN)
                all_labels.append(sub_label_remap[selected_label_map[node.file_id]])
            else:
                # Keep original cluster label
                all_labels.append(node.cluster_id)

        all_positions = np.array(all_positions) * spread

        self.worker.progress.emit(90, "Building sub-scene...")
        reverse_vocab = self.tag_interner.index_to_tag if self.tag_interner else None
        scene = SceneGraph()
        scene.build_from_data(
            all_file_ids,
            all_positions,
            all_tag_data,
            np.array(all_labels),
            min_size=min_size,
            max_size=max_size,
            tokenized=bool(self.tag_interner),
            reverse_vocab=reverse_vocab,
        )

        # Preserve original colors for non-selected nodes so their appearance
        # stays exactly the same; only the split sub-cohorts get new colors.
        if self.scene_graph is not None:
            old_nodes = self.scene_graph.nodes
            for node in scene.nodes.values():
                if node.file_id not in selected_ids:
                    old = old_nodes.get(node.file_id)
                    if old is not None:
                        node.color = old.color

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

    def _update_recluster_button_state(self):
        """Grey out the recluster button when target params equal last-used params.

        The button unlocks only after the user changes eps/min_samples away from
        the values that produced the current clusters, preventing accidental
        redundant recomputation and teaching the button's purpose.
        """
        if not hasattr(self, 'recluster_button'):
            return
        if getattr(self, '_last_used_eps', None) is None or getattr(self, '_last_used_min_samples', None) is None:
            # No data processed yet; keep button disabled until data loads
            self.recluster_button.setEnabled(False)
            return
        current_eps = self.eps_spin.value() / 100.0
        current_min = self.min_samples_spin.value()
        same = (abs(current_eps - self._last_used_eps) < 1e-9
                and current_min == self._last_used_min_samples)
        self.recluster_button.setEnabled(not same)

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
        min_size = float(self.min_size_spin.value())
        max_size = float(self.max_size_spin.value())
        spread = float(self.spread_spin.value())

        # Use existing positions for ALL nodes (no re-reduce)
        all_file_ids = []
        all_positions = []
        all_tag_data = {}
        for node in self.node_list:
            all_file_ids.append(node.file_id)
            all_positions.append(node.position)
            all_tag_data[node.file_id] = list(node.tags)
        all_positions = np.array(all_positions) * spread

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
            min_size=min_size,
            max_size=max_size,
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
        if not getattr(self, 'normalize_positions', False):
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

        min_size = float(self.min_size_spin.value())
        max_size = float(self.max_size_spin.value())
        spread = float(self.spread_spin.value())

        # Use existing positions for ALL nodes (no re-reduce)
        all_file_ids = []
        all_positions = []
        all_tag_data = {}
        for node in self.node_list:
            all_file_ids.append(node.file_id)
            all_positions.append(node.position)
            all_tag_data[node.file_id] = list(node.tags)
        all_positions = np.array(all_positions) * spread

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
            min_size=min_size,
            max_size=max_size,
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

    def _save_session(self):
        """Save the current rendered session to a hybrid .npz archive.

        Stores positions and cluster labels as compact numpy arrays, and
        metadata (file_ids, tags, colors, sizes, settings, camera) as JSON.
        This lets a future load skip query, metadata, UMAP, and DBSCAN.
        """
        import numpy as np
        import json
        from pathlib import Path

        if self.scene_graph is None or not hasattr(self, 'node_list') or not self.node_list:
            self.status_label.setText("Error: No scene to save. Load data first.")
            return

        scene = self.scene_graph
        nodes = list(scene.nodes.values())
        if not nodes:
            self.status_label.setText("Error: Scene has no nodes.")
            return

        # Build parallel arrays
        file_ids = [n.file_id for n in nodes]
        positions = np.array([n.position for n in nodes])  # (n, 3)
        cluster_labels = np.array([n.cluster_id for n in nodes])  # (n,)
        colors = np.array([n.color for n in nodes])  # (n, 3)
        sizes = np.array([n.size for n in nodes])  # (n,)

        # Build tag metadata (resolve tokenized tags to strings for portability)
        tags_list = []
        for n in nodes:
            if scene.tokenized and scene.reverse_vocab:
                resolved = []
                for t in n.tags:
                    if isinstance(t, int) and t < len(scene.reverse_vocab):
                        resolved.append(scene.reverse_vocab[t])
                    else:
                        resolved.append(str(t))
                tags_list.append(resolved)
            else:
                tags_list.append(list(n.tags))

        # Build cluster metadata
        clusters_meta = []
        for c in scene.clusters.values():
            clusters_meta.append({
                "id": c.cluster_id,
                "centroid": c.centroid.tolist(),
                "size": len(c.nodes),
                "dominant_tags": list(c.dominant_tags),
                "color": list(c.color),
                "label": c.label,
                "density": float(c.density),
            })

        # Build settings snapshot for recompute/recluster after load
        settings_meta = {
            "algorithm": self.algorithm_combo.currentText().lower(),
            "n_neighbors": self.n_neighbors_spin.value(),
            "min_dist": self.min_dist_spin.value() / 100.0,
            "n_epochs": self.n_epochs_spin.value() if hasattr(self, 'n_epochs_spin') else None,
            "learning_rate": self.learning_rate_spin.value() if hasattr(self, 'learning_rate_spin') else 1.0,
            "low_memory": self.low_memory,
            "n_jobs": self.n_jobs,
            "metric": self.metric_combo.currentText() if hasattr(self, 'metric_combo') else 'cosine',
            "eps": self.eps_spin.value() / 100.0,
            "min_samples": self.min_samples_spin.value(),
            "min_size": float(self.min_size_spin.value()),
            "max_size": float(self.max_size_spin.value()),
            "spread": float(self.spread_spin.value()),
            "min_doc_freq": self.min_doc_freq_spin.value() if hasattr(self, 'min_doc_freq_spin') else 3,
            "drop_universal_tags": self.drop_universal_checkbox.isChecked() if hasattr(self, 'drop_universal_checkbox') else False,
            "tokenized": bool(self.tag_interner),
        }

        # Camera state
        camera_meta = {
            "position": scene.camera_position.tolist(),
            "target": scene.camera_target.tolist(),
        }

        # Build metadata JSON
        metadata = {
            "version": 1,
            "node_count": len(nodes),
            "file_ids": file_ids,
            "tags": tags_list,
            "colors": colors.tolist(),
            "sizes": sizes.tolist(),
            "clusters": clusters_meta,
            "settings": settings_meta,
            "camera": camera_meta,
        }

        # Save as hybrid archive in the sessions/ subfolder with a timestamp name
        import time as _time
        from pathlib import Path

        sessions_dir = Path("sessions")
        sessions_dir.mkdir(exist_ok=True)

        # Use timestamp-based filename (user can rename later)
        timestamp = _time.strftime("%Y%m%d_%H%M%S")
        save_path = sessions_dir / f"session_{timestamp}.npz"

        np.savez_compressed(
            save_path,
            positions=positions,
            cluster_labels=cluster_labels,
            metadata=json.dumps(metadata).encode('utf-8'),
        )

        self.status_label.setText(f"Session saved to {save_path} ({len(nodes)} nodes)")
        print(f"[Session] Saved {len(nodes)} nodes to {save_path}")

    def _get_save_path(self, default_path):
        """Prompt for a save path via QFileDialog."""
        from pathlib import Path

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            str(default_path),
            "Session Archive (*.npz)",
        )
        if not filename:
            return None, None
        return Path(filename), None

    def _load_session(self):
        """Load a saved session archive directly (skips reprocessing).

        Reconstructs the SceneGraph from the hybrid .npz archive and renders
        it immediately — no query, metadata, UMAP, or DBSCAN needed.
        """
        import numpy as np
        import json
        from pathlib import Path
        from src.core.models import SceneGraph, TagNode, Cluster

        # List sessions from the sessions/ subfolder
        from pathlib import Path
        sessions_dir = Path("sessions")
        if not sessions_dir.exists():
            self.status_label.setText("No sessions folder found.")
            return

        session_files = sorted(sessions_dir.glob("*.npz"), reverse=True)
        if not session_files:
            self.status_label.setText("No saved sessions found.")
            return

        # Prompt user to pick a session by name (inline filename)
        from PySide6.QtWidgets import QInputDialog
        names = [f.name for f in session_files]
        choice, ok = QInputDialog.getItem(
            self,
            "Load Session",
            "Select a session:",
            names,
            0,
            False,
        )
        if not ok or not choice:
            return
        filename = sessions_dir / choice

        try:
            data = np.load(filename, allow_pickle=True)
            positions = data['positions']
            cluster_labels = data['cluster_labels']
            metadata = json.loads(data['metadata'].decode('utf-8'))

            file_ids = metadata['file_ids']
            tags_list = metadata['tags']
            colors = metadata['colors']
            sizes = metadata['sizes']
            clusters_meta = metadata['clusters']
            settings_meta = metadata['settings']
            camera_meta = metadata['camera']

            # Reconstruct SceneGraph
            scene = SceneGraph()
            scene.tokenized = False  # tags stored as strings
            scene.reverse_vocab = None

            nodes = []
            for i, fid in enumerate(file_ids):
                node = TagNode(
                    file_id=fid,
                    position=positions[i],
                    tags=tags_list[i],
                    score=0.0,  # recomputed on demand; not persisted
                    cluster_id=int(cluster_labels[i]),
                    color=tuple(colors[i]),
                    size=sizes[i],
                )
                nodes.append(node)
            scene.add_nodes(nodes)

            # Reconstruct clusters
            for cmeta in clusters_meta:
                cid = cmeta['id']
                cluster_nodes = [n for n in nodes if n.cluster_id == cid]
                cluster = Cluster(
                    cluster_id=cid,
                    centroid=np.array(cmeta['centroid']),
                    nodes=cluster_nodes,
                    dominant_tags=list(cmeta['dominant_tags']),
                    color=tuple(cmeta['color']),
                    label=cmeta['label'],
                    density=cmeta['density'],
                )
                scene.add_cluster(cluster)

            # Restore camera
            scene.camera_position = np.array(camera_meta['position'])
            scene.camera_target = np.array(camera_meta['target'])

            # Restore settings for recompute/recluster
            self._apply_session_settings(settings_meta)

            self.scene_graph = scene
            self.node_list = list(scene.nodes.values())
            self.tag_data = None  # not persisted; recompute needs re-query

            self.status_label.setText(f"Session loaded: {len(nodes)} nodes")
            print(f"[Session] Loaded {len(nodes)} nodes from {filename}")
            self.render_scene(scene)

        except Exception as e:
            import traceback
            self.status_label.setText(f"Error loading session: {e}")
            traceback.print_exc()

    def _apply_session_settings(self, settings_meta):
        """Apply saved settings back to the UI controls."""
        try:
            algo_idx = self.algorithm_combo.findText(settings_meta.get("algorithm", "UMAP").upper())
            if algo_idx >= 0:
                self.algorithm_combo.setCurrentIndex(algo_idx)
            self.n_neighbors_spin.setValue(settings_meta.get("n_neighbors", 15))
            self.min_dist_spin.setValue(settings_meta.get("min_dist", 0.1) * 100.0)
            if hasattr(self, 'n_epochs_spin') and settings_meta.get("n_epochs"):
                self.n_epochs_spin.setValue(settings_meta["n_epochs"])
            if hasattr(self, 'learning_rate_spin'):
                self.learning_rate_spin.setValue(settings_meta.get("learning_rate", 1.0))
            self.low_memory = settings_meta.get("low_memory", self.low_memory)
            self.n_jobs = settings_meta.get("n_jobs", self.n_jobs)
            metric_idx = self.metric_combo.findText(settings_meta.get("metric", "cosine"))
            if metric_idx >= 0:
                self.metric_combo.setCurrentIndex(metric_idx)
            self.eps_spin.setValue(settings_meta.get("eps", 0.5) * 100.0)
            self.min_samples_spin.setValue(settings_meta.get("min_samples", 10))
            self.min_size_spin.setValue(settings_meta.get("min_size", 0.03))
            self.max_size_spin.setValue(settings_meta.get("max_size", 0.08))
            self.spread_spin.setValue(settings_meta.get("spread", 1.0))
            if hasattr(self, 'min_doc_freq_spin'):
                self.min_doc_freq_spin.setValue(settings_meta.get("min_doc_freq", 3))
            if hasattr(self, 'drop_universal_checkbox'):
                self.drop_universal_checkbox.setChecked(settings_meta.get("drop_universal_tags", False))
        except Exception as e:
            print(f"Error applying session settings: {e}")

    def update_progress(self, percentage, message):
        """Update progress bar, status label, and phase label."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

        # Update phase label based on the message
        if hasattr(self, 'phase_label'):
            phase = self._phase_from_message(message)
            self.phase_label.setText(f"Phase: {phase}")

    def _phase_from_message(self, message):
        """Map a progress message to a pipeline phase name."""
        msg = message.lower()
        if 'connect' in msg:
            return "Connecting to client"
        if 'load' in msg:
            return "Loading file data"
        if 'filter' in msg:
            return "Filtering data"
        if 'vector' in msg:
            return "Vectorizing tags"
        if 'umap' in msg or 'pca' in msg or 'reduc' in msg:
            return "Reducing dimensions"
        if 'cluster' in msg:
            return "Clustering"
        if 'scene' in msg or 'build' in msg:
            return "Building scene"
        if 'complete' in msg or 'done' in msg:
            return "Complete"
        return "Working"

    def on_loading_finished(self, result):
        """Handle loading completion."""
        scene, tag_data = result

        # Sentinel: no files matched the query/filters. Show a friendly status
        # message and keep the existing scene (if any) intact.
        if scene is None and tag_data is None:
            self.load_button.setEnabled(True)
            self.recompute_button.setEnabled(True)
            self.recluster_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
            self.save_session_button.setEnabled(True)
            self.load_session_button.setEnabled(True)
            self._set_cohort_action_buttons(True)
            self.status_label.setText("No files matched the current query/filters. Adjust the query, whitelist/blacklist, or max files and try again.")
            return

        self.tag_data = tag_data  # Store for recomputation
        self.scene_graph = scene  # Store for session save
        self.load_button.setEnabled(True)
        self.recompute_button.setEnabled(True)
        self.recluster_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        self.save_session_button.setEnabled(True)
        self.load_session_button.setEnabled(True)
        self.send_to_tab_btn.setEnabled(True)
        self.time_travel_button.setEnabled(True)
        self._set_cohort_action_buttons(True)
        self.status_label.setText("Ready - Left: cohort | Ctrl+Left: node | Right: move camera | F11: Fullscreen")

        # Re-cluster results keep positions unchanged -> update colors in-place
        # (much lighter than full render_scene, avoids UI lock on 55k nodes).
        if getattr(self, '_pending_recluster', False):
            self._pending_recluster = False
            self.node_list = list(scene.nodes.values())
            self._build_base_scatter()
            self._apply_highlight_colors(self._base_colors_rgba)
            self._update_cohort_table()
            self._update_cohort_labels()
        else:
            self.render_scene(scene)

        # Record the DBSCAN params that produced this scene so the recluster
        # button greys out until the user changes them.
        self._last_used_eps = self.eps_spin.value() / 100.0
        self._last_used_min_samples = self.min_samples_spin.value()
        self._update_recluster_button_state()

    def on_loading_error(self, error_msg):
        """Handle loading error."""
        self.load_button.setEnabled(True)
        self.recompute_button.setEnabled(True)
        self.recluster_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        self.save_session_button.setEnabled(True)
        self.load_session_button.setEnabled(True)
        self._set_cohort_action_buttons(True)
        self.status_label.setText(f"Error: {error_msg}")

    def render_scene(self, scene):
        """Render the scene graph in the 3D view with V3/V5/V6 features."""
        try:
            import pyqtgraph.opengl as gl
            import numpy as np

            if not hasattr(self, 'gl_view') or not hasattr(self, 'gl_scatter'):
                return

            # Clear existing scatter
            if self.gl_scatter:
                self.gl_view.removeItem(self.gl_scatter)

            # Remove old hulls and relationship lines when re-rendering
            self._remove_cluster_hulls()
            self._remove_relationship_lines()

            # Get data
            positions = scene.get_node_positions()
            # Apply spread factor
            spread = float(self.spread_spin.value())
            positions = positions * spread

            # V5: Apply color scheme to node colors
            cluster_ids_set = set()
            for node in scene.nodes.values():
                cluster_ids_set.add(node.cluster_id)
            total_clusters = len(cluster_ids_set) if cluster_ids_set else 1

            # Generate colors based on scheme
            colors_list = []
            for node in scene.nodes.values():
                color = self._get_color_for_cluster(node.cluster_id, total_clusters)
                colors_list.append(color)
            colors = np.array(colors_list)

            sizes = scene.get_node_sizes()

            if len(positions) == 0:
                return

            # Apply transparency
            alpha = self.transparency_spin.value()
            colors_normalized = colors / 255.0
            colors_rgba = np.column_stack([colors_normalized, alpha * np.ones(len(colors))])

            # Create scatter plot
            self.gl_scatter = gl.GLScatterPlotItem(
                pos=positions,
                size=sizes,
                color=colors_rgba,
                pxMode=False
            )
            try:
                self.gl_scatter.sigClicked.connect(self.on_node_clicked)
            except (AttributeError, TypeError):
                pass  # sigClicked not available in this PyQtGraph version
            self.gl_view.addItem(self.gl_scatter)

            # Store file IDs for click handling
            self.file_ids = scene.get_file_ids()
            self.node_list = list(scene.nodes.values())

            # Build base scatter cache for efficient highlight updates
            self._build_base_scatter()

            # Auto-fit camera to data bounds only on the FIRST render.
            # On subsequent renders (re-cluster / recompute / session load),
            # preserve the current camera position so the user's view isn't reset.
            if not getattr(self, '_camera_initialized', False):
                self._fit_camera_to_data(positions)
                self._camera_initialized = True

            # Update cohort table
            self._update_cohort_table()

            # Ensure the 3D view has keyboard focus for F11 fullscreen toggle
            self.gl_view.setFocus()

            # Re-apply cluster boundaries if enabled
            if self.show_boundaries_checkbox.isChecked():
                self._update_cluster_hulls()

            # Re-apply relationship lines if enabled
            if self.show_relationships_checkbox.isChecked():
                self._update_relationship_lines()

            # Re-apply cohort labels if enabled
            if self.show_cohort_labels_checkbox.isChecked():
                self._update_cohort_labels()
                # Ensure the blink timer is running
                if hasattr(self, 'cohort_label_blink_timer'):
                    self.cohort_label_blink_visible = True
                    self.cohort_label_blink_timer.start(500)

            # Compute tag importance ranking
            self._compute_tag_importance()

        except Exception as e:
            import traceback
            print(f"Error rendering scene: {e}")
            traceback.print_exc()

    def _on_supersample_toggle(self, state):
        """Toggle live supersampling: render the view at 4x continuously.

        Renders the GL scene offscreen at 4x via renderToArray on a throttled
        timer, and displays the downsampled result in an overlay label. The
        live view stays visible underneath so the viewer remains responsive.
        """
        if not hasattr(self, 'gl_view') or self.gl_view is None:
            return
        if state == 2:  # Qt.Checked
            from PySide6.QtCore import QTimer
            if not hasattr(self, 'supersample_timer') or self.supersample_timer is None:
                self.supersample_timer = QTimer(self)
                self.supersample_timer.timeout.connect(self._render_supersampled)
            fps = self.supersample_fps_spin.value() if hasattr(self, 'supersample_fps_spin') else 10
            if fps > 0:
                self.supersample_timer.start(int(1000 / fps))
            else:
                self.supersample_timer.start(0)  # 0 = no limiter, render as fast as possible
            self._render_supersampled()
            print(f"Supersample enabled: rendering live at 4x ({fps} fps)")
        else:
            if hasattr(self, 'supersample_timer') and self.supersample_timer:
                self.supersample_timer.stop()
            if hasattr(self, 'supersample_label') and self.supersample_label:
                self.supersample_label.hide()
            if hasattr(self, 'view_3d') and self.view_3d:
                self.view_3d.show()
            print("Supersample disabled")

    def _on_supersample_fps_changed(self, value):
        """Update the supersample timer interval when FPS changes."""
        if not hasattr(self, 'supersample_timer') or self.supersample_timer is None:
            return
        if not self.supersample_checkbox.isChecked():
            return
        if value > 0:
            self.supersample_timer.start(int(1000 / value))
        else:
            self.supersample_timer.start(0)  # 0 = no limiter

    def _render_supersampled(self):
        """Render the GL scene offscreen at 4x and display downsampled.

        Called by the supersample timer to keep the overlay roughly live.
        """
        try:
            from PySide6.QtGui import QImage, QPixmap
            import numpy as np

            if not hasattr(self, 'gl_view') or self.gl_view is None:
                return
            view = self.gl_view
            w = view.width()
            h = view.height()
            if w <= 0 or h <= 0:
                return

            # Render at 4x via renderToArray (returns BGRA uint8)
            arr = view.renderToArray(size=(w * 4, h * 4))
            arr = np.ascontiguousarray(arr)
            # renderToArray returns BGRA; swap R/B for QImage ARGB32
            bgr = arr[..., 0].copy()
            arr[..., 0] = arr[..., 2]
            arr[..., 2] = bgr
            img = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format.Format_ARGB32)
            img = img.copy()  # detach from numpy buffer

            # Downsample to view size with smoothing
            downsampled = img.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            pixmap = QPixmap.fromImage(downsampled)
            if hasattr(self, 'supersample_label') and self.supersample_label:
                self.supersample_label.setPixmap(pixmap)
                self.supersample_label.resize(w, h)
                self.supersample_label.show()
                # Keep live view visible underneath (responsive)
                if hasattr(self, 'view_3d') and self.view_3d:
                    self.view_3d.show()
        except Exception as e:
            import traceback
            print(f"Supersample render failed: {e}")
            traceback.print_exc()

    def _fit_camera_to_data(self, positions):
        """Fit the camera view to the bounding box of the data.
        
        Args:
            positions: np.ndarray of shape (n, 3) with node positions
        """
        import numpy as np
        from PySide6.QtGui import QVector3D
        if len(positions) == 0:
            return
        
        # Calculate bounding box
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        
        # Calculate center and diagonal
        center = (min_pos + max_pos) / 2
        diagonal = np.linalg.norm(max_pos - min_pos)
        
        # Print diagnostic information
        print(f"\n=== 3D Position Analysis ===")
        print(f"Number of nodes: {len(positions)}")
        print(f"Position range X: [{min_pos[0]:.2f}, {max_pos[0]:.2f}]")
        print(f"Position range Y: [{min_pos[1]:.2f}, {max_pos[1]:.2f}]")
        print(f"Position range Z: [{min_pos[2]:.2f}, {max_pos[2]:.2f}]")
        print(f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"Diagonal spread: {diagonal:.2f}")
        print(f"Mean position: ({np.mean(positions[:, 0]):.2f}, {np.mean(positions[:, 1]):.2f}, {np.mean(positions[:, 2]):.2f})")
        print(f"Std deviation: ({np.std(positions[:, 0]):.2f}, {np.std(positions[:, 1]):.2f}, {np.std(positions[:, 2]):.2f})")
        print(f"===========================\n")
        
        # Set camera distance to 1.5x the diagonal for a nice overview
        distance = max(diagonal * 1.5, 10)  # Minimum distance of 10
        
        # Move camera to look at the center of the data
        # Use QVector3D for center (required by PyQtGraph)
        self.gl_view.opts['center'] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
        self.gl_view.opts['distance'] = float(distance)
        self.gl_view.opts['elevation'] = 30
        self.gl_view.opts['azimuth'] = 45
        self.gl_view.update()

    def on_node_clicked(self, scatter, ids):
        """Handle node click event to display file info.

        Args:
            scatter: The scatter plot item that was clicked
            ids: Object IDs that were clicked
        """
        if ids is None or len(ids) == 0:
            return

        # Get the first clicked ID
        node_index = ids[0]
        self.show_node_info(node_index)
    
    def _toggle_selection_highlight(self):
        """Toggle the visual highlight for selected node/cluster."""
        if self.gl_scatter is None or self.node_list is None:
            return
        
        import numpy as np
        
        # Toggle visibility state
        self.selection_visible = not self.selection_visible
        
        if self.selected_node_index is not None:
            # Single node selection
            self._highlight_single_node(self.selected_node_index)
        elif self.selected_cluster_id is not None:
            # Cluster selection
            self._highlight_cluster(self.selected_cluster_id)
    
    def _build_base_scatter(self):
        """Build and cache the base scatter (positions/sizes/base colors).

        Returns the base colors_rgba array (no highlight/dim).
        """
        import numpy as np
        positions = np.array([node.position for node in self.node_list]) * float(self.spread_spin.value())
        min_size = self.min_size_spin.value()
        max_size = self.max_size_spin.value()
        sizes = np.array([max(min_size, min(max_size, len(node.tags) * 0.015)) for node in self.node_list])
        colors = np.array([node.color for node in self.node_list]) / 255.0
        alpha = self.transparency_spin.value()
        colors_rgba = np.column_stack([colors, alpha * np.ones(len(colors))])
        self._base_positions = positions
        self._base_sizes = sizes
        self._base_colors_rgba = colors_rgba
        return positions, sizes, colors_rgba

    def _apply_highlight_colors(self, colors_rgba):
        """Update the existing scatter's colors in-place (no recreate)."""
        if self.gl_scatter is None:
            return
        self.gl_scatter.setData(color=colors_rgba)

    def _highlight_single_node(self, node_index):
        """Apply or remove highlight for a single node (in-place color update)."""
        if not (0 <= node_index < len(self.node_list)):
            return
        import numpy as np

        # Use cached base colors if available, else build
        if not hasattr(self, '_base_colors_rgba') or self._base_colors_rgba is None:
            self._build_base_scatter()
        colors_rgba = self._base_colors_rgba.copy()

        # Dimming is persistent while a selection is active (independent of blink).
        # Vectorized: dim every row's alpha except the selected node.
        if self.dim_non_selected_checkbox.isChecked():
            dim_alpha = self.dim_alpha_spin.value()
            mask = np.ones(len(colors_rgba), dtype=bool)
            mask[node_index] = False
            colors_rgba[mask, 3] = dim_alpha

        # Blink only the selected node highlight
        if self.selection_visible:
            colors_rgba[node_index] = [1.0, 1.0, 0.0, 1.0]  # Bright yellow, fully opaque

        # Update colors in-place (no scatter recreate)
        self._apply_highlight_colors(colors_rgba)

    def _highlight_cluster(self, cluster_id):
        """Apply or remove highlight for all nodes in a cluster (in-place)."""
        import numpy as np

        # Use cached base colors if available, else build
        if not hasattr(self, '_base_colors_rgba') or self._base_colors_rgba is None:
            self._build_base_scatter()
        colors_rgba = self._base_colors_rgba.copy()

        # Build the cluster-id lookup once (vectorized masks below).
        cluster_ids = np.array([node.cluster_id for node in self.node_list])

        # Dimming is persistent while a selection is active (independent of blink).
        # Vectorized: dim every node NOT in the selected cluster.
        if self.dim_non_selected_checkbox.isChecked():
            colors_rgba[cluster_ids != cluster_id, 3] = self.dim_alpha_spin.value()

        # Blink only the selected cluster highlight
        if self.selection_visible:
            colors_rgba[cluster_ids == cluster_id] = [0.0, 1.0, 1.0, 1.0]  # Bright cyan, fully opaque

        # Update colors in-place (no scatter recreate)
        self._apply_highlight_colors(colors_rgba)
    
    def _update_scatter_plot(self, positions, sizes, colors_rgba):
        """Update the scatter plot with new positions, sizes, and colors."""
        import pyqtgraph.opengl as gl
        
        # Remove old scatter (guard against already-removed item)
        if self.gl_scatter is not None:
            try:
                self.gl_view.removeItem(self.gl_scatter)
            except (ValueError, KeyError):
                pass  # Item already removed
        
        # Create new scatter
        self.gl_scatter = gl.GLScatterPlotItem(
            pos=positions,
            size=sizes,
            color=colors_rgba,
            pxMode=False
        )
        try:
            self.gl_scatter.sigClicked.connect(self.on_node_clicked)
        except (AttributeError, TypeError):
            pass
        self.gl_view.addItem(self.gl_scatter)
    
    def _reapply_selection_style(self):
        """Re-apply the current selection highlight/dim style to the scatter."""
        if self.gl_scatter is None or self.node_list is None:
            return
        if self.selected_node_index is not None:
            self._highlight_single_node(self.selected_node_index)
        elif self.selected_cluster_id is not None:
            self._highlight_cluster(self.selected_cluster_id)

    def _on_dim_toggle_changed(self):
        """Handle dim non-selected toggle changes."""
        self._reapply_selection_style()
        # Also dim/undim the cohort labels of non-selected cohorts
        self._update_cohort_labels()

    def _on_dim_alpha_changed(self):
        """Handle dim alpha parameter changes."""
        self._reapply_selection_style()
        # Also update the cohort label dim alpha
        self._update_cohort_labels()

    def toggle_split_window(self):
        """Toggle the split image preview window (F4)."""
        if self.split_window is None:
            self.split_window = TagMap3DSplitWindow(self)
            self.split_window.show()
            self._sync_split_window()
        else:
            self.split_window.close()
            self.split_window = None

    def _sync_split_window(self):
        """Sync the split window display based on current selection state."""
        if self.split_window is None or not self.split_window.isVisible():
            return
        if not hasattr(self, 'node_list') or not self.node_list:
            return

        self.split_window.clear_grid()

        # Determine selection state
        file_ids = self._get_selected_file_ids()

        if self.selected_node_index is not None and file_ids:
            # Single file selected
            self.split_window.set_title(f"File {file_ids[0]}")
            self._load_single_file(file_ids[0])
        elif self.selected_cluster_id is not None and file_ids:
            # Cohort selected - grid of thumbnails
            self.split_window.set_title(f"Cohort {self.selected_cluster_id} - {len(file_ids)} files")
            self._load_file_grid(file_ids)
        else:
            # Nothing selected - one representative image per cohort
            self.split_window.set_title("All Cohorts")
            self._load_cohort_representatives()

    def _load_single_file(self, file_id):
        """Load a single full-res file into the split window asynchronously."""
        self._pending_single_file_id = file_id
        self.single_loader = SingleFileLoader(
            self.client_combo.currentText(),
            file_id,
            parent=self
        )
        self.single_loader.pixmap_ready.connect(self._on_single_pixmap_ready)
        self.single_loader.start()

    def _on_single_pixmap_ready(self, pixmap, tooltip):
        """Display a loaded full-res single file (main thread)."""
        if self.split_window is None or not self.split_window.isVisible():
            return
        # Guard: selection may have changed while the file was loading
        if self.selected_node_index is None:
            return
        file_ids = self._get_selected_file_ids()
        if file_ids and file_ids[0] != getattr(self, '_pending_single_file_id', None):
            return
        self.split_window.show_single_image(pixmap, tooltip=tooltip)

    def _load_file_grid(self, file_ids):
        """Load a grid of thumbnails for the given file IDs asynchronously."""
        max_files = self.split_window.max_files_spin.value() if hasattr(self.split_window, 'max_files_spin') else 60
        image_size = self.split_window.image_size_spin.value() if hasattr(self.split_window, 'image_size_spin') else 200
        self.split_loader = SplitWindowLoader(
            self.client_combo.currentText(),
            file_ids[:max_files],
            image_size,
            parent=self
        )
        self.split_loader.pixmap_ready.connect(self._on_split_pixmap_ready)
        self.split_loader.start()

    def _on_split_pixmap_ready(self, pixmap, tooltip):
        """Add a loaded pixmap to the split window (main thread)."""
        if self.split_window is None or not self.split_window.isVisible():
            return
        self.split_window.add_image(pixmap, tooltip=tooltip)

    def _load_cohort_representatives(self):
        """Load one representative image per cohort asynchronously (clickable tiles)."""
        # Group nodes by cluster
        cluster_nodes = {}
        for node in self.node_list:
            if node.cluster_id != -1:
                cluster_nodes.setdefault(node.cluster_id, []).append(node)

        # Build representative file IDs with cluster info
        rep_file_ids = []
        cluster_map = {}  # file_id -> cluster_id
        for cluster_id, nodes in cluster_nodes.items():
            rep_node = nodes[0]
            rep_file_ids.append(rep_node.file_id)
            cluster_map[rep_node.file_id] = cluster_id

        image_size = self.split_window.image_size_spin.value() if hasattr(self.split_window, 'image_size_spin') else 200
        self.split_loader = SplitWindowLoader(
            self.client_combo.currentText(),
            rep_file_ids,
            image_size,
            parent=self
        )
        self.split_loader.pixmap_ready.connect(self._on_cohort_pixmap_ready)
        self.split_loader.start()
        self._cohort_cluster_map = cluster_map

    def _on_cohort_pixmap_ready(self, pixmap, tooltip):
        """Add a cohort representative tile to the split window (main thread)."""
        if self.split_window is None or not self.split_window.isVisible():
            return
        # Extract file_id from tooltip to look up cluster
        file_id = tooltip.replace("File ", "")
        cluster_id = self._cohort_cluster_map.get(file_id)
        if cluster_id is not None:
            self.split_window.add_cohort_tile(cluster_id, pixmap, tooltip=tooltip)

    def _move_camera_to_cluster(self, cluster_id):
        """Move the 3D camera to focus on a specific cluster (future interaction)."""
        if not hasattr(self, 'node_list') or not self.node_list:
            return
        import numpy as np
        cluster_nodes = [n for n in self.node_list if n.cluster_id == cluster_id]
        if not cluster_nodes:
            return
        positions = np.array([n.position for n in cluster_nodes])
        centroid = positions.mean(axis=0)
        spread = float(self.spread_spin.value())
        centroid = centroid * spread
        self._set_camera_to_waypoint(centroid, 1.0)
        self.status_label.setText(f"Camera moved to Cohort {cluster_id}")

    def _restore_base_colors(self):
        """Restore the scatter to base colors (no dimming/highlight)."""
        if self.gl_scatter is None or self.node_list is None:
            return
        # Use cached base colors if available, else build
        if not hasattr(self, '_base_colors_rgba') or self._base_colors_rgba is None:
            self._build_base_scatter()
        self._apply_highlight_colors(self._base_colors_rgba.copy())

    def clear_selection(self):
        """Clear the current selection and stop blinking."""
        self.selection_timer.stop()
        self.selected_node_index = None
        self.selected_cluster_id = None
        self.selection_visible = False
        # Clear tag state and widgets
        self.tag_query_states.clear()
        self._clear_tag_widgets()
        # Clear the cohort tag data panel
        if hasattr(self, 'selection_tags_text'):
            self.selection_tags_text.clear()

        # Disable cut/pop/re-cluster buttons (no cohort selected)
        if hasattr(self, 'cut_button'):
            self.cut_button.setEnabled(False)
        if hasattr(self, 'pop_button'):
            self.pop_button.setEnabled(False)
        if hasattr(self, 'cluster_button'):
            self.cluster_button.setEnabled(False)

        # Restore base colors (remove dimming/highlight)
        self._restore_base_colors()

        # Re-render labels (removes the forced label of the previously selected cohort)
        self._update_cohort_labels()

        # Sync split window if open
        self._sync_split_window()
    
    def show_node_info(self, node_index):
        """Display file info for a given node index.
        
        Args:
            node_index: Index into the node_list
        """
        # Clear previous selection
        self.clear_selection()
        
        if not hasattr(self, 'node_list') or not self.node_list:
            return
        
        if 0 <= node_index < len(self.node_list):
            node = self.node_list[node_index]
            
            # Clear tag state for new selection
            self.tag_query_states.clear()
            self._clear_tag_widgets()
            
            # Update text info (without tags)
            info_lines = [
                f"File ID: {node.file_id}",
                f"Cluster: {node.cluster_id}",
                f"Score: {node.score:.2f}",
                f"Position: ({node.position[0]:.1f}, {node.position[1]:.1f}, {node.position[2]:.1f})",
            ]
            self.info_text.setText("\n".join(info_lines))
            
            # Create clickable tag widgets
            tags_to_show = node.tags[:50]  # Limit to 50 tags for performance
            if self.tag_interner:
                tags_to_show = self.tag_interner.strings_to_list(tags_to_show)
            if len(node.tags) > 50:
                # Add a label to indicate more tags exist
                more_label = QLabel(f"... and {len(node.tags) - 50} more tags")
                more_label.setStyleSheet(f"color: {GRAY_33}; font-size: 10px; font-style: italic;")
                self.tag_grid.addWidget(more_label, 0, 0)
                self._tag_widgets.append(more_label)
            
            included_query, excluded_query, or_query = self._get_query_tag_states()
            for idx, tag in enumerate(tags_to_show):
                tag_widget = ClickableTag(tag)
                tag_widget.stateChanged.connect(self._on_tag_state_changed)
                # Initialize state from the current query so already-queried tags show correctly
                if tag in excluded_query:
                    tag_widget.state = 2
                elif tag in or_query:
                    tag_widget.state = 3
                elif tag in included_query:
                    tag_widget.state = 1
                else:
                    tag_widget.state = 0
                tag_widget._update_appearance()
                self.tag_query_states[tag] = tag_widget.state
                self._tag_widgets.append(tag_widget)
                self.tag_grid.addWidget(tag_widget, idx // 3, idx % 3)  # 3 columns
            
            # Force layout refresh so the tags render immediately
            self.tag_container.update()
            self.tag_grid.update()

            # Set up selection highlight
            self.selected_node_index = node_index
            self.selection_timer.start(500)  # 500ms interval for ~2Hz blink

            # Update the cohort tag data panel with the node's cluster cohort
            cluster_nodes = [n for n in self.node_list if n.cluster_id == node.cluster_id]
            self._update_selection_tags(cluster_nodes)

            # Ensure the selected node's cohort gets a label
            self._update_cohort_labels()

            # Sync split window if open
            self._sync_split_window()

    def show_cluster_info(self, node_index):
        """Display info for all nodes in the same cluster as the given node.
        
        Args:
            node_index: Index into the node_list
        """
        # Clear previous selection
        self.clear_selection()
        
        if not hasattr(self, 'node_list') or not self.node_list:
            return
        
        if 0 <= node_index < len(self.node_list):
            clicked_node = self.node_list[node_index]
            cluster_id = clicked_node.cluster_id
            
            # Find all nodes in the same cluster
            cluster_nodes = [n for n in self.node_list if n.cluster_id == cluster_id]
            
            if not cluster_nodes:
                return
            
            # Clear tag state for new selection
            self.tag_query_states.clear()
            self._clear_tag_widgets()
            
            # Count tag occurrences across files in cluster
            tag_counts = {}
            for n in cluster_nodes:
                for tag in n.tags:
                    if self.tag_interner and isinstance(tag, int):
                        tag = self.tag_interner.index_to_string(tag)
                    if tag is None:
                        continue
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Sort by count (descending), then alphabetically for ties
            sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
            
            # Update text info
            info_lines = [
                f"Cluster {cluster_id} - {len(cluster_nodes)} files",
                f"",
                f"Common Tags ({len(tag_counts)}):",
            ]
            self.info_text.setText("\n".join(info_lines))
            
            # Create clickable tag widgets for top tags
            tags_to_show = sorted_tags[:50]  # Limit to 50 tags
            if len(sorted_tags) > 50:
                more_label = QLabel(f"... and {len(sorted_tags) - 50} more tags")
                more_label.setStyleSheet(f"color: {GRAY_33}; font-size: 10px; font-style: italic;")
                self.tag_grid.addWidget(more_label, 0, 0)
                self._tag_widgets.append(more_label)
            
            included_query, excluded_query, or_query = self._get_query_tag_states()
            for idx, (tag, count) in enumerate(tags_to_show):
                tag_widget = ClickableTag(tag)
                tag_widget.setToolTip(f"Appears in {count} files ({count/len(cluster_nodes)*100:.0f}%)")
                tag_widget.stateChanged.connect(self._on_tag_state_changed)
                # Initialize state from the current query so already-queried tags show correctly
                if tag in excluded_query:
                    tag_widget.state = 2
                elif tag in or_query:
                    tag_widget.state = 3
                elif tag in included_query:
                    tag_widget.state = 1
                else:
                    tag_widget.state = 0
                tag_widget._update_appearance()
                self.tag_query_states[tag] = tag_widget.state
                self._tag_widgets.append(tag_widget)
                self.tag_grid.addWidget(tag_widget, idx // 3, idx % 3)  # 3 columns
            
            # Force layout refresh so the tags render immediately
            self.tag_container.update()
            self.tag_grid.update()

            # Set up cluster selection highlight
            self.selected_cluster_id = cluster_id
            self.selection_timer.start(500)  # 500ms interval for ~2Hz blink

            # Enable cut/pop/re-cluster buttons for the selected cohort
            if hasattr(self, 'cut_button'):
                self.cut_button.setEnabled(True)
            if hasattr(self, 'pop_button'):
                self.pop_button.setEnabled(True)
            if hasattr(self, 'cluster_button'):
                self.cluster_button.setEnabled(True)

            # Update the cohort tag data panel
            self._update_selection_tags(cluster_nodes)

            # Ensure the selected cohort gets a label (even if filtered out by label mode)
            self._update_cohort_labels()

            # Sync split window if open
            self._sync_split_window()

    def _clear_tag_widgets(self):
        """Clear all tag widgets from the grid layout synchronously."""
        # Clear the reference list first
        self._tag_widgets.clear()
        # Synchronously remove all items from the grid layout
        while self.tag_grid.count():
            item = self.tag_grid.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.hide()  # Hide immediately
                    widget.deleteLater()  # Schedule for cleanup
        # Reset container height so the layout reflows on next add
        self.tag_container.setMinimumHeight(50)
    
    def _on_tag_state_changed(self, tag_name, new_state):
        """Handle tag state change and update query field."""
        self.tag_query_states[tag_name] = new_state
        self._update_query_from_tags()
    
    def _update_query_from_tags(self):
        """Update the query_edit field based on tag states.

        Preserves query parts that are not managed by clickable tags
        (e.g., system:inbox) so they are not overwritten.

        OR-state tags (state 3) are grouped into a single bracket segment
        `[TagA, TagB]` which Hydrus interprets as an OR group. If an OR
        bracket already exists, new OR tags are appended to the FIRST
        bracket; additional brackets are left untouched.
        """
        # Parse the current query into managed/unmanaged parts.
        # We need to handle bracket groups (OR segments) as single units.
        current_query = self.query_edit.text().strip()
        unmanaged_parts = []
        or_bracket_index = None  # index in unmanaged_parts where the first managed OR bracket sits
        if current_query:
            # Split by comma, but keep bracket groups intact.
            parts = self._split_query_preserving_brackets(current_query)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # Determine if this part is an OR bracket group
                if part.startswith('[') and part.endswith(']'):
                    # It's an OR bracket group. Check if it's managed (all tags in states)
                    inner = part[1:-1].strip()
                    tags_in_group = [t.strip() for t in inner.split(',') if t.strip()]
                    # If ALL tags in the group are managed by clickable tags, it's managed
                    if tags_in_group and all(t in self.tag_query_states for t in tags_in_group):
                        # Only the FIRST managed OR bracket is rebuilt from OR states.
                        # Additional managed OR brackets are preserved as-is (ignored).
                        if or_bracket_index is None:
                            or_bracket_index = len(unmanaged_parts)
                            continue
                        else:
                            unmanaged_parts.append(part)
                            continue
                    else:
                        unmanaged_parts.append(part)
                        continue
                # Determine the tag name (strip leading '-' for excluded)
                tag_name = part[1:] if part.startswith('-') else part
                # If this tag is managed by clickable tags, skip it (will rebuild from states)
                if tag_name in self.tag_query_states:
                    continue
                unmanaged_parts.append(part)

        # Build managed parts from current tag states.
        # OR tags that appear in preserved (non-first) OR brackets are excluded
        # from the first bracket's OR collection (per requirement: only the
        # first OR bracket is managed; additional brackets are ignored).
        preserved_or_tags = set()
        for part in unmanaged_parts:
            if part.startswith('[') and part.endswith(']'):
                inner = part[1:-1].strip()
                for t in inner.split(','):
                    t = t.strip()
                    if t and not t.startswith('-'):
                        preserved_or_tags.add(t)

        included_tags = []
        excluded_tags = []
        or_tags = []
        for tag, state in self.tag_query_states.items():
            if state == 1:  # Included
                included_tags.append(tag)
            elif state == 2:  # Excluded
                excluded_tags.append(f"-{tag}")
            elif state == 3:  # OR
                if tag not in preserved_or_tags:
                    or_tags.append(tag)

        # Build the OR bracket segment
        or_segment = None
        if or_tags:
            or_segment = f"[{', '.join(or_tags)}]"

        # Combine: unmanaged (preserved) + managed (included then excluded)
        query_parts = unmanaged_parts + included_tags + excluded_tags

        # Insert OR segment at the first OR bracket position if one exists, else append
        if or_segment:
            if or_bracket_index is not None:
                # Replace the managed OR bracket with the rebuilt OR segment
                query_parts.insert(or_bracket_index, or_segment)
            else:
                # No OR bracket exists - append as new bracket segment
                query_parts.append(or_segment)

        query_text = ", ".join(query_parts)
        self.query_edit.setText(query_text)

    def _split_query_preserving_brackets(self, query):
        """Split a query string by commas, keeping bracket groups intact.

        Returns:
            list: Parts split by top-level commas (bracket groups stay whole strings).
        """
        parts = []
        depth = 0
        current = []
        for ch in query:
            if ch == '[':
                depth += 1
                current.append(ch)
            elif ch == ']':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current).strip())
        return parts

    def _query_to_api_tags(self, query):
        """Convert a query string to API-ready tags list.

        Bracket groups (OR segments) are converted to nested lists so the
        Hydrus API interprets them as OR groups.
        """
        parts = self._split_query_preserving_brackets(query)
        api_tags = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith('[') and part.endswith(']'):
                inner = part[1:-1].strip()
                tags = [t.strip() for t in inner.split(',') if t.strip()]
                api_tags.append(tags)
            else:
                api_tags.append(part)
        return api_tags

    def _get_query_tag_states(self):
        """Parse the current query_edit into included/excluded/OR tag sets.

        Returns:
            tuple: (included_tags, excluded_tags, or_tags) sets parsed from the query.
        """
        query = self.query_edit.text().strip()
        included = set()
        excluded = set()
        or_tags = set()
        if not query:
            return included, excluded, or_tags
        for part in self._split_query_preserving_brackets(query):
            part = part.strip()
            if not part:
                continue
            # Handle OR bracket group
            if part.startswith('[') and part.endswith(']'):
                inner = part[1:-1].strip()
                for tag in inner.split(','):
                    tag = tag.strip()
                    if not tag:
                        continue
                    if tag.startswith('-'):
                        excluded.add(tag[1:].strip())
                    else:
                        or_tags.add(tag.strip())
                continue
            if part.startswith('-'):
                excluded.add(part[1:].strip())
            else:
                included.add(part.strip())
        return included, excluded, or_tags

    def _on_size_changed(self):
        """Handle size parameter changes to update scatter dynamically."""
        if self.gl_scatter and hasattr(self, 'node_list') and self.node_list:
            import numpy as np
            import pyqtgraph.opengl as gl
            min_size = self.min_size_spin.value()
            max_size = self.max_size_spin.value()
            alpha = self.transparency_spin.value()
            
            # Recalculate sizes based on tag counts
            sizes = np.array([max(min_size, min(max_size, len(node.tags) * 0.015)) for node in self.node_list])
            
            # Remove old scatter and create new one with updated sizes
            self.gl_view.removeItem(self.gl_scatter)
            
            positions = np.array([node.position for node in self.node_list])
            colors = np.array([node.color for node in self.node_list]) / 255.0
            colors_rgba = np.column_stack([colors, alpha * np.ones(len(colors))])
            
            self.gl_scatter = gl.GLScatterPlotItem(
                pos=positions,
                size=sizes,
                color=colors_rgba,
                pxMode=False
            )
            try:
                self.gl_scatter.sigClicked.connect(self.on_node_clicked)
            except (AttributeError, TypeError):
                pass
            self.gl_view.addItem(self.gl_scatter)
            self._build_base_scatter()

        # Re-apply selection style if a selection is active
        self._reapply_selection_style()

    def _on_spread_changed(self):
        """Handle spread parameter changes to update positions dynamically."""
        if self.gl_scatter and hasattr(self, 'node_list') and self.node_list:
            import numpy as np
            import pyqtgraph.opengl as gl
            spread = float(self.spread_spin.value())
            min_size = self.min_size_spin.value()
            max_size = self.max_size_spin.value()
            alpha = self.transparency_spin.value()
            
            # Recalculate positions with spread
            positions = np.array([node.position for node in self.node_list]) * spread
            sizes = np.array([max(min_size, min(max_size, len(node.tags) * 0.015)) for node in self.node_list])
            colors = np.array([node.color for node in self.node_list]) / 255.0
            colors_rgba = np.column_stack([colors, alpha * np.ones(len(colors))])
            
            # Remove old scatter and create new one
            self.gl_view.removeItem(self.gl_scatter)
            self.gl_scatter = gl.GLScatterPlotItem(
                pos=positions,
                size=sizes,
                color=colors_rgba,
                pxMode=False
            )
            try:
                self.gl_scatter.sigClicked.connect(self.on_node_clicked)
            except (AttributeError, TypeError):
                pass
            self.gl_view.addItem(self.gl_scatter)
            self._build_base_scatter()

            # Re-apply relationship lines if visible (positions depend on spread)
            if self.show_relationships_checkbox.isChecked():
                self._update_relationship_lines()

        # Re-apply selection style if a selection is active
        self._reapply_selection_style()

    def _on_transparency_changed(self):
        """Handle transparency parameter changes to update scatter dynamically."""
        if self.gl_scatter and hasattr(self, 'node_list') and self.node_list:
            import numpy as np
            import pyqtgraph.opengl as gl
            spread = float(self.spread_spin.value())
            min_size = self.min_size_spin.value()
            max_size = self.max_size_spin.value()
            alpha = self.transparency_spin.value()
            
            # Recalculate positions with spread
            positions = np.array([node.position for node in self.node_list]) * spread
            sizes = np.array([max(min_size, min(max_size, len(node.tags) * 0.015)) for node in self.node_list])
            colors = np.array([node.color for node in self.node_list]) / 255.0
            colors_rgba = np.column_stack([colors, alpha * np.ones(len(colors))])
            
            # Remove old scatter and create new one
            self.gl_view.removeItem(self.gl_scatter)
            self.gl_scatter = gl.GLScatterPlotItem(
                pos=positions,
                size=sizes,
                color=colors_rgba,
                pxMode=False
            )
            try:
                self.gl_scatter.sigClicked.connect(self.on_node_clicked)
            except (AttributeError, TypeError):
                pass
            self.gl_view.addItem(self.gl_scatter)
            self._build_base_scatter()

        # Re-apply selection style if a selection is active
        self._reapply_selection_style()

    def _on_wobble_toggle(self, state):
        """Handle wobble enable/disable toggle."""
        from PySide6.QtCore import Qt
        if state == Qt.CheckState.Checked.value:
            self.wobble_timer.start()
            print("Camera wobble enabled")
        else:
            self.wobble_timer.stop()
            print("Camera wobble disabled")

    def _on_trail_toggle(self, state):
        """Handle trail enable/disable toggle."""
        from PySide6.QtCore import Qt
        enabled = state == Qt.CheckState.Checked.value
        self.trail_active = enabled
        # Enable/disable trail parameter spins
        self.trail_length_spin.setEnabled(enabled)
        self.trail_decay_spin.setEnabled(enabled)
        self.trail_max_nodes_spin.setEnabled(enabled)
        if enabled:
            self.trail_history = []
            self._ensure_trail_scatter()
        else:
            self._clear_trails()
        print(f"Trails {'enabled' if enabled else 'disabled'}")

    def _ensure_trail_scatter(self):
        """Create the trail scatter item if it doesn't exist."""
        import pyqtgraph.opengl as gl
        if self.trail_scatter is None and hasattr(self, 'gl_view'):
            self.trail_scatter = gl.GLScatterPlotItem()
            self.trail_scatter.setGLOptions('translucent')
            self.gl_view.addItem(self.trail_scatter)

    def _clear_trails(self):
        """Remove trail scatter and reset history."""
        import numpy as np
        self.trail_history = []
        if self.trail_scatter is not None:
            self.trail_scatter.setData(pos=np.zeros((0, 3)))
            self.trail_scatter.setVisible(False)

    def _update_trails(self):
        """Capture the camera position and render a fading trail of its path.

        The nodes themselves are static; it is the CAMERA that moves during
        wobble. So the trail traces the camera's position over time, giving a
        visible path of the wobble motion.
        """
        import numpy as np
        if not self.trail_active or not hasattr(self, 'gl_view'):
            return
        self._ensure_trail_scatter()
        if self.trail_scatter is None:
            return

        trail_len = self.trail_length_spin.value()
        decay = self.trail_decay_spin.value()

        # Capture the current camera position (center + distance/elevation/azimuth)
        center = self.gl_view.opts.get('center', np.array([0.0, 0.0, 0.0]))
        distance = float(self.gl_view.opts.get('distance', 200))
        elevation = float(self.gl_view.opts.get('elevation', 30))
        azimuth = float(self.gl_view.opts.get('azimuth', 45))

        # Convert spherical camera params to a 3D position
        import math
        elev_rad = math.radians(elevation)
        azim_rad = math.radians(azimuth)
        cam_pos = np.array([
            float(center[0]) + distance * math.cos(elev_rad) * math.cos(azim_rad),
            float(center[1]) + distance * math.cos(elev_rad) * math.sin(azim_rad),
            float(center[2]) + distance * math.sin(elev_rad),
        ])

        # Ring buffer of recent camera positions
        self.trail_history.append(cam_pos)
        if len(self.trail_history) > trail_len:
            self.trail_history.pop(0)

        # Build trail points with decaying alpha
        all_pos = []
        all_color = []
        all_size = []
        n_hist = len(self.trail_history)
        for k, hist_pos in enumerate(self.trail_history):
            alpha = decay ** (n_hist - 1 - k)
            if alpha < 0.05:
                continue
            all_pos.append(hist_pos)
            all_color.append((1.0, 1.0, 1.0, alpha))
            all_size.append(2.0)

        if all_pos:
            self.trail_scatter.setData(
                pos=np.array(all_pos),
                color=np.array(all_color),
                size=np.array(all_size),
            )
            self.trail_scatter.setVisible(True)
        else:
            self.trail_scatter.setVisible(False)

    def _update_wobble(self):
        """Update camera position for wobble effect."""
        import numpy as np
        if not hasattr(self, 'gl_view'):
            return
        
        # Increment time
        self.wobble_time += 0.016  # ~60fps timestep
        
        # Get wobble parameters
        speed = self.wobble_speed_spin.value()
        x_range = self.wobble_x_range_spin.value()
        y_range = self.wobble_y_range_spin.value()
        z_range = self.wobble_z_range_spin.value()
        azim_range = self.wobble_azim_range_spin.value()
        elev_range = self.wobble_elev_range_spin.value()
        
        # Calculate oscillation using sine waves with different frequencies for each axis
        t = self.wobble_time * speed
        
        # Get current camera state
        current_center = self.gl_view.opts.get('center', np.array([0, 0, 0]))
        current_distance = self.gl_view.opts.get('distance', 200)
        current_elevation = self.gl_view.opts.get('elevation', 30)
        current_azimuth = self.gl_view.opts.get('azimuth', 45)
        
        # Apply wobble offsets
        new_distance = current_distance + np.sin(t * 1.3) * z_range
        new_elevation = current_elevation + np.sin(t * 0.7) * elev_range
        new_azimuth = current_azimuth + np.sin(t * 0.5) * azim_range
        
        # Update camera position
        self.gl_view.opts['distance'] = new_distance
        self.gl_view.opts['elevation'] = new_elevation
        self.gl_view.opts['azimuth'] = new_azimuth
        self.gl_view.update()

        # Render trails if enabled
        self._update_trails()

    def keyPressEvent(self, event):
        """Handle key press events for fullscreen toggle and shortcuts."""
        # Ctrl+E: send selected cohort to Hydrus tab
        if event.key() == Qt.Key_E and event.modifiers() & Qt.ControlModifier:
            self.send_selected_to_tab()
            event.accept()
            return
        if event.key() == Qt.Key_F11:
            # Toggle fullscreen on the main window
            if self.main_window.isFullScreen():
                self.main_window.showNormal()
                print("Exited fullscreen")
            else:
                self.main_window.showFullScreen()
                print("Entered fullscreen")
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse click to select nodes."""
        # This would need proper ray-casting implementation
        # For now, just call parent
        super().mousePressEvent(event)

    def _get_color_for_cluster(self, cluster_id, total_clusters):
        """Get RGB color for a cluster based on the current color scheme.

        Args:
            cluster_id: Cluster ID (-1 for noise)
            total_clusters: Total number of clusters (for normalization)

        Returns:
            Tuple of (R, G, B) in 0-255 range
        """
        if cluster_id == -1:
            return (169, 169, 169)  # Gray for noise regardless of scheme

        scheme = self.color_scheme_combo.currentText()

        if scheme == "Pastel":
            # Use the default pastel colors from SceneGraph
            from src.core.models import SceneGraph
            colors = SceneGraph.CLUSTER_COLORS
            return colors[cluster_id % len(colors)]

        # Matplotlib-based colormaps
        try:
            import matplotlib.cm as cm
            import numpy as np

            # Get the colormap
            cmap_name = scheme.lower()
            cmap = cm.get_cmap(cmap_name)

            # Normalize cluster_id to 0-1 range
            if total_clusters > 1:
                t = cluster_id / (total_clusters - 1)
            else:
                t = 0.5

            # Get color from colormap (returns RGBA 0-1)
            rgba = cmap(t)
            r = int(rgba[0] * 255)
            g = int(rgba[1] * 255)
            b = int(rgba[2] * 255)
            return (r, g, b)
        except (ImportError, ValueError):
            # Fallback to pastel colors if matplotlib fails
            from src.core.models import SceneGraph
            colors = SceneGraph.CLUSTER_COLORS
            return colors[cluster_id % len(colors)]

    def _on_color_scheme_changed(self):
        """Handle color scheme dropdown change - recolor all nodes."""
        if not hasattr(self, 'node_list') or not self.node_list:
            return

        # Count unique clusters for normalization
        cluster_ids = set()
        for node in self.node_list:
            cluster_ids.add(node.cluster_id)
        total_clusters = len(cluster_ids) if cluster_ids else 1

        import numpy as np
        import pyqtgraph.opengl as gl

        spread = float(self.spread_spin.value())
        min_size = self.min_size_spin.value()
        max_size = self.max_size_spin.value()
        alpha = self.transparency_spin.value()

        # Recalculate positions
        positions = np.array([node.position for node in self.node_list]) * spread
        sizes = np.array([max(min_size, min(max_size, len(node.tags) * 0.015)) for node in self.node_list])

        # Generate new colors based on scheme
        colors = []
        for node in self.node_list:
            color = self._get_color_for_cluster(node.cluster_id, total_clusters)
            colors.append(color)
        colors = np.array(colors) / 255.0
        colors_rgba = np.column_stack([colors, alpha * np.ones(len(colors))])

        # Update node colors in the data for future reference
        for i, node in enumerate(self.node_list):
            node.color = tuple(int(c * 255) for c in colors[i])

        # Remove old scatter and create new one
        if self.gl_scatter:
            self.gl_view.removeItem(self.gl_scatter)
        self.gl_scatter = gl.GLScatterPlotItem(
            pos=positions,
            size=sizes,
            color=colors_rgba,
            pxMode=False
        )
        try:
            self.gl_scatter.sigClicked.connect(self.on_node_clicked)
        except (AttributeError, TypeError):
            pass
        self.gl_view.addItem(self.gl_scatter)
        self._build_base_scatter()

        # Update hull colors if visible
        if self.show_boundaries_checkbox.isChecked():
            self._update_cluster_hulls()

        # Update relationship line colors if visible
        if self.show_relationships_checkbox.isChecked():
            self._update_relationship_lines()

    # Cluster Boundary Meshes (Convex Hulls)

    def _on_boundaries_toggle(self, state):
        """Handle cluster boundaries checkbox toggle."""
        from PySide6.QtCore import Qt
        if state == Qt.CheckState.Checked.value:
            self._update_cluster_hulls()
            print("Cluster boundaries enabled")
        else:
            self._remove_cluster_hulls()
            print("Cluster boundaries disabled")

    def _on_boundary_alpha_changed(self):
        """Handle boundary opacity changes - rebuild hulls."""
        if self.show_boundaries_checkbox.isChecked():
            self._update_cluster_hulls()

    def _update_cluster_hulls(self):
        """Create convex hull meshes around clusters with 10+ nodes."""
        # Remove existing hulls first
        self._remove_cluster_hulls()

        if not hasattr(self, 'node_list') or not self.node_list:
            return

        try:
            import pyqtgraph.opengl as gl
            import numpy as np
            from scipy.spatial import ConvexHull
            from collections import defaultdict

            # Group nodes by cluster
            cluster_nodes = defaultdict(list)
            for node in self.node_list:
                if node.cluster_id != -1:  # Skip noise
                    cluster_nodes[node.cluster_id].append(node)

            spread = float(self.spread_spin.value())

            for cluster_id, nodes in cluster_nodes.items():
                if len(nodes) < 10:
                    continue  # Skip small clusters

                # Get 3D positions
                positions = np.array([n.position for n in nodes]) * spread

                # Compute convex hull
                hull = ConvexHull(positions)

                # Get cluster color
                cluster_ids = set(n.cluster_id for n in self.node_list)
                total_clusters = len(cluster_ids) if cluster_ids else 1
                color = self._get_color_for_cluster(cluster_id, total_clusters)
                color_normalized = np.array(color) / 255.0

                # Create mesh from hull vertices and simplices
                vertices = positions[hull.vertices]
                faces = hull.simplices

                # Create mesh item with configurable opacity
                boundary_alpha = self.boundary_alpha_spin.value() if hasattr(self, 'boundary_alpha_spin') else 0.15
                mesh = gl.GLMeshItem(
                    vertexes=vertices,
                    faces=faces,
                    color=(*color_normalized, boundary_alpha),  # R, G, B, A
                    shader="color",
                    renderMode="solid"
                )
                self.gl_view.addItem(mesh)
                self.cluster_hull_meshes.append(mesh)

            print(f"Created {len(self.cluster_hull_meshes)} cluster hull meshes")
        except ImportError as e:
            print(f"Error updating cluster hulls (missing dependency): {e}")
        except Exception as e:
            print(f"Error updating cluster hulls: {e}")
            import traceback
            traceback.print_exc()

    def _remove_cluster_hulls(self):
        """Remove all cluster hull meshes from the view."""
        for mesh in self.cluster_hull_meshes:
            try:
                self.gl_view.removeItem(mesh)
            except Exception:
                pass
        self.cluster_hull_meshes = []

    # V7: Relationship Lines between related clusters

    def _on_relationships_toggle(self, state):
        """Handle relationship lines checkbox toggle."""
        from PySide6.QtCore import Qt
        if state == Qt.CheckState.Checked.value:
            self.relationship_metric_combo.setEnabled(True)
            self.relationship_min_sim_spin.setEnabled(True)
            self.relationship_max_spin.setEnabled(True)
            self.relationship_alpha_spin.setEnabled(True)
            self.relationship_width_spin.setEnabled(True)
            self.relationship_color_btn.setEnabled(True)
            self._update_relationship_lines()
            print("Relationship lines enabled")
        else:
            self.relationship_metric_combo.setEnabled(False)
            self.relationship_min_sim_spin.setEnabled(False)
            self.relationship_max_spin.setEnabled(False)
            self.relationship_alpha_spin.setEnabled(False)
            self.relationship_width_spin.setEnabled(False)
            self.relationship_color_btn.setEnabled(False)
            self._remove_relationship_lines()
            print("Relationship lines disabled")

    def _pick_relationship_color(self):
        """Open color picker for relationship lines."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        color = QColorDialog.getColor(
            QColor(*self._relationship_color),
            self,
            "Select Relationship Line Color",
        )
        if color.isValid():
            self._relationship_color = (color.red(), color.green(), color.blue())
            self._update_relationship_color_button()
            if self.show_relationships_checkbox.isChecked():
                self._update_relationship_lines()

    def _update_relationship_color_button(self):
        """Update the relationship line color button background."""
        r, g, b = self._relationship_color
        self.relationship_color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                border: 1px solid {BLUE_60};
                border-radius: 3px;
            }}
        """)

    def _on_relationship_metric_changed(self):
        """Handle metric combo change - rebuild lines with new metric."""
        if self.show_relationships_checkbox.isChecked():
            self._update_relationship_lines()

    def _on_relationship_params_changed(self):
        """Handle threshold/max/opacity changes - rebuild lines."""
        if self.show_relationships_checkbox.isChecked():
            self._update_relationship_lines()

    def _on_relationship_labels_toggle(self, state):
        """Handle relationship labels checkbox toggle."""
        from PySide6.QtCore import Qt
        if state == Qt.CheckState.Checked.value and self.show_relationships_checkbox.isChecked():
            self._update_relationship_lines()
        else:
            self._remove_relationship_labels()

    def _compute_cluster_relationships(self):
        """Compute related cluster pairs with a similarity score.

        Returns:
            List of (cid1, cid2, score, shared_tag_strings) sorted by
            score descending. Score is in [0, 1] for both metrics.
        """
        import numpy as np
        from collections import Counter

        if not hasattr(self, 'node_list') or not self.node_list:
            return []

        # Group nodes by cluster, collecting tag counts and positions
        cluster_data = {}  # cluster_id -> {'positions': [...], 'tag_counts': Counter}
        for node in self.node_list:
            if node.cluster_id == -1:
                continue
            if node.cluster_id not in cluster_data:
                cluster_data[node.cluster_id] = {'positions': [], 'tag_counts': Counter()}
            cluster_data[node.cluster_id]['positions'].append(node.position)
            if self.tag_interner:
                cluster_data[node.cluster_id]['tag_counts'].update(
                    self.tag_interner.strings_to_list(node.tags)
                )
            else:
                cluster_data[node.cluster_id]['tag_counts'].update(node.tags)

        if len(cluster_data) < 2:
            return []

        # Document frequency: how many clusters contain each tag
        n_clusters = len(cluster_data)
        tag_df = Counter()
        for data in cluster_data.values():
            for tag in data['tag_counts']:
                tag_df[tag] += 1

        # Build one IDF-weighted vector per cluster (mean of member tag counts)
        metric = self.relationship_metric_combo.currentText()
        centroids = {}
        cluster_vectors = {}
        for cid, data in cluster_data.items():
            centroids[cid] = np.mean(data['positions'], axis=0)
            if metric == "IDF Cosine":
                # IDF-weighted vector: weight = log(1 + N/df)
                vec = {}
                for tag, count in data['tag_counts'].items():
                    idf = np.log(1.0 + n_clusters / tag_df[tag])
                    vec[tag] = count * idf
                cluster_vectors[cid] = vec
            else:
                # Raw count vector (shared-tag-count metric)
                cluster_vectors[cid] = dict(data['tag_counts'])

        # Pairwise similarity
        cluster_ids = list(centroids.keys())
        related_pairs = []
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cid1, cid2 = cluster_ids[i], cluster_ids[j]
                v1, v2 = cluster_vectors[cid1], cluster_vectors[cid2]
                shared = set(v1.keys()) & set(v2.keys())
                if not shared:
                    continue
                if metric == "IDF Cosine":
                    dot = sum(v1[t] * v2[t] for t in shared)
                    norm1 = np.sqrt(sum(w * w for w in v1.values()))
                    norm2 = np.sqrt(sum(w * w for w in v2.values()))
                    score = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
                else:
                    # Normalize raw shared count to [0, 1] via Jaccard-like
                    # overlap coefficient: |A ∩ B| / min(|A|, |B|)
                    score = len(shared) / min(len(v1), len(v2))
                # Top shared tags by combined weight (for labels)
                top_shared = sorted(
                    shared, key=lambda t: v1[t] + v2[t], reverse=True
                )[:3]
                related_pairs.append((cid1, cid2, score, top_shared))

        related_pairs.sort(key=lambda x: x[2], reverse=True)
        return related_pairs

    def _update_relationship_lines(self):
        """Draw lines between related clusters (static, strength-encoded)."""
        # Remove existing lines and labels first
        self._remove_relationship_lines()

        if not hasattr(self, 'gl_view') or self.gl_view is None:
            return
        if not hasattr(self, 'node_list') or not self.node_list:
            return

        try:
            import pyqtgraph.opengl as gl
            import numpy as np

            pairs = self._compute_cluster_relationships()
            if not pairs:
                print("No related clusters found")
                return

            # Apply threshold and max-lines limit
            min_sim = float(self.relationship_min_sim_spin.value())
            max_lines = int(self.relationship_max_spin.value())
            base_alpha = float(self.relationship_alpha_spin.value())
            line_width = int(self.relationship_width_spin.value())
            rel_r, rel_g, rel_b = self._relationship_color
            pairs = [p for p in pairs if p[2] >= min_sim][:max_lines]
            if not pairs:
                print(f"No cluster pairs above similarity threshold {min_sim}")
                return

            spread = float(self.spread_spin.value())

            # Precompute centroids once (avoids O(pairs x nodes) rescans)
            cluster_positions = {}
            for n in self.node_list:
                if n.cluster_id != -1:
                    cluster_positions.setdefault(n.cluster_id, []).append(n.position)
            centroids = {
                cid: np.mean(positions, axis=0) * spread
                for cid, positions in cluster_positions.items()
            }

            # Normalize scores to [0, 1] within the visible set for opacity mapping
            scores = np.array([p[2] for p in pairs])
            s_min, s_max = scores.min(), scores.max()
            s_range = (s_max - s_min) if s_max > s_min else 1.0

            self.relationship_pairs = []
            for cid1, cid2, score, top_shared in pairs:
                pos1 = centroids[cid1]
                pos2 = centroids[cid2]

                # Opacity encodes relative strength: 0.3x..1.0x of base alpha
                t = (score - s_min) / s_range
                alpha = base_alpha * (0.3 + 0.7 * t)

                # Color: user-selected relationship line color
                color = [rel_r / 255.0, rel_g / 255.0, rel_b / 255.0, alpha]

                line = gl.GLLinePlotItem(
                    pen=gl.mkPen(color=color, width=line_width),
                    pos=np.array([pos1, pos2]),
                )
                self.gl_view.addItem(line)
                self.relationship_line_items.append(line)
                self.relationship_pairs.append((cid1, cid2, score, top_shared))

            print(f"Drawn {len(self.relationship_line_items)} relationship lines")

            # Labels (separate toggle)
            if self.show_relationship_labels_checkbox.isChecked():
                self._update_relationship_labels()

        except Exception as e:
            print(f"Error updating relationship lines: {e}")
            import traceback
            traceback.print_exc()

    def _update_relationship_labels(self):
        """Show a label at the midpoint of each relationship line."""
        self._remove_relationship_labels()

        if not self.relationship_pairs:
            return

        try:
            import pyqtgraph.opengl as gl
            import numpy as np
            from PySide6.QtGui import QFont

            spread = float(self.spread_spin.value())
            font = QFont("Helvetica", 11)

            # Precompute centroids once (avoids O(pairs x nodes) rescans)
            cluster_positions = {}
            for n in self.node_list:
                if n.cluster_id != -1:
                    cluster_positions.setdefault(n.cluster_id, []).append(n.position)
            centroids = {
                cid: np.mean(positions, axis=0) * spread
                for cid, positions in cluster_positions.items()
            }

            for cid1, cid2, score, top_shared in self.relationship_pairs:
                pos1 = centroids[cid1]
                pos2 = centroids[cid2]
                midpoint = (pos1 + pos2) / 2.0

                # Label: top shared tags + score
                tag_text = ", ".join(top_shared[:3]) if top_shared else ""
                text = f"{tag_text} ({score:.2f})" if tag_text else f"{score:.2f}"

                label_item = gl.GLTextItem(
                    pos=midpoint,
                    text=text,
                    color=(255, 255, 255, 220),
                    font=font,
                    alignment=Qt.AlignHCenter | Qt.AlignVCenter,
                )
                self.gl_view.addItem(label_item)
                self.relationship_label_items.append(label_item)

        except Exception as e:
            print(f"Error updating relationship labels: {e}")
            import traceback
            traceback.print_exc()

    def _remove_relationship_labels(self):
        """Remove all relationship label items from the view."""
        for item in self.relationship_label_items:
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self.relationship_label_items = []

    def _remove_relationship_lines(self):
        """Remove all relationship lines and labels from the view."""
        self._remove_relationship_labels()
        for item in self.relationship_line_items:
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self.relationship_line_items = []
        self.relationship_pairs = []

    # Remember Last Session
    def _load_last_data(self):
        """Load data using the last saved parameters (manual button trigger)."""
        try:
            if not os.path.exists(SETTINGS_FILE):
                self.status_label.setText("No saved settings found.")
                return

            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)

            # Verify we have saved params
            if "max_files" not in settings or "client" not in settings:
                self.status_label.setText("No saved data parameters found.")
                return

            # Apply saved settings to UI
            client_idx = self.client_combo.findText(settings.get("client", ""))
            if client_idx >= 0:
                self.client_combo.setCurrentIndex(client_idx)
            self.max_files_spin.setValue(settings.get("max_files", 4096))
            tag_service_idx = self.tag_service_combo.findText(settings.get("tag_service", "auto2"))
            if tag_service_idx >= 0:
                self.tag_service_combo.setCurrentIndex(tag_service_idx)

            # Algorithm settings
            algo_idx = self.algorithm_combo.findText(settings.get("algorithm", "UMAP"))
            if algo_idx >= 0:
                self.algorithm_combo.setCurrentIndex(algo_idx)
            self.n_neighbors_spin.setValue(settings.get("n_neighbors", 15))
            self.min_dist_spin.setValue(settings.get("min_dist", 50))
            metric_idx = self.metric_combo.findText(settings.get("metric", "cosine"))
            if metric_idx >= 0:
                self.metric_combo.setCurrentIndex(metric_idx)
            self.eps_spin.setValue(settings.get("eps", 50))
            self.min_samples_spin.setValue(settings.get("min_samples", 10))

            # Filter settings
            self.query_edit.setText(settings.get("query", ""))
            self.whitelist_edit.setText(settings.get("whitelist", ""))
            self.blacklist_edit.setText(settings.get("blacklist", ""))
            self.min_doc_freq_spin.setValue(settings.get("min_doc_freq", 3))
            self.drop_universal_checkbox.setChecked(settings.get("drop_universal_tags", False))

            self.status_label.setText("Loaded last settings, starting data load...")
            self.start_loading()

        except Exception as e:
            self.status_label.setText(f"Error loading last data: {e}")
            import traceback
            traceback.print_exc()

    def _auto_load_last_data(self):
        """Auto-load the last used data on startup if enabled (timer-triggered)."""
        if not self.auto_load_checkbox.isChecked():
            return

        # Check if data is already loaded
        if self.gl_scatter is not None:
            return

        try:
            if not os.path.exists(SETTINGS_FILE):
                return
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)

            # Verify we have saved params
            if "max_files" not in settings or "client" not in settings:
                return

            self.status_label.setText("Auto-loading last data...")
            self.start_loading()
        except Exception as e:
            print(f"Auto-load failed: {e}")

    # Time Travel Animation

    def _toggle_time_travel(self):
        """Toggle time travel animation on/off."""
        if self.time_travel_active:
            self._stop_time_travel()
        else:
            self._start_time_travel()

    def _start_time_travel(self):
        """Set up waypoints and start time travel animation."""
        if not hasattr(self, 'node_list') or not self.node_list:
            return

        try:
            import numpy as np
            from collections import defaultdict

            # Group nodes by cluster and compute centroids
            cluster_data = defaultdict(list)
            for node in self.node_list:
                if node.cluster_id != -1:  # Skip noise
                    cluster_data[node.cluster_id].append(node)

            if len(cluster_data) < 2:
                self.status_label.setText("Need at least 2 clusters for Time Travel")
                return

            # Calculate centroids and sort by cluster size (largest first)
            centroids = []
            for cluster_id, nodes in cluster_data.items():
                positions = np.array([n.position for n in nodes])
                centroid = np.mean(positions, axis=0)
                spread = float(self.spread_spin.value())
                centroids.append((cluster_id, len(nodes), centroid * spread))

            # Sort by size descending, take top 10
            centroids.sort(key=lambda x: x[1], reverse=True)
            top_centroids = centroids[:10]

            # Build waypoints list
            self.time_travel_waypoints = []
            for cluster_id, size, position in top_centroids:
                cluster_name = f"Cluster {cluster_id} ({size} files)"
                self.time_travel_waypoints.append((position, cluster_name))

            if len(self.time_travel_waypoints) < 2:
                self.status_label.setText("Need at least 2 clusters for Time Travel")
                return

            # Start animation
            self.time_travel_active = True
            self.time_travel_current_index = 0
            self.time_travel_t = 0.0
            self.time_travel_frames = 0
            self.time_travel_mode = "dwell"

            # Update button
            self.time_travel_button.setText("Stop")
            self.time_travel_button.setToolTip("Stop the Time Travel animation.")

            # Start timer at ~30fps
            self.time_travel_timer.start(33)

            # Show initial cluster name
            self.status_label.setText(f"Time Travel: {self.time_travel_waypoints[0][1]}")

            # Set initial camera position
            self._set_camera_to_waypoint(self.time_travel_waypoints[0][0], 1.0)

        except Exception as e:
            print(f"Error starting time travel: {e}")
            import traceback
            traceback.print_exc()
            self._stop_time_travel()

    def _stop_time_travel(self):
        """Stop time travel animation and restore button."""
        self.time_travel_active = False
        self.time_travel_timer.stop()
        self.time_travel_button.setText("Time Travel")
        self.time_travel_button.setToolTip("Animate camera flying through cluster centroids.")
        self.status_label.setText("Time Travel stopped")

    def _update_time_travel(self):
        """Called at ~30fps to update camera position for time travel."""
        if not self.time_travel_active or not self.time_travel_waypoints:
            return

        self.time_travel_frames += 1

        if self.time_travel_mode == "dwell":
            # Dwell at current waypoint
            if self.time_travel_frames >= self.time_travel_dwell_duration:
                # Switch to orbit mode (spaceship circles the cluster)
                self.time_travel_mode = "orbit"
                self.time_travel_frames = 0
                self.time_travel_orbit_angle = 0.0
                self.time_travel_orbit_center = self.time_travel_waypoints[self.time_travel_current_index][0]
        elif self.time_travel_mode == "orbit":
            # Orbit around the current cluster centroid
            self.time_travel_orbit_angle += self.time_travel_orbit_speed
            if self.time_travel_frames >= self.time_travel_orbit_duration:
                # Done orbiting, travel to next cluster
                self.time_travel_mode = "travel"
                self.time_travel_frames = 0
                self.time_travel_t = 0.0
                self.time_travel_orbit_center = None
                return
            import numpy as np
            center = self.time_travel_orbit_center
            if center is None:
                return
            radius = self.time_travel_orbit_radius
            angle = self.time_travel_orbit_angle
            # Circular orbit around the cluster (in the XY plane, with slight Z lift)
            orbit_pos = np.array([
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                center[2] + 2.0 * np.sin(angle * 0.5),
            ])
            self._set_camera_to_waypoint(orbit_pos, 1.0)
        elif self.time_travel_mode == "travel":
            # Fly over to the next cluster along a straight line
            self.time_travel_t += 1.0 / self.time_travel_segment_duration

            if self.time_travel_t >= 1.0:
                # Reached next waypoint
                self.time_travel_current_index = (self.time_travel_current_index + 1) % len(self.time_travel_waypoints)
                # Transition to orbit around the reached cluster
                self.time_travel_mode = "orbit"
                self.time_travel_frames = 0
                self.time_travel_orbit_angle = 0.0
                self.time_travel_orbit_center = self.time_travel_waypoints[self.time_travel_current_index][0]

                # Update status
                self.status_label.setText(f"Time Travel: {self.time_travel_waypoints[self.time_travel_current_index][1]}")

                # Set camera to reached waypoint
                self._set_camera_to_waypoint(self.time_travel_waypoints[self.time_travel_current_index][0], 1.0)
                return

            # Ease in-out interpolation
            eased_t = self._ease_in_out(self.time_travel_t)

            # Get current and next waypoint
            current_idx = self.time_travel_current_index
            next_idx = (current_idx + 1) % len(self.time_travel_waypoints)
            current_pos = self.time_travel_waypoints[current_idx][0]
            next_pos = self.time_travel_waypoints[next_idx][0]

            # Straight-line fly-over toward the next cluster
            import numpy as np
            interpolated_pos = current_pos + eased_t * (next_pos - current_pos)
            self._set_camera_to_waypoint(interpolated_pos, self.time_travel_t)

    def _set_camera_to_waypoint(self, position, blend_t=1.0):
        """Set camera to look at a waypoint position.

        Args:
            position: numpy array of shape (3) with target position.
            blend_t: Blend factor for smooth transition (0.0 to 1.0).
        """
        try:
            from PySide6.QtGui import QVector3D
            import numpy as np

            current_center = self.gl_view.opts.get('center', np.array([0, 0, 0]))
            # Convert current_center to a numpy array (it may be a QVector3D)
            if not isinstance(current_center, np.ndarray):
                current_center = np.array([
                    float(current_center[0]),
                    float(current_center[1]),
                    float(current_center[2]),
                ])

            # Blend between current center and target
            if blend_t < 1.0:
                new_center = current_center + blend_t * (position - current_center)
            else:
                new_center = position

            self.gl_view.opts['center'] = QVector3D(float(new_center[0]), float(new_center[1]), float(new_center[2]))
            self.gl_view.update()
        except Exception as e:
            print(f"Error setting camera to waypoint: {e}")

    def _ease_in_out(self, t):
        """Smooth ease-in-out function for smooth transitions.

        Args:
            t: Value between 0.0 and 1.0.

        Returns:
            Eased t value with smooth acceleration/deceleration.
        """
        if t <= 0.5:
            return 2.0 * t * t
        else:
            return 1.0 - 2.0 * (1.0 - t) ** 2

    # Tag Importance Ranking (Chi-Square)

    def _compute_tag_importance(self):
        """Compute tag importance using chi-square statistic vs cluster labels.

        Shows top 10 tags ranked by how much they contribute to cluster separation.
        """
        if not hasattr(self, 'node_list') or not self.node_list:
            return

        try:
            import numpy as np
            from collections import defaultdict, Counter

            # Get cluster labels and tag data
            cluster_labels = []
            file_tags = []
            for node in self.node_list:
                cluster_labels.append(node.cluster_id)
                file_tags.append(node.tags)

            cluster_labels = np.array(cluster_labels)
            num_files = len(cluster_labels)

            if num_files == 0:
                return

            # Get unique clusters (exclude noise -1)
            unique_clusters = set(cluster_labels)
            if len(unique_clusters) < 2:
                self.tag_importance_text.setText("Need at least 2 clusters for importance ranking.")
                return

            # Build tag occurrence data
            tag_counts = Counter()  # Total occurrences of each tag
            tag_cluster_counts = defaultdict(lambda: Counter())  # Tag -> {cluster_id: count}

            for i, tags in enumerate(file_tags):
                cluster_id = cluster_labels[i]
                for tag in tags:
                    if self.tag_interner and isinstance(tag, int):
                        tag = self.tag_interner.index_to_string(tag)
                    if tag is None:
                        continue
                    tag_counts[tag] += 1
                    tag_cluster_counts[tag][cluster_id] += 1

            # Total files per cluster
            cluster_total = Counter(cluster_labels)
            total_clusters = len(unique_clusters)

            # Compute chi-square for each tag
            tag_scores = []
            for tag, total_count in tag_counts.items():
                # Skip very rare tags
                if total_count < 3:
                    continue

                # Observed counts per cluster
                observed = np.zeros(total_clusters)
                expected_per_cluster = np.zeros(total_clusters)

                cluster_list = list(unique_clusters)
                for idx, cid in enumerate(cluster_list):
                    observed[idx] = tag_cluster_counts[tag].get(cid, 0)
                    # Expected = (total_tag_count * cluster_size) / total_files
                    expected_per_cluster[idx] = (total_count * cluster_total[cid]) / num_files

                # Chi-square statistic
                chi_sq = 0.0
                for idx in range(total_clusters):
                    exp = expected_per_cluster[idx]
                    if exp > 0:
                        chi_sq += ((observed[idx] - exp) ** 2) / exp

                # Normalize by number of clusters
                normalized_score = chi_sq / total_clusters
                tag_scores.append((tag, normalized_score))

            # Sort by score descending
            tag_scores.sort(key=lambda x: x[1], reverse=True)

            # Display top 10
            if not tag_scores:
                self.tag_importance_text.setText("No tags with sufficient frequency for ranking.")
                return

            lines = []
            for rank, (tag, score) in enumerate(tag_scores[:10], 1):
                lines.append(f"{rank}. {tag} (score: {score:.2f})")

            self.tag_importance_text.setText("\n".join(lines))

        except Exception as e:
            print(f"Error computing tag importance: {e}")
            import traceback
            traceback.print_exc()
            self.tag_importance_text.setText(f"Error computing importance: {e}")
