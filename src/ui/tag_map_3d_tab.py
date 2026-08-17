"""3D Tag Space Visualization Tab Widget.

PyQt-based 3D visualization tab for exploring tag relationships.
"""

import json
import os
import tempfile
import time

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QComboBox, QProgressBar, QGroupBox, QFormLayout,
    QTextEdit, QSplitter, QScrollArea, QLineEdit, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QGridLayout, QFileDialog, QTabWidget
)
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QCloseEvent, QMouseEvent, QVector3D, QFont
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
from src.ui.tag_map_utils import compile_tag_patterns, ease_in_out

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
        """GLTextItem that stacks multi-line text in SCREEN space.

        Performance: the per-frame paint() path is kept minimal. Everything
        that depends only on (text, font, alignment) -- line splitting, font
        metrics, and per-line horizontal advances -- is computed ONCE and
        cached. Each frame we only project the world anchor to screen space
        and issue drawText calls at the cached offsets. This matters a lot
        when the view repaints continuously (e.g. camera wobble), because it
        removes QFontMetrics + horizontalAdvance from the hot path.
        """

        def _build_cache(self):
            """Precompute line layout metrics (called only when text/font change)."""
            fm = QFontMetrics(self.font)
            line_height = fm.lineSpacing()
            lines = self.text.split("\n")
            align = self.alignment
            n = len(lines)

            # Per-line horizontal offset (dx) and vertical offset (dy)
            layout = []
            for i, line in enumerate(lines):
                if not line:
                    continue
                dx = 0.0
                if align & _Qt.AlignmentFlag.AlignHCenter:
                    dx = fm.horizontalAdvance(line) / 2.0
                elif align & _Qt.AlignmentFlag.AlignRight:
                    dx = fm.horizontalAdvance(line)
                dy = i * line_height
                if align & _Qt.AlignmentFlag.AlignVCenter:
                    dy -= (n - 1) * line_height / 2.0
                elif align & _Qt.AlignmentFlag.AlignTop:
                    dy -= (n - 1) * line_height
                layout.append((line, dx, dy))

            self._label_cache = {
                "key": (self.text, self.font.pointSizeF(), int(self.alignment)),
                "layout": layout,
            }

        def paint(self):
            if len(self.text) < 1:
                return

            # Rebuild the metrics cache only when text/font/alignment changed
            cache = getattr(self, "_label_cache", None)
            key = (self.text, self.font.pointSizeF(), int(self.alignment))
            if cache is None or cache["key"] != key:
                self._build_cache()
                cache = self._label_cache
            layout = cache["layout"]
            if not layout:
                return

            self.setupGLState()
            project = self.compute_projection()
            anchor = project.map(QVector3D(*self.pos)).toPointF()

            # Off-screen culling: skip labels whose anchor is outside the
            # viewport (with a margin for text extent). Saves drawText calls
            # for cohorts that are behind/beside the camera.
            view = self.view()
            vw = view.width()
            vh = view.height()
            margin = 200.0
            if (anchor.x() < -margin or anchor.x() > vw + margin or
                    anchor.y() < -margin or anchor.y() > vh + margin):
                return

            painter = QPainter(view)
            painter.setPen(self.color)
            painter.setFont(self.font)
            painter.setRenderHints(
                QPainter.RenderHint.Antialiasing
                | QPainter.RenderHint.TextAntialiasing
            )

            for line, dx, dy in layout:
                painter.drawText(QPointF(anchor.x() - dx, anchor.y() + dy), line)
            painter.end()

    _MULTILINE_TEXT_ITEM_CLASS = _MultiLineGLTextItem
    return _MULTILINE_TEXT_ITEM_CLASS


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
        self.tag_interner = None  # TagInterner instance (set on load; None after session load)

        # Advanced settings (managed by the settings dialog)
        self.low_memory = False
        self.n_jobs = os.cpu_count() or 4
        self.use_direct_db = False
        self.client_db_paths = {}
        self.tokenize = True
        self.drop_universal = True  # Drop universal tags (managed in settings dialog)
        self.drop_empty_files = False  # Drop empty files (managed in settings dialog)
        # When on, right-click both re-centers the camera AND selects the cohort
        # under the cursor (faster navigation). Managed in the settings dialog.
        self.right_click_select_cohort = False

        # DBSCAN optimizer settings (managed by the settings dialog)
        self.opt_max_cohort_size = 500
        self.opt_max_noise_ratio = 10
        self.opt_max_attempts = 60
        self.opt_eps_min = 5
        self.opt_eps_max = 100
        self.opt_min_samples_min = 2
        self.opt_min_samples_max = 30
        # Normalize positions before DBSCAN (global + split clustering)
        self.normalize_positions = True

        # Optional external tag-score DB path (empty = scoring disabled)
        self.score_db_path = ""

        # UI scale factor in percent (applied at startup via QT_SCALE_FACTOR;
        # changing it requires an app restart to take effect).
        self.ui_scale = 100

        self.setup_ui()

        # Star twinkle effect state (must exist BEFORE load_settings, which sets
        # the twinkle spin values and can fire the connected handlers).
        self.gl_twinkle = None  # Separate scatter item for twinkling nodes
        self.twinkle_active = False
        self.twinkle_indices = np.array([], dtype=np.int32)  # indices of active twinkling nodes
        self.twinkle_birth = np.array([])  # birth time (perf_counter) per node
        self.twinkle_lifespan = np.array([])  # lifespan in seconds per node
        self.twinkle_phase = np.array([])  # random phase offset per node
        self.twinkle_timer = QTimer(self)
        self.twinkle_timer.timeout.connect(self._update_twinkle)

        # Default before load_settings (which may set it from the saved state)
        self._restore_media_viewer_open = False
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
        self.gl_highlight = None  # Separate highlight scatter item (selected points only)

        # Cohort label blink timer
        self.cohort_label_blink_timer = QTimer(self)
        self.cohort_label_blink_timer.timeout.connect(self._toggle_cohort_label_blink)
        self.cohort_label_blink_visible = True

        # Cohort labels
        self.cohort_label_items = []  # List of GLTextItem for cohort labels
        self.cohort_label_map = {}  # cluster_id -> GLTextItem (for targeted blink)

        # Auto-load last data
        self.auto_load_timer = QTimer(self)
        self.auto_load_timer.setSingleShot(True)
        self.auto_load_timer.timeout.connect(self._auto_load_last_data)
        # Start the timer so auto-load actually fires after the window is shown.
        # The handler (_auto_load_last_data) is a no-op unless the checkbox is
        # enabled and no data is loaded yet, so this is safe to always start.
        self.auto_load_timer.start(1000)

        # Restore media viewer open state (if it was open at last exit).
        # Deferred until after the main window has been shown once so the
        # saved geometry can be applied correctly.
        if getattr(self, '_restore_media_viewer_open', False):
            self._media_viewer_restore_timer = QTimer(self)
            self._media_viewer_restore_timer.setSingleShot(True)
            self._media_viewer_restore_timer.timeout.connect(
                lambda: (self.toggle_split_window(),
                         setattr(self, '_restore_media_viewer_open', False))
            )
            self._media_viewer_restore_timer.start(1500)

        # Time Travel animation
        self.time_travel_timer = QTimer(self)
        self.time_travel_timer.timeout.connect(self._update_time_travel)
        self.time_travel_active = False
        self.time_travel_waypoints = []  # List of (position, cluster_name) tuples
        self.time_travel_current_index = 0
        self.time_travel_t = 0.0
        self.time_travel_segment_duration = 120  # frames per segment (2s at 60fps)
        self.time_travel_dwell_duration = 30  # frames to dwell at each waypoint (~1s)
        self.time_travel_frames = 0
        self.time_travel_mode = "dwell"  # "dwell", "travel", or "orbit"
        # Orbit parameters (spaceship circling a cluster)
        self.time_travel_orbit_radius = 15.0  # distance from cluster centroid
        self.time_travel_orbit_speed = 0.02  # radians per frame
        self.time_travel_orbit_angle = 0.0
        self.time_travel_orbit_duration = 180  # frames to orbit (~3s at 60fps)
        self.time_travel_orbit_center = None  # current orbit center
        # Travel curve parameters (spaceship banking turn)
        self.time_travel_arc_height = 6.0  # how high the arc rises between clusters

        # Tag query state tracking for clickable tags
        self.tag_query_states = {}  # tag_name -> 0 (neutral), 1 (included), 2 (excluded)
        self._tag_widgets = []  # Store references to prevent garbage collection

        # Media viewer (split) window for image preview sync
        self.split_window = None

        # F4 toggles the media viewer from anywhere in the app, not only when
        # the 3D view has keyboard focus (the old keyPressEvent path missed
        # presses made while another widget/window had focus).
        from PySide6.QtGui import QShortcut, QKeySequence
        self._shortcuts = []

        def _add(key, slot):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)  # keep references alive

        # Plain function keys: always active
        _add("F3", self.open_settings_dialog)   # Settings window
        _add("F4", self.toggle_split_window)    # Media viewer (toggle)
        _add("F5", self.start_loading)          # Load & Compute
        _add("F6", self.start_recompute)        # Recompute (UMAP only)
        _add("F7", self.start_recluster)        # Regroup (DBSCAN only)
        _add("F12", self._take_supersample_screenshot)  # 4x snapshot -> screenshots/

        # Ctrl+ combos: skip while a text field has focus so standard
        # cut/copy/save behavior is preserved in the query/whitelist edits.
        from PySide6.QtWidgets import QLineEdit, QTextEdit

        def _guarded(slot):
            fw = self.focusWidget()
            if isinstance(fw, (QLineEdit, QTextEdit)):
                return
            slot()

        _add("Ctrl+X", lambda: _guarded(self.clear_session))          # Clear session
        _add("Ctrl+S", lambda: _guarded(self._recluster_selection))   # Split group
        _add("Ctrl+E", lambda: _guarded(self._cut_selected_cohort))   # Cut out
        _add("Ctrl+T", lambda: _guarded(self.send_selected_to_tab))   # Send to Tab

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

    def _populate_tag_services(self, client_name):
        """Dynamically populate the tag service combo with all services
        available on the given Hydrus client.

        Uses client.get_services() which returns every registered tag service
        (name + key) directly — no file fetch required. Falls back to a
        sensible default list if the client is unreachable.
        """
        if not hasattr(self, 'tag_service_combo'):
            return

        # Preserve the current selection so we can restore it after repopulating
        previous = self.tag_service_combo.currentText()

        names = []
        if client_name:
            try:
                from src.utils.utility_functions import ConnectToClient
                client = ConnectToClient(client_name)
                services_dict = client.get_services() or {}
                # Only real TAG services belong in this combo. The flat
                # 'services' dict also contains file domains (my files, trash,
                # all local files...) and rating services,
                # which are not valid tag sources and would leak into the list.
                # Hydrus categorises them for us: use only the tag categories.
                tag_categories = ("local_tags", "all_known_tags", "tag_repositories")
                seen = set()
                collected = []
                for cat in tag_categories:
                    for info in services_dict.get(cat, []) or []:
                        if isinstance(info, dict):
                            nm = info.get("name", "")
                            if nm and nm not in seen:
                                seen.add(nm)
                                collected.append(nm)
                names = sorted(collected)
            except Exception as e:
                print(f"Could not fetch tag services for '{client_name}': {e}")

        # Fallback defaults if the client is unreachable or returned nothing
        if not names:
            names = ["auto2", "local", "all known tags"]

        self.tag_service_combo.blockSignals(True)
        self.tag_service_combo.clear()
        self.tag_service_combo.addItems(names)
        # Restore previous selection if it still exists, else first item
        idx = self.tag_service_combo.findText(previous)
        if idx >= 0:
            self.tag_service_combo.setCurrentIndex(idx)
        self.tag_service_combo.blockSignals(False)

    def open_settings_dialog(self):
        """Open the advanced settings window for the 3D tag map tab.

        Window size/position are persisted to the settings file under
        "settings_dialog_geometry" (survives save_settings() because it now
        starts from the existing file contents).
        """
        from src.ui.settings_dialog import TagMap3DSettingsDialog
        dialog = TagMap3DSettingsDialog(self)
        # Restore last size/position (best effort; no-op if never saved)
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    self._restore_window_geometry(json.load(f), "settings_dialog_geometry", dialog)
        except Exception:
            pass
        dialog.exec()
        # Persist new size/position for next time
        try:
            settings = {}
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    settings = loaded
            self._save_window_geometry(settings, "settings_dialog_geometry", dialog)
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
        except Exception:
            pass

    def _refresh_client_combo(self):
        """Rebuild the client combo from clients.json, preserving selection.

        Called after the Settings window saves client changes so newly added /
        renamed / removed clients are reflected without restarting the app.
        """
        if not hasattr(self, 'client_combo'):
            return
        from src.data.clients import client_ids
        previous = self.client_combo.currentText()
        self.client_combo.blockSignals(True)
        self.client_combo.clear()
        ids = client_ids() or []
        self.client_combo.addItems(ids)
        idx = self.client_combo.findText(previous)
        if idx >= 0:
            self.client_combo.setCurrentIndex(idx)
        elif ids:
            self.client_combo.setCurrentIndex(0)
        self.client_combo.blockSignals(False)
        # Refresh tag services for the (possibly new) selected client.
        self._populate_tag_services(self.client_combo.currentText())

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
    
    def resizeEvent(self, event):
        """Reposition toggle buttons on resize."""
        super().resizeEvent(event)
        if hasattr(self, 'sidebar_toggle_btn'):
            self._position_toggle_buttons()

    def _on_cohort_threshold_changed(self, value):
        """Update cohort labels when threshold changes."""
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
            scene = getattr(self, 'scene_graph', None)
            if scene is not None and 0 <= self.selected_node_index < len(scene.file_ids):
                selected_cid = int(scene.cluster_ids[self.selected_node_index])
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

    def _on_smart_label_mode_changed(self, value):
        """Update cohort labels when the smart label mode changes."""
        self._update_cohort_labels()

    def _on_dynamic_label_size_toggled(self, state):
        """Toggle dynamic label sizing."""
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
        scene = self.scene_graph
        tag_data = self.tag_data or {}
        tag_counter = Counter()
        for i in nodes:
            tags = tag_data.get(scene.file_ids[i], [])
            if self.tag_interner:
                tag_counter.update(self.tag_interner.strings_to_list(tags))
            else:
                tag_counter.update(tags)

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

        Skips AND-shared query tags but does NOT apply the cohort threshold.
        Returns the complete ranked list so smart-labels resolution can pick
        next-in-line tags when the top ones are taken.

        Returns:
            list: All non-shared tags in dominance order (tag, count) tuples.
        """
        from collections import Counter
        scene = self.scene_graph
        tag_data = self.tag_data or {}
        tag_counter = Counter()
        for i in nodes:
            tags = tag_data.get(scene.file_ids[i], [])
            if self.tag_interner:
                tag_counter.update(self.tag_interner.strings_to_list(tags))
            else:
                tag_counter.update(tags)

        shared_tags = self._get_query_shared_tags()

        ranked = []
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
        scene = self.scene_graph
        cohort_info = {}
        for cid, idx in cluster_nodes.items():
            centroid = scene.positions[idx].mean(axis=0)
            ranked = self._get_cohort_dominant_tags_full(idx)
            cohort_info[cid] = {
                "centroid": centroid,
                "size": len(idx),
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

            # Group member indices by cluster (vectorized; skip noise)
            scene = self.scene_graph
            cids_arr = scene.cluster_ids
            cluster_nodes = {}
            for cid in np.unique(cids_arr):
                if int(cid) == -1:
                    continue  # Skip noise
                cluster_nodes[int(cid)] = np.where(cids_arr == cid)[0]

            if not cluster_nodes:
                return

            # Determine label color
            r, g, b = self._cohort_label_color

            # Compute max cohort size for dynamic scaling
            max_count = max(len(nodes) for nodes in cluster_nodes.values()) if cluster_nodes else 1
            base_size = self.cohort_label_size_spin.value()

            # Resolve the currently selected cohort (a single-node selection
            # counts as its cohort). Needed by both "Selected Only" mode and
            # the always-include-selected fallback below.
            selected_cid = self.selected_cluster_id
            if selected_cid is None and self.selected_node_index is not None:
                if 0 <= self.selected_node_index < len(scene.file_ids):
                    selected_cid = int(scene.cluster_ids[self.selected_node_index])

            # Apply label mode filter to avoid overlap noise
            mode = self.cohort_label_mode_combo.currentText()
            n = self.cohort_label_n_spin.value()
            if mode == "Selected Only":
                # Show only the selected cohort's label (lightest). No-op if none selected.
                if selected_cid is not None:
                    cluster_nodes = {cid: nodes for cid, nodes in cluster_nodes.items() if cid == selected_cid}
                else:
                    cluster_nodes = {}
            elif mode == "Selected & N neighbors":
                # Selected cohort + its N nearest neighbors (by centroid distance).
                if selected_cid is not None and selected_cid in cluster_nodes:
                    sel_centroid = scene.positions[cluster_nodes[selected_cid]].mean(axis=0)
                    others = {cid: idx for cid, idx in cluster_nodes.items() if cid != selected_cid}
                    # Rank other cohorts by centroid distance to the selection
                    def _centroid(idx):
                        return scene.positions[idx].mean(axis=0)
                    ranked = sorted(
                        others.items(),
                        key=lambda kv: float(np.linalg.norm(_centroid(kv[1]) - sel_centroid))
                    )
                    keep = {selected_cid: cluster_nodes[selected_cid]}
                    for cid, nodes in ranked[:n]:
                        keep[cid] = nodes
                    cluster_nodes = keep
                else:
                    # No valid selection -> nothing to anchor neighbors to.
                    cluster_nodes = {}
            elif mode == "Top N largest":
                # Sort cohorts by size descending, keep top N
                sorted_cohorts = sorted(cluster_nodes.items(), key=lambda kv: len(kv[1]), reverse=True)
                cluster_nodes = dict(sorted_cohorts[:n])
            elif mode == "Above size threshold":
                # Keep only cohorts with at least N files
                cluster_nodes = {cid: nodes for cid, nodes in cluster_nodes.items() if len(nodes) >= n}
            # "All cohorts" -> no filtering

            # Always include the selected cohort so it gets a label even when
            # the label mode filter excluded it (e.g. outside top N or below
            # the size threshold).
            if selected_cid is not None and selected_cid not in cluster_nodes:
                sel_idx = np.where(scene.cluster_ids == selected_cid)[0]
                if len(sel_idx) > 0:
                    cluster_nodes[selected_cid] = sel_idx

            # Smart labels: resolve duplicate labels across cohorts
            smart_label_mode = self.smart_label_mode_combo.currentText()
            smart_label_map = None
            if smart_label_mode != "Raw":
                try:
                    smart_label_map = self._apply_smart_labels(cluster_nodes)
                except Exception as e:
                    print(f"Smart labels failed, falling back to normal labels: {e}")
                    import traceback
                    traceback.print_exc()
                    smart_label_map = None

            self.cohort_label_items = []
            for cid, idx in cluster_nodes.items():
                # Compute centroid (center of cohort) from member indices
                centroid = scene.positions[idx].mean(axis=0)
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

    def _update_selection_tags(self, cluster_nodes):
        """Populate the Cohort Tag Data panel with top 20 tags for the selection.

        Args:
            cluster_nodes: List of nodes in the selected cohort/cluster.
        """
        if not hasattr(self, 'selection_tags_text'):
            return

        if len(cluster_nodes) == 0:
            self.selection_tags_text.setText("No files in selection.")
            return

        # Count tag occurrences across the selected files (cluster_nodes = member indices)
        scene = self.scene_graph
        tag_data = self.tag_data or {}
        tag_counts = {}
        for i in cluster_nodes:
            for tag in tag_data.get(scene.file_ids[i], []):
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
        # Cluster selection (vectorized mask over the scene's label array)
        elif self.selected_cluster_id is not None and getattr(self, 'scene_graph', None) is not None:
            import numpy as np
            idx = np.where(self.scene_graph.cluster_ids == self.selected_cluster_id)[0]
            file_ids.extend(int(self.scene_graph.file_ids[i]) for i in idx)
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

    def _release_session_data(self):
        """Drop all session data and free resources (CPU + GPU).

        Removes GL items from the view, clears cached numpy arrays, node
        list, scene graph and tag data, then forces a garbage collection.
        Called before starting a new load to avoid OOM when the previous
        session's memory is still held while new data is being fetched.
        """
        # Stop timers that reference session state
        for timer in (getattr(self, 'selection_timer', None),
                      getattr(self, 'cohort_label_blink_timer', None),
                      getattr(self, 'twinkle_timer', None),
                      getattr(self, 'time_travel_timer', None)):
            if timer is not None:
                try:
                    timer.stop()
                except RuntimeError:
                    pass

        # Remove GL items (frees GPU vertex buffers)
        self._remove_highlight_item()
        self._remove_twinkle_item()
        for item in list(getattr(self, 'cohort_label_items', []) or []):
            try:
                self.gl_view.removeItem(item)
            except (ValueError, KeyError, RuntimeError):
                pass
        if getattr(self, 'gl_scatter', None) is not None:
            try:
                self.gl_view.removeItem(self.gl_scatter)
            except (ValueError, KeyError, RuntimeError):
                pass

        # Clear cached numpy arrays and data structures
        for attr in ('_base_positions', '_base_sizes', '_base_colors_rgba',
                     '_base_cluster_ids'):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self.gl_scatter = None
        self.node_list = []
        self.scene_graph = None
        self.tag_data = None
        self.tag_interner = None
        if hasattr(self, 'file_ids'):
            self.file_ids = []

        # Reset selection state + info panels
        self.selected_node_index = None
        self.selected_cluster_id = None
        self.selection_visible = False
        self.cohort_label_items = []
        self.cohort_label_map = {}
        self.tag_query_states.clear()
        self._clear_tag_widgets()
        if hasattr(self, 'info_text'):
            self.info_text.clear()
        if hasattr(self, 'selection_tags_text'):
            self.selection_tags_text.clear()
        if hasattr(self, 'tag_importance_text'):
            self.tag_importance_text.clear()

        # Sync split window (clears its grid)
        if getattr(self, 'split_window', None) is not None:
            try:
                self.split_window.clear_grid()
            except RuntimeError:
                pass

        import gc
        gc.collect()

    def clear_session(self):
        """Clear button handler: drop the current session and free memory."""
        if self._is_worker_busy():
            self.status_label.setText("Please wait - a process is already running.")
            return
        self._release_session_data()
        self.recompute_button.setEnabled(False)
        self.recluster_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.send_to_tab_btn.setEnabled(False)
        self.time_travel_button.setEnabled(False)
        self._set_cohort_action_buttons(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Session cleared - memory freed. Ready for a new load.")

    def start_loading(self):
        """Start the data loading and computation process."""
        if self._is_worker_busy():
            self.status_label.setText("Please wait - a process is already running.")
            return
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
        node_size = float(self.min_size_spin.value()) / 10.0
        spread = float(self.spread_spin.value())
        whitelist = [t.strip() for t in self.whitelist_edit.text().split(',') if t.strip()]
        blacklist = [t.strip() for t in self.blacklist_edit.text().split(',') if t.strip()]
        drop_empty = getattr(self, 'drop_empty_files', False)
        query = self.query_edit.text().strip()
        min_doc_freq = self.min_doc_freq_spin.value() if hasattr(self, 'min_doc_freq_spin') else 3
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
            subsample_size=subsample_size
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
        min_doc_freq = self.min_doc_freq_spin.value() if hasattr(self, 'min_doc_freq_spin') else 3
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

    def _save_session_to_path(self, save_path):
        """Save the current scene to a hybrid .npz archive at the given path.

        Stores positions and cluster labels as compact numpy arrays, and
        metadata (file_ids, tags, colors, sizes, settings, camera) as JSON.
        This lets a future load skip query, metadata, UMAP, and DBSCAN.

        Returns True on success, False if there is no scene to save.
        """
        import numpy as np
        import json
        from pathlib import Path

        if self.scene_graph is None or not hasattr(self, 'node_list') or not self.node_list:
            return False

        scene = self.scene_graph
        n = len(scene.file_ids)
        if n == 0:
            return False

        # Build parallel arrays (direct access to the SoA arrays)
        # Convert file_ids to native Python ints (numpy int64 is not JSON-serializable)
        file_ids = [int(fid) for fid in scene.file_ids]
        positions = scene.positions  # (n, 3)
        cluster_labels = scene.cluster_ids  # (n,)
        colors = scene.colors  # (n, 3)
        sizes = scene.sizes  # (n,)

        # Build tag metadata (resolve tokenized tags to strings for portability)
        tags_list = []
        for fid in scene.file_ids:
            raw_tags = (scene.tag_data or {}).get(fid, [])
            if scene.tokenized and scene.reverse_vocab:
                resolved = []
                for t in raw_tags:
                    if isinstance(t, int) and 0 <= t < len(scene.reverse_vocab):
                        resolved.append(scene.reverse_vocab[t])
                    else:
                        resolved.append(str(t))
                tags_list.append(resolved)
            else:
                tags_list.append(list(raw_tags))

        # Build cluster metadata
        clusters_meta = []
        for c in scene.clusters.values():
            clusters_meta.append({
                "id": int(c.cluster_id),
                "centroid": c.centroid.tolist(),
                "size": c.size,
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
            "subsample_enabled": self.subsample_checkbox.isChecked() if hasattr(self, 'subsample_checkbox') else True,
            "subsample_size": self.subsample_size_spin.value() if hasattr(self, 'subsample_size_spin') else 70000,
            "eps": self.eps_spin.value() / 100.0,
            "min_samples": self.min_samples_spin.value(),
            "node_size": float(self.min_size_spin.value()) / 10.0,
            "spread": float(self.spread_spin.value()),
            "min_doc_freq": self.min_doc_freq_spin.value() if hasattr(self, 'min_doc_freq_spin') else 3,
            "drop_universal_tags": getattr(self, 'drop_universal', True),
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
            "node_count": n,
            "file_ids": file_ids,
            "tags": tags_list,
            "colors": colors.tolist(),
            "sizes": sizes.tolist(),
            "clusters": clusters_meta,
            "settings": settings_meta,
            "camera": camera_meta,
        }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            save_path,
            positions=positions,
            cluster_labels=cluster_labels,
            metadata=json.dumps(metadata).encode('utf-8'),
        )
        return True

    def _auto_save_session(self):
        """Automatically save the current scene to sessions/latest.npz.

        Called after every successful load/recompute so that "Auto load
        session" can restore it on next launch without reprocessing.
        Runs silently (no status label change) and never raises.
        """
        try:
            from pathlib import Path
            save_path = Path("sessions") / "latest.npz"
            if self._save_session_to_path(save_path):
                print(f"[Session] Auto-saved to {save_path}")
        except Exception as e:
            print(f"[Session] Auto-save failed: {e}")

    def _save_session(self):
        """Manual save (legacy; button removed). Saves with a timestamp name."""
        import time as _time
        from pathlib import Path
        sessions_dir = Path("sessions")
        timestamp = _time.strftime("%Y%m%d_%H%M%S")
        save_path = sessions_dir / f"session_{timestamp}.npz"
        if self._save_session_to_path(save_path):
            self.status_label.setText(f"Session saved to {save_path}")
        else:
            self.status_label.setText("Error: No scene to save. Load data first.")

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

    def _load_session(self, path=None):
        """Load a saved session archive directly (skips reprocessing).

        Reconstructs the SceneGraph from the hybrid .npz archive and renders
        it immediately — no query, metadata, UMAP, or DBSCAN needed.

        Args:
            path: Optional explicit path to load. If None, shows a picker.
        """
        import numpy as np
        import json
        from pathlib import Path
        from src.core.models import SceneGraph

        if path is not None:
            filename = Path(path)
            if not filename.exists():
                self.status_label.setText(f"No session found at {filename}.")
                return
        else:
            # List sessions from the sessions/ subfolder
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
            # metadata is stored as a 0-d object array holding bytes; extract it.
            meta_raw = data['metadata']
            if isinstance(meta_raw, np.ndarray):
                meta_raw = meta_raw.item()
            if isinstance(meta_raw, bytes):
                metadata = json.loads(meta_raw.decode('utf-8'))
            else:
                metadata = json.loads(str(meta_raw))

            file_ids = metadata['file_ids']
            tags_list = metadata['tags']
            colors = metadata['colors']
            sizes = metadata['sizes']
            clusters_meta = metadata['clusters']
            settings_meta = metadata['settings']
            camera_meta = metadata['camera']

            # Reconstruct SceneGraph directly into the SoA arrays (no per-node objects).
            # Tags are stored as strings in the session, so build with tokenized=False.
            tag_data = {fid: list(tags_list[i]) for i, fid in enumerate(file_ids)}
            scene = SceneGraph()
            scene.build_from_data(
                file_ids,
                np.asarray(positions),
                tag_data,
                np.asarray(cluster_labels, dtype=np.int32),
                node_size=float(sizes[0]) if len(sizes) else 0.02,
                tokenized=False,
            )

            # Restore the exact saved colors (build_from_data assigns by cluster;
            # a session may carry custom scheme colors).
            scene.colors = np.asarray(colors, dtype=np.uint8)

            # Restore camera
            scene.camera_position = np.array(camera_meta['position'])
            scene.camera_target = np.array(camera_meta['target'])

            # Restore settings for recompute/recluster
            self._apply_session_settings(settings_meta)

            self.scene_graph = scene
            self.node_list = scene.get_file_ids()
            self.tag_data = tag_data  # tags ARE persisted in the session (strings)

            n_loaded = len(file_ids)
            self.status_label.setText(f"Session loaded: {n_loaded} nodes")
            print(f"[Session] Loaded {n_loaded} nodes from {filename}")
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
            if hasattr(self, 'subsample_checkbox'):
                self.subsample_checkbox.setChecked(settings_meta.get("subsample_enabled", False))
            if hasattr(self, 'subsample_size_spin'):
                self.subsample_size_spin.setValue(settings_meta.get("subsample_size", 70000))
            self.eps_spin.setValue(settings_meta.get("eps", 0.5) * 100.0)
            self.min_samples_spin.setValue(settings_meta.get("min_samples", 10))
            node_size_actual = settings_meta.get("node_size", settings_meta.get("min_size", 0.02))
            self.min_size_spin.setValue(node_size_actual * 10.0)
            self.spread_spin.setValue(settings_meta.get("spread", 1.0))
            if hasattr(self, 'min_doc_freq_spin'):
                self.min_doc_freq_spin.setValue(settings_meta.get("min_doc_freq", 3))
            self.drop_universal = settings_meta.get("drop_universal_tags", True)
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
            self.node_list = scene.get_file_ids()
            self._build_base_scatter()
            self._apply_highlight_colors(self._base_colors_rgba)
            self._update_cohort_labels()
        else:
            self.render_scene(scene)

        # Auto-save session so "Auto load session" can restore it on next launch.
        self._auto_save_session()

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

            # Clear existing scatter, highlight, and twinkle items
            self._remove_highlight_item()
            self._remove_twinkle_item()
            if self.gl_scatter:
                self.gl_view.removeItem(self.gl_scatter)

            # Get data
            positions = scene.get_node_positions()
            # Apply spread factor
            spread = float(self.spread_spin.value())
            positions = positions * spread

            # V5: Apply color scheme to node colors (vectorized over the label array)
            cluster_ids_arr = scene.cluster_ids
            cluster_ids_set = set(np.unique(cluster_ids_arr).tolist())
            total_clusters = len(cluster_ids_set) if cluster_ids_set else 1

            # Generate colors based on scheme (one per node, by its cluster)
            colors_list = [self._get_color_for_cluster(int(cid), total_clusters) for cid in cluster_ids_arr]
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

            # Store file IDs for click handling (node_list == file ids, same index order)
            self.file_ids = scene.get_file_ids()
            self.node_list = scene.get_file_ids()

            # Build base scatter cache for efficient highlight updates
            self._build_base_scatter()

            # Auto-fit camera to data bounds only on the FIRST render.
            # On subsequent renders (re-cluster / recompute / session load),
            # preserve the current camera position so the user's view isn't reset.
            if not getattr(self, '_camera_initialized', False):
                self._fit_camera_to_data(positions)
                self._camera_initialized = True

            # Ensure the 3D view has keyboard focus for F11 fullscreen toggle
            self.gl_view.setFocus()

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

            ss = 4

            # pxMode=False point sprites get their pixel size from the widget height,
            # but we render into a 4x-taller buffer. Scale point sizes up by `ss` so
            # that after downsampling they appear at normal size (otherwise tiny).
            saved_sizes = []
            for name in ('gl_scatter', 'gl_highlight', 'gl_twinkle'):
                s = getattr(self, name, None)
                if s is None:
                    continue
                try:
                    orig = s.size
                    scaled = np.asarray(orig, dtype=np.float32) * ss
                    s.setData(size=scaled)
                    saved_sizes.append((s, orig))
                except Exception:
                    pass

            try:
                # Render at 4x via renderToArray (returns BGRA uint8)
                arr = view.renderToArray(size=(w * ss, h * ss))
            finally:
                for s, orig in saved_sizes:
                    try:
                        s.setData(size=orig)
                    except Exception:
                        pass
            arr = np.ascontiguousarray(arr)
            # renderToArray renders into an FBO that is never cleared, so its alpha
            # channel is uninitialized garbage. Force it opaque (255).
            arr[..., 3] = 255
            # NOTE: no R/B swap here. renderToArray returns BGRA bytes and Qt's
            # Format_ARGB32 stores pixels as B,G,R,A in memory on little-endian
            # Windows, so the buffer is consumed AS-IS. Swapping it (as before)
            # made Qt read red as blue and vice versa.
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

    def _take_supersample_screenshot(self):
        """Render the GL scene at 4x and save it to screenshots/ as a PNG (F12).

        Saves the FULL 4x-resolution image (not downsampled) so the snapshot is
        crisp. The point sizes are scaled up for the render then restored, same
        as the live supersample path.
        """
        try:
            from PySide6.QtGui import QImage
            import numpy as np

            if not hasattr(self, 'gl_view') or self.gl_view is None:
                self.status_label.setText("No 3D view to capture.")
                return
            view = self.gl_view
            w = view.width()
            h = view.height()
            if w <= 0 or h <= 0:
                self.status_label.setText("3D view not ready yet.")
                return

            ss = 4
            # Scale point sizes up for the taller buffer (restored afterwards).
            saved_sizes = []
            for name in ('gl_scatter', 'gl_highlight', 'gl_twinkle'):
                s = getattr(self, name, None)
                if s is None:
                    continue
                try:
                    orig = s.size
                    s.setData(size=np.asarray(orig, dtype=np.float32) * ss)
                    saved_sizes.append((s, orig))
                except Exception:
                    pass

            try:
                arr = view.renderToArray(size=(w * ss, h * ss))
            finally:
                for s, orig in saved_sizes:
                    try:
                        s.setData(size=orig)
                    except Exception:
                        pass

            arr = np.ascontiguousarray(arr)
            arr[..., 3] = 255  # FBO alpha is uninitialized; force opaque
            # No R/B swap: renderToArray returns BGRA, which matches Qt's
            # in-memory ARGB32 layout (B,G,R,A) on little-endian Windows.
            img = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format.Format_ARGB32)
            img = img.copy()

            # Save to a screenshots/ folder at the project root.
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            out_dir = os.path.join(project_root, "screenshots")
            os.makedirs(out_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"hydruxiom_{stamp}.png")
            if img.save(out_path, "PNG"):
                self.status_label.setText(f"Screenshot saved: {out_path}")
                print(f"Screenshot saved to {out_path} ({img.width()}x{img.height()})")
            else:
                self.status_label.setText("Failed to save screenshot.")
        except Exception as e:
            import traceback
            print(f"Screenshot failed: {e}")
            traceback.print_exc()
            self.status_label.setText(f"Screenshot failed: {e}")

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

    def pick_nearest_point(self, gl_view, click_pos, pick_threshold=20.0):
        """Vectorized 3D point picking using numpy matrix multiplication.

        Projects all points to screen space in one vectorized operation
        instead of iterating in a Python loop. ~100x faster at scale.

        Args:
            gl_view: The GLViewWidget instance (provides view/proj matrices).
            click_pos: QPoint of the click in widget coordinates.
            pick_threshold: Base pixel threshold for hit detection.

        Returns:
            int or None: Index of the nearest point within threshold, or None.
        """
        import numpy as np

        scatter = self.gl_scatter
        if scatter is None or scatter.pos is None or len(scatter.pos) == 0:
            return None

        positions = scatter.pos if isinstance(scatter.pos, np.ndarray) else np.array(scatter.pos)
        n = len(positions)
        if n == 0:
            return None

        # Build MVP matrix as numpy (4x4)
        view_matrix = gl_view.viewMatrix()
        proj_matrix = gl_view.currentProjection()
        mvp = proj_matrix * view_matrix

        # Extract 4x4 matrix from QMatrix4x4
        m = np.array([
            [mvp(0,0), mvp(0,1), mvp(0,2), mvp(0,3)],
            [mvp(1,0), mvp(1,1), mvp(1,2), mvp(1,3)],
            [mvp(2,0), mvp(2,1), mvp(2,2), mvp(2,3)],
            [mvp(3,0), mvp(3,1), mvp(3,2), mvp(3,3)],
        ])

        # Homogeneous coordinates: (n, 4) @ (4, 4).T -> (n, 4)
        ones = np.ones((n, 1))
        pos_h = np.hstack([positions, ones])  # (n, 4)
        clip = pos_h @ m.T  # (n, 4)

        # Perspective divide
        w = clip[:, 3]  # (n,)
        # Avoid division by zero
        w_safe = np.where(np.abs(w) < 1e-10, 1e-10, w)  # (n,)
        ndc = clip[:, :3] / w_safe[:, np.newaxis]  # (n, 3)

        # Filter: behind camera or outside NDC z range
        valid = (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1) & (w > 0)
        if not np.any(valid):
            return None

        # Convert NDC to screen coordinates
        width = gl_view.width()
        height = gl_view.height()
        screen_x = (ndc[:, 0] * 0.5 + 0.5) * width
        screen_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height

        # Distance from click to each point (vectorized)
        dx = screen_x - click_pos.x()
        dy = screen_y - click_pos.y()
        dist = np.sqrt(dx * dx + dy * dy)

        # Apply per-point size threshold if available
        if isinstance(scatter.size, np.ndarray) and len(scatter.size) == n:
            thresholds = pick_threshold + scatter.size
        else:
            base_size = scatter.size if isinstance(scatter.size, (int, float)) else 10
            thresholds = np.full(n, pick_threshold + base_size)

        # Mask: within threshold AND valid (in front of camera)
        dist[~valid] = np.inf
        dist[dist > thresholds] = np.inf

        closest_idx = int(np.argmin(dist))
        if dist[closest_idx] == np.inf:
            return None
        return closest_idx

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
        """Toggle the visual highlight for selected node/cluster.

        Uses a separate highlight scatter item (only the selected points)
        so the blink timer just toggles visibility — zero GPU upload per tick.
        """
        if self.gl_scatter is None or self.node_list is None:
            return

        # Toggle visibility state
        self.selection_visible = not self.selection_visible

        # Just toggle the highlight item's visibility (no color re-upload)
        if hasattr(self, 'gl_highlight') and self.gl_highlight is not None:
            self.gl_highlight.setVisible(self.selection_visible)
    
    def _build_base_scatter(self):
        """Build and cache the base scatter (positions/sizes/base colors/cluster_ids).

        Returns the base colors_rgba array (no highlight/dim).
        """
        import numpy as np
        scene = self.scene_graph
        positions = scene.positions * float(self.spread_spin.value())
        node_size = self.min_size_spin.value() / 10.0
        sizes = np.full(len(scene.file_ids), node_size)
        colors = scene.colors.astype(np.float64) / 255.0
        alpha = self.transparency_spin.value()
        colors_rgba = np.column_stack([colors, alpha * np.ones(len(colors))])
        # Cache cluster_ids as numpy array (avoids O(n) Python loop per blink tick)
        self._base_cluster_ids = scene.cluster_ids.astype(np.int32)
        self._base_positions = positions
        self._base_sizes = sizes
        self._base_colors_rgba = colors_rgba
        # Re-spawn twinkle nodes if active (positions may have changed)
        if getattr(self, 'twinkle_active', False):
            self._spawn_twinkle_nodes()
        return positions, sizes, colors_rgba

    def _apply_highlight_colors(self, colors_rgba):
        """Update the existing scatter's colors in-place (no recreate)."""
        if self.gl_scatter is None:
            return
        self.gl_scatter.setData(color=colors_rgba)

    def _highlight_single_node(self, node_index):
        """Apply highlight for a single node using a separate highlight item.

        The base scatter gets dimmed once (persistent). A small highlight
        scatter shows the selected node. Blink just toggles visibility.
        """
        if not (0 <= node_index < len(self.node_list)):
            return
        import numpy as np

        if not hasattr(self, '_base_colors_rgba') or self._base_colors_rgba is None:
            self._build_base_scatter()

        # Dim base scatter (persistent, applied once)
        if self.dim_non_selected_checkbox.isChecked():
            colors_rgba = self._base_colors_rgba.copy()
            dim_alpha = self.dim_alpha_spin.value()
            mask = np.ones(len(colors_rgba), dtype=bool)
            mask[node_index] = False
            colors_rgba[mask, 3] = dim_alpha
            self._apply_highlight_colors(colors_rgba)
        else:
            self._apply_highlight_colors(self._base_colors_rgba)

        # Create/update highlight item (single point)
        self._update_highlight_item(
            positions=self._base_positions[node_index:node_index+1],
            color=[self.highlight_color[0], self.highlight_color[1], self.highlight_color[2], 1.0]
        )

    def _highlight_cluster(self, cluster_id):
        """Apply highlight for a cluster using a separate highlight item.

        The base scatter gets dimmed once (persistent). A small highlight
        scatter shows the selected cluster's points. Blink just toggles visibility.
        """
        import numpy as np

        if not hasattr(self, '_base_colors_rgba') or self._base_colors_rgba is None:
            self._build_base_scatter()

        # Dim base scatter (persistent, applied once)
        if self.dim_non_selected_checkbox.isChecked():
            colors_rgba = self._base_colors_rgba.copy()
            cluster_ids = self._base_cluster_ids
            colors_rgba[cluster_ids != cluster_id, 3] = self.dim_alpha_spin.value()
            self._apply_highlight_colors(colors_rgba)
        else:
            self._apply_highlight_colors(self._base_colors_rgba)

        # Create/update highlight item (cluster points only)
        cluster_ids = self._base_cluster_ids
        mask = cluster_ids == cluster_id
        self._update_highlight_item(
            positions=self._base_positions[mask],
            color=[self.highlight_color[0], self.highlight_color[1], self.highlight_color[2], 1.0]
        )

    def _update_highlight_item(self, positions, color):
        """Create or update the separate highlight scatter item.

        Only contains the selected points (not all 2M), so GPU upload
        is tiny. Blink timer just toggles visibility.
        """
        import numpy as np
        import pyqtgraph.opengl as gl

        if len(positions) == 0:
            self._remove_highlight_item()
            return

        # Remove old highlight item if exists
        self._remove_highlight_item()

        n = len(positions)
        colors = np.tile(np.array(color, dtype=np.float32), (n, 1))
        sizes = np.full(n, self.min_size_spin.value() / 10.0 * 1.5)  # Slightly larger

        self.gl_highlight = gl.GLScatterPlotItem(
            pos=positions,
            size=sizes,
            color=colors,
            pxMode=False
        )
        self.gl_highlight.setVisible(self.selection_visible)
        self.gl_view.addItem(self.gl_highlight)

    def _remove_highlight_item(self):
        """Remove the highlight scatter item if it exists."""
        if hasattr(self, 'gl_highlight') and self.gl_highlight is not None:
            try:
                self.gl_view.removeItem(self.gl_highlight)
            except (ValueError, KeyError):
                pass
            self.gl_highlight = None
    
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

    def _pick_highlight_color(self):
        """Open color picker for the highlight/blink color."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        current = QColor(int(self.highlight_color[0]*255), int(self.highlight_color[1]*255), int(self.highlight_color[2]*255))
        color = QColorDialog.getColor(current, self, "Highlight Color")
        if color.isValid():
            self.highlight_color = (color.red()/255.0, color.green()/255.0, color.blue()/255.0)
            r, g, b = int(color.red()), int(color.green()), int(color.blue())
            self.highlight_color_btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: black; font-weight: bold;")
            # Re-apply if a selection is active
            self._reapply_selection_style()

    # ─── Star Twinkle Effect ───────────────────────────────────────────────

    def _on_twinkle_toggle(self, state):
        """Start or stop the star twinkle animation."""
        # Guard: state attrs may not exist yet during early load_settings()
        if not hasattr(self, 'twinkle_timer'):
            return
        self.twinkle_active = bool(state)
        if self.twinkle_active:
            if not hasattr(self, '_base_positions') or self._base_positions is None:
                # No scene loaded yet; disable the checkbox
                self.twinkle_checkbox.setChecked(False)
                return
            self._spawn_twinkle_nodes()
            self.twinkle_timer.start(33)  # ~30 fps
        else:
            self.twinkle_timer.stop()
            self._remove_twinkle_item()

    def _on_twinkle_param_changed(self):
        """Restart twinkle with new parameters if active."""
        if not hasattr(self, 'twinkle_active'):
            return
        if self.twinkle_active and hasattr(self, '_base_positions') and self._base_positions is not None:
            self._spawn_twinkle_nodes()

    def _spawn_twinkle_nodes(self):
        """Pick a random set of nodes and assign them lifespans/phases."""
        n_total = len(self._base_positions)
        if n_total == 0:
            return
        count = min(self.twinkle_count_spin.value(), n_total)
        now = time.perf_counter()

        self.twinkle_indices = np.random.choice(n_total, size=count, replace=False).astype(np.int32)
        lifespan_min = self.twinkle_lifespan_min_spin.value()
        lifespan_max = max(self.twinkle_lifespan_max_spin.value(), lifespan_min)
        # All nodes are born now; the random lifespan distribution naturally staggers
        # their deaths over time (short-lived ones get replaced first, long-lived later).
        self.twinkle_birth = np.full(count, now)
        self.twinkle_lifespan = np.random.uniform(lifespan_min, lifespan_max, size=count)
        self.twinkle_phase = np.random.uniform(0, 2 * np.pi, size=count)

        # Create the twinkle scatter item (positions are static, colors update per frame)
        self._remove_twinkle_item()
        import pyqtgraph.opengl as gl
        positions = self._base_positions[self.twinkle_indices]
        node_size = self.min_size_spin.value() / 10.0
        sizes = np.full(count, node_size * 1.2)
        # Initial colors: base colors of those nodes
        base_colors = self._base_colors_rgba[self.twinkle_indices] if hasattr(self, '_base_colors_rgba') else None
        if base_colors is None:
            base_colors = np.tile([0.5, 0.5, 0.5, 1.0], (count, 1))
        self.gl_twinkle = gl.GLScatterPlotItem(
            pos=positions, size=sizes, color=base_colors, pxMode=False
        )
        self.gl_view.addItem(self.gl_twinkle)

    def _update_twinkle(self):
        """Per-frame twinkle update: compute brightness, replace dead nodes."""
        if not self.twinkle_active or self.gl_twinkle is None:
            return
        now = time.perf_counter()
        freq = self.twinkle_freq_spin.value()
        brightness_mult = self.twinkle_brightness_spin.value()

        # Compute per-node blink intensity: 0.5 + 0.5 * sin(2π * freq * t + phase)
        age = now - self.twinkle_birth
        intensity = 0.5 + 0.5 * np.sin(2 * np.pi * freq * age + self.twinkle_phase)

        # Get base colors for active nodes
        if not hasattr(self, '_base_colors_rgba') or self._base_colors_rgba is None:
            return
        base = self._base_colors_rgba[self.twinkle_indices]  # (n, 4)

        # Apply brightness: scale RGB by (1 + intensity * brightness_mult), keep alpha high
        colors = base.copy()
        rgb_scale = 1.0 + intensity * brightness_mult
        colors[:, 0] = np.clip(base[:, 0] * rgb_scale, 0, 1)
        colors[:, 1] = np.clip(base[:, 1] * rgb_scale, 0, 1)
        colors[:, 2] = np.clip(base[:, 2] * rgb_scale, 0, 1)
        colors[:, 3] = np.clip(0.5 + intensity * 0.5, 0, 1)

        self.gl_twinkle.setData(color=colors)

        # Replace dead nodes (age > lifespan) with new random ones
        dead_mask = age > self.twinkle_lifespan
        if np.any(dead_mask):
            n_dead = int(np.sum(dead_mask))
            n_total = len(self._base_positions)
            # Pick new random indices (avoid current active set for variety)
            new_indices = np.random.choice(n_total, size=n_dead, replace=False).astype(np.int32)
            lifespan_min = self.twinkle_lifespan_min_spin.value()
            lifespan_max = max(self.twinkle_lifespan_max_spin.value(), lifespan_min)

            # Update arrays in place
            self.twinkle_indices[dead_mask] = new_indices
            self.twinkle_birth[dead_mask] = now
            self.twinkle_lifespan[dead_mask] = np.random.uniform(lifespan_min, lifespan_max, size=n_dead)
            self.twinkle_phase[dead_mask] = np.random.uniform(0, 2 * np.pi, size=n_dead)

            # Update positions in the scatter item (only changed points)
            new_positions = self._base_positions[new_indices]
            # Rebuild full position array for setData (cheap: only `count` points)
            all_positions = self._base_positions[self.twinkle_indices]
            self.gl_twinkle.setData(pos=all_positions)

    def _remove_twinkle_item(self):
        """Remove the twinkle scatter item if it exists."""
        if self.gl_twinkle is not None:
            try:
                self.gl_view.removeItem(self.gl_twinkle)
            except (ValueError, KeyError):
                pass
            self.gl_twinkle = None

    def toggle_split_window(self):
        """Toggle the media viewer window on/off (F4, works app-wide)."""
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
        # This is a node-selection load, not a thumbnail zoom.
        self._zoom_pending_file_id = None
        self.single_loader = SingleFileLoader(
            self.client_combo.currentText(),
            file_id,
            parent=self
        )
        self.single_loader.pixmap_ready.connect(self._on_single_pixmap_ready)
        self.single_loader.start()

    def _on_single_pixmap_ready(self, pixmap, tooltip):
        """Display a loaded full-res single file (main thread).

        Two triggers share this handler:
        - Node selection (single node in 3D view): tied to selected_node_index.
        - Thumbnail zoom (clicking a grid image): tracked via _zoom_pending_file_id,
          and NOT tied to any node selection.
        """
        if self.split_window is None or not self.split_window.isVisible():
            return

        fid = getattr(self, '_zoom_pending_file_id', None)
        if fid is not None:
            # Thumbnail zoom: show it; clicking again returns to the grid.
            self._zoom_pending_file_id = None
            self.split_window.show_single_image(pixmap, tooltip=tooltip, file_id=fid)
            return

        # Node-selection path: guard against stale loads if selection changed.
        if self.selected_node_index is None:
            return
        file_ids = self._get_selected_file_ids()
        if file_ids and file_ids[0] != getattr(self, '_pending_single_file_id', None):
            return
        self.split_window.show_single_image(pixmap, tooltip=tooltip)

    def _open_file_in_viewer(self, file_id):
        """Open a single file full-res in the media viewer (thumbnail click)."""
        if self.split_window is None or not self.split_window.isVisible():
            return
        # Remember which file we zoomed into so the ready handler can tag it.
        self._zoom_pending_file_id = str(file_id)
        self.single_loader = SingleFileLoader(
            self.client_combo.currentText(),
            file_id,
            parent=self
        )
        self.single_loader.pixmap_ready.connect(self._on_single_pixmap_ready)
        self.single_loader.start()

    def _return_to_grid(self, file_id):
        """Go back from a zoomed full-res image to the thumbnail grid."""
        if self.split_window is None or not self.split_window.isVisible():
            return
        # Rebuild whatever grid was showing before the zoom (cohort files or
        # cohort representatives) based on the current selection state.
        self._zoom_pending_file_id = None
        self._sync_split_window()

    def _load_file_grid(self, file_ids):
        """Load a grid of thumbnails for the given file IDs asynchronously."""
        max_files = self.split_window.max_files_spin.value() if hasattr(self.split_window, 'max_files_spin') else 60
        image_size = self.split_window.image_size_spin.value() if hasattr(self.split_window, 'image_size_spin') else 200
        loader = SplitWindowLoader(
            self.client_combo.currentText(),
            file_ids[:max_files],
            image_size,
            parent=self
        )
        # Identity guard: drop emissions from a superseded loader so leftover
        # thumbnails from a previous selection don't pile onto the current grid.
        def _ready(pixmap, tooltip, _l=loader):
            if self.split_loader is not _l:
                return
            self._on_split_pixmap_ready(pixmap, tooltip)
        loader.pixmap_ready.connect(_ready)
        self.split_loader = loader
        loader.start()

    def _on_split_pixmap_ready(self, pixmap, tooltip):
        """Add a loaded pixmap to the split window (main thread)."""
        if self.split_window is None or not self.split_window.isVisible():
            return
        # Pass the file id so the thumbnail can be clicked to open full-res.
        file_id = tooltip.replace("File ", "") if tooltip.startswith("File ") else None
        self.split_window.add_image(pixmap, tooltip=tooltip, file_id=file_id)

    def _load_cohort_representatives(self):
        """Load one representative image per cohort asynchronously (clickable tiles).

        Capped by the media viewer's "Max Files" spin so a scene with thousands of
        cohorts does not flood the grid / issue thousands of thumbnail requests.
        Largest cohorts are shown first.
        """
        import numpy as np
        scene = self.scene_graph
        max_files = self.split_window.max_files_spin.value() if hasattr(self.split_window, 'max_files_spin') else 28

        # Group member indices by cluster (skip noise)
        cids_arr = scene.cluster_ids
        cluster_nodes = {}
        for cid in np.unique(cids_arr):
            if int(cid) == -1:
                continue
            cluster_nodes[int(cid)] = np.where(cids_arr == cid)[0]

        # Largest cohorts first, then cap to max_files representatives.
        ordered = sorted(cluster_nodes.items(), key=lambda kv: len(kv[1]), reverse=True)[:max_files]

        # Build representative file IDs with cluster info (first member of each cohort)
        rep_file_ids = []
        cluster_map = {}  # file_id -> cluster_id
        for cluster_id, idx in ordered:
            if len(idx) == 0:
                continue
            rep_fid = scene.file_ids[idx[0]]
            rep_file_ids.append(rep_fid)
            cluster_map[rep_fid] = cluster_id

        image_size = self.split_window.image_size_spin.value() if hasattr(self.split_window, 'image_size_spin') else 200
        loader = SplitWindowLoader(
            self.client_combo.currentText(),
            rep_file_ids,
            image_size,
            parent=self
        )
        # Identity guard (same as _load_file_grid): drop emissions from a
        # superseded loader so stale tiles don't accumulate.
        def _ready(pixmap, tooltip, _l=loader):
            if self.split_loader is not _l:
                return
            self._on_cohort_pixmap_ready(pixmap, tooltip)
        loader.pixmap_ready.connect(_ready)
        self.split_loader = loader
        loader.start()
        self._cohort_cluster_map = cluster_map

    def _on_cohort_pixmap_ready(self, pixmap, tooltip):
        """Add a cohort representative tile to the split window (main thread)."""
        if self.split_window is None or not self.split_window.isVisible():
            return
        # Extract file_id from tooltip to look up cluster (keys are int hash_ids)
        raw_fid = tooltip.replace("File ", "")
        try:
            fid_key = int(raw_fid)
        except (ValueError, TypeError):
            fid_key = raw_fid
        cluster_id = self._cohort_cluster_map.get(fid_key)
        if cluster_id is not None:
            self.split_window.add_cohort_tile(cluster_id, pixmap, tooltip=tooltip)

    def _move_camera_to_cluster(self, cluster_id):
        """Move the 3D camera to focus on a specific cluster (future interaction)."""
        if not hasattr(self, 'node_list') or not self.node_list:
            return
        import numpy as np
        scene = self.scene_graph
        idx = np.where(scene.cluster_ids == int(cluster_id))[0]
        if len(idx) == 0:
            return
        centroid = scene.positions[idx].mean(axis=0)
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
        self._remove_highlight_item()
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
        
        scene = self.scene_graph
        if 0 <= node_index < len(scene.file_ids):
            fid = scene.file_ids[node_index]
            cluster_id = int(scene.cluster_ids[node_index])
            score = float(scene.scores[node_index])
            pos = scene.positions[node_index]

            # Clear tag state for new selection
            self.tag_query_states.clear()
            self._clear_tag_widgets()

            # Update text info (without tags)
            info_lines = [
                f"File ID: {fid}",
                f"Cluster: {cluster_id}",
                f"Score: {score:.2f}",
                f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
            ]
            self.info_text.setText("\n".join(info_lines))

            # Create clickable tag widgets (tags live in self.tag_data)
            node_tags = (self.tag_data or {}).get(fid, [])
            tags_to_show = list(node_tags[:50])  # Limit to 50 tags for performance
            if self.tag_interner:
                tags_to_show = self.tag_interner.strings_to_list(tags_to_show)
            if len(node_tags) > 50:
                # Add a label to indicate more tags exist
                more_label = QLabel(f"... and {len(node_tags) - 50} more tags")
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
            self.selection_visible = True
            self._highlight_single_node(node_index)  # Create highlight item + dim base
            self.selection_timer.start(500)  # 500ms interval for ~2Hz blink

            # Update the cohort tag data panel with the node's cluster cohort (member indices)
            import numpy as np
            cluster_idx = np.where(scene.cluster_ids == cluster_id)[0]
            self._update_selection_tags(cluster_idx)

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
        
        scene = self.scene_graph
        if 0 <= node_index < len(scene.file_ids):
            cluster_id = int(scene.cluster_ids[node_index])

            # Find all member indices in the same cluster
            import numpy as np
            cluster_nodes = np.where(scene.cluster_ids == cluster_id)[0]

            if len(cluster_nodes) == 0:
                return

            # Clear tag state for new selection
            self.tag_query_states.clear()
            self._clear_tag_widgets()

            # Count tag occurrences across files in cluster (tags live in self.tag_data)
            tag_data = self.tag_data or {}
            tag_counts = {}
            for i in cluster_nodes:
                for tag in tag_data.get(scene.file_ids[i], []):
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
            self.selection_visible = True
            self._highlight_cluster(cluster_id)  # Create highlight item + dim base
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
        """Split a query string by commas, keeping bracket groups intact."""
        return split_query_preserving_brackets(query)

    def _query_to_api_tags(self, query):
        """Convert a query string to API-ready tags list (OR groups as nested lists)."""
        return query_to_api_tags(query)

    def _get_query_tag_states(self):
        """Parse the current query_edit into included/excluded/OR tag sets."""
        return parse_query_tag_states(self.query_edit.text())

    def _on_size_changed(self):
        """Handle size parameter changes to update scatter dynamically."""
        if self.gl_scatter and hasattr(self, 'node_list') and self.node_list:
            import numpy as np
            import pyqtgraph.opengl as gl
            node_size = self.min_size_spin.value() / 10.0
            alpha = self.transparency_spin.value()
            
            # Uniform node size
            scene = self.scene_graph
            sizes = np.full(len(scene.file_ids), node_size)

            # Remove old scatter and create new one with updated sizes
            self.gl_view.removeItem(self.gl_scatter)

            positions = scene.positions
            colors = scene.colors.astype(np.float64) / 255.0
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
            node_size = self.min_size_spin.value() / 10.0
            alpha = self.transparency_spin.value()
            
            # Recalculate positions with spread (direct SoA array access)
            scene = self.scene_graph
            positions = scene.positions * spread
            sizes = np.full(len(scene.file_ids), node_size)
            colors = scene.colors.astype(np.float64) / 255.0
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

    def _on_transparency_changed(self):
        """Handle transparency parameter changes to update scatter dynamically."""
        if self.gl_scatter and hasattr(self, 'node_list') and self.node_list:
            import numpy as np
            import pyqtgraph.opengl as gl
            spread = float(self.spread_spin.value())
            node_size = self.min_size_spin.value() / 10.0
            alpha = self.transparency_spin.value()
            
            # Recalculate positions with spread (direct SoA array access)
            scene = self.scene_graph
            positions = scene.positions * spread
            sizes = np.full(len(scene.file_ids), node_size)
            colors = scene.colors.astype(np.float64) / 255.0
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
            # Capture the current camera angles as the base for oscillation/spin.
            self._capture_wobble_base()
            self.wobble_timer.start()
            print("Camera wobble enabled")
        else:
            self.wobble_timer.stop()
            print("Camera wobble disabled")

    def _on_wobble_continuous_toggle(self, state):
        """Handle continuous-spin toggle. Re-base so there is no camera jump."""
        if hasattr(self, 'gl_view') and self.gl_view is not None:
            self._capture_wobble_base()
        # Reset the spin accumulator so continuous mode starts from current azimuth.
        self.wobble_spin_azim = 0.0

    def _capture_wobble_base(self):
        """Store the current camera angles as the wobble/spin base."""
        if hasattr(self, 'gl_view') and self.gl_view is not None:
            self.wobble_base_azim = self.gl_view.opts.get('azimuth', 45)
            self.wobble_base_elev = self.gl_view.opts.get('elevation', 30)

    def _update_wobble(self):
        """Update camera position for wobble (oscillate) or continuous spin."""
        import numpy as np
        if not hasattr(self, 'gl_view') or self.gl_view is None:
            return

        # Pause while the user is dragging so manual orbit isn't fought.
        # Time is NOT advanced here, so the animation resumes smoothly on release.
        if getattr(self, 'wobble_user_interacting', False):
            return

        # Increment time
        dt = 0.016  # ~60fps timestep
        self.wobble_time += dt

        # Get wobble parameters
        speed = self.wobble_speed_spin.value()
        azim_range = self.wobble_azim_range_spin.value()
        elev_range = self.wobble_elev_range_spin.value()

        t = self.wobble_time * speed

        if self.wobble_continuous_checkbox.isChecked():
            # Continuous spin: azimuth rotates in one direction (wraps at 360),
            # elevation bobs gently so the camera never flips over the top.
            # The Azim/Elev Range values act as degrees-per-second.
            self.wobble_spin_azim += azim_range * dt * speed
            new_azimuth = (self.wobble_base_azim + self.wobble_spin_azim) % 360.0
            new_elevation = self.wobble_base_elev + np.sin(t * 0.4) * (elev_range * 0.5)
        else:
            # Classic sine wobble: oscillate around the base angles for parallax depth.
            new_azimuth = self.wobble_base_azim + np.sin(t * 0.5) * azim_range
            new_elevation = self.wobble_base_elev + np.sin(t * 0.7) * elev_range

        # Update camera position
        self.gl_view.opts['elevation'] = new_elevation
        self.gl_view.opts['azimuth'] = new_azimuth
        self.gl_view.update()

    def keyPressEvent(self, event):
        """Handle key press events for fullscreen toggle and shortcuts."""
        # Ctrl+E: send selected cohort to Hydrus tab
        if event.key() == Qt.Key_E and event.modifiers() & Qt.ControlModifier:
            self.send_selected_to_tab()
            event.accept()
            return
        # Ctrl+R: pop selected cohort
        if event.key() == Qt.Key_R and event.modifiers() & Qt.ControlModifier:
            self._pop_selected_cohort()
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

        import numpy as np
        import pyqtgraph.opengl as gl

        scene = self.scene_graph
        # Count unique clusters for normalization (vectorized)
        cluster_ids_set = set(np.unique(scene.cluster_ids).tolist())
        total_clusters = len(cluster_ids_set) if cluster_ids_set else 1

        spread = float(self.spread_spin.value())
        node_size = self.min_size_spin.value() / 10.0
        alpha = self.transparency_spin.value()

        # Recalculate positions (direct SoA array access)
        positions = scene.positions * spread
        sizes = np.full(len(scene.file_ids), node_size)

        # Generate new colors based on scheme (one per node, by its cluster)
        colors = np.array([self._get_color_for_cluster(int(cid), total_clusters) for cid in scene.cluster_ids]) / 255.0
        colors_rgba = np.column_stack([colors, alpha * np.ones(len(colors))])

        # Update node colors in the data for future reference (bulk array write)
        scene.colors[:] = (colors * 255).astype(np.uint8)

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
            self.max_files_spin.setValue(settings.get("max_files", 20000))
            tag_service_idx = self.tag_service_combo.findText(settings.get("tag_service", "auto2"))
            if tag_service_idx >= 0:
                self.tag_service_combo.setCurrentIndex(tag_service_idx)

            # Algorithm settings
            algo_idx = self.algorithm_combo.findText(settings.get("algorithm", "UMAP"))
            if algo_idx >= 0:
                self.algorithm_combo.setCurrentIndex(algo_idx)
            self.n_neighbors_spin.setValue(settings.get("n_neighbors", 15))
            self.min_dist_spin.setValue(settings.get("min_dist", 10))
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
            self.drop_universal = settings.get("drop_universal_tags", True)

            self.status_label.setText("Loaded last settings, starting data load...")
            self.start_loading()

        except Exception as e:
            self.status_label.setText(f"Error loading last data: {e}")
            import traceback
            traceback.print_exc()

    def _auto_load_last_data(self):
        """Auto-load the last generated session on startup if enabled.

        Loads sessions/latest.npz directly (skips query, UMAP, DBSCAN).
        If no saved session exists yet, falls back to reprocessing with the
        last saved parameters so a session gets created for next time.
        """
        if not self.auto_load_checkbox.isChecked():
            return

        # Check if data is already loaded
        if self.gl_scatter is not None:
            return

        from pathlib import Path
        latest = Path("sessions") / "latest.npz"
        if latest.exists():
            self.status_label.setText("Auto-loading last session...")
            self._load_session(path=latest)
            return

        # No saved session yet — fall back to reprocessing with saved params.
        try:
            if not os.path.exists(SETTINGS_FILE):
                return
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
            if "max_files" not in settings or "client" not in settings:
                return
            self.status_label.setText("No saved session found - processing last data...")
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

            scene = self.scene_graph
            # Group member indices by cluster and compute centroids (skip noise)
            cids_arr = scene.cluster_ids
            cluster_data = defaultdict(list)
            for cid in np.unique(cids_arr):
                if int(cid) == -1:  # Skip noise
                    continue
                cluster_data[int(cid)] = np.where(cids_arr == cid)[0]

            if len(cluster_data) < 2:
                self.status_label.setText("Need at least 2 clusters for Explore")
                return

            # Calculate centroids and sort by cluster size (largest first)
            centroids = []
            for cluster_id, idx in cluster_data.items():
                centroid = scene.positions[idx].mean(axis=0)
                spread = float(self.spread_spin.value())
                centroids.append((cluster_id, len(idx), centroid * spread))

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

            # Zoom in: reduce camera distance so clusters are visible
            self.gl_view.opts['distance'] = 50.0

            # Scale orbit radius to be visible at the new distance
            self.time_travel_orbit_radius = 15.0

            # Update button
            self.time_travel_button.setText("Stop")
            self.time_travel_button.setToolTip("Stop the Explore animation.")

            # Start timer at ~30fps
            self.time_travel_timer.start(33)

            # Show initial cluster name
            self.status_label.setText(f"Explore: {self.time_travel_waypoints[0][1]}")

            # Set initial camera position
            self._set_camera_to_waypoint(self.time_travel_waypoints[0][0], 1.0)

        except Exception as e:
            print(f"Error starting time travel: {e}")
            import traceback
            traceback.print_exc()
            self._stop_time_travel()

    def _stop_time_travel(self):
        """Stop explore animation and restore button + camera distance."""
        self.time_travel_active = False
        self.time_travel_timer.stop()
        self.time_travel_button.setText("Explore")
        self.time_travel_button.setToolTip("Animate camera flying through cluster centroids.")
        # Restore camera distance
        if hasattr(self, 'gl_view') and self.gl_view:
            self.gl_view.opts['distance'] = 200.0
            self.gl_view.update()
        self.status_label.setText("Explore stopped")

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
                self.status_label.setText(f"Explore: {self.time_travel_waypoints[self.time_travel_current_index][1]}")

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

            current_center = self.gl_view.opts.get('center', np.array([0.0, 0.0, 0.0]))
            # Convert current_center to a numpy array. It may be a QVector3D (set by
            # right-click camera-centering) which is NOT subscriptable — use .x/.y/.z.
            if isinstance(current_center, np.ndarray):
                pass
            elif hasattr(current_center, 'x'):  # QVector3D / QQuaternion-like
                current_center = np.array([float(current_center.x()),
                                           float(current_center.y()),
                                           float(current_center.z())])
            else:
                current_center = np.array([float(v) for v in current_center])

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

        Thin wrapper around :func:`src.ui.tag_map_utils.ease_in_out`.
        """
        return ease_in_out(t)

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

            scene = self.scene_graph
            # Get cluster labels and tag data (direct array access + tag_data dict)
            cluster_labels = scene.cluster_ids.astype(np.int64).copy()
            file_tags = [list((self.tag_data or {}).get(fid, [])) for fid in scene.file_ids]

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
