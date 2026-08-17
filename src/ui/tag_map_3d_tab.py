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
from src.ui.tag_map_utils import (
    compile_tag_patterns, ease_in_out, SETTINGS_FILE,
    render_cohort_tags_html, render_importance_html, render_info_rows,
)
from src.ui.gl_text_items import get_multiline_text_item_class as _get_multiline_text_item_class
from src.ui.client_services import ClientServicesMixin
from src.ui.settings_persistence import SettingsPersistenceMixin
from src.ui.ui_construction import UIConstructionMixin
from src.ui.visual_effects import VisualEffectsMixin
from src.ui.cohort_labels import CohortLabelsMixin
from src.ui.picking_highlight import PickingHighlightMixin
from src.ui.data_pipeline import DataPipelineMixin
from src.ui.cohort_ops import CohortOpsMixin


class TagMap3DTab(CohortOpsMixin, DataPipelineMixin, PickingHighlightMixin, CohortLabelsMixin, VisualEffectsMixin, UIConstructionMixin, SettingsPersistenceMixin, ClientServicesMixin, QWidget):
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
        self.low_memory = True
        self.n_jobs = os.cpu_count() or 4
        self.use_direct_db = False
        self.client_db_paths = {}
        self.tokenize = True
        self.drop_universal = True  # Drop universal tags (managed in settings dialog)
        self.drop_empty_files = False  # Drop empty files (managed in settings dialog)
        # When on, right-click both re-centers the camera AND selects the cohort
        # under the cursor (faster navigation). Managed in the settings dialog.
        self.right_click_select_cohort = False

        # Auto-center on selection: when enabled, selecting a cohort (e.g. via WASD
        # navigation) also moves the camera center to that cohort's centroid.
        # Managed in the settings dialog (UI group).
        self.auto_center_on_selection = False

        # Smooth center transition: when enabled, right-click glides the camera
        # to the new center instead of teleporting. Managed in the settings dialog.
        self.smooth_center_transition = False
        # Glide speed in scene units/second (higher = faster). Duration is
        # distance / speed, clamped so short hops stay snappy and long ones don't drag.
        self.smooth_center_speed = 1.0

        # Auto-split: after Load & Compute (or Regroup), automatically select and
        # split any cohort larger than the threshold, cycling up to max_cycles
        # times or until no oversized cohort remains. Managed in the settings dialog.
        self.auto_split_enabled = True   # master switch (Settings -> DBSCAN Optimizer)
        self.auto_split_threshold = 5000
        self.auto_split_max_cycles = 3

        # Smart Scale: when enabled, after data loads the app picks a profile by
        # file count and overwrites UMAP/DBSCAN/visualization settings with it.
        # Managed in the settings dialog (Smart Scale tab). See src/ui/smart_scale.py.
        self.smart_scale_enabled = False
        self.smart_scale_profiles = []

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
        self.auto_deorphan = "Never"  # "Never", "After Load and Compute", "After Regroup"
        self._pending_deorphan = False  # True while a deorphan worker is in flight

        # Session auto-save: instead of writing sessions/latest.npz immediately
        # after every operation (costly when iterating settings), each change
        # arms a resettable delay timer; the file is written only once no new
        # change has occurred for session_save_delay seconds. closeEvent forces
        # an immediate save so nothing is lost on exit. 0 = save immediately.
        self.session_save_delay = 60  # seconds (managed in settings dialog)
        self._session_save_timer = QTimer(self)
        self._session_save_timer.setSingleShot(True)
        self._session_save_timer.timeout.connect(self._auto_save_session)

        # Explore "helicopter orbit" mode defaults. MUST be set BEFORE load_settings()
        # so that saved values from the JSON file are not clobbered by these defaults.
        self.explore_accel = 0.06          # approach speed factor (higher = faster)
        self.explore_decel = 0.06          # deceleration into orbit (higher = snappier settle)
        self.explore_orbit_radius_base = 0.0   # base orbit distance from cohort centroid
        self.explore_orbit_size_factor = 0.2   # + this * sqrt(cohort size) for big cohorts
        self.explore_orbit_speed = 15.0  # orbit angular speed (deg/sec) while circling a cohort
        self.explore_cycles = 1          # full orbits per cohort before moving on
        self.explore_max_orbit_time = 12.0  # max seconds to orbit a cohort
        self.explore_elevation = 30.0    # camera elevation (deg) while orbiting
        self.explore_mode = "Contrast"     # "Random" | "Linear Path" | "Contrast"
        self.explore_show_path = False    # draw the planned route preview (managed in settings)

        self.load_settings()
        # Sync the main-window Smart Scale checkbox with the loaded value (the
        # widget was created in setup_ui before load_settings ran). Block signals
        # so this doesn't re-trigger the toggle handler / save.
        if hasattr(self, 'smart_scale_checkbox'):
            self.smart_scale_checkbox.blockSignals(True)
            self.smart_scale_checkbox.setChecked(bool(getattr(self, 'smart_scale_enabled', False)))
            self.smart_scale_checkbox.blockSignals(False)
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

        # Cohort label fade-out: when the selection changes (in "Selected & N
        # neighbors" mode), labels that drop out fade to transparent over ~0.3 s
        # instead of vanishing instantly, which is less disorienting.
        self.label_fade_enabled = True  # master toggle (managed in the label panel)
        self._label_fade_items = []  # list of [item, base_rgba] currently fading
        self._label_fade_timer = QTimer(self)
        self._label_fade_timer.setInterval(16)  # ~60 fps
        self._label_fade_timer.timeout.connect(self._update_label_fade)

        # WASD cohort navigation: first WASD press shows preview paths (W/S/A/D)
        # to the nearest cohort in each screen direction; subsequent presses move
        # the selection + camera there. Paths are 2D overlays that refresh as the
        # camera moves (arrow keys) or the selection changes.
        self.wasd_paths_enabled = True   # master toggle (managed in settings dialog)
        self._wasd_mode = False          # navigation active (paths visible); refresh on camera move
        self._wasd_selecting = False     # transient: True only during WASD-initiated selection
        self._wasd_targets = {}   # 'W'/'S'/'A'/'D' -> cluster_id
        self._wasd_items = []     # GL items currently drawn for the preview
        # Fade-out of WASD paths when navigation ends (labels fade over ~2 s; lines
        # are dropped immediately since GLLinePlotItem has no color-update API).
        self._wasd_fade_labels = []  # list of [GLTextItem, base_rgba] fading out
        self._wasd_fade_timer = QTimer(self)
        self._wasd_fade_timer.setInterval(16)  # ~60 fps
        self._wasd_fade_timer.timeout.connect(self._update_wasd_path_fade)

        # WASD travel history: an ordered list of (cluster_id, centroid) visited via
        # W/S/A/D navigation. _wasd_history_index points at the current position in
        # that list. Q steps back through it, E steps forward again (only available
        # after going back). The persistent translucent-turquoise trail is drawn from
        # history[0..index]. Kept until reset (Visualization Settings -> "Reset Travel
        # Trail") or a scene rebuild. Independent of the live W/S/A/D preview arrows.
        self._wasd_history = []        # list of (cluster_id, np.array(3) centroid)
        self._wasd_history_index = -1  # current position in _wasd_history (-1 = none)
        self._wasd_trail_items = []    # GL items currently drawn for the trail

        # Explore path preview: runtime state for the planned route overlay.
        self._explore_path_items = []  # GL items currently drawn for the path preview

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

        # Explore (helicopter-orbit) animation timer. The old "Time Travel" waypoint
        # fields were replaced by the _explore_* runtime state below.
        self.time_travel_timer = QTimer(self)
        self.time_travel_timer.timeout.connect(self._update_time_travel)

        # Smooth center transition animation state
        self._center_anim_active = False
        self._center_anim_start = None   # np.array(3) – starting camera center
        self._center_anim_target = None  # np.array(3) – destination camera center
        self._center_anim_t0 = 0.0      # perf_counter() at animation start
        self._center_anim_duration = 0.5  # seconds (computed per-move from distance)
        self._center_anim_timer = QTimer(self)
        self._center_anim_timer.setInterval(16)  # ~60 fps
        self._center_anim_timer.timeout.connect(self._update_center_animation)

        # Runtime explore state (not persisted).
        self._explore_active = False
        self._explore_path = []          # ordered list of cohort targets for this run
        self._explore_path_index = 0     # position in _explore_path
        self._explore_phase = "approach"   # "approach" | "orbit"
        self._explore_cohorts = []         # list of (cluster_id, size, centroid) for random picks
        self._explore_target_cid = None
        self._explore_target_centroid = None
        self._explore_target_size = 0
        self._explore_orbit_radius = 15.0
        self._explore_azimuth = 0.0        # current orbit azimuth (deg)
        self._explore_entry_azimuth = 0.0  # azimuth the approach settles into
        self._explore_side = 1             # +1 -> cohort in right third, -1 -> left third
        self._explore_cycles_done = 0
        self._explore_orbit_turns = 0.0    # accumulated orbit angle (deg) for cycle counting
        self._explore_orbit_time = 0.0     # seconds spent orbiting the current cohort
        self._explore_approach_t = 0.0     # 0..1 progress of the approach phase
        self._explore_from_center = None   # camera look-at center at approach start
        self._explore_from_dist = None
        self._explore_from_elev = None
        self._explore_from_azim = None
        # Flight-path arc: actual 3D camera positions (start -> elevated control ->
        # orbit entry) so the camera flies along a plane-like curve instead of just
        # lerping distance/elevation/azimuth independently.
        self._explore_cam_start = None     # np.array(3) camera position at approach start
        self._explore_cam_end = None       # np.array(3) orbit-entry camera position
        self._explore_cam_start_vel = None  # np.array(3) velocity at approach start (from prior orbit)
        self._explore_marker_item = None   # bright marker on the target centroid (shows what's orbited)

        # Auto-split cycle state (driven from on_loading_finished; each split is a
        # full re-cluster worker round-trip, so the loop chains across completions).
        self._auto_split_active = False
        self._auto_split_cycles_done = 0
        self._auto_split_target_id = None    # cluster id targeted by the current cycle
        self._auto_split_target_size = 0     # its size at selection time (early-stop check)
        self._auto_split_target_members = None  # set of file_ids in the target cohort;
                                                # re-cluster REMAPS labels to new ids, so
                                                # membership must be tracked by file_id.

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


    def closeEvent(self, event: QCloseEvent):
        """Handle window close event to save settings."""
        # Cancel the pending delayed auto-save and force an immediate one so the
        # latest scene is never lost on exit (the delay only exists to avoid
        # costly writes while iterating; closing means we must persist now).
        self._session_save_timer.stop()
        self._auto_save_session()
        # Wait for that background save to finish before tearing down.
        self._wait_pending_auto_save()
        self._save_client_db_path()
        self.save_settings()
        super().closeEvent(event)

    def resizeEvent(self, event):
        """Reposition toggle buttons on resize."""
        super().resizeEvent(event)
        if hasattr(self, 'sidebar_toggle_btn'):
            self._position_toggle_buttons()

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
        self.selection_tags_text.setHtml(
            render_cohort_tags_html(total_files, sorted_tags, shown=20)
        )

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

    def _session_settings_snapshot(self):
        """Read the current pipeline settings from the UI controls.

        Must be called on the GUI thread (Qt widgets are not thread-safe).
        The snapshot is passed to background session saves so they never touch
        widgets off-thread.
        """
        return {
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

    def _save_session_to_path(self, save_path, scene=None, tag_data=None, settings_meta=None, view_meta=None):
        """Save a scene to a hybrid .npz archive at the given path.

        Stores positions and cluster labels as compact numpy arrays, and
        metadata (file_ids, tags, colors, sizes, settings, camera, view) as JSON.
        This lets a future load skip query, metadata, UMAP, and DBSCAN.

        ``scene`` / ``tag_data`` / ``settings_meta`` / ``view_meta`` may be passed
        explicitly (background auto-save captures them on the GUI thread); otherwise
        the current self.scene_graph / self.tag_data / live widget values are used.

        The archive is written to a temp file and atomically renamed over
        ``save_path``, so concurrent saves or a crash mid-write can never leave
        a corrupt latest.npz behind.

        Returns True on success, False if there is no scene to save.
        """
        import numpy as np
        import json
        from pathlib import Path

        if scene is None:
            scene = self.scene_graph
        if scene is None or not getattr(scene, 'file_ids', None):
            return False

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

        # Build tag metadata (resolve tokenized tags to strings for portability).
        # Uses the explicitly passed tag_data when given: after a pop/cut the
        # scene's own reference can be stale relative to self.tag_data, and the
        # background auto-save must not depend on either being current.
        if tag_data is None:
            tag_data = scene.tag_data or {}
        tags_list = []
        for fid in scene.file_ids:
            raw_tags = tag_data.get(fid, [])
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

        # Settings snapshot for recompute/recluster after load. Captured on the
        # GUI thread by callers (widgets are not thread-safe); manual saves fall
        # back to reading the live controls here.
        if settings_meta is None:
            settings_meta = self._session_settings_snapshot()

        # Live view + selection state (passed in by callers captured on the GUI
        # thread). Restored on load so the user returns to exactly where they were.

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
            "view": view_meta,
        }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: serialize to a temp file in the same directory, then
        # rename over the target. A crash or a concurrent save can therefore
        # never leave a half-written latest.npz behind (np.savez_compressed
        # writes directly and is not atomic). The suffix MUST end in ".npz" —
        # np.savez_compressed appends ".npz" to any path that doesn't already
        # have it, which would otherwise write to a different file than the one
        # we rename.
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp.npz", dir=str(save_path.parent))
        os.close(fd)
        try:
            np.savez_compressed(
                tmp_path,
                positions=positions,
                cluster_labels=cluster_labels,
                metadata=json.dumps(metadata).encode('utf-8'),
            )
            os.replace(tmp_path, save_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return True

    def _capture_view_meta(self):
        """Capture the live camera view + current selection on the GUI thread.

        Returns a JSON-serializable dict with the actual camera state (center,
        distance, elevation, azimuth) and which cohort is selected. This reflects
        what the user is looking at right now — unlike scene.camera_position which
        is never updated from the live view. Used to persist the session so a later
        load restores exactly this viewpoint + selection.
        """
        import numpy as np
        meta = {"center": None, "distance": None, "elevation": None,
                "azimuth": None, "selected_cluster_id": None}
        try:
            view = getattr(self, 'gl_view', None)
            if view is not None:
                opts = view.opts
                center = opts.get('center')
                if hasattr(center, 'x'):  # QVector3D
                    meta["center"] = [float(center.x()), float(center.y()), float(center.z())]
                elif center is not None:
                    arr = np.asarray(center, dtype=float)
                    if arr.size == 3:
                        meta["center"] = [float(v) for v in arr.reshape(3)]
                for k in ("distance", "elevation", "azimuth"):
                    v = opts.get(k)
                    if v is not None:
                        try:
                            meta[k] = float(v)
                        except (TypeError, ValueError):
                            pass
            sel = getattr(self, 'selected_cluster_id', None)
            if sel is not None and int(sel) != -1:
                meta["selected_cluster_id"] = int(sel)

            # Session-specific navigation state (references this scene's cohort ids):
            # the WASD travel history + current position, and the Explore tour path +
            # position. Saved so a session reload restores where you were navigating /
            # exploring rather than starting over. Centroids are re-derived from the
            # loaded scene on restore (only cluster ids are stored).
            wasd_hist = getattr(self, '_wasd_history', []) or []
            if wasd_hist:
                meta["wasd_history"] = [int(cid) for cid, _c in wasd_hist]
                meta["wasd_history_index"] = int(getattr(self, '_wasd_history_index', -1))
            explore_path = getattr(self, '_explore_path', []) or []
            if explore_path:
                meta["explore_path"] = [int(p[0]) for p in explore_path]
                meta["explore_path_index"] = int(getattr(self, '_explore_path_index', 0))
        except Exception as e:
            print(f"[Session] Failed to capture view meta: {e}")
        return meta

    def _schedule_session_save(self):
        """Arm (or reset) the delayed session auto-save timer.

        Called after every operation that changes session data. Each call resets
        the single-shot timer, so the file is only written once no new change has
        occurred for ``session_save_delay`` seconds — eliminating costly writes
        while iterating settings or chaining operations. A delay of 0 saves
        immediately (legacy behavior). closeEvent bypasses this and saves now.
        """
        delay_s = int(getattr(self, 'session_save_delay', 60) or 0)
        if delay_s <= 0:
            self._auto_save_session()
            return
        # start(ms) on a running single-shot timer resets it to the full interval.
        self._session_save_timer.start(delay_s * 1000)

    def _auto_save_session(self):
        """Automatically save the current scene to sessions/latest.npz.

        Called (directly or via the delayed timer) after successful
        load/recompute/regroup/split/pop so that "Auto load session" can restore
        it on next launch without reprocessing.

        Runs in a BACKGROUND worker: serializing + compressing a large scene
        takes seconds at 100K+ files and must not freeze the UI (EXP-005). The
        scene, tag data, and settings snapshot are captured here on the GUI
        thread; the worker only reads them. A new auto-save cancels/replaces
        any still-running one (latest state wins), and closeEvent waits for a
        pending save to finish so nothing is lost on exit. Never raises.
        """
        try:
            from pathlib import Path

            scene = self.scene_graph
            if scene is None or not getattr(scene, 'file_ids', None):
                return

            # Capture everything the worker needs NOW (GUI thread). The scene's
            # own arrays are never mutated in place after build (recluster/pop/
            # deorphan all construct new SceneGraph objects), so sharing them is
            # safe; tag_data, settings, and view state are passed by value.
            save_scene = scene
            save_tag_data = self.tag_data
            save_settings = self._session_settings_snapshot()
            save_view_meta = self._capture_view_meta()

            def worker_func():
                return self._save_session_to_path(
                    Path("sessions") / "latest.npz",
                    scene=save_scene,
                    tag_data=save_tag_data,
                    settings_meta=save_settings,
                    view_meta=save_view_meta,
                )

            # Replace a still-running auto-save (its result is stale by then).
            old = getattr(self, '_autosave_worker', None)
            if old is not None and old.isRunning():
                try:
                    old.wait(2000)
                except RuntimeError:
                    pass
            self._autosave_worker = WorkerThread(worker_func)
            self._autosave_worker.finished.connect(self._on_auto_save_finished)
            self._autosave_worker.start()
        except Exception as e:
            print(f"[Session] Auto-save failed: {e}")

    def _on_auto_save_finished(self, result):
        """Background auto-save completed (worker thread signal -> GUI)."""
        if result:
            print("[Session] Auto-saved to sessions/latest.npz")
        else:
            print("[Session] Auto-save skipped (no scene)")

    def _wait_pending_auto_save(self, timeout_ms=10000):
        """Block until a pending background auto-save finishes (used on close)."""
        worker = getattr(self, '_autosave_worker', None)
        if worker is not None and worker.isRunning():
            try:
                worker.wait(timeout_ms)
            except RuntimeError:
                pass

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
            view_meta = metadata.get('view')  # live camera + selection

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

            # Restore settings for recompute/recluster
            self._apply_session_settings(settings_meta)

            self.scene_graph = scene
            self.node_list = scene.get_file_ids()
            self.tag_data = tag_data  # tags ARE persisted in the session (strings)

            n_loaded = len(file_ids)
            self.status_label.setText(f"Session loaded: {n_loaded} nodes")
            print(f"[Session] Loaded {n_loaded} nodes from {filename}")
            # Mark camera as initialized so render_scene doesn't auto-fit over the
            # saved viewpoint we're about to restore.
            self._camera_initialized = True
            self.render_scene(scene)

            # Restore the live camera view + selection captured at save time, so the
            # user returns to exactly where they were (applied after render so it wins).
            if isinstance(view_meta, dict):
                from PySide6.QtGui import QVector3D
                center = view_meta.get("center")
                if center and len(center) == 3:
                    self.gl_view.opts['center'] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
                for k in ("distance", "elevation", "azimuth"):
                    v = view_meta.get(k)
                    if v is not None:
                        try:
                            self.gl_view.opts[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                self.gl_view.update()

                # Restore the selected cohort (if it still exists in this scene).
                sel_cid = view_meta.get("selected_cluster_id")
                if sel_cid is not None:
                    member_idx = np.where(scene.cluster_ids == sel_cid)[0]
                    if len(member_idx) > 0:
                        self.show_cluster_info(int(member_idx[0]))

                # Restore the WASD travel history (re-derive centroids from this scene;
                # drop any cohort ids that no longer exist). Redraws the trail.
                wasd_cids = view_meta.get("wasd_history")
                if isinstance(wasd_cids, list) and wasd_cids:
                    hist = []
                    for cid in wasd_cids:
                        c = self._wasd_centroid(int(cid))
                        if c is not None:
                            hist.append((int(cid), np.asarray(c, dtype=float)))
                    if hist:
                        self._wasd_history = hist
                        idx = int(view_meta.get("wasd_history_index", len(hist) - 1))
                        self._wasd_history_index = max(0, min(idx, len(hist) - 1))
                        self._wasd_draw_trail()

                # Restore the Explore tour path + position (re-derive size/centroid).
                explore_cids = view_meta.get("explore_path")
                if isinstance(explore_cids, list) and explore_cids:
                    path = []
                    for cid in explore_cids:
                        c = self._wasd_centroid(int(cid))
                        if c is not None:
                            size = int(np.sum(scene.cluster_ids == int(cid)))
                            path.append((int(cid), size, np.asarray(c, dtype=float)))
                    if path:
                        self._explore_path = path
                        idx = int(view_meta.get("explore_path_index", 0))
                        self._explore_path_index = max(0, min(idx, len(path) - 1))

            # A session load is a valid "data ready" state — enable the same set
            # of action buttons that on_loading_finished() enables. Without this,
            # auto-loading a session on startup left every button greyed out.
            self.load_button.setEnabled(True)
            self.recompute_button.setEnabled(True)
            self.recluster_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
            self.deorphan_button.setEnabled(True)
            self.save_session_button.setEnabled(True)
            self.load_session_button.setEnabled(True)
            if hasattr(self, 'send_to_tab_btn'):
                self.send_to_tab_btn.setEnabled(True)
            if hasattr(self, 'time_travel_button'):
                self.time_travel_button.setEnabled(True)
            self._set_cohort_action_buttons(True)

            # Show basic scene stats now that we're idle (not processing).
            self._update_idle_status()

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
            self.deorphan_button.setEnabled(True)
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
        self.deorphan_button.setEnabled(True)
        self.save_session_button.setEnabled(True)
        self.load_session_button.setEnabled(True)
        self.send_to_tab_btn.setEnabled(True)
        self.time_travel_button.setEnabled(True)
        self._set_cohort_action_buttons(True)
        self.status_label.setText("Ready - Left: cohort | Ctrl+Left: node | Right: move camera | F11: Fullscreen")

        # Re-cluster results keep positions unchanged -> update colors in-place
        # (much lighter than full render_scene, avoids UI lock on 55k nodes).
        was_recluster = getattr(self, '_pending_recluster', False)
        if was_recluster:
            self._pending_recluster = False
            self.node_list = scene.get_file_ids()
            self._build_base_scatter()
            self._apply_highlight_colors(self._base_colors_rgba)
            self._update_cohort_labels()
        else:
            self.render_scene(scene)

        # Schedule a delayed auto-save (reset by each subsequent change) so that
        # iterating settings / chaining operations doesn't trigger costly writes;
        # closeEvent forces an immediate save so nothing is lost on exit.
        self._schedule_session_save()

        # Auto-split: start a fresh cycle after any operation that produces a new
        # clustering result (Load & Compute, Split group, Regroup, Cut out, Pop),
        # and continue the loop when a split from an in-flight cycle just finished.
        # Each auto-split is itself a re-cluster, so mid-cycle completions take the
        # "continue" branch; once the cycle ends (goal reached / max cycles / no
        # shrink) _auto_split_active is False and the next op starts a new one.
        if self._auto_split_active:
            self._auto_split_step()
        else:
            self._start_auto_split()

        # Auto-Deorphan: optionally assign noise (-1) nodes to their nearest
        # cohort after the chosen operation. Skip when this completion IS a
        # deorphan (its flags are set by start_deorphan) to avoid an infinite loop,
        # and skip while an auto-split cycle is active so both don't fight for the
        # single worker slot (auto-split takes priority).
        was_deorphan = getattr(self, '_pending_deorphan', False)
        if was_deorphan:
            self._pending_deorphan = False
        else:
            mode = getattr(self, 'auto_deorphan', "Never")
            # Don't let deorphan steal the worker slot while an auto-split cycle
            # is in flight (each split is a re-cluster round-trip).
            should_deorphan = (
                not self._auto_split_active and (
                    (mode == "After Load and Compute" and not was_recluster) or
                    (mode == "After Regroup" and was_recluster)
                )
            )
            if should_deorphan:
                self.start_deorphan()

        # Show basic scene stats in the status bar now that we're idle. If a new
        # worker (auto-split / deorphan) was just started, skip — its completion
        # will refresh the stats once processing is truly done.
        if not self._is_worker_busy():
            self._update_idle_status()

    def on_loading_error(self, error_msg):
        """Handle loading error."""
        # A failed split aborts the auto-split cycle (otherwise _auto_split_active
        # would stay True and block deorphan / future cycles).
        self._auto_split_active = False
        self.load_button.setEnabled(True)
        self.recompute_button.setEnabled(True)
        self.recluster_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        self.deorphan_button.setEnabled(True)
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

            # A full scene rebuild invalidates any WASD preview paths (they point at
            # the old scene's centroids) — drop them and exit navigation mode. The
            # persistent travel trail + explore path preview are also stale now.
            self._wasd_mode = False
            self._wasd_clear_paths()
            self._reset_wasd_trail()
            self._explore_clear_path_preview()
            # A new scene invalidates any remembered Explore tour (its cohorts no
            # longer exist), so the next Explore starts fresh from index 0.
            self._explore_path = []
            self._explore_path_index = 0

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

            # Scale text items (cohort labels) for supersample rendering.
            # Their paint() method reads _ss_factor to scale font + culling bounds.
            saved_text_factors = []
            for item in getattr(self, 'cohort_label_items', []) or []:
                try:
                    orig_factor = getattr(item, '_ss_factor', 1.0)
                    item._ss_factor = float(ss)
                    saved_text_factors.append((item, orig_factor))
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
                for item, orig_factor in saved_text_factors:
                    try:
                        item._ss_factor = orig_factor
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

            # Update text info (without tags) — styled label/value rows
            self.info_text.setHtml(render_info_rows([
                ("File ID", str(fid)),
                ("Cluster", str(cluster_id)),
                ("Score", f"{score:.2f}"),
                ("Position", f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"),
            ]))

            # Create clickable tag widgets (tags live in self.tag_data)
            node_tags = (self.tag_data or {}).get(fid, [])
            tags_to_show = list(node_tags[:42])  # Limit to 42 tags for performance
            if self.tag_interner:
                tags_to_show = self.tag_interner.strings_to_list(tags_to_show)
            if len(node_tags) > 42:
                # Add a label to indicate more tags exist
                more_label = QLabel(f"... and {len(node_tags) - 42} more tags")
                more_label.setStyleSheet("color: rgb(140, 146, 156); font-size: 10px; font-style: italic;")
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
        # If WASD navigation paths are showing and this selection change did NOT come
        # from a WASD press (WASD sets _wasd_selecting True before calling here), fade
        # them out over ~2 s instead of leaving stale arrows on screen.
        if not getattr(self, '_wasd_selecting', False):
            self._wasd_end_fade()

        # Capture + start fading any labels that are dropping due to this selection
        # switch, BEFORE clear_selection() removes them (otherwise they'd vanish
        # instantly). Only active in "Selected & N neighbors" mode with fade enabled.
        if hasattr(self, 'node_list') and self.node_list:
            _scene = self.scene_graph
            if _scene is not None and 0 <= node_index < len(_scene.file_ids):
                self._capture_dropping_labels(target_cid=int(_scene.cluster_ids[node_index]))

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
            
            # Update text info — styled label/value rows (the tag list itself is
            # the clickable grid below, so no "Common Tags:" header needed here)
            self.info_text.setHtml(render_info_rows([
                ("Cohort", f"{cluster_id}"),
                ("Files", f"{len(cluster_nodes):,}"),
                ("Unique tags", f"{len(tag_counts):,}"),
            ]))

            # Create clickable tag widgets for top tags
            tags_to_show = sorted_tags[:42]  # Limit to 42 tags
            if len(sorted_tags) > 42:
                more_label = QLabel(f"... and {len(sorted_tags) - 42} more tags")
                more_label.setStyleSheet("color: rgb(140, 146, 156); font-size: 10px; font-style: italic;")
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


    def keyPressEvent(self, event):
        """Handle key press events for fullscreen toggle and shortcuts."""
        # WASD/QE: screen-space cohort navigation (only when no modifier is held so
        # Ctrl+W / Ctrl+E etc. keep their normal meaning). W/S/A/D move to the nearest
        # cohort in that direction; Q/E step back/forward through the travel history.
        if not event.modifiers() & Qt.ControlModifier \
                and not event.modifiers() & Qt.ShiftModifier:
            key = {Qt.Key_W: 'W', Qt.Key_S: 'S', Qt.Key_A: 'A', Qt.Key_D: 'D',
                   Qt.Key_Q: 'Q', Qt.Key_E: 'E'}.get(event.key())
            if key is not None:
                # While Explore is running, Q/E steer the tour (previous / next
                # cohort), W/S adjust camera distance to the cohort. These take
                # priority over WASD travel-history navigation.
                if getattr(self, '_explore_active', False):
                    if key in ('Q', 'E'):
                        self._explore_jump(1 if key == 'E' else -1)
                        event.accept()
                        return
                    if key in ('W', 'S'):
                        # W = closer (reduce orbit radius), S = farther (increase).
                        # Multiplicative for consistent feel at any zoom level.
                        factor = 0.85 if key == 'W' else 1.18
                        self._explore_orbit_radius = max(
                            0.5, self._explore_orbit_radius * factor)
                        event.accept()
                        return
                self._wasd_mode = True
                self._wasd_handle_key(key)
                event.accept()
                return
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

    # ------------------------------------------------------------------
    # Explore — "helicopter orbit" mode
    # ------------------------------------------------------------------

    def _toggle_time_travel(self):
        """Toggle the Explore (helicopter-orbit) animation on/off."""
        if self._explore_active:
            self._stop_explore()
        else:
            self._start_explore()

    def _build_explore_cohorts(self):
        """Collect all non-noise cohorts as (cluster_id, size, centroid*spread)."""
        import numpy as np
        scene = self.scene_graph
        cids_arr = scene.cluster_ids
        spread = float(self.spread_spin.value())
        out = []
        for cid in np.unique(cids_arr):
            if int(cid) == -1:
                continue
            idx = np.where(cids_arr == cid)[0]
            centroid = scene.positions[idx].mean(axis=0) * spread
            out.append((int(cid), len(idx), centroid))
        return out

    def _build_explore_path(self, cohorts):
        """Build the ordered list of cohort targets to visit, per explore mode.

        - Random:      shuffled order (every cohort once).
        - Linear Path: start at one spatial extreme and greedily hop to the nearest
                       unvisited cohort — a short-step sweep that ends near the other
                       extreme while passing through every cohort.
        - Contrast:    greedy farthest-point sampling — each next cohort is the one
                       most distant from ALL previously visited, maximizing variety.

        Returns a list of (cluster_id, size, centroid) tuples.
        """
        import random
        import numpy as np
        mode = getattr(self, 'explore_mode', "Random")
        if len(cohorts) <= 1:
            return list(cohorts)

        centroids = np.array([c[2] for c in cohorts], dtype=float)

        if mode == "Linear Path":
            # Find the two most distant centroids (the polar extremes).
            n = len(cohorts)
            max_d, a, b = 0.0, 0, 1
            for i in range(n):
                d = np.linalg.norm(centroids - centroids[i], axis=1)
                j = int(np.argmax(d))
                if d[j] > max_d:
                    max_d, a, b = float(d[j]), i, j
            # Greedy nearest-neighbor sweep starting from extreme `a`.
            order = [a]
            remaining = set(range(n)) - {a}
            while remaining:
                last = centroids[order[-1]]
                dists = np.array([np.linalg.norm(centroids[i] - last) for i in remaining])
                next_i = list(remaining)[int(np.argmin(dists))]
                order.append(next_i)
                remaining.discard(next_i)
            return [cohorts[i] for i in order]

        if mode == "Contrast":
            # Greedy farthest-point sampling: each pick maximizes the minimum distance
            # to every already-visited centroid (most varied tour).
            n = len(cohorts)
            start = int(np.argmax([c[1] for c in cohorts]))  # begin at largest cohort (c[1] is the size)
            order = [start]
            visited = {start}
            while len(order) < n:
                best_i, best_score = None, -1.0
                for i in range(n):
                    if i in visited:
                        continue
                    min_d = float(np.min(np.linalg.norm(centroids[i] - centroids[list(visited)], axis=1)))
                    if min_d > best_score:
                        best_score, best_i = min_d, i
                order.append(best_i)
                visited.add(best_i)
            return [cohorts[i] for i in order]

        # Default: Random (shuffled).
        shuffled = list(cohorts)
        random.shuffle(shuffled)
        return shuffled

    def _start_explore(self):
        """Begin the Explore helicopter-orbit animation."""
        if not hasattr(self, 'node_list') or not self.node_list:
            return
        try:
            import random
            cohorts = self._build_explore_cohorts()
            if len(cohorts) < 1:
                self.status_label.setText("Need at least 1 cohort for Explore")
                return

            # Resume from the previous position if a path was already built and all
            # of its cohorts still exist in the current scene (i.e. we're restarting
            # after Stop, not starting fresh). Otherwise build a new path from index 0.
            prev_path = getattr(self, '_explore_path', []) or []
            current_ids = {c[0] for c in cohorts}
            if prev_path and all(p[0] in current_ids for p in prev_path):
                self._explore_path = prev_path
                # Keep the remembered position (clamped to the path length).
                idx = getattr(self, '_explore_path_index', 0)
                self._explore_path_index = max(0, min(idx, len(prev_path) - 1))
            else:
                self._explore_path = self._build_explore_path(cohorts)
                self._explore_path_index = 0
            self._explore_cohorts = cohorts
            self._explore_active = True
            # Stop the selection blink while exploring (media viewer stays synced).
            if hasattr(self, 'selection_timer'):
                self.selection_timer.stop()

            self.time_travel_button.setText("Stop")
            self.time_travel_button.setToolTip("Stop the Explore helicopter-orbit animation.")
            self.time_travel_timer.start(33)  # ~30 fps

            # Show the planned route (color-coded by phase) before flying it.
            self._explore_draw_path_preview()

            # Begin at the remembered position (resumes after Stop).
            self._explore_begin_target(self._explore_path[self._explore_path_index])
        except Exception as e:
            print(f"Error starting explore: {e}")
            import traceback
            traceback.print_exc()
            self._stop_explore()

    def _stop_explore(self):
        """Stop Explore and restore the button + selection blink."""
        self._explore_active = False
        self.time_travel_timer.stop()
        self._remove_explore_marker()
        self._explore_clear_path_preview()
        self.time_travel_button.setText("Explore")
        self.time_travel_button.setToolTip(
            "Fly the camera around random cohorts (helicopter orbit).")
        # Re-enable the selection blink if a cohort is still selected.
        if hasattr(self, 'selection_timer') and self.selected_cluster_id not in (None, -1):
            self.selection_timer.start(500)
        self.status_label.setText("Explore stopped")

    def _explore_jump(self, direction):
        """Jump the Explore tour to the next (+1) or previous (-1) cohort.

        Bound to E (next) / Q (previous) while Explore is running. Moves along the
        remembered visit path and starts a fresh approach from wherever the camera
        currently is — so it banks into the new orbit with momentum, same as the
        automatic advance when an orbit completes.
        """
        if not getattr(self, '_explore_active', False):
            return
        path = getattr(self, '_explore_path', []) or []
        n = len(path)
        if n == 0:
            return
        idx = (getattr(self, '_explore_path_index', 0) + direction) % n
        self._explore_path_index = idx
        self._explore_begin_target(path[idx])
        self._explore_draw_path_preview()

    def _explore_begin_target(self, target):
        """Start approaching a new cohort: select it, compute the orbit geometry.

        The camera look-at point is offset from the centroid so the cohort sits in
        the left or right third of the screen while we circle around it (the
        "helicopter" framing).
        """
        import random
        import numpy as np
        cid, size, centroid = target
        self._explore_target_cid = cid
        self._explore_target_size = int(size)
        self._explore_target_centroid = np.asarray(centroid, dtype=float)

        # Orbit radius scales with cohort size so big cohorts get more room. The
        # floor is tiny (0.5) so a small base value actually produces a close orbit —
        # the old max(4.0, …) forced every orbit to be at least 4 units out.
        base = float(getattr(self, 'explore_orbit_radius_base', 8.0))
        factor = float(getattr(self, 'explore_orbit_size_factor', 2.0))
        self._explore_orbit_radius = max(0.5, base + factor * np.sqrt(max(size, 1)))

        # Orbit directly around the cohort centroid (no offset) so the target stays
        # fixed at screen center while we circle it — an offset look-at made it
        # unclear which cohort was being orbited. The red marker below sits on this
        # same point, reinforcing what's being circled.
        self._explore_look_center = np.asarray(self._explore_target_centroid, dtype=float)

        # Entry azimuth: approach settles into this heading before orbiting.
        self._explore_entry_azimuth = random.uniform(0.0, 360.0)
        self._explore_cycles_done = 0
        self._explore_orbit_turns = 0.0
        self._explore_orbit_time = 0.0

        # Place a bright marker on the target centroid so it's obvious which cohort
        # we're orbiting (the camera look-at is offset, which can be confusing).
        try:
            import pyqtgraph.opengl as gl
            from PySide6.QtGui import QColor
            self._remove_explore_marker()
            marker = gl.GLScatterPlotItem(
                pos=np.array([self._explore_target_centroid]), size=14,
                color=(255, 80, 80, 255), pxMode=True)
            self.gl_view.addItem(marker)
            self._explore_marker_item = marker
        except Exception as e:
            print(f"Error creating explore marker: {e}")

        # Capture the current camera state as the approach start point, and compute
        # the orbit-entry camera position so the approach can fly a real arc between
        # them (a "flight path") instead of lerping distance/elevation/azimuth.
        opts = self.gl_view.opts
        cur_center = opts.get('center')
        if hasattr(cur_center, 'x'):
            self._explore_from_center = np.array([cur_center.x(), cur_center.y(), cur_center.z()])
        else:
            self._explore_from_center = np.asarray(cur_center, dtype=float).reshape(3)
        self._explore_from_dist = float(opts.get('distance', 100.0))
        self._explore_from_elev = float(opts.get('elevation', 20.0))
        self._explore_from_azim = float(opts.get('azimuth', 0.0))

        # Flight-path arc endpoints (actual 3D camera positions).
        try:
            cp = self.gl_view.cameraPosition()
            self._explore_cam_start = np.array([float(cp.x()), float(cp.y()), float(cp.z())])
        except Exception:
            self._explore_cam_start = None
        self._explore_cam_end = self._explore_cam_pos(
            self._explore_look_center, self._explore_orbit_radius,
            float(self.explore_elevation), self._explore_entry_azimuth)

        # Capture the camera's current velocity so the approach spline starts with
        # momentum (C1 continuous through orbit -> approach). When transitioning from
        # an active orbit, that's the tangential velocity at the current azimuth. For
        # the very first target there is no prior motion, so start_vel stays None and
        # the approach uses a gentle climb-out tangent instead.
        if self._explore_phase == "orbit":
            self._explore_cam_start_vel = self._explore_orbit_velocity(self._explore_azimuth)
        else:
            self._explore_cam_start_vel = None

        # Select the cohort (syncs media viewer + panels) then stop its blink.
        member_idx = np.where(self.scene_graph.cluster_ids == cid)[0]
        if len(member_idx) > 0:
            self.show_cluster_info(int(member_idx[0]))
            if hasattr(self, 'selection_timer'):
                self.selection_timer.stop()

        self._explore_phase = "approach"
        self._explore_approach_t = 0.0
        self.status_label.setText(
            f"Explore: Cohort {cid} ({size:,} files) — approaching")

    def _update_time_travel(self):
        """Per-frame (~30 fps) update for the Explore helicopter-orbit animation."""
        if not self._explore_active or not self._explore_cohorts:
            return
        import numpy as np

        if self._explore_phase == "approach":
            # Accelerate toward the target while orienting into the orbit. The camera
            # flies along a quadratic-Bezier arc (start -> elevated control -> orbit
            # entry) so it follows a real flight path rather than just rising/falling,
            # while the look-at point eases from the old center to the new one.
            accel = max(1e-5, float(getattr(self, 'explore_accel', 0.6)))
            decel = max(1e-5, float(getattr(self, 'explore_decel', 0.6)))
            # Approach duration (frames) from distance; higher accel/decel = faster.
            dist_now = float(np.linalg.norm(
                self._explore_look_center - self._explore_from_center))
            # No speed floor: tiny accel/decel genuinely produce a slow approach.
            speed = max(1e-4, (accel + decel) * 25.0)  # units/sec-ish
            # Cap the approach at ~3 min so an extremely small value never freezes it.
            duration_frames = min(5400, max(8, int((dist_now / speed) * 30.0)))
            self._explore_approach_t += 1.0 / duration_frames

            t = min(1.0, self._explore_approach_t)
            # Look-at center eases toward the target centroid (arrives at rest, since
            # during orbiting the aim is fixed on the cohort). Smoothstep gives zero
            # end-velocity so the view doesn't rotate abruptly at handoff.
            eased = self._smoothstep_cubic(t)
            look_center = (self._explore_from_center
                           + eased * (self._explore_look_center - self._explore_from_center))

            # Camera position follows a cubic Hermite spline from the start point to
            # the orbit-entry point. The END tangent is set to the orbit's tangential
            # velocity, so the camera arrives already moving in the orbital direction
            # at orbital speed — no full stop, no hard turn (a "rocket banking into
            # its orbit"). A small initial upward tangent gives a gentle climb-out.
            cam_start = getattr(self, '_explore_cam_start', None)
            cam_end = getattr(self, '_explore_cam_end', None)
            if cam_start is not None and cam_end is not None:
                D = duration_frames / 30.0  # approach duration in seconds
                m1 = self._explore_orbit_velocity(self._explore_entry_azimuth)
                # Start tangent: use the captured orbital velocity (momentum from the
                # previous orbit) if available; otherwise a gentle upward climb-out.
                start_vel = getattr(self, '_explore_cam_start_vel', None)
                if start_vel is not None:
                    m0 = np.asarray(start_vel, dtype=float)
                else:
                    m0 = np.array([0.0, 0.0, 0.25 * float(np.linalg.norm(m1))])
                u = t
                h00 = 2.0 * u ** 3 - 3.0 * u ** 2 + 1.0
                h10 = u ** 3 - 2.0 * u ** 2 + u
                h01 = -2.0 * u ** 3 + 3.0 * u ** 2
                h11 = u ** 3 - u ** 2
                cam_pos = (h00 * cam_start + h10 * (m0 * D)
                           + h01 * cam_end + h11 * (m1 * D))
            else:
                # Fallback: no arc data — just place the camera at the orbit entry.
                cam_pos = self._explore_cam_pos(
                    look_center, self._explore_orbit_radius,
                    float(self.explore_elevation), self._explore_entry_azimuth)

            # Derive spherical params from the spline position so setCameraPosition
            # stays consistent (distance/elevation/azimuth relative to the look-at).
            rel = cam_pos - look_center
            dist = max(0.1, float(np.linalg.norm(rel)))
            elev = float(np.degrees(np.arcsin(max(-1.0, min(1.0, rel[2] / dist)))))
            azim = float(np.degrees(np.arctan2(rel[1], rel[0])))

            self._explore_apply_camera(look_center, dist, elev, azim)

            if t >= 1.0:
                # Arrived at the orbit-entry point already moving tangentially at
                # orbital speed -> hand straight off to steady-state orbiting (C1).
                self._explore_phase = "orbit"
                self._explore_azimuth = self._explore_entry_azimuth
                self.status_label.setText(
                    f"Explore: Cohort {self._explore_target_cid} "
                    f"({self._explore_target_size:,} files) — orbiting")

        elif self._explore_phase == "orbit":
            # Circle the look-at center at a steady angular speed. Timer runs at ~30 fps.
            orbit_speed = max(0.1, float(getattr(self, 'explore_orbit_speed', 12.0)))
            azim_step = orbit_speed / 30.0  # deg per frame at ~30 fps
            self._explore_azimuth += azim_step
            self._explore_orbit_turns += azim_step
            self._explore_orbit_time += 1.0 / 30.0

            cycles = max(1, int(getattr(self, 'explore_cycles', 3)))
            max_time = float(getattr(self, 'explore_max_orbit_time', 30.0))
            # Advance when EITHER the cycle count or the max orbit time is reached.
            if self._explore_orbit_turns >= cycles * 360.0 or (max_time > 0 and self._explore_orbit_time >= max_time):
                # Done orbiting this cohort — advance along the visit path (loops).
                self._explore_path_index = (self._explore_path_index + 1) % len(self._explore_path)
                self._explore_begin_target(self._explore_path[self._explore_path_index])
                # Refresh the windowed route preview around the new current stop.
                self._explore_draw_path_preview()
                return

            self._explore_apply_camera(
                self._explore_look_center, self._explore_orbit_radius,
                float(self.explore_elevation), self._explore_azimuth)

    def _explore_orbit_velocity(self, azim_deg):
        """Tangential velocity vector (units/sec) of the orbit at `azim_deg`.

        This is d(position)/dt for the circular orbit = dP/d(azimuth) * omega, where
        omega is the constant orbital angular speed. Used as the Hermite end-tangent
        so the approach arrives already moving in the orbital direction (no stop).
        """
        import numpy as np
        R = float(self._explore_orbit_radius)
        elev = np.radians(float(self.explore_elevation))
        azim = np.radians(azim_deg)
        omega = np.radians(max(0.1, float(getattr(self, 'explore_orbit_speed', 12.0))))
        return R * omega * np.array([
            -np.cos(elev) * np.sin(azim),
             np.cos(elev) * np.cos(azim),
             0.0,
        ])

    @staticmethod
    def _explore_cam_pos(center, dist, elev_deg, azim_deg):
        """Camera world position for a given look-at center + spherical params.

        Matches pyqtgraph's euler convention: x = d·cos(elev)·cos(azim), etc. Used to
        build the flight-path arc so the camera moves along a real curve rather than
        lerping distance/elevation/azimuth independently (which looks like it just
        rises/falls instead of flying).
        """
        import numpy as np
        elev = np.radians(elev_deg)
        azim = np.radians(azim_deg)
        return np.array([
            center[0] + dist * np.cos(elev) * np.cos(azim),
            center[1] + dist * np.cos(elev) * np.sin(azim),
            center[2] + dist * np.sin(elev),
        ])

    def _remove_explore_marker(self):
        """Remove the bright target marker if present."""
        item = getattr(self, '_explore_marker_item', None)
        if item is not None:
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
            self._explore_marker_item = None

    def _explore_apply_camera(self, center, dist, elev, azim):
        """Set the camera to look at `center` from (dist, elev, azim)."""
        try:
            from PySide6.QtGui import QVector3D
            self.gl_view.setCameraPosition(
                pos=QVector3D(float(center[0]), float(center[1]), float(center[2])),
                distance=float(dist), elevation=float(elev), azimuth=float(azim))
            self.gl_view.update()
        except Exception as e:
            print(f"Error applying explore camera: {e}")

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

    # ------------------------------------------------------------------
    # Smooth center transition animation
    # ------------------------------------------------------------------

    @staticmethod
    def _smoothstep_cubic(t):
        """Cubic smoothstep: 6t⁵ − 15t⁴ + 10t³.

        Zero first and second derivative at t=0 and t=1, giving a very
        natural ease-in-out with no sudden acceleration changes.
        """
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _start_center_animation(self, target):
        """Begin a smooth camera-center glide toward *target* (np.array(3)).

        Duration is proportional to travel distance (clamped 0.25 s – 2.0 s)
        so short hops feel snappy and long traversals don't drag.
        """
        import numpy as np
        from PySide6.QtGui import QVector3D

        # Read current center (may be QVector3D or ndarray)
        cur = self.gl_view.opts.get('center', np.array([0.0, 0.0, 0.0]))
        if isinstance(cur, np.ndarray):
            start = cur.copy()
        elif hasattr(cur, 'x'):
            start = np.array([float(cur.x()), float(cur.y()), float(cur.z())])
        else:
            start = np.array([float(v) for v in cur])

        target = np.asarray(target, dtype=np.float64)
        dist = float(np.linalg.norm(target - start))

        # Duration from the exposed speed (units/s), clamped so short hops stay
        # snappy (≥ 0.25 s) and absurdly long traversals don't drag (≤ 15 s).
        # At low speeds (e.g. default 1 u/s) most moves land well under the cap,
        # so duration scales proportionally with distance.
        speed = max(0.1, float(getattr(self, 'smooth_center_speed', 1.0)))
        self._center_anim_duration = max(0.25, min(15.0, dist / speed))

        self._center_anim_start = start
        self._center_anim_target = target
        self._center_anim_t0 = time.perf_counter()
        self._center_anim_active = True
        self._center_anim_timer.start()

    def _update_center_animation(self):
        """Per-frame tick for the smooth center transition."""
        import numpy as np
        from PySide6.QtGui import QVector3D

        if not self._center_anim_active:
            return

        elapsed = time.perf_counter() - self._center_anim_t0
        raw_t = min(1.0, elapsed / self._center_anim_duration)
        eased_t = self._smoothstep_cubic(raw_t)

        new_center = (self._center_anim_start
                      + eased_t * (self._center_anim_target - self._center_anim_start))

        self.gl_view.opts['center'] = QVector3D(
            float(new_center[0]), float(new_center[1]), float(new_center[2]))
        self.gl_view.update()

        if raw_t >= 1.0:
            # Snap to exact target and stop the timer.
            t = self._center_anim_target
            self.gl_view.opts['center'] = QVector3D(
                float(t[0]), float(t[1]), float(t[2]))
            self.gl_view.update()
            self._center_anim_active = False
            self._center_anim_timer.stop()

    def _cancel_center_animation(self):
        """Stop any in-flight center animation (e.g. user grabs the camera)."""
        if self._center_anim_active:
            self._center_anim_active = False
            self._center_anim_timer.stop()

    # ------------------------------------------------------------------
    # Cohort label fade-out (selection switching)
    # ------------------------------------------------------------------

    def _start_label_fade(self, items):
        """Begin fading out the given cohort-label items over ~0.3 s.

        Each item's alpha is ramped to 0 by a timer; when it reaches 0 the item
        is removed from the GL view. Used so labels that drop out of "Selected &
        N neighbors" mode fade instead of vanishing instantly on selection change.
        """
        if not items:
            return
        # Reset any in-flight fades so rapid selection changes don't stack stale
        # entries (the previous fading labels are simply dropped now).
        self._cancel_label_fades()
        for entry in items:
            self._label_fade_items.append(entry)
        if not self._label_fade_timer.isActive():
            self._label_fade_timer.start()

    def _update_label_fade(self):
        """Per-frame tick that ramps fading labels to transparent, then removes them."""
        from PySide6.QtGui import QColor
        if not self._label_fade_items:
            self._label_fade_timer.stop()
            return
        # Alpha decrement per tick derived from the configured fade duration so a
        # full 255->0 ramp takes exactly that long at ~60 fps (16 ms/tick).
        try:
            dur_ms = max(1, int(self.label_fade_duration_spin.value()))
        except Exception:
            dur_ms = 300
        ticks = max(1, round(dur_ms / 16.0))
        step = max(1, int(round(255.0 / ticks)))
        remaining = []
        for entry in self._label_fade_items:
            item, base_rgba = entry
            a = max(0, int(base_rgba[3]) - step)
            try:
                if a <= 0:
                    self.gl_view.removeItem(item)
                else:
                    item.color = QColor(int(base_rgba[0]), int(base_rgba[1]),
                                        int(base_rgba[2]), a)
                    item.update()
                    remaining.append([item, (int(base_rgba[0]), int(base_rgba[1]),
                                             int(base_rgba[2]), a)])
            except Exception:
                # Item already gone / view torn down — drop it.
                continue
        self._label_fade_items = remaining
        if not self._label_fade_items:
            self._label_fade_timer.stop()

    def _cancel_label_fades(self):
        """Immediately remove any labels mid-fade (used on a hard label rebuild)."""
        for entry in self._label_fade_items:
            try:
                self.gl_view.removeItem(entry[0])
            except Exception:
                pass
        self._label_fade_items = []
        if self._label_fade_timer.isActive():
            self._label_fade_timer.stop()

    # ------------------------------------------------------------------
    # Auto-split oversized cohorts
    # ------------------------------------------------------------------

    def _auto_split_step(self):
        """One cycle of the auto-split loop.

        Finds the largest non-noise cohort in the current scene. If it exceeds
        ``auto_split_threshold``, selects it and runs a re-cluster (the same
        operation as the "Split group" button). The next step is driven from
        ``on_loading_finished`` once that worker completes, so each cycle is a
        full select -> split -> check round-trip.

        Stops when: no oversized cohort remains (goal reached), max cycles hit,
        or a split failed to shrink its target (early stop — further cycles with
        the same sub-DBSCAN parameters would not help).
        """
        import numpy as np

        threshold = int(getattr(self, 'auto_split_threshold', 0) or 0)
        max_cycles = int(getattr(self, 'auto_split_max_cycles', 0) or 0)
        if threshold <= 0 or max_cycles <= 0:
            self._auto_split_active = False
            return

        scene = getattr(self, 'scene_graph', None)
        if scene is None or not hasattr(self, 'node_list') or not self.node_list:
            self._auto_split_active = False
            return

        # --- Early stop: did the previous split actually shrink its target? ---
        # Re-cluster REMAPS the target's labels to new sub-cohort ids, so we track
        # membership by file_id and measure the largest cluster those files now
        # fall into. If that is still >= the old size, the split produced no net
        # reduction (DBSCAN returned a single cluster again) — stop early instead
        # of burning all max_cycles with identical parameters.
        if self._auto_split_target_members is not None:
            old_size = int(self._auto_split_target_size)
            member_mask = np.isin(scene.file_ids, list(self._auto_split_target_members))
            member_labels = scene.cluster_ids[member_mask]
            # Largest group among the members (noise -1 excluded from "cohort" size).
            non_noise = member_labels[member_labels != -1]
            if len(non_noise) > 0:
                _, counts = np.unique(non_noise, return_counts=True)
                largest_member_cluster = int(counts.max())
            else:
                largest_member_cluster = 0
            if largest_member_cluster >= old_size:
                print(f"[AutoSplit] Split did not shrink target "
                      f"({old_size} -> {largest_member_cluster}); stopping.")
                self.status_label.setText(
                    f"Auto-split: no further reduction possible after "
                    f"{self._auto_split_cycles_done} cycle(s).")
                self._auto_split_active = False
                return

        # --- Cycle budget exhausted? ---
        if self._auto_split_cycles_done >= max_cycles:
            print(f"[AutoSplit] Max cycles ({max_cycles}) reached; stopping.")
            self.status_label.setText(
                f"Auto-split: stopped after {max_cycles} cycle(s).")
            self._auto_split_active = False
            return

        # --- Find the largest non-noise cohort ---
        cluster_ids = np.asarray(scene.cluster_ids)
        valid = cluster_ids[cluster_ids != -1]
        if len(valid) == 0:
            print("[AutoSplit] No cohorts to split; stopping.")
            self.status_label.setText("Auto-split: no cohorts found.")
            self._auto_split_active = False
            return

        sizes, ids = np.unique(valid, return_counts=True)
        largest_idx = int(np.argmax(sizes))
        largest_id = int(ids[largest_idx])
        largest_size = int(sizes[largest_idx])

        if largest_size <= threshold:
            # Goal condition reached: no cohort over the threshold.
            print(f"[AutoSplit] Largest cohort is {largest_size} "
                  f"(<= {threshold}); goal reached after "
                  f"{self._auto_split_cycles_done} cycle(s).")
            self.status_label.setText(
                f"Auto-split complete: largest cohort now {largest_size:,} files.")
            self._auto_split_active = False
            return

        # --- Select the oversized cohort and split it (like "Split group") ---
        print(f"[AutoSplit] Cycle {self._auto_split_cycles_done + 1}/{max_cycles}: "
              f"splitting cohort {largest_id} ({largest_size:,} files > {threshold:,}).")
        self.status_label.setText(
            f"Auto-split cycle {self._auto_split_cycles_done + 1}/{max_cycles}: "
            f"splitting cohort {largest_id} ({largest_size:,} files)...")

        # Select it (drives _recluster_selection, which reads selected_cluster_id).
        member_idx = int(np.where(cluster_ids == largest_id)[0][0])
        self.show_cluster_info(member_idx)

        # Record membership by file_id so the early-stop check can measure how
        # much the target shrank after re-cluster remaps its labels.
        self._auto_split_target_members = set(
            int(fid) for fid in scene.file_ids[cluster_ids == largest_id])
        self._auto_split_target_size = largest_size
        self._auto_split_cycles_done += 1

        # _recluster_selection sets _pending_recluster and starts the worker;
        # on_loading_finished will call _auto_split_step() again when done.
        self._recluster_selection()

    def _start_auto_split(self):
        """Begin an auto-split cycle (called after Load & Compute / Regroup)."""
        if not getattr(self, 'auto_split_enabled', True):
            return
        max_cycles = int(getattr(self, 'auto_split_max_cycles', 0) or 0)
        threshold = int(getattr(self, 'auto_split_threshold', 0) or 0)
        if max_cycles <= 0 or threshold <= 0:
            return
        self._auto_split_active = True
        self._auto_split_cycles_done = 0
        self._auto_split_target_members = None
        self._auto_split_target_size = 0
        self._auto_split_step()

    # ------------------------------------------------------------------
    # Smart Scale (node-count-based automatic settings)
    # ------------------------------------------------------------------

    def _on_smart_scale_toggled(self, state):
        """Master toggle for Smart Scale.

        Persists the flag and, when turned on with data already loaded, applies
        the matching profile immediately so the effect is visible without a reload.
        """
        self.smart_scale_enabled = bool(state)
        self.save_settings()
        if self.smart_scale_enabled:
            scene = getattr(self, 'scene_graph', None)
            if scene is not None and len(getattr(scene, 'file_ids', [])) > 0:
                self._apply_smart_scale_to_widgets(len(scene.file_ids))

    def _smart_scale_profile_for(self, node_count):
        """Resolve the smart-scale profile for a file count (None if disabled)."""
        if not getattr(self, 'smart_scale_enabled', False):
            return None
        from src.ui.smart_scale import resolve_profile
        return resolve_profile(getattr(self, 'smart_scale_profiles', []), node_count)

    def _apply_smart_scale_to_widgets(self, node_count):
        """Apply the resolved profile's values to the setting widgets.

        Used after a fresh load so the UI reflects the parameters that were used
        for computation and so visual params (node size / transparency / spread)
        take effect on render. No-op when Smart Scale is off or no profile matches.
        """
        from src.ui.smart_scale import apply_profile_to_tab
        profile = self._smart_scale_profile_for(node_count)
        if profile is None:
            return False
        apply_profile_to_tab(self, profile)
        print(f"[SmartScale] Applied profile for endpoint "
              f"{profile.get('endpoint')} ({node_count:,} files).")
        return True

    def _smart_scale_apply_for_load(self, node_count):
        """Apply the matching profile to widgets BEFORE a load's computation.

        Called on the GUI thread from start_loading() using the expected file
        count (max_files_spin) as the size estimate, so the UMAP / DBSCAN / visual
        parameters read by the worker below reflect the size-appropriate profile.
        Returns True if a profile was applied (for logging). No-op when Smart Scale
        is disabled or no profiles are configured.
        """
        return self._apply_smart_scale_to_widgets(node_count)

    def _update_idle_status(self):
        """Show basic scene stats in the status bar while idle (not processing).

        Displays file count, cohort count, orphan (noise) count, and median cohort
        size. Called at the end of on_loading_finished only when no worker is
        running, so it's the resting state between operations.
        """
        scene = getattr(self, 'scene_graph', None)
        if scene is None or not getattr(scene, 'file_ids', None):
            self.status_label.setText("Ready - Left: cohort | Ctrl+Left: node | Right: move camera | F11: Fullscreen")
            return

        import numpy as np
        cluster_ids = np.asarray(scene.cluster_ids)
        file_count = int(len(scene.file_ids))
        orphan_count = int(np.sum(cluster_ids == -1))
        non_noise = cluster_ids[cluster_ids != -1]
        cohort_count = int(len(np.unique(non_noise))) if len(non_noise) else 0
        if cohort_count:
            _, sizes = np.unique(non_noise, return_counts=True)
            median_size = float(np.median(sizes))
            # Show as a clean integer when whole.
            median_str = str(int(median_size)) if median_size.is_integer() else f"{median_size:.1f}"
        else:
            median_str = "0"

        self.status_label.setText(
            f"Ready | Files: {file_count:,} | Cohorts: {cohort_count:,} | "
            f"Orphans: {orphan_count:,} | Median cohort: {median_str}"
        )

    # ------------------------------------------------------------------
    # WASD cohort navigation (screen-space) + preview paths
    # ------------------------------------------------------------------

    def _wasd_project_centroids(self):
        """Project every non-noise cohort centroid to screen space.

        Returns a list of (cluster_id, screen_x, screen_y) for cohorts in front of
        the camera, using the same MVP projection as point picking.
        """
        import numpy as np
        scene = getattr(self, 'scene_graph', None)
        if scene is None or not hasattr(self, 'node_list') or not self.node_list:
            return []
        cids_arr = scene.cluster_ids
        spread = float(self.spread_spin.value())
        positions = scene.positions * spread

        view_matrix = self.gl_view.viewMatrix()
        proj_matrix = self.gl_view.currentProjection()
        mvp = proj_matrix * view_matrix
        m = np.array([
            [mvp(0, 0), mvp(0, 1), mvp(0, 2), mvp(0, 3)],
            [mvp(1, 0), mvp(1, 1), mvp(1, 2), mvp(1, 3)],
            [mvp(2, 0), mvp(2, 1), mvp(2, 2), mvp(2, 3)],
            [mvp(3, 0), mvp(3, 1), mvp(3, 2), mvp(3, 3)],
        ])

        out = []
        for cid in np.unique(cids_arr):
            if int(cid) == -1:
                continue
            idx = np.where(cids_arr == cid)[0]
            centroid = positions[idx].mean(axis=0)
            p = np.array([centroid[0], centroid[1], centroid[2], 1.0]) @ m.T
            w = p[3]
            if abs(w) < 1e-10 or w <= 0:
                continue
            ndc = p[:3] / w
            if not (0.0 <= ndc[2] <= 1.0):
                continue
            width, height = self.gl_view.width(), self.gl_view.height()
            sx = (ndc[0] * 0.5 + 0.5) * width
            sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
            out.append((int(cid), float(sx), float(sy)))
        return out

    def _wasd_pick_targets(self):
        """Pick the nearest cohort in each screen direction (W/S/A/D) from the
        currently selected centroid. Returns dict {'W': cid, 'S': cid, ...}.

        Navigation is restricted to the cohorts that are CURRENTLY LABELED on
        screen (self.cohort_label_map), so WASD hops between the labels you can
        actually see rather than distant unlabeled clusters. If no labels are
        shown it falls back to all projected centroids.
        """
        import numpy as np
        scene = self.scene_graph
        selected_cid = self.selected_cluster_id
        if selected_cid is None or selected_cid == -1:
            return {}

        projected = self._wasd_project_centroids()
        sel = next((p for p in projected if p[0] == selected_cid), None)
        if sel is None:
            return {}
        sx, sy = sel[1], sel[2]

        # Restrict candidates to the labeled cohorts (what's visible on screen).
        label_map = getattr(self, 'cohort_label_map', {}) or {}
        candidate_cids = set(label_map.keys()) if label_map else None

        # Screen-space direction cones (y grows downward on screen).
        cones = {
            'W': (0.0, -1.0),   # up
            'S': (0.0, 1.0),    # down
            'A': (-1.0, 0.0),   # left
            'D': (1.0, 0.0),    # right
        }
        targets = {}
        for key, (dx, dy) in cones.items():
            best_cid, best_score = None, float('inf')
            for cid, px, py in projected:
                if cid == selected_cid:
                    continue
                if candidate_cids is not None and cid not in candidate_cids:
                    continue
                vx, vy = px - sx, py - sy
                dist = float(np.hypot(vx, vy))
                if dist < 1e-3:
                    continue
                # Angle between the vector and the cone direction.
                cosang = (vx * dx + vy * dy) / dist
                if cosang < 0.7071:  # >45° off-axis -> not in this direction
                    continue
                score = dist / max(cosang, 1e-3)  # prefer aligned & near
                if score < best_score:
                    best_score, best_cid = score, cid
            if best_cid is not None:
                targets[key] = best_cid
        return targets

    def _wasd_clear_paths(self, fade=False):
        """Remove the drawn WASD preview paths/labels.

        Args:
            fade: when True (navigation ending), the W/S/A/D labels fade out over
                ~2 s while the lines are dropped immediately (GLLinePlotItem has no
                color-update API). When False, everything is removed at once.
        """
        items = getattr(self, '_wasd_items', []) or []
        if not items:
            return
        import pyqtgraph.opengl as gl
        fading_labels = []
        for item in items:
            try:
                if fade and isinstance(item, gl.GLTextItem):
                    c = item.color
                    base_rgba = (int(c.red()), int(c.green()), int(c.blue()), max(1, int(c.alpha())))
                    fading_labels.append([item, base_rgba])
                    continue  # leave in place; the fade timer removes it
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self._wasd_items = []
        if fading_labels:
            self._start_wasd_path_fade(fading_labels)

    def _start_wasd_path_fade(self, labels):
        """Begin fading out the given WASD path labels over ~2 s."""
        # Reset any in-flight fade so rapid changes don't stack stale entries.
        for entry in getattr(self, '_wasd_fade_labels', []):
            try:
                self.gl_view.removeItem(entry[0])
            except Exception:
                pass
        self._wasd_fade_labels = list(labels)
        if not self._wasd_fade_timer.isActive():
            self._wasd_fade_timer.start()

    def _update_wasd_path_fade(self):
        """Per-frame tick ramping fading WASD labels to transparent, then removing."""
        from PySide6.QtGui import QColor
        if not getattr(self, '_wasd_fade_labels', []):
            self._wasd_fade_timer.stop()
            return
        # ~2 s at 16 ms/frame -> decrement alpha by ~8 per tick.
        step = 8
        remaining = []
        for entry in self._wasd_fade_labels:
            item, base_rgba = entry
            a = max(0, int(base_rgba[3]) - step)
            try:
                if a <= 0:
                    self.gl_view.removeItem(item)
                else:
                    item.color = QColor(int(base_rgba[0]), int(base_rgba[1]),
                                        int(base_rgba[2]), a)
                    item.update()
                    remaining.append([item, (int(base_rgba[0]), int(base_rgba[1]),
                                             int(base_rgba[2]), a)])
            except Exception:
                continue
        self._wasd_fade_labels = remaining
        if not self._wasd_fade_labels:
            self._wasd_fade_timer.stop()

    def _wasd_draw_paths(self):
        """Draw (or redraw) the W/S/A/D preview paths from the selected centroid."""
        # Hidden via settings -> nothing to draw.
        if not getattr(self, 'wasd_paths_enabled', True):
            return
        import numpy as np
        scene = self.scene_graph
        spread = float(self.spread_spin.value())
        positions = scene.positions * spread

        import pyqtgraph.opengl as gl
        # Redraw: drop the current paths immediately (no fade) so they track the
        # camera/selection crisply; fading only happens when navigation ends.
        self._wasd_clear_paths(fade=False)
        targets = self._wasd_pick_targets()
        if not targets:
            return

        sel_idx = np.where(scene.cluster_ids == self.selected_cluster_id)[0]
        if len(sel_idx) == 0:
            return
        start = positions[sel_idx].mean(axis=0)

        from PySide6.QtGui import QColor, QFont
        for key, cid in targets.items():
            idx = np.where(scene.cluster_ids == cid)[0]
            if len(idx) == 0:
                continue
            end = positions[idx].mean(axis=0)
            try:
                # NOTE: this pyqtgraph version's GLLinePlotItem takes `color` (not
                # `pen`) and its setData() doesn't accept positional x,y,z — so the
                # two endpoints are passed straight to the constructor via `pos`.
                line = gl.GLLinePlotItem(
                    pos=np.array([[start[0], start[1], start[2]],
                                  [end[0], end[1], end[2]]]),
                    color=(80, 255, 140, 255), width=6, antialias=True)
                self.gl_view.addItem(line)
                self._wasd_items.append(line)

                mid = (start + end) / 2.0
                # GLTextItem takes its position via the constructor `pos` (no setPos
                # in this pyqtgraph version).
                label = gl.GLTextItem(
                    pos=np.array([mid[0], mid[1], mid[2]]),
                    text=key, color=QColor(80, 255, 140, 255),
                    font=QFont("Helvetica", 36))
                self.gl_view.addItem(label)
                self._wasd_items.append(label)
            except Exception as e:
                print(f"Error drawing WASD path {key}: {e}")

    def _wasd_end_fade(self):
        """Fade out the current WASD paths (navigation is ending).

        Called when the selection changes by non-WASD means (left-click, session
        load) or the scene is rebuilt — i.e. whenever the preview should no longer
        be shown. No-op if there are no active paths.
        """
        if getattr(self, '_wasd_items', []):
            self._wasd_mode = False
            self._wasd_clear_paths(fade=True)

    def _wasd_select_begin(self):
        """Mark that a WASD-initiated selection is in progress (suppresses the fade)."""
        self._wasd_selecting = True

    def _wasd_handle_key(self, key):
        """Handle a WASD/QE navigation press.

        W/S/A/D: move to the nearest cohort in that screen direction (first press
        with no selection picks the nearest cohort). Q/E: step back/forward through
        the travel history (E only works after going back with Q). Each move records
        into the persistent trail and refreshes the preview paths.
        """
        import numpy as np
        scene = getattr(self, 'scene_graph', None)
        if scene is None or not hasattr(self, 'node_list') or not self.node_list:
            return

        # Navigation is active for this press; mark the selection below as WASD-
        # initiated so show_cluster_info doesn't fade out the paths we're about to draw.
        self._wasd_mode = True
        self._wasd_selecting = True

        try:
            if key in ("Q", "E"):
                # History navigation (back / forward). No new trail point is added —
                # we just move the cursor through already-visited cohorts.
                moved = self._wasd_history_back() if key == "Q" else self._wasd_history_forward()
                if not moved:
                    return
            else:
                # W/S/A/D directional movement.
                # No selection yet -> pick the nearest cohort to enter navigation mode.
                if self.selected_cluster_id is None or self.selected_cluster_id == -1:
                    cids_arr = scene.cluster_ids
                    valid = cids_arr[cids_arr != -1]
                    if len(valid) == 0:
                        return
                    # Nearest cohort to the current camera center.
                    spread = float(self.spread_spin.value())
                    positions = scene.positions * spread
                    cur = self.gl_view.opts.get('center', np.array([0.0, 0.0, 0.0]))
                    if hasattr(cur, 'x'):
                        cur = np.array([float(cur.x()), float(cur.y()), float(cur.z())])
                    else:
                        cur = np.asarray(cur, dtype=float)
                    best_cid, best_d = None, float('inf')
                    for cid in np.unique(valid):
                        idx = np.where(cids_arr == cid)[0]
                        c = positions[idx].mean(axis=0)
                        d = float(np.linalg.norm(c - cur))
                        if d < best_d:
                            best_d, best_cid = d, int(cid)
                    member_idx = int(np.where(scene.cluster_ids == best_cid)[0][0])
                    self.show_cluster_info(member_idx)
                    # Record the starting point of the travel trail.
                    self._wasd_history_add(best_cid)

                # Move to the target in the pressed direction (if one exists).
                targets = self._wasd_pick_targets()
                target_cid = targets.get(key)
                if target_cid is not None:
                    member_idx = int(np.where(scene.cluster_ids == target_cid)[0][0])
                    self.show_cluster_info(member_idx)  # selects the new cohort
                    # Optionally recenter the camera on the newly selected cohort.
                    if getattr(self, 'auto_center_on_selection', False):
                        self._recenter_camera_on_cohort(target_cid)
                    # Record this visited cohort in the travel history/trail.
                    self._wasd_history_add(target_cid)

            # (Re)draw the preview paths for the new selection.
            self._wasd_draw_paths()
        finally:
            # Clear the transient flag so a later non-WASD selection change is
            # recognized as "navigation ended" and fades these paths out. _wasd_mode
            # stays True so arrow-key camera moves keep refreshing the paths.
            self._wasd_selecting = False

    def _on_camera_moved(self):
        """Refresh WASD preview paths when the camera moves (arrow-key orbit)."""
        if getattr(self, '_wasd_mode', False) and self.selected_cluster_id not in (None, -1):
            self._wasd_draw_paths()

    # ------------------------------------------------------------------
    # WASD travel trail (persistent route of visited cohorts)
    # ------------------------------------------------------------------

    def _wasd_centroid(self, cluster_id):
        """Return the (spread-scaled) centroid of a cohort as np.array(3), or None."""
        import numpy as np
        scene = getattr(self, 'scene_graph', None)
        if scene is None or cluster_id in (None, -1):
            return None
        idx = np.where(scene.cluster_ids == cluster_id)[0]
        if len(idx) == 0:
            return None
        spread = float(self.spread_spin.value())
        return scene.positions[idx].mean(axis=0) * spread

    def _wasd_history_add(self, cluster_id):
        """Record a new WASD move in the travel history.

        Truncates any "future" entries (so a fresh W/S/A/D after going back with Q
        discards the forward branch), appends the cohort, and advances to it. The
        persistent trail is redrawn from history[0..index].
        """
        centroid = self._wasd_centroid(cluster_id)
        if centroid is None:
            return
        # Drop any future entries beyond the current position (new move forks).
        del self._wasd_history[self._wasd_history_index + 1:]
        self._wasd_history.append((cluster_id, np.asarray(centroid, dtype=float)))
        self._wasd_history_index = len(self._wasd_history) - 1
        self._wasd_draw_trail()

    def _wasd_history_back(self):
        """Step back one in the WASD travel history (Q). No-op if at the start."""
        if self._wasd_history_index <= 0:
            return False
        self._wasd_history_index -= 1
        cid, centroid = self._wasd_history[self._wasd_history_index]
        self._wasd_goto(cid)
        self._wasd_draw_trail()
        return True

    def _wasd_history_forward(self):
        """Step forward one in the WASD travel history (E). No-op if at the end."""
        if self._wasd_history_index >= len(self._wasd_history) - 1:
            return False
        self._wasd_history_index += 1
        cid, centroid = self._wasd_history[self._wasd_history_index]
        self._wasd_goto(cid)
        self._wasd_draw_trail()
        return True

    def _wasd_goto(self, cluster_id):
        """Select a cohort (syncing media viewer/panels) without touching the camera.

        Used by Q/E history navigation so the user can step through visited cohorts;
        the caller controls whether to recenter the camera.
        """
        import numpy as np
        scene = getattr(self, 'scene_graph', None)
        if scene is None or cluster_id in (None, -1):
            return
        member_idx = np.where(scene.cluster_ids == cluster_id)[0]
        if len(member_idx) == 0:
            return
        self._wasd_selecting = True
        try:
            self.show_cluster_info(int(member_idx[0]))
        finally:
            self._wasd_selecting = False

    def _wasd_draw_trail(self):
        """(Re)draw the persistent WASD travel trail from history[0..index]."""
        import numpy as np
        import pyqtgraph.opengl as gl
        # Clear existing trail items.
        for item in getattr(self, '_wasd_trail_items', []):
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self._wasd_trail_items = []

        idx = getattr(self, '_wasd_history_index', -1)
        history = getattr(self, '_wasd_history', []) or []
        pts = [c for (_cid, c) in history[:idx + 1]] if idx >= 0 else []
        if len(pts) < 2:
            return
        # One translucent turquoise line through the visited centroids so far.
        try:
            arr = np.vstack([p.reshape(1, 3) for p in pts])
            trail = gl.GLLinePlotItem(pos=arr, color=(0, 255, 255, 90), width=4, antialias=True)
            self.gl_view.addItem(trail)
            self._wasd_trail_items.append(trail)
        except Exception as e:
            print(f"Error drawing WASD trail: {e}")

    def _reset_wasd_trail(self):
        """Clear the persistent WASD travel history + trail (button in Visualization Settings)."""
        for item in getattr(self, '_wasd_trail_items', []):
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self._wasd_trail_items = []
        self._wasd_history = []
        self._wasd_history_index = -1

    # ------------------------------------------------------------------
    # Explore path preview (planned camera route, color-coded by phase)
    # ------------------------------------------------------------------

    def _explore_draw_path_preview(self):
        """Draw a windowed Explore route preview: the last 2 + next 3 stops.

        Shows two things so you can see where it's going without cluttering the view
        with the entire tour:
          - Cohort path (orange): straight lines connecting the stop centroids.
          - Camera flight path (green arcs): elevated Bezier curves between each stop's
            orbit-entry camera position — the actual path the camera flies.
        Each stop gets a blue orbit ring + numbered label. No-op when disabled or no
        path is set. Called on start and whenever the current target changes.
        """
        import numpy as np
        import pyqtgraph.opengl as gl
        self._explore_clear_path_preview()
        if not getattr(self, 'explore_show_path', True):
            return
        path = getattr(self, '_explore_path', []) or []
        n = len(path)
        if n < 1:
            return

        cur = getattr(self, '_explore_path_index', 0) % max(n, 1)
        # Window: last 2 (before current), the current stop, and next 3.
        window = []
        for k in range(-2, 4):  # -2,-1,0,1,2,3 relative to current
            idx = (cur + k) % n
            if idx not in [w[0] for w in window]:
                window.append((idx, path[idx]))

        COHORT = (255, 170, 40, 230)   # orange — cohort-to-cohort route
        FLIGHT = (90, 255, 120, 230)   # green — camera flight path
        ORBIT = (80, 160, 255, 230)    # blue — orbit ring at each stop

        base = float(self.explore_orbit_radius_base)
        factor = float(self.explore_orbit_size_factor)
        elev_deg = float(self.explore_elevation)

        def _orbit_radius(size):
            return max(0.5, base + factor * np.sqrt(max(size, 1)))

        # Precompute each stop's centroid and orbit-entry camera position.
        stops = []
        for idx, (cid, size, centroid) in window:
            c = np.asarray(centroid, dtype=float).reshape(3)
            r = _orbit_radius(size)
            # Orbit-entry azimuth is only known for the current target; use a fixed
            # heading for preview stops so the flight path is deterministic.
            cam_end = self._explore_cam_pos(c, r, elev_deg, 0.0)
            stops.append((idx, cid, size, c, r, cam_end))

        from PySide6.QtGui import QColor, QFont
        for j, (idx, cid, size, c, r, cam_end) in enumerate(stops):
            try:
                # Orbit ring at this stop.
                elev = np.radians(elev_deg)
                ring_pts = []
                for a in range(0, 361, 24):
                    az = np.radians(a)
                    ring_pts.append([c[0] + r * np.cos(elev) * np.cos(az),
                                     c[1] + r * np.cos(elev) * np.sin(az),
                                     c[2] + r * np.sin(elev)])
                self._explore_path_items.append(self._add_line(ring_pts, ORBIT, 3))

                # Numbered stop label (visit order within the window). GLTextItem takes
                # its position via the constructor `pos` (no setPos in this pyqtgraph).
                lbl = gl.GLTextItem(pos=np.array([c[0], c[1], c[2] + r * 0.6]),
                                    text=str(idx + 1), color=QColor(*ORBIT[:3], 255),
                                    font=QFont("Helvetica", 24))
                self.gl_view.addItem(lbl)
                self._explore_path_items.append(lbl)

                # Connect to the previous stop in the window: cohort line (orange,
                # centroid-to-centroid) and camera flight arc (green, elevated).
                if j > 0:
                    prev_c = stops[j - 1][3]
                    self._explore_path_items.append(
                        self._add_line([prev_c, c], COHORT, 4))
                    # Flight arc: quadratic Bezier between the two orbit-entry camera
                    # positions with an elevated control point (mirrors the real flight).
                    prev_cam = stops[j - 1][5]
                    mid = (prev_cam + cam_end) / 2.0
                    lift = max(1.5, float(np.linalg.norm(cam_end - prev_cam)) * 0.35)
                    control = mid + np.array([0.0, 0.0, lift])
                    arc_pts = []
                    for s in range(0, 21):
                        u = s / 20.0
                        p = ((1 - u) ** 2) * prev_cam + (2 * (1 - u) * u) * control + (u ** 2) * cam_end
                        arc_pts.append(p)
                    self._explore_path_items.append(self._add_line(arc_pts, FLIGHT, 3))
            except Exception as e:
                print(f"Error drawing explore path preview stop {idx}: {e}")

    def _add_line(self, points, color, width):
        """Helper: build + add a GLLinePlotItem from a list of (x,y,z) points."""
        import numpy as np
        import pyqtgraph.opengl as gl
        line = gl.GLLinePlotItem(pos=np.array(points), color=color, width=width, antialias=True)
        self.gl_view.addItem(line)
        return line

    def _explore_clear_path_preview(self):
        """Remove any drawn Explore path-preview items."""
        for item in getattr(self, '_explore_path_items', []):
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self._explore_path_items = []

    def _recenter_camera_on_cohort(self, cluster_id):
        """Move the camera center to a cohort's centroid.

        Uses the smooth-center glide when that option is enabled, otherwise an
        instant recenter. No-op if the cohort can't be resolved.
        """
        import numpy as np
        scene = getattr(self, 'scene_graph', None)
        if scene is None or cluster_id in (None, -1):
            return
        idx = np.where(scene.cluster_ids == cluster_id)[0]
        if len(idx) == 0:
            return
        spread = float(self.spread_spin.value())
        centroid = scene.positions[idx].mean(axis=0) * spread
        if getattr(self, 'smooth_center_transition', False):
            self._start_center_animation(centroid)
        else:
            from PySide6.QtGui import QVector3D
            self.gl_view.opts['center'] = QVector3D(
                float(centroid[0]), float(centroid[1]), float(centroid[2]))
            self.gl_view.update()

    # Tag Importance Ranking (Chi-Square)

    def _compute_tag_importance(self):
        """Compute tag importance using chi-square statistic vs cluster labels.

        Shows top 10 tags ranked by how much they contribute to cluster separation.

        Vectorized: builds a single sparse (cluster x tag) occurrence matrix and
        evaluates the chi-square for every tag with numpy/scipy ops instead of a
        Python loop over files x tags. The per-file pass is O(n_files) dict
        lookups only, and tokenized tags are resolved to strings solely for the
        final top 10 (the old code resolved every occurrence). At scale this is
        ~50-100x faster than the original implementation.

        Math note: with E[c,t] = tag_totals[t]*rowtot[c]/num_files,
            chi2_t = sum_c (C-E)^2/E
                   = (num_files/tag_totals[t]) * sum_c C[c,t]^2/rowtot[c] - tag_totals[t]
        which is computed over the sparse nonzeros of C only.
        """
        if not hasattr(self, 'node_list') or not self.node_list:
            return

        try:
            import numpy as np
            from scipy.sparse import csr_matrix

            scene = self.scene_graph
            cluster_labels = scene.cluster_ids.astype(np.int64)
            num_files = len(cluster_labels)

            if num_files == 0:
                return

            # Dense row index per file for every distinct label. Includes noise
            # (-1), matching the original which iterated set(cluster_labels).
            unique_clusters, dense_labels = np.unique(cluster_labels, return_inverse=True)
            total_clusters = len(unique_clusters)
            if total_clusters < 2:
                self.tag_importance_text.setText("Need at least 2 clusters for importance ranking.")
                return

            tag_data = self.tag_data or {}
            tokenized = bool(self.tag_interner)

            # Collect every tag occurrence as (file_row, tag). Per-file loop only;
            # list.extend is C-speed, so this stays fast even at millions of tags.
            rows_list = []
            flat_tags = []
            for i, fid in enumerate(scene.file_ids):
                tags = tag_data.get(fid)
                if tags:
                    rows_list.extend([i] * len(tags))
                    flat_tags.extend(tags)

            if not flat_tags:
                self.tag_importance_text.setText("No tags with sufficient frequency for ranking.")
                return

            rows = np.asarray(rows_list, dtype=np.int32)

            if tokenized:
                cols = np.asarray(flat_tags, dtype=np.int32)
                n_tags = len(self.tag_interner.index_to_tag)
                tag_names_sorted = None
            else:
                # dict.fromkeys preserves first-seen order and dedupes at C speed;
                # far faster than np.unique on an object array of strings.
                tag_names_sorted = list(dict.fromkeys(flat_tags))
                name_to_col = {t: c for c, t in enumerate(tag_names_sorted)}
                cols = np.asarray([name_to_col[t] for t in flat_tags], dtype=np.int32)
                n_tags = len(tag_names_sorted)

            # C[c, t] = occurrences of tag t among files in cluster c.
            # csr_matrix sums duplicate (row, col) pairs, so no counting pass needed.
            occ_cluster = dense_labels[rows]
            C = csr_matrix(
                (np.ones(len(rows), dtype=np.float64), (occ_cluster, cols)),
                shape=(total_clusters, n_tags),
            )
            tag_totals = np.asarray(C.sum(axis=0)).ravel()

            # Skip very rare tags (matches original: total occurrences < 3)
            keep = tag_totals >= 3
            if not keep.any():
                self.tag_importance_text.setText("No tags with sufficient frequency for ranking.")
                return

            rowtot = np.bincount(dense_labels, minlength=total_clusters).astype(np.float64)
            coo = C.tocoo()
            contrib = (coo.data ** 2) / rowtot[coo.row]
            S = np.bincount(coo.col, weights=contrib, minlength=n_tags)

            scores = np.zeros(n_tags)
            scores[keep] = ((num_files * S[keep]) / tag_totals[keep] - tag_totals[keep]) / total_clusters

            # Top 10 among kept tags, score descending (stable for ties)
            keep_idx = np.where(keep)[0]
            order = np.argsort(-scores[keep_idx], kind="stable")
            top = keep_idx[order[:10]]

            tag_scores = []
            for t in top:
                if tokenized:
                    name = self.tag_interner.index_to_string(int(t))
                else:
                    name = str(tag_names_sorted[t])
                tag_scores.append((name, float(scores[t])))

            self.tag_importance_text.setHtml(render_importance_html(tag_scores, shown=10))

        except Exception as e:
            print(f"Error computing tag importance: {e}")
            import traceback
            traceback.print_exc()
            self.tag_importance_text.setText(f"Error computing importance: {e}")
