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
from src.ui.tag_map_utils import compile_tag_patterns, ease_in_out, SETTINGS_FILE
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
        self.auto_deorphan = "Never"  # "Never", "After Load and Compute", "After Regroup"
        self._pending_deorphan = False  # True while a deorphan worker is in flight
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


    def closeEvent(self, event: QCloseEvent):
        """Handle window close event to save settings."""
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

        # Auto-save session so "Auto load session" can restore it on next launch.
        self._auto_save_session()

        # Auto-Deorphan: optionally assign noise (-1) nodes to their nearest
        # cohort after the chosen operation. Skip when this completion IS a
        # deorphan (its flags are set by start_deorphan) to avoid an infinite loop.
        was_deorphan = getattr(self, '_pending_deorphan', False)
        if was_deorphan:
            self._pending_deorphan = False
        else:
            mode = getattr(self, 'auto_deorphan', "Never")
            should_deorphan = (
                (mode == "After Load and Compute" and not was_recluster) or
                (mode == "After Regroup" and was_recluster)
            )
            if should_deorphan:
                self.start_deorphan()

    def on_loading_error(self, error_msg):
        """Handle loading error."""
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
            tags_to_show = list(node_tags[:42])  # Limit to 42 tags for performance
            if self.tag_interner:
                tags_to_show = self.tag_interner.strings_to_list(tags_to_show)
            if len(node_tags) > 42:
                # Add a label to indicate more tags exist
                more_label = QLabel(f"... and {len(node_tags) - 42} more tags")
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
            tags_to_show = sorted_tags[:42]  # Limit to 42 tags
            if len(sorted_tags) > 42:
                more_label = QLabel(f"... and {len(sorted_tags) - 42} more tags")
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
