"""Media Viewer: separate window syncing image previews to the 3D tag map.

Extracted from ``tag_map_3d_tab.py`` (monolith split, step 1). Contains:

- :class:`TagMap3DSplitWindow` - the media viewer window itself
- :class:`SplitWindowLoader`   - background thumbnail-grid loader
- :class:`SingleFileLoader`    - background full-res single-file loader

The window communicates back to its parent tab only through a handful of
callbacks (``_return_to_grid``, ``_open_file_in_viewer``,
``_move_camera_to_cluster``, geometry persistence helpers), so the tab stays
the orchestrator while this module owns all viewer UI + loading.
"""

import json
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QScrollArea, QGridLayout,
)
from PySide6.QtCore import Qt, QThread, Signal


# Settings file path (relative to project root; same file the tab uses)
SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "3d_tag_map_settings.json",
)


class TagMap3DSplitWindow(QWidget):
    """Media Viewer: separate window syncing image previews to the 3D tag map.

    Display modes based on selection state:
    - Single file selected: shows that file
    - Cohort selected: shows a grid of thumbnails for those files
    - Nothing selected: shows one representative image per existing cohort
    """

    def __init__(self, parent_tab):
        super().__init__()
        self.parent_tab = parent_tab
        self.setWindowTitle("Hydruxiom - Media Viewer")
        self.resize(900, 700)
        self.setMinimumSize(600, 400)

        # Layout: title bar + control bar + scrollable image grid
        outer = QVBoxLayout()
        outer.setContentsMargins(5, 5, 5, 5)

        self.title_label = QLabel("No Selection")
        self.title_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold; padding: 5px;")
        outer.addWidget(self.title_label)

        # Control bar for grid settings
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
        self.max_files_spin.setValue(28)
        self.max_files_spin.setToolTip("Maximum number of thumbnails to pull.")
        controls.addWidget(self.max_files_spin)

        controls.addWidget(QLabel("Image Size:"))
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(50, 800)
        self.image_size_spin.setValue(400)
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

        # Zoom state: file id currently shown full-res after a thumbnail click.
        # None = showing the grid; set = showing one full file (click to go back).
        self._zoomed_file_id = None

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
        # Reset zoom + single-file view
        self._zoomed_file_id = None
        self._single_file_pixmap = None
        self.single_file_label.clear()
        self.single_file_label.hide()
        self.scroll_area.show()

    def show_single_image(self, pixmap, tooltip="", file_id=None):
        """Show a single full-res image scaled to fill the window.

        If ``file_id`` is given (thumbnail zoom), clicking the image again goes
        back to the grid; otherwise (node selection) it just displays.
        """
        self._single_file_pixmap = pixmap
        self.single_file_label.setToolTip(tooltip)
        self._zoomed_file_id = file_id
        if file_id is not None:
            self.single_file_label.setCursor(Qt.PointingHandCursor)
        else:
            self.single_file_label.unsetCursor()
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
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
                self.columns_spin.setValue(settings.get("split_columns", 4))
                self.max_files_spin.setValue(settings.get("split_max_files", 28))
                self.image_size_spin.setValue(settings.get("split_image_size", 400))
                # Restore window position/size (best effort)
                if hasattr(self.parent_tab, '_restore_window_geometry'):
                    self.parent_tab._restore_window_geometry(
                        settings, "split_window_geometry", self
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def _save_settings(self):
        """Save split window settings to the 3D tag map settings file."""
        try:
            settings = {}
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
            settings["split_columns"] = self.columns_spin.value()
            settings["split_max_files"] = self.max_files_spin.value()
            settings["split_image_size"] = self.image_size_spin.value()
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=2)
        except (OSError, TypeError):
            pass

    def closeEvent(self, event):
        """Save settings (incl. window geometry) when the media viewer closes."""
        try:
            if hasattr(self.parent_tab, '_persist_split_window_geometry'):
                self.parent_tab._persist_split_window_geometry()
        except Exception:
            pass
        # Clear the tab's reference so F4 re-opens a fresh instance next time
        # (closing via X must behave like closing via the toggle).
        try:
            if getattr(self.parent_tab, 'split_window', None) is self:
                self.parent_tab.split_window = None
        except Exception:
            pass
        self._save_settings()
        super().closeEvent(event)

    def set_title(self, text):
        """Update the title bar text."""
        self.title_label.setText(text)

    def add_image(self, pixmap, tooltip="", file_id=None):
        """Add a single image label to the grid using configured columns/size.

        Thumbnails are clickable: clicking one opens that file full-res in this
        window (click again to return to the grid). ``file_id`` is stored as a
        widget property so mousePressEvent can look it up.
        """
        size = self.image_size_spin.value() if hasattr(self, 'image_size_spin') else 200
        label = QLabel()
        label.setPixmap(pixmap)
        label.setFixedSize(size, size)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("border: 1px solid #4050a0;")
        if tooltip:
            label.setToolTip(tooltip)
        if file_id is not None:
            label.setProperty("file_id", str(file_id))
            label.setCursor(Qt.PointingHandCursor)
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
        """Handle clicks in the media viewer.

        - Full-res image shown via thumbnail zoom: click to go back to the grid.
        - Thumbnail with a file_id: click to open that file full-res here.
        - Cohort representative tile: click to move the 3D camera to it.
        """
        if event.button() == Qt.LeftButton:
            # Zoomed-out state: clicking the big image returns to the grid.
            if self._zoomed_file_id is not None and self.single_file_label.isVisible():
                fid = self._zoomed_file_id
                self._zoomed_file_id = None
                self.parent_tab._return_to_grid(fid)
                event.accept()
                return

            widget = self.childAt(event.position().toPoint())
            if widget is not None:
                # Cohort representative tile -> move camera to that cohort.
                cluster_id = widget.property("cluster_id")
                if cluster_id is not None and cluster_id != -1:
                    self.parent_tab._move_camera_to_cluster(cluster_id)
                    event.accept()
                    return
                # Regular thumbnail with a file id -> open full-res here.
                fid = widget.property("file_id")
                if fid is not None:
                    self.parent_tab._open_file_in_viewer(fid)
                    event.accept()
                    return
        super().mousePressEvent(event)


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
