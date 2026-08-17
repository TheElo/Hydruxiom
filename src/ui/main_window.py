"""Standalone host window for the 3D Tag Space tab.

Hydruxiom is the 3D tag map extracted from HydrusForHydrus as its own app.
This window hosts the full-featured ``TagMap3DTab`` (all UI and function
logic intact) directly, with no surrounding tab widget or other features.
"""

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PySide6.QtGui import QCloseEvent


class MainWindow(QMainWindow):
    """Standalone window that hosts the 3D Tag Map tab."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hydruxiom - 3D Tag Space Explorer")
        self.resize(1600, 950)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        from src.ui.tag_map_3d_tab import TagMap3DTab
        self.tab = TagMap3DTab(self)
        layout.addWidget(self.tab)

    def closeEvent(self, event: QCloseEvent):
        """Save settings and close the split/media window on exit."""
        # Wait for a pending background session auto-save so the latest scene
        # isn't lost (the tab's own closeEvent may not fire when the main
        # window closes).
        try:
            if hasattr(self.tab, '_wait_pending_auto_save'):
                self.tab._wait_pending_auto_save()
        except Exception:
            pass
        try:
            if hasattr(self.tab, 'save_settings'):
                self.tab.save_settings()
        except Exception:
            pass
        try:
            if getattr(self.tab, 'split_window', None) is not None:
                self.tab.split_window.close()
                self.tab.split_window = None
        except Exception:
            pass
        super().closeEvent(event)
