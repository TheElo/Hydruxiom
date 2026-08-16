"""Hydruxiom application bootstrap.

Creates the QApplication (dark Fusion style, high-DPI) and shows the main
window. Called from main.py.
"""

import json
import os
import sys

from PySide6.QtWidgets import QApplication


# Project root = parent of src/ (icon + settings live here)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON_PATH = os.path.join(_PROJECT_ROOT, "icon", "hydruxiom.ico")


def _app_icon():
    """Load the app icon (best effort; None if missing)."""
    from PySide6.QtGui import QIcon
    if os.path.exists(ICON_PATH):
        return QIcon(ICON_PATH)
    return None


def _apply_ui_scale():
    """Apply the user-configured UI scale factor (settings JSON "ui_scale").

    Must run BEFORE QApplication is constructed: Qt reads QT_SCALE_FACTOR at
    startup and then uniformly scales fonts + all widgets. A value of 100
    (default) leaves OS display scaling untouched.
    """
    try:
        settings_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "3d_tag_map_settings.json",
        )
        if not os.path.exists(settings_file):
            return
        with open(settings_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        scale_pct = float(data.get("ui_scale", 100))
    except Exception:
        return
    if scale_pct > 0 and abs(scale_pct - 100.0) > 0.01:
        os.environ["QT_SCALE_FACTOR"] = f"{scale_pct / 100.0:.4f}"


def run():
    _apply_ui_scale()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Hydruxiom")

    # App icon (taskbar + window title bar on all top-level windows)
    icon = _app_icon()
    if icon is not None:
        app.setWindowIcon(icon)

    # Dark palette
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtCore import Qt
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(24, 24, 28))
    pal.setColor(QPalette.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.Base, QColor(18, 18, 22))
    pal.setColor(QPalette.AlternateBase, QColor(30, 30, 36))
    pal.setColor(QPalette.ToolTipBase, QColor(40, 40, 48))
    pal.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    pal.setColor(QPalette.Text, QColor(220, 220, 220))
    pal.setColor(QPalette.Button, QColor(35, 35, 42))
    pal.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    pal.setColor(QPalette.Highlight, QColor(64, 96, 192))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    from src.ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    return app.exec()
