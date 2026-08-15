"""Hydruxiom application bootstrap.

Creates the QApplication (dark Fusion style, high-DPI) and shows the main
window. Called from main.py.
"""

import sys

from PySide6.QtWidgets import QApplication


def run():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Hydruxiom")

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
