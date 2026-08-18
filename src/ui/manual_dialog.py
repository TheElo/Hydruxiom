"""Manual (help) window for Hydruxiom.

A tabbed, read-only reference similar in shape to the settings dialog. The
first tab covers controls and interactions; further tabs can be added as the
manual grows. Content is static text/tables — no state is written back.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget,
    QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QScrollArea,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


def _section_label(text):
    """A bold section header label."""
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #ff9d9d;")
    return lbl


def _body_text():
    """A read-only, word-wrapped text block for prose sections."""
    te = QTextEdit()
    te.setReadOnly(True)
    te.setStyleSheet(
        "QTextEdit { background-color: rgb(30, 32, 38); color: rgb(221, 221, 221);"
        " border: 1px solid rgb(44, 49, 58); padding: 6px; font-size: 12px; }"
    )
    return te


def _shortcut_table(rows):
    """A read-only two-column (Key | Action) table."""
    tbl = QTableWidget(len(rows), 2)
    for r, (key, desc) in enumerate(rows):
        key_item = QTableWidgetItem(key)
        key_item.setFont(QFont("Consolas", 10, QFont.Bold))
        key_item.setFlags(Qt.ItemFlag.NoItemFlags)
        tbl.setItem(r, 0, key_item)
        desc_item = QTableWidgetItem(desc)
        desc_item.setFlags(Qt.ItemFlag.NoItemFlags)
        tbl.setItem(r, 1, desc_item)
    tbl.horizontalHeader().setVisible(False)
    tbl.verticalHeader().setVisible(False)
    tbl.setColumnWidth(0, 120)
    tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    return tbl


class ManualDialog(QDialog):
    """Tabbed manual window. First tab: controls and interactions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hydruxiom Manual")
        self.setMinimumSize(640, 520)

        outer = QVBoxLayout()
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        tabs = QTabWidget()
        tabs.addTab(self._build_controls_tab(), "Controls & Interactions")
        # Future tabs (e.g. "Pipeline", "Settings reference") go here.
        outer.addWidget(tabs)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

        self.setLayout(outer)
        self._apply_dark_theme(tabs)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_controls_tab(self):
        """Controls & interactions: mouse, camera, keyboard, and buttons."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        # --- Mouse in the 3D view ---
        lay.addWidget(_section_label("Mouse — 3D View"))
        mouse_te = _body_text()
        mouse_te.setHtml(
            "<b>Left-click</b> a node or cohort to select it and show its info "
            "(right sidebar, Stats tab).<br>"
            "<b>Ctrl + Left-click</b> selects the single node under the cursor.<br>"
            "<b>Right-click</b> moves the camera center to the node under the cursor. "
            "If <i>Smooth center transition</i> is on, the camera glides there; if "
            "<i>Right-click also selects cohort</i> is on, it also selects that cohort.<br>"
            "<b>Double-click empty space</b> clears the current selection.<br>"
            "<b>Drag</b> to orbit the camera; <b>scroll wheel</b> to zoom."
        )
        lay.addWidget(mouse_te)

        # --- Keyboard: navigation ---
        lay.addWidget(_section_label("Keyboard — Camera & Navigation"))
        nav_te = _body_text()
        nav_te.setHtml(
            "<b>Arrow keys</b> orbit the camera (speed set by <i>Orbit Speed</i>).<br>"
            "<b>W / S / A / D</b> fly to the nearest cohort in that screen direction "
            "(preview arrows shown when <i>Show WASD navigation paths</i> is on).<br>"
            "<b>Q / E</b> step back / forward through your visited-cohort history.<br>"
            "A translucent turquoise travel trail records every cohort you visit with WASD; "
            "clear it with the <i>Reset Travel Trail</i> button (Visuals tab)."
        )
        lay.addWidget(nav_te)

        # --- Keyboard: shortcuts table ---
        lay.addWidget(_section_label("Keyboard — Shortcuts"))
        shortcut_rows = [
            ("F1", "Toggle left sidebar"),
            ("F2", "Toggle right sidebar"),
            ("F3", "Open / close the Settings window"),
            ("F4", "Toggle media viewer (split window)"),
            ("F5", "Load and Compute (full pipeline)"),
            ("F6", "Recompute (UMAP/PCA only, keeps positions)"),
            ("F7", "Regroup (DBSCAN only, keeps positions)"),
            ("F11", "Toggle fullscreen"),
            ("F12", "4x supersample snapshot → screenshots/"),
            ("Ctrl+S", "Split the selected cohort into sub-groups"),
            ("Ctrl+E", "Cut out — keep only the selected cohort"),
            ("Ctrl+R", "Pop — remove the selected cohort, keep the rest"),
            ("Ctrl+T", "Send selection to a Hydrus tab"),
            ("Ctrl+X", "Clear session (frees memory)"),
        ]
        lay.addWidget(_shortcut_table(shortcut_rows))

        # --- Buttons ---
        lay.addWidget(_section_label("Buttons"))
        btn_te = _body_text()
        btn_te.setHtml(
            "<b>Left panel (top):</b><br>"
            "&nbsp;&nbsp;• <i>⚙ Settings</i> — open the settings window (F3).<br>"
            "&nbsp;&nbsp;• <i>\U0001F4D6 Manual</i> — open this manual.<br><br>"
            "<b>Left panel (bottom, always visible):</b><br>"
            "&nbsp;&nbsp;• <i>Load and Compute</i> — run the full pipeline (query → tags → "
            "UMAP/PCA → DBSCAN).<br>"
            "&nbsp;&nbsp;• <i>Recompute</i> / <i>Regroup</i> — re-run only the reduction or "
            "only the clustering on already-loaded data.<br>"
            "&nbsp;&nbsp;• <i>\U0001F680 Optimize</i> — auto-search DBSCAN eps/min_samples.<br>"
            "&nbsp;&nbsp;• <i>Deorphan</i> — assign noise nodes to their nearest cohort.<br>"
            "&nbsp;&nbsp;• <i>Split / Pop / Cut</i> — sub-group, remove, or isolate the "
            "selected cohort (see shortcuts).<br>"
            "&nbsp;&nbsp;• <i>Clear</i> — drop session data and free memory.<br><br>"
            "<b>Right sidebar (Stats tab):</b><br>"
            "&nbsp;&nbsp;• <i>\u25B6 Explore</i> — fly the camera through cohort centroids "
            "(helicopter tour).<br>"
            "&nbsp;&nbsp;• <i>Send to Tab</i> — send selected file(s) to a named Hydrus tab.<br><br>"
            "<b>Right sidebar tabs:</b> Stats (file/cohort info + actions), Visuals "
            "(node appearance, camera wobble, cohort labels), Algorithm (UMAP/PCA + DBSCAN "
            "parameters)."
        )
        lay.addWidget(btn_te)

        # --- Sidebars & scaling note ---
        lay.addWidget(_section_label("Sidebars & Scaling"))
        side_te = _body_text()
        side_te.setHtml(
            "Both sidebars scroll vertically when their content doesn't fit; the scrollbar "
            "only appears when needed. Use the edge toggle buttons (or F1 / F2) to hide a "
            "sidebar and give the 3D view more room.<br><br>"
            "If the UI is too large for your screen, open <b>Settings → UI → Scale</b> and "
            "set a value below 100% (e.g. 75%), then restart Hydruxiom."
        )
        lay.addWidget(side_te)

        return page

    # ------------------------------------------------------------------
    def _apply_dark_theme(self, tabs):
        """Match the app's dark palette for tables and tab chrome."""
        try:
            from src.ui.ui_utils import apply_dark_theme
            apply_dark_theme(self)
        except Exception:
            pass
        tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid rgb(44, 49, 58);"
            " background-color: rgb(24, 24, 28); }"
            "QTabBar::tab { background-color: rgb(30, 32, 38); color: rgb(221, 221, 221);"
            " padding: 6px; font-size: 13px; border: none; }"
            "QTabBar::tab:selected { background-color: rgb(40, 56, 120); }"
        )
        for i in range(tabs.count()):
            w = tabs.widget(i)
            tbls = w.findChildren(QTableWidget)
            for t in tbls:
                t.setStyleSheet(
                    "QTableWidget { background-color: rgb(33, 37, 43);"
                    " color: rgb(221, 221, 221); border: 1px solid rgb(44, 49, 58);"
                    " gridline-color: rgb(44, 49, 58); }"
                )
