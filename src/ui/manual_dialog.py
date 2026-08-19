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
        tabs.addTab(self._build_performance_tab(), "Performance")
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

    def _build_performance_tab(self):
        """Performance: what is slow and which setting fixes it."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        lay.addWidget(_section_label("What Is Slow"))
        intro_te = _body_text()
        intro_te.setHtml(
            "The <b>UMAP</b> step dominates load time. Two things make it expensive: the "
            "<i>number of files</i> and, less obviously, the <i>tag vocabulary size</i> — UMAP "
            "computes distances over all tag dimensions internally (it densifies sparse input), so a "
            "50k-tag collection is ~10x more work per distance than a 5k-tag one even at the same file count.<br><br>"
            "Before every Load & Compute / Recompute, Hydruxiom estimates peak RAM and prints it to the "
            "console (<b>[RAM check]</b>). If the estimate exceeds what is safely available you get a status-bar "
            "warning — execution continues either way (it never blocks)."
        )
        lay.addWidget(intro_te)

        rows = [
            ("Pre-SVD", "ON/OFF + components spinbox (Algorithm group)",
             "Collapses tag dims to ~64 before UMAP. Makes the non-subsampled path dramatically faster and far lighter on RAM; layout quality is very close for TF-IDF-like data. Default OFF — enable it, compare your map, keep whichever you prefer."),
            ("Subsample", "ON/OFF + subset size (Algorithm group)",
             "Fit UMAP on a random subset, then project all points into that fixed space. Caps memory at any scale — but is SLOWER than plain UMAP when the full fit would have fit in RAM (projection cost grows with files × subset). Use it for very large loads or when the [RAM check] warns."),
            ("Chunked Transform", "ON/OFF (Algorithm group)",
             "Only relevant with Subsample ON: projects points in bounded chunks (~1.5 GB peak) instead of one giant call that would densify everything at once. Keep it on; uncheck only to compare against the legacy path."),
            ("CPU Cores / Low RAM", "Settings → Performance",
             "Cores parallelize UMAP's neighbor search (near-linear speedup). Low-RAM mode trades speed for lower peak memory — enable if you hit out-of-memory errors."),
            ("Min Tag Frequency", "Filter Settings (left panel)",
             "Drops rare tags from the vocabulary. Unit 'n' = absolute number of files a tag must appear in (0 disables); unit '%' = percent of the loaded collection — each unit remembers its own value. Fewer dimensions = faster UMAP and less noise; raise it on huge collections before reaching for the other knobs."),
            ("API / Direct-DB Load Threads", "Settings → Performance",
             "Concurrent connections used while loading file tags (defaults 4 API / 2 direct-DB, from benchmarks/benchmark_api_io.py: ~1.8x and ~1.7x faster than sequential). Set to 1 for the legacy single-request behavior."),
            ("API / Direct-DB Chunk Size", "Settings → Clients",
             "Files per request (API) or per query (direct-DB). Benchmarked optima: API ~8192 (network-bound, fewer bigger requests win); direct-DB flat/fast from ~512 up (default 4096). Local clients should use Direct DB mode (~30–55x faster than the API at scale)."),
        ]
        tbl = QTableWidget(len(rows), 3)
        headers = ["Setting", "Where", "What it does"]
        tbl.setHorizontalHeaderLabels(headers)
        for r, (a, b, c) in enumerate(rows):
            for col, text in enumerate((a, b, c)):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setFont(QFont("Consolas", 10, QFont.Bold))
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                tbl.setItem(r, col, item)
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        lay.addWidget(tbl)

        gpu_te = _body_text()
        gpu_te.setHtml(
            "<b>GPU UMAP:</b> the <i>GPU UMAP</i> algorithm option requires RAPIDS cuVS, which is "
            "Linux-only — on Windows it silently falls back to CPU UMAP. See "
            "<i>docs/gpu_umap_study.md</i> for options (WSL2 service / PyTorch-UMAP)."
        )
        lay.addWidget(gpu_te)

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
