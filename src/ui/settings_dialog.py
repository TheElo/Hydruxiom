"""Settings dialog for the 3D Tag Space Visualization tab.

Holds the "advanced" settings that were moved out of the main control panel:
- Low RAM toggle (UMAP low_memory)
- CPU core count (UMAP n_jobs)
- Direct DB mode toggle + per-client DB path fields

The dialog reads/writes the owning tab's plain-value attributes directly
(self.low_memory, self.n_jobs, self.use_direct_db, self.client_db_paths)
and calls tab.save_settings() on OK so the JSON settings file stays in sync.
"""

import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QSpinBox, QLineEdit, QGroupBox, QFormLayout,
)
from PySide6.QtCore import Qt


class TagMap3DSettingsDialog(QDialog):
    """Settings window for the 3D tag map tab."""

    def __init__(self, tab, parent=None):
        super().__init__(parent)
        self.tab = tab
        self.setWindowTitle("3D Tag Map Settings")
        self.setMinimumSize(520, 420)

        # Load current values from the tab
        self.low_memory = getattr(self.tab, 'low_memory', False)
        self.n_jobs = getattr(self.tab, 'n_jobs', os.cpu_count() or 4)
        self.use_direct_db = getattr(self.tab, 'use_direct_db', False)
        self.client_db_paths = dict(getattr(self.tab, 'client_db_paths', {}))

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # --- Performance group ---
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout()

        self.low_memory_checkbox = QCheckBox("Low RAM (UMAP low-memory)")
        self.low_memory_checkbox.setChecked(self.low_memory)
        self.low_memory_checkbox.setToolTip(
            "Use the UMAP low-memory algorithm.\nReduces peak memory usage but may be slower.\nEnable if you run out of memory."
        )
        perf_layout.addRow("Low RAM:", self.low_memory_checkbox)

        self.n_jobs_spin = QSpinBox()
        self.n_jobs_spin.setRange(1, 64)
        self.n_jobs_spin.setValue(self.n_jobs)
        self.n_jobs_spin.setToolTip(
            "Number of CPU cores to use for UMAP parallel NN-descent.\nHigher = faster but uses more CPU.\nDefault: all cores."
        )
        perf_layout.addRow("CPU Cores:", self.n_jobs_spin)

        perf_group.setLayout(perf_layout)
        main_layout.addWidget(perf_group)

        # --- DBSCAN Optimizer group ---
        opt_group = QGroupBox("DBSCAN Optimizer")
        opt_layout = QFormLayout()

        self.normalize_checkbox = QCheckBox("Normalize positions before DBSCAN")
        self.normalize_checkbox.setChecked(getattr(self.tab, 'normalize_positions', False))
        self.normalize_checkbox.setToolTip(
            "When enabled, positions are normalized (centered + std-scaled) before\n"
            "DBSCAN clustering. This makes eps a relative measure of local density\n"
            "rather than an absolute coordinate distance, so parameters behave\n"
            "consistently across datasets with different file counts / reducer scales.\n"
            "Applies to global clustering, re-cluster, and sub-cohort splitting."
        )
        opt_layout.addRow(self.normalize_checkbox)

        self.opt_max_cohort_size_spin = QSpinBox()
        self.opt_max_cohort_size_spin.setRange(2, 100000)
        self.opt_max_cohort_size_spin.setValue(getattr(self.tab, 'opt_max_cohort_size', 500))
        self.opt_max_cohort_size_spin.setToolTip(
            "Target maximum cohort size. Cohorts larger than this are considered\n"
            "disproportionately large and should be split into smaller sub-cohorts."
        )
        opt_layout.addRow("Max Cohort Size:", self.opt_max_cohort_size_spin)

        self.opt_max_noise_ratio_spin = QSpinBox()
        self.opt_max_noise_ratio_spin.setRange(0, 100)
        self.opt_max_noise_ratio_spin.setValue(getattr(self.tab, 'opt_max_noise_ratio', 10))
        self.opt_max_noise_ratio_spin.setSuffix("%")
        self.opt_max_noise_ratio_spin.setToolTip(
            "Target maximum noise ratio (non-cohorted nodes as % of total).\n"
            "Lower = fewer unclustered nodes. The optimizer tries to stay at or below this."
        )
        opt_layout.addRow("Max Noise Ratio:", self.opt_max_noise_ratio_spin)

        self.opt_max_attempts_spin = QSpinBox()
        self.opt_max_attempts_spin.setRange(1, 500)
        self.opt_max_attempts_spin.setValue(getattr(self.tab, 'opt_max_attempts', 60))
        self.opt_max_attempts_spin.setToolTip(
            "Maximum number of DBSCAN runs the optimizer may attempt\n"
            "to find the ideal eps/min_samples combination."
        )
        opt_layout.addRow("Max Attempts:", self.opt_max_attempts_spin)

        self.opt_eps_min_spin = QSpinBox()
        self.opt_eps_min_spin.setRange(1, 200)
        self.opt_eps_min_spin.setValue(getattr(self.tab, 'opt_eps_min', 5))
        self.opt_eps_min_spin.setToolTip("Lower bound of the EPS (%) search range.")
        opt_layout.addRow("EPS Min (%):", self.opt_eps_min_spin)

        self.opt_eps_max_spin = QSpinBox()
        self.opt_eps_max_spin.setRange(1, 200)
        self.opt_eps_max_spin.setValue(getattr(self.tab, 'opt_eps_max', 100))
        self.opt_eps_max_spin.setToolTip("Upper bound of the EPS (%) search range.")
        opt_layout.addRow("EPS Max (%):", self.opt_eps_max_spin)

        self.opt_min_samples_min_spin = QSpinBox()
        self.opt_min_samples_min_spin.setRange(2, 100)
        self.opt_min_samples_min_spin.setValue(getattr(self.tab, 'opt_min_samples_min', 2))
        self.opt_min_samples_min_spin.setToolTip("Lower bound of the Min Samples search range.")
        opt_layout.addRow("Min Samples Min:", self.opt_min_samples_min_spin)

        self.opt_min_samples_max_spin = QSpinBox()
        self.opt_min_samples_max_spin.setRange(2, 100)
        self.opt_min_samples_max_spin.setValue(getattr(self.tab, 'opt_min_samples_max', 30))
        self.opt_min_samples_max_spin.setToolTip("Upper bound of the Min Samples search range.")
        opt_layout.addRow("Min Samples Max:", self.opt_min_samples_max_spin)

        opt_group.setLayout(opt_layout)
        main_layout.addWidget(opt_group)

        # --- Direct DB group ---
        db_group = QGroupBox("Direct DB")
        db_layout = QFormLayout()

        self.direct_db_checkbox = QCheckBox("Use Direct DB (tag loading)")
        self.direct_db_checkbox.setChecked(self.use_direct_db)
        self.direct_db_checkbox.setToolTip(
            "When enabled, load tags directly from the Hydrus client DB\ninstead of the API. Much faster at scale (~99%).\nRequires a valid client DB path. Falls back to API if no path set."
        )
        db_layout.addRow(self.direct_db_checkbox)

        # Per-client DB path fields (populated from clients.json)
        from src.data.clients import client_ids
        self.client_db_path_edits = {}
        for client_id in (client_ids() or []):
            edit = QLineEdit()
            edit.setPlaceholderText(f"DB dir for {client_id} (e.g. X:\\HYDRUS\\CLIENT {client_id}\\CLIENT\\db)")
            edit.setToolTip(f"Path to the Hydrus client DB directory for {client_id}.\nUsed only when Direct DB mode is enabled.")
            edit.setText(self.client_db_paths.get(client_id, ""))
            db_layout.addRow(f"{client_id} DB Path:", edit)
            self.client_db_path_edits[client_id] = edit

        db_group.setLayout(db_layout)
        main_layout.addWidget(db_group)

        # --- Tag Score Weighting group (optional) ---
        score_group = QGroupBox("Tag Score Weighting (optional)")
        score_layout = QFormLayout()
        self.score_db_edit = QLineEdit()
        self.score_db_edit.setPlaceholderText("Path to SQLite with ExternalTagScores table (empty = off)")
        self.score_db_edit.setToolTip(
            "Optional: a SQLite file containing an ExternalTagScores (tag, score) table.\n"
            "When set, each tag's TF-IDF weight is multiplied by (1.0 + 0.1 * score),\n"
            "so high-scored tags pull the embedding more strongly.\n"
            "Leave empty to disable score weighting."
        )
        self.score_db_edit.setText(getattr(self.tab, 'score_db_path', '') or '')
        score_layout.addRow("Score DB Path:", self.score_db_edit)
        score_group.setLayout(score_layout)
        main_layout.addWidget(score_group)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.clicked.connect(self.apply_settings)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.apply_dark_theme()

    def apply_settings(self):
        """Apply dialog values back to the tab and save settings."""
        tab = self.tab

        tab.low_memory = self.low_memory_checkbox.isChecked()
        tab.n_jobs = self.n_jobs_spin.value()
        tab.use_direct_db = self.direct_db_checkbox.isChecked()

        # DBSCAN optimizer parameters
        tab.normalize_positions = self.normalize_checkbox.isChecked()
        # Sync the tab's inline normalize checkbox (if present)
        if hasattr(tab, 'normalize_checkbox'):
            tab.normalize_checkbox.setChecked(self.normalize_checkbox.isChecked())
        tab.opt_max_cohort_size = self.opt_max_cohort_size_spin.value()
        tab.opt_max_noise_ratio = self.opt_max_noise_ratio_spin.value()
        tab.opt_max_attempts = self.opt_max_attempts_spin.value()
        tab.opt_eps_min = self.opt_eps_min_spin.value()
        tab.opt_eps_max = self.opt_eps_max_spin.value()
        tab.opt_min_samples_min = self.opt_min_samples_min_spin.value()
        tab.opt_min_samples_max = self.opt_min_samples_max_spin.value()

        # Collect per-client DB paths
        paths = {}
        for client_id, edit in self.client_db_path_edits.items():
            path = edit.text().strip()
            if path:
                paths[client_id] = path
        tab.client_db_paths = paths

        # Persist DB paths to clients.json
        try:
            from src.data.direct_db import set_client_db_path
            for client_id, path in paths.items():
                set_client_db_path(client_id, path)
        except Exception as e:
            print(f"Error saving client DB paths: {e}")

        # Optional tag-score weighting
        tab.score_db_path = self.score_db_edit.text().strip()
        try:
            from src.utils.query_comperator import reload_external_tag_scores
            reload_external_tag_scores(tab.score_db_path or None)
        except Exception as e:
            print(f"Error reloading tag scores: {e}")

        # Save the rest of the settings JSON
        if hasattr(tab, 'save_settings'):
            tab.save_settings()

    def apply_dark_theme(self):
        """Apply a dark theme matching the rest of the app."""
        try:
            from src.ui.ui_utils import apply_dark_theme
            apply_dark_theme(self)
        except Exception:
            pass

        style = (
            "QSpinBox { background-color: rgb(33, 37, 43); color: rgb(255, 255, 255);"
            " border-radius: 5px; padding: 3px; }"
            "QSpinBox::up-button { subcontrol-origin: padding; subcontrol-position: top right;"
            " width: 16px; height: 16px; }"
            "QSpinBox::down-button { subcontrol-origin: padding; subcontrol-position: bottom right;"
            " width: 16px; height: 16px; }"
        )
        self.n_jobs_spin.setStyleSheet(style)
        # Style the DBSCAN optimizer spinboxes
        for spin in [
            self.opt_max_cohort_size_spin,
            self.opt_max_noise_ratio_spin,
            self.opt_max_attempts_spin,
            self.opt_eps_min_spin,
            self.opt_eps_max_spin,
            self.opt_min_samples_min_spin,
            self.opt_min_samples_max_spin,
        ]:
            spin.setStyleSheet(style)

        line_style = (
            "QLineEdit { background-color: rgb(33, 37, 43); border-radius: 5px;"
            " border: 2px solid rgb(33, 37, 43); padding: 8px; }"
            "QLineEdit:hover { border: 2px solid rgb(64, 71, 88); }"
            "QLineEdit:focus { border: 2px solid rgb(91, 101, 124); }"
        )
        for edit in self.client_db_path_edits.values():
            edit.setStyleSheet(line_style)
