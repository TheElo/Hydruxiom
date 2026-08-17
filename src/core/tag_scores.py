"""External tag-score store (pure Python — no Qt, no Hydrus).

Holds the optional per-tag score weights used by the vectorizer and scene
graph. Populated from a user-supplied SQLite DB ("score_db_path" in the 3D
settings file) via :func:`reload_external_tag_scores`. Empty by default -> no
score weighting is applied.

This module lives in ``core`` so that ``pipeline/`` and ``core`` stay importable
without Qt or Hydrus (headless benchmarks + unit tests). It was previously in
``utils/query_comperator.py``, which imported ``hydrus_api`` at module level and
broke the layering rule.
"""

import json
import os
import sqlite3


# Module-level store: tag -> float score. Empty = scoring disabled.
ExternalTagScores = {}


def _score_db_path_from_settings():
    """Read the optional score DB path from the 3D tag map settings file."""
    try:
        # Project root = parent of src/ (this file is in src/core/)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        settings_file = os.path.join(project_root, "3d_tag_map_settings.json")
        if os.path.exists(settings_file):
            with open(settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (data.get("score_db_path") or "").strip()
    except Exception:
        pass
    return ""


def reload_external_tag_scores(db_path=None):
    """(Re)load ExternalTagScores from a SQLite DB with an ExternalTagScores table.

    Args:
        db_path: Path to the SQLite file. If None, reads "score_db_path" from
            the 3D settings file. Empty/None path -> clears scores (disabled).

    Returns:
        int: number of scores loaded.
    """
    # Mutate in place (clear + update) rather than rebinding the global, so that
    # modules which did `from ... import ExternalTagScores` keep a valid reference.
    ExternalTagScores.clear()
    if db_path is None:
        db_path = _score_db_path_from_settings()
    if not db_path:
        return 0
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT tag, score FROM ExternalTagScores")
        for tag, score in cursor.fetchall():
            ExternalTagScores[tag] = score
        conn.close()
    except Exception as e:
        print(f"[tag_scores] Failed to load scores from {db_path}: {e}")
        ExternalTagScores.clear()
    return len(ExternalTagScores)
