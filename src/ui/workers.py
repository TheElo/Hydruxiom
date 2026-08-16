"""Generic background worker threads.

Extracted from ``tag_map_3d_tab.py`` (monolith split, step 2).

:class:`WorkerThread` is a thin QThread wrapper that runs any callable in the
background and reports progress / result / error via signals. Used by all
pipeline orchestration paths in the tab (load & compute, recompute, cut/pop,
re-cluster, DBSCAN optimize).
"""

from PySide6.QtCore import QThread, Signal


class WorkerThread(QThread):
    """Background worker thread for data processing."""
    progress = Signal(int, str)  # percentage, message
    finished = Signal(object)  # result
    error = Signal(str)  # error message

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
