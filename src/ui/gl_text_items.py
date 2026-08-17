"""Multi-line GLTextItem (camera-stable stacked labels).

Extracted from ``tag_map_3d_tab.py``. This is a module-level helper with no
dependency on the widget instance, so it can be imported and reused directly.
"""

# Multi-line GLTextItem (camera-stable stacked labels)
_MULTILINE_TEXT_ITEM_CLASS = None


def get_multiline_text_item_class():
    """Lazily create (and cache) a GLTextItem subclass that renders
    multi-line text stacked in SCREEN space.

    The base pyqtgraph GLTextItem renders its text via
    QPainter.drawText(QPointF, str), which treats the whole string as a
    SINGLE line -- embedded newlines are NOT rendered as line breaks (the
    lines end up concatenated, e.g. "Tag1Tag2Tag3"). To get a true stacked
    label that stays locked to the camera we override paint() to draw each
    line separately, offset in SCREEN space from the single projected world
    anchor. Because all lines share one world anchor and are offset in
    screen space, the stack does not drift as the camera moves (unlike
    offsetting each line in world space).
    """
    global _MULTILINE_TEXT_ITEM_CLASS
    if _MULTILINE_TEXT_ITEM_CLASS is not None:
        return _MULTILINE_TEXT_ITEM_CLASS

    import pyqtgraph.opengl as gl
    from PySide6.QtGui import QFontMetrics, QVector3D, QPainter
    from PySide6.QtCore import QPointF, Qt as _Qt

    class _MultiLineGLTextItem(gl.GLTextItem):
        """GLTextItem that stacks multi-line text in SCREEN space.

        Performance: the per-frame paint() path is kept minimal. Everything
        that depends only on (text, font, alignment) -- line splitting, font
        metrics, and per-line horizontal advances -- is computed ONCE and
        cached. Each frame we only project the world anchor to screen space
        and issue drawText calls at the cached offsets. This matters a lot
        when the view repaints continuously (e.g. camera wobble), because it
        removes QFontMetrics + horizontalAdvance from the hot path.
        """

        def _build_cache(self):
            """Precompute line layout metrics (called only when text/font change)."""
            fm = QFontMetrics(self.font)
            line_height = fm.lineSpacing()
            lines = self.text.split("\n")
            align = self.alignment
            n = len(lines)

            # Per-line horizontal offset (dx) and vertical offset (dy)
            layout = []
            for i, line in enumerate(lines):
                if not line:
                    continue
                dx = 0.0
                if align & _Qt.AlignmentFlag.AlignHCenter:
                    dx = fm.horizontalAdvance(line) / 2.0
                elif align & _Qt.AlignmentFlag.AlignRight:
                    dx = fm.horizontalAdvance(line)
                dy = i * line_height
                if align & _Qt.AlignmentFlag.AlignVCenter:
                    dy -= (n - 1) * line_height / 2.0
                elif align & _Qt.AlignmentFlag.AlignTop:
                    dy -= (n - 1) * line_height
                layout.append((line, dx, dy))

            self._label_cache = {
                "key": (self.text, self.font.pointSizeF(), int(self.alignment)),
                "layout": layout,
            }

        def paint(self):
            if len(self.text) < 1:
                return

            # Rebuild the metrics cache only when text/font/alignment changed
            cache = getattr(self, "_label_cache", None)
            key = (self.text, self.font.pointSizeF(), int(self.alignment))
            if cache is None or cache["key"] != key:
                self._build_cache()
                cache = self._label_cache
            layout = cache["layout"]
            if not layout:
                return

            self.setupGLState()
            project = self.compute_projection()
            anchor = project.map(QVector3D(*self.pos)).toPointF()

            # Off-screen culling: skip labels whose anchor is outside the
            # viewport (with a margin for text extent). Saves drawText calls
            # for cohorts that are behind/beside the camera.
            view = self.view()
            vw = view.width()
            vh = view.height()
            margin = 200.0
            if (anchor.x() < -margin or anchor.x() > vw + margin or
                    anchor.y() < -margin or anchor.y() > vh + margin):
                return

            painter = QPainter(view)
            painter.setPen(self.color)
            painter.setFont(self.font)
            painter.setRenderHints(
                QPainter.RenderHint.Antialiasing
                | QPainter.RenderHint.TextAntialiasing
            )

            for line, dx, dy in layout:
                painter.drawText(QPointF(anchor.x() - dx, anchor.y() + dy), line)
            painter.end()

    _MULTILINE_TEXT_ITEM_CLASS = _MultiLineGLTextItem
    return _MULTILINE_TEXT_ITEM_CLASS


# Backwards-compatible alias so existing call sites keep working.
_get_multiline_text_item_class = get_multiline_text_item_class
