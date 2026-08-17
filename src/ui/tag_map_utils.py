"""Stateless helper functions for the 3D tag map tab.

These are pure functions extracted from ``tag_map_3d_tab.py`` so they can be
reused and unit-tested without instantiating the (large) widget. They take no
``self`` and have no side effects beyond their return values.
"""

import fnmatch
import os
import re

# Settings file path (relative to project root). Shared by the 3D tag map tab
# and its mixins, so it lives here rather than in any single module.
SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "3d_tag_map_settings.json",
)


def compile_tag_patterns(tag_list):
    """Split a tag list into exact-match set and compiled wildcard patterns.

    Args:
        tag_list: List of tag strings (may contain wildcards like 'system:*')

    Returns:
        tuple: (exact_set, compiled_patterns)
            - exact_set: set of lowercase exact tag names (no wildcards)
            - compiled_patterns: list of compiled regex for wildcard patterns
    """
    exact = set()
    patterns = []
    for pattern in tag_list:
        if '*' in pattern or '?' in pattern or '[' in pattern:
            patterns.append(re.compile(fnmatch.translate(pattern.lower())))
        else:
            exact.add(pattern.lower())
    return exact, patterns


def ease_in_out(t):
    """Smooth ease-in-out function for smooth transitions.

    Args:
        t: Value between 0.0 and 1.0.

    Returns:
        Eased t value with smooth acceleration/deceleration.
    """
    if t <= 0.5:
        return 2.0 * t * t
    else:
        return 1.0 - 2.0 * (1.0 - t) ** 2


# ---------------------------------------------------------------------------
# HTML table rendering for the Stats panels (Cohort Tag Data, Tag Importance).
# Replaces the old monospace text dumps with a scannable layout: name column,
# right-aligned value columns, alternating row shading. The widgets stay
# QTextEdit so all existing setText/clear call sites keep working.
# ---------------------------------------------------------------------------

# NOTE: these colors are embedded in HTML (style="color:..." / bgcolor="...") that
# is rendered via QTextEdit.setHtml(). Qt's rich-text engine does NOT understand the
# CSS rgb(r,g,b) functional notation — it requires #RRGGBB hex. Using rgb() here
# spams "QTextHtmlParser::applyAttributes: Unknown color name" warnings and drops
# the colors. (Qt Style Sheets via setStyleSheet DO accept rgb(), so other files are
# unaffected.)
_TABLE_ROW_A = "#262A32"   # even rows (slightly lighter than the widget bg)
_TABLE_ROW_B = "#21252B"   # odd rows (widget bg)
_TABLE_HEAD_BG = "#2C313A"
_TABLE_TEXT = "#D2D6DC"
_TABLE_DIM = "#8C929C"


def _esc(text):
    """Escape a string for safe embedding in an HTML fragment."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def render_rank_table(rows, headers, value_formats=None, name_title_attr=None):
    """Render ranked rows as a compact styled HTML table.

    Args:
        rows: list of tuples — first element is the display name, remaining
            elements are values (one per header after the first).
        headers: list of column titles; headers[0] labels the name column,
            the rest label the value columns (rendered right-aligned).
        value_formats: optional list of callables applied to each value cell
            (same length as headers - 1); default is str().
        name_title_attr: optional callable(name) -> tooltip text for the name
            cell.

    Returns:
        HTML string suitable for QTextEdit.setHtml().
    """
    if not rows:
        return ""
    value_formats = value_formats or [str] * (len(headers) - 1)

    # NOTE: Qt's rich-text engine only honors ONE style attribute per element,
    # so all styling must be merged into a single style="..." string.
    def _cell_style(align_left):
        align = "" if align_left else "text-align:right; "
        return f"color:{_TABLE_TEXT}; font-size:12px; {align}padding:2px 6px;"

    parts = ['<table cellspacing="0" cellpadding="3" width="100%">']
    # Header row
    parts.append('<tr>')
    for idx, h in enumerate(headers):
        align = "" if idx == 0 else "text-align:right; "
        parts.append(
            f'<td bgcolor="{_TABLE_HEAD_BG}" '
            f'style="{align}color:{_TABLE_DIM}; font-size:10px; padding:2px 6px;">'
            f'{_esc(h)}</td>'
        )
    parts.append('</tr>')

    for i, row in enumerate(rows):
        name = row[0]
        bg = _TABLE_ROW_A if i % 2 == 0 else _TABLE_ROW_B
        title = ""
        if name_title_attr:
            tip = name_title_attr(name)
            if tip:
                title = f' title="{_esc(tip)}"'
        parts.append(f'<tr bgcolor="{bg}">')
        parts.append(
            f'<td style="{_cell_style(True)}"{title}>{_esc(name)}</td>'
        )
        for j, value in enumerate(row[1:]):
            text = value_formats[j](value)
            parts.append(f'<td style="{_cell_style(False)}">{_esc(text)}</td>')
        parts.append('</tr>')
    parts.append('</table>')
    return "".join(parts)


def render_cohort_tags_html(total_files, sorted_tags, shown=20):
    """HTML for the Cohort Tag Data panel: summary line + ranked tag table.

    Args:
        total_files: number of files in the selection (summary denominator).
        sorted_tags: list of (tag, count) tuples, already sorted descending.
        shown: how many rows to render (the rest are summarized).

    Returns:
        HTML string for QTextEdit.setHtml().
    """
    if total_files <= 0 or not sorted_tags:
        return ""
    top = sorted_tags[:shown]
    # Each row carries both value cells (count, percent) — one per column.
    rows = [(tag, count, count / total_files * 100) for tag, count in top]
    table = render_rank_table(
        rows,
        headers=["Tag", "Files", "%"],
        value_formats=[lambda c: f"{int(c):,}", lambda p: f"{p:.0f}%"],
    )
    html = [f'<p style="margin:2px 0; color:{_TABLE_TEXT}; font-size:12px;">'
            f'{total_files:,} files &middot; {len(sorted_tags):,} unique tags</p>']
    if len(sorted_tags) > shown:
        html.append(f'<p style="margin:2px 0; color:{_TABLE_DIM}; font-size:11px;">'
                    f'top {shown} of {len(sorted_tags):,}</p>')
    html.append(table)
    return "".join(html)


def render_info_rows(pairs):
    """HTML for the Selected File Info panel: bold label / value rows.

    Args:
        pairs: list of (label, value) tuples in display order.

    Returns:
        HTML string for QTextEdit.setHtml().
    """
    if not pairs:
        return ""
    parts = []
    for label, value in pairs:
        parts.append(
            f'<p style="margin:1px 0; font-size:12px;">'
            f'<span style="color:{_TABLE_DIM};">{_esc(label)}:</span> '
            f'<span style="color:{_TABLE_TEXT};">{_esc(value)}</span></p>'
        )
    return "".join(parts)


def render_importance_html(tag_scores, shown=10):
    """HTML for the Tag Importance panel: ranked score table.

    Args:
        tag_scores: list of (tag, score) tuples, already sorted descending.
        shown: how many rows to render.

    Returns:
        HTML string for QTextEdit.setHtml().
    """
    if not tag_scores:
        return ""
    top = tag_scores[:shown]
    table = render_rank_table(
        list(top),
        headers=["Tag", "Score"],
        value_formats=[lambda s: f"{s:.2f}"],
    )
    html = [table]
    if len(tag_scores) > shown:
        html.append(f'<p style="margin:2px 0; color:{_TABLE_DIM}; font-size:11px;">'
                    f'top {shown} of {len(tag_scores):,}</p>')
    return "".join(html)
