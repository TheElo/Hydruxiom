"""Tag query widgets and pure query-string parsing helpers.

Extracted from ``tag_map_3d_tab.py`` (monolith split, step 3).

- :class:`ClickableTag` - clickable tag label cycling through four visual states
- :func:`split_query_preserving_brackets` - comma-split that keeps [OR groups] intact
- :func:`query_to_api_tags` - query string -> API-ready tags (nested lists for OR)
- :func:`parse_query_tag_states` - query string -> (included, excluded, or) sets

The parsing functions are pure (no widget access) so they are unit-testable.
"""

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, Signal

from src.ui.styles import RED_A, BLUE_60


class ClickableTag(QLabel):
    """Clickable tag label that cycles through four visual states.

    States:
    - 0 (neutral): White text
    - 1 (included): Green text (added to query)
    - 2 (excluded): Red text with "-" prefix (excluded from query)
    - 3 (OR): Bright blue text (added to OR bracket group)
    """

    stateChanged = Signal(str, int)  # tag_name, new_state

    def __init__(self, tag_name, parent=None):
        if tag_name is None:
            tag_name = ""
        super().__init__(tag_name, parent)
        self.tag_name = tag_name
        self.state = 0  # 0=neutral, 1=included, 2=excluded, 3=OR
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(f"""
            ClickableTag {{
                background-color: transparent;
                color: {RED_A};
                padding: 2px 6px;
                margin: 1px;
                border-radius: 3px;
                font-size: 11px;
            }}
            ClickableTag:hover {{
                background-color: {BLUE_60};
            }}
        """)
        self.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, event):
        """Handle click to cycle through states."""
        if event.button() == Qt.LeftButton:
            self.state = (self.state + 1) % 4
            self._update_appearance()
            self.stateChanged.emit(self.tag_name, self.state)
            event.accept()
        super().mousePressEvent(event)

    def _update_appearance(self):
        """Update label text and color based on current state."""
        if self.state == 0:
            # Neutral - white text
            self.setText(self.tag_name)
            self.setStyleSheet(f"""
                ClickableTag {{
                    background-color: transparent;
                    color: {RED_A};
                    padding: 2px 6px;
                    margin: 1px;
                    border-radius: 3px;
                    font-size: 11px;
                }}
                ClickableTag:hover {{
                    background-color: {BLUE_60};
                }}
            """)
        elif self.state == 1:
            # Included - green text
            self.setText(self.tag_name)
            self.setStyleSheet(f"""
                ClickableTag {{
                    background-color: transparent;
                    color: #44ff44;
                    padding: 2px 6px;
                    margin: 1px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }}
                ClickableTag:hover {{
                    background-color: {BLUE_60};
                }}
            """)
        elif self.state == 2:
            # Excluded - red text with "-" prefix and strikethrough
            self.setText(f"-{self.tag_name}")
            self.setStyleSheet(f"""
                ClickableTag {{
                    background-color: transparent;
                    color: #ff4444;
                    padding: 2px 6px;
                    margin: 1px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                    text-decoration: line-through;
                }}
                ClickableTag:hover {{
                    background-color: {BLUE_60};
                }}
            """)
        else:
            # OR - bright blue text
            self.setText(self.tag_name)
            self.setStyleSheet(f"""
                ClickableTag {{
                    background-color: transparent;
                    color: #44aaff;
                    padding: 2px 6px;
                    margin: 1px;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }}
                ClickableTag:hover {{
                    background-color: {BLUE_60};
                }}
            """)


def split_query_preserving_brackets(query):
    """Split a query string by commas, keeping bracket groups intact.

    Returns:
        list: Parts split by top-level commas (bracket groups stay whole strings).
    """
    parts = []
    depth = 0
    current = []
    for ch in query:
        if ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return parts


def query_to_api_tags(query):
    """Convert a query string to API-ready tags list.

    Bracket groups (OR segments) are converted to nested lists so the
    Hydrus API interprets them as OR groups.
    """
    parts = split_query_preserving_brackets(query)
    api_tags = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith('[') and part.endswith(']'):
            inner = part[1:-1].strip()
            tags = [t.strip() for t in inner.split(',') if t.strip()]
            api_tags.append(tags)
        else:
            api_tags.append(part)
    return api_tags


def parse_query_tag_states(query):
    """Parse a query string into included/excluded/OR tag sets.

    Args:
        query: The raw query text (e.g. from the tab's query_edit).

    Returns:
        tuple: (included_tags, excluded_tags, or_tags) sets parsed from the query.
    """
    query = (query or "").strip()
    included = set()
    excluded = set()
    or_tags = set()
    if not query:
        return included, excluded, or_tags
    for part in split_query_preserving_brackets(query):
        part = part.strip()
        if not part:
            continue
        # Handle OR bracket group
        if part.startswith('[') and part.endswith(']'):
            inner = part[1:-1].strip()
            for tag in inner.split(','):
                tag = tag.strip()
                if not tag:
                    continue
                if tag.startswith('-'):
                    excluded.add(tag[1:].strip())
                else:
                    or_tags.add(tag.strip())
            continue
        if part.startswith('-'):
            excluded.add(part[1:].strip())
        else:
            included.add(part.strip())
    return included, excluded, or_tags
