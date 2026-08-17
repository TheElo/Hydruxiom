"""Stateless helper functions for the 3D tag map tab.

These are pure functions extracted from ``tag_map_3d_tab.py`` so they can be
reused and unit-tested without instantiating the (large) widget. They take no
``self`` and have no side effects beyond their return values.
"""

import fnmatch
import re


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
