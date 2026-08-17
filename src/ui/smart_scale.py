"""Smart Scale: node-count-based automatic settings.

When enabled, after data loads the app resolves which *profile* applies to the
current file count and overwrites a set of "smart-scalable" settings (UMAP,
DBSCAN, and visualization parameters) with that profile's values. This lets the
user tune one set of parameters for small datasets and another for large ones
without manual switching.

A profile is a dict: ``{"endpoint": <int node count>, "settings": {key: value}}``.
Profiles are ordered by endpoint ascending; the highest endpoint whose value is
<= node_count wins (the topmost row also covers everything above it, and the
bottom row covers everything below its endpoint).

The settings keys map 1:1 to widget *raw* values (what the spin boxes display),
so applying a profile is just ``widget.setValue(...)``. See SMART_SCALE_SETTINGS
for the canonical list and their tooltips.
"""

# Canonical smart-scalable settings: key -> (label, tooltip)
SMART_SCALE_SETTINGS = [
    ("n_neighbors", "UMAP N Neighbors",
     "UMAP n_neighbors for this size range."),
    ("min_dist", "UMAP Min Dist (%)",
     "UMAP min_dist as a percent (0-100)."),
    ("n_epochs", "UMAP Epochs",
     "UMAP epochs (0 = auto)."),
    ("learning_rate", "UMAP Learning Rate",
     "UMAP initial learning rate."),
    ("eps", "DBSCAN EPS (%)",
     "DBSCAN eps as a percent of data spread."),
    ("min_samples", "DBSCAN Min Samples",
     "DBSCAN min_samples."),
    ("node_size", "Node Size (x10)",
     "Node size stored x10 (widget shows value*10)."),
    ("transparency", "Transparency",
     "Node transparency (0-1)."),
    ("spread", "Spread",
     "Position spread factor."),
]

# Keys in display order (used to build table columns / apply).
SMART_SCALE_KEYS = [k for k, _label, _tip in SMART_SCALE_SETTINGS]


def default_profiles():
    """Return a sensible starter profile list.

    Two rows: one tuned for small datasets (larger nodes, tighter UMAP/DBSCAN)
    and one for large datasets (smaller nodes, looser parameters). The user can
    edit or replace these freely.
    """
    return [
        {
            "endpoint": 1000,
            "settings": {
                "n_neighbors": 15, "min_dist": 10, "n_epochs": 0,
                "learning_rate": 1.0, "eps": 40, "min_samples": 6,
                "node_size": 3.0, "transparency": 0.9, "spread": 1.0,
            },
        },
        {
            "endpoint": 100000,
            "settings": {
                "n_neighbors": 20, "min_dist": 5, "n_epochs": 32,
                "learning_rate": 2.0, "eps": 60, "min_samples": 10,
                "node_size": 1.0, "transparency": 0.7, "spread": 1.0,
            },
        },
    ]


def resolve_profile(profiles, node_count):
    """Return the profile whose endpoint is the highest value <= node_count.

    Args:
        profiles: list of {"endpoint": int, "settings": dict}. May be unsorted.
        node_count: number of files/nodes in the current dataset.

    Returns:
        The matching profile dict, or None if there are no profiles. If
        node_count is below every endpoint, the lowest-endpoint profile applies.
    """
    if not profiles:
        return None
    valid = [p for p in profiles if isinstance(p, dict) and "endpoint" in p]
    if not valid:
        return None
    # Highest endpoint that is <= node_count wins; fall back to the overall
    # lowest endpoint when node_count is below all of them.
    candidates = [p for p in valid if int(p["endpoint"]) <= int(node_count)]
    if candidates:
        return max(candidates, key=lambda p: int(p["endpoint"]))
    return min(valid, key=lambda p: int(p["endpoint"]))


def apply_profile_to_tab(tab, profile):
    """Overwrite the tab's smart-scalable widgets with a profile's values.

    Only keys present in ``profile['settings']`` are applied, so a partial
    profile leaves other settings untouched. Missing/invalid widgets are skipped
    silently (the feature is best-effort and must never break a load).
    """
    if not isinstance(profile, dict):
        return
    settings = profile.get("settings") or {}
    for key in SMART_SCALE_KEYS:
        if key not in settings:
            continue
        widget = getattr(tab, _widget_attr_for(key), None)
        if widget is None or not hasattr(widget, "setValue"):
            continue
        try:
            value = settings[key]
            # learning_rate / node_size / transparency / spread are float widgets.
            if key in ("learning_rate", "node_size", "transparency", "spread"):
                widget.setValue(float(value))
            else:
                widget.setValue(int(round(float(value))))
        except (TypeError, ValueError):
            continue


def _widget_attr_for(key):
    """Map a smart-scale setting key to the tab's widget attribute name."""
    return {
        "n_neighbors": "n_neighbors_spin",
        "min_dist": "min_dist_spin",
        "n_epochs": "n_epochs_spin",
        "learning_rate": "learning_rate_spin",
        "eps": "eps_spin",
        "min_samples": "min_samples_spin",
        "node_size": "min_size_spin",
        "transparency": "transparency_spin",
        "spread": "spread_spin",
    }[key]


def read_current_values(tab):
    """Snapshot the tab's current smart-scalable widget values as a settings dict.

    Used to pre-fill a new profile row with the user's present settings so they
    only need to change what differs for that size range.
    """
    out = {}
    for key in SMART_SCALE_KEYS:
        widget = getattr(tab, _widget_attr_for(key), None)
        if widget is not None and hasattr(widget, "value"):
            try:
                out[key] = float(widget.value())
            except (TypeError, ValueError):
                continue
    return out
