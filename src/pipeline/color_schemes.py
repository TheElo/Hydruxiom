"""Color-scheme utilities: image -> palette extraction + name validation.

Pure Python (PIL + scikit-learn only, no Qt) so it can be unit-tested headless
and reused by the UI layer without pulling in GUI imports. See
docs/features/generate-color-scheme-from-image.md for the feature design.
"""

from io import BytesIO
from typing import List, Optional, Tuple

RGB = Tuple[int, int, int]


def extract_palette_from_image(image_bytes: bytes, n_colors: int = 19) -> List[RGB]:
    """Extract a palette of exactly ``n_colors`` distinct RGB colors from an image.

    Pipeline (see feature design doc):
      1. Decode with PIL, convert to RGB, downscale to a ~64px thumbnail so the
         pixel cloud is small and noise-averaged.
      2. KMeans over the pixels -> cluster centers are the palette candidates.
      3. Sort by hue (then lightness) for a smoother ramp.

    Args:
        image_bytes: Raw image data as returned by Hydrus (get_file/get_thumbnail).
        n_colors: Number of distinct colors to produce (2..64 in practice; clamped here).

    Returns:
        List of exactly ``n_colors`` (r, g, b) tuples with values 0-255.

    Raises:
        ValueError: if the image cannot be decoded or KMeans is unavailable.
    """
    n = max(1, min(int(n_colors), 64))

    from PIL import Image
    img = Image.open(BytesIO(image_bytes))
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Downscale for speed + noise reduction; keep at least a few pixels per side.
    img.thumbnail((64, 64), Image.Resampling.LANCZOS)

    import numpy as np
    arr = np.asarray(img, dtype=np.float32).reshape(-1, 3) / 255.0
    if len(arr) < n:
        # Tiny image (fewer pixels than requested colors): pad by repeating so
        # KMeans still has enough samples; the palette will be low-diversity but valid.
        repeats = max(1, -(-n // len(arr)))  # ceil division
        arr = np.tile(arr, (repeats, 1))

    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ValueError("scikit-learn is required for palette extraction") from e

    km = KMeans(n_clusters=n, n_init=3, random_state=0)
    km.fit(arr)
    centers = np.clip(km.cluster_centers_, 0.0, 1.0)

    # Sort by hue (then lightness) for a smoother ramp instead of arbitrary order.
    import colorsys
    def _key(c):
        r, g, b = float(c[0]), float(c[1]), float(c[2])
        h, l, s = colorsys.rgb_to_hls(r, g, b)  # note: HLS (h, lightness, saturation)
        return (round(h * 360.0), round(l * 255.0))

    centers = sorted(centers.tolist(), key=_key)
    palette = [(int(round(c[0] * 255)), int(round(c[1] * 255)), int(round(c[2] * 255))) for c in centers]
    return palette


def sanitize_scheme_name(name: str, existing: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """Validate + normalize a user-entered color-scheme name.

    Args:
        name: Raw user input.
        existing: Names already in use (case-insensitive duplicate check).

    Returns:
        (clean_name, error_message): exactly one is None on success/failure.
    """
    clean = (name or "").strip()
    if not clean:
        return None, "Name cannot be empty."
    # Collapse internal whitespace runs to a single space for tidy dropdown entries.
    clean = " ".join(clean.split())
    if len(clean) > 40:
        return None, "Name too long (max 40 characters)."
    if existing and any(e.lower() == clean.lower() for e in existing):
        return None, f"A color scheme named '{clean}' already exists."
    return clean, None
