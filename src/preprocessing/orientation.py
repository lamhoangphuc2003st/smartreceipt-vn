"""
src/preprocessing/orientation.py
----------------------------------
Detect and fix 90 / 180 / 270-degree rotation in receipt photos.

Two detection strategies (tried in order):
  1. EXIF Orientation tag  — fast, exact; only available on ~6% of our dataset
  2. Hough line analysis   — works without EXIF; detects dominant line angles

Fine-angle deskewing (< 15°) is handled separately by deskew.py — this module
only corrects cardinal rotations (multiples of 90°).

Design principles:
  - Safety-first: only rotate when confidence is high (≥ 60% lines agree
    AND ≥ 10 lines detected).  Otherwise return image unchanged.
  - Never raises to caller.
  - EXIF orientation tag is stripped after correction to prevent double-apply.
"""

from __future__ import annotations

import logging
from collections import Counter

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── EXIF Orientation map ──────────────────────────────────────────────────────
# PIL tag 274 → degrees of counter-clockwise rotation needed to correct
# Values 2/4/5/7 involve mirroring (rare in phone photos) — handled separately.
_EXIF_ROTATION = {
    1: 0,    # Normal
    3: 180,  # Upside down
    6: 270,  # Rotated 90° CW (phone held portrait, phone reports landscape)
    8: 90,   # Rotated 90° CCW
}
_EXIF_MIRROR_THEN_ROTATE = {
    2: 0,    # Mirror horizontal
    4: 180,  # Mirror vertical
    5: 90,   # Mirror horizontal then rotate 90° CW
    7: 270,  # Mirror horizontal then rotate 90° CCW
}


# ── Low-level rotation ────────────────────────────────────────────────────────

def _apply_rotation(image: np.ndarray, degrees: int) -> np.ndarray:
    """
    Rotate image by exactly 0 / 90 / 180 / 270 degrees.

    Uses transpose + flip for lossless integer rotation (no interpolation,
    no quality loss).  0° is a no-op (returns original reference).
    """
    degrees = degrees % 360
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # Fallback for non-cardinal angles (shouldn't happen in normal use)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))


# ── EXIF helpers ──────────────────────────────────────────────────────────────

def _read_exif_orientation(exif_data: dict | None) -> int | None:
    """
    Extract PIL EXIF Orientation value (tag 274) from exif_data dict.

    Returns integer 1–8 or None if not present.
    exif_data is expected to be a dict from PIL's _getexif() or getexif().
    """
    if not exif_data:
        return None
    # PIL._getexif() returns {tag_id: value}; PIL.getexif() returns ExifData obj
    orientation = exif_data.get(274)  # tag 274 = Orientation
    if orientation is None:
        return None
    try:
        return int(orientation)
    except (TypeError, ValueError):
        return None


# ── Hough-based rotation detection ────────────────────────────────────────────

def _detect_rotation_hough(
    image: np.ndarray,
    min_lines: int = 10,
    confidence_threshold: float = 0.60,
) -> tuple[int, float]:
    """
    Detect cardinal rotation (0/90/180/270°) using dominant Hough line angles.

    Algorithm:
      1. Grayscale + Canny edges
      2. Probabilistic Hough Lines (HoughLinesP)
      3. Compute angle of each line segment (0–180°)
      4. Bucket into 4 quadrants:
           near-0°   (±20°) → image is portrait-upright
           near-90°  (±20°) → image is on its side (needs 90° fix)
           near-180°        → same as near-0° (Hough wraps at 180°)
      5. If dominant bucket has ≥ confidence_threshold of all lines AND
         at least min_lines lines were found → apply that rotation.
      6. Else → return (0, 0.0) — do nothing.

    Returns (rotation_degrees, confidence).

    Why Hough and not text-direction detection:
      Text detection would require OCR or a deep model.  Hough on edges is
      fast, dependency-free, and reliable for detecting 90/180/270° cases
      where all receipt content lines up in one direction.
    """
    if image is None or image.size == 0:
        return 0, 0.0

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Resize to max 800px for speed
        h, w = gray.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=max(30, min(gray.shape) // 8),
            maxLineGap=10,
        )

        if lines is None or len(lines) < min_lines:
            return 0, 0.0

        # Compute angle of each line segment in degrees [0, 180)
        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))) % 180
            angles.append(angle)

        # Bucket: near-0 (horizontal lines) vs near-90 (vertical lines)
        # Receipt text lines are horizontal → near-0° means upright
        bucket_horizontal = sum(1 for a in angles if a < 20 or a > 160)   # near 0°/180°
        bucket_vertical   = sum(1 for a in angles if 70 <= a <= 110)      # near 90°

        total = len(angles)
        conf_h = bucket_horizontal / total
        conf_v = bucket_vertical   / total

        # If vertical lines dominate → receipt is sideways → rotate 90°
        if conf_v >= confidence_threshold and conf_v > conf_h:
            return 90, conf_v

        # If horizontal lines dominate (or neither clear) → upright
        return 0, conf_h

    except Exception as exc:
        log.warning("_detect_rotation_hough() error: %s", exc)
        return 0, 0.0


# ── Main public function ──────────────────────────────────────────────────────

def fix_orientation(
    image: np.ndarray,
    exif_data: dict | None = None,
    use_hough: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Detect and fix cardinal rotation (0 / 90 / 180 / 270°).

    use_hough=False (default): EXIF-only. Safer — Hough tends to false-positive
    on portrait receipts with dominant vertical paper edges (exp_002 finding:
    14/100 images mis-rotated, causing -18pp E2E regression on group 03).

    Priority:
      1. EXIF Orientation tag (if exif_data provided) — exact, instant
      2. Hough line analysis (fallback when no EXIF)   — confidence-gated

    Returns (corrected_image, metadata_dict).
    metadata keys:
      method           : 'exif' | 'hough' | 'none'
      rotation_applied : int  (0, 90, 180, or 270)
      confidence       : float  (0.0–1.0; 1.0 for EXIF)
      warning          : str | None

    Never raises.  If unsure, returns original image with rotation_applied=0.
    """
    if image is None or image.size == 0:
        return image, {
            "method": "none",
            "rotation_applied": 0,
            "confidence": 0.0,
            "warning": "empty image",
        }

    try:
        # ── Strategy 1: EXIF ─────────────────────────────────────────────────
        exif_orientation = _read_exif_orientation(exif_data)

        if exif_orientation is not None and exif_orientation != 1:
            if exif_orientation in _EXIF_ROTATION:
                rotation = _EXIF_ROTATION[exif_orientation]
                corrected = _apply_rotation(image, rotation)
                return corrected, {
                    "method": "exif",
                    "rotation_applied": rotation,
                    "confidence": 1.0,
                    "warning": None,
                }
            elif exif_orientation in _EXIF_MIRROR_THEN_ROTATE:
                rotation = _EXIF_MIRROR_THEN_ROTATE[exif_orientation]
                flipped = cv2.flip(image, 1)   # horizontal flip
                corrected = _apply_rotation(flipped, rotation)
                return corrected, {
                    "method": "exif",
                    "rotation_applied": rotation,
                    "confidence": 1.0,
                    "warning": f"mirror+rotate applied (EXIF orientation={exif_orientation})",
                }

        # ── Strategy 2: Hough (opt-in only) ─────────────────────────────────
        # Disabled by default (use_hough=False) after exp_002 showed that Hough
        # produces 14% false-positive rotation on portrait receipts whose vertical
        # paper edges cause >60% "vertical line" votes, triggering a 90° rotation.
        if use_hough:
            rotation, confidence = _detect_rotation_hough(image)
            if rotation != 0 and confidence > 0:
                corrected = _apply_rotation(image, rotation)
                return corrected, {
                    "method": "hough",
                    "rotation_applied": rotation,
                    "confidence": confidence,
                    "warning": None,
                }

        return image, {
            "method": "none",
            "rotation_applied": 0,
            "confidence": 1.0 if exif_orientation == 1 else 0.0,
            "warning": None,
        }

    except Exception as exc:
        log.error("fix_orientation() error: %s", exc)
        return image, {
            "method": "none",
            "rotation_applied": 0,
            "confidence": 0.0,
            "warning": f"fix_orientation() error: {exc}",
        }
