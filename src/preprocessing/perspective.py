"""
src/preprocessing/perspective.py
----------------------------------
Perspective warp: given 4 corners from document_detect.py, warp the receipt
to a flat rectangle, removing keystoning from an angled phone camera.

Design principles:
  - Only runs when DocumentBounds.crop_recommended=True and corners are valid.
  - Validates the quad before warping — skips if aspect ratio is suspicious.
  - Fills background with white (255, 255, 255) — receipts are on white paper.
  - Uses LANCZOS4 for high-quality interpolation (better text readability).
  - Never raises to caller.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ASPECT_RATIO: float = 8.0   # skip warp if computed W/H or H/W > this value


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_output_dims(corners: np.ndarray) -> tuple[int, int]:
    """
    Compute the output rectangle dimensions from 4 sorted corners.

    corners order: TL, TR, BR, BL (as returned by _sort_corners_clockwise).

    Width  = max(dist(TL, TR), dist(BL, BR))
    Height = max(dist(TL, BL), dist(TR, BR))

    Using max() rather than mean() avoids shrinking the output when one
    side is foreshortened by perspective.
    """
    tl, tr, br, bl = corners.astype(np.float32)
    width_top    = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_left  = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    width  = max(1, int(round(max(width_top, width_bottom))))
    height = max(1, int(round(max(height_left, height_right))))
    return width, height


def _validate_quad(
    corners: np.ndarray,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> tuple[bool, str]:
    """
    Validate that the 4 corners produce a sensible rectangle.

    Returns (is_valid, reason_if_invalid).

    Rejects:
      - Degenerate quads (area near 0 — collinear points)
      - Extreme aspect ratios (> max_aspect_ratio in either direction)
        These indicate a false-positive quad detection.
    """
    if corners is None or corners.shape != (4, 2):
        return False, "corners must be a (4, 2) array"

    # Check area via 2D cross product (scalar): |a×b| = |ax*by - ay*bx|
    tl, tr, br, bl = corners.astype(np.float32)
    diag1 = br - tl
    diag2 = tr - bl
    area = abs(float(diag1[0] * diag2[1] - diag1[1] * diag2[0])) / 2.0
    if area < 100:   # fewer than 100 px² → degenerate
        return False, f"quad area too small ({area:.1f} px²)"

    w, h = _compute_output_dims(corners)
    if w == 0 or h == 0:
        return False, "degenerate output dimensions"
    ratio = w / h
    if ratio > max_aspect_ratio or ratio < (1.0 / max_aspect_ratio):
        return False, f"aspect ratio {ratio:.2f} exceeds limit {max_aspect_ratio}"

    return True, ""


# ── Main warp function ────────────────────────────────────────────────────────

def warp_perspective(
    image: np.ndarray,
    corners: np.ndarray | None,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> tuple[np.ndarray, dict]:
    """
    Apply perspective warp to straighten a receipt.

    corners: (4, 2) float32 in TL/TR/BR/BL order (from _sort_corners_clockwise).

    Returns (warped_image, metadata_dict).
    metadata keys:
      applied        : bool
      reason_skipped : str | None   — set when applied=False
      output_size    : (w, h) | None

    Skips warp (returns original image) when:
      - corners is None
      - validation fails (degenerate / extreme aspect ratio)

    Why LANCZOS4:
      At our typical image sizes (avg 763×974px), the compute difference vs
      INTER_LINEAR is negligible. LANCZOS4 produces visibly sharper text edges,
      which directly helps the VLM read characters and Vietnamese diacritics.
    """
    if corners is None:
        return image, {
            "applied": False,
            "reason_skipped": "no corners provided",
            "output_size": None,
        }

    if image is None or image.size == 0:
        return image, {
            "applied": False,
            "reason_skipped": "empty image",
            "output_size": None,
        }

    try:
        valid, reason = _validate_quad(corners, max_aspect_ratio)
        if not valid:
            log.debug("warp_perspective skipped: %s", reason)
            return image, {
                "applied": False,
                "reason_skipped": reason,
                "output_size": None,
            }

        w, h = _compute_output_dims(corners)
        dst_pts = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(
            image,
            M,
            (w, h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),   # white fill — receipt paper colour
        )

        return warped, {
            "applied": True,
            "reason_skipped": None,
            "output_size": (w, h),
        }

    except Exception as exc:
        log.error("warp_perspective() error: %s", exc)
        return image, {
            "applied": False,
            "reason_skipped": f"error: {exc}",
            "output_size": None,
        }
