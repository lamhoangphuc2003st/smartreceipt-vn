"""
src/preprocessing/deskew.py
-----------------------------
Fix small-angle tilt (typically ±15°) from a slightly off-angle phone camera.

This is distinct from orientation.py which handles 90/180/270° rotations.
Deskew handles continuous angles: the receipt is roughly upright but tilted.

Algorithm: Projection Profile Analysis
  - Rotate a binarized version of the image by candidate angles
  - At the correct angle, horizontal text rows produce sharp peaks in the
    row-sum projection (high row-sum variance)
  - Find the angle that maximizes variance → that is the tilt to correct

Why projection profile over Hough lines:
  - HoughLinesP gives individual edge angles — noisy on complex receipt textures
  - Projection profile integrates information over the WHOLE image — robust to
    local noise, partial legibility, and mixed-orientation content
  - Well-established in production OCR pre-processing pipelines
  - Works even when text is degraded or partially illegible

Safety rules:
  - Only apply if |angle| > min_angle_threshold (0.5°) — below this, the
    correction introduces more interpolation artifacts than it fixes
  - Only apply if |angle| < max_angle (15°) — beyond this, it's probably a
    genuine layout choice, not camera tilt
  - Skip if image has < 1% white pixels (too little content to analyze)
  - Skip if score improvement < 10% (variance ratio < 1.1)

Never raises to caller.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Parameters ────────────────────────────────────────────────────────────────

MAX_ANGLE: float = 15.0            # degrees — don't correct beyond this
ANGLE_STEP: float = 0.5            # degrees — search granularity
MIN_ANGLE_THRESHOLD: float = 0.5   # degrees — don't correct below this
MIN_CONTENT_RATIO: float = 0.01    # skip if < 1% of pixels are "content"
MIN_SCORE_RATIO: float = 1.10      # require 10% improvement in variance


# ── Internal helpers ──────────────────────────────────────────────────────────

def _rotate_image_subpixel(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by a small angle around its center.

    Fills background with white (255) — receipt paper colour.
    Uses cv2.INTER_LINEAR (adequate for binary/grayscale intermediate results).
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # Expand canvas to avoid clipping corners
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    if len(image.shape) == 2:
        fill = 255
    else:
        fill = (255, 255, 255)
    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _compute_projection_score(binary: np.ndarray, angle: float) -> float:
    """
    Rotate binary image by `angle` and return the variance of the horizontal
    row-sum projection.

    Higher variance = text rows are more aligned horizontally = better deskew.
    """
    if angle == 0.0:
        rotated = binary
    else:
        rotated = _rotate_image_subpixel(binary, angle)
    # Row sums — count white (content) pixels per row
    row_sums = np.sum(rotated == 255, axis=1).astype(np.float64)
    return float(np.var(row_sums))


def _find_skew_angle(
    binary: np.ndarray,
    max_angle: float = MAX_ANGLE,
    angle_step: float = ANGLE_STEP,
) -> tuple[float, float]:
    """
    Search for the skew angle that maximizes horizontal projection variance.

    Returns (best_angle_degrees, best_score).
    best_angle is in [-max_angle, +max_angle].
    """
    best_angle = 0.0
    best_score = _compute_projection_score(binary, 0.0)

    angles = np.arange(-max_angle, max_angle + angle_step, angle_step)
    for angle in angles:
        score = _compute_projection_score(binary, float(angle))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return best_angle, best_score


# ── Main public function ──────────────────────────────────────────────────────

def deskew(
    image: np.ndarray,
    max_angle: float = MAX_ANGLE,
    angle_step: float = ANGLE_STEP,
    min_angle_threshold: float = MIN_ANGLE_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    """
    Detect and correct small-angle tilt in a receipt image.

    Returns (deskewed_image, metadata_dict).
    metadata keys:
      angle_detected : float  — detected skew angle (degrees)
      angle_applied  : float  — rotation actually applied (0 if skipped)
      method         : 'projection' | 'skipped'
      score_before   : float  — row-sum variance before correction
      score_after    : float  — row-sum variance after correction
      warning        : str | None

    Never raises.
    """
    meta_base = {
        "angle_detected": 0.0,
        "angle_applied": 0.0,
        "method": "skipped",
        "score_before": 0.0,
        "score_after": 0.0,
        "warning": None,
    }

    if image is None or image.size == 0:
        return image, {**meta_base, "warning": "empty image"}

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Binarize via Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Check content ratio — skip if image is near-blank
        white_ratio = float(np.sum(binary == 255)) / binary.size
        if white_ratio < MIN_CONTENT_RATIO:
            return image, {
                **meta_base,
                "warning": f"image has < {MIN_CONTENT_RATIO:.0%} content — skipping deskew",
            }

        score_before = _compute_projection_score(binary, 0.0)
        best_angle, best_score = _find_skew_angle(binary, max_angle, angle_step)

        meta = {
            "angle_detected": round(best_angle, 2),
            "angle_applied": 0.0,
            "method": "skipped",
            "score_before": round(score_before, 2),
            "score_after": round(best_score, 2),
            "warning": None,
        }

        # Apply only if:
        #   - angle is meaningful (> min threshold)
        #   - angle is within safe range (< max_angle)
        #   - improvement is significant (score ratio > MIN_SCORE_RATIO)
        angle_abs = abs(best_angle)
        score_ratio = best_score / score_before if score_before > 0 else 1.0

        if angle_abs < min_angle_threshold:
            meta["warning"] = (
                f"detected angle {best_angle:.2f}° is below "
                f"threshold {min_angle_threshold}° — not correcting"
            )
            return image, meta

        if angle_abs > max_angle:
            meta["warning"] = (
                f"detected angle {best_angle:.2f}° exceeds max "
                f"{max_angle}° — not correcting (probably not deskew)"
            )
            return image, meta

        if score_ratio < MIN_SCORE_RATIO:
            meta["warning"] = (
                f"score improvement {score_ratio:.3f} < {MIN_SCORE_RATIO} — "
                "not worth correcting"
            )
            return image, meta

        # Apply rotation to the original (colour) image
        corrected = _rotate_image_subpixel(image, best_angle)
        meta["angle_applied"] = round(best_angle, 2)
        meta["method"] = "projection"
        return corrected, meta

    except Exception as exc:
        log.error("deskew() error: %s", exc)
        return image, {**meta_base, "warning": f"deskew() error: {exc}"}
