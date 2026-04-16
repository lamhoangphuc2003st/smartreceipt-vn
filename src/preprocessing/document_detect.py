"""
src/preprocessing/document_detect.py
--------------------------------------
Two responsibilities:
  1. Quality assessment  — detect blur, darkness, empty images
  2. Document detection  — find the receipt bounding quadrilateral and crop

Both are "is this image ready to process?" checks that inform downstream steps.

Design principles:
  - Never raise to caller.
  - If document detection fails, fall back to full image (crop_recommended=False).
  - Correction (detect & crop) is preferred over rejection.
  - Every warning is logged and returned in QualityFlags.warnings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Quality thresholds ────────────────────────────────────────────────────────

BLUR_THRESHOLD: float = 80.0    # Laplacian variance; below → blurry
DARK_THRESHOLD: float = 50.0    # mean pixel brightness 0–255; below → dark
EMPTY_THRESHOLD: float = 250.0  # mean brightness above AND std_dev below → truly blank
EMPTY_STD_THRESHOLD: float = 8.0  # std_dev below this → near-uniform → blank (not a receipt)

# ── Document detection parameters ─────────────────────────────────────────────

MIN_AREA_RATIO: float = 0.10    # quad must cover ≥10% of image area
ANGLE_TOLERANCE: float = 30.0   # interior angles must be within 90°±30°
CROP_PADDING: float = 0.02      # 2% padding margin when cropping to quad


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class QualityFlags:
    """Image quality assessment result."""
    is_blurry: bool
    blur_score: float        # Laplacian variance — higher = sharper
    is_dark: bool
    mean_brightness: float   # 0–255
    std_brightness: float    # pixel std_dev — low on blank/uniform images
    is_empty: bool           # near-blank / all-white image (high mean AND low std_dev)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DocumentBounds:
    """Result of document (receipt) bounding box detection."""
    corners: np.ndarray | None   # shape (4, 2) float32, TL/TR/BR/BL order, or None
    confidence: float            # 0.0–1.0
    method: str                  # 'contour_quad' | 'full_image'
    crop_recommended: bool
    warning: str | None = None


# ── Quality assessment ─────────────────────────────────────────────────────────

def assess_quality(image: np.ndarray) -> QualityFlags:
    """
    Assess image quality: blur, darkness, emptiness.

    blur_score = Laplacian variance.  Rule of thumb for receipt photos:
      < 80   → blurry (significant motion blur or out-of-focus)
      80–300 → acceptable
      > 300  → sharp

    Never raises.
    """
    warnings: list[str] = []

    if image is None or image.size == 0:
        return QualityFlags(
            is_blurry=True,
            blur_score=0.0,
            is_dark=True,
            mean_brightness=0.0,
            std_brightness=0.0,
            is_empty=True,
            warnings=["empty image passed to assess_quality()"],
        )

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))

        std_brightness = float(np.std(gray))
        is_blurry = blur_score < BLUR_THRESHOLD
        is_dark = mean_brightness < DARK_THRESHOLD
        # Truly blank images: near-uniform white (high mean AND low std_dev).
        # White receipts with dark text have mean > 200 but std_dev > 20.
        # A blank/camera-cap image has mean > 250 AND std_dev < 8.
        is_empty = mean_brightness > EMPTY_THRESHOLD and std_brightness < EMPTY_STD_THRESHOLD

        if is_blurry:
            warnings.append(
                f"image appears blurry (Laplacian variance={blur_score:.1f} < {BLUR_THRESHOLD})"
            )
        if is_dark:
            warnings.append(
                f"image is underexposed (mean brightness={mean_brightness:.1f} < {DARK_THRESHOLD})"
            )
        if is_empty:
            warnings.append(
                f"image appears blank/empty (mean={mean_brightness:.1f} > {EMPTY_THRESHOLD}"
                f", std={std_brightness:.1f} < {EMPTY_STD_THRESHOLD})"
            )

        return QualityFlags(
            is_blurry=is_blurry,
            blur_score=blur_score,
            is_dark=is_dark,
            mean_brightness=mean_brightness,
            std_brightness=std_brightness,
            is_empty=is_empty,
            warnings=warnings,
        )

    except Exception as exc:
        log.error("assess_quality() error: %s", exc)
        return QualityFlags(
            is_blurry=False,
            blur_score=0.0,
            is_dark=False,
            mean_brightness=128.0,
            std_brightness=64.0,
            is_empty=False,
            warnings=[f"assess_quality() error: {exc}"],
        )


# ── Internal contour helpers ───────────────────────────────────────────────────

def _sort_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """
    Sort 4 corner points into TL, TR, BR, BL order.

    Algorithm:
      - TL has smallest (x+y)
      - BR has largest (x+y)
      - TR has smallest (y-x)  [large x, small y]
      - BL has largest (y-x)   [small x, large y]

    Required for consistent input to perspective.py's warpPerspective.
    """
    pts = corners.reshape(4, 2).astype(np.float32)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmin(diffs)]
    bl = pts[np.argmax(diffs)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _is_valid_rectangle(quad: np.ndarray, angle_tolerance: float = ANGLE_TOLERANCE) -> bool:
    """
    Check that a 4-corner polygon has roughly right-angle corners.

    Each interior angle must be within 90° ± angle_tolerance.
    This filters out non-receipt quads (thin triangles, very oblique parallelograms).
    """
    pts = quad.reshape(4, 2).astype(np.float32)
    for i in range(4):
        p0 = pts[(i - 1) % 4]
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return False
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        if abs(angle_deg - 90.0) > angle_tolerance:
            return False
    return True


def _find_quad_contours(
    edges: np.ndarray,
    image_area: int,
    min_area_ratio: float = MIN_AREA_RATIO,
) -> list[np.ndarray]:
    """
    Find quadrilateral contours in an edge map, sorted by area (largest first).

    Returns list of (4, 1, 2) int32 arrays — the raw approxPolyDP output.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads: list[tuple[float, np.ndarray]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < image_area * min_area_ratio:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quads.append((area, approx))

    quads.sort(key=lambda x: x[0], reverse=True)
    return [q for _, q in quads]


# ── Document detection ─────────────────────────────────────────────────────────

def detect_document(
    image: np.ndarray,
    min_area_ratio: float = MIN_AREA_RATIO,
    angle_tolerance: float = ANGLE_TOLERANCE,
) -> DocumentBounds:
    """
    Detect the receipt bounding quadrilateral in an image.

    Algorithm:
      1. Downsample to ≤1024px longest side (speed — contours don't need full res)
      2. Grayscale → Gaussian blur → Canny edge detection
      3. Find largest quad contour with area > min_area_ratio of image
      4. Validate corners are roughly rectangular
      5. Scale corners back to original image size

    Fallback (safe path):
      If no valid quad found, return DocumentBounds with crop_recommended=False.
      This is the correct behaviour for receipts that fill the whole frame
      (most group 01 POS receipts) — no crop needed, the receipt IS the image.

    Never raises.
    """
    if image is None or image.size == 0:
        return DocumentBounds(
            corners=None,
            confidence=0.0,
            method="full_image",
            crop_recommended=False,
            warning="empty image",
        )

    try:
        orig_h, orig_w = image.shape[:2]
        image_area = orig_h * orig_w

        # Downsample for fast contour finding
        max_detect_side = 1024
        long_side = max(orig_h, orig_w)
        if long_side > max_detect_side:
            scale = max_detect_side / long_side
            detect_w = max(1, round(orig_w * scale))
            detect_h = max(1, round(orig_h * scale))
            small = cv2.resize(image, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
            detect_area = detect_h * detect_w
        else:
            small = image
            scale = 1.0
            detect_area = image_area

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Auto-threshold Canny using Otsu threshold value.
        # cv2.threshold returns (retval_scalar, thresholded_image).
        # We want the scalar retval, NOT the image — hence otsu_val first.
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_val = float(otsu_val) if otsu_val > 0 else 128.0
        edges = cv2.Canny(blurred, otsu_val * 0.5, otsu_val)
        # Dilate edges slightly to close small gaps in receipt border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        quads = _find_quad_contours(edges, detect_area, min_area_ratio)

        for quad in quads:
            if _is_valid_rectangle(quad, angle_tolerance):
                # Scale corners back to original image coordinates
                corners_scaled = (quad.reshape(4, 2).astype(np.float32) / scale)
                sorted_corners = _sort_corners_clockwise(corners_scaled)

                # Confidence: ratio of quad area to image area (higher = more dominant)
                quad_area = cv2.contourArea(quad) / (scale ** 2)
                confidence = min(1.0, quad_area / image_area)

                return DocumentBounds(
                    corners=sorted_corners,
                    confidence=confidence,
                    method="contour_quad",
                    crop_recommended=True,
                )

        # No valid quad found — safe fallback
        return DocumentBounds(
            corners=None,
            confidence=0.0,
            method="full_image",
            crop_recommended=False,
            warning="no valid quadrilateral found — using full image",
        )

    except Exception as exc:
        log.error("detect_document() error: %s", exc)
        return DocumentBounds(
            corners=None,
            confidence=0.0,
            method="full_image",
            crop_recommended=False,
            warning=f"detect_document() error: {exc}",
        )


# ── Crop ──────────────────────────────────────────────────────────────────────

def crop_to_document(
    image: np.ndarray,
    bounds: DocumentBounds,
    padding: float = CROP_PADDING,
) -> np.ndarray:
    """
    Crop the image to the detected document bounding box with padding.

    Returns original image if:
      - bounds.crop_recommended is False
      - bounds.corners is None
      - computed crop would be degenerate

    The padding (default 2%) prevents accidentally clipping receipt edges
    when the detected quad is very tight.
    """
    if not bounds.crop_recommended or bounds.corners is None:
        return image

    try:
        h, w = image.shape[:2]
        corners = bounds.corners

        # Compute tight bounding rectangle of the quad
        x_min = max(0, int(np.min(corners[:, 0])))
        y_min = max(0, int(np.min(corners[:, 1])))
        x_max = min(w, int(np.max(corners[:, 0])))
        y_max = min(h, int(np.max(corners[:, 1])))

        # Add padding
        pad_x = max(1, int((x_max - x_min) * padding))
        pad_y = max(1, int((y_max - y_min) * padding))
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        crop_w = x_max - x_min
        crop_h = y_max - y_min

        # Guard: degenerate crop
        if crop_w < 32 or crop_h < 32:
            return image

        return image[y_min:y_max, x_min:x_max]

    except Exception as exc:
        log.error("crop_to_document() error: %s", exc)
        return image
