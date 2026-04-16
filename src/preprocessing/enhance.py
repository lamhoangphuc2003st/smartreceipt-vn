"""
src/preprocessing/enhance.py
-----------------------------
Image enhancement for Vietnamese receipt photos before sending to Gemini.

Applies (in order, conditionally):
  1. Mode normalization   — RGBA / Palette → RGB (always)
  2. Gamma correction     — lifts shadows for dark images (if is_dark)
  3. CLAHE                — local contrast enhancement on L channel of LAB (always)
  4. NLM denoising        — edge-preserving noise removal (only if NOT blurry)
  5. Resize               — scale longest side to target_long_side (always)

All public functions are pure (no side effects, return new arrays).
Never raises — callers rely on this guarantee.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Default parameters ────────────────────────────────────────────────────────

TARGET_LONG_SIDE: int = 1600          # px — Gemini reads well up to this size
CLAHE_CLIP_LIMIT: float = 2.0         # conservative — avoids halos on receipt paper
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)
DENOISE_H: float = 6.0                # NLM filter strength (0–30 scale)
GAMMA_DARK: float = 1.8               # lifts shadows without blowing highlights


# ── Mode normalization ─────────────────────────────────────────────────────────

def normalize_mode(image: np.ndarray) -> np.ndarray:
    """
    Convert RGBA / Palette images to BGR (OpenCV's native format).
    No-op for images already in BGR or Grayscale.

    All preprocessing functions in this package expect BGR numpy arrays.
    Call this first if the source is PIL (which uses RGB mode).
    """
    if image is None or image.size == 0:
        return image
    # 4-channel (BGRA or RGBA from PIL)
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # Grayscale → BGR so downstream CLAHE (LAB) works uniformly
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


# ── Gamma correction ──────────────────────────────────────────────────────────

def apply_gamma_correction(image: np.ndarray, gamma: float = GAMMA_DARK) -> np.ndarray:
    """
    LUT-based gamma correction.  gamma > 1 → lifts shadows (brightens dark images).

    Preferred over linear brightness boost (cv2.convertScaleAbs) because
    gamma compresses highlights and lifts shadows, which matches both human
    perception and the distribution of images that VLMs were trained on.

    O(1) per image after LUT build (256-entry lookup table).
    """
    if gamma <= 0:
        return image
    inv_gamma = 1.0 / gamma
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


# ── CLAHE contrast enhancement ────────────────────────────────────────────────

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization applied on the L channel
    of the LAB colorspace.

    Why CLAHE over global histogram equalization:
      - Global EQ applies one curve to the whole image.  A receipt with bright
        paper AND dark shadows at edges gets the shadow areas inverted.
      - CLAHE's local tiles handle lighting variation correctly.
      - clipLimit=2.0 prevents halos around dark text on white paper.

    Why L channel of LAB:
      - Modifying only luminance preserves hue/saturation (colors of logos, etc.).
    """
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l_enhanced = clahe.apply(l_ch)
        enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    except Exception as exc:
        log.warning("apply_clahe failed (%s) — returning original", exc)
        return image


# ── NLM denoising ─────────────────────────────────────────────────────────────

def apply_denoise(image: np.ndarray, h: float = DENOISE_H) -> np.ndarray:
    """
    Non-Local Means denoising (edge-preserving).

    Why NLM over Gaussian:
      - NLM removes random sensor/JPEG noise without blurring text edges.
      - Gaussian smoothing blurs character strokes — bad for VLM text reading.

    IMPORTANT — only call when the image is NOT already blurry.
    Denoising a blurry image cannot recover sharpness; it only smears further.
    The pipeline.py idempotency gate enforces this.

    h=6.0 is conservative: strong enough to reduce JPEG noise, weak enough not
    to destroy Vietnamese diacritical marks (which are small and detail-rich).
    """
    try:
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=h,
            hColor=h,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    except Exception as exc:
        log.warning("apply_denoise failed (%s) — returning original", exc)
        return image


# ── Resize ────────────────────────────────────────────────────────────────────

def resize_for_vlm(
    image: np.ndarray,
    target_long_side: int = TARGET_LONG_SIDE,
) -> np.ndarray:
    """
    Resize image so its longest side equals target_long_side, preserving aspect ratio.

    Why 1600px:
      - Gemini supports up to 3072px but shows diminishing returns past 1600px
        for text-heavy documents.
      - Our dataset averages 763×974px — most images are UPSCALED, not downscaled.
        At 1600px, Vietnamese diacritical combining marks are typically 8–12px
        tall and stay distinguishable; at 1024px they can merge.

    Interpolation choice:
      - INTER_CUBIC for upscaling   → smoother than LINEAR, avoids blocky artefacts
      - INTER_AREA for downscaling  → avoids aliasing (Gaussian-equivalent anti-alias)
    """
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side == 0 or long_side == target_long_side:
        return image
    scale = target_long_side / long_side
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


# ── Main enhancement function ─────────────────────────────────────────────────

def enhance(
    image: np.ndarray,
    quality_flags: Any | None = None,   # QualityFlags from document_detect.py
    target_long_side: int = TARGET_LONG_SIDE,
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    clahe_tile_grid: tuple[int, int] = CLAHE_TILE_GRID,
    denoise_h: float = DENOISE_H,
    gamma: float = GAMMA_DARK,
    enable_denoise: bool = False,
    clahe_only_if_needed: bool = True,  # skip CLAHE on clean images (exp_003 finding)
) -> tuple[np.ndarray, dict]:
    """
    Full enhancement pipeline.  All parameters are explicit for testability.

    Returns (enhanced_image, metadata_dict).
    metadata keys:
      steps_applied : list[str]  — which steps actually ran
      input_size    : (h, w)
      output_size   : (h, w)
      warnings      : list[str]

    Never raises.
    """
    steps_applied: list[str] = []
    warnings: list[str] = []

    if image is None or image.size == 0:
        return image, {
            "steps_applied": steps_applied,
            "input_size": None,
            "output_size": None,
            "warnings": ["empty image passed to enhance()"],
        }

    input_h, input_w = image.shape[:2]
    result = image.copy()

    try:
        # Step 1: Mode normalization (always)
        result = normalize_mode(result)
        steps_applied.append("normalize_mode")

        # Step 2: Gamma correction (only for dark images)
        is_dark = getattr(quality_flags, "is_dark", False)
        if is_dark:
            result = apply_gamma_correction(result, gamma=gamma)
            steps_applied.append("gamma_correction")

        # Step 3: CLAHE — always unless clahe_only_if_needed=True and image is clean.
        # exp_003 finding: CLAHE on clean high-contrast receipts caused -10pp regression.
        # Clean images already have good contrast; CLAHE introduces subtle tile-boundary
        # artifacts that degrade Gemini's text reading on white POS thermal paper.
        is_blurry = getattr(quality_flags, "is_blurry", False)
        needs_clahe = is_dark or is_blurry or not clahe_only_if_needed
        if needs_clahe:
            result = apply_clahe(result, clip_limit=clahe_clip_limit, tile_grid=clahe_tile_grid)
            steps_applied.append("clahe")
        else:
            steps_applied.append("clahe_skipped_clean")

        # Step 4: NLM denoising (disabled by default after exp_002)
        # exp_002 finding: NLM on clean receipt images degraded E2E by ~5pp.
        # Vietnamese diacritical marks are fine-grained; NLM smoothing at h=6
        # blurs them even on "non-blurry" images. Enable only for known noisy images.
        if enable_denoise and not is_blurry:
            result = apply_denoise(result, h=denoise_h)
            steps_applied.append("denoise")
        elif enable_denoise and is_blurry:
            warnings.append("image is blurry — skipped denoising (would only smear)")

        # Step 5: Resize (always)
        result = resize_for_vlm(result, target_long_side=target_long_side)
        steps_applied.append("resize")

    except Exception as exc:
        log.error("enhance() unexpected error: %s", exc)
        warnings.append(f"enhance() partial failure: {exc}")

    output_h, output_w = result.shape[:2]
    return result, {
        "steps_applied": steps_applied,
        "input_size": (input_h, input_w),
        "output_size": (output_h, output_w),
        "warnings": warnings,
    }
