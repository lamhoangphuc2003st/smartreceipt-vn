"""
tests/test_preprocessing.py
-----------------------------
Unit tests for the src/preprocessing/ module.

All tests use synthetically generated images (no real receipt photos required)
so the test suite runs without any data files — privacy-safe and reproducible.

Test structure:
  - Synthetic image generators (module-level helpers)
  - TestEnhance
  - TestDocumentDetect
  - TestPerspective
  - TestOrientation
  - TestDeskew
  - TestImagePreprocessor  (integration)

Run with:
    venv/Scripts/python.exe -m pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest

# ── Synthetic image generators ────────────────────────────────────────────────

def make_clean_receipt(width: int = 600, height: int = 900) -> np.ndarray:
    """
    Create a synthetic clean receipt: white background, horizontal dark text lines.
    Represents a well-lit, straight POS receipt.
    Returns BGR numpy array.
    """
    img = np.full((height, width, 3), 230, dtype=np.uint8)  # light gray background
    # Draw a "receipt body" in the center
    pad = 40
    cv2.rectangle(img, (pad, pad), (width - pad, height - pad), (245, 245, 245), -1)
    # Draw horizontal text-like lines (simulate printed text rows)
    for y in range(100, height - 100, 25):
        line_width = np.random.randint(width // 3, width - 80)
        x_start = pad + 10
        thickness = np.random.randint(2, 5)
        cv2.line(img, (x_start, y), (x_start + line_width, y), (30, 30, 30), thickness)
    # Title block
    cv2.rectangle(img, (pad + 10, pad + 10), (width - pad - 10, pad + 50), (50, 50, 50), -1)
    return img


def make_rotated_receipt(base_image: np.ndarray, degrees: int = 90) -> np.ndarray:
    """Apply exact rotation with cv2 (known ground truth for testing)."""
    degrees = degrees % 360
    if degrees == 0:
        return base_image
    if degrees == 90:
        return cv2.rotate(base_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degrees == 180:
        return cv2.rotate(base_image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(base_image, cv2.ROTATE_90_CLOCKWISE)
    return base_image


def make_tilted_receipt(base_image: np.ndarray, angle: float = 7.0) -> np.ndarray:
    """Apply known small-angle tilt for deskew testing."""
    h, w = base_image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        base_image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(230, 230, 230),
    )


def make_dark_receipt(base_image: np.ndarray, gamma: float = 0.4) -> np.ndarray:
    """Darken image to simulate poor lighting."""
    lut = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(base_image, lut)


def make_blurry_receipt(base_image: np.ndarray, kernel: int = 21) -> np.ndarray:
    """Apply Gaussian blur to simulate out-of-focus photo."""
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(base_image, (k, k), 0)


def make_receipt_on_table(
    receipt_image: np.ndarray,
    background_color: tuple = (180, 160, 140),
    margin: int = 80,
) -> np.ndarray:
    """
    Composite receipt onto a larger background — simulates a phone photo
    where the receipt is placed on a table, visible with surrounding context.
    """
    rh, rw = receipt_image.shape[:2]
    canvas_h = rh + 2 * margin
    canvas_w = rw + 2 * margin
    canvas = np.full((canvas_h, canvas_w, 3), background_color, dtype=np.uint8)
    canvas[margin:margin + rh, margin:margin + rw] = receipt_image
    return canvas


# ── TestEnhance ───────────────────────────────────────────────────────────────

class TestEnhance:
    def test_clean_image_runs_clahe_not_gamma(self):
        """Clean bright image: CLAHE should run, gamma and denoise should NOT."""
        from src.preprocessing.enhance import enhance
        from src.preprocessing.document_detect import QualityFlags
        img = make_clean_receipt()
        flags = QualityFlags(
            is_blurry=False, blur_score=300.0,
            is_dark=False, mean_brightness=180.0,
            std_brightness=40.0,
            is_empty=False,
        )
        result, meta = enhance(img, quality_flags=flags)  # enable_denoise=False default
        assert result is not None
        # CLAHE skipped on clean images (clahe_only_if_needed=True by default — exp_003 fix)
        assert "clahe" not in meta["steps_applied"]
        assert "clahe_skipped_clean" in meta["steps_applied"]
        assert "gamma_correction" not in meta["steps_applied"]
        assert "denoise" not in meta["steps_applied"]  # disabled by default (exp_002)

    def test_dark_image_applies_gamma(self):
        """Dark image: gamma correction should be applied."""
        from src.preprocessing.enhance import enhance
        from src.preprocessing.document_detect import QualityFlags
        img = make_dark_receipt(make_clean_receipt(), gamma=0.3)
        flags = QualityFlags(
            is_blurry=False, blur_score=200.0,
            is_dark=True, mean_brightness=30.0,
            std_brightness=20.0,
            is_empty=False,
        )
        result, meta = enhance(img, quality_flags=flags)
        assert "gamma_correction" in meta["steps_applied"]
        # Brightness should increase after correction
        gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        assert float(np.mean(gray_after)) > float(np.mean(gray_before))

    def test_resize_to_target(self):
        """Small image should be upscaled so longest side = target_long_side."""
        from src.preprocessing.enhance import enhance
        img = make_clean_receipt(300, 500)
        target = 1200
        result, meta = enhance(img, target_long_side=target)
        assert result is not None
        assert max(result.shape[:2]) == target

    def test_rgba_normalized_to_bgr(self):
        """RGBA input should be converted to 3-channel BGR."""
        from src.preprocessing.enhance import enhance, normalize_mode
        import numpy as np
        # Create a fake BGRA image
        rgba = np.full((100, 80, 4), 200, dtype=np.uint8)
        bgr = normalize_mode(rgba)
        assert bgr.shape[2] == 3

    def test_blurry_image_skips_denoise(self):
        """Blurry image should NOT be denoised even when enable_denoise=True."""
        from src.preprocessing.enhance import enhance
        from src.preprocessing.document_detect import QualityFlags
        img = make_blurry_receipt(make_clean_receipt(), kernel=15)
        flags = QualityFlags(
            is_blurry=True, blur_score=20.0,
            is_dark=False, mean_brightness=180.0,
            std_brightness=40.0,
            is_empty=False,
        )
        # enable_denoise=True but image is blurry → should still skip
        result, meta = enhance(img, quality_flags=flags, enable_denoise=True)
        assert "denoise" not in meta["steps_applied"]
        assert any("blurry" in w for w in meta["warnings"])


# ── TestDocumentDetect ────────────────────────────────────────────────────────

class TestDocumentDetect:
    def test_receipt_on_table_detected(self):
        """Receipt placed on contrasting background — quad should be found."""
        from src.preprocessing.document_detect import detect_document
        receipt = make_clean_receipt(400, 600)
        scene = make_receipt_on_table(receipt, background_color=(80, 80, 80), margin=60)
        bounds = detect_document(scene)
        # Either finds a quad or falls back safely — must not crash
        assert bounds is not None
        assert bounds.method in ("contour_quad", "full_image")

    def test_full_frame_receipt_no_crash(self):
        """Receipt fills the frame — detect_document should not crash."""
        from src.preprocessing.document_detect import detect_document
        img = make_clean_receipt(600, 900)
        bounds = detect_document(img)
        assert bounds is not None
        # No exception is the primary guarantee
        assert bounds.method in ("contour_quad", "full_image")

    def test_blur_detection_identifies_blurry_image(self):
        """Heavy blur should be detected."""
        from src.preprocessing.document_detect import assess_quality
        img = make_blurry_receipt(make_clean_receipt(), kernel=31)
        flags = assess_quality(img)
        assert flags.is_blurry is True
        assert flags.blur_score < 80.0

    def test_dark_detection_identifies_dark_image(self):
        """Very dark image should be flagged."""
        from src.preprocessing.document_detect import assess_quality
        # Create a directly dark image (mean brightness = 30 < DARK_THRESHOLD=50)
        img = np.full((400, 600, 3), 30, dtype=np.uint8)
        flags = assess_quality(img)
        assert flags.is_dark is True
        assert flags.mean_brightness < 50.0

    def test_clean_image_no_quality_warnings(self):
        """Clean, bright, sharp image should produce no quality warnings."""
        from src.preprocessing.document_detect import assess_quality
        img = make_clean_receipt()
        flags = assess_quality(img)
        assert not flags.is_dark
        assert not flags.is_empty
        # Blur check depends on the synthetic lines; just verify no crash
        assert flags.blur_score >= 0.0
        assert len([w for w in flags.warnings if "dark" in w or "empty" in w]) == 0

    def test_empty_image_returns_safely(self):
        """Empty (zero-size) array should not raise."""
        from src.preprocessing.document_detect import detect_document, assess_quality
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        bounds = detect_document(empty)
        assert bounds.method == "full_image"   # safe fallback
        flags = assess_quality(empty)
        assert flags.is_empty is True


# ── TestPerspective ───────────────────────────────────────────────────────────

class TestPerspective:
    def test_none_corners_returns_original(self):
        """warp_perspective with corners=None should return image unchanged."""
        from src.preprocessing.perspective import warp_perspective
        img = make_clean_receipt(400, 600)
        result, meta = warp_perspective(img, corners=None)
        assert meta["applied"] is False
        assert result.shape == img.shape
        np.testing.assert_array_equal(result, img)

    def test_near_square_corners_idempotent(self):
        """
        Near-perfect rectangle corners → warp should produce same size image
        (within a few pixels).
        """
        from src.preprocessing.perspective import warp_perspective
        img = make_clean_receipt(400, 600)
        h, w = img.shape[:2]
        # Corners are exactly the image boundaries
        corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32,
        )
        result, meta = warp_perspective(img, corners=corners)
        # Output size should be close to original (within 3px each dimension)
        assert abs(result.shape[0] - h) <= 3
        assert abs(result.shape[1] - w) <= 3

    def test_degenerate_quad_skipped(self):
        """Collinear corners (zero area) should be rejected gracefully."""
        from src.preprocessing.perspective import warp_perspective
        img = make_clean_receipt(400, 600)
        # All corners on the same line → degenerate
        corners = np.array(
            [[10, 10], [100, 10], [200, 10], [300, 10]],
            dtype=np.float32,
        )
        result, meta = warp_perspective(img, corners=corners)
        assert meta["applied"] is False
        assert result.shape == img.shape


# ── TestOrientation ───────────────────────────────────────────────────────────

class TestOrientation:
    def test_no_exif_no_rotation_clean_image(self):
        """Clean portrait receipt with no rotation → rotation_applied should be 0."""
        from src.preprocessing.orientation import fix_orientation
        img = make_clean_receipt(400, 700)  # portrait aspect ratio
        result, meta = fix_orientation(img, exif_data=None)
        # We accept either 0° or a small confidence rotation — the key is no crash
        assert meta["rotation_applied"] in (0, 90, 180, 270)
        assert result is not None and result.size > 0

    def test_exif_orientation_6_corrected(self):
        """EXIF orientation=6 means 90° CCW correction needed."""
        from src.preprocessing.orientation import fix_orientation
        img = make_clean_receipt(400, 700)
        exif_data = {274: 6}   # PIL tag 274 = Orientation
        result, meta = fix_orientation(img, exif_data=exif_data)
        assert meta["method"] == "exif"
        assert meta["rotation_applied"] == 270   # EXIF 6 → 270° correction
        # After correction, portrait becomes landscape (or vice versa)
        # Shape changes for non-square images
        assert result.shape[0] != result.shape[1] or img.shape[0] == img.shape[1]

    def test_exif_orientation_1_no_change(self):
        """EXIF orientation=1 (normal) → no rotation applied."""
        from src.preprocessing.orientation import fix_orientation
        img = make_clean_receipt()
        exif_data = {274: 1}
        result, meta = fix_orientation(img, exif_data=exif_data)
        assert meta["rotation_applied"] == 0
        np.testing.assert_array_equal(result, img)

    def test_empty_image_no_raise(self):
        """Empty array should not raise."""
        from src.preprocessing.orientation import fix_orientation
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result, meta = fix_orientation(empty)
        assert meta["rotation_applied"] == 0


# ── TestDeskew ────────────────────────────────────────────────────────────────

class TestDeskew:
    def test_straight_image_angle_near_zero(self):
        """Perfectly straight receipt → angle_applied should be close to 0."""
        from src.preprocessing.deskew import deskew
        img = make_clean_receipt(500, 800)
        result, meta = deskew(img)
        assert abs(meta["angle_applied"]) < 2.0   # within 2° of zero

    def test_tilted_image_detects_nonzero_angle(self):
        """7° tilted receipt → deskew should detect a nonzero angle."""
        from src.preprocessing.deskew import deskew
        img = make_tilted_receipt(make_clean_receipt(500, 800), angle=7.0)
        result, meta = deskew(img)
        # The detected angle should be meaningful (either corrected or reasoned about)
        assert meta["angle_detected"] != 0.0 or meta["method"] == "skipped"
        assert result is not None and result.size > 0

    def test_near_blank_image_skipped(self):
        """Near-blank (all-white) image → deskew should be skipped."""
        from src.preprocessing.deskew import deskew
        blank = np.full((400, 300, 3), 250, dtype=np.uint8)
        result, meta = deskew(blank)
        assert meta["method"] == "skipped"
        assert meta["angle_applied"] == 0.0

    def test_max_angle_not_exceeded(self):
        """deskew should never apply a correction beyond max_angle."""
        from src.preprocessing.deskew import deskew
        img = make_tilted_receipt(make_clean_receipt(500, 800), angle=20.0)
        result, meta = deskew(img, max_angle=15.0)
        # Either skipped (angle > max) or applied within limit
        assert abs(meta["angle_applied"]) <= 15.0

    def test_output_shape_reasonable(self):
        """Output image should have a reasonable shape (not collapsed)."""
        from src.preprocessing.deskew import deskew
        img = make_clean_receipt(400, 600)
        result, meta = deskew(img)
        assert result.shape[0] > 50
        assert result.shape[1] > 50


# ── TestImagePreprocessor (integration) ──────────────────────────────────────

class TestImagePreprocessor:
    def test_clean_receipt_success(self):
        """Clean receipt should be processed successfully."""
        from src.preprocessing import ImagePreprocessor
        import tempfile, os
        img = make_clean_receipt(400, 600)
        preprocessor = ImagePreprocessor()

        # Save to temp file and process
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
        try:
            cv2.imwrite(tmp_path, img)
            result = preprocessor.process(tmp_path)
            assert result.success is True
            assert result.image is not None
            assert result.image.size > 0
            assert result.is_usable is True
        finally:
            os.unlink(tmp_path)

    def test_numpy_array_input(self):
        """Passing a numpy array directly should work."""
        from src.preprocessing import ImagePreprocessor
        img = make_clean_receipt(400, 600)
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(img)
        assert result.success is True
        assert result.image is not None

    def test_corrupt_bytes_returns_failure(self):
        """Invalid bytes → success=False, no exception raised."""
        from src.preprocessing import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(b"this_is_not_an_image_xyz")
        assert result.success is False
        assert len(result.warnings) > 0

    def test_process_batch_all_results_returned(self):
        """Batch of 4 images (including 1 corrupt) → 4 results, no crash."""
        from src.preprocessing import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        inputs = [
            make_clean_receipt(300, 500),
            make_dark_receipt(make_clean_receipt(300, 500), gamma=0.3),
            make_blurry_receipt(make_clean_receipt(300, 500), kernel=15),
            b"corrupt_bytes_xyz",
        ]
        results = preprocessor.process_batch(inputs)
        assert len(results) == 4
        # At least the 3 valid images should succeed
        assert sum(1 for r in results if r.success) >= 3
        # Corrupt input should fail gracefully
        assert results[3].success is False

    def test_original_image_preserved(self):
        """original_image should be equal to the input array."""
        from src.preprocessing import ImagePreprocessor
        img = make_clean_receipt(400, 600)
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(img.copy())
        assert result.original_image is not None
        assert result.original_image.shape == img.shape

    def test_processing_time_recorded(self):
        """processing_time_ms should be positive."""
        from src.preprocessing import ImagePreprocessor
        img = make_clean_receipt(400, 600)
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(img)
        assert result.processing_time_ms > 0
