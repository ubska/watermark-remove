"""Unit tests for src/detector.py — WatermarkDetector."""

import numpy as np
import pytest

from src.detector import WatermarkDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bgr_image():
    """Plain 100×200 BGR image (gradient)."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    for i in range(100):
        img[i] = [i * 2, 100, 200 - i]
    return img


@pytest.fixture
def bgra_image():
    """100×200 BGRA image with a semi-transparent band."""
    img = np.zeros((100, 200, 4), dtype=np.uint8)
    img[:, :, :3] = 128
    img[:, :, 3] = 255
    img[30:70, 50:150, 3] = 128  # semi-transparent watermark band
    return img


@pytest.fixture
def detector():
    return WatermarkDetector()


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_detector_default_thresholds():
    d = WatermarkDetector()
    assert d.fft_threshold == 0.15
    assert d.edge_threshold == 0.3


def test_detector_custom_thresholds():
    d = WatermarkDetector(fft_threshold=0.2, edge_threshold=0.5)
    assert d.fft_threshold == 0.2
    assert d.edge_threshold == 0.5


# ---------------------------------------------------------------------------
# Return type / shape
# ---------------------------------------------------------------------------

def test_detect_returns_2d_mask(detector, bgr_image):
    mask = detector.detect(bgr_image)
    assert mask.ndim == 2
    assert mask.shape == bgr_image.shape[:2]


def test_detect_mask_dtype(detector, bgr_image):
    mask = detector.detect(bgr_image)
    assert mask.dtype == np.uint8


def test_detect_mask_binary_values(detector, bgr_image):
    mask = detector.detect(bgr_image)
    unique = set(np.unique(mask).tolist())
    assert unique.issubset({0, 255})


# ---------------------------------------------------------------------------
# Method: fft
# ---------------------------------------------------------------------------

def test_fft_mask_shape(detector, bgr_image):
    mask = detector.detect(bgr_image, method="fft")
    assert mask.shape == bgr_image.shape[:2]


def test_fft_mask_binary(detector, bgr_image):
    mask = detector.detect(bgr_image, method="fft")
    assert set(np.unique(mask).tolist()).issubset({0, 255})


# ---------------------------------------------------------------------------
# Method: edge
# ---------------------------------------------------------------------------

def test_edge_mask_shape(detector, bgr_image):
    mask = detector.detect(bgr_image, method="edge")
    assert mask.shape == bgr_image.shape[:2]


def test_edge_mask_binary(detector, bgr_image):
    mask = detector.detect(bgr_image, method="edge")
    assert set(np.unique(mask).tolist()).issubset({0, 255})


# ---------------------------------------------------------------------------
# Method: alpha
# ---------------------------------------------------------------------------

def test_alpha_mask_detects_semitransparent(detector, bgra_image):
    mask = detector.detect(bgra_image, method="alpha")
    # The region [30:70, 50:150] should have some mask pixels
    roi = mask[30:70, 50:150]
    assert roi.max() == 255


def test_alpha_mask_on_bgr_returns_zeros(detector, bgr_image):
    mask = detector.detect(bgr_image, method="alpha")
    assert mask.sum() == 0


# ---------------------------------------------------------------------------
# Method: manual
# ---------------------------------------------------------------------------

def test_manual_mask_correct_region(detector, bgr_image):
    mask = detector.detect(bgr_image, method="manual", region=(10, 20, 50, 30))
    # Interior of the bounding box must be 255
    assert mask[20, 10] == 255
    assert mask[49, 59] == 255


def test_manual_mask_outside_region(detector, bgr_image):
    mask = detector.detect(bgr_image, method="manual", region=(10, 20, 50, 30))
    assert mask[0, 0] == 0


def test_manual_raises_without_region(detector, bgr_image):
    with pytest.raises(ValueError, match="region required"):
        detector.detect(bgr_image, method="manual")


# ---------------------------------------------------------------------------
# Method: auto
# ---------------------------------------------------------------------------

def test_auto_on_bgra_uses_alpha(detector, bgra_image):
    mask = detector.detect(bgra_image, method="auto")
    assert mask.shape == bgra_image.shape[:2]


def test_auto_on_bgr_returns_mask(detector, bgr_image):
    mask = detector.detect(bgr_image, method="auto")
    assert mask.shape == bgr_image.shape[:2]
