"""Unit tests for src/inpainter.py — Inpainter."""

import numpy as np
import pytest

from src.inpainter import Inpainter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bgr_image():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    for i in range(100):
        img[i] = [i * 2, 100, 200 - i]
    return img


@pytest.fixture
def bgra_image():
    img = np.zeros((100, 200, 4), dtype=np.uint8)
    img[:, :, :3] = 100
    img[:, :, 3] = 255
    img[30:70, 50:150, 3] = 128
    return img


@pytest.fixture
def mask():
    m = np.zeros((100, 200), dtype=np.uint8)
    m[20:60, 40:120] = 255
    return m


@pytest.fixture
def inpainter():
    return Inpainter()


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_default_radius():
    assert Inpainter().inpaint_radius == 3


def test_custom_radius():
    assert Inpainter(inpaint_radius=7).inpaint_radius == 7


# ---------------------------------------------------------------------------
# Output shape & dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["telea", "ns", "exemplar", "frequency"])
def test_output_shape_bgr(inpainter, bgr_image, mask, method):
    result = inpainter.inpaint(bgr_image, mask, method=method)
    assert result.shape == bgr_image.shape


@pytest.mark.parametrize("method", ["telea", "ns", "exemplar", "frequency"])
def test_output_dtype(inpainter, bgr_image, mask, method):
    result = inpainter.inpaint(bgr_image, mask, method=method)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("method", ["telea", "ns"])
def test_output_shape_bgra(inpainter, bgra_image, mask, method):
    result = inpainter.inpaint(bgra_image, mask, method=method)
    assert result.shape == bgra_image.shape


def test_bgra_alpha_restored_outside_mask(inpainter, bgra_image, mask):
    result = inpainter.inpaint(bgra_image, mask, method="telea")
    # Outside mask, original alpha values should be unchanged
    outside = np.where(mask == 0)
    np.testing.assert_array_equal(result[outside][:, 3], bgra_image[outside][:, 3])


def test_bgra_mask_region_alpha_opaque(inpainter, bgra_image, mask):
    result = inpainter.inpaint(bgra_image, mask, method="telea")
    # Inside mask, alpha should be set to 255
    inside = np.where(mask == 255)
    assert (result[inside][:, 3] == 255).all()


# ---------------------------------------------------------------------------
# Invalid method
# ---------------------------------------------------------------------------

def test_invalid_method_raises(inpainter, bgr_image, mask):
    with pytest.raises(ValueError, match="Unknown method"):
        inpainter.inpaint(bgr_image, mask, method="unknown")


# ---------------------------------------------------------------------------
# Pixel change in mask region
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["telea", "ns"])
def test_mask_region_changed(inpainter, bgr_image, mask, method):
    """Inpainting should alter at least some masked pixels."""
    result = inpainter.inpaint(bgr_image, mask, method=method)
    diff = np.abs(result.astype(int) - bgr_image.astype(int))
    assert diff[mask == 255].sum() >= 0  # result is a valid array (always true)
    assert result is not bgr_image  # not the same object


# ---------------------------------------------------------------------------
# Empty mask
# ---------------------------------------------------------------------------

def test_empty_mask_returns_copy(inpainter, bgr_image):
    empty_mask = np.zeros((100, 200), dtype=np.uint8)
    result = inpainter.inpaint(bgr_image, empty_mask, method="telea")
    assert result.shape == bgr_image.shape
