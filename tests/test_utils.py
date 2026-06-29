"""Unit tests for src/utils.py — I/O, metrics, visualisation."""

import os
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils import (
    load_image,
    save_image,
    compute_psnr,
    compute_ssim,
    print_metrics,
    show_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bgr():
    img = np.zeros((80, 160, 3), dtype=np.uint8)
    for i in range(80):
        img[i] = [i * 3, 100, 200 - i * 2]
    return img


@pytest.fixture
def noisy_bgr(sample_bgr):
    noise = np.random.randint(-20, 20, sample_bgr.shape, dtype=np.int16)
    return np.clip(sample_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# save_image / load_image
# ---------------------------------------------------------------------------

def test_save_load_png(sample_bgr, tmp_dir):
    path = os.path.join(tmp_dir, "test.png")
    save_image(sample_bgr, path)
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded, sample_bgr)


def test_save_load_tiff(sample_bgr, tmp_dir):
    path = os.path.join(tmp_dir, "test.tiff")
    save_image(sample_bgr, path)
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded, sample_bgr)


def test_save_jpeg_creates_file(sample_bgr, tmp_dir):
    path = os.path.join(tmp_dir, "test.jpg")
    save_image(sample_bgr, path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/path/image.png")


def test_save_creates_parent_dir(sample_bgr, tmp_dir):
    path = os.path.join(tmp_dir, "subdir", "nested", "out.png")
    save_image(sample_bgr, path)
    assert os.path.exists(path)


def test_save_load_bgra(tmp_dir):
    img = np.zeros((50, 100, 4), dtype=np.uint8)
    img[:, :, 3] = 128
    path = os.path.join(tmp_dir, "alpha.png")
    save_image(img, path)
    loaded = load_image(path)
    assert loaded.shape[2] == 4
    np.testing.assert_array_equal(loaded[:, :, 3], img[:, :, 3])


# ---------------------------------------------------------------------------
# compute_psnr
# ---------------------------------------------------------------------------

def test_psnr_identical_images_is_inf(sample_bgr):
    psnr = compute_psnr(sample_bgr, sample_bgr.copy())
    assert math.isinf(psnr) or psnr > 100


def test_psnr_noisy_is_finite(sample_bgr, noisy_bgr):
    psnr = compute_psnr(sample_bgr, noisy_bgr)
    assert math.isfinite(psnr)
    assert psnr > 0


def test_psnr_higher_for_less_noise(sample_bgr):
    slight_noise = sample_bgr.copy().astype(np.int16)
    slight_noise += 2
    slight = np.clip(slight_noise, 0, 255).astype(np.uint8)

    heavy_noise = sample_bgr.copy().astype(np.int16)
    heavy_noise += 50
    heavy = np.clip(heavy_noise, 0, 255).astype(np.uint8)

    assert compute_psnr(sample_bgr, slight) > compute_psnr(sample_bgr, heavy)


def test_psnr_returns_float(sample_bgr, noisy_bgr):
    assert isinstance(compute_psnr(sample_bgr, noisy_bgr), float)


# ---------------------------------------------------------------------------
# compute_ssim
# ---------------------------------------------------------------------------

def test_ssim_identical_is_one(sample_bgr):
    ssim = compute_ssim(sample_bgr, sample_bgr.copy())
    assert abs(ssim - 1.0) < 1e-6


def test_ssim_noisy_less_than_one(sample_bgr, noisy_bgr):
    ssim = compute_ssim(sample_bgr, noisy_bgr)
    assert ssim < 1.0


def test_ssim_in_valid_range(sample_bgr, noisy_bgr):
    ssim = compute_ssim(sample_bgr, noisy_bgr)
    assert -1.0 <= ssim <= 1.0


def test_ssim_returns_float(sample_bgr, noisy_bgr):
    assert isinstance(compute_ssim(sample_bgr, noisy_bgr), float)


# ---------------------------------------------------------------------------
# print_metrics
# ---------------------------------------------------------------------------

def test_print_metrics_returns_tuple(sample_bgr, noisy_bgr, capsys):
    result = print_metrics(sample_bgr, noisy_bgr)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_print_metrics_output_contains_psnr(sample_bgr, noisy_bgr, capsys):
    print_metrics(sample_bgr, noisy_bgr)
    out = capsys.readouterr().out
    assert "PSNR" in out


def test_print_metrics_output_contains_ssim(sample_bgr, noisy_bgr, capsys):
    print_metrics(sample_bgr, noisy_bgr)
    out = capsys.readouterr().out
    assert "SSIM" in out


# ---------------------------------------------------------------------------
# show_comparison
# ---------------------------------------------------------------------------

def test_show_comparison_saves_file(sample_bgr, tmp_dir):
    mask = np.zeros(sample_bgr.shape[:2], dtype=np.uint8)
    mask[10:40, 20:80] = 255
    save_path = os.path.join(tmp_dir, "comparison.png")
    show_comparison(sample_bgr, mask, sample_bgr, save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
