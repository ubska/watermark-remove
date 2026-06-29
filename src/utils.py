"""Image I/O, quality metrics (PSNR/SSIM), and comparative visualisation."""

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for scripts
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load an image from *path* and return it as a BGR(A) numpy array.

    Alpha channel is preserved when present (BGRA).
    """
    path = str(path)
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return image


def save_image(image: np.ndarray, path: str, jpeg_quality: int = 100) -> None:
    """Save *image* to *path* with lossless settings where possible.

    - PNG / TIFF: compression=0 (lossless, no quality loss)
    - JPEG:       quality=100   (maximum quality)
    - Others:     default OpenCV settings
    """
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext in (".png",):
        params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    elif ext in (".tif", ".tiff"):
        params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]  # 1 = no compression
    elif ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    else:
        params = []

    os.makedirs(Path(path).parent, exist_ok=True)
    success = cv2.imwrite(path, image, params)
    if not success:
        raise IOError(f"Failed to write image to: {path}")


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (higher = better, ∞ means identical)."""
    orig = _to_rgb_float(original)
    proc = _to_rgb_float(processed)
    if orig.shape != proc.shape:
        proc = cv2.resize(proc, (orig.shape[1], orig.shape[0]))
    return float(peak_signal_noise_ratio(orig, proc, data_range=1.0))


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Structural Similarity Index (1.0 = identical)."""
    orig = _to_rgb_float(original)
    proc = _to_rgb_float(processed)
    if orig.shape != proc.shape:
        proc = cv2.resize(proc, (orig.shape[1], orig.shape[0]))
    channel_axis = 2 if orig.ndim == 3 else None
    return float(
        structural_similarity(orig, proc, data_range=1.0, channel_axis=channel_axis)
    )


def print_metrics(original: np.ndarray, processed: np.ndarray) -> Tuple[float, float]:
    """Print PSNR and SSIM to stdout and return them as a tuple."""
    psnr = compute_psnr(original, processed)
    ssim = compute_ssim(original, processed)
    print(f"  PSNR : {psnr:.2f} dB")
    print(f"  SSIM : {ssim:.4f}")
    return psnr, ssim


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def show_comparison(
    original: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray,
    title: str = "Watermark Removal",
    save_path: Optional[str] = None,
) -> None:
    """Display (or save) a three-panel comparison: original | mask | result."""
    orig_rgb = _bgr_to_rgb(original)
    res_rgb = _bgr_to_rgb(result)
    mask_disp = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask_disp, cmap="gray")
    axes[1].set_title("Watermark Mask")
    axes[1].axis("off")

    axes[2].imshow(res_rgb)
    axes[2].set_title("Result")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Comparison saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _to_rgb_float(image: np.ndarray) -> np.ndarray:
    """Convert BGR(A) uint8 to RGB float32 in [0, 1]."""
    if image.shape[2] == 4:
        image = image[:, :, :3]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
