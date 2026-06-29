"""Full demonstration of the watermark-remover library.

Run from the repository root:
    python examples/example_usage.py
"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import WatermarkDetector
from src.inpainter import Inpainter
from src.utils import load_image, save_image, compute_psnr, compute_ssim, show_comparison


# ---------------------------------------------------------------------------
# Helper: create a synthetic watermarked image
# ---------------------------------------------------------------------------

def create_test_image(path: str = "examples/test_watermarked.png") -> np.ndarray:
    h, w = 300, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Gradient background
    for i in range(h):
        img[i] = [int(i / h * 180 + 40), int(80 + i / h * 80), int(200 - i / h * 100)]
    # Watermark text
    cv2.putText(img, "© WATERMARK", (80, 170), cv2.FONT_HERSHEY_DUPLEX,
                1.8, (220, 220, 220), 2, cv2.LINE_AA)
    save_image(img, path)
    print(f"[demo] Saved test image: {path}")
    return img


# ---------------------------------------------------------------------------
# Example 1: Auto detection + Telea inpainting
# ---------------------------------------------------------------------------

def example_auto_telea():
    print("\n=== Example 1: Auto detection + Telea ===")
    img = create_test_image()
    detector = WatermarkDetector()
    inpainter = Inpainter(inpaint_radius=5)

    mask = detector.detect(img, method="auto")
    result = inpainter.inpaint(img, mask, method="telea")
    save_image(result, "examples/result_telea.png")

    psnr = compute_psnr(img, result)
    ssim = compute_ssim(img, result)
    print(f"  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")
    show_comparison(img, mask, result, title="Telea", save_path="examples/compare_telea.png")


# ---------------------------------------------------------------------------
# Example 2: Edge detection + NS inpainting
# ---------------------------------------------------------------------------

def example_edge_ns():
    print("\n=== Example 2: Edge detection + NS ===")
    img = create_test_image()
    detector = WatermarkDetector()
    inpainter = Inpainter()

    mask = detector.detect(img, method="edge")
    result = inpainter.inpaint(img, mask, method="ns")
    save_image(result, "examples/result_ns.png")

    psnr = compute_psnr(img, result)
    ssim = compute_ssim(img, result)
    print(f"  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")
    show_comparison(img, mask, result, title="NS", save_path="examples/compare_ns.png")


# ---------------------------------------------------------------------------
# Example 3: Manual region + Exemplar inpainting
# ---------------------------------------------------------------------------

def example_manual_exemplar():
    print("\n=== Example 3: Manual region + Exemplar ===")
    img = create_test_image()
    detector = WatermarkDetector()
    inpainter = Inpainter()

    # Approximate bounding box of the watermark text
    mask = detector.detect(img, method="manual", region=(70, 130, 470, 70))
    result = inpainter.inpaint(img, mask, method="exemplar")
    save_image(result, "examples/result_exemplar.png")

    psnr = compute_psnr(img, result)
    ssim = compute_ssim(img, result)
    print(f"  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")
    show_comparison(img, mask, result, title="Exemplar", save_path="examples/compare_exemplar.png")


# ---------------------------------------------------------------------------
# Example 4: FFT detection + Frequency inpainting (periodic watermarks)
# ---------------------------------------------------------------------------

def example_fft_frequency():
    print("\n=== Example 4: FFT detection + Frequency ===")
    img = create_test_image()
    detector = WatermarkDetector()
    inpainter = Inpainter()

    mask = detector.detect(img, method="fft")
    result = inpainter.inpaint(img, mask, method="frequency")
    save_image(result, "examples/result_frequency.png")

    psnr = compute_psnr(img, result)
    ssim = compute_ssim(img, result)
    print(f"  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")
    show_comparison(img, mask, result, title="Frequency", save_path="examples/compare_frequency.png")


# ---------------------------------------------------------------------------
# Example 5: BGRA image with alpha watermark
# ---------------------------------------------------------------------------

def example_alpha_channel():
    print("\n=== Example 5: Alpha-channel watermark ===")
    h, w = 200, 400
    base = np.full((h, w, 4), [100, 150, 200, 255], dtype=np.uint8)
    # Semi-transparent overlay (alpha=128 = half-transparent watermark)
    base[50:150, 100:300, 3] = 128
    cv2.putText(base[:, :, :3], "ALPHA WM", (110, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    save_image(base, "examples/test_alpha.png")

    detector = WatermarkDetector()
    inpainter = Inpainter()

    mask = detector.detect(base, method="alpha")
    result = inpainter.inpaint(base, mask, method="telea")
    save_image(result, "examples/result_alpha.png")

    psnr = compute_psnr(base[:, :, :3], result[:, :, :3])
    ssim = compute_ssim(base[:, :, :3], result[:, :, :3])
    print(f"  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Path("examples").mkdir(exist_ok=True)
    example_auto_telea()
    example_edge_ns()
    example_manual_exemplar()
    example_fft_frequency()
    example_alpha_channel()
    print("\nAll examples completed. Check the examples/ folder.")
