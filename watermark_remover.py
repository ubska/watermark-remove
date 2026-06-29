#!/usr/bin/env python3
"""Watermark Remover — CLI entry point.

Commands
--------
remove   Remove watermark from a single image.
batch    Process all images in a directory.
demo     Synthesise a watermarked image, remove it, and show metrics.
compare  Compare two images with PSNR / SSIM metrics.
"""

import sys
from pathlib import Path

import click
import numpy as np
import cv2
from tqdm import tqdm

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.detector import WatermarkDetector
from src.inpainter import Inpainter
from src.utils import load_image, save_image, print_metrics, show_comparison


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option("1.0.0", prog_name="watermark-remover")
def cli():
    """Remove watermarks from images without losing quality."""


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--method",
    "-m",
    type=click.Choice(["telea", "ns", "exemplar", "frequency"]),
    default="telea",
    show_default=True,
    help="Inpainting algorithm.",
)
@click.option(
    "--detect",
    "-d",
    type=click.Choice(["auto", "fft", "edge", "alpha", "manual"]),
    default="auto",
    show_default=True,
    help="Watermark detection strategy.",
)
@click.option(
    "--region",
    "-r",
    nargs=4,
    type=int,
    default=None,
    metavar="X Y W H",
    help="Manual bounding box (requires --detect manual).",
)
@click.option("--radius", default=3, show_default=True, help="Inpainting radius (pixels).")
@click.option("--compare/--no-compare", default=False, help="Save a side-by-side comparison PNG.")
def remove(input_path, output_path, method, detect, region, radius, compare):
    """Remove watermark from INPUT_PATH and write result to OUTPUT_PATH."""
    click.echo(f"Loading  : {input_path}")
    image = load_image(input_path)

    detector = WatermarkDetector()
    inpainter = Inpainter(inpaint_radius=radius)

    click.echo(f"Detecting: method={detect}")
    mask = detector.detect(image, method=detect, region=region if detect == "manual" else None)
    filled = mask.sum() // 255
    click.echo(f"  Mask pixels: {filled}")

    click.echo(f"Inpainting: method={method}")
    result = inpainter.inpaint(image, mask, method=method)

    save_image(result, output_path)
    click.echo(f"Saved    : {output_path}")

    click.echo("Quality metrics (vs. watermarked input):")
    print_metrics(image, result)

    if compare:
        cmp_path = str(Path(output_path).with_suffix("")) + "_comparison.png"
        show_comparison(image, mask, result, save_path=cmp_path)


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--method", "-m",
    type=click.Choice(["telea", "ns", "exemplar", "frequency"]),
    default="telea", show_default=True,
)
@click.option(
    "--detect", "-d",
    type=click.Choice(["auto", "fft", "edge", "alpha", "manual"]),
    default="auto", show_default=True,
)
@click.option("--ext", default=".png", show_default=True, help="Output file extension.")
@click.option("--radius", default=3, show_default=True)
def batch(input_dir, output_dir, method, detect, ext, radius):
    """Process all images in INPUT_DIR and write results to OUTPUT_DIR."""
    supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in supported]
    if not paths:
        click.echo("No supported images found.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    detector = WatermarkDetector()
    inpainter = Inpainter(inpaint_radius=radius)

    psnr_list, ssim_list = [], []
    for p in tqdm(paths, desc="Processing"):
        try:
            image = load_image(str(p))
            mask = detector.detect(image, method=detect)
            result = inpainter.inpaint(image, mask, method=method)
            out_path = Path(output_dir) / (p.stem + ext)
            save_image(result, str(out_path))
            psnr, ssim = print_metrics(image, result)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        except Exception as exc:
            click.echo(f"  ERROR {p.name}: {exc}", err=True)

    if psnr_list:
        click.echo(f"\nBatch summary ({len(psnr_list)} images):")
        click.echo(f"  Avg PSNR : {np.mean(psnr_list):.2f} dB")
        click.echo(f"  Avg SSIM : {np.mean(ssim_list):.4f}")


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--output", "-o", default="demo_result.png", show_default=True)
@click.option(
    "--method", "-m",
    type=click.Choice(["telea", "ns", "exemplar", "frequency"]),
    default="telea", show_default=True,
)
def demo(output, method):
    """Synthesise a watermarked test image, remove it, and show metrics."""
    click.echo("Generating synthetic test image …")
    # Create a gradient base image
    h, w = 256, 512
    base = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        base[i, :] = [int(i / h * 200), int((w - i) / w * 150 + 50), 100]

    # Overlay a text watermark
    wm = base.copy()
    cv2.putText(wm, "WATERMARK", (60, 140), cv2.FONT_HERSHEY_DUPLEX,
                2, (200, 200, 200), 3, cv2.LINE_AA)

    detector = WatermarkDetector()
    inpainter = Inpainter()

    mask = detector.detect(wm, method="edge")
    result = inpainter.inpaint(wm, mask, method=method)

    save_image(result, output)
    click.echo(f"Result saved to: {output}")

    click.echo("\nQuality metrics (watermarked vs. cleaned):")
    print_metrics(wm, result)

    cmp_path = str(Path(output).with_suffix("")) + "_comparison.png"
    show_comparison(wm, mask, result, title="Demo: Watermark Removal", save_path=cmp_path)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("image_a", type=click.Path(exists=True))
@click.argument("image_b", type=click.Path(exists=True))
def compare(image_a, image_b):
    """Compute PSNR and SSIM between IMAGE_A (reference) and IMAGE_B."""
    a = load_image(image_a)
    b = load_image(image_b)
    click.echo(f"Reference : {image_a}")
    click.echo(f"Comparison: {image_b}")
    print_metrics(a, b)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
