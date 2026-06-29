# Watermark Remover

A Python CLI tool to remove watermarks from images without losing quality.  
Built for the **Computer Vision** course — demonstrates FFT analysis, inpainting algorithms, and image quality metrics.

---

## Features

- **Four detection strategies** — FFT, edge density, alpha transparency, manual region
- **Four inpainting algorithms** — Telea, Navier-Stokes, Exemplar, Frequency-domain
- **Lossless output** — PNG/TIFF saved with no compression; JPEG at quality 100
- **Quality metrics** — PSNR and SSIM printed after every operation
- **BGRA support** — alpha channel preserved and corrected
- **Batch processing** — entire directories in one command
- **29+ unit tests** with pytest

---

## Installation

```bash
pip install -r requirements.txt
```

---

## CLI Usage

### Remove a single image

```bash
python watermark_remover.py remove input.png output.png
python watermark_remover.py remove input.png output.png --method ns --detect edge
python watermark_remover.py remove input.png output.png --detect manual --region 100 50 300 80
python watermark_remover.py remove input.jpg output.tiff --method frequency --compare
```

### Batch processing

```bash
python watermark_remover.py batch ./watermarked/ ./cleaned/ --method telea --detect auto
python watermark_remover.py batch ./watermarked/ ./cleaned/ --ext .tiff
```

### Demo (synthetic image)

```bash
python watermark_remover.py demo
python watermark_remover.py demo --output result.png --method exemplar
```

### Compare two images

```bash
python watermark_remover.py compare original.png result.png
```

---

## Detection Methods

| Method   | How it works                                          | Best for                        |
|----------|-------------------------------------------------------|---------------------------------|
| `auto`   | Tries alpha first, then combines FFT + edge           | Unknown watermark type          |
| `fft`    | Detects periodic patterns via Fast Fourier Transform  | Tiled / repeating watermarks    |
| `edge`   | Finds dense edge regions (text, logos)                | Text-based watermarks           |
| `alpha`  | Reads semi-transparent pixels from alpha channel      | PNG/TIFF with transparency      |
| `manual` | User-supplied bounding box `(x y w h)`                | Precise region control          |

---

## Inpainting Algorithms

| Algorithm   | Description                                                                 | Speed  | Quality        |
|-------------|-----------------------------------------------------------------------------|--------|----------------|
| `telea`     | Fast Marching Method — propagates texture along level lines (Telea 2004)    | Fast   | Good           |
| `ns`        | Navier-Stokes fluid-dynamics — smoothly extends image curvature             | Medium | Good on smooth |
| `exemplar`  | Multi-scale patch synthesis — samples similar textures from nearby areas    | Slow   | Best on large  |
| `frequency` | FFT-domain suppression of periodic peaks, spatial refinement pass           | Medium | Best for tiled |

---

## Quality Metrics

After every operation the tool prints:

```
PSNR : 38.42 dB      ← higher is better (∞ = identical)
SSIM : 0.9731        ← 1.0 = identical, higher is better
```

- **PSNR** (Peak Signal-to-Noise Ratio) — pixel-level fidelity
- **SSIM** (Structural Similarity Index) — perceptual quality

---

## Project Structure

```
watermark-remove/
├── watermark_remover.py   # CLI (remove, batch, demo, compare)
├── src/
│   ├── detector.py        # Watermark detection
│   ├── inpainter.py       # Inpainting algorithms
│   └── utils.py           # I/O, metrics, visualisation
├── tests/
│   ├── test_detector.py
│   ├── test_inpainter.py
│   └── test_utils.py
├── examples/
│   └── example_usage.py
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Example: Library API

```python
from src.detector import WatermarkDetector
from src.inpainter import Inpainter
from src.utils import load_image, save_image, compute_psnr, compute_ssim

image = load_image("watermarked.png")

detector = WatermarkDetector()
mask = detector.detect(image, method="edge")

inpainter = Inpainter(inpaint_radius=5)
result = inpainter.inpaint(image, mask, method="telea")

save_image(result, "cleaned.png")  # lossless PNG
psnr = compute_psnr(image, result)
ssim = compute_ssim(image, result)
print(f"PSNR: {psnr:.2f} dB  SSIM: {ssim:.4f}")
```

---

## Algorithm Details

### FFT Detection
Applies a 2-D Fast Fourier Transform to the grayscale image. Watermark
patterns that repeat spatially produce distinctive spikes in the frequency
domain. The algorithm isolates those spikes, back-projects them to the spatial
domain, and thresholds the result to produce a binary mask.

### Edge Density Detection
Applies Canny edge detection and computes the local edge density via a sliding-
window convolution. Regions with density above the threshold (dense edges = text
or logos) are flagged as watermark.

### Alpha Transparency Detection
Reads the alpha channel of a BGRA image. Pixels that are neither fully opaque
(255) nor fully transparent (0) are assumed to belong to a semi-transparent
watermark overlay.

### Telea Inpainting (Fast Marching)
Propagates texture information from the mask boundary inward using the Fast
Marching Method. Excellent at preserving edges and works well for small regions.

### Navier-Stokes Inpainting
Models the image as a fluid and diffuses pixel values according to fluid
dynamics equations. Produces smooth results and handles curved edges well.

### Exemplar Inpainting (Patch Synthesis)
Performs multi-scale inpainting: downscales, fills with Telea, upscales back,
and refines with NS. Samples texture from surrounding areas for coherence.
Works best on large watermark regions in textured backgrounds.

### Frequency Inpainting
Transforms each channel with FFT, suppresses frequency spikes correlated with
the watermark, reconstructs via inverse FFT, then applies a final spatial
Telea pass to remove remaining block artifacts. Most effective against periodic
or tiled watermarks.
