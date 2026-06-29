"""Watermark detection using FFT, edge density, alpha transparency, and manual region."""

import numpy as np
import cv2
from typing import Optional, Tuple


class WatermarkDetector:
    """Detect watermark regions in an image using multiple strategies."""

    def __init__(self, fft_threshold: float = 0.15, edge_threshold: float = 0.3):
        self.fft_threshold = fft_threshold
        self.edge_threshold = edge_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        method: str = "auto",
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Return a binary mask (255 = watermark, 0 = background).

        Parameters
        ----------
        image:  BGR or BGRA image as numpy array.
        method: 'auto' | 'fft' | 'edge' | 'alpha' | 'manual'
        region: (x, y, w, h) used when method='manual'
        """
        if method == "manual":
            if region is None:
                raise ValueError("region required for method='manual'")
            return self._manual_mask(image, region)
        if method == "fft":
            return self._fft_mask(image)
        if method == "edge":
            return self._edge_mask(image)
        if method == "alpha":
            return self._alpha_mask(image)
        # auto: try alpha first, then combine fft + edge
        if image.shape[2] == 4:
            mask = self._alpha_mask(image)
            if mask.any():
                return mask
        fft_mask = self._fft_mask(image)
        edge_mask = self._edge_mask(image)
        combined = cv2.bitwise_or(fft_mask, edge_mask)
        return self._clean_mask(combined)

    # ------------------------------------------------------------------
    # Detection strategies
    # ------------------------------------------------------------------

    def _fft_mask(self, image: np.ndarray) -> np.ndarray:
        """Detect periodic patterns (tiled watermarks) via FFT."""
        gray = self._to_gray(image)
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-9)

        # Suppress DC component at center
        h, w = magnitude_norm.shape
        cx, cy = w // 2, h // 2
        cv2.circle(magnitude_norm, (cx, cy), max(w, h) // 10, 0, -1)

        binary = (magnitude_norm > self.fft_threshold).astype(np.uint8) * 255
        # Back-project to spatial domain via inverse FFT of the peaks
        mask_freq = np.zeros_like(fshift, dtype=complex)
        mask_freq[binary == 255] = fshift[binary == 255]
        spatial = np.abs(np.fft.ifft2(np.fft.ifftshift(mask_freq)))
        spatial_norm = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-9)
        spatial_bin = (spatial_norm > self.fft_threshold).astype(np.uint8) * 255
        return self._clean_mask(spatial_bin)

    def _edge_mask(self, image: np.ndarray) -> np.ndarray:
        """Detect dense edge regions typical of text/logo watermarks."""
        gray = self._to_gray(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        # Compute local edge density with a sliding window
        kernel = np.ones((15, 15), np.float32) / (15 * 15)
        density = cv2.filter2D(edges.astype(np.float32) / 255.0, -1, kernel)
        threshold = self.edge_threshold * density.max()
        binary = (density > threshold).astype(np.uint8) * 255
        return self._clean_mask(binary)

    def _alpha_mask(self, image: np.ndarray) -> np.ndarray:
        """Use the alpha channel to find semi-transparent overlays."""
        if image.shape[2] < 4:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        alpha = image[:, :, 3]
        # Semi-transparent pixels (not fully opaque, not fully transparent)
        mask = np.zeros_like(alpha)
        mask[(alpha > 10) & (alpha < 245)] = 255
        return self._clean_mask(mask)

    def _manual_mask(
        self, image: np.ndarray, region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Build mask from a user-supplied bounding box (x, y, w, h)."""
        x, y, w, h = region
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y : y + h, x : x + w] = 255
        return mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.shape[2] == 4:
            bgr = image[:, :, :3]
        else:
            bgr = image
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """Morphological cleanup to remove noise."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
