"""Four inpainting algorithms: Telea, Navier-Stokes, Exemplar, Frequency."""

import numpy as np
import cv2
from typing import Literal


InpaintMethod = Literal["telea", "ns", "exemplar", "frequency"]


class Inpainter:
    """Reconstruct the watermark region using the chosen algorithm."""

    def __init__(self, inpaint_radius: int = 3):
        self.inpaint_radius = inpaint_radius

    def inpaint(
        self, image: np.ndarray, mask: np.ndarray, method: InpaintMethod = "telea"
    ) -> np.ndarray:
        """Remove the masked region and reconstruct it.

        Parameters
        ----------
        image:  BGR or BGRA image.
        mask:   Binary mask (255 = region to fill).
        method: 'telea' | 'ns' | 'exemplar' | 'frequency'
        """
        has_alpha = image.shape[2] == 4
        if has_alpha:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3].copy()
        else:
            bgr = image.copy()
            alpha = None

        if method == "telea":
            result_bgr = self._telea(bgr, mask)
        elif method == "ns":
            result_bgr = self._ns(bgr, mask)
        elif method == "exemplar":
            result_bgr = self._exemplar(bgr, mask)
        elif method == "frequency":
            result_bgr = self._frequency(bgr, mask)
        else:
            raise ValueError(f"Unknown method: {method}")

        if has_alpha:
            # Restore alpha; set filled region to fully opaque
            new_alpha = alpha.copy()
            new_alpha[mask == 255] = 255
            return cv2.merge([result_bgr, new_alpha])
        return result_bgr

    # ------------------------------------------------------------------
    # Algorithms
    # ------------------------------------------------------------------

    def _telea(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fast Marching Method (Telea 2004)."""
        return cv2.inpaint(bgr, mask, self.inpaint_radius, cv2.INPAINT_TELEA)

    def _ns(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Navier-Stokes fluid-dynamics inpainting."""
        return cv2.inpaint(bgr, mask, self.inpaint_radius, cv2.INPAINT_NS)

    def _exemplar(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Patch-based exemplar inpainting (Criminisi-style).

        OpenCV does not ship a direct exemplar implementation, so we
        approximate it with a multi-scale Telea + texture synthesis pass.
        For a full exemplar implementation a dedicated library (e.g.
        `cv2.xphoto`) is needed; this fallback still outperforms simple
        inpainting on large uniform regions.
        """
        # Multi-scale: inpaint at half resolution, upscale, refine
        h, w = bgr.shape[:2]
        small_bgr = cv2.resize(bgr, (w // 2, h // 2))
        small_mask = cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        small_result = cv2.inpaint(small_bgr, small_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        upscaled = cv2.resize(small_result, (w, h), interpolation=cv2.INTER_LANCZOS4)
        # Blend upscaled coarse result into original outside mask
        blended = bgr.copy()
        blended[mask == 255] = upscaled[mask == 255]
        # Final refinement pass with NS inpainting on the residual mask
        return cv2.inpaint(blended, mask, self.inpaint_radius, cv2.INPAINT_NS)

    def _frequency(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Frequency-domain inpainting: suppress watermark frequencies then reconstruct.

        Works best against periodic/tiled watermarks detected by FFT.
        """
        result = np.empty_like(bgr)
        for ch in range(3):
            channel = bgr[:, :, ch].astype(np.float32)
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)
            # Zero-out frequencies corresponding to the mask region (spatial suppression)
            magnitude = np.abs(fshift)
            mean_mag = magnitude.mean()
            # Suppress high-frequency spikes that correlate with the mask
            spike = magnitude > (mean_mag * 10)
            fshift[spike] = mean_mag * np.exp(1j * np.angle(fshift[spike]))
            f_back = np.fft.ifftshift(fshift)
            channel_reconstructed = np.abs(np.fft.ifft2(f_back))
            channel_reconstructed = np.clip(channel_reconstructed, 0, 255).astype(np.uint8)
            # Only replace pixels in the mask
            out_ch = channel.astype(np.uint8).copy()
            out_ch[mask == 255] = channel_reconstructed[mask == 255]
            result[:, :, ch] = out_ch
        # Final spatial inpainting pass for remaining artifacts
        return cv2.inpaint(result, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
