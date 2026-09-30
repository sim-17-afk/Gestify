# ============================================================
# ghost_mode.py — Hand-Controlled Invisibility / Ghost Effect
# ============================================================
"""
Ghost Mode works in two phases:
  1. CALIBRATE — capture a clean background frame while no hand is present
  2. ACTIVE    — detect the hand bounding box; replace that region with the
                 background (makes the hand 'invisible') and apply chromatic
                 aberration + edge glitch for a sci-fi invisibility look.

Toggle with the 'G' key or the 'fist-for-3-seconds' trigger.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple
import config


class GhostMode:
    """Invisibility-cloak effect driven by hand region."""

    def __init__(self, width: int, height: int):
        self.width   = width
        self.height  = height
        self.active  = False

        self._bg     : Optional[np.ndarray] = None   # captured background
        self._fgmask = cv2.createBackgroundSubtractorMOG2(
            history   = config.GHOST_HISTORY,
            varThreshold = config.GHOST_THRESHOLD,
            detectShadows= False,
        )

    # ── Public API ──────────────────────────────────────────

    def toggle(self) -> bool:
        self.active = not self.active
        return self.active

    def capture_background(self, frame: np.ndarray) -> None:
        """Store a clean background reference (call when hand is absent)."""
        self._bg = frame.copy()

    def apply(
        self,
        frame   : np.ndarray,
        bbox    : Optional[Tuple[int,int,int,int]],   # x,y,w,h hand bounding box
    ) -> np.ndarray:
        """
        Apply the ghost effect to `frame`.
        `bbox` is the hand region detected by gesture engine.
        Returns the processed frame (BGR).
        """
        if not self.active or self._bg is None:
            return frame

        out = frame.copy()

        if bbox is not None:
            x, y, w, h = bbox
            # Expand bbox slightly for a cleaner cloak
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(self.width,  x + w + pad)
            y2 = min(self.height, y + h + pad)

            # ── Foreground mask via MOG2 ────────────────────
            fg = self._fgmask.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel)
            fg_roi  = fg[y1:y2, x1:x2]
            mask3   = cv2.cvtColor(fg_roi, cv2.COLOR_GRAY2BGR).astype(np.float32)/255.0

            # ── Replace hand region with background ─────────
            bg_roi  = self._bg[y1:y2, x1:x2].astype(np.float32)
            out_roi = out[y1:y2, x1:x2].astype(np.float32)
            blended = out_roi * (1 - mask3) + bg_roi * mask3
            out[y1:y2, x1:x2] = blended.astype(np.uint8)

        # ── Chromatic aberration overlay ───────────────────
        out = self._chromatic_aberration(out)

        # ── Subtle scan-line / glitch vignette ─────────────
        out = self._scanline_vignette(out)

        return out

    # ── Visual FX helpers ───────────────────────────────────

    @staticmethod
    def _chromatic_aberration(frame: np.ndarray, shift: int = 3) -> np.ndarray:
        """Slightly shift the red and blue channels horizontally."""
        b, g, r = cv2.split(frame)
        h, w = frame.shape[:2]
        M_r = np.float32([[1, 0,  shift], [0, 1, 0]])
        M_b = np.float32([[1, 0, -shift], [0, 1, 0]])
        r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)
        return cv2.merge([b, g, r])

    @staticmethod
    def _scanline_vignette(frame: np.ndarray) -> np.ndarray:
        """Add subtle dark scan-line stripes for sci-fi feel."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for y in range(0, h, 6):
            cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
        return cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
