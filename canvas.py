# ============================================================
# canvas.py — Air-canvas drawing / erasing layer
# ============================================================
"""
Manages a persistent BGRA drawing layer that sits on top of the camera feed.
Supports: drawing strokes, erasing, colour switching, undo/redo, and save.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple
import config


class AirCanvas:
    """Transparent drawing surface that overlays the live camera feed."""

    UNDO_LIMIT = 20

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height

        # BGRA canvas — alpha=0 means transparent
        self._layer : np.ndarray = np.zeros((height, width, 4), dtype=np.uint8)
        self._history: list[np.ndarray] = []   # undo stack
        self._redo   : list[np.ndarray] = []   # redo stack

        # Current tool state
        self.color           : Tuple[int,int,int] = config.COLORS[config.DEFAULT_COLOR]
        self.color_name      : str               = config.DEFAULT_COLOR
        self.brush_thickness : int               = config.DEFAULT_BRUSH_THICKNESS
        self.eraser_thickness: int               = config.ERASER_THICKNESS

        # Previous draw position (for stroke continuity)
        self._prev_pos: Optional[Tuple[int,int]] = None

    # ── Public API ──────────────────────────────────────────

    def draw_stroke(self, pos: Tuple[int,int]) -> None:
        """Draw a continuous stroke to pos.  Call per-frame while drawing."""
        if self._prev_pos is None:
            self._prev_pos = pos
            return
        if self._dist(pos, self._prev_pos) < config.MIN_DRAW_DISTANCE:
            return
        self._save_history()
        bgra = (*self.color, 255)
        cv2.line(self._layer, self._prev_pos, pos, bgra, self.brush_thickness,
                 lineType=cv2.LINE_AA)
        self._prev_pos = pos

    def erase_stroke(self, pos: Tuple[int,int]) -> None:
        """Erase a circular region around pos."""
        if self._prev_pos is None:
            self._prev_pos = pos
            return
        self._save_history()
        r = self.eraser_thickness // 2
        cv2.circle(self._layer, pos, r, (0, 0, 0, 0), -1)
        self._prev_pos = pos

    def lift_pen(self) -> None:
        """Call when the drawing gesture ends (finger lifted)."""
        self._prev_pos = None

    def clear(self) -> None:
        self._save_history()
        self._layer[:] = 0

    def set_color(self, name: str) -> None:
        if name in config.COLORS:
            self.color_name = name
            self.color      = config.COLORS[name]
            self._prev_pos  = None          # avoid ghost lines on colour switch

    def set_brush_size(self, size: int) -> None:
        self.brush_thickness = max(config.MIN_BRUSH,
                                   min(config.MAX_BRUSH, size))

    def undo(self) -> None:
        if self._history:
            self._redo.append(self._layer.copy())
            self._layer = self._history.pop()
            self._prev_pos = None

    def redo(self) -> None:
        if self._redo:
            self._history.append(self._layer.copy())
            self._layer = self._redo.pop()
            self._prev_pos = None

    def save(self, path: str = "canvas_output.png") -> str:
        """Save the canvas (transparent PNG)."""
        cv2.imwrite(path, self._layer)
        return path

    def composite_onto(self, frame: np.ndarray) -> np.ndarray:
        """
        Alpha-blend the canvas layer onto a BGR frame (in-place).
        Returns the modified frame.
        """
        alpha = self._layer[:, :, 3:4].astype(np.float32) / 255.0
        canvas_rgb = self._layer[:, :, :3].astype(np.float32)
        frame_f    = frame.astype(np.float32)
        frame_f    = frame_f * (1 - alpha) + canvas_rgb * alpha
        np.clip(frame_f, 0, 255, out=frame_f)
        frame[:] = frame_f.astype(np.uint8)
        return frame

    # ── Internals ───────────────────────────────────────────

    def _save_history(self) -> None:
        if len(self._history) >= self.UNDO_LIMIT:
            self._history.pop(0)
        self._history.append(self._layer.copy())
        self._redo.clear()

    @staticmethod
    def _dist(a: Tuple[int,int], b: Tuple[int,int]) -> float:
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
