# ============================================================
# smoother.py — Exponential-moving-average position smoother
# ============================================================
"""
Reduces jitter in fingertip position while keeping latency low.
"""

from __future__ import annotations
from typing import Optional, Tuple
import config


class PositionSmoother:
    """
    EMA smoother.  alpha ∈ (0, 1]:
      alpha → 1 : no smoothing (raw position)
      alpha → 0 : very heavy smoothing (lags behind)
    """

    def __init__(self, alpha: float = config.SMOOTHING_ALPHA):
        self._alpha   = alpha
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None

    def update(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        x, y = pos
        if self._sx is None:
            self._sx, self._sy = float(x), float(y)
        else:
            self._sx = self._alpha * x + (1 - self._alpha) * self._sx
            self._sy = self._alpha * y + (1 - self._alpha) * self._sy
        return int(self._sx), int(self._sy)

    def reset(self) -> None:
        self._sx = self._sy = None
