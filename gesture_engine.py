# ============================================================
# gesture_engine.py — Hand landmark analysis & gesture labels
# ============================================================
"""
Analyses a MediaPipe hand-landmark result and returns a high-level
GestureState describing what the hand is doing.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import config


# ── Landmark indices (MediaPipe convention) ─────────────────
WRIST       = 0
THUMB_TIP   = 4
INDEX_TIP   = 8
INDEX_MCP   = 5
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20
THUMB_IP    = 3
INDEX_PIP   = 6
MIDDLE_PIP  = 10
RING_PIP    = 14
PINKY_PIP   = 18


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _landmark_px(hand_landmarks, idx: int, w: int, h: int) -> Tuple[int, int]:
    lm = hand_landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)


@dataclass
class GestureState:
    # Raw fingertip positions (px)
    index_tip   : Tuple[int, int] = (0, 0)
    middle_tip  : Tuple[int, int] = (0, 0)
    thumb_tip   : Tuple[int, int] = (0, 0)
    ring_tip    : Tuple[int, int] = (0, 0)
    pinky_tip   : Tuple[int, int] = (0, 0)
    wrist       : Tuple[int, int] = (0, 0)

    # Fingers extended (True = up / extended)
    fingers_up  : List[bool] = field(default_factory=lambda: [False]*5)
    # [thumb, index, middle, ring, pinky]

    # High-level gesture labels
    gesture     : str  = "none"   # primary label
    is_drawing  : bool = False
    is_erasing  : bool = False
    is_pinching : bool = False
    is_fist     : bool = False
    is_palm     : bool = False
    is_pointing : bool = False
    is_peace    : bool = False     # index + middle up → next colour
    is_ok       : bool = False     # pinch OK sign
    is_gun      : bool = False     # index + thumb L-shape
    is_three    : bool = False     # three fingers for menu nav

    pinch_dist  : float = 0.0
    hand_height : float = 0.0      # bounding-box height in px (scale ref)

    # Smoothed drawing position (updated externally by Smoother)
    draw_pos    : Tuple[int, int] = (0, 0)


class GestureEngine:
    """Converts raw MediaPipe landmarks into a GestureState."""

    def analyse(
        self,
        hand_landmarks,
        frame_w: int,
        frame_h: int,
    ) -> GestureState:

        lm = hand_landmarks.landmark
        gs = GestureState()

        # ── Raw positions ───────────────────────────────────
        gs.index_tip  = _landmark_px(hand_landmarks, INDEX_TIP,  frame_w, frame_h)
        gs.middle_tip = _landmark_px(hand_landmarks, MIDDLE_TIP, frame_w, frame_h)
        gs.thumb_tip  = _landmark_px(hand_landmarks, THUMB_TIP,  frame_w, frame_h)
        gs.ring_tip   = _landmark_px(hand_landmarks, RING_TIP,   frame_w, frame_h)
        gs.pinky_tip  = _landmark_px(hand_landmarks, PINKY_TIP,  frame_w, frame_h)
        gs.wrist      = _landmark_px(hand_landmarks, WRIST,      frame_w, frame_h)

        # ── Hand bounding-box height ────────────────────────
        ys = [int(lm[i].y * frame_h) for i in range(21)]
        gs.hand_height = max(ys) - min(ys) + 1e-6

        # ── Fingers-up detection ────────────────────────────
        #  Thumb: compare tip x vs ip x (mirrored cam, right-hand assumption)
        thumb_up = lm[THUMB_TIP].x < lm[THUMB_IP].x

        def finger_up(tip_idx: int, pip_idx: int) -> bool:
            tip_y = lm[tip_idx].y
            pip_y = lm[pip_idx].y
            return (pip_y - tip_y) / gs.hand_height > config.FIST_FINGER_THRESHOLD

        index_up  = finger_up(INDEX_TIP,  INDEX_PIP)
        middle_up = finger_up(MIDDLE_TIP, MIDDLE_PIP)
        ring_up   = finger_up(RING_TIP,   RING_PIP)
        pinky_up  = finger_up(PINKY_TIP,  PINKY_PIP)

        gs.fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]

        # ── Pinch distance ──────────────────────────────────
        gs.pinch_dist = _dist(gs.index_tip, gs.thumb_tip)
        gs.is_pinching = gs.pinch_dist < config.PINCH_THRESHOLD

        # ── Composite gestures ──────────────────────────────
        n_up = sum(gs.fingers_up[1:])   # exclude thumb

        gs.is_fist    = n_up == 0 and not thumb_up
        gs.is_palm    = n_up >= 4
        gs.is_pointing = index_up and not middle_up and not ring_up and not pinky_up
        gs.is_peace   = index_up and middle_up and not ring_up and not pinky_up
        gs.is_three   = index_up and middle_up and ring_up and not pinky_up
        gs.is_ok      = gs.is_pinching and not middle_up and not ring_up and not pinky_up
        gs.is_gun     = index_up and thumb_up and not middle_up and not ring_up and not pinky_up

        # ── Primary label ───────────────────────────────────
        if gs.is_fist:
            gs.gesture = "fist"
        elif gs.is_palm:
            gs.gesture = "palm"
        elif gs.is_ok:
            gs.gesture = "ok"
        elif gs.is_peace:
            gs.gesture = "peace"
        elif gs.is_three:
            gs.gesture = "three"
        elif gs.is_gun:
            gs.gesture = "gun"
        elif gs.is_pointing:
            gs.gesture = "pointing"
        elif gs.is_pinching:
            gs.gesture = "pinch"
        else:
            gs.gesture = "open"

        # ── Drawing / erasing convenience flags ────────────
        gs.is_drawing = gs.is_pointing
        gs.is_erasing = gs.is_palm

        gs.draw_pos = gs.index_tip   # will be smoothed by caller
        return gs
