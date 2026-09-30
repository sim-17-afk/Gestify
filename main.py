# ============================================================
# main.py — Gesture Suite Entry Point
# ============================================================
"""
Gestify — Hand Gesture Control Suite
─────────────────────────────────────
Run:
    python main.py

Dependencies:
    pip install opencv-python mediapipe numpy Pillow

Features:
  • Air Canvas — draw with index finger
  • Gesture-based color switching (✌ peace = next color)
  • Palm eraser
  • Ghost Mode (invisibility cloak effect)
  • Gesture-controlled side menu (三 three-fingers)
  • Real-time hand tracking HUD
"""

from __future__ import annotations
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys
from typing import Optional, Tuple

import config
from gesture_engine import GestureEngine, GestureState
from canvas        import AirCanvas
from ghost_mode    import GhostMode
from smoother      import PositionSmoother
from ui_renderer   import UIRenderer


# ── MediaPipe setup ─────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles


# ── Helpers ─────────────────────────────────────────────────

def hand_bbox(hand_landmarks, w: int, h: int) -> Tuple[int,int,int,int]:
    """Return (x, y, width, height) bounding box of the hand."""
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return x1, y1, x2-x1, y2-y1


def color_name_cycle(current: str, direction: int = 1) -> str:
    names = list(config.COLORS.keys())
    idx   = names.index(current) if current in names else 0
    return names[(idx + direction) % len(names)]


# ── App state ────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.mode          : str   = "DRAW"   # DRAW | ERASE | IDLE
        self.menu_open     : bool  = False
        self.menu_sel      : int   = 0
        self.is_ghost      : bool  = False
        self.fps           : float = 0.0
        self.gesture       : str   = "none"
        self.draw_pos      : Tuple[int,int] = (0, 0)

        # Gesture debounce / hold detection
        self._last_gesture  : str   = "none"
        self._gesture_start : float = 0.0
        self._peace_last    : float = 0.0   # colour-switch debounce
        self._menu_nav_last : float = 0.0

        # Fist hold → ghost toggle
        self._fist_hold_start: Optional[float] = None
        FIST_HOLD_SECS = 1.8

        self.FIST_HOLD_SECS = FIST_HOLD_SECS

    def update_gesture(self, gest: str) -> None:
        now = time.time()
        if gest != self._last_gesture:
            self._gesture_start = now
            self._last_gesture  = gest
        self.gesture = gest

    def fist_held(self) -> bool:
        """Returns True once when fist is held for FIST_HOLD_SECS."""
        now = time.time()
        if self.gesture == "fist":
            if self._fist_hold_start is None:
                self._fist_hold_start = now
            elif now - self._fist_hold_start >= self.FIST_HOLD_SECS:
                self._fist_hold_start = None   # reset so it triggers once
                return True
        else:
            self._fist_hold_start = None
        return False

    def colour_switch_ready(self, debounce: float = 1.0) -> bool:
        now = time.time()
        if now - self._peace_last >= debounce:
            self._peace_last = now
            return True
        return False

    def menu_nav_ready(self, debounce: float = 0.6) -> bool:
        now = time.time()
        if now - self._menu_nav_last >= debounce:
            self._menu_nav_last = now
            return True
        return False


# ── Main loop ────────────────────────────────────────────────

def run() -> None:
    # ── Camera init ────────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.py")
        sys.exit(1)

    # Read actual resolution (camera may cap it)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {W}×{H}")

    # ── Sub-system init ────────────────────────────────────
    engine   = GestureEngine()
    canvas   = AirCanvas(W, H)
    ghost    = GhostMode(W, H)
    smoother = PositionSmoother()
    ui       = UIRenderer(W, H)
    state    = AppState()

    hands = mp_hands.Hands(
        max_num_hands          = config.MP_MAX_HANDS,
        min_detection_confidence = config.MP_DETECTION_CONFIDENCE,
        min_tracking_confidence  = config.MP_TRACKING_CONFIDENCE,
    )

    cv2.namedWindow("Gestify", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gestify", W, H)

    # ── FPS tracking ───────────────────────────────────────
    fps_t0     = time.time()
    fps_frames = 0

    print("[INFO] Starting Gestify — press H for help, Q to quit.")

    was_drawing = False
    was_erasing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame not received — retrying…")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)   # mirror so left/right feel natural

        # ── Ghost background capture ────────────────────────
        # Capture bg for a few frames at startup (no hand present)
        if fps_frames < 30:
            ghost.capture_background(frame)

        # ── FPS ────────────────────────────────────────────
        fps_frames += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 0.5:
            state.fps  = fps_frames / elapsed
            fps_frames = 0
            fps_t0     = time.time()

        # ── MediaPipe hand detection ────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        gs: Optional[GestureState] = None
        bbox: Optional[Tuple[int,int,int,int]] = None

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]

            # Draw skeleton (subtle)
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,80,120), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(60,60,100), thickness=1),
            )

            bbox = hand_bbox(hand_lm, W, H)
            gs   = engine.analyse(hand_lm, W, H)

            # Smooth the draw position
            smooth_pos = smoother.update(gs.draw_pos)
            gs.draw_pos = smooth_pos
            state.draw_pos = smooth_pos
            state.update_gesture(gs.gesture)

            # ── Ghost hold trigger ──────────────────────────
            if state.fist_held():
                state.is_ghost = ghost.toggle()
                ui.notify(f"👻 Ghost {'ON' if state.is_ghost else 'OFF'}")
                if state.is_ghost:
                    ghost.capture_background(frame)

            # ── Menu navigation ─────────────────────────────
            if state.menu_open:
                if gs.is_peace and state.menu_nav_ready():
                    state.menu_sel = (state.menu_sel + 1) % len(ui.MENU_ITEMS)
                if gs.is_ok:
                    _execute_menu(state, canvas, ghost, ui)
                # Three fingers closes menu
                if gs.is_three and state.menu_nav_ready(0.8):
                    state.menu_open = False
                    ui.notify("Menu closed")
            else:
                # ── Colour cycle (peace gesture) ─────────────
                if gs.is_peace and state.colour_switch_ready():
                    new_color = color_name_cycle(canvas.color_name)
                    canvas.set_color(new_color)
                    ui.notify(f"🎨 Color → {new_color}")

                # ── Open menu (three-finger) ──────────────────
                if gs.is_three and state.menu_nav_ready(0.8):
                    state.menu_open = True
                    ui.notify("≡ Menu opened")

                # ── Canvas drawing / erasing ──────────────────
                px, py = smooth_pos

                # Guard: don't draw in toolbar or status bar
                in_canvas = (config.TOOLBAR_HEIGHT < py < H - config.STATUS_BAR_HEIGHT)

                if gs.is_drawing and in_canvas:
                    canvas.draw_stroke(smooth_pos)
                    state.mode = "DRAW"
                    was_drawing = True
                elif gs.is_erasing and in_canvas:
                    canvas.erase_stroke(smooth_pos)
                    state.mode = "ERASE"
                    was_erasing = True
                else:
                    if was_drawing or was_erasing:
                        canvas.lift_pen()
                        was_drawing = was_erasing = False
                    state.mode = "IDLE"

        else:
            # No hand detected
            state.update_gesture("none")
            smoother.reset()
            canvas.lift_pen()
            was_drawing = was_erasing = False
            state.mode = "IDLE"

        # ── Ghost mode rendering ────────────────────────────
        frame = ghost.apply(frame, bbox if state.is_ghost else None)

        # ── Composite canvas onto frame ─────────────────────
        canvas.composite_onto(frame)

        # ── Draw UI HUD ────────────────────────────────────
        ui_state = {
            "gesture"   : state.gesture,
            "color_name": canvas.color_name,
            "color_bgr" : canvas.color,
            "brush_size": canvas.brush_thickness,
            "is_ghost"  : state.is_ghost,
            "fps"       : state.fps,
            "mode"      : state.mode,
            "menu_open" : state.menu_open,
            "menu_sel"  : state.menu_sel,
            "draw_pos"  : state.draw_pos if gs else None,
        }
        ui.draw(frame, ui_state)

        # ── Show ────────────────────────────────────────────
        cv2.imshow("Gestify", frame)

        # ── Key handling ────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == config.KEY_QUIT:
            print("[INFO] Quit.")
            break
        elif key == config.KEY_CLEAR:
            canvas.clear()
            ui.notify("🗑 Canvas cleared")
        elif key == config.KEY_SAVE:
            path = _save_canvas(canvas)
            ui.notify(f"💾 Saved: {os.path.basename(path)}")
        elif key == config.KEY_GHOST:
            state.is_ghost = ghost.toggle()
            if state.is_ghost:
                ghost.capture_background(frame)
            ui.notify(f"👻 Ghost {'ON' if state.is_ghost else 'OFF'}")
        elif key == config.KEY_HELP:
            ui.toggle_help()
        elif key == config.KEY_MENU:
            state.menu_open = not state.menu_open
            ui.notify("≡ Menu " + ("opened" if state.menu_open else "closed"))
        # Brush size: + / -
        elif key == ord('+') or key == ord('='):
            canvas.set_brush_size(canvas.brush_thickness + 2)
            ui.notify(f"Brush: {canvas.brush_thickness}px")
        elif key == ord('-'):
            canvas.set_brush_size(canvas.brush_thickness - 2)
            ui.notify(f"Brush: {canvas.brush_thickness}px")
        # Colour shortcuts 1-8
        elif ord('1') <= key <= ord('8'):
            cname = list(config.COLORS.keys())[key - ord('1')]
            canvas.set_color(cname)
            ui.notify(f"🎨 {cname}")
        # Undo / Redo
        elif key == ord('z'):
            canvas.undo()
            ui.notify("↩ Undo")
        elif key == ord('y'):
            canvas.redo()
            ui.notify("↪ Redo")

    # ── Cleanup ─────────────────────────────────────────────
    hands.close()
    cap.release()
    cv2.destroyAllWindows()


# ── Menu action executor ─────────────────────────────────────

def _execute_menu(state: AppState, canvas: AirCanvas,
                  ghost: GhostMode, ui: UIRenderer) -> None:
    _, action = ui.MENU_ITEMS[state.menu_sel]
    if action == "draw":
        state.mode = "DRAW"
        state.menu_open = False
        ui.notify("🖊 Draw mode")
    elif action == "erase":
        state.mode = "ERASE"
        state.menu_open = False
        ui.notify("🗑 Erase mode")
    elif action == "next_color":
        new_color = color_name_cycle(canvas.color_name)
        canvas.set_color(new_color)
        ui.notify(f"🎨 Color → {new_color}")
    elif action == "save":
        path = _save_canvas(canvas)
        ui.notify(f"💾 Saved: {os.path.basename(path)}")
        state.menu_open = False
    elif action == "clear":
        canvas.clear()
        ui.notify("🗑 Canvas cleared")
        state.menu_open = False
    elif action == "ghost":
        state.is_ghost = ghost.toggle()
        ui.notify(f"👻 Ghost {'ON' if state.is_ghost else 'OFF'}")
        state.menu_open = False
    elif action == "quit":
        cv2.destroyAllWindows()
        sys.exit(0)


def _save_canvas(canvas: AirCanvas) -> str:
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = f"canvas_{ts}.png"
    return canvas.save(path)


# ── Entry ────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
