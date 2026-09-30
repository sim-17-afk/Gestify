# ============================================================
# ui_renderer.py — Gestify HUD overlay, toolbar, menus, status bar
# ============================================================
"""
Draws all on-screen UI elements onto the camera frame using OpenCV.
Nothing here modifies the canvas layer — only the display frame.
"""

from __future__ import annotations
import cv2
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional
import config


# ── Colour palette (BGR) ────────────────────────────────────
UI_BG        = (15,  15,  30)     # dark navy
UI_ACCENT    = (80,  200, 255)    # neon cyan
UI_ACCENT2   = (200, 100, 255)    # neon purple
UI_TEXT      = (230, 230, 230)
UI_GREEN     = (80,  220, 120)
UI_RED       = (60,  60,  220)
UI_YELLOW    = (40,  220, 220)
UI_WARN      = (30,  140, 255)


def _alpha_rect(frame: np.ndarray,
                x1: int, y1: int, x2: int, y2: int,
                color: Tuple[int,int,int],
                alpha: float = 0.55,
                radius: int  = 0) -> None:
    """Draw a semi-transparent filled rectangle (glassmorphism style)."""
    overlay = frame.copy()
    if radius > 0:
        # Rounded corners via ellipse trick
        cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.ellipse(overlay, (cx, cy), (radius, radius), 0, 0, 360, color, -1)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)


def _put_text(frame: np.ndarray,
              text : str,
              pos  : Tuple[int,int],
              scale: float = config.FONT_MED,
              color: Tuple[int,int,int] = UI_TEXT,
              bold : bool = False) -> None:
    thickness = 2 if bold else 1
    cv2.putText(frame, text, pos, config.FONT, scale, color, thickness, cv2.LINE_AA)


class UIRenderer:
    """Stateful renderer — call `draw(frame, state_dict)` each frame."""

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self._notif_msg  : str   = ""
        self._notif_until: float = 0.0
        self._show_help  : bool  = False
        self._show_menu  : bool  = False
        self._menu_sel   : int   = 0
        self._pulse_t    : float = 0.0

    # ── Main entry point ────────────────────────────────────

    def draw(self, frame: np.ndarray, state: dict) -> None:
        """
        state keys expected:
          gesture, color_name, brush_size, is_ghost, fps,
          canvas_strokes, mode, menu_open, menu_sel
        """
        self._pulse_t = time.time()

        self._draw_toolbar(frame, state)
        self._draw_status_bar(frame, state)
        self._draw_gesture_indicator(frame, state)
        self._draw_cursor_ring(frame, state)

        if state.get("menu_open"):
            self._draw_gesture_menu(frame, state)

        if state.get("is_ghost"):
            self._draw_ghost_hud(frame)

        if self._show_help:
            self._draw_help_panel(frame)

        if self._notif_msg and time.time() < self._notif_until:
            self._draw_notification(frame)

    # ── Toolbar (top band) ──────────────────────────────────

    def _draw_toolbar(self, frame: np.ndarray, state: dict) -> None:
        th = config.TOOLBAR_HEIGHT
        _alpha_rect(frame, 0, 0, self.w, th, UI_BG, alpha=0.75)

        # Gradient accent line at bottom of toolbar
        for x in range(self.w):
            t   = x / self.w
            r   = int(80  + 120*t)
            g   = int(100 + 100*(1-t))
            b   = int(255 - 80*t)
            frame[th-2, x] = (b, g, r)
            frame[th-3, x] = (b//2, g//2, r//2)

        # App title
        _put_text(frame, "✦ GESTIFY", (18, 30),
                  scale=config.FONT_LARGE, color=UI_ACCENT, bold=True)
        _put_text(frame, "v2.0", (170, 30), scale=config.FONT_SMALL,
                  color=(140, 140, 160))

        # FPS badge
        fps = state.get("fps", 0)
        fps_color = UI_GREEN if fps >= 25 else (UI_WARN if fps >= 15 else UI_RED)
        fps_x = self.w - 120
        _alpha_rect(frame, fps_x-6, 8, fps_x+100, 42, (20,20,40), alpha=0.8, radius=6)
        _put_text(frame, f"FPS {fps:5.1f}", (fps_x, 32),
                  scale=config.FONT_MED, color=fps_color, bold=True)

        # Color swatches
        colors = list(config.COLORS.items())
        sw = 36
        spad = 8
        start_x = 280
        active_name = state.get("color_name", config.DEFAULT_COLOR)
        for i, (name, bgr) in enumerate(colors):
            cx = start_x + i * (sw + spad)
            cy = th // 2
            is_active = (name == active_name)
            # Glow for active
            if is_active:
                cv2.circle(frame, (cx+sw//2, cy), sw//2+5, UI_ACCENT, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx+sw//2, cy), sw//2, bgr, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx+sw//2, cy), sw//2, (255,255,255) if is_active else (80,80,80), 1, cv2.LINE_AA)

        # Mode label
        mode = state.get("mode", "DRAW")
        mode_color = UI_ACCENT if mode=="DRAW" else (UI_RED if mode=="ERASE" else UI_ACCENT2)
        mx = self.w - 260
        _alpha_rect(frame, mx-10, 8, mx+140, 42, (20,20,40), alpha=0.8, radius=6)
        _put_text(frame, f"● {mode}", (mx, 32), scale=config.FONT_MED,
                  color=mode_color, bold=True)

        # Brush size indicator
        bx = self.w - 420
        bsize = state.get("brush_size", config.DEFAULT_BRUSH_THICKNESS)
        _put_text(frame, f"⬤ {bsize}px", (bx, 32), scale=config.FONT_SMALL,
                  color=UI_TEXT)

    # ── Status bar (bottom band) ────────────────────────────

    def _draw_status_bar(self, frame: np.ndarray, state: dict) -> None:
        sh = config.STATUS_BAR_HEIGHT
        y0 = self.h - sh
        _alpha_rect(frame, 0, y0, self.w, self.h, UI_BG, alpha=0.7)

        # Gesture label
        gest = state.get("gesture", "—").upper()
        _put_text(frame, f"GESTURE: {gest}", (12, y0+20),
                  scale=config.FONT_SMALL, color=UI_ACCENT)

        # Ghost status
        ghost_txt = "👻 GHOST ON" if state.get("is_ghost") else "  GHOST OFF"
        ghost_col = UI_ACCENT2 if state.get("is_ghost") else (80,80,100)
        _put_text(frame, ghost_txt, (260, y0+20),
                  scale=config.FONT_SMALL, color=ghost_col)

        # Shortcuts hint
        hints = "[Q] Quit  [C] Clear  [S] Save  [G] Ghost  [H] Help  [M] Menu"
        _put_text(frame, hints, (480, y0+20),
                  scale=config.FONT_SMALL, color=(100,100,120))

    # ── Gesture indicator (live label badge) ────────────────

    def _draw_gesture_indicator(self, frame: np.ndarray, state: dict) -> None:
        gest = state.get("gesture", "none")
        if gest == "none":
            return
        emoji_map = {
            "pointing": "☞  DRAWING",
            "palm"    : "🖐  ERASER",
            "fist"    : "✊  FIST",
            "peace"   : "✌  NEXT COLOR",
            "three"   : "三  MENU",
            "ok"      : "OK  CONFIRM",
            "gun"     : "☞  GUN",
            "pinch"   : "◎  PINCH",
            "open"    : "〇  OPEN",
        }
        label = emoji_map.get(gest, gest.upper())
        lx, ly = 14, config.TOOLBAR_HEIGHT + 50
        _alpha_rect(frame, lx-4, ly-24, lx+200, ly+8,
                    (20,20,40), alpha=0.7, radius=4)
        _put_text(frame, label, (lx, ly),
                  scale=config.FONT_MED, color=UI_ACCENT2, bold=True)

    # ── Cursor ring at index fingertip ──────────────────────

    def _draw_cursor_ring(self, frame: np.ndarray, state: dict) -> None:
        pos = state.get("draw_pos")
        if not pos:
            return
        x, y = pos
        # Skip if inside toolbar or status bar
        if y < config.TOOLBAR_HEIGHT or y > self.h - config.STATUS_BAR_HEIGHT:
            return

        mode   = state.get("mode", "DRAW")
        color  = state.get("color_bgr", UI_ACCENT)
        pulsed = abs(math.sin(self._pulse_t * 4)) * 4

        if mode == "ERASE":
            r = config.ERASER_THICKNESS // 2
            cv2.circle(frame, (x,y), r, UI_RED,   2, cv2.LINE_AA)
            cv2.circle(frame, (x,y), 4, UI_RED, -1, cv2.LINE_AA)
        elif mode == "DRAW":
            cv2.circle(frame, (x,y), int(6+pulsed), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x,y), 3, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x,y), 8, UI_TEXT, 1, cv2.LINE_AA)

    # ── Gesture-controlled menu ─────────────────────────────

    MENU_ITEMS = [
        ("🖊  DRAW MODE",       "draw"),
        ("🗑  ERASE MODE",      "erase"),
        ("🎨  NEXT COLOR",      "next_color"),
        ("💾  SAVE CANVAS",     "save"),
        ("🗑️  CLEAR CANVAS",    "clear"),
        ("👻  TOGGLE GHOST",    "ghost"),
        ("❌  EXIT",            "quit"),
    ]

    def _draw_gesture_menu(self, frame: np.ndarray, state: dict) -> None:
        sel = state.get("menu_sel", 0)
        mw  = config.MENU_WIDTH
        x0  = self.w - mw - 10
        y0  = config.TOOLBAR_HEIGHT + 20
        item_h = 52

        total_h = len(self.MENU_ITEMS) * item_h + 20
        _alpha_rect(frame, x0-10, y0-10, x0+mw+10, y0+total_h,
                    UI_BG, alpha=0.88, radius=12)
        # Border
        cv2.rectangle(frame, (x0-10, y0-10), (x0+mw+10, y0+total_h),
                      UI_ACCENT, 1, cv2.LINE_AA)

        _put_text(frame, "≡ GESTURE MENU", (x0+4, y0+14),
                  scale=config.FONT_MED, color=UI_ACCENT, bold=True)

        for i, (label, _) in enumerate(self.MENU_ITEMS):
            iy = y0 + 30 + i * item_h
            is_sel = (i == sel)
            if is_sel:
                pulse = abs(math.sin(self._pulse_t * 3)) * 0.3
                _alpha_rect(frame, x0-8, iy-2, x0+mw+8, iy+item_h-6,
                            UI_ACCENT, alpha=0.25+pulse, radius=6)
                cv2.rectangle(frame, (x0-8, iy-2), (x0+mw+8, iy+item_h-6),
                              UI_ACCENT, 1, cv2.LINE_AA)
            col = UI_ACCENT if is_sel else UI_TEXT
            _put_text(frame, label, (x0+6, iy+26),
                      scale=config.FONT_MED, color=col, bold=is_sel)

        _put_text(frame, "✌Peace=nav  OK=select",
                  (x0+2, y0+total_h-4),
                  scale=0.38, color=(100,100,140))

    # ── Ghost HUD ───────────────────────────────────────────

    def _draw_ghost_hud(self, frame: np.ndarray) -> None:
        # Animated corner brackets
        t = self._pulse_t
        pulse = int(abs(math.sin(t*2)) * 6)
        corners = [
            ((10,  config.TOOLBAR_HEIGHT+10), (10+60+pulse, config.TOOLBAR_HEIGHT+10),
             (10,  config.TOOLBAR_HEIGHT+10), (10,  config.TOOLBAR_HEIGHT+70+pulse)),
            ((self.w-10, config.TOOLBAR_HEIGHT+10), (self.w-10-60-pulse, config.TOOLBAR_HEIGHT+10),
             (self.w-10, config.TOOLBAR_HEIGHT+10), (self.w-10, config.TOOLBAR_HEIGHT+70+pulse)),
        ]
        col = (200, 100 + pulse*5, 255)
        for (a1,a2,b1,b2) in corners:
            cv2.line(frame, a1, a2, col, 2, cv2.LINE_AA)
            cv2.line(frame, b1, b2, col, 2, cv2.LINE_AA)

        label = "◈ GHOST MODE ACTIVE"
        lw, _ = cv2.getTextSize(label, config.FONT, config.FONT_MED, 2)[0]
        lx = (self.w - lw) // 2
        ly = config.TOOLBAR_HEIGHT + 40
        _alpha_rect(frame, lx-10, ly-22, lx+lw+10, ly+8, (30,0,50), alpha=0.75, radius=4)
        _put_text(frame, label, (lx, ly),
                  scale=config.FONT_MED, color=col, bold=True)

    # ── Help panel ──────────────────────────────────────────

    HELP_LINES = [
        ("GESTURES", None),
        ("☞  Index only",     "Draw on canvas"),
        ("🖐  Open palm",     "Erase"),
        ("✌  Peace",         "Next color / Menu nav"),
        ("三  3 fingers",     "Open/close menu"),
        ("OK  Pinch",         "Select menu item"),
        ("✊  Fist (hold)",   "Toggle Ghost Mode"),
        ("", None),
        ("KEYBOARD", None),
        ("[Q]  Quit",         ""),
        ("[C]  Clear canvas", ""),
        ("[S]  Save PNG",     ""),
        ("[G]  Ghost toggle", ""),
        ("[H]  This help",    ""),
        ("[M]  Menu",         ""),
    ]

    def _draw_help_panel(self, frame: np.ndarray) -> None:
        pw, ph = 420, 420
        px = (self.w - pw) // 2
        py = (self.h - ph) // 2
        _alpha_rect(frame, px, py, px+pw, py+ph, UI_BG, alpha=0.92, radius=14)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), UI_ACCENT, 1, cv2.LINE_AA)
        _put_text(frame, "  HELP  [H to close]", (px+12, py+28),
                  scale=config.FONT_LARGE, color=UI_ACCENT, bold=True)
        y = py + 58
        for label, desc in self.HELP_LINES:
            if desc is None:
                if label:
                    _put_text(frame, label, (px+12, y),
                              scale=config.FONT_SMALL, color=UI_ACCENT2, bold=True)
                y += 22 if label else 10
                continue
            _put_text(frame, label, (px+18, y), scale=config.FONT_SMALL, color=UI_TEXT)
            if desc:
                _put_text(frame, desc, (px+230, y), scale=config.FONT_SMALL,
                          color=(150,150,180))
            y += 24

    # ── Notification banner ─────────────────────────────────

    def notify(self, msg: str, duration: float = 2.5) -> None:
        self._notif_msg   = msg
        self._notif_until = time.time() + duration

    def _draw_notification(self, frame: np.ndarray) -> None:
        msg = self._notif_msg
        progress = max(0.0, (self._notif_until - time.time()) / 2.5)
        tw, _ = cv2.getTextSize(msg, config.FONT, config.FONT_MED, 2)[0]
        nx = (self.w - tw) // 2
        ny = self.h - config.STATUS_BAR_HEIGHT - 55
        bar_w = int(tw * progress)
        _alpha_rect(frame, nx-12, ny-26, nx+tw+12, ny+10, (20,20,50), alpha=0.85, radius=6)
        cv2.rectangle(frame, (nx-12, ny+4), (nx-12+bar_w+24, ny+8), UI_ACCENT, -1)
        _put_text(frame, msg, (nx, ny), scale=config.FONT_MED, color=UI_ACCENT, bold=True)

    # ── Public accessors ────────────────────────────────────

    def toggle_help(self) -> bool:
        self._show_help = not self._show_help
        return self._show_help
