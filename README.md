# ✦ Gestify — Hand Gesture Control System

A real-time computer vision project powered by **Python**, **OpenCV**, **MediaPipe**, and **NumPy** that lets you control an application entirely with hand gestures — no mouse or keyboard needed.

---

## 🚀 Features

| Feature | Gesture | Description |
|---|---|---|
| **Air Canvas** | ☞ Index finger only | Draw strokes in the air |
| **Color Switch** | ✌ Peace sign | Cycle through 8 colors |
| **Palm Eraser** | 🖐 Open palm | Erase canvas strokes |
| **Ghost Mode** | ✊ Hold fist (1.8s) | Invisibility cloak effect |
| **Gesture Menu** | 三 Three fingers | Open nav menu |
| **Menu Select** | OK Pinch | Confirm menu action |
| **Real-time HUD** | — | FPS, mode, gesture labels |

---

## 📦 Installation

```bash
cd VISION
pip install -r requirements.txt
```

> **Requirements**: Python 3.9+, webcam

---

## ▶️ Run

```bash
python main.py
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` | Quit |
| `C` | Clear canvas |
| `S` | Save canvas as PNG |
| `G` | Toggle Ghost Mode |
| `H` | Toggle help panel |
| `M` | Toggle gesture menu |
| `+` / `-` | Increase / decrease brush size |
| `1`–`8` | Quick color select |
| `Z` | Undo |
| `Y` | Redo |

---

## 👁️ Gesture Reference

```
☞  Index finger only  →  DRAW mode
🖐  Open palm (4+ fingers)  →  ERASE mode
✌  Peace (index + middle)  →  Next color / Menu navigate
三  Three fingers  →  Open/Close gesture menu
OK  Pinch (index + thumb)  →  Select menu item
✊  Fist held 1.8s  →  Toggle Ghost Mode
```

---

## 👻 Ghost Mode

Ghost Mode creates an **invisibility cloak effect** around your hand:

1. Press `G` (or hold fist for 1.8s) to toggle
2. The app captures the background frame
3. Your hand region is replaced with the background in real-time
4. Chromatic aberration + scanline effects are applied for a sci-fi look

---

## 🗂️ Project Structure

```
VISION/
├── main.py           # Entry point & main loop
├── config.py         # All tunable constants
├── gesture_engine.py # MediaPipe → GestureState
├── canvas.py         # Air canvas drawing layer
├── ghost_mode.py     # Invisibility effect
├── smoother.py       # Position EMA smoother
├── ui_renderer.py    # Gestify HUD / overlay renderer
└── requirements.txt
```

---

## 🎨 Color Palette

8 built-in colors selectable by gesture or keyboard (1–8):

**Red · Green · Blue · Yellow · Purple · Cyan · White · Orange**

---

## ⚙️ Configuration

Edit [`config.py`](config.py) to tune:
- `CAMERA_INDEX` — change webcam (default 0)
- `SMOOTHING_ALPHA` — drawing smoothness (0=heavy, 1=raw)
- `PINCH_THRESHOLD` — pinch sensitivity
- `FIST_FINGER_THRESHOLD` — finger detection sensitivity
- `GHOST_HISTORY` / `GHOST_BLUR_KSIZE` — ghost effect quality
