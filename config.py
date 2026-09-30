# ============================================================
# config.py — Global configuration for the Hand Gesture Suite
# ============================================================

# ── Camera ─────────────────────────────────────────────────
CAMERA_INDEX     = 0
FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720
TARGET_FPS       = 30

# ── MediaPipe ──────────────────────────────────────────────
MP_MAX_HANDS              = 1
MP_DETECTION_CONFIDENCE   = 0.75
MP_TRACKING_CONFIDENCE    = 0.75

# ── UI / Layout ────────────────────────────────────────────
TOOLBAR_HEIGHT    = 80          # px — top toolbar band
MENU_WIDTH        = 220         # px — right-side gesture menu
STATUS_BAR_HEIGHT = 30          # px — bottom status bar

# ── Brush / Canvas ─────────────────────────────────────────
DEFAULT_BRUSH_THICKNESS = 8
ERASER_THICKNESS        = 60
MIN_BRUSH               = 2
MAX_BRUSH               = 50
SMOOTHING_ALPHA         = 0.4   # exponential moving average (0 = max smooth)

# ── Colors (BGR) ───────────────────────────────────────────
COLORS = {
    "Red"    : (50,  50,  220),
    "Green"  : (50,  200, 50 ),
    "Blue"   : (220, 80,  50 ),
    "Yellow" : (0,   220, 220),
    "Purple" : (200, 50,  200),
    "Cyan"   : (220, 200, 50 ),
    "White"  : (255, 255, 255),
    "Orange" : (30,  140, 255),
}
DEFAULT_COLOR = "Red"

# ── Ghost Mode ─────────────────────────────────────────────
GHOST_BLUR_KSIZE  = (25, 25)    # Gaussian blur kernel
GHOST_HISTORY     = 200         # BackgroundSubtractor history
GHOST_THRESHOLD   = 25

# ── Gesture thresholds ─────────────────────────────────────
PINCH_THRESHOLD        = 40     # px — index-thumb distance for pinch
FIST_FINGER_THRESHOLD  = 0.06   # fraction of hand height — finger-up test
MIN_DRAW_DISTANCE      = 5      # px — minimum movement to register a stroke

# ── Fonts & sizes (OpenCV) ─────────────────────────────────
FONT       = 0   # cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.45
FONT_MED   = 0.65
FONT_LARGE = 0.9

# ── Key bindings ───────────────────────────────────────────
KEY_QUIT    = ord('q')
KEY_CLEAR   = ord('c')
KEY_SAVE    = ord('s')
KEY_GHOST   = ord('g')
KEY_HELP    = ord('h')
KEY_MENU    = ord('m')
