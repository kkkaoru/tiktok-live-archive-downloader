"""User-selected caption style 20; use unless explicitly overridden.

Colours are sRGB intent, NOT calibrated Motion control values. Geometry is the
approved SVG preview's units at font size 88; scale proportionally with font size.
These constants do not certify rendered colour fidelity or AAA contrast.
"""

from typing import Final

DEFAULT_CAPTION_STYLE_ID: Final[int] = 20
# Native font selection uses its PostScript instance, never the file's basename.
FONT_FILE_NAME: Final[str] = "NotoSansJP-VariableFont_wght.ttf"
FONT_FAMILY: Final[str] = "Noto Sans JP"
FONT_FACE: Final[str] = "Bold"
FONT_POSTSCRIPT_NAME: Final[str] = "NotoSansJP-Thin_Bold"
FONT_WEIGHT_AXIS: Final[int] = 2003265652  # OpenType 'wght'.
FONT_WEIGHT: Final[float] = 700.0
# Explicit editing intent in output pixels, not a detected head measurement.
HEAD_CAPTION_GAP_PX: Final[float] = 24.0
# Spatial tolerance at the 1920-pixel reference canvas; no frame-rate-dependent lag.
HEAD_CAPTION_Y_DEADBAND_PX: Final[float] = 6.0
HEAD_CAPTION_REFERENCE_HEIGHT: Final[int] = 1920
# Motion stage canvases use at least this long edge for reference coordinates.
MOTION_REFERENCE_LONG_EDGE: Final[int] = 1920
# Approximate head extension above a measured face, not semantic hair detection.
HEAD_EXTENSION_FACE_HEIGHT_RATIO: Final[float] = 1.25
# Screen-only candidate placement when no current head is observed; not clearance proof.
CAPTION_SCREEN_INSET_PX: Final[float] = 24.0
MAX_CAPTION_LINES: Final[int] = 3
# Existing fixed-bottom FCP/Y-only entrance remains unchanged.
ENTRANCE_START_Y_SCALE: Final[float] = 0.96
ENTRANCE_DURATION_NUMERATOR: Final[int] = 1
ENTRANCE_DURATION_DENOMINATOR: Final[int] = 15
# Motion title entrance: visible uniform 22% scale-up over 0.2 seconds.
MOTION_ENTRANCE_START_SCALE: Final[float] = 0.78
MOTION_ENTRANCE_DURATION_NUMERATOR: Final[int] = 1
MOTION_ENTRANCE_DURATION_DENOMINATOR: Final[int] = 5

COLOR_SPACE: Final[str] = "sRGB"
GRADIENT_TOP_HEX: Final[str] = "#2FB0EE"
GRADIENT_BOTTOM_HEX: Final[str] = "#E3F6FF"
GRADIENT_DARK_HOLD: Final[float] = 0.75
GRADIENT_DIRECTION: Final[str] = "top_to_bottom"
GRADIENT_INTERPOLATION: Final[str] = "sRGB"

REFERENCE_FONT_SIZE: Final[float] = 88.0
WHITE_HEX: Final[str] = "#FFFFFF"
BLACK_HEX: Final[str] = "#000000"
# Centred SVG stroke widths: outer black behind white, inner black behind fill.
WHITE_STROKE_WIDTH: Final[float] = 28.8
# Keep the thin outer black rim at 3 reference units on each side.
OUTER_BLACK_STROKE_WIDTH: Final[float] = 34.8
INNER_BLACK_STROKE_WIDTH: Final[float] = 4.0
STROKE_LINE_JOIN: Final[str] = "round"

SHADOW_COLOR_HEX: Final[str] = "#000000"
SHADOW_OFFSET_X: Final[float] = 0.0
SHADOW_OFFSET_Y: Final[float] = 7.0
SHADOW_BLUR_STD_DEVIATION: Final[float] = 4.0
SHADOW_OPACITY: Final[float] = 0.6
