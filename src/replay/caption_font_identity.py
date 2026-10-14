"""Selected variable-font intent checked against hash-bound native font evidence.

This validates native selection metadata, not Motion/FCP raster consumption.
The caller must bind actual font bytes and native execution through existing receipts.
"""

from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from replay import caption_style_defaults as defaults
from replay.core import ReplayError


def require_selected_name(name: str) -> None:
    """Reject family/file names and other instances before native font lookup."""
    if name != defaults.FONT_POSTSCRIPT_NAME:
        raise ReplayError("Selected style requires the Noto Sans JP variable Bold PostScript name")


def require_selected_identity(identity: dict[str, object]) -> None:
    """Require the selected file, family, instance and weight in observed metadata."""
    url = urlsplit(str(identity.get("fontURL")))
    if (
        identity.get("postScriptName") != defaults.FONT_POSTSCRIPT_NAME
        or identity.get("family") != defaults.FONT_FAMILY
        or identity.get("face") != defaults.FONT_FACE
        or url.scheme != "file"
        or bool(url.netloc or url.query or url.fragment)
        or PurePosixPath(unquote(url.path)).name != defaults.FONT_FILE_NAME
        or identity.get("variations")
        != [{"identifier": defaults.FONT_WEIGHT_AXIS, "value": defaults.FONT_WEIGHT}]
    ):
        raise ReplayError("Native font differs from the selected variable file or weight")
