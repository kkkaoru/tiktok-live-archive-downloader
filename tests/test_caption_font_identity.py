"""Font selection metadata only; no installed font or native renderer is used."""

import pytest

from replay.caption_font_identity import require_selected_identity, require_selected_name
from replay.core import ReplayError


def test_selected_postscript_and_variable_identity() -> None:
    require_selected_name("NotoSansJP-Thin_Bold")
    require_selected_identity(
        {
            "family": "Noto Sans JP",
            "face": "Bold",
            "postScriptName": "NotoSansJP-Thin_Bold",
            "fontURL": "file:///synthetic/fonts/NotoSansJP-VariableFont_wght.ttf",
            "variations": [{"identifier": 2003265652, "value": 700}],
        }
    )


@pytest.mark.parametrize("name", ["NotoSansJP-VariableFont_wght", "Noto Sans JP", "Arial-Bold", ""])
def test_file_or_family_name_cannot_replace_native_instance(name: str) -> None:
    with pytest.raises(ReplayError, match="PostScript"):
        require_selected_name(name)


@pytest.mark.parametrize(
    "field,value",
    [
        ("family", "Helvetica"),
        ("face", "Regular"),
        ("postScriptName", "NotoSansJP-Regular"),
        ("fontURL", "file:///synthetic/NotoSansJP-Bold.ttf"),
        ("fontURL", "https://example.invalid/NotoSansJP-VariableFont_wght.ttf"),
        ("fontURL", "file://other/NotoSansJP-VariableFont_wght.ttf"),
        ("fontURL", "file:///synthetic/NotoSansJP-VariableFont_wght.ttf#alias"),
        ("variations", []),
        ("variations", [{"identifier": 2003265652, "value": 400}]),
        ("variations", [{"identifier": 1, "value": 700}]),
    ],
)
def test_changed_file_instance_axis_or_weight_is_rejected(field: str, value: object) -> None:
    identity: dict[str, object] = {
        "family": "Noto Sans JP",
        "face": "Bold",
        "postScriptName": "NotoSansJP-Thin_Bold",
        "fontURL": "file:///synthetic/fonts/NotoSansJP-VariableFont_wght.ttf",
        "variations": [{"identifier": 2003265652, "value": 700}],
    }
    identity[field] = value
    with pytest.raises(ReplayError, match="variable file or weight"):
        require_selected_identity(identity)
