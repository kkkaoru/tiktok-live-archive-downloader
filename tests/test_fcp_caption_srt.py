from fractions import Fraction
from pathlib import Path

import pytest

from replay import fcp_caption_srt as srt
from replay.core import ReplayError


@pytest.fixture
def xml(tmp_path: Path) -> Path:
    path = tmp_path / "timeline.fcpxml"
    path.write_text(
        '<fcpxml><library><event><project><sequence duration="3s" tcStart="0s">'
        '<spine><asset-clip offset="0s" start="100s" duration="1s">'
        '<title offset="100s" start="3600s" duration="1s">'
        "<text>今日は\n元気</text><text>今日は\n元気</text><text>今日は\n元気</text>"
        '</title></asset-clip><asset-clip offset="1s" start="200s" duration="2s">'
        '<title offset="200s" duration="1/2s"><text>今日は\n元気</text></title>'
        '<title offset="401/2s" duration="1/30s"><text>次</text></title>'
        '<title offset="3008/15s" duration="1s"><text>次です</text></title>'
        "</asset-clip></spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    return path


def test_exact_source_jumps_layers_prefixes_and_adjacent_merge(xml: Path) -> None:
    assert srt.caption_cues(xml) == (
        (Fraction(0), Fraction(3, 2), "今日は\n元気"),
        (Fraction(3, 2), Fraction(23, 15), "次"),
        (Fraction(23, 15), Fraction(38, 15), "次です"),
    )
    assert srt.srt_text(xml) == (
        "1\n00:00:00,000 --> 00:00:01,500\n今日は\n元気\n\n"
        "2\n00:00:01,500 --> 00:00:01,533\n次\n\n"
        "3\n00:00:01,533 --> 00:00:02,533\n次です\n"
    )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('duration="3s"', 'duration="4s"', "complete clip coverage"),
        ('tcStart="0s"', 'tcStart="3600s"', "Nonzero"),
        ('offset="1s"', 'offset="2s"', "contiguous"),
        ('duration="2s"', 'duration="0s"', "positive"),
        ('offset="401/2s"', 'offset="200s"', "Overlapping"),
        ('offset="401/2s"', 'offset="203s"', "beyond"),
        ('offset="100s" start=', 'offset="99s" start=', "beyond"),
        ('duration="1/30s"', 'duration="1/10000s"', "millisecond"),
        ('duration="1/30s"', 'duration="nope"', "Missing or invalid"),
        ('duration="1/30s"', 'duration="a/bs"', "Invalid FCP"),
        ('duration="1/30s"', 'duration="1/0s"', "Invalid FCP"),
        ('duration="1/30s"', 'duration="-1s"', "Negative"),
        ("<text>次</text>", "<text>次</text><text>別</text>", "identical"),
        ("<text>次</text>", "", "identical"),
        ("<text>次</text>", "<text> </text>", "represented"),
        ("<text>次</text>", "<text>次\n\n文</text>", "represented"),
        ("<text>次</text>", "<text>次\t文</text>", "represented"),
        ("<text>次</text>", "<text><text-style>次</text-style></text>", "plain"),
        ("<text>次</text>", "<text-style-def/>", "overrides"),
        ("<text>次</text>", "<timeMap/>", "retiming"),
        ('<title offset="401/2s"', '<title enabled="0" offset="401/2s"', "Disabled"),
        ('<asset-clip offset="0s"', '<asset-clip enabled="0" offset="0s"', "Disabled primary"),
        ('<title offset="401/2s"', '<gap><title offset="401/2s"', "direct clip"),
        ("<fcpxml>", "<wrong>", "Expected FCPXML"),
        ("</fcpxml>", "", "Invalid FCPXML"),
    ],
)
def test_rejects_unsupported_xml(xml: Path, old: str, new: str, message: str) -> None:
    xml.write_text(xml.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(ReplayError, match=message):
        srt.caption_cues(xml)


def test_bounded_text(xml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srt, "MAX_TEXT_CHARACTERS", 2)
    with pytest.raises(ReplayError, match="text exceeds"):
        srt.caption_cues(xml)


def test_native_style_layers_preserve_single_caption(xml: Path) -> None:
    expected = srt.caption_cues(xml)
    text = xml.read_text()
    text = text.replace(
        "<text>次</text>",
        '<text><text-style ref="s1">次</text-style></text>' * 3
        + '<text-style-def id="s1"><text-style fontSize="20"/></text-style-def>',
    )
    xml.write_text(text)
    assert srt.caption_cues(xml) == expected


def test_native_style_missing_definition_rejected(xml: Path) -> None:
    xml.write_text(
        xml.read_text().replace(
            "<text>次</text>", '<text><text-style ref="absent">次</text-style></text>'
        )
    )
    with pytest.raises(ReplayError, match="local definitions"):
        srt.caption_cues(xml)


def test_bounded_title_count(xml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srt, "MAX_TITLES", 1)
    with pytest.raises(ReplayError, match="bounded direct"):
        srt.caption_cues(xml)


def test_bounded_input(xml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srt, "MAX_FCPXML_BYTES", 1)
    with pytest.raises(ReplayError, match="bounded nonempty"):
        srt.caption_cues(xml)


def test_missing_input(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="bounded nonempty"):
        srt.caption_cues(tmp_path / "missing")


def test_custom_doctype_rejected(xml: Path) -> None:
    xml.write_text(
        '<!DOCTYPE fcpxml [<!ENTITY body "oops">]>' + xml.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="declarations"):
        srt.caption_cues(xml)


def test_standard_doctype_accepted(xml: Path) -> None:
    xml.write_text("<!DOCTYPE fcpxml>" + xml.read_text(encoding="utf-8"), encoding="utf-8")
    assert len(srt.caption_cues(xml)) == 3


def test_multiple_sequences_rejected(xml: Path) -> None:
    xml.write_text(
        xml.read_text(encoding="utf-8").replace(
            "</project>", '<sequence duration="0s"/></project>'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="exactly one"):
        srt.caption_cues(xml)


def test_missing_clock_context_rejected() -> None:
    with pytest.raises(ReplayError, match="clock context"):
        srt._Collector().finish_title()
