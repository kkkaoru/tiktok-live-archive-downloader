from fractions import Fraction
from pathlib import Path
from xml.etree.ElementTree import ParseError

import pytest

from replay.fcpxml_verify import AssetReference, FCPReport, clock, verify_fcpxml


@pytest.fixture
def project(tmp_path: Path) -> tuple[bytes, tuple[AssetReference, ...], Path]:
    original = tmp_path / "original.mp4"
    voice = tmp_path / "voice.wav"
    rendered = tmp_path / "rendered.mp4"
    proxy = tmp_path / "proxy.mp4"
    effect = tmp_path / "title.moti"
    original.touch()
    voice.touch()
    rendered.touch()
    proxy.touch()
    effect.touch()
    xml = f'''<fcpxml><resources>
<asset id="r2" start="0s" duration="2s">
<media-rep kind="original-media" src="{original.as_uri()}"/>
<media-rep kind="proxy-media" src="{proxy.as_uri()}"/></asset>
<asset id="r3" start="0s" duration="2s">
<media-rep kind="original-media" src="{voice.as_uri()}"/></asset>
<asset id="r4" start="0s" duration="1s">
<media-rep kind="original-media" src="{rendered.as_uri()}"/></asset>
<effect id="r5" src="{effect.as_uri()}"/></resources>
<sequence duration="1s" tcStart="0s"><spine>
<asset-clip ref="r2" offset="0s" start="0s" duration="1s" srcEnable="video">
<asset-clip ref="r3" offset="0s" start="0s" duration="1s" srcEnable="audio">
<adjust-volume amount="0dB"/></asset-clip>
<asset-clip ref="r4" offset="0s" start="0s" duration="1s" srcEnable="video"/>
<title ref="r5" offset="0s" duration="1s"/>
</asset-clip></spine></sequence></fcpxml>'''.encode()
    return (
        xml,
        (
            AssetReference("r2", original, Fraction(2), proxy),
            AssetReference("r3", voice, Fraction(2)),
            AssetReference("r4", rendered, Fraction(1)),
        ),
        effect,
    )


def test_existing_references_and_sample_clock(
    project: tuple[bytes, tuple[AssetReference, ...], Path],
) -> None:
    content, assets, effect = project
    assert verify_fcpxml(
        content, assets=assets, effect=effect, total_frames=30, gain=1
    ) == FCPReport(0, 30, 1, 1)


@pytest.mark.parametrize(
    "before,after",
    [
        (b'id="r2"', b'id="changed"'),
        (b'duration="2s"', b'duration="3s"'),
        (b'kind="proxy-media"', b'kind="other"'),
        (b'<effect id="r5"', b'<effect id="r9"'),
        (b'ref="r3" offset="0s"', b'ref="r3" offset="1s"'),
        (b'ref="r3" offset="0s" start="0s"', b'ref="r3" offset="0s" start="1s"'),
        (b'<adjust-volume amount="0dB"/>', b'<adjust-volume amount="2dB"/>'),
        (b'<adjust-volume amount="0dB"/>', b""),
        (b'ref="r4" offset="0s" start="0s"', b'ref="r4" offset="0s" start="1s"'),
        (b'ref="r3"', b'ref="other"'),
        (b'ref="r3"', b'ref="r4"'),
        (b'<sequence duration="1s"', b'<sequence duration="2s"'),
        (b'tcStart="0s"', b'tcStart="1/61s"'),
        (b'ref="r2" offset="0s"', b'ref="r2" offset="1s"'),
        (
            b'<title ref="r5" offset="0s" duration="1s"/>',
            b'<title ref="r5" offset="0s" duration="2s"/>',
        ),
        (b"<sequence ", b"<other "),
        (b"</sequence>", b"</other>"),
    ],
)
def test_changed_timeline_is_rejected(
    project: tuple[bytes, tuple[AssetReference, ...], Path], before: bytes, after: bytes
) -> None:
    content, assets, effect = project
    # The final two mutations intentionally invalidate well-formedness as well.
    with pytest.raises((ValueError, ParseError)):
        verify_fcpxml(
            content.replace(before, after), assets=assets, effect=effect, total_frames=30, gain=1
        )


def test_missing_file_stops(project: tuple[bytes, tuple[AssetReference, ...], Path]) -> None:
    content, assets, effect = project
    effect.unlink()
    with pytest.raises(ValueError, match="media reference"):
        verify_fcpxml(content, assets=assets, effect=effect, total_frames=30, gain=1)


def test_remote_media_refused(project: tuple[bytes, tuple[AssetReference, ...], Path]) -> None:
    content, assets, effect = project
    with pytest.raises(ValueError, match="media URL"):
        verify_fcpxml(
            content.replace(b"file:///", b"https://example.test/"),
            assets=assets,
            effect=effect,
            total_frames=30,
            gain=1,
        )


def test_external_entity_refused(project: tuple[bytes, tuple[AssetReference, ...], Path]) -> None:
    content, assets, effect = project
    with pytest.raises(ValueError, match="Unsafe"):
        verify_fcpxml(
            b'<!DOCTYPE fcpxml SYSTEM "remote">' + content,
            assets=assets,
            effect=effect,
            total_frames=30,
            gain=1,
        )


def test_bad_gain(project: tuple[bytes, tuple[AssetReference, ...], Path]) -> None:
    content, assets, effect = project
    with pytest.raises(ValueError, match="settings"):
        verify_fcpxml(content, assets=assets, effect=effect, total_frames=30, gain=float("nan"))


def test_wrong_root(project: tuple[bytes, tuple[AssetReference, ...], Path]) -> None:
    content, assets, effect = project
    with pytest.raises(ValueError, match="structure"):
        verify_fcpxml(
            content.replace(b"fcpxml", b"other"),
            assets=assets,
            effect=effect,
            total_frames=30,
            gain=1,
        )


def test_global_output_duration_must_match_rendered_asset(
    project: tuple[bytes, tuple[AssetReference, ...], Path],
) -> None:
    content, assets, effect = project
    with pytest.raises(ValueError, match="coverage"):
        verify_fcpxml(content, assets=assets, effect=effect, total_frames=31, gain=1)


def test_no_voice_is_not_a_valid_silent_project(
    project: tuple[bytes, tuple[AssetReference, ...], Path],
) -> None:
    content, assets, effect = project
    voice = (
        b'<asset-clip ref="r3" offset="0s" start="0s" duration="1s" srcEnable="audio">\n'
        b'<adjust-volume amount="0dB"/></asset-clip>'
    )
    with pytest.raises(ValueError, match="exactly one"):
        verify_fcpxml(
            content.replace(voice, b""), assets=assets, effect=effect, total_frames=30, gain=1
        )


def test_clock_units_required() -> None:
    with pytest.raises(ValueError, match="clock"):
        clock("30")
