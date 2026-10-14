import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from test_caption_native_layout import fake_native
from test_caption_native_layout import layout as layout

from replay import caption_native_reflow
from replay.caption_linguistics import CaptionLayoutError
from replay.caption_native_layout import NativeCaptionLayout
from replay.core import ReplayError
from replay.recordings import record


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidateOnly": True,
                "canvas": [1080, 1920],
                "coordinateSystem": "top-left canvas; calibrated text origin",
                "policy": {"frame_rate": 30},
                "inputs": {str((tmp_path / "original.mp4").resolve()): "a" * 64},
                "candidates": [
                    {"cue": {"text": "ab", "start": 0.1, "end": 0.3}, "x": 540, "y": 500},
                    {"cue": {"text": "abcd\nef", "start": 0.3, "end": 0.9}, "x": 540, "y": 500},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def exact(layout: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch) -> NativeCaptionLayout:
    def native(argv: tuple[str, ...], output: Path, timeout: int) -> None:
        fake_native(argv, output, timeout)
        raw = record(json.loads(output.read_text(encoding="utf-8")))
        record(raw["font"])["pointSize"] = 10
        output.write_text(json.dumps(raw), encoding="utf-8")

    monkeypatch.setattr(layout, "_execute", native)
    layout.expected_point_size = 10
    return layout


def test_reflow_preserves_text_clock_and_discards_old_origins(
    source: Path, exact: NativeCaptionLayout
) -> None:
    output = source.with_name("reflow.json")
    original = source.read_bytes()
    caption_native_reflow.reflow(
        source,
        output,
        layout=exact,
        duration=Fraction(1),
        rate=30,
        source_media=source.with_name("original.mp4"),
    )
    raw = record(json.loads(output.read_text()))
    assert raw["captions"] == [
        {"text": "ab", "start": 0.1, "end": 0.3},
        {"text": "ab\ncd\nef", "start": 0.3, "end": 0.9},
    ]
    assert record(raw["layout"])["changedStages"] == 1
    assert record(raw["layout"])["originsRequireRetargeting"] is True
    assert source.read_bytes() == original
    assert str(source.resolve()) in record(raw["inputs"])
    assert raw["coverage"] == {"start": 0.0, "end": 1.0}
    assert len(tuple(exact.directory.glob("*.request.json"))) == 2
    with pytest.raises(FileExistsError):
        caption_native_reflow.reflow(
            source,
            output,
            layout=exact,
            duration=Fraction(1),
            rate=30,
            source_media=source.with_name("original.mp4"),
        )


def test_absolute_segment_clock(source: Path, exact: NativeCaptionLayout) -> None:
    output = source.with_name("shifted.json")
    caption_native_reflow.reflow(
        source,
        output,
        layout=exact,
        duration=Fraction(1),
        rate=30,
        offset=Fraction(10),
        source_media=source.with_name("original.mp4"),
    )
    raw = record(json.loads(output.read_text()))
    assert raw["captions"] == [
        {"text": "ab", "start": 10.1, "end": 10.3},
        {"text": "ab\ncd\nef", "start": 10.3, "end": 10.9},
    ]


@pytest.mark.parametrize(
    "window",
    [(Fraction(-1), Fraction(1)), (Fraction(1), Fraction(2)), (Fraction(9, 10), Fraction(1))],
)
def test_window_refuses_invalid_or_empty(
    source: Path, exact: NativeCaptionLayout, window: tuple[Fraction, Fraction]
) -> None:
    with pytest.raises(ReplayError, match="window"):
        caption_native_reflow.reflow(
            source,
            source.with_name("window.json"),
            layout=exact,
            duration=Fraction(1),
            rate=30,
            source_media=source.with_name("original.mp4"),
            window=window,
        )


def test_explicit_window_is_clipped_and_coverage_is_not_full(
    source: Path, exact: NativeCaptionLayout
) -> None:
    output = source.with_name("short.json")
    caption_native_reflow.reflow(
        source,
        output,
        layout=exact,
        duration=Fraction(1),
        rate=30,
        source_media=source.with_name("original.mp4"),
        window=(Fraction(1, 5), Fraction(2, 5)),
    )
    raw = record(json.loads(output.read_text()))
    assert raw["coverage"] == {"start": 0.2, "end": 0.4}
    assert raw["captions"] == [
        {"text": "ab", "start": 0.2, "end": 0.3},
        {"text": "ab\ncd\nef", "start": 0.3, "end": 0.4},
    ]


@pytest.mark.parametrize("change", ["point", "canvas", "offset", "overlap", "width", "binding"])
def test_invalid_or_unfit_never_invents_timings(
    source: Path, exact: NativeCaptionLayout, change: str
) -> None:
    raw = record(json.loads(source.read_text()))
    if change == "point":
        exact.expected_point_size = None
    elif change == "canvas":
        raw["canvas"] = [1920, 1080]
    elif change == "overlap":
        entries = raw["candidates"]
        assert isinstance(entries, list)
        record(record(entries[1])["cue"])["start"] = 0.2
    elif change == "width":
        exact.width = 1
    elif change == "binding":
        raw["inputs"] = {}
    source.write_text(json.dumps(raw), encoding="utf-8")
    output = source.with_name("invalid.json")
    with pytest.raises((ReplayError, CaptionLayoutError)):
        caption_native_reflow.reflow(
            source,
            output,
            layout=exact,
            duration=Fraction(1),
            rate=30,
            source_media=source.with_name("original.mp4"),
            offset=Fraction(-1 if change == "offset" else 0),
        )
    assert not output.exists()


def test_cli(source: Path, exact: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caption_native_reflow, "NativeCaptionLayout", lambda **kwargs: exact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reflow",
            str(source),
            str(source.with_name("cli.json")),
            str(exact.executable),
            str(source.with_name("metrics")),
            "--duration",
            "1",
            "--source-media",
            str(source.with_name("original.mp4")),
            "--ass-font-size",
            "10",
            "--point-size",
            "10",
            "--border",
            "1",
            "--rim",
            "0.5",
            "--width",
            "24",
            "--evidence",
            str(exact.executable),
        ],
    )
    caption_native_reflow.main()
    assert source.with_name("cli.json").is_file()


def test_changed_source_is_rejected(
    source: Path, exact: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = exact.verify_inputs

    def changed() -> dict[str, str]:
        source.write_text("{}", encoding="utf-8")
        return original()

    monkeypatch.setattr(exact, "verify_inputs", changed)
    with pytest.raises(ReplayError, match="changed during reflow"):
        caption_native_reflow.reflow(
            source,
            source.with_name("bad.json"),
            layout=exact,
            duration=Fraction(1),
            rate=30,
            source_media=source.with_name("original.mp4"),
        )
