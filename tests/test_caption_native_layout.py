"""Native-process doubles and synthetic morphology; no UI or private media."""

import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from replay import caption_linguistics, caption_native_layout, stages
from replay.caption_linguistics import CaptionLayoutError
from replay.caption_native_layout import NativeCaptionLayout, NativeCaptionStyle
from replay.caption_timing import CaptionPolicy, SpokenToken, make_captions
from replay.core import ReplayError
from replay.recordings import record


def fake_native(argv: tuple[str, ...], output: Path, timeout: int) -> None:
    assert argv[1:3] == ("measure-media", "caption_text_measure")
    assert timeout == 60
    request = record(json.loads(Path(argv[-1]).read_text(encoding="utf-8")))
    texts = request["texts"]
    assert isinstance(texts, list)
    strings = [text for text in texts if isinstance(text, str)]
    assert len(strings) == len(texts)
    output.write_text(
        json.dumps(
            {
                "motionRenderVerified": False,
                "font": {
                    "postScriptName": request["fontName"],
                    "lineAdvance": request["assFontSize"],
                },
                "bounds": [
                    {
                        "text": text,
                        "width": len(text) * 10 + 4,
                        "height": 12,
                        "lineWidths": [len(text) * 10 + 4],
                    }
                    for text in strings
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NativeCaptionLayout:
    monkeypatch.setattr(
        caption_linguistics, "_atomic_ends", lambda text: frozenset(range(1, len(text) + 1))
    )
    monkeypatch.setattr(caption_linguistics, "semantic_break", lambda previous, following: False)
    executable, evidence = tmp_path / "native", tmp_path / "calibration.json"
    executable.write_bytes(b"synthetic native binary")
    evidence.write_text('{"synthetic":true}', encoding="utf-8")
    monkeypatch.setattr(
        caption_linguistics,
        "boundaries",
        lambda text: (
            tuple(range(2, len(text), 2)) + (len(text),),
            tuple(range(1, len(text) + 1)),
        ),
    )
    return NativeCaptionLayout(
        executable=executable,
        directory=tmp_path / "metrics",
        evidence=(evidence,),
        style=NativeCaptionStyle("Synthetic", 10, 1, 0.5, 14),
        width=24,
        height=26,
        execute=fake_native,
    )


def test_layout_uses_observed_leading_and_caches_lines(layout: NativeCaptionLayout) -> None:
    assert layout("ab") == "ab"
    assert layout("abcd") == "ab\ncd"
    assert layout.measure("ab\ncd") == (24, 26)
    assert layout("abcd") == "ab\ncd"
    assert len(tuple(layout.directory.glob("*.response.json"))) == 3
    assert len(layout.verify_inputs()) == 10
    assert (layout.directory.stat().st_mode & 0o777) == 0o700
    assert ((layout.directory / "inputs.json").stat().st_mode & 0o777) == 0o600


def test_three_lines_use_two_observed_advances(layout: NativeCaptionLayout) -> None:
    layout.height = 40
    assert layout.measure("ab\ncd\nef") == (24, 40)
    assert layout("abcdef") == "ab\ncd\nef"
    layout.height = 39
    with pytest.raises(CaptionLayoutError):
        layout("abcdef")


def test_shared_wrapper_batches_all_candidate_lines(layout: NativeCaptionLayout) -> None:
    wrapped = caption_linguistics.wrap_measured_lines(
        "abcdefghijkl",
        measure=layout.measure,
        width=64,
        height=26,
        prepare_lines=layout.prepare_lines,
    )
    assert wrapped == "abcdef\nghijkl"
    assert len(tuple(layout.directory.glob("*.response.json"))) == 2
    assert layout.measure(wrapped) == (64, 26)


def test_policy_preserves_native_word_clocks(layout: NativeCaptionLayout) -> None:
    cues = make_captions(
        (
            SpokenToken("ab", 0, 1, "utterance"),
            SpokenToken("cd", 1, 2, "utterance"),
            SpokenToken("ef", 2, 3, "utterance"),
        ),
        policy=CaptionPolicy(hold_seconds=0, linguistic_boundaries=True, measured_layout=layout),
    )
    assert [(cue.text, cue.start, cue.end) for cue in cues] == [
        ("ab", 0, 1),
        ("ab\ncd", 1, 2),
        ("ef", 2, 3),
    ]


def test_overflow_and_insufficient_height_fail(layout: NativeCaptionLayout) -> None:
    with pytest.raises(CaptionLayoutError):
        layout("abcdef")
    layout.height = 25
    with pytest.raises(CaptionLayoutError):
        layout("abcd")


@pytest.mark.parametrize("text", ["", "a\rb", "a\0b", "a" * 121, "a\nb\nc\nd"])
def test_measurement_text_limits(layout: NativeCaptionLayout, text: str) -> None:
    with pytest.raises(ReplayError):
        layout.measure(text)


def test_batch_resource_bound_and_cache_limit(
    layout: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout.prepare_lines(("a" * 100, "b" * 100))
    assert len(tuple(layout.directory.glob("*.response.json"))) == 2
    monkeypatch.setattr(caption_native_layout, "MAX_CACHE", 2)
    with pytest.raises(ReplayError, match="cache budget"):
        layout.measure("new")


def test_changed_bound_input_stops_publication(layout: NativeCaptionLayout) -> None:
    layout("ab")
    (layout.directory / "batch-000000.response.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="inputs changed"):
        layout.verify_inputs()


@pytest.mark.parametrize(
    "change",
    ["text", "lineWidths", "width", "height", "font", "approved", "count", "large", "request"],
)
def test_malformed_native_result_fails(
    layout: NativeCaptionLayout,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    def corrupt(argv: tuple[str, ...], output: Path, timeout: int) -> None:
        fake_native(argv, output, timeout)
        document = record(json.loads(output.read_text(encoding="utf-8")))
        bounds = document["bounds"]
        assert isinstance(bounds, list)
        mutations: dict[str, object] = {
            "text": "wrong",
            "lineWidths": [1],
            "width": True,
            "height": -1,
        }
        if change in mutations:
            record(bounds[0])[change] = mutations[change]
        elif change == "font":
            document["font"] = {"postScriptName": "Different", "lineAdvance": 10}
        elif change == "approved":
            document["motionRenderVerified"] = True
        elif change == "count":
            document["bounds"] = []
        elif change == "large":
            monkeypatch.setattr(caption_native_layout, "MAX_RESPONSE_BYTES", 1)
        else:
            Path(argv[-1]).write_text("{}", encoding="utf-8")
        output.write_text(json.dumps(document), encoding="utf-8")

    monkeypatch.setattr(layout, "_execute", corrupt)
    with pytest.raises(ReplayError):
        layout.measure("ab")


@pytest.mark.parametrize("advance", [0, 9, 41, float("nan")])
def test_invalid_style(advance: float) -> None:
    with pytest.raises(ReplayError):
        replace(NativeCaptionStyle("Synthetic", 10, 1, 0.5, 14), motion_line_advance=advance)


@pytest.mark.parametrize("width,evidence", [(0, True), (float("nan"), True), (24, False)])
def test_invalid_constructor_inputs(
    layout: NativeCaptionLayout, width: float, evidence: bool
) -> None:
    with pytest.raises(ReplayError):
        NativeCaptionLayout(
            executable=layout.executable,
            directory=layout.directory.parent / "invalid",
            style=layout.style,
            width=width,
            height=26,
            evidence=(layout.executable,) if evidence else (),
            execute=fake_native,
        )


def test_cli_keeps_measurements_unapproved(
    layout: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = layout.directory.parent / "cli"
    monkeypatch.setattr(stages, "execute_command", fake_native)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "caption_native_layout",
            str(layout.executable),
            str(output),
            "--evidence",
            str(layout.executable),
            "--text",
            "abcd",
            "--font-size",
            "10",
            "--border",
            "1",
            "--rim",
            "0.5",
            "--line-advance",
            "14",
            "--width",
            "24",
            "--height",
            "26",
        ],
    )
    runpy.run_path(caption_native_layout.__file__, run_name="__main__")
    result = record(json.loads((output / "layouts.json").read_text(encoding="utf-8")))
    assert result["layouts"] == [
        {"sourceText": "abcd", "text": "ab\ncd", "width": 24, "height": 26}
    ]
    assert result["motionRenderVerified"] is False
    assert result["captionGeometryApproved"] is False
    assert result["deliveryComplete"] is False
    request = record(json.loads((output / "batch-000000.request.json").read_text(encoding="utf-8")))
    assert request["fontName"] == "NotoSansJP-Thin_Bold"


def test_equal_character_counts_use_native_widths(
    layout: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    def proportional(argv: tuple[str, ...], output: Path, timeout: int) -> None:
        fake_native(argv, output, timeout)
        document = record(json.loads(output.read_text(encoding="utf-8")))
        request = record(json.loads(Path(argv[-1]).read_text(encoding="utf-8")))
        texts = request["texts"]
        assert isinstance(texts, list)
        strings = [text for text in texts if isinstance(text, str)]
        document["bounds"] = [
            {
                "text": text,
                "width": sum(2 if ch == "i" else 20 for ch in text),
                "height": 12,
                "lineWidths": [sum(2 if ch == "i" else 20 for ch in text)],
            }
            for text in strings
        ]
        output.write_text(json.dumps(document), encoding="utf-8")

    monkeypatch.setattr(layout, "_execute", proportional)
    assert layout("iiii") == "iiii"
    with pytest.raises(CaptionLayoutError):
        layout("WWWW")


def test_existing_output_is_never_reused(layout: NativeCaptionLayout) -> None:
    with pytest.raises(FileExistsError):
        NativeCaptionLayout(
            executable=layout.executable,
            directory=layout.directory,
            style=layout.style,
            width=24,
            height=26,
            evidence=(layout.executable,),
            execute=fake_native,
        )


@pytest.mark.parametrize("point", [True, 0, 1025, float("nan")])
def test_expected_point_size_validation(layout: NativeCaptionLayout, point: float) -> None:
    with pytest.raises(ReplayError, match="point size"):
        NativeCaptionLayout(
            executable=layout.executable,
            directory=layout.directory.parent / "invalid-point",
            style=layout.style,
            width=24,
            height=26,
            evidence=(layout.executable,),
            expected_point_size=point,
            execute=fake_native,
        )


def test_ass_size_is_not_motion_point_size(
    layout: NativeCaptionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    def points(argv: tuple[str, ...], output: Path, timeout: int) -> None:
        fake_native(argv, output, timeout)
        raw = record(json.loads(output.read_text(encoding="utf-8")))
        record(raw["font"])["pointSize"] = 7.0
        output.write_text(json.dumps(raw), encoding="utf-8")

    monkeypatch.setattr(layout, "_execute", points)
    layout.expected_point_size = 10
    with pytest.raises(ReplayError, match="differs from the Motion"):
        layout.measure("ab")
    layout.expected_point_size = 7
    assert layout.measure("ab") == (24, 12)
