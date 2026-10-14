import json
import runpy
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from replay import fcp_measured
from replay.caption_measured_plan import PositionedCue
from replay.caption_timing import CaptionCue
from replay.recordings import record
from replay.stages import file_digest


def test_native_request_pairs_may_exceed_generic_object_limit() -> None:
    bindings = {f"/synthetic/input-{index}": "a" * 64 for index in range(284)}
    assert len(fcp_measured._input_bindings(bindings)) == 284


@pytest.mark.parametrize("value", [None, {}, {1: "a" * 64}, {"/source": 1}])
def test_provenance_requires_bounded_string_bindings(value: object) -> None:
    with pytest.raises(ValueError):
        fcp_measured._input_bindings(value)


def test_provenance_retains_its_own_bound() -> None:
    with pytest.raises(ValueError):
        fcp_measured._input_bindings({f"/input-{index}": "a" * 64 for index in range(513)})


@pytest.fixture
def parts(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic evidence")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "version": 1,
                "candidateOnly": True,
                "canvas": [1080, 1920],
                "coordinateSystem": "top-left canvas; calibrated text origin",
                "policy": {"frame_rate": 30},
                "inputs": {str(source): file_digest(source)},
                "candidates": [{"cue": {"text": "あ", "start": 0, "end": 1}, "x": 540, "y": 500}],
            }
        ),
        encoding="utf-8",
    )
    raw = record(json.loads(first.read_text(encoding="utf-8")))
    raw["candidates"] = [{"cue": {"text": "い", "start": 1, "end": 2}, "x": 540, "y": 480}]
    second.write_text(json.dumps(raw), encoding="utf-8")
    return first, second, tmp_path / "combined.json"


def test_combine_retains_clocks_and_origins_without_approval(
    parts: tuple[Path, Path, Path],
) -> None:
    first, second, output = parts
    fcp_measured.combine_measured_captions(
        [(first, file_digest(first)), (second, file_digest(second))],
        output,
        width=1080,
        height=1920,
        rate=30,
        duration=Fraction(2),
        calibration=(first.parent / "source.bin", file_digest(first.parent / "source.bin")),
    )
    raw = record(json.loads(output.read_text(encoding="utf-8")))
    assert raw["candidates"] == [
        {"cue": {"text": "あ", "start": 0, "end": 1}, "x": 540, "y": 500},
        {"cue": {"text": "い", "start": 1, "end": 2}, "x": 540, "y": 480},
    ]
    assert raw["deliveryComplete"] is False
    assert raw["aaaContrastApproved"] is False
    assert output.stat().st_mode & 0o777 == 0o600
    assert len(record(raw["inputs"])) == 3


@pytest.mark.parametrize("change", ["policy", "missing", "stale", "relative", "overlap"])
def test_combine_rejects_bad_parts(parts: tuple[Path, Path, Path], change: str) -> None:
    first, second, output = parts
    raw = record(json.loads(second.read_text(encoding="utf-8")))
    if change == "policy":
        raw["policy"] = {"frame_rate": 30, "hold_seconds": 0.5}
    elif change == "missing":
        raw["inputs"] = {}
    elif change == "stale":
        (first.parent / "source.bin").write_bytes(b"changed")
    elif change == "relative":
        raw["inputs"] = {"source.bin": "0" * 64}
    else:
        raw["candidates"] = [{"cue": {"text": "い", "start": 0.5, "end": 2}, "x": 540, "y": 480}]
    second.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        fcp_measured.combine_measured_captions(
            [(first, file_digest(first)), (second, file_digest(second))],
            output,
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(2),
            calibration=(first.parent / "source.bin", file_digest(first.parent / "source.bin")),
        )
    assert not output.exists()


def test_combine_refuses_existing_output(parts: tuple[Path, Path, Path]) -> None:
    first, second, output = parts
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        fcp_measured.combine_measured_captions(
            [(first, file_digest(first)), (second, file_digest(second))],
            output,
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(2),
            calibration=(first.parent / "source.bin", file_digest(first.parent / "source.bin")),
        )
    assert output.read_bytes() == b"preserve"


def test_combine_requires_references(parts: tuple[Path, Path, Path]) -> None:
    with pytest.raises(ValueError, match="bounded"):
        fcp_measured.combine_measured_captions(
            [],
            parts[2],
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(2),
        )


@pytest.mark.parametrize(
    ("width", "height", "rate", "duration"),
    [
        (0, 1920, 30, Fraction(2)),
        (1080, 0, 30, Fraction(2)),
        (True, 1920, 30, Fraction(2)),
        (8193, 1920, 30, Fraction(2)),
        (1080, 1920, 0, Fraction(2)),
        (1080, 1920, 241, Fraction(2)),
        (1080, 1920, 30, Fraction(0)),
        (1080, 1920, 30, Fraction(121)),
    ],
)
def test_shared_api_rejects_invalid_render_bounds(
    parts: tuple[Path, Path, Path],
    width: int,
    height: int,
    rate: int,
    duration: Fraction,
) -> None:
    first, second, output = parts
    with pytest.raises(ValueError, match="bounds"):
        fcp_measured.combine_measured_captions(
            [(first, file_digest(first)), (second, file_digest(second))],
            output,
            width=width,
            height=height,
            rate=rate,
            duration=duration,
        )
    assert not output.exists()


def test_common_render_calibration_is_required(parts: tuple[Path, Path, Path]) -> None:
    first, second, output = parts
    with pytest.raises(ValueError, match="calibration is required"):
        fcp_measured.combine_measured_captions(
            [(first, file_digest(first)), (second, file_digest(second))],
            output,
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(2),
        )


def test_independent_calibration_cannot_be_combined(parts: tuple[Path, Path, Path]) -> None:
    first, second, output = parts
    other = second.parent / "other-calibration.bin"
    other.write_bytes(b"different calibration")
    raw = record(json.loads(second.read_text(encoding="utf-8")))
    raw["inputs"] = {str(other): file_digest(other)}
    second.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="share the pinned render calibration"):
        fcp_measured.combine_measured_captions(
            [(first, file_digest(first)), (second, file_digest(second))],
            output,
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(2),
            calibration=(first.parent / "source.bin", file_digest(first.parent / "source.bin")),
        )
    assert not output.exists()


def test_combine_preserves_total_budget(
    parts: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second, output = parts
    monkeypatch.setattr(fcp_measured, "MAX_TOKENS", 1)
    with pytest.raises(ValueError, match="budget"):
        fcp_measured.combine_measured_captions(
            [(first, file_digest(first)), (second, file_digest(second))],
            output,
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(2),
            calibration=(first.parent / "source.bin", file_digest(first.parent / "source.bin")),
        )


def test_combine_cli(parts: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    first, second, output = parts
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "combine",
            str(output),
            "--candidate",
            str(first),
            file_digest(first),
            "--candidate",
            str(second),
            file_digest(second),
            "--calibration",
            str(first.parent / "source.bin"),
            file_digest(first.parent / "source.bin"),
            "--width",
            "1080",
            "--height",
            "1920",
            "--rate",
            "30",
            "--frames",
            "60",
        ],
    )
    runpy.run_path(fcp_measured.__file__, run_name="__main__")
    assert output.is_file()


def test_combine_cli_rejects_zero_rate(
    parts: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second, output = parts
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "combine",
            str(output),
            "--candidate",
            str(first),
            file_digest(first),
            "--candidate",
            str(second),
            file_digest(second),
            "--calibration",
            str(first.parent / "source.bin"),
            file_digest(first.parent / "source.bin"),
            "--width",
            "1080",
            "--height",
            "1920",
            "--rate",
            "0",
            "--frames",
            "60",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        fcp_measured.main()


def _with_hold(path: Path, candidates: list[dict[str, object]]) -> None:
    raw = record(json.loads(path.read_text(encoding="utf-8")))
    raw["policy"] = {"frame_rate": 30, "hold_seconds": 0.08}
    raw["candidates"] = candidates
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_combine_trims_only_the_boundary_hold(parts: tuple[Path, Path, Path]) -> None:
    first, second, output = parts
    _with_hold(first, [{"cue": {"text": "あ", "start": 0, "end": 1.05}, "x": 540, "y": 500}])
    _with_hold(second, [{"cue": {"text": "い", "start": 1, "end": 2}, "x": 540, "y": 480}])
    fcp_measured.combine_measured_captions(
        [(first, file_digest(first)), (second, file_digest(second))],
        output,
        width=1080,
        height=1920,
        rate=30,
        duration=Fraction(2),
        calibration=(first.parent / "source.bin", file_digest(first.parent / "source.bin")),
    )
    raw = record(json.loads(output.read_text(encoding="utf-8")))
    assert raw["candidates"] == [
        {"cue": {"text": "あ", "start": 0, "end": 1}, "x": 540, "y": 500},
        {"cue": {"text": "い", "start": 1, "end": 2}, "x": 540, "y": 480},
    ]


def test_combine_rejects_overlap_longer_than_hold(parts: tuple[Path, Path, Path]) -> None:
    first, second, output = parts
    _with_hold(first, [{"cue": {"text": "あ", "start": 0, "end": 1.2}, "x": 540, "y": 500}])
    _with_hold(second, [{"cue": {"text": "い", "start": 1, "end": 2}, "x": 540, "y": 480}])
    with pytest.raises(ValueError, match="overlap"):
        fcp_measured.combine_measured_captions(
            [(first, file_digest(first)), (second, file_digest(second))],
            output,
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(2),
            calibration=(first.parent / "source.bin", file_digest(first.parent / "source.bin")),
        )


def test_hidden_prefix_retained_by_extension_is_dropped() -> None:
    first = PositionedCue(CaptionCue("ッ", 1.81, 1.82), 540, 500)
    second = PositionedCue(CaptionCue("ッパ", 1.82, 2.0), 540, 500)
    kept, hidden = fcp_measured.drop_retained_hidden_prefixes([first, second], rate=30)
    assert kept == [second]
    assert hidden == 1


def test_hidden_caption_without_extension_is_kept() -> None:
    first = PositionedCue(CaptionCue("ッ", 1.81, 1.82), 540, 500)
    second = PositionedCue(CaptionCue("パ", 1.82, 2.0), 540, 500)
    kept, hidden = fcp_measured.drop_retained_hidden_prefixes([first, second], rate=30)
    assert kept == [first, second]
    assert hidden == 0


def test_hidden_caption_is_widened_to_one_frame() -> None:
    hidden = PositionedCue(CaptionCue("あん", 3.6133333333, 3.6333333333), 540, 300)
    following = PositionedCue(CaptionCue("まり", 3.6333333333, 3.75), 540, 200)
    widened, count = fcp_measured.ensure_display_frames(
        [hidden, following], rate=30, duration=Fraction(10)
    )
    assert count == 1
    assert [(item.cue.text, item.cue.start, item.cue.end) for item in widened] == [
        ("あん", 3.6133333333333333, 3.6666666666666665),
        ("まり", 3.6666666666666665, 3.75),
    ]


def test_visible_captions_are_unchanged() -> None:
    visible = PositionedCue(CaptionCue("あ", 1.0, 1.5), 540, 300)
    unchanged, count = fcp_measured.ensure_display_frames([visible], rate=30, duration=Fraction(2))
    assert unchanged == [visible]
    assert count == 0


@pytest.mark.parametrize("rate", [24, 30, 60])
def test_dense_independent_cues_do_not_accumulate_display_delay(rate: int) -> None:
    captions = [
        PositionedCue(CaptionCue("first", 0.001, 0.002), 540, 300),
        PositionedCue(CaptionCue("second", 0.002, 0.003), 540, 300),
        PositionedCue(CaptionCue("third", 0.003, 0.1), 540, 300),
    ]
    with pytest.raises(ValueError, match="one-frame display-delay"):
        fcp_measured.ensure_display_frames(captions, rate=rate, duration=Fraction(1))


def test_final_subframe_cannot_be_extended_beyond_duration() -> None:
    caption = PositionedCue(CaptionCue("last", 0.99, 1), 540, 300)
    with pytest.raises(ValueError, match="no frame support"):
        fcp_measured.ensure_display_frames([caption], rate=30, duration=Fraction(1))
