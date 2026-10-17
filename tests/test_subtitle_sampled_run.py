"""The chunk runner audits each attempt, escalates once, and never accepts a failed audit."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from replay import subtitle_sampled_run as runner
from replay.caption_io import FrameObservation
from replay.caption_masks import TextDetection, TextRectangle

BOX = {"x": 0, "y": 100, "width": 1000, "height": 200}
PLAN: dict[str, object] = {
    "frames": [{"region": BOX}, {"classification": "unknown"}, {"region": BOX}],
    "sampling": {"rawOcr": False},
    "stageSeconds": {"ocr": 1.0},
}
CAPTION = TextDetection("決着", 1.0, TextRectangle(100, 150, 60, 30))
SMEAR = TextDetection("Ter.ne", 0.3, TextRectangle(100, 150, 60, 30))


def _tools(clean_stride: int, strides: list[int]) -> runner.Collaborators:
    def plan(stride: int, directory: Path) -> dict[str, object]:
        strides.append(stride)
        return PLAN

    def render(_plan: dict[str, object], directory: Path) -> Path:
        video = directory / "candidate.mp4"
        video.write_bytes(b"video")
        return video

    def audit(video: Path, indices: Sequence[int]) -> Mapping[int, FrameObservation]:
        stride = strides[-1]
        line = SMEAR if stride <= clean_stride else CAPTION
        return {index: FrameObservation(0.0, 0.0, 1080, 1920, (line,)) for index in indices}

    return runner.Collaborators(plan, render, audit)


def test_a_passing_sampled_attempt_is_accepted_without_escalation(tmp_path: Path) -> None:
    strides: list[int] = []
    receipt = runner.run(tmp_path / "chunk", _tools(6, strides), pins={"native": "abc"})
    assert strides == [6]
    assert receipt["accepted"] == "attempt-sampled"
    assert receipt["escalated"] is False
    assert receipt["pins"] == {"native": "abc"}
    assert receipt["qualityApproved"] is False
    stored = json.loads((tmp_path / "chunk" / "chunk-receipt.json").read_text(encoding="utf-8"))
    assert stored == receipt


def test_a_failed_audit_escalates_once_to_full_ocr(tmp_path: Path) -> None:
    strides: list[int] = []
    receipt = runner.run(tmp_path / "chunk", _tools(1, strides))
    assert strides == [6, 1]
    assert receipt["accepted"] == "attempt-full"
    assert receipt["escalated"] is True
    attempts = receipt["attempts"]
    assert isinstance(attempts, list)
    assert attempts[0]["passed"] is False
    assert attempts[0]["residualLines"] == [
        {"frame": 0, "text": "決着", "confidence": 1.0, "box": [100.0, 150.0, 60.0, 30.0]},
        {"frame": 2, "text": "決着", "confidence": 1.0, "box": [100.0, 150.0, 60.0, 30.0]},
    ]
    assert attempts[1]["passed"] is True


def test_nothing_is_accepted_when_the_full_attempt_also_fails(tmp_path: Path) -> None:
    strides: list[int] = []
    with pytest.raises(ValueError, match="nothing accepted"):
        runner.run(tmp_path / "chunk", _tools(0, strides))
    stored = json.loads((tmp_path / "chunk" / "chunk-receipt.json").read_text(encoding="utf-8"))
    assert stored["accepted"] is None
    assert stored["escalated"] is True


def test_invalid_strides_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stride"):
        runner.run(tmp_path / "a", _tools(6, []), stride=1)
    with pytest.raises(ValueError, match="stride"):
        runner.run(tmp_path / "b", _tools(6, []), audit_stride=0)


def test_a_plan_without_frame_entries_is_rejected(tmp_path: Path) -> None:
    broken = runner.Collaborators(
        plan=lambda _stride, _directory: {"frames": "none"},
        render=lambda _plan, directory: directory / "x.mp4",
        audit=lambda _video, _indices: {},
    )
    with pytest.raises(ValueError, match="per-frame"):
        runner.run(tmp_path / "chunk", broken)


def _with_colour(tools: runner.Collaborators, counts: dict[int, list[int]]) -> runner.Collaborators:
    def colour(_plan: dict[str, object], video: Path, indices: Sequence[int]) -> Sequence[int]:
        return counts[1 if video.parent.name == "attempt-full" else 0][: len(indices)]

    return runner.Collaborators(tools.plan, tools.render, tools.audit, colour)


def test_colour_residual_below_the_gate_is_recorded_and_accepted(tmp_path: Path) -> None:
    tools = _with_colour(_tools(6, []), {0: [0, 39], 1: [0, 0]})
    receipt = runner.run(tmp_path / "chunk", tools)
    attempts = receipt["attempts"]
    assert isinstance(attempts, list)
    assert attempts[0]["colourResidual"] == {"gate": 40, "flaggedFrames": [], "maximum": 39}
    assert receipt["accepted"] == "attempt-sampled"


def test_colour_residual_at_the_gate_escalates(tmp_path: Path) -> None:
    tools = _with_colour(_tools(6, []), {0: [0, 40], 1: [0, 5]})
    receipt = runner.run(tmp_path / "chunk", tools)
    attempts = receipt["attempts"]
    assert isinstance(attempts, list)
    assert attempts[0]["colourResidual"] == {"gate": 40, "flaggedFrames": [2], "maximum": 40}
    assert (receipt["accepted"], receipt["escalated"]) == ("attempt-full", True)


def _coloured_tools(flags: list[bool], strides: list[int]) -> runner.Collaborators:
    base = _tools(0, strides)

    def caption_coloured(residuals: Sequence[object]) -> list[bool]:
        return [flags[0] for _ in residuals]

    return runner.Collaborators(
        base.plan, base.render, base.audit, caption_coloured=caption_coloured
    )


def test_a_residual_without_caption_colour_is_recorded_as_interface_text(tmp_path: Path) -> None:
    receipt = runner.run(tmp_path / "chunk", _coloured_tools([False], []))
    attempts = receipt["attempts"]
    assert isinstance(attempts, list)
    assert attempts[0]["residualLines"] == []
    assert attempts[0]["interfaceText"] == [
        {"frame": 0, "text": "決着", "confidence": 1.0, "box": [100.0, 150.0, 60.0, 30.0]},
        {"frame": 2, "text": "決着", "confidence": 1.0, "box": [100.0, 150.0, 60.0, 30.0]},
    ]
    assert receipt["accepted"] == "attempt-sampled"


def test_a_caption_coloured_residual_still_fails_the_audit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Audit failed"):
        runner.run(tmp_path / "chunk", _coloured_tools([True], []))
