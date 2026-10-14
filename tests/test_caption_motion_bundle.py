"""Tests for POS-aware, output-clock Motion title bundle planning."""

import json
from pathlib import Path

import pytest

from replay.caption_motion_bundle import main, reflow
from replay.core import ReplayError
from replay.recordings import record


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _inputs(tmp_path: Path, cues: list[dict[str, object]]) -> tuple[Path, Path, Path]:
    cue_path = _write(tmp_path / "cues.json", {"version": 1, "clock": "output", "cues": cues})
    cut_path = _write(
        tmp_path / "cuts.json",
        {
            "version": 1,
            "clock": "source-presentation",
            "fps": 30,
            "cuts": [{"source_start_frame": 0, "source_end_frame": 9000, "output_start_frame": 0}],
        },
    )
    bundle_path = _write(
        tmp_path / "base.json",
        {
            "version": 1,
            "canvas": {"width": 1280, "height": 720, "rate": 30},
            "slot": {"centerX": 884.0, "left": 497.0, "right": 1271.0, "width": 774.0},
            "fontSize": 104,
        },
    )
    return cue_path, cut_path, bundle_path


def test_reflow_regroups_fragments_with_pos_and_keeps_output_clocks(tmp_path: Path) -> None:
    cues = [
        {"text": "フ", "start": 1.00, "end": 1.08},
        {"text": "ォロ", "start": 1.08, "end": 1.20},
        {"text": "ワ", "start": 1.20, "end": 1.28},
        {"text": "ー", "start": 1.28, "end": 1.36},
        {"text": "さん", "start": 1.36, "end": 1.52},
        {"text": "。", "start": 1.52, "end": 1.56},
    ]
    cue_path, cut_path, bundle_path = _inputs(tmp_path, cues)
    output = tmp_path / "candidate.json"
    result = record(reflow(cue_path, cut_path, bundle_path, output))
    raw_parts = result.get("titleParts")
    assert isinstance(raw_parts, list)
    flattened: list[dict[str, object]] = []
    for raw_part in raw_parts:
        placements = record(raw_part).get("placements")
        assert isinstance(placements, list)
        flattened.extend(record(cue) for cue in placements)
    text = [cue["text"] for cue in flattened]
    assert text[0] == "フォロワーさん。"
    start_frame = flattened[0]["startFrame"]
    end_frame = flattened[0]["endFrame"]
    assert type(start_frame) is int and start_frame == 0
    assert type(end_frame) is int and end_frame >= 16
    evidence = record(json.loads(output.with_suffix(".evidence.json").read_text(encoding="utf-8")))
    assert evidence["candidateOnly"] is True
    assert evidence["deliveryComplete"] is False
    assert record(evidence["sourceEvidence"])["sourceCueCount"] == len(cues)
    assert output.is_file()


def test_reflow_clips_display_hold_at_retained_cut_boundary(tmp_path: Path) -> None:
    cues = [
        {"text": "フ", "start": 1.00, "end": 1.08},
        {"text": "ォロ", "start": 1.08, "end": 1.20},
    ]
    cue_path, cut_path, bundle_path = _inputs(tmp_path, cues)
    cuts = json.loads(cut_path.read_text(encoding="utf-8"))
    cuts["cuts"] = [
        {"source_start_frame": 0, "source_end_frame": 33, "output_start_frame": 0},
        {"source_start_frame": 60, "source_end_frame": 9000, "output_start_frame": 33},
    ]
    _write(cut_path, cuts)
    output = tmp_path / "candidate.json"
    reflow(cue_path, cut_path, bundle_path, output)
    evidence = record(json.loads(output.with_suffix(".evidence.json").read_text()))
    evidence = record(evidence["sourceEvidence"])
    assert evidence["displayEndsClippedAtCut"] == 1


@pytest.mark.parametrize(
    "cues",
    [
        [{"text": "x", "start": 1.0, "end": 1.0}],
        [{"text": "x", "start": 2.0, "end": 2.1}, {"text": "y", "start": 1.0, "end": 1.1}],
        [{"text": "\n", "start": 1.0, "end": 1.1}],
    ],
)
def test_reflow_rejects_bad_cue_text_or_order(
    tmp_path: Path, cues: list[dict[str, object]]
) -> None:
    cue_path, cut_path, bundle_path = _inputs(tmp_path, cues)
    with pytest.raises(ReplayError):
        reflow(cue_path, cut_path, bundle_path, tmp_path / "candidate.json")


def test_reflow_rejects_noncontiguous_output_cut_clock(tmp_path: Path) -> None:
    cue_path, cut_path, bundle_path = _inputs(tmp_path, [{"text": "語", "start": 1.0, "end": 1.2}])
    cuts = json.loads(cut_path.read_text(encoding="utf-8"))
    cuts["cuts"].append(
        {"source_start_frame": 9001, "source_end_frame": 9010, "output_start_frame": 9001}
    )
    _write(cut_path, cuts)
    with pytest.raises(ReplayError, match="output clock"):
        reflow(cue_path, cut_path, bundle_path, tmp_path / "candidate.json")


def test_reflow_refuses_output_collision_and_bad_bundle(tmp_path: Path) -> None:
    cue_path, cut_path, bundle_path = _inputs(tmp_path, [{"text": "語", "start": 1.0, "end": 1.2}])
    bad_bundle = _write(tmp_path / "bad-bundle.json", {"version": 1})
    with pytest.raises(ReplayError, match="Malformed"):
        reflow(cue_path, cut_path, bad_bundle, tmp_path / "candidate.json")
    out = tmp_path / "present.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="overwrite"):
        reflow(cue_path, cut_path, bundle_path, out)


def test_reflow_splits_motion_titles_by_count_and_duration(tmp_path: Path) -> None:
    cues = [{"text": "語", "start": float(index), "end": float(index) + 0.2} for index in range(33)]
    cue_path, cut_path, bundle_path = _inputs(tmp_path, cues)
    by_count = record(reflow(cue_path, cut_path, bundle_path, tmp_path / "by-count.json"))
    count_parts = by_count["titleParts"]
    assert isinstance(count_parts, list) and len(count_parts) == 2
    first_placements = record(count_parts[0])["placements"]
    second_placements = record(count_parts[1])["placements"]
    assert isinstance(first_placements, list) and len(first_placements) == 32
    assert isinstance(second_placements, list) and len(second_placements) == 1

    spread = [{"text": "語", "start": start, "end": start + 0.2} for start in (0.0, 60.0, 130.0)]
    cue_path2 = _write(tmp_path / "spread.json", {"version": 1, "clock": "output", "cues": spread})
    by_duration = record(reflow(cue_path2, cut_path, bundle_path, tmp_path / "by-duration.json"))
    duration_parts = by_duration["titleParts"]
    assert isinstance(duration_parts, list) and len(duration_parts) == 2


def test_reflow_rejects_no_displayable_lexical_cues(tmp_path: Path) -> None:
    cue_path, cut_path, bundle_path = _inputs(tmp_path, [{"text": "ー", "start": 1.0, "end": 1.2}])
    with pytest.raises(ReplayError, match="no display cues"):
        reflow(cue_path, cut_path, bundle_path, tmp_path / "candidate.json")


def test_reflow_rejects_invalid_cut_source_order_and_outside_caption(tmp_path: Path) -> None:
    cue_path, cut_path, bundle_path = _inputs(tmp_path, [{"text": "語", "start": 1.0, "end": 1.2}])
    cuts = json.loads(cut_path.read_text(encoding="utf-8"))
    cuts["cuts"] = [
        {"source_start_frame": 20, "source_end_frame": 30, "output_start_frame": 0},
        {"source_start_frame": 10, "source_end_frame": 40, "output_start_frame": 10},
    ]
    _write(cut_path, cuts)
    with pytest.raises(ReplayError, match="unordered"):
        reflow(cue_path, cut_path, bundle_path, tmp_path / "bad-cut.json")

    cuts["cuts"] = [{"source_start_frame": 0, "source_end_frame": 9000, "output_start_frame": 0}]
    _write(cut_path, cuts)
    outside = _write(
        tmp_path / "outside.json",
        {
            "version": 1,
            "clock": "output",
            "cues": [{"text": "語", "start": 301.0, "end": 301.2}],
        },
    )
    with pytest.raises(ReplayError, match="outside"):
        reflow(outside, cut_path, bundle_path, tmp_path / "outside-plan.json")


@pytest.mark.parametrize("contents", ["not-json", "{}", ""])
def test_reflow_rejects_malformed_or_empty_input(tmp_path: Path, contents: str) -> None:
    cue_path, cut_path, bundle_path = _inputs(tmp_path, [{"text": "語", "start": 1.0, "end": 1.2}])
    cue_path.write_text(contents, encoding="utf-8")
    with pytest.raises(ReplayError):
        reflow(cue_path, cut_path, bundle_path, tmp_path / "bad-input.json")


def test_reflow_rejects_symlink_inputs(tmp_path: Path) -> None:
    cue_path, cut_path, bundle_path = _inputs(tmp_path, [{"text": "語", "start": 1.0, "end": 1.2}])
    link = tmp_path / "linked-cues.json"
    link.symlink_to(cue_path)
    with pytest.raises(ReplayError, match="canonical"):
        reflow(link, cut_path, bundle_path, tmp_path / "linked-plan.json")


def test_main_emits_candidate_evidence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cue_path, cut_path, bundle_path = _inputs(tmp_path, [{"text": "語", "start": 1.0, "end": 1.2}])
    output = tmp_path / "cli-plan.json"
    assert main([str(cue_path), str(cut_path), str(bundle_path), str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["sourceCueCount"] == 1
