import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import pytest

from replay.audio_clock import AudioFrameClock
from replay.caption_projection import CutSpan
from replay.caption_repair import ClockLabel
from replay.core import ReplayError
from replay.cut_audio_support import _binding, _cuts, _local, intersect, main, run
from replay.stages import StageError, file_digest


def test_trim_rounds_inward_rebuilds_offsets_and_preserves_internal_gap() -> None:
    clock = AudioFrameClock((1200, 8000), (4000, 6000), 8000, Fraction(1, 8000))
    result = intersect(
        (CutSpan(0, 6, 0), CutSpan(8, 20, 6)),
        (ClockLabel("声", 0.3, 0.5, "raw:1"),),
        clock,
        fps=10,
    )
    assert result.first_frame == 2
    assert result.end_frame == 17
    assert result.removed == ((0, 2), (17, 20))
    assert result.cuts == (CutSpan(2, 6, 0), CutSpan(8, 17, 4))


def test_whole_unsupported_cuts_are_removed_and_offsets_compact() -> None:
    result = intersect(
        (CutSpan(0, 1, 0), CutSpan(3, 8, 1), CutSpan(11, 12, 6)),
        (),
        AudioFrameClock((1600,), (6400,), 8000, Fraction(1, 8000)),
        fps=10,
    )
    assert result.cuts == (CutSpan(3, 8, 0),)
    assert result.removed == ((0, 1), (11, 12))


@pytest.mark.parametrize(
    "label",
    [
        ClockLabel("声", 0.1, 0.21, "a"),
        ClockLabel("長い", 0.1, 0.9, "b"),
        ClockLabel("点", 0.1, 0.1, "c"),
        ClockLabel("末尾", 0.9, 1.1, "d"),
        ClockLabel("末尾点", 1.0, 1.0, "e"),
    ],
)
def test_rejects_speech_and_unresolved_points_in_removed_frames(label: ClockLabel) -> None:
    with pytest.raises(ReplayError, match="recognized"):
        intersect(
            (CutSpan(0, 12, 0),),
            (label,),
            AudioFrameClock((1600,), (6400,), 8000, Fraction(1, 8000)),
            fps=10,
        )


def test_half_open_boundaries_and_nested_labels_protect_maximum_extent() -> None:
    clock = AudioFrameClock((1600,), (6400,), 8000, Fraction(1, 8000))
    labels = (ClockLabel("中", 0.2, 1.0, "a"), ClockLabel("点", 0.2, 0.2, "b"))
    assert intersect((CutSpan(0, 12, 0),), labels, clock, fps=10).removed == ((0, 2), (10, 12))
    with pytest.raises(ReplayError, match="recognized"):
        intersect(
            (CutSpan(0, 12, 0),),
            (ClockLabel("長い", 0.2, 1.1, "a"), ClockLabel("短い", 0.3, 0.4, "b")),
            clock,
            fps=10,
        )


@pytest.mark.parametrize("fps", [0, 121, True])
def test_invalid_fps(fps: int) -> None:
    with pytest.raises(ReplayError, match="frame rate"):
        intersect(
            (CutSpan(0, 10, 0),),
            (),
            AudioFrameClock((0,), (8000,), 8000, Fraction(1, 8000)),
            fps=fps,
        )


@pytest.mark.parametrize(
    "cuts",
    [
        (),
        (CutSpan(0, 2, 1),),
        (CutSpan(0, 4, 0), CutSpan(3, 6, 4)),
        (CutSpan(0, 4, 0), CutSpan(5, 6, 5)),
        (CutSpan(False, 4, 0),),
    ],
)
def test_invalid_cut_order_or_budget(cuts: tuple[CutSpan, ...]) -> None:
    with pytest.raises(ReplayError):
        intersect(cuts, (), AudioFrameClock((0,), (8000,), 8000, Fraction(1, 8000)), fps=10)


def test_empty_support_and_all_removed_are_explicit_failures() -> None:
    with pytest.raises(ReplayError, match="No complete"):
        intersect(
            (CutSpan(0, 10, 0),), (), AudioFrameClock((1,), (1,), 8000, Fraction(1, 8000)), fps=10
        )
    with pytest.raises(ReplayError, match="every retained"):
        intersect(
            (CutSpan(20, 30, 0),),
            (),
            AudioFrameClock((0,), (8000,), 8000, Fraction(1, 8000)),
            fps=10,
        )


def test_label_after_removed_ranges_is_unchanged() -> None:
    assert intersect(
        (CutSpan(0, 10, 0),),
        (ClockLabel("外", 2.0, 2.1, "outside-existing-cuts"),),
        AudioFrameClock((1600,), (8000,), 8000, Fraction(1, 8000)),
        fps=10,
    ).cuts == (CutSpan(2, 10, 0),)


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def bind(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    clock = write(
        tmp_path / "clock.json",
        {
            "streams": [{"sample_rate": "8000", "time_base": "1/8000"}],
            "frames": [{"pts": 1200, "nb_samples": 6400}],
        },
    )
    labels = write(
        tmp_path / "labels.json",
        {
            "clock": "source-presentation",
            "parentSHA256": {str(clock): file_digest(clock)},
            "labels": [asdict(ClockLabel("声", 0.3, 0.7, "raw"))],
        },
    )
    cuts = write(
        tmp_path / "cuts.json",
        {
            "version": 1,
            "clock": "source-presentation",
            "fps": 10,
            "sourceFrames": 10,
            "cuts": [asdict(CutSpan(0, 10, 0))],
            "sourceQuiet": [],
            "parentSHA256": {str(labels): file_digest(labels), str(clock): file_digest(clock)},
        },
    )
    return write(
        tmp_path / "config.json",
        {
            "version": 1,
            "artifacts": {"cuts": bind(cuts), "labels": bind(labels), "audioClock": bind(clock)},
        },
    )


def test_cli_preserves_inputs_and_publishes_separate_candidate(config: Path) -> None:
    output = config.parent / "edits/work/revised.json"
    before = file_digest(config.parent / "cuts.json")
    assert main([str(config), str(output)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["cuts"] == [
        {"source_start_frame": 2, "source_end_frame": 9, "output_start_frame": 0}
    ]
    assert document["audioSupport"]["removedFrames"] == [[0, 2], [9, 10]]
    assert document["deliveryComplete"] is False
    assert document["nonSpeechRemovalVerified"] is False
    assert document["sourceQuiet"] == []
    assert file_digest(config.parent / "cuts.json") == before
    with pytest.raises(ReplayError, match="new repository"):
        run(config, output)


@pytest.mark.parametrize("field,value", [("clock", "edited"), ("fps", True), ("sourceFrames", 0)])
def test_invalid_plan_header(config: Path, field: str, value: object) -> None:
    cuts = config.parent / "cuts.json"
    document = json.loads(cuts.read_text(encoding="utf-8"))
    document[field] = value
    write(cuts, document)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["artifacts"]["cuts"] = bind(cuts)
    write(config, data)
    with pytest.raises(ReplayError, match="candidate cut plan"):
        run(config, config.parent / "edits/work/new.json")


def test_rejects_mismatched_lineage_and_extent(config: Path) -> None:
    cuts = config.parent / "cuts.json"
    document = json.loads(cuts.read_text(encoding="utf-8"))
    document["parentSHA256"] = {}
    write(cuts, document)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["artifacts"]["cuts"] = bind(cuts)
    write(config, data)
    with pytest.raises(ReplayError, match="identical"):
        run(config, config.parent / "edits/work/new.json")


def test_rejects_extent(config: Path) -> None:
    cuts = config.parent / "cuts.json"
    document = json.loads(cuts.read_text(encoding="utf-8"))
    document["sourceFrames"] = 9
    write(cuts, document)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["artifacts"]["cuts"] = bind(cuts)
    write(config, data)
    with pytest.raises(ReplayError, match="extent"):
        run(config, config.parent / "edits/work/new.json")


def test_changed_pin_and_invalid_config(config: Path) -> None:
    (config.parent / "clock.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StageError):
        run(config, config.parent / "edits/work/new.json")
    write(config, {"version": 2, "artifacts": {}})
    with pytest.raises(ReplayError, match="bindings"):
        run(config, config.parent / "edits/work/new.json")


def test_paths_and_external_json_are_not_trusted(config: Path, tmp_path: Path) -> None:
    link = tmp_path / "link.json"
    link.symlink_to(config)
    with pytest.raises(ReplayError, match="canonical"):
        _local(link)
    with pytest.raises(ReplayError, match="canonical"):
        _local(tmp_path.parent / "outside.json", output=True)
    with pytest.raises(ReplayError, match="bounded"):
        _local(tmp_path)
    with pytest.raises(ReplayError, match="new repository"):
        _local(tmp_path / "not-work.json", output=True)
    with pytest.raises(ReplayError, match="explicit"):
        _binding({"path": 1, "sha256": "bad"})


@pytest.mark.parametrize(
    "rows", [None, [], [{"source_start_frame": True}], [{"source_start_frame": 0}]]
)
def test_malformed_cuts(rows: object) -> None:
    with pytest.raises(ReplayError):
        _cuts({"cuts": rows})


def test_speech_inside_decoded_audio_may_start_in_the_trimmed_partial_frame() -> None:
    # 10/6 (2026-10-22): audio from 0.901 s, first whole frame 28 (0.9333 s), speech at 0.931 s.
    clock = AudioFrameClock((7208,), (65536,), 8000, Fraction(1, 8000))
    result = intersect(
        (CutSpan(0, 300, 0),), (ClockLabel("開始", 0.931, 1.354, "a"),), clock, fps=30
    )
    assert result.removed == ((0, 28), (272, 300))
    assert result.cuts == (CutSpan(28, 272, 0),)
    with pytest.raises(ReplayError, match="recognized"):
        intersect((CutSpan(0, 300, 0),), (ClockLabel("点", 0.931, 0.931, "b"),), clock, fps=30)
