import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay import delivery_captions as captions
from replay.core import ReplayError

RATE = 30


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def window(directory: Path, index: int, first: int, frames: int) -> None:
    """A completed synthetic foreground window over `frames` 30fps source frames."""
    pts = [(first + offset) * 3000 for offset in range(frames)]
    clock = {"pts": pts, "timeBase": "1/90000", "lastDuration": 3000}
    write(directory / "configs" / f"{index:04d}.json", {"clock": clock})
    write(directory / f"evidence-{index:04d}" / "provenance.json", {"frames": frames})


def token(text: str, start: float, end: float) -> captions.Token:
    return captions.Token(text, start, end, "u")


def test_read_cuts_requires_contiguous_output(tmp_path: Path) -> None:
    path = write(
        tmp_path / "cuts.json",
        {
            "cuts": [
                {"source_start_frame": 0, "source_end_frame": 30, "output_start_frame": 0},
                {"source_start_frame": 60, "source_end_frame": 90, "output_start_frame": 31},
            ]
        },
    )
    with pytest.raises(ReplayError, match="contiguous"):
        captions.read_cuts(path)


def test_read_cuts_requires_source_order(tmp_path: Path) -> None:
    path = write(
        tmp_path / "cuts.json",
        {
            "cuts": [
                {"source_start_frame": 60, "source_end_frame": 90, "output_start_frame": 0},
                {"source_start_frame": 0, "source_end_frame": 30, "output_start_frame": 30},
            ]
        },
    )
    with pytest.raises(ReplayError, match="ordered"):
        captions.read_cuts(path)


def test_read_cuts_rejects_empty_and_non_integer(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="no cuts"):
        captions.read_cuts(write(tmp_path / "a.json", {"cuts": []}))
    bad = {"cuts": [{"source_start_frame": 0.5, "source_end_frame": 3, "output_start_frame": 0}]}
    with pytest.raises(ReplayError, match="integer frames"):
        captions.read_cuts(write(tmp_path / "b.json", bad))


def test_read_tokens_validates_rows(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="no tokens"):
        captions.read_tokens(write(tmp_path / "a.json", {"tokens": []}))
    bad = {"tokens": [{"text": "a", "start": 2, "end": 1, "utterance": "u"}]}
    with pytest.raises(ReplayError, match="Invalid output-clock token"):
        captions.read_tokens(write(tmp_path / "b.json", bad))


def test_window_index_orders_and_rejects_overlap(tmp_path: Path) -> None:
    window(tmp_path / "fg", 0, 128, 128)
    window(tmp_path / "fg", 1, 0, 128)
    windows = captions.window_index([tmp_path / "fg"], [])
    assert [item.start for item in windows] == [Fraction(0), Fraction(128, 30)]
    window(tmp_path / "other", 0, 100, 10)
    extra = (
        tmp_path / "other" / "configs" / "0000.json",
        tmp_path / "other" / "evidence-0000" / "provenance.json",
    )
    with pytest.raises(ReplayError, match="overlap"):
        captions.window_index([tmp_path / "fg"], [extra])


def test_incomplete_window_is_marked(tmp_path: Path) -> None:
    clock = {"pts": [0], "timeBase": "1/90000", "lastDuration": 3000}
    config = write(tmp_path / "configs" / "0000.json", {"clock": clock})
    window = captions.read_window(config, tmp_path / "missing" / "provenance.json")
    assert window.complete is False


def test_window_without_clock_is_rejected(tmp_path: Path) -> None:
    config = write(tmp_path / "config.json", {"clock": {"pts": []}})
    with pytest.raises(ReplayError, match="exact clock"):
        captions.read_window(config, tmp_path / "provenance.json")


def test_straddles_requires_a_full_hold_gap() -> None:
    tokens = [token("a", 0.0, 1.0), token("b", 1.05, 1.5), token("c", 2.0, 2.5)]
    starts = [item.start for item in tokens]
    assert captions._straddles(tokens, starts, 1.0) is True
    assert captions._straddles(tokens, starts, 1.0, held=False) is False
    assert captions._straddles(tokens, starts, 1.2) is True
    assert captions._straddles(tokens, starts, 1.6) is False
    assert captions._straddles(tokens, starts, 0.0) is False


def test_split_cut_prefers_hold_safe_gaps() -> None:
    tokens = [token(str(index), index * 0.5, index * 0.5 + 0.3) for index in range(120)]
    starts = [item.start for item in tokens]
    pieces = captions.split_cut(captions.Cut(0, 1800, 0), tokens, starts, RATE)
    assert [(piece.source_start, piece.source_end, piece.output_start) for piece in pieces] == [
        (0, 714, 0),
        (714, 1254, 714),
        (1254, 1800, 1254),
    ]


def test_split_cut_falls_back_to_back_to_back_tokens() -> None:
    tokens = [token(str(index), index * 0.2, index * 0.2 + 0.2) for index in range(60)]
    starts = [item.start for item in tokens]
    pieces = captions.split_cut(captions.Cut(0, 360, 0), tokens, starts, RATE)
    assert [(piece.source_start, piece.source_end) for piece in pieces] == [(0, 180), (180, 360)]


def test_split_cut_fails_without_any_boundary() -> None:
    tokens = [token("long", 0.0, 12.5)] + [token(str(index), 12.0, 12.1) for index in range(60)]
    starts = [item.start for item in tokens]
    with pytest.raises(ReplayError, match="No token-free split point"):
        captions.split_cut(captions.Cut(0, 400, 0), tokens, starts, RATE)


def test_native_partition_retains_dense_overlapping_tokens_without_legacy_splitting() -> None:
    tokens = [token("long", 0.0, 12.5)] + [token(str(index), 12.0, 12.1) for index in range(60)]
    segments = captions.partition(
        [captions.Cut(0, 400, 0)],
        tokens,
        (),
        rate=RATE,
        require_foreground=False,
        maximum_job_tokens=4096,
    )
    assert len(segments) == 1
    assert segments[0].frames == 400
    assert segments[0].jobs[0].tokens == tuple(tokens)
    assert segments[0].jobs[0].cuts == (captions.Cut(0, 400, 0),)


@pytest.mark.parametrize("budget", [0, 4097, True])
def test_partition_rejects_unbounded_token_budget(budget: int) -> None:
    with pytest.raises(ReplayError, match="budget"):
        captions.partition([], [], (), rate=RATE, maximum_job_tokens=budget)


def test_native_dense_budget_does_not_remove_duration_limit() -> None:
    tokens = [token(str(index), index * 0.5, index * 0.5 + 0.3) for index in range(480)]
    starts = [item.start for item in tokens]
    pieces = captions.split_cut(captions.Cut(0, 7200, 0), tokens, starts, RATE, maximum_tokens=4096)
    assert sum(piece.frames for piece in pieces) == 7200
    assert max(piece.frames for piece in pieces) <= 3300


def test_partition_builds_segments_with_hold_support(tmp_path: Path) -> None:
    window(tmp_path / "fg", 0, 0, 128)
    window(tmp_path / "fg", 1, 128, 128)
    windows = captions.window_index([tmp_path / "fg"], [])
    cuts = [captions.Cut(0, 90, 0), captions.Cut(120, 240, 90)]
    tokens = [token("a", 0.5, 2.98), token("b", 3.5, 4.0)]
    segments = captions.partition(cuts, tokens, windows, rate=RATE)
    assert len(segments) == 1
    job = segments[0].jobs[0]
    assert [(cut.source_start, cut.source_end) for cut in job.cuts] == [(0, 90), (120, 240)]
    assert len(job.windows) == 2
    assert segments[0].frames == 210


def test_supported_extends_into_following_frames() -> None:
    group = [captions.Cut(0, 90, 0)]
    following = [captions.Cut(120, 240, 90)]
    spans = captions._supported(group, following, [token("a", 0.5, 2.99)], RATE)
    assert spans == (captions.Cut(0, 90, 0), captions.Cut(120, 123, 90))


def test_supported_without_tokens_keeps_group() -> None:
    assert captions._supported([captions.Cut(0, 9, 0)], [], [], RATE) == (captions.Cut(0, 9, 0),)


def test_partition_rejects_unordered_tokens(tmp_path: Path) -> None:
    window(tmp_path / "fg", 0, 0, 128)
    windows = captions.window_index([tmp_path / "fg"], [])
    tokens = [token("b", 2.0, 2.5), token("a", 1.0, 1.5)]
    with pytest.raises(ReplayError, match="ordered by onset"):
        captions.partition([captions.Cut(0, 90, 0)], tokens, windows, rate=RATE)


def test_partition_requires_foreground_support() -> None:
    with pytest.raises(ReplayError, match="foreground support"):
        captions.partition([captions.Cut(0, 90, 0)], [token("a", 0.5, 1.0)], [], rate=RATE)


def test_partition_rejects_tokens_outside_cuts(tmp_path: Path) -> None:
    window(tmp_path / "fg", 0, 0, 128)
    windows = captions.window_index([tmp_path / "fg"], [])
    tokens = [token("a", 0.5, 1.0), token("late", 9.0, 9.5)]
    with pytest.raises(ReplayError, match="outside the cut plan"):
        captions.partition([captions.Cut(0, 90, 0)], tokens, windows, rate=RATE)


def plan_directory(tmp_path: Path) -> Path:
    window(tmp_path / "fg", 0, 0, 128)
    windows = captions.window_index([tmp_path / "fg"], [])
    tokens = [token("a", 0.5, 1.0), token("b", 1.5, 2.0)]
    segments = captions.partition([captions.Cut(30, 120, 0)], tokens, windows, rate=RATE)
    native, calibration = tmp_path / "native", tmp_path / "calibration.json"
    native.write_bytes(b"native")
    calibration.write_text("{}", encoding="utf-8")
    settings = captions.Settings(RATE, native, calibration, (0, 0, 1080, 1920), 540, 24)
    captions.write_jobs(tmp_path / "plan", segments, settings, workers=2)
    return tmp_path / "plan"


def test_write_jobs_uses_segment_local_clocks(tmp_path: Path) -> None:
    directory = plan_directory(tmp_path)
    manifest = json.loads((directory / "manifest-0000.json").read_text(encoding="utf-8"))
    assert manifest["spans"] == [
        {
            "sourceStart": {"value": 30, "timescale": 30},
            "outputStart": {"value": 0, "timescale": 30},
            "duration": {"value": 90, "timescale": 30},
        }
    ]
    assert json.loads((directory / "segments.json").read_text(encoding="utf-8")) == {
        "version": 1,
        "rate": 30,
        "segments": [{"segment": 0, "offsetFrame": 0, "frames": 90, "jobs": [0]}],
    }
    assert sorted(path.name for path in directory.glob("measure-*.json")) == ["measure-0-0000.json"]


def test_write_jobs_rejects_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="1-8 workers"):
        captions.write_jobs(
            tmp_path / "plan",
            [],
            captions.Settings(30, tmp_path, tmp_path, (0, 0, 1, 1), 0, 0),
            workers=0,
        )


def test_retry_preserves_failed_attempts(tmp_path: Path) -> None:
    directory = plan_directory(tmp_path)
    (directory / "measurement-0000").mkdir()
    workflows = captions.retry(directory, workers=1)
    assert (directory / "measurement-0000.failed-0").is_dir()
    stages = json.loads(workflows[0].read_text(encoding="utf-8"))["stages"]
    assert [stage["name"] for stage in stages] == ["measure-0000"]


def test_retry_without_pending_jobs_fails(tmp_path: Path) -> None:
    directory = plan_directory(tmp_path)
    write(directory / "measurement-0000" / "candidate.json", {})
    with pytest.raises(ReplayError, match="already has a candidate"):
        captions.retry(directory, workers=1)


def test_combine_waits_for_every_job(tmp_path: Path) -> None:
    directory = plan_directory(tmp_path)
    with pytest.raises(ReplayError, match="unfinished"):
        captions.write_combine(
            directory, width=1080, height=1920, calibration=tmp_path / "calibration.json"
        )


def test_combine_writes_one_stage_per_segment(tmp_path: Path) -> None:
    directory = plan_directory(tmp_path)
    write(directory / "measurement-0000" / "candidate.json", {})
    path = captions.write_combine(
        directory, width=1080, height=1920, calibration=tmp_path / "calibration.json"
    )
    stage = json.loads(path.read_text(encoding="utf-8"))["stages"][0]
    assert stage["name"] == "combine-0000"
    assert stage["command"][3:12] == [
        str(directory / "combined-0000.json"),
        "--width",
        "1080",
        "--height",
        "1920",
        "--rate",
        "30",
        "--frames",
        "90",
    ]


def test_project_records_omitted_and_zero_length_labels(tmp_path: Path) -> None:
    labels = write(
        tmp_path / "labels.json",
        {
            "labels": [
                {"text": "a", "start": 1.2, "end": 1.5, "provenance": "p:0"},
                {"text": "ー", "start": 2.0, "end": 2.0, "provenance": "p:1"},
                {"text": "gone", "start": 5.0, "end": 5.5, "provenance": "p:2"},
            ]
        },
    )
    cuts = write(
        tmp_path / "cuts.json",
        {"cuts": [{"source_start_frame": 30, "source_end_frame": 90, "output_start_frame": 0}]},
    )
    output = captions.project(labels, cuts, tmp_path / "tokens.json", rate=RATE)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["tokens"] == [
        {"text": "a", "start": 0.19999999999999996, "end": 0.5, "utterance": "p:0:cut-0"}
    ]
    assert [row["text"] for row in data["omitted"]] == ["gone"]
    assert data["zeroLengthLabels"] == [
        {"text": "ー", "start": 2.0, "end": 2.0, "provenance": "p:1"}
    ]


def test_read_labels_rejects_invalid_rows(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="no labels"):
        captions.read_labels(write(tmp_path / "a.json", {"labels": []}))
    bad = {"labels": [{"text": "a", "start": "0", "end": 1, "provenance": "p"}]}
    with pytest.raises(ReplayError, match="Invalid source-clock label"):
        captions.read_labels(write(tmp_path / "b.json", bad))


def test_window_list_pairs(tmp_path: Path) -> None:
    listing = tmp_path / "windows.txt"
    listing.write_text("/a/config.json /a/provenance.json\n", encoding="utf-8")
    assert captions.read_window_list(listing) == [
        (Path("/a/config.json"), Path("/a/provenance.json"))
    ]
    assert captions.read_window_list(None) == []
    with pytest.raises(ReplayError, match="PROVENANCE pairs"):
        captions._pairs(["/only-one"])


def test_cli_plan_project_retry_and_combine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    window(tmp_path / "fg", 0, 0, 128)
    tokens = write(
        tmp_path / "tokens.json",
        {"tokens": [{"text": "a", "start": 0.5, "end": 1.0, "utterance": "u"}]},
    )
    cuts = write(
        tmp_path / "cuts.json",
        {"cuts": [{"source_start_frame": 0, "source_end_frame": 90, "output_start_frame": 0}]},
    )
    native, calibration = tmp_path / "native", tmp_path / "calibration.json"
    native.write_bytes(b"native")
    calibration.write_text("{}", encoding="utf-8")
    plan = tmp_path / "plan"
    args = ["plan", str(tokens), str(cuts), str(plan), "--foreground", str(tmp_path / "fg")]
    args += ["--native", str(native), "--calibration", str(calibration), "--workers", "1"]
    assert captions.main(args) == 0
    assert captions.main(["retry", str(plan), "--workers", "1"]) == 0
    write(plan / "measurement-0000" / "candidate.json", {})
    assert captions.main(["combine", str(plan), "--calibration", str(calibration)]) == 0
    labels = write(
        tmp_path / "labels.json",
        {"labels": [{"text": "a", "start": 0.5, "end": 1.0, "provenance": "p"}]},
    )
    assert captions.main(["project", str(labels), str(cuts), str(tmp_path / "projected.json")]) == 0
    assert capsys.readouterr().out.strip().endswith("projected.json")


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert captions.main(["combine", str(tmp_path), "--calibration", str(tmp_path / "c")]) == 1
    assert "DELIVERY_CAPTIONS_ERROR" in capsys.readouterr().err


def test_retry_skips_jobs_owned_by_live_workflows(tmp_path: Path) -> None:
    directory = plan_directory(tmp_path)
    live = write(tmp_path / "live.json", {"stages": [{"name": "measure-0000"}]})
    with pytest.raises(ReplayError, match="already has a candidate"):
        captions.retry(directory, workers=1, exclude=[live])


def test_rewrite_rebinds_windows_of_unfinished_jobs(tmp_path: Path) -> None:
    directory = plan_directory(tmp_path)
    window(tmp_path / "rebound", 0, 0, 128)
    windows = captions.window_index([tmp_path / "rebound"], [])
    assert captions.rewrite(directory, windows, rate=RATE) == [0]
    manifest = json.loads((directory / "manifest-0000.json").read_text(encoding="utf-8"))
    assert manifest["foreground"][0]["config"]["path"] == str(
        tmp_path / "rebound" / "configs" / "0000.json"
    )
    assert (directory / "manifest-0000.json.orig-0").is_file()
    assert captions.rewrite(directory, windows, rate=RATE) == []


def test_rewrite_keeps_finished_jobs(tmp_path: Path) -> None:
    directory = plan_directory(tmp_path)
    write(directory / "measurement-0000" / "candidate.json", {})
    window(tmp_path / "rebound", 0, 0, 128)
    windows = captions.window_index([tmp_path / "rebound"], [])
    assert captions.rewrite(directory, windows, rate=RATE) == []


def test_rewrite_requires_support(tmp_path: Path) -> None:
    directory = plan_directory(tmp_path)
    with pytest.raises(ReplayError, match="foreground support"):
        captions.rewrite(directory, [], rate=RATE)


def test_span_cut_requires_integer_frames() -> None:
    span = {"sourceStart": {"value": 1.5}, "outputStart": {"value": 0}, "duration": {"value": 3}}
    with pytest.raises(ReplayError, match="integer frame"):
        captions._span_cut(span, RATE)


def test_cli_rewrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = plan_directory(tmp_path)
    window(tmp_path / "rebound", 0, 0, 128)
    args = ["rewrite", str(directory), "--foreground", str(tmp_path / "rebound")]
    assert captions.main(args) == 0
    assert capsys.readouterr().out.strip() == "[0]"


def test_incremental_plan_adds_jobs_as_windows_complete(tmp_path: Path) -> None:
    window(tmp_path / "fg", 0, 0, 128)
    window(tmp_path / "fg", 1, 128, 128)
    (tmp_path / "fg" / "evidence-0001" / "provenance.json").unlink()
    tokens = [token("a", 0.5, 1.0), token("b", 3.5, 4.0)]
    cuts = [captions.Cut(0, 90, 0), captions.Cut(150, 200, 90)]
    native, calibration = tmp_path / "native", tmp_path / "calibration.json"
    native.write_bytes(b"native")
    calibration.write_text("{}", encoding="utf-8")
    settings = captions.Settings(RATE, native, calibration, (0, 0, 1080, 1920), 540, 24)
    segments = captions.partition(
        cuts, tokens, captions.window_index([tmp_path / "fg"], []), rate=RATE
    )
    first = captions.write_jobs(tmp_path / "plan", segments, settings, workers=1)
    assert [path.name for path in first] == []
    assert sorted(path.name for path in (tmp_path / "plan").glob("manifest-*.json")) == []
    write(tmp_path / "fg" / "evidence-0001" / "provenance.json", {})
    segments = captions.partition(
        cuts, tokens, captions.window_index([tmp_path / "fg"], []), rate=RATE
    )
    second = captions.write_jobs(tmp_path / "plan", segments, settings, workers=1)
    assert [path.parent.name for path in second] == ["round-0000"]
    assert sorted(path.name for path in (tmp_path / "plan").glob("manifest-*.json")) == [
        "manifest-0000.json"
    ]
    assert captions.write_jobs(tmp_path / "plan", segments, settings, workers=1) == []


def test_incremental_plan_rejects_changed_partition(tmp_path: Path) -> None:
    directory = plan_directory(tmp_path)
    window(tmp_path / "fg2", 0, 0, 128)
    segments = captions.partition(
        [captions.Cut(30, 150, 0)],
        [token("a", 0.5, 1.0)],
        captions.window_index([tmp_path / "fg2"], []),
        rate=RATE,
    )
    settings = captions.Settings(
        RATE, tmp_path / "native", tmp_path / "calibration.json", (0, 0, 1080, 1920), 540, 24
    )
    with pytest.raises(ReplayError, match="partition changed"):
        captions.write_jobs(directory, segments, settings, workers=1)


def test_same_frame_onsets_merge_with_union_clock() -> None:
    tokens = [token("き", 1.01, 1.02), token("は", 1.02, 1.1), token("い", 2.0, 2.2)]
    merged, records = captions.merge_same_frame_onsets(tokens, rate=RATE)
    assert merged == (
        captions.Token("きは", 1.01, 1.1, "u"),
        captions.Token("い", 2.0, 2.2, "u"),
    )
    assert records == [
        {
            "merged": [
                {"text": "き", "start": 1.01, "end": 1.02, "utterance": "u"},
                {"text": "は", "start": 1.02, "end": 1.1, "utterance": "u"},
            ],
            "into": {"text": "きは", "start": 1.01, "end": 1.1, "utterance": "u"},
        }
    ]
