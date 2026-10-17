"""The density audit reports unshowable token timing without moving or inventing any time."""

import json
from pathlib import Path

import pytest

from replay import caption_density_audit as audit
from replay.delivery_captions import Cut


def token(text: str, start: float, end: float) -> audit.Token:
    return audit.Token(text, start, end, "u")


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def rows(*items: tuple[str, float, float]) -> dict[str, object]:
    return {"tokens": [{"text": t, "start": s, "end": e, "utterance": "u"} for t, s, e in items]}


def test_coarse_tokens_exceed_the_focused_limit() -> None:
    found = audit.coarse_tokens([token("あ", 0.0, 2.8), token("い", 3.0, 5.9)])
    assert [(f.kind, f.text, f.start) for f in found] == [("coarse-token", "い", 3.0)]


def test_dense_burst_needs_four_onsets_inside_three_frames() -> None:
    dense = [token(str(i), 1.0 + i * 0.02, 1.2 + i * 0.02) for i in range(4)]
    sparse = [token("x", 5.0, 5.1), token("y", 5.5, 5.6)]
    found = audit.dense_bursts([*dense, *sparse])
    assert [(f.kind, f.text, f.start) for f in found] == [("dense-burst", "0123", 1.0)]


def test_identical_onsets_report_neighbours_that_begin_together() -> None:
    found = audit.identical_onsets(
        [token("3", 2.0, 2.2), token("2", 2.0, 2.3), token("a", 3.0, 3.1)]
    )
    assert [(f.kind, f.text, f.end) for f in found] == [("identical-onset", "3|2", 2.3)]


def test_histogram_counts_frames_by_onsets() -> None:
    tokens = [token("a", 0.01, 0.1), token("b", 0.02, 0.1), token("c", 0.5, 0.6)]
    assert audit.frame_histogram(tokens) == {1: 1, 2: 1}


def test_source_seconds_map_through_retained_cuts() -> None:
    cuts = [Cut(30, 90, 0), Cut(120, 180, 60)]
    assert audit.source_seconds(cuts, 0.0) == 1.0
    assert audit.source_seconds(cuts, 2.5) == 4.5
    assert audit.source_seconds(cuts, 99.0) is None


def test_windows_merge_pad_and_split_to_the_limit() -> None:
    tokens = [token("あ", 5.0, 5.1)]
    close = [
        audit.Finding("coarse-token", 10.0, 10.5, "a", ""),
        audit.Finding("dense-burst", 12.0, 12.2, "b", ""),
    ]
    merged = audit.windows(close, tokens, duration=100.0)
    assert [(w.start, w.end, w.kinds) for w in merged] == [
        (8.0, 14.2, ("coarse-token", "dense-burst"))
    ]
    long = audit.windows(
        [audit.Finding("coarse-token", 10.0, 190.0, "z", "")], tokens, duration=300.0
    )
    assert [(w.start, w.end) for w in long] == [
        (8.0, 54.0),
        (54.0, 100.0),
        (100.0, 146.0),
        (146.0, 192.0),
    ]


def test_windows_clip_to_the_timeline() -> None:
    window = audit.windows([audit.Finding("coarse-token", 0.5, 1.0, "a", "")], [], duration=2.0)
    assert [(w.start, w.end) for w in window] == [(0.0, 2.0)]


def test_probe_findings_read_density_records(tmp_path: Path) -> None:
    probe = tmp_path / "violations.jsonl"
    probe.write_text(
        json.dumps(
            {
                "kind": "density",
                "cueStart": 90.0,
                "text": "食べる",
                "previous": "じゃん",
                "delay": 0.038,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    found = audit.probe_findings(probe, offset=600.0)
    assert [(f.kind, f.start, f.text) for f in found] == [
        ("display-density", 687.5, "じゃん / 食べる")
    ]


def test_probe_findings_reject_other_or_invalid_records(tmp_path: Path) -> None:
    probe = tmp_path / "violations.jsonl"
    probe.write_text(
        json.dumps({"kind": "other", "cueStart": 1.0, "text": "a", "delay": 0.1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="paging probe"):
        audit.probe_findings(probe, offset=0.0)


def test_read_tokens_rejects_invalid_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid projected token"):
        audit.read_tokens(
            write(
                tmp_path / "t.json",
                {"tokens": [{"text": "a", "start": True, "end": 1, "utterance": "u"}]},
            )
        )
    with pytest.raises(ValueError, match="missing"):
        audit.read_tokens(write(tmp_path / "e.json", {"tokens": []}))


def test_report_locates_findings_per_segment_without_timing_changes(tmp_path: Path) -> None:
    tokens = write(
        tmp_path / "tokens.json",
        rows(("あ", 0.0, 0.2), ("い", 0.2, 3.5), ("う", 4.0, 4.2)),
    )
    cuts = write(
        tmp_path / "cuts.json",
        {"cuts": [{"source_start_frame": 30, "source_end_frame": 180, "output_start_frame": 0}]},
    )
    segments = write(
        tmp_path / "segments.json",
        {
            "segments": [
                {"segment": 0, "offsetFrame": 0, "frames": 90},
                {"segment": 1, "offsetFrame": 90, "frames": 60},
            ]
        },
    )
    result = audit.report(tokens=tokens, cuts=cuts, segments=segments)
    assert result["timingChanged"] is False
    assert result["findings"] == [
        {
            "kind": "coarse-token",
            "text": "い",
            "detail": "duration 3.300s",
            "outputStart": 0.2,
            "outputEnd": 3.5,
            "sourceStart": 1.2,
            "sourceEnd": 4.466666666666667,
        }
    ]
    assert result["segments"] == [
        {
            "segment": 0,
            "outputStart": 0.0,
            "outputEnd": 3.0,
            "coarseTokens": 1,
            "denseBursts": 0,
            "identicalOnsets": 0,
            "displayDensity": 0,
        }
    ]


def test_report_rejects_a_bad_segment_index(tmp_path: Path) -> None:
    tokens = write(tmp_path / "tokens.json", rows(("あ", 0.0, 0.2)))
    cuts = write(
        tmp_path / "cuts.json",
        {"cuts": [{"source_start_frame": 0, "source_end_frame": 30, "output_start_frame": 0}]},
    )
    bad = write(
        tmp_path / "bad.json", {"segments": [{"segment": "0", "offsetFrame": 0, "frames": 3}]}
    )
    with pytest.raises(ValueError, match="segment row"):
        audit.report(tokens=tokens, cuts=cuts, segments=bad)
    with pytest.raises(ValueError, match="index is missing"):
        audit.report(
            tokens=tokens, cuts=cuts, segments=write(tmp_path / "none.json", {"segments": []})
        )


def test_report_places_probe_findings_at_the_segment_offset(tmp_path: Path) -> None:
    tokens = write(tmp_path / "tokens.json", rows(("あ", 0.0, 0.2), ("い", 4.0, 4.2)))
    cuts = write(
        tmp_path / "cuts.json",
        {"cuts": [{"source_start_frame": 0, "source_end_frame": 300, "output_start_frame": 0}]},
    )
    segments = write(
        tmp_path / "segments.json",
        {
            "segments": [
                {"segment": 0, "offsetFrame": 0, "frames": 90},
                {"segment": 1, "offsetFrame": 90, "frames": 210},
            ]
        },
    )
    probe = tmp_path / "violations.jsonl"
    probe.write_text(
        json.dumps(
            {"kind": "density", "cueStart": 4.0, "text": "b", "previous": "a", "delay": 0.04}
        )
        + "\n",
        encoding="utf-8",
    )
    result = audit.report(tokens=tokens, cuts=cuts, segments=segments, probe=(probe, 1))
    assert result["segments"] == [
        {
            "segment": 1,
            "outputStart": 3.0,
            "outputEnd": 10.0,
            "coarseTokens": 0,
            "denseBursts": 0,
            "identicalOnsets": 0,
            "displayDensity": 1,
        }
    ]
    with pytest.raises(ValueError, match="unknown recognition segment"):
        audit.report(tokens=tokens, cuts=cuts, segments=segments, probe=(probe, 7))
