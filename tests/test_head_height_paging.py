import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_caption_native_layout import layout as layout
from test_caption_native_reflow import exact as exact
from test_fcp_head_near import paired as paired

from replay import caption_native_paging as paging
from replay import head_height_paging as height
from replay.caption_measure_cache import CachedCaptionLayout
from replay.caption_native_layout import NativeCaptionLayout
from replay.caption_scale import CaptionScale, scale_fields
from replay.caption_timing import SpokenToken
from replay.caption_token_sources import TokenSource
from replay.core import ReplayError
from replay.fcp_caption_retarget import RetainedSpan
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_tracking_caption import MissingFaceRange, TrackingPosition
from replay.head_near_placement import FaceSizeObservation
from replay.stages import file_digest


def test_headroom_uses_derived_scale_without_changing_raw_face(paired: FaceScaleSeries) -> None:
    derived = replace(
        paired,
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 100, 100),
        ),
    ).with_current_scale(reference_time=Fraction(0))
    assert (
        height.minimum_headroom(
            derived,
            (RetainedSpan(Fraction(0), Fraction(0), Fraction(0), Fraction(2)),),
            start=Fraction(0),
            end=Fraction(2),
            preference=height.HeadroomPreference(Fraction(0), 500, 12),
        )
        == 176
    )
    assert derived.tops == paired.tops
    assert derived.bottoms == paired.bottoms


def test_missing_geometry_uses_canvas_without_old_reference_headroom(
    paired: FaceScaleSeries,
) -> None:
    gaps = (
        MissingFaceRange(Fraction(1), Fraction(4), 90, Fraction(0), "last-known-hold", "missing"),
    )
    stale = FaceScaleSeries(
        replace(
            paired.tops,
            positions=(TrackingPosition(Fraction(0), 0, 900), TrackingPosition(Fraction(3), 0, 10)),
            coverage_end=Fraction(4),
            derived_ranges=gaps,
        ),
        replace(
            paired.bottoms,
            positions=(
                TrackingPosition(Fraction(0), 0, 1300),
                TrackingPosition(Fraction(3), 0, 500),
            ),
            coverage_end=Fraction(4),
            derived_ranges=gaps,
        ),
    )
    assert (
        height.minimum_headroom(
            stale,
            (RetainedSpan(Fraction(3), Fraction(0), Fraction(0), Fraction(1)),),
            start=Fraction(0),
            end=Fraction(1),
            preference=height.HeadroomPreference(Fraction(0), 500, 12),
        )
        == 1896
    )
    assert stale.tops.positions[-1].y == 10


@pytest.mark.parametrize("head_top, expected", [(500, 276), (0, 12)])
def test_prefer_headroom_but_keep_one_native_line(
    paired: FaceScaleSeries,
    head_top: float,
    expected: float,
) -> None:
    assert (
        height.minimum_headroom(
            paired,
            (RetainedSpan(Fraction(0), Fraction(0), Fraction(0), Fraction(2)),),
            start=Fraction(0),
            end=Fraction(2),
            preference=height.HeadroomPreference(Fraction(0), head_top, 12),
        )
        == expected
    )


def test_single_line_native_height(exact: NativeCaptionLayout) -> None:
    assert (
        height.single_line_height(
            {"captions": [{"text": "ab\ncd"}]},
            CachedCaptionLayout(exact),
        )
        == 12
    )


def test_scaled_part_fits_actual_height_without_double_scale(
    exact: NativeCaptionLayout, tmp_path: Path
) -> None:
    scale = CaptionScale(24, 12, 12, 12, 2, 0.2)
    part: dict[str, object] = {"captions": [{"text": "ab", **scale_fields(scale)}]}
    result, changed = height.repage_segment(
        captions=tmp_path,
        segment={},
        part=part,
        layout=CachedCaptionLayout(exact),
        maximum_height=6,
        rate=30,
    )
    assert result == part
    assert changed is False


@pytest.mark.parametrize("part", [{}, {"captions": [{"text": 1}]}])
def test_single_line_height_rejects_missing_text(
    exact: NativeCaptionLayout,
    part: dict[str, object],
) -> None:
    with pytest.raises(ReplayError):
        height.single_line_height(part, CachedCaptionLayout(exact))


def test_height_reflow_keeps_adopted_token_clock(
    exact: NativeCaptionLayout,
    tmp_path: Path,
) -> None:
    tokens = tmp_path / "adopted.json"
    tokens.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "output",
                "tokens": [
                    {"text": "ab", "start": 0.25, "end": 0.5, "utterance": "u"},
                    {"text": "cd", "start": 0.5, "end": 1, "utterance": "u"},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "adoption.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "tokens": {"path": str(tokens), "sha256": file_digest(tokens)},
            }
        ),
        encoding="utf-8",
    )
    source = TokenSource(0, 0, tokens, file_digest(tokens), manifest, file_digest(manifest))
    (tmp_path / "combined-0000.json").write_text(
        json.dumps(
            {
                "inputs": {str(tokens): file_digest(tokens), str(manifest): file_digest(manifest)},
            }
        ),
        encoding="utf-8",
    )
    result, changed = height.repage_segment(
        captions=tmp_path,
        segment={"segment": 0, "jobs": [0], "frames": 60, "offsetFrame": 90},
        part={"captions": [{"text": "ab\ncd"}], "coverage": {"start": 3, "end": 5}},
        layout=CachedCaptionLayout(exact),
        maximum_height=12,
        rate=30,
        token_sources=(source,),
    )
    assert changed is True
    assert result["captions"] == [
        {"start": 3.25, "end": 3.5, "text": "ab"},
        {"start": 3.5, "end": 4.08, "text": "cd"},
    ]


def test_measured_height_repages_at_original_onsets(exact: NativeCaptionLayout) -> None:
    cues, _, _ = paging.paged_cues(
        (SpokenToken("ab", 0, 0.5, "u"), SpokenToken("cd", 0.5, 1, "u")),
        layout=exact,
        rate=30,
        duration=Fraction(2),
        maximum_height=12,
    )
    assert [(cue.text, cue.start) for cue in cues] == [("ab", 0), ("cd", 0.5)]
    with pytest.raises(ReplayError, match="height"):
        paging.paged_cues(
            (), layout=exact, rate=30, duration=Fraction(2), maximum_height=float("nan")
        )


@pytest.mark.parametrize("fresh", [False, True])
def test_repage_segment_and_reuse_unchanged(
    exact: NativeCaptionLayout, tmp_path: Path, fresh: bool
) -> None:
    tokens = tmp_path / "tokens-0000.json"
    tokens.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "output",
                "tokens": [
                    {"text": "ab", "start": 0, "end": 0.5, "utterance": "u"},
                    {"text": "cd", "start": 0.5, "end": 1, "utterance": "u"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "combined-0000.json").write_text(
        json.dumps({"inputs": {str(tokens): file_digest(tokens)}}), encoding="utf-8"
    )
    part: dict[str, object] = {
        "captions": [{"text": "ab\ncd", "start": 0, "end": 1}],
        "coverage": {"start": 0, "end": 2},
    }
    segment: dict[str, object] = {"segment": 0, "frames": 60, "offsetFrame": 0, "jobs": [0]}
    if fresh:
        binding = tmp_path / "recognition-0000.json"
        binding.write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "recognition-inputs",
                    "segment": 0,
                    "jobs": [0],
                    "measurementsPerformed": False,
                    "inputs": {str(tokens): file_digest(tokens)},
                }
            )
        )
        segment["recognitionInput"] = {"path": str(binding), "sha256": file_digest(binding)}
        (tmp_path / "combined-0000.json").unlink()
    wrapped = CachedCaptionLayout(exact)
    unchanged, changed = height.repage_segment(
        captions=tmp_path, segment=segment, part=part, layout=wrapped, maximum_height=30, rate=30
    )
    assert changed is False
    assert unchanged["captions"] == [{"text": "ab\ncd", "start": 0, "end": 1}]
    result, changed = height.repage_segment(
        captions=tmp_path, segment=segment, part=part, layout=wrapped, maximum_height=12, rate=30
    )
    assert changed is True
    assert result["captions"] == [
        {"start": 0.0, "end": 0.5, "text": "ab"},
        {"start": 0.5, "end": 1.08, "text": "cd"},
    ]


@pytest.mark.parametrize("part", [{}, {"captions": [{"text": 2}]}])
def test_repage_invalid_part(
    exact: NativeCaptionLayout, tmp_path: Path, part: dict[str, object]
) -> None:
    with pytest.raises(ReplayError):
        height.repage_segment(
            captions=tmp_path,
            segment={},
            part=part,
            layout=CachedCaptionLayout(exact),
            maximum_height=12,
            rate=30,
        )


@pytest.mark.parametrize("long", [False, True])
def test_make_layout_reuses_pinned_metrics(
    exact: NativeCaptionLayout, tmp_path: Path, long: bool
) -> None:
    exact.measure("ab")
    reference = exact.directory / "inputs.json"
    raw = json.loads(reference.read_text(encoding="utf-8"))
    raw["expectedPointSize"] = 10
    reference.write_text(json.dumps(raw), encoding="utf-8")
    result = height.make_layout(
        reference=reference,
        executable=exact.executable,
        output=tmp_path / "reuse",
        pinned_inputs=exact.verify_inputs(),
        cached_directory=exact.directory,
        long_executable=exact.executable if long else None,
    )
    assert result.measure("ab") == (24, 12)
    raw["style"]["font_name"] = None
    reference.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="font"):
        height.make_layout(
            reference=reference,
            executable=exact.executable,
            output=tmp_path / "bad",
            pinned_inputs={},
            cached_directory=exact.directory,
        )


def test_head_layout_reuses_inherited_and_fresh_measurements(
    exact: NativeCaptionLayout, tmp_path: Path
) -> None:
    exact.measure("ab")
    reference = exact.directory / "inputs.json"
    raw = json.loads(reference.read_text(encoding="utf-8"))
    raw["expectedPointSize"] = 10
    reference.write_text(json.dumps(raw), encoding="utf-8")
    fresh = NativeCaptionLayout(
        executable=exact.executable,
        directory=tmp_path / "fresh",
        style=exact.style,
        width=exact.width,
        height=4096,
        evidence=(reference,),
        expected_point_size=10,
        execute=exact._execute,
    )
    fresh.measure("cd")
    pins = exact.verify_inputs() | fresh.verify_inputs()
    result = height.make_layout(
        reference=reference,
        executable=exact.executable,
        output=tmp_path / "head",
        pinned_inputs=pins,
        cached_directory=fresh.directory,
    )
    assert result.measure("ab\ncd") == (24, 26)
    assert not tuple((tmp_path / "head").glob("*.request.json"))
    result.verify_inputs()
    del pins[str(exact.directory / "batch-000000.response.json")]
    with pytest.raises(ReplayError, match="pinned response"):
        height.make_layout(
            reference=reference,
            executable=exact.executable,
            output=tmp_path / "missing-pair",
            pinned_inputs=pins,
            cached_directory=fresh.directory,
        )


def test_head_layout_rejects_unbounded_or_absent_requests(
    exact: NativeCaptionLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact.measure("ab")
    reference = exact.directory / "inputs.json"
    raw = json.loads(reference.read_text(encoding="utf-8"))
    raw["expectedPointSize"] = 10
    reference.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(height, "MAX_CACHE", 0)
    with pytest.raises(ReplayError, match="bounded pinned"):
        height.make_layout(
            reference=reference,
            executable=exact.executable,
            output=tmp_path / "over-budget",
            pinned_inputs=exact.verify_inputs(),
            cached_directory=exact.directory,
        )
