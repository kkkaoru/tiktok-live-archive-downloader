"""Synthetic observed tracker envelopes and explicit head calibration."""

import plistlib
import runpy
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_tracking_caption as tracking
from replay.core import ReplayError
from replay.fcp_tracking_caption import (
    FacePositionSeries,
    TrackingCalibration,
    TrackingPosition,
    apply_tracker_to_titles,
    face_position_series,
    main,
    parse_tracking_positions,
    read_native_tracker,
    title_position_samples,
)
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import TrackingJob

TRACK = b"""<parameter-container><param key="200"><param key="201"><keyframeAnimation>
<keyframe time="0s" value="90 10"/><keyframe time="1/2s" value="-90 20"/>
<keyframe time="1s" value="70 30"/>
</keyframeAnimation></param></param></parameter-container>"""
CALIBRATION = TrackingCalibration(Fraction(0), 400, 2, 100, 20, 1000, (100, 200, 100, 150))
PROJECT = b"""<fcpxml><resources>
<locator id="r4" url="Contents/Effects/ObjectTracking/abc.plist"/></resources>
<asset-clip><object-tracker><tracking-shape id="tr1" dataLocator="r4"/></object-tracker>
<title ref="r5" offset="0s" start="3600s" duration="1s"><text>Caption</text>
<param name="reveal" value="1"/><adjust-transform position="99 99" scale="1 1">
<param name="position"><keyframeAnimation/></param></adjust-transform></title>
<title ref="r6" offset="0s" start="3600s" duration="1s"/>
</asset-clip></fcpxml>"""


def test_projects_only_y_into_title_clock_without_line_count_compensation() -> None:
    title = ET.fromstring('<title offset="5/4s" start="3600s" duration="1/2s"/>')
    samples = parse_tracking_positions(TRACK, frame_duration=Fraction(1, 2))
    assert title_position_samples(
        title, samples, calibration=CALIBRATION, tracker_source_start=Fraction(1)
    ) == (("3600s", "0 1"), ("14401/4s", "0 0"), ("7201/2s", "0 -1"))


def test_final_exclusive_title_end_uses_last_observed_frame() -> None:
    title = ET.fromstring('<title offset="1s" start="3600s" duration="1/2s"/>')
    assert title_position_samples(
        title,
        parse_tracking_positions(TRACK, frame_duration=Fraction(1, 2)),
        calibration=CALIBRATION,
        tracker_source_start=Fraction(0),
    ) == (("3600s", "0 -2"),)


def test_native_face_vfr_positions_preserve_actual_top_and_exclusive_end() -> None:
    observations = [
        {
            "presentationValue": 0,
            "presentationTimescale": 1000,
            "width": 1000,
            "height": 1000,
            "method": "detection",
            "detectionAttempted": True,
            "faces": [
                {"confidence": 0.8, "region": {"x": 100, "y": 400, "width": 100, "height": 200}}
            ],
        },
        {
            "presentationValue": 33,
            "presentationTimescale": 1000,
            "width": 1000,
            "height": 1000,
            "method": "tracking",
            "detectionAttempted": False,
            "faces": [
                {"confidence": 0.9, "region": {"x": 110, "y": 410, "width": 100, "height": 200}}
            ],
        },
        {
            "presentationValue": 67,
            "presentationTimescale": 1000,
            "width": 1000,
            "height": 1000,
            "method": "tracking",
            "detectionAttempted": False,
            "faces": [
                {"confidence": 0.9, "region": {"x": 120, "y": 420, "width": 100, "height": 200}}
            ],
        },
    ]
    clock = VideoFrameClock((0, 33, 67), Fraction(1, 1000), 33)
    series = face_position_series(observations, clock, width=1000, height=1000)
    assert series.coverage_end == Fraction(1, 10)
    assert series.positions[1] == TrackingPosition(Fraction(33, 1000), 110, 410)
    bottom = face_position_series(observations, clock, width=1000, height=1000, anchor="bottom")
    assert bottom.positions[1].y == 610
    assert bottom.anchor == "bottom"
    assert title_position_samples(
        ET.fromstring('<title offset="67/1000s" start="3600s" duration="33/1000s"/>'),
        series.positions,
        calibration=replace(CALIBRATION, tracker_y_to_pixels=1),
        tracker_source_start=Fraction(0),
        observed_end=series.coverage_end,
    ) == (("3600s", "0 0"),)
    observations[2]["faces"] = []
    observations[2]["method"] = "missing"
    with pytest.raises(ReplayError, match="one observed face"):
        face_position_series(observations, clock, width=1000, height=1000)


def observation_series(
    values: tuple[float | None, ...], *, start: int = 0
) -> tracking.FaceObservationSeries:
    return tracking.FaceObservationSeries(
        tuple(
            tracking.FaceObservation(
                Fraction(start + index),
                None if value is None else TrackingPosition(Fraction(start + index), 10, value),
                "missing" if value is None else None,
            )
            for index, value in enumerate(values)
        ),
        Fraction(start + len(values)),
        1080,
        1920,
    )


def test_candidate_hold_keeps_first_middle_last_unknown_provenance() -> None:
    raw = observation_series((None, 100, None, None, 200, None))
    with pytest.raises(ReplayError, match="one observed face"):
        tracking.resolve_face_observations(raw)
    result = tracking.resolve_face_observations(raw, missing_policy="hold")
    assert [point.y for point in result.positions] == [100, 100, 100, 100, 200, 200]
    assert [
        (gap.start, gap.end, gap.frames, gap.derived_from, gap.method)
        for gap in result.derived_ranges
    ] == [
        (Fraction(0), Fraction(1), 1, Fraction(1), "leading-observation-backfill"),
        (Fraction(2), Fraction(4), 2, Fraction(1), "last-known-hold"),
        (Fraction(5), Fraction(6), 1, Fraction(4), "last-known-hold"),
    ]
    assert raw.samples[2].position is None


def test_long_missing_window_is_derived_only_after_global_join() -> None:
    left = observation_series((100, None))
    lost = observation_series((None,) * 100, start=2)
    right = observation_series((200, 210), start=102)
    joined = tracking.merge_face_observations((left, lost, right))
    result = tracking.resolve_face_observations(joined, missing_policy="hold")
    assert len(result.positions) == 104
    assert result.derived_ranges[0].frames == 101
    assert result.derived_ranges[0].end == 102
    with pytest.raises(ReplayError, match="No observed face"):
        tracking.resolve_face_observations(lost, missing_policy="hold")


def test_candidate_ambiguous_faces_are_not_arbitrarily_selected() -> None:
    raw = observation_series((100, None))
    raw = replace(raw, samples=(raw.samples[0], replace(raw.samples[1], reason="ambiguous")))
    result = tracking.resolve_face_observations(raw, missing_policy="hold")
    assert result.derived_ranges[0].reason == "ambiguous"


def test_raw_seed_overlap_uses_next_observation_even_when_missing() -> None:
    left = observation_series((100, 110))
    right = observation_series((None, 200), start=1)
    with pytest.raises(ReplayError, match="clock gap"):
        tracking.merge_face_observations((left, right))
    raw = tracking.merge_face_observations((left, right), seed_overlap="next")
    result = tracking.resolve_face_observations(raw, missing_policy="hold")
    assert [point.y for point in result.positions] == [100, 100, 200]
    assert result.derived_ranges[0].start == 1


@pytest.mark.parametrize("mode", ["empty", "jump", "canvas", "anchor", "order", "end"])
def test_candidate_windows_reject_unknown_geometry_or_clock(mode: str) -> None:
    first = observation_series((100, 110))
    second = observation_series((120, 130), start=2)
    if mode == "jump":
        second = observation_series((120, 130), start=3)
    elif mode == "canvas":
        second = replace(second, width=999)
    elif mode == "anchor":
        second = replace(second, anchor="bottom")
    elif mode == "order":
        second = replace(second, samples=tuple(reversed(second.samples)))
    elif mode == "end":
        second = replace(second, coverage_end=Fraction(3))
    with pytest.raises(ReplayError):
        tracking.merge_face_observations(() if mode == "empty" else (first, second))


def test_resolver_rejects_invalid_clock_and_missing_reason() -> None:
    raw = observation_series((100, None))
    with pytest.raises(ReplayError, match="clock"):
        tracking.resolve_face_observations(
            replace(raw, coverage_end=Fraction(1)), missing_policy="hold"
        )
    with pytest.raises(ReplayError, match="reason"):
        tracking.resolve_face_observations(
            replace(raw, samples=(raw.samples[0], replace(raw.samples[1], reason=None))),
            missing_policy="hold",
        )


def test_explicit_bottom_anchor_does_not_treat_bbox_shrink_as_head_motion() -> None:
    frames = [
        {
            "presentationValue": index,
            "presentationTimescale": 30,
            "width": 1080,
            "height": 1920,
            "method": "detection",
            "detectionAttempted": True,
            "faces": [
                {"confidence": 0.9, "region": {"x": 300, "y": top, "width": 300, "height": height}}
            ],
        }
        for index, (top, height) in enumerate(((400, 200), (430, 170)))
    ]
    clock = VideoFrameClock((0, 1), Fraction(1, 30), 1)
    top = face_position_series(frames, clock, width=1080, height=1920)
    bottom = face_position_series(frames, clock, width=1080, height=1920, anchor="bottom")
    assert top.positions[1].y - top.positions[0].y == 30
    assert bottom.positions[1].y - bottom.positions[0].y == 0


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (b'key="201"', b'key="202"', "exactly one"),
        (b"90 10", b"nan 10", "finite"),
        (b"90 10", b"a b", "Invalid tracker"),
        (b"90 10", b"90", "two finite"),
        (b"1/2s", b"3/4s", "consecutive"),
        (b"1/2s", b"0s", "consecutive"),
        (b"0s", b"-1s", "rational"),
        (b"<parameter-container>", b'<!ENTITY x "y"><parameter-container>', "declarations"),
        (b"parameter-container", b"other", "root"),
        (b"</parameter-container>", b"", "Invalid tracking XML"),
    ],
)
def test_rejects_unknown_tracking_structure(old: bytes, new: bytes, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        parse_tracking_positions(TRACK.replace(old, new), frame_duration=Fraction(1, 2))


@pytest.mark.parametrize("clock", [Fraction(0), Fraction(2)])
def test_rejects_invalid_clock(clock: Fraction) -> None:
    with pytest.raises(ReplayError, match="frame duration"):
        parse_tracking_positions(TRACK, frame_duration=clock)


def test_reads_selected_native_envelope(tmp_path: Path) -> None:
    directory = tmp_path / "Contents/Effects/ObjectTracking"
    directory.mkdir(parents=True)
    (directory / "abc.plist").write_bytes(
        plistlib.dumps({"id": "abc", "type": "fcpxmld.data.effects.object-tracking", "data": TRACK})
    )
    (tmp_path / "Info.fcpxml").write_text(
        '<fcpxml><resources><locator id="r4" url="Contents/Effects/ObjectTracking/abc.plist"/>'
        '</resources><tracking-shape id="tr1" dataLocator="r4"/></fcpxml>',
        encoding="utf-8",
    )
    result = read_native_tracker(tmp_path, tracker_id="tr1", frame_duration=Fraction(1, 2))
    assert len(result) == 3
    assert result[1].y == 20
    with pytest.raises(ReplayError, match="selection"):
        read_native_tracker(tmp_path, tracker_id="tr2", frame_duration=Fraction(1, 2))
    (directory / "abc.plist").write_bytes(plistlib.dumps({"id": "wrong"}))
    with pytest.raises(ReplayError, match="envelope"):
        read_native_tracker(tmp_path, tracker_id="tr1", frame_duration=Fraction(1, 2))


@pytest.mark.parametrize(
    "calibration",
    [
        replace(CALIBRATION, source_head_top=float("nan")),
        replace(CALIBRATION, tracker_y_to_pixels=0),
        replace(CALIBRATION, gap=-1),
        replace(CALIBRATION, canvas_height=0),
        replace(CALIBRATION, roi=(0, 0, 0, 1)),
        replace(CALIBRATION, source_head_top=1001),
    ],
)
def test_rejects_unmeasured_calibration(calibration: TrackingCalibration) -> None:
    with pytest.raises(ReplayError, match="calibration"):
        title_position_samples(
            ET.fromstring('<title offset="0s" start="3600s" duration="1s"/>'),
            parse_tracking_positions(TRACK, frame_duration=Fraction(1, 2)),
            calibration=calibration,
            tracker_source_start=Fraction(0),
        )


def test_refuses_extrapolation_and_off_canvas() -> None:
    samples = parse_tracking_positions(TRACK, frame_duration=Fraction(1, 2))
    with pytest.raises(ReplayError, match="outside observed"):
        title_position_samples(
            ET.fromstring('<title offset="0s" start="3600s" duration="2s"/>'),
            samples,
            calibration=CALIBRATION,
            tracker_source_start=Fraction(0),
        )


def test_refuses_off_canvas() -> None:
    samples = parse_tracking_positions(TRACK, frame_duration=Fraction(1, 2))
    with pytest.raises(ReplayError, match="leaves the canvas"):
        title_position_samples(
            ET.fromstring('<title offset="0s" start="3600s" duration="1s"/>'),
            samples,
            calibration=replace(CALIBRATION, source_head_top=0),
            tracker_source_start=Fraction(0),
        )


def test_rejects_unvalidated_direct_samples() -> None:
    with pytest.raises(ReplayError, match="calibration"):
        title_position_samples(
            ET.fromstring('<title offset="0s" start="3600s" duration="1s"/>'),
            (TrackingPosition(Fraction(1), 0, 10), TrackingPosition(Fraction(0), 0, 20)),
            calibration=CALIBRATION,
            tracker_source_start=Fraction(0),
        )


def test_writes_native_position_schema_preserving_other_title_content() -> None:
    result = apply_tracker_to_titles(
        PROJECT,
        parse_tracking_positions(TRACK, frame_duration=Fraction(1, 2)),
        tracker_id="tr1",
        title_ref="r5",
        calibration=CALIBRATION,
        tracker_source_start=Fraction(0),
    )
    root = ET.fromstring(result)
    assert root.findall(".//title[@ref='r5']/adjust-transform")[0].attrib == {"scale": "1 1"}
    assert root.findall(".//title[@ref='r5']/adjust-transform/param")[0].attrib == {
        "name": "position"
    }
    assert root.findall(".//keyframe")[0].attrib == {
        "time": "3600s",
        "value": "0 2",
        "curve": "linear",
    }
    assert root.findall(".//title[@ref='r5']/param")[0].attrib == {"name": "reveal", "value": "1"}
    assert root.findtext(".//title[@ref='r5']/text") == "Caption"
    assert root.find(".//title[@ref='r6']/adjust-transform") is None


@pytest.mark.parametrize(
    "old,new,message",
    [
        (b'id="tr1"', b'id="tr2"', "tracker-owning"),
        (b"<asset-clip>", b"<asset-clip><timeMap/>", "time remapping"),
        (b"<fcpxml>", b"<fcpxml><timeMap/>", "Nested or time-remapped"),
        (b'ref="r5"', b'ref="r7"', "No direct"),
        (b'scale="1 1"', b'scale="2 2"', "identity"),
        (
            b"<adjust-transform position=",
            b"<adjust-transform/><adjust-transform position=",
            "Ambiguous",
        ),
    ],
)
def test_writer_refuses_ambiguous_or_retimed_selection(
    old: bytes, new: bytes, message: str
) -> None:
    with pytest.raises(ReplayError, match=message):
        apply_tracker_to_titles(
            PROJECT.replace(old, new),
            parse_tracking_positions(TRACK, frame_duration=Fraction(1, 2)),
            tracker_id="tr1",
            title_ref="r5",
            calibration=CALIBRATION,
            tracker_source_start=Fraction(0),
        )


def test_cli_writes_new_bundle_without_media_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.fcpxmld"
    destination = tmp_path / "candidate.fcpxmld"
    directory = source / "Contents/Effects/ObjectTracking"
    directory.mkdir(parents=True)
    (source / "Info.fcpxml").write_bytes(PROJECT)
    (directory / "abc.plist").write_bytes(
        plistlib.dumps({"id": "abc", "type": "fcpxmld.data.effects.object-tracking", "data": TRACK})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fcp_tracking_caption",
            str(source),
            str(destination),
            "--tracker-id",
            "tr1",
            "--title-ref",
            "r5",
            "--frame-duration",
            "1/2s",
            "--tracker-source-start",
            "0s",
            "--reference-time",
            "0s",
            "--head-top",
            "400",
            "--y-to-pixels",
            "2",
            "--local-bottom",
            "100",
            "--gap",
            "20",
            "--canvas-height",
            "1000",
            "--roi",
            "100",
            "200",
            "100",
            "150",
        ],
    )
    monkeypatch.delitem(sys.modules, "replay.fcp_tracking_caption")
    runpy.run_module("replay.fcp_tracking_caption", run_name="__main__")
    assert (destination / "Contents/Effects/ObjectTracking/abc.plist").is_file()
    assert ET.parse(destination / "Info.fcpxml").findall(".//keyframe")[0].get("value") == "0 2"
    with pytest.raises(FileExistsError):
        runpy.run_module("replay.fcp_tracking_caption", run_name="__main__")


def test_covered_titles_only_removes_unobserved_cues() -> None:
    source = PROJECT.replace(b'offset="0s"', b'offset="2s"', 1)
    with pytest.raises(ReplayError, match="No caption titles lie"):
        apply_tracker_to_titles(
            source,
            parse_tracking_positions(TRACK, frame_duration=Fraction(1, 2)),
            tracker_id="tr1",
            title_ref="r5",
            calibration=CALIBRATION,
            tracker_source_start=Fraction(0),
            covered_titles_only=True,
        )


def test_cli_native_face_mode_removes_rejected_fcp_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.fcpxml"
    media = tmp_path / "source.mp4"
    media.write_bytes(b"synthetic source")
    project = PROJECT.replace(b"<asset-clip>", b'<asset-clip ref="r2">').replace(
        b"<resources>",
        (
            '<resources><asset id="r2"><media-rep kind="original-media" '
            f'src="{media.as_uri()}"/></asset>'
        ).encode(),
    )
    source.write_bytes(project)
    destination = tmp_path / "face.fcpxmld"
    series = FacePositionSeries(
        (TrackingPosition(Fraction(0), 0, 400), TrackingPosition(Fraction(1), 10, 420)),
        Fraction(2),
        1000,
        1000,
    )
    monkeypatch.setattr(
        "replay.fcp_tracking_caption.read_verified_face_positions", lambda config, evidence: series
    )
    job = TrackingJob(
        tmp_path / "config.json",
        "0" * 64,
        media,
        tmp_path / "native",
        "0" * 64,
        "0" * 64,
        VideoFrameClock((0, 1), Fraction(1), 1),
        1000,
        1000,
        0.5,
    )
    monkeypatch.setattr("replay.fcp_tracking_caption.load_job", lambda config: job)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fcp_tracking_caption",
            str(source),
            str(destination),
            "--tracker-id",
            "tr1",
            "--title-ref",
            "r5",
            "--frame-duration",
            "1/2s",
            "--tracker-source-start",
            "0s",
            "--reference-time",
            "0s",
            "--head-top",
            "400",
            "--y-to-pixels",
            "1",
            "--local-bottom",
            "100",
            "--canvas-height",
            "1000",
            "--roi",
            "100",
            "200",
            "100",
            "150",
            "--face-config",
            "config.json",
            "--face-evidence",
            "evidence",
            "--covered-titles-only",
        ],
    )
    main()
    root = ET.parse(destination / "Info.fcpxml")
    assert root.find(".//object-tracker") is None
    assert root.find("resources/locator") is None
    assert root.findall(".//keyframe")[0].get("value") == "0 2.4"
    assert not (destination / "Contents").exists()


def test_asset_selection_binds_exact_verified_source(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"synthetic source")
    other = tmp_path / "other.mp4"
    other.write_bytes(b"different")
    project = PROJECT.replace(b"<asset-clip>", b'<asset-clip ref="r2">').replace(
        b"<resources>",
        (
            '<resources><asset id="r2"><media-rep kind="original-media" '
            f'src="{media.as_uri()}"/></asset>'
        ).encode(),
    )
    project = project.replace(b'<title ref="r5"', b'<asset-clip ref="r2"/><title ref="r5"', 1)
    samples = parse_tracking_positions(TRACK, frame_duration=Fraction(1, 2))
    result = apply_tracker_to_titles(
        project,
        samples,
        tracker_id=None,
        asset_ref="r2",
        source_path=media,
        title_ref="r5",
        calibration=CALIBRATION,
        tracker_source_start=Fraction(0),
    )
    assert ET.fromstring(result).findall(".//keyframe")[0].get("value") == "0 2"
    with pytest.raises(ReplayError, match="differs from verified"):
        apply_tracker_to_titles(
            project,
            samples,
            tracker_id=None,
            asset_ref="r2",
            source_path=other,
            title_ref="r5",
            calibration=CALIBRATION,
            tracker_source_start=Fraction(0),
        )
    with pytest.raises(ReplayError, match="source binding"):
        apply_tracker_to_titles(
            project,
            samples,
            tracker_id=None,
            asset_ref="r2",
            title_ref="r5",
            calibration=CALIBRATION,
            tracker_source_start=Fraction(0),
        )
