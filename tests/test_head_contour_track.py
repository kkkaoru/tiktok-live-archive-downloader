import json
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from replay.head_contour_track import ContourSeed, contour_seed, main, run, track_contour
from replay.local_asr import fingerprint
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image


@pytest.fixture
def frame() -> Image:
    image = np.random.default_rng(19).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    return np.asarray(cv2.GaussianBlur(image, (5, 5), 0), np.uint8)


@pytest.fixture
def seed() -> ContourSeed:
    return ContourSeed(Fraction(1), 96, 96, np.array([[32, 32], [48, 40], [64, 48]], np.float32))


def test_translation_preserves_separate_tracking_provenance(
    frame: Image, seed: ContourSeed
) -> None:
    translated = np.asarray(
        cv2.warpAffine(frame, np.array([[1, 0, 2], [0, 1, 0]], np.float32), (96, 96)), np.uint8
    )
    result = track_contour(
        seed, VideoFrameClock((30, 31), Fraction(1, 30), 1), [(30, frame), (31, translated)]
    )
    assert result[0].provenance == "measured-foreground-contour"
    assert result[1].provenance == "tracked-foreground-contour"
    assert result[1].supported_points == 3
    assert result[1].unknown_points == 0
    np.testing.assert_allclose(
        np.asarray(result[1].points, np.float32), [[34, 32], [50, 40], [66, 48]], atol=0.1
    )
    assert result[1].candidate_top == pytest.approx(32, abs=0.1)


def test_rotation_uses_actual_point_correspondences(frame: Image, seed: ContourSeed) -> None:
    rotated = np.asarray(
        cv2.warpAffine(frame, cv2.getRotationMatrix2D((48, 48), 2, 1), (96, 96)), np.uint8
    )
    result = track_contour(
        seed, VideoFrameClock((30, 31), Fraction(1, 30), 1), [(30, frame), (31, rotated)]
    )
    assert result[1].supported_points == 3
    np.testing.assert_allclose(
        np.asarray(result[1].points, np.float32),
        np.array([[31.45, 32.57], [47.72, 40], [63.99, 47.44]], np.float32),
        atol=0.3,
    )


@pytest.mark.parametrize("blank_source", [False, True])
def test_occlusion_or_blank_stops_without_hold(
    frame: Image, seed: ContourSeed, blank_source: bool
) -> None:
    blank = np.zeros((96, 96, 3), np.uint8)
    first = blank if blank_source else frame
    result = track_contour(
        seed,
        VideoFrameClock((30, 31, 32), Fraction(1, 30), 1),
        [(30, first), (31, blank), (32, frame)],
    )
    assert len(result) == 2
    assert result[-1].candidate_top is None
    assert result[-1].reason == "lk_roundtrip_or_appearance_unsupported"
    assert result[-1].unknown_points == 3


def test_edge_seed_is_unknown_not_clamped(frame: Image) -> None:
    seed = ContourSeed(Fraction(1), 96, 96, np.array([[1, 12], [12, 12], [24, 12]], np.float32))
    result = track_contour(seed, VideoFrameClock((30,), Fraction(1, 30), 1), [(30, frame)])
    assert result[0].reason == "seed_outside_lk_support"
    assert result[0].candidate_top is None
    assert result[0].points[0] == (1.0, 12.0)


def test_wrong_missing_or_extra_decoded_clocks_fail(frame: Image, seed: ContourSeed) -> None:
    clock = VideoFrameClock((30,), Fraction(1, 30), 1)
    with pytest.raises(ValueError, match="clock"):
        track_contour(seed, clock, [(31, frame)])
    with pytest.raises(ValueError, match="incomplete"):
        track_contour(seed, clock, [])
    with pytest.raises(ValueError, match="clock"):
        track_contour(seed, clock, [(30, frame), (31, frame)])
    with pytest.raises(ValueError, match="geometry"):
        track_contour(seed, clock, [(30, np.zeros((32, 32, 3), np.uint8))])
    with pytest.raises(ValueError, match="contract"):
        track_contour(seed, VideoFrameClock((31,), Fraction(1, 30), 1), [])


@pytest.fixture
def response() -> list[dict[str, object]]:
    return [
        {
            "actual": {"value": 30, "timescale": 30},
            "requested": {"value": 1, "timescale": 1},
            "width": 96,
            "height": 96,
            "captionPlacementApproved": False,
            "headTop": {
                "status": "candidate",
                "semanticHeadVerified": False,
                "captionPlacementApproved": False,
                "contourScope": "upper-component-in-face-x-range",
                "upperContour": [{"x": 32, "y": 32}, {"x": 48, "y": 40}, {"x": 64, "y": 48}],
            },
        }
    ]


def test_seed_reads_measured_points_without_bbox(response: list[dict[str, object]]) -> None:
    seed = contour_seed(response, clock=VideoFrameClock((30,), Fraction(1, 30), 1))
    assert seed.time == Fraction(1)
    assert seed.points.tolist() == [[32, 32], [48, 40], [64, 48]]


@pytest.mark.parametrize(
    "fault",
    [
        "clock",
        "timescale",
        "canvas",
        "approved",
        "scope",
        "points",
        "duplicate",
        "order",
        "outside",
        "frames",
    ],
)
def test_seed_rejects_unbound_or_fabricated_geometry(
    response: list[dict[str, object]], fault: str
) -> None:
    if fault == "clock":
        response[0]["actual"] = {"value": 31, "timescale": 30}
    elif fault == "timescale":
        response[0]["actual"] = {"value": 1, "timescale": 0}
    elif fault == "canvas":
        response[0]["width"] = 1
    elif fault == "frames":
        response.clear()
    else:
        head = response[0]["headTop"]
        assert isinstance(head, dict)
        changes: dict[str, dict[str, object]] = {
            "approved": {"semanticHeadVerified": True},
            "scope": {"contourScope": "bbox-corners"},
            "points": {"upperContour": []},
            "duplicate": {"upperContour": [{"x": 32, "y": 32}] * 3},
            "order": {"upperContour": [{"x": 64, "y": 32}, {"x": 48, "y": 40}, {"x": 32, "y": 48}]},
            "outside": {
                "upperContour": [{"x": 32, "y": 32}, {"x": 48, "y": 40}, {"x": 96, "y": 48}]
            },
        }
        head.update(changes[fault])
    with pytest.raises(ValueError):
        contour_seed(response, clock=VideoFrameClock((30,), Fraction(1, 30), 1))


@pytest.fixture
def config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: list[dict[str, object]],
    frame: Image,
) -> Path:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic source")
    executable = tmp_path / "native"
    executable.write_bytes(b"synthetic executable")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "path": str(source),
                "samples": [{"value": 30, "timescale": 30}],
                "headFace": {"presentation": {"value": 1, "timescale": 1}},
            }
        ),
        encoding="utf-8",
    )
    result = tmp_path / "response.json"
    result.write_text(json.dumps(response), encoding="utf-8")
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": {"path": str(source), "sha256": fingerprint(source)},
                "nativeExecutable": {"path": str(executable), "sha256": fingerprint(executable)},
                "ffmpeg": {"path": str(executable), "sha256": fingerprint(executable)},
                "nativeRequest": {"path": str(request), "sha256": fingerprint(request)},
                "nativeResponse": {"path": str(result), "sha256": fingerprint(result)},
                "clock": {"pts": [30, 31], "timeBase": "1/30", "lastDuration": 1},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "replay.head_contour_track.decode_context", Mock(return_value=(frame, frame))
    )
    return path


def test_pinned_run_is_not_face_or_placement_approval(config: Path) -> None:
    output = config.parent / "tracks"
    assert main([str(config), str(output)]) == 0
    report = json.loads((output / "contour-tracks.json").read_text(encoding="utf-8"))
    assert report["completeTrack"] is True
    assert report["semanticHeadVerified"] is False
    assert report["captionPlacementApproved"] is False
    assert report["headContourComplete"] is False
    assert report["observedFrames"] == 2
    assert report["observations"][1]["provenance"] == "tracked-foreground-contour"
    with pytest.raises(ValueError, match="new checkout"):
        run(config, output)


def test_changed_seed_pin_fails(config: Path) -> None:
    (config.parent / "response.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        run(config, config.parent / "tracks")


def test_inputs_changed_during_decode_fail(
    config: Path, monkeypatch: pytest.MonkeyPatch, frame: Image
) -> None:
    def changed(*args: object, **kwargs: object) -> tuple[Image, Image]:
        (config.parent / "source.bin").write_bytes(b"changed")
        return frame, frame

    monkeypatch.setattr("replay.head_contour_track.decode_context", changed)
    with pytest.raises(ValueError, match="changed during"):
        run(config, config.parent / "tracks")


@pytest.mark.parametrize("fault", ["version", "base", "path", "sha", "nativeclock", "jsonbudget"])
def test_invalid_job_fails(config: Path, fault: str) -> None:
    raw = json.loads(config.read_text(encoding="utf-8"))
    if fault == "version":
        raw["version"] = 2
    elif fault == "base":
        raw["clock"]["timeBase"] = None
    elif fault == "path":
        raw["source"]["path"] = "source.bin"
    elif fault == "sha":
        raw["source"]["sha256"] = "invalid"
    elif fault == "nativeclock":
        request = config.parent / "request.json"
        request.write_text(
            json.dumps(
                {
                    "path": str(config.parent / "source.bin"),
                    "samples": [],
                    "headFace": {"presentation": {"value": 1, "timescale": 1}},
                }
            ),
            encoding="utf-8",
        )
        raw["nativeRequest"]["sha256"] = fingerprint(request)
    elif fault == "jsonbudget":
        config.write_text(" " * (2 * 1024 * 1024 + 1), encoding="utf-8")
        with pytest.raises(ValueError, match="budget"):
            run(config, config.parent / "tracks")
        return
    config.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        run(config, config.parent / "tracks")
