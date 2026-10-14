import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from replay import head_contour_track as tracker
from replay import head_partial_pixel_track as pixels
from replay import head_upper_texture as upper
from replay.core import ReplayError
from replay.delivery_mask_quality import integer, rows
from replay.head_seed import ContourTrackMetrics, Coordinates
from replay.head_texture_track import (
    TextureFit,
    TextureMotion,
    derive_contour_motion,
    foreground_features,
)
from replay.recordings import record
from replay.stages import _uncached_file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image


@pytest.fixture
def image() -> Image:
    random = np.random.default_rng(51).integers(0, 255, (160, 160, 3), dtype=np.uint8)
    return np.asarray(cv2.GaussianBlur(random, (3, 3), 0), np.uint8)


@pytest.fixture
def response() -> list[dict[str, object]]:
    return [
        {
            "requested": {"value": 30, "timescale": 30},
            "actual": {"value": 30, "timescale": 30},
            "width": 160,
            "height": 160,
            "observationPresent": True,
            "instanceLabels": [1],
            "maskThreshold8Bit": 1,
            "captionPlacementApproved": False,
            "unionMask": {
                "width": 160,
                "height": 160,
                "nonzeroPixels": 16000,
                "nonzeroBounds": {"x": 16, "y": 16, "width": 144, "height": 144},
            },
            "headTop": {
                "status": "clippedSupport",
                "semanticHeadVerified": False,
                "captionPlacementApproved": False,
                "contourScope": "upper-component-in-face-x-range",
                "faceCoveredPixels": 3000,
                "facePixels": 3360,
                "upperContour": [
                    {"x": 28, "y": 20},
                    {"x": 60, "y": 16},
                    {"x": 92, "y": 18},
                    {"x": 124, "y": 20},
                ],
                "interiorRegions": [
                    {"x": 24, "y": 32, "width": 16, "height": 16},
                    {"x": 56, "y": 32, "width": 16, "height": 16},
                    {"x": 88, "y": 32, "width": 16, "height": 16},
                    {"x": 120, "y": 32, "width": 16, "height": 16},
                    {"x": 24, "y": 80, "width": 16, "height": 16},
                    {"x": 56, "y": 80, "width": 16, "height": 16},
                    {"x": 88, "y": 80, "width": 16, "height": 16},
                    {"x": 120, "y": 80, "width": 16, "height": 16},
                    {"x": 24, "y": 112, "width": 16, "height": 16},
                    {"x": 56, "y": 112, "width": 16, "height": 16},
                    {"x": 88, "y": 112, "width": 16, "height": 16},
                    {"x": 120, "y": 112, "width": 16, "height": 16},
                    {"x": 24, "y": 128, "width": 16, "height": 16},
                    {"x": 0, "y": 32, "width": 16, "height": 16},
                ],
                "upperSupport": {
                    "status": "interiorMeasured",
                    "minimumY": 16,
                    "contourTouchesTopEdge": False,
                    "contourTouchesBottomEdge": False,
                    "contourTouchesSideEdge": False,
                },
            },
        }
    ]


@pytest.fixture
def seed(response: list[dict[str, object]]) -> pixels.PartialPixelSeed:
    result = pixels.read_seed(
        response,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        width=160,
        height=160,
        face=(24, 112, 112, 30),
    )
    assert result is not None
    return result


def detector_frame(pts: int, observed: bool) -> dict[str, object]:
    region = {"x": 24.0, "y": 112.0, "width": 112.0, "height": 30.0}
    return {
        "faceObserved": observed,
        "faceFrame": {
            "presentationValue": pts,
            "presentationTimescale": 30,
            "method": "detection" if pts == 30 else "missing",
            "faces": [{"region": region, "confidence": 0.6}] if observed else [],
        },
        "objects": [
            {
                "stage": "detection",
                "disposition": "accepted",
                "displayTopLeft": region,
                "confidence": 0.6,
            }
        ]
        if pts == 30
        else [],
    }


def test_actual_detector_and_exact_outward_request() -> None:
    clock = VideoFrameClock((30, 31), Fraction(1, 30), 1)
    face, flags = pixels.detector_seed(
        [detector_frame(30, True), detector_frame(31, False)], clock=clock
    )
    assert face == (24.0, 112.0, 112.0, 30.0)
    assert flags == (True, False)
    assert pixels.seed_request(
        source=Path("/source"), clock=clock, face=(24.2, 112.1, 111.4, 29.6)
    ) == {
        "path": "/source",
        "samples": [{"value": 30, "timescale": 30}],
        "headFace": {
            "presentation": {"value": 30, "timescale": 30},
            "x": 24,
            "y": 112,
            "width": 112,
            "height": 30,
            "minimumFaceCoverage": 0.5,
        },
    }


@pytest.mark.parametrize(
    "fault", ["short", "clock", "flag", "missing", "two", "different", "number"]
)
def test_rejected_stale_or_inconsistent_detector_never_seeds(fault: str) -> None:
    frames = [detector_frame(30, True), detector_frame(31, False)]
    if fault == "short":
        frames.pop()
    elif fault == "clock":
        record(frames[0]["faceFrame"])["presentationValue"] = 29
    elif fault == "flag":
        frames[1]["faceObserved"] = True
    elif fault == "missing":
        record(frames[0]["faceFrame"])["method"] = "tracking"
    elif fault == "two":
        frames[0]["objects"] = rows(frames[0]["objects"]) * 2
    elif fault == "different":
        record(rows(record(frames[0]["faceFrame"])["faces"])[0])["region"] = {"x": 0}
    else:
        record(record(rows(frames[0]["objects"])[0])["displayTopLeft"])["width"] = True
    with pytest.raises((ValueError, TypeError)):
        pixels.detector_seed(frames, clock=VideoFrameClock((30, 31), Fraction(1, 30), 1))


@pytest.mark.parametrize("face", [(-1, 2, 3, 4), (1, 2, 0, 4), (1, 2, 3, float("nan"))])
def test_invalid_rectangle_is_not_clamped(face: pixels.Face) -> None:
    with pytest.raises(ValueError):
        pixels.seed_request(
            source=Path("/source"), clock=VideoFrameClock((30,), Fraction(1, 30), 1), face=face
        )


def test_nonreciprocal_native_clock_rejected() -> None:
    with pytest.raises(ValueError, match="reciprocal"):
        pixels.seed_request(
            source=Path("/source"),
            clock=VideoFrameClock((30,), Fraction(2, 31), 1),
            face=(1, 2, 3, 4),
        )


@pytest.mark.parametrize(
    "clock",
    [
        VideoFrameClock(tuple(range(257)), Fraction(1, 30), 1),
        VideoFrameClock((0, 11), Fraction(1), 1),
    ],
)
def test_finite_clock_guard_unchanged(clock: VideoFrameClock) -> None:
    with pytest.raises(ValueError, match="budget"):
        pixels.validate_clock(clock)


def test_measured_bands_exclude_body_and_outside_ui(seed: pixels.PartialPixelSeed) -> None:
    assert seed.native_status == "clippedSupport"
    assert seed.upper_regions == (
        (24, 32, 16, 16),
        (56, 32, 16, 16),
        (88, 32, 16, 16),
        (120, 32, 16, 16),
    )
    assert seed.anchor_regions == (
        (24, 112, 16, 16),
        (56, 112, 16, 16),
        (88, 112, 16, 16),
        (120, 112, 16, 16),
    )
    np.testing.assert_array_equal(seed.contour.points, [[28, 20], [60, 16], [92, 18], [124, 20]])


def test_face_interior_uses_real_texture_when_old_upper_anchor_is_flat(
    image: Image, seed: pixels.PartialPixelSeed
) -> None:
    source = image.copy()
    source[64:104] = 0
    assert len(foreground_features(source, ((24, 80, 16, 16), (56, 80, 16, 16)))) == 0
    result = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, source), (31, source)),
    )
    assert result[-1].reason is None
    assert result[0].anchor_features >= 12
    assert result[-1].candidate_upper_y == pytest.approx(16, abs=0.1)
    assert result[-1].face_observed is False


def test_missing_face_interior_does_not_return_to_upper_anchor(
    response: list[dict[str, object]],
) -> None:
    head = record(response[0]["headTop"])
    head["interiorRegions"] = [
        item for item in rows(head["interiorRegions"]) if integer_region_y(item) < 112
    ]
    assert (
        pixels.read_seed(
            response,
            clock=VideoFrameClock((30,), Fraction(1, 30), 1),
            width=160,
            height=160,
            face=(24, 112, 112, 30),
        )
        is None
    )


def integer_region_y(item: object) -> int:
    y = record(item)["y"]
    assert type(y) is int
    return y


def test_only_fully_measured_face_interior_is_selected(
    response: list[dict[str, object]],
) -> None:
    head = record(response[0]["headTop"])
    head["interiorRegions"] = [
        *rows(head["interiorRegions"]),
        {"x": 0, "y": 112, "width": 16, "height": 16},
        {"x": 24, "y": 104, "width": 16, "height": 16},
        {"x": 120, "y": 128, "width": 16, "height": 16},
        {"x": 24, "y": 144, "width": 16, "height": 16},
    ]
    seed = pixels.read_seed(
        response,
        clock=VideoFrameClock((30,), Fraction(1, 30), 1),
        width=160,
        height=160,
        face=(24, 112, 112, 30),
    )
    assert seed is not None
    assert seed.native_status == "clippedSupport"
    assert seed.anchor_regions == tuple((x, 112, 16, 16) for x in (24, 56, 88, 120))


@pytest.mark.parametrize(
    "fault", ["missing", "oneband", "apex", "margin", "many", "extent", "fields"]
)
def test_missing_or_clipped_pixel_support_not_fabricated(
    response: list[dict[str, object]], fault: str
) -> None:
    head = record(response[0]["headTop"])
    if fault == "missing":
        record(head["upperSupport"])["status"] = "canvasClipped"
    elif fault == "oneband":
        head["interiorRegions"] = rows(head["interiorRegions"])[:4]
    elif fault == "apex":
        head["interiorRegions"] = [
            record(rows(head["interiorRegions"])[0]),
            record(rows(head["interiorRegions"])[8]),
        ]
    elif fault == "margin":
        head["upperContour"] = [{"x": 28, "y": 2}, {"x": 60, "y": 2}, {"x": 92, "y": 2}]
        record(head["upperSupport"])["minimumY"] = 2
    elif fault == "many":
        head["interiorRegions"] = rows(head["interiorRegions"]) * 30
    elif fault == "extent":
        head["interiorRegions"] = [{"x": 155, "y": 32, "width": 16, "height": 16}]
    else:
        head["interiorRegions"] = [{"x": 24, "y": 32, "width": 16, "height": 16, "guessed": True}]
    if fault in {"many", "extent", "fields"}:
        with pytest.raises(ValueError):
            pixels.read_seed(
                response,
                clock=VideoFrameClock((30,), Fraction(1, 30), 1),
                width=160,
                height=160,
                face=(24, 112, 112, 30),
            )
    else:
        assert (
            pixels.read_seed(
                response,
                clock=VideoFrameClock((30,), Fraction(1, 30), 1),
                width=160,
                height=160,
                face=(24, 112, 112, 30),
            )
            is None
        )


def test_missing_face_can_have_separate_current_pixel_support(
    image: Image, seed: pixels.PartialPixelSeed
) -> None:
    donor = np.asarray(
        cv2.warpAffine(image, np.array([[1, 0, -2], [0, 1, 2]], np.float32), (160, 160)), np.uint8
    )
    result = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, donor)),
    )
    assert result[1].reason is None
    assert result[1].face_observed is False
    assert result[1].provenance == "current-pixel-derived-candidate"
    assert result[1].candidate_upper_y == pytest.approx(18, abs=0.3)


def test_stationary_upper_hud_and_moving_head_are_ambiguous(
    image: Image, seed: pixels.PartialPixelSeed
) -> None:
    donor = np.asarray(
        cv2.warpAffine(image, np.array([[1, 0, 0], [0, 1, 10]], np.float32), (160, 160)), np.uint8
    )
    donor[:64] = image[:64]
    result = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, donor)),
    )
    assert result[-1].candidate_upper_y is None
    assert result[-1].reason in {"competing_upper_and_head_motion", "texture_support_lost"}


def test_chest_and_outside_hud_motion_not_used(image: Image, seed: pixels.PartialPixelSeed) -> None:
    donor = image.copy()
    donor[144:] = 0
    donor[:8] = 255
    donor[:, :8] = 255
    result = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, donor)),
    )
    assert result[-1].reason is None
    assert result[-1].candidate_upper_y == pytest.approx(16, abs=0.1)


@pytest.mark.parametrize("reason", ["source-cut", "scene", "zoom", "geometry"])
def test_exact_resets_stop_without_old_seed(
    image: Image, seed: pixels.PartialPixelSeed, reason: str
) -> None:
    result = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, image)),
        resets=((31, reason),),
    )
    assert len(result) == 2
    assert result[1].candidate_upper_y is None
    assert result[1].reason == reason


def test_occlusion_and_flat_texture_stop(image: Image, seed: pixels.PartialPixelSeed) -> None:
    result = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, np.zeros_like(image))),
    )
    assert result[-1].candidate_upper_y is None
    assert result[-1].reason is not None
    blank = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30,), Fraction(1, 30), 1),
        face_flags=(True,),
        frames=((30, np.zeros_like(image)),),
    )
    assert blank[0].reason == "independent_texture_unavailable"


def test_scale_is_rejected_by_existing_motion_limits(
    image: Image, seed: pixels.PartialPixelSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cv2,
        "estimateAffinePartial2D",
        Mock(return_value=(np.array([[1.2, 0, 0], [0, 1.2, 0]]), np.ones((50, 1)))),
    )
    result = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, image)),
    )
    assert result[-1].candidate_upper_y is None
    assert result[-1].reason == "motion_deformation_exceeded"


@pytest.mark.parametrize(
    "fault", ["seed", "flags", "reset", "order", "clock", "geometry", "incomplete", "extra"]
)
def test_bad_tracking_inputs_fail(image: Image, seed: pixels.PartialPixelSeed, fault: str) -> None:
    clock = VideoFrameClock((30, 31), Fraction(1, 30), 1)
    flags: tuple[bool, ...] = (True, False)
    frames: tuple[tuple[int, Image], ...] = ((30, image), (31, image))
    resets: tuple[tuple[int, str], ...] = ()
    if fault == "seed":
        seed = replace(seed, contour=replace(seed.contour, time=Fraction(0)))
    elif fault == "flags":
        flags = (True,)
    elif fault == "reset":
        resets = ((32, "scene"),)
    elif fault == "order":
        resets = ((31, "scene"), (30, "scene"))
    elif fault == "clock":
        frames = ((29, image), (31, image))
    elif fault == "geometry":
        frames = ((30, np.zeros((32, 32, 3), np.uint8)), (31, image))
    elif fault == "incomplete":
        frames = ((30, image),)
    else:
        frames = ((30, image), (31, image), (32, image))
    with pytest.raises(ValueError):
        pixels.track_pixels(seed, clock=clock, face_flags=flags, frames=frames, resets=resets)


@pytest.fixture
def config(tmp_path: Path, response: list[dict[str, object]]) -> Path:
    source = tmp_path / "source"
    source.write_bytes(b"source fixture")
    native = tmp_path / "native"
    native.write_bytes(b"native fixture")
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"ffmpeg fixture")
    native_request = tmp_path / "native-request.json"
    native_request.write_text(
        json.dumps(
            pixels.seed_request(
                source=source,
                clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
                face=(24, 112, 112, 30),
            )
        ),
        encoding="utf-8",
    )
    native_response = tmp_path / "native-response.json"
    native_response.write_text(json.dumps(response), encoding="utf-8")
    detector_request = tmp_path / "detector-request.json"
    detector_request.write_text(json.dumps({"path": str(source)}), encoding="utf-8")
    detector_response = tmp_path / "detector-response.json"
    detector_response.write_text(
        json.dumps([detector_frame(30, True), detector_frame(31, False)]), encoding="utf-8"
    )
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": {"path": str(source), "sha256": _uncached_file_digest(source)},
                "nativeExecutable": {"path": str(native), "sha256": _uncached_file_digest(native)},
                "nativeRequest": {
                    "path": str(native_request),
                    "sha256": _uncached_file_digest(native_request),
                },
                "nativeResponse": {
                    "path": str(native_response),
                    "sha256": _uncached_file_digest(native_response),
                },
                "detectorRequest": {
                    "path": str(detector_request),
                    "sha256": _uncached_file_digest(detector_request),
                },
                "detectorResponse": {
                    "path": str(detector_response),
                    "sha256": _uncached_file_digest(detector_response),
                },
                "ffmpeg": {"path": str(ffmpeg), "sha256": _uncached_file_digest(ffmpeg)},
                "clock": {"pts": [30, 31], "timeBase": "1/30", "lastDuration": 1},
                "width": 160,
                "height": 160,
                "resets": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_run_verifier_and_false_approvals(
    config: Path, image: Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upper, "_code_paths", lambda: ())
    monkeypatch.setattr(tracker, "_decode", lambda *_: iter(((30, image), (31, image))))
    output = config.parent / "result"
    pixels.main([str(config), str(output)])
    pixels.main([str(config), str(output), "--verify"])
    report = json.loads((output / "partial-pixel-track.json").read_text(encoding="utf-8"))
    assert report["requestedFrames"] == 2
    assert report["completeTrack"] is True
    assert report["observations"][1]["face_observed"] is False
    assert report["headNearVerified"] is False
    assert report["captionPlacementApproved"] is False
    assert report["seedAnchorPolicy"] == "same-pts-measured-face-interior-v1"
    assert report["seedDetectorRegion"] == [24, 112, 112, 30]
    assert (output / "partial-pixel-track.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("fault", ["existing", "schema", "clock", "detector-source", "request"])
def test_run_binding_guards(config: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    raw = json.loads(config.read_text(encoding="utf-8"))
    if fault == "schema":
        raw["unexpected"] = True
    elif fault == "clock":
        raw["clock"]["timeBase"] = 30
    elif fault == "detector-source":
        request = Path(raw["detectorRequest"]["path"])
        request.write_text('{"path":"/other"}', encoding="utf-8")
        raw["detectorRequest"]["sha256"] = _uncached_file_digest(request)
    elif fault == "request":
        request = Path(raw["nativeRequest"]["path"])
        request.write_text("{}", encoding="utf-8")
        raw["nativeRequest"]["sha256"] = _uncached_file_digest(request)
    config.write_text(json.dumps(raw), encoding="utf-8")
    output = config.parent / "result"
    if fault == "existing":
        output.mkdir()
    with pytest.raises(ValueError):
        pixels.run(config, output)


def test_missing_seed_emits_request_not_old_height(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upper, "_code_paths", lambda: ())
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: None)
    output = config.parent / "missing"
    pixels.run(config, output)
    pixels.verify(config, output)
    report = json.loads((output / "partial-pixel-track.json").read_text(encoding="utf-8"))
    assert report["completeTrack"] is False
    assert report["measurementRequestReason"] == "measured_independent_upper_seed_unavailable"
    assert report["observations"] == []


def test_verifier_rejects_promoted_approval(
    config: Path, image: Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upper, "_code_paths", lambda: ())
    monkeypatch.setattr(tracker, "_decode", lambda *_: iter(((30, image), (31, image))))
    output = config.parent / "result"
    path = pixels.run(config, output)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["headNearVerified"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="unapproved"):
        pixels.verify(config, output)


def test_generic_producer_output_binding(config: Path) -> None:
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["nativeResponse"]["sha256"] = None
    config.write_text(json.dumps(raw), encoding="utf-8")
    output = config.parent / "bound.json"
    pixels.main([str(config), str(output), "--bind-native"])
    bound = json.loads(output.read_text(encoding="utf-8"))
    assert bound["nativeResponse"]["sha256"] == _uncached_file_digest(
        Path(raw["nativeResponse"]["path"])
    )
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("fault", ["bound", "path", "outside", "existing", "operation"])
def test_invalid_or_duplicate_producer_binding(config: Path, fault: str) -> None:
    raw = json.loads(config.read_text(encoding="utf-8"))
    if fault != "bound":
        raw["nativeResponse"]["sha256"] = None
    if fault == "path":
        raw["nativeResponse"]["path"] = 5
    if fault == "outside":
        raw["nativeResponse"]["path"] = "/outside"
    config.write_text(json.dumps(raw), encoding="utf-8")
    output = config.parent / "bound.json"
    if fault == "existing":
        output.write_bytes(b"preserved")
    if fault == "operation":
        with pytest.raises(SystemExit):
            pixels.main([str(config), str(output), "--bind-native", "--verify"])
    else:
        with pytest.raises(ValueError):
            pixels.bind_native(config, output)


def test_low_confidence_detection_not_reaccepted() -> None:
    frame = detector_frame(30, True)
    record(rows(frame["objects"])[0])["confidence"] = 0.2
    record(rows(record(frame["faceFrame"])["faces"])[0])["confidence"] = 0.2
    with pytest.raises(ValueError, match="actual accepted"):
        pixels.detector_seed([frame], clock=VideoFrameClock((30,), Fraction(1, 30), 1))


def test_physical_long_frame_extent_is_unknown(image: Image, seed: pixels.PartialPixelSeed) -> None:
    result = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30, 90), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (90, image)),
    )
    assert result[-1].candidate_upper_y is None
    assert result[-1].face_observed is False
    assert result[-1].reason == "unsupported-frame-extent"


@pytest.fixture
def approximate_seed(seed: pixels.PartialPixelSeed) -> pixels.PartialPixelSeed:
    return replace(
        seed,
        anchor_regions=tuple((x, y, 16, 16) for y in (64, 80, 96, 112) for x in (24, 56, 88, 120)),
    )


def test_face_motion_approximation_preserves_missing_and_exposes_every_model(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    donor = np.asarray(
        cv2.warpAffine(image, np.array([[1, 0, 0], [0, 1, 2]], np.float32), (160, 160)), np.uint8
    )
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, donor)),
    )
    assert result[-1].reason is None
    assert result[-1].face_observed is False
    assert result[-1].estimated_upper_y == pytest.approx(18, abs=0.3)
    assert result[-1].maximum_model_disagreement is not None
    assert result[-1].maximum_model_disagreement < 0.5
    assert len(result[-1].models) == 2
    model = result[-1].models[0]
    assert model["acceptedByExistingMotionContract"] is True
    assert model["reason"] is None
    assert model["minimumInliers"] == 12
    assert model["minimumInlierRatio"] == 0.75
    assert model["roundtripLimitPx"] == 0.5
    assert model["appearanceLimit"] == 12
    assert record(model["deformationGuard"])["status"] == "passed-by-existing-contract"
    assert record(model["deformationGuard"])["scaleFromAcceptedControls"] == pytest.approx(
        1, abs=0.01
    )
    assert len(rows(model["roundtripError"])) == len(rows(model["sourceFeatures"]))
    assert len(rows(model["appearanceError"])) == len(rows(model["currentFeatures"]))


@pytest.mark.parametrize("policy", ["legacy", "uncertain", "corroborated"])
def test_disjoint_face_cohorts_are_not_upper_hud_or_chest(
    image: Image, approximate_seed: pixels.PartialPixelSeed, policy: str
) -> None:
    left, right = pixels.face_feature_cohorts(image, approximate_seed)
    assert min(len(left), len(right)) >= 12
    assert not set(map(tuple, left)).intersection(map(tuple, right))
    assert np.all(left[:, 1] >= 64)
    assert np.all(right[:, 1] < 128)
    donor = np.asarray(
        cv2.warpAffine(image, np.array([[1, 0, 0], [0, 1, 2]], np.float32), (160, 160)), np.uint8
    )
    donor[:56] = image[:56]
    donor[144:] = 0
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, donor)),
        maximum_target_y_uncertainty_px=pixels.target_y_budget(160) if policy != "legacy" else None,
        corroborated=policy == "corroborated",
    )
    assert result[-1].reason is None
    assert result[-1].estimated_upper_y == pytest.approx(18, abs=0.3)


def test_current_refresh_replenishes_accepted_cohorts_without_stale_features(
    image: Image, approximate_seed: pixels.PartialPixelSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = derive_contour_motion
    calls: list[int] = []

    def lose_some_support(
        source: Image, donor: Image, features: Coordinates, contour: Coordinates
    ) -> TextureMotion:
        calls.append(len(features))
        motion = original(source, donor, features, contour)
        assert motion.reason is None
        accepted = np.asarray(motion.inliers).copy()
        accepted[::5] = False
        return replace(motion, inliers=tuple(bool(v) for v in accepted))

    monkeypatch.setattr(pixels, "derive_contour_motion", lose_some_support)
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31, 32, 33), Fraction(1, 30), 1),
        face_flags=(True, False, False, False),
        frames=((30, image), (31, image), (32, image), (33, image)),
    )
    assert len(result) == 4
    assert result[-1].reason is None
    assert result[-1].face_observed is False
    assert min(calls[4:]) >= min(calls[:2]) - 2
    assert result[-1].feature_refresh["status"] == "current-pixel-features-selected"
    assert result[-1].feature_refresh["newFaceObservation"] is False
    assert result[-1].feature_refresh["nativeInteriorRemeasured"] is False
    assert result[-1].estimated_upper_y == pytest.approx(16, abs=0.1)


def test_refresh_uses_current_pixels_in_both_projected_measured_interiors(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    controls = approximate_seed.contour.points
    moved = controls + np.asarray((3, 4), np.float32)
    current = image.copy()
    current[:64] = 255
    current[144:] = 255
    cohorts, audit = pixels.refresh_face_features(
        current, approximate_seed, controls, (moved, moved)
    )
    assert audit["status"] == "current-pixel-features-selected"
    assert min(len(cohorts[0]), len(cohorts[1])) >= 12
    assert np.all(cohorts[0][:, 1] >= 68)
    assert np.all(cohorts[0][:, 1] < 132)
    assert np.all(cohorts[1][:, 0] >= 27)
    assert np.all(cohorts[1][:, 0] < 139)
    assert not set(map(tuple, cohorts[0])).intersection(map(tuple, cohorts[1]))
    assert audit["maximumInteriorProjectionDisagreementPx"] == pytest.approx(0)


def test_refresh_does_not_fill_gaps_between_measured_tiles(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    current = np.zeros_like(image)
    current[64:128, 40:56] = image[64:128, 40:56]
    current[64:128, 72:88] = image[64:128, 72:88]
    controls = approximate_seed.contour.points
    cohorts, audit = pixels.refresh_face_features(
        current, approximate_seed, controls, (controls, controls)
    )
    assert audit["reason"] == "current_face_texture_unavailable"
    assert len(cohorts[0]) == 0
    assert len(cohorts[1]) == 0


@pytest.mark.parametrize("fault", ["disagreement", "outside", "blank"])
def test_refresh_rejects_unsupported_geometry_or_texture(
    image: Image, approximate_seed: pixels.PartialPixelSeed, fault: str
) -> None:
    controls = approximate_seed.contour.points
    first, second = controls.copy(), controls.copy()
    current = image
    if fault == "disagreement":
        second += (0, 7)
    elif fault == "outside":
        first += (200, 200)
        second += (200, 200)
    else:
        current = np.zeros_like(image)
    _, audit = pixels.refresh_face_features(current, approximate_seed, controls, (first, second))
    assert audit["status"] == "unsupported"
    assert audit["newFaceObservation"] is False
    assert audit["reason"] in {
        "competing_face_interior_geometry",
        "current_face_interior_unavailable",
        "current_face_texture_unavailable",
    }


def test_refresh_inward_square_survives_similarity_rotation_and_scale(
    approximate_seed: pixels.PartialPixelSeed,
) -> None:
    controls = approximate_seed.contour.points
    linear = np.asarray(((1, -0.1), (0.1, 1)), np.float32)
    moved = controls @ linear.T + (5, 0)
    regions, corners = pixels._projected_interior(approximate_seed.anchor_regions, controls, moved)
    assert len(regions) == 16
    assert corners.shape == (64, 2)
    assert regions[0] == (23, 68, 14, 14)


@pytest.mark.parametrize("policy", ["legacy", "uncertain", "corroborated"])
def test_current_refresh_handles_local_face_change_without_following_static_ui(
    image: Image, approximate_seed: pixels.PartialPixelSeed, policy: str
) -> None:
    first = np.asarray(
        cv2.warpAffine(image, np.array([[1, 0, 0], [0, 1, 2]], np.float32), (160, 160)), np.uint8
    )
    first[:56] = image[:56]
    first[144:] = image[144:]
    first[86:90, 64:68] = 0
    second = np.asarray(
        cv2.warpAffine(first, np.array([[1, 0, 0], [0, 1, 2]], np.float32), (160, 160)), np.uint8
    )
    second[:56] = image[:56]
    second[144:] = image[144:]
    second[90:94, 96:100] = 255
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31, 32), Fraction(1, 30), 1),
        face_flags=(True, False, False),
        frames=((30, image), (31, first), (32, second)),
        maximum_target_y_uncertainty_px=pixels.target_y_budget(160) if policy != "legacy" else None,
        corroborated=policy == "corroborated",
    )
    if policy == "corroborated":
        # The tiny canvas allocates only 2/3 px of conditional target error.
        # Local expression changes may exceed that basis bound; static UI is
        # never used to replace the rejected facial corroboration.
        assert result[-1].reason == "corroboration_target_budget_exceeded"
        assert result[-1].estimated_upper_y is None
        assert result[-1].feature_refresh["status"] == "not-attempted"
        return
    assert result[-1].reason is None
    assert result[-1].estimated_upper_y == pytest.approx(20, abs=0.4)
    assert result[-1].feature_refresh["status"] == "current-pixel-features-selected"
    assert result[-1].face_observed is False


def test_rejected_models_never_trigger_feature_refresh(
    image: Image, approximate_seed: pixels.PartialPixelSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    refresh = Mock(side_effect=AssertionError("Refresh may not repair a rejected model"))
    monkeypatch.setattr(pixels, "refresh_face_features", refresh)
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, np.zeros_like(image))),
    )
    assert result[-1].estimated_upper_y is None
    assert result[-1].feature_refresh["status"] == "not-attempted"
    assert result[-1].feature_refresh["reason"] == "texture_support_lost"
    refresh.assert_not_called()


@pytest.mark.parametrize("reason", ["source-cut", "scene", "zoom", "geometry"])
def test_approximate_current_motion_resets_never_hold(
    image: Image, approximate_seed: pixels.PartialPixelSeed, reason: str
) -> None:
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, image)),
        resets=((31, reason),),
    )
    assert result[-1].reason == reason
    assert result[-1].estimated_upper_y is None
    assert result[-1].models == ()


def test_approximate_occlusion_reports_unknown_models(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, np.zeros_like(image))),
    )
    assert result[-1].estimated_upper_y is None
    assert result[-1].reason is not None
    assert result[-1].models[0]["acceptedByExistingMotionContract"] is False
    assert record(result[-1].models[0]["deformationGuard"])["scaleFromAcceptedControls"] is None


def test_approximate_deformation_never_passes_from_counts(
    image: Image, approximate_seed: pixels.PartialPixelSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cv2,
        "estimateAffinePartial2D",
        Mock(
            return_value=(
                np.array([[1.2, 0, 0], [0, 1.2, 0]]),
                np.ones((50, 1)),
            )
        ),
    )
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, image)),
    )
    assert result[-1].reason == "motion_deformation_exceeded"
    assert result[-1].estimated_upper_y is None
    assert result[-1].models[1]["acceptedByExistingMotionContract"] is False


def test_approximate_independent_models_must_agree(
    image: Image, approximate_seed: pixels.PartialPixelSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = derive_contour_motion
    first = original(
        image,
        image,
        pixels.face_feature_cohorts(image, approximate_seed)[0],
        approximate_seed.contour.points,
    )
    assert first.contour is not None
    second = replace(first, contour=first.contour + np.array([0, 7], np.float32))
    monkeypatch.setattr(pixels, "derive_contour_motion", Mock(side_effect=[first, second]))
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, image)),
    )
    assert result[-1].reason == "competing_face_motion"
    assert result[-1].estimated_upper_y is None
    assert result[-1].maximum_model_disagreement == pytest.approx(7)


@pytest.mark.parametrize(
    "fault", ["flags", "seed", "reset", "clock", "geometry", "short", "extra", "apex"]
)
def test_approximate_bad_clock_or_support_is_not_fabricated(
    image: Image, approximate_seed: pixels.PartialPixelSeed, fault: str
) -> None:
    seed = approximate_seed
    flags: tuple[bool, ...] = (True, False)
    frames: tuple[tuple[int, Image], ...] = ((30, image), (31, image))
    resets: tuple[tuple[int, str], ...] = ()
    if fault == "flags":
        flags = (True,)
    elif fault == "seed":
        seed = replace(seed, contour=replace(seed.contour, time=Fraction(0)))
    elif fault == "reset":
        resets = ((32, "scene"),)
    elif fault == "clock":
        frames = ((29, image), (31, image))
    elif fault == "geometry":
        frames = ((30, np.zeros((32, 32, 3), np.uint8)),)
    elif fault == "short":
        frames = ((30, image),)
    elif fault == "extra":
        frames = ((30, image), (31, image), (32, image))
    else:
        seed = replace(
            seed,
            contour=replace(
                seed.contour, points=np.array([[40, 16], [60, 40], [80, 60]], np.float32)
            ),
        )
    with pytest.raises(ValueError):
        pixels.track_face_motion(
            seed,
            clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
            face_flags=flags,
            frames=frames,
            resets=resets,
        )


def test_approximate_blank_seed_and_physical_gap_remain_unknown(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    blank = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30,), Fraction(1, 30), 1),
        face_flags=(True,),
        frames=((30, np.zeros_like(image)),),
    )
    assert blank[0].reason == "independent_face_texture_unavailable"
    assert blank[0].estimated_upper_y is None
    gap = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 90), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (90, image)),
    )
    assert gap[-1].reason == "unsupported-frame-extent"
    assert gap[-1].estimated_upper_y is None


@pytest.fixture
def approximation_config(config: Path) -> Path:
    raw = record(json.loads(config.read_text(encoding="utf-8")))
    source = record(raw["source"])
    image = config.parent / "association.png"
    image.write_bytes(b"synthetic source-image association")
    observation = config.parent / "association.json"
    observation.write_text(
        json.dumps(
            {
                "version": 1,
                "source": {
                    "expectedSHA256FromExistingBoundConfig": source["sha256"],
                    "identityBeforeAfterEqual": True,
                },
                "operation": {
                    "commandExit": 0,
                    "framesWritten": 1,
                    "showinfoFirstPTS": 30,
                    "timeBase": "1/30",
                    "image": str(image),
                    "imageSHA256": _uncached_file_digest(image),
                },
                "visualObservation": {
                    "directlyViewed": True,
                    "pixelPerfectHairSegmentation": False,
                    "acceptedOriginalDetectorRectangle": [24.0, 112.0, 112.0, 30.0],
                },
            }
        ),
        encoding="utf-8",
    )
    raw["approximateAssociation"] = {
        "path": str(observation),
        "sha256": _uncached_file_digest(observation),
    }
    config.write_text(json.dumps(raw), encoding="utf-8")
    return config


def test_explicit_approximate_cli_binding_and_verifier(
    approximation_config: Path,
    image: Image,
    approximate_seed: pixels.PartialPixelSeed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(upper, "_code_paths", lambda: ())
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: approximate_seed)
    monkeypatch.setattr(tracker, "_decode", lambda *_: iter(((30, image), (31, image))))
    output = approximation_config.parent / "approximate"
    pixels.main([str(approximation_config), str(output), "--approx-face-motion"])
    pixels.main([str(approximation_config), str(output), "--approx-face-motion", "--verify"])
    report = record(json.loads((output / "partial-pixel-track.json").read_text(encoding="utf-8")))
    assert report["protocol"] == "derived-face-motion-approximate-head-v1"
    assert report["completeTrack"] is True
    assert report["currentMeasuredUpperPixels"] is False
    assert report["headNearVerified"] is False
    assert record(rows(report["observations"])[1])["face_observed"] is False
    with pytest.raises(ValueError):
        pixels.verify(approximation_config, output)
    record(rows(record(rows(report["observations"])[1])["models"])[0])[
        "acceptedByExistingMotionContract"
    ] = False
    (output / "partial-pixel-track.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="complete current face models"):
        pixels.verify(approximation_config, output, approximate=True)


@pytest.mark.parametrize("fault", ["source", "clock", "image", "association", "semantic"])
def test_approximate_seed_association_is_explicit_and_source_bound(
    approximation_config: Path, fault: str
) -> None:
    raw = record(json.loads(approximation_config.read_text(encoding="utf-8")))
    path = Path(str(record(raw["approximateAssociation"])["path"]))
    observation = record(json.loads(path.read_text(encoding="utf-8")))
    if fault == "source":
        record(observation["source"])["expectedSHA256FromExistingBoundConfig"] = "different"
    elif fault == "clock":
        record(observation["operation"])["showinfoFirstPTS"] = 29
    elif fault == "image":
        record(observation["operation"])["image"] = 4
    elif fault == "association":
        record(observation["visualObservation"])["directlyViewed"] = False
    else:
        record(observation["visualObservation"])["pixelPerfectHairSegmentation"] = True
    path.write_text(json.dumps(observation), encoding="utf-8")
    raw["approximateAssociation"] = {"path": str(path), "sha256": _uncached_file_digest(path)}
    approximation_config.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="association"):
        pixels.run(approximation_config, approximation_config.parent / "bad", approximate=True)


def test_approximate_binding_cannot_launch_a_new_native_probe(approximation_config: Path) -> None:
    with pytest.raises(SystemExit):
        pixels.main(
            [
                str(approximation_config),
                str(approximation_config.parent / "bad"),
                "--approx-face-motion",
                "--bind-native",
            ]
        )


def test_approximate_model_audit_has_a_finite_output_budget(
    approximation_config: Path,
    image: Image,
    approximate_seed: pixels.PartialPixelSeed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(upper, "_code_paths", lambda: ())
    monkeypatch.setattr(pixels, "MAX_APPROX_JSON_BYTES", 64)
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: approximate_seed)
    monkeypatch.setattr(tracker, "_decode", lambda *_: iter(((30, image), (31, image))))
    with pytest.raises(ValueError, match="bounded JSON"):
        pixels.run(approximation_config, approximation_config.parent / "bounded", approximate=True)


def test_approximate_verifier_rejects_promoted_original_face_flag(
    approximation_config: Path,
    image: Image,
    approximate_seed: pixels.PartialPixelSeed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(upper, "_code_paths", lambda: ())
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: approximate_seed)
    monkeypatch.setattr(tracker, "_decode", lambda *_: iter(((30, image), (31, image))))
    output = approximation_config.parent / "face-flag"
    path = pixels.run(approximation_config, output, approximate=True)
    report = record(json.loads(path.read_text(encoding="utf-8")))
    record(rows(report["observations"])[1])["face_observed"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="original face observations"):
        pixels.verify(approximation_config, output, approximate=True)


@pytest.fixture
def uncertainty_fit() -> TextureFit:
    source = np.asarray(
        [(x, y) for y in (700, 850, 1000, 1150, 1300) for x in (100, 220, 340, 460, 580, 700)],
        np.float32,
    )
    current = source + (1, 2)
    current[20:, 0] += 2.5
    current = np.asarray(current, np.float32)
    flags = np.ones(len(source), np.bool_)
    metrics = ContourTrackMetrics(
        current,
        flags,
        flags,
        flags,
        flags,
        np.zeros(len(source), np.float32),
        np.zeros(len(source), np.float32),
    )
    errors = np.linalg.norm(current - source - (1, 2), axis=1)
    inliers = tuple(bool(v) for v in errors <= 1.5)
    return TextureFit(
        source,
        metrics,
        ((1, 0, 1), (0, 1, 2)),
        inliers,
        tuple(float(v) for v in errors),
        (600, 600),
        (600, 450),
        "motion_support_insufficient",
    )


@pytest.fixture
def uncertainty_controls() -> Coordinates:
    return np.array(((200, 340), (300, 340), (400, 340)), np.float32)


def _uncertainty_audit(fitted: TextureFit, controls: Coordinates) -> dict[str, object]:
    decision = pixels.approximate_model(
        fitted, controls, maximum_target_y_uncertainty_px=8, width=1080, height=1920
    )
    motion = TextureMotion(
        None,
        fitted.metrics.points,
        fitted.inliers,
        fitted.residuals,
        fitted.metrics,
        fitted.reason,
        fitted,
    )
    audit = {
        **pixels._model_audit(fitted.source, controls, motion),
        "fitMatrix": fitted.matrix,
        "sourceControls": controls.tolist(),
        "approximateHeadContract": decision.audit,
    }
    return record(json.loads(json.dumps(audit)))


def test_conditional_y_contract_keeps_rejected_legacy_and_clearance_allocation(
    uncertainty_fit: TextureFit, uncertainty_controls: Coordinates
) -> None:
    audit = _uncertainty_audit(uncertainty_fit, uncertainty_controls)
    assert audit["acceptedByExistingMotionContract"] is False
    assert audit["reason"] == "motion_support_insufficient"
    result = pixels.validate_approximate_model(
        audit, maximum_target_y_uncertainty_px=8, width=1080, height=1920
    )
    assert result.accepted
    assert result.transferred_controls is not None
    np.testing.assert_allclose(result.transferred_controls, uncertainty_controls + (1, 2))
    assert result.audit["consensusCount"] == 30
    assert float(str(result.audit["conditionalBoundPx"])) <= 8
    assert result.audit["notTrueHairErrorGuarantee"] is True
    assert pixels.target_y_budget(1920) == 8
    assert pixels.target_y_budget(960) == 4


def test_extrapolation_not_local_residual_alone_controls_decision(
    uncertainty_fit: TextureFit, uncertainty_controls: Coordinates
) -> None:
    nearby = pixels.approximate_model(
        uncertainty_fit,
        uncertainty_controls,
        maximum_target_y_uncertainty_px=8,
        width=1080,
        height=1920,
    )
    distant = pixels.approximate_model(
        uncertainty_fit,
        uncertainty_controls - (0, 300),
        maximum_target_y_uncertainty_px=8,
        width=1080,
        height=1920,
    )
    assert nearby.accepted
    assert not distant.accepted
    assert distant.reason == "approximate_consensus_insufficient"
    assert float(str(distant.audit["targetSensitivityGain"])) > float(
        str(nearby.audit["targetSensitivityGain"])
    )


@pytest.mark.parametrize("fault", ["missing", "cluster", "nonrigid", "scale", "clip"])
def test_approximate_uncertainty_rejects_unsupported_models(
    uncertainty_fit: TextureFit, uncertainty_controls: Coordinates, fault: str
) -> None:
    fitted = uncertainty_fit
    controls = uncertainty_controls
    if fault == "missing":
        fitted = replace(fitted, matrix=None)
    elif fault == "cluster":
        source = fitted.source.copy()
        source[:20, 1] = 1000
        fitted = replace(fitted, source=source)
    elif fault == "nonrigid":
        current = fitted.metrics.points.copy()
        current[20:, 0] += 25
        fitted = replace(fitted, metrics=replace(fitted.metrics, points=current))
    elif fault == "scale":
        fitted = replace(
            fitted,
            metrics=replace(fitted.metrics, points=np.asarray(fitted.source * 1.2, np.float32)),
        )
    else:
        controls = np.asarray(controls + (0, -350), np.float32)
        # Zero residual permits isolating clipping from the consensus guard.
        fitted = replace(
            fitted,
            metrics=replace(fitted.metrics, points=np.asarray(fitted.source + (1, 2), np.float32)),
        )
    result = pixels.approximate_model(
        fitted, controls, maximum_target_y_uncertainty_px=8, width=1080, height=1920
    )
    assert not result.accepted
    assert result.transferred_controls is None


@pytest.mark.parametrize("budget", [0, -1, 9, float("nan")])
def test_uncertainty_budget_cannot_exceed_generic_clearance(
    uncertainty_fit: TextureFit, uncertainty_controls: Coordinates, budget: float
) -> None:
    with pytest.raises(ValueError, match="budget"):
        pixels.approximate_model(
            uncertainty_fit,
            uncertainty_controls,
            maximum_target_y_uncertainty_px=budget,
            width=1080,
            height=1920,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "coordinates",
        "statuses",
        "support",
        "matrix",
        "residual",
        "decision",
        "bool",
        "budget",
        "negative",
        "canvas",
    ],
)
def test_uncertainty_validator_rejects_invented_audit(
    uncertainty_fit: TextureFit, uncertainty_controls: Coordinates, fault: str
) -> None:
    audit = _uncertainty_audit(uncertainty_fit, uncertainty_controls)
    if fault == "coordinates":
        audit["sourceControls"] = [[1, 2]]
    elif fault == "statuses":
        audit["forwardStatus"] = []
    elif fault == "support":
        audit["supported"] = [False] * 30
    elif fault == "matrix":
        audit["fitMatrix"] = [[1, 0, 1], [0, 2, 2]]
    elif fault == "residual":
        audit["residuals"] = [0] * 30
    elif fault == "decision":
        contract = record(audit["approximateHeadContract"])
        contract["conditionalBoundPx"] = 0
        audit["approximateHeadContract"] = contract
    elif fault == "bool":
        audit["canvasStatus"] = [1] * 30
    elif fault == "negative":
        audit["roundtripError"] = [-1] * 30
    elif fault == "canvas":
        audit["currentFeatures"] = [[0, 0]] * 30
    else:
        contract = record(audit["approximateHeadContract"])
        contract["maximumTargetYModelUncertaintyPx"] = 2
        audit["approximateHeadContract"] = contract
    with pytest.raises(ValueError):
        pixels.validate_approximate_model(
            audit, maximum_target_y_uncertainty_px=8, width=1080, height=1920
        )


@pytest.mark.parametrize("reason", ["source-cut", "scene", "zoom", "geometry"])
def test_uncertain_motion_keeps_reset_missing_and_actual_pixels(
    image: Image, approximate_seed: pixels.PartialPixelSeed, reason: str
) -> None:
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, image)),
        resets=((31, reason),),
        maximum_target_y_uncertainty_px=pixels.target_y_budget(160),
    )
    assert result[-1].reason == reason
    assert result[-1].face_observed is False
    assert result[-1].estimated_upper_y is None


def test_uncertain_current_pixels_transport_and_refresh(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    donor = np.asarray(
        cv2.warpAffine(image, np.array([[1, 0, 1], [0, 1, 1]], np.float32), (160, 160)), np.uint8
    )
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, donor)),
        maximum_target_y_uncertainty_px=pixels.target_y_budget(160),
    )
    assert result[-1].reason is None
    assert result[-1].face_observed is False
    assert result[-1].estimated_upper_y == pytest.approx(17, abs=0.4)
    assert result[-1].feature_refresh["status"] == "current-pixel-features-selected"
    assert record(result[-1].models[0]["approximateHeadContract"])["accepted"] is True


def test_uncertain_cli_is_distinct_and_finite(
    approximation_config: Path,
    image: Image,
    approximate_seed: pixels.PartialPixelSeed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = record(json.loads(approximation_config.read_text(encoding="utf-8")))
    raw["approximateYPolicy"] = {
        "version": 1,
        "maximumTargetYModelUncertaintyPx": pixels.target_y_budget(160),
    }
    approximation_config.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: approximate_seed)
    monkeypatch.setattr(tracker, "_decode", lambda *_args: iter(((30, image), (31, image))))
    output = approximation_config.parent / "uncertain"
    pixels.main([str(approximation_config), str(output), "--approx-target-y"])
    pixels.main([str(approximation_config), str(output), "--approx-target-y", "--verify"])
    report = record(json.loads((output / "partial-pixel-track.json").read_text(encoding="utf-8")))
    assert report["protocol"] == pixels.UNCERTAIN_PROTOCOL
    assert report["currentMeasuredUpperPixels"] is False
    assert report["completeTrack"] is True
    with pytest.raises(SystemExit):
        pixels.main(
            [str(approximation_config), str(output), "--approx-target-y", "--approx-face-motion"]
        )
    with pytest.raises(SystemExit):
        pixels.main([str(approximation_config), str(output), "--approx-target-y", "--bind-native"])


def test_independently_valid_but_conflicting_models_reject(
    image: Image, seed: pixels.PartialPixelSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = foreground_features(image, seed.upper_regions)
    original = derive_contour_motion(image, image, features, seed.contour.points)
    assert original.contour is not None
    competing = replace(original, contour=original.contour + np.array([0, 7], np.float32))
    monkeypatch.setattr(pixels, "derive_contour_motion", Mock(side_effect=[original, competing]))
    result = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, image)),
    )
    assert result[-1].candidate_upper_y is None
    assert result[-1].reason == "competing_upper_and_head_motion"


def _pair_fit(
    fitted: TextureFit, *, source_shift: float = 0, contaminated: bool = False
) -> TextureFit:
    source = np.asarray(fitted.source + (source_shift, 0), np.float32)
    current = np.asarray(source + (1, 2), np.float32)
    if contaminated:
        current[20, 0] += 2.5
        current[21:, 0] += 20
    residual = np.linalg.norm(current - source - (1, 2), axis=1)
    selected = residual <= 1.5
    span = np.ptp(source[selected], axis=0)
    return replace(
        fitted,
        source=source,
        metrics=replace(fitted.metrics, points=current),
        inliers=tuple(bool(v) for v in selected),
        residuals=tuple(float(v) for v in residual),
        inlier_span=(float(span[0]), float(span[1])),
        reason="motion_support_insufficient" if contaminated else None,
    )


def test_complete_primary_and_independent_basis_do_not_relabel_weak_cohort(
    uncertainty_fit: TextureFit, uncertainty_controls: Coordinates
) -> None:
    weak = _uncertainty_audit(_pair_fit(uncertainty_fit, contaminated=True), uncertainty_controls)
    primary = _uncertainty_audit(_pair_fit(uncertainty_fit, source_shift=0.5), uncertainty_controls)
    assert record(weak["approximateHeadContract"])["accepted"] is False
    assert record(weak["approximateHeadContract"])["consensusRatio"] == 0.7
    result = pixels.validate_model_pair(
        [weak, primary], maximum_target_y_uncertainty_px=8, width=1080, height=1920
    )
    assert result.accepted
    assert result.audit["primaryIndex"] == 1
    assert result.audit["completeModelAccepted"] == [False, True]
    assert result.estimated_upper_y == 342
    evidence = record(rows(result.audit["corroborators"])[0])
    assert evidence["accepted"] is True
    assert evidence["wholeCohortAccepted"] is False
    assert evidence["wholeCohortReason"] == "approximate_consensus_insufficient"
    assert evidence["originalBasisMaximumResidualPx"] == 0
    assert evidence["basisCount"] == 20
    assert evidence["notWholeCohortApproval"] is True


@pytest.mark.parametrize("fault", ["both-weak", "overlap", "different-target", "clustered"])
def test_corroborated_pair_rejects_missing_primary_or_independent_spatial_agreement(
    uncertainty_fit: TextureFit, uncertainty_controls: Coordinates, fault: str
) -> None:
    weak = _pair_fit(uncertainty_fit, contaminated=True)
    strong = _pair_fit(uncertainty_fit, source_shift=0.5)
    controls = uncertainty_controls
    if fault == "both-weak":
        strong = _pair_fit(uncertainty_fit, source_shift=0.5, contaminated=True)
    elif fault == "overlap":
        strong = _pair_fit(uncertainty_fit)
    elif fault == "different-target":
        controls = np.asarray(controls + (0, 7), np.float32)
    else:
        source = weak.source.copy()
        source[:20, 1] = 1000
        current = np.asarray(source + (1, 2), np.float32)
        current[20, 0] += 2.5
        current[21:, 0] += 20
        weak = replace(weak, source=source, metrics=replace(weak.metrics, points=current))
    result = pixels.validate_model_pair(
        [_uncertainty_audit(weak, controls), _uncertainty_audit(strong, uncertainty_controls)],
        maximum_target_y_uncertainty_px=8,
        width=1080,
        height=1920,
    )
    assert not result.accepted
    assert result.transferred_controls is None
    assert result.estimated_upper_y is None
    assert (
        result.reason
        == {
            "both-weak": "complete_primary_unavailable",
            "overlap": "corroboration_cohorts_overlap",
            "different-target": "competing_corroborated_target",
            "clustered": "corroboration_basis_unavailable",
        }[fault]
    )


@pytest.mark.parametrize("fault", ["missing", "residual", "deformation", "budget", "clipped"])
def test_corroborator_retains_strict_basis_deformation_and_conditional_target_bounds(
    uncertainty_fit: TextureFit, uncertainty_controls: Coordinates, fault: str
) -> None:
    fitted = _pair_fit(uncertainty_fit, contaminated=True)
    controls = uncertainty_controls
    if fault == "missing":
        fitted = replace(fitted, matrix=None)
    elif fault == "residual":
        fitted = replace(fitted, residuals=(2.0,) * len(fitted.source))
    elif fault == "deformation":
        fitted = replace(fitted, matrix=((1.2, 0, 1), (0, 1.2, 2)))
    elif fault == "budget":
        current = fitted.metrics.points.copy()
        current[:20:2, 0] += 1.4
        current[1:20:2, 0] -= 1.4
        fitted = replace(fitted, metrics=replace(fitted.metrics, points=current))
        controls = np.asarray(uncertainty_controls - (0, 300), np.float32)
    else:
        controls = np.asarray(uncertainty_controls - (0, 350), np.float32)
    decision = pixels.approximate_model(
        fitted, controls, maximum_target_y_uncertainty_px=8, width=1080, height=1920
    )
    if fault == "budget":
        decision = pixels.approximate_model(
            fitted, controls, maximum_target_y_uncertainty_px=2, width=1080, height=1920
        )
    result = pixels._corroborating_model(fitted, controls, decision, width=1080, height=1920)
    assert not result.accepted
    assert result.transferred_controls is None
    assert (
        result.reason
        == {
            "missing": "corroboration_basis_unavailable",
            "residual": "corroboration_original_basis_residual_exceeded",
            "deformation": "corroboration_deformation_exceeded",
            "budget": "corroboration_target_budget_exceeded",
            "clipped": "corroboration_target_clipped",
        }[fault]
    )


@pytest.mark.parametrize("corroborated", [False, True])
def test_current_corroborated_motion_preserves_missing_reset_and_current_pixels(
    image: Image, approximate_seed: pixels.PartialPixelSeed, corroborated: bool
) -> None:
    donor = np.asarray(
        cv2.warpAffine(image, np.array([[1, 0, 1], [0, 1, 1]], np.float32), (160, 160)), np.uint8
    )
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31, 32), Fraction(1, 30), 1),
        face_flags=(True, False, False),
        frames=((30, image), (31, donor), (32, donor)),
        resets=((32, "source-cut"),),
        maximum_target_y_uncertainty_px=pixels.target_y_budget(160),
        corroborated=corroborated,
    )
    assert result[1].reason is None
    assert result[1].face_observed is False
    assert result[1].estimated_upper_y == pytest.approx(17, abs=0.4)
    assert result[-1].reason == "source-cut"
    assert result[-1].estimated_upper_y is None
    if corroborated:
        assert record(result[1].feature_refresh["modelPairDecision"])["accepted"] is True
        assert result[1].feature_refresh["nextFrameRequiresCompleteModels"] is False


def test_corroborated_cli_and_verifier_require_authentic_pair_decision(
    approximation_config: Path,
    image: Image,
    approximate_seed: pixels.PartialPixelSeed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = record(json.loads(approximation_config.read_text(encoding="utf-8")))
    raw["approximateYPolicy"] = {
        "version": 1,
        "maximumTargetYModelUncertaintyPx": pixels.target_y_budget(160),
    }
    approximation_config.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: approximate_seed)
    monkeypatch.setattr(tracker, "_decode", lambda *_args: iter(((30, image), (31, image))))
    output = approximation_config.parent / "corroborated"
    pixels.main([str(approximation_config), str(output), "--approx-corroborated-y"])
    pixels.main([str(approximation_config), str(output), "--approx-corroborated-y", "--verify"])
    path = output / "partial-pixel-track.json"
    report = record(json.loads(path.read_text(encoding="utf-8")))
    assert report["protocol"] == pixels.CORROBORATED_PROTOCOL
    assert report["currentMeasuredUpperPixels"] is False
    assert report["completeTrack"] is True
    observation = record(rows(report["observations"])[1])
    refresh = record(observation["feature_refresh"])
    pair = record(refresh["modelPairDecision"])
    pair["primaryIndex"] = 1 - integer(pair["primaryIndex"])
    refresh["modelPairDecision"] = pair
    observation["feature_refresh"] = refresh
    report["observations"] = [rows(report["observations"])[0], observation]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="decision differs"):
        pixels.verify(approximation_config, output, corroborated=True)
    with pytest.raises(SystemExit):
        pixels.main(
            [str(approximation_config), str(output), "--approx-corroborated-y", "--approx-target-y"]
        )
    with pytest.raises(ValueError, match="explicit target Y budget"):
        pixels.track_face_motion(
            approximate_seed,
            clock=VideoFrameClock((30,), Fraction(1, 30), 1),
            face_flags=(True,),
            frames=((30, image),),
            corroborated=True,
        )


def test_corroborated_rejected_missing_fit_remains_an_unapproved_diagnostic(
    approximation_config: Path,
    image: Image,
    approximate_seed: pixels.PartialPixelSeed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = record(json.loads(approximation_config.read_text(encoding="utf-8")))
    raw["approximateYPolicy"] = {
        "version": 1,
        "maximumTargetYModelUncertaintyPx": pixels.target_y_budget(160),
    }
    approximation_config.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: approximate_seed)
    monkeypatch.setattr(
        tracker, "_decode", lambda *_args: iter(((30, image), (31, np.zeros_like(image))))
    )
    output = approximation_config.parent / "unsupported-corroborated"
    pixels.run(approximation_config, output, corroborated=True)
    pixels.verify(approximation_config, output, corroborated=True)
    path = output / "partial-pixel-track.json"
    report = record(json.loads(path.read_text(encoding="utf-8")))
    assert report["completeTrack"] is False
    assert report["headNearVerified"] is False
    observation = record(rows(report["observations"])[1])
    assert observation["estimated_upper_y"] is None
    assert record(record(observation["feature_refresh"])["modelPairDecision"])["accepted"] is False
    observation["estimated_upper_y"] = 16
    report["observations"] = [rows(report["observations"])[0], observation]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        pixels.verify(approximation_config, output, corroborated=True)


def test_spatial_current_motion_retains_missing_status_and_supported_endpoint_history(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    first = np.asarray(
        cv2.warpAffine(image, np.array([[1, 0, 0], [0, 1, 1]], np.float32), (160, 160)), np.uint8
    )
    second = np.asarray(
        cv2.warpAffine(first, np.array([[1, 0, 0], [0, 1, 1]], np.float32), (160, 160)), np.uint8
    )
    first[:56] = image[:56]
    second[:56] = image[:56]
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31, 32), Fraction(1, 30), 1),
        face_flags=(True, False, False),
        frames=((30, image), (31, first), (32, second)),
        maximum_target_y_uncertainty_px=pixels.target_y_budget(160),
        corroborated=True,
        spatial_stable=True,
    )
    assert len(result) == 3
    assert result[-1].reason is None
    assert result[-1].face_observed is False
    assert result[-1].estimated_upper_y == pytest.approx(18, abs=0.3)
    refresh = result[-1].feature_refresh
    assert refresh["selectionHistoryPTS"] == 31
    selection = record(refresh["spatialSelection"])
    assert sum(integer(v) for v in rows(selection["cellSelectedContinued"])) >= 24
    assert selection["nextFrameRevalidatesEverySelectedPoint"] is True
    assert record(refresh["modelPairDecision"])["accepted"] is True


@pytest.mark.parametrize("reason", ["source-cut", "scene", "zoom", "geometry"])
def test_spatial_policy_never_refreshes_after_reset(
    image: Image, approximate_seed: pixels.PartialPixelSeed, reason: str
) -> None:
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, image)),
        resets=((31, reason),),
        maximum_target_y_uncertainty_px=pixels.target_y_budget(160),
        corroborated=True,
        spatial_stable=True,
    )
    assert result[-1].reason == reason
    assert result[-1].face_observed is False
    assert result[-1].estimated_upper_y is None
    assert result[-1].feature_refresh["status"] == "not-attempted"


def test_spatial_blank_face_does_not_follow_static_hud_or_refresh_on_rejection(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    current = image.copy()
    current[56:144] = 0
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, current)),
        maximum_target_y_uncertainty_px=pixels.target_y_budget(160),
        corroborated=True,
        spatial_stable=True,
    )
    assert result[-1].estimated_upper_y is None
    assert result[-1].reason == "complete_primary_unavailable"
    assert result[-1].feature_refresh["status"] == "not-attempted"
    assert result[-1].face_observed is False


def test_spatial_policy_requires_real_accepted_history_and_distinct_contract(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    with pytest.raises(ValueError, match="distinct primary"):
        pixels.track_face_motion(
            approximate_seed,
            clock=VideoFrameClock((30,), Fraction(1, 30), 1),
            face_flags=(True,),
            frames=((30, image),),
            spatial_stable=True,
        )
    with pytest.raises(ValueError, match="accepted current"):
        pixels.refresh_face_features(
            image,
            approximate_seed,
            approximate_seed.contour.points,
            (approximate_seed.contour.points, approximate_seed.contour.points),
            spatial_stable=True,
        )
    with pytest.raises(ValueError, match="actual accepted fit"):
        pixels._continued_current_features(
            TextureMotion(
                None,
                np.empty((0, 2), np.float32),
                (),
                (),
                derive_contour_motion(
                    image,
                    image,
                    foreground_features(image, approximate_seed.anchor_regions),
                    approximate_seed.contour.points,
                ).metrics,
                "missing",
            )
        )


def test_spatial_cli_shared_references_roundtrip_exact_guards_and_detect_tampering(
    approximation_config: Path,
    image: Image,
    approximate_seed: pixels.PartialPixelSeed,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = record(json.loads(approximation_config.read_text(encoding="utf-8")))
    raw["approximateYPolicy"] = {
        "version": 1,
        "maximumTargetYModelUncertaintyPx": pixels.target_y_budget(160),
    }
    raw["spatialFeaturePolicy"] = {"version": 1, "kind": pixels.SPATIAL_POLICY}
    approximation_config.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: approximate_seed)
    monkeypatch.setattr(tracker, "_decode", lambda *_args: iter(((30, image), (31, image))))
    output = approximation_config.parent / "spatial"
    pixels.main([str(approximation_config), str(output), "--approx-spatial-stable-y"])
    pixels.main([str(approximation_config), str(output), "--approx-spatial-stable-y", "--verify"])
    path = output / "partial-pixel-track.json"
    report = record(json.loads(path.read_text(encoding="utf-8")))
    assert report["protocol"] == "derived-face-motion-spatial-stable-head-v1"
    assert report["headNearVerified"] is False
    encoded = record(rows(report["observations"])[1])
    assert "originalMeasuredInteriorRegions" not in record(encoded["feature_refresh"])
    expanded = pixels.spatial_observations(report)
    assert record(rows(expanded[1]["models"])[0])["minimumInlierRatio"] == 0.75
    assert record(expanded[1]["feature_refresh"])["originalMeasuredInteriorRegions"] == [
        list(r) for r in approximate_seed.anchor_regions
    ]
    record(record(report["spatialModelGuardDefaults"])["model"])["minimumInlierRatio"] = 0.7
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="guard references"):
        pixels.verify(approximation_config, output, spatial_stable=True)
    with pytest.raises(SystemExit):
        pixels.main(
            [
                str(approximation_config),
                str(output),
                "--approx-spatial-stable-y",
                "--approx-corroborated-y",
            ]
        )


@pytest.fixture
def spatial_report(
    approximation_config: Path,
    approximate_seed: pixels.PartialPixelSeed,
    image: Image,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    raw = record(json.loads(approximation_config.read_text(encoding="utf-8")))
    raw["approximateYPolicy"] = {
        "version": 1,
        "maximumTargetYModelUncertaintyPx": pixels.target_y_budget(160),
    }
    raw["spatialFeaturePolicy"] = {"version": 1, "kind": pixels.SPATIAL_POLICY}
    approximation_config.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: approximate_seed)
    monkeypatch.setattr(tracker, "_decode", lambda *_args: iter(((30, image), (31, image))))
    return pixels.run(
        approximation_config, approximation_config.parent / "spatial-fixture", spatial_stable=True
    )


@pytest.mark.parametrize(
    "fault",
    [
        "history",
        "selection-order",
        "counts",
        "empty-support",
        "duplicate-origin",
        "wrong-cohort",
        "nested-origin",
        "guard-shadow",
        "guard-reference",
        "tile-reference",
        "tile-shadow",
        "model-reference",
        "artifact-root",
    ],
)
def test_spatial_tampering_cannot_relabel_current_support(
    spatial_report: Path, approximation_config: Path, fault: str
) -> None:
    report = record(json.loads(spatial_report.read_text(encoding="utf-8")))
    observation = record(rows(report["observations"])[1])
    refresh = record(observation["feature_refresh"])
    selection = record(refresh["spatialSelection"])
    if fault == "history":
        refresh["selectionHistoryPTS"] = 29
    elif fault == "selection-order":
        selection["selectionBeforeNextFrame"] = False
    elif fault == "counts":
        selection["selectedFeatureCounts"] = [129, 0]
    elif fault == "empty-support":
        selection["cellEligiblePixels"] = [0, 0, 0, 0]
    elif fault == "duplicate-origin":
        rows(selection["selectedOrigins"])[1] = rows(selection["selectedOrigins"])[0]
        selection["selectedFeatureCounts"] = [
            len(rows(rows(selection["selectedOrigins"])[0])),
            len(rows(rows(selection["selectedOrigins"])[0])),
        ]
        selection["cellSelectedContinued"] = [
            sum(integer(v) for v in rows(selection["selectedFeatureCounts"])),
            0,
            0,
            0,
        ]
    elif fault == "wrong-cohort":
        groups = rows(selection["selectedOrigins"])
        group = rows(groups[0])
        group[0] = ["continued-1", 0]
        groups[0] = group
        selection["selectedOrigins"] = groups
    elif fault == "nested-origin":
        groups = rows(selection["selectedOrigins"])
        group = rows(groups[0])
        group[0] = ["fresh", []]
        groups[0] = group
        selection["selectedOrigins"] = groups
    elif fault == "tile-reference":
        refresh["originalMeasuredInteriorRegionsRef"] = "other-seed"
    elif fault == "tile-shadow":
        refresh["originalMeasuredInteriorRegions"] = []
    elif fault == "model-reference":
        record(rows(observation["models"])[0])["index"] = 99
    elif fault == "artifact-root":
        report["spatialModelArtifactRoot"] = str(spatial_report.parent / "absent")
    else:
        reference = record(rows(report["spatialModelRecords"])[0])
        path = Path(str(reference["path"]))
        payload = record(json.loads(path.read_text(encoding="utf-8")))
        model = record(rows(payload["models"])[0])
        if fault == "guard-shadow":
            model["minimumInlierRatio"] = 0.7
        else:
            model["sharedGuardRef"] = "old-reference"
        path.write_text(json.dumps(payload), encoding="utf-8")
        reference["sha256"] = _uncached_file_digest(path)
    spatial_report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises((ValueError, ReplayError)):
        pixels.verify(approximation_config, spatial_report.parent, spatial_stable=True)


@pytest.mark.parametrize(
    "policy",
    [
        {"version": True, "kind": pixels.SPATIAL_POLICY},
        {"version": 1, "kind": "old-face-reference"},
        {"version": 1, "kind": pixels.SPATIAL_POLICY, "fallback": True},
    ],
)
def test_spatial_config_requires_exact_opt_in_policy(policy: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Explicit current"):
        pixels._validate_spatial_policy(policy)


def test_spatial_models_are_bounded_per_record_and_reused_once(
    tmp_path: Path,
) -> None:
    report: dict[str, object] = {
        "observations": [{"models": [{"serial": index} for index in range(70)]}]
    }
    pixels._write_spatial_models(report, tmp_path)
    assert len(rows(report["spatialModelRecords"])) == 3
    loaded = pixels._read_spatial_models(report)
    assert tuple(len(group) for group in loaded) == (32, 32, 6)
    assert record(loaded[2][5])["serial"] == 69


@pytest.mark.parametrize("fault", ["count", "single-byte-budget", "total-byte-budget"])
def test_spatial_model_serialization_never_widens_byte_or_count_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    count = 513 if fault == "count" else 1
    report: dict[str, object] = {"observations": [{"models": [{"field": "actual"}] * count}]}
    if fault == "single-byte-budget":
        monkeypatch.setattr(pixels, "MAX_APPROX_JSON_BYTES", 1)
    elif fault == "total-byte-budget":
        monkeypatch.setattr(pixels, "MAX_SPATIAL_MODEL_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="bounded"):
        pixels._write_spatial_models(report, tmp_path)
    assert not list(tmp_path.glob("models-*.json"))


def test_current_upper_arc_leaving_canvas_is_unknown(
    image: Image, seed: pixels.PartialPixelSeed, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cv2,
        "estimateAffinePartial2D",
        Mock(return_value=(np.array([[1, 0, -40], [0, 1, 0]]), np.ones((50, 1)))),
    )
    result = pixels.track_pixels(
        seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, image)),
    )
    assert result[-1].candidate_upper_y is None
    assert result[-1].reason == "derived_contour_clipped"


def test_local_current_models_keep_independent_tile_owners_and_raw_missing(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    donor = np.asarray(
        cv2.warpAffine(image, np.array([[1, 0, 0], [0, 1, 1]], np.float32), (160, 160)), np.uint8
    )
    donor[:56] = image[:56]
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, donor)),
        maximum_target_y_uncertainty_px=pixels.target_y_budget(160),
        corroborated=True,
        spatial_stable=True,
        local_support=True,
    )
    assert len(result) == 2
    assert result[-1].reason is None
    assert result[-1].face_observed is False
    assert result[-1].estimated_upper_y == pytest.approx(17, abs=0.3)
    refresh = result[-1].feature_refresh
    assert refresh["crossOwnerFullInteriorAgreementRequired"] is False
    assert refresh["currentTargetAgreementRequired"] is True
    assert refresh["newFaceObservation"] is False
    local = record(record(refresh["spatialSelection"])["predeclaredLocalSupport"])
    assert local["fitDependentSelection"] is False
    assert local["ownershipSelectedBeforeFit"] is True


@pytest.mark.parametrize("overlap", [False, True])
def test_local_models_reject_ambiguous_owners_but_allow_distinct_supported_geometry(
    image: Image, approximate_seed: pixels.PartialPixelSeed, overlap: bool
) -> None:
    controls = approximate_seed.contour.points
    first = cv2.getRotationMatrix2D((80, 16), 2 if overlap else 0, 1 if overlap else 0.96)
    second = cv2.getRotationMatrix2D((80, 16), -2 if overlap else 0, 1 if overlap else 1.04)
    moved = (controls @ first[:, :2].T + first[:, 2], controls @ second[:, :2].T + second[:, 2])
    assert float(np.linalg.norm(moved[0] - moved[1], axis=1).max()) < 6
    _, legacy = pixels.refresh_face_features(image, approximate_seed, controls, moved)
    cohorts, local = pixels.refresh_face_features(
        image,
        approximate_seed,
        controls,
        moved,
        continuing=(np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)),
        spatial_stable=True,
        local_support=True,
    )
    assert legacy["reason"] == "competing_face_interior_geometry"
    if overlap:
        assert local["reason"] == "current_local_ownership_unsupported"
        assert local["spatialSelection"] is None
    else:
        assert local["reason"] is None
        assert min(len(group) for group in cohorts) >= 12
    assert float(str(local["maximumCrossOwnerProjectionDifferencePx"])) > 6
    assert local["modelAgreementLimitPx"] == 6
    assert local["nextFrameRequiresCompletePrimaryAndCorroboration"] is True


@pytest.mark.parametrize("fault", ["raw-missing", "clock", "geometry", "blank-seed"])
def test_local_source_or_missing_geometry_is_not_fabricated(
    image: Image, approximate_seed: pixels.PartialPixelSeed, fault: str
) -> None:
    clock = VideoFrameClock((30,), Fraction(1, 30), 1)
    flags = (False,) if fault == "raw-missing" else (True,)
    frame_pts = 31 if fault == "clock" else 30
    current = image[:, :-1] if fault == "geometry" else image
    if fault == "blank-seed":
        current = np.zeros_like(image)
        result = pixels.track_face_motion(
            approximate_seed,
            clock=clock,
            face_flags=flags,
            frames=((frame_pts, current),),
            maximum_target_y_uncertainty_px=pixels.target_y_budget(160),
            corroborated=True,
            spatial_stable=True,
            local_support=True,
        )
        assert result[0].estimated_upper_y is None
        assert result[0].reason == "independent_face_texture_unavailable"
    else:
        with pytest.raises(ValueError):
            pixels.track_face_motion(
                approximate_seed,
                clock=clock,
                face_flags=flags,
                frames=((frame_pts, current),),
                maximum_target_y_uncertainty_px=pixels.target_y_budget(160),
                corroborated=True,
                spatial_stable=True,
                local_support=True,
            )


@pytest.mark.parametrize("reason", ["source-cut", "scene", "zoom", "geometry"])
def test_local_reset_ends_support_without_a_held_position(
    image: Image, approximate_seed: pixels.PartialPixelSeed, reason: str
) -> None:
    result = pixels.track_face_motion(
        approximate_seed,
        clock=VideoFrameClock((30, 31), Fraction(1, 30), 1),
        face_flags=(True, False),
        frames=((30, image), (31, image)),
        resets=((31, reason),),
        maximum_target_y_uncertainty_px=pixels.target_y_budget(160),
        corroborated=True,
        spatial_stable=True,
        local_support=True,
    )
    assert result[-1].reason == reason
    assert result[-1].estimated_upper_y is None
    assert result[-1].feature_refresh["status"] == "not-attempted"


@pytest.mark.parametrize("fault", ["blank", "outside", "owner-crossing"])
def test_local_refresh_failure_retains_unknown_and_actual_failure_reason(
    image: Image, approximate_seed: pixels.PartialPixelSeed, fault: str
) -> None:
    controls = approximate_seed.contour.points
    moved = controls.copy() + ((200, 200) if fault == "outside" else (0, 0))
    current = np.zeros_like(image) if fault == "blank" else image
    continuing = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    if fault == "owner-crossing":
        continuing = (np.asarray(((30, 120),), np.float32), continuing[1])
    _, audit = pixels.refresh_face_features(
        current,
        approximate_seed,
        controls,
        (moved, moved),
        continuing=continuing,
        spatial_stable=True,
        local_support=True,
    )
    assert audit["status"] == "unsupported"
    assert audit["reason"] in {
        "current_local_texture_unavailable",
        "current_local_ownership_unsupported",
    }
    assert audit["spatialSelection"] is None
    assert audit["nativeInteriorRemeasured"] is False


def test_local_contract_requires_its_explicit_current_spatial_parent(
    image: Image, approximate_seed: pixels.PartialPixelSeed
) -> None:
    with pytest.raises(ValueError, match="distinct spatial"):
        pixels.track_face_motion(
            approximate_seed,
            clock=VideoFrameClock((30,), Fraction(1, 30), 1),
            face_flags=(True,),
            frames=((30, image),),
            local_support=True,
        )
    with pytest.raises(ValueError, match="supported spatial"):
        pixels.refresh_face_features(
            image,
            approximate_seed,
            approximate_seed.contour.points,
            (approximate_seed.contour.points, approximate_seed.contour.points),
            local_support=True,
        )


@pytest.fixture
def local_report(
    approximation_config: Path,
    approximate_seed: pixels.PartialPixelSeed,
    image: Image,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    raw = record(json.loads(approximation_config.read_text(encoding="utf-8")))
    raw["approximateYPolicy"] = {
        "version": 1,
        "maximumTargetYModelUncertaintyPx": pixels.target_y_budget(160),
    }
    raw["spatialFeaturePolicy"] = {"version": 1, "kind": pixels.SPATIAL_POLICY}
    raw["localFeaturePolicy"] = {"version": 1, "kind": pixels.LOCAL_POLICY}
    approximation_config.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: approximate_seed)
    monkeypatch.setattr(tracker, "_decode", lambda *_args: iter(((30, image), (31, image))))
    path = pixels.run(
        approximation_config, approximation_config.parent / "local-fixture", local_support=True
    )
    pixels.verify(approximation_config, path.parent, local_support=True)
    return path


def test_local_cli_has_distinct_protocol_and_lossless_support_material(
    local_report: Path, approximation_config: Path
) -> None:
    pixels.main(
        [
            str(approximation_config),
            str(local_report.parent),
            "--approx-local-support-y",
            "--verify",
        ]
    )
    report = record(json.loads(local_report.read_text(encoding="utf-8")))
    assert report["protocol"] == pixels.LOCAL_PROTOCOL
    assert report["completeTrack"] is True
    assert report["headNearVerified"] is False
    expanded = pixels.spatial_observations(report, local_support=True)
    assert expanded[1]["face_observed"] is False
    assert len(rows(expanded[1]["models"])) == 2
    assert record(rows(expanded[1]["models"])[0])["minimumInlierRatio"] == 0.75
    with pytest.raises(ValueError, match="policy or guard"):
        pixels.spatial_observations(report)


@pytest.mark.parametrize(
    "fault",
    [
        "owners",
        "split",
        "budget",
        "fit-selection",
        "missing-index",
        "duplicate-index",
        "source-point",
        "foreign-point",
        "shape",
        "original-count",
        "geometry",
        "tile-shadow",
    ],
)
def test_local_tampering_cannot_omit_or_change_selected_current_support(
    local_report: Path, approximation_config: Path, fault: str
) -> None:
    report = record(json.loads(local_report.read_text(encoding="utf-8")))
    observation = record(rows(report["observations"])[0])
    selection = record(record(observation["feature_refresh"])["spatialSelection"])
    support = record(selection["predeclaredLocalSupport"])
    if fault == "owners":
        support["originalTileOwners"] = [1] * 16
    elif fault == "split":
        support["seedAreaMedianY"] = 1
    elif fault == "budget":
        support["maximumFeaturesPerOwner"] = 129
    elif fault == "fit-selection":
        support["fitDependentSelection"] = True
    elif fault == "missing-index":
        groups = rows(support["originalSelectedIndices"])
        group = rows(groups[0])
        group.pop()
        groups[0] = group
        support["originalSelectedIndices"] = groups
    elif fault == "duplicate-index":
        groups = rows(support["originalSelectedIndices"])
        group = rows(groups[0])
        group[0] = group[1]
        groups[0] = group
        support["originalSelectedIndices"] = groups
    elif fault == "source-point":
        groups = rows(selection["currentSelectedFeatures"])
        group = rows(groups[0])
        group[0] = [30, 110]
        groups[0] = group
        selection["currentSelectedFeatures"] = groups
    elif fault == "foreign-point":
        originals = rows(support["originalSelectedFeatures"])
        group = rows(originals[0])
        group[0] = [1, 1]
        originals[0] = group
        support["originalSelectedFeatures"] = originals
    elif fault == "shape":
        selection["currentSelectedFeatures"] = [[], [], []]
    elif fault == "original-count":
        record(support["originalSelection"])["selectedFeatureCounts"] = [129, 0]
    elif fault == "geometry":
        report["localGeometry"] = {"width": 159, "height": 160}
    else:
        report["seedAnchorRegions"] = [[1, 1, 1]]
    local_report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        pixels.verify(approximation_config, local_report.parent, local_support=True)


@pytest.mark.parametrize(
    "policy",
    [
        {"version": True, "kind": pixels.LOCAL_POLICY},
        {"version": 1, "kind": "old-reference"},
        {"version": 1, "kind": pixels.LOCAL_POLICY, "fallback": True},
    ],
)
def test_local_config_rejects_implicit_or_untyped_policy(policy: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Explicit predeclared"):
        pixels._validate_local_policy(policy)


@pytest.mark.parametrize("fault", ["model-source", "projection", "projection-policy", "origin"])
def test_local_transition_requires_the_previously_selected_owner_pixels(
    local_report: Path, approximation_config: Path, fault: str
) -> None:
    report = record(json.loads(local_report.read_text(encoding="utf-8")))
    observation = record(rows(report["observations"])[1])
    refresh = record(observation["feature_refresh"])
    if fault == "model-source":
        reference = record(rows(report["spatialModelRecords"])[0])
        path = Path(str(reference["path"]))
        payload = record(json.loads(path.read_text(encoding="utf-8")))
        model = record(rows(payload["models"])[0])
        features = rows(model["sourceFeatures"])
        features[0] = [30, 110]
        model["sourceFeatures"] = features
        path.write_text(json.dumps(payload), encoding="utf-8")
        reference["sha256"] = _uncached_file_digest(path)
    elif fault == "projection":
        projections = rows(refresh["derivedSupportRegions"])
        projections[0] = [16, 16, 8, 8]
        refresh["derivedSupportRegions"] = projections
    elif fault == "projection-policy":
        refresh["nextFrameRequiresCompletePrimaryAndCorroboration"] = False
    else:
        selection = record(refresh["spatialSelection"])
        groups = rows(selection["selectedOrigins"])
        group = rows(groups[0])
        group[0] = ["fresh", 255]
        groups[0] = group
        selection["selectedOrigins"] = groups
    local_report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        pixels.verify(approximation_config, local_report.parent, local_support=True)


@pytest.mark.parametrize("fault", ["seed-tiles", "controls", "after-reset", "initial-y"])
def test_local_control_chain_is_bound_to_the_actual_native_seed(
    approximation_config: Path,
    approximate_seed: pixels.PartialPixelSeed,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: approximate_seed)
    raw = record(json.loads(approximation_config.read_text(encoding="utf-8")))
    report: dict[str, object] = {
        "seedAnchorRegions": [list(tile) for tile in approximate_seed.anchor_regions]
    }
    observations: tuple[dict[str, object], ...] = (
        {"models": [], "estimated_upper_y": 16, "reason": None, "feature_refresh": {}},
    )
    if fault == "seed-tiles":
        report["seedAnchorRegions"] = []
    elif fault == "controls":
        observations = (
            {
                "models": [{"sourceControls": [[10, 10], [20, 10], [30, 10]]}],
                "reason": None,
                "feature_refresh": {},
            },
        )
    elif fault == "after-reset":
        observations = (
            {"models": [], "estimated_upper_y": None, "reason": "scene", "feature_refresh": {}},
            {"models": [], "estimated_upper_y": 16, "reason": None, "feature_refresh": {}},
        )
    else:
        observations[0]["estimated_upper_y"] = 40
    with pytest.raises(ValueError):
        pixels._verify_local_controls(raw, report, observations)


def test_local_missing_native_seed_is_unresolved_without_fake_observations(
    approximation_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pixels, "read_seed", lambda *_args, **_kwargs: None)
    raw = record(json.loads(approximation_config.read_text(encoding="utf-8")))
    pixels._verify_local_controls(raw, {"seedAnchorRegions": []}, ())
    with pytest.raises(ValueError, match="actual measured seed"):
        pixels._verify_local_controls(raw, {"seedAnchorRegions": []}, ({"models": []},))


@pytest.mark.parametrize("fault", ["coordinate-bound", "index-bound", "tile-dimensions"])
def test_local_metadata_bounds_are_preserved(fault: str) -> None:
    with pytest.raises(ValueError):
        if fault == "coordinate-bound":
            pixels._selected_coordinates([[0, 0]] * 129)
        elif fault == "index-bound":
            pixels._local_reference_rows([[0, 128]])
        else:
            pixels._local_region([24, 24, 16])
