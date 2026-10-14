"""Synthetic mask accounting; no real media or model inference."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from replay import delivery_mask_quality as quality
from replay import subtitle_quality_observe as observation
from replay.caption_io import FrameObservation
from replay.core import ReplayError
from replay.recordings import record
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image
from replay.subtitle_prepared import ImageContext
from replay.subtitle_retry import RepairEvidence


@pytest.fixture
def observed_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    background = np.random.default_rng(42).integers(20, 200, (192, 192, 3), dtype=np.uint8)
    original = background.copy()
    original[90:100, 90:100] = 255
    source, native, model, ffmpeg = (
        tmp_path / name for name in ("source", "native", "model", "ffmpeg")
    )
    source.write_bytes(b"source")
    native.write_bytes(b"native")
    model.write_bytes(b"model")
    ffmpeg.write_bytes(b"ffmpeg")
    monkeypatch.setattr(shutil, "which", lambda name: str(ffmpeg))

    def decode(
        source: Path, *, clock: VideoFrameClock, width: int, height: int, ffmpeg: Path
    ) -> tuple[Image, ...]:
        return background, original, background

    monkeypatch.setattr(observation, "decode_context", decode)

    class Detector:
        def __init__(self, model: Path, *, sha256: str) -> None:
            pass

        def detect(self, images: Sequence[Image]) -> tuple[Image, ...]:
            return tuple(np.zeros(image.shape[:2], np.uint8) for image in images)

    monkeypatch.setattr(quality, "LearnedTextDetector", Detector)

    def ocr(
        images: tuple[Image, ...], *, executable: Path, work_directory: Path
    ) -> tuple[ImageContext, ...]:
        return tuple(
            ImageContext(FrameObservation(0, 0, image.shape[1], image.shape[0], ()), 0)
            for image in images
        )

    monkeypatch.setattr(observation, "image_observations", ocr)
    producer = document(
        tmp_path / "producer.json",
        {
            "source": str(source),
            "sourceSHA256": quality.digest(source),
            "nativeExecutable": str(native),
            "nativeSHA256": quality.digest(native),
            "width": 192,
            "height": 192,
            "evidenceFrames": [1],
            "clock": {"pts": [0, 1, 2], "timeBase": "1/30", "lastDuration": 1},
            "learnedTextDetector": {
                "backend": "ppocrv3-db",
                "model": str(model),
                "sha256": quality.digest(model),
            },
        },
    )
    pixels = tmp_path / "pixels.npz"
    np.savez(
        pixels,
        source=original[90:100, 90:100],
        candidate=background[90:100, 90:100],
        glyph=np.full((10, 10), 255, np.uint8),
        repair=np.full((10, 10), 255, np.uint8),
    )
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"candidate")
    modules = tuple(
        Path(quality.__file__).with_name(f"subtitle_{name}.py")
        for name in (
            "candidate",
            "prepared",
            "detector",
            "stream",
            "repair",
            "inpaint",
            "video",
            "evidence",
            "clock",
            "plan",
            "retry",
        )
    )
    inputs = {
        str(path): quality.digest(path)
        for path in (*modules, source, native, model, ffmpeg, producer)
    }
    candidate = document(
        tmp_path / "candidate.json",
        {
            "inputs": inputs,
            "videoSHA256": quality.digest(video),
            "evidenceRuntime": {"numpy": np.__version__, "opencv": cv2.__version__},
            "render": {"frames": 3},
            "losslessEvidence": {
                "capturePoint": "after-encoder-pipe-write-before-lossy-encode",
                "pixelFormat": "bgr24",
                "frames": [
                    {
                        **pin(pixels),
                        "frame": 1,
                        "sourcePTS": 1,
                        "candidatePTS": 1,
                        "glyphPixels": 100,
                        "repairPixels": 100,
                        "methods": ["glyph-local-softening"],
                        "timeBase": "1/30",
                        "region": {"x": 90, "y": 90, "width": 10, "height": 10},
                    }
                ],
            },
        },
    )
    return document(
        tmp_path / "observe.json",
        {
            "version": 1,
            "producer": pin(producer),
            "candidate": pin(candidate),
            "contextProducer": pin(producer),
        },
    )


@pytest.mark.parametrize("fault", [None, "pin", "path", "link", "missing", "relative"])
def test_explicit_source_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"authorized synthetic source")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    producer: dict[str, object] = {"source": str(source), "sourceSHA256": quality.digest(source)}
    allowed: dict[str, object] = {"path": str(source), "sha256": quality.digest(source)}
    with pytest.raises(ReplayError):
        quality.observation_source(producer, None)
    if fault == "pin":
        allowed["sha256"] = "a" * 64
    elif fault == "path":
        allowed["path"] = str(checkout)
    elif fault == "link":
        linked = tmp_path / "linked"
        linked.symlink_to(source)
        producer["source"] = allowed["path"] = str(linked)
    elif fault == "missing":
        producer["source"] = allowed["path"] = str(tmp_path / "absent")
    elif fault == "relative":
        producer["source"] = allowed["path"] = "../source"
    if fault is None:
        assert quality.observation_source(producer, allowed) == source
    else:
        with pytest.raises(ReplayError):
            quality.observation_source(producer, allowed)


@pytest.mark.parametrize("method", ["glyph", "expanded-glyph", "glyph-local-softening"])
def test_bounded_nonsoftened_comparison(observed_request: Path, method: str) -> None:
    request = quality.read_json(observed_request)
    request["repairMethod"] = method
    document(observed_request, request)
    output = observed_request.parent / "comparison"
    if method == "glyph-local-softening":
        with pytest.raises(ReplayError, match="non-softened"):
            quality.observe_capture(
                observed_request, sha256=quality.digest(observed_request), directory=output
            )
        return
    result = quality.observe_capture(
        observed_request, sha256=quality.digest(observed_request), directory=output
    )
    assert result["actualMethods"] == [method]
    assert result["comparisonScope"] == "lossless-ROI-only-not-encoded-producer"
    assert result["accepted"] is False
    with np.load(output / "support-00000001.npz") as pixels:
        assert pixels["candidate"].shape == (10, 10, 3)
        assert np.count_nonzero(pixels["glyph"]) == 100


def test_observe_capture_internal_native_link(observed_request: Path) -> None:
    root = observed_request.parent
    linked = root / "native-link"
    linked.symlink_to(root / "native")
    producer = quality.read_json(root / "producer.json")
    producer["nativeExecutable"] = str(linked)
    document(root / "producer.json", producer)
    candidate = quality.read_json(root / "candidate.json")
    inputs = record(candidate["inputs"])
    inputs[str(linked)] = inputs.pop(str(root / "native"))
    inputs[str(root / "producer.json")] = quality.digest(root / "producer.json")
    candidate["inputs"] = inputs
    document(root / "candidate.json", candidate)
    repin(observed_request, "producer", root / "producer.json")
    repin(observed_request, "contextProducer", root / "producer.json")
    repin(observed_request, "candidate", root / "candidate.json")
    result = quality.observe_capture(
        observed_request,
        sha256=quality.digest(observed_request),
        directory=root / "linked-observations",
    )
    assert result["unknownPixels"] == [0]


def test_observe_capture_authorized_source(observed_request: Path) -> None:
    producer = quality.read_json(observed_request.parent / "producer.json")
    request = quality.read_json(observed_request)
    request["authorizedSource"] = {"path": producer["source"], "sha256": producer["sourceSHA256"]}
    document(observed_request, request)
    result = quality.observe_capture(
        observed_request,
        sha256=quality.digest(observed_request),
        directory=observed_request.parent / "authorized-observations",
    )
    assert result["unknownPixels"] == [0]


def test_observe_capture(observed_request: Path) -> None:
    result = quality.observe_capture(
        observed_request,
        sha256=quality.digest(observed_request),
        directory=observed_request.parent / "observations",
    )
    assert result["accepted"] is False
    assert record(result["sampleQuality"])["accepted"] is True
    assert result["unknownPixels"] == [0]
    assert result["fullCoverageVerified"] is False
    assert result["actualCompositorProvenanceVerified"] is False
    assert result["contextOrigin"] == [0, 0]
    assert result["sourcePTS"] == [1]
    assert result["actualMethods"] == ["glyph-local-softening"]
    assert result["qualityMethod"] == "glyph"
    with np.load(observed_request.parent / "observations/support-00000001.npz") as support:
        assert support["previous_xy"][95, 95].tolist() == [95.0, 95.0]
        assert np.count_nonzero(support["unknown"]) == 0


@pytest.mark.parametrize(
    "fault",
    ["runtime", "pin", "clock", "capture", "frame", "source", "archive", "pixels", "version"],
)
def test_observe_capture_rejects(observed_request: Path, fault: str) -> None:
    root = observed_request.parent
    candidate_path = root / "candidate.json"
    candidate = quality.read_json(candidate_path)
    if fault == "runtime":
        candidate["evidenceRuntime"] = {}
    elif fault == "pin":
        record(candidate["inputs"])[str(root / "source")] = "a" * 64
        (root / "source").write_bytes(b"changed")
    elif fault in {"clock", "capture", "frame"}:
        capture = record(candidate["losslessEvidence"])
        entries = quality.rows(capture["frames"])
        entry = record(entries[0])
        if fault == "clock":
            entry["sourcePTS"] = 9
        elif fault == "frame":
            entry["frame"] = 10
        else:
            capture["pixelFormat"] = "rgb24"
        capture["frames"] = [entry]
        candidate["losslessEvidence"] = capture
    elif fault == "source":
        context = quality.read_json(root / "producer.json")
        context["sourceSHA256"] = "a" * 64
        context_path = document(root / "context.json", context)
        repin(observed_request, "contextProducer", context_path)
    elif fault in {"archive", "pixels"}:
        archive = root / "wrong.npz"
        if fault == "archive":
            np.savez(archive, other=np.zeros((2, 2), np.uint8))
        else:
            np.savez(
                archive,
                source=np.zeros((10, 10, 3), np.uint8),
                candidate=np.zeros((10, 10, 3), np.uint8),
                glyph=np.zeros((10, 10), np.uint8),
                repair=np.zeros((10, 10), np.uint8),
            )
        capture = record(candidate["losslessEvidence"])
        entry = record(quality.rows(capture["frames"])[0])
        capture["frames"] = [{**entry, **pin(archive)}]
        candidate["losslessEvidence"] = capture
    else:
        data = quality.read_json(observed_request)
        data["version"] = 2
        document(observed_request, data)
    document(candidate_path, candidate)
    repin(observed_request, "candidate", candidate_path)
    with pytest.raises((ReplayError, ValueError)):
        quality.observe_capture(
            observed_request,
            sha256=quality.digest(observed_request),
            directory=root / "observations",
        )


def test_observe_cli(observed_request: Path) -> None:
    output = observed_request.parent / "result.json"
    assert (
        quality.main(
            [
                str(observed_request),
                str(output),
                "--sha256",
                quality.digest(observed_request),
                "--observe",
            ]
        )
        == 1
    )
    assert quality.read_json(output)["accepted"] is False


def document(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": quality.digest(path)}


@pytest.fixture
def request_file(tmp_path: Path) -> Path:
    media = tmp_path / "candidate.mp4"
    media.write_bytes(b"synthetic receipt-only media; not decode evidence")
    producer = document(
        tmp_path / "producer.json",
        {
            "source": str(tmp_path / "original.mp4"),
            "sourceSHA256": "a" * 64,
            "outputDirectory": str(tmp_path),
            "width": 16,
            "height": 16,
            "clock": {"pts": [0, 1, 2], "timeBase": "1/30", "lastDuration": 1},
            "masks": [
                {
                    "startSeconds": 0,
                    "endSeconds": 0.1,
                    "region": {"x": 0, "y": 0, "width": 16, "height": 16},
                }
            ],
        },
    )
    candidate = document(
        tmp_path / "candidate.json",
        {
            "inputs": {
                str(producer): quality.digest(producer),
                str(tmp_path / "original.mp4"): "a" * 64,
            },
            "videoSHA256": quality.digest(media),
            "activeSourceFrames": [0, 1, 2],
            "detectedSupportPixels": [4, 0, 4],
            "emptyDetections": 1,
            "render": {"frames": 3, "quality_measured": False, "quality_passed": False},
        },
    )
    index = document(
        tmp_path / "index.json",
        {
            "ranges": [
                {
                    "media": str(media),
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": 3,
                    "frames": 3,
                    "timeBase": "1/30",
                }
            ]
        },
    )
    manifest = document(
        tmp_path / "manifest.json",
        {
            "version": 1,
            "index": pin(index),
            "chunks": [{"candidate": pin(candidate), "producer": pin(producer)}],
        },
    )
    cuts = document(
        tmp_path / "cuts.json",
        {"cuts": [{"source_start_frame": 0, "source_end_frame": 3, "output_start_frame": 0}]},
    )
    render = document(
        tmp_path / "render.json", {"rate": 30, "cuts": str(cuts), "sourceChunks": str(index)}
    )
    return document(
        tmp_path / "request.json",
        {
            "version": 1,
            "renderConfig": pin(render),
            "assemblyManifest": pin(manifest),
            "protectedCuts": pin(cuts),
            "renderIndex": pin(index),
        },
    )


def repin(request: Path, name: str, path: Path) -> None:
    data = quality.read_json(request)
    data[name] = pin(path)
    document(request, data)


def test_accounting_keeps_quality_false(request_file: Path) -> None:
    result = quality.account(request_file, sha256=quality.digest(request_file))
    assert result["accepted"] is False
    measured = record(result["measurements"])
    assert measured["selectedMaskFrames"] == 3
    assert measured["selectedEmpty"] == 1
    assert measured["selectedNativeFrames"] == 3
    assert measured["measuredFrames"] == 0
    assert measured["missingMaskFrames"] == 3
    assert measured["readableFrames"] is None
    assert measured["maximumUnstableFraction"] is None
    assert measured["finalChainVerified"] is False
    assert record(result["checks"])["losslessEvidenceVerified"] is False
    assert record(result["checks"])["qualityComplete"] is False
    assert "lossless pre-encode" in str(result["limitations"])


def test_dropped_empty_is_not_selected(request_file: Path) -> None:
    cuts = request_file.parent / "cuts.json"
    document(
        cuts,
        {
            "cuts": [
                {"source_start_frame": 0, "source_end_frame": 1, "output_start_frame": 0},
                {"source_start_frame": 2, "source_end_frame": 3, "output_start_frame": 1},
            ]
        },
    )
    repin(request_file, "protectedCuts", cuts)
    result = quality.account(request_file, sha256=quality.digest(request_file))
    assert record(result["measurements"])["selectedEmpty"] == 0
    assert record(result["measurements"])["selectedMaskFrames"] == 2
    assert result["accepted"] is False


@pytest.mark.parametrize(
    "change", ["missing", "changed", "partial", "clock", "unsafe", "version", "cuts"]
)
def test_invalid_accounting_fails(request_file: Path, change: str) -> None:
    root = request_file.parent
    if change == "missing":
        (root / "candidate.mp4").unlink()
    elif change == "changed":
        (root / "candidate.mp4").write_bytes(b"changed")
    elif change == "partial":
        receipt = quality.read_json(root / "candidate.json")
        receipt["activeSourceFrames"] = [0]
        document(root / "candidate.json", receipt)
        manifest = quality.read_json(root / "manifest.json")
        manifest["chunks"] = [
            {"candidate": pin(root / "candidate.json"), "producer": pin(root / "producer.json")}
        ]
        document(root / "manifest.json", manifest)
        repin(request_file, "assemblyManifest", root / "manifest.json")
    elif change == "clock":
        render = quality.read_json(root / "render.json")
        render["rate"] = 60
        document(root / "render.json", render)
        repin(request_file, "renderConfig", root / "render.json")
    elif change == "unsafe":
        data = quality.read_json(request_file)
        data["renderConfig"] = {"path": "/outside/forbidden", "sha256": "a" * 64}
        document(request_file, data)
    elif change == "version":
        data = quality.read_json(request_file)
        data["version"] = 2
        document(request_file, data)
    else:
        document(
            root / "cuts.json",
            {"cuts": [{"source_start_frame": 0, "source_end_frame": 3, "output_start_frame": 1}]},
        )
        repin(request_file, "protectedCuts", root / "cuts.json")
    with pytest.raises((ReplayError, OSError, ValueError)):
        quality.account(request_file, sha256=quality.digest(request_file))


@pytest.mark.parametrize("partial", [False, True])
def test_execution_rows_are_not_provenance(request_file: Path, partial: bool) -> None:
    result = quality.account(request_file, sha256=quality.digest(request_file))
    frames = record(result["measurements"])["projectedFrames"]
    execution = document(
        request_file.parent / "execution.json",
        {
            "renderConfigSHA256": quality.digest(request_file.parent / "render.json"),
            "assemblyManifestSHA256": quality.digest(request_file.parent / "manifest.json"),
            "frames": [] if partial else frames,
        },
    )
    repin(request_file, "execution", execution)
    if partial:
        with pytest.raises(ReplayError, match="execution observation"):
            quality.account(request_file, sha256=quality.digest(request_file))
    else:
        result = quality.account(request_file, sha256=quality.digest(request_file))
        assert record(result["checks"])["executionRowsMatchProtectedCuts"] is True
        assert record(result["checks"])["actualCompositorProvenanceVerified"] is False
        assert result["accepted"] is False


def test_cli_retains_failure_and_refuses_overwrite(request_file: Path) -> None:
    output = request_file.parent / "report.json"
    args = [str(request_file), str(output), "--sha256", quality.digest(request_file)]
    assert quality.main(args) == 1
    before = output.read_bytes()
    assert quality.main(args) == 1
    assert output.read_bytes() == before
    failure = request_file.parent / "failure.json"
    assert quality.main([str(request_file), str(failure), "--sha256", "wrong"]) == 1
    assert quality.read_json(failure)["checks"] == {"accounting": False}
    assert quality.main([str(request_file), "relative.json", "--sha256", "wrong"]) == 1


@pytest.fixture
def frame() -> quality.LosslessFrame:
    return quality.LosslessFrame(
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.ones((16, 16), dtype=np.bool_),
        np.ones((16, 16), dtype=np.bool_),
        np.zeros((16, 16), dtype=np.bool_),
        False,
        Fraction(1, 30),
        Fraction(1, 30),
    )


def test_batched_independent_measurement(frame: quality.LosslessFrame) -> None:
    calls: list[int] = []

    def detector(images: Sequence[Image]) -> tuple[Image, ...]:
        calls.append(len(images))
        return tuple(np.zeros(image.shape[:2], dtype=np.uint8) for image in images)

    evidence = quality.measure_batch([frame, frame], detect=detector)
    assert calls == [2]
    assert quality.summarize_quality(evidence, expected=2) == {
        "measuredFrames": 2,
        "expectedFrames": 2,
        "technicallyInvalidFrames": 0,
        "readableFrames": 0,
        "maximumResidualFraction": 0.0,
        "maximumUnstableFraction": 0.0,
        "complete": True,
        "accepted": True,
    }
    assert quality.summarize_quality(evidence, expected=3)["accepted"] is False
    assert quality.summarize_quality([], expected=0)["accepted"] is False


@pytest.mark.parametrize(
    "kind", ["empty", "large", "pts", "canvas", "support", "detector", "dtype"]
)
def test_invalid_quality_frames(frame: quality.LosslessFrame, kind: str) -> None:
    frames = [frame]
    if kind == "empty":
        frames = []
    elif kind == "large":
        frames = [frame] * 9
    elif kind == "pts":
        frames = [replace(frame, candidate_pts=Fraction(2, 30))]
    elif kind == "canvas":
        frames = [replace(frame, candidate=np.zeros((1, 1, 3), dtype=np.uint8))]
    elif kind == "support":
        frames = [
            replace(
                frame,
                source=np.zeros((8, 8, 3), dtype=np.uint8),
                candidate=np.zeros((8, 8, 3), dtype=np.uint8),
            )
        ]
    with pytest.raises(ValueError):
        quality.measure_batch(
            frames,
            detect=lambda images: (
                () if kind == "detector" else (np.zeros((16, 16), dtype=np.float64),)
            ),
        )


@pytest.mark.parametrize(
    "evidence",
    [
        RepairEvidence(True, 0.006, 0, False),
        RepairEvidence(True, 0, 0.021, False),
        RepairEvidence(True, 0, 0, True),
        RepairEvidence(False, 0, 0, False),
        RepairEvidence(True),
    ],
)
def test_quality_never_uses_cosmetic_fallback(evidence: RepairEvidence) -> None:
    assert quality.summarize_quality([evidence], expected=1)["accepted"] is False


def test_empty_support_is_unknown(frame: quality.LosslessFrame) -> None:
    empty = replace(frame, original_glyph=np.zeros((16, 16), dtype=np.bool_))
    evidence = quality.measure_batch(
        [empty], detect=lambda images: (np.zeros((16, 16), dtype=np.uint8),)
    )
    assert evidence == (RepairEvidence(False, None, 0.0, False),)
    assert quality.summarize_quality(evidence, expected=1)["accepted"] is False


def test_zero_over_zero_never_passes(frame: quality.LosslessFrame) -> None:
    empty = replace(
        frame,
        original_glyph=np.zeros((16, 16), dtype=np.bool_),
        repair_roi=np.zeros((16, 16), dtype=np.bool_),
    )
    evidence = quality.measure_batch(
        [empty], detect=lambda images: (np.zeros((16, 16), dtype=np.uint8),)
    )
    assert evidence == (RepairEvidence(False, None, None, False),)
    assert quality.summarize_quality(evidence, expected=1)["accepted"] is False


def test_real_small_ffmpeg_decode_is_not_lossless_support(request_file: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    root = request_file.parent
    media = root / "fixture.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "color=black:size=16x16:rate=30",
            "-frames:v",
            "3",
            "-c:v",
            "libx264",
            str(media),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    decoded = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(media),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    ).stdout
    assert len(decoded) == 2304
    (root / "candidate.mp4").write_bytes(media.read_bytes())
    receipt = quality.read_json(root / "candidate.json")
    receipt["videoSHA256"] = quality.digest(root / "candidate.mp4")
    document(root / "candidate.json", receipt)
    manifest = quality.read_json(root / "manifest.json")
    manifest["chunks"] = [
        {"candidate": pin(root / "candidate.json"), "producer": pin(root / "producer.json")}
    ]
    document(root / "manifest.json", manifest)
    repin(request_file, "assemblyManifest", root / "manifest.json")
    assert quality.account(request_file, sha256=quality.digest(request_file))["accepted"] is False


@pytest.mark.parametrize("value", [None, "", "relative", "/outside", "bad\npath"])
def test_unsafe_paths(value: object) -> None:
    with pytest.raises(ReplayError):
        quality.local_path(value)


def test_symlink_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = document(tmp_path / "target.json", {})
    linked = tmp_path / "link.json"
    linked.symlink_to(target)
    with pytest.raises(ReplayError):
        quality.local_path(str(linked))
    with pytest.raises(ReplayError):
        quality.integer(True)
    with pytest.raises(ReplayError):
        quality.rows(None)
    with pytest.raises(ReplayError):
        quality.reference({"path": str(target)}, {})
    monkeypatch.setattr(quality, "MAX_JSON_BYTES", 1)
    with pytest.raises(ReplayError):
        quality.read_json(target)


@pytest.mark.parametrize("changed", [False, True])
def test_assembled_index_receipt(request_file: Path, changed: bool) -> None:
    root = request_file.parent
    joined = root / "joined.mov"
    joined.write_bytes(b"synthetic assembled bytes")
    index = document(
        root / "joined-index.json",
        {
            "ranges": [
                {
                    "media": str(joined),
                    "sha256": quality.digest(joined),
                    "frames": 3,
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": 3,
                    "timeBase": "1/30",
                }
            ]
        },
    )
    receipt = document(
        root / "assembly-receipt.json",
        {
            "inputs": {
                str(root / "manifest.json"): quality.digest(root / "manifest.json"),
                str(root / "candidate.mp4"): quality.digest(root / "candidate.mp4"),
            },
            "output": pin(joined),
            "indexSHA256": quality.digest(index),
            "exactPresentationClockVerified": not changed,
            "compressedPacketsIdentical": True,
        },
    )
    render = quality.read_json(root / "render.json")
    render["sourceChunks"] = str(index)
    document(root / "render.json", render)
    repin(request_file, "renderConfig", root / "render.json")
    repin(request_file, "renderIndex", index)
    repin(request_file, "assemblyReceipt", receipt)
    if changed:
        with pytest.raises(ReplayError, match="Assembly receipt/index/clock"):
            quality.account(request_file, sha256=quality.digest(request_file))
    else:
        assert (
            quality.account(request_file, sha256=quality.digest(request_file))["accepted"] is False
        )


def test_residual_counts_displaced_text_and_outside_changes(frame: quality.LosslessFrame) -> None:
    glyph = np.zeros((16, 16), dtype=np.bool_)
    glyph[0, 0] = True
    independent = np.zeros((16, 16), dtype=np.uint8)
    independent[1, 1] = 255
    evidence = quality.measure_batch(
        [replace(frame, original_glyph=glyph)], detect=lambda images: (independent,)
    )
    assert evidence == (RepairEvidence(True, 1.0, 0.0, False),)
    roi = np.zeros((16, 16), dtype=np.bool_)
    roi[0, 0] = True
    candidate = frame.candidate.copy()
    candidate[15, 15] = 255
    changed = replace(frame, candidate=candidate, original_glyph=glyph, repair_roi=roi)
    evidence = quality.measure_batch(
        [changed], detect=lambda images: (np.zeros((16, 16), dtype=np.uint8),)
    )
    assert evidence == (RepairEvidence(False, 0.0, 0.0, False),)
