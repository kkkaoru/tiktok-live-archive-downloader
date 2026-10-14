import json
import shutil
from pathlib import Path
from typing import NamedTuple

import pytest
from pytest import MonkeyPatch

from replay import subtitle_cleanup_workflow as flow
from replay.local_asr import fingerprint
from replay.recordings import record


class Usage(NamedTuple):
    total: int
    used: int
    free: int


def document(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": fingerprint(path)}


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    source, native, model = tmp_path / "source.mp4", tmp_path / "native", tmp_path / "model"
    source.write_bytes(b"synthetic original")
    native.write_bytes(b"synthetic executable")
    native.chmod(0o700)
    model.write_bytes(b"synthetic model")
    clock = document(
        tmp_path / "clock.json", {"pts": list(range(10)), "timeBase": "1", "lastDuration": 1}
    )
    review = document(tmp_path / "review.json", {"kind": "synthetic source classification"})
    profile = tmp_path / "profile"
    profile.mkdir()
    inputs = document(
        profile / "inputs.json",
        {
            "sourceSHA256": fingerprint(source),
            "nativeSHA256": fingerprint(native),
            "nativeExecutable": str(native),
            "measurementTool": "video_text_recognize",
            "fullFrame": True,
        },
    )
    frames = [
        {
            "requestedTimeSeconds": index,
            "actualTimeSeconds": index,
            "width": 100,
            "height": 100,
            "lines": [
                {
                    "text": "字幕" + word,
                    "confidence": 0.9,
                    "region": {"x": 0, "y": 40, "width": 100, "height": 20},
                }
            ],
        }
        for index, word in enumerate("春夏秋冬雨雲花風月空")
    ]
    first = document(profile / "0000.result.json", frames[:8])
    second = document(profile / "0001.result.json", frames[8:])
    complete = document(
        profile / "complete.json",
        {
            "observations": 10,
            "results": {"0000.result.json": first["sha256"], "0001.result.json": second["sha256"]},
        },
    )
    path = tmp_path / "settings.json"
    document(
        path,
        {
            "version": 1,
            "source": {"path": str(source), "sha256": fingerprint(source)},
            "sourceClock": clock,
            "classificationEvidence": review,
            "native": {"path": str(native), "sha256": fingerprint(native)},
            "decoder": {"path": str(native), "sha256": fingerprint(native)},
            "model": {"path": str(model), "sha256": fingerprint(model)},
            "profileInputs": inputs,
            "profileComplete": complete,
            "chunkFrames": 2,
        },
    )
    return path


@pytest.fixture
def chunk(settings: Path) -> Path:
    output = settings.parent / "planned"
    flow.prepare(settings, output, workers=2, limit=3)
    path = output / "chunk-00000" / "chunk.json"
    ocr = path.parent / "ocr"
    ocr.mkdir()
    producer = json.loads((settings.parent / "profile" / "inputs.json").read_text())
    document(ocr / "inputs.json", producer)
    observations = json.loads((settings.parent / "profile" / "0000.result.json").read_text())[:2]
    batch = document(ocr / "0000.result.json", observations)
    document(
        ocr / "complete.json", {"observations": 2, "results": {"0000.result.json": batch["sha256"]}}
    )
    return path


def test_prepare_resumes_identical_plan_and_limits_inflight(settings: Path) -> None:
    output = settings.parent / "planned"
    flow.prepare(settings, output, workers=2, limit=3)
    before = (output / "worker-00.json").read_bytes()
    flow.prepare(settings, output, workers=2, limit=3)
    assert (output / "worker-00.json").read_bytes() == before
    plan = record(json.loads((output / "plan.json").read_text()))
    assert plan["plannedFrames"] == 6
    assert plan["totalSourceFrames"] == 10
    assert plan["maximumActiveChunks"] == 2
    assert plan["decodedFramesOnDisk"] is False
    assert plan["wholeSourcePlanned"] is False
    workflow = json.loads(before)
    assert workflow["stages"][3]["inputs"][-1].endswith("/chunk-00000/repair/candidate.json")


def test_verified_collector_connects_to_candidate(chunk: Path) -> None:
    flow.check_ocr(chunk)
    flow.build_candidate(chunk)
    flow.build_candidate(chunk)
    candidate = json.loads((chunk.parent / "candidate-config.json").read_text())
    assert candidate["clock"] == {"pts": [0, 1], "timeBase": "1", "lastDuration": 1}
    assert candidate["evidenceFrames"] == [0, 1]
    assert candidate["learnedTextDetector"]["backend"] == "ppocrv3-db"
    admission = json.loads((chunk.parent / "admission.json").read_text())
    assert admission["unknownSourcePTS"] == []
    assert admission["removalQualityApproved"] is False


def test_explicit_media_build_and_repair_preserve_local_evidence(chunk: Path) -> None:
    media_root = chunk.parents[2] / "approved-media"
    media_root.mkdir()
    media = media_root / "candidate.mp4"
    data = json.loads(chunk.read_text(encoding="utf-8"))
    data.update(mediaOutput=str(media), allowedMediaRoot=str(media_root))
    document(chunk, data)
    flow.build_candidate(chunk)
    producer = chunk.parent / "candidate-config.json"
    candidate = json.loads(producer.read_text(encoding="utf-8"))
    assert candidate["mediaOutput"] == str(media)
    assert candidate["outputDirectory"] == str(chunk.parent / "repair")
    (chunk.parent / "repair").mkdir()
    media.write_bytes(b"synthetic repaired")
    receipt = {
        "render": {"frames": 2},
        "mediaOutput": str(media),
        "videoSHA256": fingerprint(media),
        "inputs": {str(producer): fingerprint(producer)},
    }
    document(chunk.parent / "repair/candidate.json", receipt)
    flow.check_repair(chunk)
    receipt["mediaOutput"] = str(chunk.parent / "repair/candidate.mp4")
    document(chunk.parent / "repair/candidate.json", receipt)
    with pytest.raises(ValueError, match="destination"):
        flow.check_repair(chunk)


def test_capacity_guard_rejects_before_repair(chunk: Path, monkeypatch: MonkeyPatch) -> None:
    usage = Usage(100 * 1024**3, 90 * 1024**3, 10 * 1024**3)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage)
    with pytest.raises(ValueError, match="Insufficient free"):
        flow.build_candidate(chunk)
    assert not (chunk.parent / "candidate-config.json").exists()


def test_invalid_active_budget(chunk: Path) -> None:
    data = json.loads(chunk.read_text())
    data["maximumActiveChunks"] = 9
    document(chunk, data)
    with pytest.raises(ValueError, match="capacity budget"):
        flow.build_candidate(chunk)


def test_no_ocr_text_remains_unknown(chunk: Path) -> None:
    ocr = chunk.parent / "ocr"
    frames = json.loads((ocr / "0000.result.json").read_text())
    frames[0]["lines"] = []
    frames[1]["lines"] = []
    batch = document(ocr / "0000.result.json", frames)
    document(
        ocr / "complete.json", {"observations": 2, "results": {"0000.result.json": batch["sha256"]}}
    )
    flow.build_candidate(chunk)
    admission = json.loads((chunk.parent / "admission.json").read_text())
    candidate = json.loads((chunk.parent / "candidate-config.json").read_text())
    assert admission["unknownSourcePTS"] == [0, 1]
    assert candidate["masks"] == []
    assert "evidenceFrames" not in candidate


@pytest.mark.parametrize("actual", [0, 0.5])
def test_wrong_or_duplicate_observed_clock(chunk: Path, actual: float) -> None:
    ocr = chunk.parent / "ocr"
    frames = json.loads((ocr / "0000.result.json").read_text())
    frames[1]["actualTimeSeconds"] = actual
    batch = document(ocr / "0000.result.json", frames)
    document(
        ocr / "complete.json", {"observations": 2, "results": {"0000.result.json": batch["sha256"]}}
    )
    with pytest.raises(ValueError, match="PTS"):
        flow.check_ocr(chunk)


def test_changed_collector_media_rejected(chunk: Path) -> None:
    inputs = chunk.parent / "ocr" / "inputs.json"
    producer = json.loads(inputs.read_text())
    producer["sourceSHA256"] = "other"
    document(inputs, producer)
    with pytest.raises(ValueError, match="provenance"):
        flow.check_ocr(chunk)


@pytest.mark.parametrize("results", [{}, {"0001.result.json": "wrong"}])
def test_missing_batch(chunk: Path, results: dict[str, str]) -> None:
    document(chunk.parent / "ocr" / "complete.json", {"observations": 2, "results": results})
    with pytest.raises(ValueError):
        flow.check_ocr(chunk)


def test_wrong_observation_total(chunk: Path) -> None:
    path = chunk.parent / "ocr" / "complete.json"
    complete = json.loads(path.read_text())
    complete["observations"] = 3
    document(path, complete)
    with pytest.raises(ValueError, match="total"):
        flow.check_ocr(chunk)


def test_truncated_internal_batch(settings: Path) -> None:
    directory = settings.parent / "profile"
    path = directory / "0000.result.json"
    frames = json.loads(path.read_text())[:7]
    ref = document(path, frames)
    complete = json.loads((directory / "complete.json").read_text())
    complete["results"]["0000.result.json"] = ref["sha256"]
    document(directory / "complete.json", complete)
    with pytest.raises(ValueError, match="internal"):
        flow.collector_frames(
            directory / "inputs.json", directory / "complete.json", json.loads(settings.read_text())
        )


@pytest.mark.parametrize(("workers", "limit"), [(0, 1), (9, 1), (2, 0)])
def test_invalid_worker_budget(settings: Path, workers: int, limit: int) -> None:
    with pytest.raises(ValueError, match="worker"):
        flow.prepare(settings, settings.parent / "out", workers=workers, limit=limit)


def test_changed_source_is_not_planned(settings: Path) -> None:
    (settings.parent / "source.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source media"):
        flow.prepare(settings, settings.parent / "out", workers=1)


@pytest.mark.parametrize(("key", "value"), [("version", 2), ("chunkFrames", "two")])
def test_invalid_settings(settings: Path, key: str, value: object) -> None:
    data = json.loads(settings.read_text())
    data[key] = value
    document(settings, data)
    with pytest.raises(ValueError):
        flow.prepare(settings, settings.parent / "out", workers=1)


def test_changed_plan_is_preserved(settings: Path) -> None:
    output = settings.parent / "out"
    flow.prepare(settings, output, workers=1, limit=1)
    (output / "chunk-00000" / "times.json").write_text("changed")
    with pytest.raises(ValueError, match="Existing"):
        flow.prepare(settings, output, workers=1, limit=1)


def test_linked_metadata_refused(settings: Path) -> None:
    link = settings.parent / "linked.json"
    link.symlink_to(settings)
    with pytest.raises(ValueError, match="canonical"):
        flow.prepare(link, settings.parent / "out", workers=1)


def test_missing_metadata_refused(settings: Path) -> None:
    with pytest.raises(ValueError, match="bounded"):
        flow.prepare(settings.parent / "absent.json", settings.parent / "out", workers=1)


def test_repair_receipt_identity(chunk: Path) -> None:
    flow.build_candidate(chunk)
    directory = chunk.parent / "repair"
    directory.mkdir()
    video = directory / "candidate.mp4"
    video.write_bytes(b"synthetic repaired")
    document(
        directory / "candidate.json",
        {
            "render": {"frames": 2},
            "videoSHA256": fingerprint(video),
            "inputs": {str(chunk): fingerprint(chunk)},
        },
    )
    flow.check_repair(chunk)
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        flow.check_repair(chunk)


@pytest.mark.parametrize("frames", [1, 3])
def test_repair_wrong_frame_count(chunk: Path, frames: int) -> None:
    directory = chunk.parent / "repair"
    directory.mkdir()
    document(directory / "candidate.json", {"render": {"frames": frames}})
    with pytest.raises(ValueError, match="count"):
        flow.check_repair(chunk)


def test_cli_prepare(settings: Path) -> None:
    assert (
        flow.main(
            [
                "prepare",
                str(settings),
                "--output",
                str(settings.parent / "out"),
                "--limit",
                "1",
                "--workers",
                "2",
            ]
        )
        == 0
    )


def test_cli_check_and_build(chunk: Path) -> None:
    assert flow.main(["check-ocr", str(chunk)]) == 0
    assert flow.main(["build", str(chunk)]) == 0


def test_cli_requires_output(settings: Path) -> None:
    with pytest.raises(SystemExit):
        flow.main(["prepare", str(settings)])
