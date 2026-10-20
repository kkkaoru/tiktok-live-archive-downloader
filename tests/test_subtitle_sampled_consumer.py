"""The consumer adapter writes hash-bound metadata only for validated sampled chunks."""

import dataclasses
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay import subtitle_sampled_consumer as consumer
from replay.caption_io import FrameObservation
from replay.caption_masks import MaskPolicy, TextDetection, TextRectangle
from replay.core import ReplayError
from replay.recordings import record
from replay.subtitle_cleanup_plan import CaptionSearchProfile
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_profile import SubtitleBands
from replay.subtitle_sampled_admission import admission

WIDTH = 1080
HEIGHT = 1920
BASE = Fraction(1, 90000)
PTS = (0, 3000, 6000, 9000)
PROFILE = CaptionSearchProfile(
    SubtitleBands(WIDTH, HEIGHT, primary=(42,), secondary=()), (0, 1283, 1077, 167)
)
LINE = TextDetection("こんにちは世界です", 0.9, TextRectangle(200, 1330, 600, 48))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clock(pts: tuple[int, ...]) -> dict[str, object]:
    return {"pts": list(pts), "timeBase": "1/90000", "lastDuration": 3000}


def _chunk(root: Path, index: int, pts: tuple[int, ...], pins: dict[str, str]) -> None:
    clock = VideoFrameClock(pts, BASE, 3000)
    sampled = {
        i: FrameObservation(float(p * BASE), float(p * BASE), WIDTH, HEIGHT, (LINE,))
        for i, p in enumerate(pts)
    }
    plan = admission(
        sampled=sampled,
        clock=clock,
        profile=PROFILE,
        policy=MaskPolicy(),
        stride=1,
        triggers=(),
        detected=[None] * len(pts),
        radius=1,
    )
    name = f"chunk-{index:05d}"
    state = root / "state" / name / "attempt-sampled"
    state.mkdir(parents=True)
    (state / "admission.json").write_text(json.dumps(plan), encoding="utf-8")
    receipt = {
        "version": 1,
        "kind": "sampled-ocr-repair-chunk",
        "attempts": [
            {
                "directory": "attempt-sampled",
                "sampling": plan["sampling"],
                "auditedFrames": 1,
                "residualLines": [],
                "passed": True,
            }
        ],
        "accepted": "attempt-sampled",
        "escalated": False,
        "regionsFromSampling": True,
        "pins": pins,
        "qualityApproved": False,
        "deliveryComplete": False,
    }
    (root / "state" / name / "chunk-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    media = root / "media" / name / "attempt-sampled"
    media.mkdir(parents=True)
    (media / "candidate.mp4").write_bytes(f"video-{index}".encode())
    config = {
        "source": str(root / "source.mp4"),
        "sourceSHA256": pins["source"],
        "native": str(root / "native"),
        "width": WIDTH,
        "height": HEIGHT,
        "clock": _clock(pts),
        "mediaOutput": str(root / "media" / name),
    }
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / f"{name}.json").write_text(json.dumps(config), encoding="utf-8")


def _world(root: Path) -> consumer.Adaptation:
    source, native = root / "source.mp4", root / "native"
    source.write_bytes(b"source")
    native.write_bytes(b"native")
    pins = {"source": _sha(source), "native": _sha(native)}
    clock = root / "clock.json"
    clock.write_text(json.dumps(_clock(PTS)), encoding="utf-8")
    base = root / "settings.json"
    base.write_text(
        json.dumps(
            {
                "version": 1,
                "chunkFrames": 2,
                "source": {"path": str(source), "sha256": pins["source"]},
                "native": {"path": "/old/native", "sha256": "0" * 64},
                "sourceClock": {"path": str(clock), "sha256": _sha(clock)},
            }
        ),
        encoding="utf-8",
    )
    for index, chunk_pts in enumerate((PTS[:2], PTS[2:])):
        _chunk(root, index, chunk_pts, pins)
    return consumer.Adaptation(
        configs=root / "configs",
        states=root / "state",
        settings=base,
        output=root / "out",
        media_root=root / "media",
        chunk_frames=2,
    )


def _plan(world: consumer.Adaptation) -> dict[str, object]:
    return record(json.loads(consumer.adapt(world).read_text(encoding="utf-8")))


def test_adapt_writes_plan_and_per_chunk_layout(tmp_path: Path) -> None:
    world = _world(tmp_path)
    plan = _plan(world)
    assert (plan["chunks"], plan["plannedFrames"], plan["qualityApproved"]) == (2, 4, False)
    candidate = json.loads((world.output / "chunk-00001/repair/candidate.json").read_text("utf-8"))
    media = tmp_path / "media/chunk-00001/attempt-sampled/candidate.mp4"
    assert candidate["mediaOutput"] == str(media)
    assert candidate["videoSHA256"] == _sha(media)
    assert candidate["sourceStart"] == "1/15"
    assert candidate["render"] == {"frames": 2}
    assert candidate["deliveryComplete"] is False


def test_chunk_and_producer_bind_the_new_settings_and_native(tmp_path: Path) -> None:
    world = _world(tmp_path)
    consumer.adapt(world)
    chunk = json.loads((world.output / "chunk-00000/chunk.json").read_text("utf-8"))
    producer = json.loads((world.output / "chunk-00000/candidate-config.json").read_text("utf-8"))
    settings = json.loads((world.output / "settings.json").read_text("utf-8"))
    assert chunk["settings"]["sha256"] == _sha(world.output / "settings.json")
    assert settings["native"]["sha256"] == _sha(tmp_path / "native")
    assert settings["sampledRepair"]["qualityApproved"] is False
    assert producer["sampledEligibilityValidated"] is True
    assert producer["allowedMediaRoot"] == str(tmp_path / "media")


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    world = _world(tmp_path)
    consumer.adapt(world)
    with pytest.raises(FileExistsError):
        consumer.adapt(world)


def test_native_changed_after_the_run_is_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    (tmp_path / "native").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from the receipt pins"):
        consumer.adapt(world)


def test_missing_media_is_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    (tmp_path / "media/chunk-00001/attempt-sampled/candidate.mp4").unlink()
    with pytest.raises(ValueError, match="media is missing"):
        consumer.adapt(world)


def test_media_outside_the_root_is_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    elsewhere = consumer.Adaptation(
        world.configs, world.states, world.settings, world.output, tmp_path / "other", 2
    )
    (tmp_path / "other").mkdir()
    with pytest.raises(ValueError, match="outside the media root"):
        consumer.adapt(elsewhere)


def test_a_moved_clock_with_the_pinned_hash_is_used(tmp_path: Path) -> None:
    world = _world(tmp_path)
    settings = json.loads(world.settings.read_text("utf-8"))
    pinned = Path(settings["sourceClock"]["path"])
    moved = tmp_path / "moved-clock.json"
    moved.write_bytes(pinned.read_bytes())
    pinned.unlink()
    with pytest.raises(ReplayError, match="regular local input"):
        consumer.adapt(world)
    moved_world = dataclasses.replace(world, output=tmp_path / "out-2", source_clock=moved)
    assert _plan(moved_world)["plannedFrames"] == 4
    written = json.loads((tmp_path / "out-2/settings.json").read_text("utf-8"))
    assert written["sourceClock"] == {"path": str(moved), "sha256": _sha(moved)}


def test_a_clock_differing_from_the_pin_is_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    other = tmp_path / "other-clock.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the settings pin"):
        consumer.adapt(dataclasses.replace(world, source_clock=other))


def test_chunk_clock_must_match_the_source_plan(tmp_path: Path) -> None:
    world = _world(tmp_path)
    path = tmp_path / "configs/chunk-00001.json"
    config = json.loads(path.read_text("utf-8"))
    config["clock"]["pts"] = [6000, 9001]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="clock or canvas"):
        consumer.adapt(world)


def test_non_integer_chunk_clock_is_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    path = tmp_path / "configs/chunk-00000.json"
    config = json.loads(path.read_text("utf-8"))
    config["clock"]["pts"] = [0, "3000"]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="non-integer"):
        consumer.adapt(world)


def test_missing_clock_fields_are_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    path = tmp_path / "configs/chunk-00000.json"
    config = json.loads(path.read_text("utf-8"))
    config["clock"] = {"pts": [0]}
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="exact source clock"):
        consumer.adapt(world)


def test_non_text_receipt_pins_are_rejected(tmp_path: Path) -> None:
    world = _world(tmp_path)
    path = tmp_path / "state/chunk-00000/chunk-receipt.json"
    receipt = json.loads(path.read_text("utf-8"))
    receipt["pins"]["source"] = 5
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="must be text"):
        consumer.adapt(world)


def test_failed_audit_chunk_is_not_adapted(tmp_path: Path) -> None:
    world = _world(tmp_path)
    path = tmp_path / "state/chunk-00001/chunk-receipt.json"
    receipt = json.loads(path.read_text("utf-8"))
    receipt["attempts"][0]["residualLines"] = [{"frame": 1}]
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="residual-text audit"):
        consumer.adapt(world)
