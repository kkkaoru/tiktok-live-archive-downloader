"""Synthetic metadata planning; decoded media approval remains a separate stage."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import delivery_source_assembly
from replay import subtitle_cleanup_assembly as planner
from replay.recordings import record
from replay.subtitle_cleanup_plan import clock_chunks


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass(frozen=True)
class Fixture:
    root: Path
    plan: Path
    settings: Path
    source_clock: Path
    output: Path

    def directory(self, index: int) -> Path:
        return self.root / f"chunk-{index:05d}"

    def producer(self, index: int) -> Path:
        return self.directory(index) / "candidate-config.json"

    def receipt(self, index: int) -> Path:
        return self.directory(index) / "repair/candidate.json"

    def update_producer(self, index: int, key: str, value: object) -> None:
        path = self.producer(index)
        data = read(path)
        data[key] = value
        write(path, data)
        receipt = read(self.receipt(index))
        record(receipt["inputs"])[str(path)] = pin(path)["sha256"]
        write(self.receipt(index), receipt)


def fixture(
    root: Path, *, pts: tuple[int, ...] = (0, 33, 67, 102, 140), budget: int = 3
) -> Fixture:
    root.mkdir()
    source, native = root / "original.mp4", root / "native"
    source.write_bytes(b"original source")
    native.write_bytes(b"native executable identity")
    source_clock = write(
        root / "physical-clock.json", {"pts": list(pts), "timeBase": "1/1000", "lastDuration": 31}
    )
    settings = write(
        root / "settings.json",
        {
            "version": 1,
            "source": pin(source),
            "native": pin(native),
            "sourceClock": pin(source_clock),
            "chunkFrames": budget,
        },
    )
    clocks = clock_chunks(read(source_clock), maximum_frames=budget)
    plan = write(
        root / "plan.json",
        {
            "version": 1,
            "settings": pin(settings),
            "wholeSourcePlanned": True,
            "chunks": len(clocks),
            "totalSourceFrames": len(pts),
            "plannedFrames": len(pts),
        },
    )
    for index, clock in enumerate(clocks):
        directory = root / f"chunk-{index:05d}"
        clock_data = {
            "pts": list(clock.pts),
            "timeBase": str(clock.time_base),
            "lastDuration": clock.last_duration,
        }
        write(
            directory / "chunk.json",
            {
                "version": 1,
                "settings": pin(settings),
                "clock": clock_data,
                "ocrDirectory": str(directory / "ocr"),
            },
        )
        media = directory / "repair/candidate.mp4"
        media.parent.mkdir()
        media.write_bytes(b"synthetic, not approved or decoded media")
        producer = write(
            directory / "candidate-config.json",
            {
                "source": str(source),
                "sourceSHA256": pin(source)["sha256"],
                "nativeExecutable": str(native),
                "nativeSHA256": pin(native)["sha256"],
                "width": 32,
                "height": 64,
                "clock": clock_data,
                "outputDirectory": str(media.parent),
            },
        )
        write(
            media.with_suffix(".json"),
            {
                "inputs": {
                    str(producer): pin(producer)["sha256"],
                    str(source): pin(source)["sha256"],
                    str(native): pin(native)["sha256"],
                },
                "render": {"frames": len(clock.pts)},
                "sourceStart": str(clock.pts[0] * clock.time_base),
                "videoSHA256": pin(media)["sha256"],
                "qualityApproval": False,
                "deliveryComplete": False,
            },
        )
    output = root / "prepared"
    output.mkdir()
    return Fixture(root, plan, settings, source_clock, output)


@pytest.fixture
def ready(tmp_path: Path) -> Fixture:
    return fixture(tmp_path / "cleanup")


def test_explicit_media_is_bound_by_planner_and_delivery_consumer(ready: Fixture) -> None:
    media_root = ready.root.parent / "approved-media"
    media_root.mkdir()
    media = media_root / "candidate.mp4"
    media.write_bytes((ready.directory(0) / "repair/candidate.mp4").read_bytes())
    ready.update_producer(0, "mediaOutput", str(media))
    ready.update_producer(0, "allowedMediaRoot", str(media_root))
    receipt = read(ready.receipt(0))
    receipt["mediaOutput"] = str(media)
    write(ready.receipt(0), receipt)
    manifest = planner.prepare(ready.plan, ready.output)
    chunks = delivery_source_assembly.load_chunks(
        manifest, sha256=pin(manifest)["sha256"], bindings={}
    )
    assert chunks[0].media == media_root / "candidate.mp4"
    assert chunks[0].clock.pts == (0, 33, 67)
    receipt["mediaOutput"] = str(ready.directory(0) / "repair/candidate.mp4")
    write(ready.receipt(0), receipt)
    with pytest.raises(ValueError, match="destination"):
        planner.prepare(ready.plan, ready.output)


@pytest.mark.parametrize(
    "pts,budget", [((0, 33, 67, 102, 140), 3), ((0, 5, 13, 46, 50, 71, 99), 2), ((0,), 1)]
)
def test_exact_vfr_and_internal_edges_without_media_approval(
    tmp_path: Path, pts: tuple[int, ...], budget: int
) -> None:
    data = fixture(tmp_path / "cleanup", pts=pts, budget=budget)
    manifest = planner.prepare(data.plan, data.output)
    prepared = read(data.output / "preparation.json")
    assert prepared["physicalSourceFrames"] == len(pts)
    assert prepared["physicalSourceTerminalPTS"] == pts[-1] + 31
    assert prepared["canvas"] == [32, 64]
    assert prepared["candidateOnly"] is True
    assert prepared["sourceCaptionQualityApproved"] is False
    assert prepared["mediaBytesVerified"] is False
    assert prepared["deliveryComplete"] is False
    chunks = delivery_source_assembly.load_chunks(
        manifest, sha256=pin(manifest)["sha256"], bindings={}
    )
    assert tuple(pts for chunk in chunks for pts in chunk.clock.pts) == pts
    assert all(
        left.clock.pts[-1] + left.clock.last_duration == right.clock.pts[0]
        for left, right in zip(chunks, chunks[1:], strict=False)
    )
    assert manifest.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "key,value",
    [
        ("wholeSourcePlanned", False),
        ("version", True),
        ("chunks", 1),
        ("chunks", True),
        ("totalSourceFrames", 4),
        ("plannedFrames", 4),
    ],
)
def test_incomplete_or_wrong_plan_rejected(ready: Fixture, key: str, value: object) -> None:
    raw = read(ready.plan)
    raw[key] = value
    write(ready.plan, raw)
    with pytest.raises(ValueError):
        planner.prepare(ready.plan, ready.output)
    assert list(ready.output.iterdir()) == []


@pytest.mark.parametrize("target", ["receipt", "producer", "media", "empty-media"])
def test_uncompleted_chunk_is_never_guessed(ready: Fixture, target: str) -> None:
    path = {
        "receipt": ready.receipt(1),
        "producer": ready.producer(1),
        "media": ready.directory(1) / "repair/candidate.mp4",
        "empty-media": ready.directory(1) / "repair/candidate.mp4",
    }[target]
    if target == "empty-media":
        path.write_bytes(b"")
    else:
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        planner.prepare(ready.plan, ready.output)
    assert list(ready.output.iterdir()) == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("source", "/wrong/original.mp4"),
        ("sourceSHA256", "b" * 64),
        ("nativeExecutable", "/wrong/native"),
        ("nativeSHA256", "c" * 64),
        ("outputDirectory", "/wrong/repair"),
        ("width", True),
        ("height", 0),
        ("width", 16385),
        ("height", 65),
        ("clock", {"pts": [102, 140], "lastDuration": 30, "timeBase": "1/1000"}),
    ],
)
def test_source_native_clock_geometry_boundaries(ready: Fixture, key: str, value: object) -> None:
    ready.update_producer(1, key, value)
    with pytest.raises(ValueError):
        planner.prepare(ready.plan, ready.output)
    assert list(ready.output.iterdir()) == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("render", {"frames": 1}),
        ("render", {"frames": True}),
        ("sourceStart", "0"),
        ("videoSHA256", "invalid"),
        ("inputs", {}),
    ],
)
def test_receipt_binds_exact_native_source_and_recognized_clock(
    ready: Fixture, key: str, value: object
) -> None:
    raw = read(ready.receipt(1))
    raw[key] = value
    write(ready.receipt(1), raw)
    with pytest.raises(ValueError):
        planner.prepare(ready.plan, ready.output)


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", True),
        ("settings", {}),
        ("ocrDirectory", "/wrong/ocr"),
        ("clock", {"pts": [102, 140], "timeBase": "1/1000", "lastDuration": 30}),
    ],
)
def test_chunk_declaration_must_match_full_clock(ready: Fixture, key: str, value: object) -> None:
    path = ready.directory(1) / "chunk.json"
    raw = read(path)
    raw[key] = value
    write(path, raw)
    with pytest.raises(ValueError):
        planner.prepare(ready.plan, ready.output)


def test_immutable_settings_pin(ready: Fixture) -> None:
    write(ready.settings, {"version": 2})
    with pytest.raises(ValueError, match="pin changed"):
        planner.prepare(ready.plan, ready.output)


@pytest.mark.parametrize("pts", [[], [0, 33, 67, 102]])
def test_empty_or_truncated_physical_clock_rejected(ready: Fixture, pts: list[int]) -> None:
    clock = read(ready.source_clock)
    clock["pts"] = pts
    write(ready.source_clock, clock)
    settings = read(ready.settings)
    settings["sourceClock"] = pin(ready.source_clock)
    write(ready.settings, settings)
    plan = read(ready.plan)
    plan["settings"] = pin(ready.settings)
    write(ready.plan, plan)
    with pytest.raises(ValueError):
        planner.prepare(ready.plan, ready.output)
    assert list(ready.output.iterdir()) == []


def test_nonzero_source_and_excessive_chunk_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nonzero = fixture(tmp_path / "nonzero", pts=(5, 7))
    with pytest.raises(ValueError, match="zero-origin"):
        planner.prepare(nonzero.plan, nonzero.output)
    large = fixture(tmp_path / "large")
    monkeypatch.setattr(planner, "MAX_CHUNKS", 1)
    with pytest.raises(ValueError, match="complete"):
        planner.prepare(large.plan, large.output)


@pytest.mark.parametrize("which", ["source", "native", "sourceClock"])
def test_invalid_settings_references(ready: Fixture, which: str) -> None:
    raw = read(ready.settings)
    raw[which] = {"path": "relative", "sha256": "a" * 64, "extra": True}
    write(ready.settings, raw)
    plan = read(ready.plan)
    plan["settings"] = pin(ready.settings)
    write(ready.plan, plan)
    with pytest.raises(ValueError):
        planner.prepare(ready.plan, ready.output)


@pytest.mark.parametrize(
    "text", ["[]", "", '{"version":1,"version":1}', '{"version": 1, "settings": []}']
)
def test_invalid_bounded_json(ready: Fixture, text: str) -> None:
    ready.plan.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        planner.prepare(ready.plan, ready.output)


def test_json_limit_and_no_linked_inputs(ready: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(planner, "MAX_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="bounded metadata"):
        planner.prepare(ready.plan, ready.output)
    monkeypatch.undo()
    linked = ready.root / "linked-plan.json"
    linked.symlink_to(ready.plan)
    with pytest.raises(ValueError, match="canonical"):
        planner.prepare(linked, ready.output)


def test_output_collision_keeps_bytes_and_no_partial_manifest(ready: Fixture) -> None:
    old = ready.output / "manifest.json"
    old.write_bytes(b"retained old proof")
    with pytest.raises(ValueError, match="absent"):
        planner.prepare(ready.plan, ready.output)
    assert old.read_bytes() == b"retained old proof"
    assert not (ready.output / "index.json").exists()


def test_output_and_source_references_cannot_escape(ready: Fixture) -> None:
    with pytest.raises(ValueError, match="checkout-local"):
        planner.prepare(ready.plan, Path("/outside/assembly-output"))
    with pytest.raises(ValueError, match="canonical absolute"):
        planner._name("/example/../escape")
    with pytest.raises(ValueError, match="canonical absolute"):
        planner._name("/example/new\nline")


def test_work_scope_and_linked_output(ready: Fixture) -> None:
    with pytest.raises(ValueError, match="local work"):
        planner._bytes(Path(planner.__file__).resolve())
    with pytest.raises(ValueError, match="local work"):
        planner.prepare(ready.plan, Path.cwd().resolve() / "src")
    alias = ready.root / "alias"
    alias.symlink_to(ready.output, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical"):
        planner.prepare(ready.plan, alias)


def test_changed_open_file_is_rejected(ready: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    original = os.fstat
    calls = 0

    def stat_and_change(descriptor: int) -> os.stat_result:
        nonlocal calls
        observed = original(descriptor)
        calls += 1
        if calls == 1:
            ready.plan.write_bytes(ready.plan.read_bytes() + b" ")
        return observed

    monkeypatch.setattr(os, "fstat", stat_and_change)
    with pytest.raises(ValueError, match="during reading"):
        planner._bytes(ready.plan)


def test_metadata_rechecked_at_terminal(ready: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    original = planner._bytes
    count = 0

    def changed(path: Path, *, code: bool = False) -> bytes:
        nonlocal count
        data = original(path, code=code)
        if path == ready.plan:
            count += 1
            return data + b" " if count == 2 else data
        return data

    monkeypatch.setattr(planner, "_bytes", changed)
    with pytest.raises(ValueError, match="before publication"):
        planner.prepare(ready.plan, ready.output)
    assert list(ready.output.iterdir()) == []


def test_conflicting_repeat_read_rejected(ready: Fixture) -> None:
    with pytest.raises(ValueError, match="fingerprint changed"):
        planner._read(ready.plan, {str(ready.plan): "a" * 64})


def test_main_and_required_domain_verification(
    ready: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    assert planner.main([str(ready.plan), str(ready.output)]) == 0
    assert capsys.readouterr().out.strip() == str(ready.output / "manifest.json")
    prepared = read(ready.output / "preparation.json")
    assert "decoded PTS verification" in str(prepared["requiredNextStage"])
    with pytest.raises(ValueError, match="absent"):
        planner.prepare(ready.plan, ready.output)
