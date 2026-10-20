"""The generic cleanup runner derives chunk jobs, bounds execution and assembles by description."""

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import delivery_source_assembly_parallel as assembly_runner
from replay import gpu_lease
from replay import source_clean_run as runner
from replay import subtitle_cleanup_assembly as preparation
from replay import subtitle_sampled_consumer as consumer
from replay.core import ReplayError
from replay.workflow_parallel import Load


@pytest.fixture(autouse=True)
def private_gpu_leases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Chunk jobs lease from a test directory, never the checkout's shared one."""
    monkeypatch.setattr(gpu_lease, "root", lambda: tmp_path / "gpu-leases")


def description(root: Path, **changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "version": 1,
        "recording": "replay-1",
        "source": "/media/source.mp4",
        "sourceSHA256": "a" * 64,
        "clock": str(root / "clock.json"),
        "width": 1080,
        "height": 1920,
        "profileDirectory": "/profile",
        "native": "/native",
        "detectorProgram": "/model.pt",
        "detectorProgramSHA256": "b" * 64,
        "settings": str(root / "settings.json"),
        "work": str(root / "edits/work/clean-replay-1"),
        "mediaRoot": str(root / "media"),
        "ffmpeg": "/bin/ffmpeg",
        "ffprobe": "/bin/ffprobe",
        "chunkFrames": 2,
        "streams": 2,
        "freeFloorGiB": 48,
        "stride": 6,
        "auditStride": 5,
        "workers": 2,
        "repairWorkers": 4,
        "radius": 6,
    }
    return {**base, **changes}


def make(tmp_path: Path, **changes: object) -> runner.Recording:
    (tmp_path / "clock.json").write_text(
        json.dumps(
            {"pts": [0, 3000, 6000, 9000, 12000], "timeBase": "1/90000", "lastDuration": 3000}
        ),
        encoding="utf-8",
    )
    path = tmp_path / "description.json"
    path.write_text(json.dumps(description(tmp_path, **changes)), encoding="utf-8")
    return runner.load(path, root=tmp_path)


def test_load_rejects_unknown_keys_relative_paths_and_foreign_work(tmp_path: Path) -> None:
    for bad in (
        {"extra": 1},
        {"native": "native"},
        {"work": str(tmp_path / "elsewhere")},
        {"streams": 99},
    ):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(description(tmp_path, **bad)), encoding="utf-8")
        with pytest.raises(ValueError):
            runner.load(path, root=tmp_path)


def test_configs_write_one_job_per_chunk_with_colour_gate_on(tmp_path: Path) -> None:
    recording = make(tmp_path)
    assert runner.configs(recording) == 3
    job = json.loads((recording.work / "configs/chunk-00001.json").read_text(encoding="utf-8"))
    assert job["clock"]["pts"] == [6000, 9000]
    assert (job["colourKey"], job["colourGate"]) == (True, True)
    assert job["output"].endswith("state/chunk-00001")
    assert job["mediaOutput"].endswith("media/replay-1/sampled/chunk-00001")
    assert runner.configs(recording) == 3


YOLO = {"kind": "yolo", "weights": "/w.pt", "sha256": "d" * 64, "configDirectory": "/c",
        "work": "/edits", "imgsz": 640, "masks": True}  # fmt: skip


def test_a_yolo_region_detector_is_passed_to_every_chunk_job(tmp_path: Path) -> None:
    recording = make(tmp_path, regionDetector=YOLO)
    runner.configs(recording)
    job = json.loads((recording.work / "configs/chunk-00002.json").read_text(encoding="utf-8"))
    assert job["regionDetector"] == YOLO


def test_the_band_audit_tile_is_passed_to_every_chunk_job(tmp_path: Path) -> None:
    recording = make(tmp_path, auditTile=3)
    runner.configs(recording)
    job = json.loads((recording.work / "configs/chunk-00000.json").read_text(encoding="utf-8"))
    assert job["auditTile"] == 3


def test_an_invalid_audit_tile_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="auditTile"):
        make(tmp_path, auditTile=7)


def test_an_invalid_yolo_region_detector_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="regionDetector"):
        make(tmp_path, regionDetector={"kind": "yolo"})


def test_configs_refuse_a_changed_existing_job(tmp_path: Path) -> None:
    recording = make(tmp_path)
    runner.configs(recording)
    changed = make(tmp_path, stride=8)
    with pytest.raises(ValueError, match="different content"):
        runner.configs(changed)


def test_run_skips_finished_chunks_stops_below_the_floor_and_records_exits(tmp_path: Path) -> None:
    recording = make(tmp_path)
    runner.configs(recording)
    (recording.work / "state/chunk-00000").mkdir(parents=True)
    (recording.work / "state/chunk-00000/chunk-receipt.json").write_text("{}", encoding="utf-8")
    seen: list[str] = []

    def fake(command: Sequence[str], log: Path) -> int:
        seen.append(Path(command[-1]).name)
        return 0 if "00001" in command[-1] else 3

    summary = runner.run(recording, first=0, last=2, runner=fake, free=lambda: 100 * runner.GIB)
    results = summary["results"]
    assert isinstance(results, dict)
    assert results["0"] == {"skipped": True}
    assert results["1"]["exit"] == 0
    assert results["2"]["exit"] == 3
    assert sorted(seen) == ["chunk-00001.json", "chunk-00002.json"]


IDLE = Load(cpus=10, load=1.0, memory_free_percent=95, gpu_percent=10)
BUSY = Load(cpus=10, load=9.5, memory_free_percent=95, gpu_percent=10)


def concurrency(recording: runner.Recording, measured: Load) -> int:
    active: list[int] = []
    peak: list[int] = []
    lock = threading.Lock()

    def fake(_command: Sequence[str], _log: Path) -> int:
        with lock:
            active.append(1)
            peak.append(len(active))
        threading.Event().wait(0.5)
        with lock:
            active.pop()
        return 0

    runner.configs(recording)
    runner.run(
        recording, first=0, last=2, runner=fake, free=lambda: 100 * runner.GIB,
        measure=lambda: measured, poll_seconds=0.01,
    )  # fmt: skip
    return max(peak)


def test_an_idle_machine_runs_up_to_max_streams(tmp_path: Path) -> None:
    assert concurrency(make(tmp_path, streams=1, maxStreams=3), IDLE) == 3


def test_a_busy_machine_keeps_only_the_base_streams(tmp_path: Path) -> None:
    assert concurrency(make(tmp_path, streams=1, maxStreams=3), BUSY) == 1


def test_an_interrupted_chunk_is_retired_and_run_again(tmp_path: Path) -> None:
    recording = make(tmp_path)
    runner.configs(recording)
    stale = recording.work / "state" / "chunk-00000" / "attempt-sampled"
    stale.mkdir(parents=True)
    media = recording.media / "sampled" / "chunk-00000"
    media.mkdir(parents=True)
    runner.run(
        recording, first=0, last=0, runner=lambda _c, _l: 0, free=lambda: 100 * runner.GIB,
        measure=lambda: IDLE, poll_seconds=0.01,
    )  # fmt: skip
    assert not stale.parent.exists() and not media.exists()
    (retired,) = (recording.work / "interrupted").iterdir()
    assert (retired / "attempt-sampled").is_dir()


def test_free_memory_must_cover_one_more_job_above_the_floor(tmp_path: Path) -> None:
    floor_only = Load(cpus=10, load=1.0, memory_free_percent=runner.MEMORY_FREE_MIN, gpu_percent=10)
    assert concurrency(make(tmp_path, streams=3, maxStreams=3), floor_only) == 1


def test_max_streams_defaults_to_streams_and_must_cover_them(tmp_path: Path) -> None:
    assert make(tmp_path).max_streams == 2
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(description(tmp_path, maxStreams=1)), encoding="utf-8")
    with pytest.raises(ValueError, match="maxStreams"):
        runner.load(path, root=tmp_path)


def test_max_streams_is_not_copied_into_chunk_jobs(tmp_path: Path) -> None:
    recording = make(tmp_path, maxStreams=4)
    runner.configs(recording)
    job = json.loads((recording.work / "configs/chunk-00000.json").read_text(encoding="utf-8"))
    assert "maxStreams" not in job


def test_run_does_not_start_chunks_below_the_free_space_floor(tmp_path: Path) -> None:
    recording = make(tmp_path)
    runner.configs(recording)
    summary = runner.run(
        recording, first=0, last=2, runner=lambda *_a: 0, free=lambda: 10 * runner.GIB
    )
    assert summary["results"] == {
        "0": {"notStarted": True},
        "1": {"notStarted": True},
        "2": {"notStarted": True},
    }


def test_a_rerun_keeps_its_own_summary_beside_the_earlier_one(tmp_path: Path) -> None:
    recording = make(tmp_path)
    runner.configs(recording)
    runner.run(recording, first=0, last=2, runner=lambda *_a: 1, free=lambda: 10 * runner.GIB)
    runner.run(recording, first=0, last=2, runner=lambda *_a: 0, free=lambda: 100 * runner.GIB)
    assert sorted(path.name for path in recording.work.glob("queue-summary-*")) == [
        "queue-summary-0-2-attempt-2.json",
        "queue-summary-0-2.json",
    ]


def test_assemble_adapts_prepares_joins_and_pins_the_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = make(tmp_path)
    calls: list[str] = []

    def adapt(adaptation: object) -> Path:
        calls.append("adapt")
        (recording.work / "consumer").mkdir(parents=True, exist_ok=True)
        plan = recording.work / "consumer/plan.json"
        plan.write_text("{}", encoding="utf-8")
        return plan

    def prepare(plan: Path, output: Path) -> Path:
        calls.append("prepare")
        manifest = output / "manifest.json"
        manifest.write_text('{"m": 1}', encoding="utf-8")
        return manifest

    def join(manifest: Path, **keywords: object) -> Path:
        calls.append("join")
        evidence = Path(str(keywords["evidence"]))
        evidence.mkdir()
        (evidence / "receipt.json").write_text('{"r": 1}', encoding="utf-8")
        (evidence / "single-source-index.json").write_text('{"i": 1}', encoding="utf-8")
        return evidence / "receipt.json"

    monkeypatch.setattr(consumer, "adapt", adapt)
    monkeypatch.setattr(preparation, "prepare", prepare)
    monkeypatch.setattr(assembly_runner, "assemble_parallel", join)
    cleanup = runner.assemble(recording)
    pins = json.loads(cleanup.read_text(encoding="utf-8"))
    manifest = recording.work / "consumer/assembly-preparation/manifest.json"
    assert calls == ["adapt", "prepare", "join"]
    assert pins["assemblyManifest"] == {
        "path": str(manifest),
        "sha256": hashlib.sha256(b'{"m": 1}').hexdigest(),
    }
    assert set(pins) == {"assemblyManifest", "assemblyReceipt", "cleanIndex"}


def test_main_runs_the_configs_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording = make(tmp_path)
    monkeypatch.chdir(tmp_path)
    code = runner.main([str(tmp_path / "description.json"), "configs"])
    assert code == 0
    assert len(list((recording.work / "configs").glob("chunk-*.json"))) == 3


def test_main_run_reports_failed_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording = make(tmp_path)
    runner.configs(recording)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "subprocess_runner", lambda *_a: 1)
    monkeypatch.setattr(runner, "free_bytes", lambda: 100 * runner.GIB)
    assert runner.main([str(tmp_path / "description.json"), "run", "--last", "0"]) == 1


def machine(tmp_path: Path) -> Path:
    path = tmp_path / "defaults.json"
    path.write_text(
        json.dumps(
            {
                "native": "/native",
                "detectorProgram": "/model.pt",
                "detectorProgramSHA256": "b" * 64,
                "ffmpeg": "/bin/ffmpeg",
                "ffprobe": "/bin/ffprobe",
                "mediaRoot": str(tmp_path / "media"),
                "chunkFrames": 900,
                "streams": 3,
                "freeFloorGiB": 48,
                "stride": 6,
                "auditStride": 5,
                "workers": 2,
                "repairWorkers": 4,
                "radius": 6,
            }
        ),
        encoding="utf-8",
    )
    return path


def bootstrap(tmp_path: Path, name: str = "replay-9") -> Path:
    source = tmp_path / "media-in" / f"{name}.mp4"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"x")
    directory = tmp_path / "boot" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source.sha256").write_text(f"{'c' * 64}  {source}\n", encoding="utf-8")
    return directory


def test_describe_derives_a_valid_description_from_bootstrap_and_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "canvas", lambda _ffprobe, _source: (1080, 1920))
    destination = runner.describe(bootstrap(tmp_path), machine(tmp_path), root=tmp_path)
    loaded = runner.load(destination, root=tmp_path)
    assert loaded.text("recording") == "replay-9-v1"
    assert loaded.media == tmp_path / "media/replay-9-v1"
    assert loaded.text("sourceSHA256") == "c" * 64
    assert (loaded.number("width"), loaded.number("height")) == (1080, 1920)
    assert loaded.text("clock").endswith("replay-9/cleanup-bootstrap-v1/physical-clock.json")
    assert loaded.work == tmp_path / "edits/work/clean-replay-9-v1"


def test_describe_carries_a_yolo_region_detector_from_the_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "canvas", lambda _ffprobe, _source: (1080, 1920))
    defaults = machine(tmp_path)
    data = json.loads(defaults.read_text(encoding="utf-8"))
    data["regionDetector"] = YOLO
    defaults.write_text(json.dumps(data), encoding="utf-8")
    destination = runner.describe(bootstrap(tmp_path), defaults, root=tmp_path)
    assert runner.load(destination, root=tmp_path).data["regionDetector"] == YOLO


def test_describe_rejects_foreign_defaults_and_bad_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "canvas", lambda _ffprobe, _source: (1080, 1920))
    defaults = machine(tmp_path)
    data = json.loads(defaults.read_text(encoding="utf-8"))
    data["extra"] = 1
    defaults.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported cleanup defaults"):
        runner.describe(bootstrap(tmp_path), defaults, root=tmp_path)
    clean = machine(tmp_path)
    boot = bootstrap(tmp_path, "replay-8")
    (boot / "source.sha256").write_text("short  /nowhere.mp4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid recorded source hash"):
        runner.describe(boot, clean, root=tmp_path)


def test_canvas_reads_the_first_video_stream(tmp_path: Path) -> None:
    import subprocess

    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "color=c=black:s=320x240:r=30:d=1", "-pix_fmt", "yuv420p", "-y", str(video)],
        check=True, timeout=60,
    )  # fmt: skip
    assert runner.canvas("ffprobe", video) == (320, 240)


def test_main_describe_requires_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="needs --defaults"):
        runner.main([str(tmp_path / "boot"), "describe"])


def test_main_describe_writes_a_new_revision_beside_an_existing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "canvas", lambda _ffprobe, _source: (1080, 1920))
    boot, defaults = bootstrap(tmp_path), machine(tmp_path)
    runner.main([str(boot), "describe", "--defaults", str(defaults)])
    runner.main([str(boot), "describe", "--defaults", str(defaults), "--revision", "v2"])
    assert capsys.readouterr().out.splitlines()[-1].endswith("clean-replay-9-v2.description.json")
    with pytest.raises(ValueError, match="alphanumeric"):
        runner.main([str(boot), "describe", "--defaults", str(defaults), "--revision", "../x"])
