"""The generic cleanup runner derives chunk jobs, bounds execution and assembles by description."""

import hashlib
import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import delivery_source_assembly_parallel as assembly_runner
from replay import failure_notify, gpu_lease, memory_admission
from replay import recording_priority as priority
from replay import source_clean_run as runner
from replay import subtitle_cleanup_assembly as preparation
from replay import subtitle_sampled_consumer as consumer
from replay.core import ReplayError
from replay.workflow_parallel import MEMORY_FREE_MIN, Load


@pytest.fixture(autouse=True)
def private_gpu_leases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Chunk jobs lease from a test directory, never the checkout's shared one."""
    monkeypatch.setattr(gpu_lease, "root", lambda: tmp_path / "gpu-leases")
    monkeypatch.setattr(memory_admission, "root", lambda: tmp_path / "memory-reservations")
    monkeypatch.setattr(memory_admission, "pressure", lambda: (95, 1))
    monkeypatch.setattr(failure_notify, "notify", lambda *_args, **_kwargs: None)


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
        measure=lambda: measured, poll_seconds=0.01, settle_seconds=0.03,
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
        measure=lambda: IDLE, poll_seconds=0.01, settle_seconds=0.03,
    )  # fmt: skip
    assert not stale.parent.exists() and not media.exists()
    (retired,) = (recording.work / "interrupted").iterdir()
    assert (retired / "attempt-sampled").is_dir()


def test_free_memory_must_cover_one_more_job_above_the_floor(tmp_path: Path) -> None:
    floor_only = Load(cpus=10, load=1.0, memory_free_percent=MEMORY_FREE_MIN, gpu_percent=10)
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
        assert keywords["probes"] == recording.work / "probes"
        evidence = Path(str(keywords["evidence"]))
        evidence.mkdir()
        (evidence / "receipt.json").write_text('{"r": 1}', encoding="utf-8")
        (evidence / "single-source-index.json").write_text('{"i": 1}', encoding="utf-8")
        return evidence / "receipt.json"

    monkeypatch.setattr(consumer, "adapt", adapt)
    monkeypatch.setattr(preparation, "prepare", prepare)
    monkeypatch.setattr(assembly_runner, "assemble_parallel", join)
    (recording.work / "probes/cached").mkdir(parents=True)
    cleanup = runner.assemble(recording)
    assert not (recording.work / "probes").exists()
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
    assert loaded.text("yoloBoxes").endswith("replay-9/yolo-boxes-v1")


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


def test_chunks_lease_under_the_pipeline_recording_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = make(tmp_path, recording="youtube-Eg9-gl3Z954-p1")
    runner.configs(recording)
    keys: list[tuple[str, str, str]] = []
    real = gpu_lease.Waiter

    def waiter(directory: Path, replay_id: str, *, step: str, job: str) -> gpu_lease.Waiter:
        keys.append((replay_id, step, job))
        return real(directory, replay_id, step=step, job=job)

    monkeypatch.setattr(gpu_lease, "Waiter", waiter)
    runner.run(recording, first=0, last=0, runner=lambda *_a: 0, free=lambda: 100 * runner.GIB)
    assert keys == [("youtube-Eg9-gl3Z954", "cleanup chunk-00000", "cleanup-chunk")]


def test_a_settle_longer_than_the_poll_wakes_when_it_ends_and_is_logged(tmp_path: Path) -> None:
    recording = make(tmp_path, streams=2, maxStreams=2)
    runner.configs(recording)
    both = threading.Barrier(2, timeout=5)
    ticks = [0.0]

    def clock() -> float:
        ticks[0] += 1.0  # one settle second per poll
        return ticks[0]

    def job(_command: Sequence[str], _log: Path) -> int:
        both.wait()  # the first chunk runs until the second one is admitted
        return 0

    runner.run(
        recording, first=0, last=1, runner=job, free=lambda: 100 * runner.GIB,
        measure=lambda: TIGHT, poll_seconds=0.01, settle_seconds=3.0, clock=clock,
    )  # fmt: skip
    rows = (recording.work / "logs" / "admission.jsonl").read_text(encoding="utf-8").splitlines()
    waits = [(row["settleLeft"], row["lease"] is None) for row in map(json.loads, rows[1:])]
    assert waits == [(2.0, True), (1.0, True), (0.0, False)]


def test_a_settling_wait_ends_with_the_settle_and_a_leased_one_polls_leases_fast() -> None:
    assert runner.poll_wait(5.0, leased=False, settle_left=0.0) == 5.0
    assert runner.poll_wait(5.0, leased=True, settle_left=0.0) == 2.0
    assert runner.poll_wait(5.0, leased=False, settle_left=0.5) == 0.5
    assert runner.poll_wait(0.01, leased=False, settle_left=0.5) == 0.01


STARVED = Load(cpus=10, load=1.0, memory_free_percent=0, gpu_percent=10)


TIGHT = Load(cpus=10, load=1.0, memory_free_percent=45, gpu_percent=10)  # one more peak fits


def test_the_settle_is_skipped_while_free_memory_covers_two_more_peaks(tmp_path: Path) -> None:
    recording = make(tmp_path, streams=2, maxStreams=2)
    runner.configs(recording)
    both = threading.Barrier(2, timeout=5)

    def job(_command: Sequence[str], _log: Path) -> int:
        both.wait()  # the first chunk runs until the second one is admitted
        return 0

    runner.run(
        recording, first=0, last=1, runner=job, free=lambda: 100 * runner.GIB,
        measure=lambda: IDLE, poll_seconds=0.01, settle_seconds=0.4, clock=lambda: 0.0,
    )  # fmt: skip
    # The clock never moves, so only the skip can admit the second chunk while the first runs.
    rows = (recording.work / "logs" / "admission.jsonl").read_text(encoding="utf-8").splitlines()
    waits = [(row["settleLeft"], row["lease"] is None) for row in map(json.loads, rows[1:])]
    assert waits == [(0.0, False)]
    assert runner.SETTLE_FREE_PERCENT == MEMORY_FREE_MIN + 2 * runner.JOB_PERCENT


def test_a_waiting_chunk_holds_free_leases_only_while_it_could_start(tmp_path: Path) -> None:
    leases = tmp_path / "gpu-leases"
    leases.mkdir()
    (leases / "settings.json").write_text('{"capacity": 2, "priority": ["1"]}', "utf-8")
    blocker = gpu_lease.Waiter(leases, "9", start=9).try_acquire()
    assert blocker is not None
    recording = make(tmp_path, streams=2, maxStreams=2)
    runner.configs(recording)
    reached = {2: threading.Event(), 3: threading.Event()}
    proceed = {2: threading.Event(), 3: threading.Event()}
    calls = [0]
    seen: dict[str, bool] = {}

    def measure() -> Load:
        calls[0] += 1
        if calls[0] in reached:
            reached[calls[0]].set()
            assert proceed[calls[0]].wait(5)
        return STARVED if calls[0] == 2 else IDLE

    def job(_command: Sequence[str], _log: Path) -> int:
        if seen:
            return 0
        assert reached[2].wait(5)  # the second chunk tried for a lease while it could start
        gpu_lease.release(blocker)
        fair = gpu_lease.Waiter(leases, "9", start=9)
        seen["yieldsWhileAdmissible"] = fair.try_acquire() is None
        fair.close()
        proceed[2].set()
        assert reached[3].wait(5)  # it was refused on memory and now waits again
        lease = gpu_lease.Waiter(leases, "9", start=9).try_acquire()
        seen["takesWhileNotAdmitted"] = lease is not None
        if lease is not None:
            gpu_lease.release(lease)
        proceed[3].set()
        return 0

    summary = runner.run(
        recording, first=0, last=1, runner=job, free=lambda: 100 * runner.GIB,
        measure=measure, poll_seconds=0.01, settle_seconds=0.0, leases=leases,
    )  # fmt: skip
    assert seen == {"yieldsWhileAdmissible": True, "takesWhileNotAdmitted": True}
    results = summary["results"]
    assert isinstance(results, dict)
    assert results["0"]["exit"] == 0
    assert results["1"]["exit"] == 0


def accepting(recording: runner.Recording, accepted: dict[str, object]) -> runner.Runner:
    """A chunk job that leaves a receipt naming ``accepted[chunk]`` and that attempt's media."""

    def job(command: Sequence[str], _log: Path) -> int:
        name = Path(command[-1]).stem
        if name not in accepted:
            return 3
        state = recording.work / "state" / name
        state.mkdir(parents=True)
        (state / "chunk-receipt.json").write_text(
            json.dumps({"accepted": accepted[name]}), encoding="utf-8"
        )
        media = recording.media / "sampled" / name / "attempt-a" / consumer.MEDIA_NAME
        media.parent.mkdir(parents=True)
        media.write_bytes(b"media")
        return 0

    return job


def test_accepted_chunks_are_probed_for_the_assembly_during_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = make(tmp_path)
    runner.configs(recording)
    probed: list[tuple[Path, Path, Path]] = []

    def prefetch(media: Path, *, probes: Path, ffprobe: Path) -> Path:
        probed.append((media, probes, ffprobe))
        return probes

    monkeypatch.setattr(assembly_runner, "prefetch", prefetch)
    job = accepting(recording, {"chunk-00000": "attempt-a", "chunk-00001": None})
    runner.run(recording, first=0, last=2, runner=job, free=lambda: 100 * runner.GIB)
    assert probed == [
        (
            tmp_path / "media/replay-1/sampled/chunk-00000/attempt-a/candidate.mp4",
            recording.work / "probes",
            Path("/bin/ffprobe"),
        )
    ]


def test_a_failed_probe_ahead_leaves_the_probe_to_the_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = make(tmp_path)
    runner.configs(recording)

    def prefetch(_media: Path, **_keywords: object) -> Path:
        raise ReplayError("ffprobe failed")

    monkeypatch.setattr(assembly_runner, "prefetch", prefetch)
    job = accepting(recording, {"chunk-00000": "attempt-a"})
    summary = runner.run(recording, first=0, last=0, runner=job, free=lambda: 100 * runner.GIB)
    results = summary["results"]
    assert isinstance(results, dict)
    assert results["0"]["exit"] == 0
    assert capsys.readouterr().err == (
        "chunk-00000: probe ahead failed, the assembly probes it: ffprobe failed\n"
    )


def test_a_chunk_config_without_media_output_is_reported(tmp_path: Path) -> None:
    recording = make(tmp_path)
    runner.configs(recording)
    state = recording.work / "state/chunk-00000"
    state.mkdir(parents=True)
    (state / "chunk-receipt.json").write_text('{"accepted": "attempt-a"}', encoding="utf-8")
    (recording.work / "configs/chunk-00000.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="names no media output"):
        runner.accepted_media(recording, "chunk-00000")


def test_a_settling_chunk_keeps_its_place_ahead_of_lower_ranks(tmp_path: Path) -> None:
    leases = tmp_path / "gpu-leases"
    leases.mkdir()
    (leases / "settings.json").write_text('{"capacity": 3, "priority": ["1"]}', "utf-8")
    first = gpu_lease.Waiter(leases, "9", start=9).try_acquire()
    second = gpu_lease.Waiter(leases, "9", start=9).try_acquire()
    assert first is not None and second is not None
    recording = make(tmp_path, streams=3, maxStreams=3)
    runner.configs(recording)
    tried = threading.Event()
    started = threading.Event()
    checked = threading.Event()
    settling = threading.Event()
    calls = [0]
    seen: dict[str, bool] = {}
    lock = threading.Lock()
    now = [0.0]

    def measure() -> Load:
        if len(list((leases / "waiting").glob("1.*"))) == 2:
            tried.set()  # both later chunks asked for a lease and hold their place
        if not tried.is_set():
            now[0] += 0.5  # the first settles pass; the clock then stops until the check
        if started.is_set():
            settling.set()
        return TIGHT

    def job(_command: Sequence[str], _log: Path) -> int:
        with lock:
            calls[0] += 1
            order = calls[0]
        if order == 2:
            started.set()
            assert checked.wait(5)  # keeps its lease while the lower rank asks
        if order != 1:
            return 0
        assert tried.wait(5)
        gpu_lease.release(first)
        assert started.wait(5)  # one later chunk took it; the last one now settles
        assert settling.wait(5)  # the clock is stopped, so the last chunk is still settling
        gpu_lease.release(second)
        lower = gpu_lease.Waiter(leases, "9", start=9)
        seen["yieldsDuringSettle"] = lower.try_acquire() is None
        lower.close()
        checked.set()
        now[0] += 10.0  # the settle ends
        return 0

    summary = runner.run(
        recording, first=0, last=2, runner=job, free=lambda: 100 * runner.GIB,
        measure=measure, poll_seconds=0.01, settle_seconds=1.0, leases=leases,
        clock=lambda: now[0],
    )  # fmt: skip
    assert seen == {"yieldsDuringSettle": True}
    results = summary["results"]
    assert isinstance(results, dict)
    assert results["2"]["exit"] == 0


def test_chunks_share_the_machine_memory_budget_and_get_a_measured_cap(tmp_path: Path) -> None:
    reservations = tmp_path / "memory-reservations"
    (reservations).mkdir()
    (reservations / "settings.json").write_text('{"budgetGiB": 25}', "utf-8")
    # Two measured runs of 8 GiB: 10 GiB reserved, capped at 16.
    memory_admission.record_peak(reservations, "cleanup-chunk", 8 * runner.GIB)
    memory_admission.record_peak(reservations, "cleanup-chunk", 8 * runner.GIB)
    recording = make(tmp_path, streams=3, maxStreams=3)
    runner.configs(recording)
    active: list[int] = []
    peak: list[int] = []
    commands: list[list[str]] = []
    lock = threading.Lock()
    pair = threading.Barrier(2, timeout=5)

    def job(command: Sequence[str], _log: Path) -> int:
        with lock:
            commands.append(list(command[:3]))
            active.append(1)
            peak.append(len(active))
            order = len(commands)
        if order <= 2:
            pair.wait()  # the two that fit the budget run together
        time.sleep(0.3)  # gives a third, over-budget chunk the chance to start too
        with lock:
            active.pop()
        return 0

    runner.run(
        recording, first=0, last=2, runner=job, free=lambda: 100 * runner.GIB,
        measure=lambda: IDLE, poll_seconds=0.01, settle_seconds=0.0,
    )  # fmt: skip
    assert max(peak) == 2  # 3 x 10 GiB would exceed the 25 GiB budget
    assert commands[0] == [
        "/usr/bin/env",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4274",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4274",
    ]
    assert memory_admission.reserved(reservations) == 0


def box_decisions(recording: runner.Recording) -> list[tuple[int, str]]:
    rows = (recording.work / "logs" / "admission.jsonl").read_text(encoding="utf-8").splitlines()
    return [
        (int(row["chunk"]), str(row["boxes"])) for row in map(json.loads, rows) if "boxes" in row
    ]


def test_chunks_start_once_yolo_track_streamed_their_boxes(tmp_path: Path) -> None:
    boxes = tmp_path / "boxes"
    boxes.mkdir()
    (boxes / "range-000000000000-000000009000.json").write_text("{}", "utf-8")
    recording = make(tmp_path, yoloBoxes=str(boxes))
    runner.configs(recording)
    job = json.loads((recording.work / "configs/chunk-00000.json").read_text("utf-8"))
    assert job["yoloBoxes"] == str(boxes)

    def finish_the_pass(_command: Sequence[str], _log: Path) -> int:
        (boxes / "complete.json").write_text("{}", "utf-8")  # chunk 2's frames never came
        return 0

    runner.run(
        recording, first=0, last=2, runner=finish_the_pass, free=lambda: 100 * runner.GIB,
        measure=lambda: IDLE, poll_seconds=0.01, settle_seconds=0.0,
    )  # fmt: skip
    assert sorted(box_decisions(recording)) == [(0, "stored"), (1, "stored"), (2, "fallback")]


def test_a_stalled_box_stream_stops_every_chunk_from_waiting(tmp_path: Path) -> None:
    recording = make(tmp_path, yoloBoxes=str(tmp_path / "never-written"))
    runner.configs(recording)
    ticks = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        ticks[0] += 1.0
        return ticks[0]

    runner.run(
        recording, first=0, last=2, runner=lambda *_a: 0, free=lambda: 100 * runner.GIB,
        measure=lambda: IDLE, poll_seconds=0.01, settle_seconds=0.0, boxes_stall_seconds=0.5,
        clock=clock, sleep=sleeps.append,
    )  # fmt: skip
    assert set(sleeps) == {0.01}  # one poll without progress, then the stall stops every chunk
    assert sorted(box_decisions(recording)) == [(0, "fallback"), (1, "fallback"), (2, "fallback")]


def test_a_relative_boxes_directory_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(description(tmp_path, yoloBoxes="boxes")), encoding="utf-8")
    with pytest.raises(ValueError, match="yoloBoxes must be an absolute path"):
        runner.load(path, root=tmp_path)


def test_a_finished_yolo_track_without_boxes_never_makes_chunks_wait(tmp_path: Path) -> None:
    (tmp_path / "replay-9").mkdir()
    (tmp_path / "replay-9" / "yolo-track-v2.json").write_text("{}", "utf-8")  # an older pass
    recording = make(tmp_path, yoloBoxes=str(tmp_path / "replay-9" / "yolo-boxes-v1"))
    runner.configs(recording)
    sleeps: list[float] = []
    runner.run(
        recording, first=0, last=2, runner=lambda *_a: 0, free=lambda: 100 * runner.GIB,
        measure=lambda: IDLE, poll_seconds=0.01, settle_seconds=0.0, boxes_stall_seconds=60.0,
        sleep=sleeps.append,
    )  # fmt: skip
    assert sleeps == []
    assert sorted(box_decisions(recording)) == [(0, "fallback"), (1, "fallback"), (2, "fallback")]


def test_a_failing_chunk_is_notified_while_the_others_go_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        failure_notify, "notify", lambda *args, **kw: sent.append((*args, kw["cls"]))
    )
    recording = make(tmp_path)
    runner.configs(recording)

    def job(command: Sequence[str], _log: Path) -> int:
        return 4 if command[-1].endswith("chunk-00001.json") else 0

    summary = runner.run(recording, first=0, last=2, runner=job, free=lambda: 100 * runner.GIB)
    results = summary["results"]
    assert isinstance(results, dict)
    assert [results[key]["exit"] for key in ("0", "1", "2")] == [0, 4, 0]
    assert sent == [("1", "cleanup chunk-00001", "chunk")]


def test_a_drained_recording_starts_no_new_chunk(tmp_path: Path) -> None:
    recording = make(tmp_path)
    runner.configs(recording)
    priority.drain(tmp_path, "1", "drain")  # gpu_lease.root() is tmp_path / "gpu-leases"
    summary = runner.run(
        recording, first=0, last=1, runner=lambda *_a: 0, free=lambda: 100 * runner.GIB
    )
    assert summary["results"] == {
        "0": {"notStarted": True, "drained": "drain"},
        "1": {"notStarted": True, "drained": "drain"},
    }


def test_an_admitted_chunk_polls_fast_for_a_freed_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leases = tmp_path / "gpu-leases"
    leases.mkdir()
    (leases / "settings.json").write_text('{"capacity": 1}', "utf-8")
    blocker = gpu_lease.Waiter(leases, "9", start=9).try_acquire()
    assert blocker is not None
    monkeypatch.setattr(runner, "LEASE_POLL_SECONDS", 0.01)
    recording = make(tmp_path, streams=1, maxStreams=1)
    runner.configs(recording)
    polls = [0]

    def clock() -> float:
        polls[0] += 1
        if polls[0] == 2:
            gpu_lease.release(blocker)  # freed between the first and second lease poll
        return 0.0

    runner.run(
        recording, first=0, last=0, runner=lambda *_a: 0, free=lambda: 100 * runner.GIB,
        measure=lambda: IDLE, poll_seconds=600.0, settle_seconds=0.0, leases=leases,
        clock=clock,
    )  # fmt: skip
    rows = (recording.work / "logs" / "admission.jsonl").read_text(encoding="utf-8").splitlines()
    held = [row["lease"] is not None for row in map(json.loads, rows)]
    assert held == [False, True]  # the second lease poll, not the 600 s one
