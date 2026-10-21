"""One command binds a recording and runs its steps idempotently, with forced re-runs."""

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from replay import gpu_lease, recording_caption_steps, recording_steps
from replay import recording_pipeline as pipeline
from replay.core import ReplayError
from replay.stages import Stage, execution_lock

STEP_SCRIPT = """
import json, pathlib, sys
step, config = sys.argv[1], json.loads(pathlib.Path(sys.argv[2]).read_text())
steps = pathlib.Path(config["steps"])
with (steps / "ran.txt").open("a") as log:
    log.write(step + "\\n")
from replay import recording_caption_steps, recording_steps
back = step in recording_caption_steps.STEPS
graph = recording_caption_steps.BACK if back else recording_steps.FRONT
graph.steps[step] = lambda p, r: {}
recording_steps.execute(step, pathlib.Path(sys.argv[2]), run=lambda c, l: None, graph=graph)
"""


@pytest.fixture(autouse=True)
def private_gpu_leases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GPU stages lease from a test directory, never the checkout's shared one."""
    monkeypatch.setattr(gpu_lease, "root", lambda: tmp_path / "gpu-leases")
    monkeypatch.setenv(gpu_lease.STARTED_VARIABLE, "1")


@pytest.fixture(autouse=True)
def no_storage_binding(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Path]]:
    """Fixture edit roots are minimal; storage binding is tested on its own."""
    bound: list[tuple[Path, Path]] = []
    monkeypatch.setattr(pipeline, "bind_delivery_storage", lambda r, d: bound.append((r, d)))
    return bound


@pytest.fixture(autouse=True)
def zero_origin_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture sources are not media; the lead-in copy is tested on its own."""
    monkeypatch.setattr(pipeline, "zero_origin_source", lambda _d, root: pipeline.edit_source(root))


def defaults(tmp_path: Path, **changes: object) -> Path:
    data: dict[str, object] = {
        "version": 1,
        "root": str(tmp_path / "edits/work/recordings"),
        "python": "/py",
        "providerPython": "/provider",
        "native": "/native",
        "ffmpeg": "/ffmpeg",
        "ffprobe": "/ffprobe",
        "references": {name: f"/ref/{name}.json" for name in sorted(recording_steps.REFERENCES)},
        "yolo": {"weights": "/w.pt", "sha256": "a" * 64, "configDirectory": "/c", "step": "2"},
        "maxParallel": 6,
        "gpuSlots": 3,
        "deliveryRoot": str(tmp_path / "deliveries"),
        "captions": {
            "references": {name: "/r" for name in recording_caption_steps.REFERENCES},
            "relocations": "/t.tsv",
            "mediaRoot": str(tmp_path / "media"),
            "placement": {"regionTop": 1, "regionBottom": 2, "clearance": 3, "maximumDrift": 4},
            "motionFinish": [],
            "compressor": {"preset": "/p", "inFlight": 1, "submitWorkers": 1},
            "compose": {"server": "/s", "serverSha256": "a", "workers": 1},
        },
        **changes,
    }
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def edit_root(tmp_path: Path) -> Path:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    root = tmp_path / "edits/replay-42"
    root.mkdir(parents=True)
    (root / "edit-config.json").write_text(json.dumps({"source": str(source)}), "utf-8")
    return root


def fake_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every step through a tiny script that records its name and writes a receipt."""
    script = tmp_path / "step.py"
    script.write_text(STEP_SCRIPT, encoding="utf-8")

    def stage_of(layout: pipeline.Layout, python: str, name: str) -> Stage:
        receipts = [layout.steps / f"{p}.json" for p in pipeline.DEPENDENCIES[name]]
        return Stage(
            name=name,
            command=(sys.executable, str(script), name, str(layout.pipeline)),
            inputs=(layout.pipeline, *receipts),
            outputs=(layout.steps / f"{name}.json",),
            timeout_seconds=60,
        )

    monkeypatch.setattr(pipeline, "stage_of", stage_of)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))
    edit_root(tmp_path)
    return defaults(tmp_path)


def ran(tmp_path: Path) -> list[str]:
    text = (tmp_path / "edits/work/recordings/replay-42/steps/ran.txt").read_text("utf-8")
    return text.split()


def test_prepare_writes_the_pipeline_configuration(tmp_path: Path) -> None:
    edit_root(tmp_path)
    layout = pipeline.prepare("42", defaults(tmp_path), checkout=tmp_path)
    config = recording_steps.load(layout.pipeline)
    assert config.text("source") == str(tmp_path / "source.mp4")
    assert config.work == layout.directory / "replay-42"
    assert config.steps == layout.steps


def test_prepare_is_idempotent_and_refuses_changed_content(tmp_path: Path) -> None:
    edit_root(tmp_path)
    path = defaults(tmp_path)
    first = pipeline.prepare("42", path, checkout=tmp_path)
    assert pipeline.prepare("42", path, checkout=tmp_path) == first
    with pytest.raises(ReplayError, match="different content"):
        pipeline.prepare("42", defaults(tmp_path, ffmpeg="/other"), checkout=tmp_path)


def test_run_executes_every_step_then_skips_them_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fake_steps(tmp_path, monkeypatch)
    pipeline.prepare("42", path, checkout=tmp_path)
    first = pipeline.run("42", path, checkout=tmp_path)
    assert set(first.values()) == {"completed"}
    assert sorted(ran(tmp_path)) == sorted(pipeline.DEPENDENCIES)
    assert ran(tmp_path)[0] == "clocks"
    assert ran(tmp_path).index("finish") > ran(tmp_path).index("compose")
    second = pipeline.run("42", path, checkout=tmp_path)
    assert set(second.values()) == {"reused"}
    assert len(ran(tmp_path)) == 16


def test_forcing_a_step_redoes_it_and_its_dependents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fake_steps(tmp_path, monkeypatch)
    pipeline.prepare("42", path, checkout=tmp_path)
    pipeline.run("42", path, checkout=tmp_path)
    results = pipeline.run("42", path, checkout=tmp_path, forcing=pipeline.Forcing(("separation",)))
    assert results == {
        "clocks": "reused",
        "separation": "completed",
        "cleanup-bootstrap": "reused",
        "cleanup": "reused",
        "asr": "completed",
        "yolo-track": "reused",
        "retire-intermediates": "completed",
        "cut-plan": "completed",
        "paging": "completed",
        "placement": "completed",
        "caption-parts": "completed",
        "caption-render": "completed",
        "compose": "completed",
        "finish": "completed",
        "publish": "completed",
        "upload": "completed",
    }


def test_forcing_everything_redoes_every_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fake_steps(tmp_path, monkeypatch)
    pipeline.prepare("42", path, checkout=tmp_path)
    pipeline.run("42", path, checkout=tmp_path)
    results = pipeline.run("42", path, checkout=tmp_path, forcing=pipeline.Forcing(every=True))
    assert set(results.values()) == {"completed"}
    assert len(ran(tmp_path)) == 32


def test_a_second_run_of_the_same_recording_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fake_steps(tmp_path, monkeypatch)
    layout = pipeline.prepare("42", path, checkout=tmp_path)
    with (
        execution_lock(layout.directory, "pipeline"),
        pytest.raises(ReplayError, match="Another run"),
    ):
        pipeline.run("42", path, checkout=tmp_path)


def test_downstream_includes_every_dependent_step() -> None:
    assert pipeline.downstream(["cleanup-bootstrap"]) == [
        "cleanup-bootstrap",
        "cleanup",
        "retire-intermediates",
        "compose",
        "finish",
        "publish",
        "upload",
    ]
    assert pipeline.downstream([]) == []
    with pytest.raises(ReplayError, match="Unknown steps: nope"):
        pipeline.downstream(["nope"])


def test_forcing_arguments_round_trip() -> None:
    assert pipeline.Forcing(("asr", "cleanup")).arguments() == [
        "--force",
        "asr",
        "--force",
        "cleanup",
    ]
    assert pipeline.Forcing(every=True).arguments() == ["--force-all"]


def test_a_new_recording_is_bound_through_edit_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_k: calls.append(command))
    assert pipeline.bind("42", "/py", tmp_path) == tmp_path / "edits/replay-42"
    assert calls == [
        ["/py", "-B", "-m", "replay.edit_completion", "init", str(tmp_path / "edits/replay-42"),
         "--engine", str(tmp_path / "src/replay"), "--executor-scope", str(tmp_path)],
    ]  # fmt: skip


def test_a_replay_id_must_be_digits(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="TikTok replay ID or youtube-<video id>"):
        pipeline.bind("../x", "/py", tmp_path)


def test_a_youtube_recording_must_be_bound_from_its_original_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("no download job"))
    with pytest.raises(ReplayError, match="Bind youtube-Eg9-gl3Z954 first"):
        pipeline.bind("youtube-Eg9-gl3Z954", "/py", tmp_path)
    root = tmp_path / "edits/youtube-Eg9-gl3Z954"
    root.mkdir(parents=True)
    (root / "edit-config.json").write_text("{}", encoding="utf-8")
    assert pipeline.bind("youtube-Eg9-gl3Z954", "/py", tmp_path) == root


def test_a_missing_source_is_refused(tmp_path: Path) -> None:
    root = edit_root(tmp_path)
    (tmp_path / "source.mp4").unlink()
    with pytest.raises(ReplayError, match="no existing source"):
        pipeline.edit_source(root)


def test_a_pipeline_root_outside_the_work_tree_is_refused(tmp_path: Path) -> None:
    edit_root(tmp_path)
    with pytest.raises(ReplayError, match="edits/work"):
        pipeline.prepare("42", defaults(tmp_path, root=str(tmp_path / "x")), checkout=tmp_path)


@pytest.mark.parametrize(
    "change",
    [{"version": 2}, {"references": {}}, {"maxParallel": 0}, {"gpuSlots": True}],
)
def test_invalid_defaults_are_refused(tmp_path: Path, change: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        pipeline.load_defaults(defaults(tmp_path, **change))


def test_start_launches_the_run_detached_with_its_forcing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edit_root(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_k: calls.append(command))
    path = defaults(tmp_path)
    forcing = pipeline.Forcing(("asr",))
    assert pipeline.start("42", path, checkout=tmp_path, forcing=forcing) == "pipeline-v1"
    directory = tmp_path / "edits/work/recordings/replay-42"
    assert calls == [
        [str(tmp_path / "scripts/tmux-job.sh"), str(directory), "pipeline-v1", "/py", "-B", "-m",
         "replay.recording_pipeline", "run", "42", "--defaults", str(path), "--force", "asr"],
    ]  # fmt: skip


def test_a_launch_name_skips_earlier_launches(tmp_path: Path) -> None:
    (tmp_path / "pipeline-v1.launch").mkdir()
    assert pipeline.launch_name(pipeline.Layout(tmp_path)) == "pipeline-v2"


def test_too_many_launches_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "MAX_LAUNCHES", 1)
    (tmp_path / "pipeline-v1.launch").mkdir()
    with pytest.raises(ReplayError, match="Too many"):
        pipeline.launch_name(pipeline.Layout(tmp_path))


def test_the_cli_prepares_and_reports_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    edit_root(tmp_path)
    path = defaults(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert pipeline.main(["prepare", "42", "--defaults", str(path)]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "edits/work/recordings/replay-42")
    assert pipeline.main(["prepare", "4x", "--defaults", str(path)]) == 1
    assert "Pipeline stopped (ReplayError)" in capsys.readouterr().err


def test_the_cli_runs_and_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = fake_steps(tmp_path, monkeypatch)
    assert pipeline.main(["run", "42", "--defaults", str(path), "--force-all"]) == 0
    assert "pipeline wall seconds" in capsys.readouterr().out
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: None)
    assert pipeline.main(["start", "42", "--defaults", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "pipeline-v1"


def test_sigterm_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    edit_root(tmp_path)
    monkeypatch.chdir(tmp_path)

    def interrupted(*_a: object, **_k: object) -> pipeline.Layout:
        handler = signal.getsignal(signal.SIGTERM)
        if callable(handler):
            handler(signal.SIGTERM, None)
        raise AssertionError("unreachable")

    monkeypatch.setattr(pipeline, "prepare", interrupted)
    assert pipeline.main(["prepare", "42", "--defaults", str(defaults(tmp_path))]) == 1
    assert "Pipeline stopped (StageError)" in capsys.readouterr().err


def test_delivery_storage_is_bound_once_under_the_delivery_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    calls: list[tuple[Path, Path]] = []

    class Config:
        storage_root: Path | None = None

    monkeypatch.setattr(pipeline, "load_config", lambda _root: Config())
    monkeypatch.setattr(pipeline, "bind_storage", lambda r, d: calls.append((r, d)))
    root = tmp_path / "edits/replay-42"
    pipeline.bind_delivery_storage(root, tmp_path / "deliveries")
    assert calls == [(root, tmp_path / "deliveries/replay-42")]
    assert (tmp_path / "deliveries/replay-42").is_dir()
    Config.storage_root = tmp_path / "bound"
    pipeline.bind_delivery_storage(root, tmp_path / "deliveries")
    assert len(calls) == 1


def test_a_late_video_source_is_rebound_to_its_lead_in_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = edit_root(tmp_path)
    monkeypatch.undo()
    calls: list[tuple[Path, Path, str, str]] = []
    bound: list[tuple[Path, Path]] = []

    def prepared(source: Path, directory: Path, *, ffmpeg: str, ffprobe: str) -> Path:
        calls.append((source, directory, ffmpeg, ffprobe))
        return directory / source.name

    class Config:
        source = tmp_path / "source.mp4"
        storage_root: Path | None = tmp_path / "deliveries/replay-42"

    monkeypatch.setattr("replay.source_lead_in.prepared", prepared)
    monkeypatch.setattr("replay.source_decodable.prepared", lambda source, *_a, **_k: source)
    monkeypatch.setattr(pipeline, "load_config", lambda _root: Config)
    monkeypatch.setattr(pipeline, "bind_source", lambda r, s: bound.append((r, s)))
    data: dict[str, object] = {"ffmpeg": "/ffmpeg", "ffprobe": "/ffprobe"}
    copy = tmp_path / "deliveries/source-lead-in/replay-42/source.mp4"
    assert pipeline.zero_origin_source(data, root) == copy
    assert calls == [
        (tmp_path / "source.mp4", tmp_path / "deliveries/source-lead-in/replay-42", "/ffmpeg",
         "/ffprobe")
    ]  # fmt: skip
    assert bound == [(root, copy)]
    monkeypatch.setattr("replay.source_lead_in.prepared", lambda source, *_a, **_k: source)
    assert pipeline.zero_origin_source(data, root) == tmp_path / "source.mp4"
    assert bound == [(root, copy)]
    Config.storage_root = None
    with pytest.raises(ReplayError, match="no bound storage"):
        pipeline.zero_origin_source(data, root)


def test_an_undecodable_source_is_copied_for_the_pipeline_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = edit_root(tmp_path)
    monkeypatch.undo()
    bound: list[tuple[Path, Path]] = []

    class Config:
        source = tmp_path / "source.mp4"
        storage_root: Path | None = tmp_path / "deliveries/youtube-Eg9-gl3Z954"

    monkeypatch.setattr(
        "replay.source_decodable.prepared", lambda source, directory, **_k: directory / source.name
    )
    monkeypatch.setattr("replay.source_lead_in.prepared", lambda source, *_a, **_k: source)
    monkeypatch.setattr(pipeline, "load_config", lambda _root: Config)
    monkeypatch.setattr(pipeline, "bind_source", lambda r, s: bound.append((r, s)))
    data: dict[str, object] = {"ffmpeg": "/ffmpeg", "ffprobe": "/ffprobe"}
    assert pipeline.zero_origin_source(data, root) == (
        tmp_path / "deliveries/source-decodable/replay-42/source.mp4"
    )
    assert bound == []
