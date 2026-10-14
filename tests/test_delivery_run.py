import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import delivery_render, delivery_retention, delivery_run
from replay.core import ReplayError


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def defaults(tmp_path: Path) -> dict[str, object]:
    return {
        "version": 1,
        "workRoot": str(tmp_path / "work"),
        "storageRoot": str(tmp_path / "storage"),
        "deliveryRoot": str(tmp_path / "deliveries"),
        "viewerRoot": str(tmp_path / "edits"),
        "measureNative": "/native",
        "foregroundNative": "/foreground-native",
        "calibration": "/calibration.json",
        "motionTemplate": "/template.moti",
        "motionFinish": ["--top", "0", "0", "0"],
        "alphaPreset": "/alpha.setting",
        "encodePreset": "/encode.setting",
        "finalPreset": "/final.setting",
        "encoding": {"codec": "hevc"},
        "separationPython": "/mel/python",
        "separationModelHome": "/mel",
        "summaryExecutable": str(tmp_path / "missing-summary"),
        "minimumTop": 300,
        "unknownTop": 900,
        "foregroundWorkers": 4,
        "measureWorkers": 4,
    }


def recording(tmp_path: Path, source: str, *, vocals: bool) -> delivery_run.Recording:
    data: dict[str, object] = {
        "version": 1,
        "recording": "replay-1",
        "source": str(tmp_path / source),
        "labels": str(tmp_path / "labels.json"),
        "cuts": str(tmp_path / "cuts.json"),
        "sourceRenderConfigs": [str(tmp_path / "renders" / "*.json")],
    }
    if vocals:
        data["vocals"] = str(tmp_path / "vocals.wav")
    return delivery_run.load_recording(write(tmp_path / "delivery.json", data))


class Recorder:
    """Creates each step's declared output so the next step can run."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> int:
        self.commands.append(list(command))
        module = command[2] if len(command) > 2 else ""
        if module == "replay.delivery_media":
            action = command[3]
            target = {
                "proxy": command[-1],
                "clock": command[-1],
                "audio-clock": command[-1],
                "chunk-index": command[4],
                "foreground-plan": f"{command[-1]}/configs/0000.json",
            }[action]
            write(Path(target), {})
        if module == "replay.delivery_captions":
            write(Path(command[-1]), {})
        if "replay.delivery_audio" in command:
            write(Path(command[-1]) / "vocals.wav", {})
        return 0


def test_prepare_builds_every_input_once(tmp_path: Path) -> None:
    write(tmp_path / "renders" / "0000.json", {})
    runner = Recorder()
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mkv", vocals=False),
        defaults(tmp_path),
        runner=runner,
        probe=lambda _path: (720, 1280),
    )
    config = delivery.prepare()
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["audio"]["source"] == str(
        tmp_path / "storage" / "replay-1" / "separation" / "vocals.wav"
    )
    assert data["viewerRoot"] == str(tmp_path / "edits" / "replay-1")
    steps = [command[3] if len(command) > 3 else command[0] for command in runner.commands]
    assert steps[:3] == ["proxy", "clock", "foreground-plan"]
    runner.commands.clear()
    assert delivery.prepare() == config
    assert runner.commands == []


def test_native_canvas_source_needs_no_proxy(tmp_path: Path) -> None:
    write(tmp_path / "renders" / "0000.json", {})
    runner = Recorder()
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True),
        defaults(tmp_path),
        runner=runner,
        probe=lambda _path: (1080, 1920),
    )
    delivery.prepare()
    assert [command[3] for command in runner.commands[:2]] == ["clock", "foreground-plan"]


def test_missing_render_configs_fail(tmp_path: Path) -> None:
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True),
        defaults(tmp_path),
        runner=Recorder(),
        probe=lambda _path: (1080, 1920),
    )
    with pytest.raises(ReplayError, match="render configs"):
        delivery.prepare()


def test_failed_step_stops(tmp_path: Path) -> None:
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True),
        defaults(tmp_path),
        runner=lambda _command: 1,
        probe=lambda _path: (1080, 1920),
    )
    with pytest.raises(ReplayError, match="video clock"):
        delivery.prepare()


def test_measure_runs_both_supervisors_together(tmp_path: Path) -> None:
    launched: list[list[list[str]]] = []

    def launcher(batch: Sequence[Sequence[str]]) -> list[int]:
        launched.append([list(command) for command in batch])
        return [0, 0]

    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True), defaults(tmp_path), launcher=launcher
    )
    delivery.measure()
    assert [command[2] for command in launched[0]] == [
        "replay.delivery_foreground",
        "replay.delivery_measure",
    ]


def test_measure_failure_is_reported(tmp_path: Path) -> None:
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True),
        defaults(tmp_path),
        launcher=lambda batch: [1 for _ in batch],
    )
    with pytest.raises(ReplayError, match="Measurement phase failed"):
        delivery.measure()


def test_finish_runs_pipeline_and_timeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summaries: list[Path] = []
    monkeypatch.setattr(
        delivery_run.Delivery, "summarize", lambda _self, config: summaries.append(config)
    )
    runner = Recorder()
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True), defaults(tmp_path), runner=runner
    )
    delivery.finish(tmp_path / "render-config.json")
    assert [command[2] for command in runner.commands] == [
        "replay.delivery_pipeline",
        "replay.delivery_fcpxml",
        "replay.delivery_retention",
    ]
    assert runner.commands[-1][3:] == [
        str(tmp_path / "render-config.json"),
        "--fcpxml",
        str(tmp_path / "storage" / "replay-1" / "replay-1.fcpxml"),
        "--apply",
    ]
    assert summaries == [tmp_path / "render-config.json"]


def test_finish_skips_rendering_once_intermediates_are_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(delivery_retention, "is_retired", lambda _config: True)
    runner = Recorder()
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True), defaults(tmp_path), runner=runner
    )
    delivery.finish(tmp_path / "render-config.json")
    assert runner.commands == []


def test_finish_reports_a_failed_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(delivery_run.Delivery, "summarize", lambda _self, _config: None)
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True),
        defaults(tmp_path),
        runner=lambda command: 1 if command[2] == "replay.delivery_retention" else 0,
    )
    with pytest.raises(ReplayError, match="intermediate retention failed"):
        delivery.finish(tmp_path / "render-config.json")


def test_finish_reports_pipeline_failure(tmp_path: Path) -> None:
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True), defaults(tmp_path), runner=lambda _c: 1
    )
    with pytest.raises(ReplayError, match="Render pipeline failed"):
        delivery.finish(tmp_path / "render-config.json")


def test_invalid_descriptions_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="delivery.json"):
        delivery_run.load_recording(write(tmp_path / "d.json", {"version": 1}))
    with pytest.raises(ReplayError, match="defaults.json"):
        delivery_run.load_defaults(write(tmp_path / "x.json", {"version": 1}))
    data = {
        "version": 1,
        "recording": "other",
        "source": "/s",
        "labels": "/l",
        "cuts": "/c",
        "sourceRenderConfigs": [],
    }
    with pytest.raises(ReplayError, match="replay-<id>"):
        delivery_run.load_recording(write(tmp_path / "e.json", data))


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing.json"
    assert delivery_run.main([str(missing), "--defaults", str(missing)]) == 1
    assert "DELIVERY_RUN_ERROR" in capsys.readouterr().err


def test_default_runner_and_launcher() -> None:
    assert delivery_run._run(["/usr/bin/true"]) == 0
    assert delivery_run._launch([["/usr/bin/true"], ["/usr/bin/false"]]) == [0, 1]


def test_probe_canvas_reads_dimensions(tmp_path: Path) -> None:
    movie = tmp_path / "probe.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x96:duration=0.1",
            str(movie),
        ],
        check=True,
    )
    assert delivery_run._probe_canvas(movie) == (64, 96)


def test_finish_summarizes_the_delivery_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary"
    summary.write_text("", encoding="utf-8")
    summary.chmod(0o700)
    settings = defaults(tmp_path)
    settings["summaryExecutable"] = str(summary)
    write(tmp_path / "storage" / "replay-1" / "replay-1.fcpxml", {})

    class Render:
        viewer_root = tmp_path / "edits" / "replay-1"

    monkeypatch.setattr(delivery_render, "load_config", lambda _path: Render())
    monkeypatch.setattr(delivery_render, "delivery_stem", lambda _config: "stem")
    runner = Recorder()
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True), settings, runner=runner
    )
    delivery.finish(tmp_path / "render-config.json")
    assert runner.commands[-2] == [
        str(summary),
        "generate",
        "--library",
        str(tmp_path / "edits"),
        "--transcript",
        str(tmp_path / "edits" / "replay-1" / "revisions" / "native-v1" / "delivery" / "stem.srt"),
    ]


@pytest.mark.parametrize("value", [None, 3, "", "relative-summary"])
def test_summary_rejects_invalid_executable_setting(tmp_path: Path, value: object) -> None:
    settings = defaults(tmp_path)
    settings["summaryExecutable"] = value
    delivery = delivery_run.Delivery(recording(tmp_path, "a.mp4", vocals=True), settings)
    with pytest.raises(ReplayError, match="summaryExecutable must be an absolute path"):
        delivery.summarize(tmp_path / "render-config.json")


def test_finish_requires_summary_executable(tmp_path: Path) -> None:
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True), defaults(tmp_path), runner=Recorder()
    )
    with pytest.raises(ReplayError, match="existing executable file"):
        delivery.finish(tmp_path / "render-config.json")


def test_summary_rejects_nonexecutable_file(tmp_path: Path) -> None:
    summary = tmp_path / "summary"
    summary.write_text("", encoding="utf-8")
    summary.chmod(0o600)
    settings = defaults(tmp_path)
    settings["summaryExecutable"] = str(summary)
    delivery = delivery_run.Delivery(recording(tmp_path, "a.mp4", vocals=True), settings)
    with pytest.raises(ReplayError, match="existing executable file"):
        delivery.summarize(tmp_path / "render-config.json")


@pytest.mark.parametrize("exit_code", [1, 2, -9])
def test_summary_failure_can_resume_without_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    settings = defaults(tmp_path)
    settings["summaryExecutable"] = "/usr/bin/true"

    class Render:
        viewer_root = tmp_path / "edits" / "replay-1"

    monkeypatch.setattr(delivery_render, "load_config", lambda _path: Render())
    monkeypatch.setattr(delivery_render, "delivery_stem", lambda _config: "stem")
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True),
        settings,
        runner=lambda command: exit_code if command[0] == "/usr/bin/true" else 0,
    )
    with pytest.raises(ReplayError, match="Transcript summary generation failed"):
        delivery.finish(tmp_path / "render-config.json")
    runner = Recorder()
    delivery.runner = runner
    delivery.summarize(tmp_path / "render-config.json")
    assert runner.commands == [
        [
            "/usr/bin/true",
            "generate",
            "--library",
            str(tmp_path / "edits"),
            "--transcript",
            str(tmp_path / "edits/replay-1/revisions/native-v1/delivery/stem.srt"),
        ]
    ]


def test_summary_launch_error_retains_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = defaults(tmp_path)
    settings["summaryExecutable"] = "/usr/bin/true"

    class Render:
        viewer_root = tmp_path / "edits" / "replay-1"

    def fail(_command: Sequence[str]) -> int:
        raise PermissionError("Cannot execute")

    monkeypatch.setattr(delivery_render, "load_config", lambda _path: Render())
    monkeypatch.setattr(delivery_render, "delivery_stem", lambda _config: "stem")
    delivery = delivery_run.Delivery(
        recording(tmp_path, "a.mp4", vocals=True), settings, runner=fail
    )
    with pytest.raises(ReplayError, match="could not be started") as caught:
        delivery.summarize(tmp_path / "render-config.json")
    assert isinstance(caught.value.__cause__, PermissionError)


def test_run_and_cli_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def prepare(_self: object) -> Path:
        calls.append("prepare")
        return tmp_path

    monkeypatch.setattr(delivery_run.Delivery, "prepare", prepare)
    monkeypatch.setattr(delivery_run.Delivery, "measure", lambda _self: calls.append("measure"))
    monkeypatch.setattr(delivery_run.Delivery, "finish", lambda _self, _c: calls.append("finish"))
    description = write(
        tmp_path / "delivery.json",
        {
            "version": 1,
            "recording": "replay-1",
            "source": "/s.mp4",
            "labels": "/l.json",
            "cuts": "/c.json",
            "sourceRenderConfigs": ["/r/*.json"],
        },
    )
    settings = write(tmp_path / "defaults.json", defaults(tmp_path))
    assert delivery_run.main([str(description), "--defaults", str(settings)]) == 0
    assert calls == ["prepare", "measure", "finish"]
