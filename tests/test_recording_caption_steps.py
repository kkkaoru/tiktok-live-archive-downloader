"""Caption steps rebind the reference plans, call the producers in order and reset on force."""

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from replay import caption_gradient_audit as gradient_audit
from replay import compressor_guard, pipeline_completion, r2_fetch
from replay import recording_caption_steps as captions
from replay import recording_steps as steps
from replay import recording_viewer_steps as viewer
from replay.caption_compose import Part
from replay.core import ReplayError

ROLE_NAMES = ("directory", "stereo", "mono", "monoReceipt", "clock", "clockReceipt", "video",
              "labels", "asr", "source")  # fmt: skip


def configure(tmp_path: Path) -> Path:
    """A pipeline whose front half finished, with caption settings and reference files."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    work = tmp_path / "replay-123"
    for name in ("separation-v1/pipeline", "asr-v1/prepared/state", "asr-v1/prepared/asr", "state"):
        (work / name).mkdir(parents=True, exist_ok=True)
    (work / "audio-clock.json").write_text("{}", encoding="utf-8")
    (work / "source.sha256").write_text(f"{'b' * 64}  {source}\n", encoding="utf-8")
    (work / "separation-v1/pipeline/vocals.wav").write_bytes(b"v")
    data = {
        "version": 1, "recording": "123", "editRoot": str(tmp_path / "edit"),
        "source": str(source), "work": str(work), "steps": str(tmp_path / "steps"),
        "python": "/py", "providerPython": "/provider", "native": "/native",
        "ffmpeg": "/ffmpeg", "ffprobe": "/ffprobe",
        "references": {name: "/r" for name in steps.REFERENCES},
        "yolo": {"weights": "/w.pt", "sha256": "a" * 64, "configDirectory": "/c", "step": "2"},
    }  # fmt: skip
    (tmp_path / "steps").mkdir()
    for name in ("clocks", "asr", "yolo-track"):
        (tmp_path / "steps" / f"{name}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "steps/cleanup.json").write_text(
        json.dumps({"outputs": {"cleanSource": str(tmp_path / "clean.mov")}}), encoding="utf-8"
    )
    references = reference_files(tmp_path)
    settings = {
        "version": 1,
        "references": references,
        "relocations": str(relocations(tmp_path)),
        "mediaRoot": str(tmp_path / "media"),
        "placement": {"regionTop": 120, "regionBottom": 1800, "clearance": 60, "maximumDrift": 150},
        "motionFinish": ["--font-size", "82"],
        "compressor": {"preset": "/p.compressorsetting", "inFlight": 2, "submitWorkers": 1},
        "compose": {"server": "/server", "serverSha256": "c" * 64, "workers": 3},
    }
    (tmp_path / "captions.json").write_text(json.dumps(settings), encoding="utf-8")
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def reference_files(tmp_path: Path) -> dict[str, str]:
    old = tmp_path / "old"
    old.mkdir()
    roles = {name: str(old / name) for name in ROLE_NAMES}
    (tmp_path / "roles.json").write_text(json.dumps(roles), encoding="utf-8")
    stage = {
        "name": "only",
        "command": ["true", *(f"{old}/{name}" for name in ROLE_NAMES)],
        "inputs": [f"{old}/directory/in.json"],
        "outputs": [f"{old}/directory/out.json"],
        "timeout_seconds": 10,
    }
    workflow = {"version": 1, "stages": [stage]}
    (tmp_path / "workflow.json").write_text(json.dumps(workflow), encoding="utf-8")
    closure = tmp_path / "closure.json"
    code = tmp_path / "code.py"
    code.write_text("x = 1\n", encoding="utf-8")
    closure.write_text(json.dumps({"inputs": {str(code): "0" * 64}}), encoding="utf-8")
    font = tmp_path / "font.json"
    font.write_text("{}", encoding="utf-8")
    paging = {
        "version": 1,
        "inputs": {
            "source": {"path": "/old.mp4", "sha256": "0"},
            "codeClosure": {"path": str(closure), "sha256": "0"},
            "fontReceipt": {"path": str(font), "sha256": steps.sha256(font)},
        },
        "futurePaths": {},
        "evidence": {"0": {"path": str(font), "sha256": steps.sha256(font)}},
    }
    (tmp_path / "paging.json").write_text(json.dumps(paging), encoding="utf-8")
    calibration = tmp_path / "envelope.json"
    calibration.write_text("{}", encoding="utf-8")
    return {
        "cutPlanWorkflow": str(tmp_path / "workflow.json"),
        "cutPlanRoles": str(tmp_path / "roles.json"),
        "pagingConfig": str(tmp_path / "paging.json"),
        "calibration": str(calibration),
        "motionTemplate": str(tmp_path / "template.moti"),
    }


def relocations(tmp_path: Path) -> Path:
    table = tmp_path / "RELOCATED.tsv"
    table.write_text(
        "name\told_path\tnew_path\tfiles\tbytes\tmoved_at\n"
        f"moved\t{tmp_path / 'gone'}\t{tmp_path / 'ssd'}\t1\t1\tx\n",
        encoding="utf-8",
    )
    return table


class Producers:
    """Records commands and writes what the real producers would."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.commands: list[list[str]] = []
        self.opened: list[str] = []

    def __call__(self, command: Sequence[str], log: Path) -> None:
        if command[0] == "open":
            self.opened = list(command)
            return
        self.commands.append(list(command))
        work = self.tmp / "replay-123"
        name = command[command.index("-m") + 1]
        arguments = list(command[command.index("-m") + 2 :])
        if name == "replay.workflow" and "cut-plan-v1" in arguments[0]:
            for output in ("supported-cuts.json", "native-config.json"):
                (work / "cut-plan-v1" / output).write_text("{}", encoding="utf-8")
        elif name == "replay.caption_native_continuation":
            prepared = Path(arguments[1])
            for directory in ("paging", "tempo"):
                (prepared / directory).mkdir(parents=True)
            paging = ["true", "-m", "replay.caption_native_paging", "a"]
            verify = ["true", "-m", "replay.caption_reflow_verify", "r"]
            stage = {"name": "s", "command": paging, "verify_command": verify,
                     "inputs": [str(prepared / "i")],
                     "outputs": [str(prepared / "o")], "timeout_seconds": 1}  # fmt: skip
            (prepared / "workflow.json").write_text(json.dumps({"version": 1, "stages": [stage]}))
        elif name == "replay.workflow":
            prepared = work / "paging-v1/prepared"
            (prepared / "paging/reflow.json").write_text("{}", encoding="utf-8")
            (prepared / "tempo/candidate.wav").write_bytes(b"t")
        elif name == "replay.caption_fixed_y_job":
            placed = {"plannedRuns": {"placedCues": 9, "failedRuns": [{}]}}
            Path(arguments[4]).write_text(json.dumps(placed), encoding="utf-8")
        elif name == "replay.speech_onsets":
            Path(arguments[2]).write_text('{"tokens": []}', encoding="utf-8")
        elif name == "replay.caption_parts":
            Path(arguments[1]).mkdir()
            for output in ("parts.json", "jobs.json"):
                (Path(arguments[1]) / output).write_text("[]", encoding="utf-8")
        elif name == "replay.caption_compose" and arguments[0] == "plan":
            Path(arguments[5]).mkdir(parents=True)
            groups = '[{"frames": 5}, {"frames": 2}]'
            (Path(arguments[5]) / "groups.json").write_text(groups, encoding="utf-8")
            video = {"encoding": {"averageBitRate": 8_200_000}}
            (Path(arguments[5]) / "g000.json").write_text(json.dumps({"recipe": {"video": video}}))
        elif name == "replay.caption_compose" and arguments[0] == "finish":
            Path(arguments[3]).write_bytes(b"movie")


def modules(producers: Producers) -> list[str]:
    return [command[command.index("-m") + 1] for command in producers.commands]


def run_through(path: Path, producers: Producers, *names: str) -> None:
    for name in names:
        steps.execute(name, path, run=producers, graph=captions.BACK)


def test_the_cut_plan_rebinds_every_reference_role_and_runs_the_workflow(tmp_path: Path) -> None:
    path = configure(tmp_path)
    producers = Producers(tmp_path)
    receipt = steps.execute("cut-plan", path, run=producers, graph=captions.BACK)
    plan = tmp_path / "replay-123/cut-plan-v1"
    written = json.loads((plan / "workflow-v1.json").read_text(encoding="utf-8"))
    assert written["stages"][0]["command"][1] == str(plan)
    assert written["stages"][0]["command"][-1] == str(tmp_path / "source.mp4")
    assert written["stages"][0]["outputs"] == [str(plan / "out.json")]
    native = json.loads((plan / "native-template.json").read_text(encoding="utf-8"))
    assert native["expectedInputs"]["source"]["sha256"] == "b" * 64
    assert modules(producers) == ["replay.workflow"]
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert sorted(data["inputs"]) == sorted(
        [str(path), str(tmp_path / "captions.json"),
         str(tmp_path / "steps/clocks.json"), str(tmp_path / "steps/asr.json")]
    )  # fmt: skip


def test_a_reference_path_the_workflow_lacks_is_refused() -> None:
    roles = {name: f"/old/{name}" for name in ROLE_NAMES}
    with pytest.raises(ReplayError, match="does not use its clockReceipt path"):
        captions.rebind_workflow("/old/directory", roles, {n: Path("/new") for n in ROLE_NAMES})


def test_a_reference_path_left_after_substitution_is_refused() -> None:
    roles = {name: f"/old/{name}" for name in ROLE_NAMES}
    text = " ".join(roles.values())
    new = {name: Path(f"/new/{name}") for name in ROLE_NAMES}
    new["asr"] = Path("/old/source")
    with pytest.raises(ReplayError, match="remain in the workflow: source"):
        captions.rebind_workflow(text, roles, new)


def test_the_reference_recording_rebinds_onto_itself() -> None:
    roles = {name: f"/old/{name}" for name in ROLE_NAMES}
    text = " ".join(roles.values())
    new = {name: Path(f"/old/{name}") for name in ROLE_NAMES}
    assert captions.rebind_workflow(text, roles, new) == text


def test_incomplete_reference_roles_are_refused() -> None:
    with pytest.raises(ReplayError, match="roles are incomplete"):
        captions.rebind_workflow("", {"directory": "/d"}, {})


def test_paging_rebinds_the_configuration_and_refreshes_the_closure(tmp_path: Path) -> None:
    path = configure(tmp_path)
    producers = Producers(tmp_path)
    run_through(path, producers, "cut-plan", "paging")
    root = tmp_path / "replay-123/paging-v1"
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["inputs"]["source"]["path"] == str(tmp_path / "source.mp4")
    assert config["inputs"]["codeClosure"]["path"] == str(root / "CODE-CLOSURE.json")
    assert config["futurePaths"]["cuts"].endswith("cut-plan-v1/supported-cuts.json")
    closure = json.loads((root / "CODE-CLOSURE.json").read_text(encoding="utf-8"))
    lines = Path(captions.__file__).with_name("caption_native_paging_lines.py")
    stable = Path(captions.__file__).with_name("caption_stable_lines.py")
    assert closure == {
        "inputs": {
            str(tmp_path / "code.py"): steps.sha256(tmp_path / "code.py"),
            str(lines): steps.sha256(lines),
            str(stable): steps.sha256(stable),
        }
    }
    plan = json.loads((root / "prepared/workflow.json").read_text(encoding="utf-8"))
    assert plan["stages"][0]["command"] == [
        "true", "-m", "replay.caption_native_paging_lines", "--maximum-lines", "2",
        "--module", "replay.caption_native_paging", "--", "a",
    ]  # fmt: skip
    assert plan["stages"][0]["verify_command"][3:7] == [
        "--maximum-lines", "2", "--module", "replay.caption_reflow_verify",
    ]  # fmt: skip
    captions.limit_paging_lines(root / "prepared/workflow.json")  # an applied limit is kept
    again = json.loads((root / "prepared/workflow.json").read_text(encoding="utf-8"))
    assert again["stages"][0]["command"] == [
        "true", "-m", "replay.caption_native_paging_lines", "--maximum-lines", "2",
        "--module", "replay.caption_native_paging", "--", "a",
    ]  # fmt: skip
    assert modules(producers)[1:] == ["replay.caption_native_continuation", "replay.workflow"]
    assert producers.commands[-1][1].startswith("PYTHONPATH=")


def test_a_relocated_reference_is_copied_back_after_its_digest_matches(tmp_path: Path) -> None:
    table = relocations(tmp_path)
    moved = tmp_path / "ssd/sub/a.json"
    moved.parent.mkdir(parents=True)
    moved.write_text("{}", encoding="utf-8")
    target = tmp_path / "gone/sub/a.json"
    captions.restore(target, steps.sha256(moved), table)
    assert target.read_text(encoding="utf-8") == "{}"


def test_a_relocated_reference_with_another_digest_is_refused(tmp_path: Path) -> None:
    table = relocations(tmp_path)
    (tmp_path / "ssd").mkdir()
    (tmp_path / "ssd/b.json").write_text("x", encoding="utf-8")
    with pytest.raises(ReplayError, match="differs from its pin"):
        captions.restore(tmp_path / "gone/b.json", "0" * 64, table)


def test_a_missing_unrelocated_reference_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="was not relocated"):
        captions.restore(tmp_path / "elsewhere.json", "0" * 64, relocations(tmp_path))


def test_placement_runs_over_the_whole_output_with_the_yolo_track(tmp_path: Path) -> None:
    path = configure(tmp_path)
    producers = Producers(tmp_path)
    run_through(path, producers, "cut-plan", "paging")
    cuts = {"cuts": [{"source_start_frame": 10, "source_end_frame": 40, "output_start_frame": 60}]}
    (tmp_path / "replay-123/cut-plan-v1/supported-cuts.json").write_text(json.dumps(cuts), "utf-8")
    receipt = steps.execute("placement", path, run=producers, graph=captions.BACK)
    command = producers.commands[-1]
    assert command[command.index("--end-seconds") + 1] == "90/30"
    assert command[command.index("--yolo-track") + 1].endswith("yolo-track-v2.json")
    assert command[-4:] == ["--max-lines", "2", "--unknown-person", "region-top"]
    outputs = json.loads(receipt.read_text(encoding="utf-8"))["outputs"]
    assert (outputs["placedCues"], outputs["failedRuns"]) == (9, 1)
    assert outputs["calibration"]["status"].startswith("provisional")


def test_onsets_align_the_tokens_in_the_ctc_environment_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    producers = Producers(tmp_path)
    monkeypatch.chdir(tmp_path)
    for name in ("cut-plan", "paging"):
        (tmp_path / "steps" / f"{name}.json").write_text("{}", encoding="utf-8")
    receipt = steps.execute("onsets", path, run=producers, graph=captions.BACK)
    command = producers.commands[-1]
    assert command[0] == "/usr/bin/env"
    # The package the step runs (here src/), not the working directory's.
    assert f"PYTHONPATH={Path(captions.__file__).resolve().parents[1]}" in command
    assert f"TMPDIR={tmp_path / 'replay-123/speech-onsets-scratch'}" in command
    assert command[command.index("-m") - 2 :] == [
        str(tmp_path / "edits/.venv-ctc-alignment/bin/python"), "-B",
        "-m", "replay.speech_onsets",
        str(tmp_path / "replay-123/cut-plan-v1/projected-tokens.json"),
        str(tmp_path / "replay-123/paging-v1/prepared/tempo/candidate.wav"),
        str(tmp_path / "replay-123/speech-onsets-v1.json"),
        "--model", str(tmp_path / "edits/models/japanese-wav2vec2-large-rs35kh-2162c9ed"),
    ]  # fmt: skip
    outputs = json.loads(receipt.read_text(encoding="utf-8"))["outputs"]
    assert outputs["onsets"]["path"] == str(tmp_path / "replay-123/speech-onsets-v1.json")
    receipt.unlink()
    steps.execute("onsets", path, run=producers, graph=captions.BACK)
    assert len(producers.commands) == 1  # an existing onsets file is reused


def test_parts_render_compose_and_finish_chain_the_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    producers = Producers(tmp_path)
    (tmp_path / "replay-123/placement-v1.json").write_text("{}", encoding="utf-8")
    upstream = ("cut-plan", "paging", "placement", "onsets")
    for name in upstream:
        (tmp_path / "steps" / f"{name}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(captions, "load_parts", lambda *_a: [])
    monkeypatch.setattr(captions, "packet_count", lambda _p: -1)  # the join's count is used
    monkeypatch.setattr(captions, "rendered_gradient", lambda _parts: {"lines": 3})
    run_through(path, producers, "caption-parts", "caption-render", "compose", "finish")
    assert modules(producers) == [
        "replay.caption_parts", "replay.compressor_batch", "replay.caption_compose",
        "replay.caption_compose", "replay.caption_compose",
    ]  # fmt: skip
    parts = producers.commands[0]
    assert parts[parts.index("--") + 1 :] == ["--font-size", "82"]
    assert parts[parts.index("--tokens") + 1] == str(tmp_path / "replay-123/speech-onsets-v1.json")
    assert producers.opened == ["open", "-g", "-a", "/Applications/Compressor Creator Studio.app"]
    assert producers.commands[2][producers.commands[2].index("plan") + 3] == str(
        tmp_path / "clean.mov"
    )
    planned = producers.commands[2]
    assert planned[-4:] == [
        "--rate-source", str(tmp_path / "source.mp4"),
        "--frame-clock", str(tmp_path / "replay-123/video-clock.json"),
    ]  # fmt: skip
    composed = json.loads((tmp_path / "steps/compose.json").read_text(encoding="utf-8"))
    assert composed["outputs"]["averageBitRate"] == 8_200_000
    render = producers.commands[3]
    assert render[render.index("--tempo") :] == [
        "--tempo", str(tmp_path / "replay-123/paging-v1/prepared/tempo/candidate.wav"),
        "--audio", str(tmp_path / "media/replay-123/final/audio/tempo.m4a"),
        "--ffmpeg", "/ffmpeg",
    ]  # fmt: skip
    finish = json.loads((tmp_path / "steps/finish.json").read_text(encoding="utf-8"))
    assert finish["outputs"]["packets"] == 7
    assert (tmp_path / "media/replay-123/final/candidate.mp4").read_bytes() == b"movie"


def test_a_recording_palette_reaches_the_caption_parts(tmp_path: Path) -> None:
    path = configure(tmp_path)
    settings = json.loads((tmp_path / "captions.json").read_text(encoding="utf-8"))
    green = {"gradientTop": "#B6DF9E", "gradientBottom": "#6DBE3C"}
    (tmp_path / "captions.json").write_text(json.dumps({**settings, "palette": green}), "utf-8")
    producers = Producers(tmp_path)
    (tmp_path / "replay-123/placement-v1.json").write_text("{}", encoding="utf-8")
    for name in ("cut-plan", "paging", "placement", "onsets"):
        (tmp_path / "steps" / f"{name}.json").write_text("{}", encoding="utf-8")
    run_through(path, producers, "caption-parts")
    parts = producers.commands[0]
    assert parts[parts.index("--palette") : parts.index("--")] == [
        "--palette", "#B6DF9E", "#6DBE3C"
    ]  # fmt: skip
    (tmp_path / "captions.json").write_text(
        json.dumps({**settings, "palette": {"gradientTop": "#000000"}}), "utf-8"
    )
    with pytest.raises(ReplayError, match="exactly gradientTop"):
        captions.load_captions(steps.load(path))


def test_forcing_the_render_removes_only_the_rendered_movies(tmp_path: Path) -> None:
    path = configure(tmp_path)
    pipeline = steps.load(path)
    rendered = tmp_path / "media/replay-123/alpha/w000-part-0000/compressor-batch-v1"
    rendered.mkdir(parents=True)
    removed = steps.reset(pipeline, "caption-render", captions.BACK)
    assert [item["path"] for item in removed] == [str(rendered)]
    assert rendered.parent.exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"version": 2}, "Unsupported caption"),
        ({"references": {}}, "Incomplete caption references"),
        ({"compose": {}}, "Incomplete compose settings"),
        ({"motionFinish": [1]}, "list of arguments"),
        ({"mediaRoot": "relative"}, "absolute"),
    ],
)
def test_invalid_caption_settings_are_refused(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    path = configure(tmp_path)
    settings = json.loads((tmp_path / "captions.json").read_text(encoding="utf-8"))
    (tmp_path / "captions.json").write_text(json.dumps({**settings, **change}), "utf-8")
    with pytest.raises(ReplayError, match=message):
        captions.load_captions(steps.load(path))


def test_a_source_digest_for_another_file_is_refused(tmp_path: Path) -> None:
    path = configure(tmp_path)
    (tmp_path / "replay-123/source.sha256").write_text(f"{'b' * 64}  /other\n", "utf-8")
    with pytest.raises(ReplayError, match="does not name"):
        captions.source_digest(steps.load(path))


def test_an_empty_cut_plan_has_no_duration(tmp_path: Path) -> None:
    (tmp_path / "cuts.json").write_text('{"cuts": []}', encoding="utf-8")
    with pytest.raises(ReplayError, match="empty"):
        captions.output_seconds(tmp_path / "cuts.json")


def test_a_missing_closure_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "c.json").write_text('{"inputs": {"/missing.py": "0"}}', encoding="utf-8")
    with pytest.raises(ReplayError, match="missing"):
        captions.refreshed_closure(tmp_path / "c.json", tmp_path / "out.json")


def test_the_cli_runs_one_caption_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = configure(tmp_path)
    monkeypatch.setattr(steps, "Producer", lambda: Producers(tmp_path))
    assert captions.main(["cut-plan", str(path)]) == 0
    assert (tmp_path / "steps/cut-plan.json").exists()


def test_a_paging_plan_without_a_paging_stage_is_refused(tmp_path: Path) -> None:
    plan = tmp_path / "workflow.json"
    plan.write_text(json.dumps({"version": 1, "stages": [{"command": ["true"]}]}), "utf-8")
    with pytest.raises(ReplayError, match="no paging stage"):
        captions.limit_paging_lines(plan)


def test_the_render_gradient_is_measured_on_parts_until_enough_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def frames(path: str) -> list[str]:
        seen.append(path)
        return [path, path]

    monkeypatch.setattr(gradient_audit, "frames", frames)
    monkeypatch.setattr(gradient_audit, "line_measures", lambda _frame: [(80, 30.0)] * 10)
    parts = [Part(f"p{index}", 0, 1, Path(f"/r{index}.mov")) for index in range(3)]
    assert captions.rendered_gradient(parts) == {
        "lines": 20,
        "weakestLift": 30.0,
        "minimumLift": 12.0,
    }
    assert seen == ["/r0.mov"]


def test_the_mono_receipt_is_the_context_streams_own_when_present(tmp_path: Path) -> None:
    assert captions.mono_receipt(tmp_path) == tmp_path / "state/mono.json"
    (tmp_path / "mono-receipt.json").write_text("{}", encoding="utf-8")
    assert captions.mono_receipt(tmp_path) == tmp_path / "mono-receipt.json"


def test_upload_runs_the_web_viewer_sync_under_its_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = steps.load(configure(tmp_path))
    (tmp_path / "steps" / "publish.json").write_text(
        '{"outputs": {"delivery": "/d/movie.mp4"}}', encoding="utf-8"
    )
    monkeypatch.setattr(viewer, "WEB_VIEWER", tmp_path / "web")
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/bun")
    monkeypatch.setattr(time, "monotonic", lambda: 3.0)
    checked: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        viewer, "require_uploaded", lambda video, movie, root: checked.append((video, movie))
    )
    (tmp_path / "web").mkdir()
    commands: list[list[str]] = []
    outputs = captions.upload(pipeline, lambda command, _log: commands.append(list(command)))
    assert commands == [["/bin/bun", str(tmp_path / "web/src/bin/sync.ts"), "--only", "movie"]]
    log = tmp_path / "steps/logs/upload.log"
    assert log.read_text(encoding="utf-8") == "PHASE lock 0.0 s\n"
    assert checked == [("movie", Path("/d/movie.mp4"))]
    assert outputs == {"published": {"delivery": "/d/movie.mp4"}}
    assert (tmp_path / "web/sync.lock").exists()


def test_upload_without_bun_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = steps.load(configure(tmp_path))
    (tmp_path / "steps" / "publish.json").write_text(
        '{"outputs": {"delivery": "/d/movie.mp4"}}', encoding="utf-8"
    )
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(ReplayError, match="bun is required"):
        captions.upload(pipeline, lambda _command, _log: None)


def test_source_probe_records_the_original_once(tmp_path: Path) -> None:
    captions.source_probe(tmp_path, "/m/2026-10-14_18-30-45_JST_replay-1.mp4")
    captions.source_probe(tmp_path, "/m/2026-10-14_18-30-45_JST_replay-1.mp4")
    assert json.loads((tmp_path / "evidence/source-probe.json").read_text(encoding="utf-8")) == {
        "format": {"filename": "/m/2026-10-14_18-30-45_JST_replay-1.mp4"}
    }
    with pytest.raises(ReplayError, match="different original"):
        captions.source_probe(tmp_path, "/m/other.mp4")


def test_source_probe_follows_a_relocated_original(tmp_path: Path) -> None:
    captions.source_probe(tmp_path, "/m/2026-10-07_18-30-20_JST_replay-1.mp4")
    captions.source_probe(tmp_path, "/m/replay-1/2026-10-07_18-30-20_JST_replay-1.mp4")
    assert json.loads((tmp_path / "evidence/source-probe.json").read_text(encoding="utf-8")) == {
        "format": {"filename": "/m/replay-1/2026-10-07_18-30-20_JST_replay-1.mp4"}
    }


def test_publish_writes_the_storage_source_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = steps.load(configure(tmp_path))
    (tmp_path / "steps" / "finish.json").write_text(
        '{"outputs": {"movie": {"path": "/f/candidate.mp4", "sha256": "abcdef0123456789"}}}',
        encoding="utf-8",
    )
    storage = tmp_path / "storage"
    delivery = storage / "revisions/pipeline-abcdef012345/delivery/movie.mp4"
    index = storage / "revisions/pipeline-abcdef012345/metadata/pipeline-evidence.json"
    index.parent.mkdir(parents=True)
    index.write_text('{"source": "/m/source.mp4"}', encoding="utf-8")
    seen: list[str] = []

    def publish(_root: Path, **options: object) -> tuple[Path, Path]:
        seen.append(str(options["revision"]))
        return delivery, index

    monkeypatch.setattr(pipeline_completion, "publish", publish)
    released: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        viewer, "release_older", lambda root, published: released.append((root, published))
    )
    outputs = captions.publish(pipeline, lambda _command, _log: None)
    assert seen == ["pipeline-abcdef012345"]
    assert released == [(tmp_path / "edit", delivery)]
    assert outputs["delivery"] == str(delivery)
    assert json.loads((storage / "evidence/source-probe.json").read_text(encoding="utf-8")) == {
        "format": {"filename": "/m/source.mp4"}
    }


def test_an_unverified_candidate_selection_is_moved_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "time_ns", lambda: 42)
    selection = tmp_path / "evidence/candidate-selection.json"
    selection.parent.mkdir()
    selection.write_text('{"status": "unverified-candidate"}', encoding="utf-8")
    retired = captions.retire_selection(tmp_path)
    assert retired == tmp_path / "evidence/retired/candidate-selection-retired-42.json"
    assert retired.read_text(encoding="utf-8") == '{"status": "unverified-candidate"}'
    assert not selection.exists()
    assert captions.retire_selection(tmp_path) is None


def test_a_selection_of_another_status_is_kept(tmp_path: Path) -> None:
    selection = tmp_path / "evidence/candidate-selection.json"
    selection.parent.mkdir()
    selection.write_text('{"status": "verified"}', encoding="utf-8")
    assert captions.retire_selection(tmp_path) is None
    assert selection.read_text(encoding="utf-8") == '{"status": "verified"}'


def renders(tmp_path: Path, *, gate: object, evidence: str) -> steps.Pipeline:
    """A recording that published, summarized, and still holds its render media."""
    pipeline = steps.load(configure(tmp_path))
    media = tmp_path / "media/replay-123"
    (media / "alpha/w001").mkdir(parents=True)
    (media / "alpha/w001/part.mov").write_bytes(b"alpha")
    (media / "compose").mkdir()
    (media / "compose/g001.mov").write_bytes(b"composed")
    (media / "final").mkdir()
    (media / "final/candidate.mp4").write_bytes(b"movie")
    index = tmp_path / "storage/metadata/pipeline-evidence.json"
    index.parent.mkdir(parents=True)
    index.write_text(evidence, encoding="utf-8")
    published = {"delivery": str(media / "final/candidate.mp4"), "evidence": {"path": str(index)}}
    (tmp_path / "steps/publish.json").write_text(
        json.dumps({"outputs": published}), encoding="utf-8"
    )
    (tmp_path / "steps/summary.json").write_text(
        json.dumps({"outputs": {"gate": gate}}), encoding="utf-8"
    )
    return pipeline


def test_renders_are_retired_after_the_gate_passed(tmp_path: Path) -> None:
    pipeline = renders(tmp_path, gate={"complete": True}, evidence="{}")
    outputs = captions.retire_renders(pipeline, lambda _command, _log: None)
    assert outputs == {
        "removed": [
            {"path": str(tmp_path / "media/replay-123/alpha"), "bytes": 5},
            {"path": str(tmp_path / "media/replay-123/compose"), "bytes": 8},
        ],
        "bytes": 13,
    }
    assert (tmp_path / "media/replay-123/final/candidate.mp4").read_bytes() == b"movie"


def test_renders_renamed_aside_are_retired_with_them(tmp_path: Path) -> None:
    pipeline = renders(tmp_path, gate={"complete": True}, evidence="{}")
    media = tmp_path / "media/replay-123"
    (media / "compose.groups300-20261023").mkdir()
    (media / "compose.groups300-20261023/g001.mov").write_bytes(b"old")
    (media / "alpha.v1").mkdir()
    (media / "alpha.v1/part.mov").write_bytes(b"older")
    outputs = captions.retire_renders(pipeline, lambda _command, _log: None)
    assert outputs == {
        "removed": [
            {"path": str(tmp_path / "media/replay-123/alpha"), "bytes": 5},
            {"path": str(tmp_path / "media/replay-123/compose"), "bytes": 8},
            {"path": str(tmp_path / "media/replay-123/alpha.v1"), "bytes": 5},
            {"path": str(tmp_path / "media/replay-123/compose.groups300-20261023"), "bytes": 3},
        ],
        "bytes": 21,
    }
    assert (tmp_path / "media/replay-123/final/candidate.mp4").read_bytes() == b"movie"


def test_a_renamed_render_named_by_the_evidence_is_kept(tmp_path: Path) -> None:
    media = tmp_path / "media/replay-123"
    pipeline = renders(tmp_path, gate={"complete": True}, evidence=f'["{media}/compose.old/g.mov"]')
    (media / "compose.old").mkdir()
    (media / "compose.old/g.mov").write_bytes(b"old")
    with pytest.raises(ReplayError, match="still references"):
        captions.retire_renders(pipeline, lambda _command, _log: None)
    assert (media / "compose.old/g.mov").read_bytes() == b"old"


def test_renders_named_by_the_evidence_are_kept(tmp_path: Path) -> None:
    media = tmp_path / "media/replay-123"
    pipeline = renders(tmp_path, gate={"complete": True}, evidence=f'["{media}/compose/g001.mov"]')
    with pytest.raises(ReplayError, match="still references"):
        captions.retire_renders(pipeline, lambda _command, _log: None)
    assert (media / "compose/g001.mov").read_bytes() == b"composed"


def test_an_older_summary_passes_the_gate_before_renders_are_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = renders(tmp_path, gate=None, evidence="{}")
    gated: list[str] = []

    def gate(passed: steps.Pipeline) -> dict[str, object]:
        gated.append(passed.text("recording"))
        return {}

    monkeypatch.setattr(viewer, "gate", gate)
    outputs = captions.retire_renders(pipeline, lambda _command, _log: None)
    assert gated == ["123"]
    assert outputs["bytes"] == 13


def test_a_failed_gate_keeps_the_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = renders(tmp_path, gate={"complete": False}, evidence="{}")

    def refuse(_pipeline: steps.Pipeline) -> dict[str, object]:
        raise ReplayError("The completion gate did not pass")

    monkeypatch.setattr(viewer, "gate", refuse)
    with pytest.raises(ReplayError, match="gate did not pass"):
        captions.retire_renders(pipeline, lambda _command, _log: None)
    assert (tmp_path / "media/replay-123/alpha/w001/part.mov").read_bytes() == b"alpha"


def test_the_in_flight_override_replaces_the_configured_count(tmp_path: Path) -> None:
    override = tmp_path / "in-flight.json"
    assert captions.in_flight(2, override) == 2
    override.write_text('{"inFlight": 3}', encoding="utf-8")
    assert captions.in_flight(2, override) == 3


def test_an_in_flight_override_beyond_four_is_refused(tmp_path: Path) -> None:
    override = tmp_path / "in-flight.json"
    override.write_text('{"inFlight": 5}', encoding="utf-8")
    with pytest.raises(ReplayError, match="inFlight integer from 1 to 4"):
        captions.in_flight(2, override)
    override.write_text('{"inFlight": "3"}', encoding="utf-8")
    with pytest.raises(ReplayError, match="inFlight integer from 1 to 4"):
        captions.in_flight(2, override)


def test_finish_counts_a_movie_left_by_an_earlier_run_and_refuses_a_changed_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    pipeline = steps.load(path)
    movie = captions.final_movie(pipeline)
    movie.parent.mkdir(parents=True)
    movie.write_bytes(b"movie")
    monkeypatch.setattr(captions, "packet_count", lambda _p: 9)
    assert captions.finish(pipeline, lambda _c, _l: None)["packets"] == 9
    groups = tmp_path / "groups.json"
    groups.write_text('[{"frames": 4}]', encoding="utf-8")
    receipt = {"outputs": {"groups": {"path": str(groups), "sha256": "0" * 64}}}
    (tmp_path / "steps" / "compose.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ReplayError, match="differs from the compose receipt's pin"):
        captions.planned_frames(pipeline)


def test_finish_fetches_an_offloaded_candidate_back_before_counting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = steps.load(configure(tmp_path))
    movie = captions.final_movie(pipeline)
    asked: list[tuple[Path, Path]] = []

    def restore(path: Path, *, root: Path) -> bool:
        asked.append((path, root))
        path.parent.mkdir(parents=True)
        path.write_bytes(b"movie")
        return True

    monkeypatch.setattr(r2_fetch, "restore_if_offloaded", restore)
    monkeypatch.setattr(captions, "packet_count", lambda _p: 9)
    assert captions.finish(pipeline, lambda _c, _l: None)["packets"] == 9
    assert asked == [(movie, pipeline.path_of("editRoot"))]


def compressor_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[steps.Pipeline, list[str]]:
    pipeline = steps.load(configure(tmp_path))
    (tmp_path / "replay-123" / "caption-parts-v1").mkdir(parents=True)
    (tmp_path / "replay-123" / "caption-parts-v1" / "parts.json").write_text("[]", "utf-8")
    monkeypatch.setattr(captions, "load_parts", lambda *_a: [])
    monkeypatch.setattr(captions, "rendered_gradient", lambda _parts: {"lines": 3})
    events: list[str] = []
    monkeypatch.setattr(compressor_guard, "restart", lambda _app: events.append("restart"))
    return pipeline, events


def test_caption_render_restarts_compressor_once_when_its_controller_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, events = compressor_steps(tmp_path, monkeypatch)

    def run(command: Sequence[str], log: Path) -> None:
        if "replay.compressor_batch" not in command:
            return
        events.append("batch")
        if events.count("batch") == 1:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as stream:
                stream.write("COMPRESSOR_BATCH_ERROR … job controller is down\n")
            raise ReplayError("Producer failed (1)")

    captions.caption_render(pipeline, run)
    assert events == ["batch", "restart", "batch"]


def test_caption_render_does_not_restart_for_other_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, events = compressor_steps(tmp_path, monkeypatch)

    def run(command: Sequence[str], _log: Path) -> None:
        if "replay.compressor_batch" in command:
            raise ReplayError("Producer failed (1)")

    with pytest.raises(ReplayError):
        captions.caption_render(pipeline, run)
    assert events == []


def test_caption_render_resubmits_parts_reported_successful_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, events = compressor_steps(tmp_path, monkeypatch)
    monkeypatch.setattr(compressor_guard, "unverified", lambda *_a: ["w000-part-0001"])
    monkeypatch.setattr(compressor_guard, "retire", lambda _s, names: events.extend(names))

    def run(command: Sequence[str], _log: Path) -> None:
        if "replay.compressor_batch" in command:
            events.append("batch")

    captions.caption_render(pipeline, run)
    assert events == ["batch", "w000-part-0001", "batch"]


def refusing_run(output: Path, events: list[str]) -> Callable[[Sequence[str], Path], None]:
    """A batch whose first run refuses ``output`` as untracked, as compressor_batch does."""

    def run(command: Sequence[str], log: Path) -> None:
        if "replay.compressor_batch" not in command:
            return
        events.append("batch")
        if events.count("batch") == 1:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"COMPRESSOR_BATCH_ERROR Refusing to overwrite an untracked output: {output}\n"
                )
            raise ReplayError("Producer failed (1)")

    return run


def test_caption_render_sets_an_untracked_output_aside_and_submits_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, events = compressor_steps(tmp_path, monkeypatch)
    part = tmp_path / "media/replay-123/alpha/w028-part-0000"
    output = part / "compressor-batch-v1/motion-alpha.mov"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"")
    captions.caption_render(pipeline, refusing_run(output, events))
    assert events == ["batch", "batch"]
    assert list(output.parent.iterdir()) == []
    assert [path.name.startswith("motion-alpha.unverified-") for path in part.glob("*.mov")] == [
        True
    ]


def test_caption_render_never_moves_a_refused_file_outside_its_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline, events = compressor_steps(tmp_path, monkeypatch)
    output = tmp_path / "elsewhere/compressor-batch-v1/motion-alpha.mov"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"")
    with pytest.raises(ReplayError):
        captions.caption_render(pipeline, refusing_run(output, events))
    assert events == ["batch"]
    assert output.read_bytes() == b""


def test_a_render_is_counted_once_per_version(tmp_path: Path) -> None:
    movie = tmp_path / "motion-alpha.mov"
    movie.write_bytes(b"abc")
    counted: list[Path] = []

    def count(path: Path) -> int:
        counted.append(path)
        return 3

    packets = captions.remembered(count)
    assert [packets(movie), packets(movie)] == [3, 3]
    movie.write_bytes(b"abcdef")  # resubmitted: a new version is counted again
    assert packets(movie) == 3
    assert packets(tmp_path / "absent.mov") == 3
    assert counted == [movie, movie, tmp_path / "absent.mov"]


def test_closure_files_map_to_the_package_this_step_runs() -> None:
    package = Path(captions.__file__).resolve().parent
    assert captions.running_copy("/old/checkout/src/replay/stages.py") == str(package / "stages.py")
    assert captions.running_copy("/old/checkout/src/replay/sub/x.py") == str(package / "sub/x.py")
    assert captions.running_copy("/usr/lib/python3/os.py") == "/usr/lib/python3/os.py"


def native_render(tmp_path: Path) -> dict[str, object]:
    return {"binary": "/native/caption-alpha-render", "binarySha256": "a" * 64,
            "floor": "/pinned/floor.json", "sample": 2, "workers": 2}  # fmt: skip


def test_native_render_gates_then_renders_the_same_parts_instead_of_compressor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    settings = json.loads((tmp_path / "captions.json").read_text(encoding="utf-8"))
    (tmp_path / "captions.json").write_text(
        json.dumps({**settings, "nativeRender": native_render(tmp_path)}), "utf-8"
    )
    producers = Producers(tmp_path)
    (tmp_path / "steps/caption-parts.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(captions, "load_parts", lambda *_a: [])
    monkeypatch.setattr(captions, "rendered_gradient", lambda _parts: {"lines": 3})
    native_work = tmp_path / "replay-123/caption-render-v1/native"
    native_work.mkdir(parents=True)
    (native_work / "receipt.json").write_text(
        json.dumps({"renderer": "mixed", "fallback": {"w000-part-0000": "refused"}}), "utf-8"
    )
    monkeypatch.setattr(captions, "in_flight", lambda configured, _override: int(str(configured)))
    hashed: list[Path] = []
    monkeypatch.setattr(compressor_guard, "record_hashes", lambda *a: hashed.append(a[2]))
    run_through(path, producers, "caption-render")
    assert modules(producers) == ["replay.caption_render_native"]
    assert producers.opened == ["open", "-g", "-a", "/Applications/Compressor Creator Studio.app"]
    command = producers.commands[0]
    parts = tmp_path / "replay-123/caption-parts-v1"
    assert command[command.index("-m") + 2 :] == [
        "run", str(parts / "parts.json"), str(parts / "jobs.json"),
        "--binary", "/native/caption-alpha-render", "--binary-sha256", "a" * 64,
        "--work", str(native_work), "--floor-json", "/pinned/floor.json", "--sample", "2",
        "--state", str(tmp_path / "replay-123/caption-render-v1/state"), "--workers", "2",
        "--in-flight", "2", "--submit-workers", "1",
    ]  # fmt: skip
    assert hashed == [tmp_path / "replay-123/caption-render-v1/verified-outputs.json"]
    receipt = json.loads((tmp_path / "steps/caption-render.json").read_text(encoding="utf-8"))
    outputs = receipt["outputs"]
    assert (outputs["renderer"], outputs["fallback"]) == ("mixed", {"w000-part-0000": "refused"})
    assert outputs["native"]["path"] == str(native_work / "receipt.json")


def test_incomplete_native_render_settings_are_refused(tmp_path: Path) -> None:
    path = configure(tmp_path)
    settings = json.loads((tmp_path / "captions.json").read_text(encoding="utf-8"))
    (tmp_path / "captions.json").write_text(
        json.dumps({**settings, "nativeRender": {"binary": "/b"}}), "utf-8"
    )
    with pytest.raises(ReplayError, match="Incomplete nativeRender settings"):
        captions.load_captions(steps.load(path))


def test_a_voice_level_policy_reaches_compose_and_finish_and_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    settings = json.loads((tmp_path / "captions.json").read_text(encoding="utf-8"))
    level = {"targetLufs": -18, "truePeakDbtp": -1}
    (tmp_path / "captions.json").write_text(
        json.dumps({**settings, "voiceLoudness": level}), "utf-8"
    )
    producers = Producers(tmp_path)
    (tmp_path / "steps/caption-render.json").write_text("{}", encoding="utf-8")
    sidecar = tmp_path / "media/replay-123/final/audio/tempo.m4a.loudness.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text('{"gainDb": -9.4}', encoding="utf-8")
    monkeypatch.setattr(captions, "packet_count", lambda _p: -1)
    run_through(path, producers, "compose", "finish")
    render, finished = producers.commands[1], producers.commands[2]
    assert render[-4:] == ["--loudness-target", "-18", "--true-peak-ceiling", "-1"]
    assert finished[-4:] == ["--loudness-target", "-18", "--true-peak-ceiling", "-1"]
    composed = json.loads((tmp_path / "steps/compose.json").read_text(encoding="utf-8"))
    assert composed["outputs"]["voiceLoudness"]["path"] == str(sidecar)


def test_an_invalid_voice_level_policy_is_refused(tmp_path: Path) -> None:
    path = configure(tmp_path)
    settings = json.loads((tmp_path / "captions.json").read_text(encoding="utf-8"))
    (tmp_path / "captions.json").write_text(
        json.dumps({**settings, "voiceLoudness": {"targetLufs": "loud", "truePeakDbtp": -1}}),
        "utf-8",
    )
    with pytest.raises(ReplayError, match="Incomplete voiceLoudness settings"):
        captions.load_captions(steps.load(path))


def test_captions_read_the_stock_phrase_filtered_labels_when_the_asr_step_wrote_them(
    tmp_path: Path,
) -> None:
    assert captions.asr_labels(tmp_path) == tmp_path / "labels-silence-v1.json"
    (tmp_path / "labels-hallucination-v1.json").write_text("{}", encoding="utf-8")
    assert captions.asr_labels(tmp_path) == tmp_path / "labels-hallucination-v1.json"
