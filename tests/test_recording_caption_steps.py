"""Caption steps rebind the reference plans, call the producers in order and reset on force."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import caption_gradient_audit as gradient_audit
from replay import pipeline_completion
from replay import recording_caption_steps as captions
from replay import recording_steps as steps
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
        elif name == "replay.caption_parts":
            Path(arguments[1]).mkdir()
            for output in ("parts.json", "jobs.json"):
                (Path(arguments[1]) / output).write_text("[]", encoding="utf-8")
        elif name == "replay.caption_compose" and arguments[0] == "plan":
            Path(arguments[5]).mkdir(parents=True)
            (Path(arguments[5]) / "groups.json").write_text("[]", encoding="utf-8")
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


def test_parts_render_compose_and_finish_chain_the_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    producers = Producers(tmp_path)
    (tmp_path / "replay-123/placement-v1.json").write_text("{}", encoding="utf-8")
    upstream = ("cut-plan", "paging", "placement")
    for name in upstream:
        (tmp_path / "steps" / f"{name}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(captions, "load_parts", lambda *_a: [])
    monkeypatch.setattr(captions, "packet_count", lambda _p: 7)
    monkeypatch.setattr(captions, "rendered_gradient", lambda _parts: {"lines": 3})
    run_through(path, producers, "caption-parts", "caption-render", "compose", "finish")
    assert modules(producers) == [
        "replay.caption_parts", "replay.compressor_batch", "replay.caption_compose",
        "replay.caption_compose", "replay.caption_compose",
    ]  # fmt: skip
    parts = producers.commands[0]
    assert parts[parts.index("--") + 1 :] == ["--font-size", "82"]
    assert producers.opened == ["open", "-g", "-a", "/Applications/Compressor Creator Studio.app"]
    assert producers.commands[2][producers.commands[2].index("plan") + 3] == str(
        tmp_path / "clean.mov"
    )
    finish = json.loads((tmp_path / "steps/finish.json").read_text(encoding="utf-8"))
    assert finish["outputs"]["packets"] == 7
    assert (tmp_path / "media/replay-123/final/candidate.mp4").read_bytes() == b"movie"


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
    monkeypatch.setattr(gradient_audit, "line_lifts", lambda _frame: [30.0] * 10)
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
    monkeypatch.setattr(captions, "WEB_VIEWER", tmp_path / "web")
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/bun")
    (tmp_path / "web").mkdir()
    commands: list[list[str]] = []
    outputs = captions.upload(pipeline, lambda command, _log: commands.append(list(command)))
    assert commands == [["/bin/bun", str(tmp_path / "web/src/bin/sync.ts")]]
    assert outputs == {"published": {"delivery": "/d/movie.mp4"}}
    assert (tmp_path / "web/sync.lock").exists()


def test_upload_without_bun_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = steps.load(configure(tmp_path))
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
    outputs = captions.publish(pipeline, lambda _command, _log: None)
    assert seen == ["pipeline-abcdef012345"]
    assert outputs["delivery"] == str(delivery)
    assert json.loads((storage / "evidence/source-probe.json").read_text(encoding="utf-8")) == {
        "format": {"filename": "/m/source.mp4"}
    }
