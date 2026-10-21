"""The caption half of a recording's automatic pipeline: cuts, paging, placement, alpha, composite.

The steps continue ``replay.recording_steps`` on the same steps directory, with the settings of
``captions.json`` beside ``pipeline.json`` (a changed caption setting marks only these steps
stale). Each runs the existing producers and keeps their checks:

- ``cut-plan``: the quiet-cut and recognition-input workflow of a reference recording, rebound to
  this recording by exact path substitution (every reference path must be replaced), run with
  ``replay.workflow``;
- ``paging``: ``caption_native_continuation`` from the reference paging configuration with this
  recording's source, voice PCM, audio clock and cuts, then its tempo and paging workflow;
- ``placement``: ``caption_fixed_y_job`` over the whole output with the YOLO face track;
- ``caption-parts``: ``replay.caption_parts`` (Motion parts and their Compressor jobs);
- ``caption-render``: ``compressor_batch`` renders every part's alpha movie;
- ``compose``: ``replay.caption_compose plan/render`` over the cleaned source;
- ``finish``: ``replay.caption_compose finish`` adds the tempo-matched voice and joins the groups.
- ``upload``: the web viewer's ``sync`` (``web-viewer/src/bin/sync.ts``) uploads what changed,
  including the published delivery, to Cloudflare; one sync runs at a time on this checkout.

Resumable like the front half: an output directory without its completion file is redone, sub
workflows reuse finished stages, parts and groups that exist are reused, Compressor continues
from its state. Pinned reference files that were moved to the external work root are copied back
to their recorded path when their digest matches (``RELOCATED-TO-SSD.tsv``).

The style-20 calibration is the surviving envelope (``provisional``): receipts say so.
"""

import csv
import fcntl
import json
import shutil
import sysconfig
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from replay import caption_gradient_audit as gradient_audit
from replay import compressor_batch
from replay import pipeline_completion as completion
from replay import recording_steps as front
from replay.caption_compose import Part, load_parts, packet_count
from replay.core import ReplayError
from replay.recording_steps import Graph, Pipeline, Runner, module, pin, sha256, workflow
from replay.recordings import record

KEYS = frozenset(
    {
        "version",
        "references",
        "relocations",
        "mediaRoot",
        "placement",
        "motionFinish",
        "compressor",
        "compose",
    }
)
REFERENCES = frozenset(
    {"cutPlanWorkflow", "cutPlanRoles", "pagingConfig", "calibration", "motionTemplate"}
)
ROLES = frozenset(
    {"directory", "stereo", "mono", "monoReceipt", "clock", "clockReceipt", "video", "labels",
     "asr", "source"}
)  # fmt: skip
PLACEMENT_KEYS = frozenset({"regionTop", "regionBottom", "clearance", "maximumDrift"})
COMPRESSOR_KEYS = frozenset({"preset", "inFlight", "submitWorkers"})
COMPOSE_KEYS = frozenset({"server", "serverSha256", "workers"})
SECTIONS = (
    ("placement", PLACEMENT_KEYS),
    ("compressor", COMPRESSOR_KEYS),
    ("compose", COMPOSE_KEYS),
)
RATE = 30
# Every page paging can produce is placed, and a run whose person no observation bounds goes
# to the region top instead of being dropped: a delivery shows every caption.
MAX_PAGE_LINES = 2  # the top-anchored Motion template has two text layers
GRADIENT_LINES = 20  # rendered caption lines the gradient audit measures
PAGING_MODULE, LINES_MODULE = "replay.caption_native_paging", "replay.caption_native_paging_lines"
VERIFY_MODULE = "replay.caption_reflow_verify"
UNKNOWN_PERSON = "region-top"
CUT_TRANSFORMS = ("concatenate-decoded-frames", "duration-preserving-resample")
CALIBRATION_STATUS = "provisional: surviving style-20 envelope values, not delivery-approved"
RESTORE_LIMIT = 1 << 26  # only small pinned references are copied back


@dataclass(frozen=True, slots=True)
class Captions:
    """A validated ``captions.json``."""

    path: Path
    data: dict[str, object]

    def reference(self, key: str) -> Path:
        return Path(str(record(self.data["references"])[key]))

    def section(self, key: str) -> dict[str, object]:
        return record(self.data[key])

    def media(self, pipeline: Pipeline) -> Path:
        return Path(str(self.data["mediaRoot"])) / front.recording_name(pipeline.text("recording"))


def captions_path(pipeline: Pipeline) -> Path:
    return pipeline.path.parent / "captions.json"


def load_captions(pipeline: Pipeline) -> Captions:
    path = captions_path(pipeline)
    data = record(json.loads(path.read_text(encoding="utf-8")))
    if set(data) != KEYS or data["version"] != 1:
        raise ReplayError("Unsupported caption pipeline settings")
    if set(record(data["references"])) != REFERENCES:
        raise ReplayError("Incomplete caption references")
    for name, keys in SECTIONS:
        if set(record(data[name])) != keys:
            raise ReplayError(f"Incomplete {name} settings")
    finish = data["motionFinish"]
    if not isinstance(finish, list) or not all(isinstance(item, str) for item in finish):
        raise ReplayError("motionFinish must be a list of arguments")
    if not Path(str(data["mediaRoot"])).is_absolute():
        raise ReplayError("mediaRoot must be an absolute path")
    return Captions(path.resolve(), data)


def read_json(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: object) -> None:
    front.write_new(path, value)


def relocated(path: Path, table: Path) -> Path | None:
    """Where the external work root holds ``path`` after a recorded relocation, if anywhere."""
    with table.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            old, new = Path(row["old_path"]), Path(row["new_path"])
            if path.is_relative_to(old):
                return new / path.relative_to(old)
    return None


def restore(path: Path, digest: str, table: Path) -> None:
    """Make a pinned file present at its recorded path, copying a relocated identical file."""
    if path.exists():
        return
    moved = relocated(path, table)
    if moved is None or not moved.is_file():
        raise ReplayError(f"Pinned reference {path} is missing and was not relocated")
    if moved.stat().st_size > RESTORE_LIMIT or sha256(moved) != digest:
        raise ReplayError(f"Relocated copy of {path} is too large or differs from its pin")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.restoring")
    shutil.copy2(moved, partial)
    if sha256(partial) != digest:
        partial.unlink()
        raise ReplayError(f"Restored copy of {path} differs from its pin")
    partial.rename(path)


def source_digest(pipeline: Pipeline) -> str:
    """The source digest the clocks step recorded (``source.sha256``)."""
    digest, _, name = (pipeline.work / "source.sha256").read_text("utf-8").partition("  ")
    if name.strip() != pipeline.text("source") or len(digest) != 64:
        raise ReplayError("source.sha256 does not name this recording's source")
    return digest


def mono_receipt(prepared: Path) -> Path:
    """The recognition input's receipt: context-stream recognition writes its own."""
    own = prepared / "mono-receipt.json"
    return own if own.exists() else prepared / "state" / "mono.json"


def cut_roles(pipeline: Pipeline, directory: Path) -> dict[str, Path]:
    """This recording's file for every role of the reference cut-plan workflow."""
    work, prepared = pipeline.work, pipeline.work / "asr-v1" / "prepared"
    return {
        "directory": directory,
        "stereo": work / "separation-v1" / "pipeline" / "vocals.wav",
        "mono": prepared / "mono.wav",
        "monoReceipt": mono_receipt(prepared),
        "clock": work / "audio-clock.json",
        "clockReceipt": work / "state" / "source-audio-clock.json",
        "video": work / "video-clock.json",
        "labels": prepared / "labels-silence-v1.json",
        "asr": prepared / "asr" / "inputs.json",
        "source": pipeline.path_of("source"),
    }


def cut_templates(roles: dict[str, Path], *, clock_sha: str, source_sha: str) -> dict[str, object]:
    """The five templates the cut-plan workflow prepares its configurations from."""
    directory = roles["directory"]
    lineage = str(directory / "lineage.json")
    return {
        "lineage-template.json": {
            "version": 1, "prepareKind": "lineage", "clock": "continuous-decoded-pcm",
            "transforms": list(CUT_TRANSFORMS),
            "artifactPaths": {"stereoPCM": str(roles["stereo"]), "monoPCM": str(roles["mono"]),
                              "audioClock": str(roles["clock"])},
        },
        "projection-template.json": {
            "version": 1, "prepareKind": "projection", "inputClock": "continuous-decoded-pcm",
            "transforms": list(CUT_TRANSFORMS),
            "artifactPaths": {"labels": str(roles["labels"]), "pcm": str(roles["mono"]),
                              "audioClock": str(roles["clock"]), "asrInputs": str(roles["asr"]),
                              "clockReceipt": str(roles["clockReceipt"]),
                              "pcmReceipt": str(roles["monoReceipt"]), "lineageEvidence": lineage},
        },
        "quiet-template.json": {
            "version": 1, "prepareKind": "quiet", "pcmClock": "continuous-decoded-pcm",
            "fps": RATE,
            "artifactPaths": {"labels": str(directory / "source-labels.json"),
                              "stereoPCM": str(roles["stereo"]), "audioClock": str(roles["clock"]),
                              "videoClock": str(roles["video"]),
                              "projectionConfig": str(directory / "projection-config.json"),
                              "lineageEvidence": lineage},
        },
        "support-template.json": {
            "version": 1, "kind": "cut-audio-support",
            "artifactPaths": {"cuts": str(directory / "quiet" / "cuts.json"),
                              "labels": str(directory / "source-labels.json"),
                              "audioClock": str(roles["clock"])},
            "expectedInputs": {"audioClock": {"path": str(roles["clock"]), "sha256": clock_sha}},
        },
        "native-template.json": {
            "version": 1, "kind": "native-inputs",
            "artifactPaths": {"cuts": str(directory / "supported-cuts.json"),
                              "projectedTokens": str(directory / "projected-tokens.json"),
                              "source": str(roles["source"])},
            "expectedInputs": {"source": {"path": str(roles["source"]), "sha256": source_sha}},
        },
    }  # fmt: skip


def rebind_workflow(text: str, reference: dict[str, str], roles: dict[str, Path]) -> str:
    """The reference workflow with every reference role path replaced by this recording's."""
    if set(reference) != ROLES:
        raise ReplayError("The cut-plan reference roles are incomplete")
    # Longest first, so a path that extends another (a receipt beside its file) stays whole.
    for name in sorted(reference, key=lambda role: (-len(reference[role]), role)):
        if reference[name] not in text:
            raise ReplayError(f"The reference workflow does not use its {name} path")
        text = text.replace(reference[name], str(roles[name]))
    # The reference recording rebinds onto itself: a role whose path is unchanged is not leftover.
    leftover = [
        name for name, path in reference.items() if path in text and path != str(roles[name])
    ]
    if leftover:
        raise ReplayError(f"Reference paths remain in the workflow: {', '.join(leftover)}")
    return text


def cut_plan(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    captions, directory = load_captions(pipeline), pipeline.work / "cut-plan-v1"
    plan, log = directory / "workflow-v1.json", pipeline.logs / "cut-plan.log"
    front.clear_partial(directory, plan)
    if not directory.exists():
        roles = cut_roles(pipeline, directory)
        directory.mkdir(mode=0o700)
        templates = cut_templates(
            roles, clock_sha=sha256(roles["clock"]), source_sha=source_digest(pipeline)
        )
        for name, value in templates.items():
            write_json(directory / name, value)
        reference = {k: str(v) for k, v in read_json(captions.reference("cutPlanRoles")).items()}
        text = captions.reference("cutPlanWorkflow").read_text(encoding="utf-8")
        plan.write_text(rebind_workflow(text, reference, roles), encoding="utf-8")
    run(workflow(pipeline, plan, directory / "state-v1"), log)
    return {
        "cuts": pin(directory / "supported-cuts.json"),
        "nativeConfig": pin(directory / "native-config.json"),
    }


def refreshed_closure(reference: Path, output: Path) -> Path:
    """The reference closure's files plus the line-limit entry point, with current digests."""
    files = [*record(read_json(reference).get("inputs")), *entry_modules()]
    missing = [name for name in files if not Path(name).is_file()]
    if missing:
        raise ReplayError(f"Code closure files are missing: {', '.join(missing[:3])}")
    write_json(output, {"inputs": {name: sha256(Path(name)) for name in sorted(files)}})
    return output


def paging_config(
    pipeline: Pipeline, captions: Captions, closure: Path, cuts: Path, native: Path
) -> dict[str, object]:
    """The reference paging configuration rebound to this recording's inputs."""
    data = read_json(captions.reference("pagingConfig"))
    work = pipeline.work
    changes = {
        "source": pipeline.path_of("source"),
        "stereoPCM": work / "separation-v1" / "pipeline" / "vocals.wav",
        "audioClock": work / "audio-clock.json",
        "codeClosure": closure,
    }
    inputs = dict(record(data["inputs"]))
    kept = [value for name, value in inputs.items() if name not in changes]
    table = Path(str(captions.data["relocations"]))
    for value in (*kept, *record(data["evidence"]).values()):
        ref = record(value)
        restore(Path(str(ref["path"])), str(ref["sha256"]), table)
    inputs.update({name: pin(path) for name, path in changes.items()})
    return {
        **data,
        "inputs": inputs,
        "futurePaths": {"cuts": str(cuts), "nativeConfig": str(native)},
    }


def lines_module() -> str:
    return str(Path(__file__).resolve().with_name("caption_native_paging_lines.py"))


def entry_modules() -> tuple[str, ...]:
    """The paging entry point and the stable-line module it installs (both stage inputs)."""
    return lines_module(), str(Path(__file__).resolve().with_name("caption_stable_lines.py"))


def limited(command: list[str], module: str) -> list[str]:
    """``command`` running ``module`` through the line-limit entry point (unchanged if done)."""
    if module not in command or LINES_MODULE in command:
        return command
    at = command.index(module)
    entry = [LINES_MODULE, "--maximum-lines", str(MAX_PAGE_LINES), "--module", module, "--"]
    return [*command[:at], *entry, *command[at + 1 :]]


def limit_paging_lines(plan: Path) -> None:
    """Make the paging stage and its verifier use the renderer's line limit (idempotent)."""
    data = read_json(plan)
    stages = [record(stage) for stage in record_list(data.get("stages"))]
    paging = [stage for stage in stages if PAGING_MODULE in record_list(stage.get("command"))]
    if not paging:
        if not any(LINES_MODULE in record_list(stage.get("command")) for stage in stages):
            raise ReplayError("The paging plan has no paging stage")
        return
    for stage in paging:
        stage["command"] = limited([str(i) for i in record_list(stage["command"])], PAGING_MODULE)
        verify = [str(item) for item in record_list(stage.get("verify_command", []))]
        stage["verify_command"] = limited(verify, VERIFY_MODULE)
        inputs = [str(item) for item in record_list(stage.get("inputs"))]
        stage["inputs"] = [*inputs, *(name for name in entry_modules() if name not in inputs)]
    plan.write_text(json.dumps({**data, "stages": stages}, indent=2) + "\n", "utf-8")


def package_environment() -> list[str]:
    """``env`` arguments that let the paging stages' pinned interpreter import this package.

    ``caption_native_continuation`` pins the base interpreter, not the virtual environment, so its
    stages find ``replay`` and its dependencies only through ``PYTHONPATH``.
    """
    paths = (Path.cwd().resolve() / "src", Path(sysconfig.get_paths()["purelib"]))
    return [front.ENV, f"PYTHONPATH={':'.join(str(path) for path in paths)}"]


def paging(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    captions, root = load_captions(pipeline), pipeline.work / "paging-v1"
    config, prepared = root / "config.json", root / "prepared"
    log, cuts = pipeline.logs / "paging.log", pipeline.work / "cut-plan-v1"
    front.clear_partial(root, config)
    if not root.exists():
        root.mkdir(mode=0o700)
        reference = record(read_json(captions.reference("pagingConfig"))["inputs"])
        closure = refreshed_closure(
            Path(str(record(reference["codeClosure"])["path"])), root / "CODE-CLOSURE.json"
        )
        value = paging_config(
            pipeline, captions, closure, cuts / "supported-cuts.json", cuts / "native-config.json"
        )
        write_json(config, value)
    front.clear_partial(prepared, prepared / "workflow.json")
    if not prepared.exists():
        run(module(pipeline, "caption_native_continuation", str(config), str(prepared)), log)
    limit_paging_lines(prepared / "workflow.json")  # idempotent
    run(
        [*package_environment(), *workflow(pipeline, prepared / "workflow.json", root / "state")],
        log,
    )
    return {
        "maximumLines": MAX_PAGE_LINES,
        "reflow": pin(prepared / "paging" / "reflow.json"),
        "tempo": pin(prepared / "tempo" / "candidate.wav"),
    }


def output_seconds(cuts: Path) -> str:
    """The whole output duration in seconds, as an exact fraction of whole frames."""
    rows = read_json(cuts).get("cuts")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("The cut plan is empty")
    last = record(rows[-1])
    start, end = int(str(last["source_start_frame"])), int(str(last["source_end_frame"]))
    return f"{int(str(last['output_start_frame'])) + end - start}/{RATE}"


def placement(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    captions, output = load_captions(pipeline), pipeline.work / "placement-v1.json"
    cuts = pipeline.work / "cut-plan-v1" / "supported-cuts.json"
    settings = captions.section("placement")
    if not output.exists():
        paging_root = pipeline.work / "paging-v1" / "prepared" / "paging"
        run(
            module(pipeline, "caption_fixed_y_job", str(paging_root), "-", str(cuts),
                   str(captions.reference("calibration")), str(output),
                   "--start-seconds", "0", "--end-seconds", output_seconds(cuts),
                   "--region-top", str(settings["regionTop"]),
                   "--region-bottom", str(settings["regionBottom"]),
                   "--clearance", str(settings["clearance"]),
                   "--maximum-drift", str(settings["maximumDrift"]),
                   "--yolo-track", str(pipeline.work / "yolo-track-v2.json"),
                   "--max-lines", str(MAX_PAGE_LINES), "--unknown-person", UNKNOWN_PERSON),
            pipeline.logs / "placement.log",
        )  # fmt: skip
    planned = record(read_json(output).get("plannedRuns"))
    failed = planned.get("failedRuns")
    if not isinstance(failed, list):
        raise ReplayError("The placement candidate records no failed-runs list")
    return {
        "candidate": pin(output),
        "placedCues": planned.get("placedCues"),
        "failedRuns": len(failed),
        "calibration": {**pin(captions.reference("calibration")), "status": CALIBRATION_STATUS},
    }


def caption_parts_step(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    captions, output = load_captions(pipeline), pipeline.work / "caption-parts-v1"
    if not (output / "jobs.json").exists():
        finish = [str(item) for item in record_list(captions.data["motionFinish"])]
        run(
            module(pipeline, "caption_parts", str(pipeline.work / "placement-v1.json"),
                   str(output), str(captions.media(pipeline) / "alpha"),
                   "--template", str(captions.reference("motionTemplate")),
                   "--preset", str(captions.section("compressor")["preset"]), "--", *finish),
            pipeline.logs / "caption-parts.log",
        )  # fmt: skip
    return {"parts": pin(output / "parts.json"), "jobs": pin(output / "jobs.json")}


def record_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ReplayError("Expected a list")
    return value


def caption_render(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    """Render every part's alpha movie; Compressor continues from its recorded state."""
    settings = load_captions(pipeline).section("compressor")
    parts = pipeline.work / "caption-parts-v1"
    # Compressor fails CLI jobs ("job controller is down") unless its app is running, and nothing
    # reopens it after a restart; open it in the background (no-op when already running).
    app = compressor_batch.COMPRESSOR.parents[2]
    run(["open", "-g", "-a", str(app)], pipeline.logs / "caption-render.log")
    run(
        module(pipeline, "compressor_batch", str(parts / "jobs.json"),
               "--state", str(pipeline.work / "caption-render-v1" / "state"),
               "--in-flight", str(settings["inFlight"]),
               "--submit-workers", str(settings["submitWorkers"]),
               "--retry-controller-failure"),
        pipeline.logs / "caption-render.log",
    )  # fmt: skip
    states = pipeline.work / "caption-render-v1" / "state"
    rendered = load_parts(parts / "parts.json", packet_count, states)  # recorded and complete
    return {
        "parts": len(rendered),
        "frames": sum(part.frames for part in rendered),
        "gradient": rendered_gradient(rendered),
    }


def rendered_gradient(parts: Sequence[Part]) -> dict[str, object]:
    """The style gradient measured on the first parts' rendered pixels; missing raises."""
    lifts: list[float] = []
    for part in parts:
        if len(lifts) >= GRADIENT_LINES:
            break
        frames = gradient_audit.frames(str(part.render))
        lifts += [lift for frame in frames for lift in gradient_audit.line_lifts(frame)]
    return gradient_audit.audit(lifts)


def clean_source(pipeline: Pipeline) -> Path:
    """The assembled clean source the cleanup step recorded."""
    receipt = read_json(pipeline.steps / "cleanup.json")
    return Path(str(record(receipt["outputs"])["cleanSource"]))


def compose(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    captions = load_captions(pipeline)
    settings, root = captions.section("compose"), pipeline.work / "compose-v1"
    requests, log = root / "requests", pipeline.logs / "compose.log"
    if not (requests / "groups.json").exists():
        root.mkdir(mode=0o700, exist_ok=True)
        run(
            module(pipeline, "caption_compose", "plan",
                   str(pipeline.work / "cut-plan-v1" / "supported-cuts.json"),
                   str(pipeline.work / "caption-parts-v1" / "parts.json"),
                   str(clean_source(pipeline)), str(captions.media(pipeline) / "compose"),
                   str(requests), "--states",
                   str(pipeline.work / "caption-render-v1" / "state")),
            log,
        )  # fmt: skip
    run(
        module(pipeline, "caption_compose", "render", str(requests),
               "--server", str(settings["server"]),
               "--server-sha256", str(settings["serverSha256"]),
               "--workers", str(settings["workers"])),
        log,
    )  # fmt: skip
    return {"groups": pin(requests / "groups.json")}


def final_movie(pipeline: Pipeline) -> Path:
    return load_captions(pipeline).media(pipeline) / "final" / "candidate.mp4"


def finish(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    output = final_movie(pipeline)
    if not output.exists():
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        run(
            module(pipeline, "caption_compose", "finish",
                   str(pipeline.work / "compose-v1" / "requests"),
                   str(pipeline.work / "paging-v1" / "prepared" / "tempo" / "candidate.wav"),
                   str(output), "--ffmpeg", pipeline.text("ffmpeg")),
            pipeline.logs / "finish.log",
        )  # fmt: skip
    return {"movie": pin(output), "packets": packet_count(output)}


def publish(pipeline: Pipeline, _run: Runner) -> dict[str, object]:
    """Publish the movie as a dated delivery of the recording, with its SRT and evidence."""
    finished = read_json(pipeline.steps / "finish.json")
    movie = record(record(finished["outputs"])["movie"])
    delivery, index = completion.publish(
        pipeline.path_of("editRoot"),
        movie=Path(str(movie["path"])),
        steps=pipeline.steps,
        candidate=pipeline.work / "placement-v1.json",
        revision=f"pipeline-{str(movie['sha256'])[:12]}",
    )
    source_probe(delivery.parents[3], str(read_json(index)["source"]))
    return {"delivery": str(delivery), "evidence": pin(index)}


def source_probe(storage: Path, source: str) -> None:
    """Record the original for TranscriptViewer, which dates a replay from this probe."""
    probe = storage / "evidence" / "source-probe.json"
    if probe.exists():
        if record(read_json(probe).get("format")).get("filename") != source:
            raise ReplayError(f"{probe} belongs to a different original")
        return
    probe.parent.mkdir(mode=0o700, exist_ok=True)
    write_json(probe, {"format": {"filename": source}})


WEB_VIEWER = Path(__file__).resolve().parents[2] / "web-viewer"


def upload(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    """Sync the web viewer (the new delivery and anything else not uploaded yet)."""
    bun = shutil.which("bun")
    if bun is None:
        raise ReplayError("bun is required to upload to the web viewer")
    with (WEB_VIEWER / "sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # concurrent recordings share one sync state
        run([bun, str(WEB_VIEWER / "src" / "bin" / "sync.ts")], pipeline.logs / "upload.log")
    return {"published": read_json(pipeline.steps / "publish.json")["outputs"]}


# Each step's upstream steps (front-half steps included); their receipts are its inputs.
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "cut-plan": ("clocks", "asr"),
    "paging": ("cut-plan",),
    "placement": ("paging", "yolo-track"),
    "caption-parts": ("placement",),
    "caption-render": ("caption-parts",),
    "compose": ("caption-render", "cleanup"),
    "finish": ("compose",),
    "publish": ("finish",),
    "upload": ("publish",),
}
STEPS: dict[str, front.StepFunction] = {
    "cut-plan": cut_plan,
    "paging": paging,
    "placement": placement,
    "caption-parts": caption_parts_step,
    "caption-render": caption_render,
    "compose": compose,
    "finish": finish,
    "publish": publish,
    "upload": upload,
}


def step_outputs(pipeline: Pipeline, step: str) -> tuple[Path, ...]:
    """What a forced re-run of ``step`` removes besides its receipt."""
    work, media = pipeline.work, load_captions(pipeline).media(pipeline)
    owned: dict[str, tuple[Path, ...]] = {
        "cut-plan": (work / "cut-plan-v1",),
        "paging": (work / "paging-v1",),
        "placement": (work / "placement-v1.json",),
        "caption-parts": (work / "caption-parts-v1", media / "alpha"),
        "caption-render": (
            work / "caption-render-v1",
            *sorted((media / "alpha").glob("*/compressor-*")),
        ),
        "compose": (work / "compose-v1", media / "compose"),
        "finish": (media / "final",),
        "publish": (),  # a published delivery is never deleted by a forced re-run
        "upload": (),
    }
    return owned[step]


def configurations(pipeline: Pipeline) -> tuple[Path, ...]:
    return (pipeline.path, captions_path(pipeline))


BACK = Graph(STEPS, DEPENDENCIES, configurations, step_outputs)


def main(argv: Sequence[str] | None = None) -> int:
    return front.main(argv, graph=BACK)


if __name__ == "__main__":
    raise SystemExit(main())
