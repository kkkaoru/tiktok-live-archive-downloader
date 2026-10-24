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
- ``onsets``: ``replay.speech_onsets`` times each recognized character by CTC alignment against
  the tempo voice track (the delivered audio), in the pinned CTC alignment environment;
- ``caption-parts``: ``replay.caption_parts`` (Motion parts and their Compressor jobs);
- ``caption-render``: ``compressor_batch`` renders every part's alpha movie;
- ``compose``: ``replay.caption_compose plan/render`` over the cleaned source;
- ``finish``: ``replay.caption_compose finish`` adds the tempo-matched voice and joins the groups.
- ``upload-original``, ``excitement``, ``summarize``, ``summary`` and ``clips``: the viewer extras
  (``replay.recording_viewer_steps``); ``summarize`` runs beside the upload, first passing the
  completion gate and superseding incomplete older revisions;
- ``retire-renders``: after the gate passed, deletes the render intermediates (``alpha/``,
  ``compose/`` under the media root, ~43 GB a recording); ``final/`` is the delivery's hardlink;
- ``upload``: the web viewer's ``sync`` (``web-viewer/src/bin/sync.ts``) uploads what changed,
  including the published delivery, to Cloudflare; one sync runs at a time on this checkout.

Resumable like the front half: an output directory without its completion file is redone, sub
workflows reuse finished stages, parts and groups that exist are reused, Compressor continues
from its state. Pinned reference files that were moved to the external work root are copied back
to their recorded path when their digest matches (``RELOCATED-TO-SSD.tsv``).

The style-20 calibration is the surviving envelope (``provisional``): receipts say so.
"""

import csv
import json
import shutil
import sysconfig
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from replay import caption_gradient_audit as gradient_audit
from replay import caption_palette, compressor_batch, compressor_guard, gpu_lease, r2_fetch
from replay import pipeline_completion as completion
from replay import recording_steps as front
from replay import recording_viewer_steps as viewer
from replay.candidate_publication import SELECTION_NAME
from replay.caption_compose import Part, load_parts, packet_count, voice
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
# Runtime override of the Compressor jobs in flight, outside every hashed configuration (so no
# finished receipt goes stale): ``{"inFlight": n}``, read at each caption-render launch.
IN_FLIGHT_OVERRIDE = "in-flight.json"
MAX_IN_FLIGHT_OVERRIDE = 4  # 2 in flight was chosen for Compressor stability; raise in steps
COMPOSE_KEYS = frozenset({"server", "serverSha256", "workers"})
# Optional, off when absent: caption-render draws the finished Motion parts natively
# (``replay.caption_render_native``); parts it cannot verify, or all parts when the parity gate
# on sampled parts fails, are rendered by Motion + Compressor instead.
NATIVE_RENDER_KEYS = frozenset({"binary", "binarySha256", "floor", "sample", "workers"})
# Optional, off when absent: the delivered voice gets one static gain from one EBU R128
# measurement of the tempo PCM, toward the target, capped so its true peak stays at the ceiling.
VOICE_LOUDNESS_KEYS = frozenset({"targetLufs", "truePeakDbtp"})
OPTIONAL = frozenset({"palette", "nativeRender", "voiceLoudness"})
# Settings the pipeline defaults may carry for every new recording (a palette is per recording).
DEFAULT_OPTIONAL = OPTIONAL - {"palette"}
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
# The pinned CTC alignment environment (``--extra ctc-alignment``) and model, in the checkout.
ALIGNER_PYTHON = Path("edits/.venv-ctc-alignment/bin/python")
ALIGNER_MODEL = Path("edits/models/japanese-wav2vec2-large-rs35kh-2162c9ed")
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
    if set(data) - OPTIONAL != KEYS or data["version"] != 1:
        raise ReplayError("Unsupported caption pipeline settings")
    if "palette" in data:
        caption_palette.validated(data["palette"])
    if "nativeRender" in data and set(record(data["nativeRender"])) != NATIVE_RENDER_KEYS:
        raise ReplayError("Incomplete nativeRender settings")
    if "voiceLoudness" in data:
        level = record(data["voiceLoudness"])
        if set(level) != VOICE_LOUDNESS_KEYS or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in level.values()
        ):
            raise ReplayError("Incomplete voiceLoudness settings")
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


def asr_labels(prepared: Path) -> Path:
    """The asr step's labels: without stock phrases since 2026-10-24, else the older ones."""
    filtered = prepared / "labels-hallucination-v1.json"
    return filtered if filtered.exists() else prepared / "labels-silence-v1.json"


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
        "labels": asr_labels(prepared),
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


def running_copy(name: str) -> str:
    """``name`` under the package this process runs (src/replay, or its code snapshot)."""
    parts = Path(name).parts
    for at in range(len(parts) - 1):
        if parts[at : at + 2] == ("src", "replay"):
            return str(Path(__file__).resolve().parent.joinpath(*parts[at + 2 :]))
    return name


def refreshed_closure(reference: Path, output: Path) -> Path:
    """The reference closure's files plus the line-limit entry point, with current digests."""
    files = [*map(running_copy, record(read_json(reference).get("inputs"))), *entry_modules()]
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
    # The package this step runs (src/, or its code snapshot), then the site packages.
    paths = (Path(__file__).resolve().parents[1], Path(sysconfig.get_paths()["purelib"]))
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


def onsets(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    """Per-character speech onsets for the caption reveal (the file appears only when complete)."""
    output, checkout = onsets_path(pipeline), Path.cwd().resolve()
    if not output.exists():
        scratch = pipeline.work / "speech-onsets-scratch"
        scratch.mkdir(mode=0o700, exist_ok=True)
        settings = {"PYTHONPATH": Path(__file__).resolve().parents[1], "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                    "HF_HOME": scratch, "TMPDIR": scratch, "XDG_CACHE_HOME": scratch}  # fmt: skip
        run(
            [front.ENV, *(f"{key}={value}" for key, value in settings.items()),
             str(checkout / ALIGNER_PYTHON), "-B", "-m", "replay.speech_onsets",
             str(pipeline.work / "cut-plan-v1" / "projected-tokens.json"),
             str(tempo_audio(pipeline)), str(output), "--model", str(checkout / ALIGNER_MODEL)],
            pipeline.logs / "onsets.log",
        )  # fmt: skip
    return {"onsets": pin(output)}


def onsets_path(pipeline: Pipeline) -> Path:
    return pipeline.work / "speech-onsets-v1.json"


def caption_parts_step(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    captions, output = load_captions(pipeline), pipeline.work / "caption-parts-v1"
    if not (output / "jobs.json").exists():
        finish = [str(item) for item in record_list(captions.data["motionFinish"])]
        colours = record(captions.data.get("palette", {}))
        recoloured = [f"{colours[key]}" for key in ("gradientTop", "gradientBottom") if colours]
        run(
            module(pipeline, "caption_parts", str(pipeline.work / "placement-v1.json"),
                   str(output), str(captions.media(pipeline) / "alpha"),
                   "--template", str(captions.reference("motionTemplate")),
                   "--preset", str(captions.section("compressor")["preset"]),
                   "--tokens", str(onsets_path(pipeline)),
                   *(["--palette", *recoloured] if recoloured else []), "--", *finish),
            pipeline.logs / "caption-parts.log",
        )  # fmt: skip
    return {"parts": pin(output / "parts.json"), "jobs": pin(output / "jobs.json")}


def record_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ReplayError("Expected a list")
    return value


def in_flight(configured: object, override: Path) -> int:
    """The override file's ``inFlight`` (1-4) when present, else the configured value."""
    if not override.is_file():
        return int(str(configured))
    value = record(json.loads(override.read_text(encoding="utf-8"))).get("inFlight")
    if type(value) is not int or not 1 <= value <= MAX_IN_FLIGHT_OVERRIDE:
        raise ReplayError(f"{override} needs an inFlight integer from 1 to 4")
    return value


def caption_render(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    """Render every part's alpha movie; Compressor continues from its recorded state."""
    captions = load_captions(pipeline)
    if "nativeRender" in captions.data:
        return native_caption_render(pipeline, run, captions)
    settings = captions.section("compressor")
    parts = pipeline.work / "caption-parts-v1"
    # Compressor fails CLI jobs ("job controller is down") unless its app is running, and nothing
    # reopens it after a restart; open it in the background (no-op when already running).
    app = compressor_batch.COMPRESSOR.parents[2]
    log = pipeline.logs / "caption-render.log"
    run(["open", "-g", "-a", str(app)], log)
    batch = module(pipeline, "compressor_batch", str(parts / "jobs.json"),
                   "--state", str(pipeline.work / "caption-render-v1" / "state"),
                   "--in-flight", str(in_flight(settings["inFlight"], gpu_lease.root().parent
                                                / "compressor-leases" / IN_FLIGHT_OVERRIDE)),
                   "--submit-workers", str(settings["submitWorkers"]),
                   "--retry-controller-failure")  # fmt: skip
    since = log.stat().st_size if log.exists() else 0
    try:
        run(batch, log)
    except ReplayError:
        refused = compressor_guard.untracked(log, since, captions.media(pipeline))
        if refused is not None:
            # A Failed job's partial output whose state was retired: set it aside, submit again.
            compressor_guard.set_aside(refused, time.time_ns())
        elif compressor_guard.controller_down(log, since):
            # Circuit breaker: stop submitting, restart Compressor, then resume the batch once.
            compressor_guard.restart(app)
        else:
            raise
        run(batch, log)
    states = pipeline.work / "caption-render-v1" / "state"
    packets = remembered(packet_count)  # verification and loading read each render once
    missing = compressor_guard.unverified(parts / "parts.json", states, packets)
    if missing:
        # "Success without output": those parts are submitted once more.
        compressor_guard.retire(states, missing)
        run(batch, log)
    compressor_guard.record_hashes(
        parts / "parts.json", states, pipeline.work / "caption-render-v1" / "verified-outputs.json"
    )
    rendered = load_parts(parts / "parts.json", packets, states)  # recorded and complete
    return {
        "parts": len(rendered),
        "frames": sum(part.frames for part in rendered),
        "gradient": rendered_gradient(rendered),
    }


def remembered(count: compressor_guard.Packets) -> compressor_guard.Packets:
    """``count`` once per file version (path, size, mtime), so a resubmitted render is counted
    again but an unchanged one is read only once."""
    seen: dict[tuple[str, int, int], int] = {}

    def counted(path: Path) -> int:
        try:
            status = path.stat()
        except OSError:
            return count(path)
        key = (str(path), status.st_size, status.st_mtime_ns)
        if key not in seen:
            seen[key] = count(path)
        return seen[key]

    return counted


def native_caption_render(pipeline: Pipeline, run: Runner, captions: Captions) -> dict[str, object]:
    """The same parts, alpha movies and job states, drawn natively from the finished ``.moti``.

    ``replay.caption_render_native run`` gates on sampled parts first; a failed gate renders
    every part with Motion + Compressor, and a part the native reader cannot verify is rendered
    by them alone. Each fallback and its reason is in the receipt.
    """
    settings, compressor = captions.section("nativeRender"), captions.section("compressor")
    parts, root = pipeline.work / "caption-parts-v1", pipeline.work / "caption-render-v1"
    states, log = root / "state", pipeline.logs / "caption-render.log"
    # The gate and any fallback render with Compressor, which needs its app running.
    run(["open", "-g", "-a", str(compressor_batch.COMPRESSOR.parents[2])], log)
    count = in_flight(
        compressor["inFlight"], gpu_lease.root().parent / "compressor-leases" / IN_FLIGHT_OVERRIDE
    )
    run(
        module(pipeline, "caption_render_native", "run", str(parts / "parts.json"),
               str(parts / "jobs.json"), "--binary", str(settings["binary"]),
               "--binary-sha256", str(settings["binarySha256"]),
               "--work", str(root / "native"), "--floor-json", str(settings["floor"]),
               "--sample", str(settings["sample"]), "--state", str(states),
               "--workers", str(settings["workers"]), "--in-flight", str(count),
               "--submit-workers", str(compressor["submitWorkers"])),
        log,
    )  # fmt: skip
    compressor_guard.record_hashes(parts / "parts.json", states, root / "verified-outputs.json")
    rendered = load_parts(parts / "parts.json", packet_count, states)  # recorded and complete
    receipt = read_json(root / "native" / "receipt.json")
    return {
        "parts": len(rendered),
        "frames": sum(part.frames for part in rendered),
        "gradient": rendered_gradient(rendered),
        "renderer": receipt["renderer"],
        "fallback": receipt["fallback"],
        "native": pin(root / "native" / "receipt.json"),
    }


def rendered_gradient(parts: Sequence[Part]) -> dict[str, object]:
    """The style gradient measured on the first parts' rendered pixels; missing raises."""
    measures: list[tuple[int, float]] = []
    for part in parts:
        if len(gradient_audit.full_lines(measures)) >= GRADIENT_LINES:
            break
        frames = gradient_audit.frames(str(part.render))
        measures += [line for frame in frames for line in gradient_audit.line_measures(frame)]
    return gradient_audit.audit(gradient_audit.full_lines(measures))


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
                   str(pipeline.work / "caption-render-v1" / "state"),
                   "--rate-source", str(pipeline.path_of("source")),
                   "--frame-clock", str(pipeline.work / "video-clock.json")),
            log,
        )  # fmt: skip
    final = final_movie(pipeline)
    final.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    run(
        module(pipeline, "caption_compose", "render", str(requests),
               "--server", str(settings["server"]),
               "--server-sha256", str(settings["serverSha256"]),
               "--workers", str(settings["workers"]),
               "--tempo", str(tempo_audio(pipeline)), "--audio", str(voice(final)),
               "--ffmpeg", pipeline.text("ffmpeg"), *level_arguments(captions)),
        log,
    )  # fmt: skip
    first = requests / "g000.json"
    video = record(record(read_json(first)["recipe"])["video"]) if first.exists() else {}
    levelled = voice(final).with_name(f"{voice(final).name}.loudness.json")
    return {
        "groups": pin(requests / "groups.json"),
        "averageBitRate": record(video.get("encoding", {})).get("averageBitRate"),
        **({"voiceLoudness": pin(levelled)} if levelled.exists() else {}),
    }


def level_arguments(captions: Captions) -> list[str]:
    """``caption_compose`` arguments of the recording's voice level policy, if it has one."""
    if "voiceLoudness" not in captions.data:
        return []
    level = captions.section("voiceLoudness")
    return ["--loudness-target", str(level["targetLufs"]),
            "--true-peak-ceiling", str(level["truePeakDbtp"])]  # fmt: skip


def final_movie(pipeline: Pipeline) -> Path:
    return load_captions(pipeline).media(pipeline) / "final" / "candidate.mp4"


def tempo_audio(pipeline: Pipeline) -> Path:
    return pipeline.work / "paging-v1" / "prepared" / "tempo" / "candidate.wav"


def finish(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    output = final_movie(pipeline)
    r2_fetch.restore_if_offloaded(output, root=pipeline.path_of("editRoot"))  # an R2-only delivery
    if output.exists():  # a movie from an earlier run: count it
        return {"movie": pin(output), "packets": packet_count(output)}
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    run(
        module(pipeline, "caption_compose", "finish",
               str(pipeline.work / "compose-v1" / "requests"), str(tempo_audio(pipeline)),
               str(output), "--ffmpeg", pipeline.text("ffmpeg"),
               *level_arguments(load_captions(pipeline))),
        pipeline.logs / "finish.log",
    )  # fmt: skip
    # caption_compose's join publishes the movie only after counting exactly the planned frames
    # in it, so the count is the plan's: the groups file the compose receipt pins.
    return {"movie": pin(output), "packets": planned_frames(pipeline)}


def planned_frames(pipeline: Pipeline) -> int:
    """Frames of the compose plan, read from the groups file only while it matches its pin."""
    groups = record(record(read_json(pipeline.steps / "compose.json")["outputs"])["groups"])
    path = Path(str(groups["path"]))
    if sha256(path) != groups["sha256"]:
        raise ReplayError(f"{path} differs from the compose receipt's pin")
    return sum(
        int(str(record(row)["frames"])) for row in record_list(json.loads(path.read_text("utf-8")))
    )


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
    retire_selection(delivery.parents[3])
    # Before upload, excitement and summarize start: with an older revision still current the
    # web viewer's catalog kept the old delivery and the upload sent no movie.
    viewer.release_older(pipeline.path_of("editRoot"), delivery)
    return {"delivery": str(delivery), "evidence": pin(index)}


def retire_selection(storage: Path) -> Path | None:
    """Move an older flow's unverified candidate selection aside (never deleted).

    The gate admits that selection as a current delivery: once its movie was retired the gate
    failed on the missing file, and while it exists it outranks the pipeline's delivery.
    """
    selection = storage / "evidence" / SELECTION_NAME
    if not selection.is_file() or read_json(selection).get("status") != "unverified-candidate":
        return None
    retired = storage / "evidence" / "retired" / f"{selection.stem}-retired-{time.time_ns()}.json"
    retired.parent.mkdir(mode=0o700, exist_ok=True)
    selection.rename(retired)
    return retired


def source_probe(storage: Path, source: str) -> None:
    """Record the original for TranscriptViewer, which dates a replay from this probe.

    A probe naming the same file (dated, replay-ID name) at another path predates a relocation of
    the original and is rebound to the current path.
    """
    probe = storage / "evidence" / "source-probe.json"
    if probe.exists():
        recorded = str(record(read_json(probe).get("format")).get("filename"))
        if recorded == source:
            return
        if Path(recorded).name != Path(source).name:
            raise ReplayError(f"{probe} belongs to a different original")
        probe.unlink()
    probe.parent.mkdir(mode=0o700, exist_ok=True)
    write_json(probe, {"format": {"filename": source}})


def upload(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    """Sync the web viewer: this delivery's movie, every changed data file and the catalog.

    Other recordings' movies wait for their own upload steps (or a full sync); the catalog is
    published only once no listed movie is missing from the bucket.
    """
    published = record(read_json(pipeline.steps / "publish.json")["outputs"])
    movie = Path(str(published["delivery"]))
    stem = movie.stem  # the web viewer's video ID
    viewer.sync(run, pipeline.logs / "upload.log", "--only", stem)
    viewer.require_uploaded(stem, movie, root=pipeline.path_of("editRoot"))
    return {"published": published}


def retire_renders(pipeline: Pipeline, _run: Runner) -> dict[str, object]:
    """Delete the alpha and compose media once the gate passed and the delivery needs neither."""
    gate = record(read_json(pipeline.steps / "summary.json").get("outputs")).get("gate")
    if not isinstance(gate, dict) or gate.get("complete") is not True:
        viewer.gate(pipeline)  # a summary from before the gate moved there; raises unless passed
    media = load_captions(pipeline).media(pipeline)
    published = record(read_json(pipeline.steps / "publish.json")["outputs"])
    evidence = Path(str(record(published["evidence"])["path"])).read_text(encoding="utf-8")
    delivered = Path(str(published["delivery"])).resolve()
    # Siblings renamed aside (e.g. compose.groups300-20261023 after a regroup) retire with them.
    targets = [
        *(media / name for name in ("alpha", "compose")),
        *sorted(media.glob("alpha.*")),
        *sorted(media.glob("compose.*")),
    ]
    for target in targets:
        if delivered.is_relative_to(target.resolve()) or str(target) in evidence:
            raise ReplayError(f"The delivery still references {target}; renders are kept")
    removed = [
        {"path": str(target), "bytes": front.remove(target)}
        for target in targets
        if target.exists() and not target.is_symlink()
    ]
    return {"removed": removed, "bytes": sum(int(str(item["bytes"])) for item in removed)}


# Each step's upstream steps (front-half steps included); their receipts are its inputs.
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "upload-original": ("clocks",),
    "cut-plan": ("clocks", "asr"),
    "paging": ("cut-plan",),
    "placement": ("paging", "yolo-track"),
    "onsets": ("paging",),
    "caption-parts": ("placement", "onsets"),
    "caption-render": ("caption-parts",),
    "compose": ("caption-render", "cleanup"),
    "finish": ("compose",),
    "publish": ("finish",),
    "excitement": ("publish",),
    "upload": ("publish", "excitement"),
    "summarize": ("publish",),
    "summary": ("upload",),
    "clips": ("upload", "summary"),
    "retire-renders": ("summary",),
}
STEPS: dict[str, front.StepFunction] = {
    "upload-original": viewer.upload_original,
    "cut-plan": cut_plan,
    "paging": paging,
    "placement": placement,
    "onsets": onsets,
    "caption-parts": caption_parts_step,
    "caption-render": caption_render,
    "compose": compose,
    "finish": finish,
    "publish": publish,
    "excitement": viewer.excitement,
    "upload": upload,
    "summarize": viewer.summarize,
    "summary": viewer.summary,
    "clips": viewer.clips,
    "retire-renders": retire_renders,
}


def step_outputs(pipeline: Pipeline, step: str) -> tuple[Path, ...]:
    """What a forced re-run of ``step`` removes besides its receipt."""
    work, media = pipeline.work, load_captions(pipeline).media(pipeline)
    owned: dict[str, tuple[Path, ...]] = {
        "upload-original": (),
        "cut-plan": (work / "cut-plan-v1",),
        "paging": (work / "paging-v1",),
        "placement": (work / "placement-v1.json",),
        "onsets": (onsets_path(pipeline), work / "speech-onsets-scratch"),
        "caption-parts": (work / "caption-parts-v1", media / "alpha"),
        "caption-render": (
            work / "caption-render-v1",
            *sorted((media / "alpha").glob("*/compressor-*")),
        ),
        "compose": (work / "compose-v1", media / "compose"),
        "finish": (media / "final",),
        "publish": (),  # a published delivery is never deleted by a forced re-run
        "excitement": (),  # sidecars beside a delivery stay with it
        "upload": (),
        "summarize": (),
        "summary": (),
        "clips": (),
        "retire-renders": (),
    }
    return owned[step]


def configurations(pipeline: Pipeline) -> tuple[Path, ...]:
    return (pipeline.path, captions_path(pipeline))


BACK = Graph(STEPS, DEPENDENCIES, configurations, step_outputs)


def main(argv: Sequence[str] | None = None) -> int:
    return front.main(argv, graph=BACK)


if __name__ == "__main__":
    raise SystemExit(main())
