"""One step of a recording's automatic pipeline: the hand-offs agents used to make by hand.

Each step runs the existing producers of one stage in order, with the paths and references the
recording pipeline configuration names, and finishes by writing its receipt. Every producer
keeps its own checks; a step adds none and skips none. ``replay.recording_pipeline`` runs the
steps.

Steps are idempotent and resumable. A step reuses everything an earlier attempt finished and
redoes only what an interruption left unfinished:

- a producer output directory without its completion file (a prepare cut short) is removed and
  produced again;
- in a ``replay.workflow`` sub-plan, completed stages are reused; a stage left ``running`` by an
  interruption (its lock free) loses its receipt and declared outputs and runs again;
- caption removal reuses every chunk with a receipt and redoes only chunks without one;
- separation continues with ``--resume`` (the pipeline's own matching-resume checks).

``reset`` removes one step's outputs and receipt for a forced re-run.

Steps and producers:

- ``clocks``: ``replay_preparation_plan prepare`` and its video and audio clock stages; writes
  ``source.sha256`` for the cleanup description;
- ``cleanup-bootstrap``: ``cleanup_bootstrap_request``, ``replay_cleanup_plan prepare`` and the
  bootstrap workflow up to ``profile-settings``;
- ``cleanup``: ``source_clean_run describe/configs/run/assemble`` (revision ``p1``); a source
  whose profile finds no caption band instead gets a 240-frame ``caption_absence`` survey in
  ``cleanup-bootstrap`` and is composed unchanged;
- ``yolo-track``: ``yolo_track`` over every frame of the video clock;
- ``separation``: ``separation_plan_refresh`` with the reference template set to the GPU device,
  then ``separation_window_pipeline`` in the separation provider environment;
- ``asr``: ``metal_asr_config``, ``vocal_metal_asr_pipeline prepare``, its workflow, and
  ``asr_silence_exclude``;
- ``retire-intermediates``: after ASR and cleanup assembly passed, deletes their bulky
  intermediates (separation chunks and model input, cleanup chunk media) and records each path.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from replay import caption_absence as absence
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import Stage, execution_lock
from replay.workflow import load_stages

KEYS = frozenset(
    {
        "version",
        "recording",
        "editRoot",
        "source",
        "work",
        "steps",
        "python",
        "providerPython",
        "native",
        "ffmpeg",
        "ffprobe",
        "references",
        "yolo",
    }
)
REFERENCES = frozenset(
    {
        "preparationTools",
        "bootstrapRequest",
        "separationConfig",
        "separationTemplate",
        "asrConfig",
        "cleanupDefaults",
    }
)
YOLO_KEYS = frozenset({"weights", "sha256", "configDirectory", "step"})
CLOCK_STAGES = ("source-video-clock", "source-audio-clock")
BOOTSTRAP_TARGET = "profile-settings"
SILENCE_ARGUMENTS = ("--minimum-seconds", "5", "--maximum-amplitude", "8", "--clamp-partial")
# Recognition: "context-streams" (pause-cut chunks with previous-text context, one persistent
# model; replay.asr_context_streams, adopted 2026-10-18) or "windows" (60/30 s windows without
# context, one process per window). Code is not a receipt input: switching needs --force asr.
ASR_METHOD = "context-streams"
SEPARATION_DEVICE = "mps"
YOLO_DEVICE, YOLO_DECODER = "mps", "videotoolbox"  # detector on the GPU, media-engine decode
# The producers check their code pins against these closures (``code_inputs`` in each).
PREPARATION_CLOSURE = ("separation_pipeline_prepare", "code_dependency_inventory")
PLAN_CLOSURE = ("separation_window_pipeline", "code_dependency_inventory")
CLEANUP_REVISION = "p1"
CAPTION_ABSENCE = "caption-absence-v1"
HASH_BLOCK = 1 << 24
ENV = "/usr/bin/env"
STOP_GRACE_SECONDS = 15

Runner = Callable[[Sequence[str], Path], None]
StepFunction = Callable[["Pipeline", Runner], dict[str, object]]


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A validated recording pipeline configuration."""

    path: Path
    data: dict[str, object]

    def text(self, key: str) -> str:
        value = self.data[key]
        if not isinstance(value, str) or not value:
            raise ReplayError(f"Expected text for {key}")
        return value

    def path_of(self, key: str) -> Path:
        return Path(self.text(key))

    def reference(self, key: str) -> Path:
        return Path(str(record(self.data["references"])[key]))

    def yolo(self, key: str) -> str:
        return str(record(self.data["yolo"])[key])

    @property
    def work(self) -> Path:
        return self.path_of("work")

    @property
    def steps(self) -> Path:
        return self.path_of("steps")

    @property
    def logs(self) -> Path:
        return self.steps / "logs"

    @property
    def description(self) -> Path:
        name = f"clean-{self.work.name}-{CLEANUP_REVISION}.description.json"
        return Path.cwd().resolve() / "edits/work" / name


@dataclass(frozen=True, slots=True)
class Graph:
    """A set of steps: what each runs, what it waits for, its configuration files and outputs.

    A step's receipt binds the digests of its configuration files and upstream receipts, so a
    changed setting or upstream result marks it stale. Upstream steps may belong to another graph
    sharing the same steps directory.
    """

    steps: Mapping[str, StepFunction]
    dependencies: Mapping[str, tuple[str, ...]]
    configurations: Callable[["Pipeline"], tuple[Path, ...]]
    outputs: Callable[["Pipeline", str], tuple[Path, ...]]


def load(path: Path) -> Pipeline:
    data = record(json.loads(path.read_text(encoding="utf-8")))
    references, yolo = record(data.get("references")), record(data.get("yolo"))
    if set(data) != KEYS or data["version"] != 1:
        raise ReplayError("Unsupported recording pipeline configuration")
    if set(references) != REFERENCES or set(yolo) != YOLO_KEYS:
        raise ReplayError("Incomplete pipeline references or detector settings")
    pipeline = Pipeline(path.resolve(), data)
    for key in ("editRoot", "source", "work", "steps", "python", "providerPython", "native"):
        if not pipeline.path_of(key).is_absolute():
            raise ReplayError(f"{key} must be an absolute path")
    return pipeline


class Producer:
    """Runs producers one at a time; ``stop`` ends the current one, gently first."""

    def __init__(self) -> None:
        self._child: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def __call__(self, command: Sequence[str], log: Path) -> None:
        """Run with output appended to the step log; a failure stops the step."""
        log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with log.open("ab") as stream:
            stream.write(f"$ {' '.join(command)}\n".encode())
            stream.flush()
            with subprocess.Popen(
                list(command), stdin=subprocess.DEVNULL, stdout=stream, stderr=stream
            ) as child:
                with self._lock:
                    self._child = child
                code = child.wait()
                with self._lock:
                    self._child = None
        if code != 0:
            raise ReplayError(f"Producer failed ({code}); inspect {log}")

    def stop(self) -> None:
        """SIGTERM lets a producer stop its own children; SIGKILL follows after a grace."""
        with self._lock:
            child = self._child
        if child is None:
            return
        child.terminate()
        try:
            child.wait(timeout=STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            child.kill()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def module(pipeline: Pipeline, name: str, *arguments: str, provider: bool = False) -> list[str]:
    python = pipeline.text("providerPython" if provider else "python")
    return [python, "-B", "-m", f"replay.{name}", *arguments]


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def remove(path: Path) -> int:
    """Delete a file or tree; return the bytes it held (0 when absent)."""
    if path.is_symlink() or path.is_file():
        size = path.lstat().st_size
        path.unlink()
        return size
    if not path.exists():
        return 0
    size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    shutil.rmtree(path)
    return size


def read_receipt(pipeline: Pipeline, step: str) -> dict[str, object]:
    return record(json.loads((pipeline.steps / f"{step}.json").read_text(encoding="utf-8")))


def clear_partial(directory: Path, complete: Path) -> None:
    """Remove a producer's output directory that lacks its completion file."""
    if directory.exists() and not complete.exists():
        remove(directory)


def owned_directory(
    directory: Path, name: str, stages: Sequence[Stage], plan: Path, state: Path
) -> bool:
    """True when ``directory`` holds only the files of stage ``name`` (not the plan or state).

    Consumers reading the stage's own outputs there do not share the directory.
    """
    if plan.is_relative_to(directory) or state.is_relative_to(directory):
        return False
    own = {path for stage in stages if stage.name == name for path in stage.outputs}
    others = (path for stage in stages if stage.name != name
              for path in (*stage.inputs, *stage.outputs) if path not in own)  # fmt: skip
    return not any(path.is_relative_to(directory) for path in others)


def clear_interrupted(plan: Path, state: Path) -> list[str]:
    """Let a sub-plan rerun stages an interruption left behind; return their names.

    That is a stage whose receipt says ``running``, or one without a receipt whose outputs are
    incomplete while a directory only it writes to exists (partial files of a stopped run).
    """
    if not plan.exists():
        return []
    cleared: list[str] = []
    stages = load_stages(plan)
    for stage in stages:
        receipt = state / f"{stage.name}.json"
        if receipt.is_file():
            status = record(json.loads(receipt.read_text(encoding="utf-8"))).get("status")
            if status != "running":
                continue
        elif all(output.exists() for output in stage.outputs) or not any(
            output.parent.exists()
            and owned_directory(output.parent, stage.name, stages, plan, state)
            for output in stage.outputs
        ):
            continue
        with execution_lock(state, stage.name) as free:
            if not free:
                raise ReplayError(f"Stage {stage.name} is still running elsewhere")
            for output in stage.outputs:
                remove(output)
                # Producers that create their output directory refuse one left behind (partial
                # files included); it is removed only when no other stage reads or writes there.
                if owned_directory(output.parent, stage.name, stages, plan, state):
                    remove(output.parent)
            receipt.unlink(missing_ok=True)
        cleared.append(stage.name)
    return cleared


def workflow(pipeline: Pipeline, plan: Path, state: Path, *targets: str) -> list[str]:
    clear_interrupted(plan, state)
    selected = [argument for target in targets for argument in ("--target", target)]
    return module(pipeline, "workflow", str(plan), "--state", str(state), *selected)


def clocks(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    work, log = pipeline.work, pipeline.logs / "clocks.log"
    clear_partial(work, work / "workflow.json")
    if not work.exists():
        tools = str(pipeline.reference("preparationTools"))
        edit_root = pipeline.text("editRoot")
        prepare = module(
            pipeline, "replay_preparation_plan", "prepare", edit_root, tools, str(work)
        )
        run(prepare, log)
    run(workflow(pipeline, work / "workflow.json", work / "state", *CLOCK_STAGES), log)
    source = pipeline.path_of("source")
    hashed = work / "source.sha256"
    if not hashed.exists():
        hashed.write_text(f"{sha256(source)}  {source}\n", encoding="utf-8")
    return {
        "videoClock": pin(work / "video-clock.json"),
        "audioClock": pin(work / "audio-clock.json"),
        "preparation": pin(work / "preparation.json"),
    }


def cleanup_bootstrap(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    work, log = pipeline.work, pipeline.logs / "cleanup-bootstrap.log"
    request_root, bootstrap = work / "cleanup-preparation-v1", work / "cleanup-bootstrap-v1"
    request = request_root / "BOOTSTRAP-REQUEST-v1.json"
    clear_partial(request_root, request)
    if not request.exists():
        reference = str(pipeline.reference("bootstrapRequest"))
        run(
            module(pipeline, "cleanup_bootstrap_request", str(work), pipeline.text("source"),
                   reference, str(request_root), "--native", pipeline.text("native"),
                   "--ffprobe", pipeline.text("ffprobe")),
            log,
        )  # fmt: skip
    clear_partial(bootstrap, bootstrap / "workflow.json")
    if not bootstrap.exists():
        run(module(pipeline, "replay_cleanup_plan", "prepare", str(request), str(bootstrap)), log)
    try:
        run(workflow(pipeline, bootstrap / "workflow.json", bootstrap / "state", BOOTSTRAP_TARGET),
            log)  # fmt: skip
    except ReplayError:
        if not absence.no_band(request, bootstrap):
            raise
        return {"captionFree": caption_free(pipeline, run, request)}
    return {
        "settings": pin(bootstrap / "settings.json"),
        "clock": pin(bootstrap / "physical-clock.json"),
    }


def caption_free(pipeline: Pipeline, run: Runner, request: Path) -> dict[str, str]:
    """Survey a source whose cleanup profile found no caption band; pin its absence summary."""
    root, log = pipeline.work / CAPTION_ABSENCE, pipeline.logs / "cleanup-bootstrap.log"
    survey, output = root / "request.json", root / "survey"
    clear_partial(root, survey)
    if not survey.exists():
        root.mkdir(mode=0o700)
        absence.survey_request(request, survey)
    clear_partial(output, output / "workflow.json")
    if not output.exists():
        run(module(pipeline, "replay_cleanup_plan", "prepare", str(survey), str(output)), log)
    try:
        run(workflow(pipeline, output / "workflow.json", output / "state", "profile-ocr"), log)
    except ReplayError:
        # The stage's own verifier refuses a band-free profile; any other failure stands.
        if not absence.no_band(survey, output):
            raise
    summary = root / "summary.json"
    if not summary.exists():
        write_new(summary, absence.summary(survey, output))
    return pin(summary)


def cleanup_layout(description: Path) -> tuple[Path, Path]:
    """(work, media) directories a cleanup description names."""
    loaded = record(json.loads(description.read_text(encoding="utf-8")))
    return Path(str(loaded["work"])), Path(str(loaded["mediaRoot"])) / str(loaded["recording"])


def clear_partial_chunks(work: Path, media: Path) -> None:
    """Chunks without a receipt were interrupted: their state and media are redone.

    A chunk whose receipt accepted no attempt failed its audit; its state is kept under
    ``rejected/`` as evidence and the chunk is redone with the current code.
    """
    states = work / "state"
    for chunk in sorted(states.glob("chunk-*")):
        receipt = chunk / "chunk-receipt.json"
        if not receipt.exists():
            remove(chunk)
            remove(media / "sampled" / chunk.name)
        elif record(json.loads(receipt.read_text(encoding="utf-8"))).get("accepted", "") is None:
            (work / "rejected").mkdir(mode=0o700, exist_ok=True)
            chunk.rename(work / "rejected" / f"{chunk.name}-{time.time_ns()}")
            remove(media / "sampled" / chunk.name)
    # A chunk can be cut before its state exists; its log alone blocks the redo.
    for log in sorted((work / "logs").glob("chunk-?????.log")):
        if not (states / log.stem / "chunk-receipt.json").exists():
            log.rename(log.with_name(f"{log.stem}.interrupted-{time.time_ns()}.log"))


def cleanup(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    bootstrap = record(read_receipt(pipeline, "cleanup-bootstrap").get("outputs", {}))
    if "captionFree" in bootstrap:
        # Nothing to remove: the measured source is composed unchanged.
        return {"captionFree": bootstrap["captionFree"], "cleanSource": pipeline.text("source")}
    log, description = pipeline.logs / "cleanup.log", pipeline.description
    if not description.exists():
        defaults = str(pipeline.reference("cleanupDefaults"))
        run(
            module(pipeline, "source_clean_run", str(pipeline.work), "describe", "--defaults",
                   defaults, "--revision", CLEANUP_REVISION),
            log,
        )  # fmt: skip
    work, media = cleanup_layout(description)
    run(module(pipeline, "source_clean_run", str(description), "configs"), log)  # idempotent
    clear_partial_chunks(work, media)
    run(module(pipeline, "source_clean_run", str(description), "run"), log)
    consumer = work / "consumer" / "source-cleanup.json"
    if not consumer.exists():
        remove(work / "consumer")
        remove(media / "assembly")
        run(module(pipeline, "source_clean_run", str(description), "assemble"), log)
    return {
        "description": pin(description),
        "sourceCleanup": pin(consumer),
        "cleanSource": str(media / "assembly" / "source.mov"),
        "chunkMedia": str(media / "sampled"),
    }


def frame_count(clock: Path) -> int:
    """Decoded frames in the ``ffprobe -show_frames`` video clock."""
    frames = record(json.loads(clock.read_text(encoding="utf-8"))).get("frames")
    if not isinstance(frames, list) or not frames:
        raise ReplayError("The video clock has no frames")
    return len(frames)


def yolo_track(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    output = pipeline.work / "yolo-track-v2.json"
    if not output.exists():
        frames = frame_count(pipeline.work / "video-clock.json")
        run(
            module(pipeline, "yolo_track", pipeline.text("source"), str(output),
                   "--weights", pipeline.yolo("weights"),
                   "--weights-sha256", pipeline.yolo("sha256"),
                   "--config-directory", pipeline.yolo("configDirectory"),
                   "--work", str(Path.cwd().resolve() / "edits"), "--end", str(frames),
                   "--step", pipeline.yolo("step"), "--device", YOLO_DEVICE,
                   "--decoder", YOLO_DECODER, "--ffmpeg", pipeline.text("ffmpeg"),
                   "--ffprobe", pipeline.text("ffprobe")),
            pipeline.logs / "yolo-track.log",
        )  # fmt: skip
    return {"track": pin(output)}


def rebound(reference: Path, destination: Path, roots: tuple[str, ...], **changes: str) -> Path:
    """Write the reference with its code pins taken from the current closure of ``roots``.

    ``separation_plan_refresh`` only re-hashes the reference's pinned files and refuses a
    closure whose membership changed since the reference was written; the producers compare
    the pins with exactly this closure, so a new recording binds to the code that runs it.
    """
    data = record(json.loads(reference.read_text(encoding="utf-8")))
    data["codeInputs"] = [
        {"path": str(item.path), "sha256": item.sha256} for item in capture(roots)
    ]
    remove(destination)  # a copy from an earlier attempt may pin an older closure
    write_new(destination, {**data, **changes})
    return destination


def provider_environment(namespace: Path) -> list[str]:
    """``env`` arguments the separation provider runs with: offline pinned model, local scratch."""
    checkout = Path.cwd().resolve()
    models = checkout / "edits" / "models" / "mel-roformer"
    scratch, cache = namespace / "scratch", namespace / "xdg-cache"
    for directory in (scratch, cache):
        directory.mkdir(mode=0o700, exist_ok=True)
    settings = {
        "PYTHONPATH": checkout / "src",
        "HF_HOME": models,
        "HF_HUB_CACHE": models / "hub",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": scratch,
        "TMP": scratch,
        "TEMP": scratch,
        "XDG_CACHE_HOME": cache,
    }
    return [ENV, *(f"{key}={value}" for key, value in settings.items())]


def current_pins(plan: Path) -> bool:
    """True when the plan pins exactly the current separation pipeline closure."""
    pinned = record(json.loads(plan.read_text(encoding="utf-8"))).get("codeInputs")
    current = [{"path": str(item.path), "sha256": item.sha256} for item in capture(PLAN_CLOSURE)]
    return pinned == current


def separation(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    work, log = pipeline.work, pipeline.logs / "separation.log"
    namespace = work / "separation-v1"
    plan = namespace / "PIPELINE-PLAN-v1.json"
    clear_partial(namespace, plan)
    if plan.exists() and not (namespace / "pipeline").exists() and not current_pins(plan):
        remove(namespace)  # prepare-plan can write a plan before failing its closure check
    if not namespace.exists():
        config = rebound(
            pipeline.reference("separationConfig"),
            work / "separation-config.json",
            PREPARATION_CLOSURE,
        )
        template = rebound(
            pipeline.reference("separationTemplate"),
            work / "separation-template-gpu.json",
            PLAN_CLOSURE,
            device=SEPARATION_DEVICE,
        )
        run(
            module(pipeline, "separation_plan_refresh", str(config), str(template),
                   str(namespace), "--work", str(work), "--source", pipeline.text("source")),
            log,
        )  # fmt: skip
    vocals = namespace / "pipeline" / "vocals.wav"
    if not vocals.exists():
        resume = ["--resume"] if (namespace / "pipeline").exists() else []
        command = module(pipeline, "separation_window_pipeline", str(plan), *resume, provider=True)
        run([*provider_environment(namespace), *command], log)
    return {"plan": pin(plan), "vocals": pin(vocals)}


def asr(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    work, log = pipeline.work, pipeline.logs / "asr.log"
    root = work / "asr-v1"
    config, prepared = root / "CONFIG-v1.json", root / "prepared"
    plan = work / "separation-v1" / "PIPELINE-PLAN-v1.json"
    if not config.exists():
        root.mkdir(mode=0o700, exist_ok=True)
        reference = str(pipeline.reference("asrConfig"))
        run(module(pipeline, "metal_asr_config", reference, str(plan), str(config)), log)
    if ASR_METHOD == "context-streams":
        recognized(pipeline, run, config, prepared, log)
        return silence_excluded(pipeline, run, prepared, log)
    clear_partial(prepared, prepared / "workflow.json")
    if not prepared.exists():
        prepared.mkdir(mode=0o700)
        run(
            module(pipeline, "vocal_metal_asr_pipeline", "prepare", str(config),
                   "--output", str(prepared)),
            log,
        )  # fmt: skip
    run(workflow(pipeline, prepared / "workflow.json", prepared / "state", "assemble"), log)
    return silence_excluded(pipeline, run, prepared, log)


def silence_excluded(
    pipeline: Pipeline, run: Runner, prepared: Path, log: Path
) -> dict[str, object]:
    """Recognition labels without those inside digital silence; the step's outputs."""
    labels = prepared / "labels-silence-v1.json"
    if not labels.exists():
        run(
            module(pipeline, "asr_silence_exclude", str(prepared / "labels.json"),
                   str(prepared / "mono.wav"), str(labels), *SILENCE_ARGUMENTS),
            log,
        )  # fmt: skip
    return {"labels": pin(labels), "mono": pin(prepared / "mono.wav")}


def recognized(pipeline: Pipeline, run: Runner, config: Path, prepared: Path, log: Path) -> None:
    """Context-stream recognition: mono PCM from the vocals, then labels and their manifest.

    The windowed recognizer's workflow re-binds the separation input by file identity, which a
    retired and rebuilt intermediate cannot satisfy; this path needs only the retained vocals.
    """
    labels, manifest = prepared / "labels.json", prepared / "asr" / "inputs.json"
    if labels.exists():
        return
    prepared.mkdir(mode=0o700, parents=True, exist_ok=True)
    settings = record(json.loads(config.read_text(encoding="utf-8")))
    vocals = pipeline.work / "separation-v1" / "pipeline" / "vocals.wav"
    ffmpeg = str(record(settings["ffmpeg"])["path"])
    run(module(pipeline, "asr_context_streams", "mono", str(vocals), ffmpeg,
               str(prepared / "mono.wav"), str(prepared / "mono-receipt.json")), log)  # fmt: skip
    manifest.unlink(missing_ok=True)  # an interrupted publication left no labels
    model = record(settings["model"])
    directory = Path(str(record(model["config"])["path"])).parent
    pipeline_run = prepared / "context-streams"
    arguments = [str(prepared / "mono.wav"), str(directory), str(pipeline_run), str(labels),
                 str(manifest)]  # fmt: skip
    run(module(pipeline, "asr_context_streams", "publish", *arguments), log)


def retire_intermediates(pipeline: Pipeline, _run: Runner) -> dict[str, object]:
    """Delete intermediates whose consumers passed; the receipts of those consumers are inputs."""
    cleaned = record(read_receipt(pipeline, "cleanup").get("outputs"))
    namespace = pipeline.work / "separation-v1"
    targets = (
        namespace / "pipeline" / "chunks",
        namespace / "model-input.wav",
        *((Path(str(cleaned["chunkMedia"])),) if "chunkMedia" in cleaned else ()),
    )
    removed = [
        {"path": str(target), "bytes": remove(target)}
        for target in targets
        if target.exists() and not target.is_symlink()
    ]
    return {"removed": removed, "bytes": sum(int(str(item["bytes"])) for item in removed)}


# Each step's upstream steps; their receipts are the step's inputs.
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "clocks": (),
    "separation": ("clocks",),
    "cleanup-bootstrap": ("clocks",),
    "cleanup": ("cleanup-bootstrap",),
    "asr": ("separation",),
    "yolo-track": ("clocks",),
    "retire-intermediates": ("asr", "cleanup"),
}
STEPS: dict[str, StepFunction] = {
    "clocks": clocks,
    "cleanup-bootstrap": cleanup_bootstrap,
    "cleanup": cleanup,
    "yolo-track": yolo_track,
    "separation": separation,
    "asr": asr,
    "retire-intermediates": retire_intermediates,
}


def step_outputs(pipeline: Pipeline, step: str) -> tuple[Path, ...]:
    """What a forced re-run of ``step`` removes besides its receipt."""
    work = pipeline.work
    if step == "cleanup":
        if not pipeline.description.exists():
            return ()
        clean_work, media = cleanup_layout(pipeline.description)
        return (clean_work, media, pipeline.description)
    owned = {
        "clocks": (work,),
        "cleanup-bootstrap": (
            work / "cleanup-preparation-v1",
            work / "cleanup-bootstrap-v1",
            work / CAPTION_ABSENCE,
        ),
        "yolo-track": (work / "yolo-track-v2.json",),
        "separation": (
            work / "separation-v1",
            work / "separation-config.json",
            work / "separation-template-gpu.json",
        ),
        "asr": (work / "asr-v1",),
        "retire-intermediates": (),
    }
    return owned[step]


def pipeline_configuration(pipeline: Pipeline) -> tuple[Path, ...]:
    return (pipeline.path,)


FRONT = Graph(STEPS, DEPENDENCIES, pipeline_configuration, step_outputs)


def reset(pipeline: Pipeline, step: str, graph: Graph = FRONT) -> list[dict[str, object]]:
    """Remove a step's outputs and receipt so the next run computes it again."""
    targets = (*graph.outputs(pipeline, step), pipeline.steps / f"{step}.json")
    return [{"path": str(path), "bytes": remove(path)} for path in targets if path.exists()]


def inputs(pipeline: Pipeline, step: str, graph: Graph = FRONT) -> dict[str, str]:
    """Digests of the configuration and the upstream receipts this step is computed from."""
    upstream = graph.dependencies[step]
    for name in upstream:
        if not (pipeline.steps / f"{name}.json").is_file():
            raise ReplayError(f"The {name} step has not passed; {step} waits for it")
    paths = [*graph.configurations(pipeline), *(pipeline.steps / f"{n}.json" for n in upstream)]
    return {str(path): sha256(path) for path in paths}


def finished(pipeline: Pipeline, step: str, graph: Graph = FRONT) -> bool:
    """True when the step's receipt matches its current inputs; a stale receipt is refused."""
    receipt = pipeline.steps / f"{step}.json"
    if not receipt.exists():
        return False
    recorded = record(json.loads(receipt.read_text(encoding="utf-8"))).get("inputs")
    if recorded != inputs(pipeline, step, graph):
        raise ReplayError(f"The {step} receipt predates its inputs; force the step to redo it")
    return True


def execute(step: str, path: Path, *, run: Runner, graph: Graph = FRONT) -> Path:
    """Run one step and write its receipt with its inputs and wall time."""
    pipeline = load(path)
    receipt = pipeline.steps / f"{step}.json"
    if receipt.exists():
        raise ReplayError(f"The {step} receipt exists; the pipeline reuses it")
    bound = inputs(pipeline, step, graph)
    started = time.time()
    outputs = graph.steps[step](pipeline, run)
    ended = time.time()
    write_new(
        receipt,
        {
            "version": 1,
            "step": step,
            "recording": pipeline.text("recording"),
            "inputs": bound,
            "startedAt": round(started, 1),
            "endedAt": round(ended, 1),
            "wallSeconds": round(ended - started, 1),
            "outputs": outputs,
            "qualityApproved": False,
        },
    )
    return receipt


def main(argv: Sequence[str] | None = None, graph: Graph = FRONT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=tuple(graph.steps))
    parser.add_argument("pipeline", type=Path)
    args = parser.parse_args(argv)
    producer = Producer()

    def stop(_signum: int, _frame: FrameType | None) -> None:
        producer.stop()
        raise ReplayError("Step stopped; its finished work is reused on the next run")

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        print(execute(args.step, args.pipeline, run=producer, graph=graph))
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
