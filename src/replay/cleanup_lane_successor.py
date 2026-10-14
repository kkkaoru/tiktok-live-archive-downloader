"""Plan quiescent cleanup lane successors without replacing original evidence."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from replay import code_dependency_inventory, subtitle_candidate, subtitle_observations
from replay import subtitle_cleanup_assembly as assembly
from replay import subtitle_cleanup_workflow as cleanup
from replay.caption_io import native_ocr_frames
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.sequence_journal import bounded_read, decode, encode, publish
from replay.stages import Stage, _fingerprint, execution_lock, file_digest
from replay.subtitle_cleanup_plan import clock_chunks
from replay.workflow import load_stages, parse_stages, stage_graph

MAX_CHUNKS = 512
MAX_PREFIX_FILES = 10000


@dataclass(frozen=True)
class Prepared:
    plan: Path
    workflows: tuple[Path, ...]
    states: tuple[Path, ...]


def _object(value: object) -> dict[str, object]:
    return assembly._object(value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected bounded successor list")
    return list(value)


def _strings(value: object) -> list[str]:
    items = _list(value)
    result: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError("Expected successor string list")
        result.append(item)
    return result


def _path(value: object, *, work: bool = True) -> Path:
    path = Path(assembly._name(value))
    root = Path.cwd().resolve()
    if work and not path.is_relative_to(root / "edits/work"):
        raise ValueError("Successor metadata must remain in repository work")
    if (
        not work
        and path.is_relative_to(root)
        and any(path.is_relative_to(root / name) for name in ("private", "downloads"))
    ):
        raise ValueError("Private or download inputs are not accepted")
    return path


def _read(path: Path, pins: dict[str, str]) -> dict[str, object]:
    data = bounded_read(path)
    digest = hashlib.sha256(data).hexdigest()
    if pins.get(str(path), digest) != digest:
        raise ValueError("Successor metadata changed")
    pins[str(path)] = digest
    return decode(data)


def _ref(value: object, pins: dict[str, str]) -> tuple[Path, dict[str, object]]:
    ref = _object(value)
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Expected exact metadata reference")
    path = _path(ref["path"])
    data = _read(path, pins)
    if pins[str(path)] != assembly._sha(ref["sha256"]):
        raise ValueError("Referenced metadata changed")
    return path, data


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(bounded_read(path)).hexdigest()}


def _absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"Occupied successor output; preserve and reconcile: {path}")


@contextmanager
def _unlocked(path: Path) -> Iterator[None]:
    """Inspect an existing cooperative lock without creating/replacing it."""
    if not path.exists() and not path.is_symlink():
        yield
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Invalid canonical worker lock")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Original worker or collector is active") from error
        yield
    finally:
        os.close(descriptor)


def _quiescent(handoff: dict[str, object], states: tuple[Path, ...], *, fresh: bool) -> None:
    if (
        set(handoff)
        != {
            "version",
            "observedAtNs",
            "maximumAgeSeconds",
            "quiesced",
            "oldWorkersDisabled",
            "actors",
            "oldStates",
            "evidence",
        }
        or type(handoff.get("version")) is not int
        or handoff.get("version") != 1
        or handoff.get("quiesced") is not True
        or (
            handoff.get("oldWorkersDisabled") is not True
            or handoff.get("oldStates") != [str(path) for path in states]
        )
    ):
        raise ValueError("Explicit matching Root worker handoff required")
    observed, age = handoff["observedAtNs"], handoff["maximumAgeSeconds"]
    if type(observed) is not int or type(age) is not int or not 1 <= age <= 3600:
        raise ValueError("Invalid handoff observation clock")
    if fresh and not 0 <= time.time_ns() - observed <= age * 1000000000:
        raise ValueError("Handoff observation expired")
    actors = handoff["actors"]
    if not isinstance(actors, list) or not 1 <= len(actors) <= 64:
        raise ValueError("Bounded exact original actors required")
    seen: set[int] = set()
    for raw in actors:
        actor = _object(raw)
        pid = actor.get("pid")
        if (
            set(actor) != {"pid", "birth"}
            or type(pid) is not int
            or not 1 <= pid <= 2147483647
            or pid in seen
            or not isinstance(actor.get("birth"), str)
            or not actor["birth"]
        ):
            raise ValueError("Invalid or duplicate original actor identity")
        seen.add(pid)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("Original actor PID is still present; no takeover")
    evidence = handoff["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 64:
        raise ValueError("Handoff requires retained owner evidence")
    for ref in evidence:
        _ref(ref, {})
    with ExitStack() as stack:
        for state in states:
            stack.enter_context(_unlocked(state / ".workflow.lock"))


def _prefix(
    directory: Path, settings: dict[str, object], pins: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Validate only published batches; retain failed attempts as unconsumed evidence."""
    ocr = directory / "ocr"
    if not ocr.exists():
        return [], []
    _path(str(ocr))
    with _unlocked(ocr / "collector.lock"):
        entries = sorted(ocr.iterdir())
        if len(entries) > MAX_PREFIX_FILES:
            raise ValueError("Excessive OCR prefix inventory")
        times = subtitle_observations.observation_times(
            json.loads(bounded_read(directory / "times.json"))
        )
        inputs = ocr / "inputs.json"
        if not inputs.is_file() or inputs.is_symlink():
            raise ValueError("Partial OCR lacks its original input manifest")
        producer = _read(inputs, pins)
        source, native = _object(settings["source"]), _object(settings["native"])
        expected = {
            "source": source["path"],
            "sourceSHA256": source["sha256"],
            "planSHA256": _pin(directory / "times.json")["sha256"],
            "collectorSHA256": _pin(Path(subtitle_observations.__file__).resolve())["sha256"],
            "parserSHA256": _pin(
                Path(subtitle_observations.__file__).with_name("caption_io.py").resolve()
            )["sha256"],
            "headParserSHA256": None,
            "measurementTool": "video_text_recognize",
            "fullFrame": True,
            "backend": "native-cli",
            "nativeExecutable": native["path"],
            "nativeSHA256": native["sha256"],
        }
        if producer != expected:
            raise ValueError("Original OCR source/clock/native/collector inputs changed")
        prefix: list[str] = [str(inputs)]
        attempts: list[str] = []
        count = 0
        for entry in entries:
            if entry.name in {"inputs.json", "complete.json", "collector.lock"}:
                continue
            if re.fullmatch(r"\d{4}\.result\.json", entry.name):
                if entry.name != f"{count:04d}.result.json":
                    raise ValueError("OCR prefix gap or extra published result")
                batch = times[count * 8 : (count + 1) * 8]
                if not batch:
                    raise ValueError("OCR prefix exceeds physical clock")
                subtitle_observations._validate_result(entry, batch, native=True)
                if (
                    tuple(
                        frame.actual_time
                        for frame in native_ocr_frames(json.loads(bounded_read(entry)))
                    )
                    != batch
                ):
                    raise ValueError("Published OCR prefix actual PTS differ")
                pins[str(entry)] = _pin(entry)["sha256"]
                prefix.append(str(entry))
                count += 1
            elif entry.name.startswith("attempt-") and entry.is_dir() and not entry.is_symlink():
                attempts.append(str(entry))
                files = sorted(entry.iterdir())
                if not files or any(
                    item.name not in {"request.json", "response.json", "stderr.log"}
                    for item in files
                ):
                    raise ValueError(
                        "Unsupported native attempt inventory; retain for owner review"
                    )
                for item in files:
                    _path(str(item))
                    pins[str(item)] = _pin(item)["sha256"]
            else:
                raise ValueError("Unsupported or linked OCR prefix; retain for owner review")
        if (ocr / "complete.json").exists():
            _read(ocr / "complete.json", pins)
            cleanup.check_ocr(directory / "chunk.json")
        return prefix, attempts


def _historical_fingerprint(stage: Stage, inputs: dict[str, str]) -> str:
    """Recompute a Stage fingerprint with caller-supplied input digests.

    This intentionally mirrors stages._fingerprint (verified equivalent by
    test_stage_fingerprint_matches_shared_runner) so pre-recovery receipts can
    be checked without editing the shared runner: stages.py itself is never
    given a compat override, keeping its closure byte-identical.
    """
    definition = {
        "command": stage.command,
        "verifyCommand": stage.verify_command,
        "inputs": inputs,
        "outputs": [str(path) for path in stage.outputs],
        "timeoutSeconds": stage.timeout_seconds,
        "workingDirectory": str(Path.cwd().resolve()),
    }
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode("utf-8")).hexdigest()


def _complete(
    stage: Stage, state: Path, *, historical_input_sets: list[dict[str, str]] | None = None
) -> None:
    receipt = decode(bounded_read(state / f"{stage.name}.json"))
    outputs = {str(path): file_digest(path) for path in stage.outputs}
    matches = receipt == {
        "status": "complete",
        "fingerprint": _fingerprint(stage),
        "outputs": outputs,
    }
    for historical in historical_input_sets or ():
        matches = matches or receipt == {
            "status": "complete",
            "fingerprint": _historical_fingerprint(stage, historical),
            "outputs": outputs,
        }
    if not matches or not stage.verify_command:
        raise ValueError("Genuine unchanged complete Stage evidence required")


def _code() -> dict[str, str]:
    return {
        str(item.path): item.sha256
        for item in code_dependency_inventory.capture(
            (
                "cleanup_lane_successor",
                "code_dependency_inventory",
            )
        )
    }


def _command(operation: str, plan: Path, index: int) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "replay.cleanup_lane_successor",
        operation,
        str(plan),
        "--chunk",
        str(index),
    ]


def prepare(request: Path, output: Path) -> Prepared:
    """Metadata-only selection after explicit quiescence; never infer completion."""
    _path(str(request))
    _path(str(output))
    if not output.is_dir():
        raise ValueError("Existing empty successor purpose directory required")
    pins: dict[str, str] = {}
    raw = _read(request, pins)
    if (
        set(raw)
        not in (
            {"version", "originalPlan", "oldWorkers", "handoff", "workers"},
            {"version", "originalPlan", "oldWorkers", "handoff", "workers", "previousGeneration"},
        )
        or type(raw.get("version")) is not int
        or raw.get("version") != 1
    ):
        raise ValueError("Invalid successor request schema")
    workers = raw["workers"]
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Successor lane budget must be 1..8")
    original, original_data = _ref(raw["originalPlan"], pins)
    settings_path, settings = _ref(original_data.get("settings"), pins)
    _, source_clock = _ref(settings.get("sourceClock"), pins)
    clocks = clock_chunks(
        source_clock, maximum_frames=assembly._positive(settings.get("chunkFrames"))
    )
    frames = sum(len(clock.pts) for clock in clocks)
    if (
        type(original_data.get("version")) is not int
        or original_data.get("version") != 1
        or not 1 <= len(clocks) <= MAX_CHUNKS
        or clocks[0].pts[0] != 0
        or original_data.get("wholeSourcePlanned") is not True
        or original_data.get("chunks") != len(clocks)
        or original_data.get("plannedFrames") != frames
        or original_data.get("totalSourceFrames") != frames
    ):
        raise ValueError("Original plan must cover every physical PTS exactly once")
    old = raw["oldWorkers"]
    if not isinstance(old, list) or not 1 <= len(old) <= 8:
        raise ValueError("Explicit original worker plans/states required")
    stages: dict[str, tuple[Stage, Path]] = {}
    states: list[Path] = []
    for item in old:
        row = _object(item)
        if set(row) != {"workflow", "state"}:
            raise ValueError("Invalid original worker binding")
        workflow, _ = _ref(row["workflow"], pins)
        state = _path(row["state"])
        if state in states or not state.is_dir():
            raise ValueError("Duplicate or missing original state")
        states.append(state)
        for stage in load_stages(workflow):
            if stage.name in stages:
                raise ValueError("Duplicate original chunk producer")
            stages[stage.name] = stage, state
    previous_rows: dict[int, dict[str, object]] = {}
    previous_generation = raw.get("previousGeneration")
    if previous_generation is not None:
        prior = _object(previous_generation)
        if set(prior) != {"plan", "workers", "evidence"}:
            raise ValueError("Invalid previous successor generation")
        previous_path, previous_data = _ref(prior["plan"], pins)
        if (
            previous_data.get("version") != 1
            or previous_data.get("originalPlan") != str(original)
            or previous_data.get("settings") != str(settings_path)
        ):
            raise ValueError("Previous successor original source plan differs")
        previous_inputs = _object(previous_data.get("inputs"))
        previous_request_path = _path(previous_data["request"])
        _, previous_request = _ref(
            {
                "path": str(previous_request_path),
                "sha256": previous_inputs.get(str(previous_request_path)),
            },
            pins,
        )
        if previous_request.get("originalPlan") != raw["originalPlan"] or (
            previous_request.get("oldWorkers") != raw["oldWorkers"]
        ):
            raise ValueError("Previous successor ownership differs")
        evidence = prior["evidence"]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 64:
            raise ValueError("Previous generation requires retained pre-fix evidence")
        for reference in evidence:
            _ref(reference, pins)
        declared = prior["workers"]
        if not isinstance(declared, list) or not 1 <= len(declared) <= 8:
            raise ValueError("Explicit previous worker workflows/states required")
        previous_workflows: list[str] = []
        previous_states: list[str] = []
        for binding in declared:
            item = _object(binding)
            if set(item) != {"workflow", "state"}:
                raise ValueError("Invalid previous worker binding")
            workflow, _ = _ref(item["workflow"], pins)
            state = _path(item["state"])
            if state in states or not state.is_dir():
                raise ValueError("Duplicate or missing previous state")
            states.append(state)
            previous_workflows.append(str(workflow))
            previous_states.append(str(state))
        if previous_workflows != previous_data.get("workflows") or (
            previous_states != previous_data.get("states")
        ):
            raise ValueError("Previous workflows/states differ from their immutable plan")
        for old_row in _list(previous_data.get("chunks")):
            row = _object(old_row)
            index = row.get("index")
            if type(index) is not int or index in previous_rows:
                raise ValueError("Previous generation has invalid duplicate chunks")
            previous_rows[index] = row
        if set(previous_rows) != set(range(len(clocks))):
            raise ValueError("Previous generation does not cover the original physical clock")
        # Pin each immutable previous config, not its historical reader's current
        # source bytes. No old receipt is rewritten or promoted to current code.
        old_inputs = _object(previous_data.get("inputs"))
        for row in previous_rows.values():
            name = str(row.get("chunk"))
            if name not in old_inputs or _pin(_path(name))["sha256"] != old_inputs[name]:
                raise ValueError("Previous selected chunk bytes changed")
        pins[str(previous_path)] = _pin(previous_path)["sha256"]
    handoff_path, handoff = _ref(raw["handoff"], pins)
    _quiescent(handoff, tuple(states), fresh=True)
    rows: list[dict[str, object]] = []
    selected: list[tuple[Path, dict[str, object]]] = []
    expected_names: set[str] = set()
    for index, clock in enumerate(clocks):
        directory = original.parent / f"chunk-{index:05d}"
        chunk = directory / "chunk.json"
        data = _read(chunk, pins)
        if (
            data.get("clock") != assembly._clock_data(clock)
            or data.get("settings")
            != {"path": str(settings_path), "sha256": pins[str(settings_path)]}
            or data.get("ocrDirectory") != str(directory / "ocr")
        ):
            raise ValueError("Original chunk clock/settings/canonical output changed")
        times = directory / "times.json"
        actual_times = subtitle_observations.observation_times(json.loads(bounded_read(times)))
        if actual_times != tuple(float(pts * clock.time_base) for pts in clock.pts):
            raise ValueError("Original chunk times differ")
        pins[str(times)] = _pin(times)["sha256"]
        original_stages = [
            stages.get(directory.name + "-" + operation)
            for operation in ("ocr", "admission", "repair")
        ]
        if any(stage is None for stage in original_stages):
            raise ValueError("Missing original chunk stage")
        old_rows: list[dict[str, object]] = []
        complete = True
        for pair in original_stages:
            if pair is None:
                raise ValueError("Missing original stage")
            stage, state = pair
            expected_names.add(stage.name)
            base = cleanup._stages(chunk, settings, None)[len(old_rows)]
            expected = parse_stages({"version": 1, "stages": [base]})[0]
            if (
                stage.command != expected.command
                or stage.outputs != expected.outputs
                or stage.verify_command != expected.verify_command
                or stage.timeout_seconds != expected.timeout_seconds
                or stage.adopt_existing
                or (
                    stage.inputs != expected.inputs
                    and not (
                        len(old_rows) == 0
                        and stage.inputs[:-1] == expected.inputs
                        and stage.inputs[-1]
                        in {
                            original.parent / f"chunk-{j:05d}/repair/candidate.json"
                            for j in range(index)
                        }
                    )
                )
            ):
                raise ValueError("Original stage does not match canonical cleanup producer")
            with _unlocked(state / f"{stage.name}.lock"):
                receipt_path = state / f"{stage.name}.json"
                receipt = _read(receipt_path, pins) if receipt_path.exists() else None
            if receipt is not None and (
                receipt.get("status") not in {"running", "complete"}
                or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("fingerprint"))) is None
            ):
                raise ValueError("Unsupported original Stage receipt")
            if receipt is not None and receipt.get("status") == "running" and len(old_rows) > 0:
                raise ValueError("Partial admission/repair Stage requires explicit reconciliation")
            complete = complete and receipt is not None and receipt.get("status") == "complete"
            old_rows.append({"state": str(state), "name": stage.name})
        prefix, attempts = _prefix(directory, settings, pins)
        if complete:
            mode, new_chunk = "complete-original", chunk
        else:
            if (directory / "ocr/complete.json").exists():
                raise ValueError(
                    "OCR-only completion requires explicit original Stage reconciliation"
                )
            if any(
                (directory / name).exists() or (directory / name).is_symlink()
                for name in (
                    "ocr-batches.json",
                    "admission.json",
                    "eligibility.json",
                    "candidate-config.json",
                    "repair",
                    "scratch",
                )
            ):
                raise ValueError(
                    f"Unsupported partial admission/repair: {directory}; preserve for owner review"
                )
            mode = "resume-ocr" if prefix else "pending"
            new_chunk = directory / "chunk-successor-v1.json"
            successor = dict(data)
            successor["maximumActiveChunks"] = min(workers, len(clocks))
            if new_chunk.exists() or new_chunk.is_symlink():
                prior_row = previous_rows.get(index)
                if (
                    prior_row is None
                    or prior_row.get("chunk") != str(new_chunk)
                    or (
                        _read(new_chunk, pins) != successor
                        or prior_row.get("original") != str(chunk)
                        or prior_row.get("oldStages") != old_rows
                    )
                ):
                    raise ValueError("Occupied successor config lacks matching prior generation")
            else:
                if previous_rows and previous_rows[index].get("chunk") == str(new_chunk):
                    raise ValueError("Previous selected successor config is missing")
                selected.append((new_chunk, successor))
        rows.append(
            {
                "index": index,
                "mode": mode,
                "original": str(chunk),
                "chunk": str(new_chunk),
                "oldStages": old_rows,
                "prefix": prefix,
                "unpublishedAttemptsRetainedNotConsumed": attempts,
            }
        )
    if set(stages) != expected_names:
        raise ValueError("Original workers contain extra or omitted chunks")
    plan = output / "successor.json"
    workflows = tuple(output / f"worker-{i:02d}.json" for i in range(min(workers, len(rows))))
    new_states = tuple(output / f"state-{i:02d}" for i in range(len(workflows)))
    for path in (plan, *workflows, *new_states, output / "generation.json"):
        _absent(path)
    code = _code()
    for path, value in selected:
        publish(path, encode(value))
        pins[str(path)] = _pin(path)["sha256"]
    lane_stages: list[list[dict[str, object]]] = [[] for _ in workflows]
    previous: list[Path | None] = [None for _ in workflows]
    for index, row in enumerate(rows):
        lane = index % len(workflows)
        chunk = Path(str(row["chunk"]))
        common = [str(plan), str(Path(__file__).resolve()), str(chunk)]
        if row["mode"] == "complete-original":
            result = output / f"reused-{index:05d}.json"
            definitions = [
                {
                    "name": chunk.parent.name + "-retained",
                    "command": _command("reuse", plan, index),
                    "inputs": common + [str(chunk.parent / "repair/candidate.json")],
                    "outputs": [str(result)],
                    "verify_command": _command("verify-reuse", plan, index),
                    "timeout_seconds": 3600,
                }
            ]
            if previous[lane] is not None:
                definitions[0]["inputs"] = [
                    *common,
                    str(chunk.parent / "repair/candidate.json"),
                    str(previous[lane]),
                ]
        else:
            definitions = cleanup._stages(chunk, settings, previous[lane])
            for definition, operation in zip(
                definitions, ("ocr", "admission", "repair"), strict=True
            ):
                definition["command"] = _command(operation, plan, index)
                definition["verify_command"] = _command("verify-" + operation, plan, index)
                definition["inputs"] = list(
                    dict.fromkeys([*_strings(definition["inputs"]), *common])
                )
            if row["prefix"]:
                definitions[0]["outputs"] = [str(chunk.parent / "ocr/complete.json")]
                definitions[0]["inputs"] = list(
                    dict.fromkeys([*_strings(definitions[0]["inputs"]), *_strings(row["prefix"])])
                )
            result = chunk.parent / "repair/candidate.json"
        row["successorStages"] = [item["name"] for item in definitions]
        row["lane"] = lane
        previous[lane] = result
        lane_stages[lane].extend(definitions)
    for items in lane_stages:
        stage_graph(parse_stages({"version": 1, "stages": items}))
    for name, digest in {**pins, **code}.items():
        if _pin(Path(name))["sha256"] != digest:
            raise ValueError("Input changed before successor publication")
    _quiescent(handoff, tuple(states), fresh=True)
    publish(
        plan,
        encode(
            {
                "version": 1,
                "request": str(request),
                "originalPlan": str(original),
                "handoff": str(handoff_path),
                "settings": str(settings_path),
                "oldStates": [str(path) for path in states],
                "previousGeneration": previous_generation,
                "workflows": [str(path) for path in workflows],
                "states": [str(path) for path in new_states],
                "workers": len(workflows),
                "chunks": rows,
                "inputs": pins,
                "readerCode": code,
                "qualityApproved": False,
                "deliveryComplete": False,
            }
        ),
    )
    for path, items in zip(workflows, lane_stages, strict=True):
        publish(path, encode({"version": 1, "stages": items}))
    return Prepared(plan, workflows, new_states)


def _selected_inputs(data: dict[str, object], index: int) -> set[str]:
    chunks = _list(data.get("chunks"))
    if type(index) is not int or not 0 <= index < len(chunks):
        raise ValueError("Unknown successor chunk")
    row = _object(chunks[index])
    directory = _path(row["original"]).parent
    selected = {
        str(_path(data[field])) for field in ("request", "originalPlan", "handoff", "settings")
    }
    request = _read(_path(data["request"]), {})
    for worker in _list(request["oldWorkers"]):
        selected.add(str(_path(_object(_object(worker)["workflow"])["path"])))
    for stage in _list(row["oldStages"]):
        original = _object(stage)
        selected.add(str(_path(original["state"]) / (str(original["name"]) + ".json")))
    handoff = _read(_path(data["handoff"]), {})
    for reference in _list(handoff["evidence"]):
        selected.add(str(_path(_object(reference)["path"])))
    settings = _read(_path(data["settings"]), {})
    shared_directories: list[Path] = []
    for field in (
        "native",
        "decoder",
        "model",
        "sourceClock",
        "classificationEvidence",
        "profileInputs",
        "profileComplete",
    ):
        shared_path = _path(_object(settings[field])["path"], work=False)
        selected.add(str(shared_path))
        if field in {"profileInputs", "profileComplete"}:
            shared_directories.append(shared_path.parent)
    for name in _object(data["inputs"]):
        path = _path(name)
        if path.is_relative_to(directory) or any(
            path.is_relative_to(parent) for parent in shared_directories
        ):
            selected.add(name)
    return selected


def _load(plan: Path, index: int | None = None) -> dict[str, object]:
    data = _read(_path(str(plan)), {})
    if data.get("version") != 1 or _object(data.get("readerCode")) != _code():
        raise ValueError("Successor reader code changed")
    selected = None if index is None else _selected_inputs(data, index)
    for name, digest in _object(data.get("inputs")).items():
        if selected is not None and name not in selected:
            continue
        if _pin(_path(name))["sha256"] != assembly._sha(digest):
            raise ValueError("Successor input changed")
    states = tuple(_path(name) for name in _strings(data["oldStates"]))
    _quiescent(_read(_path(data["handoff"]), {}), states, fresh=False)
    return data


@contextmanager
def _owned(plan: Path, index: int) -> Iterator[tuple[dict[str, object], dict[str, object]]]:
    data = _load(plan, index)
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or type(index) is not int or not 0 <= index < len(chunks):
        raise ValueError("Unknown successor chunk")
    row = _object(chunks[index])
    # A phase consumes only one selected chunk and its shared authorities.
    # Entry/end _load independently hash that exact closure; prepare/assemble
    # retain the complete all-chunk check. Generated files remain ordinary.
    settings = cleanup._settings(_path(data["settings"]))
    paths = [
        plan,
        _path(data["settings"]),
        _path(row["chunk"]),
        _path(row["original"]),
        _path(row["original"]).parent / "times.json",
    ]
    paths.extend(_path(name, work=False) for name in _object(data["readerCode"]))
    paths.extend(
        _path(_object(settings[field])["path"], work=False)
        for field in (
            "source",
            "native",
            "decoder",
            "model",
            "sourceClock",
            "classificationEvidence",
            "profileInputs",
            "profileComplete",
        )
    )
    paths.extend(_path(name) for name in _strings(row["prefix"]))
    if row["mode"] == "complete-original":
        for old_stage, _ in _old_stages(data, row):
            paths.extend(old_stage.inputs + old_stage.outputs)
    cache = InputHashCache(tuple(dict.fromkeys(paths)), allow_uncached_overflow=True)
    cache.prime()
    with input_hash_context(cache), ExitStack() as stack:
        for state in (_path(name) for name in _strings(data["oldStates"])):
            # Keep shared probes through the whole operation. Original writers
            # require LOCK_EX; other successor readers may safely coexist.
            stack.enter_context(_unlocked(state / ".workflow.lock"))
        old_stages = row["oldStages"]
        if not isinstance(old_stages, list):
            raise ValueError("Missing canonical stage leases")
        for item in old_stages:
            stage = _object(item)
            if not stack.enter_context(execution_lock(_path(stage["state"]), str(stage["name"]))):
                raise ValueError("Original chunk lease was reacquired")
        cache.guard()
        states = tuple(_path(name) for name in _strings(data["oldStates"]))
        _quiescent(_read(_path(data["handoff"]), {}), states, fresh=False)
        yield data, row
        cache.verify()
    # Independent fresh bytes of the same actually consumed metadata scope.
    _load(plan, index)


def _old_stages(data: dict[str, object], row: dict[str, object]) -> tuple[tuple[Stage, Path], ...]:
    wanted = {
        str(_object(item)["name"]): _path(_object(item)["state"])
        for item in _list(row["oldStages"])
    }
    request = _read(_path(data["request"]), {})
    old_workers = request["oldWorkers"]
    if not isinstance(old_workers, list):
        raise ValueError("Missing old worker definitions")
    found: list[tuple[Stage, Path]] = []
    for item in old_workers:
        workflow, _ = _ref(_object(item)["workflow"], {})
        for stage in load_stages(workflow):
            if stage.name in wanted:
                found.append((stage, wanted[stage.name]))
    return tuple(found)


def _retained(data: dict[str, object], row: dict[str, object]) -> None:
    for stage, state in _old_stages(data, row):
        _complete(stage, state)
    chunk = _path(row["original"])
    cleanup.check_ocr(chunk)
    # The original admission Stage's own verifier checks same-byte existing
    # artifacts. _complete above requires all four outputs before this call.
    cleanup.build_candidate(chunk)
    cleanup.check_repair(chunk)


def phase(plan: Path, index: int, operation: str) -> None:
    """Finite existing-domain adapter; never submit work outside an ordinary Stage."""
    with _owned(plan, index) as (data, row):
        chunk = _path(row["chunk"])
        settings = cleanup._settings(_path(data["settings"]))
        source = _object(settings["source"])
        if file_digest(_path(source["path"], work=False)) != source["sha256"]:
            raise ValueError("Source changed before cleanup successor")
        if operation in {"reuse", "verify-reuse"}:
            if row["mode"] != "complete-original":
                raise ValueError("Original completion is absent")
            _retained(data, row)
            result = plan.parent / f"reused-{index:05d}.json"
            value = {
                "chunk": row["original"],
                "originalCompleteVerified": True,
                "qualityApproved": False,
            }
            if operation == "reuse":
                publish(result, encode(value))
            elif _read(result, {}) != value:
                raise ValueError("Retained verification result differs")
        elif operation == "ocr":
            for old_stage, state in _old_stages(data, row):
                receipt = state / (old_stage.name + ".json")
                if old_stage.name.endswith("-ocr") and receipt.exists():
                    saved = _read(receipt, {})
                    if saved != {"status": "running", "fingerprint": _fingerprint(old_stage)}:
                        raise ValueError("Partial original OCR Stage input binding changed")
            usage = shutil.disk_usage(chunk.parent)
            seeds, _ = cleanup.collector_frames(
                _path(_object(settings["profileInputs"])["path"]),
                _path(_object(settings["profileComplete"])["path"]),
                settings,
            )
            _, _, clock = cleanup._chunk(chunk)
            reserve = (
                len(clock.pts) * seeds[0].width * seeds[0].height * 3 * int(str(data["workers"]))
            )
            if usage.free < max(30 * 1024**3, usage.total // 20) + reserve:
                raise ValueError("Insufficient successor capacity for all active lanes")
            subtitle_observations.collect_observations(
                _path(source["path"], work=False),
                chunk.parent / "times.json",
                chunk.parent / "ocr",
                native_executable=_path(_object(settings["native"])["path"], work=False),
            )
        elif operation == "admission":
            cleanup.build_candidate(chunk)
        elif operation == "repair":
            subtitle_candidate.render_candidate(chunk.parent / "candidate-config.json")
        elif operation in {"verify-ocr", "verify-admission", "verify-repair"}:
            {
                "verify-ocr": cleanup.check_ocr,
                "verify-admission": cleanup.build_candidate,
                "verify-repair": cleanup.check_repair,
            }[operation](chunk)
        else:
            raise ValueError("Unknown successor phase")
        if file_digest(_path(source["path"], work=False)) != source["sha256"]:
            raise ValueError("Source changed during cleanup successor")
        cleanup._settings(_path(data["settings"]))


def assemble(plan: Path, output: Path) -> Path:
    """Require genuine complete successor stages, then delegate the full-clock index."""
    data = _load(plan)
    workflows, states = data["workflows"], data["states"]
    if not isinstance(workflows, list) or not isinstance(states, list):
        raise ValueError("Missing successor workflows/states")
    for workflow, state in zip(workflows, states, strict=True):
        with _unlocked(_path(str(state)) / ".workflow.lock"):
            for stage in load_stages(_path(str(workflow))):
                _complete(stage, _path(str(state)))
    manifest = assembly.prepare(_path(data["originalPlan"]), _path(str(output)))
    _load(plan)
    publish(
        output / "successor-generation.json",
        encode(
            {
                "version": 1,
                "successor": _pin(plan),
                "manifest": _pin(manifest),
                "maximumActiveChunks": data["workers"],
                "chunks": data["chunks"],
                "allPhysicalPTSAccountedByCanonicalAssembly": True,
                "qualityApproved": False,
                "mediaBytesVerified": False,
            }
        ),
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "prepare",
            "assemble",
            "ocr",
            "admission",
            "repair",
            "verify-ocr",
            "verify-admission",
            "verify-repair",
            "reuse",
            "verify-reuse",
        ),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk", type=int)
    args = parser.parse_args(argv)
    if args.operation in {"prepare", "assemble"}:
        if args.output is None:
            parser.error("output required")
        {"prepare": prepare, "assemble": assemble}[args.operation](args.input, args.output)
    else:
        if args.chunk is None:
            parser.error("chunk required")
        phase(args.input, args.chunk, args.operation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
