"""Exclusive cleanup recovery generations with original evidence and new Stage lineage."""

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from replay import cleanup_lane_successor as old
from replay import (
    code_dependency_inventory,
    code_generation_compat,
    delivery_source_assembly,
    subtitle_candidate,
    subtitle_observations,
)
from replay import subtitle_admission_verification as admission_reader
from replay import subtitle_cleanup_assembly as assembly
from replay import subtitle_cleanup_workflow as cleanup
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.sequence_journal import bounded_read, encode, publish
from replay.stages import Stage, execution_lock, file_digest
from replay.subtitle_cleanup_plan import clock_chunks
from replay.workflow import MAX_PLAN_BYTES, load_stages, parse_stages, stage_graph


def _code() -> dict[str, str]:
    return {
        str(row.path): row.sha256
        for row in code_dependency_inventory.capture(("cleanup_generation_recovery", "workflow"))
    }


# Reviewed original recovery producer. This pin permits retained evidence only;
# it never authorizes pending work or changes any original receipt.
RETAINED_RECOVERY_SHA = "477a786df8baf1d267a44f5f424a1a375b78fb6224050935bc981d79843a4198"


def _retained_code(recorded: object, current: dict[str, str]) -> None:
    pins = old._object(recorded)
    normalized = dict(pins)
    for name, digest in pins.items():
        if Path(name).name == "cleanup_generation_recovery.py" and digest == RETAINED_RECOVERY_SHA:
            normalized[name] = current.get(name)
    if not code_generation_compat.historical_generation(normalized, current):
        raise ValueError("Unreviewed retained reader code")
    code_generation_compat.verify_recorded_pins(normalized, current)


def _retained_digest(name: str, recorded: object, current: str) -> bool:
    if Path(name).parent != Path(__file__).resolve().parent:
        return False
    return (
        Path(name).name == "cleanup_generation_recovery.py" and recorded == RETAINED_RECOVERY_SHA
    ) or code_generation_compat.accepted_input_digest(name, recorded, current)


def retained_complete(stage: Stage, state: Path, recorded: object) -> None:
    """Check immutable receipts against reviewed code and unchanged data/output pins."""
    code = old._object(recorded)
    inputs: dict[str, str] = {}
    for path in stage.inputs:
        name, current = str(path), file_digest(path)
        digest = code.get(name, current)
        if digest != current and not _retained_digest(name, digest, current):
            raise ValueError("Unreviewed retained Stage input")
        inputs[name] = assembly._sha(digest)
    old._complete(stage, state, historical_input_sets=[inputs])


def _retained_successor(plan: Path, index: int) -> dict[str, object]:
    data = old._read(plan, {})
    if data.get("version") != 1:
        raise ValueError("Unsupported retained successor")
    _retained_code(data.get("readerCode"), old._code())
    selected = old._selected_inputs(data, index)
    for name, digest in old._object(data["inputs"]).items():
        if name not in selected:
            continue
        current = old._pin(old._path(name, work=False))["sha256"]
        if current != assembly._sha(digest) and not _retained_digest(name, digest, current):
            raise ValueError("Retained successor input changed")
    states = tuple(old._path(name) for name in old._strings(data["oldStates"]))
    old._quiescent(old._read(old._path(data["handoff"]), {}), states, fresh=False)
    return data


@contextmanager
def _retained_successor_owned(plan: Path, index: int) -> Iterator[None]:
    data = _retained_successor(plan, index)
    row = old._object(old._list(data["chunks"])[index])
    settings = cleanup._settings(old._path(data["settings"]))
    paths = [plan, old._path(row["chunk"]), old._path(row["original"])]
    paths.extend(
        path
        for name in old._selected_inputs(data, index)
        for path in [old._path(name, work=False)]
        if path.stat().st_size > 0
    )
    paths.extend(old._path(name, work=False) for name in old._object(data["readerCode"]))
    paths.extend(
        old._path(old._object(settings[field])["path"], work=False)
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
    paths.extend(old._path(name) for name in old._strings(row["prefix"]))
    paths.append(old._path(row["original"]).parent / "times.json")
    if row["mode"] == "complete-original":
        for stage, _ in old._old_stages(data, row):
            paths.extend(stage.inputs + stage.outputs)
    cache = InputHashCache(tuple(dict.fromkeys(paths)), allow_uncached_overflow=True)
    cache.prime()
    with input_hash_context(cache), ExitStack() as stack:
        for state in old._strings(data["oldStates"]):
            stack.enter_context(old._unlocked(old._path(state) / ".workflow.lock"))
        for item in old._list(row["oldStages"]):
            stage_row = old._object(item)
            if not stack.enter_context(
                execution_lock(old._path(stage_row["state"]), str(stage_row["name"]))
            ):
                raise ValueError("Original chunk lease was reacquired")
        cache.guard()
        _retained_successor(plan, index)
        yield
        cache.verify()
    if _retained_successor(plan, index) != data:
        raise ValueError("Retained successor changed")


def _graph_segments(items: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    segments: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for item in items:
        if len(encode({"version": 1, "stages": [item]})) > MAX_PLAN_BYTES:
            raise ValueError("Single recovery stage exceeds workflow byte bound")
        if len(encode({"version": 1, "stages": [*current, item]})) > MAX_PLAN_BYTES:
            segments.append(current)
            current = []
        current.append(item)
    if current:
        segments.append(current)
    return segments


def _terminal_ref(value: object) -> tuple[Path, bytes]:
    ref = old._object(value)
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Expected exact terminal reference")
    path = old._path(ref["path"])
    content = bounded_read(path)
    if (
        hashlib.sha256(content).hexdigest() != assembly._sha(ref["sha256"])
        or old._pin(path)["sha256"] != ref["sha256"]
    ):
        raise ValueError("Terminal reference changed")
    return path, content


def _process_absence(paths: tuple[Path, ...]) -> None:
    pattern = "|".join(re.escape(str(path)) for path in paths)
    if not paths or len(pattern.encode()) > 128 * 1024:
        raise ValueError("Invalid bounded process scope")
    result = subprocess.run(
        ["/usr/bin/pgrep", "-f", pattern],
        capture_output=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 1 or result.stdout or result.stderr:
        raise ValueError("Previous generation process absence is not established")


def _terminal_observation(request: Path) -> dict[str, object]:
    pins: dict[str, str] = {}
    raw = old._read(request, pins)
    if (
        set(raw)
        != {"version", "kind", "successor", "oldWorkersDisabled", "maximumAgeSeconds", "terminals"}
        or raw.get("version") != 1
        or raw.get("kind") != "terminal-generation-observation-request"
        or raw.get("oldWorkersDisabled") is not True
    ):
        raise ValueError("Explicit terminal generation owner request required")
    age = raw["maximumAgeSeconds"]
    if type(age) is not int or not 1 <= age <= 3600:
        raise ValueError("Invalid owner observation freshness")
    _, legacy = old._ref(raw["successor"], pins)
    workflows, states = old._strings(legacy["workflows"]), old._strings(legacy["states"])
    terminals = old._list(raw["terminals"])
    if not 1 <= len(states) <= 8 or len(terminals) != len(states) or len(workflows) != len(states):
        raise ValueError("Incomplete terminal generation coverage")
    paths: list[Path] = [request, old._path(old._object(raw["successor"])["path"])]
    scope = [old._path(name) for name in workflows + states]
    scopes = {old._path(old._object(row)["chunk"]).parent for row in old._list(legacy["chunks"])}
    scope.extend(sorted(scopes))
    with ExitStack() as stack:
        for index, item in enumerate(terminals):
            row = old._object(item)
            if set(row) != {"workflow", "state", "status", "log"} or row["state"] != states[index]:
                raise ValueError("Terminal state identity differs")
            workflow, _ = _terminal_ref(row["workflow"])
            status, content = _terminal_ref(row["status"])
            log, _ = _terminal_ref(row["log"])
            if (
                str(workflow) != workflows[index]
                or re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\s*", content) is None
                or int(content) > 255
            ):
                raise ValueError("Terminal workflow or exit status differs")
            paths.extend((workflow, status, log))
            for value in (row["workflow"], row["status"], row["log"]):
                reference = old._object(value)
                pins[str(reference["path"])] = assembly._sha(reference["sha256"])
            if not old._path(states[index]).is_dir():
                raise ValueError("Original state directory is missing")
            stack.enter_context(old._unlocked(old._path(states[index]) / ".workflow.lock"))
        for directory in scopes:
            stack.enter_context(old._unlocked(directory / "ocr/collector.lock"))
        _process_absence(tuple(scope))
        if any(old._pin(Path(name))["sha256"] != sha for name, sha in pins.items()):
            raise ValueError("Terminal owner inputs changed during observation")
        identities = {
            str(path): list(
                (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            )
            for path in dict.fromkeys(paths)
            for info in [path.stat()]
        }
        code = _code()
        _process_absence(tuple(scope))
    return {
        "version": 2,
        "kind": "terminal-generation-observation",
        "request": old._pin(request),
        "successor": raw["successor"],
        "oldWorkersDisabled": True,
        "states": states,
        "maximumAgeSeconds": age,
        "terminals": terminals,
        "identities": identities,
        "processScope": [str(path) for path in scope],
        "matchingProcesses": [],
        "historicalActorBirthRecorded": False,
        "readerCode": code,
    }


def observe_owner(request: Path, output: Path) -> Path:
    """Record current exact terminal/lock/process evidence without inventing exited actors."""
    value = _terminal_observation(request)
    value["observedAtNs"] = time.time_ns()
    old._path(str(output))
    publish(output, encode(value))
    return output


def _owner_guard(
    handoff: dict[str, object],
    states: tuple[Path, ...],
    *,
    fresh: bool,
    retained: bool = False,
    owner_observation: Path | None = None,
) -> None:
    if handoff.get("kind") != "terminal-generation-observation":
        old._quiescent(handoff, states, fresh=fresh)
        return
    observed, age = handoff.get("observedAtNs"), handoff.get("maximumAgeSeconds")
    if type(observed) is not int or type(age) is not int or not 1 <= age <= 3600:
        raise ValueError("Invalid terminal owner clock")
    if fresh and not 0 <= time.time_ns() - observed <= age * 10**9:
        raise ValueError("Terminal owner observation expired")
    path, _ = old._ref(handoff.get("request"), {})
    current = _terminal_observation(path)
    saved = dict(handoff)
    del saved["observedAtNs"]
    if retained:
        _retained_code(saved.get("readerCode"), _code())
        saved["readerCode"] = current["readerCode"]
        if owner_observation is not None:
            replacement = old._read(owner_observation, {})
            _owner_guard(replacement, states, fresh=False)
            replacement = dict(replacement)
            replacement.pop("observedAtNs")
            if replacement != current:
                raise ValueError("Replacement owner observation differs")
            saved["identities"] = replacement["identities"]
    if current != saved or current["states"] != [str(path) for path in states]:
        raise ValueError("Terminal generation owner evidence changed")


def _load(plan: Path, *, retained: bool = False) -> dict[str, object]:
    data = old._read(plan, {})
    if data.get("version") != 1 or data.get("kind") != "cleanup-recovery-generation":
        raise ValueError("Unsupported recovery generation")
    if retained:
        _retained_code(data.get("readerCode"), _code())
    elif data.get("readerCode") != _code():
        raise ValueError("Recovery code changed")
    for name, sha in old._object(data["inputs"]).items():
        current = file_digest(old._path(name, work=False))
        if current != assembly._sha(sha) and not (
            retained and _retained_digest(name, sha, current)
        ):
            raise ValueError("Recovery input changed")
    workers = data.get("workers")
    rows = old._list(data["chunks"])
    legacy = old._read(old._path(data["successor"]), {})
    originals = old._list(legacy["chunks"])
    if type(workers) is not int or not 1 <= workers <= 8 or len(rows) != len(originals):
        raise ValueError("Recovery coverage or concurrency changed")
    for index, (item, original_item) in enumerate(zip(rows, originals, strict=True)):
        row, original = old._object(item), old._object(original_item)
        mode = row.get("mode")
        expected = (
            str(plan.parent / f"chunk-{index:05d}" / "chunk.json")
            if mode == "pending"
            else original["chunk"]
        )
        if (
            mode not in {"pending", "retained-completed", "adopted-existing"}
            or row.get("index") != index
            or row.get("lane") != index % workers
            or row.get("originalChunk") != original["chunk"]
            or row.get("chunk") != expected
            or row.get("proof") != str(plan.parent / f"verified-{index:05d}.json")
            or row.get("oldStageNames") != original["successorStages"]
            or row.get("oldState") != old._strings(legacy["states"])[int(str(original["lane"]))]
        ):
            raise ValueError("Recovery selection identity changed")
        selected = old._read(old._path(row["chunk"]), {})
        prior = old._read(old._path(original["chunk"]), {})
        if selected.get("clock") != prior.get("clock") or selected.get("settings") != prior.get(
            "settings"
        ):
            raise ValueError("Recovery source clock or settings changed")
        if mode == "pending" and (
            selected.get("maximumActiveChunks") != workers
            or selected.get("ocrDirectory") != str(Path(str(expected)).parent / "ocr")
        ):
            raise ValueError("Recovery pending capacity or OCR location changed")
    return data


@contextmanager
def _owned(
    plan: Path, index: int, *, retained: bool = False, owner_observation: Path | None = None
) -> Iterator[tuple[dict[str, object], dict[str, object]]]:
    data = _load(plan, retained=retained)
    rows = old._list(data["chunks"])
    if type(index) is not int or not 0 <= index < len(rows):
        raise ValueError("Unknown recovery chunk")
    row = old._object(rows[index])
    legacy = old._path(data["successor"])
    states = tuple(old._path(value) for value in old._strings(data["oldStates"]))
    handoff = old._read(old._path(data["handoff"]), {})
    with ExitStack() as stack:
        _owner_guard(
            handoff, states, fresh=False, retained=retained, owner_observation=owner_observation
        )
        for state in states:
            stack.enter_context(old._unlocked(state / ".workflow.lock"))
        if not stack.enter_context(execution_lock(plan.parent, f"recovery-lane-{row['lane']}")):
            raise ValueError("Recovery lane already active")
        for name in old._strings(row["oldStageNames"]):
            if not stack.enter_context(execution_lock(old._path(row["oldState"]), name)):
                raise ValueError("Previous generation chunk is active")
        if retained and data["readerCode"] != _code():
            stack.enter_context(_retained_successor_owned(legacy, index))
        else:
            stack.enter_context(old._owned(legacy, index))
        yield data, row
        _owner_guard(
            handoff, states, fresh=False, retained=retained, owner_observation=owner_observation
        )
        _load(plan, retained=retained)


def _prior_complete(
    data: dict[str, object], row: dict[str, object], *, ocr_only: bool = False
) -> None:
    workflows, states = old._strings(data["workflows"]), old._strings(data["states"])
    lane = row.get("lane")
    if type(lane) is not int or not 0 <= lane < len(workflows):
        raise ValueError("Invalid previous lane")
    names = old._strings(row["successorStages"])
    if ocr_only:
        names = [name for name in names if name.endswith("-ocr")]
        if len(names) != 1:
            raise ValueError("Missing original completed OCR Stage")
    selected = [stage for stage in load_stages(Path(workflows[lane])) if stage.name in names]
    if len(selected) != len(names):
        raise ValueError("Missing previous Stage lineage")
    for stage in selected:
        old._complete(stage, Path(states[lane]))


def _copy_ocr(source: Path, target: Path) -> None:
    cleanup.check_ocr(source)
    raw, _, _ = cleanup._chunk(source)
    origin = old._path(raw["ocrDirectory"])
    complete = cleanup._read(origin / "complete.json")
    names = ["inputs.json", "complete.json", *old._object(complete["results"])]
    target.mkdir()
    for name in names:
        # collector_frames already enforces contiguous canonical result names.
        publish(target / name, bounded_read(origin / name))


def _row(
    index: int,
    original: dict[str, object],
    *,
    legacy: dict[str, object],
    output: Path,
    workers: int,
    adopted: dict[int, object],
    pins: dict[str, str],
) -> dict[str, object]:
    chunk = old._path(original["chunk"])
    if original.get("index") != index:
        raise ValueError("Previous chunk order differs")
    raw = old._read(chunk, pins)
    candidate = chunk.parent / "repair/candidate.mp4"
    receipt = chunk.parent / "repair/candidate.json"
    if candidate.exists() != receipt.exists():
        raise ValueError("Partial previous repair must be preserved for explicit diagnosis")
    proof: object = None
    repair_evidence: object = None
    if candidate.exists():
        if index in adopted:
            _, proof_data = old._ref(adopted[index], pins)
            if proof_data != admission_reader.verify_admission(chunk, verify_repair=True):
                raise ValueError("Adopted repair proof differs")
            proof = adopted[index]
            mode = "adopted-existing"
        else:
            _prior_complete(legacy, original)
            mode = "retained-completed"
        selected = chunk
        repair_evidence = {
            "receipt": old._pin(receipt),
            "media": {
                "path": str(candidate),
                "sha256": assembly._sha(cleanup._read(receipt)["videoSHA256"]),
            },
        }
    else:
        if index in adopted:
            raise ValueError("Adoption requires existing repair")
        mode = "pending"
        directory = output / f"chunk-{index:05d}"
        directory.mkdir()
        selected = directory / "chunk.json"
        times = chunk.parent / "times.json"
        pins[str(times)] = old._pin(times)["sha256"]
        publish(directory / "times.json", bounded_read(times))
        raw = dict(raw)
        raw["maximumActiveChunks"] = workers
        raw["ocrDirectory"] = str(directory / "ocr")
        publish(selected, encode(raw))
        if (chunk.parent / "ocr/complete.json").exists():
            _prior_complete(legacy, original, ocr_only=True)
            _copy_ocr(chunk, directory / "ocr")
            complete = cleanup._read(directory / "ocr/complete.json")
            for name in ["inputs.json", "complete.json", *old._object(complete["results"])]:
                path = directory / "ocr" / name
                pins[str(path)] = old._pin(path)["sha256"]
    pins[str(selected)] = old._pin(selected)["sha256"]
    return {
        "index": index,
        "lane": index % workers,
        "mode": mode,
        "originalChunk": str(chunk),
        "chunk": str(selected),
        "adoptedProof": proof,
        "repairEvidence": repair_evidence,
        "oldStageNames": original["successorStages"],
        "oldState": old._strings(legacy["states"])[int(str(original["lane"]))],
        "proof": str(output / f"verified-{index:05d}.json"),
    }


def _preparation_paths(request: Path) -> tuple[Path, ...]:
    raw = old._read(request, {})
    legacy_path, legacy = old._ref(raw["successor"], {})
    settings_path = old._path(legacy["settings"])
    settings = cleanup._settings(settings_path)
    paths = [request, legacy_path, settings_path]
    paths.extend(
        Path(str(old._object(settings[name])["path"]))
        for name in (
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
    selected: set[str] = set()
    for item in old._list(legacy["chunks"]):
        row = old._object(item)
        chunk = old._path(row["chunk"])
        names = old._strings(row["successorStages"])
        if (chunk.parent / "repair/candidate.mp4").exists():
            selected.update(names)
        elif (chunk.parent / "ocr/complete.json").exists():
            selected.update(name for name in names if name.endswith("-ocr"))
    for name in old._strings(legacy["workflows"]):
        workflow = old._path(name)
        paths.append(workflow)
        for stage in load_stages(workflow):
            if stage.name in selected:
                paths.extend(path for path in stage.inputs + stage.outputs if path.is_file())
    return tuple(dict.fromkeys(paths))


def prepare(request: Path, output: Path) -> Path:
    """Reuse immutable hashes only within this one entry/terminal-checked operation."""
    cache = InputHashCache(_preparation_paths(request), allow_uncached_overflow=True)
    cache.prime()
    with input_hash_context(cache):
        result = _prepare(request, output)
        cache.verify()
    return result


def _prepare(request: Path, output: Path) -> Path:
    """Create new immutable lane definitions; no OCR/render dispatch or old writes."""
    pins: dict[str, str] = {}
    raw = old._read(request, pins)
    if set(raw) != {"version", "successor", "handoff", "workers", "adopted"} or raw["version"] != 1:
        raise ValueError("Invalid recovery request")
    workers = raw["workers"]
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Invalid new generation concurrency")
    legacy_path, legacy = old._ref(raw["successor"], pins)
    owner_path, owner = old._ref(raw["handoff"], pins)
    states = tuple(old._path(value) for value in old._strings(legacy["states"]))
    _owner_guard(owner, states, fresh=True)
    old._load(legacy_path)
    originals = old._list(legacy["chunks"])
    if not 1 <= len(originals) <= assembly.MAX_CHUNKS:
        raise ValueError("Invalid recovery chunk coverage")
    workers = min(workers, len(originals))
    adopted: dict[int, object] = {}
    for item in old._list(raw["adopted"]):
        item_row = old._object(item)
        index = item_row.get("index")
        if (
            set(item_row) != {"index", "proof"}
            or type(index) is not int
            or not 0 <= index < len(originals)
            or index in adopted
        ):
            raise ValueError("Invalid or duplicate adoption")
        adopted[index] = item_row["proof"]
    old._path(str(output))
    old._absent(output)
    output.mkdir()
    plan = output / "generation.json"
    rows = [
        _row(
            index,
            old._object(row),
            legacy=legacy,
            output=output,
            workers=workers,
            adopted=adopted,
            pins=pins,
        )
        for index, row in enumerate(originals)
    ]
    code = _code()
    settings_path = old._path(legacy["settings"])
    settings = cleanup._settings(settings_path)
    authorities = [str(settings_path), str(legacy_path), str(owner_path)]
    authorities.extend(
        str(old._object(settings[name])["path"])
        for name in (
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
    definitions: list[list[dict[str, object]]] = [[] for _ in range(workers)]
    previous: dict[int, str] = {}
    for row in rows:
        lane, index = int(str(row["lane"])), int(str(row["index"]))
        command = [sys.executable, "-B", "-m", "replay.cleanup_generation_recovery"]
        inputs = [str(plan), str(request), str(row["chunk"]), *authorities, *code]
        if lane in previous:
            inputs.append(previous[lane])
        outputs = [row["proof"]]
        if row["mode"] == "pending":
            outputs.extend(
                str(Path(str(row["chunk"])).parent / "repair" / name)
                for name in ("candidate.mp4", "candidate.json")
            )
        definitions[lane].append(
            {
                "name": f"recovery-{index:05d}",
                "command": [*command, "run", str(plan), "--index", str(index)],
                "verify_command": [*command, "verify", str(plan), "--index", str(index)],
                "inputs": list(dict.fromkeys(inputs)),
                "outputs": outputs,
                "timeout_seconds": 7200,
            }
        )
        previous[lane] = str(row["proof"])
    segments = [
        (lane, number, items)
        for lane, definition in enumerate(definitions)
        for number, items in enumerate(_graph_segments(definition))
    ]
    workflows = [output / f"lane-{lane:02d}-part-{number:02d}.json" for lane, number, _ in segments]
    new_states = [output / f"state-{lane:02d}-part-{number:02d}" for lane, number, _ in segments]
    for _, _, items in segments:
        stage_graph(parse_stages({"version": 1, "stages": items}))
    publish(
        plan,
        encode(
            {
                "version": 1,
                "kind": "cleanup-recovery-generation",
                "successor": str(legacy_path),
                "handoff": str(owner_path),
                "oldStates": [str(path) for path in states],
                "settings": legacy["settings"],
                "originalPlan": legacy["originalPlan"],
                "workers": workers,
                "chunks": rows,
                "inputs": pins,
                "readerCode": code,
                "workflows": [str(path) for path in workflows],
                "states": [str(path) for path in new_states],
                "workflowLanes": [lane for lane, _, _ in segments],
            }
        ),
    )
    for workflow, (_, _, items) in zip(workflows, segments, strict=True):
        publish(workflow, encode({"version": 1, "stages": items}))
    parents: list[list[dict[str, object]]] = [[] for _ in range(workers)]
    for index, (workflow, state, (lane, number, _)) in enumerate(
        zip(workflows, new_states, segments, strict=True)
    ):
        stages = load_stages(workflow)
        child_outputs = {path for stage in stages for path in stage.outputs}
        child_inputs = {path for stage in stages for path in stage.inputs} - child_outputs
        parents[lane].append(
            {
                "name": f"segment-{number:02d}",
                "command": [
                    sys.executable,
                    "-B",
                    "-m",
                    "replay.workflow",
                    str(workflow),
                    "--state",
                    str(state),
                ],
                "verify_command": [
                    sys.executable,
                    "-B",
                    "-m",
                    "replay.cleanup_generation_recovery",
                    "verify-segment",
                    str(plan),
                    "--index",
                    str(index),
                ],
                "inputs": [str(path) for path in sorted(child_inputs | {workflow})],
                "outputs": [str(path) for path in sorted(child_outputs)],
                "timeout_seconds": 86400,
            }
        )
    for lane, items in enumerate(parents):
        stage_graph(parse_stages({"version": 1, "stages": items}))
        content = encode({"version": 1, "stages": items})
        if len(content) > MAX_PLAN_BYTES:
            raise ValueError("Recovery parent exceeds workflow byte bound")
        publish(output / f"lane-{lane:02d}.json", content)
    _load(plan)
    return plan


def verify_segment(plan: Path, index: int, *, owner_observation: Path | None = None) -> None:
    """Verify genuine child checkpoints and every selected current domain proof."""
    data = _load(plan, retained=True)
    workflows, states = old._strings(data["workflows"]), old._strings(data["states"])
    if not 0 <= index < len(workflows):
        raise ValueError("Unknown recovery segment")
    for stage in load_stages(Path(workflows[index])):
        retained_complete(stage, Path(states[index]), data["readerCode"])
        if re.fullmatch(r"recovery-[0-9]{5}", stage.name) is None:
            raise ValueError("Invalid recovery segment stage")
        phase(
            plan,
            int(stage.name.removeprefix("recovery-")),
            verify=True,
            owner_observation=owner_observation,
        )


def _pending(chunk: Path) -> None:
    raw, settings, clock = cleanup._chunk(chunk)
    source, native = old._object(settings["source"]), old._object(settings["native"])
    if file_digest(Path(str(source["path"]))) != source["sha256"]:
        raise ValueError("Source changed before recovery production")
    ocr = old._path(raw["ocrDirectory"])
    if not (ocr / "complete.json").exists():
        seeds, _ = cleanup.collector_frames(
            Path(str(old._object(settings["profileInputs"])["path"])),
            Path(str(old._object(settings["profileComplete"])["path"])),
            settings,
        )
        usage = shutil.disk_usage(chunk.parent)
        reserve = (
            len(clock.pts)
            * seeds[0].width
            * seeds[0].height
            * 3
            * int(str(raw["maximumActiveChunks"]))
        )
        if usage.free < max(30 * 1024**3, usage.total // 20) + reserve:
            raise ValueError("Insufficient recovery concurrency capacity")
        subtitle_observations.collect_observations(
            Path(str(source["path"])),
            chunk.parent / "times.json",
            ocr,
            native_executable=Path(str(native["path"])),
        )
    cleanup.build_candidate(chunk)
    subtitle_candidate.render_candidate(chunk.parent / "candidate-config.json")


def _repair_evidence(chunk: Path) -> dict[str, object]:
    receipt = chunk.parent / "repair/candidate.json"
    media = chunk.parent / "repair/candidate.mp4"
    expected = assembly._sha(cleanup._read(receipt)["videoSHA256"])
    if file_digest(media) != expected:
        raise ValueError("Retained repair bytes changed")
    return {"receipt": old._pin(receipt), "media": {"path": str(media), "sha256": expected}}


def phase(
    plan: Path, index: int, *, verify: bool = False, owner_observation: Path | None = None
) -> None:
    """Run one finite new Stage or strictly verify its selected original/new result."""
    if owner_observation is not None and not verify:
        raise ValueError("Replacement observation permits verification only")
    with _owned(plan, index, retained=verify, owner_observation=owner_observation) as (data, row):
        chunk = old._path(row["chunk"])
        if row["mode"] != "pending" and _repair_evidence(chunk) != row["repairEvidence"]:
            raise ValueError("Retained repair evidence changed")
        if row["mode"] == "retained-completed":
            legacy = old._read(old._path(data["successor"]), {})
            prior_row = old._object(old._list(legacy["chunks"])[index])
            if verify and data["readerCode"] != _code():
                lane = int(str(prior_row["lane"]))
                names = old._strings(prior_row["successorStages"])
                stages = [
                    stage
                    for stage in load_stages(Path(old._strings(legacy["workflows"])[lane]))
                    if stage.name in names
                ]
                if len(stages) != len(names):
                    raise ValueError("Missing previous Stage lineage")
                for stage in stages:
                    retained_complete(
                        stage, Path(old._strings(legacy["states"])[lane]), legacy["readerCode"]
                    )
            else:
                _prior_complete(legacy, prior_row)
        elif row["mode"] == "adopted-existing":
            _, original_proof = old._ref(row["adoptedProof"], {})
            if original_proof != admission_reader.verify_admission(chunk, verify_repair=True):
                raise ValueError("Adopted repair proof changed")
        if not verify and row["mode"] == "pending":
            _pending(chunk)
        result = admission_reader.verify_admission(chunk, verify_repair=True)
        repair = _repair_evidence(chunk)
        if row["mode"] != "pending" and repair != row["repairEvidence"]:
            raise ValueError("Retained repair changed during verification")
        value = {
            "version": 1,
            "generation": old._pin(plan),
            "index": index,
            "mode": row["mode"],
            "originalChunk": row["originalChunk"],
            "adoptedProof": row["adoptedProof"],
            "verification": result,
            "repairEvidence": repair,
            "qualityApproved": False,
        }
        proof = old._path(row["proof"])
        if verify:
            if old._read(proof, {}) != value:
                raise ValueError("Recovery result differs")
        else:
            publish(proof, encode(value))


def assemble(
    plan: Path, output: Path, *, retained: bool = False, owner_observation: Path | None = None
) -> Path:
    """Publish verified candidate refs, without producing or changing any repair."""
    if owner_observation is not None and not retained:
        raise ValueError("Replacement observation requires retained assembly")
    data = _load(plan, retained=retained)
    for workflow, state in zip(
        old._strings(data["workflows"]), old._strings(data["states"]), strict=True
    ):
        for stage in load_stages(Path(workflow)):
            if retained:
                retained_complete(stage, Path(state), data["readerCode"])
            else:
                old._complete(stage, Path(state))
    pins: dict[str, str] = {}
    settings_path = old._path(data["settings"])
    settings = assembly._read(settings_path, pins)
    _, source_clock = assembly._ref(settings["sourceClock"], pins)
    clocks = clock_chunks(source_clock, maximum_frames=assembly._positive(settings["chunkFrames"]))
    chunks = old._list(data["chunks"])
    if len(chunks) != len(clocks) or clocks[0].pts[0] != 0:
        raise ValueError("Recovery does not cover whole original clock")
    ranges: list[dict[str, object]] = []
    refs: list[dict[str, object]] = []
    canvases: set[tuple[int, int]] = set()
    for index, (raw, clock) in enumerate(zip(chunks, clocks, strict=True)):
        row = old._object(raw)
        if owner_observation is None:
            phase(plan, index, verify=True)
        else:
            phase(plan, index, verify=True, owner_observation=owner_observation)
        chunk = old._path(row["chunk"])
        # Original successor chunk names differ, but its unchanged original chunk.json
        # remains beside the same producer. New pending directories use chunk.json.
        selected, reference, canvas = assembly._chunk(
            chunk.parent,
            clock,
            settings_path,
            assembly._identity(settings["source"]),
            assembly._identity(settings["native"]),
            pins,
        )
        ranges.append(selected)
        refs.append(reference)
        canvases.add(canvas)
    if len(canvases) != 1:
        raise ValueError("Recovery chunk geometry differs")
    old._path(str(output))
    old._absent(output)
    output.mkdir()
    index_path, manifest = output / "index.json", output / "manifest.json"
    publish(index_path, encode({"ranges": ranges, "purpose": "explicit recovery selected results"}))
    publish(manifest, encode({"version": 1, "index": old._pin(index_path), "chunks": refs}))
    # Existing canonical verifier binds every media SHA and original producer clock.
    delivery_source_assembly.load_chunks(manifest, sha256=old._pin(manifest)["sha256"], bindings={})
    publish(
        output / "lineage.json",
        encode(
            {
                "version": 1,
                "generation": old._pin(plan),
                "manifest": old._pin(manifest),
                "selectedResults": chunks,
                "requiredNextStage": (
                    "delivery_source_assembly full packet/decoded PTS verification"
                ),
                "qualityApproved": False,
                "deliveryComplete": False,
            }
        ),
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("prepare", "run", "verify", "assemble", "observe-owner", "verify-segment"),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--owner-observation", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args(argv)
    try:
        if args.operation in {"prepare", "assemble", "observe-owner"}:
            if args.output is None:
                raise ValueError("Output required")
            if args.operation == "prepare":
                result = prepare(args.input, args.output)
            elif args.operation == "assemble":
                result = assemble(args.input, args.output)
            else:
                result = observe_owner(args.input, args.output)
            print(result)
        else:
            if args.index is None:
                raise ValueError("Index required")
            if args.operation == "verify-segment":
                verify_segment(args.input, args.index, owner_observation=args.owner_observation)
            else:
                phase(
                    args.input,
                    args.index,
                    verify=args.operation == "verify",
                    owner_observation=args.owner_observation,
                )
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"RECOVERY_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
