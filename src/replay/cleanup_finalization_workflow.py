"""Connect quiescent cleanup lanes to ordinary exact-clock assembly workflows.

Preparation reads metadata only. Runtime never polls, restarts a lane, runs OCR,
or treats an occupied partial publication as complete. Original receipts remain
unchanged; the child uses the existing lossless assembly and readonly domain.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from replay import cleanup_lane_successor as successor
from replay import code_dependency_inventory, workflow
from replay import delivery_source_assembly as media
from replay import subtitle_cleanup_assembly as assembly
from replay import subtitle_cleanup_workflow as cleanup
from replay.core import ReplayError
from replay.fcp_transformed_source import clean_source
from replay.recordings import record
from replay.sequence_journal import bounded_read, decode, encode, publish
from replay.stages import StageError, file_digest
from replay.subtitle_cleanup_plan import clock_chunks
from replay.subtitle_eligibility import _reference as tool_reference

PROTOCOL = "cleanup-finalization-workflow-v1"
MAX_SECONDS = 43200
FINAL_FILES = ("index.json", "manifest.json", "preparation.json", "successor-generation.json")
MODES = ("finalize", "prepare-child", "run-child")


@dataclass(frozen=True)
class Job:
    operation: Path
    root: Path
    plan: Path
    lanes: tuple[tuple[Path, Path, Path], ...]
    code: dict[str, str]
    tools: tuple[dict[str, object], dict[str, object]]
    timeout: int


def _path(value: object) -> Path:
    return successor._path(value)


def _read(path: Path) -> dict[str, object]:
    return decode(bounded_read(_path(str(path))))


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(bounded_read(path)).hexdigest()}


def _ref(value: object) -> Path:
    ref = record(value)
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Expected exact finalization metadata reference")
    path = _path(ref["path"])
    if _pin(path)["sha256"] != assembly._sha(ref["sha256"]):
        raise ValueError("Finalization metadata pin changed")
    return path


def _code() -> dict[str, str]:
    return {
        str(item.path): item.sha256
        for item in code_dependency_inventory.capture(("cleanup_finalization_workflow",))
    }


def _request(
    path: Path,
) -> tuple[
    Path, tuple[tuple[Path, Path, Path], ...], tuple[dict[str, object], dict[str, object]], int
]:
    raw = _read(path)
    if set(raw) != {"version", "successor", "lanes", "ffmpeg", "ffprobe", "timeoutSeconds"} or (
        type(raw["version"]) is not int or raw["version"] != 1
    ):
        raise ValueError("Invalid cleanup finalization request")
    timeout = raw["timeoutSeconds"]
    if type(timeout) is not int or not 1 <= timeout <= MAX_SECONDS:
        raise ValueError("Expected bounded finalization timeout")
    plan = _ref(raw["successor"])
    data = _read(plan)
    rows = raw["lanes"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        raise ValueError("Expected explicit bounded lane bindings")
    lanes: list[tuple[Path, Path, Path]] = []
    for raw_row in rows:
        row = record(raw_row)
        if set(row) != {"workflow", "state", "status"}:
            raise ValueError("Invalid cleanup lane terminal binding")
        lanes.append((_ref(row["workflow"]), _path(row["state"]), _path(row["status"])))
    if (
        len({item for lane in lanes for item in lane}) != len(lanes) * 3
        or [str(lane[0]) for lane in lanes] != data.get("workflows")
        or [str(lane[1]) for lane in lanes] != data.get("states")
    ):
        raise ValueError("Lane bindings differ from the original successor")
    tools = record(raw["ffmpeg"]), record(raw["ffprobe"])
    for tool in tools:
        if set(tool) != {"path", "sha256"}:
            raise ValueError("Expected exact installed tool reference")
        media._path(tool["path"])
        assembly._sha(tool["sha256"])
    return plan, tuple(lanes), tools, timeout


def _load(operation: Path) -> Job:
    raw = _read(operation)
    if set(raw) != {"version", "protocol", "request", "workRoot", "readerCode"} or (
        raw.get("version") != 1 or raw.get("protocol") != PROTOCOL or raw["readerCode"] != _code()
    ):
        raise ValueError("Finalization operation or reader code changed")
    plan, lanes, tools, timeout = _request(_ref(raw["request"]))
    code = {name: assembly._sha(sha) for name, sha in record(raw["readerCode"]).items()}
    return Job(operation, _path(raw["workRoot"]), plan, lanes, code, tools, timeout)


@contextmanager
def _ready(job: Job) -> Iterator[dict[str, object]]:
    """Hold shared locks against lane reentry throughout each finite operation."""
    with ExitStack() as stack:
        for _, state, status in job.lanes:
            stack.enter_context(successor._unlocked(state / ".workflow.lock"))
            if bounded_read(status).strip() != b"0":
                raise ValueError("All original cleanup lane outcomes must be terminal zero")
        data = successor._load(job.plan)
        for graph, state, _ in job.lanes:
            for stage in workflow.load_stages(graph):
                successor._complete(stage, state)
        settings = cleanup._settings(_path(data["settings"]))
        source = record(settings["source"])
        if file_digest(media._path(source["path"])) != source["sha256"]:
            raise ValueError("Original cleanup source bytes changed")
        for tool in job.tools:
            tool_reference(tool, {}, executable=True)
        yield data
        if any(bounded_read(status).strip() != b"0" for _, _, status in job.lanes):
            raise ValueError("Cleanup lane outcome changed during finalization")
        if file_digest(media._path(source["path"])) != source["sha256"]:
            raise ValueError("Original cleanup source changed during finalization")
        cleanup._settings(_path(data["settings"]))
        for tool in job.tools:
            tool_reference(tool, {}, executable=True)
        successor._load(job.plan)
        _load(job.operation)


def _finalized(job: Job, data: dict[str, object]) -> dict[str, object]:
    """Reconstruct metadata using original typed assembly helpers, without writes."""
    output = job.root / "finalized"
    inputs: dict[str, str] = {}
    original = _path(data["originalPlan"])
    plan = assembly._read(original, inputs)
    settings_path, settings = assembly._ref(plan.get("settings"), inputs)
    source, native = assembly._identity(settings["source"]), assembly._identity(settings["native"])
    clock_path, clock = assembly._ref(settings["sourceClock"], inputs)
    clocks = clock_chunks(clock, maximum_frames=assembly._positive(settings["chunkFrames"]))
    frames = sum(len(item.pts) for item in clocks)
    if (
        not clocks
        or len(clocks) > assembly.MAX_CHUNKS
        or clocks[0].pts[0] != 0
        or plan.get("wholeSourcePlanned") is not True
        or plan.get("chunks") != len(clocks)
        or plan.get("totalSourceFrames") != frames
        or plan.get("plannedFrames") != frames
    ):
        raise ValueError("Finalization must retain the entire original physical clock")
    rows: list[dict[str, object]] = []
    refs: list[dict[str, object]] = []
    canvases: set[tuple[int, int]] = set()
    for index, frame_clock in enumerate(clocks):
        row, ref, canvas = assembly._chunk(
            original.parent / f"chunk-{index:05d}",
            frame_clock,
            settings_path,
            source,
            native,
            inputs,
        )
        rows.append(row)
        refs.append(ref)
        canvases.add(canvas)
    if len(canvases) != 1:
        raise ValueError("Finalized cleanup geometry differs")
    code = {
        str(Path(assembly.__file__).with_name(name).resolve()): hashlib.sha256(
            assembly._bytes(Path(assembly.__file__).with_name(name).resolve(), code=True)
        ).hexdigest()
        for name in assembly.CODE_FILES
    }
    inputs.update(code)
    manifest = output / "manifest.json"
    if _read(output / "index.json") != {
        "purpose": "ordered source-clock cleanup candidates; quality unapproved",
        "ranges": rows,
    } or _read(manifest) != {"version": 1, "index": _pin(output / "index.json"), "chunks": refs}:
        raise ValueError("Finalized index/manifest differs from original chunk domains")
    if _read(output / "preparation.json") != {
        "version": 1,
        "inputs": inputs,
        "currentPreparationCode": code,
        "sourceIdentity": {"path": source[0], "sha256": source[1]},
        "sourceClock": {"path": str(clock_path), "sha256": inputs[str(clock_path)]},
        "chunks": len(clocks),
        "physicalSourceFrames": frames,
        "physicalSourceTerminalPTS": clocks[-1].pts[-1] + clocks[-1].last_duration,
        "timeBase": str(clocks[0].time_base),
        "canvas": list(canvases.pop()),
        "manifest": _pin(manifest),
        "exactPreparedSourceClockVerified": True,
        "candidateOnly": True,
        "requiredNextStage": (
            "delivery_source_assembly full media, packets and decoded PTS verification"
        ),
        "mediaBytesVerified": False,
        "sourceCaptionQualityApproved": False,
        "deliveryComplete": False,
    } or _read(output / "successor-generation.json") != {
        "version": 1,
        "successor": _pin(job.plan),
        "manifest": _pin(manifest),
        "maximumActiveChunks": data["workers"],
        "chunks": data["chunks"],
        "allPhysicalPTSAccountedByCanonicalAssembly": True,
        "qualityApproved": False,
        "mediaBytesVerified": False,
    }:
        raise ValueError("Finalization provenance or generation differs")
    if any(
        hashlib.sha256(assembly._bytes(Path(name), code=name in code)).hexdigest() != digest
        for name, digest in inputs.items()
    ):
        raise ValueError("Finalization metadata changed during readonly verification")
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "operation": _pin(job.operation),
        "outputs": {str(output / name): _pin(output / name)["sha256"] for name in FINAL_FILES},
    }


def _command(mode: str, operation: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "replay.cleanup_finalization_workflow",
        mode,
        str(operation),
    ]


def _child(job: Job) -> dict[str, object]:
    root = job.root
    command = [
        sys.executable,
        "-B",
        "-m",
        "replay.delivery_source_assembly",
        str(root / "finalized/manifest.json"),
        str(root / "clean-source.mov"),
        str(root / "evidence"),
        "--sha256",
        _pin(root / "finalized/manifest.json")["sha256"],
        "--ffmpeg",
        str(job.tools[0]["path"]),
        "--ffprobe",
        str(job.tools[1]["path"]),
    ]
    return {
        "version": 1,
        "stages": [
            {
                "name": "assemble-clean-source",
                "command": command,
                "inputs": list(
                    dict.fromkeys(
                        [
                            str(job.operation),
                            str(job.plan),
                            *job.code,
                            *[str(root / "finalized" / name) for name in FINAL_FILES],
                            *[str(tool["path"]) for tool in job.tools],
                        ]
                    )
                ),
                "outputs": [
                    str(root / "clean-source.mov"),
                    str(root / "evidence/receipt.json"),
                    str(root / "evidence/single-source-index.json"),
                ],
                "timeout_seconds": job.timeout,
                "verify_command": _command("verify-assembly", job.operation),
            }
        ],
    }


def _prepared(job: Job) -> dict[str, object]:
    graph = job.root / "child/workflow.json"
    if _read(graph) != _child(job):
        raise ValueError("Prepared child graph differs from exact generated manifest")
    workflow.stage_graph(workflow.load_stages(graph))
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "operation": _pin(job.operation),
        "workflow": _pin(graph),
        "finalized": _pin(job.root / "finalized.json"),
    }


def _assembled(job: Job) -> dict[str, object]:
    root = job.root
    manifest = root / "finalized/manifest.json"
    clean_source(
        {
            "assemblyManifest": _pin(manifest),
            "assemblyReceipt": _pin(root / "evidence/receipt.json"),
            "cleanIndex": _pin(root / "evidence/single-source-index.json"),
        }
    )
    receipt = _read(root / "evidence/receipt.json")
    if record(receipt["output"])["path"] != str(root / "clean-source.mov"):
        raise ValueError("Assembled output belongs to another namespace")
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "operation": _pin(job.operation),
        "assemblyReceipt": _pin(root / "evidence/receipt.json"),
        "cleanIndex": _pin(root / "evidence/single-source-index.json"),
        "output": receipt["output"],
        "qualityApproved": False,
        "avSyncApproved": False,
    }


def _completed(job: Job) -> dict[str, object]:
    if _read(job.root / "prepared.json") != _prepared(job):
        raise ValueError("Prepared child checkpoint changed")
    stage = workflow.load_stages(job.root / "child/workflow.json")[0]
    successor._complete(stage, job.root / "child-state")
    result = _assembled(job)
    result["childStageReceipt"] = _pin(job.root / "child-state/assemble-clean-source.json")
    return result


def prepare(config: Path, output: Path) -> Path:
    """Create an ordinary finite parent graph without reading source/media bytes."""
    _path(str(output))
    if not output.is_dir() or tuple(output.iterdir()):
        raise ValueError("Finalization preparation requires an existing empty work root")
    plan, lanes, _, timeout = _request(config)
    code = _code()
    operation = output / "operation.json"
    publish(
        operation,
        encode(
            {
                "version": 1,
                "protocol": PROTOCOL,
                "request": _pin(config),
                "workRoot": str(output),
                "readerCode": code,
            }
        ),
    )
    for name in ("finalized", "child"):
        (output / name).mkdir()
    common = list(
        dict.fromkeys(
            [
                str(operation),
                str(config),
                str(plan),
                *code,
                *[str(item) for lane in lanes for item in (lane[0], lane[2])],
            ]
        )
    )
    finalized = [str(output / "finalized" / name) for name in FINAL_FILES] + [
        str(output / "finalized.json")
    ]
    prepared = [str(output / "child/workflow.json"), str(output / "prepared.json")]
    final = [
        str(output / "clean-source.mov"),
        str(output / "evidence/receipt.json"),
        str(output / "evidence/single-source-index.json"),
        str(output / "completion.json"),
        str(output / "child-state/assemble-clean-source.json"),
    ]
    phases = (
        (MODES[0], common, finalized),
        (MODES[1], common + finalized, prepared),
        (MODES[2], common + finalized + prepared, final),
    )
    graph = {
        "version": 1,
        "stages": [
            {
                "name": mode,
                "command": _command(mode, operation),
                "inputs": inputs,
                "outputs": outputs,
                "timeout_seconds": timeout,
                "verify_command": _command("verify-" + mode, operation),
            }
            for mode, inputs, outputs in phases
        ],
    }
    workflow.stage_graph(workflow.parse_stages(graph))
    path = output / "workflow.json"
    publish(path, encode(graph))
    _load(operation)
    return path


def run(operation: Path, *, mode: str) -> None:
    """One finite Stage operation or readonly verifier; never wait for workers."""
    job = _load(operation)
    with _ready(job) as data:
        if mode == "finalize":
            successor.assemble(job.plan, job.root / "finalized")
            publish(job.root / "finalized.json", encode(_finalized(job, data)))
        elif mode == "verify-finalize":
            if _read(job.root / "finalized.json") != _finalized(job, data):
                raise ValueError("Finalized checkpoint changed")
        elif mode in {
            "prepare-child",
            "verify-prepare-child",
            "run-child",
            "verify-run-child",
            "verify-assembly",
        }:
            if _read(job.root / "finalized.json") != _finalized(job, data):
                raise ValueError("Finalized checkpoint changed")
            _continue(job, mode)
        else:
            raise ValueError("Unknown finalization mode")


def _continue(job: Job, mode: str) -> None:
    if mode == "prepare-child":
        graph = job.root / "child/workflow.json"
        publish(graph, encode(_child(job)))
        if workflow.main([str(graph), "--state", str(job.root / "child-state"), "--check"]) != 0:
            raise ValueError("Normal assembly child preflight failed")
        publish(job.root / "prepared.json", encode(_prepared(job)))
    elif mode == "verify-prepare-child":
        if _read(job.root / "prepared.json") != _prepared(job):
            raise ValueError("Prepared checkpoint changed")
    elif mode == "verify-assembly":
        _assembled(job)
    elif mode == "run-child":
        if _read(job.root / "prepared.json") != _prepared(job):
            raise ValueError("Prepared checkpoint changed")
        if (job.root / "completion.json").exists():
            if _read(job.root / "completion.json") != _completed(job):
                raise ValueError("Completed child cannot be replaced or reissued")
            return
        stage = workflow.load_stages(job.root / "child/workflow.json")[0]
        if any(path.exists() or path.is_symlink() for path in stage.outputs):
            # Only a genuine completed child can reenter the ordinary verifier.
            # A failed/partial namespace remains intact for explicit diagnosis.
            successor._complete(stage, job.root / "child-state")
        chunks = media.load_chunks(
            job.root / "finalized/manifest.json",
            sha256=_pin(job.root / "finalized/manifest.json")["sha256"],
            bindings={},
        )
        usage = shutil.disk_usage(job.root)
        budget = sum(chunk.media.stat().st_size for chunk in chunks) + 64 * 1024**2
        if usage.free < max(30 * 1024**3, usage.total // 20) + budget:
            raise ValueError("Insufficient headroom for exact-clock source assembly")
        # The ordinary runner owns cancellation and stops its stage child group.
        if (
            workflow.main(
                [str(job.root / "child/workflow.json"), "--state", str(job.root / "child-state")]
            )
            != 0
        ):
            raise ValueError("Normal assembly child workflow did not complete")
        publish(job.root / "completion.json", encode(_completed(job)))
    elif mode == "verify-run-child":
        if _read(job.root / "completion.json") != _completed(job):
            raise ValueError("Completion checkpoint changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("prepare", *MODES, *["verify-" + mode for mode in MODES], "verify-assembly"),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "prepare":
            if args.output is None:
                raise ValueError("Preparation requires --output")
            print(prepare(args.input, args.output))
        else:
            if args.output is not None:
                raise ValueError("Runtime uses the pinned operation only")
            run(args.input, mode=args.mode)
    except (OSError, ValueError, ReplayError, StageError) as error:
        print(f"CLEANUP_FINALIZATION_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
