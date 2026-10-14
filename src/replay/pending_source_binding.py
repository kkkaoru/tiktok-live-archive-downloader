"""Bind completed cleanup and semantic artifacts without guessing future hashes."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from replay import code_dependency_inventory as inventory
from replay import completion_barrier as barrier
from replay import source_caption_binding as binding
from replay import workflow
from replay.artifact_json import _json, _path, _pin
from replay.caption_consumer_binding import _read as _caption_receipt
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference, _save
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
FUTURE_FIELDS = frozenset({"assemblyManifest", "assemblyReceipt", "cleanIndex", "semanticXML"})
FIXED_FIELDS = frozenset(
    {"version", "terminalFrameCell", "removeSourceMasks", "voice", "voiceReceipt"}
)
REQUEST_FIELDS = frozenset(
    {
        "version",
        "kind",
        "fixedFields",
        "futurePinFields",
        "requiredCompletions",
        "configOutput",
        "bindingOutput",
        "bindingConsumerKind",
        "readyForRender",
    }
)


@dataclass(frozen=True)
class Pending:
    request: Path
    config: dict[str, object]
    completions: tuple[Path, ...]
    config_output: Path
    binding_output: Path

    @property
    def receipt(self) -> Path:
        return self.binding_output.with_suffix(".pending-source.json")


def _local(value: object) -> Path:
    path = _path(value)
    if not path.is_relative_to(ROOT / "edits/work"):
        raise ReplayError("Pending binding paths must stay inside checkout edits/work")
    return path


def _intent(request: Path, sha256: str) -> dict[str, object]:
    _local(str(request))
    _pin({"path": str(request), "sha256": sha256})
    data = _json(request)
    fixed = record(data.get("fixedFields"))
    future = record(data.get("futurePinFields"))
    if (
        set(data) != REQUEST_FIELDS
        or data.get("version") != 1
        or data.get("kind") != "pending-actual-pins-not-executable"
        or data.get("bindingConsumerKind") != "transformed-source-config"
        or data.get("readyForRender") is not False
        or set(fixed) != FIXED_FIELDS
        or set(future) != FUTURE_FIELDS
        or fixed.get("version") != 1
        or fixed.get("removeSourceMasks") is not True
        or not isinstance(fixed.get("terminalFrameCell"), bool)
    ):
        raise ReplayError("Invalid pending source binding schema or intent")
    _pin(fixed.get("voice"))
    _pin(fixed.get("voiceReceipt"))
    return data


def _load(request: Path, sha256: str) -> Pending:
    data = _intent(request, sha256)
    fixed = record(data["fixedFields"])
    future = record(data["futurePinFields"])
    raw = data.get("requiredCompletions")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ReplayError("Require cleanup and semantic completion artifacts")
    completions = tuple(_local(value) for value in raw)
    config: dict[str, object] = {
        **fixed,
        **{key: _reference(_local(value)) for key, value in future.items()},
    }
    outputs = (_local(data.get("configOutput")), _local(data.get("bindingOutput")))
    all_outputs = (*outputs, outputs[1].with_suffix(".pending-source.json"))
    input_paths = {
        request,
        *completions,
        *(_pin(value) for value in config.values() if isinstance(value, dict)),
    }
    if (
        len(set(completions)) != 2
        or len(set(all_outputs)) != 3
        or input_paths.intersection(all_outputs)
    ):
        raise ReplayError("Pending binding paths must be distinct")
    return Pending(request, config, completions, *outputs)


def _evidence(pending: Pending) -> tuple[binding.Cleanup, dict[str, str]]:
    original, clean, duration = binding.clean_source(pending.config)
    paths = tuple(
        _pin(pending.config[key]) for key in (*sorted(FUTURE_FIELDS), "voice", "voiceReceipt")
    )
    # Semantic head receipts include per-title measurement lineage. Use the
    # existing caption receipt budget, then retain the ordinary cleanup limit.
    completion_rows = tuple(_caption_receipt(path) for path in pending.completions)
    assembly = _reference(_pin(pending.config["assemblyReceipt"]))
    semantic = _reference(_pin(pending.config["semanticXML"]))
    cleanup_rows = tuple(row for row in completion_rows if row.get("assemblyReceipt") == assembly)
    head_rows = tuple(row for row in completion_rows if row.get("semanticXML") == semantic)
    if len(cleanup_rows) != 1 or len(head_rows) != 1 or cleanup_rows[0] is head_rows[0]:
        raise ReplayError("Completion artifacts do not bind actual cleanup and semantic outputs")
    cleanup, head = cleanup_rows[0], head_rows[0]
    cleanup_path = pending.completions[completion_rows.index(cleanup)]
    if _json(cleanup_path) != cleanup:
        raise ReplayError("Cleanup completion changed during validation")
    child = _pin(cleanup.get("childStageReceipt"))
    if (
        cleanup.get("version") != 1
        or cleanup.get("output") != _reference(clean)
        or _json(child).get("status") != "complete"
        or head.get("readyForRender") is not False
        or head.get("requiredNextStage") != "verified-clean-source-substitution"
    ):
        raise ReplayError("Upstream completion is incomplete or belongs to different media")
    voice_receipt = _json(_pin(pending.config["voiceReceipt"]))
    if voice_receipt.get("outputSHA256") != record(pending.config["voice"]).get("sha256"):
        raise ReplayError("Processed voice completion differs from pinned voice")
    snapshot = {
        str(path): file_digest(path)
        for path in (
            pending.request,
            *paths,
            *pending.completions,
            child,
            original,
            clean,
        )
    }
    binding.clean_source(pending.config)
    _unchanged(snapshot)
    return binding.Cleanup(pending.config_output, original, clean, duration), snapshot


def _unchanged(inputs: dict[str, str]) -> None:
    if any(file_digest(Path(path)) != digest for path, digest in inputs.items()):
        raise ReplayError("Pending source binding inputs changed during validation")


def _validate(pending: Pending, cleanup: binding.Cleanup) -> None:
    binding.validate_binding(
        pending.binding_output,
        consumer_inputs=(pending.binding_output, cleanup.clean, pending.config_output),
        consumer_command=(
            "python",
            "-m",
            "replay.fcp_transformed_source",
            str(pending.config_output),
        ),
    )


def bind(request: Path, *, sha256: str) -> Path:
    """Publish only genuine completed pins; existing output artifacts are never rewritten."""
    pending = _load(request, sha256)
    if any(
        path.exists() for path in (pending.config_output, pending.binding_output, pending.receipt)
    ):
        raise ReplayError("Pending binding output already exists; verify retained evidence")
    cleanup, inputs = _evidence(pending)
    if inputs[str(request)] != sha256:
        raise ReplayError("Pending request changed during validation")
    _unchanged(inputs)
    _save(pending.config_output, pending.config)
    binding.create_binding(
        cleanup,
        consumer=pending.config_output,
        kind="transformed-source-config",
        output=pending.binding_output,
    )
    _validate(pending, cleanup)
    _unchanged(inputs)
    _save(
        pending.receipt,
        {
            "version": 1,
            "kind": "actual-source-binding",
            "request": _reference(request),
            "config": _reference(pending.config_output),
            "binding": _reference(pending.binding_output),
            "inputs": inputs,
            "readyForRender": False,
        },
    )
    return pending.receipt


def verify(request: Path, *, sha256: str) -> Path:
    """Recheck original pending intent, completions, provenance and consumer binding."""
    pending = _load(request, sha256)
    cleanup, inputs = _evidence(pending)
    if inputs[str(request)] != sha256:
        raise ReplayError("Pending request changed during validation")
    if _json(pending.config_output) != pending.config:
        raise ReplayError("Bound transform config differs from pending intent")
    _validate(pending, cleanup)
    expected: dict[str, object] = {
        "version": 1,
        "kind": "actual-source-binding",
        "request": _reference(request),
        "config": _reference(pending.config_output),
        "binding": _reference(pending.binding_output),
        "inputs": inputs,
        "readyForRender": False,
    }
    if _json(pending.receipt) != expected:
        raise ReplayError("Pending binding completion evidence changed")
    _unchanged(inputs)
    return pending.receipt


def prepare_graph(
    request: Path, *, sha256: str, captures: tuple[Path, ...], template: Path, output: Path
) -> Path:
    """Prepare barriers, actual binding and the existing ordinary consumer stages."""
    data = _intent(request, sha256)
    _local(str(output))
    if not output.is_dir() or tuple(output.iterdir()) or not 1 <= len(captures) <= 2:
        raise ReplayError("Require empty existing metadata root and one or two actor captures")
    consumer = _json(template)
    workflow.load_stages(template)
    code = tuple(item.path for item in inventory.capture(("pending_source_binding",)))
    stages: list[dict[str, object]] = []
    proofs: list[Path] = []
    for number, capture in enumerate(captures):
        captured = _json(capture)
        producer = _pin(captured.get("workflow"))
        producer_stages = barrier._stages(barrier.Pin(producer, file_digest(producer)))
        nested = tuple(sorted(barrier._workflow_paths(producer_stages)))
        declared = {path for stage in producer_stages for path in stage.inputs}
        for child in nested:
            child_stages = barrier._stages(barrier.Pin(child, file_digest(child)))
            declared.update(path for stage in child_stages for path in stage.inputs)
        sources = sorted(
            path
            for path in declared
            if path.suffix in barrier.CODE_SUFFIXES or path.name == "METADATA"
        )
        barrier_path = output / f"barrier-{number:02d}.json"
        proof = output / f"released-{number:02d}.json"
        _save(
            barrier_path,
            {
                "version": 1,
                "kind": "completion-barrier",
                "workflow": _reference(producer),
                "capture": _reference(capture),
                "stateDirectory": str(_local(captured.get("checkpoint")).parent),
                "terminalStatus": captured.get("statusPath"),
                "lastStage": captured.get("stage"),
                "codeInputs": [_reference(path) for path in sources],
                "nestedWorkflows": [_reference(path) for path in nested],
                "waitSeconds": 28800,
                "pollSeconds": 30,
                "output": str(proof),
            },
        )
        barrier.load(barrier_path, file_digest(barrier_path))
        command = [str(ROOT / ".venv/bin/python"), "-m", "replay.completion_barrier"]
        arguments = [str(barrier_path), "--sha256", file_digest(barrier_path)]
        stages.append(
            {
                "name": f"wait-producer-{number:02d}",
                "inputs": [
                    str(path)
                    for path in dict.fromkeys(
                        (barrier_path, capture, producer, *nested, *sources, *code)
                    )
                ],
                "outputs": [str(proof)],
                "command": [*command, "wait", *arguments],
                "verify_command": [*command, "verify", *arguments],
                "timeout_seconds": 30000,
            }
        )
        proofs.append(proof)
    fixed = record(data["fixedFields"])
    config = _local(data.get("configOutput"))
    bound = _local(data.get("bindingOutput"))
    future = tuple(_local(value) for value in record(data["futurePinFields"]).values())
    completions = data.get("requiredCompletions")
    if not isinstance(completions, list) or len(completions) != 2:
        raise ReplayError("Require two pending completions")
    inputs = (
        request,
        *proofs,
        *future,
        *(_local(value) for value in completions),
        _pin(fixed["voice"]),
        _pin(fixed["voiceReceipt"]),
        *code,
    )
    command = [str(ROOT / ".venv/bin/python"), "-m", "replay.pending_source_binding"]
    arguments = [str(request), "--sha256", sha256]
    binding_stage: dict[str, object] = {
        "name": "bind-actual-source",
        "inputs": [str(path) for path in dict.fromkeys(inputs)],
        "outputs": [str(config), str(bound), str(bound.with_suffix(".pending-source.json"))],
        "command": [*command, "bind", *arguments],
        "verify_command": [*command, "verify", *arguments],
        "timeout_seconds": 7200,
    }
    raw_stages = consumer.get("stages")
    if not isinstance(raw_stages, list):
        raise ReplayError("Consumer template has no stages")
    child = output / "child-workflow.json"
    _save(child, {**consumer, "stages": [binding_stage, *raw_stages]})
    child_stages = workflow.load_stages(child)
    child_state = output / "child-state"
    command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "replay.workflow",
        str(child),
        "--state",
        str(child_state),
    ]
    stages.append(
        {
            "name": "run-clean-consumer",
            "inputs": [str(path) for path in (child, request, *proofs, *code)],
            "outputs": [str(path) for stage in child_stages for path in stage.outputs]
            + [str(child_state / f"{child_stages[-1].name}.json")],
            "command": command,
            "verify_command": command,
            "timeout_seconds": 15000,
        }
    )
    _save(output / "workflow.json", {"version": 1, "stages": stages})
    parsed = workflow._load_workflow(output / "workflow.json")
    workflow._check_external_inputs(parsed.stages, parsed.paths)
    return output / "workflow.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bind", "verify", "prepare"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--capture", type=Path, action="append", default=[])
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.template is None or args.output is None:
            parser.error("prepare requires --template and --output")
        print(
            prepare_graph(
                args.request,
                sha256=args.sha256,
                captures=tuple(args.capture),
                template=args.template,
                output=args.output,
            )
        )
    else:
        print({"bind": bind, "verify": verify}[args.mode](args.request, sha256=args.sha256))


if __name__ == "__main__":
    main()
