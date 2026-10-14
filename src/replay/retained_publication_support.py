"""Verify retained assembly provenance across a terminal publication boundary."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from replay import cleanup_finalization_workflow as common
from replay import cleanup_lane_successor as original
from replay import code_dependency_inventory as inventory
from replay import recovery_source_finalization as recovery
from replay import retained_cleanup_finalization as retained
from replay import workflow
from replay.fcp_part_assembly import _save
from replay.recordings import record
from replay.stages import Stage, _fingerprint


def child_document(job: common.Job) -> dict[str, object]:
    """Derive the ordinary child without relaxing any workflow output guard.

    The original factory also validates already-published output link counts.
    Publication repair needs its exact definition before that repair. The common
    factory remains the source of the command, tools, inputs and output paths.
    """
    data = common._child(job)
    stage = record(original._list(data["stages"])[0])
    removed = {str(job.root / "finalized" / name) for name in common.FINAL_FILES}
    stage["inputs"] = (
        [name for name in original._strings(stage["inputs"]) if name not in removed]
        + [str(job.root / "finalized" / name) for name in recovery.FILES]
        + [str(job.root / "finalized.json")]
    )
    stage["verify_command"] = retained._command("verify-assembly", job.operation)
    return {"version": 1, "stages": [stage]}


def original_stage(job: common.Job) -> Stage:
    """Require the immutable child graph and prepared checkpoint to match."""
    graph = job.root / "child/workflow.json"
    if common._read(graph) != child_document(job):
        raise ValueError("Retained publication child definition changed")
    expected = {
        "version": 1,
        "protocol": retained.PROTOCOL,
        "operation": common._pin(job.operation),
        "workflow": common._pin(graph),
        "finalized": common._pin(job.root / "finalized.json"),
    }
    if common._read(job.root / "prepared.json") != expected:
        raise ValueError("Retained publication prepared checkpoint changed")
    # Parse this exact definition only. A normal graph is validated after the
    # approved alias retirement; its single-link output guard stays unchanged.
    return workflow._stage(record(original._list(child_document(job)["stages"])[0]))


def check_pending_child(job: common.Job, checkpoint: Path) -> None:
    """A terminal actor's unchanged running checkpoint is never success evidence."""
    if checkpoint != job.root / "child-state/assemble-clean-source.json":
        raise ValueError("Retained publication checkpoint belongs to another child")
    if common._read(checkpoint) != {
        "status": "running",
        "fingerprint": _fingerprint(original_stage(job)),
    }:
        raise ValueError("Retained publication pending child fingerprint changed")
    common._assembled(job)


def pending_job(operation: Path, checkpoint: Path) -> common.Job:
    """Load the exact retained operation and its unchanged pending child."""
    job = retained._load(operation)
    check_pending_child(job, checkpoint)
    return job


@contextmanager
def pending_child_ready(operation: Path, checkpoint: Path) -> Iterator[common.Job]:
    """Keep all original generation, source, model, native and assembly guards."""
    job = retained._load(operation)
    with retained._ready(job) as data:
        if common._read(job.root / "finalized.json") != retained._seal(job, data, create=False):
            raise ValueError("Retained publication finalized checkpoint changed")
        check_pending_child(job, checkpoint)
        yield job
        check_pending_child(job, checkpoint)


def adopted_document(job: common.Job, extra_inputs: tuple[Path, ...]) -> dict[str, object]:
    """Register existing media through the original readonly assembly verifier."""
    stage = original_stage(job)
    ordinary = retained._command("verify-assembly", job.operation)
    command = [*ordinary[:3], "replay.code_inventory_memo_run", "run", *ordinary[3:]]
    return {
        "version": 1,
        "stages": [
            {
                "name": stage.name,
                "inputs": [str(path) for path in dict.fromkeys((*stage.inputs, *extra_inputs))],
                "outputs": [str(path) for path in stage.outputs],
                "command": command,
                "verify_command": command,
                "timeout_seconds": stage.timeout_seconds,
                "adopt_existing": True,
            }
        ],
    }


def child_inputs() -> tuple[Path, ...]:
    """Capture the current publication helper dependency closure."""
    return tuple(
        row.path
        for row in inventory.capture(
            ("publication_completion_seal", "verified_hardlink_publication")
        )
    )


def prepare_adoption(operation: Path, output: Path) -> Path:
    """Prepare fresh readonly metadata; its normal check follows alias retirement."""
    root = Path(__file__).resolve().parents[2] / "edits/work"
    if output != output.resolve() or not output.is_relative_to(root):
        raise ValueError("Adoption metadata must remain canonical checkout work")
    job = retained._load(operation)
    if output in {operation, job.root / "child/workflow.json"}:
        raise ValueError("Adoption metadata cannot replace original producer graph")
    _save(output, adopted_document(job, child_inputs()))
    return output


def completed(
    job: common.Job, graph: Path, state: Path, extra_inputs: tuple[Path, ...]
) -> dict[str, object]:
    """Require a genuine new normal child receipt, preserving the old checkpoint."""
    if common._read(graph) != adopted_document(job, extra_inputs):
        raise ValueError("Retained publication adopted child definition changed")
    stage = workflow.load_stages(graph)[0]
    original._complete(stage, state)
    result = common._assembled(job)
    result["protocol"] = retained.PROTOCOL
    result["childStageReceipt"] = common._pin(state / f"{stage.name}.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare-adoption",))
    parser.add_argument("operation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_adoption(args.operation, args.output))


if __name__ == "__main__":
    main()
