"""Seal a genuinely completed assembly after a proven publication-alias repair."""

import argparse
from contextlib import ExitStack
from pathlib import Path

from replay import cleanup_finalization_workflow as common
from replay import cleanup_lane_successor as locks
from replay import code_inventory_memo_run as memo
from replay import retained_cleanup_finalization as retained
from replay import retained_publication_support as publication
from replay import subtitle_session_cleanup_finalization as finalization
from replay import verified_hardlink_publication as retirement
from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference, _save
from replay.input_hash_cache import InputHashCache
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
FIELDS = frozenset(
    {
        "version",
        "kind",
        "operation",
        "retirementRequest",
        "completionOutput",
        "proofOutput",
        "leases",
    }
)
RETAINED_FIELDS = FIELDS | {"childWorkflow", "childState"}
Job = finalization.Job | common.Job


def _local(value: object) -> Path:
    path = _path(value)
    if not path.is_relative_to(ROOT / "edits/work"):
        raise ReplayError("Completion seal paths must remain checkout-local")
    return path


def _cache_paths(job: Job, request: Path) -> tuple[Path, ...]:
    manifest = _json(job.root / "finalized/manifest.json")
    index = _pin(manifest.get("index"))
    receipt = _json(job.root / "evidence/receipt.json")
    rows = _json(index).get("ranges")
    chunks = manifest.get("chunks")
    if (
        not isinstance(rows, list)
        or not isinstance(chunks, list)
        or not 1 <= len(rows) == len(chunks) <= 512
    ):
        raise ReplayError("Seal requires bounded matching assembly chunk references")
    paths = [
        request,
        job.operation,
        index,
        job.root / "clean-source.mov",
        job.root / "prepared.json",
        job.root / "finalized.json",
        job.root / "finalized/manifest.json",
        job.root / "child/workflow.json",
        job.root / "child-state/assemble-clean-source.json",
        job.root / "evidence/receipt.json",
        job.root / "evidence/single-source-index.json",
    ]
    paths.extend(_path(record(row).get("media")) for row in rows)
    paths.extend(
        _path(record(record(chunk).get(key)).get("path"))
        for chunk in chunks
        for key in ("candidate", "producer")
    )
    paths.extend(_path(path) for path in record(receipt.get("probeEvidence")))
    paths.extend(Path(path) for path in job.code)
    return tuple(dict.fromkeys(paths))


def retained_child_inputs() -> tuple[Path, ...]:
    """Bind the current publication helpers and their complete source closure."""
    return publication.child_inputs()


def _retained_expected(job: common.Job, graph: Path, state: Path) -> dict[str, object]:
    if common._read(job.root / "prepared.json") != retained._prepared(job):
        raise ReplayError("Retained completion prepared checkpoint changed")
    return publication.completed(job, graph, state, retained_child_inputs())


def execute(request: Path, *, sha256: str, verify: bool = False) -> Path:
    """Publish the exact original completion schema without changing failed status."""
    _local(str(request))
    _pin({"path": str(request), "sha256": sha256})
    data = _json(request)
    version = data.get("version")
    if (
        set(data) != (RETAINED_FIELDS if version == 2 else FIELDS)
        or version not in (1, 2)
        or data.get("kind") != "publication-completion-seal"
    ):
        raise ReplayError("Invalid publication completion seal request")
    operation = _pin(data.get("operation"))
    retirement_request = _pin(data.get("retirementRequest"))
    completion = _local(data.get("completionOutput"))
    proof = _local(data.get("proofOutput"))
    if completion == proof or proof in {request, operation, retirement_request}:
        raise ReplayError("Completion seal paths overlap")
    leases = data.get("leases")
    if not isinstance(leases, list) or not 1 <= len(leases) <= 8:
        raise ReplayError("Require original producer writer leases")
    with ExitStack() as stack:
        for path in sorted({_local(value) for value in leases}):
            stack.enter_context(locks._unlocked(path))
        retired = retirement.execute(
            retirement_request, sha256=file_digest(retirement_request), verify=True
        )
        stack.enter_context(memo.capture_scope())
        job = retained._load(operation) if version == 2 else finalization._load(operation)
        if completion != job.root / "completion.json":
            raise ReplayError("Seal must use the genuine original completion namespace")
        graph: Path | None = None
        state: Path | None = None
        extra: tuple[Path, ...] = ()
        if isinstance(job, common.Job):
            graph = _pin(data.get("childWorkflow"))
            state = _local(data.get("childState"))
            if state == job.root / "child-state" or not state.is_relative_to(proof.parent):
                raise ReplayError("Retained seal requires a fresh child state under repair root")
            generation = stack.enter_context(retained._ready(job))
            if common._read(job.root / "finalized.json") != retained._seal(
                job, generation, create=False
            ):
                raise ReplayError("Retained completion finalized checkpoint changed")
            extra = (graph, state / "assemble-clean-source.json", *retained_child_inputs())
        else:
            stack.enter_context(finalization._ready(job))
        cache = InputHashCache((*_cache_paths(job, request), *extra), allow_uncached_overflow=True)
        # _ready owns the digest context. Prime and verify this additional
        # snapshot directly; original producer checks keep their existing cache.
        cache.prime()
        if isinstance(job, common.Job):
            if graph is None or state is None:
                raise ReplayError("Retained completion child proof is absent")
            expected = _retained_expected(job, graph, state)
        else:
            expected = finalization._completed(job)
        if verify:
            if _json(completion) != expected:
                raise ReplayError("Sealed completion differs from genuine child")
        else:
            if completion.exists() or proof.exists():
                raise ReplayError("Completion seal already exists; verify retained outputs")
            _save(completion, expected)
        if isinstance(job, common.Job):
            if (
                graph is None
                or state is None
                or _json(completion) != _retained_expected(job, graph, state)
            ):
                raise ReplayError("Retained completion differs from genuine adopted child")
        else:
            finalization._continue(job, "verify-run-child")
        inputs = cache.verify()
        evidence: dict[str, object] = {
            "version": 1,
            "kind": "verified-publication-completion-seal",
            "request": _reference(request),
            "operation": _reference(operation),
            "retirement": _reference(retired),
            "completion": _reference(completion),
            "inputs": inputs,
            "publicationBoundaryRepaired": True,
            "originalOuterStatusRewritten": False,
            "mediaReassembled": False,
            "qualityApproved": False,
        }
        if verify:
            if _json(proof) != evidence:
                raise ReplayError("Completion seal proof changed")
        else:
            _save(proof, evidence)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(execute(args.request, sha256=args.sha256, verify=args.mode == "verify"))


if __name__ == "__main__":
    main()
