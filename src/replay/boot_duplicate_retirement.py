"""Exact BOOT-duplicate retirement adapter; no discovery by file names."""

import argparse
import ast
import hashlib
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from replay import compressed_part_retirement as kernel
from replay import fcp_native_admission as admission
from replay import part_retirement_verification_batch as batch
from replay import part_retirement_verification_readonly as readonly
from replay.native_sequence import ContractError
from replay.sequence_adapters import path_value, pin_value, reference, save_identical
from replay.sequence_journal import Pin, bounded_read, decode, encode, publish
from replay.stages import execution_lock


@dataclass(frozen=True)
class Plan:
    request: Pin
    root: Path
    boot: Path
    retirement: Pin
    verification: Pin
    mandatory: Pin
    disposition: Pin
    archive: tuple[Pin, ...]
    code: tuple[Pin, ...]
    leases: tuple[Path, ...]
    source: Path
    survivor: Path
    part: int


class Guards(Protocol):
    def before(self, plan: Plan) -> None: ...

    def after(self, plan: Plan) -> Pin: ...

    def unused(self, source: Path) -> None: ...

    def birth(self, pid: int) -> str: ...

    def now_ns(self) -> int: ...


class OriginalGuards:
    def before(self, plan: Plan) -> None:
        request = readonly.load(plan.mandatory.path)
        phase, parts, _ = batch._manifest(request.manifest.path, request.manifest.sha256)
        if (
            phase != "before-delete"
            or len(parts) != 1
            or parts[0].number != plan.part
            or parts[0].config.path != plan.verification.path
            or parts[0].config.sha256 != plan.verification.sha256
        ):
            raise ContractError("Mandatory completed domain is not this exact BOOT part")
        readonly.verify(request)

    def after(self, plan: Plan) -> Pin:
        manifest = plan.root / "after-delete-manifest.json"
        if not manifest.exists():
            batch.prepare((plan.verification.path,), manifest, phase="after-delete")
        bound = Pin(manifest, reference(manifest)["sha256"])
        phase, parts, _ = batch._manifest(bound.path, bound.sha256)
        if (
            phase != "after-delete"
            or len(parts) != 1
            or parts[0].config.path != plan.verification.path
            or parts[0].config.sha256 != plan.verification.sha256
        ):
            raise ContractError("Occupied post-delete domain differs")
        output = plan.root / "after-delete"
        result = batch.run(bound.path, bound.sha256, output, resume=output.exists())
        save_identical(
            plan.root / "after-delete-readonly.json",
            {
                "version": 1,
                "kind": "readonly-retirement-verification",
                "manifest": reference(manifest),
                "operation": reference(output / "operation.json"),
                "receipt": reference(result),
                "checkpoints": [
                    {
                        "part": plan.part,
                        "artifact": reference(output / f"part-{plan.part:04d}.json"),
                    }
                ],
            },
        )
        return Pin(result, reference(result)["sha256"])

    def unused(self, source: Path) -> None:
        kernel._unused(source)

    def birth(self, pid: int) -> str:
        return admission.SystemHost().birth(pid)

    def now_ns(self) -> int:
        return time.time_ns()


def code_paths() -> tuple[Path, ...]:
    """Bound this adapter's actual import closure, including original lazy imports."""
    private = Path(__file__).resolve().parent
    pending = {Path(__file__).resolve(), kernel.ROOT / "src/replay/__init__.py"}
    found: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if len(found) >= 512:
            raise ContractError("Retirement code closure exceeds its bounded scope")
        found.add(path)
        for node in ast.walk(ast.parse(bounded_read(path).decode("utf-8"))):
            names: tuple[str, ...] = ()
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1 or path.parent != kernel.ROOT / "src/replay":
                    raise ContractError("Relative import escapes the replay package")
                names = (
                    (f"replay.{node.module}",)
                    if node.module is not None
                    else tuple(f"replay.{alias.name}" for alias in node.names)
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "replay":
                names = tuple(f"replay.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module,)
            elif isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            for name in names:
                if name.startswith("replay."):
                    pending.add(kernel.ROOT / "src" / (name.replace(".", "/") + ".py"))
                elif (private / f"{name}.py").is_file():
                    pending.add(private / f"{name}.py")
    return tuple(sorted(found))


def load(path: Path) -> Plan:
    request = Pin(path, reference(path)["sha256"])
    raw = decode(bounded_read(path))
    if (
        set(raw)
        != {
            "version",
            "kind",
            "root",
            "boot",
            "retirement",
            "verification",
            "mandatoryBefore",
            "disposition",
            "archive",
            "code",
            "leases",
        }
        or raw.get("version") != 1
        or raw.get("kind") != "exact-boot-duplicate-retirement"
    ):
        raise ContractError("Unsupported explicit BOOT retirement request")
    root, boot = path_value(raw["root"]), path_value(raw["boot"])
    retirement, verification = pin_value(raw["retirement"]), pin_value(raw["verification"])
    retired = decode(bounded_read(retirement.path))
    if (
        retired.get("kind") != "verified-staging-retirement"
        or retired.get("stagingRetirementAuthorized") is not True
        or retired.get("deletionPerformed") is not False
    ):
        raise ContractError("Original explicit staging retirement authority required")
    verify = decode(bounded_read(verification.path))
    if verify.get("retirement") != reference(retirement.path):
        raise ContractError("Verification does not bind original retirement")
    source_ref, survivor_ref = retired.get("source"), retired.get("movie")
    if not isinstance(source_ref, dict) or not isinstance(survivor_ref, dict):
        raise ContractError("Exact source and surviving publication required")
    source, survivor = kernel._ref(source_ref).path, kernel._ref(survivor_ref).path
    part = retired.get("part")
    if type(part) is not int or part <= 0 or not isinstance(part, int):
        raise ContractError("Positive ordered part required")
    work = kernel.ROOT / "edits" / "work"
    if (
        not source.is_relative_to(work)
        or source == survivor
        or not root.is_relative_to(work)
        or not root.is_dir()
        or not boot.is_dir()
        or boot.stat().st_dev != kernel.ROOT.stat().st_dev
    ):
        raise ContractError("Only exact redundant BOOT work copy is eligible")
    archives, leases = raw["archive"], raw["leases"]
    if (
        not isinstance(archives, list)
        or not 1 <= len(archives) <= 1024
        or not isinstance(leases, list)
        or not 1 <= len(leases) <= 64
    ):
        raise ContractError("Explicit archival evidence and ownership scopes required")
    locked = tuple(path_value(value) for value in leases)
    if len(set(locked)) != len(locked) or any(
        not item.is_dir() or not item.is_relative_to(work) for item in locked
    ):
        raise ContractError("Invalid declared workflow ownership roots")
    archive = tuple(pin_value(value) for value in archives)
    required_archive = tuple(
        pin_value(retired.get(name))
        for name in ("publication", "completion", "frameProof", "intent", "config")
    )
    if not set(required_archive) <= set(archive):
        raise ContractError(
            "Original publication/completion/proof/intent/config must remain archived"
        )
    code_raw = raw["code"]
    if not isinstance(code_raw, list) or not 1 <= len(code_raw) <= 512:
        raise ContractError("Explicit current code closure is required")
    code = tuple(pin_value(value) for value in code_raw)
    if len({ref.path for ref in code}) != len(code) or {ref.path for ref in code} != set(
        code_paths()
    ):
        raise ContractError("Exact current retirement import closure required")
    return Plan(
        request,
        root,
        boot,
        retirement,
        verification,
        pin_value(raw["mandatoryBefore"]),
        pin_value(raw["disposition"]),
        archive,
        code,
        locked,
        source,
        survivor,
        part,
    )


def disposition(plan: Plan, guards: Guards, observation: Pin | None = None) -> tuple[Pin, ...]:
    selected = plan.disposition if observation is None else observation
    selected.verify()
    raw = decode(bounded_read(selected.path))
    retired = decode(bounded_read(plan.retirement.path))
    if (
        set(raw) != {"version", "source", "owner", "ownerReleased", "observedAtNs", "references"}
        or raw.get("version") != 1
        or raw.get("source") != retired.get("source")
        or raw.get("ownerReleased") is not True
    ):
        raise ContractError("Exact owner source-release disposition required")
    owner = raw["owner"]
    if (
        not isinstance(owner, dict)
        or set(owner) != {"pid", "birth"}
        or type(owner["pid"]) is not int
        or not isinstance(owner["pid"], int)
        or not isinstance(owner["birth"], str)
        or guards.birth(owner["pid"]) != owner["birth"]
    ):
        raise ContractError("Disposition owner process identity changed")
    instant = raw["observedAtNs"]
    if (
        type(instant) is not int
        or not isinstance(instant, int)
        or not 0 < instant <= guards.now_ns()
        or guards.now_ns() - instant > kernel.MAX_AGE_NS
    ):
        raise ContractError("Disposition is stale or future dated")
    scopes = raw["references"]
    if not isinstance(scopes, dict) or set(scopes) != kernel.SCOPES:
        raise ContractError("All current/queued/resume/FCP/Viewer/diagnostic scopes required")
    refs: list[Pin] = []
    for row in scopes.values():
        if (
            not isinstance(row, dict)
            or set(row) != {"scope", "resolved", "requiresMaterializedSource", "evidence"}
            or row["resolved"] is not True
            or row["requiresMaterializedSource"] is not False
            or not isinstance(row["scope"], str)
            or not row["scope"]
        ):
            raise ContractError("Unresolved or active source dependency")
        evidence = row["evidence"]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 1024:
            raise ContractError("Every dependency scope needs concrete pinned evidence")
        refs.extend(pin_value(value) for value in evidence)
    return tuple(refs)


def resume_observation(plan: Plan, observation: Pin, guards: Guards) -> tuple[Pin, ...]:
    """A fresh post-only owner observation never replaces original effect authority."""
    observation.verify()
    raw = decode(bounded_read(observation.path))
    operation = plan.root / "unlink-operation.json"
    expected = {
        "version": 1,
        "kind": "exact-boot-post-only-resume",
        "request": reference(plan.request.path),
        "operation": reference(operation),
        "source": decode(bounded_read(plan.retirement.path)).get("source"),
        "survivorIdentity": list(kernel._identity(plan.survivor)),
        "archive": [{"path": str(ref.path), "sha256": ref.sha256} for ref in plan.archive],
        "code": [{"path": str(ref.path), "sha256": ref.sha256} for ref in plan.code],
        "leases": [str(path) for path in plan.leases],
    }
    if set(raw) != {*expected, "disposition"} or any(
        raw.get(key) != value for key, value in expected.items()
    ):
        raise ContractError("Fresh resume must bind the exact immutable original operation")
    fresh = pin_value(raw["disposition"])
    _absent(plan.source)
    return (
        observation,
        fresh,
        Pin(operation, reference(operation)["sha256"]),
        *disposition(plan, guards, fresh),
    )


def _absent(source: Path) -> None:
    if source.exists() or source.is_symlink():
        raise ContractError("Actual absence is required before post-delete verification")


def commit(
    plan: Plan, guards: Guards, *, resume: Pin | None = None, post_only: bool = False
) -> Path:
    """One exact unlink under original domain, ownership and FD checks; resumable post stage."""
    operation = plan.root / "unlink-operation.json"
    released = plan.root / "released.json"
    with ExitStack() as stack:
        name = hashlib.sha256(str(plan.source).encode()).hexdigest()
        for directory, key in ((plan.root, name), *((item, ".workflow") for item in plan.leases)):
            if not stack.enter_context(execution_lock(directory, key)):
                raise ContractError("Retirement or a declared current workflow is active")
        refs = (
            plan.request,
            plan.retirement,
            plan.verification,
            plan.mandatory,
            plan.disposition,
            *plan.archive,
            *plan.code,
        )
        for ref in refs:
            ref.verify()
        if post_only or resume is not None:
            _absent(plan.source)
            if not operation.is_file():
                raise ContractError("Post-only continuation requires the original unlink operation")
        live = (
            disposition(plan, guards)
            if resume is None
            else resume_observation(plan, resume, guards)
        )
        if operation.exists():
            prior = decode(bounded_read(operation))
            if (
                set(prior)
                != {
                    "version",
                    "kind",
                    "request",
                    "sourceIdentity",
                    "survivorIdentity",
                    "parentIdentity",
                }
                or prior.get("version") != 1
                or prior.get("kind") != "exact-boot-unlink-operation"
                or prior.get("request") != reference(plan.request.path)
                or prior.get("survivorIdentity") != list(kernel._identity(plan.survivor))
            ):
                raise ContractError("Occupied operation belongs to another request")
            _absent(plan.source)
        else:
            source, survivor = kernel._identity(plan.source), kernel._identity(plan.survivor)
            receipt = decode(bounded_read(plan.retirement.path))
            if (
                list(source[:4]) != receipt.get("sourceIdentity")
                or list(survivor[:4]) != receipt.get("destinationIdentity")
                or source[:2] == survivor[:2]
                or source[0] != plan.boot.stat().st_dev
            ):
                raise ContractError("Exact BOOT/survivor identities differ from original proof")
            guards.before(plan)
            if (
                kernel._identity(plan.source) != source
                or kernel._identity(plan.survivor) != survivor
            ):
                raise ContractError("Domain verification changed media identities")
            disposition(plan, guards)
            guards.unused(plan.source)
            parent = os.open(plan.source.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, parent)
            parent_info = os.fstat(parent)
            descriptor = os.open(
                plan.source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            stack.callback(os.close, descriptor)
            opened = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
            ) != source:
                raise ContractError("Opened source FD differs")
            for ref in (*refs, *live):
                ref.verify()
            intent = {
                "version": 1,
                "kind": "exact-boot-unlink-operation",
                "request": reference(plan.request.path),
                "sourceIdentity": list(source),
                "survivorIdentity": list(survivor),
                "parentIdentity": [parent_info.st_dev, parent_info.st_ino],
            }
            publish(operation, encode(intent))
            named = os.stat(plan.source.name, dir_fd=parent, follow_symlinks=False)
            if (
                (
                    named.st_dev,
                    named.st_ino,
                    named.st_size,
                    named.st_mtime_ns,
                    named.st_ctime_ns,
                    named.st_nlink,
                )
                != source
                or kernel._identity(plan.source) != source
                or kernel._identity(plan.survivor) != survivor
                or (plan.source.parent.stat().st_dev, plan.source.parent.stat().st_ino)
                != (parent_info.st_dev, parent_info.st_ino)
            ):
                raise ContractError("Exact source namespace changed before commit")
            os.unlink(plan.source.name, dir_fd=parent)
            os.fsync(parent)
            _absent(plan.source)
        # This original AFTER domain is never called while the source exists.
        post = guards.after(plan)
        post.verify()
        _absent(plan.source)
        for ref in (*refs, *live):
            ref.verify()
        if resume is not None:
            resume_observation(plan, resume, guards)
            save_identical(
                plan.root / f"resume-{resume.sha256}.json",
                {
                    "version": 1,
                    "kind": "verified-post-only-resume-observation",
                    "observation": {"path": str(resume.path), "sha256": resume.sha256},
                    "operation": reference(operation),
                    "postDelete": reference(post.path),
                },
            )
        # A successful original post-domain return maps to the existing release
        # consumer's status contract; no pre-delete process exit is reused here.
        status = plan.root / "post-delete.status"
        if status.exists():
            if bounded_read(status) != b"0\n":
                raise ContractError("Occupied post-delete status differs")
        else:
            publish(status, b"0\n")
        receipt = decode(bounded_read(plan.retirement.path))
        survivor_ref = kernel._ref(receipt.get("movie"))
        audit = plan.root / "retirement-audit.json"
        save_identical(
            audit,
            {
                "version": 1,
                "part": plan.part,
                "stagingPath": str(plan.source),
                "stagingAbsent": True,
                "historyUnchanged": True,
                "retirementSHA256": plan.retirement.sha256,
                "qualityApproved": False,
                "prepareConsumerPostDeleteExitCodes": [0],
                "survivingMovie": str(survivor_ref.path),
                "survivingSHA256": survivor_ref.sha256,
                "postDeleteDomain": reference(post.path),
                "statusMeaning": "Successful original after-delete batch API return",
            },
        )
        save_identical(
            released,
            {
                "version": 1,
                "kind": "exact-boot-retirement-release",
                "request": reference(plan.request.path),
                "operation": reference(operation),
                "postDelete": reference(post.path),
                "postDeleteStatus": reference(status),
                "retirementAudit": reference(audit),
                "sourceAbsent": True,
                "qualityApproved": False,
            },
        )
        return released


def prepare_graph(request: Path, output: Path, python: Path, *, resume: Pin | None = None) -> Path:
    plan = load(request)
    command = [str(python), "-B", str(Path(__file__).resolve()), "execute", str(request)]
    if resume is not None:
        resume.verify()
        command.extend(["--resume-observation", str(resume.path), "--resume-sha256", resume.sha256])
    inputs = [
        str(ref.path)
        for ref in (
            plan.request,
            plan.retirement,
            plan.verification,
            plan.mandatory,
            plan.disposition,
            *plan.archive,
            *plan.code,
        )
    ]
    code = (
        Path(__file__).resolve(),
        Path(kernel.__file__).resolve(),
        Path(batch.__file__).resolve(),
        Path(readonly.__file__).resolve(),
    )
    inputs.extend(str(path) for path in code)
    if resume is not None:
        inputs.append(str(resume.path))
        inputs.append(str(pin_value(decode(bounded_read(resume.path)).get("disposition")).path))
    save_identical(
        output,
        {
            "version": 1,
            "stages": [
                {
                    "name": "exact-boot-retirement",
                    "command": command,
                    "verify_command": [*command[:3], "verify", *command[4:]],
                    "adopt_existing": resume is not None,
                    "inputs": sorted(set(inputs)),
                    "outputs": [str(plan.root / "released.json")],
                    "timeout_seconds": 3600,
                }
            ],
        },
    )
    return output


def verify(plan: Plan) -> Path:
    """Read-only acceptance; never calls the post-delete producer or unlink."""
    _absent(plan.source)
    for ref in (
        plan.request,
        plan.retirement,
        plan.verification,
        plan.mandatory,
        plan.disposition,
        *plan.archive,
        *plan.code,
    ):
        ref.verify()
    released = plan.root / "released.json"
    result = decode(bounded_read(released))
    operation = plan.root / "unlink-operation.json"
    original = decode(bounded_read(operation))
    if (
        result.get("kind") != "exact-boot-retirement-release"
        or result.get("request") != reference(plan.request.path)
        or result.get("operation") != reference(operation)
        or result.get("sourceAbsent") is not True
        or original.get("request") != reference(plan.request.path)
        or original.get("survivorIdentity") != list(kernel._identity(plan.survivor))
    ):
        raise ContractError("Complete original unlink and post-delete release required")
    for name in ("postDelete", "postDeleteStatus", "retirementAudit"):
        pin_value(result.get(name)).verify()
    if bounded_read(pin_value(result.get("postDeleteStatus")).path) != b"0\n":
        raise ContractError("Post-delete status is not successful")
    request = readonly.load(plan.root / "after-delete-readonly.json")
    if request.receipt.path != pin_value(result.get("postDelete")).path:
        raise ContractError("Readonly post-domain differs from the released receipt")
    phase, parts, _ = batch._manifest(request.manifest.path, request.manifest.sha256)
    if (
        phase != "after-delete"
        or len(parts) != 1
        or parts[0].number != plan.part
        or parts[0].config.path != plan.verification.path
        or parts[0].config.sha256 != plan.verification.sha256
    ):
        raise ContractError("Readonly post-domain is not the exact retired part")
    readonly.verify(request)
    _absent(plan.source)
    return released


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("execute", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--resume-observation", type=Path)
    parser.add_argument("--resume-sha256")
    args = parser.parse_args(argv)
    plan = load(args.request)
    if (args.resume_observation is None) != (args.resume_sha256 is None):
        raise ContractError("Resume observation requires its exact pinned SHA256")
    resume = (
        None
        if args.resume_observation is None
        else pin_value({"path": str(args.resume_observation), "sha256": args.resume_sha256})
    )
    if args.operation == "verify":
        _absent(plan.source)
        # Verification can never become a producer after an interrupted pre-delete intent.
        if (
            not (plan.root / "unlink-operation.json").is_file()
            or not (plan.root / "released.json").is_file()
        ):
            raise ContractError("Complete original unlink and post-delete release required")
    print(
        verify(plan)
        if args.operation == "verify"
        else commit(plan, OriginalGuards(), resume=resume)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
