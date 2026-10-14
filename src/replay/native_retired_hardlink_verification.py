"""Seal completed hardlink-retirement evidence without repeating media verification."""

import argparse
import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from replay.artifact_json import _json
from replay.core import ReplayError
from replay.fcp_part_assembly import _save
from replay.foreground_evidence import _bound_bytes
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.leaf_retirement import _row
from replay.recordings import record
from replay.stages import _fingerprint, file_digest
from replay.workflow import load_stages

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "edits/work"
MAX_BYTES = 8 * 1024 * 1024
MAX_PINS = 512


@dataclass(frozen=True)
class Reference:
    path: Path
    sha256: str

    def json(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class VerifiedRetirement:
    alias: Path
    keeper: Path
    owner: str
    original_registry: Reference
    pins: tuple[Reference, ...]


class Evidence:
    def __init__(self) -> None:
        self.pins: dict[Path, str] = {}

    def read(self, value: object) -> dict[str, object]:
        ref = reference(value)
        if len(self.pins) >= MAX_PINS or (
            ref.path in self.pins and self.pins[ref.path] != ref.sha256
        ):
            raise ReplayError("Conflicting or excessive retirement evidence")
        self.pins[ref.path] = ref.sha256
        try:
            return record(
                json.loads(_bound_bytes(ref.path, ref.sha256, MAX_BYTES), object_pairs_hook=_unique)
            )
        except ValueError as error:
            raise ReplayError("Pinned retirement evidence changed or is invalid") from error

    def verify(self) -> None:
        if any(file_digest(path) != digest for path, digest in self.pins.items()):
            raise ReplayError("Retirement evidence changed")


def _unique(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise ReplayError("Duplicate retirement JSON key")
        result[key] = value
    return result


def _text(value: object) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= 512 or "\0" in value:
        raise ReplayError("Expected bounded retirement text")
    return value


def _local(value: object) -> Path:
    text = _text(value)
    path = Path(text)
    if (
        not path.is_absolute()
        or not path.is_relative_to(WORK)
        or path.parent != path.parent.resolve()
    ):
        raise ReplayError("Retirement evidence requires canonical repository paths")
    return path


def reference(value: object) -> Reference:
    raw = record(value)
    digest = _text(raw.get("sha256"))
    if set(raw) != {"path", "sha256"} or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise ReplayError("Expected exact immutable retirement reference")
    return Reference(_local(raw.get("path")), digest)


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def _identity(path: Path) -> dict[str, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ReplayError("Retained keeper must be a regular single-link file")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "modifiedAtNs": info.st_mtime_ns,
        "changedAtNs": info.st_ctime_ns,
        "nlink": info.st_nlink,
    }


def _absent(alias: Path) -> None:
    try:
        alias.lstat()
    except FileNotFoundError:
        return
    raise ReplayError("Retired alias reappeared or is a dangling symlink")


def _original(evidence: Evidence, raw: dict[str, object]) -> tuple[Reference, str]:
    original = reference(raw.get("originalRegistry"))
    registry = evidence.read(original.json())
    outputs = registry.get("outputs")
    if (
        registry.get("kind") != "native-admission-registry"
        or registry.get("complete") is not True
        or not isinstance(outputs, list)
    ):
        raise ReplayError("Original complete admission registry is required")
    matches = [record(row) for row in outputs if record(row).get("path") == raw.get("alias")]
    if len(matches) != 1 or matches[0].get("state") != "retained":
        raise ReplayError("Original retained alias row is required")
    return original, _text(matches[0].get("owner"))


def _completed(evidence: Evidence, raw: dict[str, object]) -> dict[str, object]:
    workflow = reference(raw.get("workflow"))
    evidence.read(workflow.json())
    stages = [stage for stage in load_stages(workflow.path) if stage.name == raw.get("stage")]
    if len(stages) != 1 or not stages[0].verify_command:
        raise ReplayError("Exact stage with post-command verifier is required")
    stage = stages[0]
    request_ref = reference(raw.get("retirementRequest"))
    suffix = (
        "-m",
        "replay.verified_hardlink_publication",
        "verify",
        str(request_ref.path),
        "--sha256",
        request_ref.sha256,
    )
    run_suffix = (
        "-m",
        "replay.verified_hardlink_publication",
        "run",
        str(request_ref.path),
        "--sha256",
        request_ref.sha256,
    )
    if stage.verify_command[-6:] != suffix or stage.command[-6:] != run_suffix:
        raise ReplayError("Completed stage must invoke the exact existing hardlink verifier")
    checkpoint = evidence.read(raw.get("completedStage"))
    expected = {str(path): file_digest(path) for path in stage.outputs}
    if (
        set(checkpoint) != {"status", "fingerprint", "outputs"}
        or checkpoint.get("status") != "complete"
        or checkpoint.get("fingerprint") != _fingerprint(stage)
        or checkpoint.get("outputs") != expected
    ):
        raise ReplayError("Completed post-command stage or its input/output pins changed")
    return {
        "name": stage.name,
        "command": list(stage.command),
        "verifyCommand": list(stage.verify_command),
        "fingerprint": checkpoint["fingerprint"],
        "workingDirectory": str(Path.cwd().resolve()),
        "timeoutSeconds": stage.timeout_seconds,
        "inputs": {str(path): file_digest(path) for path in stage.inputs},
        "outputs": expected,
    }


def _domain(evidence: Evidence, raw: dict[str, object]) -> dict[str, int]:
    request_ref = reference(raw.get("retirementRequest"))
    request = evidence.read(request_ref.json())
    proof = evidence.read(raw.get("retirementProof"))
    alias, keeper = _local(raw.get("alias")), _local(raw.get("keeper"))
    if (
        request.get("kind") != "verified-hardlink-publication-retirement"
        or request.get("alias") != str(alias)
        or request.get("keeper") != str(keeper)
        or request.get("output") != reference(raw.get("retirementProof")).path.as_posix()
    ):
        raise ReplayError("Retirement request does not bind the exact endpoints")
    policy = evidence.read(request.get("policy"))
    keep_paths = policy.get("keepPaths")
    if (
        policy.get("eligiblePaths") != [str(alias)]
        or policy.get("eligibleTrees") != []
        or not isinstance(keep_paths, list)
        or str(keeper) not in keep_paths
    ):
        raise ReplayError("Owner policy must approve only this alias and retain its keeper")
    inventory = evidence.read(request.get("inventory"))
    rows = inventory.get("rows")
    if not isinstance(rows, list):
        raise ReplayError("Original hardlink inventory is required")
    matches = [
        record(row)
        for row in rows
        if record(row).get("path") in (str(alias), alias.relative_to(ROOT).as_posix())
    ]
    if len(matches) != 1:
        raise ReplayError("Original alias inventory must be unique")
    row = {**matches[0], "path": str(alias), "classification": "eligible"}
    metadata = _row(row)
    identity = _identity(keeper)
    if metadata.nlink != 2 or (metadata.dev, metadata.inode, metadata.size, metadata.mtime_ns) != (
        identity["device"],
        identity["inode"],
        identity["bytes"],
        identity["modifiedAtNs"],
    ):
        raise ReplayError("Keeper differs from the original hardlink inode")
    disposition = evidence.read(proof.get("disposition"))
    log = evidence.read(proof.get("retirementLog"))
    if disposition != {
        "version": 1,
        "kind": "verified-hardlink-alias-disposition",
        "request": request_ref.json(),
        "rows": [row],
    } or log != {"path": str(alias), "removed": True, "kind": "file", "bytes": metadata.size}:
        raise ReplayError("Retirement disposition or actual executor log changed")
    if (
        proof.get("kind") != "verified-hardlink-publication-retired"
        or proof.get("version") != 1
        or proof.get("request") != request_ref.json()
        or proof.get("alias") != str(alias)
        or proof.get("keeper") != _ref(keeper)
        or proof.get("logicalAliasBytes") != metadata.size
        or proof.get("physicalBytesReclaimed") != 0
        or proof.get("originalOuterStatusRewritten") is not False
        or proof.get("qualityApproved") is not False
    ):
        raise ReplayError("Actual hardlink retirement proof changed")
    _absent(alias)
    if _identity(keeper) != identity:
        raise ReplayError("Keeper changed while validating its content hash")
    return identity


def _seal(request: Path) -> dict[str, object]:
    """Seal actual completed verification, not merely an unlink command receipt."""
    evidence = Evidence()
    raw = evidence.read(_ref(_local(str(request))))
    if (
        set(raw)
        != {
            "version",
            "kind",
            "originalRegistry",
            "workflow",
            "stage",
            "completedStage",
            "retirementRequest",
            "retirementProof",
            "alias",
            "keeper",
        }
        or raw.get("version") != 1
        or raw.get("kind") != "native-retired-hardlink-seal-request"
    ):
        raise ReplayError("Unsupported retired hardlink seal request")
    original, owner = _original(evidence, raw)
    completed = _completed(evidence, raw)
    identity = _domain(evidence, raw)
    proof_outputs = record(completed["outputs"])
    proof_ref = reference(raw["retirementProof"])
    if proof_outputs.get(str(proof_ref.path)) != proof_ref.sha256:
        raise ReplayError("Retirement proof is not the completed stage output")
    evidence.verify()
    if _identity(_local(raw["keeper"])) != identity:
        raise ReplayError("Keeper changed before sealing")
    _absent(_local(raw["alias"]))
    result = {
        "version": 1,
        "kind": "native-verified-retired-hardlink",
        "request": _ref(request),
        "alias": raw["alias"],
        "keeper": raw["keeper"],
        "owner": owner,
        "originalRegistry": original.json(),
        "completedStage": completed,
        "keeperIdentity": identity,
        "keeperSHA256": record(evidence.read(raw["retirementProof"])["keeper"])["sha256"],
        "evidence": [
            {"path": str(path), "sha256": digest} for path, digest in evidence.pins.items()
        ],
        "qualityApproved": False,
        "physicalBytesReclaimed": 0,
    }
    return result


def seal(request: Path, output: Path) -> VerifiedRetirement:
    """Validate input bytes before and after creating a non-overwriting seal."""
    request = _local(str(request))
    evidence = Evidence()
    raw = evidence.read(_ref(request))
    workflow = reference(raw.get("workflow"))
    stages = [stage for stage in load_stages(workflow.path) if stage.name == raw.get("stage")]
    if len(stages) != 1:
        raise ReplayError("Require one exact retirement stage")
    paths = tuple(
        dict.fromkeys((request, workflow.path, _local(raw.get("keeper")), *stages[0].inputs))
    )
    cache = InputHashCache(tuple(path for path in paths if path.stat().st_size > 0))
    with input_hash_context(cache):
        cache.prime()
        result = _seal(request)
        cache.verify()
    _save(_local(str(output)), result)
    return verify(output, alias=_local(result["alias"]), owner=_text(result["owner"]))


def _historical_completed(
    evidence: Evidence, request: dict[str, object], completed: dict[str, object]
) -> None:
    workflow = reference(request.get("workflow"))
    evidence.read(workflow.json())
    stages = [stage for stage in load_stages(workflow.path) if stage.name == request.get("stage")]
    if len(stages) != 1:
        raise ReplayError("Sealed verifier stage no longer matches its pinned workflow")
    stage = stages[0]
    inputs = record(completed.get("inputs"))
    if set(inputs) != {str(path) for path in stage.inputs} or any(
        not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
        for value in inputs.values()
    ):
        raise ReplayError("Sealed historical input hashes are invalid")
    definition = {
        "command": stage.command,
        "verifyCommand": stage.verify_command,
        "inputs": inputs,
        "outputs": [str(path) for path in stage.outputs],
        "timeoutSeconds": stage.timeout_seconds,
        "workingDirectory": completed.get("workingDirectory"),
    }
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode("utf-8")).hexdigest()
    if (
        completed.get("name") != stage.name
        or completed.get("command") != list(stage.command)
        or completed.get("verifyCommand") != list(stage.verify_command)
        or completed.get("timeoutSeconds") != stage.timeout_seconds
        or completed.get("workingDirectory") != str(Path.cwd().resolve())
        or completed.get("fingerprint") != digest
    ):
        raise ReplayError("Sealed historical verifier definition changed")


def verify(seal_path: Path, *, alias: Path, owner: str) -> VerifiedRetirement:
    """Revalidate immutable receipt pins and the hash-bound inode identity."""
    evidence = Evidence()
    raw = evidence.read(_ref(_local(str(seal_path))))
    if (
        set(raw)
        != {
            "version",
            "kind",
            "request",
            "alias",
            "keeper",
            "owner",
            "originalRegistry",
            "completedStage",
            "keeperIdentity",
            "keeperSHA256",
            "evidence",
            "qualityApproved",
            "physicalBytesReclaimed",
        }
        or raw.get("version") != 1
        or raw.get("kind") != "native-verified-retired-hardlink"
        or raw.get("alias") != str(alias)
        or raw.get("owner") != owner
        or raw.get("qualityApproved") is not False
        or raw.get("physicalBytesReclaimed") != 0
    ):
        raise ReplayError("Exact sealed alias/owner evidence is required")
    pins = raw.get("evidence")
    if not isinstance(pins, list) or not 1 <= len(pins) < MAX_PINS:
        raise ReplayError("Bounded sealed retirement pins are required")
    for value in pins:
        evidence.read(value)
    request = evidence.read(raw.get("request"))
    proof_ref = reference(request.get("retirementProof"))
    proof = evidence.read(proof_ref.json())
    completed = record(raw.get("completedStage"))
    checkpoint = evidence.read(request.get("completedStage"))
    if (
        checkpoint.get("status") != "complete"
        or checkpoint.get("fingerprint") != completed.get("fingerprint")
        or checkpoint.get("outputs") != completed.get("outputs")
        or record(completed.get("outputs")).get(str(proof_ref.path)) != proof_ref.sha256
        or proof.get("keeper") != {"path": raw.get("keeper"), "sha256": raw.get("keeperSHA256")}
    ):
        raise ReplayError("Sealed completed verifier provenance changed")
    if (
        request.get("kind") != "native-retired-hardlink-seal-request"
        or request.get("version") != 1
        or request.get("alias") != str(alias)
        or request.get("keeper") != raw.get("keeper")
        or request.get("originalRegistry") != raw.get("originalRegistry")
    ):
        raise ReplayError("Sealed request endpoints or original registry changed")
    digest = raw.get("keeperSHA256")
    if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
        raise ReplayError("Sealed keeper digest is invalid")
    identity = record(raw.get("keeperIdentity"))
    if set(identity) != {"device", "inode", "bytes", "modifiedAtNs", "changedAtNs", "nlink"} or any(
        type(value) is not int or value < 0 for value in identity.values()
    ):
        raise ReplayError("Sealed keeper identity is invalid")
    _historical_completed(evidence, request, completed)
    keeper = _local(raw.get("keeper"))
    _absent(alias)
    if _identity(keeper) != raw.get("keeperIdentity"):
        raise ReplayError("Hash-bound keeper identity changed since actual verification")
    original, selected_owner = _original(evidence, raw)
    if selected_owner != owner:
        raise ReplayError("Original alias custodian changed")
    evidence.verify()
    _absent(alias)
    if _identity(keeper) != raw.get("keeperIdentity"):
        raise ReplayError("Keeper changed during retirement verification")
    return VerifiedRetirement(
        alias,
        keeper,
        owner,
        original,
        tuple(Reference(path, digest) for path, digest in evidence.pins.items()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    creator = commands.add_parser("seal")
    creator.add_argument("request", type=Path)
    creator.add_argument("output", type=Path)
    checker = commands.add_parser("verify")
    checker.add_argument("seal", type=Path)
    args = parser.parse_args()
    if args.operation == "seal":
        seal(args.request.absolute(), args.output.absolute())
    else:
        raw = _json(args.seal)
        verify(args.seal.absolute(), alias=_local(raw.get("alias")), owner=_text(raw.get("owner")))


if __name__ == "__main__":
    main()
