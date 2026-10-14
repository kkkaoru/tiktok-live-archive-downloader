"""Verify failed reader quiescence without awarding completion or releasing rows.

Historical source manifests are retained evidence, never executable backends or
claims that current source equals the failed generation. Existing completed
owner seals and their producer contract remain untouched.
"""

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from replay import native_completed_owner_verification as completed
from replay import native_retired_hardlink_verification as refs
from replay.core import ReplayError
from replay.fcp_part_assembly import _save
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest
from replay.workflow import load_stages

MAX_PINS = 512
MAX_FD_PARENTS = 16
MAX_FD_BYTES = 1024 * 1024
REQUEST_KEYS = frozenset(
    {
        "version",
        "kind",
        "owner",
        "originalRegistry",
        "capture",
        "failureRelease",
        "sourceEvidence",
    }
)


class Host(completed.Host, Protocol):
    def open_paths(self, paths: tuple[Path, ...]) -> tuple[str, ...]: ...


class SystemHost(completed.SystemHost):
    def open_paths(self, paths: tuple[Path, ...]) -> tuple[str, ...]:
        parents = tuple(dict.fromkeys(path.parent for path in paths))
        if not 1 <= len(parents) <= MAX_FD_PARENTS:
            raise ReplayError("Failed-reader FD parent count is outside its bound")
        wanted = {str(path) for path in paths}
        opened: set[str] = set()
        for parent in parents:
            result = subprocess.run(
                ["/usr/sbin/lsof", "-nP", "-Fpn", "+d", str(parent)],
                check=False,
                capture_output=True,
                timeout=5,
            )
            if (
                result.returncode not in (0, 1)
                or result.stderr
                or len(result.stdout) > MAX_FD_BYTES
            ):
                raise ReplayError("Failed-reader scoped FD observation was incomplete")
            try:
                names = result.stdout.decode("utf-8").splitlines()
            except UnicodeDecodeError as error:
                raise ReplayError("Invalid scoped FD observation") from error
            opened.update(
                line[1:]
                for line in names
                if line.startswith("n")
                and (line[1:] in wanted or line[1:].removesuffix(" (deleted)") in wanted)
            )
        return tuple(sorted(opened))


@dataclass(frozen=True)
class QuiescentOwner:
    owner: str
    process: dict[str, object]
    original_registry: refs.Reference
    endpoints: dict[str, object]
    pins: tuple[refs.Reference, ...]


class Evidence:
    def __init__(self) -> None:
        self.pins: dict[Path, str] = {}

    def bytes(self, value: object) -> bytes:
        ref = refs.reference(value)
        if (
            len(self.pins) >= MAX_PINS
            or ref.path in self.pins
            and self.pins[ref.path] != ref.sha256
        ):
            raise ReplayError("Conflicting or excessive failed-reader evidence")
        try:
            data = _bound_bytes(ref.path, ref.sha256, refs.MAX_BYTES)
        except ValueError as error:
            raise ReplayError("Failed-reader pinned evidence changed") from error
        self.pins[ref.path] = ref.sha256
        return data

    def read(self, value: object) -> dict[str, object]:
        return record(json.loads(self.bytes(value)))

    def verify(self) -> None:
        if any(file_digest(path) != digest for path, digest in self.pins.items()):
            raise ReplayError("Failed-reader evidence changed during observation")


def _rows(value: object) -> list[object]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_PINS:
        raise ReplayError("Expected bounded nonempty failed-reader evidence")
    return value


def _failure(
    evidence: Evidence,
    raw: dict[str, object],
    capture: dict[str, object],
    completion: completed.Completion,
) -> dict[str, object]:
    release = evidence.read(raw["failureRelease"])
    status = record(release.get("status"))
    exit_code = status.get("exit")
    if (
        release.get("version") != 1
        or release.get("kind") != "terminal-failed-writer-release-observation"
        or release.get("capture") != raw["capture"]
        or release.get("parent") != capture.get("parent")
        or release.get("processGroups") != list(completion.process_groups)
        or release.get("matches") != []
        or release.get("parentAbsent") is not True
        or release.get("capturedGroupsAbsent") is not True
        or release.get("completedStage") is not False
        or release.get("qualityApproved") is not False
        or type(exit_code) is not int
        or not 1 <= exit_code <= 255
        or status.get("path") != str(completion.status)
    ):
        raise ReplayError("Expected exact failed terminal capture without completion credit")
    status_pin = {"path": status.get("path"), "sha256": status.get("sha256")}
    if evidence.bytes(status_pin).decode("utf-8").strip() != str(exit_code):
        raise ReplayError("Failed-reader actual terminal status differs")
    checkpoint = record(release.get("checkpoint"))
    checkpoint_pin = {"path": checkpoint.get("path"), "sha256": checkpoint.get("sha256")}
    if (
        checkpoint.get("path") != capture.get("checkpoint")
        or checkpoint.get("sha256") != capture.get("initialCheckpointSHA256")
        or checkpoint.get("status") != "running"
        or evidence.read(checkpoint_pin) != capture.get("initialCheckpoint")
    ):
        raise ReplayError("Failed-reader retained running checkpoint differs from live capture")
    evidence.bytes(release.get("failureLog"))
    if release.get("layoutFailures") is not None:
        evidence.read(release["layoutFailures"])
    return status_pin


def _source_evidence(
    evidence: Evidence,
    raw: dict[str, object],
    capture: dict[str, object],
    completion: completed.Completion,
) -> None:
    evidence.read({"path": str(completion.workflow), "sha256": completion.workflow_sha256})
    stages = load_stages(completion.workflow)
    selected = [stage for stage in stages if stage.name == capture.get("stage")]
    if len(selected) != 1:
        raise ReplayError("Captured failed stage is not uniquely declared")
    declared = set(selected[0].inputs)
    manifests = _rows(raw["sourceEvidence"])
    for value in manifests:
        manifest = refs.reference(value)
        if manifest.path not in declared:
            raise ReplayError("Historical source evidence is not a declared stage input")
        inputs = record(evidence.read(value).get("inputs"))
        if not 1 <= len(inputs) <= MAX_PINS or any(
            Path(path) not in declared
            or not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            for path, digest in inputs.items()
        ):
            raise ReplayError("Historical source identity map differs from declared inputs")


def _facts(request: Path, host: Host) -> dict[str, object]:
    evidence = Evidence()
    raw = evidence.read(refs._ref(request))
    if (
        set(raw) != REQUEST_KEYS
        or raw.get("version") != 1
        or raw.get("kind") != "native-failed-owner-request"
    ):
        raise ReplayError("Expected explicit failed-reader request")
    registry = evidence.read(raw["originalRegistry"])
    owner_name = refs._text(raw["owner"])
    owner = completed._owner(registry, owner_name)
    capture = evidence.read(raw["capture"])
    capture_ref = refs.reference(raw["capture"])
    parent, completion = completed.load_capture(capture_ref.path, capture_ref.sha256)
    process: dict[str, object] = {"pid": parent.pid, "birth": parent.birth}
    if owner.get("process") != process:
        raise ReplayError("Failed-reader original owner process differs")
    status = _failure(evidence, raw, capture, completion)
    _source_evidence(evidence, raw, capture, completion)
    paths = completed._paths(registry, owner_name)
    owned_rows = [
        record(value)
        for value in _rows(registry.get("outputs"))
        if record(value).get("owner") == owner_name
    ]
    if any(row.get("state") != "planned" for row in owned_rows):
        raise ReplayError("Failed-reader quiescence supports only preserved planned endpoints")
    endpoints: dict[str, object] = {str(path): {"exists": False} for path in paths}
    completed._absent(host, parent=parent, completion=completion)
    if any(completed._endpoint(path) != {"exists": False} for path in paths) or host.open_paths(
        paths
    ):
        raise ReplayError("Failed-reader endpoint exists or has an observed open descriptor")
    evidence.verify()
    completed._absent(host, parent=parent, completion=completion)
    if any(completed._endpoint(path) != {"exists": False} for path in paths) or host.open_paths(
        paths
    ):
        raise ReplayError("Failed-reader endpoint changed during quiescence verification")
    evidence.verify()
    return {
        "version": 1,
        "kind": "native-failed-owner-quiescence",
        "owner": owner_name,
        "process": process,
        "originalRegistry": raw["originalRegistry"],
        "capture": raw["capture"],
        "request": refs._ref(request),
        "outerStatus": status,
        "producerSHA256": file_digest(Path(__file__).resolve()),
        "endpoints": endpoints,
        "pins": [refs.Reference(path, digest).json() for path, digest in evidence.pins.items()],
        "completedStage": False,
        "qualityApproved": False,
        "completionCredit": False,
        "futureAllocationBytes": 0,
        "protectedReservationsReleased": False,
        "descriptorScope": "exact-endpoint-names-in-known-parent-directories",
        "globalTransientDescriptorCoverage": False,
    }


def seal(request: Path, output: Path, *, host: Host) -> QuiescentOwner:
    """Publish only a failed actor's verified quiescence, never stage completion."""
    _save(output, _facts(request, host))
    raw = record(json.loads(output.read_bytes()))
    return verify(output, owner=refs._text(raw["owner"]), process=record(raw["process"]), host=host)


def verify(path: Path, *, owner: str, process: dict[str, object], host: Host) -> QuiescentOwner:
    evidence = Evidence()
    raw = evidence.read(refs._ref(path))
    request = refs.reference(raw.get("request"))
    if (
        raw.get("owner") != owner
        or raw.get("process") != process
        or raw != _facts(request.path, host)
    ):
        raise ReplayError("Failed-reader quiescence seal or evidence changed")
    evidence.verify()
    return QuiescentOwner(
        owner,
        process,
        refs.reference(raw["originalRegistry"]),
        record(raw["endpoints"]),
        tuple(refs.reference(item) for item in _rows(raw["pins"])),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        seal(args.request.absolute(), args.output.absolute(), host=SystemHost())
    except (ReplayError, OSError, ValueError, subprocess.SubprocessError):
        print("NATIVE_FAILED_OWNER_QUIESCENCE_ERROR")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
