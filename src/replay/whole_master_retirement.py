"""Evidence-only lifecycle for a reconstructable whole MOV; never delete media.

``prepare REQUEST INTENT`` validates fresh retained bytes and completed owners.
Root separately rechecks references/identity and performs authorized cleanup.
``complete REQUEST INTENT DELETION_RECORD OUTCOME`` requires actual absence.
``verify REQUEST RECORD`` rechecks either immutable phase without a producer.
Rebuilding uses the ordinary checked join workflow in a new absent revision.
Historical whole verification is never reported as fresh verification after
retirement, and a lossy final movie is never the byte replacement for the MOV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.native_export_receipt import save_new
from replay.recordings import record
from replay.stages import execution_lock, file_digest

ROOT = Path(__file__).resolve().parents[2]
MEDIA_ROOT = Path("/Volumes/m.2-ssd/movie")
TOOL_ROOT = Path("/opt/homebrew/Cellar/ffmpeg")
MAX_METADATA_BYTES = 8 * 1024**2
MAX_PARTS = 512
MAX_REFS = 1024
MAX_LEASES = 32
MAX_FRAMES = 1_000_000
MAX_MEDIA_BYTES = 256 * 1024**3
MAX_DISPOSITION_AGE_NS = 3600 * 1_000_000_000
REFERENCE_ROLES = frozenset({"current", "queued", "resume", "fcp", "viewer", "diagnostic"})
DOMAIN_FIELDS = frozenset(
    {"frames", "rate", "width", "height", "sampleRate", "channels", "sampleFrames", "pcm32SHA256"}
)
CODE_NAMES = frozenset(
    {
        "whole_master_retirement.py",
        "native_export_receipt.py",
        "input_hash_cache.py",
        "core.py",
        "recordings.py",
        "stages.py",
        "delivery_render.py",
        "native_join.py",
        "media.py",
        "subtitle_clock.py",
    }
)
RETAINED_ROLES = frozenset(
    {
        "voiceProof",
        "assembly",
        "assemblyLineage",
        "editableXML",
        "wholeWorkflow",
        "sourceBefore",
        "sourceAfter",
    }
)
Identity = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class Reference:
    path: Path
    sha256: str

    def json(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class Part:
    media: Reference
    frames: int


@dataclass(frozen=True)
class Claim:
    role: str
    artifact: Reference
    pointer: str


@dataclass(frozen=True)
class Request:
    source: Reference
    master: Reference
    master_bytes: int
    plan: Reference
    parts: tuple[Part, ...]
    voice: Reference
    retained: tuple[Reference, ...]
    code: tuple[Reference, ...]
    tools: tuple[Reference, ...]
    domain: dict[str, object]
    claims: tuple[Claim, ...]
    disposition: Reference
    state: Path
    leases: tuple[Path, ...]


def _path(value: object, *, work: bool = False) -> Path:
    if not isinstance(value, str) or not value or any(c in value for c in "\0\r\n"):
        raise ReplayError("Expected a canonical retirement path")
    path = Path(value)
    allowed = (
        (ROOT / "edits/work",)
        if work
        else (ROOT / "edits", ROOT / "src/replay", MEDIA_ROOT, TOOL_ROOT)
    )
    if (
        not path.is_absolute()
        or str(path) != value
        or ".." in path.parts
        or not any(path.is_relative_to(root) for root in allowed)
        or path.resolve() != path
    ):
        raise ReplayError("Retirement path escapes its declared scope or traverses a link")
    return path


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise ReplayError("Expected a SHA256 pin")
    return value


def _ref(value: object) -> Reference:
    raw = record(value)
    if set(raw) != {"path", "sha256"}:
        raise ReplayError("Reference fields must be exact")
    return Reference(_path(raw["path"]), _sha(raw["sha256"]))


def _integer(value: object, maximum: int = MAX_MEDIA_BYTES) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ReplayError("Expected a positive bounded integer")
    return value


def _rows(value: object, maximum: int = MAX_REFS) -> list[object]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ReplayError("Expected a nonempty bounded inventory")
    return list(value)


def _identity(path: Path) -> Identity:
    _path(str(path))
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or not 0 < info.st_size <= MAX_MEDIA_BYTES
        or info.st_nlink < 1
    ):
        raise ReplayError("Retirement inputs must be nonempty regular files")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ReplayError("Retirement JSON contains duplicate keys")
    return dict(pairs)


def _nonfinite(value: str) -> NoReturn:
    raise ReplayError(f"Nonfinite retirement JSON value: {value}")


def _read(path: Path, digest: str | None = None) -> object:
    before = _identity(path)
    if before[2] > MAX_METADATA_BYTES:
        raise ReplayError("Retirement metadata exceeds its byte budget")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != before[:2]:
            raise ReplayError("Retirement metadata changed before opening")
        data = handle.read(MAX_METADATA_BYTES + 1)
        after = os.fstat(handle.fileno())
    if (
        len(data) != before[2]
        or _identity(path) != before
        or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        != before
        or (digest is not None and hashlib.sha256(data).hexdigest() != digest)
    ):
        raise ReplayError("Retirement metadata bytes changed")
    return json.loads(data, object_pairs_hook=_unique, parse_constant=_nonfinite)


def _pointer(value: object, pointer: str) -> object:
    current = value
    for raw in pointer.split("/")[1:]:
        name = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or name not in current:
            raise ReplayError("Historical projection does not identify a recorded value")
        current = current[name]
    return current


def _domain(value: object) -> dict[str, object]:
    raw = record(value)
    if set(raw) != DOMAIN_FIELDS:
        raise ReplayError("Whole domain fields must be exact")
    result: dict[str, object] = {
        name: _integer(item, MAX_FRAMES if name == "frames" else MAX_MEDIA_BYTES)
        for name, item in raw.items()
        if name != "pcm32SHA256"
    }
    result["pcm32SHA256"] = _sha(raw["pcm32SHA256"])
    rate = _integer(raw["rate"], 240)
    frames = _integer(raw["frames"], MAX_FRAMES)
    sample_rate = _integer(raw["sampleRate"])
    samples = _integer(raw["sampleFrames"])
    if samples * rate != frames * sample_rate:
        raise ReplayError("Whole video and processed voice clocks differ")
    return result


def _expected(request: Request) -> dict[str, object]:
    return {
        **request.domain,
        "masterPath": str(request.master.path),
        "masterSHA256": request.master.sha256,
        "masterBytes": request.master_bytes,
        "fullAVVerified": True,
        "allPTSVerified": True,
        "wholeVoicePCMEqual": True,
    }


def load_request(path: Path) -> Request:
    """Load exact generic recipe/projections; no child, media hash or write."""
    path = _path(str(path), work=True)
    raw = record(_read(path))
    required = {
        "version",
        "master",
        "masterBytes",
        "joinPlan",
        "parts",
        "voice",
        "retained",
        "code",
        "tools",
        "domain",
        "historical",
        "disposition",
        "state",
        "leases",
    }
    if set(raw) != required or type(raw["version"]) is not int or raw["version"] != 1:
        raise ReplayError("Expected the exact version-one retirement request")
    parts: list[Part] = []
    for item in _rows(raw["parts"], MAX_PARTS):
        part = record(item)
        if set(part) != {"media", "frames"}:
            raise ReplayError("Part fields must be exact")
        parts.append(Part(_ref(part["media"]), _integer(part["frames"], MAX_FRAMES)))
    claims: list[Claim] = []
    for role, item in record(raw["historical"]).items():
        claim = record(item)
        pointer = claim.get("pointer")
        if (
            set(claim) != {"artifact", "pointer"}
            or not isinstance(pointer, str)
            or not pointer.startswith("/")
            or not 1 < len(pointer) <= 1024
            or re.search(r"~(?![01])", pointer)
        ):
            raise ReplayError("Historical claims require bounded object JSON pointers")
        claims.append(Claim(role, _ref(claim["artifact"]), pointer))
    retained = record(raw["retained"])
    if set(retained) != RETAINED_ROLES:
        raise ReplayError("Voice, editable assembly and historical workflow evidence are required")
    request = Request(
        Reference(path, file_digest(path)),
        _ref(raw["master"]),
        _integer(raw["masterBytes"]),
        _ref(raw["joinPlan"]),
        tuple(parts),
        _ref(raw["voice"]),
        tuple(_ref(retained[name]) for name in sorted(retained)),
        tuple(_ref(item) for item in _rows(raw["code"])),
        tuple(_ref(item) for item in _rows(raw["tools"], 2)),
        _domain(raw["domain"]),
        tuple(claims),
        _ref(raw["disposition"]),
        _path(raw["state"], work=True),
        tuple(_path(item, work=True) for item in _rows(raw["leases"], MAX_LEASES)),
    )
    if (
        len({part.media.path for part in request.parts}) != len(request.parts)
        or sum(part.frames for part in request.parts) != request.domain["frames"]
        or {claim.role for claim in request.claims} != set(_expected(request))
        or {ref.path for ref in request.code} != {ROOT / "src/replay" / name for name in CODE_NAMES}
        or {ref.path.name for ref in request.tools} != {"ffmpeg", "ffprobe"}
        or any(not ref.path.is_relative_to(TOOL_ROOT) for ref in request.tools)
        or len(set(request.leases)) != len(request.leases)
        or request.state in request.leases
        or not request.state.is_dir()
        or any(not path.is_dir() for path in request.leases)
    ):
        raise ReplayError("Retirement recipe, code, tools or ownership namespace differs")
    return request


def _disposition(
    request: Request, ref: Reference, *, current: bool = True
) -> tuple[Reference, ...]:
    raw = record(_read(ref.path, ref.sha256))
    if (
        set(raw) != {"version", "master", "owner", "ownerReleased", "observedAtNs", "references"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or raw["ownerReleased"] is not True
        or _ref(raw["master"]) != request.master
    ):
        raise ReplayError("Completed owner disposition does not bind this master")
    owner, instant = raw["owner"], raw["observedAtNs"]
    if (
        not isinstance(owner, str)
        or not 1 <= len(owner) <= 256
        or type(instant) is not int
        or instant <= 0
        or instant > time.time_ns()
        or (current and time.time_ns() - instant > MAX_DISPOSITION_AGE_NS)
    ):
        raise ReplayError("Owner/reference observations are stale or invalid")
    roles = record(raw["references"])
    if set(roles) != REFERENCE_ROLES:
        raise ReplayError("All current, queued, resume, app and diagnostic scopes are required")
    evidence: list[Reference] = [ref]
    for role in roles.values():
        row = record(role)
        scope = row.get("scope")
        if (
            set(row) != {"scope", "resolved", "requiresMaterializedMaster", "evidence"}
            or not isinstance(scope, str)
            or not 1 <= len(scope) <= 4096
            or row["resolved"] is not True
            or row["requiresMaterializedMaster"] is not False
        ):
            raise ReplayError("A master reference remains active, unresolved or required")
        evidence.extend(_ref(item) for item in _rows(row["evidence"]))
    return tuple(evidence)


def _inventory(refs: tuple[Reference, ...]) -> tuple[Reference, ...]:
    bound: dict[Path, Reference] = {}
    for ref in refs:
        if ref.path in bound and bound[ref.path] != ref:
            raise ReplayError("Conflicting retirement input pins")
        _identity(ref.path)
        bound[ref.path] = ref
    if len(bound) > MAX_REFS:
        raise ReplayError("Retirement input inventory exceeds its budget")
    return tuple(bound[path] for path in sorted(bound))


def _inputs(
    request: Request, *, current: bool = True, code: tuple[Reference, ...] | None = None
) -> tuple[Reference, ...]:
    refs = _inventory(
        (
            request.source,
            request.plan,
            request.voice,
            *request.retained,
            *(request.code if code is None else code),
            *request.tools,
            *(part.media for part in request.parts),
            *(claim.artifact for claim in request.claims),
            *_disposition(request, request.disposition, current=current),
        )
    )
    if request.master.path in {ref.path for ref in refs}:
        raise ReplayError("The reconstruction/consumer recipe still requires the master")
    return refs


def _runtime_transition(
    request: Request, intent: Reference, deletion: Reference, transition: Reference
) -> tuple[Reference, ...]:
    """Bind an explicit owner-API migration; archived source is evidence, never code."""
    raw = record(_read(transition.path, transition.sha256))
    if (
        set(raw) != {"version", "scope", "request", "intent", "deletion", "changes", "runtime"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or raw["scope"] != "supplemental-current-owner-v1"
        or _ref(raw["request"]) != request.source
        or _ref(raw["intent"]) != intent
        or _ref(raw["deletion"]) != deletion
    ):
        raise ReplayError("Runtime transition must bind the exact original retirement")
    changes = _rows(raw["changes"], 1)
    if len(changes) != 1:
        raise ReplayError("Only the explicit supplemental-owner module migration is supported")
    change = record(changes[0])
    if set(change) != {"original", "archive", "current"}:
        raise ReplayError("Runtime transition fields must be exact")
    old, archive, new = (_ref(change[key]) for key in ("original", "archive", "current"))
    if (
        old not in request.code
        or old.path != ROOT / "src/replay/whole_master_retirement.py"
        or new.path != old.path
        or new.sha256 == old.sha256
        or archive.sha256 != old.sha256
        or archive.path == old.path
        or not archive.path.is_relative_to(ROOT / "edits/work")
        or archive.path.stat().st_size > MAX_METADATA_BYTES
    ):
        raise ReplayError("Unapproved runtime change or missing exact original source archive")
    runtime = tuple(_ref(item) for item in _rows(raw["runtime"]))
    actual = tuple(
        Reference(source.path, source.sha256)
        for source in capture(("whole_master_retirement",), package=ROOT / "src/replay")
    )
    if len(set(runtime)) != len(runtime) or set(runtime) != set(actual) or new not in runtime:
        raise ReplayError("Current runtime closure differs from the explicit transition pins")
    # Every unlisted original code pin remains enforced at its original path.
    return (transition, archive, *runtime, *(ref for ref in request.code if ref != old))


def _domains(request: Request) -> None:
    rows = _rows(_read(request.plan.path, request.plan.sha256), MAX_PARTS)
    expected = [
        {"sourcePath": str(part.media.path), "durationFrames": part.frames}
        for part in request.parts
    ]
    if rows != expected:
        raise ReplayError("Ordered join plan differs from the retained reconstruction recipe")
    values = _expected(request)
    for claim in request.claims:
        actual = _pointer(_read(claim.artifact.path, claim.artifact.sha256), claim.pointer)
        wanted = values[claim.role]
        if type(actual) is not type(wanted) or actual != wanted:
            raise ReplayError(f"Historical whole domain differs: {claim.role}")


@contextmanager
def _owned(request: Request) -> Iterator[None]:
    with ExitStack() as stack:
        lock_name = hashlib.sha256(str(request.master.path).encode()).hexdigest()
        owned = stack.enter_context(execution_lock(request.state, lock_name))
        if not owned:
            raise ReplayError("The retirement namespace is already owned")
        for directory in request.leases:
            if not stack.enter_context(execution_lock(directory, ".workflow")):
                raise ReplayError("A declared workflow owner is still active")
        yield


def _absent(request: Request) -> None:
    _path(str(request.master.path))
    if request.master.path.exists() or request.master.path.is_symlink():
        raise ReplayError("Retired whole master must actually be absent")


def _intent(request: Request, intent: Reference) -> dict[str, object]:
    raw = record(_read(intent.path, intent.sha256))
    if (
        raw.get("kind") != "whole-master-retirement-intent"
        or raw.get("request") != request.source.json()
        or raw.get("master") != request.master.json()
        or raw.get("masterMaterialized") is not True
        or raw.get("deletionAuthorized") is not False
    ):
        raise ReplayError("Original prepared intent is missing or belongs to another recipe")
    identity = _rows(raw.get("masterIdentity"), 6)
    if (
        len(identity) != 6
        or any(type(item) is not int or item < 0 for item in identity)
        or identity[2] != request.master_bytes
        or identity[5] != 1
    ):
        raise ReplayError("Prepared master identity differs")
    prepared_at = raw.get("preparedAtNs")
    if type(prepared_at) is not int or not 0 < prepared_at <= time.time_ns():
        raise ReplayError("Prepared chronology is invalid")
    return raw


def _deletion(
    request: Request, intent: Reference, ref: Reference, *, current: bool = True
) -> tuple[Reference, ...]:
    raw = record(_read(ref.path, ref.sha256))
    prepared = _intent(request, intent)
    if (
        set(raw)
        != {
            "version",
            "intent",
            "master",
            "masterIdentity",
            "removed",
            "operation",
            "observedAtNs",
            "disposition",
        }
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or _ref(raw["intent"]) != intent
        or _ref(raw["master"]) != request.master
        or raw["masterIdentity"] != prepared["masterIdentity"]
        or raw["removed"] is not True
        or raw["operation"] != "exact-path-unlink"
    ):
        raise ReplayError("External deletion result does not bind the prepared exact path")
    instant, prepared_at = raw["observedAtNs"], prepared.get("preparedAtNs")
    if (
        type(instant) is not int
        or type(prepared_at) is not int
        or not prepared_at <= instant <= time.time_ns()
    ):
        raise ReplayError("Deletion chronology is invalid")
    return (intent, ref, *_disposition(request, _ref(raw["disposition"]), current=current))


def _document(
    request: Request,
    inputs: tuple[Reference, ...],
    *,
    intent: Reference | None,
    deletion: Reference | None,
) -> dict[str, object]:
    before = intent is None
    result: dict[str, object] = {
        "version": 1,
        "kind": "whole-master-retirement-intent"
        if before
        else "retired-reconstructable-whole-master",
        "request": request.source.json(),
        "master": request.master.json(),
        "masterMaterialized": before,
        "deletionAuthorized": False,
        "historicalWholeVerified": True,
        "freshWholeVerified": False,
        "reconstructableInputsVerified": True,
        "byteIdenticalRebuildVerified": False,
        "rebuildPolicy": "new-nonoverwriting-revision",
        "domain": request.domain,
        "inputs": [ref.json() for ref in inputs],
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
    }
    if before:
        info = _identity(request.master.path)
        if info[2] != request.master_bytes or info[5] != 1:
            raise ReplayError("Original whole master size differs")
        result.update({"masterIdentity": list(info), "preparedAtNs": time.time_ns()})
    else:
        _absent(request)
        if intent is None or deletion is None:
            raise ReplayError("After-delete evidence requires both immutable handoff records")
        result.update({"intent": intent.json(), "deletion": deletion.json()})
    return result


def run(
    request_path: Path,
    output: Path,
    *,
    intent: Path | None = None,
    deletion: Path | None = None,
    verify: bool = False,
    current_owner: Path | None = None,
    runtime_transition: Path | None = None,
) -> Path:
    """Check fresh initial/terminal bytes and domains; only write exclusive evidence."""
    request = load_request(request_path)
    output = _path(str(output), work=True)
    if not output.parent.is_dir() or (intent is None) != (deletion is None):
        raise ReplayError("Retirement output/phase is incomplete")
    if current_owner is not None and intent is None:
        raise ReplayError("Supplemental current owner requires original intent and deletion")
    if runtime_transition is not None and current_owner is None:
        raise ReplayError("Runtime transition requires explicit current owner evidence")
    if not verify and (output.exists() or output.is_symlink()):
        raise ReplayError("Retirement output is occupied; reconcile before retry")
    intent_ref = (
        None if intent is None else Reference(_path(str(intent), work=True), file_digest(intent))
    )
    deletion_ref = (
        None
        if deletion is None
        else Reference(_path(str(deletion), work=True), file_digest(deletion))
    )
    owner_ref = (
        None
        if current_owner is None
        else Reference(_path(str(current_owner), work=True), file_digest(current_owner))
    )
    transition_ref = (
        None
        if runtime_transition is None
        else Reference(_path(str(runtime_transition), work=True), file_digest(runtime_transition))
    )
    with _owned(request):
        # Immutable retirement verification is historical ownership evidence,
        # while all retained input bytes are still read freshly in this operation.
        current = owner_ref is None and not (verify and intent_ref is not None)
        extras = (
            ()
            if intent_ref is None or deletion_ref is None
            else _deletion(request, intent_ref, deletion_ref, current=current)
        )
        master = (request.master,) if intent_ref is None else ()
        supplemental = () if owner_ref is None else _disposition(request, owner_ref)
        code = None
        if transition_ref is not None and intent_ref is not None and deletion_ref is not None:
            code = _runtime_transition(request, intent_ref, deletion_ref, transition_ref)
        refs = _inventory(
            (*_inputs(request, current=current, code=code), *master, *extras, *supplemental)
        )
        if intent_ref is not None and request.master.path in {ref.path for ref in refs}:
            raise ReplayError("Retired input inventory still requires the materialized master")
        cache = InputHashCache((ref.path for ref in refs), allow_uncached_overflow=True)
        cache.prime()
        with input_hash_context(cache):
            if (
                any(file_digest(ref.path) != ref.sha256 for ref in refs)
                or load_request(request_path) != request
            ):
                raise ReplayError("Retirement input bytes changed")
            _domains(request)
            wanted = _document(request, refs, intent=intent_ref, deletion=deletion_ref)
            if owner_ref is not None:
                wanted.update({"version": 2, "currentOwner": owner_ref.json()})
                if transition_ref is not None:
                    wanted["runtimeTransition"] = transition_ref.json()
            if verify:
                found = record(_read(output))
                if intent_ref is None:
                    _intent(request, Reference(output, file_digest(output)))
                    wanted["preparedAtNs"] = found.get("preparedAtNs")
                if found != wanted:
                    raise ReplayError("Immutable retirement evidence differs from current domain")
            _disposition(request, request.disposition, current=current)
            if owner_ref is not None:
                _disposition(request, owner_ref)
            cache.verify()
            if (
                transition_ref is not None
                and intent_ref is not None
                and deletion_ref is not None
                and _runtime_transition(request, intent_ref, deletion_ref, transition_ref) != code
            ):
                raise ReplayError("Runtime transition changed at terminal boundary")
        # Check path state once more after fresh terminal bytes, before publication.
        if intent_ref is None:
            if list(_identity(request.master.path)) != wanted["masterIdentity"]:
                raise ReplayError("Master identity changed before intent publication")
        else:
            _absent(request)
        if not verify:
            save_new(output, wanted)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "complete", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--intent", type=Path)
    parser.add_argument("--deletion", type=Path)
    parser.add_argument("--current-owner", type=Path)
    parser.add_argument("--runtime-transition", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "complete" and (args.intent is None or args.deletion is None):
        parser.error("complete requires --intent and --deletion")
    if args.phase == "prepare" and (args.intent is not None or args.deletion is not None):
        parser.error("prepare has no after-delete inputs")
    try:
        result = run(
            args.request,
            args.output,
            intent=args.intent,
            deletion=args.deletion,
            verify=args.phase == "verify",
            current_owner=args.current_owner,
            runtime_transition=args.runtime_transition,
        )
    except (ReplayError, OSError, ValueError) as error:
        parser.exit(1, f"Whole-master retirement failed: {error}\n")
    print(json.dumps({"evidence": str(result), "mediaDeleted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
