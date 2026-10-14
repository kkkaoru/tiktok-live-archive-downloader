"""Retire one verified ProRes part while retaining its lossy HEVC derivation.

Native completion is historical after retirement, never a byte-identical HEVC
replacement claim. Rebuilding ProRes requires the retained native recipe and a
new revision. This lifecycle never renders, encodes, or operates applications.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

from replay import part_master_publication as publication
from replay import whole_technical_batch as metadata
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.native_export_receipt import save_new
from replay.recordings import record
from replay.stages import execution_lock, file_digest
from replay.whole_technical_batch import Reference

ROOT = Path(__file__).resolve().parents[2]
MEDIA_ROOT = Path("/Volumes/m.2-ssd/movie")
MAX_REFS = 1024
MAX_PARTS = 128
MAX_BYTES = 1024**4
MAX_FD_BYTES = 256 * 1024
MAX_AGE_NS = 3600 * 1_000_000_000
HEVC_CONTAINERS = frozenset({".mp4", ".m4v", ".mov"})
SCOPES = frozenset({"current", "queued", "resume", "FCP", "Viewer", "diagnostic"})
RECIPE_REFS = frozenset(
    {
        "cleanSource",
        "sourceOriginal",
        "partXML",
        "partition",
        "processedVoice",
        "motionTemplate",
        "fontAsset",
        "cleanupEvidence",
    }
)
Identity = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class Native:
    completion: Reference
    frame_proof: Reference
    publication: Reference | None
    staging_retirement: Reference | None


@dataclass(frozen=True)
class Request:
    reference: Reference
    part: int
    source: Reference
    source_bytes: int
    planner: Reference
    native: Native
    recipe: Reference
    disposition: Reference
    state: Path
    leases: tuple[Path, ...]
    instruction: Reference


@dataclass(frozen=True)
class RetiredSource:
    inventory: tuple[Reference, ...]
    rebuild_required: bool = True


def _path(value: object, *, work: bool = False, source: bool = False) -> Path:
    path = metadata._path(value, work=work)
    roots = (ROOT / "edits", MEDIA_ROOT) if source else (ROOT, MEDIA_ROOT, Path("/opt/homebrew"))
    if not any(path.is_relative_to(root) for root in roots):
        raise ReplayError("Part retirement path is outside its authorized scope")
    return path


def _ref(value: object) -> Reference:
    ref = metadata._reference(value)
    _path(str(ref.path))
    return ref


def _evidence_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_mode,
    )


def _evidence_ref(value: object) -> Reference:
    """Allow only the current configured Python binary outside retained-data roots."""
    ref = metadata._reference(value)
    if any(ref.path.is_relative_to(root) for root in (ROOT, MEDIA_ROOT, Path("/opt/homebrew"))):
        return _ref(value)
    alias = ROOT / ".venv/bin/python"
    patch = ROOT / ".python-version"
    try:
        expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        runtime = (
            Path(sys.base_prefix)
            / "bin"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
        ).resolve(strict=True)
        link_before = alias.lstat()
        link_target = os.readlink(alias)
        before = ref.path.lstat()
        if (
            not stat.S_ISLNK(link_before.st_mode)
            or alias.parent.resolve(strict=True) != alias.parent
            or alias.resolve(strict=True) != ref.path
            or ref.path != runtime
            or metadata._bounded_bytes(patch).strip() != expected.encode("ascii")
            or not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= MAX_BYTES
            or not os.access(ref.path, os.X_OK)
            or file_digest(ref.path) != ref.sha256
        ):
            raise ReplayError("Configured Python executable evidence differs")
        result = subprocess.run(
            [str(ref.path), "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=5,
            cwd=ROOT,
            env={"PATH": os.defpath, "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        after = ref.path.lstat()
        if (
            result.returncode != 0
            or result.stdout != f"Python {expected}\n".encode("ascii")
            or result.stderr
            or _evidence_identity(alias.lstat()) != _evidence_identity(link_before)
            or os.readlink(alias) != link_target
            or alias.resolve(strict=True) != ref.path
            or _evidence_identity(before) != _evidence_identity(after)
        ):
            raise ReplayError("Configured Python executable evidence changed")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ReplayError("Configured Python executable evidence is unavailable") from error
    return ref


def _current(path: Path) -> Reference:
    return Reference(_path(str(path)), file_digest(path))


def _read(ref: Reference) -> dict[str, object]:
    _path(str(ref.path))
    return metadata._metadata(ref.path, ref.sha256)


def _rows(value: object) -> list[object]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_REFS:
        raise ReplayError("Expected a bounded nonempty retained inventory")
    return list(value)


def _integer(value: object, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ReplayError("Expected a positive bounded part integer")
    return value


def _identity(path: Path) -> Identity:
    _path(str(path), source=True)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_BYTES or info.st_nlink != 1:
        raise ReplayError("The retired source must be one regular nonsymlink file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink


def _identity_value(value: object) -> Identity:
    if (
        not isinstance(value, list)
        or len(value) != 6
        or any(type(item) is not int for item in value)
    ):
        raise ReplayError("Source identity must contain six filesystem integers")
    a, b, c, d, e, f = value
    if not all(isinstance(item, int) for item in (a, b, c, d, e, f)):
        raise ReplayError("Source identity has invalid integers")
    if min(a, b, c, d, e) < 0 or c == 0 or f != 1:
        raise ReplayError("Source identity is outside its domain")
    return a, b, c, d, e, f


def load_request(path: Path) -> Request:
    """Read exact policy and metadata without media operations or deletion."""
    path = _path(str(path), work=True)
    reference = _current(path)
    raw = _read(reference)
    required = {
        "version",
        "part",
        "source",
        "sourceBytes",
        "planner",
        "native",
        "recipe",
        "disposition",
        "state",
        "leases",
        "approval",
    }
    if set(raw) != required or type(raw["version"]) is not int or raw["version"] != 1:
        raise ReplayError("Expected exact version-one part retirement request")
    native, approval = record(raw["native"]), record(raw["approval"])
    if set(native) != {"completion", "frameProof", "publication", "stagingRetirement"} or (
        set(approval) != {"instruction", "scope", "deletionAuthorized"}
        or approval["scope"] != "verified-hevc-part-source"
        or approval["deletionAuthorized"] is not True
    ):
        raise ReplayError("Native provenance or explicit approved scope is incomplete")
    publication_ref = None if native["publication"] is None else _ref(native["publication"])
    staging = None if native["stagingRetirement"] is None else _ref(native["stagingRetirement"])
    leases = tuple(_path(item, work=True) for item in _rows(raw["leases"]))
    state = _path(raw["state"], work=True)
    source = _ref(raw["source"])
    _path(str(source.path), source=True)
    if (
        source.path.suffix.lower() != ".mov"
        or len(set(leases)) != len(leases)
        or state in leases
        or any(not directory.is_dir() for directory in (state, *leases))
        or (staging is not None and publication_ref is None)
    ):
        raise ReplayError("Part source or declared ownership namespaces are invalid")
    return Request(
        reference,
        _integer(raw["part"], MAX_PARTS),
        source,
        _integer(raw["sourceBytes"], MAX_BYTES),
        _ref(raw["planner"]),
        Native(_ref(native["completion"]), _ref(native["frameProof"]), publication_ref, staging),
        _ref(raw["recipe"]),
        _ref(raw["disposition"]),
        state,
        leases,
        _ref(approval["instruction"]),
    )


def _inventory(refs: tuple[Reference, ...], *, source: Reference) -> tuple[Reference, ...]:
    result: dict[Path, Reference] = {}
    for ref in refs:
        if ref.path == source.path or (ref.path in result and result[ref.path] != ref):
            raise ReplayError("Retained inventory requires the retired source or conflicts")
        result[ref.path] = ref
    if not result or len(result) > MAX_REFS:
        raise ReplayError("Retained part inventory exceeds its budget")
    return tuple(result[path] for path in sorted(result))


def _disposition(request: Request, *, current: bool) -> tuple[Reference, ...]:
    raw = _read(request.disposition)
    if set(raw) != {
        "version",
        "source",
        "owner",
        "ownerReleased",
        "observedAtNs",
        "references",
    } or (
        raw["version"] != 1
        or type(raw["version"]) is not int
        or _ref(raw["source"]) != request.source
        or raw["ownerReleased"] is not True
        or not isinstance(raw["owner"], str)
        or not 1 <= len(raw["owner"]) <= 256
    ):
        raise ReplayError("Completed owner disposition does not bind this part source")
    instant = raw["observedAtNs"]
    if (
        type(instant) is not int
        or not 0 < instant <= time.time_ns()
        or (current and time.time_ns() - instant > MAX_AGE_NS)
    ):
        raise ReplayError("Part reference observations are stale or future dated")
    scopes = record(raw["references"])
    if set(scopes) != SCOPES:
        raise ReplayError("All six live, queued, resume and application scopes are required")
    refs = [request.disposition]
    for value in scopes.values():
        row = record(value)
        scope = row.get("scope")
        if (
            set(row) != {"scope", "resolved", "requiresMaterializedSource", "evidence"}
            or not isinstance(scope, str)
            or not 1 <= len(scope) <= 4096
            or row["resolved"] is not True
            or row["requiresMaterializedSource"] is not False
        ):
            raise ReplayError("A current or resumable source reference is unresolved")
        refs.extend(_ref(item) for item in _rows(row["evidence"]))
    return tuple(refs)


def _recipe(request: Request) -> tuple[Reference, ...]:
    raw = _read(request.recipe)
    if set(raw) != {
        "version",
        "kind",
        "part",
        "source",
        "inputs",
        "outputPolicy",
        *RECIPE_REFS,
    } or (
        type(raw["version"]) is not int
        or raw["version"] != 1
        or raw["kind"] != "native-part-rebuild-recipe"
        or raw["part"] != request.part
        or _ref(raw["source"]) != request.source
        or raw["outputPolicy"] != "new-nonoverwriting-revision"
    ):
        raise ReplayError("Retained native rerender recipe does not bind this ordered part")
    required = tuple(_ref(raw[name]) for name in sorted(RECIPE_REFS))
    refs = tuple(_ref(item) for item in _rows(raw["inputs"]))
    if not set(required).issubset(refs):
        raise ReplayError("Native recipe omits required clean, editable, voice or font inputs")
    xml, partition = _ref(raw["partXML"]), _ref(raw["partition"])
    if _ref(_read(request.native.frame_proof).get("partition")) != partition:
        raise ReplayError("Native rerender recipe differs from the completed part partition")
    rows = _rows(_read(partition).get("parts"))
    selected = record(rows[request.part - 1]) if request.part <= len(rows) else {}
    if (
        request.part > len(rows)
        or _ref({"path": selected.get("path"), "sha256": selected.get("sha256")}) != xml
    ):
        raise ReplayError("Native recipe XML differs from its ordered partition")
    root = ET.fromstring(metadata._bounded_bytes(xml.path))
    bound = {ref.path for ref in refs}
    media = root.findall("./resources/asset/media-rep")
    paths = {_xml_media(item.get("src")) for item in media}
    if not paths or not paths.issubset(bound):
        raise ReplayError("Native recipe leaves an XML media dependency unretained")
    if not {_ref(raw["cleanSource"]).path, _ref(raw["processedVoice"]).path}.issubset(paths):
        raise ReplayError(
            "Native XML does not consume the retained clean source and processed voice"
        )
    return (request.recipe, *refs)


def _xml_media(value: str | None) -> Path:
    if value is None:
        raise ReplayError("Native XML media has no URL")
    uri = urlparse(value)
    if uri.scheme != "file" or uri.netloc not in ("", "localhost") or uri.query or uri.fragment:
        raise ReplayError("Native XML media must use one local file URL")
    return _path(unquote(uri.path))


def _native(request: Request) -> tuple[Reference, ...]:
    native = request.native
    completion = _read(native.completion)
    original = _ref(completion.get("movie"))
    if native.publication is None:
        publication._completed(native.completion.path)
        if original != request.source or _ref(completion.get("frameProof")) != native.frame_proof:
            raise ReplayError("Direct native completion identifies a different source")
        extra: tuple[Reference, ...] = ()
    else:
        completion_path, intent_path = publication.verify_published(
            native.publication.path,
            native.frame_proof.path,
            part=request.part,
            retirement=None
            if native.staging_retirement is None
            else native.staging_retirement.path,
        )
        published = _read(native.publication)
        if (
            completion_path != native.completion.path
            or _ref(published.get("movie")) != request.source
        ):
            raise ReplayError("Published native completion identifies a different source")
        extra = (native.publication, _current(intent_path))
    if completion.get("part") != request.part:
        raise ReplayError("Native completion is for a different ordered part")
    proof = _read(native.frame_proof)
    if proof.get("part") != request.part or _ref(proof.get("movie")) != request.source:
        raise ReplayError("Native frame proof does not bind this source")
    inputs = record(completion.get("inputs"))
    refs = tuple(
        _evidence_ref({"path": path, "sha256": digest})
        for path, digest in inputs.items()
        if Path(path) != original.path
    )
    retired = () if native.staging_retirement is None else (native.staging_retirement,)
    return (
        native.completion,
        native.frame_proof,
        _ref(completion.get("frameProof")),
        *extra,
        *retired,
        *refs,
    )


def _acceptance(request: Request) -> dict[str, object]:
    # Import only during current-source admission; readonly retired consumers do
    # not call the planner, which itself imports this module.
    from replay.compressed_part_delivery import part_acceptance

    accepted = part_acceptance(request.planner.path, number=request.part)
    _accepted(request, accepted)
    return accepted


def _accepted(request: Request, value: object) -> tuple[Reference, ...]:
    raw = record(value)
    if (
        type(raw.get("version")) is not int
        or raw.get("version") != 1
        or raw.get("kind") != "accepted-compressed-part"
        or type(raw.get("part")) is not int
        or raw.get("part") != request.part
        or raw.get("accepted") is not True
        or _ref(raw.get("source")) != request.source
    ):
        raise ReplayError("HEVC replacement has not passed its measured part acceptance")
    movie, evidence = _ref(raw.get("movie")), _ref(raw.get("evidence"))
    if movie.path == request.source.path or movie.path.suffix.lower() not in HEVC_CONTAINERS:
        raise ReplayError("HEVC replacement must be a distinct supported intermediate container")
    refs = tuple(_evidence_ref(item) for item in _rows(raw.get("inputs")))
    if request.source not in refs or movie not in refs or evidence not in refs:
        raise ReplayError("HEVC acceptance omits its source, replacement or proof")
    return tuple(ref for ref in refs if ref.path != request.source.path)


@contextmanager
def _owned(request: Request) -> Iterator[None]:
    with ExitStack() as stack:
        name = hashlib.sha256(str(request.source.path).encode()).hexdigest()
        if not stack.enter_context(execution_lock(request.state, name)):
            raise ReplayError("Part retirement is already owned")
        for directory in request.leases:
            if not stack.enter_context(execution_lock(directory, ".workflow")):
                raise ReplayError("A declared current or resumable workflow is active")
        yield


@contextmanager
def _fresh(refs: tuple[Reference, ...]) -> Iterator[None]:
    cache = InputHashCache((ref.path for ref in refs), allow_uncached_overflow=True)
    cache.prime()
    with input_hash_context(cache):
        if any(file_digest(ref.path) != ref.sha256 for ref in refs):
            raise ReplayError("Part retirement input bytes changed")
        yield
        cache.verify()


def _unused(path: Path) -> None:
    result = subprocess.run(
        ["/usr/sbin/lsof", "-n", "-P", "-Fpf", "--", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=15,
        env={**os.environ, "LC_ALL": "C"},
    )
    if (
        result.returncode != 1
        or result.stdout
        or result.stderr
        or len(result.stdout) > MAX_FD_BYTES
    ):
        raise ReplayError("Part source is open or its file-descriptor use is unknown")


def _output(path: Path) -> Path:
    path = _path(str(path), work=True)
    if not path.parent.is_dir() or path.exists() or path.is_symlink():
        raise ReplayError("Retirement evidence output must be absent in an existing work directory")
    return path


def prepare(request_path: Path, output: Path) -> Path:
    """Freshly admit source/replacement/recipe and reserve immutable deletion intent."""
    request, output = load_request(request_path), _output(output)
    with _owned(request):
        identity = _identity(request.source.path)
        if identity[2] != request.source_bytes:
            raise ReplayError("Original part size changed")
        accepted = _acceptance(request)
        refs = _inventory(
            (
                request.reference,
                request.planner,
                request.instruction,
                *_native(request),
                *_accepted(request, accepted),
                *_recipe(request),
                *_disposition(request, current=True),
            ),
            source=request.source,
        )
        with _fresh((*refs, request.source)):
            _unused(request.source.path)
            if _identity(request.source.path) != identity:
                raise ReplayError("Original part identity changed during admission")
            parent = request.source.path.parent.stat()
        save_new(
            output,
            {
                "version": 1,
                "kind": "compressed-part-retirement-intent",
                "request": request.reference.json(),
                "part": request.part,
                "source": request.source.json(),
                "sourceIdentity": list(identity),
                "parentIdentity": [parent.st_dev, parent.st_ino],
                "acceptance": accepted,
                "inventory": [ref.json() for ref in refs],
                "preparedAtNs": time.time_ns(),
                "deletionAuthorized": True,
                "historicalNativePartVerified": True,
                "byteIdenticalReplacement": False,
            },
        )
    return output


def _intent(request: Request, ref: Reference) -> dict[str, object]:
    raw = _read(ref)
    fields = {
        "version",
        "kind",
        "request",
        "part",
        "source",
        "sourceIdentity",
        "parentIdentity",
        "acceptance",
        "inventory",
        "preparedAtNs",
        "deletionAuthorized",
        "historicalNativePartVerified",
        "byteIdenticalReplacement",
    }
    if (
        set(raw) != fields
        or type(raw.get("version")) is not int
        or raw.get("version") != 1
        or raw.get("kind") != "compressed-part-retirement-intent"
        or _ref(raw.get("request")) != request.reference
        or type(raw.get("part")) is not int
        or raw.get("part") != request.part
        or _ref(raw.get("source")) != request.source
        or raw.get("deletionAuthorized") is not True
        or raw.get("historicalNativePartVerified") is not True
        or raw.get("byteIdenticalReplacement") is not False
        or _identity_value(raw.get("sourceIdentity"))[2] != request.source_bytes
        or type(raw.get("preparedAtNs")) is not int
    ):
        raise ReplayError("Prepared retirement intent does not bind this source")
    parent = raw.get("parentIdentity")
    if (
        not isinstance(parent, list)
        or len(parent) != 2
        or any(type(item) is not int or item < 0 for item in parent)
    ):
        raise ReplayError("Prepared intent has an invalid canonical parent identity")
    _accepted(request, raw.get("acceptance"))
    return raw


def _retained(
    request: Request, intent: dict[str, object], *, current: bool
) -> tuple[Reference, ...]:
    refs = _inventory(
        tuple(_evidence_ref(item) for item in _rows(intent.get("inventory"))),
        source=request.source,
    )
    expected = (
        request.reference,
        request.planner,
        request.instruction,
        *_accepted(request, intent.get("acceptance")),
        *_recipe(request),
        *_disposition(request, current=current),
    )
    if not set(expected).issubset(refs):
        raise ReplayError("Prepared intent omits retained recipe or reference closure")
    return refs


def _absent(source: Reference) -> None:
    _path(str(source.path), source=True)
    if source.path.exists() or source.path.is_symlink():
        raise ReplayError("Retired ProRes source is present; reconcile its new revision")


def _operation(request: Request, intent: Reference, output: Path) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "exact-prores-part-unlink-operation",
        "request": request.reference.json(),
        "intent": intent.json(),
        "source": request.source.json(),
        "result": str(output),
    }


def _unlink(request: Request, prepared: dict[str, object]) -> None:
    source = request.source.path
    expected = _identity_value(prepared.get("sourceIdentity"))
    _unused(source)
    with ExitStack() as stack:
        parent = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, parent)
        parent_info = os.fstat(parent)
        if prepared.get("parentIdentity") != [parent_info.st_dev, parent_info.st_ino]:
            raise ReplayError("Canonical source parent changed")
        info = os.stat(source.name, dir_fd=parent, follow_symlinks=False)
        if _identity(source) != expected or (info.st_dev, info.st_ino) != expected[:2]:
            raise ReplayError("Exact source identity changed before unlink")
        descriptor = os.open(
            source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        stream = stack.enter_context(os.fdopen(descriptor, "rb"))
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != expected[:2]:
            raise ReplayError("Opened source identity differs")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != request.source.sha256 or _identity(source) != expected:
            raise ReplayError("Exact source bytes changed before unlink")
        _path(str(source), source=True)
        named_parent = source.parent.stat()
        named = os.stat(source.name, dir_fd=parent, follow_symlinks=False)
        if (named_parent.st_dev, named_parent.st_ino) != (
            parent_info.st_dev,
            parent_info.st_ino,
        ) or (
            named.st_dev,
            named.st_ino,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
            named.st_nlink,
        ) != expected:
            raise ReplayError("Source namespace changed immediately before unlink")
        os.unlink(source.name, dir_fd=parent)
        os.fsync(parent)


def execute(request_path: Path, intent_path: Path, output: Path) -> Path:
    """Execute one admitted exact unlink, reconciling immutable interrupted operations."""
    request = load_request(request_path)
    intent = _current(_path(str(intent_path), work=True))
    output = _path(str(output), work=True)
    operation_path = output.with_suffix(output.suffix + ".operation.json")
    with _owned(request):
        if output.exists() or output.is_symlink():
            current = _current(output)
            accepted = _intent(request, intent)
            movie = _ref(record(accepted["acceptance"])["movie"])
            verify(current, source=request.source, movie=movie, part=request.part)
            if _read(current).get("intent") != intent.json():
                raise ReplayError("Existing retirement result binds a different intent")
            return output
        _output(output)
        prepared = _intent(request, intent)
        operation = _operation(request, intent, output)
        resumed = operation_path.exists() or operation_path.is_symlink()
        if resumed and _read(_current(operation_path)) != operation:
            raise ReplayError("Occupied unlink operation differs; do not retry deletion")
        present = request.source.path.exists() or request.source.path.is_symlink()
        if not present and not resumed:
            raise ReplayError("Source disappeared without this reserved operation")
        refs = _retained(request, prepared, current=present)
        with _fresh((*refs, intent)):
            if present:
                if _identity(request.source.path) != _identity_value(prepared["sourceIdentity"]):
                    raise ReplayError("Original source identity differs from prepared intent")
                if not resumed:
                    save_new(_output(operation_path), operation)
                _unlink(request, prepared)
            _absent(request.source)
        operation_ref = _current(operation_path)
        if _read(operation_ref) != operation:
            raise ReplayError("Reserved unlink operation changed during execution")
        save_new(
            output,
            {
                "version": 1,
                "kind": "retired-prores-part",
                "request": request.reference.json(),
                "part": request.part,
                "source": request.source.json(),
                "movie": record(prepared["acceptance"])["movie"],
                "intent": intent.json(),
                "operation": operation_ref.json(),
                "inventory": [ref.json() for ref in refs],
                "masterMaterialized": False,
                "historicalNativePartVerified": True,
                "freshOriginalVerified": False,
                "byteIdenticalReplacement": False,
                "rebuildRequired": True,
                "rebuildPolicy": "new-nonoverwriting-revision",
                "unlinkObserved": present,
                "interruptedAbsenceRecovered": resumed and not present,
                "qualityApproved": False,
                "syncApproved": False,
                "deliveryApproved": False,
            },
        )
    return output


def verify(
    provenance: Reference, *, source: Reference, movie: Reference, part: int
) -> RetiredSource:
    """Freshly verify retained bytes; original native completion remains historical."""
    raw = _read(provenance)
    fields = {
        "version",
        "kind",
        "request",
        "part",
        "source",
        "movie",
        "intent",
        "operation",
        "inventory",
        "masterMaterialized",
        "historicalNativePartVerified",
        "freshOriginalVerified",
        "byteIdenticalReplacement",
        "rebuildRequired",
        "rebuildPolicy",
        "unlinkObserved",
        "interruptedAbsenceRecovered",
        "qualityApproved",
        "syncApproved",
        "deliveryApproved",
    }
    if (
        set(raw) != fields
        or type(raw.get("version")) is not int
        or raw.get("version") != 1
        or raw.get("kind") != "retired-prores-part"
        or type(raw.get("part")) is not int
        or raw.get("part") != part
        or _ref(raw.get("source")) != source
        or _ref(raw.get("movie")) != movie
        or raw.get("masterMaterialized") is not False
        or raw.get("historicalNativePartVerified") is not True
        or raw.get("freshOriginalVerified") is not False
        or raw.get("byteIdenticalReplacement") is not False
        or raw.get("rebuildRequired") is not True
        or raw.get("rebuildPolicy") != "new-nonoverwriting-revision"
        or any(
            raw.get(name) is not False
            for name in ("qualityApproved", "syncApproved", "deliveryApproved")
        )
        or type(raw.get("unlinkObserved")) is not bool
        or type(raw.get("interruptedAbsenceRecovered")) is not bool
        or raw["unlinkObserved"] == raw["interruptedAbsenceRecovered"]
    ):
        raise ReplayError("Retired part provenance does not bind this historical source and HEVC")
    request_ref = _ref(raw.get("request"))
    request = load_request(request_ref.path)
    if request.reference != request_ref or request.source != source or request.part != part:
        raise ReplayError("Retired part request changed")
    intent_ref, operation_ref = _ref(raw.get("intent")), _ref(raw.get("operation"))
    intent = _intent(request, intent_ref)
    if _ref(record(intent["acceptance"])["movie"]) != movie or _read(operation_ref) != _operation(
        request, intent_ref, provenance.path
    ):
        raise ReplayError("Retired part operation or accepted replacement differs")
    refs = _retained(request, intent, current=False)
    if raw.get("inventory") != [ref.json() for ref in refs]:
        raise ReplayError("Retired part retained inventory differs from admitted intent")
    inventory = _inventory((*refs, provenance, intent_ref, operation_ref), source=source)
    with _fresh(inventory):
        _absent(source)
    return RetiredSource(inventory)


def main(argv: list[str] | None = None) -> int:
    """Dispatch explicit prepare/execute/verify; never infer approval from a filename."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--intent", type=Path)
    args = parser.parse_args(argv)
    try:
        request = load_request(args.request)
        if args.mode == "verify":
            result = _read(_current(args.output))
            verify(
                _current(args.output),
                source=request.source,
                movie=_ref(result["movie"]),
                part=request.part,
            )
        elif args.mode == "prepare":
            if args.intent is not None:
                raise ReplayError("Preparation does not accept an earlier intent")
            prepare(args.request, args.output)
        else:
            if args.intent is None:
                raise ReplayError("Execution requires its immutable prepared intent")
            execute(args.request, args.intent, args.output)
        return 0
    except (ReplayError, OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Compressed part retirement failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
