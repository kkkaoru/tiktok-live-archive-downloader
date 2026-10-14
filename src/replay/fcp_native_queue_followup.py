"""Bind actual queue observations to existing followup and release contracts.

This adapter never dispatches apps, runs workflows, deletes media or retries an
uncertain effect. Native observations remain at their original paths and times.
Release is a read-only decision; the caller retains GUI and execution ownership.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from replay import fcp_native_queue as native_queue
from replay import fcp_part_followup as followup
from replay.artifact_json import _path
from replay.core import ReplayError
from replay.fcp_native_parts import _idle
from replay.foreground_evidence import _bound_bytes
from replay.input_hash_cache import MAX_INPUTS, InputHashCache, input_hash_context
from replay.native_export_receipt import _dispatch, native_payload, save_new
from replay.part_master_publication import verify_published
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "edits" / "work"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_PARTS = 1024
MAX_OBSERVATION_SECONDS = 30
REQUEST_KEYS = frozenset(
    {
        "version",
        "kind",
        "queue",
        "ready",
        "queueOrdinal",
        "assemblyPart",
        "assembly",
        "publishedAssembly",
        "sourceCleanup",
        "exportRequest",
        "exportResponse",
        "background",
        "ffmpeg",
        "ffprobe",
        "audioSampleRate",
        "audioChannels",
        "receiptOutput",
    }
)
RELEASE_KEYS = frozenset(
    {
        "version",
        "kind",
        "preparation",
        "terminal",
        "publication",
        "retirement",
        "postDeleteStatus",
        "retirementAudit",
        "history",
        "consumer",
        "capacity",
        "successorQueue",
        "successorOrdinal",
    }
)


@dataclass(frozen=True)
class Ref:
    path: Path
    sha256: str

    def json(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class Release:
    assembly_part: int
    successor_part: int
    successor_project: str
    successor_movie: Path
    evidence: tuple[Ref, ...]
    eligible: bool = True
    quality_approved: bool = False


class Inputs:
    """Keep explicit pins and recheck bytes; never infer an absent input's hash."""

    def __init__(self) -> None:
        self.refs: dict[Path, str] = {}

    def add(self, value: object) -> Ref:
        raw = record(value)
        path, digest = _path(raw.get("path")), raw.get("sha256")
        if path.is_relative_to(ROOT / "private") or path.is_relative_to(ROOT / "downloads"):
            raise ReplayError("Queue evidence cannot read private or download data")
        if (
            set(raw) != {"path", "sha256"}
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ReplayError("Expected an exact path and SHA256 reference")
        if path in self.refs and self.refs[path] != digest:
            raise ReplayError("Conflicting immutable input pins")
        if path not in self.refs and len(self.refs) >= MAX_INPUTS:
            raise ReplayError("Followup input inventory exceeds its bounded budget")
        self.refs[path] = digest
        return Ref(path, digest)

    def current(self, path: Path) -> Ref:
        return self.add({"path": str(_path(str(path))), "sha256": file_digest(path)})

    def read(self, ref: Ref) -> dict[str, object]:
        if not ref.path.is_relative_to(ROOT / "edits"):
            raise ReplayError("Queue metadata must remain within checkout edit evidence")
        return record(
            json.loads(
                _bound_bytes(ref.path, ref.sha256, MAX_JSON_BYTES), object_pairs_hook=_unique
            )
        )

    def verify(self) -> dict[str, str]:
        if any(file_digest(path) != digest for path, digest in self.refs.items()):
            raise ReplayError("Followup evidence changed during acceptance")
        return {str(path): digest for path, digest in self.refs.items()}


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError("Duplicate JSON evidence key")
        result[key] = value
    return result


def _integer(value: object, *, maximum: int = MAX_PARTS) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ReplayError("Expected a positive bounded integer")
    return value


def _work(path: Path) -> Path:
    path = _path(str(path))
    if not path.is_relative_to(WORK) or path == WORK:
        raise ReplayError("New adapter artifacts require canonical repository work storage")
    return path


def _schema(
    data: dict[str, object],
    *,
    kind: str,
    keys: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    if (
        data.get("version") != 1
        or data.get("kind") != kind
        or not keys <= set(data) <= keys | optional
    ):
        raise ReplayError("Unsupported or incomplete queue followup contract")


def _member(data: dict[str, object], ordinal: int) -> dict[str, object]:
    rows = data.get("parts")
    if not isinstance(rows, list) or not 1 <= ordinal <= len(rows) <= MAX_PARTS:
        raise ReplayError("Queue or assembly member is absent")
    return record(rows[ordinal - 1])


def _queue(inputs: Inputs, value: object) -> tuple[native_queue.Queue, dict[str, object]]:
    ref = inputs.add(value)
    data = inputs.read(ref)
    for core in native_queue._rows(data.get("corePins"), MAX_PARTS):
        inputs.add(core)
    for member in native_queue._rows(data.get("parts"), MAX_PARTS):
        _request_pins(inputs, record(member))
    queue = native_queue.load(ref.path)
    for path, digest in queue.pins:
        inputs.add({"path": str(path), "sha256": digest})
    return queue, data


def _request_pins(inputs: Inputs, member: dict[str, object]) -> None:
    for item in native_queue._rows(member.get("operations"), len(native_queue.PHASES)):
        ref = inputs.add(record(item).get("request"))
        if not ref.path.is_relative_to(ROOT / "edits"):
            raise ReplayError("Native requests must remain within checkout edit evidence")


def _part_identity(member: dict[str, object]) -> tuple[int, str, Path]:
    part, project = _integer(member.get("assemblyPart")), member.get("project")
    if not isinstance(project, str) or not project:
        raise ReplayError("Expected an exact queue project")
    return part, project, _path(member.get("movie"))


def _preparation(request: Path, inputs: Inputs) -> tuple[dict[str, object], dict[str, object]]:
    raw = inputs.read(inputs.current(_work(request)))
    _schema(
        raw,
        kind="native-queue-followup-request",
        keys=REQUEST_KEYS,
        optional=frozenset({"approvedOriginal"}),
    )
    if "approvedOriginal" in raw:
        inputs.add(raw["approvedOriginal"])
    queue, descriptor = _queue(inputs, raw["queue"])
    ordinal, part = _integer(raw["queueOrdinal"]), _integer(raw["assemblyPart"])
    member = _member(descriptor, ordinal)
    actual_part, project, movie = _part_identity(member)
    ready_ref = inputs.add(raw["ready"])
    ready = inputs.read(ready_ref)
    if (
        ordinal != 1
        or actual_part != part
        or ready_ref.path != queue.root / "followup-ready.json"
        or ready.get("queueSHA256") != queue.digest
        or ready.get("queueOrdinal") != ordinal
        or ready.get("assemblyPart") != part
        or ready.get("project") != project
        or ready.get("movie") != str(movie)
        or ready.get("processedParts") != 1
        or ready.get("remainingParts") != len(queue.parts) - 1
        or ready.get("completionVerified") is not False
        or ready.get("qualityApproved") is not False
    ):
        raise ReplayError("Ready evidence does not bind the explicit first queue member")
    for name in ("exportRequest", "exportResponse", "background"):
        if ready.get(name) != raw[name]:
            raise ReplayError("Actual native evidence differs from the ready marker")
    export = next(op for op in queue.parts[ordinal - 1].operations if op.phase == "export")
    paths = {name: inputs.add(raw[name]).path for name in followup.PIN_NAMES}
    if paths["exportRequest"] != export.request or any(
        not paths[name].is_relative_to(queue.root) for name in ("exportResponse", "background")
    ):
        raise ReplayError("Native request or observations differ from the original queue events")
    staged = inputs.read(inputs.add(raw["assembly"]))
    published = inputs.read(inputs.add(raw["publishedAssembly"]))
    if any(
        staged.get(name) != published.get(name) for name in ("partition", "voice", "voiceReceipt")
    ):
        raise ReplayError("Staging and SSD assembly clocks differ")
    if _member(staged, part).get("movie") != str(movie):
        raise ReplayError("Queue movie does not match the explicit staging assembly part")
    partition = inputs.read(inputs.add(staged.get("partition")))
    part_row = _member(partition, part)
    xml = inputs.add({"path": part_row.get("path"), "sha256": part_row.get("sha256")})
    if not xml.path.is_relative_to(ROOT / "edits"):
        raise ReplayError("Partition XML must remain within checkout edit evidence")
    root = ET.fromstring(_bound_bytes(xml.path, xml.sha256, MAX_JSON_BYTES))
    element = root.find("./library/event/project")
    if element is None or element.get("name") != project:
        raise ReplayError("Queue project does not match the selected partition XML")
    native = native_payload(inputs.read(inputs.add(raw["exportResponse"])))
    if _dispatch(inputs.read(inputs.add(raw["exportRequest"])), native, project) != movie:
        raise ReplayError("Actual dispatch output does not match the staging movie")
    background = native_payload(inputs.read(inputs.add(raw["background"])))
    _idle(background)
    if (
        not movie.is_file()
        or movie.is_symlink()
        or movie.stat().st_size <= 0
        or background.get("pid") != native.get("pid")
        or paths["background"].stat().st_mtime_ns
        < max(movie.stat().st_mtime_ns, paths["exportResponse"].stat().st_mtime_ns)
    ):
        raise ReplayError("Background observation belongs to another native process")
    config = {
        "version": 1,
        "part": part,
        **{name: raw[name] for name in followup.PIN_NAMES},
        "audioSampleRate": _integer(raw["audioSampleRate"], maximum=384000),
        "audioChannels": _integer(raw["audioChannels"], maximum=64),
        "receiptOutput": str(_work(_path(raw["receiptOutput"]))),
    }
    if "approvedOriginal" in raw:
        config["approvedOriginal"] = raw["approvedOriginal"]
    identity: dict[str, object] = {
        "queueOrdinal": ordinal,
        "assemblyPart": part,
        "project": project,
        "movie": str(movie),
        "queue": raw["queue"],
        "ready": raw["ready"],
    }
    return config, identity


def prepare(request: Path, output: Path) -> Path:
    """Prepare the existing factory graph from actual evidence; run nothing."""
    output = _work(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ReplayError("Adapter output is occupied or its parent is absent; reconcile first")
    inputs = Inputs()
    config, identity = _preparation(request, inputs)
    inputs.current(Path(__file__).resolve())
    inputs.verify()
    output.mkdir(mode=0o700)
    config_path = output / "followup-config.json"
    save_new(config_path, config)
    graph = followup.prepare(config_path, output / "prepared", integrated=True)
    save_new(
        output / "preparation.json",
        {
            "version": 1,
            "kind": "native-queue-followup-preparation",
            **identity,
            "inputs": inputs.verify(),
            "config": inputs.current(config_path).json(),
            "workflow": inputs.current(graph).json(),
            "operation": inputs.current(graph.parent / "operation.json").json(),
            "qualityApproved": False,
        },
    )
    return graph


def prepare_from_ready(
    *,
    template: Path,
    queue: Path,
    ready: Path,
    native_complete: Path,
    request_output: Path,
    prepared_output: Path,
) -> Path:
    """Bind completed native observations to the existing full-AV factory."""
    inputs = Inputs()
    request_output, prepared_output = _work(request_output), _work(prepared_output)
    if any(path.exists() or path.is_symlink() for path in (request_output, prepared_output)):
        raise ReplayError("Followup outputs are occupied; preserve existing evidence")
    if not request_output.parent.is_dir() or not prepared_output.parent.is_dir():
        raise ReplayError("Followup output parents must already exist")
    base = inputs.read(inputs.current(_work(template)))
    expected = REQUEST_KEYS - {
        "version",
        "kind",
        "queue",
        "ready",
        "queueOrdinal",
        "assemblyPart",
        "exportRequest",
        "exportResponse",
        "background",
    }
    if not expected <= set(base) <= expected | {"approvedOriginal"}:
        raise ReplayError("Expected an exact followup template")
    queue_ref, ready_ref = inputs.current(_work(queue)), inputs.current(_work(ready))
    observed = inputs.read(ready_ref)
    checkpoint = inputs.read(inputs.current(_work(native_complete)))
    config = inputs.read(inputs.add(checkpoint.get("config")))
    if (
        checkpoint.get("version") != 1
        or checkpoint.get("kind") != "native-delivery-pump-checkpoint"
        or checkpoint.get("qualityApproved") is not False
        or checkpoint.get("outputs") != [ready_ref.json()]
        or config.get("kind") != "native-delivery-pump"
        or config.get("queue") != queue_ref.json()
    ):
        raise ReplayError("Actual native-complete checkpoint does not bind this queue and ready")
    for value in native_queue._rows(checkpoint.get("inputs"), MAX_INPUTS):
        inputs.add(value)
    request = {
        "version": 1,
        "kind": "native-queue-followup-request",
        "queue": queue_ref.json(),
        "ready": ready_ref.json(),
        "queueOrdinal": _integer(observed.get("queueOrdinal")),
        "assemblyPart": _integer(observed.get("assemblyPart")),
        **base,
        **{name: observed.get(name) for name in ("exportRequest", "exportResponse", "background")},
    }
    inputs.verify()
    save_new(request_output, request)
    return prepare(request_output, prepared_output)


def _refs(inputs: Inputs, value: object) -> tuple[Ref, ...]:
    if not isinstance(value, list) or not 1 <= len(value) < MAX_INPUTS:
        raise ReplayError("Expected a nonempty bounded evidence inventory")
    refs = tuple(inputs.add(item) for item in value)
    if len({ref.path for ref in refs}) != len(refs):
        raise ReplayError("Evidence inventory must be unique")
    return refs


def _terminal(
    inputs: Inputs, prep: dict[str, object], value: object
) -> tuple[dict[str, object], Path]:
    config = inputs.read(inputs.add(prep.get("config")))
    operation_ref = inputs.add(prep.get("operation"))
    operation = inputs.read(operation_ref)
    terminal_ref = inputs.add(value)
    terminal = inputs.read(terminal_ref)
    refs = _refs(Inputs(), operation.get("inputs"))
    expected = {str(ref.path): ref.sha256 for ref in refs}
    if (
        operation.get("version") != 1
        or operation.get("kind") != "bounded-native-part-followup"
        or operation.get("config") != str(inputs.add(prep.get("config")).path)
        or terminal_ref.path != operation_ref.path.parent / "verified-followup.json"
        or terminal.get("version") != 1
        or terminal.get("kind") != "verified-integrated-native-followup"
        or terminal.get("operation") != operation_ref.json()
        or terminal.get("part") != prep.get("assemblyPart")
        or terminal.get("qualityApproved") is not False
        or terminal.get("inputs") != expected
    ):
        raise ReplayError("Full followup terminal does not bind the prepared operation")
    movie = _path(prep.get("movie"))
    if str(movie) not in expected:
        raise ReplayError("Original staging movie is missing from the full input inventory")
    for ref in refs:
        if ref.path != movie:
            inputs.add(ref.json())
    outputs = _refs(inputs, terminal.get("outputs"))
    graph = inputs.read(inputs.add(prep.get("workflow")))
    rows = graph.get("stages")
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or record(rows[0]).get("outputs")
        != [*[str(ref.path) for ref in outputs], str(terminal_ref.path)]
    ):
        raise ReplayError("Terminal output inventory differs from the prepared workflow")
    if config.get("part") != prep.get("assemblyPart"):
        raise ReplayError("Prepared part identity changed")
    return config, _path(operation.get("semantic"))


def _retirement(inputs: Inputs, raw: dict[str, object], prep: dict[str, object]) -> Ref:
    part, movie = _integer(prep.get("assemblyPart")), _path(prep.get("movie"))
    publication_ref, retirement_ref = inputs.add(raw["publication"]), inputs.add(raw["retirement"])
    published, retired = inputs.read(publication_ref), inputs.read(retirement_ref)
    terminal = inputs.read(inputs.add(raw["terminal"]))
    if (
        retired.get("part") != part
        or retired.get("stagingRetirementAuthorized") is not True
        or retired.get("deletionPerformed") is not False
        or retired.get("source") != published.get("source")
        or record(published.get("source")).get("path") != str(movie)
        or record(published.get("source")).get("sha256")
        != record(terminal.get("inputs")).get(str(movie))
        or retired.get("publication") != publication_ref.json()
        or movie.exists()
        or movie.is_symlink()
    ):
        raise ReplayError("Verified physical staging retirement is required")
    status_ref = inputs.add(raw["postDeleteStatus"])
    if (
        _bound_bytes(status_ref.path, status_ref.sha256, 16).strip() != b"0"
        or status_ref.path.stat().st_mtime_ns < retirement_ref.path.stat().st_mtime_ns
    ):
        raise ReplayError("Post-delete consumer verification has not completed")
    audit_ref = inputs.add(raw["retirementAudit"])
    audit = inputs.read(audit_ref)
    codes = audit.get("prepareConsumerPostDeleteExitCodes")
    history = _refs(inputs, raw["history"])
    required = tuple(
        inputs.add(published.get(name)) for name in ("completion", "frameProof", "intent")
    )
    if (
        audit.get("version") != 1
        or audit.get("part") != part
        or audit.get("stagingPath") != str(movie)
        or audit.get("stagingAbsent") is not True
        or audit.get("historyUnchanged") is not True
        or audit.get("retirementSHA256") != retirement_ref.sha256
        or audit.get("qualityApproved") is not False
        or not isinstance(codes, list)
        or not 1 <= len(codes) <= MAX_PARTS
        or any(type(code) is not int or code != 0 for code in codes)
        or audit_ref.path.stat().st_mtime_ns < status_ref.path.stat().st_mtime_ns
        or audit.get("survivingMovie") != record(published.get("movie")).get("path")
        or audit.get("survivingSHA256") != record(published.get("movie")).get("sha256")
        or not {ref.path for ref in required}.issubset({ref.path for ref in history})
    ):
        raise ReplayError("Retirement audit or immutable history binding is incomplete")
    return retirement_ref


def _consumer(inputs: Inputs, value: object, *, config: dict[str, object], retirement: Ref) -> None:
    graph = inputs.read(inputs.add(value))
    rows = graph.get("stages")
    assembly = record(config.get("publishedAssembly")).get("path")
    retired = inputs.read(retirement)
    required = {
        assembly,
        str(retirement.path),
        record(retired.get("movie")).get("path"),
        record(retired.get("frameProof")).get("path"),
    }
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PARTS:
        raise ReplayError("Expected an explicit bounded retirement consumer graph")
    matched = False
    for item in rows:
        row = record(item)
        declared = row.get("inputs")
        command, verify = row.get("command"), row.get("verify_command")
        if (
            not isinstance(declared, list)
            or not isinstance(command, list)
            or not isinstance(verify, list)
        ):
            raise ReplayError("Consumer stage needs explicit inputs and a domain verifier")
        if (
            required.issubset({item for item in declared if isinstance(item, str)})
            and "replay.fcp_part_assembly" in command
            and "assemble" in command
            and assembly in command
            and "replay.fcp_part_assembly" in verify
            and "verify-assembly" in verify
            and assembly in verify
        ):
            matched = True
    if not matched:
        raise ReplayError("Current assembly consumer does not require the retirement proof")


def _successor(
    inputs: Inputs, raw: dict[str, object], prep: dict[str, object], host: native_queue.Host
) -> tuple[int, str, Path]:
    previous, old = _queue(inputs, prep.get("queue"))
    original = _part_identity(_member(old, _integer(prep.get("queueOrdinal"))))
    if original != (prep.get("assemblyPart"), prep.get("project"), _path(prep.get("movie"))):
        raise ReplayError("Preparation no longer binds its original queue member")
    next_member = _member(old, _integer(prep.get("queueOrdinal")) + 1)
    successor, new = _queue(inputs, raw["successorQueue"])
    ordinal = _integer(raw["successorOrdinal"])
    selected = _member(new, ordinal)
    identity = _part_identity(selected)
    if (
        ordinal != 1
        or identity != _part_identity(next_member)
        or successor.target != previous.target
    ):
        raise ReplayError("Successor does not match the next explicit queued assembly part")
    movie = identity[2]
    if any(
        path.exists() or path.is_symlink()
        for path in (
            movie,
            movie.with_name(f".{movie.name}.native-queue-reservation.json"),
            successor.root / "owner.json",
            native_queue.GUI_LEASE_ROOT / ".fcp-native-queue-owner.json",
        )
    ):
        raise ReplayError(
            "Successor has reserved or uncertain effects; reconcile instead of replaying"
        )
    if host.birth(successor.target.pid) != successor.target.birth:
        raise ReplayError("Successor GUI process identity changed")
    inventory_ref = inputs.add(raw["capacity"])
    inventory = inputs.read(inventory_ref)
    captured, age = inventory.get("capturedAtNs"), inventory.get("maximumAgeSeconds")
    if (
        inventory_ref.path != successor.inventory
        or type(captured) is not int
        or type(age) is not int
        or not 1 <= age <= MAX_OBSERVATION_SECONDS
        or not 0 <= host.now_ns() - captured <= age * 1_000_000_000
        or inventory.get("activeRenders") != 0
    ):
        raise ReplayError("Fresh idle capacity observation is required")
    count, used = inventory.get("stagingCount"), inventory.get("stagingBytes")
    maximum, byte_limit = inventory.get("maximumStagingCount"), inventory.get("maximumStagingBytes")
    if (
        type(count) is not int
        or count < 0
        or type(used) is not int
        or used < 0
        or type(maximum) is not int
        or not 1 <= maximum <= MAX_PARTS
        or type(byte_limit) is not int
        or byte_limit <= 0
        or count + 1 > maximum
        or used + successor.peak_bytes > byte_limit
        or any(
            host.free(path) - successor.peak_bytes < max(successor.floor_bytes, host.floor(path))
            for path in (successor.boot, movie.parent)
        )
    ):
        raise ReplayError("Successor exceeds measured staging or disk capacity")
    return identity


def verify_release(request: Path, *, host: native_queue.Host) -> Release:
    """Read existing proof and fresh observations; never start the successor."""
    inputs = Inputs()
    raw = inputs.read(inputs.current(_work(request)))
    _schema(raw, kind="native-queue-nextpart-release-request", keys=RELEASE_KEYS)
    prep = inputs.read(inputs.add(raw["preparation"]))
    if (
        prep.get("version") != 1
        or prep.get("kind") != "native-queue-followup-preparation"
        or prep.get("qualityApproved") is not False
    ):
        raise ReplayError("Expected immutable unapproved adapter preparation evidence")
    pins = record(prep.get("inputs"))
    if pins.get(str(Path(__file__).resolve())) != file_digest(Path(__file__).resolve()):
        raise ReplayError("Preparation requires the exact independent adapter code pin")
    for path, digest in pins.items():
        inputs.add({"path": path, "sha256": digest})
    for name in ("queue", "ready"):
        ref = inputs.add(prep.get(name))
        if pins.get(str(ref.path)) != ref.sha256:
            raise ReplayError("Preparation is missing its original queue or ready pin")
    inputs.current(Path(__file__).resolve())
    config, semantic = _terminal(inputs, prep, raw["terminal"])
    if inputs.add(raw["publication"]).path != _path(config.get("receiptOutput")):
        raise ReplayError("Release publication differs from the prepared exclusive destination")
    retirement = _retirement(inputs, raw, prep)
    _consumer(inputs, raw["consumer"], config=config, retirement=retirement)
    successor = _successor(inputs, raw, prep, host)
    cache = InputHashCache(inputs.refs, allow_uncached_overflow=True)
    with input_hash_context(cache):
        cache.prime()
        inputs.verify()
        verify_published(
            inputs.add(raw["publication"]).path,
            inputs.add(inputs.read(inputs.add(raw["publication"])).get("frameProof")).path,
            part=_integer(prep.get("assemblyPart")),
            retirement=retirement.path,
        )
        followup.validate_cleanup(inputs.add(prep.get("config")).path, semantic)
        cache.guard()
        _retirement(inputs, raw, prep)
        if _successor(inputs, raw, prep, host) != successor:
            raise ReplayError("Successor identity changed during release validation")
        cache.verify()
    return Release(
        _integer(prep.get("assemblyPart")),
        *successor,
        tuple(Ref(path, digest) for path, digest in inputs.refs.items()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare-from-ready",))
    parser.add_argument("template", type=Path)
    parser.add_argument("queue", type=Path)
    parser.add_argument("ready", type=Path)
    parser.add_argument("request_output", type=Path)
    parser.add_argument("prepared_output", type=Path)
    parser.add_argument("--native-complete", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(
            prepare_from_ready(
                template=args.template,
                queue=args.queue,
                ready=args.ready,
                native_complete=args.native_complete,
                request_output=args.request_output,
                prepared_output=args.prepared_output,
            )
        )
    except (ReplayError, OSError, ValueError) as error:
        print(f"Native followup preparation stopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
