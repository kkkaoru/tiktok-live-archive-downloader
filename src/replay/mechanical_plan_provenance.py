"""Bind mechanical request generation to bounded code and duration observations.

Source identities complement the existing fresh cleanup/native media digests. They
are not sample-content, frame-count, or delivery evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from math import floor
from pathlib import Path

from replay.core import ReplayError
from replay.delivery_render import Config
from replay.recordings import record
from replay.stages import Stage, StageError

VERSION = 1
MAX_BYTES = 4 * 1024 * 1024
MAX_PROBE_BYTES = 16 * 1024
MAX_MODULES = 192
MAX_INPUTS = 256
MAX_PROBES = 4096
PROBE_TIMEOUT = 60
ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src/replay"
SEEDS = (
    "__init__",
    "mechanical_plan_provenance",
    "mechanical_lane",
    "delivery_pipeline",
    "native_edit_parts",
    "native_invocation",
    "workflow",
)
PROBE_ARGUMENTS = ("-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0")
VERIFY_MODULE = "replay.mechanical_plan_provenance"


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    bytes: int
    mtime_ns: int
    ctime_ns: int
    mode: int

    def json(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "bytes": self.bytes,
            "mtimeNs": self.mtime_ns,
            "ctimeNs": self.ctime_ns,
            "mode": self.mode,
        }


def identity(path: Path) -> FileIdentity:
    """Observe a canonical regular file without reading media bytes."""
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ReplayError("Plan provenance requires canonical absolute files")
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise ReplayError("Plan provenance requires nonempty regular files")
    return FileIdentity(
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


def bounded_bytes(path: Path) -> bytes:
    before = identity(path)
    if before.bytes > MAX_BYTES:
        raise ReplayError("Plan provenance metadata exceeds its byte budget")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.device, before.inode):
            raise ReplayError("Plan provenance file changed before open")
        value = stream.read(MAX_BYTES + 1)
    if len(value) > MAX_BYTES or identity(path) != before:
        raise ReplayError("Plan provenance file changed during reading")
    return value


@dataclass(frozen=True, slots=True)
class Pin:
    path: Path
    sha256: str
    identity: FileIdentity

    def json(self) -> dict[str, object]:
        return {"path": str(self.path), "sha256": self.sha256, "identity": self.identity.json()}


def pin(path: Path) -> Pin:
    before = identity(path)
    raw = bounded_bytes(path)
    if identity(path) != before:
        raise ReplayError("Plan provenance pin changed")
    return Pin(path, hashlib.sha256(raw).hexdigest(), before)


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError("Ambiguous duplicate plan-provenance field")
        result[key] = value
    return result


def read(path: Path) -> dict[str, object]:
    value: object = json.loads(bounded_bytes(path), object_pairs_hook=_pairs)
    return record(value)


def _same(value: object, expected: object) -> bool:
    """Preserve JSON number types instead of Python's bool/int/float equivalence."""
    return json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(
        expected, sort_keys=True, allow_nan=False
    )


def code_paths() -> tuple[Path, ...]:
    """Follow explicit replay imports only, including lazy/relative imports."""
    pending = set(SEEDS)
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        if not re.fullmatch(r"(?:[a-z][a-z0-9_]*|__init__)", name) or len(found) >= MAX_MODULES:
            raise ReplayError("Mechanical helper import closure exceeds its scope")
        found.add(name)
        path = SOURCE_ROOT / f"{name}.py"
        for node in ast.walk(ast.parse(bounded_bytes(path).decode("utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1:
                    raise ReplayError("Mechanical helper import escapes replay")
                pending.update(
                    (node.module,) if node.module else (item.name for item in node.names)
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "replay":
                pending.update(item.name for item in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("replay.")
            ):
                pending.add(node.module.removeprefix("replay."))
            elif isinstance(node, ast.Import):
                pending.update(
                    item.name.removeprefix("replay.")
                    for item in node.names
                    if item.name.startswith("replay.")
                )
    return tuple(SOURCE_ROOT / f"{name}.py" for name in sorted(found))


def probe_tool() -> Path:
    return executable("ffprobe")


def executable(name: str) -> Path:
    selected = shutil.which(name)
    if selected is None:
        raise ReplayError(f"Explicit local {name} is unavailable")
    path = Path(selected).resolve(strict=True)
    identity(path)
    if not os.access(path, os.X_OK):
        raise ReplayError("Duration probe is not executable")
    return path


def parse_result(stdout: str, stderr: str, returncode: int) -> Fraction:
    if (
        type(returncode) is not int
        or returncode != 0
        or stderr.strip()
        or len(stdout.encode()) > MAX_PROBE_BYTES
        or len(stderr.encode()) > MAX_PROBE_BYTES
    ):
        raise ReplayError("Duration probe failed or exceeded its result budget")
    try:
        seconds = Fraction(stdout.strip())
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Could not read the source duration") from error
    if seconds <= 0:
        raise ReplayError("Could not read the source duration")
    return Fraction(floor(seconds * 1000), 1000)


@dataclass(frozen=True, slots=True)
class Observation:
    source: Path
    identity: FileIdentity
    argv: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int
    limit: Fraction

    def json(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "identity": self.identity.json(),
            "argv": list(self.argv),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "limitSeconds": str(self.limit),
        }


class Probes:
    """One generation's identity-guarded observations; no process-global path cache."""

    def __init__(self) -> None:
        self.tool = pin(probe_tool())
        self.observations: dict[Path, Observation] = {}

    def check(self, *, fresh: bool = False) -> None:
        if (
            identity(self.tool.path) != self.tool.identity
            or (fresh and pin(self.tool.path) != self.tool)
            or any(identity(path) != item.identity for path, item in self.observations.items())
        ):
            raise ReplayError("Duration source or probe executable changed")

    def limit(self, source: Path) -> Fraction:
        self.check()
        before = identity(source)
        if source not in self.observations:
            if len(self.observations) >= MAX_PROBES:
                raise ReplayError("Duration observation count exceeds its budget")
            argv = (str(self.tool.path), *PROBE_ARGUMENTS, str(source))
            result = subprocess.run(
                argv, check=False, capture_output=True, text=True, timeout=PROBE_TIMEOUT
            )
            limit = parse_result(result.stdout, result.stderr, result.returncode)
            if identity(source) != before or identity(self.tool.path) != self.tool.identity:
                raise ReplayError("Duration source or probe executable changed during probe")
            self.observations[source] = Observation(
                source, before, argv, result.stdout, result.stderr, result.returncode, limit
            )
        return self.observations[source].limit


def receipt_path(request: Path, inputs: Sequence[Path] | None = None) -> Path:
    candidates = (
        tuple(request.parent.glob(f"{request.stem}.mechanical-plan-*.json"))
        if inputs is None
        else inputs
    )
    matching = [
        path
        for path in candidates
        if path.parent == request.parent
        and re.fullmatch(
            re.escape(request.stem) + r"\.mechanical-plan-[a-f0-9]{64}\.json", path.name
        )
    ]
    if len(matching) != 1:
        raise ReplayError("Unbound or ambiguous legacy mechanical plan must be regenerated")
    path = matching[0]
    if pin(path).sha256 != path.name.removesuffix(".json").rsplit("-", 1)[1]:
        raise ReplayError("Mechanical provenance receipt changed")
    return path


def _clips(request: Path) -> list[dict[str, object]]:
    value = record(read(request).get("recipe")).get("clips")
    if not isinstance(value, list) or not 1 <= len(value) <= 60:
        raise ReplayError("Plan provenance requires bounded source clips")
    return [record(row) for row in value]


def _sources(request: Path) -> tuple[Path, ...]:
    sources: list[Path] = []
    for row in _clips(request):
        value = row.get("sourcePath")
        if not isinstance(value, str):
            raise ReplayError("Plan provenance source path is malformed")
        sources.append(Path(value))
    return tuple(dict.fromkeys(sources))


def _extend(stage: dict[str, object], paths: Sequence[Path]) -> None:
    inputs = stage.get("inputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or any(not isinstance(item, str) for item in inputs)
    ):
        raise ReplayError("Mechanical consumer inputs are malformed")
    result = list(dict.fromkeys([*inputs, *map(str, paths)]))
    if len(result) > MAX_INPUTS:
        raise ReplayError("Mechanical provenance exceeds the stage input budget")
    stage["inputs"] = result


class Generation:
    def __init__(self) -> None:
        self.probes = Probes()
        self.code = tuple(pin(path) for path in code_paths())
        self.python = pin(Path(sys.executable).resolve(strict=True))
        self._artifacts: dict[Path, Pin] = {}
        self._selections: list[str] = []

    def record_clip(self, value: dict[str, object]) -> None:
        self._selections.append(json.dumps(value, sort_keys=True))

    def finish(self) -> None:
        self.check(fresh=True)
        if self._selections:
            raise ReplayError("Mechanical generation left unbound serialized selections")

    def check(self, *, fresh: bool = False) -> None:
        self.probes.check(fresh=fresh)
        if (
            any(identity(item.path) != item.identity for item in self.code)
            or identity(self.python.path) != self.python.identity
            or any(identity(path) != saved.identity for path, saved in self._artifacts.items())
            or (
                fresh
                and (
                    tuple(pin(item.path) for item in self.code) != self.code
                    or pin(self.python.path) != self.python
                    or any(pin(path) != saved for path, saved in self._artifacts.items())
                )
            )
        ):
            raise ReplayError("Mechanical generation helper or Python changed")

    def bind(self, stages: list[dict[str, object]]) -> None:
        for stage in stages:
            inputs = stage.get("inputs")
            if not isinstance(inputs, list) or not inputs or not isinstance(inputs[0], str):
                raise ReplayError("Mechanical consumer request is absent")
            request = Path(inputs[0])
            request_pin = pin(request)
            sources = _sources(request)
            selected = [json.dumps(row, sort_keys=True) for row in _clips(request)]
            if selected != self._selections[: len(selected)]:
                raise ReplayError(
                    "Mechanical request differs from the actual serialized selections"
                )
            if any(path not in self.probes.observations for path in sources):
                raise ReplayError("Mechanical request lacks its actual duration observation")
            self.check()
            document = {
                "version": VERSION,
                "request": request_pin.json(),
                "code": [item.json() for item in self.code],
                "python": self.python.json(),
                "tool": self.probes.tool.json(),
                "probes": [self.probes.observations[path].json() for path in sources],
                "sampleContentVerified": False,
                "frameContentVerified": False,
            }
            raw = (json.dumps(document, sort_keys=True) + "\n").encode()
            if len(raw) > MAX_BYTES:
                raise ReplayError("Mechanical provenance receipt exceeds its byte budget")
            digest = hashlib.sha256(raw).hexdigest()
            path = request.with_name(f"{request.stem}.mechanical-plan-{digest}.json")
            with path.open("x", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(raw.decode())
            self._artifacts.update({request: request_pin, path: pin(path)})
            del self._selections[: len(selected)]
            _extend(
                stage,
                (path, *(item.path for item in self.code), self.python.path, self.probes.tool.path),
            )
            consumer = bind_verifier(stage)
            self._artifacts[consumer.path] = consumer
        self.check()


def _parent(request: Path) -> tuple[Path, Path]:
    manifest = request.parent.parent / "manifest.json"
    data = read(manifest)
    source = record(data.get("sourceRequest"))
    value = source.get("path")
    if not isinstance(value, str):
        raise ReplayError("Native part lacks original mechanical request provenance")
    original = Path(value)
    if source.get("sha256") != pin(original).sha256:
        raise ReplayError("Original mechanical request changed")
    parts = data.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_PROBES:
        raise ReplayError("Mechanical partition manifest is malformed")
    selected = [record(item) for item in parts if record(item).get("path") == str(request)]
    if len(selected) != 1 or selected[0].get("sha256") != pin(request).sha256:
        raise ReplayError("Native part does not match its original partition manifest")
    return original, manifest


def bind_render(stages: list[dict[str, object]]) -> None:
    for stage in stages:
        inputs = stage.get("inputs")
        if not isinstance(inputs, list) or not inputs or not isinstance(inputs[0], str):
            raise ReplayError("Native part request is absent")
        original, manifest = _parent(Path(inputs[0]))
        data = read(receipt_path(original))
        code = data.get("code")
        if not isinstance(code, list):
            raise ReplayError("Unbound legacy mechanical plan must be regenerated")
        paths = [Path(str(record(item).get("path"))) for item in code]
        _extend(
            stage,
            (
                original,
                manifest,
                receipt_path(original),
                *paths,
                Path(str(record(data.get("python")).get("path"))),
                Path(str(record(data.get("tool")).get("path"))),
            ),
        )
        bind_verifier(stage)


def _validate_stages(stages: Sequence[Stage]) -> None:
    code = tuple(pin(path) for path in code_paths())
    python = pin(Path(sys.executable).resolve(strict=True))
    checked: dict[Path, Pin] = {}
    sources: dict[Path, FileIdentity] = {}
    for stage in stages:
        request = stage.inputs[0]
        request_pin = pin(request)
        manifest_pin = (
            pin(request.parent.parent / "manifest.json")
            if "replay.native_invocation" in stage.command
            else None
        )
        original, manifest = (
            _parent(request) if "replay.native_invocation" in stage.command else (request, None)
        )
        path = receipt_path(original, stage.inputs)
        original_pin = request_pin if original == request else pin(original)
        receipt_pin = pin(path)
        data = read(path)
        if set(data) != {
            "version",
            "request",
            "code",
            "python",
            "tool",
            "probes",
            "sampleContentVerified",
            "frameContentVerified",
        } or (
            type(data.get("version")) is not int
            or data["version"] != VERSION
            or data["sampleContentVerified"] is not False
            or data["frameContentVerified"] is not False
            or not _same(data["request"], original_pin.json())
            or not _same(data["code"], [item.json() for item in code])
            or not _same(data["python"], python.json())
        ):
            raise ReplayError("Mechanical request/helper provenance differs")
        if record(data["tool"]).get("path") != str(probe_tool()):
            raise ReplayError("Mechanical selected probe executable changed")
        tool = pin(probe_tool())
        if not _same(data["tool"], tool.json()) or not os.access(tool.path, os.X_OK):
            raise ReplayError("Mechanical probe executable provenance differs")
        required = {original, path, python.path, tool.path, *(item.path for item in code)}
        if manifest is not None:
            required.add(manifest)
        if not required.issubset(stage.inputs) or str(request) not in stage.command:
            raise ReplayError("Mechanical consumer omits required provenance inputs")
        checked.update({item.path: item for item in (original_pin, request_pin, receipt_pin, tool)})
        if manifest_pin is not None:
            checked[manifest_pin.path] = manifest_pin
        probes = data["probes"]
        if not isinstance(probes, list) or len(probes) != len(_sources(original)):
            raise ReplayError("Mechanical duration observations are incomplete")
        for source, raw in zip(_sources(original), probes, strict=True):
            observed = record(raw)
            source_identity = identity(source)
            if set(observed) != {
                "source",
                "identity",
                "argv",
                "stdout",
                "stderr",
                "returncode",
                "limitSeconds",
            } or (
                observed.get("source") != str(source)
                or not _same(observed.get("identity"), source_identity.json())
                or observed.get("argv") != [str(tool.path), *PROBE_ARGUMENTS, str(source)]
                or not isinstance(observed.get("stdout"), str)
                or not isinstance(observed.get("stderr"), str)
                or type(observed.get("returncode")) is not int
            ):
                raise ReplayError("Mechanical source/probe observation differs")
            stdout, stderr, returncode = (
                observed["stdout"],
                observed["stderr"],
                observed["returncode"],
            )
            if (
                not isinstance(stdout, str)
                or not isinstance(stderr, str)
                or not isinstance(returncode, int)
            ):
                raise ReplayError("Malformed duration probe result")
            if observed["limitSeconds"] != str(parse_result(stdout, stderr, returncode)):
                raise ReplayError("Mechanical parsed duration differs from its raw result")
            sources[source] = source_identity
    if (
        tuple(pin(item.path) for item in code) != code
        or pin(python.path) != python
        or any(pin(path) != item for path, item in checked.items())
        or any(identity(path) != observed for path, observed in sources.items())
    ):
        raise ReplayError("Mechanical verifier helper or Python changed")


def _stage_document(stage: Stage) -> dict[str, object]:
    return {
        "name": stage.name,
        "command": list(stage.command),
        "verify_command": list(stage.verify_command),
        "inputs": list(map(str, stage.inputs)),
        "outputs": list(map(str, stage.outputs)),
        "timeout_seconds": stage.timeout_seconds,
        "adopt_existing": stage.adopt_existing,
    }


def _parse_stage(document: object) -> Stage:
    from replay.workflow import parse_stages

    [stage] = parse_stages({"version": 1, "stages": [document]})
    return stage


def _domain(stage: Stage, *, execute: bool) -> None:
    """Only the original native verifier or deterministic partition domain is allowed."""
    from replay import native_invocation

    argv = stage.command
    request = stage.inputs[0]
    if (
        len(argv) < 3
        or argv[1] != "-m"
        or Path(argv[0]).resolve(strict=True) != Path(sys.executable).resolve(strict=True)
    ):
        raise ReplayError("Mechanical consumer does not use the configured Python")
    if argv[2] == "replay.native_edit_parts":
        expected = (
            argv[0],
            "-m",
            argv[2],
            str(request),
            str(stage.outputs[0].parent),
            "--sha256",
            pin(request).sha256,
        )
        if (
            argv != expected
            or stage.outputs != (stage.outputs[0].parent / "manifest.json",)
            or stage.verify_command
        ):
            raise ReplayError("Mechanical partition consumer arguments differ")
        if execute:
            _verify_partition(request, stage.outputs[0])
    elif argv[2] == "replay.native_invocation":
        if (
            len(argv) != 7
            or argv[3] != "run"
            or argv[5] != str(request)
            or Path(argv[4]) not in stage.inputs
            or stage.outputs != native_invocation.paths(request)
            or stage.verify_command != (*argv[:3], "verify", *argv[4:])
        ):
            raise ReplayError("Original native read-only verifier arguments differ")
        if execute:
            result = subprocess.run(
                stage.verify_command,
                stdin=subprocess.DEVNULL,
                check=False,
                timeout=stage.timeout_seconds,
            )
            if result.returncode != 0:
                raise ReplayError("Original native domain verifier failed")
    elif _is_join(stage):
        if (
            len(argv) != 9
            or argv[4] != str(request)
            or argv[5:8] != ("--output", str(stage.outputs[0]), "--rate")
            or not argv[8].isdigit()
            or not 1 <= int(argv[8]) <= 60
            or len(stage.outputs) != 1
            or stage.verify_command
        ):
            raise ReplayError("Mechanical join consumer arguments differ")
        if execute:
            _verify_join(stage)
    else:
        raise ReplayError("Unsupported mechanical consumer domain")


def _verify_partition(request: Path, manifest: Path) -> None:
    """Read the exact original partition algorithm's outputs; never call its exporter."""
    from replay import native_edit_parts

    request_pin, manifest_pin = pin(request), pin(manifest)
    document = read(request)
    data = read(manifest)
    expected = native_edit_parts.partition(document)
    join = manifest.with_name("join-plan.json")
    if set(data) != {
        "version",
        "sourceRequest",
        "parts",
        "joinPlan",
        "renderVerified",
        "losslessAssemblyVerified",
        "deliveryComplete",
    } or (
        type(data["version"]) is not int
        or data["version"] != 1
        or data["sourceRequest"] != {"path": str(request), "sha256": request_pin.sha256}
        or any(
            data[key] is not False
            for key in ("renderVerified", "losslessAssemblyVerified", "deliveryComplete")
        )
    ):
        raise ReplayError("Original partition manifest differs")
    parts = data["parts"]
    if not isinstance(parts, list) or len(parts) != len(expected):
        raise ReplayError("Original partition outputs are incomplete")
    destination = document.get("outputDirectory")
    if not isinstance(destination, str):
        raise ReplayError("Original partition output directory is absent")
    namespace = hashlib.sha256(str(manifest.parent).encode()).hexdigest()
    checked = [request_pin, manifest_pin, pin(join)]
    for index, (raw, (offset, frames, recipe)) in enumerate(zip(parts, expected, strict=True)):
        path = manifest.parent / f"part-{index:04d}/request.json"
        saved = pin(path)
        if not _same(
            raw,
            {
                "path": str(path),
                "sha256": saved.sha256,
                "offsetFrame": offset,
                "frames": frames,
            },
        ) or not _same(
            read(path),
            {
                **recipe,
                "outputDirectory": str(Path(destination) / f"parts-{namespace}" / path.parent.name),
            },
        ):
            raise ReplayError("Original partition request differs")
        checked.append(saved)
    if data["joinPlan"] != {"path": str(join), "sha256": checked[2].sha256} or not _same(
        json.loads(bounded_bytes(join), object_pairs_hook=_pairs),
        [{"durationFrames": item[1]} for item in expected],
    ):
        raise ReplayError("Original partition join plan differs")
    if any(pin(item.path) != item for item in checked):
        raise ReplayError("Original partition output changed during verification")


def bind_verifier(row: dict[str, object]) -> Pin:
    original = _parse_stage(row)
    _domain(original, execute=False)
    document: dict[str, object] = {"version": VERSION}
    if _is_join(original):
        contract = _join_contract(original)
        _extend(row, _join_paths(contract))
        original = _parse_stage(row)
        document["join"] = contract
    document["stage"] = _stage_document(original)
    raw = (json.dumps(document, sort_keys=True) + "\n").encode()
    if len(raw) > MAX_BYTES:
        raise ReplayError("Mechanical consumer contract exceeds its byte budget")
    digest = hashlib.sha256(raw).hexdigest()
    path = original.inputs[0].with_name(
        f"{original.inputs[0].stem}.mechanical-consumer-{digest}.json"
    )
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw.decode())
    saved = pin(path)
    _extend(row, (path,))
    row["verify_command"] = [
        original.command[0],
        "-m",
        VERIFY_MODULE,
        str(path),
        "--sha256",
        digest,
    ]
    return saved


def _consumer(path: Path, sha256: str) -> tuple[Stage, Pin]:
    saved = pin(path)
    if saved.sha256 != sha256 or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ReplayError("Mechanical consumer contract changed")
    data = read(path)
    stage = _parse_stage(data.get("stage"))
    if (
        set(data) != ({"version", "stage", "join"} if _is_join(stage) else {"version", "stage"})
        or type(data["version"]) is not int
        or data["version"] != VERSION
    ):
        raise ReplayError("Unsupported mechanical consumer contract")
    if path != stage.inputs[0].with_name(
        f"{stage.inputs[0].stem}.mechanical-consumer-{sha256}.json"
    ):
        raise ReplayError("Mechanical consumer contract request differs")
    _domain(stage, execute=False)
    return stage, saved


def validate_stages(stages: Sequence[Stage]) -> None:
    originals: list[Stage] = []
    contracts: list[Pin] = []
    for stage in stages:
        argv = stage.verify_command
        if (
            len(argv) != 6
            or argv[:3] != (stage.command[0], "-m", VERIFY_MODULE)
            or argv[4] != "--sha256"
        ):
            raise ReplayError("Unbound legacy mechanical plan must be regenerated")
        original, saved = _consumer(Path(argv[3]), argv[5])
        expected = {
            **_stage_document(original),
            "inputs": [*map(str, original.inputs), str(saved.path)],
            "verify_command": list(argv),
        }
        if _stage_document(stage) != expected:
            raise ReplayError("Mechanical stage differs from its sealed consumer contract")
        originals.append(original)
        contracts.append(saved)
    _validate_stages(originals)
    if any(pin(item.path) != item for item in contracts):
        raise ReplayError("Mechanical consumer contract changed during validation")


def verify(path: Path, sha256: str) -> None:
    stage, saved = _consumer(path, sha256)
    if _is_join(stage):
        _validate_join(stage, read(path)["join"])
    else:
        _validate_stages((stage,))
    _domain(stage, execute=True)
    if _is_join(stage):
        _validate_join(stage, read(path)["join"])
    else:
        _validate_stages((stage,))
    if pin(saved.path) != saved:
        raise ReplayError("Mechanical consumer contract changed during domain verification")


def _is_join(stage: Stage) -> bool:
    return stage.command[1:4] in (
        ("-m", "replay.mechanical_join", "join-manifest"),
        ("-m", "replay.delivery_render", "join-files"),
    )


def _join_parts(stage: Stage) -> tuple[tuple[Path, int], ...]:
    from replay import mechanical_join

    path = stage.inputs[0]
    if stage.command[2] == "replay.mechanical_join":
        return tuple(
            (part.path, part.frames) for part in mechanical_join.parse_manifest(read(path))
        )
    raw: object = json.loads(bounded_bytes(path), object_pairs_hook=_pairs)
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_PROBES:
        raise ReplayError("Mechanical classic join plan is incomplete")
    parts: list[tuple[Path, int]] = []
    for value in raw:
        row = record(value)
        source, frames = row.get("sourcePath"), row.get("durationFrames")
        if (
            set(row) != {"sourcePath", "durationFrames"}
            or not isinstance(source, str)
            or type(frames) is not int
            or frames <= 0
        ):
            raise ReplayError("Mechanical classic join plan differs")
        parts.append((Path(source), frames))
    if len({path for path, _ in parts}) != len(parts):
        raise ReplayError("Mechanical classic join repeats a part")
    return tuple(parts)


def _join_contract(stage: Stage) -> dict[str, object]:
    plan = pin(stage.inputs[0])
    code = tuple(pin(path) for path in code_paths())
    python = pin(Path(sys.executable).resolve(strict=True))
    tools = tuple(pin(executable(name)) for name in ("ffprobe", "ffmpeg"))
    sources = [
        {"path": str(path), "identity": identity(path).json()} for path, _ in _join_parts(stage)
    ]
    if (
        pin(stage.inputs[0]) != plan
        or tuple(pin(item.path) for item in code) != code
        or tuple(pin(item.path) for item in tools) != tools
        or pin(python.path) != python
        or sources
        != [
            {"path": str(path), "identity": identity(path).json()} for path, _ in _join_parts(stage)
        ]
    ):
        raise ReplayError("Mechanical join generation inputs changed")
    return {
        "plan": plan.json(),
        "code": [item.json() for item in code],
        "python": python.json(),
        "tools": [item.json() for item in tools],
        "sources": sources,
    }


def _join_paths(contract: dict[str, object]) -> tuple[Path, ...]:
    code, tools = contract["code"], contract["tools"]
    if not isinstance(code, list) or not isinstance(tools, list):
        raise ReplayError("Mechanical join code/tool pins are incomplete")
    return tuple(
        Path(str(record(item).get("path"))) for item in [*code, *tools, contract["python"]]
    )


def _validate_join(stage: Stage, contract: object) -> None:
    expected = _join_contract(stage)
    required = set(_join_paths(expected))
    if stage.command[2] == "replay.delivery_render":
        required.update(path for path, _ in _join_parts(stage))
    if not _same(contract, expected) or not required.issubset(stage.inputs):
        raise ReplayError("Mechanical join helper/tool/plan/source provenance differs")


def validate_join_stage(stage: Stage) -> tuple[Pin, ...]:
    """Validate genuine new join pins; historical base plans have no such claim."""
    argv = stage.verify_command
    if (
        len(argv) != 6
        or argv[:3] != (stage.command[0], "-m", VERIFY_MODULE)
        or argv[4] != "--sha256"
    ):
        raise ReplayError("Unbound mechanical join verifier must be regenerated")
    original, saved = _consumer(Path(argv[3]), argv[5])
    expected = {
        **_stage_document(original),
        "inputs": [*map(str, original.inputs), str(saved.path)],
        "verify_command": list(argv),
    }
    if not _is_join(original) or _stage_document(stage) != expected:
        raise ReplayError("Mechanical join differs from its sealed consumer contract")
    _validate_join(original, read(saved.path)["join"])
    if pin(saved.path) != saved:
        raise ReplayError("Mechanical join consumer contract changed")
    return tuple(
        pin(path)
        for path in (
            Path(sys.executable).resolve(strict=True),
            executable("ffprobe"),
            executable("ffmpeg"),
        )
    )


def join_needs_manifest(rendered: Sequence[tuple[Path, int]], config: Config) -> bool:
    from replay import mechanical_join

    paths = {
        config.work / "join-plan.json",
        config.cuts,
        config.chunks,
        *config.cut_evidence,
        *code_paths(),
        Path(sys.executable).resolve(strict=True),
        executable("ffprobe"),
        executable("ffmpeg"),
    }
    for request, frames in rendered:
        response = read(request.with_name("render.response.json"))
        movie = response.get("outputPath")
        if not isinstance(movie, str) or response.get("frameCount") != frames:
            raise ReplayError("Native join response differs from its plan")
        paths.add(Path(movie))
    # A distinct content-addressed consumer contract is also a required input.
    return mechanical_join.needs_manifest(0, len(paths) + 1 - mechanical_join.JOIN_FIXED_INPUTS)


def _verify_join(stage: Stage) -> None:
    from replay import mechanical_join, native_join

    if stage.command[2] == "replay.mechanical_join":
        parts = mechanical_join.verify_parts(mechanical_join.parse_manifest(read(stage.inputs[0])))
    else:
        parts = tuple(native_join.PartFile(path, frames) for path, frames in _join_parts(stage))
    stream = native_join.verify_parts(parts)
    output = native_join.probe_video(stage.outputs[0])
    if (output.codec, output.time_base, output.index, output.packets) != (
        stream.codec,
        stream.time_base,
        stream.index,
        sum(part.frames for part in parts),
    ):
        raise ReplayError("Joined output differs from original native part domain")


@contextmanager
def join_fixes() -> Iterator[None]:
    """Bind only newly generated joins before their original exclusive writer runs."""
    from replay import delivery_render, mechanical_join

    original_write, original_stage = delivery_render._write_new, mechanical_join.join_stage

    def write(path: Path, value: object) -> Path:
        if path.name == "join-workflow.json":
            data = record(value)
            rows = data.get("stages")
            if (
                set(data) != {"version", "stages"}
                or type(data["version"]) is not int
                or data["version"] != 1
                or not isinstance(rows, list)
                or len(rows) != 1
            ):
                raise ReplayError("Mechanical join workflow is malformed")
            row = record(rows[0])
            if not _is_join(_parse_stage(row)):
                raise ReplayError("Mechanical join workflow has a different producer")
            bind_verifier(row)
            value = {"version": 1, "stages": [row]}
        return original_write(path, value)

    def join_stage(
        manifest: Path,
        output: Path,
        rate: int,
        command: Sequence[str],
        extra_inputs: Sequence[Path],
    ) -> dict[str, object]:
        row = original_stage(manifest, output, rate, command, extra_inputs)
        row["timeout_seconds"] = delivery_render.JOIN_TIMEOUT_SECONDS
        bind_verifier(row)
        return row

    delivery_render._write_new, mechanical_join.join_stage = write, join_stage
    try:
        yield
    finally:
        delivery_render._write_new, mechanical_join.join_stage = original_write, original_stage


class Arguments(argparse.Namespace):
    consumer: Path
    sha256: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consumer", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv, namespace=Arguments())
    try:
        verify(args.consumer, args.sha256)
    except (ReplayError, StageError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Mechanical provenance refusal: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
