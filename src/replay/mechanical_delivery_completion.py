"""Explicit mechanical evidence adapter; never renders or manufactures receipts.

The adapter label is distinct from the existing, genuine Motion/Compressor v2
index backend. Recorded execution, current content, numerical quality, sync and
standard publication are separate outcomes. Historical records are read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import ModuleType

from replay import aac_presentation_acceptance as aac
from replay import delivery_completion_v2 as standard
from replay import frame_grid, input_hash_snapshot, source_caption_binding
from replay import native_delivery_evidence as evidence
from replay import stages as stage_contract
from replay.core import ReplayError
from replay.edit_completion import REQUIRED_STAGES, EditConfig, load_config
from replay.frame_grid import select
from replay.input_hash_snapshot import MAX_INPUTS, InputHashSnapshot
from replay.mechanical_join import parse_manifest
from replay.stages import StageError, file_digest

BACKEND = "mechanical-motion-compositor-whole"
STANDARD_BACKEND = "motion-compressor-direct"
VERSION = 1
MAX_PRODUCTIONS = 4096
MAX_PARTS = 4096
MAX_SELECTIONS = 256
MAX_FRAMES = 1_000_000
INDEX_KEYS = frozenset(
    {
        "version",
        "backend",
        "recording",
        "configuration",
        "context",
        "domains",
        "productions",
        "currentInputs",
    }
)
OPTIONAL_KEYS = frozenset({"standardIndex", "standardReferences", "observation", "externalInputs"})
CONTEXT_REFERENCES = ("source", "renderConfig", "clean", "voice", "cuts", "movie", "srt")
CONTEXT_KEYS = frozenset((*CONTEXT_REFERENCES, "frames", "fps", "width", "height"))
DOMAINS = frozenset({"cleanup", "asr", "separation", "composition", "sync"})
REQUIRED_ROLES = frozenset((*REQUIRED_STAGES, "compose", "join", "encode", "technical"))
ROLES = REQUIRED_ROLES | {"separation-source"}
PRODUCTION_KEYS = frozenset(
    {"role", "workflow", "stage", "checkpoint", "inventory", "outputs", "cwd"}
)
ERRORS = (ReplayError, StageError, ValueError, OSError, KeyError, ZeroDivisionError)
LIMITATIONS = (
    "Local receipts are evidence, not authenticated attestations.",
    "Current endpoint hashes do not prove that bytes were never changed and restored.",
    "Readability, speech semantics, summary accuracy and Viewer interaction are not approved.",
    "An adapter assessment does not publish media or replace the standard register gate.",
)


def code_inputs() -> tuple[Path, ...]:
    """Named consumer entry points; callers must also pin their declared dependencies."""
    return tuple(
        _code_path(module)
        for module in (
            aac,
            standard,
            evidence,
            source_caption_binding,
            frame_grid,
            input_hash_snapshot,
            stage_contract,
        )
    ) + (Path(__file__).resolve(strict=True),)


def _code_path(module: ModuleType) -> Path:
    name = getattr(module, "__file__", None)
    if not isinstance(name, str):
        raise ValueError("Reader module has no source path")
    return Path(name).resolve(strict=True)


@dataclass(frozen=True)
class Context:
    data: dict[str, object]
    config: EditConfig
    rows: dict[str, dict[str, object]]

    def reference(self, name: str) -> dict[str, str]:
        return evidence.reference(self.data[name])

    def path(self, name: str) -> Path:
        return Path(self.reference(name)["path"])

    def count(self, name: str) -> int:
        return evidence.integer(self.data[name])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact(value: object, keys: frozenset[str], name: str) -> dict[str, object]:
    result = evidence.mapping(value)
    require(set(result) == keys, f"Unsupported {name} fields")
    return result


def _positive(value: object) -> int:
    result = evidence.integer(value)
    require(result > 0, "Expected positive integer")
    return result


def _list(value: object, maximum: int, name: str, *, nonempty: bool = False) -> list[object]:
    if not isinstance(value, list) or not int(nonempty) <= len(value) <= maximum:
        raise ValueError(f"Invalid/budgeted {name}")
    return list(value)


def _fraction(value: object) -> Fraction:
    require(isinstance(value, str), "Expected explicit rational clock")
    result = Fraction(str(value))
    require(result >= 0, "Negative clock")
    return result


def _inventory(value: object) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for raw in _list(value, MAX_INPUTS, "current inputs", nonempty=True):
        item = evidence.mapping(raw)
        if item.get("bytes") == 0:
            row = _exact(
                item,
                frozenset({"path", "bytes", "sha256", "device", "inode", "mtimeNs", "ctimeNs"}),
                "empty process stream",
            )
            empty_path = Path(evidence.string(row["path"]))
            require(
                type(row["bytes"]) is int
                and empty_path.is_absolute()
                and str(empty_path) == row["path"]
                and ".." not in empty_path.parts
                and empty_path.suffix in {".stdout", ".stderr"}
                and row["sha256"] == hashlib.sha256(b"").hexdigest(),
                "Only original empty process streams may have zero bytes",
            )
            for key in ("device", "inode", "mtimeNs", "ctimeNs"):
                _positive(row[key])
        else:
            row = next(iter(evidence.fingerprint_rows([item]).values()))
        path = evidence.string(row["path"])
        require(path not in result, "Duplicate current input")
        result[path] = row
    return result


def _pin(rows: Mapping[str, dict[str, object]], value: object) -> dict[str, str]:
    ref = evidence.reference(value)
    require(set(evidence.mapping(value)) == {"path", "sha256"}, "Expected exact reference")
    require(
        ref["path"] in rows and rows[ref["path"]]["sha256"] == ref["sha256"],
        "Reference absent from current declared closure",
    )
    return ref


def _identity(row: dict[str, object]) -> None:
    path = Path(evidence.string(row["path"]))
    require(path.resolve(strict=True) == path, "Current input is linked")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "Current input is not a regular file")
    require(
        (info.st_size, info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns)
        == tuple(
            evidence.integer(row[key]) for key in ("bytes", "device", "inode", "mtimeNs", "ctimeNs")
        ),
        "Current identity differs; stat equality is not content verification",
    )


def _bind(data: dict[str, object], root: Path, movie: Path) -> Context:
    require(
        INDEX_KEYS <= set(data) <= INDEX_KEYS | OPTIONAL_KEYS
        and type(data["version"]) is int
        and data["version"] == VERSION
        and data["backend"] == BACKEND,
        "Unsupported mechanical index/backend",
    )
    config = load_config(root)
    rows = _inventory(data["currentInputs"])
    configuration = _pin(rows, data["configuration"])
    require(Path(configuration["path"]) == root / "edit-config.json", "Different edit config")
    context = Context(_exact(data["context"], CONTEXT_KEYS, "context"), config, rows)
    require(data["recording"] == root.name, "Different recording")
    for name in CONTEXT_REFERENCES:
        _pin(rows, context.data[name])
    require(
        context.path("source") == config.source and context.path("movie") == movie,
        "Different source/movie",
    )
    require(context.path("srt") == movie.with_suffix(".srt"), "Different canonical movie SRT")
    for name in ("frames", "fps", "width", "height"):
        _positive(context.data[name])
    require(context.count("frames") <= MAX_FRAMES, "Edited frame budget exceeded")
    require(set(evidence.mapping(data["domains"])) <= DOMAINS, "Unknown mechanical domain")
    _list(data["productions"], MAX_PRODUCTIONS, "productions")
    external = evidence.hashes(data.get("externalInputs", {}))
    for path, digest in external.items():
        _pin(rows, {"path": path, "sha256": digest})
    for path in rows:
        candidate = Path(path)
        require(
            candidate == config.source
            or candidate.is_relative_to(root.parents[1])
            or (config.storage_root is not None and candidate.is_relative_to(config.storage_root))
            or path in external,
            "Input escapes repository/source/storage/exact external pins",
        )
        require(
            candidate == config.source
            or not {"private", "downloads"}.intersection(candidate.parts),
            "Undeclared user-data input",
        )
    return context


def _productions(
    data: dict[str, object], context: Context, reader: evidence.MetadataReader
) -> tuple[set[str], dict[str, str], dict[str, dict[str, str]]]:
    roles: set[str] = set()
    outputs: dict[str, str] = {}
    dependencies: dict[str, dict[str, str]] = {}
    names: set[tuple[str, str]] = set()
    for ref in _list(data["productions"], MAX_PRODUCTIONS, "productions"):
        item = _exact(reader.read(_pin(context.rows, ref)), PRODUCTION_KEYS, "production")
        role, name = evidence.string(item["role"]), evidence.string(item["stage"])
        require(role in ROLES and (role, name) not in names, "Unknown/duplicate producer role")
        names.add((role, name))
        graph = reader.read(_pin(context.rows, item["workflow"]))
        stages = [
            stage for stage in evidence.rows(graph.get("stages")) if stage.get("name") == name
        ]
        require(len(stages) == 1, "Missing/ambiguous original producer")
        stage = stages[0]
        inventory = evidence.hashes(
            reader.read(_pin(context.rows, item["inventory"])).get("inputs")
        )
        inputs = {path: inventory[path] for path in evidence.strings(stage.get("inputs"))}
        expected = evidence.hashes(item["outputs"])
        for path, digest in inputs.items():
            _pin(context.rows, {"path": path, "sha256": digest})
            require(
                path not in outputs or outputs[path] == digest,
                "Producer/consumer input hash differs",
            )
        for path, digest in expected.items():
            _pin(context.rows, {"path": path, "sha256": digest})
            require(path not in outputs, "Multiple original output producers")
        evidence.stage_provenance(
            stage=stage,
            checkpoint=reader.read(_pin(context.rows, item["checkpoint"])),
            input_hashes=inputs,
            cwd=evidence.string(item["cwd"]),
            expected_outputs=expected,
        )
        roles.add(role)
        outputs.update(expected)
        dependencies.update({path: inputs for path in expected})
    return roles, outputs, dependencies


def _domain(
    data: dict[str, object], name: str, context: Context, reader: evidence.MetadataReader
) -> dict[str, object]:
    domains = evidence.mapping(data["domains"])
    require(name in domains, f"Missing actual {name} evidence")
    return reader.read(_pin(context.rows, domains[name]))


def _separation(
    raw: dict[str, object],
    context: Context,
    outputs: dict[str, str],
    dependencies: dict[str, dict[str, str]],
) -> None:
    item = _exact(
        raw,
        frozenset(
            {"version", "source", "modelInput", "firstPassVoice", "modelPins", "producerOutput"}
        ),
        "separation",
    )
    require(type(item["version"]) is int and item["version"] == 1, "Wrong separation version")
    evidence.same(item["source"], context.reference("source"))
    first = _pin(context.rows, item["firstPassVoice"])
    model_input = _pin(context.rows, item["modelInput"])
    require(
        first["path"] != model_input["path"], "Independent separation/reference source required"
    )
    require(first == _pin(context.rows, item["producerOutput"]), "Wrong separation output")
    require(
        outputs.get(first["path"]) == first["sha256"], "No original separation output checkpoint"
    )
    pins = evidence.hashes(item["modelPins"])
    require(len(pins) >= 2, "Missing actual separation model/config pins")
    for path, digest in pins.items():
        _pin(context.rows, {"path": path, "sha256": digest})
    source = context.reference("source")
    require(
        dependencies.get(model_input["path"], {}).get(source["path"]) == source["sha256"],
        "Original model-input PCM producer does not bind the original source",
    )
    original = dependencies.get(first["path"], {})
    require(
        original.get(model_input["path"]) == model_input["sha256"]
        and all(original.get(path) == digest for path, digest in pins.items()),
        "Original separation producer omits actual model input/config/weights",
    )


def _source_selection(
    raw: dict[str, object], context: Context, reader: evidence.MetadataReader
) -> None:
    keys = frozenset({"media", "proof", "start", "frames", "nativeDuration"})
    item = _exact(raw, keys, "frame selection")
    media = _pin(context.rows, item["media"])
    evidence.same(media, context.reference("clean"))
    proof = reader.read(_pin(context.rows, item["proof"]))
    evidence.same(proof.get("media"), media)
    require(
        proof.get("kind") == "frame-grid-media-inventory" and proof.get("version") == 1,
        "Missing actual frame-grid inventory",
    )
    count, ticks, base = (_positive(proof.get(key)) for key in ("frames", "ticks", "timebase"))
    require(count <= MAX_FRAMES, "Source frame budget exceeded")
    pts = _list(proof.get("pts"), MAX_FRAMES, "source PTS", nonempty=True)
    require(len(pts) == count, "Incomplete actual source PTS")
    rate = context.count("fps")
    require(
        all(
            Fraction(evidence.integer(value), base) == Fraction(index, rate)
            for index, value in enumerate(pts)
        ),
        "Source frame grid differs",
    )
    frames, start = _positive(item["frames"]), _fraction(item["start"])
    canonical = select(
        rate=rate,
        media_frames=count,
        media_ticks=ticks,
        timebase=base,
        start_seconds=start,
        duration_seconds=Fraction(frames, rate),
    )
    require(
        start * rate == canonical.start and canonical.frames == frames,
        "Unrepresentable canonical selection",
    )
    duration = _fraction(item["nativeDuration"])
    require(
        duration > 0 and 0 <= Fraction(frames, rate) - duration <= Fraction(1, 2 * rate),
        "Native duration clamp exceeds half-frame",
    )


def _composition(
    raw: dict[str, object],
    context: Context,
    reader: evidence.MetadataReader,
    outputs: dict[str, str],
    dependencies: dict[str, dict[str, str]],
) -> dict[str, object]:
    item = _exact(
        raw, frozenset({"version", "parts", "joinPlan", "joinFormat", "joinedMovie"}), "composition"
    )
    require(type(item["version"]) is int and item["version"] == 1, "Wrong composition version")
    parts = _list(item["parts"], MAX_PARTS, "native parts", nonempty=True)
    plan = _pin(context.rows, item["joinPlan"])
    if item["joinFormat"] == "manifest":
        manifest = parse_manifest(reader.read(plan))
    else:
        require(item["joinFormat"] == "join-files", "Unsupported join format")
        path = Path(plan["path"])
        require(path.stat().st_size <= evidence.MAX_RECORD_BYTES, "Join plan byte budget exceeded")
        require(file_digest(path) == plan["sha256"], "Join plan content changed")
        reader.bytes_read += path.stat().st_size
        require(reader.bytes_read <= evidence.MAX_TOTAL_BYTES, "Metadata byte budget exceeded")
        rows = _list(json.loads(path.read_text()), MAX_PARTS, "join plan", nonempty=True)
        manifest = parse_manifest(
            {
                "version": 1,
                "parts": [
                    {
                        **evidence.mapping(row),
                        "sha256": evidence.reference(part["movie"])["sha256"],
                    }
                    for row, part in zip(rows, map(evidence.mapping, parts), strict=True)
                ],
            }
        )
    require(len(manifest) == len(parts), "Join manifest/native part coverage differs")
    total = 0
    for number, (raw_part, joined) in enumerate(zip(parts, manifest, strict=True), 1):
        part = _exact(
            raw_part,
            frozenset(
                {
                    "request",
                    "response",
                    "movie",
                    "proof",
                    "tool",
                    "frames",
                    "selections",
                    "sourceCaptionBinding",
                }
            ),
            "native part",
        )
        movie = _pin(context.rows, part["movie"])
        frames = _positive(part["frames"])
        require(
            str(joined.path) == movie["path"]
            and joined.sha256 == movie["sha256"]
            and joined.frames == frames,
            "Join part identity/order/frame count differs",
        )
        response = reader.read(_pin(context.rows, part["response"]))
        require(
            response.get("outputPath") == movie["path"]
            and type(response.get("frameCount")) is int
            and response.get("frameCount") == frames,
            "Actual native output differs",
        )
        duration = float(Fraction(frames, context.count("fps")))
        require(
            all(
                type(response.get(key)) in (int, float)
                and math.isfinite(float(str(response[key])))
                and response[key] == duration
                for key in ("expectedDurationSeconds", "actualDurationSeconds")
            ),
            "Native output clock differs",
        )
        request = _pin(context.rows, part["request"])
        evidence.native_part(
            reader.read(_pin(context.rows, part["proof"])),
            partition=dict(request),
            movie=dict(movie),
            part=number,
            frames=frames,
            fps=context.count("fps"),
            width=context.count("width"),
            height=context.count("height"),
            tool=dict(_pin(context.rows, part["tool"])),
        )
        require(
            outputs.get(movie["path"]) == movie["sha256"],
            "Native movie not an original declared output",
        )
        binding = reader.read(_pin(context.rows, part["sourceCaptionBinding"]))
        evidence.same(binding.get("consumer"), request)
        require(
            evidence.mapping(binding.get("consumer")).get("kind") == "native-request",
            "Wrong production consumer binding kind",
        )
        original = dependencies.get(movie["path"], {})
        clean = context.reference("clean")
        binding_ref = evidence.reference(part["sourceCaptionBinding"])
        require(
            all(
                original.get(ref["path"]) == ref["sha256"] for ref in (request, clean, binding_ref)
            ),
            "Original compositor does not bind actual clean/request/production inputs",
        )
        selections = _list(part["selections"], MAX_SELECTIONS, "source selections", nonempty=True)
        recipe = evidence.mapping(reader.read(request).get("recipe"))
        clips = evidence.rows(recipe.get("clips"))
        require(len(clips) == len(selections), "Actual native request selection count differs")
        require(
            evidence.mapping(recipe.get("video")).get("masks") in (None, []),
            "Fixed blur forbidden",
        )
        for clip, selection in zip(clips, map(evidence.mapping, selections), strict=True):
            require(
                clip.get("sourcePath") == evidence.reference(selection["media"])["path"]
                and clip.get("startSeconds") == float(_fraction(selection["start"]))
                and clip.get("durationSeconds") == float(_fraction(selection["nativeDuration"])),
                "Actual seconds-only request differs from declared selection",
            )
        for raw_selection in selections:
            _source_selection(evidence.mapping(raw_selection), context, reader)
        require(
            sum(_positive(evidence.mapping(value).get("frames")) for value in selections) == frames,
            "Native source selection extent differs",
        )
        total += frames
    joined_movie = _pin(context.rows, item["joinedMovie"])
    require(
        outputs.get(joined_movie["path"]) == joined_movie["sha256"]
        and total == context.count("frames"),
        "Join output/whole frame clock differs",
    )
    return {
        "nativePartsAndIntendedClockBound": True,
        "exactSelectedSourceContentVerified": False,
        "pending": "Seconds-only native selection can intersect, hold or repeat source frames; "
        "an exact selected sample-content producer proof is required",
    }


def _standard(
    index: object, context: Context, reader: evidence.MetadataReader, *, root: Path
) -> Path:
    ref = _pin(context.rows, index)
    data = reader.read(ref)
    required = {
        "version",
        "backend",
        "recording",
        "configurationSHA256",
        "sourceSHA256",
        "finalMovieSHA256",
        *standard.REFERENCES,
    }
    require(
        set(data) == required
        and type(data["version"]) is int
        and data["version"] == 2
        and data["backend"] == STANDARD_BACKEND,
        "Not a genuine existing Motion/Compressor v2 index",
    )
    require(
        data["recording"] == root.name
        and data["configurationSHA256"] == context.rows[str(root / "edit-config.json")]["sha256"]
        and data["sourceSHA256"] == context.reference("source")["sha256"]
        and data["finalMovieSHA256"] == context.reference("movie")["sha256"],
        "Standard index source/revision mismatch",
    )
    evidence.same(data["renderConfig"], context.reference("renderConfig"))
    for name in standard.REFERENCES:
        receipt = reader.read(_pin(context.rows, data[name]))
        if name == "renderConfig":
            require(
                receipt.get("cuts") == str(context.path("cuts"))
                and evidence.mapping(receipt.get("audio")).get("source")
                == str(context.path("voice"))
                and all(
                    receipt.get(key) == context.count(field)
                    for key, field in (("rate", "fps"), ("width", "width"), ("height", "height"))
                ),
                "Standard render source-clock/voice/canvas differs from mechanical context",
            )
            continue
        for row in evidence.rows(receipt.get("inputs")):
            _pin(context.rows, {"path": row.get("path"), "sha256": row.get("sha256")})
        require(
            receipt.get("inputsAfter") == receipt.get("inputs"),
            "Standard receipt requires unchanged original terminal inputs",
        )
        if name == "technical":
            expected = evidence.mapping(receipt.get("expected"))
            require(
                all(
                    expected.get(key) == context.count(field)
                    for key, field in (
                        ("frames", "frames"),
                        ("rate", "fps"),
                        ("width", "width"),
                        ("height", "height"),
                    )
                ),
                "Final technical frame clock/canvas differs from composed selection",
            )
        if name == "captionAudit":
            parts = evidence.rows(evidence.mapping(receipt.get("measurements")).get("parts"))
            require(bool(parts), "No actual painted alpha parts")
            for part in parts:
                require(
                    part.get("motionSourcePlanStyleVerified") is True,
                    "No recorded selected-style Motion source comparison",
                )
                state = reader.read(_pin(context.rows, part.get("producerState")))
                require(
                    state.get("status") == "Successful"
                    and state.get("jobID") == part.get("jobID")
                    and state.get("submissionVersion") == 2
                    and state.get("terminalInputsVerified") is True
                    and state.get("terminalRuntimeVerified") is True,
                    "No original successful Motion alpha producer/runtime",
                )
                pins = evidence.rows(state.get("inputs"))
                require(
                    any(Path(evidence.string(pin.get("path"))).suffix == ".moti" for pin in pins),
                    "No actual Motion source input",
                )
    return Path(ref["path"])


def _references(
    data: dict[str, object], context: Context, reader: evidence.MetadataReader
) -> dict[str, Path]:
    if "standardIndex" in data:
        document = reader.read(_pin(context.rows, data["standardIndex"]))
        require(
            type(document.get("version")) is int
            and document.get("version") == 2
            and document.get("backend") == STANDARD_BACKEND
            and document.get("recording") == data["recording"],
            "Cannot assemble a v2 shell by relabeling a different backend",
        )
        values = {name: document[name] for name in standard.REFERENCES if name in document}
    else:
        values = evidence.mapping(data.get("standardReferences"))
        require(
            set(values) <= set(standard.REFERENCES) and "renderConfig" in values,
            "Missing/unknown standard references",
        )
    evidence.same(values["renderConfig"], context.reference("renderConfig"))
    return {name: Path(_pin(context.rows, value)["path"]) for name, value in values.items()}


def _nested(value: object, context: Context) -> None:
    if isinstance(value, dict):
        item = evidence.mapping(value)
        if "path" in item and "sha256" in item:
            _pin(context.rows, {key: item[key] for key in ("path", "sha256")})
        for child in item.values():
            _nested(child, context)
    elif isinstance(value, list):
        for child in value:
            _nested(child, context)
    elif isinstance(value, str) and value.startswith("/"):
        path = Path(value)
        if path.is_file():
            require(value in context.rows, "Nested reader input absent from declared closure")


def _preflight(context: Context, reader: evidence.MetadataReader) -> None:
    # Standard readers have inferred local artifacts as well as explicit refs.
    # Scan only the finite declared JSON files, never directories or media.
    for name, row in context.rows.items():
        path = Path(name)
        if path.suffix != ".json":
            continue
        size = evidence.integer(row["bytes"])
        require(size <= evidence.MAX_RECORD_BYTES, "Declared JSON record byte budget exceeded")
        reader.bytes_read += size
        require(reader.bytes_read <= evidence.MAX_TOTAL_BYTES, "Metadata byte budget exceeded")
        require(file_digest(path) == row["sha256"], "Declared JSON content changed")
        _nested(json.loads(path.read_text(), object_pairs_hook=evidence.unique_object), context)


def _sync(
    raw: dict[str, object], context: Context, reader: evidence.MetadataReader, *, verify: bool
) -> bool:
    item = _exact(raw, frozenset({"request", "result"}), "sync")
    request, result = (_pin(context.rows, item[key]) for key in ("request", "result"))
    document = reader.read(result)
    evidence.same(document.get("request"), request)
    report = evidence.mapping(document.get("assessment"))
    evidence.same(report.get("candidate"), context.reference("movie"))
    reader.read(request)
    for suffix in (".decoded.json", ".packets.json"):
        path = result["path"] + suffix
        require(path in context.rows, "Missing original derived AAC input")
    if not verify:
        return False
    aac.verify_presentation(
        aac.Reference(Path(request["path"]), request["sha256"]), Path(result["path"])
    )
    return report.get("presentationSyncAccepted") is True


def _closure(reader: evidence.MetadataReader, context: Context, index: object) -> dict[Path, str]:
    pins = {Path(path): evidence.digest(row["sha256"]) for path, row in context.rows.items()}
    original = evidence.reference(index)
    pins[Path(original["path"])] = original["sha256"]
    for path, (digest, _) in reader.cache.items():
        require(
            Path(path) in pins and pins[Path(path)] == digest,
            "Metadata dependency missing from declared current closure",
        )
    require(len(pins) <= MAX_INPUTS, "Complete scope input budget exceeded")
    return pins


@dataclass(frozen=True)
class _Current:
    snapshot: InputHashSnapshot
    empty: dict[Path, tuple[int, ...]]

    def finish(self) -> None:
        self.snapshot.finish()
        for path, identity in self.empty.items():
            require(
                _empty_identity(path) == identity,
                "Original empty process stream changed",
            )


def _empty_identity(path: Path) -> tuple[int, ...]:
    require(path.resolve(strict=True) == path, "Empty process stream is linked")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size == 0 and stream.read(1) == b"",
            "Original empty process stream changed",
        )
        identity = evidence.file_identity(before)
        require(
            identity
            == evidence.file_identity(os.fstat(stream.fileno()))
            == evidence.file_identity(path.lstat()),
            "Original empty process stream changed during read",
        )
    return identity


def _snapshot(pins: Mapping[Path, str], budget: int) -> _Current:
    require(
        type(budget) is int and budget > 0,
        "Fresh content verification requires an explicit byte allowance",
    )
    total = sum(path.stat().st_size for path in pins)
    require(total <= budget, "Declared content exceeds admitted byte allowance")
    empty = {path: _empty_identity(path) for path in pins if path.stat().st_size == 0}
    require(
        all(
            path.suffix in {".stdout", ".stderr"}
            and path.resolve(strict=True) == path
            and pins[path] == hashlib.sha256(b"").hexdigest()
            for path in empty
        ),
        "Invalid empty process stream",
    )
    return _Current(
        InputHashSnapshot.capture(
            (path, digest) for path, digest in pins.items() if path not in empty
        ),
        empty,
    )


def assess(
    index: object,
    *,
    root: Path,
    movie: Path,
    verify_content: bool = False,
    content_byte_budget: int = 0,
) -> dict[str, object]:
    """Reconstruct real stages; strict success also requires admitted fresh content."""
    reader = evidence.MetadataReader()
    data = reader.read(index)
    context = _bind(data, root, movie)
    for row in context.rows.values():
        _identity(row)
    reader.read(data["configuration"])
    roles, outputs, dependencies = _productions(data, context, reader)
    failures: dict[str, list[str]] = {stage: [] for stage in REQUIRED_STAGES}
    for role in REQUIRED_ROLES - roles:
        affected = (role,) if role in REQUIRED_STAGES else REQUIRED_STAGES
        for stage in affected:
            failures[stage].append(f"Missing original {role} production checkpoint")
    validators = {
        "cleanup": ("source_caption_mask",),
        "asr": ("transcription",),
        "separation": ("bgm_removal",),
        "composition": ("silence_cut", "dynamic_captions"),
    }
    domain_results: dict[str, object] = {}
    for name, stages in validators.items():
        try:
            raw = _domain(data, name, context, reader)
            if name == "cleanup":
                evidence.same(raw.get("source"), context.reference("source"))
                domain_results[name] = evidence.cleanup_evidence(raw, reader=reader)
            elif name == "asr":
                evidence.same(raw.get("cuts"), context.reference("cuts"))
                domain_results[name] = evidence.asr_evidence(raw, reader=reader)
            elif name == "separation":
                _separation(raw, context, outputs, dependencies)
                domain_results[name] = {"originalIndependentSeparationBound": True}
            else:
                composition = _composition(raw, context, reader, outputs, dependencies)
                domain_results[name] = composition
                failures["silence_cut"].append(evidence.string(composition["pending"]))
        except ERRORS as error:
            domain_results[name] = {"bound": False, "reason": str(error)}
            for stage in stages:
                failures[stage].append(f"{name}: {error}")
    standard_index: Path | None = None
    try:
        require("standardIndex" in data, "Genuine standard v2 index not supplied")
        standard_index = _standard(data["standardIndex"], context, reader, root=root)
    except ERRORS as error:
        for stage in REQUIRED_STAGES:
            failures[stage].append(f"standard contract: {error}")
    if verify_content:
        _preflight(context, reader)
        for path in code_inputs():
            require(str(path) in context.rows, "Named current reader code input missing")
    pins = _closure(reader, context, index)
    snapshot = _snapshot(pins, content_byte_budget) if verify_content else None
    standard_complete = False
    sync_complete = False
    if snapshot is not None and standard_index is not None:
        try:
            standard_failures = standard.assess(
                standard_index,
                delivery=movie,
                root=root,
                source=context.config.source,
                storage=context.config.storage_root,
            )
            standard_complete = not standard_failures
            for stage, message in standard_failures.items():
                require(stage in failures, "Unknown standard failure stage")
                failures[stage].append(f"standard strict: {message}")
        except ERRORS as error:
            for stage in REQUIRED_STAGES:
                failures[stage].append(f"standard provenance: {error}")
        try:
            cleanup = source_caption_binding.require_cleanup(context.path("renderConfig"))
            require(
                cleanup.clean == context.path("clean")
                and cleanup.original == context.path("source"),
                "Actual production cleanup source differs from mechanical context",
            )
        except ERRORS as error:
            failures["source_caption_mask"].append(f"actual production binding: {error}")
    try:
        sync_complete = _sync(
            _domain(data, "sync", context, reader), context, reader, verify=verify_content
        )
        require(sync_complete, "Original AAC presentation/content domain is incomplete")
    except ERRORS as error:
        for stage in REQUIRED_STAGES:
            failures[stage].append(f"sync: {error}")
    _closure(reader, context, index)
    current_crypto = snapshot is not None
    if snapshot is not None:
        snapshot.finish()
    else:
        for stage in REQUIRED_STAGES:
            failures[stage].append("Fresh current content verification not performed")
    strict = standard_complete and sync_complete and current_crypto and not any(failures.values())
    return {
        "version": VERSION,
        "backend": BACKEND,
        "standardBackend": STANDARD_BACKEND,
        "recording": root.name,
        "recordedStages": {
            stage: {
                "executionVerified": stage in roles,
                "strictVerified": current_crypto and sync_complete and not failures[stage],
            }
            for stage in REQUIRED_STAGES
        },
        "domains": domain_results,
        "currentCryptoVerified": current_crypto,
        "standardContractComplete": standard_complete,
        "presentationSyncVerified": sync_complete,
        "strictComplete": strict,
        "strictFailures": {stage: messages for stage, messages in failures.items() if messages},
        "publicationPerformed": False,
        "qualityApproved": False,
        "deliveryApproved": False,
        "metadataBytesRead": reader.bytes_read,
        "limitations": list(LIMITATIONS),
    }


def _write(output: Path, value: object, *, root: Path) -> Path:
    require(
        output.is_absolute() and output.resolve() == output, "Output path is linked or relative"
    )
    require(
        output.is_relative_to(root.parents[1] / "edits/work") or output.is_relative_to(root),
        "Output must stay in checkout work/recording",
    )
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return output


def prepare(request: object, *, root: Path, movie: Path, output: Path) -> Path:
    """Write a distinct validated mechanical index, including truthful pending data."""
    data = evidence.MetadataReader().read(request)
    _bind(data, root, movie)
    return _write(output, data, root=root)


def prepare_standard(
    index: object, *, root: Path, movie: Path, output: Path, content_byte_budget: int
) -> Path:
    """Use the existing public v2 assembler; missing proofs remain incomplete shells.

    This operation does not relabel a mechanical receipt or approve its body.
    Supplied receipts retain their original pins; the old index is retained.
    """
    reader = evidence.MetadataReader()
    data = reader.read(index)
    context = _bind(data, root, movie)
    references = _references(data, context, reader)
    for reference in references.values():
        reader.read({"path": str(reference), "sha256": context.rows[str(reference)]["sha256"]})
    _preflight(context, reader)
    snapshot = _snapshot(_closure(reader, context, index), content_byte_budget)
    require(not output.exists() and not output.is_symlink(), "Refusing standard index overwrite")
    result = standard.assemble(
        root,
        movie,
        output,
        references,
    )
    snapshot.finish()
    return result


def observe(
    index: object, *, root: Path, movie: Path, output: Path, content_byte_budget: int
) -> Path:
    """Retain one fresh assessment; it cannot grant crypto to a future check."""
    result = assess(
        index, root=root, movie=movie, verify_content=True, content_byte_budget=content_byte_budget
    )
    return _write(
        output,
        {
            "kind": "mechanical-current-input-observation",
            "index": evidence.reference(index),
            "assessment": result,
        },
        root=root,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("prepare", "observe", "assess", "check", "prepare-standard")
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("movie", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--content-byte-budget", type=int, default=0)
    args = parser.parse_args(argv)
    reference = {"path": str(args.index.absolute()), "sha256": args.sha256}
    try:
        options = {"root": args.root.absolute(), "movie": args.movie.absolute()}
        if args.operation in {"prepare", "observe", "prepare-standard"}:
            require(isinstance(args.output, Path), "Operation requires new output")
            if args.operation == "prepare":
                output = prepare(reference, output=args.output.absolute(), **options)
            elif args.operation == "observe":
                output = observe(
                    reference,
                    output=args.output.absolute(),
                    content_byte_budget=args.content_byte_budget,
                    **options,
                )
            else:
                output = prepare_standard(
                    reference,
                    output=args.output.absolute(),
                    content_byte_budget=args.content_byte_budget,
                    **options,
                )
            print(output)
            return 0
        result = assess(
            reference,
            verify_content=args.operation == "check",
            content_byte_budget=args.content_byte_budget,
            **options,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return int(args.operation == "check" and result["strictComplete"] is not True)
    except ERRORS as error:
        print(json.dumps({"strictComplete": False, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
