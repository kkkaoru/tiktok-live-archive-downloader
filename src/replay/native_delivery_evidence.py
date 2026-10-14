"""Read original native delivery evidence without running or approving media.

This optional assessment is not connected to edit_completion. Original successful
execution, metadata integrity, current media crypto and quality are distinct.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
from collections.abc import Callable
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from replay.caption_io import mlx_groups
from replay.caption_projection import CutSpan, project_tokens
from replay.caption_timing import SpokenToken
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_plan import repair_spans

DOMAINS = (
    "cleanup",
    "voice",
    "cuts",
    "asr",
    "caption-layout",
    "native-parts",
    "whole-assembly",
    "final-delivery",
)
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_CAPTION_LAYOUT_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 192 * 1024 * 1024
MAX_DOMAIN_BYTES = 128 * 1024 * 1024
MAX_RECORDS = 4096
ORIGINAL_TOKEN_EPSILON = 1e-7


def fingerprint_rows(value: object) -> dict[str, dict[str, object]]:
    """Validate original seven-field identity rows without opening their files."""
    result: dict[str, dict[str, object]] = {}
    required = {"path", "bytes", "sha256", "device", "inode", "mtimeNs", "ctimeNs"}
    items = rows(value)
    if len(items) > 1024:
        raise ValueError("Fingerprint inventory budget exceeded")
    for row in items:
        if set(row) != required:
            raise ValueError("Unsupported fingerprint row schema")
        path = string(row["path"])
        if (
            not path.startswith("/")
            or path.startswith("//")
            or str(Path(path)) != path
            or ".." in Path(path).parts
        ):
            raise ValueError("Fingerprint path must be lexically canonical")
        if path in result:
            raise ValueError("Duplicate fingerprint path")
        digest(row["sha256"])
        for key in ("bytes", "device", "inode", "mtimeNs", "ctimeNs"):
            if integer(row[key]) <= 0:
                raise ValueError("Fingerprint identity fields must be positive")
        result[path] = row
    return result


def require_unchanged_fingerprints(before: object, after: object) -> None:
    if fingerprint_rows(before) != fingerprint_rows(after):
        raise ValueError("Final verifier input closure changed")


def fingerprint_hashes(value: object) -> dict[str, str]:
    return {path: digest(row["sha256"]) for path, row in fingerprint_rows(value).items()}


def checkpoint_evidence(config: dict[str, object], reader: "MetadataReader") -> dict[str, object]:
    """Reconstruct the final verification checkpoint, including its whole-phase inputs.

    This is not a parser for individual join/assembly producer checkpoints.
    Their records are bound as inputs, without asserting their fingerprints.
    """
    graph = reader.read(config.get("workflow"))
    matches = [row for row in rows(graph.get("stages")) if row.get("name") == config.get("stage")]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous original workflow stage")
    stage = matches[0]
    command = strings(stage.get("command"))
    if len(command) < 6 or command[1:4] != ["-B", "-m", "replay.delivery_verification"]:
        raise ValueError("Expected original typed final-delivery verifier stage")
    inventory = hashes(reader.read(config.get("inventory")).get("inputs"))
    declared = strings(stage.get("inputs"))
    if any(path not in inventory for path in declared):
        raise ValueError("Original stage inventory is incomplete")
    return stage_provenance(
        stage=stage,
        checkpoint=reader.read(config.get("checkpoint")),
        input_hashes={path: inventory[path] for path in declared},
        cwd=string(config.get("cwd")),
        expected_outputs=hashes(config.get("outputs")),
    )


def domain_evidence(
    name: str, config: dict[str, object], reader: "MetadataReader"
) -> dict[str, object]:
    if name == "final-delivery" and config.get("contract") == "normalized-combined-v3":
        return normalized_combined_evidence(config, reader)
    if name == "whole-assembly" and config.get("contract") == "original-split-video-whole-v1":
        return split_video_whole_evidence(config, reader)
    if name == "cleanup":
        return cleanup_evidence(config, reader=reader)
    if name == "asr":
        return asr_evidence(config, reader=reader)
    if name == "cuts":
        if config.get("contract") == "prospective-protected-grid-v1":
            return protected_grid_evidence(config, reader)
        if config.get("contract") == "original-protected-grid-with-source-evidence-v1":
            return parent_probed_cut_evidence(config, reader)
        if config.get("contract") == "original-native-grid-v2":
            return original_native_grid_evidence(config, reader)
        if "contract" in config:
            raise ValueError("Unsupported original cut contract")
        return cut_domain(
            reader.read(config.get("receipt")),
            integer(config.get("frames")),
            integer(config.get("fps")),
        )
    if name == "voice":
        if config.get("contract") == "mixed-tempo-candidate":
            return mixed_voice_evidence(config, reader)
        return voice_domain(
            reader.read(config.get("receipt")),
            expected_inputs=hashes(config.get("expectedInputs")),
            output=mapping(config.get("output")),
            frames=integer(config.get("frames")),
            fps=integer(config.get("fps")),
        )
    if name == "caption-layout":
        result = head_lineage(
            reader.read(config.get("headReceipt"), caption_layout=True),
            reader.read(config.get("headConfig")),
        )
        required = hashes(config.get("refreshInputs"))
        head_ref = reference(config.get("headReceipt"))
        if required.get(head_ref["path"]) != head_ref["sha256"]:
            raise ValueError("Refresh must bind original head receipt")
        result["refresh"] = caption_lineage(
            reader.read(config.get("refreshReceipt"), caption_layout=True),
            consumer=mapping(config.get("output")),
            required_inputs=required,
        )
        return result
    if name == "native-parts":
        if config.get("contract") in {"original-assembly-probe-v1", "original-native-batch-v1"}:
            return original_native_parts(mapping(config.get("frames")), reader)
        if "contract" in config:
            raise ValueError("Unsupported native part contract")
        result = native_parts(mapping(config.get("frames")), reader=reader)
        publications = rows(mapping(config.get("frames")).get("publications"))
        completions = [published_completion(ref, reader) for ref in publications]
        if [item["part"] for item in completions] != list(range(1, len(publications) + 1)):
            raise ValueError("Native completion part inventory differs")
        result["nativeCompletions"] = completions
        result["pending"] = [
            "original executor checkpoints",
            "current media crypto",
            "caption visual quality",
        ]
        if "retirements" not in config:
            result["retirementDisposition"] = "not supplied; no current materialization inferred"
            return result
        retired = rows(config.get("retirements"))
        if len(retired) != len(publications):
            raise ValueError("Retirement/publication coverage differs")
        records: list[dict[str, object]] = []
        for number, (retirement_ref, publication_ref) in enumerate(
            zip(retired, publications, strict=True), 1
        ):
            retirement = reader.read(reader.read(retirement_ref).get("retirementReceipt"))
            same(retirement.get("publication"), publication_ref)
            item = publication_evidence(retirement_ref, reader=reader)
            if item["part"] != number:
                raise ValueError("Retirement part coverage differs")
            records.append(item)
        result["publicationRetirement"] = records
        return result
    if name in {"whole-assembly", "final-delivery"}:
        result = recorded({"scope": "original completed native workflow checkpoint"})
        result["originalStage"] = checkpoint_evidence(mapping(config.get("stageEvidence")), reader)
        if name == "whole-assembly":
            if config.get("contract") == "original-technical-full-av-reuse-v1":
                result["wholeRecords"] = original_technical_whole(config, reader)
                return result
            if config.get("contract") == "original-whole-observations-v1":
                result["wholeRecords"] = original_whole_observations(config, reader)
                return result
            if config.get("contract") == "original-whole-phases-v1":
                result["wholeRecords"] = original_whole_phases(config, reader)
                return result
            if "contract" in config:
                raise ValueError("Unsupported whole phase contract")
            inventory = hashes(
                reader.read(mapping(config.get("stageEvidence")).get("inventory")).get("inputs")
            )
            result["wholeRecords"] = whole_records(
                config.get("wholeRecords"), inventory, mapping(config.get("source")), reader=reader
            )
            if mapping(result["wholeRecords"]).get("completionRecordsBound") is not True:
                raise ValueError("Original whole phases must be bound")
        if name == "final-delivery":
            receipt_ref = reference(config.get("receipt"))
            outputs = hashes(mapping(config.get("stageEvidence")).get("outputs"))
            if outputs.get(receipt_ref["path"]) != receipt_ref["sha256"]:
                raise ValueError("Final receipt is not an original stage output")
            receipt = reader.read(receipt_ref)
            if (
                receipt.get("version") != 1
                or receipt.get("fullDecode") is not True
                or receipt.get("technicalAccepted") is not True
            ):
                raise ValueError("Original final decode was incomplete")
            require_unchanged_fingerprints(receipt.get("inputs"), receipt.get("inputsAfter"))
            expected = mapping(receipt.get("expected"))
            if integer(expected.get("frames")) != integer(config.get("frames")) or integer(
                expected.get("rate")
            ) != integer(config.get("fps")):
                raise ValueError("Final edited frame clock differs")
            result["originalApprovals"] = mapping(receipt.get("approvals"))
            if config.get("successContract") == "original-executor-request-v1":
                result["compressor"] = original_executor_request(config, reader)
            elif config.get("successContract") == "native-status-source-master-v1":
                result["compressor"] = native_success_execution(config, reader)
            elif config.get("successContract") == "explicit-native-success-observation-v1":
                result["compressor"] = explicit_native_success(config, reader)
            elif config.get("successContract") == "original-native-success-candidate-v1":
                result["compressor"] = original_success_candidate(config, reader)
            else:
                result["compressor"] = compressor_execution(
                    submit=reader.read(config.get("submit"), native_submit=True),
                    status=reader.read(config.get("status")),
                    request=reader.read(config.get("request")),
                    success=reader.read(config.get("success")),
                    output_path=reference(config.get("output"))["path"],
                )
            result["submissionDiagnostics"] = reader.submit_diagnostics.get(
                reference(config.get("submit"))["path"], []
            )
        return result
    raise ValueError("Unsupported native domain")


def published_completion(value: object, reader: "MetadataReader") -> dict[str, object]:
    publication = reader.read(value)
    if publication.get("version") != 1 or publication.get("kind") != "published-native-part":
        raise ValueError("Unsupported native publication")
    completion = reader.read(publication.get("completion"))
    intent = reader.read(publication.get("intent"))
    if completion.get("version") != 1 or completion.get("kind") != "verified-native-part":
        raise ValueError("Unsupported native completion")
    if intent.get("kind") != "native-part-publication-intent":
        raise ValueError("Unsupported publication intent")
    same(completion.get("movie"), publication.get("source"))
    same(intent.get("source"), publication.get("source"))
    same(intent.get("completion"), publication.get("completion"))
    movie, source = reference(publication.get("movie")), reference(publication.get("source"))
    if movie["sha256"] != source["sha256"] or intent.get("destination") != movie["path"]:
        raise ValueError("Publication byte identity differs")
    if (
        completion.get("completionVerified") is not True
        or completion.get("shareObservedIdle") is not True
        or integer(completion.get("fullAVDecodeExitCode")) != 0
    ):
        raise ValueError("Native completion is partial")
    part = integer(completion.get("part"))
    if part <= 0:
        raise ValueError("Invalid native part")
    hashes(completion.get("inputs"))
    return {
        "part": part,
        "nativeDispatchVerified": completion.get("nativeDispatchVerified") is True,
        "recordedFullAVDecodeVerified": True,
        "currentCryptoVerified": False,
        "qualityApproved": False,
    }


def assess(
    config: dict[str, object], *, reader: "MetadataReader | None" = None
) -> dict[str, object]:
    """Assess only original typed record contracts; missing domains remain pending."""
    if (
        integer(config.get("version")) != 1
        or config.get("backend") != "native-fcp-partitioned-whole"
    ):
        raise ValueError("Unsupported native evidence descriptor")
    domains = mapping(config.get("domains"))
    if set(domains) - set(DOMAINS):
        raise ValueError("Unknown native domain")
    active = reader if reader is not None else MetadataReader()
    context = mapping(config.get("context"))
    for key in ("source", "cuts", "voice", "semanticXML", "wholeMaster", "finalMovie"):
        reference(context.get(key))
    if integer(context.get("frames")) <= 0 or integer(context.get("fps")) <= 0:
        raise ValueError("Expected exact positive edited clock")
    results: dict[str, object] = {}
    for name in DOMAINS:
        if name not in domains:
            results[name] = {
                "stageApplied": False,
                "recordedEvidenceBound": False,
                "pending": ["original domain evidence not supplied"],
            }
            continue
        entry = mapping(domains[name])
        if name == "whole-assembly" and entry.get("contract") == "original-split-video-whole-v1":
            if "native-parts" in domains:
                native = mapping(mapping(domains["native-parts"]).get("frames"))
                same(entry.get("partition"), native.get("partition"))
                if rows(entry.get("publications")) != [
                    mapping(item.get("publication")) for item in rows(native.get("parts"))
                ]:
                    raise ValueError("Split whole uses another native publication inventory")
            if "final-delivery" in domains:
                final = mapping(domains["final-delivery"])
                if final.get("contract") != "normalized-combined-v3":
                    raise ValueError("Split whole requires its original normalized final contract")
                same(
                    entry.get("audioEvidence"),
                    mapping(final.get("audioCompressor")).get("outputEvidence"),
                )
        if name == "cuts" and entry.get("contract") in {
            "original-protected-grid-with-source-evidence-v1",
            "original-native-grid-v2",
        }:
            same(
                entry.get("sourceClockEvidence"),
                mapping(domains.get("cleanup")).get("assemblyReceipt"),
            )
        active.begin_domain(name)
        bind_context(name, entry, context, active)
        result = domain_evidence(name, entry, active)
        result["recordedEvidenceBound"] = True
        result["currentCryptoVerified"] = False
        result["qualityApproved"] = False
        result["syncApproved"] = False
        results[name] = result
    return {
        "version": 1,
        "backend": config["backend"],
        "domains": results,
        "candidateAvailable": None,
        "currentCryptoVerified": False,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
        "finished": False,
        "metadataBytesRead": active.bytes_read,
        "metadataRecordsRead": len(active.cache),
        "domainMetadata": active.domain_metadata,
        "scope": "recorded native stages, not formal completion or current media verification",
    }


def bind_context(
    name: str, entry: dict[str, object], context: dict[str, object], reader: "MetadataReader"
) -> None:
    """Reject mixing individually valid evidence from different recordings or revisions."""
    if name == "cleanup":
        same(entry.get("approvedOriginal", entry.get("source")), context.get("source"))
    elif name == "asr":
        same(entry.get("cuts"), context.get("cuts"))
    elif name in {"voice", "cuts", "final-delivery"}:
        if name == "cuts" and entry.get("contract") == "original-native-grid-v2":
            same(entry.get("source"), context.get("source"))
        for key in ("frames", "fps"):
            if integer(entry.get(key)) != integer(context.get(key)):
                raise ValueError("Domain edited clock differs from descriptor")
        key = {"voice": "voice", "cuts": "cuts", "final-delivery": "finalMovie"}[name]
        same(entry.get("receipt" if name == "cuts" else "output"), context.get(key))
        if name == "final-delivery":
            if entry.get("contract") == "normalized-combined-v3":
                video = mapping(entry.get("videoCompressor"))
                if video.get("contract") == "original-split-native-request-v1":
                    same(video.get("source"), context.get("wholeMaster"))
                else:
                    same(
                        reader.read(video.get("success")).get("source"), context.get("wholeMaster")
                    )
                audio = mapping(reader.read(entry.get("observerConfig")).get("audio"))
                same(audio.get("pcm"), context.get("voice"))
                return
            source_key = (
                "sourceMaster"
                if entry.get("successContract") == "native-status-source-master-v1"
                else "master"
                if entry.get("successContract") == "explicit-native-success-observation-v1"
                else "source"
            )
            if entry.get("successContract") == "original-executor-request-v1":
                same(reader.read(entry.get("postConfig")).get("master"), context.get("wholeMaster"))
            elif entry.get("successContract") == "original-native-success-candidate-v1":
                same(entry.get("source"), context.get("wholeMaster"))
                if (
                    reader.read(entry.get("success")).get("wholeMasterSha256")
                    != reference(context.get("wholeMaster"))["sha256"]
                ):
                    raise ValueError("Original candidate master digest differs from context")
            else:
                same(reader.read(entry.get("success")).get(source_key), context.get("wholeMaster"))
            media = reference(context.get("finalMovie"))
            if (
                fingerprint_hashes(reader.read(entry.get("receipt")).get("inputs")).get(
                    media["path"]
                )
                != media["sha256"]
            ):
                raise ValueError("Final decode does not bind selected final movie bytes")
    elif name == "caption-layout":
        same(entry.get("output"), context.get("semanticXML"))
        same(reader.read(entry.get("headConfig")).get("cuts"), context.get("cuts"))
    elif name == "native-parts":
        frame_config = mapping(entry.get("frames"))
        same(frame_config.get("semanticXML"), context.get("semanticXML"))
        for key in ("frames", "fps"):
            if integer(frame_config.get(key)) != integer(context.get(key)):
                raise ValueError("Native partition clock differs from descriptor")
    elif name == "whole-assembly":
        same(entry.get("source"), context.get("wholeMaster"))
        if entry.get("contract") in {
            "original-split-video-whole-v1",
            "original-technical-full-av-reuse-v1",
        }:
            same(entry.get("voice"), context.get("voice"))
            same(entry.get("semanticXML"), context.get("semanticXML"))
            for key in ("frames", "fps"):
                if integer(entry.get(key)) != integer(context.get(key)):
                    raise ValueError("Split whole clock differs from descriptor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("--sha256", required=True, help="Expected descriptor SHA256")
    args = parser.parse_args()
    reader = MetadataReader()
    config = reader.read({"path": str(args.descriptor.absolute()), "sha256": args.sha256})
    print(json.dumps(assess(config, reader=reader), indent=2, sort_keys=True))


class MetadataReader:
    """Bounded JSON-only reads; references to media/model/tool files are never opened."""

    def __init__(self) -> None:
        self.bytes_read = 0
        self.cache: dict[str, tuple[str, dict[str, object]]] = {}
        self.identities: dict[str, tuple[int, ...]] = {}
        self.domain_metadata: dict[str, dict[str, int]] = {}
        self.active_domain: str | None = None
        self.domain_paths: set[str] = set()
        self.parse_modes: dict[str, str] = {}
        self.submit_diagnostics: dict[str, list[str]] = {}

    def begin_domain(self, name: str) -> None:
        if name not in DOMAINS or name in self.domain_metadata:
            raise ValueError("Unsupported or repeated metadata domain")
        self.active_domain = name
        self.domain_paths = set()
        self.domain_metadata[name] = {"bytes": 0, "records": 0}

    def _charge_domain(self, path: str, size: int) -> None:
        if self.active_domain is None or path in self.domain_paths:
            return
        totals = self.domain_metadata[self.active_domain]
        if totals["bytes"] + size > MAX_DOMAIN_BYTES:
            raise ValueError("Evidence domain byte budget exceeded")
        totals["bytes"] += size
        totals["records"] += 1
        self.domain_paths.add(path)

    def read(
        self,
        value: object,
        *,
        native_submit: bool = False,
        native_status: bool = False,
        caption_layout: bool = False,
        join_plan: bool = False,
    ) -> dict[str, object]:
        ref = reference(value)
        path = Path(ref["path"])
        if (
            (join_plan and (native_submit or native_status or caption_layout))
            or (caption_layout and (native_submit or native_status))
            or (native_submit and native_status)
        ):
            raise ValueError("Caption layout cannot use native submission parsing")
        parse_mode = (
            "join-plan"
            if join_plan
            else "submit"
            if native_submit
            else "status"
            if native_status
            else "json"
        )
        limit = MAX_CAPTION_LAYOUT_BYTES if caption_layout else MAX_RECORD_BYTES
        if path.suffix not in {".json", ".log"} or path.resolve(strict=True) != path:
            raise ValueError("Evidence must be canonical JSON, not media or a symlink")
        if path.suffix == ".log" and not (native_submit or native_status):
            raise ValueError("Log metadata requires typed native submission context")
        cached = self.cache.get(str(path))
        if cached is not None:
            if self.identities[str(path)][2] > limit:
                raise ValueError("Evidence record size exceeds caller context")
            if self.parse_modes[str(path)] != parse_mode:
                raise ValueError("Conflicting evidence parser context")
            if cached[0] != ref["sha256"]:
                raise ValueError("Conflicting evidence SHA")
            if file_identity(path.stat()) != self.identities[str(path)]:
                raise ValueError("Evidence identity changed after initial read")
            self._charge_domain(str(path), self.identities[str(path)][2])
            return cached[1]
        if len(self.cache) >= MAX_RECORDS:
            raise ValueError("Evidence record budget exceeded")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
                raise ValueError("Evidence record size/type is unsupported")
            if self.bytes_read + before.st_size > MAX_TOTAL_BYTES:
                raise ValueError("Evidence total byte budget exceeded")
            self._charge_domain(str(path), before.st_size)
            data = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        if (
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or file_identity(path.stat()) != file_identity(after)
            or path.resolve(strict=True) != path
        ):
            raise ValueError("Evidence identity changed during read")
        self.bytes_read += len(data)
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("Evidence SHA differs")
        if native_submit or native_status:
            result, diagnostics = submit_log_envelope(data)
            self.submit_diagnostics[str(path)] = diagnostics
        else:
            decoded = json.loads(data, object_pairs_hook=unique_object)
            result = {"rows": rows(decoded)} if join_plan else mapping(decoded)
        self.parse_modes[str(path)] = parse_mode
        self.cache[str(path)] = (ref["sha256"], result)
        self.identities[str(path)] = file_identity(after)
        return result


def submit_log_envelope(data: bytes) -> tuple[dict[str, object], list[str]]:
    """Parse one original Executor submission, retaining bounded standard diagnostics."""
    warning = "Settings sync failed; local Executor remains available. Run executor-sync status."
    text = data.decode("utf-8")
    diagnostics: list[str] = []
    if text.startswith(warning + "\n"):
        text = text[len(warning) + 1 :]
        diagnostics.append(warning)
    value, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(text.lstrip())
    tail = text.lstrip()[end:].strip()
    if tail == warning:
        diagnostics.append(warning)
    elif tail:
        raise ValueError("Unexpected trailing native submission log content")
    result = mapping(value)
    official_payload(result)
    return result, diagnostics


def file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Expected string-keyed object")
    return {key: item for key, item in value.items()}


def string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected nonempty string")
    return value


def integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Expected nonnegative integer")
    return value


def digest(value: object) -> str:
    text = string(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError("Invalid SHA256")
    return text


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def strings(value: object) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 1024:
        raise ValueError("Expected bounded nonempty string list")
    return [string(item) for item in value]


def hashes(value: object) -> dict[str, str]:
    result = {key: digest(item) for key, item in mapping(value).items()}
    if not result or len(result) > 1024 or any(not Path(key).is_absolute() for key in result):
        raise ValueError("Invalid retained input hashes")
    return result


def stage_provenance(
    *,
    stage: dict[str, object],
    checkpoint: dict[str, object],
    input_hashes: dict[str, str],
    cwd: str,
    expected_outputs: dict[str, str],
) -> dict[str, object]:
    """Reconstruct stages._fingerprint from retained hashes, not current file reads."""
    inputs = strings(stage.get("inputs"))
    outputs = strings(stage.get("outputs"))
    if len(set(inputs)) != len(inputs) or set(inputs) != set(input_hashes):
        raise ValueError("Incomplete or duplicate original input closure")
    if set(outputs) != set(expected_outputs) or len(set(outputs)) != len(outputs):
        raise ValueError("Output closure differs")
    if not Path(cwd).is_absolute():
        raise ValueError("Expected explicit historical working directory")
    verifier = stage.get("verify_command", [])
    verify_command = [] if isinstance(verifier, list) and not verifier else strings(verifier)
    definition = {
        "command": strings(stage.get("command")),
        "verifyCommand": verify_command,
        "inputs": hashes(input_hashes),
        "outputs": outputs,
        "timeoutSeconds": integer(stage.get("timeout_seconds")),
        "workingDirectory": cwd,
    }
    fingerprint = hashlib.sha256(json.dumps(definition, sort_keys=True).encode("utf-8")).hexdigest()
    if checkpoint.get("status") != "complete" or checkpoint.get("fingerprint") != fingerprint:
        raise ValueError("Original stage was incomplete or its fingerprint differs")
    if hashes(checkpoint.get("outputs")) != hashes(expected_outputs):
        raise ValueError("Original completed output SHA differs")
    return {
        "historicalExecutionVerified": True,
        "currentCryptoVerified": False,
        "fingerprint": fingerprint,
        "originalInputCount": len(inputs),
    }


def rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value or len(value) > 100000:
        raise ValueError("Expected bounded nonempty rows")
    return [mapping(row) for row in value]


def recorded(details: dict[str, object]) -> dict[str, object]:
    return {
        "stageApplied": True,
        "executionRecorded": True,
        "executionVerified": False,
        "currentCryptoVerified": False,
        "qualityApproved": False,
        "details": details,
        "pending": ["original producer checkpoint reconstruction", "quality evidence"],
    }


def cut_domain(record: dict[str, object], expected_frames: int, fps: int) -> dict[str, object]:
    if (
        record.get("version") != 1
        or record.get("clock") != "source-presentation"
        or integer(record.get("fps")) != fps
        or (fps <= 0)
        or (expected_frames <= 0)
    ):
        raise ValueError("Wrong cut clock")
    return cut_frame_map(record, expected_frames)


def protected_grid_evidence(
    config: dict[str, object], reader: "MetadataReader"
) -> dict[str, object]:
    record = reader.read(config.get("receipt"))
    fps, frames = integer(config.get("fps")), integer(config.get("frames"))
    if (
        fps <= 0
        or frames <= 0
        or any(key in record for key in ("version", "clock", "fps"))
        or record.get("grid") != f"prospective {fps}fps; NOT yet normalized media"
        or record.get("normalizationVerified") is not False
        or not isinstance(record.get("cutApproved"), bool)
        or integer(record.get("outputFrames")) != frames
        or relative_input_hashes(record.get("inputs"))
        != relative_input_hashes(config.get("expectedInputs"))
    ):
        raise ValueError("Original prospective cut contract differs")
    start = Fraction(string(record.get("decodedSourceStart")))
    extent = Fraction(string(record.get("decodedSourceExtent")))
    count = extent * fps
    if (
        start != 0
        or extent <= 0
        or integer(record.get("sourceFrames")) != -(-count.numerator // count.denominator)
    ):
        raise ValueError("Decoded extent and prospective grid differ")
    quiet_ref = reference(config.get("quiet"))
    original_path = string(config.get("quietInput"))
    if (
        Path(original_path).is_absolute()
        or ".." in Path(original_path).parts
        or str(Path.cwd() / original_path) != quiet_ref["path"]
        or relative_input_hashes(record.get("inputs")).get(original_path) != quiet_ref["sha256"]
    ):
        raise ValueError("Original quiet input identity differs")
    quiet = reader.read(quiet_ref)
    if digest(quiet.get("sourceSHA256")) != digest(record.get("audioSHA256")):
        raise ValueError("Quiet audio and cut source differ")
    result = cut_frame_map(record, frames)
    result["originalNormalizationVerified"] = False
    result["originalCutApproved"] = record["cutApproved"]
    result["pending"] = [
        "original protected speech decisions",
        "current source crypto",
        "normalization approval",
    ]
    return result


def original_absolute_pins(value: object) -> dict[str, str]:
    """A finite original multi-window inventory, distinct from ordinary 1024-pin records."""
    result = {key: digest(item) for key, item in mapping(value).items()}
    if not 1 <= len(result) <= 4096 or any(
        not Path(key).is_absolute()
        or str(Path(key)) != key
        or ".." in Path(key).parts
        or key.startswith("//")
        for key in result
    ):
        raise ValueError("Invalid original absolute input inventory")
    return result


def original_native_grid_evidence(
    config: dict[str, object], reader: "MetadataReader"
) -> dict[str, object]:
    """Validate the original video-only terminal-grid restriction without normalizing media."""
    record = reader.read(config.get("receipt"))
    inputs = original_absolute_pins(record.get("inputs"))
    fps, frames = integer(config.get("fps")), integer(config.get("frames"))
    if (
        fps <= 0
        or frames <= 0
        or integer(record.get("outputFrames")) != frames
        or inputs != original_absolute_pins(config.get("expectedInputs"))
        or record.get("grid")
        != f"prospective conservative full-supported {fps}fps frames; NOT yet normalized media"
        or any(key in record for key in ("version", "clock", "fps"))
        or record.get("normalizationVerified") is not False
        or record.get("cutApproved") is not False
        or record.get("deliveryComplete") is not False
        or record.get("newInferencePerformed") is not False
        or record.get("originalSpeechProtectionRetained") is not True
    ):
        raise ValueError("Unsupported original native terminal grid")
    for key in ("parent", "quiet", "endpoints", "kernel"):
        ref = reference(config.get(key))
        if inputs.get(ref["path"]) != ref["sha256"]:
            raise ValueError("Original terminal-grid dependency differs")
    parent = reader.read(config.get("parent"))
    original = original_absolute_pins(parent.get("inputs"))
    if any(inputs.get(path) != checksum for path, checksum in original.items()):
        raise ValueError("Original terminal grid lost prior producer pins")
    if parent.get("grid") != f"prospective {fps}fps; NOT yet normalized media" or any(
        parent.get(key) != record.get(key)
        for key in (
            "audioSHA256",
            "nativeVideoEnd",
            "nativeAudioEnd",
            "observedVFRSourceFrames",
            "normalizationVerified",
            "cutApproved",
            "deliveryComplete",
        )
    ):
        raise ValueError("Original terminal-grid parent clock differs")
    video, voice = (
        Fraction(string(record.get("nativeVideoEnd"))),
        Fraction(string(record.get("nativeAudioEnd"))),
    )
    source_frames = integer(record.get("sourceFrames"))
    supported = Fraction(source_frames, fps)
    policy = mapping(record.get("terminalGridPolicy"))
    if (
        video <= 0
        or voice <= 0
        or source_frames != (video * fps).numerator // (video * fps).denominator
        or Fraction(string(policy.get("supportedGridEnd"))) != supported
        or Fraction(string(policy.get("omittedTerminalVideoSeconds"))) != video - supported
        or not 0 <= video - supported < Fraction(1, fps)
        or not 0 < Fraction(string(policy.get("originalAudioEnd"))) <= supported
        or not 0 <= Fraction(string(policy.get("protectedSpeechEndIncludingGuard"))) <= supported
        or voice > supported
        or policy.get("oldPlanPreserved") is not True
        or policy.get("actualAudioOrProtectedSpeechTrimmed") is not False
        or policy.get("notMeasuredSilence") is not True
        or policy.get("audioPaddingApproved") is not False
    ):
        raise ValueError("Original terminal policy trims audio or protected source time")
    old_count = max(video, voice) * fps
    if integer(parent.get("sourceFrames")) != -(-old_count.numerator // old_count.denominator):
        raise ValueError("Original parent prospective extent differs")
    delta = integer(parent.get("sourceFrames")) - source_frames
    old_cuts, new_cuts = rows(parent.get("cuts")), rows(record.get("cuts"))
    expected_tail = dict(old_cuts[-1])
    expected_tail["source_end_frame"] = source_frames
    if (
        delta not in {0, 1}
        or len(old_cuts) != len(new_cuts)
        or old_cuts[:-1] != new_cuts[:-1]
        or integer(old_cuts[-1].get("source_end_frame")) != integer(parent.get("sourceFrames"))
        or new_cuts[-1] != expected_tail
        or integer(parent.get("outputFrames")) - frames != delta
    ):
        raise ValueError("Original terminal restriction changed retained cuts")
    cut_frame_map(parent, integer(parent.get("outputFrames")))
    quiet = reader.read(config.get("quiet"))
    if (
        digest(quiet.get("sourceSHA256")) != digest(record.get("audioSHA256"))
        or Fraction(integer(quiet.get("samples")), integer(quiet.get("sampleRate"))) != voice
    ):
        raise ValueError("Original quiet audio extent differs")
    endpoints = reader.read(config.get("endpoints"))
    same(
        {"path": endpoints.get("source"), "sha256": endpoints.get("sourceSHA256")},
        config.get("source"),
    )
    streams = rows(mapping(endpoints.get("metadata")).get("streams"))
    if len(streams) != 1 or streams[0].get("codec_type") != "audio":
        raise ValueError("Original endpoint requires one audio stream")
    base = Fraction(string(streams[0].get("time_base")))
    packets = rows(mapping(endpoints.get("tailPackets")).get("packets"))
    previous = -1
    end = 0
    for packet in packets:
        pts, duration = integer(packet.get("pts")), integer(packet.get("duration"))
        if (
            packet.get("codec_type") != "audio"
            or integer(packet.get("stream_index")) != integer(streams[0].get("index"))
            or pts <= previous
            or duration <= 0
            or base <= 0
        ):
            raise ValueError("Original audio endpoint packets differ")
        previous, end = pts, pts + duration
    if end * base != Fraction(string(policy.get("originalAudioEnd"))):
        raise ValueError("Original measured audio end differs")
    assembly = reader.read(config.get("sourceClockEvidence"))
    if assembly.get("exactPresentationClockVerified") is not True or integer(
        assembly.get("frames")
    ) != integer(record.get("observedVFRSourceFrames")):
        raise ValueError("Original physical frame count differs from cleanup")
    result = cut_frame_map(record, frames)
    result.update(
        originalNormalizationVerified=False,
        originalCutApproved=False,
        terminalGridPolicy=policy,
        physicalSourceFrames=record["observedVFRSourceFrames"],
        prospectiveSourceFrames=source_frames,
        pending=[
            "original serialization checkpoint",
            "protected speech decisions",
            "current source crypto",
            "normalization approval",
        ],
    )
    return result


def original_relative_ref(inputs: dict[str, str], value: object, lexical: object) -> dict[str, str]:
    """Bind an explicitly named original relative pin without rewriting it."""
    ref = reference(value)
    name = string(lexical)
    if (
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or str(Path(name)) != name
        or str(Path.cwd() / name) != ref["path"]
        or inputs.get(name) != ref["sha256"]
    ):
        raise ValueError("Original relative reference differs")
    return ref


def original_decoded_extent(record: dict[str, object]) -> tuple[Fraction, int]:
    """Read a bounded retained whole probe, not the per-window clock API."""
    streams = rows(record.get("streams"))
    frames = record.get("frames")
    if (
        len(streams) != 1
        or integer(streams[0].get("index")) != 0
        or integer(streams[0].get("start_pts")) != 0
        or record.get("programs") != []
        or record.get("stream_groups") != []
        or not isinstance(frames, list)
        or not 1 <= len(frames) <= 500_000
    ):
        raise ValueError("Unsupported original whole-frame observation")
    base = Fraction(string(streams[0].get("time_base")))
    previous = -1
    end = 0
    for raw in frames:
        row = mapping(raw)
        pts, duration = integer(row.get("pts")), integer(row.get("duration"))
        if (
            set(row) != {"pts", "best_effort_timestamp", "duration"}
            or integer(row.get("best_effort_timestamp")) != pts
            or pts <= previous
            or (previous == -1 and pts != 0)
            or duration <= 0
            or base <= 0
        ):
            raise ValueError("Original decoded PTS or duration differs")
        previous, end = pts, pts + duration
    return end * base, len(frames)


def parent_probed_cut_evidence(
    config: dict[str, object], reader: "MetadataReader"
) -> dict[str, object]:
    """Preserve a prospective replan bound to its original parent probe."""
    record = reader.read(config.get("receipt"))
    inputs = relative_input_hashes(record.get("inputs"))
    fps, frames = integer(config.get("fps")), integer(config.get("frames"))
    if (
        fps <= 0
        or frames <= 0
        or any(
            key in record
            for key in ("version", "clock", "fps", "decodedSourceExtent", "decodedSourceStart")
        )
        or record.get("grid") != f"prospective {fps}fps; NOT yet normalized media"
        or record.get("normalizationVerified") is not False
        or record.get("cutApproved") is not False
        or record.get("deliveryComplete") is not False
        or record.get("previousRecognitionProtectionRetained") is not True
        or integer(record.get("outputFrames")) != frames
        or inputs != relative_input_hashes(config.get("expectedInputs"))
    ):
        raise ValueError("Original parent-probed cut contract differs")
    parent = reader.read(
        original_relative_ref(inputs, config.get("parent"), config.get("parentInput"))
    )
    if (
        parent.get("grid") != record["grid"]
        or parent.get("normalizationVerified") is not False
        or parent.get("cutApproved") is not False
        or parent.get("deliveryComplete") is not False
        or integer(parent.get("sourceFrames")) != integer(record.get("sourceFrames"))
        or digest(parent.get("audioSHA256")) != digest(record.get("audioSHA256"))
    ):
        raise ValueError("Original parent source or prospective grid differs")
    cut_frame_map(parent, integer(parent.get("outputFrames")))
    parent_inputs = relative_input_hashes(parent.get("inputs"))
    probe = reader.read(
        original_relative_ref(
            parent_inputs, config.get("decodedProbe"), config.get("decodedProbeInput")
        )
    )
    extent, observed = original_decoded_extent(probe)
    source_frames = integer(record.get("sourceFrames"))
    count = extent * fps
    assembly = reader.read(config.get("sourceClockEvidence"))
    if (
        source_frames != -(-count.numerator // count.denominator)
        or observed != source_frames
        or assembly.get("version") != 1
        or integer(assembly.get("frames")) != observed
        or assembly.get("exactPresentationClockVerified") is not True
        or assembly.get("compressedPacketsIdentical") is not True
        or assembly.get("originalSourceIndependentlyDecoded") is not False
    ):
        raise ValueError("Original probe and cleanup source clocks differ")
    quiet_ref = original_relative_ref(inputs, config.get("quiet"), config.get("quietInput"))
    original_relative_ref(parent_inputs, quiet_ref, config.get("quietInput"))
    if digest(reader.read(quiet_ref).get("sourceSHA256")) != digest(record.get("audioSHA256")):
        raise ValueError("Original quiet audio differs")
    result = cut_frame_map(record, frames)
    result.update(
        originalNormalizationVerified=False,
        originalCutApproved=False,
        originalSourceIndependentlyDecoded=False,
        decodedSourceExtent=str(extent),
        observedSourceFrames=observed,
        pending=[
            "original serialization checkpoint",
            "protected speech decisions",
            "current source crypto",
            "normalization approval",
        ],
    )
    return result


def relative_input_hashes(value: object) -> dict[str, str]:
    """Retain old lexical input names without canonical repinning."""
    result = {key: digest(item) for key, item in mapping(value).items()}
    if not 1 <= len(result) <= 1024 or any(
        Path(key).is_absolute() or ".." in Path(key).parts or str(Path(key)) != key or not key
        for key in result
    ):
        raise ValueError("Invalid original relative input pins")
    return result


def cut_frame_map(record: dict[str, object], expected_frames: int) -> dict[str, object]:
    source_frames = integer(record.get("sourceFrames"))
    output_end = previous_source = 0
    cuts = rows(record.get("cuts"))
    for cut in cuts:
        start = integer(cut.get("source_start_frame"))
        end = integer(cut.get("source_end_frame"))
        output = integer(cut.get("output_start_frame"))
        if start < previous_source or not start < end <= source_frames or output != output_end:
            raise ValueError("Nonmonotonic or discontinuous cut map")
        previous_source = end
        output_end += end - start
    if output_end != expected_frames:
        raise ValueError("Cut endpoint differs from final clock")
    return recorded(
        {
            "cuts": len(cuts),
            "outputFrames": output_end,
            "nonSpeechRemovalVerified": record.get("nonSpeechRemovalVerified") is True,
        }
    )


def voice_domain(
    record: dict[str, object],
    *,
    expected_inputs: dict[str, str],
    output: dict[str, object],
    frames: int,
    fps: int,
) -> dict[str, object]:
    if (
        record.get("version") != 1
        or record.get("kind") != "context-atempo-exact-cut-slot-candidate"
    ):
        raise ValueError("Unsupported voice producer")
    if hashes(record.get("inputs")) != hashes(expected_inputs):
        raise ValueError("Voice source, tool, code or clock lineage differs")
    if string(record.get("output")) != string(output.get("path")) or digest(
        record.get("outputSHA256")
    ) != digest(output.get("sha256")):
        raise ValueError("Voice output differs from assembly")
    rate = integer(record.get("sampleRate"))
    samples = integer(record.get("outputSamples"))
    if (
        rate <= 0
        or fps <= 0
        or frames <= 0
        or (integer(record.get("channels")) != 2)
        or (integer(record.get("outputStartSample")) != 0)
        or (integer(record.get("outputEndSample")) != samples)
        or (Fraction(samples, rate) != Fraction(frames, fps))
    ):
        raise ValueError("Voice and final clocks differ")
    endpoint = 0
    cuts = rows(record.get("cuts"))
    for cut in cuts:
        start = integer(cut.get("output_start_sample"))
        end = integer(cut.get("output_end_sample"))
        if (
            start != endpoint
            or end <= start
            or (
                cut.get("filter_applied") is not True
                and not exact_tempo_passthrough(cut, end - start)
            )
            or (integer(cut.get("source_start_sample")) < 0)
            or (integer(cut.get("source_end_sample")) <= integer(cut.get("source_start_sample")))
        ):
            raise ValueError("Incomplete or degraded voice slots")
        if (
            integer(cut.get("center_crop_end_sample"))
            - integer(cut.get("center_crop_start_sample"))
            != end - start
        ):
            raise ValueError("Voice crop length differs")
        endpoint = end
    if endpoint != samples:
        raise ValueError("Voice slots do not cover whole output")
    return recorded(
        {
            "cuts": len(cuts),
            "samples": samples,
            "sampleRate": rate,
            "recordedAVSyncVerified": record.get("AVSyncVerified") is True,
            "recordedSourceCorrespondenceVerified": record.get("sourceCorrespondenceVerified")
            is True,
        }
    )


def exact_tempo_passthrough(cut: dict[str, object], samples: int) -> bool:
    """The original tempo producer copies a ratio-one slot without filtering."""
    return (
        cut.get("filter_applied") is False
        and integer(cut.get("source_end_sample")) - integer(cut.get("source_start_sample"))
        == samples
        and all(
            integer(cut.get(key)) == 0
            for key in (
                "source_context_samples",
                "synthetic_left_context_samples",
                "synthetic_right_context_samples",
                "tail_error_samples",
                "center_crop_start_sample",
            )
        )
        and all(
            integer(cut.get(key)) == samples
            for key in (
                "filter_output_samples",
                "expected_filter_output_samples",
                "center_crop_end_sample",
            )
        )
    )


def archived_command_stage(
    config: dict[str, object], reader: MetadataReader, *, required_command: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Recognize literal retained phase wrappers; never execute a historical command."""
    graph = reader.read(config.get("workflow"))
    matches = [row for row in rows(graph.get("stages")) if row.get("name") == config.get("stage")]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous original stage")
    stage = matches[0]
    command = strings(stage.get("command"))
    commands = [command]
    if len(command) == 3 and command[:2] == ["/bin/bash", "-c"]:
        lines = command[2].splitlines()
        if not lines or lines[0] != "set -euo pipefail":
            raise ValueError("Unsupported recorded shell wrapper")
        commands = [shlex.split(line) for line in lines[1:]]
    unwrapped: list[list[str]] = []
    for item in commands:
        if len(item) > 7 and item[1:4] == ["-B", "-m", "replay.phase_timer"] and item[6] == "--":
            item = item[7:]
        unwrapped.append(item)
    if required_command not in unwrapped:
        raise ValueError("Recorded typed producer command differs")
    inventory = hashes(reader.read(config.get("inventory")).get("inputs"))
    declared = strings(stage.get("inputs"))
    if any(path not in inventory for path in declared):
        raise ValueError("Original stage inventory is incomplete")
    outputs = hashes(config.get("outputs"))
    stage_provenance(
        stage=stage,
        checkpoint=reader.read(config.get("checkpoint")),
        input_hashes={path: inventory[path] for path in declared},
        cwd=string(config.get("cwd")),
        expected_outputs=outputs,
    )
    return {path: inventory[path] for path in declared}, outputs


def normalized_combined_evidence(
    config: dict[str, object], reader: MetadataReader
) -> dict[str, object]:
    """Assess archived combined-v3 domains, not a fabricated v1 decode receipt."""
    output = reference(config.get("output"))
    observation_ref, technical_ref, audit_ref, normal_ref = (
        reference(config.get(key))
        for key in ("observation", "receipt", "audioAudit", "normalization")
    )
    observation = reader.read(observation_ref)
    technical, audit, normal = (
        reader.read(technical_ref),
        reader.read(audit_ref),
        reader.read(normal_ref),
    )
    snapshot = reader.read(observation.get("snapshot"))
    process = reader.read(observation.get("process"))
    if (
        observation.get("version") != 3
        or observation.get("kind") != "single-pass-whole-observation"
        or observation.get("videoContract") != "h264"
        or snapshot.get("version") != 3
        or snapshot.get("videoContract") != "h264"
        or integer(process.get("returncode")) != 0
        or process.get("command") != snapshot.get("command")
        or observation.get("inputs") != snapshot.get("inputs")
    ):
        raise ValueError("Combined observation/process/snapshot differs")
    pins: dict[str, str] = {}
    for row in rows(snapshot.get("inputs")):
        ref = reference(row)
        if ref["path"] in pins:
            raise ValueError("Duplicate original combined input")
        pins[ref["path"]] = ref["sha256"]
    if pins.get(output["path"]) != output["sha256"]:
        raise ValueError("Combined final movie differs")
    frames, fps = integer(config.get("frames")), integer(config.get("fps"))
    if frames <= 0 or fps <= 0:
        raise ValueError("Invalid combined edited clock")
    expected = {
        "version": 3,
        "videoContract": "h264",
        "technicalAccepted": True,
        "fullDecode": True,
        "decodePasses": 1,
        "decoder": "ffprobe-combined-frames-packets",
        "videoClock": {
            "frames": frames,
            "start": "0",
            "end": str(Fraction(frames, fps)),
            "exactCFR": True,
        },
        "inputs": [output],
        "deliveryApproved": False,
    }
    if technical != expected:
        raise ValueError("Combined original technical clock/schema differs")
    if (
        normal.get("version") != 1
        or normal.get("kind") != "observed-aac-leading-exclusion-to-presentation-zero"
        or normal.get("output") != output["path"]
        or normal.get("timestampField") != "best_effort_timestamp"
    ):
        raise ValueError("Unsupported original normalization")
    inputs = mapping(normal.get("inputs"))
    if set(inputs) != {"audio", "video", "packetProbe", "decodedProbe", "tagProbe", "ffmpeg"}:
        raise ValueError("Unsupported normalization input roles")
    presentation = mapping(audit.get("presentationNormalization"))
    same(presentation.get("proof"), normal_ref)
    rate, samples, priming = (
        integer(normal.get("sampleRate")),
        integer(normal.get("decodedSamples")),
        integer(normal.get("primingSamples")),
    )
    if (
        rate <= 0
        or samples <= 0
        or priming < 0
        or integer(normal.get("preRollTicks")) < 0
        or integer(normal.get("tailDiscardSamples")) < 0
        or audit.get("version") != 3
        or audit.get("videoContract") != "h264"
        or audit.get("candidateMuxReady") is not True
        or any(
            audit.get(key) is not True
            for key in ("orderedPayloadEqual", "sourceClockPreserved", "internalPrimingPreserved")
        )
        or audit.get("sourceClockTimestampField") != "best_effort_timestamp"
        or audit.get("sampleRate") != rate
        or audit.get("decodedSamples") != samples
        or Fraction(string(audit.get("expectedEnd"))) != Fraction(frames, fps)
        or presentation.get("primingSamples") != priming
        or presentation.get("preRollTicks") != normal.get("preRollTicks")
        or presentation.get("sourceClockPreservedUnderDeclaredMapping") is not True
    ):
        raise ValueError("Normalized recorded audio clock differs")
    stages = mapping(config.get("stages"))
    observer_ref = reference(config.get("observerConfig"))
    observer = reader.read(observer_ref)
    tools = mapping(observer.get("tools"))
    audio_source = mapping(observer.get("audio"))
    same(audio_source.get("movie"), inputs["audio"])
    same(tools.get("ffmpeg"), inputs["ffmpeg"])
    audio_proof = reader.read(audio_source.get("evidence"))
    same(audio_proof.get("packetProbe"), inputs["packetProbe"])
    same(audio_proof.get("decodedProbe"), inputs["decodedProbe"])
    normal_config_ref = reference(normal.get("config"))
    normal_config = reader.read(normal_config_ref)
    for key, value in inputs.items():
        same(normal_config.get(key), value)
    if normal_config.get("output") != output["path"] or normal_config.get(
        "timestampField"
    ) != normal.get("timestampField"):
        raise ValueError("Original normalization config differs")
    scale = integer(normal_config.get("videoTrackTimescale"))
    ticks = integer(normal.get("preRollTicks"))
    if (
        normal_config.get("version") != 1
        or normal_config.get("videoTag") != "avc1"
        or not 1 <= scale <= 1_000_000
        or Fraction(string(normal.get("timeBase"))) * ticks != Fraction(priming, rate)
    ):
        raise ValueError("Original normalization timescale differs")
    digest(normal.get("sourceFrameClockSha256"))
    expected_mux = [
        reference(inputs["ffmpeg"])["path"],
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-copyts",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        reference(inputs["video"])["path"],
        "-i",
        reference(inputs["audio"])["path"],
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        "-bsf:a:0",
        f"setts=pts=PTS-{ticks}:dts=DTS-{ticks}",
        "-avoid_negative_ts",
        "disabled",
        "-use_editlist",
        "1",
        "-tag:v",
        "avc1",
        "-video_track_timescale",
        str(scale),
        "-movie_timescale",
        str(rate),
        "-movflags",
        "+faststart",
        output["path"],
    ]
    if normal.get("producerCommand") != expected_mux:
        raise ValueError("Original normalization copy-only command differs")
    ffprobe = reference(tools.get("ffprobe"))
    expected_probe = [
        ffprobe["path"],
        "-v",
        "error",
        "-err_detect",
        "explode",
        "-protocol_whitelist",
        "file,pipe",
        "-show_error",
        "-show_frames",
        "-show_packets",
        "-show_streams",
        "-show_format",
        "-show_data_hash",
        "sha256",
        "-show_entries",
        "frame=media_type,stream_index,pts,duration,pkt_duration,nb_samples,decode_error_flags:packet=stream_index,pts,dts,duration,pos,size,data_hash:packet_side_data=side_data_type,skip_samples,discard_padding:stream:format:error",
        "-of",
        "json",
        output["path"],
    ]
    if process.get("command") != expected_probe or pins.get(ffprobe["path"]) != ffprobe["sha256"]:
        raise ValueError("Original full combined decoder command differs")
    observe_command, audio_command = (
        strings(config.get("observationCommand")),
        strings(config.get("audioCommand")),
    )
    base = [
        string(config.get("python")),
        "-B",
        "-m",
        "replay.compressed_audio_verification",
        observer_ref["path"],
        str(Path(observation_ref["path"]).parent),
    ]
    if (
        observe_command
        != [*base, "--observe", "--video-contract", "h264", ">", technical_ref["path"]]
        or audio_command
        != [
            *base,
            audit_ref["path"],
            "--audio-result",
            "--video-contract",
            "h264",
            "--source-timestamp-field",
            "best_effort_timestamp",
            "--normalization-binding",
            normal_ref["path"],
            "--normalization-sha256",
            normal_ref["sha256"],
        ]
        or observer.get("output") != output["path"]
    ):
        raise ValueError("Normalized typed consumer command/config differs")
    observe_inputs, observe_outputs = archived_command_stage(
        mapping(stages.get("observation")),
        reader,
        required_command=strings(config.get("observationCommand")),
    )
    audit_inputs, audit_outputs = archived_command_stage(
        mapping(stages.get("audio")), reader, required_command=strings(config.get("audioCommand"))
    )
    mux_inputs, mux_outputs = archived_command_stage(
        mapping(stages.get("normalization")),
        reader,
        required_command=strings(normal.get("producerCommand")),
    )
    if (
        observe_outputs.get(technical_ref["path"]) != technical_ref["sha256"]
        or observe_outputs.get(observation_ref["path"]) != observation_ref["sha256"]
        or audit_outputs.get(audit_ref["path"]) != audit_ref["sha256"]
        or mux_outputs.get(output["path"]) != output["sha256"]
        or audit_inputs.get(normal_ref["path"]) != normal_ref["sha256"]
        or mux_inputs.get(normal_ref["path"]) != normal_ref["sha256"]
        or mux_inputs.get(normal_config_ref["path"]) != normal_config_ref["sha256"]
        or observe_inputs.get(observer_ref["path"]) != observer_ref["sha256"]
        or audit_inputs.get(observer_ref["path"]) != observer_ref["sha256"]
    ):
        raise ValueError("Normalized original stage output binding differs")
    for key in ("raw", "snapshot", "process"):
        ref = reference(observation.get(key))
        if observe_outputs.get(ref["path"]) != ref["sha256"]:
            raise ValueError("Combined original artifact was not completed")
    for path, sha in pins.items():
        if observe_inputs.get(path) != sha:
            raise ValueError("Combined original input closure differs")
    for value in inputs.values():
        ref = reference(value)
        if mux_inputs.get(ref["path"]) != ref["sha256"]:
            raise ValueError("Normalization original input closure differs")
    compressors: dict[str, object] = {}
    for role in ("video", "audio"):
        original = mapping(config.get(role + "Compressor"))
        if original.get("contract") == "original-split-native-request-v1":
            compressors[role] = split_native_request_evidence(
                original,
                reader,
                output=reference(inputs[role]),
                input_hashes=mux_inputs,
            )
            if role == "audio":
                same(original.get("source"), audio_source.get("pcm"))
                same(original.get("outputEvidence"), audio_source.get("evidence"))
        else:
            compressors[role] = compressor_execution(
                submit=reader.read(original.get("submit"), native_submit=True),
                status=reader.read(original.get("status")),
                request=reader.read(original.get("request")),
                success=reader.read(original.get("success")),
                output_path=reference(inputs[role])["path"],
            )
    result = recorded(
        {
            "scope": "original normalized combined-v3 execution",
            "frames": frames,
            "decodedSamples": samples,
            "originalAudioMuxAccepted": audit.get("audioMuxAccepted") is True,
            "rawObservationReparsed": False,
        }
    )
    result["pending"] = [
        "current media crypto",
        "fresh raw observation validation",
        "speech synchronization and quality",
    ]
    result["originalCompressors"] = compressors
    return result


def mixed_voice_evidence(config: dict[str, object], reader: MetadataReader) -> dict[str, object]:
    """Bind the original measured/copy mixed producer without reopening PCM."""
    record = reader.read(config.get("receipt"))
    producer_ref = reference(config.get("producerConfig"))
    producer = reader.read(producer_ref)
    pins = hashes(config.get("expectedInputs"))
    if (
        record.get("version") != 1
        or producer.get("version") != 1
        or record.get("kind") != "mixed-tempo-candidate"
        or record.get("preflightOnly") is not False
        or hashes(record.get("inputs")) != pins
    ):
        raise ValueError("Unsupported or incomplete mixed voice producer")
    if pins.get(producer_ref["path"]) != producer_ref["sha256"]:
        raise ValueError("Mixed producer config is not an original input")
    roles = {
        key: reference(producer.get(key))
        for key in (
            "reference",
            "candidateMixture",
            "candidateStem",
            "fallback",
            "fallbackReceipt",
            "selection",
            "mapping",
        )
    }
    required = [*roles.values(), *[reference(item) for item in rows(producer.get("evidence"))]]
    if not required[7:] or any(pins.get(ref["path"]) != ref["sha256"] for ref in required):
        raise ValueError("Mixed original source/evidence binding differs")
    output = reference(config.get("output"))
    rate, samples = integer(record.get("sampleRate")), integer(record.get("outputSamples"))
    frames, fps = integer(config.get("frames")), integer(config.get("fps"))
    if (
        rate <= 0
        or samples <= 0
        or frames <= 0
        or fps <= 0
        or record.get("channels") != 2
        or record.get("outputSHA256") != output["sha256"]
        or Fraction(samples, rate) != Fraction(frames, fps)
    ):
        raise ValueError("Mixed output/sample clock differs")
    fallback = reader.read(roles["fallbackReceipt"])
    if (
        fallback.get("kind") != "context-atempo-exact-cut-slot-candidate"
        or fallback.get("output") != roles["fallback"]["path"]
        or fallback.get("outputSHA256") != roles["fallback"]["sha256"]
        or fallback.get("outputSamples") != samples
        or fallback.get("sampleRate") != rate
        or fallback.get("channels") != 2
    ):
        raise ValueError("Mixed fallback clock or source differs")
    selection = reader.read(roles["selection"])
    same(selection.get("source"), roles["reference"])
    clock = reference(selection.get("clock"))
    if (
        selection.get("coordinateSpace") != "native-presentation-seconds"
        or pins.get(clock["path"]) != clock["sha256"]
    ):
        raise ValueError("Mixed selection clock differs")
    support = reader.read(roles["mapping"])
    support_pins = hashes(support.get("inputs"))
    if (
        support.get("kind") != "selected-pcm-correspondence-diagnostic"
        or support.get("sampleRate") != rate
        or support.get("channels") != 2
        or any(
            support_pins.get(roles[key]["path"]) != roles[key]["sha256"]
            for key in ("reference", "candidateMixture", "selection")
        )
        or support_pins.get(clock["path"]) != clock["sha256"]
    ):
        raise ValueError("Mixed measured support binding differs")
    cuts, old = rows(record.get("cuts")), rows(fallback.get("cuts"))
    if not cuts or len(cuts) != len(old):
        raise ValueError("Mixed cut inventory differs")
    endpoint = 0
    copied = 0
    for cut, original in zip(cuts, old, strict=True):
        start, end = integer(cut.get("outputStartSample")), integer(cut.get("outputEndSample"))
        if (
            start != endpoint
            or end <= start
            or original.get("output_start_sample") != start
            or original.get("output_end_sample") != end
        ):
            raise ValueError("Mixed cut output clock differs")
        digest(cut.get("pcmSHA256"))
        if cut.get("sourceClock") == "selected-output":
            same(
                {"path": cut.get("sourcePath"), "sha256": cut.get("sourceSHA256")},
                roles["fallback"],
            )
            if (
                cut.get("tempo") is not None
                or cut.get("fallbackReadSpan") != [start, end]
                or cut.get("sourceSpans") != [[start, end]]
            ):
                raise ValueError("Mixed copied fallback span differs")
            copied += 1
        elif cut.get("sourceClock") == "measured-continuous-pcm":
            same(
                {"path": cut.get("sourcePath"), "sha256": cut.get("sourceSHA256")},
                roles["candidateStem"],
            )
            tempo = mapping(cut.get("tempo"))
            low, high = (
                integer(tempo.get("source_start_sample")),
                integer(tempo.get("source_end_sample")),
            )
            if (
                low < 0
                or high <= low
                or cut.get("sourceSpans") != [[low, high]]
                or cut.get("fallbackReadSpan") is not None
                or tempo.get("output_start_sample") != start
                or tempo.get("output_end_sample") != end
                or integer(tempo.get("center_crop_end_sample"))
                - integer(tempo.get("center_crop_start_sample"))
                != end - start
                or (
                    tempo.get("filter_applied") is not True
                    and not (tempo.get("filter_applied") is False and high - low == end - start)
                )
            ):
                raise ValueError("Mixed measured tempo clock differs")
            source_low = integer(original.get("source_start_sample"))
            source_high = integer(original.get("source_end_sample"))
            margin = (
                integer(tempo.get("source_context_samples")) if high - low != end - start else 0
            )
            if not any(
                integer(span.get("start")) <= source_low - margin
                and integer(span.get("end")) >= source_high + margin
                and low - source_low
                == integer(span.get("candidate_start")) - integer(span.get("start"))
                and high - source_high == low - source_low
                for span in rows(support.get("measuredSupport"))
            ):
                raise ValueError("Mixed source mapping lacks measured support")
        else:
            raise ValueError("Unsupported mixed source clock")
        endpoint = end
    if (
        endpoint != samples
        or record.get("fallbackCuts") != copied
        or record.get("newStemCuts") != len(cuts) - copied
    ):
        raise ValueError("Mixed coverage/count differs")
    return recorded(
        {
            "cuts": len(cuts),
            "copiedFallbackCuts": copied,
            "samples": samples,
            "sampleRate": rate,
            "scope": "original mixed-tempo recorded execution",
            "originalQualityAccepted": record.get("qualityAccepted") is True,
        }
    )


def reference(value: object) -> dict[str, str]:
    item = mapping(value)
    path = string(item.get("path"))
    if not Path(path).is_absolute():
        raise ValueError("Absolute recorded reference required")
    return {"path": path, "sha256": digest(item.get("sha256"))}


def same(value: object, expected: object) -> None:
    if reference(value) != reference(expected):
        raise ValueError("Original reference differs")


def cleanup_assembly(
    receipt: dict[str, object],
    index: dict[str, object],
    index_ref: dict[str, object],
    clean: dict[str, object],
) -> dict[str, object]:
    if (
        integer(receipt.get("version")) != 1
        or receipt.get("exactPresentationClockVerified") is not True
    ):
        raise ValueError("No successful presentation-clock assembly record")
    same(receipt.get("output"), clean)
    if digest(receipt.get("indexSHA256")) != reference(index_ref)["sha256"]:
        raise ValueError("Wrong assembled index")
    ranges = rows(index.get("ranges"))
    if len(ranges) != 1:
        raise ValueError("Expected original single-source index")
    item = ranges[0]
    same({"path": item.get("media"), "sha256": item.get("sha256")}, clean)
    frames = integer(receipt.get("frames"))
    if frames <= 0 or integer(item.get("frames")) != frames:
        raise ValueError("Cleanup frame coverage differs")
    if integer(item.get("sourceEndPTS")) <= integer(item.get("sourceFirstPTS")):
        raise ValueError("Invalid original cleanup PTS")
    result = recorded({"assemblyFrames": frames, "scope": "lossless clean-source assembly only"})
    result["pending"] = [
        "per-chunk local repair admission",
        "cleanup quality evidence",
        "original producer checkpoint reconstruction",
    ]
    return result


def native_part(
    proof: dict[str, object],
    *,
    partition: dict[str, object],
    movie: dict[str, object],
    part: int,
    frames: int,
    fps: int,
    width: int,
    height: int,
    tool: dict[str, object],
) -> dict[str, object]:
    if integer(proof.get("version")) != 1 or integer(proof.get("part")) != part or part <= 0:
        raise ValueError("Wrong part identity")
    same(proof.get("partition"), partition)
    same(proof.get("movie"), movie)
    provenance = mapping(proof.get("probeProvenance"))
    if integer(provenance.get("version")) != 1 or integer(provenance.get("decoderThreads")) != 0:
        raise ValueError("Unsupported frame probe provenance")
    same(provenance.get("executable"), tool)
    return native_part_clock(
        proof,
        partition=partition,
        movie=movie,
        part=part,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
    )


def native_part_clock(
    proof: dict[str, object],
    *,
    partition: dict[str, object],
    movie: dict[str, object],
    part: int,
    frames: int,
    fps: int,
    width: int,
    height: int,
) -> dict[str, object]:
    """Validate retained frame observations; this does not establish tool provenance."""
    if integer(proof.get("version")) != 1 or integer(proof.get("part")) != part or part <= 0:
        raise ValueError("Wrong part identity")
    same(proof.get("partition"), partition)
    same(proof.get("movie"), movie)
    observed = mapping(proof.get("observed"))
    streams = rows(observed.get("streams"))
    if len(streams) != 1 or frames <= 0 or fps <= 0:
        raise ValueError("Invalid observed video scope")
    video = streams[0]
    base = Fraction(string(video.get("time_base")))
    if (
        video.get("codec_type") != "video"
        or integer(video.get("width")) != width
        or integer(video.get("height")) != height
        or (integer(video.get("start_pts")) != 0)
        or (Fraction(string(video.get("avg_frame_rate"))) != fps)
        or (base <= 0)
        or (int(string(video.get("nb_read_frames"))) != frames)
        or (integer(video.get("duration_ts")) * base != Fraction(frames, fps))
    ):
        raise ValueError("Part clock or geometry differs")
    timestamps = rows(observed.get("frames"))
    if len(timestamps) != frames:
        raise ValueError("Missing decoded frame observations")
    for number, frame in enumerate(timestamps):
        if integer(frame.get("best_effort_timestamp")) * base != Fraction(number, fps):
            raise ValueError("Part is not exact contiguous CFR")
    return recorded(
        {
            "part": part,
            "decodedFrameObservations": frames,
            "scope": "retained video frame proof, not native dispatch/full AV acceptance",
        }
    )


def caption_lineage(
    refresh: dict[str, object], *, consumer: dict[str, object], required_inputs: dict[str, str]
) -> dict[str, object]:
    if (
        integer(refresh.get("version")) != 1
        or refresh.get("captionMediaClockUnchanged") is not True
    ):
        raise ValueError("Caption clock preservation is not recorded")
    same(refresh.get("output"), consumer)
    actual = mapping(refresh.get("inputs"))
    if len(actual) > 100000:
        raise ValueError("Caption input inventory exceeds bound")
    expected = hashes(required_inputs)
    if any((actual.get(path) != sha for path, sha in expected.items())):
        raise ValueError("Old or unbound caption/template/font/POS input")
    result = recorded(
        {"boundInputCount": len(expected), "scope": "original caption refresh lineage"}
    )
    result["pending"] = [
        "ASR model and projection provenance",
        "complete POS timing contract",
        "actual style/font/Y quality",
        "original producer checkpoint reconstruction",
    ]
    return result


def head_lineage(receipt: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    """Recorded application of exact configured reflow and template, not rendered quality."""
    if integer(receipt.get("version")) != 1 or receipt.get("allCaptionStagesRetained") is not True:
        raise ValueError("Incomplete original caption stage retention")
    inputs = mapping(receipt.get("inputs"))
    if len(inputs) > 100000:
        raise ValueError("Caption inventory exceeds bound")
    refs: dict[str, object] = {}
    for key in ("reflow", "cuts", "template", "measurementReference"):
        ref = reference(config.get(key))
        if inputs.get(ref["path"]) != ref["sha256"]:
            raise ValueError("Configured caption input differs from original execution")
        refs[key] = ref
    return {
        "stageApplied": True,
        "executionRecorded": True,
        "executionVerified": False,
        "currentCryptoVerified": False,
        "qualityApproved": False,
        "boundInputs": refs,
        "scope": "original full caption reflow/template application",
        "pending": [
            "POS semantic quality",
            "painted style/font/Y quality",
            "original head producer checkpoint reconstruction",
        ],
    }


def original_part_stage(
    evidence: dict[str, object],
    *,
    output: dict[str, str],
    part: int,
    reuse: bool,
    reader: MetadataReader,
    semantic: dict[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    graph = reader.read(evidence.get("workflow"))
    matches = [
        item for item in rows(graph.get("stages")) if item.get("name") == evidence.get("stage")
    ]
    if len(matches) != 1:
        raise ValueError("Missing original native stage")
    stage = matches[0]
    command = strings(stage.get("command"))
    if not reuse and command[1:5] == ["-B", "-m", "replay.fcp_part_followup", "frames"]:
        if (
            semantic is None
            or len(command) != 7
            or not Path(command[0]).is_absolute()
            or command[6] != semantic["path"]
            or strings(stage.get("verify_command")) != [*command[:4], "verify-frames", *command[5:]]
        ):
            raise ValueError("Unsupported original followup frames command")
        inputs = hashes(reader.read(evidence.get("inventory")).get("inputs"))
        if inputs.get(semantic["path"]) != semantic["sha256"] or command[5] not in inputs:
            raise ValueError("Original followup frames input differs")
        config = reader.read({"path": command[5], "sha256": inputs[command[5]]})
        assembly_ref = reference(config.get("assembly"))
        if (
            integer(config.get("part")) != part
            or inputs.get(assembly_ref["path"]) != assembly_ref["sha256"]
        ):
            raise ValueError("Original followup frames assembly differs")
        stage_provenance(
            stage=stage,
            checkpoint=reader.read(evidence.get("checkpoint")),
            input_hashes=inputs,
            cwd=string(evidence.get("cwd")),
            expected_outputs={output["path"]: output["sha256"]},
        )
        return reader.read(assembly_ref), inputs
    expected_module = "replay.fcp_part_reuse" if reuse else "replay.fcp_part_assembly"
    if command[1:4] != ["-B", "-m", expected_module] or not Path(command[0]).is_absolute():
        raise ValueError("Unsupported original native command")
    if reuse:
        if len(command) != 5:
            raise ValueError("Unsupported original reuse arguments")
        config_path = command[4]
    else:
        if len(command) != 8 or command[4] != "probe" or command[6:] != ["--part", str(part)]:
            raise ValueError("Unsupported original probe arguments")
        config_path = command[5]
    inventory = hashes(reader.read(evidence.get("inventory")).get("inputs"))
    declared = strings(stage.get("inputs"))
    if any(path not in inventory for path in declared) or config_path not in declared:
        raise ValueError("Incomplete original native inventory")
    inputs = {path: inventory[path] for path in declared}
    source_name = "fcp_part_reuse.py" if reuse else "fcp_part_assembly.py"
    if not any(Path(path).name == source_name for path in inputs):
        raise ValueError("Original native producer pin absent")
    config = reader.read({"path": config_path, "sha256": inputs[config_path]})
    assembly_path = reference(config.get("targetAssembly"))["path"] if reuse else config_path
    if strings(stage.get("verify_command")) != [
        command[0],
        "-B",
        "-m",
        "replay.fcp_part_assembly",
        "verify",
        assembly_path,
        "--part",
        str(part),
    ]:
        raise ValueError("Original native verifier differs")
    stage_provenance(
        stage=stage,
        checkpoint=reader.read(evidence.get("checkpoint")),
        input_hashes=inputs,
        cwd=string(evidence.get("cwd")),
        expected_outputs={output["path"]: output["sha256"]},
    )
    return config, inputs


def original_part_reuse(
    item: dict[str, object],
    *,
    selected: dict[str, object],
    published: dict[str, object],
    publication: dict[str, object],
    partition: dict[str, str],
    part: int,
    reader: MetadataReader,
) -> None:
    selected_ref = reference(item.get("selectedProof"))
    config, inputs = original_part_stage(
        mapping(item.get("reuseStage")), output=selected_ref, part=part, reuse=True, reader=reader
    )
    reuse = mapping(selected.get("crossRevisionReuse"))
    if reader.read(reuse.get("config")) != config:
        raise ValueError("Original reuse configuration differs")
    same(config.get("publication"), item.get("publication"))
    if config.get("version") != 1 or integer(config.get("part")) != part:
        raise ValueError("Original reuse part differs")
    target_ref = reference(config.get("targetAssembly"))
    target = reader.read(target_ref)
    same(target.get("partition"), partition)
    target_parts = rows(target.get("parts"))
    if (
        part > len(target_parts)
        or target_parts[part - 1].get("proof") != selected_ref["path"]
        or target_parts[part - 1].get("movie") != reference(publication.get("movie"))["path"]
    ):
        raise ValueError("Original reuse target differs")
    same(reuse.get("priorFrameProof"), publication.get("frameProof"))
    same(reuse.get("priorPartition"), published.get("partition"))
    same(reuse.get("publication"), item.get("publication"))
    same(reuse.get("completion"), publication.get("completion"))
    same(reuse.get("intent"), publication.get("intent"))
    reuse_inputs = hashes(reuse.get("inputs"))
    for ref in (
        reference(reuse.get("config")),
        target_ref,
        partition,
        reference(publication.get("frameProof")),
        reference(item.get("publication")),
    ):
        if (
            inputs.get(ref["path"]) != ref["sha256"]
            or reuse_inputs.get(ref["path"]) != ref["sha256"]
        ):
            raise ValueError("Original reuse binding differs")
    if any(
        reuse.get(key) is not True
        for key in ("projectDisplayNameOnlyIgnored", "partXMLIdentical", "frameRangesUnchanged")
    ):
        raise ValueError("Original reuse equivalence not recorded")
    expected = {key: published[key] for key in ("version", "part", "movie", "observed")}
    expected.update(partition=partition, audioIgnored=True, crossRevisionReuse=reuse)
    if json.dumps(selected, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Original reuse transformed frame record differs")


def original_integrated_part(
    evidence: dict[str, object],
    *,
    output: dict[str, str],
    part: int,
    semantic: dict[str, str],
    reader: MetadataReader,
) -> tuple[dict[str, object], dict[str, str]]:
    """Read the original integrated producer, never manufacture a separate probe stage."""
    graph = reader.read(evidence.get("workflow"))
    stages = [row for row in rows(graph.get("stages")) if row.get("name") == evidence.get("stage")]
    if len(stages) != 1:
        raise ValueError("Missing original integrated stage")
    stage = stages[0]
    command = strings(stage.get("command"))
    operation_ref, terminal_ref = (
        reference(evidence.get("operation")),
        reference(evidence.get("terminal")),
    )
    if (
        len(command) != 7
        or not Path(command[0]).is_absolute()
        or command[1:5] != ["-B", "-m", "replay.fcp_part_followup", "run-integrated"]
        or command[5:] != [operation_ref["path"], semantic["path"]]
        or strings(stage.get("verify_command")) != [*command[:4], "verify-integrated", *command[5:]]
    ):
        raise ValueError("Unsupported original integrated command")
    inventory = hashes(reader.read(evidence.get("inventory")).get("inputs"))
    operation, terminal = reader.read(operation_ref), reader.read(terminal_ref)
    operation_inputs = reference_inventory(operation.get("inputs"))
    if (
        operation.get("version") != 1
        or operation.get("kind") != "bounded-native-part-followup"
        or operation.get("semantic") != semantic["path"]
        or operation_inputs.get(semantic["path"]) != semantic["sha256"]
        or inventory.get(operation_ref["path"]) != operation_ref["sha256"]
        or terminal.get("version") != 1
        or terminal.get("kind") != "verified-integrated-native-followup"
        or integer(terminal.get("part")) != part
        or hashes(terminal.get("inputs"))
        != {**operation_inputs, operation_ref["path"]: operation_ref["sha256"]}
    ):
        raise ValueError("Original integrated operation/terminal binding differs")
    same(terminal.get("operation"), operation_ref)
    outputs = reference_inventory(terminal.get("outputs"))
    if len(outputs) != 6 or outputs.get(output["path"]) != output["sha256"]:
        raise ValueError("Original integrated proof output differs")
    stage_provenance(
        stage=stage,
        checkpoint=reader.read(evidence.get("checkpoint")),
        input_hashes=inventory,
        cwd=string(evidence.get("cwd")),
        expected_outputs={**outputs, terminal_ref["path"]: terminal_ref["sha256"]},
    )
    config_path = string(operation.get("config"))
    if config_path not in operation_inputs:
        raise ValueError("Original integrated config pin absent")
    config = reader.read({"path": config_path, "sha256": operation_inputs[config_path]})
    if integer(config.get("part")) != part:
        raise ValueError("Original integrated config part differs")
    assembly_ref = reference(config.get("assembly"))
    if operation_inputs.get(assembly_ref["path"]) != assembly_ref["sha256"]:
        raise ValueError("Original integrated assembly pin differs")
    return reader.read(assembly_ref), operation_inputs


def reference_inventory(value: object) -> dict[str, str]:
    refs = [reference(row) for row in rows(value)]
    result = {row["path"]: row["sha256"] for row in refs}
    if len(result) != len(refs):
        raise ValueError("Duplicate original reference inventory")
    return hashes(result)


def original_batch_part(
    evidence: dict[str, object],
    *,
    part: int,
    semantic: dict[str, str],
    reader: MetadataReader,
) -> tuple[dict[str, object], dict[str, str], dict[str, str]]:
    """Read completed batch/step records without executing or inventing child stages."""
    plan_ref, terminal_ref, operation_ref = (
        reference(evidence.get(key)) for key in ("plan", "terminal", "operation")
    )
    plan, terminal, operation = (
        reader.read(plan_ref),
        reader.read(terminal_ref),
        reader.read(operation_ref),
    )
    stage_ref = mapping(evidence.get("stageEvidence"))
    graph = reader.read(stage_ref.get("workflow"))
    matches = [
        row for row in rows(graph.get("stages")) if row.get("name") == stage_ref.get("stage")
    ]
    if len(matches) != 1:
        raise ValueError("Original batch stage is ambiguous")
    stage = matches[0]
    command = strings(stage.get("command"))
    if (
        len(command) != 8
        or not Path(command[0]).is_absolute()
        or command[1:5] != ["-B", "-m", "replay.fcp_part_followup_batch", "run"]
        or command[5:] != [plan_ref["path"], "--semantic", semantic["path"]]
        or strings(stage.get("verify_command")) != [*command[:4], "verify", *command[5:]]
    ):
        raise ValueError("Original batch command differs")
    inventory = hashes(reader.read(stage_ref.get("inventory")).get("inputs"))
    if inventory.get(plan_ref["path"]) != plan_ref["sha256"]:
        raise ValueError("Original batch plan was not an input")
    stage_provenance(
        stage=stage,
        checkpoint=reader.read(stage_ref.get("checkpoint")),
        input_hashes=inventory,
        cwd=string(stage_ref.get("cwd")),
        expected_outputs={terminal_ref["path"]: terminal_ref["sha256"]},
    )
    parts = rows(plan.get("parts"))
    common = reference_inventory(plan.get("common"))
    expected = dict(common)
    numbers: set[int] = set()
    selected: dict[str, object] | None = None
    expected_checkpoints: list[str] = []
    for row in parts:
        number = integer(row.get("part"))
        if (
            number < 1
            or number in numbers
            or row.get("retirement") is not None
            or row.get("adoption") != []
        ):
            raise ValueError("Unsupported or duplicate original batch part")
        numbers.add(number)
        for key in ("operation", "native"):
            ref = reference(row.get(key))
            if ref["path"] in expected and expected[ref["path"]] != ref["sha256"]:
                raise ValueError("Conflicting original batch inputs")
            expected[ref["path"]] = ref["sha256"]
        expected_checkpoints.extend(
            str(Path(string(plan.get("checkpoints"))) / f"part-{number:04d}-{step}.json")
            for step in ("frames", "complete", "publication-config", "publish")
        )
        if number == part:
            selected = row
    checkpoints = reference_inventory(terminal.get("checkpoints"))
    if (
        plan.get("version") != 1
        or plan.get("kind") != "bounded-native-followup-batch"
        or not 1 <= len(parts) <= 64
        or selected is None
        or plan.get("terminal") != terminal_ref["path"]
        or terminal.get("version") != 1
        or terminal.get("kind") != "verified-common-context-native-batch"
        or hashes(terminal.get("inputs")) != expected
        or set(checkpoints) != set(expected_checkpoints)
        or integer(terminal.get("executed")) + integer(terminal.get("adopted")) != len(parts) * 4
    ):
        raise ValueError("Original batch terminal/coverage differs")
    same(selected.get("operation"), operation_ref)
    operation_inputs = reference_inventory(operation.get("inputs"))
    if (
        operation.get("version") != 1
        or operation.get("kind") != "bounded-native-part-followup"
        or operation.get("semantic") != semantic["path"]
        or operation_inputs.get(semantic["path"]) != semantic["sha256"]
        or any(expected.get(path) != sha for path, sha in operation_inputs.items())
    ):
        raise ValueError("Original batch operation inputs differ")
    refs = rows(evidence.get("checkpoints"))
    if len(refs) != 4:
        raise ValueError("Original batch step inventory incomplete")
    outputs: dict[str, str] = {}
    native = reference(selected.get("native"))
    for step, raw in zip(
        ("frames", "complete", "publication-config", "publish"), refs, strict=True
    ):
        ref = reference(raw)
        if (
            ref["path"]
            != str(Path(string(plan.get("checkpoints"))) / f"part-{part:04d}-{step}.json")
            or checkpoints.get(ref["path"]) != ref["sha256"]
        ):
            raise ValueError("Original batch step pin differs")
        record = reader.read(ref)
        step_outputs = reference_inventory(record.get("outputs"))
        if (
            {key: value for key, value in record.items() if key != "outputs"}
            != {
                "version": 1,
                "kind": "pending-common-context-native-step",
                "common": plan.get("common"),
                "part": part,
                "operation": operation_ref,
                "nativeSHA256": native["sha256"],
                "step": step,
                "qualityApproved": False,
            }
            or len(step_outputs) != (3 if step == "publish" else 1)
            or set(outputs) & set(step_outputs)
        ):
            raise ValueError("Original batch step identity/output differs")
        outputs.update(step_outputs)
    config_path = string(operation.get("config"))
    if config_path not in operation_inputs:
        raise ValueError("Original batch config pin absent")
    config = reader.read({"path": config_path, "sha256": operation_inputs[config_path]})
    assembly_ref = reference(config.get("assembly"))
    if (
        integer(config.get("part")) != part
        or operation_inputs.get(assembly_ref["path"]) != assembly_ref["sha256"]
    ):
        raise ValueError("Original batch assembly/part differs")
    return reader.read(assembly_ref), operation_inputs, outputs


def original_native_parts(config: dict[str, object], reader: MetadataReader) -> dict[str, object]:
    """Bind pre-provenance probes and explicit publication/reuse transformations."""
    partition_ref = reference(config.get("partition"))
    partition = reader.read(partition_ref)
    same(partition.get("source"), config.get("semanticXML"))
    if partition.get("version") != 1 or partition.get("globalCaptionCuesEqual") is not True:
        raise ValueError("Original native partition differs")
    parts, items = rows(partition.get("parts")), rows(config.get("parts"))
    if len(parts) != len(items) or len(parts) > 64:
        raise ValueError("Incomplete original native inventory")
    fps, total = integer(config.get("fps")), integer(config.get("frames"))
    if fps <= 0 or total <= 0:
        raise ValueError("Invalid original native clock")
    endpoint = Fraction(0)
    movies: set[str] = set()
    for number, (part, item) in enumerate(zip(parts, items, strict=True), 1):
        start = Fraction(string(part.get("globalStart")).removesuffix("s"))
        duration = Fraction(string(part.get("duration")).removesuffix("s"))
        count = duration * fps
        if start != endpoint or duration <= 0 or count.denominator != 1:
            raise ValueError("Noncontiguous original native clock")
        publication = reader.read(item.get("publication"))
        completion = published_completion(item.get("publication"), reader)
        if completion["part"] != number:
            raise ValueError("Original native completion part differs")
        original_ref = reference(item.get("originalProof"))
        original = reader.read(original_ref)
        same(reader.read(publication.get("completion")).get("frameProof"), original_ref)
        if (
            "probeProvenance" in original and "originalBatch" not in item
        ) or "crossRevisionReuse" in original:
            raise ValueError("Expected original pre-provenance direct proof")
        if "originalBatch" in item:
            if "originalFollowup" in item or "probeStage" in item:
                raise ValueError("Ambiguous original native batch")
            assembly, inputs, terminal_outputs = original_batch_part(
                mapping(item["originalBatch"]),
                part=number,
                semantic=reference(config.get("semanticXML")),
                reader=reader,
            )
            for bound_ref in (
                original_ref,
                reference(item.get("publication")),
                reference(publication.get("completion")),
                reference(publication.get("frameProof")),
                reference(publication.get("movie")),
            ):
                if terminal_outputs.get(bound_ref["path"]) != bound_ref["sha256"]:
                    raise ValueError("Original batch publication output differs")
            batch_probe_provenance(
                mapping(item["originalBatch"]), original=original, inputs=inputs, reader=reader
            )
        elif "originalFollowup" in item:
            if "probeStage" in item:
                raise ValueError("Ambiguous original native producer")
            assembly, inputs = original_integrated_part(
                mapping(item["originalFollowup"]),
                output=original_ref,
                part=number,
                semantic=reference(config.get("semanticXML")),
                reader=reader,
            )
            terminal = reader.read(mapping(item["originalFollowup"]).get("terminal"))
            terminal_outputs = reference_inventory(terminal.get("outputs"))
            for bound_ref in (
                reference(item.get("publication")),
                reference(publication.get("completion")),
                reference(publication.get("frameProof")),
                reference(publication.get("movie")),
            ):
                if terminal_outputs.get(bound_ref["path"]) != bound_ref["sha256"]:
                    raise ValueError("Original integrated publication output differs")
        else:
            assembly, inputs = original_part_stage(
                mapping(item.get("probeStage")),
                output=original_ref,
                part=number,
                reuse=False,
                reader=reader,
                semantic=reference(config.get("semanticXML")),
            )
        same(assembly.get("partition"), original.get("partition"))
        original_partition = reference(original.get("partition"))
        if inputs.get(original_partition["path"]) != original_partition["sha256"]:
            raise ValueError("Original probe partition was not bound")
        original_movie = reference(original.get("movie"))
        if inputs.get(original_movie["path"]) != original_movie["sha256"]:
            raise ValueError("Original probe movie was not bound")
        same(original_movie, publication.get("source"))
        published = reader.read(publication.get("frameProof"))
        expected = dict(original)
        expected.update(
            movie=publication.get("movie"),
            sourceFrameProof=original_ref,
            publicationIntent=publication.get("intent"),
        )
        if json.dumps(published, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("Original publication transformed frame record differs")
        selected = reader.read(item.get("selectedProof"))
        if "reuseStage" in item:
            original_part_reuse(
                item,
                selected=selected,
                published=published,
                publication=publication,
                partition=partition_ref,
                part=number,
                reader=reader,
            )
        else:
            same(item.get("selectedProof"), publication.get("frameProof"))
            if json.dumps(selected, sort_keys=True) != json.dumps(published, sort_keys=True):
                raise ValueError("Selected direct published proof differs")
        movie = reference(publication.get("movie"))
        if movie["path"] in movies:
            raise ValueError("Duplicate original native movie")
        movies.add(movie["path"])
        native_part_clock(
            selected,
            partition=dict(partition_ref),
            movie=dict(movie),
            part=number,
            frames=count.numerator,
            fps=fps,
            width=integer(config.get("width")),
            height=integer(config.get("height")),
        )
        endpoint += duration
    if endpoint != Fraction(total, fps) or endpoint != Fraction(
        string(partition.get("duration")).removesuffix("s")
    ):
        raise ValueError("Original native whole endpoint differs")
    result = recorded(
        {
            "parts": len(parts),
            "frames": total,
            "scope": "original probe checkpoints and publication/reuse frame records",
            "historicalProbeToolExecutionPin": None,
        }
    )
    result["pending"] = [
        "historical probe tool execution pin",
        "current media crypto",
        "caption visual quality",
    ]
    return result


def batch_probe_provenance(
    batch: dict[str, object],
    *,
    original: dict[str, object],
    inputs: dict[str, str],
    reader: MetadataReader,
) -> None:
    if batch.get("probeContract") == "original-unpinned-frame-probe-v1":
        operation = reader.read(batch.get("operation"))
        path = string(operation.get("config"))
        if path not in inputs or "probeProvenance" in original:
            raise ValueError("Original unpinned probe contract differs")
        configured = reader.read({"path": path, "sha256": inputs[path]})
        if "probeDecoderThreads" in configured:
            raise ValueError("Pinned probe cannot be relabeled as an older probe")
        return
    if "probeContract" in batch:
        raise ValueError("Unsupported original probe contract")
    provenance = mapping(original.get("probeProvenance"))
    tool = reference(provenance.get("executable"))
    if (
        provenance.get("version") != 1
        or provenance.get("decoderThreads") != 0
        or inputs.get(tool["path"]) != tool["sha256"]
    ):
        raise ValueError("Original batch probe tool pin differs")


def native_parts(
    config: dict[str, object], *, reader: MetadataReader | None = None
) -> dict[str, object]:
    read = (reader if reader is not None else MetadataReader()).read
    "Aggregate original publication/frame records without touching MOVs."
    partition_ref = mapping(config.get("partition"))
    partition = read(partition_ref)
    same(partition.get("source"), config.get("semanticXML"))
    if (
        integer(partition.get("version")) != 1
        or partition.get("globalCaptionCuesEqual") is not True
    ):
        raise ValueError("Partition caption equivalence not recorded")
    parts = rows(partition.get("parts"))
    publications = rows(config.get("publications"))
    if len(parts) != len(publications) or len(parts) > 64:
        raise ValueError("Incomplete native part inventory")
    fps = integer(config.get("fps"))
    total = integer(config.get("frames"))
    if fps <= 0 or total <= 0:
        raise ValueError("Invalid whole frame clock")
    endpoint = Fraction(0)
    seen: set[str] = set()
    for number, (part, publication_ref) in enumerate(zip(parts, publications, strict=True), 1):
        start = Fraction(string(part.get("globalStart")).removesuffix("s"))
        duration = Fraction(string(part.get("duration")).removesuffix("s"))
        count = duration * fps
        if start != endpoint or duration <= 0 or count.denominator != 1:
            raise ValueError("Noncontiguous or non-frame partition")
        publication = read(publication_ref)
        if (
            integer(publication.get("version")) != 1
            or publication.get("kind") != "published-native-part"
        ):
            raise ValueError("Unsupported native publication")
        movie = reference(publication.get("movie"))
        if movie["path"] in seen:
            raise ValueError("Duplicate native movie")
        seen.add(movie["path"])
        native_part(
            read(publication.get("frameProof")),
            partition=partition_ref,
            movie=dict(movie),
            part=number,
            frames=count.numerator,
            fps=fps,
            width=integer(config.get("width")),
            height=integer(config.get("height")),
            tool=mapping(config.get("tool")),
        )
        endpoint += duration
    if endpoint != Fraction(total, fps) or endpoint != Fraction(
        string(partition.get("duration")).removesuffix("s")
    ):
        raise ValueError("Whole partition endpoint differs")
    result = recorded(
        {
            "parts": len(parts),
            "frames": total,
            "scope": "retained partition/publication/video frame observations",
        }
    )
    result["pending"] = [
        "native dispatch and full AV completion chain",
        "publication/retirement domain reconstruction",
        "caption visual quality",
    ]
    return result


def bound(parent: dict[str, object], ref: object) -> None:
    item = reference(ref)
    if hashes(parent.get("parentSHA256")).get(item["path"]) != item["sha256"]:
        raise ValueError("Original parent pin differs or is missing")


def recognition(
    inputs: dict[str, object],
    complete: dict[str, object],
    parent: dict[str, object],
    *,
    input_ref: object,
    complete_ref: object,
    expected_inputs: dict[str, str],
) -> int:
    bound(parent, input_ref)
    bound(parent, complete_ref)
    if (
        integer(inputs.get("version")) != 1
        or inputs.get("clock") != "input-pcm-not-projected"
        or inputs.get("device") != "cpu"
        or (inputs.get("dtypeContract") != "provider-default-fp16-unchanged")
    ):
        raise ValueError("Unsupported original ASR runtime/clock")
    if hashes(inputs.get("inputs")) != hashes(expected_inputs):
        raise ValueError("Original source/model/runtime pins differ")
    return recognition_windows(inputs, complete, parent, input_ref=input_ref)


def recognition_windows(
    inputs: dict[str, object],
    complete: dict[str, object],
    parent: dict[str, object],
    *,
    input_ref: object,
) -> int:
    """Validate the original batch window inventory, without inventing runtime fields."""
    samples = integer(inputs.get("samples"))
    window = integer(inputs.get("windowSamples"))
    stride = integer(inputs.get("strideSamples"))
    if samples <= 0 or not 0 < stride <= window or integer(inputs.get("sampleRate")) != 16000:
        raise ValueError("Invalid original window clock")
    count = max(0, (samples - window + stride - 1) // stride) + 1
    raw = complete.get("rawSHA256")
    if (
        count > 10000
        or integer(complete.get("windows")) != count
        or (not isinstance(raw, list))
        or (len(raw) != count)
    ):
        raise ValueError("Incomplete or duplicate window count")
    parent_pins = hashes(parent.get("parentSHA256"))
    directory = Path(reference(input_ref)["path"]).parent
    expected_paths = {str(directory / f"{i * stride:012d}.raw.json") for i in range(count)}
    actual_paths = {
        path
        for path in parent_pins
        if Path(path).parent == directory and path.endswith(".raw.json")
    }
    if actual_paths != expected_paths:
        raise ValueError("Missing or extra original windows")
    for number, checksum in enumerate(raw):
        path = str(directory / f"{number * stride:012d}.raw.json")
        if parent_pins[path] != digest(checksum):
            raise ValueError("Raw window order or digest differs")
    return count


def asr_evidence(
    config: dict[str, object], *, reader: MetadataReader | None = None
) -> dict[str, object]:
    read = (reader if reader is not None else MetadataReader()).read
    if integer(config.get("version")) != 1:
        raise ValueError("Unsupported descriptor")
    if config.get("contract") == "legacy-native-selection-v1":
        return legacy_asr_evidence(config, read=read)
    if config.get("contract") == "original-native-context-chain-v1":
        return native_context_asr_evidence(config, read=read)
    if config.get("contract") == "original-native-owned-window-plan-v1":
        return owned_window_asr_evidence(config, read=read)
    if "contract" in config:
        raise ValueError("Unsupported original ASR contract")
    model = hashes(config.get("modelPins"))
    runtime = hashes(config.get("runtimePins"))
    if config.get("modelRepository") != "mlx-community/whisper-large-v3-mlx":
        raise ValueError("Expected explicitly requested non-turbo large-v3")
    if {Path(path).name for path in model} != {"config.json", "weights.npz"}:
        raise ValueError("Unsupported original model files")
    if len({Path(path).parent for path in model}) != 1:
        raise ValueError("Mixed model revisions")
    if len(model) != 2 or len(runtime) != 2 or set(model) & set(runtime):
        raise ValueError("Expected exact config/weights and two original runtime pins")
    stages = config.get("recognitions")
    if not isinstance(stages, list) or not 1 <= len(stages) <= 32:
        raise ValueError("Expected bounded original recognitions")
    parent_ref = config.get("refinedLabels")
    parent = read(parent_ref)
    if parent.get("clock") != "input-pcm-not-projected":
        raise ValueError("Refinement already projected")
    refinement = mapping(parent.get("refinement"))
    if (
        integer(refinement.get("clockOffsetApplications")) != 1
        or integer(refinement.get("sourceProjectionApplications")) != 0
    ):
        raise ValueError("Refinement clock transform differs")
    counts: list[int] = []
    seen: set[str] = set()
    for raw_stage in stages:
        stage = mapping(raw_stage)
        input_ref, complete_ref = (stage.get("inputs"), stage.get("complete"))
        path = reference(input_ref)["path"]
        if path in seen:
            raise ValueError("Duplicate original recognition")
        seen.add(path)
        source = reference(stage.get("source"))
        if source["path"] in model or source["path"] in runtime:
            raise ValueError("Source/model identity collision")
        counts.append(
            recognition(
                read(input_ref),
                read(complete_ref),
                parent,
                input_ref=input_ref,
                complete_ref=complete_ref,
                expected_inputs=model | runtime | {source["path"]: source["sha256"]},
            )
        )
    source_labels = read(config.get("sourceLabels"))
    if source_labels.get("clock") != "source-presentation":
        raise ValueError("Expected source-projected labels")
    bound(source_labels, parent_ref)
    projected = read(config.get("projectedTokens"))
    if projected.get("clock") != "output":
        raise ValueError("Expected edited output tokens")
    same(projected.get("labels"), config.get("sourceLabels"))
    same(projected.get("cuts"), config.get("cuts"))
    result = recorded(
        {
            "originalWindows": counts,
            "historicalModelPinsConfirmed": True,
            "modelPins": model,
            "runtimePins": runtime,
            "currentModelAvailabilityVerified": False,
            "scope": "original recognition completion and referenced clock lineage",
        }
    )
    result["pending"] = [
        "current media/model crypto",
        "all token numerical projection validation",
        "speech semantics/accuracy",
        "original executor checkpoint reconstruction",
    ]
    return result


def original_batch_asr(
    config: dict[str, object], *, read: Callable[[object], dict[str, object]]
) -> tuple[dict[str, str], dict[str, str], dict[str, object], int]:
    """Retain the exact pre-device-field producer schema and original model pins."""
    model, runtime = hashes(config.get("modelPins")), hashes(config.get("runtimePins"))
    if (
        config.get("modelRepository") != "mlx-community/whisper-large-v3-mlx"
        or len(model) != 2
        or {Path(path).name for path in model} != {"config.json", "weights.npz"}
        or len({Path(path).parent for path in model}) != 1
        or len(runtime) != 1
        or {Path(path).name for path in runtime} != {"batch_asr.py"}
        or set(model) & set(runtime)
    ):
        raise ValueError("Unsupported original model/runtime identity")
    source = reference(config.get("source"))
    inputs, complete = read(config.get("inputs")), read(config.get("complete"))
    assembled = read(config.get("assembledLabels"))
    if (
        set(inputs)
        != {"version", "clock", "inputs", "sampleRate", "samples", "windowSamples", "strideSamples"}
        or integer(inputs.get("version")) != 1
        or inputs.get("clock") != "input-pcm-not-projected"
        or assembled.get("clock") != "input-pcm-not-projected"
        or source["path"] in model | runtime
        or hashes(inputs.get("inputs")) != model | runtime | {source["path"]: source["sha256"]}
    ):
        raise ValueError("Original ASR inputs or clock differ")
    bound(assembled, config.get("inputs"))
    bound(assembled, config.get("complete"))
    count = recognition_windows(inputs, complete, assembled, input_ref=config.get("inputs"))
    return model, runtime, inputs, count


def original_parent_pins(record: dict[str, object]) -> dict[str, str]:
    """Index original absolute/relative names without changing their recorded bytes."""
    result: dict[str, str] = {}
    raw = mapping(record.get("parentSHA256"))
    if not 1 <= len(raw) <= 4096:
        raise ValueError("Original ancestry budget exceeded")
    for name, value in raw.items():
        path = Path(name)
        if not name or str(path) != name or ".." in path.parts or name.startswith("//"):
            raise ValueError("Unsupported original lexical parent")
        absolute = str(path if path.is_absolute() else Path.cwd() / path)
        if absolute in result:
            raise ValueError("Duplicate original lexical parent identity")
        result[absolute] = digest(value)
    return result


def original_source_labels(
    record: dict[str, object], duration: Fraction, *, expected_clock: str | None = None
) -> int:
    """Validate source-clock bounds while preserving recorded overlaps and zero spans."""
    if record.get("clock") not in (None, "input-pcm-not-projected", expected_clock):
        raise ValueError("Original label clock was projected again")
    if "sourceDuration" in record and Fraction(str(record["sourceDuration"])) != duration:
        raise ValueError("Original label source duration differs")
    labels = rows(record.get("labels"))
    for row in labels:
        string(row.get("text"))
        start, end = Fraction(str(row.get("start"))), Fraction(str(row.get("end")))
        if not 0 <= start <= end <= duration:
            raise ValueError("Original label lies outside its source clock")
    return len(labels)


def owned_window_asr_evidence(
    config: dict[str, object], *, read: Callable[[object], dict[str, object]]
) -> dict[str, object]:
    """Read the original per-provider window lineage, without inventing batch completion."""
    owned, base = read(config.get("sourceLabels")), read(config.get("baseLabels"))
    parents = original_parent_pins(owned)
    base_ref = reference(config.get("baseLabels"))
    if parents.get(base_ref["path"]) != base_ref["sha256"]:
        raise ValueError("Owned labels lack their original base")
    inventory = read(config.get("windowInventory"))
    same(inventory.get("sourceLabels"), config.get("sourceLabels"))
    same(inventory.get("providerBinding"), config.get("providerBinding"))
    provider = read(config.get("providerBinding"))
    if owned.get("clock") != "native-source-presentation-already-projected" or provider.get(
        "clock"
    ) != owned.get("clock"):
        raise ValueError("Original owned source clock differs")
    model = hashes(config.get("modelPins"))
    if (
        config.get("modelRepository") != "mlx-community/whisper-large-v3-mlx"
        or len(model) != 2
        or {Path(path).name for path in model} != {"config.json", "weights.npz"}
        or len({Path(path).parent for path in model}) != 1
    ):
        raise ValueError("Original owned model policy differs")
    model_names = {Path(path).name: sha for path, sha in model.items()}
    plan_refs = [reference(item) for item in rows(config.get("plans"))]
    if not 1 <= len(plan_refs) <= 16 or len({ref["path"] for ref in plan_refs}) != len(plan_refs):
        raise ValueError("Original provider plans duplicate or exceed budget")
    plans = {ref["path"]: read(ref) for ref in plan_refs}
    plan_hashes = {ref["path"]: ref["sha256"] for ref in plan_refs}
    indexed = rows(inventory.get("plans"))
    providers = rows(provider.get("providers"))
    if (
        len(indexed) != len(plans)
        or len(providers) != len(plans)
        or {reference(row.get("plan"))["path"] for row in indexed} != set(plans)
    ):
        raise ValueError("Original provider inventory coverage differs")
    total = 0
    model_bound: set[str] = set()
    sample_counts: set[int] = set()
    for row, actual_provider in zip(indexed, providers, strict=True):
        ref = reference(row.get("plan"))
        plan = plans[ref["path"]]
        if (
            plan_hashes.get(ref["path"]) != ref["sha256"]
            or parents.get(ref["path"]) != ref["sha256"]
            or row.get("provider") != actual_provider
        ):
            raise ValueError("Original provider plan binding differs")
        if (
            plan.get("clock") != owned.get("clock")
            or integer(plan.get("sampleRate")) != 16000
            or integer(plan.get("samples")) <= 0
        ):
            raise ValueError("Original provider sample clock differs")
        sample_counts.add(integer(plan.get("samples")))
        same(row.get("source"), {"path": plan.get("source"), "sha256": plan.get("sourceSHA256")})
        if "modelConfigSHA256" in plan:
            if (
                plan.get("modelConfigSHA256") != model_names["config.json"]
                or plan.get("modelWeightsSHA256") != model_names["weights.npz"]
            ):
                raise ValueError("Original provider model pins differ")
            model_bound.add(ref["sha256"])
        planned, windows = rows(plan.get("windows")), rows(row.get("windows"))
        if (
            len(planned) != len(windows)
            or len(windows) != integer(actual_provider.get("windows"))
            or len(windows) > 2048
        ):
            raise ValueError("Original provider window coverage differs")
        if Path(ref["path"]).parent.name != actual_provider.get("input"):
            raise ValueError("Original provider input directory differs")
        for number, (expected, window) in enumerate(zip(planned, windows, strict=True)):
            owned_window_clock(
                window,
                expected=expected,
                number=number,
                provider=actual_provider,
                plan_ref=ref,
                parents=parents,
                samples=integer(plan.get("samples")),
                read=read,
            )
        total += len(windows)
    if (
        len(sample_counts) != 1
        or total != integer(inventory.get("totalWindows"))
        or integer(inventory.get("totalRawClockReferences")) != 2 * total
    ):
        raise ValueError("Original provider source duration or count differs")
    for ref in plan_refs:
        plan = plans[ref["path"]]
        if ref["sha256"] in model_bound:
            continue
        parent = digest(plan.get("parentPlanSHA256"))
        seen = {ref["sha256"]}
        while parent not in model_bound:
            if parent in seen:
                raise ValueError("Cyclic original plan model ancestry")
            seen.add(parent)
            candidates = [plans[path] for path, sha in plan_hashes.items() if sha == parent]
            if len(candidates) != 1:
                raise ValueError("Original donor model ancestry absent")
            parent = digest(candidates[0].get("parentPlanSHA256"))
    duration = Fraction(next(iter(sample_counts)), 16000)
    original_source_labels(
        base, duration, expected_clock="native-source-presentation-already-projected"
    )
    labels = original_source_labels(
        owned, duration, expected_clock="native-source-presentation-already-projected"
    )
    owned_context_selections(owned, indexed=indexed, duration=duration)
    repair = read(config.get("refinementConfig"))
    plan_ref = reference(repair.get("clockEvidence"))
    if plan_hashes.get(plan_ref["path"]) != plan_ref["sha256"]:
        raise ValueError("Refinement source plan differs")
    refinement_config = dict(config)
    refinement_config["normalization"] = plan_ref
    refinement_config["source"] = {
        "path": plans[plan_ref["path"]].get("source"),
        "sha256": plans[plan_ref["path"]].get("sourceSHA256"),
    }
    refinement = context_refinement_evidence(refinement_config, model=model, read=read)
    result = recorded(
        {
            "originalWindows": total,
            "originalProviders": len(plans),
            "sourceLabelCount": labels,
            "historicalModelPinsConfirmed": True,
            "modelPins": model,
            "historicalFullRuntimePins": None,
            "originalBatchCompletion": None,
            "secondSourceProjectionApplied": False,
            "refinement": refinement,
        }
    )
    result["pending"] = [
        "original full provider execution/runtime attestation",
        "all native phrase ownership numerical replay",
        "speech semantics and word-level timing",
        "current model/media crypto",
    ]
    return result


def owned_context_selections(
    record: dict[str, object], *, indexed: list[dict[str, object]], duration: Fraction
) -> None:
    selections = record.get("nativeContextSelections", [])
    if not isinstance(selections, list) or len(selections) > 10000:
        raise ValueError("Original provider selections exceed budget")
    for value in selections:
        selection = mapping(value)
        providers = [
            row
            for row in indexed
            if mapping(row.get("provider")).get("raw") == selection.get("provider")
        ]
        if len(providers) != 1:
            raise ValueError("Original selected provider is not bound")
        windows = rows(providers[0].get("windows"))
        number = integer(selection.get("window"))
        if not 0 <= number < len(windows) or not 0 <= integer(selection.get("first")) < integer(
            selection.get("end")
        ):
            raise ValueError("Original selected provider window differs")
        original_source_labels({"labels": selection.get("before")}, duration)
        original_source_labels({"labels": selection.get("after")}, duration)
        window = mapping(windows[number].get("plannedWindow"))
        left, right = (
            integer(window.get("startSample")) / 16000,
            integer(window.get("endSample")) / 16000,
        )
        if any(
            float(Fraction(str(row.get("start")))) < left - ORIGINAL_TOKEN_EPSILON
            or float(Fraction(str(row.get("end")))) > right + ORIGINAL_TOKEN_EPSILON
            for row in rows(selection.get("after"))
        ):
            raise ValueError("Original provider selection exceeds its recognition window")


def owned_window_clock(
    window: dict[str, object],
    *,
    expected: dict[str, object],
    number: int,
    provider: dict[str, object],
    plan_ref: dict[str, str],
    parents: dict[str, str],
    samples: int,
    read: Callable[[object], dict[str, object]],
) -> None:
    raw, clock = reference(window.get("raw")), reference(window.get("clock"))
    directory = Path(plan_ref["path"]).parent.parent / string(provider.get("raw"))
    if (
        integer(window.get("index")) != number
        or window.get("plannedWindow") != expected
        or Path(raw["path"]).parent != directory
        or Path(clock["path"]).parent != directory
        or Path(raw["path"]).name != f"{number:04d}.raw.json"
        or Path(clock["path"]).name != f"{number:04d}.clock.json"
        or parents.get(raw["path"]) != raw["sha256"]
        or parents.get(clock["path"]) != clock["sha256"]
    ):
        raise ValueError("Original raw/window provenance differs")
    observed = read(clock)
    projection = mapping(window.get("observedClock"))
    expected_keys = {
        "sourceStartSample",
        "sourceEndSample",
        "sampleRate",
        "device",
        "nativeLocalClocks",
        "humanReviewed",
        "textAutomaticallyReplaced",
        "emptyRecognitionIsNotSilence",
    }
    if (
        set(projection) - expected_keys
        or set(observed) - (expected_keys | {"elapsedSeconds", "seed"})
        or any(observed.get(key) != value for key, value in projection.items())
    ):
        raise ValueError("Original clock metadata projection differs")
    if "seed" in observed:
        integer(observed["seed"])
    if "elapsedSeconds" in observed and Fraction(str(observed["elapsedSeconds"])) < 0:
        raise ValueError("Original recognition elapsed time differs")
    start, end = integer(expected.get("startSample")), integer(expected.get("endSample"))
    digest(expected.get("inputSHA256"))
    if (
        not 0 <= start < end <= samples
        or observed.get("sourceStartSample") != start
        or observed.get("sourceEndSample") != end
        or observed.get("sampleRate") != 16000
        or observed.get("nativeLocalClocks") is not True
        or observed.get("emptyRecognitionIsNotSilence") is not True
    ):
        raise ValueError("Original recognition local clock differs")


def native_context_asr_evidence(
    config: dict[str, object], *, read: Callable[[object], dict[str, object]]
) -> dict[str, object]:
    """Bind retained batch completion and a pinned, source-clock native label ancestry."""
    model, runtime, inputs, count = original_batch_asr(config, read=read)
    normalization = read(config.get("normalization"))
    if (
        normalization.get("clock") != "native-source-presentation-already-projected"
        or normalization.get("noSecondClockProjection") is not True
    ):
        raise ValueError("Original context source projection differs")
    refs = [
        reference(config.get("sourceLabels")),
        *[reference(item) for item in rows(config.get("nativeContextParents"))],
        reference(config.get("assembledLabels")),
    ]
    by_path = {ref["path"]: ref for ref in refs}
    if len(by_path) != len(refs) or not 3 <= len(refs) <= 32:
        raise ValueError("Duplicate or oversized native ancestry")
    records = {path: read(ref) for path, ref in by_path.items()}
    positions = {ref["path"]: number for number, ref in enumerate(refs)}
    duration = Fraction(integer(inputs.get("samples")), 16000)
    root = refs[0]["path"]
    reached: set[str] = {root}
    pending = [root]
    label_count = 0
    while pending:
        path = pending.pop()
        record = records[path]
        if "labels" not in record:
            native_point_collection(record, source=reference(config.get("source")))
            continue
        current_count = original_source_labels(record, duration)
        if path == root:
            label_count = current_count
        parents = original_parent_pins(record)
        for parent_path, ref in by_path.items():
            if parent_path not in parents:
                continue
            if parents[parent_path] != ref["sha256"]:
                raise ValueError("Native context parent digest differs")
            if positions[parent_path] <= positions[path]:
                raise ValueError("Cyclic or reordered native ancestry")
            if parent_path not in reached:
                reached.add(parent_path)
                pending.append(parent_path)
        context_selections(record, parents=parents)
    if reached != set(by_path):
        raise ValueError("Native context ancestry is disconnected")
    refinement = context_refinement_evidence(config, model=model, read=read)
    result = recorded(
        {
            "originalWindows": count,
            "historicalModelPinsConfirmed": True,
            "modelPins": model,
            "runtimePins": runtime,
            "sourceLabelCount": label_count,
            "originalLabelAncestryRecords": len(refs),
            "nativeSelectionModelExecutionVerified": False,
            "secondSourceProjectionApplied": False,
            "refinement": refinement,
        }
    )
    result["pending"] = [
        "native selection model/runtime execution pins",
        "all selection numerical replay",
        "current model/media crypto",
        "all prepared token numerical projection",
        "speech semantics/accuracy",
    ]
    return result


def selected_original_token(token: SpokenToken, start: Fraction, end: Fraction) -> bool:
    if (
        token.end <= float(start) + ORIGINAL_TOKEN_EPSILON
        or token.start >= float(end) - ORIGINAL_TOKEN_EPSILON
    ):
        return False
    if (
        token.start < float(start) - ORIGINAL_TOKEN_EPSILON
        or token.end > float(end) + ORIGINAL_TOKEN_EPSILON
    ):
        raise ValueError("Original token crosses replacement boundary")
    return True


def context_refinement_evidence(
    config: dict[str, object], *, model: dict[str, str], read: Callable[[object], dict[str, object]]
) -> dict[str, object]:
    """Verify a retained single local-ASR overlay without opening PCM or running ASR."""
    repair = read(config.get("refinementConfig"))
    receipt = read(config.get("refinementReceipt"))
    if (
        repair.get("version") not in {1, 2}
        or receipt.get("version") != 1
        or receipt.get("configuration") != repair
    ):
        raise ValueError("Original local refinement configuration differs")
    bindings = hashes(receipt.get("inputs"))
    for value in (
        config.get("refinementConfig"),
        *[
            repair.get(key)
            for key in (
                "window",
                "audio",
                "inputs",
                "recognition",
                "clockEvidence",
                "cuts",
                "manifest",
            )
        ],
    ):
        ref = reference(value)
        if bindings.get(ref["path"]) != ref["sha256"]:
            raise ValueError("Original local refinement input binding differs")
    if repair.get("version") == 2:
        for key in ("recognitionClock", "recognitionComplete"):
            ref = reference(repair.get(key))
            if bindings.get(ref["path"]) != ref["sha256"]:
                raise ValueError("Original batch refinement input binding differs")
    same(repair.get("clockEvidence"), config.get("normalization"))
    same(repair.get("cuts"), config.get("cuts"))
    window, original_inputs = read(repair.get("window")), read(repair.get("inputs"))
    source, audio = reference(config.get("source")), reference(repair.get("audio"))
    same({"path": window.get("source"), "sha256": window.get("sourceSHA256")}, source)
    start, end = integer(window.get("startSample")), integer(window.get("endSample"))
    if repair.get("version") == 2:
        batch_refinement_inputs(
            repair, receipt=receipt, model=model, samples=end - start, read=read
        )
    else:
        model_root = string(original_inputs.get("model"))
        if (
            original_inputs.get("version") != 1
            or original_inputs.get("clock") != "continuous-input-pcm"
            or integer(original_inputs.get("sample_rate")) != 16000
            or integer(original_inputs.get("samples")) != end - start
            or original_inputs.get("source") != audio["path"]
            or digest(original_inputs.get("source_sha256")) != audio["sha256"]
            or model
            != {
                str(Path(model_root) / "config.json"): digest(original_inputs.get("config_sha256")),
                str(Path(model_root) / "weights.npz"): digest(
                    original_inputs.get("weights_sha256")
                ),
            }
        ):
            raise ValueError("Original refinement model or PCM clock differs")
    if (
        window.get("version") != 1
        or window.get("clock") != "continuous-input-pcm"
        or window.get("sourceClockProjected") is not False
        or integer(window.get("sampleRate")) != 16000
        or not start < end
        or digest(window.get("windowSHA256")) != audio["sha256"]
        or digest(repair.get("sourceSHA256")) != source["sha256"]
    ):
        raise ValueError("Original refinement model or PCM clock differs")
    interval = mapping(repair.get("replacement"))
    first, last = integer(interval.get("startSample")), integer(interval.get("endSample"))
    if not start <= first < last <= end:
        raise ValueError("Original replacement exceeds PCM window")
    raw_ref = reference(repair.get("recognition"))
    groups = mlx_groups(read(raw_ref), duration=(end - start) / 16000)
    source_tokens = tuple(
        SpokenToken(
            group.text,
            group.start + start / 16000,
            group.end + start / 16000,
            f"refine:{raw_ref['sha256']}:mlx:"
            + ",".join(str(index) for index in group.source_segments),
        )
        for group in groups
    )
    chosen = tuple(
        token
        for token in source_tokens
        if selected_original_token(token, Fraction(first, 16000), Fraction(last, 16000))
    )
    if not chosen or [asdict(token) for token in chosen] != receipt.get("selectedSourceTokens"):
        raise ValueError("Original selected recognition groups differ")
    manifest = read(repair.get("manifest"))
    old_ref = reference(manifest.get("tokens"))
    if bindings.get(old_ref["path"]) != old_ref["sha256"]:
        raise ValueError("Original token input pin absent")
    old = read(old_ref)
    sources = read(config.get("tokenSources"))
    replacements = rows(sources.get("replacements"))
    if sources.get("version") != 1 or len(replacements) != 1:
        raise ValueError("Single refinement requires one explicit overlay")
    replacement = replacements[0]
    same(replacement.get("tokens"), receipt.get("tokens"))
    tokens = read(replacement.get("tokens"))
    adopted = read(replacement.get("adoptedManifest"))
    expected_manifest = dict(receipt if repair.get("version") == 2 else manifest)
    if repair.get("version") == 1:
        expected_manifest["tokens"] = reference(replacement.get("tokens"))
    if (
        adopted != expected_manifest
        or old.get("version") != 1
        or tokens.get("version") != 1
        or old.get("clock") != "output"
        or tokens.get("clock") != "output"
    ):
        raise ValueError("Original adopted manifest/token clock differs")
    index = read(config.get("segmentIndex"))
    rate = integer(index.get("rate"))
    segments = rows(index.get("segments"))
    number = integer(replacement.get("segment"))
    if index.get("version") != 1 or not 1 <= rate <= 120 or number >= len(segments):
        raise ValueError("Original prepared segment inventory differs")
    segment = segments[number]
    if (
        integer(segment.get("segment")) != number
        or integer(repair.get("segmentOffsetFrame")) != integer(segment.get("offsetFrame"))
        or integer(manifest.get("outputFrameRate")) != rate
        or integer(replacement.get("job")) not in integer_list(segment.get("jobs"))
    ):
        raise ValueError("Original replacement segment/job clock differs")
    cut_record = read(config.get("cuts"))
    cuts = tuple(
        CutSpan(
            integer(row.get("source_start_frame")),
            integer(row.get("source_end_frame")),
            integer(row.get("output_start_frame")),
        )
        for row in rows(cut_record.get("cuts"))
    )
    projected = project_tokens(chosen, cuts, fps=rate)
    if projected.boundary_conflicts or [
        asdict(token) for token in projected.omitted
    ] != receipt.get("omittedByCuts"):
        raise ValueError("Original refinement cut projection differs")
    removed = integer_list(receipt.get("removedTokenIndices"))
    old_rows = rows(old.get("tokens"))
    if len(set(removed)) != len(removed) or max(removed) >= len(old_rows):
        raise ValueError("Original removed token inventory differs")
    offset = integer(segment.get("offsetFrame")) / rate
    ranges = [
        (
            max(Fraction(first, 16000), Fraction(cut.source_start_frame, rate))
            + Fraction(cut.output_start_frame - cut.source_start_frame, rate)
            - Fraction(integer(segment.get("offsetFrame")), rate),
            min(Fraction(last, 16000), Fraction(cut.source_end_frame, rate))
            + Fraction(cut.output_start_frame - cut.source_start_frame, rate)
            - Fraction(integer(segment.get("offsetFrame")), rate),
        )
        for cut in cuts
        if max(Fraction(first, 16000), Fraction(cut.source_start_frame, rate))
        < min(Fraction(last, 16000), Fraction(cut.source_end_frame, rate))
    ]
    expected_removed = [
        number
        for number, row in enumerate(old_rows)
        if any(
            selected_original_token(
                SpokenToken(
                    string(row.get("text")),
                    float(Fraction(str(row.get("start")))),
                    float(Fraction(str(row.get("end")))),
                    string(row.get("utterance")),
                ),
                left,
                right,
            )
            for left, right in ranges
        )
    ]
    if (
        removed != expected_removed
        or not projected.tokens
        or any(
            token.start - offset < -ORIGINAL_TOKEN_EPSILON
            or token.end - offset > integer(segment.get("frames")) / rate + ORIGINAL_TOKEN_EPSILON
            for token in projected.tokens
        )
    ):
        raise ValueError("Original refinement changes another job or token interval")
    new_rows = [
        asdict(SpokenToken(token.text, token.start - offset, token.end - offset, token.utterance))
        for token in projected.tokens
    ]
    expected = sorted(
        [*[row for number, row in enumerate(old_rows) if number not in removed], *new_rows],
        key=lambda row: Fraction(str(row["start"])),
    )
    if tokens.get("tokens") != expected:
        raise ValueError("Original refined output changed outside selected tokens")
    return {
        "modelPinsConfirmed": True,
        "sourceOffsetApplications": 1,
        "editedProjectionApplications": 1,
        "segment": number,
        "job": replacement["job"],
        "qualityApproved": False,
    }


def batch_refinement_inputs(
    repair: dict[str, object],
    *,
    receipt: dict[str, object],
    model: dict[str, str],
    samples: int,
    read: Callable[[object], dict[str, object]],
) -> None:
    """Validate the original v2 one-window batch contract, without opening PCM/model."""
    inputs, clock, complete = (
        read(repair.get(key)) for key in ("inputs", "recognitionClock", "recognitionComplete")
    )
    pins = hashes(inputs.get("inputs"))
    audio, raw = reference(repair.get("audio")), reference(repair.get("recognition"))
    source_plan = read(repair.get("clockEvidence"))
    window = read(repair.get("window"))
    runtime = {
        path: sha for path, sha in pins.items() if path not in model and path != audio["path"]
    }
    if (
        inputs.get("version") != 1
        or inputs.get("clock") != "input-pcm-not-projected"
        or inputs.get("device") != "cpu"
        or inputs.get("dtypeContract") != "provider-default-fp16-unchanged"
        or integer(inputs.get("samples")) != samples
        or integer(inputs.get("sampleRate")) != 16000
        or integer(inputs.get("windowSamples")) != 960000
        or integer(inputs.get("strideSamples")) != 480000
        or pins.get(audio["path"]) != audio["sha256"]
        or any(pins.get(path) != sha for path, sha in model.items())
        or len(runtime) != 2
        or {Path(path).name for path in runtime} != {"batch_asr.py", "asr_runtime.py"}
        or clock.get("startSample") != 0
        or clock.get("endSample") != samples
        or clock.get("sampleRate") != 16000
        or clock.get("emptyRecognitionIsNotSilence") is not True
        or clock.get("rawSHA256") != raw["sha256"]
        or complete.get("rawSHA256") != [raw["sha256"]]
        or complete.get("windows") != 1
        or complete.get("deliveryComplete") is not False
        or source_plan.get("clock") != "native-source-presentation-already-projected"
        or source_plan.get("source") != window.get("source")
        or source_plan.get("sourceSHA256") != window.get("sourceSHA256")
        or source_plan.get("sampleRate") != 16000
        or integer(source_plan.get("samples")) < integer(window.get("endSample"))
        or receipt.get("recognitionInputKind") != "batch-asr-single-window"
        or receipt.get("sourceClockProjection")
        != "identity-native-presentation-no-second-projection"
    ):
        raise ValueError("Original batch refinement model/clock/completion differs")


def integer_list(value: object) -> list[int]:
    if not isinstance(value, list) or not 1 <= len(value) <= 100000:
        raise ValueError("Expected bounded nonempty integer list")
    return [integer(item) for item in value]


def context_selections(record: dict[str, object], *, parents: dict[str, str]) -> None:
    """Validate recorded selection reference bindings without approving recognition."""
    if "nativeContextSelections" not in record:
        return
    selections = record["nativeContextSelections"]
    if not isinstance(selections, list) or len(selections) > 10000:
        raise ValueError("Invalid original selection inventory")
    for raw_selection in selections:
        item = mapping(raw_selection)
        raw_name = string(item.get("raw"))
        raw_path = str(Path(raw_name) if Path(raw_name).is_absolute() else Path.cwd() / raw_name)
        if ".." in Path(raw_name).parts or raw_path not in parents:
            raise ValueError("Native context selection raw pin absent")
        if "rawSHA256" in item and digest(item["rawSHA256"]) != parents[raw_path]:
            raise ValueError("Native context selection raw digest differs")
        if "clock" in item:
            name = string(item["clock"])
            path = str(Path(name) if Path(name).is_absolute() else Path.cwd() / name)
            if ".." in Path(name).parts or path not in parents:
                raise ValueError("Native context selection clock pin absent")
        selection = mapping(item.get("selection"))
        if integer(selection.get("original_start")) >= integer(selection.get("original_end")):
            raise ValueError("Native context selection interval differs")
        rows(selection.get("replacement"))
        rows(item.get("before"))


def native_point_collection(record: dict[str, object], *, source: dict[str, str]) -> None:
    if set(record) != {"deliveryComplete", "inputs", "speechAccuracyVerified", "windows"}:
        raise ValueError("Unsupported original native collection")
    if hashes(record.get("inputs")).get(source["path"]) != source["sha256"]:
        raise ValueError("Native collection source differs")
    windows = rows(record.get("windows"))
    for number, row in enumerate(windows):
        if (
            set(row) != {"window", "rawSHA256", "validated"}
            or integer(row.get("window")) != number
            or row.get("validated") is not True
        ):
            raise ValueError("Incomplete original native point inventory")
        digest(row.get("rawSHA256"))


def legacy_asr_evidence(
    config: dict[str, object], *, read: Callable[[object], dict[str, object]]
) -> dict[str, object]:
    """Bind original fixed-window ASR and sample-clock native selection records."""
    model, runtime, inputs, count = original_batch_asr(config, read=read)
    source = reference(config.get("source"))
    selected = read(config.get("sourceLabels"))
    plan, selection = read(config.get("nativePlan")), read(config.get("nativeComplete"))
    bound(selected, config.get("nativePlan"))
    bound(selected, config.get("nativeComplete"))
    selection_inputs = hashes(selection.get("inputs"))
    for key in ("inputs", "complete", "assembledLabels", "normalization"):
        ref = reference(config.get(key))
        if selection_inputs.get(ref["path"]) != ref["sha256"]:
            raise ValueError("Native selection original ancestry differs")
    normalization = read(config.get("normalization"))
    if (
        normalization.get("clock") != "native-source-presentation-already-projected"
        or normalization.get("noSecondClockProjection") is not True
        or "clock" in selected
        or Fraction(str(selected.get("sourceDuration")))
        != Fraction(integer(inputs.get("samples")), 16000)
    ):
        raise ValueError("Original source projection contract differs")
    same({"path": plan.get("source"), "sha256": plan.get("sourceSHA256")}, source)
    model_root = string(plan.get("model"))
    if model != {
        str(Path(model_root) / "config.json"): digest(plan.get("modelConfigSHA256")),
        str(Path(model_root) / "weights.npz"): digest(plan.get("modelWeightsSHA256")),
    }:
        raise ValueError("Native selection substituted model")
    windows, results = rows(plan.get("windows")), rows(selection.get("windows"))
    if not 1 <= len(windows) <= 10000 or len(results) != len(windows):
        raise ValueError("Native selection windows incomplete")
    seen: set[str] = set()
    for number, (window, result) in enumerate(zip(windows, results, strict=True)):
        if (
            result.get("status") != "complete"
            or {
                key: value
                for key, value in result.items()
                if key not in {"raw", "rawSHA256", "status"}
            }
            != window
            or integer(window.get("index")) != number
            or integer(window.get("sampleRate")) != 16000
            or not 0
            <= integer(window.get("startSample"))
            < integer(window.get("endSample"))
            <= integer(inputs.get("samples"))
        ):
            raise ValueError("Native window status or sample clock differs")
        raw = reference({"path": result.get("raw"), "sha256": result.get("rawSHA256")})
        if raw["path"] in seen:
            raise ValueError("Duplicate native window result")
        seen.add(raw["path"])
        bound(selected, raw)
        bound(selected, {"path": window.get("audio"), "sha256": window.get("audioSHA256")})
    projected = read(config.get("projectedTokens"))
    if projected.get("clock") != "output":
        raise ValueError("Expected edited token clock")
    same(projected.get("labels"), config.get("sourceLabels"))
    same(projected.get("cuts"), config.get("cuts"))
    result = recorded(
        {
            "originalWindows": count,
            "nativeSelectionWindows": len(windows),
            "historicalModelPinsConfirmed": True,
            "modelPins": model,
            "runtimePins": runtime,
            "secondSourceProjectionApplied": False,
        }
    )
    result["pending"] = [
        "current model/media crypto",
        "native executor runtime checkpoint",
        "speech semantics/accuracy",
        "all selected token numerical projection validation",
    ]
    return result


def chunk(
    producer: dict[str, object],
    candidate: dict[str, object],
    eligibility: dict[str, object],
    source: dict[str, object],
) -> tuple[int, int, Fraction, Fraction]:
    same({"path": producer.get("source"), "sha256": producer.get("sourceSHA256")}, source)
    inputs = hashes(candidate.get("inputs"))
    source_ref = reference(source)
    if inputs.get(source_ref["path"]) != source_ref["sha256"]:
        raise ValueError("Repair source is not the approved original")
    same(candidate.get("sourceEligibility"), producer.get("sourceEligibility"))
    eligibility_ref = reference(producer.get("sourceEligibility"))
    if inputs.get(eligibility_ref["path"]) != eligibility_ref["sha256"]:
        raise ValueError("Repair does not bind eligibility")
    if (
        integer(eligibility.get("version")) != 1
        or eligibility.get("kind") != "source-caption-eligibility"
        or digest(eligibility.get("sourceSHA256")) != source_ref["sha256"]
    ):
        raise ValueError("Eligibility source identity differs")
    clock = mapping(producer.get("clock"))
    raw_pts = clock.get("pts")
    if not isinstance(raw_pts, list) or not 1 <= len(raw_pts) <= 10000:
        raise ValueError("Invalid chunk frame inventory")
    pts = [integer(value) for value in raw_pts]
    base = Fraction(string(clock.get("timeBase")))
    last = integer(clock.get("lastDuration"))
    if base <= 0 or last <= 0 or any((a >= b for a, b in zip(pts, pts[1:], strict=False))):
        raise ValueError("Nonmonotonic source clock")
    if Fraction(string(candidate.get("sourceStart"))) != pts[0] * base:
        raise ValueError("Repair source start differs")
    if integer(mapping(candidate.get("render")).get("frames")) != len(pts):
        raise ValueError("Incomplete repair render")
    observations = rows(eligibility.get("frames"))
    by_pts = {integer(row.get("sourcePTS")): row for row in observations}
    if len(by_pts) != len(observations) or set(by_pts) != set(pts):
        raise ValueError("Eligibility omits or duplicates frame observations")
    active = candidate.get("activeSourceFrames")
    if not isinstance(active, list) or len(active) > len(pts):
        raise ValueError("Invalid active frame inventory")
    active_frames = [integer(value) for value in active]
    if len(set(active_frames)) != len(active_frames):
        raise ValueError("Duplicate repaired frames")
    for frame in active_frames:
        if (
            not 0 <= frame < len(pts)
            or by_pts[pts[frame]].get("classification") != "overlay-subtitle"
        ):
            raise ValueError("Repair applied to unresolved or non-subtitle frame")
    return (len(pts), len(active_frames), pts[0] * base, (pts[-1] + last) * base)


def approved_original_identity(config: dict[str, object]) -> tuple[int, ...] | None:
    """Check only an explicit original alias, never reinterpret historical producer pins."""
    if "approvedOriginal" not in config:
        return None
    if config.get("contract") != "recorded-local-text-detection-v1":
        raise ValueError("Approved original requires the typed local producer contract")
    original, approved = reference(config.get("source")), reference(config.get("approvedOriginal"))
    alias, canonical = Path(original["path"]), Path(approved["path"])
    if original["sha256"] != approved["sha256"] or original["path"] == approved["path"]:
        raise ValueError("Approved original is not the exact historical alias digest")
    if canonical.resolve(strict=True) != canonical or alias.resolve(strict=True) != canonical:
        raise ValueError("Approved original alias target differs")
    identity = canonical.stat()
    if not stat.S_ISREG(identity.st_mode) or identity.st_size <= 0:
        raise ValueError("Approved original is not a nonempty regular source")
    return file_identity(identity)


def cleanup_evidence(
    config: dict[str, object], *, reader: MetadataReader | None = None
) -> dict[str, object]:
    read = (reader if reader is not None else MetadataReader()).read
    if integer(config.get("version")) != 1:
        raise ValueError("Unsupported descriptor")
    approved_identity = approved_original_identity(config)
    manifest_ref = reference(config.get("manifest"))
    manifest = read(manifest_ref)
    assembly = read(config.get("assemblyReceipt"))
    bindings = hashes(assembly.get("inputs"))
    if bindings.get(manifest_ref["path"]) != manifest_ref["sha256"]:
        raise ValueError("Assembly does not bind original cleanup manifest")
    chunks = rows(manifest.get("chunks"))
    if len(chunks) > 1024:
        raise ValueError("Cleanup chunk budget exceeded")
    runtimes = cleanup_runtime_catalog(config, chunks)
    frames = active = 0
    endpoint: Fraction | None = None
    seen: set[str] = set()
    for item in chunks:
        producer_ref, candidate_ref = (
            reference(item.get("producer")),
            reference(item.get("candidate")),
        )
        for ref in (producer_ref, candidate_ref):
            if bindings.get(ref["path"]) != ref["sha256"] or ref["path"] in seen:
                raise ValueError("Unbound or duplicate cleanup producer")
            seen.add(ref["path"])
        producer, candidate = (read(producer_ref), read(candidate_ref))
        video = str(Path(string(producer.get("outputDirectory"))) / "candidate.mp4")
        if bindings.get(video) != digest(candidate.get("videoSHA256")):
            raise ValueError("Assembled repair movie digest differs")
        if config.get("contract") == "recorded-local-text-detection-v1":
            count, changed, start, end = local_detection_chunk(
                producer,
                candidate,
                source=mapping(config.get("source")),
                producer_ref=producer_ref,
                runtime=runtimes[producer_ref["path"]],
            )
        else:
            if "contract" in config:
                raise ValueError("Unsupported cleanup contract")
            count, changed, start, end = chunk(
                producer,
                candidate,
                read(producer.get("sourceEligibility")),
                mapping(config.get("source")),
            )
        if endpoint is not None and start != endpoint:
            raise ValueError("Cleanup chunks have a clock gap or overlap")
        endpoint = end
        frames += count
        active += changed
    if (
        frames != integer(assembly.get("frames"))
        or assembly.get("exactPresentationClockVerified") is not True
    ):
        raise ValueError("Cleanup assembly frame total differs")
    if approved_original_identity(config) != approved_identity:
        raise ValueError("Approved original changed during cleanup evidence assessment")
    result = recorded(
        {
            "chunks": len(chunks),
            "frames": frames,
            "repairActiveFrames": active,
            "inactiveFramesNotApproved": frames - active,
            "scope": (
                "original recorded local detection and repair execution; "
                "softened_frames counts cosmetic decisions, not positive mask support"
                if config.get("contract") == "recorded-local-text-detection-v1"
                else "original recorded per-frame eligibility and local repair execution"
            ),
        }
    )
    result["pending"] = [
        "OCR/classification semantic truth",
        "repair visual quality",
        "original executor checkpoint reconstruction",
        "current media crypto",
    ]
    return result


def cleanup_runtime_catalog(
    config: dict[str, object], chunks: list[dict[str, object]]
) -> dict[str, dict[str, str]]:
    """Keep archived producer path identities; never use those bytes as a backend."""
    if "runtimePinsByProducer" not in config:
        if config.get("contract") != "recorded-local-text-detection-v1":
            return {}
        runtime = hashes(config.get("runtimePins"))
        return {reference(row.get("producer"))["path"]: runtime for row in chunks}
    if config.get("contract") != "recorded-local-text-detection-v1":
        raise ValueError("Per-producer runtime pins require original local detection")
    expected = {
        reference(row.get("producer"))["path"]: reference(row.get("producer"))["sha256"]
        for row in chunks
    }
    baseline = hashes(config.get("runtimePins"))
    by_name = {Path(path).name: sha for path, sha in baseline.items()}
    if len(baseline) != 3 or set(by_name) != {
        "subtitle_candidate.py",
        "subtitle_prepared.py",
        "subtitle_detector.py",
    }:
        raise ValueError("Incomplete original local detection runtime")
    result: dict[str, dict[str, str]] = {}
    for row in rows(config.get("runtimePinsByProducer")):
        if set(row) != {"producer", "runtimePins"}:
            raise ValueError("Unknown per-producer runtime schema")
        producer, runtime = reference(row["producer"]), hashes(row["runtimePins"])
        if (
            producer["path"] in result
            or expected.get(producer["path"]) != producer["sha256"]
            or len(runtime) != 3
            or {Path(path).name: sha for path, sha in runtime.items()} != by_name
        ):
            raise ValueError("Per-producer runtime identity or code bytes differ")
        result[producer["path"]] = runtime
    if set(result) != set(expected) or len(expected) != len(chunks):
        raise ValueError("Per-producer runtime coverage differs from original chunks")
    return result


def local_detection_chunk(
    producer: dict[str, object],
    candidate: dict[str, object],
    *,
    source: dict[str, object],
    producer_ref: dict[str, str],
    runtime: dict[str, str],
) -> tuple[int, int, Fraction, Fraction]:
    """Validate original detected-support execution, never classifier/quality approval."""
    same({"path": producer.get("source"), "sha256": producer.get("sourceSHA256")}, source)
    detector = mapping(producer.get("learnedTextDetector"))
    if (
        set(detector) != {"backend", "model", "sha256"}
        or detector.get("backend") != "ppocrv3-db"
        or candidate.get("learnedTextDetector") != detector
        or candidate.get("detectionBackend") != "ppocrv3-db"
        or producer.get("visionTextRegions") is not True
        or producer.get("prepareVisionContext") is not False
        or candidate.get("prepareVisionContext") is not False
        or candidate.get("visionTextRegions") is not None
        or "sourceEligibility" in producer
        or "sourceEligibility" in candidate
    ):
        raise ValueError("Unsupported original local detection selection")
    if len(runtime) != 3 or {Path(path).name for path in runtime} != {
        "subtitle_candidate.py",
        "subtitle_prepared.py",
        "subtitle_detector.py",
    }:
        raise ValueError("Incomplete original local detection runtime")
    expected = dict(runtime)
    for ref in (
        reference(source),
        producer_ref,
        reference(
            {"path": producer.get("nativeExecutable"), "sha256": producer.get("nativeSHA256")}
        ),
        reference({"path": detector.get("model"), "sha256": detector.get("sha256")}),
    ):
        if ref["path"] in expected:
            raise ValueError("Overlapping original local detection roles")
        expected[ref["path"]] = ref["sha256"]
    if hashes(candidate.get("inputs")) != expected:
        raise ValueError("Original local detection source/model/runtime pins differ")
    clock_data = mapping(producer.get("clock"))
    raw_pts = clock_data.get("pts")
    if not isinstance(raw_pts, list) or not 1 <= len(raw_pts) <= 10000:
        raise ValueError("Invalid local detection clock")
    clock = VideoFrameClock(
        tuple(integer(value) for value in raw_pts),
        Fraction(string(clock_data.get("timeBase"))),
        integer(clock_data.get("lastDuration")),
    )
    spans = repair_spans(
        producer.get("masks"),
        frames=len(clock.pts),
        width=integer(producer.get("width")),
        height=integer(producer.get("height")),
        source_clock=clock,
    )
    active = [frame for span in spans for frame in range(span.start_frame, span.end_frame)]
    render = mapping(candidate.get("render"))
    if (
        candidate.get("activeSourceFrames") != active
        or integer(render.get("frames")) != len(clock.pts)
        or integer(render.get("attempts")) != len(active)
        or Fraction(string(candidate.get("sourceStart"))) != clock.pts[0] * clock.time_base
    ):
        raise ValueError("Local detection frame membership differs")
    counts = candidate.get("detectedSupportPixels")
    if not isinstance(counts, list) or len(counts) != len(active):
        raise ValueError("Missing frame-local detected support")
    pixels = [integer(value) for value in counts]
    areas = [
        span.width * span.height for span in spans for _ in range(span.start_frame, span.end_frame)
    ]
    if (
        any(value < 0 or value > area for value, area in zip(pixels, areas, strict=True))
        or integer(candidate.get("emptyDetections")) != pixels.count(0)
        # The original renderer counts unmeasured cosmetic-fallback decisions,
        # including an empty detector mask, not positive-support masks.
        or integer(render.get("softened_frames")) != len(active)
        or candidate.get("emptyDetectionIsNotAbsence") is not True
    ):
        raise ValueError("Local detected-support accounting differs")
    for key in ("learnedObservedRegions", "learnedCanvasIntersections"):
        values = candidate.get(key)
        if (
            not isinstance(values, list)
            or len(values) != len(active)
            or any(integer(value) < 0 for value in values)
        ):
            raise ValueError("Missing frame-local detector observations")
    return (
        len(clock.pts),
        len(active),
        clock.pts[0] * clock.time_base,
        (clock.pts[-1] + clock.last_duration) * clock.time_base,
    )


def followup_history(
    config: dict[str, object], *, reader: MetadataReader | None = None
) -> dict[str, object]:
    read = (reader if reader is not None else MetadataReader()).read
    if integer(config.get("version")) != 1:
        raise ValueError("Unsupported history descriptor")
    graph_ref = reference(config.get("workflow"))
    terminal_ref = reference(config.get("terminal"))
    graph, terminal = (read(graph_ref), read(terminal_ref))
    binding = read(config.get("binding"))
    pins = hashes(binding.get("inputs"))
    if pins.get(graph_ref["path"]) != graph_ref["sha256"]:
        raise ValueError("Original supervisor does not bind workflow")
    kind = string(terminal.get("kind"))
    if kind not in {"verified-integrated-native-followup", "verified-common-context-native-batch"}:
        raise ValueError("Unsupported original terminal contract")
    selected = [
        stage for stage in rows(graph.get("stages")) if stage.get("name") == config.get("stage")
    ]
    if len(selected) != 1:
        raise ValueError("Missing or ambiguous original stage")
    stage = selected[0]
    declared = strings(stage.get("inputs"))
    recorded = mapping(terminal.get("inputs"))
    subset = {path: recorded[path] for path in declared if path in recorded}
    selected_pins = hashes(subset) if subset else {}
    for path, value in selected_pins.items():
        if path in pins and pins[path] != value:
            raise ValueError("Supervisor and original terminal input pins differ")
        pins[path] = value
    if "originalMetadata" in config:
        for item in rows(config["originalMetadata"]):
            ref = reference(item)
            if ref["path"] not in declared or ref["path"] in pins:
                raise ValueError("Supplemental metadata must fill one missing original input")
            read(ref)
            pins[ref["path"]] = ref["sha256"]
    if any(path not in pins for path in declared):
        raise ValueError("Historical input inventory is incomplete")
    expected: dict[str, str] = {terminal_ref["path"]: terminal_ref["sha256"]}
    output_rows = (
        rows(terminal.get("outputs")) if kind == "verified-integrated-native-followup" else []
    )
    for raw in output_rows:
        ref = reference(raw)
        if ref["path"] in expected:
            raise ValueError("Duplicate original output reference")
        expected[ref["path"]] = ref["sha256"]
    result = stage_provenance(
        stage=stage,
        checkpoint=read(config.get("checkpoint")),
        input_hashes={path: pins[path] for path in declared},
        cwd=string(config.get("cwd")),
        expected_outputs=expected,
    )
    result.update(
        {
            "qualityApproved": False,
            "currentOutputsCryptoVerified": False,
            "scope": "original completed followup stage checkpoint; child domains separate",
        }
    )
    return result


def publication_evidence(
    retirement_ref: object, *, reader: MetadataReader | None = None
) -> dict[str, object]:
    read = (reader if reader is not None else MetadataReader()).read
    result = read(retirement_ref)
    if integer(result.get("version")) != 1:
        raise ValueError("Unsupported retirement result")
    for key in ("producerExit", "mandatoryConsumerExit", "postDeleteConsumerExit"):
        if integer(result.get(key)) != 0:
            raise ValueError("Incomplete retirement consumers")
    if result.get("deletionActuallyPerformed") is not True:
        raise ValueError("Prepare is not deletion")
    retirement = read(result.get("retirementReceipt"))
    if retirement.get("kind") != "verified-staging-retirement":
        raise ValueError("Unknown retirement contract")
    publication = read(retirement.get("publication"))
    completion = read(retirement.get("completion"))
    intent = read(retirement.get("intent"))
    if (
        publication.get("kind") != "published-native-part"
        or completion.get("kind") != "verified-native-part"
    ):
        raise ValueError("Wrong native completion/publication contract")
    if intent.get("kind") != "native-part-publication-intent":
        raise ValueError("Wrong publication intent")
    same(publication.get("completion"), retirement.get("completion"))
    same(publication.get("intent"), retirement.get("intent"))
    same(intent.get("completion"), retirement.get("completion"))
    same(publication.get("movie"), retirement.get("movie"))
    same(publication.get("source"), retirement.get("source"))
    same(completion.get("movie"), publication.get("source"))
    same(intent.get("source"), publication.get("source"))
    same(publication.get("frameProof"), retirement.get("frameProof"))
    source, movie = (reference(publication.get("source")), reference(publication.get("movie")))
    if (
        source["sha256"] != movie["sha256"]
        or digest(result.get("sourceSHA256")) != source["sha256"]
    ):
        raise ValueError("Publication changed media bytes")
    if (
        result.get("source") != source["path"]
        or result.get("replacement") != movie["path"]
        or intent.get("destination") != movie["path"]
    ):
        raise ValueError("Retirement target differs")
    part = integer(result.get("part"))
    if part < 1 or any(integer(item.get("part")) != part for item in (retirement, completion)):
        raise ValueError("Part identity differs")
    if (
        completion.get("completionVerified") is not True
        or integer(completion.get("fullAVDecodeExitCode")) != 0
    ):
        raise ValueError("Native technical completion missing")
    if completion.get("shareObservedIdle") is not True:
        raise ValueError("Native writer completion unknown")
    attempt = read(result.get("postDeleteSupervisorAttempt"))
    if attempt.get("status") != "complete" or integer(attempt.get("code")) != 0:
        raise ValueError("Post-delete consumer is incomplete")
    if string(result.get("mandatoryConsumerWorkflow")) not in strings(attempt.get("command")):
        raise ValueError("Wrong post-delete workflow")
    absence = read(result.get("actualAbsenceRecord"))
    legacy_absence = "bothNotExistsAndNotSymlink" in absence
    absence_key = (
        "bothNotExistsAndNotSymlink"
        if legacy_absence
        else "actualAbsenceBothNotExistsAndNotSymlink"
    )
    if (
        absence.get("source") != source["path"]
        or absence.get("replacement") != movie["path"]
        or absence.get("deletionActuallyPerformed") is not True
        or (absence.get(absence_key) is not True)
    ):
        raise ValueError("Wrong deletion observation")
    hashes(completion.get("inputs"))
    if legacy_absence:
        mapping(read(absence.get("historyManifest")))
    elif (
        digest(absence.get("sourceSHA256")) != source["sha256"]
        or absence.get("replacementIdentityUnchanged") is not True
    ):
        raise ValueError("Replacement identity observation differs")
    return {
        "part": part,
        "publicationRecorded": True,
        "retirementRecorded": True,
        "nativeDispatchVerified": completion.get("nativeDispatchVerified") is True,
        "recordedFullAVDecodeVerified": True,
        "currentCryptoVerified": False,
        "qualityApproved": False,
        "checkpointReconstructionPending": True,
        "historyManifestBound": legacy_absence,
    }


def official_payload(envelope: dict[str, object]) -> dict[str, object]:
    if envelope.get("ok") is not True:
        raise ValueError("Native envelope failed")
    data = mapping(envelope.get("data"))
    structured = mapping(data.get("structuredContent"))
    if data.get("isError") is not False or structured.get("truncated") is not False:
        raise ValueError("Native response is incomplete")
    if type(structured.get("exitStatus")) is not int or structured.get("exitStatus") != 0:
        raise ValueError("Native command failed")
    return structured


def original_success_candidate(
    config: dict[str, object], reader: MetadataReader
) -> dict[str, object]:
    """Bind the original stable-stat candidate to the later independently hashed movie."""
    inventory = hashes(
        reader.read(mapping(config.get("stageEvidence")).get("inventory")).get("inputs")
    )
    for key in ("success", "request", "submit"):
        ref = reference(config.get(key))
        if inventory.get(ref["path"]) != ref["sha256"]:
            raise ValueError("Original final inventory lacks native candidate input")
    status_ref = reference(config.get("status"))
    if status_ref["path"] in inventory and inventory[status_ref["path"]] != status_ref["sha256"]:
        raise ValueError("Original native status pin conflicts")
    success, request = reader.read(config.get("success")), reader.read(config.get("request"))
    master, output = reference(config.get("source")), reference(config.get("output"))
    submitted = official_payload(reader.read(config.get("submit"), native_submit=True))
    observed = official_payload(reader.read(config.get("status"), native_status=True))
    batch = mapping(mapping(json.loads(string(submitted.get("stdout")))).get("batch"))
    jobs = rows(batch.get("jobs"))
    statuses: object = json.loads(string(observed.get("stdout")))
    if len(jobs) != 1 or not isinstance(statuses, list) or len(statuses) != 1:
        raise ValueError("Ambiguous original candidate job")
    terminal = mapping(mapping(statuses[0]).get("jobStatus"))
    job, batch_id = string(jobs[0].get("jobID")), string(batch.get("batchID"))
    status_path = Path(string(success.get("nativeStatusEvidence")))
    origin = Path(reference(config.get("success"))["path"]).parent
    expected_path = status_path if status_path.is_absolute() else origin / status_path
    if (
        success.get("version") != 1
        or success.get("kind") != "native-whole-compressor-success-candidate"
        or success.get("nativeStatus") != "Successful"
        or integer(success.get("percentComplete")) != 100
        or integer(success.get("nativeCLIExitStatus")) != 0
        or success.get("stableStatObservedTwice") is not True
        or success.get("jobID") != job
        or success.get("batchID") != batch_id
        or terminal.get("jobid") != job
        or terminal.get("batchid") != batch_id
        or terminal.get("status") != "Successful"
        or terminal.get("percentComplete") != "100"
        or os.path.normpath(expected_path) != reference(config.get("status"))["path"]
        or request.get("version") != 1
        or request.get("kind") != "whole-compressor-request"
    ):
        raise ValueError("Original native candidate status differs")
    raw = reader.read(request.get("request"))
    pins = hashes(request.get("inputs"))
    if (
        success.get("wholeMasterSha256") != master["sha256"]
        or request.get("wholeMaster") != master["path"]
        or pins.get(master["path"]) != master["sha256"]
        or raw.get("sourcePath") != master["path"]
        or raw.get("presetPath") not in pins
        or submitted.get("outputPath") != output["path"]
        or success.get("movie") != output["path"]
    ):
        raise ValueError("Original native candidate source or output differs")
    receipt = reader.read(config.get("receipt"))
    actual = fingerprint_rows(receipt.get("inputs")).get(output["path"])
    recorded_stat = mapping(success.get("observedStableStat"))
    if actual is None or actual["sha256"] != output["sha256"]:
        raise ValueError("Independent original candidate digest differs")
    if (
        integer(success.get("movieBytes")) != actual["bytes"]
        or any(integer(recorded_stat.get(key)) != actual[key] for key in ("device", "inode"))
        or any(
            integer(recorded_stat.get(key + "Seconds"))
            != integer(actual[key + "Ns"]) // 1_000_000_000
            for key in ("mtime", "ctime")
        )
    ):
        raise ValueError("Independent original candidate identity differs")
    return {
        "oneWholeJobRecorded": True,
        "jobUUID": job,
        "batchUUID": batch_id,
        "officialSuccessful": True,
        "originalNativeToolCompletionVerified": success.get("mcpCompletionVerified") is True,
        "laterIndependentOutputBound": True,
        "historicalStatusExecutionPinVerified": False,
        "statusDirectOriginalInput": status_ref["path"] in inventory,
        "currentCryptoVerified": False,
        "qualityApproved": False,
        "globalDuplicateAbsenceVerified": False,
    }


def explicit_native_success(config: dict[str, object], reader: MetadataReader) -> dict[str, object]:
    """Read the original explicit status observation, without inventing early output SHA."""
    success_ref, request_ref, submit_ref, status_ref = (
        reference(config.get(key)) for key in ("success", "request", "submit", "status")
    )
    stage = mapping(config.get("stageEvidence"))
    inventory = hashes(reader.read(stage.get("inventory")).get("inputs"))
    for ref in (success_ref, request_ref, submit_ref):
        if inventory.get(ref["path"]) != ref["sha256"]:
            raise ValueError("Original final checkpoint lacks native observation input")
    success, request = reader.read(success_ref), reader.read(request_ref)
    if (
        success.get("version") != 1
        or success.get("kind") != "explicit-native-whole-compressor-success-observation"
        or success.get("nativeStatus") != "Successful"
        or integer(success.get("nativePercentComplete")) != 100
        or integer(success.get("statusJobExit")) != 0
        or request.get("version") != 1
        or request.get("kind") != "whole-compressor-request"
    ):
        raise ValueError("Unsupported explicit native success observation")
    status_relative = Path(string(success.get("statusEvidence")))
    cwd = Path(string(stage.get("cwd")))
    if (
        status_relative.is_absolute()
        or ".." in status_relative.parts
        or str(status_relative) != success.get("statusEvidence")
        or str(cwd / status_relative) != status_ref["path"]
    ):
        raise ValueError("Explicit native status evidence path differs")
    submitted = official_payload(reader.read(submit_ref, native_submit=True))
    observed = official_payload(reader.read(status_ref, native_status=True))
    batch = mapping(mapping(json.loads(string(submitted.get("stdout")))).get("batch"))
    jobs = rows(batch.get("jobs"))
    statuses: object = json.loads(string(observed.get("stdout")))
    if len(jobs) != 1 or not isinstance(statuses, list) or len(statuses) != 1:
        raise ValueError("Ambiguous explicit native job")
    job, batch_id = string(jobs[0].get("jobID")), string(batch.get("batchID"))
    terminal = mapping(mapping(statuses[0]).get("jobStatus"))
    if (
        success.get("jobID") != job
        or success.get("batchID") != batch_id
        or terminal.get("jobid") != job
        or terminal.get("batchid") != batch_id
        or terminal.get("status") != "Successful"
        or terminal.get("percentComplete") != "100"
    ):
        raise ValueError("Explicit native job/status differs")
    master, output = reference(success.get("master")), reference(config.get("output"))
    request_inputs = hashes(request.get("inputs"))
    raw = reader.read(request.get("request"))
    if (
        request.get("wholeMaster") != master["path"]
        or request_inputs.get(master["path"]) != master["sha256"]
        or raw.get("sourcePath") != master["path"]
        or raw.get("presetPath") not in request_inputs
        or submitted.get("outputPath") != output["path"]
        or success.get("movie") != output["path"]
    ):
        raise ValueError("Explicit native source/output differs")
    same(master, config.get("source"))
    receipt = reader.read(config.get("receipt"))
    movie_row = fingerprint_rows(receipt.get("inputs")).get(output["path"])
    if (
        movie_row is None
        or movie_row["sha256"] != output["sha256"]
        or movie_row["bytes"] != integer(success.get("observedStableSizeBytes"))
        or movie_row["inode"] != integer(success.get("observedStableInode"))
        or integer(movie_row["mtimeNs"]) // 1_000_000_000
        != integer(success.get("observedStableMtimeSeconds"))
    ):
        raise ValueError("Later independent output differs from native observation")
    return {
        "oneWholeJobRecorded": True,
        "jobUUID": job,
        "batchUUID": batch_id,
        "officialSuccessful": True,
        "originalNativeToolCompletionVerified": success.get("nativeToolCompletionVerified") is True,
        "laterIndependentOutputBound": True,
        "historicalStatusExecutionPinVerified": False,
        "currentCryptoVerified": False,
        "qualityApproved": False,
        "globalDuplicateAbsenceVerified": False,
    }


def native_success_execution(
    config: dict[str, object], reader: MetadataReader
) -> dict[str, object]:
    """Join original native status/sourceMaster to later independent output proof."""
    success = reader.read(config.get("success"))
    if (
        integer(success.get("version")) != 1
        or integer(success.get("nativeSubmissionCount")) != 1
        or integer(success.get("resubmissions")) != 0
        or success.get("independentFullAVPending") is not True
    ):
        raise ValueError("Unsupported original native-success schema")
    for key, target in (("officialStatus", "status"), ("requestReceipt", "request")):
        same(success.get(key), config.get(target))
    request = reader.read(config.get("request"))
    if request.get("version") != 1 or request.get("kind") != "whole-compressor-request":
        raise ValueError("Unsupported original whole request")
    raw_request = reader.read(success.get("request"))
    submission_result = reader.read(success.get("submitResult"))
    same(submission_result.get("request"), success.get("request"))
    same(submission_result.get("requestReceipt"), config.get("request"))
    same(submission_result.get("rawSubmitLog"), config.get("submit"))
    raw_submit = reader.read(config.get("submit"), native_submit=True)
    if (
        submission_result.get("submitEnvelope") != raw_submit
        or submission_result.get("nativeSubmissionCount") != 1
    ):
        raise ValueError("Original submission envelope differs")
    submitted = official_payload(raw_submit)
    observed = official_payload(reader.read(config.get("status")))
    batches = mapping(mapping(json.loads(string(submitted.get("stdout")))).get("batch"))
    jobs = rows(batches.get("jobs"))
    statuses: object = json.loads(string(observed.get("stdout")))
    if len(jobs) != 1 or not isinstance(statuses, list) or len(statuses) != 1:
        raise ValueError("Ambiguous original native job")
    status = mapping(mapping(statuses[0]).get("jobStatus"))
    job, batch = string(jobs[0].get("jobID")), string(batches.get("batchID"))
    if (
        status != mapping(success.get("nativeStatus"))
        or status.get("status") != "Successful"
        or status.get("percentComplete") != "100"
        or status.get("jobid") != job
        or status.get("batchid") != batch
        or submission_result.get("jobID") != job
        or submission_result.get("batchID") != batch
    ):
        raise ValueError("Original native job/status differs")
    source = reference(success.get("sourceMaster"))
    output = reference(config.get("output"))
    if (
        request.get("wholeMaster") != source["path"]
        or hashes(request.get("inputs")).get(source["path"]) != source["sha256"]
        or raw_request.get("sourcePath") != source["path"]
        or submitted.get("outputPath") != output["path"]
        or submission_result.get("outputPath") != output["path"]
        or mapping(success.get("output")).get("path") != output["path"]
    ):
        raise ValueError("Original native source/output differs")
    early_output = mapping(success.get("output"))
    if (
        early_output.get("sha256") is not None
        or early_output.get("hashPendingIndependentReader") is not True
    ):
        raise ValueError("Original pending output hash schema differs")
    receipt = reader.read(config.get("receipt"))
    if fingerprint_hashes(receipt.get("inputs")).get(output["path"]) != output["sha256"]:
        raise ValueError("Later independent receipt does not bind output")
    return {
        "oneWholeJobRecorded": True,
        "jobUUID": job,
        "batchUUID": batch,
        "officialSuccessful": True,
        "earlyOutputHashPending": True,
        "laterIndependentOutputBound": True,
        "currentCryptoVerified": False,
        "qualityApproved": False,
        "globalDuplicateAbsenceVerified": False,
    }


def original_executor_request(
    config: dict[str, object], reader: MetadataReader
) -> dict[str, object]:
    """Bind retained raw Executor request/status to the later original final checkpoint."""
    inventory = hashes(
        reader.read(mapping(config.get("stageEvidence")).get("inventory")).get("inputs")
    )
    post_ref, status_ref = reference(config.get("postConfig")), reference(config.get("status"))
    if any(inventory.get(ref["path"]) != ref["sha256"] for ref in (post_ref, status_ref)):
        raise ValueError("Original final checkpoint does not bind native status/post config")
    post = reader.read(post_ref)
    master, movie, preset = (reference(post.get(key)) for key in ("master", "movie", "preset"))
    same(master, config.get("source"))
    same(movie, config.get("output"))
    if (
        inventory.get(movie["path"]) != movie["sha256"]
        or inventory.get(preset["path"]) != preset["sha256"]
    ):
        raise ValueError("Original final movie or preset pin differs")
    request = reader.read(config.get("request"))
    if set(request) != {
        "batchName",
        "bundleID",
        "sourcePath",
        "presetPath",
        "outputDirectory",
        "outputName",
    }:
        raise ValueError("Unsupported original Executor request schema")
    if request.get("sourcePath") != master["path"] or request.get("presetPath") != preset["path"]:
        raise ValueError("Original Executor source or preset differs")
    directory, name = (
        Path(string(request.get("outputDirectory"))),
        string(request.get("outputName")),
    )
    output = Path(movie["path"])
    if (
        not directory.is_absolute()
        or str(directory) != request["outputDirectory"]
        or ".." in directory.parts
        or Path(name).name != name
        or name != output.name
        or output.parent.parent != directory
    ):
        raise ValueError("Original Executor output scope differs")
    string(request.get("batchName"))
    string(request.get("bundleID"))
    submitted = official_payload(reader.read(config.get("submit"), native_submit=True))
    observed = official_payload(reader.read(status_ref))
    batch = mapping(mapping(json.loads(string(submitted.get("stdout")))).get("batch"))
    jobs = rows(batch.get("jobs"))
    statuses: object = json.loads(string(observed.get("stdout")))
    if len(jobs) != 1 or not isinstance(statuses, list) or len(statuses) != 1:
        raise ValueError("Ambiguous original Executor job")
    job = mapping(mapping(statuses[0]).get("jobStatus"))
    job_id, batch_id = string(jobs[0].get("jobID")), string(batch.get("batchID"))
    if (
        submitted.get("outputPath") != movie["path"]
        or job.get("jobid") != job_id
        or job.get("batchid") != batch_id
        or job.get("status") != "Successful"
        or job.get("percentComplete") != "100"
    ):
        raise ValueError("Original Executor job/state/output differs")
    return {
        "oneWholeJobRecorded": True,
        "jobUUID": job_id,
        "batchUUID": batch_id,
        "officialSuccessful": True,
        "laterIndependentOutputBound": True,
        "originalRawRequestExecutionFingerprintVerified": False,
        "historicalNativeRuntimePinVerified": False,
        "currentCryptoVerified": False,
        "qualityApproved": False,
        "globalDuplicateAbsenceVerified": False,
    }


def retained_phase_binding(
    evidence: dict[str, object],
    reader: MetadataReader,
) -> tuple[dict[str, object], dict[str, str], dict[str, str]]:
    """Bind an immutable original stage; its domain meaning is checked by the caller."""
    graph = reader.read(evidence.get("workflow"))
    stages = [row for row in rows(graph.get("stages")) if row.get("name") == evidence.get("stage")]
    if len(stages) != 1:
        raise ValueError("Original phase is missing or ambiguous")
    stage = stages[0]
    inputs = hashes(reader.read(evidence.get("inventory")).get("inputs"))
    outputs = hashes(evidence.get("outputs"))
    stage_provenance(
        stage=stage,
        checkpoint=reader.read(evidence.get("checkpoint")),
        input_hashes=inputs,
        cwd=string(evidence.get("cwd")),
        expected_outputs=outputs,
    )
    return stage, inputs, outputs


def split_video_whole_evidence(
    config: dict[str, object], reader: MetadataReader
) -> dict[str, object]:
    """Preserve a video-only master and a separately once-encoded whole voice."""
    master, voice, semantic = (
        reference(config.get(key)) for key in ("source", "voice", "semanticXML")
    )
    frames, fps = integer(config.get("frames")), integer(config.get("fps"))
    plan_ref, partition_ref = reference(config.get("joinPlan")), reference(config.get("partition"))
    plan = rows(reader.read(plan_ref, join_plan=True).get("rows"))
    partition = reader.read(partition_ref)
    same(partition.get("source"), semantic)
    parts = rows(partition.get("parts"))
    publications = rows(config.get("publications"))
    join_stage, join_inputs, join_outputs = retained_phase_binding(
        mapping(config.get("join")), reader
    )
    command = strings(join_stage.get("command"))
    if (
        len(command) > 7
        and command[1:4] == ["-B", "-m", "replay.phase_timer"]
        and command[6] == "--"
    ):
        command = command[7:]
    if len(command) > 2 and command[0] == "/usr/bin/env" and command[1].startswith("PATH="):
        directories = command[1].removeprefix("PATH=").split(":")
        if any(not Path(path).is_absolute() or ".." in Path(path).parts for path in directories):
            raise ValueError("Original join PATH is not absolute")
        command = command[2:]
    if (
        len(command) != 10
        or not Path(command[0]).is_absolute()
        or command[1:]
        != [
            "-B",
            "-m",
            "replay.delivery_render",
            "join-files",
            plan_ref["path"],
            "--output",
            master["path"],
            "--rate",
            str(fps),
        ]
        or join_outputs.get(master["path"]) != master["sha256"]
        or join_inputs.get(plan_ref["path"]) != plan_ref["sha256"]
        or join_inputs.get(partition_ref["path"]) != partition_ref["sha256"]
        or partition.get("globalCaptionCuesEqual") is not True
        or not 1 <= len(plan) == len(parts) == len(publications) <= 64
        or frames <= 0
        or fps <= 0
    ):
        raise ValueError("Original split join command or coverage differs")
    total = 0
    seen: set[str] = set()
    for number, (row, part, pub_ref) in enumerate(zip(plan, parts, publications, strict=True), 1):
        publication = reader.read(pub_ref)
        movie = reference(publication.get("movie"))
        completion = published_completion(pub_ref, reader)
        count = integer(row.get("durationFrames"))
        if (
            set(row) != {"sourcePath", "durationFrames"}
            or count <= 0
            or row.get("sourcePath") != movie["path"]
            or movie["path"] in seen
            or completion.get("part") != number
            or join_inputs.get(movie["path"]) != movie["sha256"]
            or join_inputs.get(reference(pub_ref)["path"]) != reference(pub_ref)["sha256"]
            or Fraction(string(part.get("globalStart")).removesuffix("s")) != Fraction(total, fps)
            or Fraction(string(part.get("duration")).removesuffix("s")) != Fraction(count, fps)
        ):
            raise ValueError("Original ordered join publication/clock differs")
        seen.add(movie["path"])
        total += count
    if total != frames or Fraction(string(partition.get("duration")).removesuffix("s")) != Fraction(
        frames, fps
    ):
        raise ValueError("Original split whole endpoint differs")
    receipt_ref = reference(config.get("packetReceipt"))
    receipt = reader.read(receipt_ref)
    packet_stage, packet_inputs, packet_outputs = retained_phase_binding(
        mapping(config.get("packetStage")), reader
    )
    # The original jq serialization command is retained by its exact fingerprint;
    # its typed output is assessed here, not executed or presented as a new decode.
    packet_command = strings(packet_stage.get("command"))
    if (
        len(packet_command) != 3
        or packet_command[:2] != ["/bin/bash", "-c"]
        or not packet_command[2].startswith("set -euo pipefail\n")
    ):
        raise ValueError("Unsupported original packet acceptance producer")
    same(receipt.get("master"), master)
    same(receipt.get("producerCheckpoint"), mapping(config.get("join")).get("checkpoint"))
    raw = reference(receipt.get("observations"))
    pins = hashes(receipt.get("inputs"))
    video = mapping(receipt.get("video"))
    base = Fraction(string(video.get("timeBase")))
    if (
        receipt.get("version") != 2
        or receipt.get("kind") != "whole-video-only-packet-clock-acceptance"
        or packet_outputs.get(receipt_ref["path"]) != receipt_ref["sha256"]
        or packet_inputs.get(master["path"]) != master["sha256"]
        or packet_inputs.get(raw["path"]) != raw["sha256"]
        or any(packet_inputs.get(path) != sha for path, sha in pins.items())
        or receipt.get("packetClockAccepted") is not True
        or receipt.get("audioStreams") != 0
        or any(
            receipt.get(key) is not False
            for key in (
                "rejoined",
                "newDecode",
                "fullDecode",
                "qualityApproved",
                "syncApproved",
                "deliveryApproved",
            )
        )
        or video.get("codec") != "prores"
        or video.get("exactPacketCFR") is not True
        or integer(video.get("frames")) != frames
        or integer(video.get("startPTS")) != 0
        or base <= 0
        or integer(video.get("endPTS")) * base != Fraction(frames, fps)
    ):
        raise ValueError("Original video-only packet acceptance differs")
    audio_ref = reference(config.get("audioEvidence"))
    audio = reader.read(audio_ref)
    same(audio.get("source"), voice)
    audio_movie = reference(audio.get("movie"))
    if audio.get("encodingPasses") != 1:
        raise ValueError("Original whole AAC was not encoded once")
    for key in ("nativeStatus", "fullAV", "decodedProbe", "packetProbe"):
        reference(audio.get(key))
    result = recorded(
        {
            "frames": frames,
            "parts": len(plan),
            "videoOnlyMaster": master,
            "wholeVoice": voice,
            "wholeAAC": audio_movie,
            "recordedEncodingPasses": 1,
            "wholeMasterFullDecodeVerified": False,
            "wholeVoicePCMEqual": False,
            "packetObservationReparsed": False,
        }
    )
    result["pending"] = [
        "current media crypto",
        "raw packet observation independent revalidation",
        "native provider before/after provenance",
        "speech synchronization and quality",
    ]
    return result


def original_technical_whole(
    config: dict[str, object], reader: MetadataReader
) -> dict[str, object]:
    """Bind the original typed reuse receipt and each actually completed technical stage."""
    final_inputs = hashes(
        reader.read(mapping(config.get("stageEvidence")).get("inventory")).get("inputs")
    )
    result_ref, receipt_ref = (
        reference(config.get("wholeResult")),
        reference(config.get("technicalReceipt")),
    )
    anchors = (result_ref, receipt_ref)
    if not any(final_inputs.get(ref["path"]) == ref["sha256"] for ref in anchors) or any(
        ref["path"] in final_inputs and final_inputs[ref["path"]] != ref["sha256"]
        for ref in anchors
    ):
        raise ValueError("Final stage lacks original whole technical evidence")
    summary, receipt = reader.read(result_ref), reader.read(receipt_ref)
    manifest = reader.read(config.get("technicalManifest"))
    technical = reader.read(config.get("technicalConfig"))
    same(receipt.get("manifest"), config.get("technicalManifest"))
    same(manifest.get("config"), config.get("technicalConfig"))
    same(receipt.get("workflow"), config.get("technicalWorkflow"))
    same(manifest.get("technicalWorkflow"), config.get("technicalWorkflow"))
    master, voice = reference(config.get("source")), reference(config.get("voice"))
    summary_master = mapping(summary.get("wholeMaster"))
    same({"path": summary_master.get("path"), "sha256": summary_master.get("sha256")}, master)
    names = ["whole-lossless-join", "whole-frame-clock", "whole-metadata", "joined-pcm-hash"]
    phases = rows(config.get("phases"))
    if (
        receipt.get("version") != 1
        or receipt.get("kind") != "verified-whole-technical-batch"
        or receipt.get("domains") != names[1:]
        or technical.get("targets") != names[1:]
        or [row.get("stage") for row in phases] != names
        or receipt.get("checkpoints") != [row.get("checkpoint") for row in phases[1:]]
        or integer(summary.get("technicalRunCode")) != 0
        or integer(summary.get("freshReadonlyVerifyCode")) != 0
        or summary.get("originalStrictFullAVVerified") is not True
        or summary.get("processedVoicePCMIdentityVerified") is not True
        or summary.get("allVideoPTSVerified") is not True
        or summary.get("technicalReceiptSha256") != receipt_ref["sha256"]
        or summary.get("technicalManifestSha256")
        != reference(config.get("technicalManifest"))["sha256"]
        or integer(summary.get("expectedVideoFrames")) != integer(config.get("frames"))
        or integer(summary.get("videoFramesPerSecond")) != integer(config.get("fps"))
    ):
        raise ValueError("Original technical phase/result coverage differs")
    admission = mapping(technical.get("admission"))
    if admission.get("fullAVReuse") != {
        "version": 1,
        "kind": "completed-canonical-ffmpeg-full-av",
        "stages": names[:1],
    }:
        raise ValueError("Original full AV reuse opt-in differs")
    pins = hashes(receipt.get("inputs"))
    required = reference_inventory(admission.get("inputs"))
    if any(pins.get(path) != sha for path, sha in required.items()):
        raise ValueError("Original technical admission pins differ")
    reuse_rows = rows(receipt.get("reusedFullAV"))
    if len(reuse_rows) != 1:
        raise ValueError("Ambiguous original full AV reuse")
    reuse = reuse_rows[0]
    same(reuse.get("media"), master)
    same(reuse.get("checkpoint"), phases[0].get("checkpoint"))
    decoder = reference(reuse.get("decoder"))
    strict = [
        decoder["path"],
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-protocol_whitelist",
        "file",
        "-f",
        "mov",
        "-enable_drefs",
        "0",
        "-use_absolute_path",
        "0",
        "-i",
        master["path"],
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    if (
        reuse.get("kind") != "completed-canonical-ffmpeg-full-av"
        or reuse.get("stage") != names[0]
        or reuse.get("newDecodePerformed") is not False
        or reuse.get("verifyCommand") != strict
    ):
        raise ValueError("Original canonical full AV reuse differs")
    for phase in phases:
        graph = reader.read(phase.get("workflow"))
        matches = [
            row for row in rows(graph.get("stages")) if row.get("name") == phase.get("stage")
        ]
        if len(matches) != 1:
            raise ValueError("Ambiguous original technical phase")
        stage = matches[0]
        inventory = hashes(reader.read(phase.get("inventory")).get("inputs"))
        inputs = {
            path: inventory[path] for path in strings(stage.get("inputs")) if path in inventory
        }
        outputs = hashes(phase.get("outputs"))
        stage_provenance(
            stage=stage,
            checkpoint=reader.read(phase.get("checkpoint")),
            input_hashes=inputs,
            cwd=string(phase.get("cwd")),
            expected_outputs=outputs,
        )
        if any(path in pins and pins[path] != sha for path, sha in inputs.items()):
            raise ValueError("Original technical input identity differs")
        checked = dict(stage)
        if phase.get("stage") == names[0]:
            if (
                stage.get("verify_command") != strict
                or inputs.get(decoder["path"]) != decoder["sha256"]
            ):
                raise ValueError("Original decode executor/checkpoint differs")
            checked["verify_command"] = [
                decoder["path"],
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "2",
                *strict[13:],
            ]
        elif phase.get("stage") == names[1]:
            verify = strings(stage.get("verify_command"))
            if len(verify) != 7:
                raise ValueError("Original frame verifier differs")
            expression, count = re.subn(
                r"\(\$s\.width==[1-9][0-9]* and \$s\.height==[1-9][0-9]* and ", "(", verify[5]
            )
            if count != 1:
                raise ValueError("Original frame dimensions absent")
            checked["verify_command"] = [*verify[:5], expression, verify[6]]
        whole_observation_command(
            checked,
            master=master,
            voice=voice,
            inputs=inputs,
            outputs=outputs,
            fps=integer(config.get("fps")),
        )
    return {
        "completionRecordsBound": True,
        "completedPhaseCount": 4,
        "individualWholeProducerFingerprintsReconstructed": True,
        "originalFullAVReused": True,
        "newDecodePerformed": False,
        "currentOutputsCryptoVerified": False,
    }


def original_whole_observations(
    config: dict[str, object], reader: MetadataReader
) -> dict[str, object]:
    """Bind original voice observations and joined media through six exact checkpoints."""
    final_inputs = hashes(
        reader.read(mapping(config.get("stageEvidence")).get("inventory")).get("inputs")
    )
    result_ref = reference(config.get("wholeResult"))
    if final_inputs.get(result_ref["path"]) != result_ref["sha256"]:
        raise ValueError("Original whole summary was not a final input")
    whole = reader.read(result_ref)
    master, voice = reference(config.get("source")), reference(config.get("voice"))
    same(whole.get("movie"), master)
    same(whole.get("semanticXML"), config.get("semanticXML"))
    same(whole.get("workflow"), config.get("wholeWorkflow"))
    fps = integer(config.get("fps"))
    if (
        whole.get("kind") != "technically-verified-whole-candidate"
        or whole.get("wholeFullAVDecodeVerified") is not True
        or whole.get("wholeVoicePCMEqual") is not True
        or integer(whole.get("frames")) != integer(config.get("frames"))
        or integer(whole.get("rate")) != fps
        or fps <= 0
    ):
        raise ValueError("Original whole result clock or status differs")
    names = [
        "expected-whole-voice-metadata",
        "expected-whole-voice-pcm-hash",
        "whole-lossless-join",
        "whole-frame-clock",
        "whole-metadata",
        "joined-pcm-hash",
    ]
    graph = reader.read(config.get("wholeWorkflow"))
    stages = [row for row in rows(graph.get("stages")) if row.get("name") in names]
    phases = rows(config.get("phases"))
    if [row.get("name") for row in stages] != names or [
        row.get("stage") for row in phases
    ] != names:
        raise ValueError("Original whole observation coverage differs")
    prior: dict[str, str] = {}
    for stage, phase in zip(stages, phases, strict=True):
        inputs = hashes(reader.read(phase.get("inventory")).get("inputs"))
        outputs = hashes(phase.get("outputs"))
        stage_provenance(
            stage=stage,
            checkpoint=reader.read(phase.get("checkpoint")),
            input_hashes=inputs,
            cwd=string(phase.get("cwd")),
            expected_outputs=outputs,
        )
        whole_observation_command(
            stage, master=master, voice=voice, outputs=outputs, inputs=inputs, fps=fps
        )
        if any(path in inputs and inputs[path] != sha for path, sha in prior.items()):
            raise ValueError("Original whole predecessor pin differs")
        prior.update(outputs)
    if prior.get(master["path"]) != master["sha256"]:
        raise ValueError("Original whole join master differs")
    return {
        "completionRecordsBound": True,
        "completedPhaseCount": 6,
        "individualWholeProducerFingerprintsReconstructed": True,
        "finalStageDirectCheckpointMembership": False,
        "currentOutputsCryptoVerified": False,
        "historicalJoinExecutorPinVerified": False,
    }


def whole_observation_command(
    stage: dict[str, object],
    *,
    master: dict[str, str],
    voice: dict[str, str],
    outputs: dict[str, str],
    inputs: dict[str, str],
    fps: int,
) -> None:
    """Recognize only the original voice/FFmpeg observation commands, without execution."""
    name, command = string(stage.get("name")), strings(stage.get("command"))
    verifier = stage.get("verify_command", [])
    verify = [] if isinstance(verifier, list) and not verifier else strings(verifier)
    if name.startswith("expected-whole-voice-"):
        if inputs.get(voice["path"]) != voice["sha256"] or command[0] not in inputs or verify:
            raise ValueError("Original voice observation binding differs")
        if name == "expected-whole-voice-metadata":
            expected = [
                command[0],
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                "-o",
                command[-2],
                voice["path"],
            ]
            output = command[-2]
            tool = "ffprobe"
        elif name == "expected-whole-voice-pcm-hash":
            expected = [
                command[0],
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-i",
                voice["path"],
                "-map",
                "0:a:0",
                "-c:a",
                "pcm_s32le",
                "-f",
                "streamhash",
                "-hash",
                "sha256",
                command[-1],
            ]
            output, tool = command[-1], "ffmpeg"
        else:
            raise ValueError("Unsupported original voice observation")
        if command != expected or Path(command[0]).name != tool or set(outputs) != {output}:
            raise ValueError("Original voice observation command differs")
        return
    checked = dict(stage)
    if name == "whole-lossless-join":
        if len(command) != 14 or command[0] != "/usr/bin/env" or not verify:
            raise ValueError("Unsupported original join environment")
        tool_dir = str(Path(verify[0]).parent)
        if (
            command[1]
            not in {
                f"PATH={tool_dir}:/usr/bin:/bin:/usr/sbin:/sbin",
                f"PATH={tool_dir}:/usr/bin:/bin",
            }
            or verify[0] not in inputs
            or str(Path(tool_dir) / "ffprobe") not in inputs
            or inputs.get(voice["path"]) != voice["sha256"]
            or command[-1] != voice["path"]
            or command[11] != str(fps)
        ):
            raise ValueError("Original join tool/voice/rate binding differs")
        checked["command"] = command[2:]
    elif name == "whole-frame-clock":
        if len(command) < 5 or command[3:5] != ["-threads", "0"]:
            raise ValueError("Original frame observer arguments differ")
        # Only compare otherwise identical semantics; the fingerprint retains original argv.
        checked["command"] = [*command[:4], "2", *command[5:]]
    elif name not in {"whole-metadata", "joined-pcm-hash"}:
        raise ValueError("Unsupported original whole observation")
    original_whole_command(name, checked, master, outputs, inputs, fps)


def original_whole_phases(config: dict[str, object], reader: MetadataReader) -> dict[str, object]:
    """Reconstruct older whole stages separately; never add them to final-stage inputs."""
    final_inputs = hashes(
        reader.read(mapping(config.get("stageEvidence")).get("inventory")).get("inputs")
    )
    result_ref = reference(config.get("wholeResult"))
    if final_inputs.get(result_ref["path"]) != result_ref["sha256"]:
        raise ValueError("Original whole summary was not a final input")
    whole = reader.read(result_ref)
    master = reference(config.get("source"))
    same(whole.get("master"), master)
    workflow = reader.read(config.get("wholeWorkflow"))
    inventories = reader.read(config.get("wholeInventories"))
    phases = rows(config.get("phases"))
    names = [
        "whole-assembly-evidence",
        "whole-lossless-join",
        "whole-frame-clock",
        "whole-metadata",
        "joined-pcm-hash",
    ]
    stages = rows(workflow.get("stages"))
    if [row.get("name") for row in stages] != names or [
        row.get("stage") for row in phases
    ] != names:
        raise ValueError("Original whole phase coverage differs")
    outputs: dict[str, str] = {}
    for stage, phase in zip(stages, phases, strict=True):
        name = string(stage.get("name"))
        inputs = hashes(inventories.get(name))
        checkpoint = reader.read(phase.get("checkpoint"))
        expected = hashes(phase.get("outputs"))
        stage_provenance(
            stage=stage,
            checkpoint=checkpoint,
            input_hashes=inputs,
            cwd=string(phase.get("cwd")),
            expected_outputs=expected,
        )
        original_whole_command(name, stage, master, expected, inputs, integer(config.get("fps")))
        for path, sha in outputs.items():
            if path in inputs and inputs[path] != sha:
                raise ValueError("Original whole predecessor pin differs")
        outputs.update(expected)
    if outputs.get(master["path"]) != master["sha256"]:
        raise ValueError("Original whole join master differs")
    return {
        "completionRecordsBound": True,
        "completedPhaseCount": 5,
        "individualWholeProducerFingerprintsReconstructed": True,
        "finalStageDirectCheckpointMembership": False,
        "currentOutputsCryptoVerified": False,
        "historicalJoinExecutorPinVerified": False,
    }


def original_whole_command(
    name: str,
    stage: dict[str, object],
    master: dict[str, str],
    outputs: dict[str, str],
    inputs: dict[str, str],
    fps: int,
) -> None:
    command, verify = strings(stage.get("command")), strings(stage.get("verify_command"))
    if name == "whole-assembly-evidence":
        if (
            len(command) not in {7, 9}
            or command[1:5] != ["-B", "-m", "replay.fcp_part_assembly", "assemble"]
            or command[5] not in inputs
            or command[6] not in outputs
            or verify != [*command[:4], "verify-assembly", command[5], command[6]]
        ):
            raise ValueError("Unsupported original whole assembly command")
    elif name == "whole-lossless-join":
        if (
            len(command) != 12
            or command[1:5] != ["-B", "-m", "replay.delivery_render", "join-files"]
            or command[5] not in inputs
            or command[6:9:2] != ["--output", "--rate"]
            or command[7] != master["path"]
            or command[10] != "--audio"
            or command[11] not in inputs
            or outputs != {master["path"]: master["sha256"]}
            or len(verify) != 16
            or verify[1:7] != ["-nostdin", "-v", "error", "-xerror", "-threads", "2"]
            or verify[7:]
            != ["-i", master["path"], "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"]
        ):
            raise ValueError("Unsupported original whole join command")
        if not command[9].isdigit() or int(command[9]) <= 0:
            raise ValueError("Invalid original whole rate")
    elif name in {"whole-frame-clock", "whole-metadata"}:
        if (
            Path(command[0]).name != "ffprobe"
            or command[0] not in inputs
            or command[-1] != master["path"]
            or command[-3] != "-o"
            or command[-2] not in outputs
            or len(verify) != 7
            or Path(verify[0]).name != "jq"
            or verify[1:3] != ["-e", "--slurpfile"]
            or verify[4] not in inputs
            or verify[-1] != command[-2]
        ):
            raise ValueError("Unsupported original whole observation command")
        if name == "whole-frame-clock":
            expression = (
                "($plan[0]|map(.durationFrames)|add) as $n | .streams[0] as $s | "
                '($s.time_base|split("/")|map(tonumber)) as $tb | '
                f'($s.start_pts==0 and $s.avg_frame_rate=="{fps}/1" and '
                f"($s.nb_frames|tonumber)==$n and ($s.duration_ts*$tb[0]*{fps})==($n*$tb[1]) "
                "and (.frames|length)==$n and (.frames|to_entries|all("
                f".value.best_effort_timestamp*$tb[0]*{fps}==.key*$tb[1])))"
            )
            expected_command = [
                command[0],
                "-v",
                "error",
                "-threads",
                "2",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_streams",
                "-show_entries",
                "frame=best_effort_timestamp,duration:stream=codec_name,width,height,time_base,start_pts,duration_ts,nb_frames,avg_frame_rate",
                "-of",
                "json",
                "-o",
                command[-2],
                master["path"],
            ]
        else:
            expression = (
                '$expected[0].streams[0] as $e | [.streams[]|select(.codec_type=="audio")] as $a | '
                "($a|length)==1 and ($a[0]|{codec_name,channels,sample_rate,"
                "time_base,duration_ts})==($e|{codec_name,channels,sample_rate,"
                "time_base,duration_ts}) and $a[0].start_pts==0"
            )
            expected_command = [
                command[0],
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                "-o",
                command[-2],
                master["path"],
            ]
        if command != expected_command or verify[5] != expression or fps <= 0:
            raise ValueError("Original whole observation verifier semantics differ")
    else:
        if (
            len(command) != 16
            or Path(command[0]).name != "ffmpeg"
            or command[0] not in inputs
            or command[1:7] != ["-nostdin", "-v", "error", "-n", "-i", master["path"]]
            or command[7:16]
            != [
                "-map",
                "0:a:0",
                "-c:a",
                "pcm_s32le",
                "-f",
                "streamhash",
                "-hash",
                "sha256",
                command[-1],
            ]
        ):
            raise ValueError("Unsupported original whole PCM command")
        if (
            len(verify) != 3
            or Path(verify[0]).name != "cmp"
            or verify[2] != command[-1]
            or inputs.get(verify[1]) != outputs.get(command[-1])
            or command[-1] not in outputs
        ):
            raise ValueError("Original whole PCM equality differs")


def split_native_request_evidence(
    config: dict[str, object],
    reader: MetadataReader,
    *,
    output: dict[str, str],
    input_hashes: dict[str, str],
) -> dict[str, object]:
    """Keep raw requested output distinct from the later retained native output."""
    request = reader.read(config.get("request"))
    source, preset = reference(config.get("source")), reference(config.get("preset"))
    if set(request) != {
        "batchName",
        "bundleID",
        "sourcePath",
        "presetPath",
        "outputDirectory",
        "outputName",
    }:
        raise ValueError("Unsupported original split native request")
    directory = Path(string(request.get("outputDirectory")))
    name = string(request.get("outputName"))
    if (
        request.get("sourcePath") != source["path"]
        or request.get("presetPath") != preset["path"]
        or not directory.is_absolute()
        or str(directory) != request["outputDirectory"]
        or ".." in directory.parts
        or Path(name).name != name
        or input_hashes.get(output["path"]) != output["sha256"]
    ):
        raise ValueError("Original split request source/output binding differs")
    string(request.get("batchName"))
    string(request.get("bundleID"))
    if config.get("payloadContract") == "recorded-native-submit-status-v1":
        submitted, observed = reader.read(config.get("submit")), reader.read(config.get("status"))
        if (
            set(submitted)
            != {
                "version",
                "submittedOnce",
                "observedUTC",
                "jobID",
                "batchID",
                "completionVerified",
                "exitStatus",
                "outputPath",
                "retrySafe",
                "stdout",
                "truncated",
            }
            or set(observed)
            != {
                "observedUTC",
                "completionVerified",
                "exitStatus",
                "retrySafe",
                "truncated",
                "stdout",
                "explicitSuccessful",
                "jobID",
                "batchID",
                "encodingWallNativeSeconds",
                "outputPath",
                "outputBytes",
                "newAACEncodingPasses",
                "fullAVVerified",
                "allPTSVerified",
            }
            or submitted.get("version") != 1
            or submitted.get("submittedOnce") is not True
            or observed.get("explicitSuccessful") is not True
            or any(
                record.get("exitStatus") != 0 or record.get("truncated") is not False
                for record in (submitted, observed)
            )
            or any(
                record.get("completionVerified") is not False
                or record.get("retrySafe") is not False
                for record in (submitted, observed)
            )
            or observed.get("outputPath") != submitted.get("outputPath")
            or integer(observed.get("outputBytes")) <= 0
            or observed.get("newAACEncodingPasses") != 0
            or observed.get("fullAVVerified") is not False
            or observed.get("allPTSVerified") is not False
        ):
            raise ValueError("Unsupported original recorded native status payload")
        string(submitted.get("observedUTC"))
        string(observed.get("observedUTC"))
        integer(observed.get("encodingWallNativeSeconds"))
    elif "payloadContract" in config:
        raise ValueError("Unsupported original native payload contract")
    else:
        submitted = official_payload(reader.read(config.get("submit"), native_submit=True))
        observed = official_payload(reader.read(config.get("status"), native_status=True))
    batch = mapping(mapping(json.loads(string(submitted.get("stdout")))).get("batch"))
    jobs = rows(batch.get("jobs"))
    statuses: object = json.loads(string(observed.get("stdout")))
    if len(jobs) != 1 or not isinstance(statuses, list) or len(statuses) != 1:
        raise ValueError("Ambiguous original split native job")
    job = mapping(mapping(statuses[0]).get("jobStatus"))
    job_id, batch_id = string(jobs[0].get("jobID")), string(batch.get("batchID"))
    if config.get("payloadContract") == "recorded-native-submit-status-v1" and any(
        row.get("jobID") != job_id or row.get("batchID") != batch_id
        for row in (submitted, observed)
    ):
        raise ValueError("Original recorded native UUID disagrees with stdout")
    reported = Path(string(submitted.get("outputPath")))
    if (
        job.get("jobid") != job_id
        or job.get("batchid") != batch_id
        or job.get("status") != "Successful"
        or job.get("percentComplete") != "100"
        or reported.name != name
        or reported.parent.parent != directory
        or ".." in reported.parts
        or not reported.is_absolute()
    ):
        raise ValueError("Original split native UUID/status/requested output differs")
    if "outputEvidence" in config:
        evidence_ref = reference(config["outputEvidence"])
        evidence = reader.read(evidence_ref)
        same(evidence.get("source"), source)
        same(evidence.get("movie"), output)
        if (
            evidence.get("encodingPasses") != 1
            or input_hashes.get(evidence_ref["path"]) != evidence_ref["sha256"]
        ):
            raise ValueError("Original whole audio evidence is not an input")
        reference(evidence.get("nativeStatus"))
    else:
        # Video submission reports the actual path. The preparation receipt binds
        # the original source/preset; no late output SHA is inserted into it.
        if str(reported) != output["path"]:
            raise ValueError("Original video output path differs")
        receipt_ref = reference(config.get("requestReceipt"))
        receipt = reader.read(receipt_ref)
        pins = hashes(receipt.get("inputs"))
        if (
            receipt.get("version") != 1
            or receipt.get("kind") != "whole-compressor-request"
            or receipt.get("wholeMaster") != source["path"]
            or pins.get(source["path"]) != source["sha256"]
            or pins.get(preset["path"]) != preset["sha256"]
        ):
            raise ValueError("Original video request source/preset pins differ")
    return {
        "oneWholeJobRecorded": True,
        "jobUUID": job_id,
        "batchUUID": batch_id,
        "officialSuccessful": True,
        "requestedOutputName": name,
        "submittedOutputPath": str(reported),
        "independentlyRecordedOutput": output,
        "dispatchOutputMatchesRetained": str(reported) == output["path"],
        "nativeProviderBeforeAfterVerified": False,
        "originalRawRequestExecutionFingerprintVerified": False,
        "currentCryptoVerified": False,
        "qualityApproved": False,
        "globalDuplicateAbsenceVerified": False,
    }


def compressor_execution(
    *,
    submit: dict[str, object],
    status: dict[str, object],
    request: dict[str, object],
    success: dict[str, object],
    output_path: str,
) -> dict[str, object]:
    if request.get("version") != 1 or request.get("kind") != "whole-compressor-request":
        raise ValueError("Unsupported whole request evidence")
    submission = official_payload(submit)
    observed = official_payload(status)
    batch = mapping(mapping(json.loads(string(submission.get("stdout")))).get("batch"))
    jobs = batch.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise ValueError("Expected one whole-video job")
    job_id = string(mapping(jobs[0]).get("jobID"))
    batch_id = string(batch.get("batchID"))
    statuses: object = json.loads(string(observed.get("stdout")))
    if not isinstance(statuses, list) or len(statuses) != 1:
        raise ValueError("Ambiguous official job status")
    job = mapping(mapping(statuses[0]).get("jobStatus"))
    if (
        job.get("jobid") != job_id
        or job.get("batchid") != batch_id
        or job.get("status") != "Successful"
        or (job.get("percentComplete") != "100")
        or (submission.get("outputPath") != output_path)
        or (success.get("jobUUID") != job_id)
        or (success.get("batchUUID") != batch_id)
        or (mapping(success.get("output")).get("path") != output_path)
    ):
        raise ValueError("Native UUID, state or output mismatch")
    source = mapping(success.get("source"))
    master = string(request.get("wholeMaster"))
    if source.get("path") != master or hashes(request.get("inputs")).get(master) != digest(
        source.get("sha256")
    ):
        raise ValueError("Whole input differs from request provenance")
    return {
        "oneWholeJobRecorded": True,
        "jobUUID": job_id,
        "batchUUID": batch_id,
        "officialSuccessful": True,
        "qualityApproved": False,
        "adapterCompletionVerified": observed.get("completionVerified"),
        "globalDuplicateAbsenceVerified": False,
    }


def whole_records(
    refs: object, inputs: dict[str, str], source: dict[str, object], *, reader: MetadataReader
) -> dict[str, object]:
    read = reader.read
    "Bind original completion records through the final stage input fingerprint."
    if refs is None:
        return {"completionRecordsBound": False, "pending": "Original four checkpoint refs absent"}
    records = mapping(refs)
    if set(records) != {"join", "clock", "metadata", "pcm"}:
        raise ValueError("Whole checkpoint inventory incomplete")
    paths = [string(mapping(value).get("path")) for value in records.values()]
    if len(set(paths)) != 4:
        raise ValueError("Whole phases require distinct original checkpoints")
    for role, value in records.items():
        ref = mapping(value)
        if inputs.get(string(ref.get("path"))) != digest(ref.get("sha256")):
            raise ValueError("Whole checkpoint was not an original final-stage input")
        record = read(ref)
        digest(record.get("fingerprint"))
        outputs = hashes(record.get("outputs"))
        if record.get("status") != "complete":
            raise ValueError("Whole checkpoint incomplete")
        if role == "join" and outputs != {string(source.get("path")): digest(source.get("sha256"))}:
            raise ValueError("Compressor source differs from completed whole join")
    return {
        "completionRecordsBound": True,
        "completedPhaseCount": 4,
        "currentOutputsCryptoVerified": False,
        "individualWholeProducerFingerprintsReconstructed": False,
    }


if __name__ == "__main__":
    main()
