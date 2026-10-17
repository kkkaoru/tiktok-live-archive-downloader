"""Explicit native-FCP candidate assessment; not connected to edit_completion yet.

Current stat identity, original execution and strict quality are separate. No
media/model hashing, rendering, recovery or historical backend execution occurs.
"""

import argparse
import json
import os
import stat
import sys
from pathlib import Path

from replay import delivery_completion_v2 as measured
from replay import native_delivery_evidence as evidence
from replay import native_derived_revision as derived
from replay.core import ReplayError
from replay.edit_completion import REQUIRED_STAGES, load_config
from replay.input_hash_snapshot import InputHashSnapshot
from replay.stages import StageError, file_digest

BACKEND = "native-fcp-partitioned-whole"
INDEX_KEYS = {
    "version",
    "backend",
    "recording",
    "configuration",
    "selection",
    "descriptor",
    "identities",
    "master",
}
IDENTITIES = {"source", "movie", "srt", "semanticXML", "template", "modelConfig", "modelWeights"}
OPTIONAL_KEYS = {"observation", "measurements", "derivedRevision"}
MEASURED_STAGES = {
    "audioQuality": "bgm_removal",
    "cutExecution": "silence_cut",
    "maskQuality": "source_caption_mask",
}
STAGE_DOMAINS = {
    "bgm_removal": ("voice", "whole-assembly", "final-delivery"),
    "silence_cut": ("cuts", "native-parts", "whole-assembly"),
    "source_caption_mask": ("cleanup", "native-parts"),
    "transcription": ("asr", "caption-layout", "final-delivery"),
    "dynamic_captions": ("caption-layout", "native-parts", "whole-assembly", "final-delivery"),
}
STRICT_GAPS = {
    "bgm_removal": [
        "separation model/source lineage",
        "post-AAC full protected-speech and music measurements",
    ],
    "silence_cut": ["protected quiet decisions bound to actual selected execution"],
    "source_caption_mask": [
        "selected-frame local repair region/compositor provenance",
        "complete residual/readability/instability measurements",
    ],
    "transcription": ["canonical SRT equivalence to all actual displayed caption states"],
    "dynamic_captions": [
        "all native painted caption states/font/bounds/head-near measurements",
        "exact editable FCP import observation",
    ],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity(value: object) -> dict[str, object]:
    rows = evidence.fingerprint_rows([value])
    return next(iter(rows.values()))


def reference_of(row: dict[str, object]) -> dict[str, str]:
    return {"path": evidence.string(row.get("path")), "sha256": evidence.digest(row.get("sha256"))}


def observe_identity(row: dict[str, object]) -> None:
    """Compare current metadata, never infer fresh content SHA from stable stat."""
    path = Path(evidence.string(row["path"]))
    require(path.resolve(strict=True) == path, "Current identity path is linked")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), "Current identity is not a regular file")
    # The device number is renumbered by remounts and differs on another volume, so it is
    # recorded as evidence but not compared; size, inode and both times still bind the file.
    actual = (info.st_size, info.st_ino, info.st_mtime_ns, info.st_ctime_ns)
    expected = tuple(evidence.integer(row[key]) for key in ("bytes", "inode", "mtimeNs", "ctimeNs"))
    require(actual == expected, "Current file identity differs from retained evidence")


def child(root: Path, value: object) -> Path:
    relative = Path(evidence.string(value))
    require(
        not relative.is_absolute() and ".." not in relative.parts and str(relative) == value,
        "Selection requires a canonical relative path",
    )
    return root / relative


def master_phase(
    value: object, context: dict[str, object], reader: evidence.MetadataReader
) -> dict[str, object]:
    descriptor = evidence.mapping(value)
    require(set(descriptor) == {"phase", "reference"}, "Unsupported master phase descriptor")
    phase = descriptor.get("phase")
    if phase == "historical":
        evidence.same(descriptor.get("reference"), context.get("wholeMaster"))
        return {"phase": phase, "freshWholeVerified": False, "currentMaterialization": None}
    require(phase == "retired", "Unsupported master phase")
    outcome = reader.read(descriptor.get("reference"))
    require(
        outcome.get("version") == 1
        and outcome.get("kind") == "retired-reconstructable-whole-master",
        "Unsupported retirement outcome",
    )
    evidence.same(outcome.get("master"), context.get("wholeMaster"))
    require(
        outcome.get("masterMaterialized") is False
        and outcome.get("freshWholeVerified") is False
        and outcome.get("byteIdenticalRebuildVerified") is False,
        "Retired master cannot be freshly materialized",
    )
    intent = reader.read(outcome.get("intent"))
    deletion = reader.read(outcome.get("deletion"))
    common = {
        "version",
        "kind",
        "request",
        "master",
        "masterMaterialized",
        "deletionAuthorized",
        "historicalWholeVerified",
        "freshWholeVerified",
        "reconstructableInputsVerified",
        "byteIdenticalRebuildVerified",
        "rebuildPolicy",
        "domain",
        "inputs",
        "qualityApproved",
        "syncApproved",
        "deliveryApproved",
    }
    require(
        set(intent) == common | {"masterIdentity", "preparedAtNs"}
        and set(outcome) == common | {"intent", "deletion"}
        and set(deletion)
        == {
            "version",
            "intent",
            "master",
            "masterIdentity",
            "removed",
            "operation",
            "observedAtNs",
            "disposition",
        },
        "Unsupported immutable retirement schema",
    )
    for document in (intent, outcome):
        require(
            type(document.get("version")) is int
            and document.get("deletionAuthorized") is False
            and document.get("historicalWholeVerified") is True
            and document.get("reconstructableInputsVerified") is True
            and document.get("freshWholeVerified") is False
            and document.get("byteIdenticalRebuildVerified") is False
            and document.get("rebuildPolicy") == "new-nonoverwriting-revision"
            and all(
                document.get(key) is False
                for key in ("qualityApproved", "syncApproved", "deliveryApproved")
            ),
            "Unsupported retirement evidence claims",
        )
    require(
        intent.get("version") == 1
        and intent.get("kind") == "whole-master-retirement-intent"
        and intent.get("masterMaterialized") is True
        and type(deletion.get("version")) is int
        and deletion.get("version") == 1
        and deletion.get("removed") is True
        and deletion.get("operation") == "exact-path-unlink",
        "Incomplete immutable retirement chain",
    )
    evidence.same(intent.get("master"), context.get("wholeMaster"))
    evidence.same(deletion.get("master"), context.get("wholeMaster"))
    evidence.same(deletion.get("intent"), outcome.get("intent"))
    evidence.same(intent.get("request"), outcome.get("request"))
    require(
        intent.get("masterIdentity") == deletion.get("masterIdentity"),
        "Retirement source identity differs",
    )
    identity = intent.get("masterIdentity")
    require(
        isinstance(identity, list)
        and len(identity) == 6
        and all(type(item) is int and item >= 0 for item in identity)
        and identity[2] > 0
        and identity[5] == 1,
        "Invalid retired master identity",
    )
    require(
        intent.get("domain") == outcome.get("domain"),
        "Retirement reconstruction lineage differs",
    )
    retirement_inputs(intent, deletion, outcome, reader)
    return {
        "phase": phase,
        "recordedRetirementBound": True,
        "freshWholeVerified": False,
        "currentMaterialization": None,
        "currentReconstructibilityVerified": False,
    }


def retirement_inputs(
    intent: dict[str, object],
    deletion: dict[str, object],
    outcome: dict[str, object],
    reader: evidence.MetadataReader,
) -> None:
    """Check the original producer's exact before/after inventory transformation."""

    def inventory(value: object) -> dict[str, dict[str, str]]:
        refs = [evidence.reference(item) for item in evidence.rows(value)]
        result = {ref["path"]: ref for ref in refs}
        require(0 < len(refs) <= 1024 and len(result) == len(refs), "Invalid retirement inventory")
        return result

    before, after = inventory(intent.get("inputs")), inventory(outcome.get("inputs"))
    master = evidence.reference(intent.get("master"))
    request = evidence.reference(intent.get("request"))
    require(before.get(master["path"]) == master, "Prepared inventory lacks original master")
    require(before.get(request["path"]) == request, "Prepared inventory lacks original request")
    del before[master["path"]]
    disposition = reader.read(deletion.get("disposition"))
    require(
        set(disposition)
        == {"version", "master", "owner", "ownerReleased", "observedAtNs", "references"}
        and disposition.get("version") == 1
        and disposition.get("ownerReleased") is True,
        "Invalid final retirement disposition",
    )
    evidence.same(disposition.get("master"), master)
    evidence.string(disposition.get("owner"))
    prepared, removed, observed = (
        intent.get("preparedAtNs"),
        deletion.get("observedAtNs"),
        disposition.get("observedAtNs"),
    )
    require(
        type(prepared) is int
        and type(removed) is int
        and type(observed) is int
        and 0 < prepared <= removed
        and 0 < observed <= removed,
        "Invalid retirement chronology",
    )
    roles = evidence.mapping(disposition.get("references"))
    require(
        set(roles) == {"current", "queued", "resume", "fcp", "viewer", "diagnostic"},
        "Missing retirement reference scopes",
    )
    extras: list[object] = [
        outcome.get("intent"),
        outcome.get("deletion"),
        deletion.get("disposition"),
    ]
    for value in roles.values():
        row = evidence.mapping(value)
        require(
            set(row) == {"scope", "resolved", "requiresMaterializedMaster", "evidence"}
            and row.get("resolved") is True
            and row.get("requiresMaterializedMaster") is False,
            "Unresolved retirement reference",
        )
        evidence.string(row.get("scope"))
        extras.extend(evidence.rows(row.get("evidence")))
    for value in extras:
        ref = evidence.reference(value)
        require(ref["path"] != master["path"], "Retired inventory still requires master")
        require(
            ref["path"] not in before or before[ref["path"]] == ref, "Conflicting retirement pin"
        )
        before[ref["path"]] = ref
    require(before == after, "Retirement reconstruction lineage differs")


def bind_index(
    data: dict[str, object], *, root: Path, delivery: Path, reader: evidence.MetadataReader
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, object]]:
    require(
        INDEX_KEYS <= set(data) <= INDEX_KEYS | OPTIONAL_KEYS
        and type(data.get("version")) is int
        and data["version"] == 3
        and data.get("backend") == BACKEND
        and data.get("recording") == root.name,
        "Unsupported native v3 index",
    )
    config_ref = evidence.reference(data.get("configuration"))
    require(
        Path(config_ref["path"]) == root / "edit-config.json",
        "Index configuration belongs to another recording",
    )
    config = reader.read(config_ref)
    require(
        config.get("stages") == dict.fromkeys(REQUIRED_STAGES, True),
        "Every required stage must remain enabled",
    )
    descriptor = reader.read(data.get("descriptor"))
    context = evidence.mapping(descriptor.get("context"))
    raw = evidence.mapping(data.get("identities"))
    require(set(raw) == IDENTITIES, "Missing or unknown current identity role")
    ids = {key: identity(value) for key, value in raw.items()}
    require(
        len({row["path"] for row in ids.values()}) == len(ids), "Duplicate current identity roles"
    )
    for key, target_key in (
        ("source", "source"),
        ("movie", "finalMovie"),
        ("semanticXML", "semanticXML"),
    ):
        if key == "movie" and "derivedRevision" in data:
            derived.validate(
                data["derivedRevision"],
                descriptor=data["descriptor"],
                movie=reference_of(ids[key]),
                reader=reader,
            )
        else:
            evidence.same(reference_of(ids[key]), context.get(target_key))
    require(
        config.get("source") == ids["source"]["path"]
        and config.get("source_size") == ids["source"]["bytes"],
        "Configuration source identity differs",
    )
    if "source_sha256" in config:
        require(
            config["source_sha256"] == ids["source"]["sha256"],
            "Configuration source digest differs",
        )
    storage = Path(evidence.string(config.get("storage_root")))
    require(
        storage.is_absolute() and storage.resolve(strict=True) == storage,
        "Invalid configured storage scope",
    )
    selection_ref = evidence.reference(data.get("selection"))
    require(
        Path(selection_ref["path"]) == storage / "evidence" / "candidate-selection.json",
        "Selection is outside recording storage",
    )
    selected = reader.read(selection_ref)
    require(
        selected.get("version") == 1 and selected.get("status") == "unverified-candidate",
        "Unsupported selected candidate header",
    )
    require(
        selected.get("editConfigSha256") == config_ref["sha256"]
        and selected.get("fcpxml") == ids["semanticXML"]["path"]
        and selected.get("fcpxmlSha256") == ids["semanticXML"]["sha256"],
        "Selected configuration/semantic revision differs",
    )
    named = child(storage, selected.get("movie"))
    movie_target = child(storage, selected.get("movieTarget"))
    require(
        delivery.absolute() == named
        and movie_target == Path(evidence.string(ids["movie"]["path"]))
        and named.resolve(strict=True) == movie_target,
        "Selected movie/target identity differs",
    )
    require(
        selected.get("movieSize") == ids["movie"]["bytes"]
        and selected.get("movieMtimeNanoseconds") == ids["movie"]["mtimeNs"],
        "Selected movie observation differs",
    )
    require(
        child(storage, selected.get("transcript")) == Path(evidence.string(ids["srt"]["path"]))
        and selected.get("transcriptSha256") == ids["srt"]["sha256"],
        "Selected transcript differs",
    )
    domains = evidence.mapping(descriptor.get("domains"))
    captions = evidence.mapping(domains.get("caption-layout"))
    head = reader.read(captions.get("headConfig"))
    evidence.same(reference_of(ids["template"]), head.get("template"))
    model = evidence.hashes(evidence.mapping(domains.get("asr")).get("modelPins"))
    require(
        model
        == {
            evidence.string(ids[k]["path"]): evidence.digest(ids[k]["sha256"])
            for k in ("modelConfig", "modelWeights")
        },
        "Original model pins differ from index",
    )
    phase = master_phase(data.get("master"), context, reader)
    return descriptor, ids, phase


def observation_rows(data: dict[str, object], ids: dict[str, dict[str, object]]) -> None:
    """A persisted observation is historical current observation, never fresh crypto."""
    if "observation" not in data:
        return
    observed = evidence.mapping(data["observation"])
    require(
        set(observed) == {"version", "kind", "expected", "identities", "initialAndTerminalSHA"}
        and observed.get("version") == 1
        and observed.get("kind") == "current-identity-observation"
        and observed.get("initialAndTerminalSHA") is True,
        "Invalid current observation contract",
    )
    require(observed.get("identities") == ids, "Observation identity rows differ")
    require(
        observed.get("expected") == {role: reference_of(row) for role, row in ids.items()},
        "Original expected pins differ from current observation",
    )


def measurement_failures(
    data: dict[str, object],
    *,
    descriptor: dict[str, object],
    root: Path,
    ids: dict[str, dict[str, object]],
    reader: evidence.MetadataReader,
    verify_content: bool,
) -> tuple[dict[str, list[str]], list[str]]:
    failures = {stage: list(gaps) for stage, gaps in STRICT_GAPS.items()}
    refs = evidence.mapping(data.get("measurements", {}))
    require(set(refs) <= set(MEASURED_STAGES), "Unsupported native quality measurement")
    if "derivedRevision" in data:
        require("audioQuality" not in refs, "Conflicting derived audio measurement")
        failures["dynamic_captions"].append(
            "onset-held vertical placement and Y-neutral entrance actual paint"
        )
        if verify_content:
            # bind_index recomputes the pinned derived proof before current
            # movie hashing. Original model lineage remains a separate failure.
            failures["bgm_removal"] = ["separation model/source lineage"]
    if not verify_content:
        return failures, []
    config = reader.read(data["configuration"])
    store = measured.Evidence(
        root,
        Path(evidence.string(ids["source"]["path"])),
        Path(evidence.string(config["storage_root"])),
    )
    passed: list[str] = []
    for kind, ref in refs.items():
        stage = MEASURED_STAGES[kind]
        try:
            receipt = reader.read(ref)
            measured.validate_native_measurement(
                kind,
                store=store,
                receipt=receipt,
                descriptor=Path(evidence.reference(data["descriptor"])["path"]),
                context=evidence.mapping(descriptor["context"]),
                movie=Path(evidence.string(ids["movie"]["path"])),
            )
            if kind == "audioQuality":
                # V2's numerical post-AAC validator does not establish the
                # original separation model. Do not clear that separate guard.
                failures[stage] = ["separation model/source lineage"]
            else:
                failures.pop(stage)
                passed.append(stage)
        except (ReplayError, ValueError, OSError, ZeroDivisionError) as error:
            failures[stage] = [f"{kind}: {error}"]
    store.verify_unchanged()
    return failures, passed


def assess(
    index: object, *, root: Path, delivery: Path, verify_content: bool = False
) -> dict[str, object]:
    """Validate the selected native candidate and report concrete strict stage gaps.

    The caller must retain normal edit-config/acquisition authorization checks.
    No quality receipt contract is invented here; strict completion remains false.
    """
    require(root.is_absolute() and root.resolve(strict=True) == root, "Edit root must be canonical")
    reader = evidence.MetadataReader()
    data = reader.read(index)
    descriptor, ids, phase = bind_index(data, root=root, delivery=delivery, reader=reader)
    observation_rows(data, ids)
    snapshot = (
        InputHashSnapshot.capture(
            (Path(evidence.string(row["path"])), evidence.digest(row["sha256"]))
            for row in ids.values()
        )
        if verify_content
        else None
    )
    for row in ids.values():
        observe_identity(row)
    recorded = evidence.assess(descriptor, reader=reader)
    domains = evidence.mapping(recorded["domains"])
    stages: dict[str, object] = {}
    failures, measured_stages = measurement_failures(
        data,
        descriptor=descriptor,
        root=root,
        ids=ids,
        reader=reader,
        verify_content=verify_content,
    )
    for stage, names in STAGE_DOMAINS.items():
        missing = [
            name
            for name in names
            if evidence.mapping(domains[name]).get("stageApplied") is not True
        ]
        stages[stage] = {
            "stageApplied": not missing,
            "recordedDomains": list(names),
            "missingDomains": missing,
        }
        if missing:
            failures.setdefault(stage, []).extend(
                "missing recorded domain: " + name for name in missing
            )
    if snapshot is not None:
        snapshot.finish()
    for row in ids.values():
        observe_identity(row)
    return {
        "version": 3,
        "backend": BACKEND,
        "candidateAvailable": True,
        "currentStatIdentityVerified": True,
        "currentCryptoVerified": snapshot is not None,
        "recordedRequiredStagesComplete": all(
            evidence.mapping(stage)["stageApplied"] is True for stage in stages.values()
        ),
        "measuredStagesVerified": measured_stages,
        "recordedStages": stages,
        "evidence": recorded,
        "derivedRevision": data.get("derivedRevision"),
        "master": phase,
        "strictFailures": failures,
        "strictComplete": not failures and snapshot is not None,
        "qualityApproved": False,
        "syncApproved": False,
        "narrativeApproved": False,
        "outcome": "available-unverified-native-candidate",
        "productionDispatchConnected": True,
    }


def prepare_observation(index: object, *, root: Path, delivery: Path, output: Path) -> Path:
    """Publish new current rows against original pins, without rewriting provenance."""
    require(
        output.is_absolute() and output.parent.resolve(strict=True) == output.parent,
        "Observation output parent must be canonical",
    )
    require(
        output.is_relative_to(root.parents[1] / "edits/work"),
        "Observation output must be repository work data",
    )
    require(not output.exists() and not output.is_symlink(), "Observation output exists")
    require(output.suffix == ".json", "Observation output must be JSON")
    reader = evidence.MetadataReader()
    data = reader.read(index)
    raw = evidence.mapping(data.get("identities"))
    require(set(raw) == IDENTITIES, "Observation requires seven original expected roles")
    expected = {role: evidence.reference(ref) for role, ref in raw.items()}
    # Admission uses fresh stat only as proposed rows; bind_index checks every
    # expected digest against the original descriptor before any content read.
    ids: dict[str, dict[str, object]] = {}
    for role, ref in expected.items():
        path = Path(ref["path"])
        require(path.resolve(strict=True) == path, "Linked observation input")
        info = path.stat()
        require(stat.S_ISREG(info.st_mode), "Observation input must be a regular file")
        ids[role] = {
            **ref,
            "bytes": info.st_size,
            "device": info.st_dev,
            "inode": info.st_ino,
            "mtimeNs": info.st_mtime_ns,
            "ctimeNs": info.st_ctime_ns,
        }
    data["identities"] = ids
    bind_index(data, root=root, delivery=delivery, reader=reader)
    snapshot = InputHashSnapshot.capture(
        (Path(ref["path"]), ref["sha256"]) for ref in expected.values()
    )
    snapshot.finish()
    for row in ids.values():
        observe_identity(row)
    data["observation"] = {
        "version": 1,
        "kind": "current-identity-observation",
        "expected": expected,
        "identities": ids,
        "initialAndTerminalSHA": True,
    }
    descriptor_fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor_fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return output


def verify_generated_observation(
    request: object, generated: Path, *, root: Path, delivery: Path
) -> dict[str, object]:
    """Reconcile a generated index against its pinned original request, read-only.

    The workflow owns the generated output SHA. This verifier never regenerates
    an observation and never infers fresh media crypto from the retained index.
    """
    require(
        generated.is_absolute() and generated.resolve(strict=True) == generated,
        "Generated observation must be canonical and unlinked",
    )
    info = generated.stat()
    require(
        generated.suffix == ".json"
        and stat.S_ISREG(info.st_mode)
        and 0 < info.st_size <= evidence.MAX_RECORD_BYTES,
        "Generated observation must be bounded JSON metadata",
    )
    reader = evidence.MetadataReader()
    original = reader.read(request)
    require("observation" not in original, "Original request must not contain an observation")
    reference = {"path": str(generated), "sha256": file_digest(generated)}
    current = reader.read(reference)
    require(
        set(current) == set(original) | {"observation"}, "Generated schema differs from request"
    )
    require(
        {key: value for key, value in original.items() if key != "identities"}
        == {
            key: value for key, value in current.items() if key not in {"identities", "observation"}
        },
        "Generated index differs from original request",
    )
    expected = {
        role: evidence.reference(ref)
        for role, ref in evidence.mapping(original.get("identities")).items()
    }
    require(
        evidence.mapping(current.get("observation")).get("expected") == expected,
        "Generated original expected pins differ",
    )
    result = assess(reference, root=root, delivery=delivery)
    # Keep request/output bytes stable across the complete metadata/domain read.
    reader.read(request)
    require(file_digest(generated) == reference["sha256"], "Generated observation changed")
    result["observationVerified"] = True
    result["originalRequest"] = evidence.reference(request)
    result["generatedIndex"] = reference
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native FCP recorded and strict completion")
    parser.add_argument("index", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--observe-output", type=Path)
    mode.add_argument("--verify-content", action="store_true")
    mode.add_argument("--verify-observation", action="store_true")
    mode.add_argument("--verify-generated-observation", type=Path)
    args = parser.parse_args(argv)
    reference = {"path": str(args.index), "sha256": args.sha256}
    try:
        # Same acquisition/storage authority as the mandatory edit-completion gate.
        load_config(args.root)
        if args.verify_generated_observation is not None:
            result = verify_generated_observation(
                reference, args.verify_generated_observation, root=args.root, delivery=args.delivery
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.observe_output is not None:
            path = prepare_observation(
                reference, root=args.root, delivery=args.delivery, output=args.observe_output
            )
            print(json.dumps({"path": str(path), "sha256": file_digest(path)}))
            return 0
        result = assess(
            reference, root=args.root, delivery=args.delivery, verify_content=args.verify_content
        )
        if args.verify_observation:
            require(
                "observation" in evidence.MetadataReader().read(reference),
                "Missing current observation provenance",
            )
            result["observationVerified"] = True
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if args.verify_observation or result["strictComplete"] is True else 2
    except (ReplayError, StageError, ValueError, OSError) as error:
        print(f"NATIVE_COMPLETION_ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
