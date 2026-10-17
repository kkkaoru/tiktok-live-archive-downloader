import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import delivery_completion_v2 as measured
from replay import native_delivery_completion as c
from replay import native_delivery_evidence as e
from replay import native_derived_revision as derived
from replay.edit_completion import REQUIRED_STAGES
from replay.input_hash_snapshot import InputHashSnapshot
from replay.stages import StageError


def put(path: Path, data: dict[str, object]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(data) + "\n").encode()
    path.write_bytes(content)
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def row(path: Path) -> dict[str, object]:
    info = path.stat()
    return {
        "path": str(path),
        "bytes": info.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtimeNs": info.st_mtime_ns,
        "ctimeNs": info.st_ctime_ns,
    }


@dataclass
class Case:
    root: Path
    delivery: Path
    index: dict[str, object]
    descriptor: dict[str, object]
    config: dict[str, object]
    selected: dict[str, object]
    ids: dict[str, dict[str, object]]

    def bind(self) -> dict[str, str]:
        self.index["configuration"] = put(self.root / "edit-config.json", self.config)
        self.selected["editConfigSha256"] = e.reference(self.index["configuration"])["sha256"]
        self.index["selection"] = put(
            self.root / "storage/evidence/candidate-selection.json", self.selected
        )
        self.index["descriptor"] = put(self.root / "descriptor.json", self.descriptor)
        return put(self.root / "index.json", self.index)


@pytest.fixture
def case(tmp_path: Path) -> Case:
    root = tmp_path / "recording"
    root.mkdir()
    storage = root / "storage"
    storage.mkdir()
    paths = {key: root / key for key in c.IDENTITIES}
    paths.update(movie=storage / "final.mov", srt=storage / "final.srt")
    for path in paths.values():
        path.write_bytes(b"bounded synthetic bytes\n")
    ids = {key: row(path) for key, path in paths.items()}
    config: dict[str, object] = {
        "version": 1,
        "source": str(paths["source"]),
        "source_size": paths["source"].stat().st_size,
        "storage_root": str(storage),
        "stages": dict.fromkeys(REQUIRED_STAGES, True),
    }
    head = put(root / "head.json", {"template": c.reference_of(ids["template"])})
    context: dict[str, object] = {
        "source": c.reference_of(ids["source"]),
        "finalMovie": c.reference_of(ids["movie"]),
        "semanticXML": c.reference_of(ids["semanticXML"]),
        "wholeMaster": {"path": str(root / "absent-master.mov"), "sha256": "b" * 64},
    }
    descriptor: dict[str, object] = {
        "version": 1,
        "backend": c.BACKEND,
        "context": context,
        "domains": {
            "caption-layout": {"headConfig": head},
            "asr": {
                "modelPins": {
                    str(paths[k]): ids[k]["sha256"] for k in ("modelConfig", "modelWeights")
                }
            },
        },
    }
    selected: dict[str, object] = {
        "version": 1,
        "status": "unverified-candidate",
        "movie": "final.mov",
        "movieTarget": "final.mov",
        "transcript": "final.srt",
        "transcriptSha256": ids["srt"]["sha256"],
        "fcpxml": ids["semanticXML"]["path"],
        "fcpxmlSha256": ids["semanticXML"]["sha256"],
        "movieSize": ids["movie"]["bytes"],
        "movieMtimeNanoseconds": ids["movie"]["mtimeNs"],
    }
    index: dict[str, object] = {
        "version": 3,
        "backend": c.BACKEND,
        "recording": root.name,
        "configuration": {},
        "selection": {},
        "descriptor": {},
        "identities": ids,
        "master": {"phase": "historical", "reference": context["wholeMaster"]},
    }
    return Case(root, paths["movie"], index, descriptor, config, selected, ids)


def recorded_result(*, missing: str | None = None) -> dict[str, object]:
    return {
        "domains": {name: {"stageApplied": name != missing} for name in e.DOMAINS},
        "qualityApproved": False,
    }


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate new index/dispatch boundaries. Existing reader owns 173 real contract regressions.
    monkeypatch.setattr(e, "assess", lambda config, *, reader: recorded_result())


def test_native_no_alpha_is_available_not_strict(case: Case, recorded: None) -> None:
    result = c.assess(case.bind(), root=case.root, delivery=case.delivery)
    assert result["candidateAvailable"] is True
    assert result["strictComplete"] is False
    assert result["currentCryptoVerified"] is False
    assert result["outcome"] == "available-unverified-native-candidate"
    assert set(e.mapping(result["strictFailures"])) == set(REQUIRED_STAGES)
    assert (
        e.mapping(e.mapping(result["recordedStages"])["dynamic_captions"])["stageApplied"] is True
    )
    assert e.mapping(result["master"])["currentMaterialization"] is None


@pytest.mark.parametrize("verify_content", [False, True])
def test_derived_revision_keeps_original_domains_and_quality_gaps(
    case: Case, recorded: None, monkeypatch: pytest.MonkeyPatch, verify_content: bool
) -> None:
    context = e.mapping(case.descriptor["context"])
    context["finalMovie"] = {"path": str(case.root / "old.mov"), "sha256": "b" * 64}
    case.descriptor["context"] = context
    case.index["derivedRevision"] = {"request": "synthetic"}
    monkeypatch.setattr(derived, "validate", lambda *args, **kwargs: {"validated": True})
    result = c.assess(
        case.bind(), root=case.root, delivery=case.delivery, verify_content=verify_content
    )
    assert result["recordedRequiredStagesComplete"] is True
    assert result["strictComplete"] is False
    assert result["qualityApproved"] is False
    gaps = e.mapping(result["strictFailures"])
    if verify_content:
        assert gaps["bgm_removal"] == ["separation model/source lineage"]
    else:
        assert gaps["bgm_removal"] == [
            "separation model/source lineage",
            "post-AAC full protected-speech and music measurements",
        ]
    caption_gaps = gaps["dynamic_captions"]
    assert isinstance(caption_gaps, list)
    assert "onset-held vertical placement and Y-neutral entrance actual paint" in caption_gaps


def test_derived_revision_does_not_skip_proof_validation(case: Case) -> None:
    case.index["derivedRevision"] = {"version": 1}
    with pytest.raises(ValueError, match="derived revision"):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


def test_derived_revision_conflicting_audio_measurement_rejected(
    case: Case, recorded: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    case.index["derivedRevision"] = {"request": "synthetic"}
    case.index["measurements"] = {"audioQuality": {}}
    monkeypatch.setattr(derived, "validate", lambda *args, **kwargs: {"validated": True})
    with pytest.raises(ValueError, match="Conflicting derived audio"):
        c.assess(case.bind(), root=case.root, delivery=case.delivery, verify_content=True)


def test_real_reader_failure_is_not_converted_to_success(case: Case) -> None:
    # The intentionally incomplete descriptor must reach the actual canonical reader and reject.
    with pytest.raises(ValueError):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


@pytest.mark.parametrize(
    "role", ["movie", "srt", "source", "semanticXML", "template", "modelConfig", "modelWeights"]
)
def test_changed_current_identity_refused(case: Case, recorded: None, role: str) -> None:
    ref = case.bind()
    Path(e.string(case.ids[role]["path"])).write_bytes(b"changed current contents")
    with pytest.raises(ValueError, match="identity differs"):
        c.assess(ref, root=case.root, delivery=case.delivery)


@pytest.mark.parametrize("role", ["movie", "semanticXML", "template", "modelWeights"])
def test_changed_recorded_digest_refused(case: Case, role: str) -> None:
    case.ids[role]["sha256"] = "c" * 64
    with pytest.raises(ValueError):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 2),
        ("version", True),
        ("backend", "motion-compressor"),
        ("recording", "other"),
        ("qualityApproved", True),
    ],
)
def test_malformed_or_boolean_approval_index(case: Case, key: str, value: object) -> None:
    case.index[key] = value
    with pytest.raises(ValueError, match="v3"):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


def test_incomplete_parts_remain_stage_missing(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        e, "assess", lambda config, *, reader: recorded_result(missing="native-parts")
    )
    result = c.assess(case.bind(), root=case.root, delivery=case.delivery)
    stage = e.mapping(e.mapping(result["recordedStages"])["dynamic_captions"])
    assert stage["stageApplied"] is False
    assert stage["missingDomains"] == ["native-parts"]
    assert result["strictComplete"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("movie", "../escape.mov"),
        ("movieTarget", "/escape.mov"),
        ("movieSize", 1),
        ("movieMtimeNanoseconds", 1),
        ("transcriptSha256", "a" * 64),
        ("transcript", "other.srt"),
        ("fcpxmlSha256", "d" * 64),
        ("status", "complete"),
    ],
)
def test_selection_cannot_rebind_revision(case: Case, field: str, value: object) -> None:
    case.selected[field] = value
    with pytest.raises(ValueError):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


@pytest.mark.parametrize(
    "field,value",
    [("source_size", 1), ("source_sha256", "a" * 64), ("stages", {}), ("storage_root", "relative")],
)
def test_config_source_and_stage_scope(case: Case, field: str, value: object) -> None:
    case.config[field] = value
    with pytest.raises(ValueError):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


def test_missing_identity_and_wrong_caller(case: Case) -> None:
    ref = case.bind()
    with pytest.raises(ValueError):
        c.assess(ref, root=case.root, delivery=case.root / "wrong.mov")
    with pytest.raises(ValueError):
        c.assess(ref, root=Path("relative"), delivery=case.delivery)
    del case.ids["srt"]
    with pytest.raises(ValueError):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


def test_duplicate_identity_and_link_rejected(case: Case) -> None:
    case.ids["srt"] = case.ids["movie"]
    with pytest.raises(ValueError, match="Duplicate"):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


def test_identity_survives_a_renumbered_device(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"real")
    renumbered = row(target) | {"device": 1}
    c.observe_identity(renumbered)


def test_identity_still_rejects_a_changed_inode(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"real")
    changed = row(target) | {"inode": 1}
    with pytest.raises(ValueError, match="differs"):
        c.observe_identity(changed)


def test_identity_link_cannot_replace_retained_file(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"real")
    link = tmp_path / "alias"
    link.symlink_to(target)
    linked = row(target) | {"path": str(link)}
    with pytest.raises(ValueError, match="linked"):
        c.observe_identity(linked)


def test_during_assessment_mutation_refused(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    def mutate(config: dict[str, object], *, reader: e.MetadataReader) -> dict[str, object]:
        case.delivery.write_bytes(b"mutated after first observation")
        return recorded_result()

    monkeypatch.setattr(e, "assess", mutate)
    with pytest.raises(ValueError, match="identity differs"):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


def retirement(case: Case) -> dict[str, str]:
    master = e.mapping(case.descriptor["context"])["wholeMaster"]
    request = {"path": str(case.root / "request.json"), "sha256": "f" * 64}
    intent_data: dict[str, object] = {
        "version": 1,
        "kind": "whole-master-retirement-intent",
        "master": master,
        "request": request,
        "masterIdentity": [1, 2, 3, 4, 5, 1],
        "domain": {"frames": 60},
        "inputs": [master, request, {"path": "/retained", "sha256": "a" * 64}],
        "preparedAtNs": 1,
        "masterMaterialized": True,
        "deletionAuthorized": False,
        "historicalWholeVerified": True,
        "reconstructableInputsVerified": True,
        "freshWholeVerified": False,
        "byteIdenticalRebuildVerified": False,
        "rebuildPolicy": "new-nonoverwriting-revision",
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
    }
    intent = put(case.root / "intent.json", intent_data)
    disposition = put(
        case.root / "disposition.json",
        {
            "version": 1,
            "master": master,
            "owner": "synthetic-owner",
            "ownerReleased": True,
            "observedAtNs": 2,
            "references": {
                role: {
                    "scope": role,
                    "resolved": True,
                    "requiresMaterializedMaster": False,
                    "evidence": [{"path": "/retained", "sha256": "a" * 64}],
                }
                for role in ("current", "queued", "resume", "fcp", "viewer", "diagnostic")
            },
        },
    )
    deletion = put(
        case.root / "deletion.json",
        {
            "version": 1,
            "removed": True,
            "operation": "exact-path-unlink",
            "observedAtNs": 3,
            "disposition": disposition,
            "master": master,
            "intent": intent,
            "masterIdentity": [1, 2, 3, 4, 5, 1],
        },
    )
    return put(
        case.root / "outcome.json",
        {
            "version": 1,
            "kind": "retired-reconstructable-whole-master",
            "master": master,
            "masterMaterialized": False,
            "freshWholeVerified": False,
            "byteIdenticalRebuildVerified": False,
            "deletionAuthorized": False,
            "historicalWholeVerified": True,
            "reconstructableInputsVerified": True,
            "rebuildPolicy": "new-nonoverwriting-revision",
            "qualityApproved": False,
            "syncApproved": False,
            "deliveryApproved": False,
            "intent": intent,
            "deletion": deletion,
            "request": request,
            "domain": intent_data["domain"],
            "inputs": [
                request,
                {"path": "/retained", "sha256": "a" * 64},
                intent,
                deletion,
                disposition,
            ],
        },
    )


def test_immutable_retired_master_is_not_required_to_exist(case: Case, recorded: None) -> None:
    case.index["master"] = {"phase": "retired", "reference": retirement(case)}
    result = c.assess(case.bind(), root=case.root, delivery=case.delivery)
    assert e.mapping(result["master"])["recordedRetirementBound"] is True
    assert e.mapping(result["master"])["freshWholeVerified"] is False
    assert result["strictComplete"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing-master",
        "missing-request",
        "extra",
        "no-delete",
        "unknown-scope",
        "unresolved",
        "changed-pin",
        "chronology",
        "master-needed",
    ],
)
def test_retirement_exact_transform_rejects_mutations(case: Case, mutation: str) -> None:
    outcome = e.mapping(json.loads(Path(retirement(case)["path"]).read_text()))
    intent = e.mapping(json.loads(Path(e.reference(outcome["intent"])["path"]).read_text()))
    deletion = e.mapping(json.loads(Path(e.reference(outcome["deletion"])["path"]).read_text()))
    disposition = e.mapping(
        json.loads(Path(e.reference(deletion["disposition"])["path"]).read_text())
    )
    inputs = e.rows(intent["inputs"])
    if mutation == "duplicate":
        intent["inputs"] = inputs + inputs[:1]
    elif mutation == "missing-master":
        intent["inputs"] = inputs[1:]
    elif mutation == "missing-request":
        intent["inputs"] = inputs[:1] + inputs[2:]
    elif mutation == "extra":
        outcome["inputs"] = e.rows(outcome["inputs"]) + [{"path": "/unknown", "sha256": "c" * 64}]
    elif mutation == "no-delete":
        outcome["inputs"] = e.rows(outcome["inputs"]) + inputs[:1]
    elif mutation == "unknown-scope":
        disposition["references"] = {}
    elif mutation == "chronology":
        deletion["observedAtNs"] = 0
    else:
        roles = e.mapping(disposition["references"])
        row = e.mapping(roles["current"])
        if mutation == "unresolved":
            row["resolved"] = False
        elif mutation == "master-needed":
            row["evidence"] = inputs[:1]
        else:
            row["evidence"] = [{"path": "/retained", "sha256": "b" * 64}]
        roles["current"] = row
        disposition["references"] = roles
    deletion["disposition"] = put(case.root / "changed-disposition.json", disposition)
    with pytest.raises(ValueError):
        c.retirement_inputs(intent, deletion, outcome, e.MetadataReader())


@pytest.mark.parametrize(
    "key,value",
    [
        ("masterMaterialized", True),
        ("freshWholeVerified", True),
        ("byteIdenticalRebuildVerified", True),
        ("kind", "prepared"),
        ("domain", {}),
        ("inputs", []),
        ("unknown", True),
        ("historicalWholeVerified", False),
    ],
)
def test_retirement_cannot_be_relabelled_fresh(case: Case, key: str, value: object) -> None:
    ref = retirement(case)
    value_data = e.mapping(json.loads(Path(ref["path"]).read_text())) | {key: value}
    new = put(case.root / "changed-outcome.json", value_data)
    case.index["master"] = {"phase": "retired", "reference": new}
    with pytest.raises(ValueError):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


def test_changed_cut_clock_reaches_original_reader(case: Case) -> None:
    context = e.mapping(case.descriptor["context"])
    cuts = put(
        case.root / "cuts.json",
        {
            "version": 1,
            "clock": "source-presentation",
            "fps": 30,
            "sourceFrames": 90,
            "cuts": [{"source_start_frame": 0, "source_end_frame": 60, "output_start_frame": 0}],
        },
    )
    context.update(
        cuts=cuts,
        voice={"path": str(case.root / "voice.wav"), "sha256": "a" * 64},
        frames=60,
        fps=30,
    )
    case.descriptor["context"] = context
    domains = e.mapping(case.descriptor["domains"])
    domains["cuts"] = {"receipt": cuts, "frames": 61, "fps": 30}
    case.descriptor["domains"] = domains
    with pytest.raises(ValueError, match="Domain edited clock differs"):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


def observation(case: Case) -> dict[str, str]:
    case.index["identities"] = {role: c.reference_of(value) for role, value in case.ids.items()}
    output = case.root.parents[1] / "edits/work" / case.root.parent.name / "observed.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    c.prepare_observation(case.bind(), root=case.root, delivery=case.delivery, output=output)
    return {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def test_missing_historical_stat_observation_is_distinct(case: Case, recorded: None) -> None:
    ref = observation(case)
    data = e.mapping(json.loads(Path(ref["path"]).read_text()))
    assert e.mapping(data["observation"])["kind"] == "current-identity-observation"
    assert "inode" not in e.mapping(e.mapping(e.mapping(data["observation"])["expected"])["source"])
    result = c.assess(ref, root=case.root, delivery=case.delivery)
    assert result["currentCryptoVerified"] is False
    live = c.assess(ref, root=case.root, delivery=case.delivery, verify_content=True)
    assert live["currentCryptoVerified"] is True
    assert live["strictComplete"] is False


@pytest.mark.parametrize("field", ["kind", "expected", "identities", "initialAndTerminalSHA"])
def test_observation_tampering(case: Case, field: str) -> None:
    ref = observation(case)
    data = e.mapping(json.loads(Path(ref["path"]).read_text()))
    e.mapping(data["observation"])[field] = None
    # mapping returns a validated dictionary; explicitly retain replacement.
    changed = e.mapping(data["observation"]) | {field: None}
    data["observation"] = changed
    with pytest.raises(ValueError):
        c.assess(put(case.root / "changed.json", data), root=case.root, delivery=case.delivery)


def test_observation_is_nonoverwriting(case: Case) -> None:
    ref = observation(case)
    with pytest.raises(ValueError, match="exists"):
        c.prepare_observation(
            case.bind(), root=case.root, delivery=case.delivery, output=Path(ref["path"])
        )


def test_observation_rejects_unknown_expected_and_scope(case: Case) -> None:
    case.index["identities"] = {"source": c.reference_of(case.ids["source"])}
    output = case.root.parents[1] / "edits/work/absent.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="seven"):
        c.prepare_observation(case.bind(), root=case.root, delivery=case.delivery, output=output)
    with pytest.raises(ValueError, match="repository work"):
        c.prepare_observation(
            case.bind(), root=case.root, delivery=case.delivery, output=case.root / "outside.json"
        )


def test_observation_wrong_expected_sha_rejected(case: Case) -> None:
    case.ids["source"]["sha256"] = "a" * 64
    with pytest.raises(ValueError):
        observation(case)


def test_failed_terminal_hash_never_publishes(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: InputHashSnapshot) -> dict[Path, str]:
        raise StageError("terminal mutation")

    monkeypatch.setattr(InputHashSnapshot, "finish", fail)
    with pytest.raises(StageError, match="terminal mutation"):
        observation(case)
    assert not (
        case.root.parents[1] / "edits/work" / case.root.parent.name / "observed.json"
    ).exists()


def test_cut_measurement_clears_only_measured_stage(case: Case, recorded: None) -> None:
    cuts = put(case.root / "cuts.json", {})
    context = e.mapping(case.descriptor["context"]) | {"frames": 60, "fps": 30, "cuts": cuts}
    case.descriptor["context"] = context
    descriptor = put(case.root / "descriptor.json", case.descriptor)
    receipt = put(
        case.root / "cut-measurement.json",
        {
            "inputs": [
                row(Path(descriptor["path"])),
                case.ids["source"],
                case.ids["movie"],
                row(Path(cuts["path"])),
            ],
            "selectionPlanMatchesProtectedCuts": True,
            "recordedExecutionMatchesPlan": True,
            "selectedFrames": 60,
            "sourceFrames": 90,
            "removedFrames": 30,
        },
    )
    case.index["measurements"] = {"cutExecution": receipt}
    metadata = c.assess(case.bind(), root=case.root, delivery=case.delivery)
    assert metadata["measuredStagesVerified"] == []
    result = c.assess(case.bind(), root=case.root, delivery=case.delivery, verify_content=True)
    assert result["measuredStagesVerified"] == ["silence_cut"]
    assert "silence_cut" not in e.mapping(result["strictFailures"])
    assert result["strictComplete"] is False


def test_failed_quality_receipt_does_not_clear_stage(case: Case, recorded: None) -> None:
    case.index["measurements"] = {
        "audioQuality": put(case.root / "bad-quality.json", {"accepted": True})
    }
    result = c.assess(case.bind(), root=case.root, delivery=case.delivery, verify_content=True)
    assert result["measuredStagesVerified"] == []
    assert "bgm_removal" in e.mapping(result["strictFailures"])


def test_unknown_measurement_rejected(case: Case, recorded: None) -> None:
    case.index["measurements"] = {"approved": True}
    with pytest.raises(ValueError, match="Unsupported"):
        c.assess(case.bind(), root=case.root, delivery=case.delivery)


def test_post_aac_measurement_does_not_invent_separation_model(
    case: Case, recorded: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Numeric validator has contrasted real schema tests in test_delivery_completion_v2.
    monkeypatch.setattr(measured, "validate_native_measurement", lambda *args, **kwargs: None)
    case.index["measurements"] = {"audioQuality": put(case.root / "measured-audio.json", {})}
    result = c.assess(case.bind(), root=case.root, delivery=case.delivery, verify_content=True)
    assert e.mapping(result["strictFailures"])["bgm_removal"] == ["separation model/source lineage"]
    assert result["measuredStagesVerified"] == []


def test_cli_reports_missing_strict_quality(
    case: Case, recorded: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(c, "load_config", lambda root: None)
    ref = case.bind()
    assert (
        c.main(
            [
                ref["path"],
                "--sha256",
                ref["sha256"],
                "--root",
                str(case.root),
                "--delivery",
                str(case.delivery),
            ]
        )
        == 2
    )
    assert (
        c.main(
            [
                ref["path"],
                "--sha256",
                "0" * 64,
                "--root",
                str(case.root),
                "--delivery",
                str(case.delivery),
            ]
        )
        == 1
    )


def test_cli_observation_produces_only_new_index(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(c, "load_config", lambda root: None)
    case.index["identities"] = {role: c.reference_of(value) for role, value in case.ids.items()}
    output = case.root.parents[1] / "edits/work/cli-observed.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    ref = case.bind()
    assert (
        c.main(
            [
                ref["path"],
                "--sha256",
                ref["sha256"],
                "--root",
                str(case.root),
                "--delivery",
                str(case.delivery),
                "--observe-output",
                str(output),
            ]
        )
        == 0
    )


def test_cli_readonly_observation_verifier_is_not_strict(
    case: Case, recorded: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(c, "load_config", lambda root: None)
    ref = observation(case)
    assert (
        c.main(
            [
                ref["path"],
                "--sha256",
                ref["sha256"],
                "--root",
                str(case.root),
                "--delivery",
                str(case.delivery),
                "--verify-observation",
            ]
        )
        == 0
    )
    result = e.mapping(json.loads(capsys.readouterr().out))
    assert result["observationVerified"] is True
    assert result["strictComplete"] is False
    assert result["currentCryptoVerified"] is False
    ordinary = case.bind()
    assert (
        c.main(
            [
                ordinary["path"],
                "--sha256",
                ordinary["sha256"],
                "--root",
                str(case.root),
                "--delivery",
                str(case.delivery),
                "--verify-observation",
            ]
        )
        == 1
    )


def test_generated_verifier_reuses_pinned_request_without_producer(
    case: Case, recorded: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = observation(case)
    request = case.bind()
    monkeypatch.setattr(c, "load_config", lambda root: None)
    monkeypatch.setattr(
        InputHashSnapshot, "capture", lambda refs: pytest.fail("No fresh media hashing")
    )
    result = c.verify_generated_observation(
        request, Path(generated["path"]), root=case.root, delivery=case.delivery
    )
    assert result["observationVerified"] is True
    assert result["currentCryptoVerified"] is False
    assert result["strictComplete"] is False
    assert (
        c.main(
            [
                request["path"],
                "--sha256",
                request["sha256"],
                "--root",
                str(case.root),
                "--delivery",
                str(case.delivery),
                "--verify-generated-observation",
                generated["path"],
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "field",
    ["configuration", "selection", "descriptor", "master", "recording", "version", "unknown"],
)
def test_generated_changed_request_fields_rejected(case: Case, recorded: None, field: str) -> None:
    generated = observation(case)
    request = case.bind()
    content = e.mapping(json.loads(Path(generated["path"]).read_text())) | {field: None}
    modified = put(case.root / "modified.json", content)
    with pytest.raises(ValueError):
        c.verify_generated_observation(
            request, Path(modified["path"]), root=case.root, delivery=case.delivery
        )


@pytest.mark.parametrize("role", ["source", "movie", "srt", "modelWeights"])
def test_generated_expected_identity_changed(case: Case, recorded: None, role: str) -> None:
    generated = observation(case)
    request = case.bind()
    content = e.mapping(json.loads(Path(generated["path"]).read_text()))
    observed = e.mapping(content["observation"])
    refs = e.mapping(observed["expected"])
    refs[role] = {"path": str(case.root / "other"), "sha256": "a" * 64}
    observed["expected"] = refs
    content["observation"] = observed
    modified = put(case.root / "modified.json", content)
    with pytest.raises(ValueError, match="expected pins"):
        c.verify_generated_observation(
            request, Path(modified["path"]), root=case.root, delivery=case.delivery
        )


def test_generated_missing_wrong_digest_and_symlink(case: Case, recorded: None) -> None:
    generated = observation(case)
    request = case.bind()
    with pytest.raises(ValueError):
        c.verify_generated_observation(
            request | {"sha256": "0" * 64},
            Path(generated["path"]),
            root=case.root,
            delivery=case.delivery,
        )
    with pytest.raises(FileNotFoundError):
        c.verify_generated_observation(
            request, case.root / "missing.json", root=case.root, delivery=case.delivery
        )
    link = case.root / "linked.json"
    link.symlink_to(generated["path"])
    with pytest.raises(ValueError, match="unlinked"):
        c.verify_generated_observation(request, link, root=case.root, delivery=case.delivery)


def test_generated_partial_or_nonmetadata_rejected(case: Case, recorded: None) -> None:
    generated = observation(case)
    request = case.bind()
    ordinary = case.root / "output.mov"
    ordinary.write_bytes(b"not JSON")
    with pytest.raises(ValueError, match="bounded JSON"):
        c.verify_generated_observation(request, ordinary, root=case.root, delivery=case.delivery)
    with pytest.raises(ValueError, match="must not contain"):
        c.verify_generated_observation(
            generated, Path(generated["path"]), root=case.root, delivery=case.delivery
        )
