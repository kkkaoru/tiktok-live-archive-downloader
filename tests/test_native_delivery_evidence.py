import copy
import hashlib
import json
import shlex
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from replay import native_delivery_evidence as e
from replay.caption_timing import SpokenToken
from replay.native_delivery_evidence import mapping, rows


@pytest.fixture
def technical_reuse_fixture(
    whole_observation_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = whole_observation_fixture
    stages = rows(records["/graph"]["stages"])[2:]
    strict = [
        "/ffmpeg",
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
        "/master",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    stages[0]["verify_command"] = strict
    frame_verify = e.strings(stages[1]["verify_command"])
    frame_verify[5] = frame_verify[5].replace(
        "($s.start_pts", "($s.width==720 and $s.height==1280 and $s.start_pts"
    )
    stages[1]["verify_command"] = frame_verify
    records["/graph"]["stages"] = stages
    phases = rows(config["phases"])[2:]

    def retain(stage: dict[str, object], phase: dict[str, object]) -> dict[str, object]:
        inputs = e.hashes(records[e.reference(phase["inventory"])["path"]]["inputs"])
        definition = {
            "command": stage["command"],
            "verifyCommand": stage["verify_command"],
            "inputs": inputs,
            "outputs": stage["outputs"],
            "timeoutSeconds": 30,
            "workingDirectory": "/repo",
        }
        records[e.reference(phase["checkpoint"])["path"]]["fingerprint"] = hashlib.sha256(
            json.dumps(definition, sort_keys=True).encode()
        ).hexdigest()
        return {**phase, "workflow": asr_evidence_ref("/graph")}

    phases = [retain(stage, phase) for stage, phase in zip(stages, phases, strict=True)]
    reuse = {
        "kind": "completed-canonical-ffmpeg-full-av",
        "stage": "whole-lossless-join",
        "checkpoint": phases[0]["checkpoint"],
        "decoder": asr_evidence_ref("/ffmpeg"),
        "media": asr_evidence_ref("/master"),
        "newDecodePerformed": False,
        "verifyCommand": strict,
    }
    config.update(
        phases=phases,
        technicalReceipt=asr_evidence_ref("/technical"),
        technicalManifest=asr_evidence_ref("/technical-manifest"),
        technicalConfig=asr_evidence_ref("/technical-config"),
        technicalWorkflow=asr_evidence_ref("/graph"),
    )
    records["/technical"] = {
        "version": 1,
        "kind": "verified-whole-technical-batch",
        "manifest": asr_evidence_ref("/technical-manifest"),
        "workflow": asr_evidence_ref("/graph"),
        "domains": ["whole-frame-clock", "whole-metadata", "joined-pcm-hash"],
        "checkpoints": [phase["checkpoint"] for phase in phases[1:]],
        "inputs": {"/ffmpeg": "a" * 64},
        "reusedFullAV": [reuse],
    }
    records["/technical-manifest"] = {
        "version": 1,
        "config": asr_evidence_ref("/technical-config"),
        "technicalWorkflow": asr_evidence_ref("/graph"),
    }
    records["/technical-config"] = {
        "targets": ["whole-frame-clock", "whole-metadata", "joined-pcm-hash"],
        "admission": {
            "inputs": [asr_evidence_ref("/ffmpeg")],
            "fullAVReuse": {
                "version": 1,
                "kind": "completed-canonical-ffmpeg-full-av",
                "stages": ["whole-lossless-join"],
            },
        },
    }
    records["/result"] = {
        "wholeMaster": asr_evidence_ref("/master"),
        "technicalRunCode": 0,
        "freshReadonlyVerifyCode": 0,
        "originalStrictFullAVVerified": True,
        "processedVoicePCMIdentityVerified": True,
        "allVideoPTSVerified": True,
        "technicalReceiptSha256": "a" * 64,
        "technicalManifestSha256": "a" * 64,
        "expectedVideoFrames": 2,
        "videoFramesPerSecond": 30,
    }
    final_path = e.reference(mapping(config["stageEvidence"])["inventory"])["path"]
    records[final_path]["inputs"] = {
        **mapping(records[final_path]["inputs"]),
        "/technical": "a" * 64,
    }
    return config, records


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "finalpin",
        "summary",
        "framecount",
        "manifest",
        "missingphase",
        "wrongcheckpoint",
        "optin",
        "pins",
        "duplicate",
        "master",
        "decoder",
        "newdecode",
        "verify",
        "stagepartial",
        "fingerprint",
        "dimension",
        "voice",
    ],
)
def test_original_technical_reuse_and_actual_phases(
    technical_reuse_fixture: tuple[dict[str, object], dict[str, dict[str, object]]], change: str
) -> None:
    config, records = technical_reuse_fixture
    if change == "finalpin":
        records[e.reference(mapping(config["stageEvidence"])["inventory"])["path"]]["inputs"] = {}
    elif change == "summary":
        records["/result"]["processedVoicePCMIdentityVerified"] = False
    elif change == "framecount":
        config["frames"] = 3
    elif change == "manifest":
        records["/technical-manifest"]["technicalWorkflow"] = asr_evidence_ref("/other")
    elif change == "missingphase":
        config["phases"] = rows(config["phases"])[:3]
    elif change == "wrongcheckpoint":
        records["/technical"]["checkpoints"] = [asr_evidence_ref("/oldcopy")]
    elif change == "optin":
        records["/technical-config"]["admission"] = {"fullAVReuse": {}}
    elif change == "pins":
        records["/technical"]["inputs"] = {"/ffmpeg": "b" * 64}
    elif change == "duplicate":
        records["/technical"]["reusedFullAV"] = rows(records["/technical"]["reusedFullAV"]) * 2
    elif change in {"master", "decoder", "newdecode", "verify"}:
        reuse = rows(records["/technical"]["reusedFullAV"])[0]
        reuse[
            {
                "master": "media",
                "decoder": "decoder",
                "newdecode": "newDecodePerformed",
                "verify": "verifyCommand",
            }[change]
        ] = True if change == "newdecode" else asr_evidence_ref("/other")
        records["/technical"]["reusedFullAV"] = [reuse]
    elif change == "stagepartial":
        records["/check-whole-frame-clock"]["status"] = "running"
    elif change == "fingerprint":
        records["/check-whole-frame-clock"]["fingerprint"] = "b" * 64
    elif change == "dimension":
        stages = rows(records["/graph"]["stages"])
        stages[1]["verify_command"] = ["jq"]
        records["/graph"]["stages"] = stages
    elif change == "voice":
        config["voice"] = asr_evidence_ref("/other")
    if change != "none":
        with pytest.raises(ValueError):
            e.original_technical_whole(config, e.MetadataReader())
        return
    result = e.original_technical_whole(config, e.MetadataReader())
    assert result["completedPhaseCount"] == 4
    assert result["newDecodePerformed"] is False
    assert result["currentOutputsCryptoVerified"] is False


@pytest.fixture
def batch_refinement_fixture(
    native_context_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = native_context_data
    records["/repair"].update(
        version=2,
        recognitionClock=asr_evidence_ref("/batch-clock"),
        recognitionComplete=asr_evidence_ref("/batch-complete"),
    )
    records["/repair-receipt"].update(
        configuration=copy.deepcopy(records["/repair"]),
        recognitionInputKind="batch-asr-single-window",
        sourceClockProjection="identity-native-presentation-no-second-projection",
    )
    records["/repair-receipt"]["inputs"] = {
        **mapping(records["/repair-receipt"]["inputs"]),
        "/batch-clock": "a" * 64,
        "/batch-complete": "a" * 64,
    }
    records["/adopted"] = copy.deepcopy(records["/repair-receipt"])
    records["/normalization"].update(
        source="/pcm", sourceSHA256="a" * 64, sampleRate=16000, samples=100
    )
    records["/local-inputs"] = {
        "version": 1,
        "inputs": {
            "/audio": "a" * 64,
            "/model/revision/config.json": "a" * 64,
            "/model/revision/weights.npz": "b" * 64,
            "/runtime/batch_asr.py": "a" * 64,
            "/runtime/asr_runtime.py": "a" * 64,
        },
        "clock": "input-pcm-not-projected",
        "samples": 100,
        "sampleRate": 16000,
        "windowSamples": 960000,
        "strideSamples": 480000,
        "device": "cpu",
        "dtypeContract": "provider-default-fp16-unchanged",
    }
    records["/batch-clock"] = {
        "startSample": 0,
        "endSample": 100,
        "sampleRate": 16000,
        "rawSHA256": "a" * 64,
        "emptyRecognitionIsNotSilence": True,
    }
    records["/batch-complete"] = {"windows": 1, "rawSHA256": ["a" * 64], "deliveryComplete": False}
    return config, records


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "model",
        "clock",
        "complete",
        "runtime",
        "dtype",
        "projection",
        "adopted",
        "source",
        "pin",
        "segment",
        "output",
    ],
)
def test_original_batch_refinement_contract(
    batch_refinement_fixture: tuple[dict[str, object], dict[str, dict[str, object]]], change: str
) -> None:
    config, records = batch_refinement_fixture
    if change == "model":
        records["/local-inputs"]["inputs"] = {
            **mapping(records["/local-inputs"]["inputs"]),
            "/model/revision/weights.npz": "c" * 64,
        }
    elif change == "clock":
        records["/batch-clock"]["startSample"] = 1
    elif change == "complete":
        records["/batch-complete"]["windows"] = 2
    elif change == "runtime":
        records["/local-inputs"]["inputs"] = {
            **mapping(records["/local-inputs"]["inputs"]),
            "/unknown.py": "a" * 64,
        }
    elif change == "dtype":
        records["/local-inputs"]["dtypeContract"] = "changed"
    elif change == "projection":
        records["/repair-receipt"]["sourceClockProjection"] = "twice"
    elif change == "adopted":
        records["/adopted"] = {"version": 1, "tokens": asr_evidence_ref("/new-tokens")}
    elif change == "source":
        records["/normalization"]["sourceSHA256"] = "b" * 64
    elif change == "pin":
        records["/repair-receipt"]["inputs"] = {
            **mapping(records["/repair-receipt"]["inputs"]),
            "/batch-clock": "b" * 64,
        }
    elif change == "segment":
        records["/index"]["segments"] = [{"segment": 0, "offsetFrame": 1, "frames": 1, "jobs": [0]}]
    elif change == "output":
        records["/new-tokens"]["tokens"] = [
            {"text": "invented", "start": 0, "end": 0.005, "utterance": "old"}
        ]
    if change != "none":
        with pytest.raises(ValueError):
            e.context_refinement_evidence(
                config,
                model=e.hashes(config["modelPins"]),
                read=lambda ref: records[e.reference(ref)["path"]],
            )
        return
    result = e.context_refinement_evidence(
        config,
        model=e.hashes(config["modelPins"]),
        read=lambda ref: records[e.reference(ref)["path"]],
    )
    assert result["sourceOffsetApplications"] == 1
    assert result["qualityApproved"] is False


@pytest.fixture
def owned_asr_fixture(
    batch_refinement_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = batch_refinement_fixture
    window = {"startSample": 0, "endSample": 100, "inputSHA256": "a" * 64}
    clock: dict[str, object] = {
        "sourceStartSample": 0,
        "sourceEndSample": 100,
        "sampleRate": 16000,
        "nativeLocalClocks": True,
        "emptyRecognitionIsNotSilence": True,
    }
    providers = [
        {"input": "base", "raw": "base-raw", "windows": 1},
        {"input": "donor", "raw": "donor-raw", "windows": 1},
    ]
    plans = [asr_evidence_ref("/base/plan.json"), {"path": "/donor/plan.json", "sha256": "b" * 64}]
    records["/base/plan.json"] = {
        "clock": "native-source-presentation-already-projected",
        "sampleRate": 16000,
        "samples": 100,
        "source": "/pcm",
        "sourceSHA256": "a" * 64,
        "modelConfigSHA256": "a" * 64,
        "modelWeightsSHA256": "b" * 64,
        "windows": [window],
    }
    records["/donor/plan.json"] = {
        "clock": "native-source-presentation-already-projected",
        "sampleRate": 16000,
        "samples": 100,
        "source": "/pcm",
        "sourceSHA256": "a" * 64,
        "parentPlanSHA256": "a" * 64,
        "windows": [window],
    }
    records["/base-raw/0000.clock.json"] = copy.deepcopy(clock)
    records["/donor-raw/0000.clock.json"] = copy.deepcopy(clock)
    records["/providers"] = {
        "clock": "native-source-presentation-already-projected",
        "providers": providers,
    }
    records["/owned"] = {
        "clock": "native-source-presentation-already-projected",
        "sourceDuration": 0.00625,
        "labels": [{"text": "hi", "start": 0, "end": 0.005}],
        "parentSHA256": {
            "/base-labels": "a" * 64,
            "/base/plan.json": "a" * 64,
            "/donor/plan.json": "b" * 64,
            "/base-raw/0000.raw.json": "a" * 64,
            "/base-raw/0000.clock.json": "a" * 64,
            "/donor-raw/0000.raw.json": "a" * 64,
            "/donor-raw/0000.clock.json": "a" * 64,
        },
    }
    records["/base-labels"] = copy.deepcopy(records["/owned"])
    records["/window-inventory"] = {
        "sourceLabels": asr_evidence_ref("/owned"),
        "providerBinding": asr_evidence_ref("/providers"),
        "totalWindows": 2,
        "totalRawClockReferences": 4,
        "plans": [
            {
                "plan": plan,
                "provider": provider,
                "source": asr_evidence_ref("/pcm"),
                "windows": [
                    {
                        "index": 0,
                        "raw": asr_evidence_ref("/" + e.string(provider["raw"]) + "/0000.raw.json"),
                        "clock": asr_evidence_ref(
                            "/" + e.string(provider["raw"]) + "/0000.clock.json"
                        ),
                        "plannedWindow": window,
                        "observedClock": clock,
                    }
                ],
            }
            for plan, provider in zip(plans, providers, strict=True)
        ],
    }
    config.update(
        contract="original-native-owned-window-plan-v1",
        sourceLabels=asr_evidence_ref("/owned"),
        baseLabels=asr_evidence_ref("/base-labels"),
        providerBinding=asr_evidence_ref("/providers"),
        windowInventory=asr_evidence_ref("/window-inventory"),
        plans=plans,
    )
    records["/repair"]["clockEvidence"] = plans[0]
    records["/repair-receipt"]["configuration"] = copy.deepcopy(records["/repair"])
    records["/repair-receipt"]["inputs"] = {
        **mapping(records["/repair-receipt"]["inputs"]),
        "/base/plan.json": "a" * 64,
    }
    records["/adopted"] = copy.deepcopy(records["/repair-receipt"])
    return config, records


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "clock",
        "model",
        "turbo",
        "duplicate",
        "missing",
        "source",
        "rawpin",
        "rawdir",
        "interval",
        "count",
        "parent",
        "cycle",
        "refinement",
        "duration",
    ],
)
def test_original_owned_provider_windows(
    owned_asr_fixture: tuple[dict[str, object], dict[str, dict[str, object]]], change: str
) -> None:
    config, records = owned_asr_fixture
    if change == "clock":
        records["/owned"]["clock"] = "edited"
    elif change == "model":
        records["/base/plan.json"]["modelWeightsSHA256"] = "c" * 64
    elif change == "turbo":
        config["modelRepository"] = "turbo"
    elif change == "duplicate":
        config["plans"] = [asr_evidence_ref("/base/plan.json"), asr_evidence_ref("/base/plan.json")]
    elif change == "missing":
        records["/window-inventory"]["plans"] = rows(records["/window-inventory"]["plans"])[:1]
    elif change == "source":
        records["/base/plan.json"]["sourceSHA256"] = "b" * 64
    elif change == "rawpin":
        records["/owned"]["parentSHA256"] = {
            **mapping(records["/owned"]["parentSHA256"]),
            "/base-raw/0000.raw.json": "b" * 64,
        }
    elif change == "rawdir":
        records["/providers"]["providers"] = [
            {"input": "base", "raw": "other", "windows": 1},
            rows(records["/providers"]["providers"])[1],
        ]
    elif change == "interval":
        records["/base-raw/0000.clock.json"]["sourceEndSample"] = 99
    elif change == "count":
        records["/window-inventory"]["totalWindows"] = 3
    elif change == "parent":
        records["/donor/plan.json"]["parentPlanSHA256"] = "c" * 64
    elif change == "cycle":
        records["/donor/plan.json"]["parentPlanSHA256"] = "b" * 64
    elif change == "refinement":
        records["/repair"]["clockEvidence"] = asr_evidence_ref("/unknown")
    elif change == "duration":
        records["/owned"]["sourceDuration"] = 1
    if change != "none":
        with pytest.raises(ValueError):
            e.owned_window_asr_evidence(config, read=lambda ref: records[e.reference(ref)["path"]])
        return
    result = e.owned_window_asr_evidence(config, read=lambda ref: records[e.reference(ref)["path"]])
    assert mapping(result["details"])["originalWindows"] == 2
    assert mapping(result["details"])["historicalFullRuntimePins"] is None
    assert result["qualityApproved"] is False


@pytest.fixture
def original_candidate_fixture(
    explicit_success_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = explicit_success_fixture
    records["/success"] = {
        "version": 1,
        "kind": "native-whole-compressor-success-candidate",
        "jobID": "job",
        "batchID": "batch",
        "nativeStatus": "Successful",
        "percentComplete": 100,
        "nativeCLIExitStatus": 0,
        "nativeStatusEvidence": "status",
        "stableStatObservedTwice": True,
        "wholeMasterSha256": "a" * 64,
        "movie": "/delivery/unique/movie.mp4",
        "movieBytes": 1,
        "observedStableStat": {"device": 1, "inode": 1, "mtimeSeconds": 1, "ctimeSeconds": 1},
        "mcpCompletionVerified": False,
    }
    records["/inventory"]["inputs"] = {
        **mapping(records["/inventory"]["inputs"]),
        "/status": "a" * 64,
    }
    records["/receipt"] = {
        "inputs": [
            identity_row("/delivery/unique/movie.mp4")
            | {
                "bytes": 1,
                "device": 1,
                "inode": 1,
                "mtimeNs": 1_000_000_000,
                "ctimeNs": 1_000_000_000,
            }
        ]
    }
    return config, records


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "kind",
        "status",
        "percent",
        "job",
        "batch",
        "size",
        "device",
        "inode",
        "mtime",
        "ctime",
        "master",
        "movie",
        "missingpin",
        "digest",
        "request",
        "stable",
        "statusevidence",
    ],
)
def test_original_candidate_status_and_later_identity(
    original_candidate_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    change: str,
) -> None:
    config, records = original_candidate_fixture
    fields = {
        "kind": "kind",
        "status": "nativeStatus",
        "percent": "percentComplete",
        "job": "jobID",
        "batch": "batchID",
        "size": "movieBytes",
        "master": "wholeMasterSha256",
        "movie": "movie",
        "stable": "stableStatObservedTwice",
        "statusevidence": "nativeStatusEvidence",
    }
    if change in fields:
        records["/success"][fields[change]] = 0
    elif change in {"device", "inode", "mtime", "ctime"}:
        records["/success"]["observedStableStat"] = {
            **mapping(records["/success"]["observedStableStat"]),
            change + ("Seconds" if change in {"mtime", "ctime"} else ""): 2,
        }
    elif change == "missingpin":
        records["/inventory"]["inputs"] = {
            key: value
            for key, value in mapping(records["/inventory"]["inputs"]).items()
            if key != "/success"
        }
    elif change == "digest":
        records["/receipt"]["inputs"] = [
            {**rows(records["/receipt"]["inputs"])[0], "sha256": "b" * 64}
        ]
    elif change == "request":
        records["/request"]["sourcePath"] = "/wrong"
    if change != "none":
        with pytest.raises(ValueError):
            e.original_success_candidate(config, e.MetadataReader())
        return
    result = e.original_success_candidate(config, e.MetadataReader())
    assert result["laterIndependentOutputBound"] is True
    assert result["originalNativeToolCompletionVerified"] is False
    assert result["currentCryptoVerified"] is False


@pytest.mark.parametrize("conflict", [False, True])
def test_original_candidate_status_indirect_original_binding(
    original_candidate_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    conflict: bool,
) -> None:
    config, records = original_candidate_fixture
    pins = mapping(records["/inventory"]["inputs"])
    pins.pop("/status")
    if conflict:
        pins["/status"] = "b" * 64
    records["/inventory"]["inputs"] = pins
    if conflict:
        with pytest.raises(ValueError, match="status pin conflicts"):
            e.original_success_candidate(config, e.MetadataReader())
        return
    result = e.original_success_candidate(config, e.MetadataReader())
    assert result["statusDirectOriginalInput"] is False
    assert result["historicalStatusExecutionPinVerified"] is False


@pytest.mark.parametrize("conflict", [False, True])
def test_technical_receipt_anchors_summary(
    technical_reuse_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    conflict: bool,
) -> None:
    config, records = technical_reuse_fixture
    inventory = e.reference(mapping(config["stageEvidence"])["inventory"])["path"]
    pins = mapping(records[inventory]["inputs"])
    pins.pop("/result")
    if conflict:
        pins["/result"] = "b" * 64
    records[inventory]["inputs"] = pins
    if conflict:
        with pytest.raises(ValueError):
            e.original_technical_whole(config, e.MetadataReader())
        return
    assert e.original_technical_whole(config, e.MetadataReader())["originalFullAVReused"] is True


@pytest.mark.parametrize("path", ["PATH=/:/usr/bin:/bin", "PATH=/:/untrusted:/usr/bin:/bin"])
def test_original_join_explicit_system_path_variants(
    whole_observation_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    path: str,
) -> None:
    config, records = whole_observation_fixture
    stage = rows(records["/graph"]["stages"])[2]
    command = e.strings(stage["command"])
    command[1] = path
    stage["command"] = command
    pins = e.hashes(records["/inputs-whole-lossless-join"]["inputs"])
    args = {
        "master": e.reference(config["source"]),
        "voice": e.reference(config["voice"]),
        "outputs": e.hashes(rows(config["phases"])[2]["outputs"]),
        "inputs": pins,
    }
    if "untrusted" in path:
        with pytest.raises(ValueError, match="tool/voice/rate"):
            e.whole_observation_command(stage, fps=30, **args)
        return
    e.whole_observation_command(stage, fps=30, **args)


@pytest.mark.parametrize("change", ["none", "missing", "newpin", "unexpected", "unsupported"])
def test_explicit_original_unpinned_batch_probe(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    records: dict[str, dict[str, object]] = {
        "/operation": {"config": "/configuration"},
        "/configuration": {},
    }
    monkeypatch.setattr(
        e.MetadataReader, "read", lambda self, value, **kwargs: records[e.reference(value)["path"]]
    )
    batch: dict[str, object] = {
        "probeContract": "original-unpinned-frame-probe-v1",
        "operation": asr_evidence_ref("/operation"),
    }
    inputs = {"/configuration": "a" * 64}
    original: dict[str, object] = {}
    if change == "missing":
        inputs = {}
    elif change == "newpin":
        records["/configuration"]["probeDecoderThreads"] = 0
    elif change == "unexpected":
        original["probeProvenance"] = {}
    elif change == "unsupported":
        batch["probeContract"] = "unknown"
    if change != "none":
        with pytest.raises(ValueError):
            e.batch_probe_provenance(
                batch, original=original, inputs=inputs, reader=e.MetadataReader()
            )
        return
    e.batch_probe_provenance(batch, original=original, inputs=inputs, reader=e.MetadataReader())


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "provider",
        "window",
        "outside",
        "unknown-clock",
        "negative-elapsed",
        "seed",
        "summary",
    ],
)
def test_owned_original_selection_and_clock_metadata(
    owned_asr_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    change: str,
) -> None:
    config, records = owned_asr_fixture
    records["/owned"]["nativeContextSelections"] = [
        {
            "provider": "base-raw",
            "window": 0,
            "first": 0,
            "end": 1,
            "before": [{"text": "old", "start": 0, "end": 0.005}],
            "after": [{"text": "new", "start": 0, "end": 0.005}],
        }
    ]
    records["/base-raw/0000.clock.json"].update(elapsedSeconds=0.25, seed=0)
    if change in {"provider", "window", "outside"}:
        selected = rows(records["/owned"]["nativeContextSelections"])[0]
        if change == "outside":
            selected["after"] = [{"text": "bad", "start": 0, "end": 0.0063}]
        else:
            selected[change] = "unknown" if change == "provider" else 10
        records["/owned"]["nativeContextSelections"] = [selected]
    elif change == "unknown-clock":
        records["/base-raw/0000.clock.json"]["newClock"] = "edited"
    elif change == "negative-elapsed":
        records["/base-raw/0000.clock.json"]["elapsedSeconds"] = -1
    elif change == "seed":
        records["/base-raw/0000.clock.json"]["seed"] = "unknown"
    elif change == "summary":
        records["/base-raw/0000.clock.json"]["nativeLocalClocks"] = False
    if change != "none":
        with pytest.raises(ValueError):
            e.owned_window_asr_evidence(config, read=lambda ref: records[e.reference(ref)["path"]])
        return
    result = e.owned_window_asr_evidence(config, read=lambda ref: records[e.reference(ref)["path"]])
    assert result["stageApplied"] is True
    assert result["executionVerified"] is False


@pytest.fixture
def original_native_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    def ref(name: str) -> dict[str, str]:
        return {"path": "/" + name, "sha256": "a" * 64}

    original = retained_domains_proof()
    original.pop("probeProvenance")
    original.update(movie=ref("staging"), partition=ref("partition"), audioIgnored=True)
    publication: dict[str, object] = {
        "version": 1,
        "kind": "published-native-part",
        "source": ref("staging"),
        "movie": ref("movie"),
        "frameProof": ref("published"),
        "completion": ref("completion"),
        "intent": ref("intent"),
    }
    published = {
        **original,
        "movie": ref("movie"),
        "sourceFrameProof": ref("original"),
        "publicationIntent": ref("intent"),
    }
    records: dict[str, dict[str, object]] = {
        "/partition": {
            "version": 1,
            "source": ref("semantic"),
            "globalCaptionCuesEqual": True,
            "duration": "1/15s",
            "parts": [{"globalStart": "0s", "duration": "1/15s"}],
        },
        "/original": original,
        "/published": published,
        "/publication": publication,
        "/completion": {
            "version": 1,
            "kind": "verified-native-part",
            "movie": ref("staging"),
            "completionVerified": True,
            "shareObservedIdle": True,
            "fullAVDecodeExitCode": 0,
            "part": 1,
            "inputs": {"/staging": "a" * 64},
            "nativeDispatchVerified": False,
            "frameProof": ref("original"),
        },
        "/intent": {
            "kind": "native-part-publication-intent",
            "source": ref("staging"),
            "completion": ref("completion"),
            "destination": "/movie",
        },
        "/assembly": {"partition": ref("partition")},
    }
    command = [
        "/python",
        "-B",
        "-m",
        "replay.fcp_part_assembly",
        "probe",
        "/assembly",
        "--part",
        "1",
    ]
    verify = [
        "/python",
        "-B",
        "-m",
        "replay.fcp_part_assembly",
        "verify",
        "/assembly",
        "--part",
        "1",
    ]
    inputs = {
        "/assembly": "a" * 64,
        "/partition": "a" * 64,
        "/staging": "a" * 64,
        "/fcp_part_assembly.py": "a" * 64,
    }
    stage: dict[str, object] = {
        "name": "probe",
        "command": command,
        "verify_command": verify,
        "inputs": list(inputs),
        "outputs": ["/original"],
        "timeout_seconds": 30,
    }
    definition = {
        "command": command,
        "verifyCommand": verify,
        "inputs": inputs,
        "outputs": ["/original"],
        "timeoutSeconds": 30,
        "workingDirectory": "/repo",
    }
    records["/graph"] = {"stages": [stage]}
    records["/inventory"] = {"inputs": inputs}
    records["/checkpoint"] = {
        "status": "complete",
        "outputs": {"/original": "a" * 64},
        "fingerprint": hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
    }
    config: dict[str, object] = {
        "partition": ref("partition"),
        "semanticXML": ref("semantic"),
        "fps": 30,
        "frames": 2,
        "width": 640,
        "height": 360,
        "parts": [
            {
                "publication": ref("publication"),
                "originalProof": ref("original"),
                "selectedProof": ref("published"),
                "probeStage": {
                    "workflow": ref("graph"),
                    "inventory": ref("inventory"),
                    "checkpoint": ref("checkpoint"),
                    "stage": "probe",
                    "cwd": "/repo",
                },
            }
        ],
    }
    monkeypatch.setattr(
        e.MetadataReader, "read", lambda self, value: records[str(mapping(value)["path"])]
    )
    return config, records


def test_original_native_probe_recorded_without_invented_tool(
    original_native_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, _records = original_native_fixture
    result = e.domain_evidence(
        "native-parts",
        {"contract": "original-assembly-probe-v1", "frames": config},
        e.MetadataReader(),
    )
    assert result["stageApplied"] is True
    assert result["executionVerified"] is False
    assert mapping(result["details"])["historicalProbeToolExecutionPin"] is None


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [
        ("/checkpoint", "status", "running"),
        ("/checkpoint", "fingerprint", "b" * 64),
        ("/checkpoint", "outputs", {"/original": "b" * 64}),
        ("/inventory", "inputs", {"/unrelated": "a" * 64}),
        ("/assembly", "partition", {"path": "/wrong", "sha256": "a" * 64}),
        ("/published", "observed", {}),
        ("/original", "probeProvenance", {}),
        ("/original", "crossRevisionReuse", {}),
        ("/publication", "source", {"path": "/wrong", "sha256": "a" * 64}),
        ("/partition", "globalCaptionCuesEqual", False),
        ("/partition", "duration", "1s"),
    ],
)
def test_original_native_rejects_changed_records(
    original_native_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    path: str,
    field: str,
    value: object,
) -> None:
    config, records = original_native_fixture
    records[path][field] = value
    with pytest.raises(ValueError):
        e.original_native_parts(config, e.MetadataReader())


@pytest.mark.parametrize(
    "command",
    [
        ["/python", "-B", "-m", "arbitrary", "probe", "/assembly", "--part", "1"],
        ["python", "-B", "-m", "replay.fcp_part_assembly", "probe", "/assembly", "--part", "1"],
        ["/python", "-B", "-m", "replay.fcp_part_assembly", "probe", "/assembly", "--part", "2"],
        ["/python", "-B", "-m", "replay.fcp_part_assembly", "verify", "/assembly", "--part", "1"],
    ],
)
def test_original_native_rejects_other_command(
    original_native_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    command: list[str],
) -> None:
    config, records = original_native_fixture
    stage = rows(records["/graph"]["stages"])[0]
    stage["command"] = command
    records["/graph"]["stages"] = [stage]
    with pytest.raises(ValueError):
        e.original_native_parts(config, e.MetadataReader())


@pytest.fixture
def original_reuse_fixture(
    original_native_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = original_native_fixture

    def ref(name: str) -> dict[str, str]:
        return {"path": "/" + name, "sha256": "a" * 64}

    records["/target-partition"] = copy.deepcopy(records["/partition"])
    config["partition"] = ref("target-partition")
    records["/target"] = {
        "partition": ref("target-partition"),
        "parts": [{"proof": "/selected", "movie": "/movie"}],
    }
    records["/reuse-config"] = {
        "version": 1,
        "part": 1,
        "targetAssembly": ref("target"),
        "publication": ref("publication"),
    }
    inputs = {
        "/reuse-config": "a" * 64,
        "/target": "a" * 64,
        "/target-partition": "a" * 64,
        "/published": "a" * 64,
        "/publication": "a" * 64,
        "/fcp_part_reuse.py": "a" * 64,
    }
    reuse: dict[str, object] = {
        "config": ref("reuse-config"),
        "priorFrameProof": ref("published"),
        "priorPartition": ref("partition"),
        "publication": ref("publication"),
        "completion": ref("completion"),
        "intent": ref("intent"),
        "inputs": inputs,
        "projectDisplayNameOnlyIgnored": True,
        "partXMLIdentical": True,
        "frameRangesUnchanged": True,
        "qualityApproved": False,
    }
    published = records["/published"]
    records["/selected"] = {
        "version": 1,
        "part": 1,
        "movie": ref("movie"),
        "observed": published["observed"],
        "partition": ref("target-partition"),
        "audioIgnored": True,
        "crossRevisionReuse": reuse,
    }
    command = ["/python", "-B", "-m", "replay.fcp_part_reuse", "/reuse-config"]
    verify = ["/python", "-B", "-m", "replay.fcp_part_assembly", "verify", "/target", "--part", "1"]
    stage: dict[str, object] = {
        "name": "reuse",
        "command": command,
        "verify_command": verify,
        "inputs": list(inputs),
        "outputs": ["/selected"],
        "timeout_seconds": 30,
    }
    definition = {
        "command": command,
        "verifyCommand": verify,
        "inputs": inputs,
        "outputs": ["/selected"],
        "timeoutSeconds": 30,
        "workingDirectory": "/repo",
    }
    records["/reuse-graph"] = {"stages": [stage]}
    records["/reuse-inventory"] = {"inputs": inputs}
    records["/reuse-checkpoint"] = {
        "status": "complete",
        "outputs": {"/selected": "a" * 64},
        "fingerprint": hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
    }
    item = rows(config["parts"])[0]
    item["selectedProof"] = ref("selected")
    item["reuseStage"] = {
        "stage": "reuse",
        "cwd": "/repo",
        "workflow": ref("reuse-graph"),
        "inventory": ref("reuse-inventory"),
        "checkpoint": ref("reuse-checkpoint"),
    }
    config["parts"] = [item]
    return config, records


def test_original_native_reuse_preserves_recorded_unknowns(
    original_reuse_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, _records = original_reuse_fixture
    result = e.original_native_parts(config, e.MetadataReader())
    assert result["stageApplied"] is True
    assert result["executionVerified"] is False
    assert result["qualityApproved"] is False


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [
        ("/reuse-config", "part", 2),
        ("/reuse-config", "publication", {"path": "/other", "sha256": "a" * 64}),
        ("/target", "parts", [{"proof": "/other", "movie": "/movie"}]),
        ("/selected", "observed", {}),
        ("/selected", "audioIgnored", False),
        ("/reuse-checkpoint", "status", "running"),
        ("/reuse-inventory", "inputs", {"/other": "a" * 64}),
    ],
)
def test_original_native_reuse_rejects_change(
    original_reuse_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    path: str,
    field: str,
    value: object,
) -> None:
    config, records = original_reuse_fixture
    records[path][field] = value
    with pytest.raises(ValueError):
        e.original_native_parts(config, e.MetadataReader())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("partXMLIdentical", False),
        ("frameRangesUnchanged", False),
        ("priorPartition", {"path": "/other", "sha256": "a" * 64}),
        ("inputs", {"/other": "a" * 64}),
    ],
)
def test_original_native_reuse_rejects_unbound_transform(
    original_reuse_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    field: str,
    value: object,
) -> None:
    config, records = original_reuse_fixture
    reuse = mapping(records["/selected"]["crossRevisionReuse"])
    reuse[field] = value
    records["/selected"]["crossRevisionReuse"] = reuse
    with pytest.raises(ValueError):
        e.original_native_parts(config, e.MetadataReader())


@pytest.fixture
def original_executor_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    def ref(path: str) -> dict[str, str]:
        return {"path": path, "sha256": "a" * 64}

    output = "/delivery/unique/movie.mp4"
    records: dict[str, dict[str, object]] = {
        "/inventory": {
            "inputs": {
                "/post": "a" * 64,
                "/status": "a" * 64,
                output: "a" * 64,
                "/preset": "a" * 64,
            }
        },
        "/post": {"master": ref("/master"), "movie": ref(output), "preset": ref("/preset")},
        "/request": {
            "batchName": "whole",
            "bundleID": "compressor",
            "sourcePath": "/master",
            "presetPath": "/preset",
            "outputDirectory": "/delivery",
            "outputName": "movie.mp4",
        },
        "/submit": execution_evidence_envelope(
            {"batch": {"batchID": "batch", "jobs": [{"jobID": "job"}]}}
        ),
        "/status": execution_evidence_envelope(
            [
                {
                    "jobStatus": {
                        "jobid": "job",
                        "batchid": "batch",
                        "status": "Successful",
                        "percentComplete": "100",
                    }
                }
            ]
        ),
    }
    data = mapping(records["/submit"]["data"])
    structured = mapping(data["structuredContent"])
    structured["outputPath"] = output
    data["structuredContent"] = structured
    records["/submit"]["data"] = data
    config: dict[str, object] = {
        "stageEvidence": {"inventory": ref("/inventory")},
        "postConfig": ref("/post"),
        "source": ref("/master"),
        "output": ref(output),
        "request": ref("/request"),
        "submit": ref("/submit"),
        "status": ref("/status"),
    }
    monkeypatch.setattr(
        e.MetadataReader, "read", lambda self, value, **kwargs: records[str(mapping(value)["path"])]
    )
    return config, records


def test_original_executor_preserves_unknown_execution(
    original_executor_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, _records = original_executor_fixture
    result = e.original_executor_request(config, e.MetadataReader())
    assert result["officialSuccessful"] is True
    assert result["originalRawRequestExecutionFingerprintVerified"] is False
    assert result["qualityApproved"] is False


@pytest.fixture
def explicit_success_fixture(
    original_executor_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = original_executor_fixture

    def ref(path: str) -> dict[str, str]:
        return {"path": path, "sha256": "a" * 64}

    records["/request-receipt"] = {
        "version": 1,
        "kind": "whole-compressor-request",
        "request": ref("/request"),
        "wholeMaster": "/master",
        "inputs": {"/master": "a" * 64, "/preset": "a" * 64},
    }
    records["/success"] = {
        "version": 1,
        "kind": "explicit-native-whole-compressor-success-observation",
        "jobID": "job",
        "batchID": "batch",
        "nativeStatus": "Successful",
        "nativePercentComplete": 100,
        "statusJobExit": 0,
        "statusEvidence": "status",
        "master": ref("/master"),
        "movie": "/delivery/unique/movie.mp4",
        "observedStableSizeBytes": 1,
        "observedStableInode": 1,
        "observedStableMtimeSeconds": 1,
        "nativeToolCompletionVerified": False,
    }
    records["/receipt"] = {
        "inputs": [
            identity_row("/delivery/unique/movie.mp4")
            | {
                "bytes": 1,
                "inode": 1,
                "mtimeNs": 1_000_000_000,
            }
        ]
    }
    inventory = mapping(records["/inventory"]["inputs"])
    for path in ("/request-receipt", "/success", "/submit"):
        inventory[path] = "a" * 64
    records["/inventory"]["inputs"] = inventory
    config.update(
        request=ref("/request-receipt"),
        success=ref("/success"),
        receipt=ref("/receipt"),
        stageEvidence={"inventory": ref("/inventory"), "cwd": "/"},
    )
    return config, records


def test_explicit_native_success_keeps_original_completion_false(
    explicit_success_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, _ = explicit_success_fixture
    result = e.explicit_native_success(config, e.MetadataReader())
    assert result["officialSuccessful"] is True
    assert result["laterIndependentOutputBound"] is True
    assert result["originalNativeToolCompletionVerified"] is False
    assert result["historicalStatusExecutionPinVerified"] is False
    assert result["qualityApproved"] is False


@pytest.mark.parametrize(
    "path,field,value",
    [
        ("/success", "version", 2),
        ("/success", "kind", "other"),
        ("/success", "nativeStatus", "Failed"),
        ("/success", "nativePercentComplete", 99),
        ("/success", "statusJobExit", 1),
        ("/success", "statusEvidence", "../status"),
        ("/success", "statusEvidence", "/status"),
        ("/success", "statusEvidence", "other"),
        ("/success", "jobID", "other"),
        ("/success", "batchID", "other"),
        ("/success", "movie", "/other.mp4"),
        ("/success", "observedStableSizeBytes", 2),
        ("/success", "observedStableInode", 2),
        ("/success", "observedStableMtimeSeconds", 2),
        ("/request-receipt", "wholeMaster", "/other"),
        ("/request", "sourcePath", "/other"),
        ("/request", "presetPath", "/other"),
        ("/request-receipt", "version", 2),
        ("/request-receipt", "kind", "other"),
        ("/receipt", "inputs", []),
        ("/inventory", "inputs", {}),
    ],
)
def test_explicit_native_success_rejects_changed_originals(
    explicit_success_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    path: str,
    field: str,
    value: object,
) -> None:
    config, records = explicit_success_fixture
    records[path][field] = value
    with pytest.raises(ValueError):
        e.explicit_native_success(config, e.MetadataReader())


@pytest.mark.parametrize("changed_master", [False, True])
def test_original_executor_context_uses_original_post_config(
    original_executor_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    changed_master: bool,
) -> None:
    config, records = original_executor_fixture
    records["/receipt"] = {
        "inputs": [
            {
                "path": "/delivery/unique/movie.mp4",
                "sha256": "a" * 64,
                "bytes": 1,
                "device": 1,
                "inode": 1,
                "mtimeNs": 1,
                "ctimeNs": 1,
            }
        ]
    }
    config.update(
        successContract="original-executor-request-v1",
        frames=30,
        fps=30,
        receipt={"path": "/receipt", "sha256": "a" * 64},
    )
    context = {
        "frames": 30,
        "fps": 30,
        "finalMovie": config["output"],
        "wholeMaster": {"path": "/other" if changed_master else "/master", "sha256": "a" * 64},
    }
    if changed_master:
        with pytest.raises(ValueError):
            e.bind_context("final-delivery", config, context, e.MetadataReader())
    else:
        e.bind_context("final-delivery", config, context, e.MetadataReader())


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [
        ("/request", "sourcePath", "/other"),
        ("/request", "presetPath", "/other"),
        ("/request", "outputDirectory", "/delivery/../else"),
        ("/request", "outputName", "other.mp4"),
        ("/request", "extra", True),
        ("/submit", "ok", False),
        ("/inventory", "inputs", {"/post": "b" * 64}),
        ("/post", "movie", {"path": "/wrong", "sha256": "a" * 64}),
    ],
)
def test_original_executor_rejects_mismatch(
    original_executor_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    path: str,
    field: str,
    value: object,
) -> None:
    config, records = original_executor_fixture
    records[path][field] = value
    with pytest.raises(ValueError):
        e.original_executor_request(config, e.MetadataReader())


@pytest.fixture
def original_whole_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    def ref(path: str) -> dict[str, str]:
        return {"path": path, "sha256": "a" * 64}

    records: dict[str, dict[str, object]] = {
        "/result": {"master": ref("/master")},
        "/final-inventory": {"inputs": {"/result": "a" * 64}},
        "/inventories": {},
    }
    stages: list[dict[str, object]] = []

    def phase(
        name: str, command: list[str], verify: list[str], inputs: list[str], outputs: list[str]
    ) -> dict[str, object]:
        pins = dict.fromkeys(inputs, "a" * 64)
        result = dict.fromkeys(outputs, "a" * 64)
        stage: dict[str, object] = {
            "name": name,
            "command": command,
            "verify_command": verify,
            "inputs": inputs,
            "outputs": outputs,
            "timeout_seconds": 30,
        }
        definition = {
            "command": command,
            "verifyCommand": verify,
            "inputs": pins,
            "outputs": outputs,
            "timeoutSeconds": 30,
            "workingDirectory": "/repo",
        }
        records["/" + name] = {
            "status": "complete",
            "outputs": result,
            "fingerprint": hashlib.sha256(
                json.dumps(definition, sort_keys=True).encode()
            ).hexdigest(),
        }
        records["/inventories"][name] = pins
        stages.append(stage)
        return {"stage": name, "checkpoint": ref("/" + name), "outputs": result, "cwd": "/repo"}

    phases = [
        phase(
            "whole-assembly-evidence",
            ["/python", "-B", "-m", "replay.fcp_part_assembly", "assemble", "/config", "/xml"],
            [
                "/python",
                "-B",
                "-m",
                "replay.fcp_part_assembly",
                "verify-assembly",
                "/config",
                "/xml",
            ],
            ["/config"],
            ["/xml"],
        ),
        phase(
            "whole-lossless-join",
            [
                "/python",
                "-B",
                "-m",
                "replay.delivery_render",
                "join-files",
                "/plan",
                "--output",
                "/master",
                "--rate",
                "30",
                "--audio",
                "/voice",
            ],
            [
                "/ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "2",
                "-i",
                "/master",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            ["/plan", "/voice"],
            ["/master"],
        ),
        phase(
            "whole-frame-clock",
            [
                "/ffprobe",
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
                "/frames",
                "/master",
            ],
            [
                "/jq",
                "-e",
                "--slurpfile",
                "plan",
                "/plan",
                "($plan[0]|map(.durationFrames)|add) as $n | .streams[0] as $s | "
                '($s.time_base|split("/")|map(tonumber)) as $tb | '
                '($s.start_pts==0 and $s.avg_frame_rate=="30/1" and '
                "($s.nb_frames|tonumber)==$n and ($s.duration_ts*$tb[0]*30)==($n*$tb[1]) "
                "and (.frames|length)==$n and (.frames|to_entries|all("
                ".value.best_effort_timestamp*$tb[0]*30==.key*$tb[1])))",
                "/frames",
            ],
            ["/ffprobe", "/master", "/plan"],
            ["/frames"],
        ),
        phase(
            "whole-metadata",
            [
                "/ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                "-o",
                "/metadata",
                "/master",
            ],
            [
                "/jq",
                "-e",
                "--slurpfile",
                "expected",
                "/expected",
                '$expected[0].streams[0] as $e | [.streams[]|select(.codec_type=="audio")] as $a | '
                "($a|length)==1 and ($a[0]|{codec_name,channels,sample_rate,"
                "time_base,duration_ts})==($e|{codec_name,channels,sample_rate,"
                "time_base,duration_ts}) and $a[0].start_pts==0",
                "/metadata",
            ],
            ["/ffprobe", "/master", "/expected"],
            ["/metadata"],
        ),
        phase(
            "joined-pcm-hash",
            [
                "/ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-i",
                "/master",
                "-map",
                "0:a:0",
                "-c:a",
                "pcm_s32le",
                "-f",
                "streamhash",
                "-hash",
                "sha256",
                "/pcm",
            ],
            ["/cmp", "/expected-pcm", "/pcm"],
            ["/ffmpeg", "/master", "/expected-pcm"],
            ["/pcm"],
        ),
    ]
    records["/graph"] = {"stages": stages}
    config: dict[str, object] = {
        "stageEvidence": {"inventory": ref("/final-inventory")},
        "wholeResult": ref("/result"),
        "source": ref("/master"),
        "wholeWorkflow": ref("/graph"),
        "wholeInventories": ref("/inventories"),
        "fps": 30,
        "phases": phases,
    }
    monkeypatch.setattr(
        e.MetadataReader, "read", lambda self, value: records[str(mapping(value)["path"])]
    )
    return config, records


def test_original_whole_independent_phase_inventory(
    original_whole_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, _records = original_whole_fixture
    result = e.original_whole_phases(config, e.MetadataReader())
    assert result["completedPhaseCount"] == 5
    assert result["finalStageDirectCheckpointMembership"] is False
    assert result["currentOutputsCryptoVerified"] is False


@pytest.mark.parametrize(
    "role",
    [
        "whole-assembly-evidence",
        "whole-lossless-join",
        "whole-frame-clock",
        "whole-metadata",
        "joined-pcm-hash",
    ],
)
def test_original_whole_rejects_partial_phase(
    original_whole_fixture: tuple[dict[str, object], dict[str, dict[str, object]]], role: str
) -> None:
    config, records = original_whole_fixture
    records["/" + role]["status"] = "running"
    with pytest.raises(ValueError):
        e.original_whole_phases(config, e.MetadataReader())


@pytest.mark.parametrize(
    "role",
    [
        "whole-assembly-evidence",
        "whole-lossless-join",
        "whole-frame-clock",
        "whole-metadata",
        "joined-pcm-hash",
    ],
)
def test_original_whole_rejects_arbitrary_verifier(
    original_whole_fixture: tuple[dict[str, object], dict[str, dict[str, object]]], role: str
) -> None:
    _config, records = original_whole_fixture
    stage = next(item for item in rows(records["/graph"]["stages"]) if item["name"] == role)
    stage["verify_command"] = ["/bin/true"]
    with pytest.raises(ValueError):
        e.original_whole_command(
            role,
            stage,
            {"path": "/master", "sha256": "a" * 64},
            e.hashes(records["/" + role]["outputs"]),
            e.hashes(records["/inventories"][role]),
            30,
        )


def test_original_whole_rejects_changed_grid(
    original_whole_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, _records = original_whole_fixture
    config["fps"] = 25
    with pytest.raises(ValueError, match="semantics"):
        e.original_whole_phases(config, e.MetadataReader())


@pytest.mark.parametrize("status", ["Failed", "Processing"])
def test_original_executor_percent_complete_is_not_success(
    original_executor_fixture: tuple[dict[str, object], dict[str, dict[str, object]]], status: str
) -> None:
    config, records = original_executor_fixture
    records["/status"] = execution_evidence_envelope(
        [
            {
                "jobStatus": {
                    "jobid": "job",
                    "batchid": "batch",
                    "status": status,
                    "percentComplete": "100",
                }
            }
        ]
    )
    with pytest.raises(ValueError):
        e.original_executor_request(config, e.MetadataReader())


def retained_stage_fixture(
    tmp_path: Path, name: str, command: list[str], inputs: dict[str, str], outputs: dict[str, str]
) -> dict[str, object]:
    wrapped = [
        "/bin/bash",
        "-c",
        "set -euo pipefail\n"
        + shlex.join(
            ["/python", "-B", "-m", "replay.phase_timer", "/phase.jsonl", "phase", "--", *command]
        ),
    ]
    stage: dict[str, object] = {
        "name": name,
        "command": wrapped,
        "verify_command": ["/verify"],
        "inputs": list(inputs),
        "outputs": list(outputs),
        "timeout_seconds": 30,
    }
    definition = {
        "command": wrapped,
        "verifyCommand": ["/verify"],
        "inputs": inputs,
        "outputs": list(outputs),
        "timeoutSeconds": 30,
        "workingDirectory": "/repo",
    }
    return {
        "stage": name,
        "cwd": "/repo",
        "outputs": outputs,
        "workflow": followup_history_put(tmp_path, name + "-graph.json", {"stages": [stage]}),
        "inventory": followup_history_put(tmp_path, name + "-inputs.json", {"inputs": inputs}),
        "checkpoint": followup_history_put(
            tmp_path,
            name + "-checkpoint.json",
            {
                "status": "complete",
                "outputs": outputs,
                "fingerprint": hashlib.sha256(
                    json.dumps(definition, sort_keys=True).encode()
                ).hexdigest(),
            },
        ),
    }


def normalized_compressor_fixture(tmp_path: Path, role: str, output: str) -> dict[str, object]:
    submit = execution_evidence_envelope(
        {"batch": {"batchID": "batch", "jobs": [{"jobID": "job"}]}}
    )
    data = mapping(submit["data"])
    structured = mapping(data["structuredContent"])
    structured["outputPath"] = output
    data["structuredContent"] = structured
    submit["data"] = data
    return {
        "submit": followup_history_put(tmp_path, role + "-submit.json", submit),
        "status": followup_history_put(
            tmp_path,
            role + "-status.json",
            execution_evidence_envelope(
                [
                    {
                        "jobStatus": {
                            "jobid": "job",
                            "batchid": "batch",
                            "status": "Successful",
                            "percentComplete": "100",
                        }
                    }
                ]
            ),
        ),
        "request": followup_history_put(
            tmp_path,
            role + "-request.json",
            {
                "version": 1,
                "kind": "whole-compressor-request",
                "wholeMaster": "/whole.mov",
                "inputs": {"/whole.mov": "a" * 64},
            },
        ),
        "success": followup_history_put(
            tmp_path,
            role + "-success.json",
            {
                "jobUUID": "job",
                "batchUUID": "batch",
                "output": {"path": output},
                "source": {"path": "/whole.mov", "sha256": "a" * 64},
            },
        ),
    }


def normalized_fixture(tmp_path: Path) -> dict[str, object]:
    output = {"path": "/normalized.mp4", "sha256": "a" * 64}
    audio = {"path": "/audio.m4a", "sha256": "b" * 64}
    ffmpeg = {"path": "/ffmpeg", "sha256": "c" * 64}
    ffprobe = {"path": "/ffprobe", "sha256": "d" * 64}
    refs = {
        "audio": audio,
        "video": {"path": "/video.mp4", "sha256": "e" * 64},
        "packetProbe": {"path": "/packets.json", "sha256": "f" * 64},
        "decodedProbe": {"path": "/decoded.json", "sha256": "1" * 64},
        "tagProbe": {"path": "/tag.json", "sha256": "2" * 64},
        "ffmpeg": ffmpeg,
    }
    normal_config = followup_history_put(
        tmp_path,
        "normal-config.json",
        {
            **refs,
            "version": 1,
            "videoTag": "avc1",
            "videoTrackTimescale": 3000,
            "output": "/normalized.mp4",
            "timestampField": "best_effort_timestamp",
        },
    )
    mux_command = [
        "/ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-copyts",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        "/video.mp4",
        "-i",
        "/audio.m4a",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        "-bsf:a:0",
        "setts=pts=PTS-2112:dts=DTS-2112",
        "-avoid_negative_ts",
        "disabled",
        "-use_editlist",
        "1",
        "-tag:v",
        "avc1",
        "-video_track_timescale",
        "3000",
        "-movie_timescale",
        "48000",
        "-movflags",
        "+faststart",
        "/normalized.mp4",
    ]
    normal = followup_history_put(
        tmp_path,
        "normal.json",
        {
            "version": 1,
            "kind": "observed-aac-leading-exclusion-to-presentation-zero",
            "output": "/normalized.mp4",
            "timestampField": "best_effort_timestamp",
            "inputs": refs,
            "config": normal_config,
            "sampleRate": 48000,
            "decodedSamples": 95999,
            "primingSamples": 2112,
            "preRollTicks": 2112,
            "tailDiscardSamples": 255,
            "producerCommand": mux_command,
            "timeBase": "1/48000",
            "sourceFrameClockSha256": "4" * 64,
        },
    )
    audio_proof = followup_history_put(
        tmp_path,
        "audio-proof.json",
        {"packetProbe": refs["packetProbe"], "decodedProbe": refs["decodedProbe"]},
    )
    observer_config = followup_history_put(
        tmp_path,
        "observer.json",
        {
            "output": "/normalized.mp4",
            "tools": {"ffmpeg": ffmpeg, "ffprobe": ffprobe},
            "audio": {"movie": audio, "evidence": audio_proof},
        },
    )
    probe = [
        "/ffprobe",
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
        "/normalized.mp4",
    ]
    pins = {ref["path"]: ref["sha256"] for ref in (output, ffprobe, observer_config, audio)}
    snapshot = followup_history_put(
        tmp_path,
        "snapshot.json",
        {
            "version": 3,
            "videoContract": "h264",
            "command": probe,
            "inputs": [{"path": path, "sha256": sha} for path, sha in pins.items()],
        },
    )
    process = followup_history_put(tmp_path, "process.json", {"command": probe, "returncode": 0})
    raw = {"path": str(tmp_path / "combined.stdout"), "sha256": "3" * 64}
    observation = followup_history_put(
        tmp_path,
        "observation.json",
        {
            "version": 3,
            "kind": "single-pass-whole-observation",
            "videoContract": "h264",
            "raw": raw,
            "snapshot": snapshot,
            "process": process,
            "inputs": [{"path": path, "sha256": sha} for path, sha in pins.items()],
        },
    )
    technical = followup_history_put(
        tmp_path,
        "technical.json",
        {
            "version": 3,
            "videoContract": "h264",
            "technicalAccepted": True,
            "fullDecode": True,
            "decodePasses": 1,
            "decoder": "ffprobe-combined-frames-packets",
            "videoClock": {"frames": 60, "start": "0", "end": "2", "exactCFR": True},
            "inputs": [output],
            "deliveryApproved": False,
        },
    )
    audit = followup_history_put(
        tmp_path,
        "audit.json",
        {
            "version": 3,
            "videoContract": "h264",
            "candidateMuxReady": True,
            "orderedPayloadEqual": True,
            "sourceClockPreserved": True,
            "internalPrimingPreserved": True,
            "sourceClockTimestampField": "best_effort_timestamp",
            "sampleRate": 48000,
            "decodedSamples": 95999,
            "expectedEnd": "2",
            "audioMuxAccepted": False,
            "presentationNormalization": {
                "proof": normal,
                "primingSamples": 2112,
                "preRollTicks": 2112,
                "sourceClockPreservedUnderDeclaredMapping": True,
            },
        },
    )
    base = [
        "/python",
        "-B",
        "-m",
        "replay.compressed_audio_verification",
        observer_config["path"],
        str(tmp_path),
    ]
    observe_command = [*base, "--observe", "--video-contract", "h264", ">", technical["path"]]
    audio_command = [
        *base,
        audit["path"],
        "--audio-result",
        "--video-contract",
        "h264",
        "--source-timestamp-field",
        "best_effort_timestamp",
        "--normalization-binding",
        normal["path"],
        "--normalization-sha256",
        normal["sha256"],
    ]
    observe_outputs = {
        ref["path"]: ref["sha256"] for ref in (observation, technical, snapshot, process, raw)
    }
    mux_inputs = {ref["path"]: ref["sha256"] for ref in (*refs.values(), normal, normal_config)}
    return {
        "contract": "normalized-combined-v3",
        "output": output,
        "frames": 60,
        "fps": 30,
        "observation": observation,
        "receipt": technical,
        "audioAudit": audit,
        "normalization": normal,
        "observerConfig": observer_config,
        "python": "/python",
        "observationCommand": observe_command,
        "audioCommand": audio_command,
        "videoCompressor": normalized_compressor_fixture(tmp_path, "video", "/video.mp4"),
        "audioCompressor": normalized_compressor_fixture(tmp_path, "audio", "/audio.m4a"),
        "stages": {
            "observation": retained_stage_fixture(
                tmp_path, "observe", observe_command, pins, observe_outputs
            ),
            "audio": retained_stage_fixture(
                tmp_path,
                "audio",
                audio_command,
                {
                    normal["path"]: normal["sha256"],
                    observer_config["path"]: observer_config["sha256"],
                },
                {audit["path"]: audit["sha256"]},
            ),
            "normalization": retained_stage_fixture(
                tmp_path, "mux", mux_command, mux_inputs, {"/normalized.mp4": "a" * 64}
            ),
        },
    }


def test_normalized_original_combined_contract(tmp_path: Path) -> None:
    result = e.domain_evidence("final-delivery", normalized_fixture(tmp_path), e.MetadataReader())
    assert result["stageApplied"] is True
    assert result["qualityApproved"] is False
    assert result["currentCryptoVerified"] is False


@pytest.mark.parametrize(
    "role,key,value",
    [
        ("receipt", "version", 1),
        ("receipt", "fullDecode", False),
        ("audioAudit", "sourceClockPreserved", False),
        ("audioAudit", "decodedSamples", 96000),
        ("audioAudit", "expectedEnd", "3"),
        ("audioAudit", "candidateMuxReady", False),
        ("normalization", "output", "/other.mp4"),
        ("normalization", "timestampField", "pts"),
        ("observation", "version", 1),
        ("observation", "inputs", []),
    ],
)
def test_normalized_changed_original_records(
    tmp_path: Path, role: str, key: str, value: object
) -> None:
    config = normalized_fixture(tmp_path)
    record = mapping(json.loads(Path(e.reference(config[role])["path"]).read_text()))
    record[key] = value
    config[role] = followup_history_put(tmp_path, "changed.json", record)
    with pytest.raises(ValueError):
        e.domain_evidence("final-delivery", config, e.MetadataReader())


@pytest.mark.parametrize(
    "role,key,value",
    [
        ("observation", "status", "running"),
        ("audio", "fingerprint", "0" * 64),
        ("normalization", "outputs", {}),
    ],
)
def test_normalized_incomplete_original_stage(
    tmp_path: Path, role: str, key: str, value: object
) -> None:
    config = normalized_fixture(tmp_path)
    stages = mapping(config["stages"])
    stage = mapping(stages[role])
    record = mapping(json.loads(Path(e.reference(stage["checkpoint"])["path"]).read_text()))
    record[key] = value
    stage["checkpoint"] = followup_history_put(tmp_path, "changed-checkpoint.json", record)
    stages[role] = stage
    config["stages"] = stages
    with pytest.raises(ValueError, match="Original|retained input hashes"):
        e.domain_evidence("final-delivery", config, e.MetadataReader())


def test_normalized_absent_raw_is_archived_not_redecoded(tmp_path: Path) -> None:
    config = normalized_fixture(tmp_path)
    assert not (tmp_path / "combined.stdout").exists()
    result = e.domain_evidence("final-delivery", config, e.MetadataReader())
    assert result["currentCryptoVerified"] is False
    assert "fresh raw observation validation" in e.strings(result["pending"])


@pytest.mark.parametrize("field", ["observationCommand", "audioCommand"])
def test_normalized_wrong_typed_command(tmp_path: Path, field: str) -> None:
    config = normalized_fixture(tmp_path)
    config[field] = ["/python", "-m", "unknown"]
    with pytest.raises(ValueError, match="command/config"):
        e.domain_evidence("final-delivery", config, e.MetadataReader())


def mixed_fixture(tmp_path: Path) -> dict[str, object]:
    source = {"path": "/source.wav", "sha256": "a" * 64}
    stem = {"path": "/stem.wav", "sha256": "b" * 64}
    fallback = {"path": "/fallback.wav", "sha256": "c" * 64}
    clock = {"path": "/clock.json", "sha256": "d" * 64}
    selection = followup_history_put(
        tmp_path,
        "selection.json",
        {
            "source": source,
            "clock": clock,
            "coordinateSpace": "native-presentation-seconds",
        },
    )
    support = followup_history_put(
        tmp_path,
        "support.json",
        {
            "kind": "selected-pcm-correspondence-diagnostic",
            "sampleRate": 100,
            "channels": 2,
            "inputs": {
                source["path"]: source["sha256"],
                selection["path"]: selection["sha256"],
                clock["path"]: clock["sha256"],
            },
            "measuredSupport": [{"start": 0, "end": 200, "candidate_start": 10}],
        },
    )
    old = followup_history_put(
        tmp_path,
        "fallback.json",
        {
            "kind": "context-atempo-exact-cut-slot-candidate",
            "output": "/fallback.wav",
            "outputSHA256": "c" * 64,
            "outputSamples": 200,
            "sampleRate": 100,
            "channels": 2,
            "cuts": [
                {
                    "source_start_sample": 0,
                    "source_end_sample": 100,
                    "output_start_sample": 0,
                    "output_end_sample": 100,
                },
                {
                    "source_start_sample": 100,
                    "source_end_sample": 200,
                    "output_start_sample": 100,
                    "output_end_sample": 200,
                },
            ],
        },
    )
    producer = followup_history_put(
        tmp_path,
        "producer.json",
        {
            "version": 1,
            "reference": source,
            "candidateMixture": source,
            "candidateStem": stem,
            "fallback": fallback,
            "fallbackReceipt": old,
            "selection": selection,
            "mapping": support,
            "evidence": [clock],
        },
    )
    inputs = {
        r["path"]: r["sha256"]
        for r in (producer, source, stem, fallback, old, selection, support, clock)
    }
    receipt = followup_history_put(
        tmp_path,
        "mixed.json",
        {
            "version": 1,
            "kind": "mixed-tempo-candidate",
            "preflightOnly": False,
            "inputs": inputs,
            "sampleRate": 100,
            "channels": 2,
            "outputSamples": 200,
            "outputSHA256": "e" * 64,
            "fallbackCuts": 1,
            "newStemCuts": 1,
            "qualityAccepted": False,
            "cuts": [
                {
                    "outputStartSample": 0,
                    "outputEndSample": 100,
                    "sourcePath": "/stem.wav",
                    "sourceSHA256": "b" * 64,
                    "sourceClock": "measured-continuous-pcm",
                    "sourceSpans": [[10, 110]],
                    "fallbackReadSpan": None,
                    "pcmSHA256": "f" * 64,
                    "tempo": {
                        "source_start_sample": 10,
                        "source_end_sample": 110,
                        "output_start_sample": 0,
                        "output_end_sample": 100,
                        "center_crop_start_sample": 0,
                        "center_crop_end_sample": 100,
                        "source_context_samples": 0,
                        "filter_applied": False,
                    },
                },
                {
                    "outputStartSample": 100,
                    "outputEndSample": 200,
                    "sourcePath": "/fallback.wav",
                    "sourceSHA256": "c" * 64,
                    "sourceClock": "selected-output",
                    "sourceSpans": [[100, 200]],
                    "fallbackReadSpan": [100, 200],
                    "pcmSHA256": "f" * 64,
                    "tempo": None,
                },
            ],
        },
    )
    return {
        "contract": "mixed-tempo-candidate",
        "receipt": receipt,
        "producerConfig": producer,
        "expectedInputs": inputs,
        "output": {"path": "/mixed.wav", "sha256": "e" * 64},
        "frames": 60,
        "fps": 30,
    }


def test_mixed_original_clock_domains(tmp_path: Path) -> None:
    result = e.domain_evidence("voice", mixed_fixture(tmp_path), e.MetadataReader())
    assert result["stageApplied"] is True
    assert result["currentCryptoVerified"] is False
    assert result["qualityApproved"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 2),
        ("preflightOnly", True),
        ("inputs", {}),
        ("sampleRate", 0),
        ("outputSamples", 199),
        ("channels", 1),
        ("outputSHA256", "0" * 64),
        ("cuts", []),
        ("fallbackCuts", 0),
        ("newStemCuts", 2),
    ],
)
def test_mixed_rejects_changed_receipt(tmp_path: Path, key: str, value: object) -> None:
    config = mixed_fixture(tmp_path)
    path = Path(e.reference(config["receipt"])["path"])
    record = mapping(json.loads(path.read_text()))
    record[key] = value
    config["receipt"] = followup_history_put(tmp_path, "changed.json", record)
    with pytest.raises(ValueError):
        e.domain_evidence("voice", config, e.MetadataReader())


@pytest.mark.parametrize(
    "number,key,value",
    [
        (0, "sourceClock", "output"),
        (0, "outputStartSample", 1),
        (0, "sourceSpans", [[0, 100]]),
        (0, "sourceSHA256", "0" * 64),
        (0, "fallbackReadSpan", [0, 100]),
        (1, "fallbackReadSpan", [0, 100]),
        (1, "tempo", {}),
        (1, "sourceSpans", [[99, 200]]),
    ],
)
def test_mixed_rejects_cut_changes(tmp_path: Path, number: int, key: str, value: object) -> None:
    config = mixed_fixture(tmp_path)
    record = mapping(json.loads(Path(e.reference(config["receipt"])["path"]).read_text()))
    cuts = rows(record["cuts"])
    cuts[number][key] = value
    record["cuts"] = cuts
    config["receipt"] = followup_history_put(tmp_path, "changed.json", record)
    with pytest.raises(ValueError):
        e.domain_evidence("voice", config, e.MetadataReader())


def evidence_context() -> dict[str, object]:
    return {
        **{
            key: {"path": "/" + key, "sha256": "a" * 64}
            for key in ("source", "cuts", "voice", "semanticXML", "wholeMaster", "finalMovie")
        },
        "frames": 60,
        "fps": 30,
    }


def test_absent_domains_are_not_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config: dict[str, object] = {
        "version": 1,
        "backend": "native-fcp-partitioned-whole",
        "context": evidence_context(),
        "domains": {},
    }
    ref = followup_history_put(tmp_path, "descriptor.json", config)
    monkeypatch.setattr(sys, "argv", ["evidence", ref["path"], "--sha256", ref["sha256"]])
    e.main()
    result = mapping(json.loads(capsys.readouterr().out))
    assert result["finished"] is False
    assert result["candidateAvailable"] is None
    assert all(mapping(row)["stageApplied"] is False for row in mapping(result["domains"]).values())
    changes: list[dict[str, object]] = [
        {"version": 2},
        {"domains": {"unknown": {}}},
        {"context": {**evidence_context(), "fps": 0}},
    ]
    for changed in changes:
        with pytest.raises(ValueError):
            e.assess(config | changed)


def test_metadata_bound_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = followup_history_put(tmp_path, "record.json", {"value": 1})
    reader = e.MetadataReader()
    assert reader.read(ref) == {"value": 1}
    count = reader.bytes_read
    assert reader.read(ref) == {"value": 1}
    assert reader.bytes_read == count
    with pytest.raises(ValueError, match="Conflicting"):
        reader.read(ref | {"sha256": "0" * 64})
    Path(ref["path"]).write_text('{"value":2}')
    with pytest.raises(ValueError, match="identity"):
        reader.read(ref)
    with pytest.raises(ValueError, match="SHA"):
        e.MetadataReader().read(ref)
    link = tmp_path / "link.json"
    link.symlink_to(ref["path"])
    with pytest.raises(ValueError, match="canonical"):
        e.MetadataReader().read(ref | {"path": str(link)})
    media = tmp_path / "movie.mov"
    media.write_bytes(b"never decode")
    with pytest.raises(ValueError, match="not media"):
        e.MetadataReader().read(ref | {"path": str(media)})
    empty = tmp_path / "empty.json"
    empty.touch()
    with pytest.raises(ValueError, match="size"):
        e.MetadataReader().read(ref | {"path": str(empty)})
    monkeypatch.setattr(e, "MAX_RECORDS", 0)
    with pytest.raises(ValueError, match="record budget"):
        e.MetadataReader().read(ref)
    monkeypatch.setattr(e, "MAX_RECORDS", 5)
    monkeypatch.setattr(e, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        e.MetadataReader().read(ref)


def final_descriptor(tmp_path: Path, native: dict[str, dict[str, object]]) -> dict[str, object]:
    inventory: dict[str, str] = {"/tool": "b" * 64, "/final.mp4": "a" * 64}
    whole: dict[str, object] = {}
    for key in ("join", "clock", "metadata", "pcm"):
        ref = followup_history_put(
            tmp_path,
            key + ".json",
            {
                "status": "complete",
                "fingerprint": "c" * 64,
                "outputs": {"/whole.mov" if key == "join" else "/" + key: "a" * 64},
            },
        )
        whole[key] = ref
        inventory[ref["path"]] = ref["sha256"]
    receipt = followup_history_put(
        tmp_path,
        "receipt.json",
        {
            "version": 1,
            "fullDecode": True,
            "technicalAccepted": True,
            "inputs": [identity_row(path, sha) for path, sha in inventory.items()],
            "inputsAfter": [identity_row(path, sha) for path, sha in inventory.items()],
            "expected": {"frames": 60, "rate": 30},
            "approvals": {"quality": False},
        },
    )
    stage: dict[str, object] = {
        "name": "decode",
        "command": ["/python", "-B", "-m", "replay.delivery_verification", "/config", "/final.mp4"],
        "verify_command": ["/python", "verify"],
        "inputs": list(inventory),
        "outputs": [receipt["path"]],
        "timeout_seconds": 30,
    }
    definition = {
        "command": stage["command"],
        "verifyCommand": stage["verify_command"],
        "inputs": inventory,
        "outputs": stage["outputs"],
        "timeoutSeconds": 30,
        "workingDirectory": "/repository",
    }
    checkpoint = followup_history_put(
        tmp_path,
        "checkpoint.json",
        {
            "status": "complete",
            "fingerprint": hashlib.sha256(
                json.dumps(definition, sort_keys=True).encode()
            ).hexdigest(),
            "outputs": {receipt["path"]: receipt["sha256"]},
        },
    )
    result: dict[str, object] = {
        "stageEvidence": {
            "workflow": followup_history_put(tmp_path, "workflow.json", {"stages": [stage]}),
            "inventory": followup_history_put(tmp_path, "inventory.json", {"inputs": inventory}),
            "stage": "decode",
            "checkpoint": checkpoint,
            "cwd": "/repository",
            "outputs": {receipt["path"]: receipt["sha256"]},
        },
        "receipt": receipt,
        "frames": 60,
        "fps": 30,
        "output": {"path": "/final.mp4", "sha256": "a" * 64},
        "source": {"path": "/whole.mov", "sha256": "a" * 64},
        "wholeRecords": whole,
    }
    for key in ("submit", "status", "request", "success"):
        result[key] = followup_history_put(tmp_path, key + ".json", native[key])
    return result


def test_original_whole_and_final_not_fresh(
    tmp_path: Path, execution_evidence_native: dict[str, dict[str, object]]
) -> None:
    final = final_descriptor(tmp_path, execution_evidence_native)
    context = evidence_context() | {"wholeMaster": final["source"], "finalMovie": final["output"]}
    result = e.assess(
        {
            "version": 1,
            "backend": "native-fcp-partitioned-whole",
            "context": context,
            "domains": {"final-delivery": final, "whole-assembly": final},
        }
    )
    assert result["finished"] is False
    assert mapping(mapping(result["domains"])["final-delivery"])["recordedEvidenceBound"] is True
    stage = mapping(final["stageEvidence"])
    with pytest.raises(ValueError, match="ambiguous"):
        e.checkpoint_evidence(stage | {"stage": "wrong"}, e.MetadataReader())
    for changed in ({"frames": 61}, {"wholeRecords": None}):
        name = "final-delivery" if "frames" in changed else "whole-assembly"
        with pytest.raises(ValueError):
            e.domain_evidence(name, final | changed, e.MetadataReader())


def native_success_fixture(
    tmp_path: Path, native: dict[str, dict[str, object]]
) -> dict[str, object]:
    config = final_descriptor(tmp_path, native)
    raw_request = followup_history_put(
        tmp_path, "native-request.json", {"sourcePath": "/whole.mov"}
    )
    result = followup_history_put(
        tmp_path,
        "native-submit-result.json",
        {
            "request": raw_request,
            "requestReceipt": config["request"],
            "rawSubmitLog": config["submit"],
            "submitEnvelope": native["submit"],
            "nativeSubmissionCount": 1,
            "jobID": "job",
            "batchID": "batch",
            "outputPath": "/final.mp4",
        },
    )
    config["successContract"] = "native-status-source-master-v1"
    config["success"] = followup_history_put(
        tmp_path,
        "native-success.json",
        {
            "version": 1,
            "nativeSubmissionCount": 1,
            "resubmissions": 0,
            "independentFullAVPending": True,
            "officialStatus": config["status"],
            "requestReceipt": config["request"],
            "request": raw_request,
            "submitResult": result,
            "nativeStatus": {
                "jobid": "job",
                "batchid": "batch",
                "status": "Successful",
                "percentComplete": "100",
            },
            "sourceMaster": {"path": "/whole.mov", "sha256": "a" * 64},
            "output": {"path": "/final.mp4", "sha256": None, "hashPendingIndependentReader": True},
        },
    )
    return config


def test_original_native_success_then_later_hash(
    tmp_path: Path, execution_evidence_native: dict[str, dict[str, object]]
) -> None:
    config = native_success_fixture(tmp_path, execution_evidence_native)
    context = evidence_context() | {"wholeMaster": config["source"], "finalMovie": config["output"]}
    result = e.assess(
        {
            "version": 1,
            "backend": "native-fcp-partitioned-whole",
            "context": context,
            "domains": {"final-delivery": config},
        }
    )
    assert result["finished"] is False
    assert mapping(mapping(result["domains"])["final-delivery"])["stageApplied"] is True


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 2),
        ("resubmissions", 1),
        ("independentFullAVPending", False),
        (
            "nativeStatus",
            {
                "jobid": "other",
                "batchid": "batch",
                "status": "Successful",
                "percentComplete": "100",
            },
        ),
        (
            "nativeStatus",
            {"jobid": "job", "batchid": "batch", "status": "Failed", "percentComplete": "100"},
        ),
        ("sourceMaster", {"path": "/other.mov", "sha256": "a" * 64}),
        ("sourceMaster", {"path": "/whole.mov", "sha256": "b" * 64}),
        (
            "output",
            {"path": "/final.mp4", "sha256": "a" * 64, "hashPendingIndependentReader": False},
        ),
    ],
)
def test_native_success_original_mutations(
    tmp_path: Path, execution_evidence_native: dict[str, dict[str, object]], key: str, value: object
) -> None:
    config = native_success_fixture(tmp_path, execution_evidence_native)
    record = mapping(json.loads(Path(e.reference(config["success"])["path"]).read_text()))
    record[key] = value
    config["success"] = followup_history_put(tmp_path, "changed-native-success.json", record)
    with pytest.raises(ValueError):
        e.domain_evidence("final-delivery", config, e.MetadataReader())


def test_source_revision_mismatch(tmp_path: Path) -> None:
    receipt = followup_history_put(tmp_path, "cuts.json", domain_composer_cuts())
    context = evidence_context() | {"cuts": receipt}
    config = {
        "version": 1,
        "backend": "native-fcp-partitioned-whole",
        "context": context,
        "domains": {"cuts": {"receipt": receipt, "frames": 60, "fps": 30}},
    }
    assert e.assess(config)["finished"] is False
    for name, entry in (
        ("cleanup", {"source": {"path": "/other", "sha256": "a" * 64}}),
        ("cuts", {"receipt": receipt, "frames": 59, "fps": 30}),
    ):
        with pytest.raises(ValueError):
            e.assess(config | {"domains": {name: entry}})


def domain_composer_voice() -> dict[str, object]:
    return {
        "version": 1,
        "kind": "context-atempo-exact-cut-slot-candidate",
        "inputs": {"/original": "a" * 64},
        "output": "/voice.wav",
        "outputSHA256": "b" * 64,
        "sampleRate": 48000,
        "channels": 2,
        "outputStartSample": 0,
        "outputEndSample": 96000,
        "outputSamples": 96000,
        "cuts": [
            {
                "source_start_sample": 10,
                "source_end_sample": 80000,
                "output_start_sample": 0,
                "output_end_sample": 96000,
                "center_crop_start_sample": 20,
                "center_crop_end_sample": 96020,
                "filter_applied": True,
            }
        ],
    }


def domain_composer_assess(record: dict[str, object]) -> dict[str, object]:
    return e.voice_domain(
        record,
        expected_inputs={"/original": "a" * 64},
        output={"path": "/voice.wav", "sha256": "b" * 64},
        frames=60,
        fps=30,
    )


def test_domain_composer_applied_not_quality_or_fresh() -> None:
    result = domain_composer_assess(domain_composer_voice())
    assert result["stageApplied"] is True
    assert result["qualityApproved"] is False
    assert result["executionVerified"] is False


def passthrough_voice() -> dict[str, object]:
    record = domain_composer_voice()
    cuts = e.rows(record["cuts"])
    cuts[0].update(
        {
            "source_end_sample": 96010,
            "filter_applied": False,
            "center_crop_start_sample": 0,
            "center_crop_end_sample": 96000,
            "source_context_samples": 0,
            "synthetic_left_context_samples": 0,
            "synthetic_right_context_samples": 0,
            "tail_error_samples": 0,
            "filter_output_samples": 96000,
            "expected_filter_output_samples": 96000,
        }
    )
    record["cuts"] = cuts
    return record


def test_exact_original_tempo_passthrough_is_not_degraded() -> None:
    result = domain_composer_assess(passthrough_voice())
    assert result["stageApplied"] is True
    assert result["qualityApproved"] is False


@pytest.fixture
def prospective_cut_data(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    quiet_path = str(Path.cwd() / "edits/quiet.json")
    record = domain_composer_cuts()
    for key in ("version", "clock", "fps"):
        del record[key]
    record.update(
        {
            "grid": "prospective 30fps; NOT yet normalized media",
            "normalizationVerified": False,
            "cutApproved": False,
            "outputFrames": 60,
            "decodedSourceStart": "0",
            "decodedSourceExtent": str(Fraction(e.integer(record["sourceFrames"]), 30)),
            "inputs": {"edits/quiet.json": "a" * 64},
            "audioSHA256": "b" * 64,
        }
    )
    records: dict[str, dict[str, object]] = {
        "/cuts": record,
        quiet_path: {"sourceSHA256": "b" * 64},
    }
    config: dict[str, object] = {
        "contract": "prospective-protected-grid-v1",
        "receipt": asr_evidence_ref("/cuts"),
        "fps": 30,
        "frames": 60,
        "expectedInputs": {"edits/quiet.json": "a" * 64},
        "quietInput": "edits/quiet.json",
        "quiet": asr_evidence_ref(quiet_path),
    }
    monkeypatch.setattr(
        e.MetadataReader, "read", lambda self, value: records[e.reference(value)["path"]]
    )
    return config, records


def test_original_prospective_cut_map_is_recorded_not_normalized(
    prospective_cut_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    result = e.domain_evidence("cuts", prospective_cut_data[0], e.MetadataReader())
    assert result["stageApplied"] is True
    assert result["originalNormalizationVerified"] is False
    assert result["originalCutApproved"] is False
    assert result["qualityApproved"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("grid", "30fps"),
        ("normalizationVerified", True),
        ("cutApproved", None),
        ("version", 1),
        ("decodedSourceStart", "1"),
        ("decodedSourceExtent", "1"),
        ("outputFrames", 59),
        ("sourceFrames", 1),
        ("inputs", {}),
        ("audioSHA256", "a" * 64),
    ],
)
def test_original_prospective_cut_mutation(
    prospective_cut_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    key: str,
    value: object,
) -> None:
    config, records = prospective_cut_data
    records["/cuts"][key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("cuts", config, e.MetadataReader())


@pytest.mark.parametrize("value", ["../quiet.json", "/edits/quiet.json", "edits/other.json"])
def test_original_quiet_lexical_binding_rejects_escape(
    prospective_cut_data: tuple[dict[str, object], dict[str, dict[str, object]]], value: str
) -> None:
    config, _ = prospective_cut_data
    config["quietInput"] = value
    with pytest.raises(ValueError):
        e.domain_evidence("cuts", config, e.MetadataReader())


@pytest.fixture
def parent_probed_cut_data(
    prospective_cut_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = prospective_cut_data
    record = records["/cuts"]
    record.pop("decodedSourceStart")
    record.pop("decodedSourceExtent")
    record.update(
        sourceFrames=3,
        outputFrames=2,
        deliveryComplete=False,
        previousRecognitionProtectionRetained=True,
    )
    record["cuts"] = [{"source_start_frame": 0, "source_end_frame": 2, "output_start_frame": 0}]
    inputs = e.mapping(record["inputs"])
    inputs["edits/parent.json"] = "a" * 64
    record["inputs"] = inputs
    parent_path = str(Path.cwd() / "edits/parent.json")
    probe_path = str(Path.cwd() / "edits/probe.json")
    parent = copy.deepcopy(record)
    parent["inputs"] = {"edits/quiet.json": "a" * 64, "edits/probe.json": "a" * 64}
    records[parent_path] = parent
    records[probe_path] = {
        "programs": [],
        "stream_groups": [],
        "streams": [{"index": 0, "start_pts": 0, "time_base": "1/1000"}],
        "frames": [
            {"pts": 0, "best_effort_timestamp": 0, "duration": 33},
            {"pts": 33, "best_effort_timestamp": 33, "duration": 34},
            {"pts": 67, "best_effort_timestamp": 67, "duration": 33},
        ],
    }
    records["/source-clock"] = {
        "version": 1,
        "frames": 3,
        "exactPresentationClockVerified": True,
        "compressedPacketsIdentical": True,
        "originalSourceIndependentlyDecoded": False,
    }
    config.update(
        contract="original-protected-grid-with-source-evidence-v1",
        frames=2,
        expectedInputs=copy.deepcopy(inputs),
        parent=asr_evidence_ref(parent_path),
        parentInput="edits/parent.json",
        decodedProbe=asr_evidence_ref(probe_path),
        decodedProbeInput="edits/probe.json",
        sourceClockEvidence=asr_evidence_ref("/source-clock"),
    )
    return config, records


def test_parent_probed_cut_records_original_extent_without_normalization(
    parent_probed_cut_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, _ = parent_probed_cut_data
    result = e.domain_evidence("cuts", config, e.MetadataReader())
    assert result["stageApplied"] is True
    assert result["decodedSourceExtent"] == "1/10"
    assert result["observedSourceFrames"] == 3
    assert result["originalNormalizationVerified"] is False
    assert result["currentCryptoVerified"] is False


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("/cuts", "previousRecognitionProtectionRetained", False),
        ("/cuts", "deliveryComplete", True),
        ("/cuts", "decodedSourceExtent", "1/10"),
        ("/cuts", "outputFrames", 3),
        ("/cuts", "sourceFrames", 4),
        ("/cuts", "inputs", {}),
        ("parent", "audioSHA256", "f" * 64),
        ("parent", "grid", "normalized"),
        ("parent", "sourceFrames", 4),
        ("parent", "outputFrames", 3),
        ("/source-clock", "frames", 4),
        ("/source-clock", "exactPresentationClockVerified", False),
        ("/source-clock", "originalSourceIndependentlyDecoded", True),
        ("quiet", "sourceSHA256", "f" * 64),
    ],
)
def test_parent_probed_cut_rejects_changed_record(
    parent_probed_cut_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    target: str,
    key: str,
    value: object,
) -> None:
    config, records = parent_probed_cut_data
    path = e.reference(config[target])["path"] if target in {"parent", "quiet"} else target
    records[path][key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("cuts", config, e.MetadataReader())


@pytest.mark.parametrize(
    "key,value",
    [
        ("parentInput", "../parent.json"),
        ("parentInput", "/edits/parent.json"),
        ("parentInput", "edits/./parent.json"),
        ("parentInput", "edits/other.json"),
        ("decodedProbeInput", "edits/other.json"),
        ("fps", 0),
        ("frames", 0),
    ],
)
def test_parent_probed_cut_rejects_changed_configuration(
    parent_probed_cut_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    key: str,
    value: object,
) -> None:
    config, _ = parent_probed_cut_data
    config[key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("cuts", config, e.MetadataReader())


@pytest.mark.parametrize(
    "key,value", [("pts", 0), ("duration", 0), ("best_effort_timestamp", 32), ("extra", 1)]
)
def test_parent_probed_cut_rejects_frame_mutation(
    parent_probed_cut_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    key: str,
    value: object,
) -> None:
    config, records = parent_probed_cut_data
    probe = records[e.reference(config["decodedProbe"])["path"]]
    frames = rows(probe["frames"])
    frames[1][key] = value
    probe["frames"] = frames
    with pytest.raises(ValueError):
        e.domain_evidence("cuts", config, e.MetadataReader())


@pytest.mark.parametrize(
    "key,value",
    [
        ("programs", [{}]),
        ("stream_groups", [{}]),
        ("streams", [{"index": 1, "start_pts": 0, "time_base": "1/1000"}]),
        ("frames", []),
        ("frames", "bad"),
    ],
)
def test_parent_probed_cut_rejects_probe_schema(
    parent_probed_cut_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    key: str,
    value: object,
) -> None:
    config, records = parent_probed_cut_data
    records[e.reference(config["decodedProbe"])["path"]][key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("cuts", config, e.MetadataReader())


@pytest.fixture
def native_grid_data(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    inputs = {
        "/parent": "a" * 64,
        "/quiet": "a" * 64,
        "/endpoints": "a" * 64,
        "/kernel": "a" * 64,
        "/original-producer": "b" * 64,
    }
    policy: dict[str, object] = {
        "originalAudioEnd": "4/5",
        "protectedSpeechEndIncludingGuard": "9/10",
        "supportedGridEnd": "1",
        "omittedTerminalVideoSeconds": "1/20",
        "oldPlanPreserved": True,
        "actualAudioOrProtectedSpeechTrimmed": False,
        "notMeasuredSilence": True,
        "audioPaddingApproved": False,
    }
    record: dict[str, object] = {
        "grid": "prospective conservative full-supported 10fps frames; NOT yet normalized media",
        "inputs": inputs,
        "sourceFrames": 10,
        "outputFrames": 10,
        "nativeVideoEnd": "21/20",
        "nativeAudioEnd": "9/10",
        "observedVFRSourceFrames": 9,
        "audioSHA256": "c" * 64,
        "normalizationVerified": False,
        "cutApproved": False,
        "deliveryComplete": False,
        "newInferencePerformed": False,
        "originalSpeechProtectionRetained": True,
        "terminalGridPolicy": policy,
        "cuts": [{"source_start_frame": 0, "source_end_frame": 10, "output_start_frame": 0}],
    }
    parent = copy.deepcopy(record)
    parent.update(
        grid="prospective 10fps; NOT yet normalized media",
        inputs={"/original-producer": "b" * 64},
        sourceFrames=11,
        outputFrames=11,
        cuts=[{"source_start_frame": 0, "source_end_frame": 11, "output_start_frame": 0}],
    )
    records: dict[str, dict[str, object]] = {
        "/native-cuts": record,
        "/parent": parent,
        "/quiet": {"sourceSHA256": "c" * 64, "samples": 9, "sampleRate": 10},
        "/endpoints": {
            "source": "/source",
            "sourceSHA256": "a" * 64,
            "metadata": {"streams": [{"index": 1, "codec_type": "audio", "time_base": "1/10"}]},
            "tailPackets": {
                "packets": [{"codec_type": "audio", "stream_index": 1, "pts": 0, "duration": 8}]
            },
        },
        "/assembly": {"frames": 9, "exactPresentationClockVerified": True},
    }
    config: dict[str, object] = {
        "contract": "original-native-grid-v2",
        "receipt": asr_evidence_ref("/native-cuts"),
        "parent": asr_evidence_ref("/parent"),
        "quiet": asr_evidence_ref("/quiet"),
        "endpoints": asr_evidence_ref("/endpoints"),
        "kernel": asr_evidence_ref("/kernel"),
        "source": asr_evidence_ref("/source"),
        "sourceClockEvidence": asr_evidence_ref("/assembly"),
        "fps": 10,
        "frames": 10,
        "expectedInputs": copy.deepcopy(inputs),
    }
    monkeypatch.setattr(
        e.MetadataReader, "read", lambda self, value: records[e.reference(value)["path"]]
    )
    return config, records


def test_native_terminal_grid_keeps_physical_and_prospective_counts_distinct(
    native_grid_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    result = e.domain_evidence("cuts", native_grid_data[0], e.MetadataReader())
    assert result["stageApplied"] is True
    assert result["physicalSourceFrames"] == 9
    assert result["prospectiveSourceFrames"] == 10
    assert result["originalNormalizationVerified"] is False
    assert result["currentCryptoVerified"] is False


@pytest.mark.parametrize(
    "record,key,value",
    [
        ("/native-cuts", "normalizationVerified", True),
        ("/native-cuts", "version", 1),
        ("/native-cuts", "outputFrames", 11),
        ("/native-cuts", "newInferencePerformed", True),
        ("/native-cuts", "inputs", {"/other": "a" * 64}),
        ("/parent", "inputs", {"/original-producer": "f" * 64}),
        ("/parent", "nativeAudioEnd", "1"),
        ("/parent", "sourceFrames", 12),
        ("/parent", "outputFrames", 12),
        (
            "/parent",
            "cuts",
            [{"source_start_frame": 1, "source_end_frame": 11, "output_start_frame": 0}],
        ),
        ("/quiet", "samples", 8),
        ("/quiet", "sourceSHA256", "f" * 64),
        ("/endpoints", "sourceSHA256", "f" * 64),
        (
            "/endpoints",
            "metadata",
            {"streams": [{"index": 1, "codec_type": "video", "time_base": "1/10"}]},
        ),
        (
            "/endpoints",
            "tailPackets",
            {"packets": [{"codec_type": "audio", "stream_index": 1, "pts": 0, "duration": 7}]},
        ),
        ("/assembly", "frames", 10),
        ("/assembly", "exactPresentationClockVerified", False),
    ],
)
def test_original_native_grid_rejects_changed_source_or_clock(
    native_grid_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    record: str,
    key: str,
    value: object,
) -> None:
    config, records = native_grid_data
    records[record][key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("cuts", config, e.MetadataReader())


@pytest.mark.parametrize(
    "key,value",
    [
        ("originalAudioEnd", "11/10"),
        ("protectedSpeechEndIncludingGuard", "11/10"),
        ("supportedGridEnd", "11/10"),
        ("omittedTerminalVideoSeconds", "0"),
        ("actualAudioOrProtectedSpeechTrimmed", True),
        ("oldPlanPreserved", False),
        ("notMeasuredSilence", False),
        ("audioPaddingApproved", True),
    ],
)
def test_original_native_terminal_policy_rejects_audio_or_protected_trim(
    native_grid_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    key: str,
    value: object,
) -> None:
    config, records = native_grid_data
    policy = mapping(records["/native-cuts"]["terminalGridPolicy"])
    policy[key] = value
    records["/native-cuts"]["terminalGridPolicy"] = policy
    with pytest.raises(ValueError):
        e.domain_evidence("cuts", config, e.MetadataReader())


@pytest.mark.parametrize(
    "value", [{"relative": "a" * 64}, {"/parent/../escape": "a" * 64}, {"//double": "a" * 64}, {}]
)
def test_original_absolute_inventory_rejects_ambiguous_names(value: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        e.original_absolute_pins(value)


@pytest.mark.parametrize(
    "key",
    [
        "source_end_sample",
        "center_crop_start_sample",
        "center_crop_end_sample",
        "source_context_samples",
        "synthetic_left_context_samples",
        "synthetic_right_context_samples",
        "tail_error_samples",
        "filter_output_samples",
        "expected_filter_output_samples",
    ],
)
@pytest.mark.parametrize("missing", [False, True])
def test_exact_passthrough_rejects_missing_or_changed_clock(key: str, missing: bool) -> None:
    record = passthrough_voice()
    cuts = e.rows(record["cuts"])
    if missing:
        del cuts[0][key]
    else:
        cuts[0][key] = e.integer(cuts[0][key]) + 1
    record["cuts"] = cuts
    with pytest.raises(ValueError):
        domain_composer_assess(record)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kind", "raw-audio"),
        ("version", 2),
        ("inputs", {"/original": "c" * 64}),
        ("output", "/wrong.wav"),
        ("outputSHA256", "c" * 64),
        ("sampleRate", 44100),
        ("channels", 1),
        ("outputStartSample", 1),
        ("outputEndSample", 1),
        ("outputSamples", 1),
        ("cuts", []),
    ],
)
def test_domain_composer_voice_refuses_changed_or_incomplete(key: str, value: object) -> None:
    record = domain_composer_voice()
    record[key] = value
    with pytest.raises(ValueError):
        domain_composer_assess(record)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("output_start_sample", 1),
        ("output_end_sample", 0),
        ("filter_applied", False),
        ("source_start_sample", -1),
        ("source_end_sample", 0),
        ("center_crop_end_sample", 96019),
    ],
)
def test_domain_composer_voice_slot_mutations(key: str, value: object) -> None:
    record = domain_composer_voice()
    domain_composer_cuts = e.rows(record["cuts"])
    domain_composer_cuts[0][key] = value
    record["cuts"] = domain_composer_cuts
    with pytest.raises(ValueError):
        domain_composer_assess(record)


def domain_composer_cuts() -> dict[str, object]:
    return {
        "version": 1,
        "clock": "source-presentation",
        "fps": 30,
        "sourceFrames": 100,
        "cuts": [
            {"source_start_frame": 0, "source_end_frame": 20, "output_start_frame": 0},
            {"source_start_frame": 40, "source_end_frame": 80, "output_start_frame": 20},
        ],
    }


def test_domain_composer_cut_clock_and_quality() -> None:
    assert e.cut_domain(domain_composer_cuts(), 60, 30)["stageApplied"] is True
    for frames, fps in [(61, 30), (60, 24), (0, 30)]:
        with pytest.raises(ValueError):
            e.cut_domain(domain_composer_cuts(), frames, fps)


@pytest.mark.parametrize("field", ["source_start_frame", "source_end_frame", "output_start_frame"])
def test_domain_composer_cut_invalid_geometry(field: str) -> None:
    record = domain_composer_cuts()
    items = e.rows(record["cuts"])
    items[1][field] = -1
    record["cuts"] = items
    with pytest.raises(ValueError):
        e.cut_domain(record, 60, 30)


def retained_domains_ref(name: str = "media") -> dict[str, object]:
    return {"path": "/" + name, "sha256": "a" * 64}


@pytest.mark.parametrize(
    "changed", [None, "reflow", "cuts", "template", "measurementReference", "incomplete"]
)
def test_retained_domains_head_caption_lineage(changed: str | None) -> None:
    keys = ("reflow", "cuts", "template", "measurementReference")
    config: dict[str, object] = {key: retained_domains_ref(key) for key in keys}
    inputs = {"/" + key: "a" * 64 for key in keys}
    if changed in keys:
        inputs["/" + str(changed)] = "b" * 64
    receipt: dict[str, object] = {
        "version": 1,
        "allCaptionStagesRetained": changed != "incomplete",
        "inputs": inputs,
    }
    if changed is not None:
        with pytest.raises(ValueError):
            e.head_lineage(receipt, config)
    else:
        result = e.head_lineage(receipt, config)
        assert result["stageApplied"] is True
        assert result["qualityApproved"] is False
        assert result["currentCryptoVerified"] is False


def test_retained_domains_cleanup_record_is_not_local_repair_approval() -> None:
    record: dict[str, object] = {
        "version": 1,
        "exactPresentationClockVerified": True,
        "output": retained_domains_ref(),
        "indexSHA256": "a" * 64,
        "frames": 10,
    }
    index: dict[str, object] = {
        "ranges": [
            {
                "media": "/media",
                "sha256": "a" * 64,
                "frames": 10,
                "sourceFirstPTS": 0,
                "sourceEndPTS": 30000,
            }
        ]
    }
    result = e.cleanup_assembly(
        record, index, retained_domains_ref("index"), retained_domains_ref()
    )
    assert result["qualityApproved"] is False
    assert result["pending"] == [
        "per-chunk local repair admission",
        "cleanup quality evidence",
        "original producer checkpoint reconstruction",
    ]
    for key, value in [
        ("version", 2),
        ("exactPresentationClockVerified", False),
        ("indexSHA256", "b" * 64),
        ("frames", 11),
    ]:
        changed = copy.deepcopy(record)
        changed[key] = value
        with pytest.raises(ValueError):
            e.cleanup_assembly(
                changed, index, retained_domains_ref("index"), retained_domains_ref()
            )
    for field, value in [("frames", 9), ("sourceEndPTS", 0), ("media", "/wrong")]:
        modified = rows(index["ranges"])
        modified[0] = {**modified[0], field: value}
        with pytest.raises(ValueError):
            e.cleanup_assembly(
                record, {"ranges": modified}, retained_domains_ref("index"), retained_domains_ref()
            )
    with pytest.raises(ValueError):
        e.cleanup_assembly(
            record, {"ranges": [{}, {}]}, retained_domains_ref("index"), retained_domains_ref()
        )


def retained_domains_proof() -> dict[str, object]:
    return {
        "version": 1,
        "part": 1,
        "partition": retained_domains_ref("partition"),
        "movie": retained_domains_ref(),
        "probeProvenance": {
            "version": 1,
            "decoderThreads": 0,
            "executable": retained_domains_ref("tool"),
        },
        "observed": {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 360,
                    "start_pts": 0,
                    "avg_frame_rate": "30/1",
                    "time_base": "1/3000",
                    "nb_read_frames": "2",
                    "duration_ts": 200,
                }
            ],
            "frames": [{"best_effort_timestamp": 0}, {"best_effort_timestamp": 100}],
        },
    }


def retained_domains_assess(record: dict[str, object]) -> dict[str, object]:
    return e.native_part(
        record,
        partition=retained_domains_ref("partition"),
        movie=retained_domains_ref(),
        part=1,
        frames=2,
        fps=30,
        width=640,
        height=360,
        tool=retained_domains_ref("tool"),
    )


def test_retained_domains_part_success_and_identity() -> None:
    assert retained_domains_assess(retained_domains_proof())["stageApplied"] is True
    for key, value in [
        ("version", 2),
        ("part", 2),
        ("movie", retained_domains_ref("wrong")),
        ("partition", retained_domains_ref("wrong")),
        ("probeProvenance", {"version": 2}),
    ]:
        record = retained_domains_proof()
        record[key] = value
        with pytest.raises(ValueError):
            retained_domains_assess(record)
    with pytest.raises(ValueError):
        e.reference({"path": "relative", "sha256": "a" * 64})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("width", 641),
        ("height", 361),
        ("codec_type", "audio"),
        ("start_pts", 1),
        ("avg_frame_rate", "24/1"),
        ("time_base", "0"),
        ("nb_read_frames", "3"),
        ("duration_ts", 201),
    ],
)
def test_retained_domains_native_video_mutations(key: str, value: object) -> None:
    record = retained_domains_proof()
    observed = mapping(record["observed"])
    video = rows(observed["streams"])[0]
    video[key] = value
    observed["streams"] = [video]
    record["observed"] = observed
    with pytest.raises(ValueError):
        retained_domains_assess(record)


def test_retained_domains_partial_and_wrong_pts() -> None:
    for changes in [
        {"frames": [{"best_effort_timestamp": 0}]},
        {"frames": [{"best_effort_timestamp": 0}, {"best_effort_timestamp": 101}]},
        {"streams": [{}, {}]},
    ]:
        record = retained_domains_proof()
        record["observed"] = {**mapping(record["observed"]), **mapping(changes)}
        with pytest.raises(ValueError):
            retained_domains_assess(record)


def test_retained_domains_caption_lineage_does_not_approve_style_or_model() -> None:
    record: dict[str, object] = {
        "version": 1,
        "captionMediaClockUnchanged": True,
        "output": retained_domains_ref(),
        "inputs": {"/font": "a" * 64},
    }
    result = e.caption_lineage(
        record, consumer=retained_domains_ref(), required_inputs={"/font": "a" * 64}
    )
    assert result["qualityApproved"] is False
    for changes in [
        {"captionMediaClockUnchanged": False},
        {"output": retained_domains_ref("old")},
        {"inputs": {"/font": "b" * 64}},
        {"inputs": dict.fromkeys(map(str, range(100001)), "a")},
    ]:
        with pytest.raises(ValueError):
            e.caption_lineage(
                {**record, **mapping(changes)},
                consumer=retained_domains_ref(),
                required_inputs={"/font": "a" * 64},
            )


def test_retained_domains_whole_native_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    partition: dict[str, object] = {
        "version": 1,
        "source": retained_domains_ref("semantic"),
        "globalCaptionCuesEqual": True,
        "duration": "1/15s",
        "parts": [{"globalStart": "0/1s", "duration": "1/15s"}],
    }
    publication: dict[str, object] = {
        "version": 1,
        "kind": "published-native-part",
        "movie": retained_domains_ref(),
        "frameProof": retained_domains_ref("proof"),
    }
    records = {
        "/partition": partition,
        "/publication": publication,
        "/proof": retained_domains_proof(),
    }

    def read(value: object) -> dict[str, object]:
        return records[str(mapping(value)["path"])]

    monkeypatch.setattr(e.MetadataReader, "read", lambda self, value: read(value))
    config: dict[str, object] = {
        "version": 1,
        "kind": "native-parts-frame-records",
        "partition": retained_domains_ref("partition"),
        "semanticXML": retained_domains_ref("semantic"),
        "publications": [retained_domains_ref("publication")],
        "fps": 30,
        "frames": 2,
        "width": 640,
        "height": 360,
        "tool": retained_domains_ref("tool"),
    }
    path = tmp_path / "whole.json"
    path.write_text(json.dumps(config))
    assert e.native_parts(config)["stageApplied"] is True
    publication.update(
        {
            "source": retained_domains_ref("staging"),
            "completion": retained_domains_ref("completion"),
            "intent": retained_domains_ref("intent"),
        }
    )
    records["/completion"] = {
        "version": 1,
        "kind": "verified-native-part",
        "part": 1,
        "movie": retained_domains_ref("staging"),
        "completionVerified": True,
        "shareObservedIdle": True,
        "fullAVDecodeExitCode": 0,
        "inputs": {"/tool": "a" * 64},
    }
    records["/intent"] = {
        "kind": "native-part-publication-intent",
        "source": retained_domains_ref("staging"),
        "completion": retained_domains_ref("completion"),
        "destination": retained_domains_ref()["path"],
    }
    completed = e.domain_evidence("native-parts", {"frames": config}, e.MetadataReader())
    assert completed["stageApplied"] is True
    assert completed["retirementDisposition"] == "not supplied; no current materialization inferred"
    with pytest.raises(ValueError):
        e.domain_evidence("native-parts", {"frames": config, "retirements": []}, e.MetadataReader())
    for change in [
        {
            "publications": [
                retained_domains_ref("publication"),
                retained_domains_ref("publication"),
            ]
        },
        {"fps": 0},
        {"frames": 3},
    ]:
        with pytest.raises(ValueError):
            e.native_parts({**config, **mapping(change)})
    partition["globalCaptionCuesEqual"] = False
    with pytest.raises(ValueError):
        e.native_parts(config)
    partition["globalCaptionCuesEqual"] = True
    partition["parts"] = [{"globalStart": "1/30s", "duration": "1/15s"}]
    with pytest.raises(ValueError):
        e.native_parts(config)
    partition["parts"] = [{"globalStart": "0/1s", "duration": "1/15s"}]
    publication["kind"] = "unverified"
    with pytest.raises(ValueError):
        e.native_parts(config)
    publication["kind"] = "published-native-part"
    partition["parts"] = [
        {"globalStart": "0/1s", "duration": "1/15s"},
        {"globalStart": "1/15s", "duration": "1/15s"},
    ]
    config["publications"] = [
        retained_domains_ref("publication"),
        retained_domains_ref("publication"),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        e.native_parts(config)


def asr_evidence_ref(path: str) -> dict[str, object]:
    return {"path": path, "sha256": "a" * 64}


@pytest.fixture
def asr_evidence_data(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    model = {"/model/revision/config.json": "a" * 64, "/model/revision/weights.npz": "b" * 64}
    runtime = {"/code/batch.py": "c" * 64, "/code/runtime.py": "d" * 64}
    records: dict[str, dict[str, object]] = {
        "/asr/inputs": {
            "version": 1,
            "clock": "input-pcm-not-projected",
            "device": "cpu",
            "dtypeContract": "provider-default-fp16-unchanged",
            "inputs": model | runtime | {"/pcm": "a" * 64},
            "samples": 100,
            "sampleRate": 16000,
            "windowSamples": 60,
            "strideSamples": 30,
        },
        "/asr/complete": {"windows": 3, "rawSHA256": ["a" * 64, "b" * 64, "c" * 64]},
        "/refined": {
            "clock": "input-pcm-not-projected",
            "refinement": {"clockOffsetApplications": 1, "sourceProjectionApplications": 0},
            "parentSHA256": {
                "/asr/inputs": "a" * 64,
                "/asr/complete": "a" * 64,
                "/asr/000000000000.raw.json": "a" * 64,
                "/asr/000000000030.raw.json": "b" * 64,
                "/asr/000000000060.raw.json": "c" * 64,
            },
        },
        "/source": {"clock": "source-presentation", "parentSHA256": {"/refined": "a" * 64}},
        "/tokens": {
            "clock": "output",
            "labels": asr_evidence_ref("/source"),
            "cuts": asr_evidence_ref("/cuts"),
        },
    }
    config: dict[str, object] = {
        "version": 1,
        "modelRepository": "mlx-community/whisper-large-v3-mlx",
        "modelPins": model,
        "runtimePins": runtime,
        "refinedLabels": asr_evidence_ref("/refined"),
        "sourceLabels": asr_evidence_ref("/source"),
        "projectedTokens": asr_evidence_ref("/tokens"),
        "cuts": asr_evidence_ref("/cuts"),
        "recognitions": [
            {
                "inputs": asr_evidence_ref("/asr/inputs"),
                "complete": asr_evidence_ref("/asr/complete"),
                "source": asr_evidence_ref("/pcm"),
            }
        ],
    }

    def read(value: object) -> dict[str, object]:
        return records[str(mapping(value)["path"])]

    monkeypatch.setattr(e.MetadataReader, "read", lambda self, value: read(value))
    return (config, records)


def test_asr_evidence_recorded_success(
    asr_evidence_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    result = e.domain_evidence("asr", asr_evidence_data[0], e.MetadataReader())
    assert result["executionRecorded"] is True
    assert result["currentCryptoVerified"] is False
    assert result["qualityApproved"] is False


@pytest.mark.parametrize(
    ("path", "key", "value"),
    [
        ("/asr/inputs", "clock", "output"),
        ("/asr/inputs", "device", "gpu"),
        ("/asr/inputs", "inputs", {"/turbo": "a" * 64}),
        ("/asr/inputs", "samples", 0),
        ("/asr/complete", "windows", 2),
        ("/asr/complete", "rawSHA256", ["a" * 64] * 3),
        ("/refined", "parentSHA256", {"/other": "a" * 64}),
        ("/refined", "clock", "source-presentation"),
        (
            "/refined",
            "refinement",
            {"clockOffsetApplications": 2, "sourceProjectionApplications": 0},
        ),
        ("/source", "clock", "output"),
        ("/source", "parentSHA256", {"/refined": "b" * 64}),
        ("/tokens", "clock", "source-presentation"),
        ("/tokens", "cuts", asr_evidence_ref("/old-cuts")),
    ],
)
def test_asr_evidence_rejects_original_boundary_changes(
    asr_evidence_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    path: str,
    key: str,
    value: object,
) -> None:
    config, records = asr_evidence_data
    records[path][key] = value
    with pytest.raises(ValueError):
        e.asr_evidence(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", 2),
        ("modelRepository", "turbo"),
        ("modelPins", {"/model/config.json": "a" * 64, "/other/weights.npz": "b" * 64}),
        ("modelPins", {"/model/turbo": "a" * 64}),
        ("runtimePins", {"/code": "a" * 64}),
        ("recognitions", []),
    ],
)
def test_asr_evidence_descriptor_rejection(
    asr_evidence_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    key: str,
    value: object,
) -> None:
    config, _ = asr_evidence_data
    config[key] = value
    with pytest.raises(ValueError):
        e.asr_evidence(config)


def test_asr_evidence_duplicate_and_missing_windows(
    asr_evidence_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, records = asr_evidence_data
    duplicate = copy.deepcopy(config)
    stages = config["recognitions"]
    assert isinstance(stages, list)
    duplicate["recognitions"] = [mapping(stages[0])] * 2
    with pytest.raises(ValueError):
        e.asr_evidence(duplicate)
    pins = mapping(records["/refined"]["parentSHA256"])
    del pins["/asr/000000000030.raw.json"]
    records["/refined"]["parentSHA256"] = pins
    with pytest.raises(ValueError):
        e.asr_evidence(config)


def test_asr_evidence_refinement_cannot_substitute_model(
    asr_evidence_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, records = asr_evidence_data
    records["/refine/inputs"] = copy.deepcopy(records["/asr/inputs"])
    records["/refine/complete"] = copy.deepcopy(records["/asr/complete"])
    pins = mapping(records["/refine/inputs"]["inputs"])
    pins["/model/revision/weights.npz"] = "f" * 64
    records["/refine/inputs"]["inputs"] = pins
    parent = mapping(records["/refined"]["parentSHA256"])
    parent["/refine/inputs"] = parent["/refine/complete"] = "a" * 64
    records["/refined"]["parentSHA256"] = parent
    stages = config["recognitions"]
    assert isinstance(stages, list)
    config["recognitions"] = [
        *stages,
        {
            "inputs": asr_evidence_ref("/refine/inputs"),
            "complete": asr_evidence_ref("/refine/complete"),
            "source": asr_evidence_ref("/pcm"),
        },
    ]
    with pytest.raises(ValueError, match="source/model/runtime"):
        e.asr_evidence(config)


def cleanup_evidence_ref(path: str) -> dict[str, object]:
    return {"path": path, "sha256": "a" * 64}


@pytest.fixture
def cleanup_evidence_data(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    records: dict[str, dict[str, object]] = {
        "/manifest": {
            "chunks": [
                {
                    "producer": cleanup_evidence_ref("/producer"),
                    "candidate": cleanup_evidence_ref("/candidate"),
                }
            ]
        },
        "/assembly": {
            "inputs": dict.fromkeys(
                ["/manifest", "/producer", "/candidate", "/repair/candidate.mp4"], "a" * 64
            ),
            "frames": 2,
            "exactPresentationClockVerified": True,
        },
        "/producer": {
            "source": "/source",
            "sourceSHA256": "a" * 64,
            "sourceEligibility": cleanup_evidence_ref("/eligibility"),
            "outputDirectory": "/repair",
            "clock": {"pts": [0, 3], "timeBase": "1/90", "lastDuration": 3},
        },
        "/candidate": {
            "inputs": {"/source": "a" * 64, "/eligibility": "a" * 64},
            "sourceEligibility": cleanup_evidence_ref("/eligibility"),
            "sourceStart": "0",
            "render": {"frames": 2},
            "activeSourceFrames": [0],
            "videoSHA256": "a" * 64,
        },
        "/eligibility": {
            "version": 1,
            "kind": "source-caption-eligibility",
            "sourceSHA256": "a" * 64,
            "frames": [
                {"sourcePTS": 0, "classification": "overlay-subtitle"},
                {"sourcePTS": 3, "classification": "unknown"},
            ],
        },
    }

    def read(value: object) -> dict[str, object]:
        return records[str(mapping(value)["path"])]

    monkeypatch.setattr(e.MetadataReader, "read", lambda self, value: read(value))
    return (
        {
            "version": 1,
            "manifest": cleanup_evidence_ref("/manifest"),
            "assemblyReceipt": cleanup_evidence_ref("/assembly"),
            "source": cleanup_evidence_ref("/source"),
        },
        records,
    )


def test_cleanup_evidence_unknown_not_repaired(
    cleanup_evidence_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    result = e.domain_evidence("cleanup", cleanup_evidence_data[0], e.MetadataReader())
    assert mapping(result["details"])["inactiveFramesNotApproved"] == 1
    assert result["qualityApproved"] is False


@pytest.fixture
def local_detection_data(
    cleanup_evidence_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = cleanup_evidence_data
    runtime = dict.fromkeys(
        [
            "/archive/subtitle_candidate.py",
            "/archive/subtitle_prepared.py",
            "/archive/subtitle_detector.py",
        ],
        "b" * 64,
    )
    detector: dict[str, object] = {
        "backend": "ppocrv3-db",
        "model": "/detector.onnx",
        "sha256": "c" * 64,
    }
    config.update({"contract": "recorded-local-text-detection-v1", "runtimePins": runtime})
    records["/producer"].pop("sourceEligibility")
    records["/producer"].update(
        {
            "nativeExecutable": "/native",
            "nativeSHA256": "d" * 64,
            "width": 10,
            "height": 10,
            "prepareVisionContext": False,
            "visionTextRegions": True,
            "learnedTextDetector": detector,
            "masks": [
                {
                    "startSeconds": 0,
                    "endSeconds": 0.06,
                    "region": {"x": 1, "y": 2, "width": 8, "height": 8},
                }
            ],
        }
    )
    records["/candidate"].pop("sourceEligibility")
    records["/candidate"].update(
        {
            "inputs": runtime
            | {
                "/source": "a" * 64,
                "/producer": "a" * 64,
                "/native": "d" * 64,
                "/detector.onnx": "c" * 64,
            },
            "render": {"frames": 2, "attempts": 2, "softened_frames": 2},
            "activeSourceFrames": [0, 1],
            "detectedSupportPixels": [3, 0],
            "emptyDetections": 1,
            "emptyDetectionIsNotAbsence": True,
            "learnedTextDetector": detector,
            "detectionBackend": "ppocrv3-db",
            "prepareVisionContext": False,
            "visionTextRegions": None,
            "learnedObservedRegions": [1, 0],
            "learnedCanvasIntersections": [0, 0],
            "fullDecodeVerified": False,
            "qualityApproval": False,
        }
    )
    return config, records


def test_original_local_detection_is_recorded_without_quality_or_new_eligibility(
    local_detection_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    result = e.domain_evidence("cleanup", local_detection_data[0], e.MetadataReader())
    assert mapping(result["details"])["repairActiveFrames"] == 2
    assert result["stageApplied"] is True
    assert result["qualityApproved"] is False
    assert result["currentCryptoVerified"] is False


@pytest.fixture
def legacy_asr_data(
    asr_evidence_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = asr_evidence_data
    runtime = {"/code/batch_asr.py": "c" * 64}
    config.update(
        {
            "contract": "legacy-native-selection-v1",
            "runtimePins": runtime,
            "source": asr_evidence_ref("/pcm"),
            "inputs": asr_evidence_ref("/asr/inputs"),
            "complete": asr_evidence_ref("/asr/complete"),
            "assembledLabels": asr_evidence_ref("/refined"),
            "nativePlan": asr_evidence_ref("/plan"),
            "nativeComplete": asr_evidence_ref("/native-complete"),
            "normalization": asr_evidence_ref("/normalization"),
        }
    )
    inputs = records["/asr/inputs"]
    del inputs["device"]
    del inputs["dtypeContract"]
    inputs["inputs"] = mapping(config["modelPins"]) | runtime | {"/pcm": "a" * 64}
    records["/refined"].pop("refinement")
    window: dict[str, object] = {
        "index": 0,
        "startSample": 10,
        "endSample": 40,
        "sampleRate": 16000,
        "audio": "/window.wav",
        "audioSHA256": "a" * 64,
    }
    records["/plan"] = {
        "source": "/pcm",
        "sourceSHA256": "a" * 64,
        "model": "/model/revision",
        "modelConfigSHA256": "a" * 64,
        "modelWeightsSHA256": "b" * 64,
        "windows": [window],
    }
    records["/native-complete"] = {
        "inputs": {
            "/asr/inputs": "a" * 64,
            "/asr/complete": "a" * 64,
            "/refined": "a" * 64,
            "/normalization": "a" * 64,
        },
        "windows": [
            window | {"raw": "/native/raw.json", "rawSHA256": "a" * 64, "status": "complete"}
        ],
    }
    records["/source"] = {
        "sourceDuration": 100 / 16000,
        "parentSHA256": {
            "/plan": "a" * 64,
            "/native-complete": "a" * 64,
            "/native/raw.json": "a" * 64,
            "/window.wav": "a" * 64,
        },
    }
    records["/normalization"] = {
        "clock": "native-source-presentation-already-projected",
        "noSecondClockProjection": True,
    }
    return config, records


def test_legacy_original_asr_does_not_invent_runtime_or_projection(
    legacy_asr_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    result = e.domain_evidence("asr", legacy_asr_data[0], e.MetadataReader())
    assert result["stageApplied"] is True
    assert result["qualityApproved"] is False
    assert mapping(result["details"])["originalWindows"] == 3
    pending = result["pending"]
    assert isinstance(pending, list)
    assert "native executor runtime checkpoint" in pending


@pytest.fixture
def native_context_data(
    legacy_asr_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = legacy_asr_data
    config.update(
        contract="original-native-context-chain-v1",
        nativeContextParents=[asr_evidence_ref("/points"), asr_evidence_ref("/collection")],
    )
    labels = [{"text": "hello", "start": 0, "end": 0.005}]
    records["/refined"]["labels"] = copy.deepcopy(labels)
    records["/source"]["labels"] = copy.deepcopy(labels)
    records["/source"]["parentSHA256"] = {
        "/points": "a" * 64,
        "/native/raw.json": "a" * 64,
        "/native/clock.json": "a" * 64,
    }
    records["/source"]["nativeContextSelections"] = [
        {
            "raw": "/native/raw.json",
            "clock": "/native/clock.json",
            "rawSHA256": "a" * 64,
            "before": copy.deepcopy(labels),
            "selection": {
                "original_start": 0,
                "original_end": 1,
                "replacement": copy.deepcopy(labels),
            },
        }
    ]
    records["/points"] = {
        "clock": "input-pcm-not-projected",
        "labels": copy.deepcopy(labels),
        "parentSHA256": {"/refined": "a" * 64, "/collection": "a" * 64},
    }
    records["/collection"] = {
        "deliveryComplete": False,
        "speechAccuracyVerified": False,
        "inputs": {"/pcm": "a" * 64},
        "windows": [{"window": 0, "rawSHA256": "a" * 64, "validated": True}],
    }
    config.update(
        refinementConfig=asr_evidence_ref("/repair"),
        refinementReceipt=asr_evidence_ref("/repair-receipt"),
        tokenSources=asr_evidence_ref("/sources"),
        segmentIndex=asr_evidence_ref("/index"),
    )
    repair: dict[str, object] = {
        "version": 1,
        "sourceSHA256": "a" * 64,
        "window": asr_evidence_ref("/window"),
        "audio": asr_evidence_ref("/audio"),
        "inputs": asr_evidence_ref("/local-inputs"),
        "recognition": asr_evidence_ref("/local-raw"),
        "clockEvidence": asr_evidence_ref("/normalization"),
        "cuts": config["cuts"],
        "manifest": asr_evidence_ref("/old-manifest"),
        "segmentOffsetFrame": 0,
        "replacement": {"startSample": 0, "endSample": 100},
    }
    records["/repair"] = repair
    chosen = {
        "text": "hello",
        "start": 0.0,
        "end": 0.005,
        "utterance": "refine:" + "a" * 64 + ":mlx:0",
    }
    records["/repair-receipt"] = {
        "version": 1,
        "configuration": copy.deepcopy(repair),
        "inputs": {
            "/repair": "a" * 64,
            "/window": "a" * 64,
            "/audio": "a" * 64,
            "/local-inputs": "a" * 64,
            "/local-raw": "a" * 64,
            "/normalization": "a" * 64,
            "/cuts": "a" * 64,
            "/old-manifest": "a" * 64,
            "/old-tokens": "a" * 64,
        },
        "selectedSourceTokens": [chosen],
        "tokens": asr_evidence_ref("/new-tokens"),
        "omittedByCuts": [],
        "removedTokenIndices": [0],
    }
    records["/window"] = {
        "version": 1,
        "source": "/pcm",
        "sourceSHA256": "a" * 64,
        "startSample": 0,
        "endSample": 100,
        "sampleRate": 16000,
        "windowSHA256": "a" * 64,
        "clock": "continuous-input-pcm",
        "sourceClockProjected": False,
    }
    records["/local-inputs"] = {
        "version": 1,
        "clock": "continuous-input-pcm",
        "model": "/model/revision",
        "config_sha256": "a" * 64,
        "weights_sha256": "b" * 64,
        "source": "/audio",
        "source_sha256": "a" * 64,
        "sample_rate": 16000,
        "samples": 100,
    }
    records["/local-raw"] = {
        "language": "ja",
        "text": "hello",
        "segments": [
            {
                "start": 0,
                "end": 0.005,
                "text": "hello",
                "words": [{"word": "hello", "start": 0, "end": 0.005, "probability": 0.9}],
            }
        ],
    }
    records["/old-manifest"] = {
        "version": 1,
        "tokens": asr_evidence_ref("/old-tokens"),
        "outputFrameRate": 30,
    }
    records["/adopted"] = {
        "version": 1,
        "tokens": asr_evidence_ref("/new-tokens"),
        "outputFrameRate": 30,
    }
    records["/old-tokens"] = {
        "version": 1,
        "clock": "output",
        "tokens": [{"text": "old", "start": 0, "end": 0.005, "utterance": "old"}],
    }
    records["/new-tokens"] = {
        "version": 1,
        "clock": "output",
        "tokens": [{**chosen, "utterance": str(chosen["utterance"]) + ":cut-0"}],
    }
    records["/sources"] = {
        "version": 1,
        "replacements": [
            {
                "segment": 0,
                "job": 0,
                "tokens": asr_evidence_ref("/new-tokens"),
                "adoptedManifest": asr_evidence_ref("/adopted"),
            }
        ],
    }
    records["/index"] = {
        "version": 1,
        "rate": 30,
        "segments": [{"segment": 0, "offsetFrame": 0, "frames": 1, "jobs": [0]}],
    }
    records["/cuts"] = {
        "cuts": [{"source_start_frame": 0, "source_end_frame": 1, "output_start_frame": 0}]
    }
    return config, records


def test_original_context_chain_preserves_missing_native_execution_pins(
    native_context_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, _ = native_context_data
    result = e.domain_evidence("asr", config, e.MetadataReader())
    assert result["stageApplied"] is True
    assert result["executionVerified"] is False
    assert mapping(result["details"])["originalWindows"] == 3
    assert mapping(result["details"])["nativeSelectionModelExecutionVerified"] is False
    assert mapping(result["details"])["secondSourceProjectionApplied"] is False


@pytest.mark.parametrize(
    "record,key,value",
    [
        ("/source", "clock", "output"),
        ("/source", "sourceDuration", 10),
        ("/source", "labels", []),
        ("/source", "labels", [{"text": "hello", "start": 0, "end": 1}]),
        ("/source", "parentSHA256", {"/points": "b" * 64}),
        ("/source", "parentSHA256", {"/missing": "a" * 64}),
        ("/source", "nativeContextSelections", "bad"),
        ("/points", "parentSHA256", {"/source": "a" * 64}),
        ("/collection", "inputs", {"/pcm": "b" * 64}),
        ("/collection", "windows", []),
        ("/collection", "windows", [{"window": 0, "rawSHA256": "a" * 64, "validated": False}]),
        ("/collection", "version", 1),
        ("/normalization", "noSecondClockProjection", False),
    ],
)
def test_original_context_chain_rejects_clock_parent_and_completion_changes(
    native_context_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    record: str,
    key: str,
    value: object,
) -> None:
    config, records = native_context_data
    records[record][key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("asr", config, e.MetadataReader())


@pytest.mark.parametrize(
    "key,value",
    [
        ("raw", "../raw"),
        ("raw", "/wrong"),
        ("rawSHA256", "f" * 64),
        ("clock", "/wrong"),
        ("clock", "../clock"),
        ("selection", {"original_start": 1, "original_end": 1}),
        ("before", []),
    ],
)
def test_original_context_selection_requires_actual_parent_pins(
    native_context_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    key: str,
    value: object,
) -> None:
    config, records = native_context_data
    selections = rows(records["/source"]["nativeContextSelections"])
    selections[0][key] = value
    records["/source"]["nativeContextSelections"] = selections
    with pytest.raises(ValueError):
        e.domain_evidence("asr", config, e.MetadataReader())


def test_original_context_keeps_empty_selection_as_no_new_inference(
    native_context_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, records = native_context_data
    records["/source"]["nativeContextSelections"] = []
    assert e.domain_evidence("asr", config, e.MetadataReader())["stageApplied"] is True


def test_original_context_rejects_duplicate_ancestor(
    native_context_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, _ = native_context_data
    config["nativeContextParents"] = [asr_evidence_ref("/points"), asr_evidence_ref("/points")]
    with pytest.raises(ValueError, match="Duplicate"):
        e.domain_evidence("asr", config, e.MetadataReader())


@pytest.mark.parametrize(
    "parents", [{"../bad": "a" * 64}, {"a/./b": "a" * 64}, {"//wrong": "a" * 64}, {}]
)
def test_original_parent_names_reject_escape_or_ambiguous_spelling(parents: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        e.original_parent_pins({"parentSHA256": parents})


def test_original_parent_names_preserve_relative_evidence_without_repinning() -> None:
    assert (
        e.original_parent_pins({"parentSHA256": {"edits/parent.json": "a" * 64}})[
            str(Path.cwd() / "edits/parent.json")
        ]
        == "a" * 64
    )


def test_original_parent_names_reject_duplicate_relative_absolute_identity() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        e.original_parent_pins(
            {
                "parentSHA256": {
                    "edits/parent.json": "a" * 64,
                    str(Path.cwd() / "edits/parent.json"): "a" * 64,
                }
            }
        )


@pytest.mark.parametrize(
    "record,key,value",
    [
        ("/repair", "version", 2),
        ("/repair-receipt", "inputs", {}),
        ("/window", "sourceClockProjected", True),
        ("/window", "sourceSHA256", "f" * 64),
        ("/window", "endSample", 0),
        ("/local-inputs", "model", "/turbo"),
        ("/local-inputs", "config_sha256", "f" * 64),
        ("/local-inputs", "samples", 99),
        ("/repair-receipt", "selectedSourceTokens", []),
        ("/adopted", "foreground", "changed"),
        ("/new-tokens", "clock", "source"),
        ("/old-tokens", "clock", "source"),
        ("/index", "rate", 0),
        ("/index", "segments", [{"segment": 0, "offsetFrame": 1, "frames": 1, "jobs": [0]}]),
        ("/repair-receipt", "removedTokenIndices", [1]),
        ("/repair-receipt", "removedTokenIndices", [0, 0]),
        ("/repair-receipt", "omittedByCuts", [{"text": "invented"}]),
        ("/new-tokens", "tokens", []),
        ("/sources", "replacements", []),
    ],
)
def test_native_context_refinement_rejects_changed_model_clock_or_output(
    native_context_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    record: str,
    key: str,
    value: object,
) -> None:
    config, records = native_context_data
    records[record][key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("asr", config, e.MetadataReader())


def test_original_token_selector_rejects_straddling_group() -> None:
    with pytest.raises(ValueError, match="crosses"):
        e.selected_original_token(SpokenToken("word", 0, 2, "source"), Fraction(1), Fraction(3))


def test_original_token_selector_does_not_select_outside_interval() -> None:
    assert (
        e.selected_original_token(SpokenToken("word", 0, 1, "source"), Fraction(2), Fraction(3))
        is False
    )


@pytest.fixture
def approved_original_data(tmp_path: Path) -> dict[str, object]:
    canonical = tmp_path / "source.bin"
    canonical.write_bytes(b"synthetic source")
    alias = tmp_path / "historical.bin"
    alias.symlink_to(canonical)
    return {
        "contract": "recorded-local-text-detection-v1",
        "source": asr_evidence_ref(str(alias)),
        "approvedOriginal": asr_evidence_ref(str(canonical)),
    }


def test_approved_original_observes_only_explicit_same_target(
    approved_original_data: dict[str, object],
) -> None:
    before = e.approved_original_identity(approved_original_data)
    assert before is not None
    assert before[2] == 16
    assert e.approved_original_identity(approved_original_data) == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("contract", "other"),
        ("approvedOriginal", {"path": "/other", "sha256": "b" * 64}),
        ("source", {"path": "/other", "sha256": "f" * 64}),
    ],
)
def test_approved_original_rejects_wrong_contract_or_digest(
    approved_original_data: dict[str, object], key: str, value: object
) -> None:
    approved_original_data[key] = value
    with pytest.raises(ValueError):
        e.approved_original_identity(approved_original_data)


def test_approved_original_rejects_retarget_even_when_bytes_equal(
    approved_original_data: dict[str, object], tmp_path: Path
) -> None:
    alias = Path(e.reference(approved_original_data["source"])["path"])
    other = tmp_path / "other.bin"
    other.write_bytes(b"synthetic source")
    alias.unlink()
    alias.symlink_to(other)
    with pytest.raises(ValueError, match="target differs"):
        e.approved_original_identity(approved_original_data)


def test_approved_original_rejects_noncanonical_destination(
    approved_original_data: dict[str, object],
) -> None:
    approved_original_data["approvedOriginal"] = approved_original_data["source"]
    with pytest.raises(ValueError):
        e.approved_original_identity(approved_original_data)


def test_approved_original_rejects_empty_source(approved_original_data: dict[str, object]) -> None:
    Path(e.reference(approved_original_data["approvedOriginal"])["path"]).write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty"):
        e.approved_original_identity(approved_original_data)


def test_cleanup_rechecks_approved_original_after_metadata(
    local_detection_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = iter([(1, 2, 3), (1, 2, 4)])
    monkeypatch.setattr(e, "approved_original_identity", lambda config: next(observations))
    with pytest.raises(ValueError, match="changed during"):
        e.cleanup_evidence(local_detection_data[0])


@pytest.mark.parametrize(
    "record,key,value",
    [
        ("/asr/inputs", "version", 2),
        ("/asr/inputs", "clock", "output"),
        ("/asr/inputs", "inputs", {}),
        ("/refined", "clock", "source-presentation"),
        ("/refined", "parentSHA256", {}),
        ("/asr/complete", "windows", 2),
        ("/source", "parentSHA256", {}),
        ("/source", "sourceDuration", 100),
        ("/source", "clock", "output"),
        ("/normalization", "noSecondClockProjection", False),
        ("/normalization", "clock", "output"),
        ("/native-complete", "inputs", {}),
        ("/plan", "model", "/turbo"),
        ("/plan", "sourceSHA256", "b" * 64),
        ("/plan", "windows", []),
        ("/native-complete", "windows", []),
        ("/tokens", "clock", "input-pcm-not-projected"),
        ("/tokens", "labels", asr_evidence_ref("/other")),
    ],
)
def test_legacy_original_asr_rejects_changed_original_contract(
    legacy_asr_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    record: str,
    key: str,
    value: object,
) -> None:
    config, records = legacy_asr_data
    records[record][key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("asr", config, e.MetadataReader())


@pytest.mark.parametrize(
    "key,value",
    [("modelRepository", "turbo"), ("modelPins", {}), ("runtimePins", {}), ("contract", "unknown")],
)
def test_legacy_original_asr_rejects_descriptor_substitution(
    legacy_asr_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    key: str,
    value: object,
) -> None:
    config, _ = legacy_asr_data
    config[key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("asr", config, e.MetadataReader())


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "running"),
        ("index", 1),
        ("endSample", 101),
        ("rawSHA256", "b" * 64),
        ("sampleRate", 48000),
    ],
)
def test_legacy_native_window_failure(
    legacy_asr_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    key: str,
    value: object,
) -> None:
    config, records = legacy_asr_data
    windows = rows(records["/native-complete"]["windows"])
    windows[0][key] = value
    records["/native-complete"]["windows"] = windows
    with pytest.raises(ValueError):
        e.domain_evidence("asr", config, e.MetadataReader())


@pytest.mark.parametrize(
    ("record", "key", "value"),
    [
        ("/producer", "sourceSHA256", "f" * 64),
        ("/producer", "visionTextRegions", False),
        ("/producer", "prepareVisionContext", True),
        ("/producer", "learnedTextDetector", {"backend": "fixed-rectangle"}),
        ("/producer", "sourceEligibility", {"path": "/new", "sha256": "a" * 64}),
        ("/producer", "clock", {"pts": [3, 0], "timeBase": "1/90", "lastDuration": 3}),
        ("/producer", "clock", {"pts": [], "timeBase": "1/90", "lastDuration": 3}),
        (
            "/producer",
            "masks",
            [
                {
                    "startSeconds": 0,
                    "endSeconds": 1,
                    "region": {"x": 9, "y": 0, "width": 5, "height": 3},
                }
            ],
        ),
        ("/producer", "masks", []),
        ("/producer", "nativeExecutable", "/archive/subtitle_candidate.py"),
        ("/candidate", "inputs", {"/source": "a" * 64}),
        ("/candidate", "sourceStart", "1/30"),
        ("/candidate", "activeSourceFrames", [1, 1]),
        ("/candidate", "render", {"frames": 1, "attempts": 2, "softened_frames": 1}),
        ("/candidate", "render", {"frames": 2, "attempts": 2, "softened_frames": 1}),
        ("/candidate", "detectedSupportPixels", [3]),
        ("/candidate", "detectedSupportPixels", [101, 0]),
        ("/candidate", "detectedSupportPixels", [65, 0]),
        (
            "/candidate",
            "learnedTextDetector",
            {"backend": "ppocrv3-db", "model": "/wrong.onnx", "sha256": "c" * 64},
        ),
        ("/candidate", "detectedSupportPixels", [-1, 0]),
        ("/candidate", "emptyDetections", 0),
        ("/candidate", "emptyDetectionIsNotAbsence", False),
        ("/candidate", "learnedObservedRegions", [1]),
        ("/candidate", "learnedCanvasIntersections", [0, -1]),
        ("/candidate", "detectionBackend", "fixed-rectangle"),
        ("/candidate", "videoSHA256", "f" * 64),
        ("/assembly", "frames", 3),
    ],
)
def test_original_local_detection_rejects_changed_provenance_or_incomplete_application(
    local_detection_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    record: str,
    key: str,
    value: object,
) -> None:
    config, records = local_detection_data
    records[record][key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("cleanup", config, e.MetadataReader())


@pytest.mark.parametrize("change", ["runtime", "unknown-contract"])
def test_original_local_detection_requires_explicit_supported_contract(
    local_detection_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    change: str,
) -> None:
    config = local_detection_data[0]
    config["runtimePins" if change == "runtime" else "contract"] = (
        {"/unknown.py": "a" * 64} if change == "runtime" else "unknown"
    )
    with pytest.raises(ValueError):
        e.domain_evidence("cleanup", config, e.MetadataReader())


@pytest.mark.parametrize(
    ("path", "key", "value"),
    [
        ("/producer", "sourceSHA256", "b" * 64),
        ("/candidate", "inputs", {"/wrong": "a" * 64}),
        ("/candidate", "inputs", {"/source": "a" * 64}),
        ("/candidate", "sourceStart", "1"),
        ("/candidate", "render", {"frames": 1}),
        ("/candidate", "activeSourceFrames", [0, 0]),
        ("/candidate", "activeSourceFrames", [1]),
        ("/candidate", "activeSourceFrames", [3]),
        ("/candidate", "activeSourceFrames", [0, 1, 2]),
        ("/candidate", "videoSHA256", "b" * 64),
        ("/eligibility", "kind", "unclassified"),
        ("/eligibility", "frames", [{"sourcePTS": 0}, {"sourcePTS": 0}]),
        ("/producer", "clock", {"pts": [], "timeBase": "1/90", "lastDuration": 3}),
        ("/producer", "clock", {"pts": [0, 0], "timeBase": "1/90", "lastDuration": 3}),
        ("/assembly", "frames", 3),
        ("/assembly", "inputs", {"/manifest": "b" * 64}),
    ],
)
def test_cleanup_evidence_mutations(
    cleanup_evidence_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    path: str,
    key: str,
    value: object,
) -> None:
    config, records = cleanup_evidence_data
    records[path][key] = value
    with pytest.raises(ValueError):
        e.cleanup_evidence(config)


def test_cleanup_evidence_duplicate_and_gap(
    cleanup_evidence_data: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, records = cleanup_evidence_data
    records["/manifest"]["chunks"] = [
        {
            "producer": cleanup_evidence_ref("/producer"),
            "candidate": cleanup_evidence_ref("/candidate"),
        }
    ] * 2
    with pytest.raises(ValueError):
        e.cleanup_evidence(config)
    records["/producer2"] = copy.deepcopy(records["/producer"])
    records["/candidate2"] = copy.deepcopy(records["/candidate"])
    records["/manifest"]["chunks"] = [
        {
            "producer": cleanup_evidence_ref("/producer"),
            "candidate": cleanup_evidence_ref("/candidate"),
        },
        {
            "producer": cleanup_evidence_ref("/producer2"),
            "candidate": cleanup_evidence_ref("/candidate2"),
        },
    ]
    records["/assembly"]["inputs"] = {
        **mapping(records["/assembly"]["inputs"]),
        "/producer2": "a" * 64,
        "/candidate2": "a" * 64,
    }
    with pytest.raises(ValueError, match="gap or overlap"):
        e.cleanup_evidence(config)


def followup_history_put(root: Path, name: str, value: object) -> dict[str, str]:
    path = root / name
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def followup_history_descriptor(root: Path, mutation: str = "") -> dict[str, object]:
    metadata = followup_history_put(root, "metadata.json", {"plan": 1})
    terminal = followup_history_put(
        root,
        "terminal.json",
        {
            "inputs": {"/python": "a" * 64} if mutation != "input" else {"/other": "a" * 64},
            "kind": "other"
            if mutation == "kind"
            else "verified-common-context-native-batch"
            if mutation == "batch"
            else "verified-integrated-native-followup",
            "outputs": [{"path": "/archived/movie.mov", "sha256": "b" * 64}]
            * (2 if mutation == "duplicate" else 1),
        },
    )
    stage: dict[str, object] = {
        "name": "followup",
        "command": ["/python", "run-integrated"],
        "verify_command": ["/python", "verify-integrated"],
        "inputs": ["/python"],
        "outputs": ["/archived/movie.mov", terminal["path"]],
        "timeout_seconds": 30,
    }
    if mutation == "batch":
        stage["outputs"] = [terminal["path"]]
    if mutation == "supplement":
        stage["inputs"] = ["/python", metadata["path"]]
    graph = followup_history_put(root, "graph.json", {"stages": [stage]})
    pins = {"/python": "a" * 64, graph["path"]: graph["sha256"]}
    if mutation == "workflow":
        pins[graph["path"]] = "c" * 64
    if mutation == "input":
        del pins["/python"]
    if mutation == "conflict":
        pins["/python"] = "d" * 64
    fingerprint_inputs = {"/python": "a" * 64}
    if mutation == "supplement":
        fingerprint_inputs[metadata["path"]] = metadata["sha256"]
    definition = {
        "command": stage["command"],
        "verifyCommand": stage["verify_command"],
        "inputs": fingerprint_inputs,
        "outputs": stage["outputs"],
        "timeoutSeconds": 30,
        "workingDirectory": "/repository",
    }
    checkpoint = {
        "status": "running" if mutation == "running" else "complete",
        "fingerprint": hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
        "outputs": {"/archived/movie.mov": "b" * 64, terminal["path"]: terminal["sha256"]},
    }
    if mutation == "batch":
        checkpoint["outputs"] = {terminal["path"]: terminal["sha256"]}
    result: dict[str, object] = {
        "version": 2 if mutation == "version" else 1,
        "workflow": graph,
        "terminal": terminal,
        "binding": followup_history_put(root, "binding.json", {"inputs": pins}),
        "checkpoint": followup_history_put(root, "checkpoint.json", checkpoint),
        "stage": "missing" if mutation == "stage" else "followup",
        "cwd": "/repository",
    }
    if mutation == "supplement":
        result["originalMetadata"] = [metadata]
    return result


def test_followup_history_archived_checkpoint_requires_no_media_or_executable(
    tmp_path: Path,
) -> None:
    result = e.followup_history(followup_history_descriptor(tmp_path))
    assert result["historicalExecutionVerified"] is True
    assert result["currentOutputsCryptoVerified"] is False
    assert result["qualityApproved"] is False


@pytest.mark.parametrize(
    "mutation",
    ["kind", "duplicate", "workflow", "input", "running", "version", "stage", "conflict"],
)
def test_followup_history_reject_changed_original_contract(tmp_path: Path, mutation: str) -> None:
    with pytest.raises(ValueError):
        e.followup_history(followup_history_descriptor(tmp_path, mutation))


@pytest.mark.parametrize("kind", ["batch", "supplement"])
def test_followup_history_original_metadata_contract(tmp_path: Path, kind: str) -> None:
    assert (
        e.followup_history(followup_history_descriptor(tmp_path, kind))[
            "historicalExecutionVerified"
        ]
        is True
    )


def test_followup_history_supplement_cannot_override_inventory(tmp_path: Path) -> None:
    config = followup_history_descriptor(tmp_path)
    config["originalMetadata"] = [{"path": "/python", "sha256": "a" * 64}]
    with pytest.raises(ValueError, match="missing original"):
        e.followup_history(config)


def publication_evidence_fixture(root: Path, change: str = "") -> dict[str, str]:
    source = {"path": "/old.mov", "sha256": "a" * 64}
    movie = {"path": "/ssd.mov", "sha256": "a" * 64}
    completion = followup_history_put(
        root,
        "completion.json",
        {
            "version": 1,
            "kind": "bad" if change == "completion" else "verified-native-part",
            "part": 2 if change == "part" else 1,
            "movie": source,
            "completionVerified": change != "verify",
            "fullAVDecodeExitCode": 1 if change == "decode" else 0,
            "shareObservedIdle": change != "idle",
            "nativeDispatchVerified": False,
            "inputs": {"/tool": "c" * 64},
        },
    )
    intent = followup_history_put(
        root,
        "intent.json",
        {
            "kind": "bad" if change == "intent" else "native-part-publication-intent",
            "source": source,
            "destination": "/wrong" if change == "destination" else movie["path"],
            "completion": completion,
        },
    )
    proof = {"path": "/proof.json", "sha256": "b" * 64}
    publication = followup_history_put(
        root,
        "publication.json",
        {
            "version": 1,
            "kind": "published-native-part",
            "source": source,
            "movie": movie,
            "completion": completion,
            "intent": intent,
            "frameProof": proof,
        },
    )
    retirement = followup_history_put(
        root,
        "retirement.json",
        {
            "kind": "bad" if change == "retirement" else "verified-staging-retirement",
            "part": 1,
            "source": source,
            "movie": movie,
            "completion": completion,
            "intent": intent,
            "publication": publication,
            "frameProof": proof,
        },
    )
    attempt = followup_history_put(
        root,
        "attempt.json",
        {
            "status": "running" if change == "attempt" else "complete",
            "code": 0,
            "command": ["/python", "/wrong" if change == "workflow" else "/workflow.json"],
        },
    )
    absence = followup_history_put(
        root,
        "absence.json",
        {
            "source": source["path"],
            "replacement": movie["path"],
            "deletionActuallyPerformed": True,
            "bothNotExistsAndNotSymlink": change != "absence",
            "historyManifest": followup_history_put(root, "history.json", {"records": []}),
        },
    )
    return followup_history_put(
        root,
        "result.json",
        {
            "version": 2 if change == "version" else 1,
            "part": 1,
            "producerExit": 1 if change == "producer" else 0,
            "mandatoryConsumerExit": 0,
            "postDeleteConsumerExit": 0,
            "deletionActuallyPerformed": change != "prepare",
            "retirementReceipt": retirement,
            "source": source["path"],
            "replacement": movie["path"],
            "sourceSHA256": "f" * 64 if change == "bytes" else source["sha256"],
            "postDeleteSupervisorAttempt": attempt,
            "mandatoryConsumerWorkflow": "/workflow.json",
            "actualAbsenceRecord": absence,
        },
    )


def test_publication_evidence_recorded_retirement_does_not_read_absent_media(
    tmp_path: Path,
) -> None:
    result = e.publication_evidence(publication_evidence_fixture(tmp_path))
    assert result["publicationRecorded"] is True
    assert result["retirementRecorded"] is True
    assert result["nativeDispatchVerified"] is False
    assert result["currentCryptoVerified"] is False
    assert result["qualityApproved"] is False


@pytest.mark.parametrize(
    "change",
    [
        "completion",
        "verify",
        "decode",
        "idle",
        "intent",
        "destination",
        "part",
        "retirement",
        "attempt",
        "workflow",
        "absence",
        "version",
        "producer",
        "prepare",
        "bytes",
    ],
)
def test_publication_evidence_reject_incomplete_or_changed_lineage(
    tmp_path: Path, change: str
) -> None:
    with pytest.raises(ValueError):
        e.publication_evidence(publication_evidence_fixture(tmp_path, change))


@pytest.mark.parametrize("changed", [False, True])
def test_publication_evidence_new_absence_schema_keeps_history_gap(
    tmp_path: Path, changed: bool
) -> None:
    ref = publication_evidence_fixture(tmp_path)
    result = json.loads(Path(ref["path"]).read_text())
    absence = json.loads(Path(result["actualAbsenceRecord"]["path"]).read_text())
    del absence["bothNotExistsAndNotSymlink"]
    del absence["historyManifest"]
    absence.update(
        actualAbsenceBothNotExistsAndNotSymlink=True,
        replacementIdentityUnchanged=not changed,
        sourceSHA256="a" * 64,
    )
    result["actualAbsenceRecord"] = followup_history_put(tmp_path, "new-absence.json", absence)
    new_ref = followup_history_put(tmp_path, "new-result.json", result)
    if changed:
        with pytest.raises(ValueError, match="Replacement identity"):
            e.publication_evidence(new_ref)
    else:
        assert e.publication_evidence(new_ref)["historyManifestBound"] is False


def execution_evidence_envelope(payload: object) -> dict[str, object]:
    return {
        "ok": True,
        "data": {
            "isError": False,
            "structuredContent": {
                "truncated": False,
                "exitStatus": 0,
                "stdout": json.dumps(payload),
                "completionVerified": False,
                "outputPath": "/final.mp4",
            },
        },
    }


@pytest.fixture
def execution_evidence_native() -> dict[str, dict[str, object]]:
    return {
        "submit": execution_evidence_envelope(
            {"batch": {"batchID": "batch", "jobs": [{"jobID": "job"}]}}
        ),
        "status": execution_evidence_envelope(
            [
                {
                    "jobStatus": {
                        "jobid": "job",
                        "batchid": "batch",
                        "status": "Successful",
                        "percentComplete": "100",
                    }
                }
            ]
        ),
        "request": {
            "version": 1,
            "kind": "whole-compressor-request",
            "wholeMaster": "/whole.mov",
            "inputs": {"/whole.mov": "a" * 64},
        },
        "success": {
            "jobUUID": "job",
            "batchUUID": "batch",
            "output": {"path": "/final.mp4"},
            "source": {"path": "/whole.mov", "sha256": "a" * 64},
        },
    }


@pytest.fixture
def execution_evidence_stage() -> dict[str, object]:
    return {
        "name": "decode",
        "command": ["/tool", "decode"],
        "verify_command": ["/check"],
        "inputs": ["/tool"],
        "outputs": ["/receipt"],
        "timeout_seconds": 30,
    }


def execution_evidence_checkpoint(
    execution_evidence_stage: dict[str, object], output_hash: str = "b" * 64
) -> dict[str, object]:
    definition = {
        "command": execution_evidence_stage["command"],
        "verifyCommand": execution_evidence_stage["verify_command"],
        "inputs": {"/tool": "a" * 64},
        "outputs": execution_evidence_stage["outputs"],
        "timeoutSeconds": 30,
        "workingDirectory": "/repository",
    }
    return {
        "status": "complete",
        "fingerprint": hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
        "outputs": {e.strings(execution_evidence_stage["outputs"])[0]: output_hash},
    }


def test_execution_evidence_stage(execution_evidence_stage: dict[str, object]) -> None:
    result = e.stage_provenance(
        stage=execution_evidence_stage,
        checkpoint=execution_evidence_checkpoint(execution_evidence_stage),
        input_hashes={"/tool": "a" * 64},
        cwd="/repository",
        expected_outputs={"/receipt": "b" * 64},
    )
    assert result["historicalExecutionVerified"] is True
    assert result["currentCryptoVerified"] is False


@pytest.mark.parametrize("omitted", [True, False])
def test_original_stage_optional_verifier_default(
    execution_evidence_stage: dict[str, object], omitted: bool
) -> None:
    stage = dict(execution_evidence_stage)
    stage["verify_command"] = []
    checkpoint = execution_evidence_checkpoint(stage)
    if omitted:
        stage.pop("verify_command")
    result = e.stage_provenance(
        stage=stage,
        checkpoint=checkpoint,
        input_hashes={"/tool": "a" * 64},
        cwd="/repository",
        expected_outputs={"/receipt": "b" * 64},
    )
    assert result["historicalExecutionVerified"] is True
    assert result["currentCryptoVerified"] is False


@pytest.mark.parametrize("value", [None, "", [1], [""], {}])
def test_original_stage_rejects_malformed_optional_verifier(
    execution_evidence_stage: dict[str, object], value: object
) -> None:
    stage = dict(execution_evidence_stage)
    stage["verify_command"] = value
    with pytest.raises(ValueError):
        e.stage_provenance(
            stage=stage,
            checkpoint={},
            input_hashes={"/tool": "a" * 64},
            cwd="/repository",
            expected_outputs={"/receipt": "b" * 64},
        )


@pytest.mark.parametrize(
    "change",
    [
        {"command": ["/other"]},
        {"verify_command": ["/other"]},
        {"inputs": ["/tool", "/tool"]},
        {"outputs": ["/different"]},
        {"timeout_seconds": 31},
    ],
)
def test_execution_evidence_changed_stage(
    execution_evidence_stage: dict[str, object], change: dict[str, object]
) -> None:
    original = execution_evidence_checkpoint(execution_evidence_stage)
    execution_evidence_stage.update(change)
    with pytest.raises(ValueError):
        e.stage_provenance(
            stage=execution_evidence_stage,
            checkpoint=original,
            input_hashes={"/tool": "a" * 64},
            cwd="/repository",
            expected_outputs={"/receipt": "b" * 64},
        )


@pytest.mark.parametrize(
    "change",
    [{"status": "running"}, {"outputs": {"/receipt": "c" * 64}}, {"fingerprint": "0" * 64}],
)
def test_execution_evidence_changed_checkpoint(
    execution_evidence_stage: dict[str, object], change: dict[str, object]
) -> None:
    record = execution_evidence_checkpoint(execution_evidence_stage)
    record.update(change)
    with pytest.raises(ValueError):
        e.stage_provenance(
            stage=execution_evidence_stage,
            checkpoint=record,
            input_hashes={"/tool": "a" * 64},
            cwd="/repository",
            expected_outputs={"/receipt": "b" * 64},
        )


def test_execution_evidence_wrong_tool(execution_evidence_stage: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        e.stage_provenance(
            stage=execution_evidence_stage,
            checkpoint=execution_evidence_checkpoint(execution_evidence_stage),
            input_hashes={"/tool": "c" * 64},
            cwd="/repository",
            expected_outputs={"/receipt": "b" * 64},
        )


def test_execution_evidence_official(
    execution_evidence_native: dict[str, dict[str, object]],
) -> None:
    result = e.compressor_execution(
        submit=execution_evidence_native["submit"],
        status=execution_evidence_native["status"],
        request=execution_evidence_native["request"],
        success=execution_evidence_native["success"],
        output_path="/final.mp4",
    )
    assert result["officialSuccessful"] is True
    assert result["adapterCompletionVerified"] is False
    assert result["globalDuplicateAbsenceVerified"] is False


@pytest.mark.parametrize(
    "status,job,percent",
    [
        ("Failed", "job", "100"),
        ("Successful", "other", "100"),
        ("Running", "job", "100"),
        ("Successful", "job", "99"),
    ],
)
def test_execution_evidence_bad_official(
    execution_evidence_native: dict[str, dict[str, object]], status: str, job: str, percent: str
) -> None:
    observed = execution_evidence_envelope(
        [
            {
                "jobStatus": {
                    "jobid": job,
                    "batchid": "batch",
                    "status": status,
                    "percentComplete": percent,
                }
            }
        ]
    )
    with pytest.raises(ValueError):
        e.compressor_execution(
            submit=execution_evidence_native["submit"],
            status=observed,
            request=execution_evidence_native["request"],
            success=execution_evidence_native["success"],
            output_path="/final.mp4",
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"ok": False},
        {"ok": True, "data": {"isError": True, "structuredContent": {}}},
        {
            "ok": True,
            "data": {"isError": False, "structuredContent": {"truncated": False, "exitStatus": 1}},
        },
    ],
)
def test_execution_evidence_false_envelope(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        e.official_payload(bad)


def test_execution_evidence_invalid_list_and_cwd(
    execution_evidence_stage: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        e.strings([])
    with pytest.raises(ValueError, match="working directory"):
        e.stage_provenance(
            stage=execution_evidence_stage,
            checkpoint=execution_evidence_checkpoint(execution_evidence_stage),
            input_hashes={"/tool": "a" * 64},
            cwd="relative",
            expected_outputs={"/receipt": "b" * 64},
        )


def test_published_completion_without_retirement(tmp_path: Path) -> None:
    result_ref = publication_evidence_fixture(tmp_path)
    reader = e.MetadataReader()
    retired = reader.read(reader.read(result_ref)["retirementReceipt"])
    result = e.published_completion(retired["publication"], reader)
    assert result == {
        "part": 1,
        "nativeDispatchVerified": False,
        "recordedFullAVDecodeVerified": True,
        "currentCryptoVerified": False,
        "qualityApproved": False,
    }


@pytest.mark.parametrize(
    "change", ["completion", "verify", "decode", "idle", "intent", "destination"]
)
def test_published_completion_rejects_partial_original(tmp_path: Path, change: str) -> None:
    result_ref = publication_evidence_fixture(tmp_path, change)
    reader = e.MetadataReader()
    retired = reader.read(reader.read(result_ref)["retirementReceipt"])
    with pytest.raises(ValueError):
        e.published_completion(retired["publication"], reader)


def test_whole_anchor_requires_original_final_verifier(
    tmp_path: Path, execution_evidence_native: dict[str, dict[str, object]]
) -> None:
    config = final_descriptor(tmp_path, execution_evidence_native)
    stage = mapping(config["stageEvidence"])
    reader = e.MetadataReader()
    workflow = reader.read(stage["workflow"])
    original = rows(workflow["stages"])[0]
    command = e.strings(original["command"])
    command[3] = "replay.fcp_part_assembly"
    altered = followup_history_put(
        tmp_path, "wrong-stage.json", {"stages": [{**original, "command": command}]}
    )
    with pytest.raises(ValueError, match="typed final-delivery"):
        e.checkpoint_evidence(stage | {"workflow": altered}, e.MetadataReader())


def identity_row(path: str = "/final.mp4", sha: str = "a" * 64) -> dict[str, object]:
    return {
        "path": path,
        "sha256": sha,
        "bytes": 100,
        "device": 1,
        "inode": 2,
        "mtimeNs": 3,
        "ctimeNs": 4,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "/different.mp4"),
        ("sha256", "b" * 64),
        ("bytes", 101),
        ("device", 2),
        ("inode", 3),
        ("mtimeNs", 4),
        ("ctimeNs", 5),
    ],
)
def test_full_original_identity_not_digest_only(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="input closure changed"):
        e.require_unchanged_fingerprints([identity_row()], [identity_row() | {field: value}])


@pytest.mark.parametrize(
    "row",
    [
        identity_row() | {"bytes": 0},
        identity_row() | {"inode": True},
        identity_row() | {"extra": 1},
        identity_row("relative"),
        identity_row("/a/../final.mp4"),
        identity_row("/a//final.mp4"),
        identity_row() | {"ctimeNs": -1},
    ],
)
def test_original_identity_rejects_unsupported_rows(row: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        e.fingerprint_rows([row])


def test_original_fingerprint_list_unique_and_bounded() -> None:
    with pytest.raises(ValueError):
        e.fingerprint_rows([identity_row(), identity_row()])
    with pytest.raises(ValueError):
        e.fingerprint_rows({"/final.mp4": "a" * 64})
    with pytest.raises(ValueError):
        e.fingerprint_rows([identity_row()] * 1025)


@pytest.mark.parametrize("content", [b"not JSON", b'{"x":1,"x":2}', b"[]", b"\x00\x01media"])
def test_pinned_log_requires_json_object(tmp_path: Path, content: bytes) -> None:
    p = tmp_path / "native.log"
    p.write_bytes(content)
    with pytest.raises(ValueError):
        e.MetadataReader().read(
            {"path": str(p), "sha256": hashlib.sha256(content).hexdigest()}, native_submit=True
        )


def test_domain_and_aggregate_budgets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = followup_history_put(tmp_path, "native.json", {"ok": True})
    reader = e.MetadataReader()
    reader.begin_domain("cleanup")
    reader.read(ref)
    reader.read(ref)
    assert reader.domain_metadata["cleanup"]["records"] == 1
    monkeypatch.setattr(e, "MAX_DOMAIN_BYTES", 1)
    reader.begin_domain("asr")
    with pytest.raises(ValueError, match="domain byte"):
        reader.read(ref)
    with pytest.raises(ValueError, match="repeated"):
        reader.begin_domain("asr")
    monkeypatch.setattr(e, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="total byte"):
        e.MetadataReader().read(ref)
    monkeypatch.setattr(e, "MAX_RECORD_BYTES", 1)
    with pytest.raises(ValueError, match="size"):
        e.MetadataReader().read(ref)


def test_cached_domain_conflicting_pin_rejected(tmp_path: Path) -> None:
    ref = followup_history_put(tmp_path, "record.json", {"value": 1})
    reader = e.MetadataReader()
    reader.begin_domain("cleanup")
    reader.read(ref)
    reader.begin_domain("asr")
    with pytest.raises(ValueError, match="Conflicting"):
        reader.read(ref | {"sha256": "b" * 64})
    with pytest.raises(ValueError, match="canonical"):
        e.fingerprint_rows([identity_row("//final.mp4")])


def test_caption_refresh_limit_is_scoped_and_cache_cannot_launder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = followup_history_put(tmp_path, "refresh.json", {"padding": "x" * 100})
    monkeypatch.setattr(e, "MAX_RECORD_BYTES", 64)
    monkeypatch.setattr(e, "MAX_CAPTION_LAYOUT_BYTES", 256)
    with pytest.raises(ValueError, match="size"):
        e.MetadataReader().read(ref)
    reader = e.MetadataReader()
    reader.begin_domain("caption-layout")
    assert reader.read(ref, caption_layout=True)["padding"] == "x" * 100
    assert reader.read(ref, caption_layout=True)["padding"] == "x" * 100
    with pytest.raises(ValueError, match="caller context"):
        reader.read(ref)
    with pytest.raises(ValueError, match="Conflicting evidence SHA"):
        reader.read(ref | {"sha256": "b" * 64}, caption_layout=True)
    monkeypatch.setattr(e, "MAX_CAPTION_LAYOUT_BYTES", 100)
    with pytest.raises(ValueError, match="size"):
        e.MetadataReader().read(ref, caption_layout=True)
    with pytest.raises(ValueError, match="caller context"):
        reader.read(ref, caption_layout=True)


@pytest.mark.parametrize("budget", ["MAX_DOMAIN_BYTES", "MAX_TOTAL_BYTES"])
def test_caption_refresh_retains_domain_and_total_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: str
) -> None:
    ref = followup_history_put(tmp_path, "refresh.json", {"padding": "x" * 100})
    monkeypatch.setattr(e, budget, 64)
    reader = e.MetadataReader()
    reader.begin_domain("caption-layout")
    with pytest.raises(ValueError, match="byte budget"):
        reader.read(ref, caption_layout=True)


def test_caption_refresh_does_not_enable_logs_or_submit_parser(tmp_path: Path) -> None:
    ref = followup_history_put(tmp_path, "refresh.log", {"padding": "x"})
    with pytest.raises(ValueError, match="typed native"):
        e.MetadataReader().read(ref, caption_layout=True)
    with pytest.raises(ValueError, match="cannot use"):
        e.MetadataReader().read(ref, caption_layout=True, native_submit=True)


@pytest.mark.parametrize(
    "prefix,suffix", [(False, False), (True, False), (False, True), (True, True)]
)
def test_original_submit_diagnostic_envelope(
    tmp_path: Path,
    execution_evidence_native: dict[str, dict[str, object]],
    prefix: bool,
    suffix: bool,
) -> None:
    warning = "Settings sync failed; local Executor remains available. Run executor-sync status."
    data = (
        (warning + "\n" if prefix else "")
        + json.dumps(execution_evidence_native["submit"])
        + ("\n" + warning if suffix else "")
        + "\n"
    ).encode()
    path = tmp_path / "original.log"
    path.write_bytes(data)
    ref = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    reader = e.MetadataReader()
    assert reader.read(ref, native_submit=True) == execution_evidence_native["submit"]
    assert len(reader.submit_diagnostics[str(path)]) == int(prefix) + int(suffix)
    with pytest.raises(ValueError, match="typed"):
        reader.read(ref)


@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ("altered diagnostic\n", ""),
        ("", " unexpected"),
        ("", "{}"),
        (
            "Settings sync failed; local Executor remains available. Run executor-sync status.\n"
            * 2,
            "",
        ),
        (
            "",
            "\nSettings sync failed; local Executor remains available. Run executor-sync status.\n"
            * 2,
        ),
    ],
)
def test_submit_extra_text_refused(
    execution_evidence_native: dict[str, dict[str, object]], prefix: str, suffix: str
) -> None:
    with pytest.raises(ValueError):
        e.submit_log_envelope(
            (prefix + json.dumps(execution_evidence_native["submit"]) + suffix).encode()
        )


def test_submit_false_envelope_not_success() -> None:
    with pytest.raises(ValueError):
        e.submit_log_envelope(b'{"ok":false}')


@pytest.mark.parametrize(
    "mutation", ["none", "missing", "duplicate", "producer", "code", "extra", "contract"]
)
def test_original_runtime_catalog_exact_membership(
    local_detection_data: tuple[dict[str, object], dict[str, dict[str, object]]],
    mutation: str,
) -> None:
    config, _records = local_detection_data
    producer = {"path": "/producer", "sha256": "a" * 64}
    runtime = {
        "/frozen/subtitle_candidate.py": "b" * 64,
        "/frozen/subtitle_prepared.py": "b" * 64,
        "/frozen/subtitle_detector.py": "b" * 64,
    }
    entry: dict[str, object] = {"producer": producer, "runtimePins": runtime}
    config["runtimePinsByProducer"] = [entry]
    chunks: list[dict[str, object]] = [{"producer": dict(producer)}]
    if mutation == "none":
        result = e.cleanup_runtime_catalog(config, chunks)
        assert result["/producer"]["/frozen/subtitle_candidate.py"] == "b" * 64
        assert "/archive/subtitle_candidate.py" not in result["/producer"]
        return
    if mutation == "missing":
        config["runtimePinsByProducer"] = []
    elif mutation == "duplicate":
        config["runtimePinsByProducer"] = [entry, entry]
    elif mutation == "producer":
        producer["sha256"] = "c" * 64
    elif mutation == "code":
        runtime["/frozen/subtitle_detector.py"] = "c" * 64
    elif mutation == "extra":
        entry["unknown"] = True
    else:
        config["contract"] = "modern"
    with pytest.raises(ValueError):
        e.cleanup_runtime_catalog(config, chunks)


@pytest.mark.parametrize("pcm", [False, True])
@pytest.mark.parametrize("mutation", ["none", "voice", "tool", "argv", "output", "verifier"])
def test_original_voice_observation_commands(pcm: bool, mutation: str) -> None:
    command = (
        [
            "/ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            "/voice",
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s32le",
            "-f",
            "streamhash",
            "-hash",
            "sha256",
            "/out",
        ]
        if pcm
        else ["/ffprobe", "-v", "error", "-show_streams", "-of", "json", "-o", "/out", "/voice"]
    )
    stage: dict[str, object] = {
        "name": "expected-whole-voice-pcm-hash" if pcm else "expected-whole-voice-metadata",
        "command": command,
    }
    inputs = {"/voice": "a" * 64, command[0]: "b" * 64}
    outputs = {"/out": "c" * 64}
    if mutation == "voice":
        inputs["/voice"] = "d" * 64
    elif mutation == "tool":
        inputs.pop(command[0])
    elif mutation == "argv":
        command[1] = "invalid"
    elif mutation == "output":
        outputs = {"/different": "c" * 64}
    elif mutation == "verifier":
        stage["verify_command"] = ["/producer"]
    if mutation != "none":
        with pytest.raises(ValueError):
            e.whole_observation_command(
                stage,
                master={"path": "/master", "sha256": "a" * 64},
                voice={"path": "/voice", "sha256": "a" * 64},
                outputs=outputs,
                inputs=inputs,
                fps=30,
            )
        return
    e.whole_observation_command(
        stage,
        master={"path": "/master", "sha256": "a" * 64},
        voice={"path": "/voice", "sha256": "a" * 64},
        outputs=outputs,
        inputs=inputs,
        fps=30,
    )
    stage["verify_command"] = []
    e.whole_observation_command(
        stage,
        master={"path": "/master", "sha256": "a" * 64},
        voice={"path": "/voice", "sha256": "a" * 64},
        outputs=outputs,
        inputs=inputs,
        fps=30,
    )


@pytest.fixture
def original_integrated_fixture(
    original_native_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = original_native_fixture

    def ref(path: str) -> dict[str, str]:
        return {"path": path, "sha256": "a" * 64}

    pins = {
        "/config": "a" * 64,
        "/assembly": "a" * 64,
        "/semantic": "a" * 64,
        "/partition": "a" * 64,
        "/staging": "a" * 64,
    }
    refs = [{"path": path, "sha256": sha} for path, sha in pins.items()]
    output_paths = [
        "/original",
        "/completion",
        "/publication-config",
        "/movie",
        "/published",
        "/publication",
    ]
    records["/operation"] = {
        "version": 1,
        "kind": "bounded-native-part-followup",
        "semantic": "/semantic",
        "config": "/config",
        "inputs": refs,
    }
    records["/terminal"] = {
        "version": 1,
        "kind": "verified-integrated-native-followup",
        "operation": ref("/operation"),
        "part": 1,
        "inputs": {**pins, "/operation": "a" * 64},
        "outputs": [ref(path) for path in output_paths],
    }
    records["/config"] = {"part": 1, "assembly": ref("/assembly")}
    command = [
        "/python",
        "-B",
        "-m",
        "replay.fcp_part_followup",
        "run-integrated",
        "/operation",
        "/semantic",
    ]
    verify = [*command[:4], "verify-integrated", *command[5:]]
    inventory = {"/operation": "a" * 64, "/semantic": "a" * 64}
    stage: dict[str, object] = {
        "name": "followup",
        "command": command,
        "verify_command": verify,
        "inputs": list(inventory),
        "outputs": [*output_paths, "/terminal"],
        "timeout_seconds": 30,
    }
    definition = {
        "command": command,
        "verifyCommand": verify,
        "inputs": inventory,
        "outputs": [*output_paths, "/terminal"],
        "timeoutSeconds": 30,
        "workingDirectory": "/repo",
    }
    records["/graph"] = {"stages": [stage]}
    records["/inventory"] = {"inputs": inventory}
    records["/checkpoint"] = {
        "status": "complete",
        "fingerprint": hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
        "outputs": dict.fromkeys([*output_paths, "/terminal"], "a" * 64),
    }
    item = rows(config["parts"])[0]
    item.pop("probeStage")
    item["originalFollowup"] = {
        "workflow": ref("/graph"),
        "inventory": ref("/inventory"),
        "checkpoint": ref("/checkpoint"),
        "operation": ref("/operation"),
        "terminal": ref("/terminal"),
        "stage": "followup",
        "cwd": "/repo",
    }
    config["parts"] = [item]
    return config, records


@pytest.mark.parametrize(
    "mutation",
    ["none", "part", "kind", "inputs", "outputs", "operation", "assembly", "checkpoint", "command"],
)
def test_original_integrated_part_retained_contract(
    original_integrated_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    mutation: str,
) -> None:
    config, records = original_integrated_fixture
    if mutation == "none":
        result = e.original_native_parts(config, e.MetadataReader())
        assert result["stageApplied"] is True
        assert result["executionVerified"] is False
        return
    if mutation == "part":
        records["/terminal"]["part"] = 2
    elif mutation == "kind":
        records["/operation"]["kind"] = "other"
    elif mutation == "inputs":
        records["/terminal"]["inputs"] = []
    elif mutation == "outputs":
        records["/terminal"]["outputs"] = []
    elif mutation == "operation":
        records["/terminal"]["operation"] = {"path": "/wrong", "sha256": "a" * 64}
    elif mutation == "assembly":
        records["/config"]["assembly"] = {"path": "/wrong", "sha256": "a" * 64}
    elif mutation == "checkpoint":
        records["/checkpoint"]["status"] = "running"
    else:
        stages = rows(records["/graph"]["stages"])
        stages[0]["command"] = ["/producer"]
        records["/graph"]["stages"] = stages
    with pytest.raises(ValueError):
        e.original_native_parts(config, e.MetadataReader())


@pytest.fixture
def whole_observation_fixture(
    original_whole_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = original_whole_fixture

    def ref(path: str) -> dict[str, str]:
        return {"path": path, "sha256": "a" * 64}

    stages = rows(records["/graph"]["stages"])[1:]
    join = e.strings(stages[0]["command"])
    stages[0]["command"] = ["/usr/bin/env", "PATH=/:/usr/bin:/bin:/usr/sbin:/sbin", *join]
    frame = e.strings(stages[1]["command"])
    frame[4] = "0"
    stages[1]["command"] = frame
    voice_path = join[-1]
    records["/result"] = {
        "kind": "technically-verified-whole-candidate",
        "movie": ref("/master"),
        "semanticXML": ref("/xml"),
        "workflow": ref("/graph"),
        "frames": 2,
        "rate": 30,
        "wholeFullAVDecodeVerified": True,
        "wholeVoicePCMEqual": True,
    }
    config.update(voice=ref(voice_path), semanticXML=ref("/xml"), frames=2)
    metadata = {
        "name": "expected-whole-voice-metadata",
        "command": [
            "/ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            "-o",
            "/expected-metadata",
            voice_path,
        ],
        "inputs": ["/ffprobe", voice_path],
        "outputs": ["/expected-metadata"],
        "timeout_seconds": 30,
    }
    pcm = {
        "name": "expected-whole-voice-pcm-hash",
        "command": [
            "/ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            voice_path,
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s32le",
            "-f",
            "streamhash",
            "-hash",
            "sha256",
            "/expected-pcm",
        ],
        "inputs": ["/ffmpeg", voice_path],
        "outputs": ["/expected-pcm"],
        "timeout_seconds": 30,
    }
    all_stages: list[dict[str, object]] = [metadata, pcm, *stages]

    def retained_phase(stage: dict[str, object]) -> dict[str, object]:
        name = e.string(stage["name"])
        pins = dict.fromkeys(e.strings(stage["inputs"]), "a" * 64)
        if name == "whole-lossless-join":
            pins.update({"/ffmpeg": "a" * 64, "/ffprobe": "a" * 64})
            stage["inputs"] = list(pins)
        outputs = dict.fromkeys(e.strings(stage["outputs"]), "a" * 64)
        records["/inputs-" + name] = {"inputs": pins}
        definition = {
            "command": stage["command"],
            "verifyCommand": stage.get("verify_command", []),
            "inputs": pins,
            "outputs": stage["outputs"],
            "timeoutSeconds": 30,
            "workingDirectory": "/repo",
        }
        records["/check-" + name] = {
            "status": "complete",
            "outputs": outputs,
            "fingerprint": hashlib.sha256(
                json.dumps(definition, sort_keys=True).encode()
            ).hexdigest(),
        }
        return {
            "stage": name,
            "inventory": ref("/inputs-" + name),
            "checkpoint": ref("/check-" + name),
            "outputs": outputs,
            "cwd": "/repo",
        }

    config["phases"] = [retained_phase(stage) for stage in all_stages]
    records["/graph"]["stages"] = all_stages
    return config, records


@pytest.mark.parametrize(
    "mutation",
    ["none", "clock", "summary", "partial", "predecessor", "incomplete", "voice", "env", "thread"],
)
def test_original_whole_six_exact_checkpoints(
    whole_observation_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    mutation: str,
) -> None:
    config, records = whole_observation_fixture
    if mutation == "none":
        result = e.original_whole_observations(config, e.MetadataReader())
        assert result["completedPhaseCount"] == 6
        assert result["currentOutputsCryptoVerified"] is False
        return
    if mutation == "clock":
        config["frames"] = 3
    elif mutation == "summary":
        records["/final-inventory"]["inputs"] = {"/wrong": "a" * 64}
    elif mutation == "partial":
        config["phases"] = rows(config["phases"])[1:]
    elif mutation == "predecessor":
        records["/inputs-whole-metadata"]["inputs"] = {"/master": "b" * 64}
    elif mutation == "incomplete":
        records["/check-whole-lossless-join"]["status"] = "running"
    elif mutation == "voice":
        config["voice"] = {"path": "/wrong", "sha256": "a" * 64}
    else:
        stages = rows(records["/graph"]["stages"])
        stage = stages[2 if mutation == "env" else 3]
        command = e.strings(stage["command"])
        command[1 if mutation == "env" else 4] = "invalid"
        stage["command"] = command
        records["/graph"]["stages"] = stages
    with pytest.raises(ValueError):
        e.original_whole_observations(config, e.MetadataReader())


@pytest.mark.parametrize("mutation", ["none", "verifier", "semantic", "assembly", "part"])
def test_original_followup_frame_checkpoint(
    original_native_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    mutation: str,
) -> None:
    config, records = original_native_fixture
    command = ["/python", "-B", "-m", "replay.fcp_part_followup", "frames", "/config", "/semantic"]
    verify = [*command[:4], "verify-frames", *command[5:]]
    pins = {
        "/config": "a" * 64,
        "/assembly": "a" * 64,
        "/semantic": "a" * 64,
        "/partition": "a" * 64,
        "/staging": "a" * 64,
    }
    stage: dict[str, object] = {
        "name": "probe",
        "command": command,
        "verify_command": verify,
        "inputs": list(pins),
        "outputs": ["/original"],
        "timeout_seconds": 30,
    }
    records["/config"] = {"part": 1, "assembly": {"path": "/assembly", "sha256": "a" * 64}}
    records["/graph"] = {"stages": [stage]}
    records["/inventory"] = {"inputs": pins}
    definition = {
        "command": command,
        "verifyCommand": verify,
        "inputs": pins,
        "outputs": ["/original"],
        "timeoutSeconds": 30,
        "workingDirectory": "/repo",
    }
    records["/checkpoint"] = {
        "status": "complete",
        "outputs": {"/original": "a" * 64},
        "fingerprint": hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
    }
    if mutation == "none":
        assert e.original_native_parts(config, e.MetadataReader())["stageApplied"] is True
        return
    if mutation == "verifier":
        verify[4] = "frames"
    elif mutation == "semantic":
        pins["/semantic"] = "b" * 64
    elif mutation == "assembly":
        records["/config"]["assembly"] = {"path": "/other", "sha256": "a" * 64}
    else:
        records["/config"]["part"] = 2
    with pytest.raises(ValueError):
        e.original_native_parts(config, e.MetadataReader())


@pytest.mark.parametrize(
    "content", [b"{}{}", b'{"a":1,"a":2}', b"prefix {}", b"[]", b'{"success":false}']
)
def test_native_status_log_rejects_non_envelope(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "status.log"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        e.MetadataReader().read(
            {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}, native_status=True
        )


def test_native_status_log_cannot_launder_parser_context(
    tmp_path: Path,
    execution_evidence_native: dict[str, object],
) -> None:
    content = json.dumps(execution_evidence_native["status"]).encode()
    path = tmp_path / "status.log"
    path.write_bytes(content)
    ref = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    reader = e.MetadataReader()
    assert reader.read(ref, native_status=True) == execution_evidence_native["status"]
    with pytest.raises(ValueError, match="parser context"):
        reader.read(ref, native_submit=True)
    with pytest.raises(ValueError, match="typed native"):
        reader.read(ref)


def test_original_integrated_rejects_unbound_publication(
    original_integrated_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, records = original_integrated_fixture
    outputs = rows(records["/terminal"]["outputs"])
    outputs[-1]["path"] = "/other-publication"
    records["/terminal"]["outputs"] = outputs
    checkpoint = records["/checkpoint"]
    mapping_outputs = e.hashes(checkpoint["outputs"])
    mapping_outputs["/other-publication"] = mapping_outputs.pop("/publication")
    checkpoint["outputs"] = mapping_outputs
    stages = rows(records["/graph"]["stages"])
    stages[0]["outputs"] = list(mapping_outputs)
    records["/graph"]["stages"] = stages
    definition = {
        "command": stages[0]["command"],
        "verifyCommand": stages[0]["verify_command"],
        "inputs": records["/inventory"]["inputs"],
        "outputs": stages[0]["outputs"],
        "timeoutSeconds": 30,
        "workingDirectory": "/repo",
    }
    checkpoint["fingerprint"] = hashlib.sha256(
        json.dumps(definition, sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="publication output"):
        e.original_native_parts(config, e.MetadataReader())


@pytest.fixture
def original_batch_fixture(
    original_integrated_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config, records = original_integrated_fixture
    ref = asr_evidence_ref
    operation = records["/operation"]
    common = copy.deepcopy(rows(operation["inputs"]))
    common.append(ref("/ffprobe"))
    operation["inputs"] = copy.deepcopy(common)
    part: dict[str, object] = {
        "part": 1,
        "operation": ref("/operation"),
        "native": ref("/staging"),
        "retirement": None,
        "adoption": [],
    }
    records["/plan"] = {
        "version": 1,
        "kind": "bounded-native-followup-batch",
        "common": common,
        "parts": [part],
        "checkpoints": "/steps",
        "terminal": "/terminal",
    }
    steps = ("frames", "complete", "publication-config", "publish")
    outputs = (
        ("/original",),
        ("/completion",),
        ("/publication-config",),
        ("/movie", "/published", "/publication"),
    )
    step_refs = []
    for step, output_paths in zip(steps, outputs, strict=True):
        path = f"/steps/part-0001-{step}.json"
        step_refs.append(ref(path))
        records[path] = {
            "version": 1,
            "kind": "pending-common-context-native-step",
            "common": common,
            "part": 1,
            "operation": ref("/operation"),
            "nativeSHA256": "a" * 64,
            "step": step,
            "qualityApproved": False,
            "outputs": [ref(output) for output in output_paths],
        }
    pins = {str(row["path"]): str(row["sha256"]) for row in common}
    records["/terminal"] = {
        "version": 1,
        "kind": "verified-common-context-native-batch",
        "inputs": {**pins, "/operation": "a" * 64},
        "checkpoints": step_refs,
        "executed": 3,
        "adopted": 1,
    }
    command = [
        "/python",
        "-B",
        "-m",
        "replay.fcp_part_followup_batch",
        "run",
        "/plan",
        "--semantic",
        "/semantic",
    ]
    verify = [*command[:4], "verify", *command[5:]]
    inventory = {"/plan": "a" * 64, "/semantic": "a" * 64}
    stage: dict[str, object] = {
        "name": "followup-batch",
        "command": command,
        "verify_command": verify,
        "inputs": list(inventory),
        "outputs": ["/terminal"],
        "timeout_seconds": 30,
    }
    records["/graph"] = {"stages": [stage]}
    records["/inventory"] = {"inputs": inventory}
    definition = {
        "command": command,
        "verifyCommand": verify,
        "inputs": inventory,
        "outputs": ["/terminal"],
        "timeoutSeconds": 30,
        "workingDirectory": "/repo",
    }
    records["/checkpoint"] = {
        "status": "complete",
        "outputs": {"/terminal": "a" * 64},
        "fingerprint": hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
    }
    item = rows(config["parts"])[0]
    item.pop("originalFollowup")
    item["originalBatch"] = {
        "plan": ref("/plan"),
        "terminal": ref("/terminal"),
        "operation": ref("/operation"),
        "checkpoints": step_refs,
        "stageEvidence": {
            "workflow": ref("/graph"),
            "checkpoint": ref("/checkpoint"),
            "inventory": ref("/inventory"),
            "stage": "followup-batch",
            "cwd": "/repo",
        },
    }
    config["parts"] = [item]
    provenance = {"version": 1, "executable": ref("/ffprobe"), "decoderThreads": 0}
    records["/original"]["probeProvenance"] = provenance
    records["/published"]["probeProvenance"] = copy.deepcopy(provenance)
    return config, records


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "terminal",
        "count",
        "step",
        "operation",
        "model",
        "duplicate",
        "missing",
        "publication",
    ],
)
def test_original_batch_retained_steps(
    original_batch_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    mutation: str,
) -> None:
    config, records = original_batch_fixture
    if mutation == "none":
        assert e.original_native_parts(config, e.MetadataReader())["stageApplied"] is True
        return
    if mutation == "terminal":
        records["/checkpoint"]["status"] = "running"
    elif mutation == "count":
        records["/terminal"]["executed"] = 0
    elif mutation == "step":
        records["/steps/part-0001-frames.json"]["nativeSHA256"] = "f" * 64
    elif mutation == "operation":
        records["/operation"]["semantic"] = "/other"
    elif mutation == "model":
        records["/original"]["probeProvenance"] = {}
    elif mutation == "duplicate":
        records["/plan"]["parts"] = [*rows(records["/plan"]["parts"])] * 2
    elif mutation == "missing":
        records["/terminal"]["checkpoints"] = []
    else:
        records["/steps/part-0001-publish.json"]["outputs"] = [asr_evidence_ref("/wrong")]
    with pytest.raises(ValueError):
        e.original_native_parts(config, e.MetadataReader())


@pytest.fixture
def split_whole_fixture(
    original_native_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    _, records = original_native_fixture
    ref = asr_evidence_ref

    def phase(
        name: str, command: list[str], inputs: dict[str, str], outputs: dict[str, str]
    ) -> dict[str, object]:
        stage: dict[str, object] = {
            "name": name,
            "command": command,
            "inputs": list(inputs),
            "outputs": list(outputs),
            "timeout_seconds": 30,
        }
        definition = {
            "command": command,
            "verifyCommand": [],
            "inputs": inputs,
            "outputs": list(outputs),
            "timeoutSeconds": 30,
            "workingDirectory": "/repo",
        }
        records[f"/{name}-graph"] = {"stages": [stage]}
        records[f"/{name}-inventory"] = {"inputs": inputs}
        records[f"/{name}-checkpoint"] = {
            "status": "complete",
            "fingerprint": hashlib.sha256(
                json.dumps(definition, sort_keys=True).encode()
            ).hexdigest(),
            "outputs": outputs,
        }
        return {
            "workflow": ref(f"/{name}-graph"),
            "inventory": ref(f"/{name}-inventory"),
            "checkpoint": ref(f"/{name}-checkpoint"),
            "outputs": outputs,
            "stage": name,
            "cwd": "/repo",
        }

    join = phase(
        "join",
        [
            "/python",
            "-B",
            "-m",
            "replay.delivery_render",
            "join-files",
            "/join-plan",
            "--output",
            "/master",
            "--rate",
            "30",
        ],
        dict.fromkeys(["/join-plan", "/partition", "/movie", "/publication"], "a" * 64),
        {"/master": "a" * 64},
    )
    packet_inputs = dict.fromkeys(["/master", "/packets", "/join-checkpoint"], "a" * 64)
    packet = phase(
        "packet",
        ["/bin/bash", "-c", "set -euo pipefail\nrecord-original"],
        packet_inputs,
        {"/packet-receipt": "a" * 64},
    )
    records["/join-plan"] = {"rows": [{"sourcePath": "/movie", "durationFrames": 2}]}
    records["/packet-receipt"] = {
        "version": 2,
        "kind": "whole-video-only-packet-clock-acceptance",
        "master": ref("/master"),
        "producerCheckpoint": ref("/join-checkpoint"),
        "observations": ref("/packets"),
        "inputs": packet_inputs,
        "audioStreams": 0,
        "packetClockAccepted": True,
        "rejoined": False,
        "newDecode": False,
        "fullDecode": False,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
        "video": {
            "codec": "prores",
            "exactPacketCFR": True,
            "frames": 2,
            "startPTS": 0,
            "endPTS": 200,
            "timeBase": "1/3000",
        },
    }
    records["/audio-evidence"] = {
        "source": ref("/voice"),
        "movie": ref("/aac"),
        "encodingPasses": 1,
        **{
            key: ref("/" + key) for key in ("nativeStatus", "fullAV", "decodedProbe", "packetProbe")
        },
    }
    config: dict[str, object] = {
        "contract": "original-split-video-whole-v1",
        "source": ref("/master"),
        "voice": ref("/voice"),
        "semanticXML": ref("/semantic"),
        "frames": 2,
        "fps": 30,
        "joinPlan": ref("/join-plan"),
        "partition": ref("/partition"),
        "publications": [ref("/publication")],
        "join": join,
        "packetStage": packet,
        "packetReceipt": ref("/packet-receipt"),
        "audioEvidence": ref("/audio-evidence"),
    }
    monkeypatch.setattr(
        e.MetadataReader, "read", lambda self, value, **kwargs: records[str(mapping(value)["path"])]
    )
    return config, records


@pytest.mark.parametrize(
    "record,key,value",
    [
        ("/packet-receipt", "audioStreams", 1),
        ("/packet-receipt", "fullDecode", True),
        ("/packet-receipt", "packetClockAccepted", False),
        ("/packet-receipt", "inputs", {"/changed": "b" * 64}),
        ("/packet-receipt", "version", 1),
        ("/audio-evidence", "encodingPasses", 2),
        ("/audio-evidence", "source", {"path": "/wrong", "sha256": "a" * 64}),
        ("/join-plan", "rows", []),
        ("/join-plan", "rows", [{"sourcePath": "/movie", "durationFrames": 3}]),
        ("/join-checkpoint", "status", "running"),
        ("/packet-checkpoint", "outputs", {}),
    ],
)
def test_split_whole_rejects_false_or_changed_records(
    split_whole_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
    record: str,
    key: str,
    value: object,
) -> None:
    config, records = split_whole_fixture
    records[record][key] = value
    with pytest.raises(ValueError):
        e.domain_evidence("whole-assembly", config, e.MetadataReader())


def test_split_whole_preserves_video_only_limits(
    split_whole_fixture: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    config, _ = split_whole_fixture
    result = e.domain_evidence("whole-assembly", config, e.MetadataReader())
    assert result["stageApplied"] is True
    assert result["executionVerified"] is False
    details = mapping(result["details"])
    assert details["wholeMasterFullDecodeVerified"] is False
    assert details["wholeVoicePCMEqual"] is False


def test_join_plan_parser_does_not_relax_object_reader(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    data = b'[{"sourcePath":"/movie","durationFrames":2}]'
    path.write_bytes(data)
    ref = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    reader = e.MetadataReader()
    assert reader.read(ref, join_plan=True) == {
        "rows": [{"sourcePath": "/movie", "durationFrames": 2}]
    }
    with pytest.raises(ValueError, match="context"):
        reader.read(ref)
    with pytest.raises(ValueError):
        e.MetadataReader().read(ref)
    with pytest.raises(ValueError):
        reader.read(ref, join_plan=True, native_submit=True)


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Settings sync failed; local Executor remains available. Run executor-sync status.",
        "arbitrary diagnostic",
    ],
)
def test_original_status_log_uses_exact_diagnostic_parser(tmp_path: Path, diagnostic: str) -> None:
    path = tmp_path / "status.log"
    envelope = execution_evidence_envelope([])
    data = (diagnostic + "\n" + json.dumps(envelope)).encode()
    path.write_bytes(data)
    ref = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    if diagnostic == "arbitrary diagnostic":
        with pytest.raises(ValueError):
            e.MetadataReader().read(ref, native_status=True)
    else:
        assert e.MetadataReader().read(ref, native_status=True) == envelope


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize(
    "mutation", ["none", "source", "failed100", "false-envelope", "job", "output", "count", "pin"]
)
def test_original_split_native_request_preserves_output_disposition(
    tmp_path: Path, audio: bool, mutation: str
) -> None:
    role = "audio" if audio else "video"
    requested = "/delivery/job/file.mp4"
    output = e.reference(asr_evidence_ref("/delivery/job/file.m4a" if audio else requested))
    config = normalized_compressor_fixture(tmp_path, role, requested)
    request: dict[str, object] = {
        "batchName": "whole",
        "bundleID": "com.apple.Compressor",
        "sourcePath": "/whole.mov",
        "presetPath": "/preset",
        "outputDirectory": "/delivery",
        "outputName": "file.mp4",
    }
    if mutation == "source":
        request["sourcePath"] = "/other.mov"
    if mutation == "output":
        request["outputName"] = "different.mp4"
    config.update(
        contract="original-split-native-request-v1",
        source=asr_evidence_ref("/whole.mov"),
        preset=asr_evidence_ref("/preset"),
        request=followup_history_put(tmp_path, "raw-request.json", request),
    )
    inputs = {output["path"]: output["sha256"]}
    if audio:
        proof = followup_history_put(
            tmp_path,
            "audio-evidence.json",
            {
                "source": asr_evidence_ref("/whole.mov"),
                "movie": output,
                "encodingPasses": 2 if mutation == "count" else 1,
                "nativeStatus": config["status"],
            },
        )
        config["outputEvidence"] = proof
        inputs[str(proof["path"])] = str(proof["sha256"])
    else:
        config["requestReceipt"] = followup_history_put(
            tmp_path,
            "request-receipt.json",
            {
                "version": 1,
                "kind": "whole-compressor-request",
                "wholeMaster": "/whole.mov",
                "inputs": {
                    "/whole.mov": "a" * 64,
                    "/preset": "b" * 64 if mutation == "count" else "a" * 64,
                },
            },
        )
    if mutation in {"failed100", "job", "false-envelope"}:
        status = execution_evidence_envelope(
            [
                {
                    "jobStatus": {
                        "jobid": "other" if mutation == "job" else "job",
                        "batchid": "batch",
                        "status": "Failed" if mutation == "failed100" else "Successful",
                        "percentComplete": "100",
                    }
                }
            ]
        )
        if mutation == "false-envelope":
            status["ok"] = False
        config["status"] = followup_history_put(tmp_path, "changed-status.json", status)
    if mutation == "pin":
        inputs[output["path"]] = "f" * 64
    if mutation != "none":
        with pytest.raises(ValueError):
            e.split_native_request_evidence(
                config, e.MetadataReader(), output=output, input_hashes=inputs
            )
        return
    result = e.split_native_request_evidence(
        config, e.MetadataReader(), output=output, input_hashes=inputs
    )
    assert result["dispatchOutputMatchesRetained"] is not audio
    assert result["nativeProviderBeforeAfterVerified"] is False
    assert result["officialSuccessful"] is True
    assert result["independentlyRecordedOutput"] == output


@pytest.mark.parametrize(
    "mutation", ["none", "job", "truncated", "partial", "schema", "contract", "source"]
)
def test_original_recorded_native_payload_contract(tmp_path: Path, mutation: str) -> None:
    output = e.reference(asr_evidence_ref("/delivery/job/file.mp4"))
    config = normalized_compressor_fixture(tmp_path, "video", output["path"])
    reader = e.MetadataReader()
    submitted = e.official_payload(reader.read(config["submit"]))
    observed = e.official_payload(reader.read(config["status"]))
    submit: dict[str, object] = {
        **submitted,
        "version": 1,
        "submittedOnce": True,
        "observedUTC": "recorded",
        "jobID": "job",
        "batchID": "batch",
        "retrySafe": False,
    }
    status: dict[str, object] = {
        **observed,
        "observedUTC": "recorded",
        "jobID": "job",
        "batchID": "batch",
        "retrySafe": False,
        "explicitSuccessful": True,
        "encodingWallNativeSeconds": 1,
        "outputPath": output["path"],
        "outputBytes": 10,
        "newAACEncodingPasses": 0,
        "fullAVVerified": False,
        "allPTSVerified": False,
    }
    if mutation == "job":
        status["jobID"] = "other"
    elif mutation == "truncated":
        submit["truncated"] = True
    elif mutation == "partial":
        status["explicitSuccessful"] = False
    elif mutation == "schema":
        status["invented"] = True
    config.update(
        contract="original-split-native-request-v1",
        payloadContract="unknown" if mutation == "contract" else "recorded-native-submit-status-v1",
        source=asr_evidence_ref("/whole.mov"),
        preset=asr_evidence_ref("/preset"),
        submit=followup_history_put(tmp_path, "recorded-submit.json", submit),
        status=followup_history_put(tmp_path, "recorded-status.json", status),
        request=followup_history_put(
            tmp_path,
            "raw.json",
            {
                "batchName": "whole",
                "bundleID": "com.apple.Compressor",
                "sourcePath": "/whole.mov",
                "presetPath": "/preset",
                "outputDirectory": "/delivery",
                "outputName": "file.mp4",
            },
        ),
        requestReceipt=followup_history_put(
            tmp_path,
            "receipt.json",
            {
                "version": 1,
                "kind": "whole-compressor-request",
                "wholeMaster": "/whole.mov",
                "inputs": {
                    "/whole.mov": "b" * 64 if mutation == "source" else "a" * 64,
                    "/preset": "a" * 64,
                },
            },
        ),
    )
    if mutation != "none":
        with pytest.raises(ValueError):
            e.split_native_request_evidence(
                config,
                e.MetadataReader(),
                output=output,
                input_hashes={output["path"]: output["sha256"]},
            )
        return
    assert (
        e.split_native_request_evidence(
            config,
            e.MetadataReader(),
            output=output,
            input_hashes={output["path"]: output["sha256"]},
        )["officialSuccessful"]
        is True
    )
