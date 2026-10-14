"""Synthetic regressions for immutable native audio revision evidence."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from replay import native_audio_revision_evidence as module
from replay import native_completion as legacy
from replay import native_delivery_evidence as evidence
from replay.bgm_separation import SeparationPolicy
from replay.core import ReplayError


def ref(name: str) -> dict[str, str]:
    return {"path": f"/fixture/{name}.json", "sha256": "a" * 64}


class Reader(evidence.MetadataReader):
    def __init__(self, records: dict[str, dict[str, object]]) -> None:
        super().__init__()
        self.records = records

    def read(
        self,
        value: object,
        *,
        native_submit: bool = False,
        native_status: bool = False,
        caption_layout: bool = False,
        join_plan: bool = False,
    ) -> dict[str, object]:
        return self.records[evidence.reference(value)["path"]]


def fixture() -> tuple[dict[str, str], Reader, dict[str, dict[str, object]]]:
    records: dict[str, dict[str, object]] = {}

    def put(name: str, value: dict[str, object]) -> dict[str, str]:
        records[ref(name)["path"]] = value
        return ref(name)

    speech: dict[str, object] = {
        "correlation": 1.0,
        "level_delta_db_p05": 0.0,
        "level_delta_db_p50": 0.0,
    }
    quality: dict[str, object] = {
        "policy": asdict(SeparationPolicy()),
        "minimumSpeechCorrelation": legacy.MINIMUM_CORRELATION,
        "speechGatePassed": True,
        "musicGatePassed": True,
        "speech": speech,
        "protectedSamples": 44100,
        "windows": [
            {
                "startSample": 0,
                "endSample": 44100,
                "protectedSamples": 44100,
                "speech": speech,
                "speechGatePassed": True,
                "beforeInstrumentalDBFS": -70.0,
                "afterInstrumentalDBFS": -70.0,
                "music": {"passed": True},
            }
        ],
    }
    descriptor = put(
        "descriptor",
        {"context": {"frames": 30, "fps": 30, "voice": ref("voice"), "finalMovie": ref("old")}},
    )
    composition = put(
        "composition",
        {
            "kind": "video-composition-timing-observation",
            "candidate": ref("new"),
            "aacAssessment": {
                "candidate": ref("new"),
                "source": ref("voice"),
                "originalPCMExtentAccepted": True,
                "packetDecodedPresentationAccepted": True,
                "trackEditMappingAccepted": True,
                "primingAppliedAgain": False,
                "decodedStart": "0",
                "decodedEnd": "1",
                "decodedSamples": 44100,
            },
        },
    )
    movie = {**ref("new"), "bytes": 100}
    full = put(
        "fullAV",
        {
            "technicalAccepted": True,
            "fullDecode": True,
            "inputs": [movie],
            "inputsAfter": [movie],
            "audioEndMinusVideoEnd": "0",
            "audioSpanMinusVideoDuration": "0",
            "expected": {"frames": 30, "rate": 30},
        },
    )
    qrequest = put("qualityRequest", {"composition": composition})
    qref = put(
        "quality",
        {
            "kind": "native-tempo-full-audio-quality-observation",
            "currentModelDevice": "mps",
            "protection": "all-selected-samples-conservative",
            "protectedSamples": 44100,
            "request": qrequest,
            "inputs": {ref("new")["path"]: movie},
            "measurements": quality,
        },
    )
    checkpoint = put(
        "qualityCheckpoint", {"status": "complete", "outputs": {qref["path"]: qref["sha256"]}}
    )
    declared: dict[str, object] = {
        "sourceObservation": ref("hash0"),
        "candidateObservation": ref("hash1"),
    }
    packet_request = put("packetRequest", declared)
    packets = put(
        "packets",
        {
            "packets": 30,
            "orderedPayloadsEqual": True,
            "rationalClocksEqual": True,
            "extradataAndGeometryEqual": True,
            "request": packet_request,
            "evidence": declared,
        },
    )
    stages: list[object] = []
    checkpoints: list[object] = []
    for index, name in enumerate(("old", "new")):
        observation = ref(f"hash{index}")
        stages.append(
            {
                "command": [
                    "ffmpeg",
                    "-i",
                    ref(name)["path"],
                    "-map",
                    "0:v:0",
                    "-c:v",
                    "copy",
                    "-threads",
                    "1",
                    "-hash",
                    "sha256",
                    "-f",
                    "framehash",
                    observation["path"],
                ],
                "inputs": [ref(name)["path"]],
                "outputs": [observation["path"]],
            }
        )
        checkpoints.append(
            put(
                f"packetCheckpoint{index}",
                {"status": "complete", "outputs": {observation["path"]: observation["sha256"]}},
            )
        )
    workflow = put("packetWorkflow", {"stages": stages})
    comparison = put(
        "comparisonCheckpoint",
        {
            "status": "complete",
            "outputs": {packets["path"]: packets["sha256"]},
        },
    )
    request = put(
        "request",
        {
            "version": 1,
            "kind": module.KIND,
            "descriptor": descriptor,
            "composition": composition,
            "fullAV": full,
            "quality": qref,
            "qualityCheckpoint": checkpoint,
            "packets": packets,
            "packetWorkflow": workflow,
            "packetCheckpoints": checkpoints,
            "comparisonCheckpoint": comparison,
        },
    )
    return request, Reader(records), records


def test_complete_numerical_is_not_delivery() -> None:
    request, reader, _ = fixture()
    result = module.assess(request, reader)
    assert result["numericalAudioMeasurementVerified"] is True
    assert result["strictComplete"] is False
    assert result["currentMediaCryptoVerified"] is False
    assert evidence.mapping(result["strictFailures"])["bgm_removal"] == [
        "separation model/source lineage"
    ]


@pytest.mark.parametrize(
    ("role", "field", "value"),
    [
        ("request", "kind", "unknown"),
        ("request", "extra", True),
        ("composition", "kind", "fake"),
        ("fullAV", "fullDecode", False),
        ("fullAV", "inputsAfter", []),
        ("fullAV", "expected", {"frames": 31, "rate": 30}),
        ("fullAV", "inputs", []),
        ("fullAV", "inputs", None),
        ("quality", "currentModelDevice", "cpu"),
        ("quality", "protectedSamples", 44000),
        ("quality", "inputs", {}),
        ("qualityCheckpoint", "status", "running"),
        ("qualityCheckpoint", "outputs", {}),
        ("qualityRequest", "composition", ref("other")),
        ("packets", "orderedPayloadsEqual", False),
        ("packets", "evidence", {}),
        ("packetWorkflow", "stages", []),
        ("packetWorkflow", "stages", None),
        ("packetCheckpoint0", "status", "running"),
        ("packetCheckpoint1", "outputs", {}),
    ],
)
def test_mutated_record_rejected(role: str, field: str, value: object) -> None:
    request, reader, records = fixture()
    records[ref(role)["path"]][field] = value
    with pytest.raises((ValueError, KeyError)):
        module.assess(request, reader)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", ref("wrong")),
        ("decodedSamples", 44099),
        ("decodedStart", "1/1000"),
        ("decodedEnd", "1001/1000"),
        ("primingAppliedAgain", True),
        ("trackEditMappingAccepted", False),
    ],
)
def test_audio_clock_mutation(field: str, value: object) -> None:
    request, reader, records = fixture()
    record = records[ref("composition")["path"]]
    record["aacAssessment"] = {**evidence.mapping(record["aacAssessment"]), field: value}
    with pytest.raises(ValueError):
        module.assess(request, reader)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy", {}),
        ("speechGatePassed", False),
        ("speech", {}),
        ("windows", []),
        ("windows", None),
        ("protectedSamples", 44099),
    ],
)
def test_quality_aggregate_mutation(field: str, value: object) -> None:
    request, reader, records = fixture()
    record = records[ref("quality")["path"]]
    record["measurements"] = {**evidence.mapping(record["measurements"]), field: value}
    with pytest.raises((ValueError, ReplayError)):
        module.assess(request, reader)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("startSample", 1),
        ("endSample", 44101),
        ("protectedSamples", 44099),
        ("speechGatePassed", False),
        ("music", {"passed": False}),
        ("beforeInstrumentalDBFS", -1.0),
    ],
)
def test_quality_window_mutation(field: str, value: object) -> None:
    request, reader, records = fixture()
    quality = evidence.mapping(records[ref("quality")["path"]]["measurements"])
    rows = quality["windows"]
    assert isinstance(rows, list)
    row = evidence.mapping(rows[0])
    row[field] = value
    if field == "beforeInstrumentalDBFS":
        row["afterInstrumentalDBFS"] = -1.0
    rows[0] = row
    with pytest.raises(ValueError):
        module.assess(request, reader)


def test_cli_exclusive_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, reader, _ = fixture()
    directory = tmp_path / "edits/work"
    directory.mkdir(parents=True)
    output = directory / "proof.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evidence, "MetadataReader", lambda: reader)
    monkeypatch.setattr(
        sys, "argv", ["proof", request["path"], str(output), "--sha256", request["sha256"]]
    )
    module.main()
    assert json.loads(output.read_text())["strictComplete"] is False
    with pytest.raises(FileExistsError):
        module.main()
    reader.records[str(output)] = evidence.mapping(json.loads(output.read_text()))
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--verify"])
    module.main()
    reader.records[str(output)]["strictComplete"] = True
    with pytest.raises(ValueError, match="Revision proof changed"):
        module.main()


@pytest.mark.parametrize("replacement", [None, ["ffmpeg", "-i", "wrong"], ["ffmpeg"]])
def test_packet_producer_command_rejected(replacement: object) -> None:
    request, reader, records = fixture()
    workflow = records[ref("packetWorkflow")["path"]]
    stages = workflow["stages"]
    assert isinstance(stages, list)
    stages[0] = {**evidence.mapping(stages[0]), "command": replacement}
    with pytest.raises(ValueError):
        module.assess(request, reader)


def test_changed_pinned_request_rejected(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    path.write_text("{}\n")
    with pytest.raises(ValueError):
        module.assess({"path": str(path), "sha256": "0" * 64}, evidence.MetadataReader())


def test_successor_rational_verifier_does_not_complete_old_stage() -> None:
    request, reader, records = fixture()
    records[ref("packetCheckpoint1")["path"]] = {"status": "running"}
    stages = records[ref("packetWorkflow")["path"]]["stages"]
    assert isinstance(stages, list)
    stages[1] = {
        **evidence.mapping(stages[1]),
        "verify_command": [
            "/usr/bin/cmp",
            ref("hash0")["path"],
            ref("hash1")["path"],
        ],
    }
    assert module.assess(request, reader)["originalPacketStageCompletionPromoted"] is False
    records[ref("comparisonCheckpoint")["path"]]["status"] = "running"
    with pytest.raises(ValueError, match="Rational packet comparison"):
        module.assess(request, reader)


@pytest.mark.parametrize("fps", [0, 29])
def test_original_clock_rejected(fps: int) -> None:
    request, reader, records = fixture()
    records[ref("descriptor")["path"]]["context"] = {
        **evidence.mapping(records[ref("descriptor")["path"]]["context"]),
        "fps": fps,
    }
    with pytest.raises(ValueError):
        module.assess(request, reader)
