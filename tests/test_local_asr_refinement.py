"""Synthetic checks for local ASR refinement provenance and word support."""

import json
import wave
from pathlib import Path

import pytest

from replay import local_asr_refinement as adapter
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import StageError, _fingerprint, file_digest
from replay.workflow import load_stages


def save(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": file_digest(path)}


def pcm(path: Path, count: int) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x00" * count)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    model_config = save(model / "config.json", {"synthetic": True})
    weights = model / "weights.npz"
    weights.write_bytes(b"synthetic weights")
    monkeypatch.setattr(adapter, "DEFAULT_MODEL", model)
    monkeypatch.setattr(adapter, "DEFAULT_CONFIG_SHA256", model_config["sha256"])
    monkeypatch.setattr(adapter, "DEFAULT_WEIGHTS_SHA256", file_digest(weights))
    whole, audio = tmp_path / "whole.wav", tmp_path / "window.wav"
    pcm(whole, 640000)
    pcm(audio, 160000)
    original_inputs = save(
        tmp_path / "original-inputs.json",
        {"samples": 640000, "inputs": {str(whole): file_digest(whole)}},
    )
    labels = save(
        tmp_path / "labels.json",
        {
            "version": 1,
            "clock": "input-pcm-not-projected",
            "parentSHA256": {original_inputs["path"]: original_inputs["sha256"]},
            "labels": [
                {"text": "前", "start": 29, "end": 30, "provenance": "old:0"},
                {"text": "粗", "start": 31, "end": 33, "provenance": "old:1"},
                {"text": "後", "start": 34, "end": 35, "provenance": "old:2"},
            ],
            "humanReviewed": False,
            "speechAccuracyVerified": False,
            "deliveryComplete": False,
        },
    )
    evidence = tmp_path / "asr"
    evidence.mkdir()
    local = save(
        evidence / "inputs.json",
        {
            "version": 1,
            "source": str(audio),
            "model": str(model),
            "source_sha256": file_digest(audio),
            "config_sha256": model_config["sha256"],
            "weights_sha256": file_digest(weights),
            "sample_rate": 16000,
            "samples": 160000,
            "clock": "continuous-input-pcm",
            "human_reviewed": False,
        },
    )
    raw = save(
        evidence / "raw-transcript.json",
        {
            "language": "ja",
            "text": "新語未",
            "segments": [
                {
                    "start": 1,
                    "end": 3,
                    "text": "新語未",
                    "words": [
                        {"word": "新", "start": 1, "end": 2, "probability": 0.9},
                        {"word": "語", "start": 2, "end": 3, "probability": 0.9},
                        {"word": "未", "start": 2.5, "end": 2.5, "probability": 0.9},
                    ],
                }
            ],
        },
    )
    validation = save(
        evidence / "validation.json",
        {
            "word_groups": 3,
            "unresolved_points": 1,
            "human_reviewed": False,
            "source_clock_projected": False,
        },
    )
    window = save(
        tmp_path / "window.json",
        {
            "version": 1,
            "source": str(whole),
            "sourceSHA256": file_digest(whole),
            "startSample": 480000,
            "endSample": 640000,
            "sampleRate": 16000,
            "windowSHA256": file_digest(audio),
            "clock": "continuous-input-pcm",
            "sourceClockProjected": False,
        },
    )
    producer = Path(adapter.__file__).with_name("local_asr.py").resolve()
    parser = producer.with_name("caption_io.py")
    source_pins = {str(producer): file_digest(producer), str(parser): file_digest(parser)}
    before = save(tmp_path / "before.json", source_pins)
    after = save(tmp_path / "after.json", {"hashes": source_pins, "differences": {}})
    plan = save(
        tmp_path / "plan.json",
        {
            "version": 1,
            "stages": [
                {
                    "name": "recognize",
                    "inputs": [
                        str(audio),
                        str(model / "config.json"),
                        str(weights),
                        str(producer),
                        str(parser),
                    ],
                    "outputs": [local["path"], raw["path"], validation["path"]],
                    "command": [
                        "python",
                        "-m",
                        "replay.local_asr",
                        str(audio),
                        str(evidence),
                        "--device",
                        "auto",
                    ],
                    "timeout_seconds": 30,
                }
            ],
        },
    )
    state = tmp_path / "state"
    state.mkdir()
    checkpoint = save(
        state / "recognize.json",
        {
            "status": "complete",
            "fingerprint": _fingerprint(load_stages(Path(plan["path"]))[0]),
            "outputs": {
                local["path"]: local["sha256"],
                raw["path"]: raw["sha256"],
                validation["path"]: validation["sha256"],
            },
        },
    )
    status = tmp_path / "job.status"
    status.write_text("0\n", encoding="ascii")
    end = save(
        tmp_path / "end.json",
        {
            "exitCode": 0,
            "completed": True,
            "reservationReleased": True,
        },
    )
    observed = {
        str(audio): file_digest(audio),
        local["path"]: local["sha256"],
        raw["path"]: raw["sha256"],
        validation["path"]: validation["sha256"],
        **source_pins,
    }
    pins = tmp_path / "pins.sha256"
    pins.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in observed.items()), encoding="utf-8"
    )
    path = tmp_path / "refine.json"
    save(
        path,
        {
            "version": 1,
            "stage": "recognize",
            "targetProvenance": "old:1",
            "artifacts": {
                "labels": labels,
                "originalInputs": original_inputs,
                "window": window,
                "audio": {"path": str(audio), "sha256": file_digest(audio)},
                "inputs": local,
                "raw": raw,
                "validation": validation,
                "plan": plan,
                "checkpoint": checkpoint,
                "status": {"path": str(status), "sha256": file_digest(status)},
                "end": end,
                "sourceBefore": before,
                "sourceAfter": after,
                "pins": {"path": str(pins), "sha256": file_digest(pins)},
            },
        },
    )
    return path


def rewrite_artifact(config: Path, name: str, value: object) -> None:
    document = record(json.loads(config.read_text(encoding="utf-8")))
    artifacts = record(document["artifacts"])
    binding = record(artifacts[name])
    artifacts[name] = save(Path(str(binding["path"])), value)
    save(config, document)


def test_candidate_preserves_neighbors_and_offset(config: Path, tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    assert adapter.run(config, output) == 5
    assert adapter.verify(config, output) == 5
    candidate = record(json.loads(output.read_text(encoding="utf-8")))
    assert candidate["labels"] == [
        {"text": "前", "start": 29.0, "end": 30.0, "provenance": "old:0"},
        {"text": "新", "start": 31.0, "end": 32.0, "provenance": "000000480000:0"},
        {"text": "語", "start": 32.0, "end": 33.0, "provenance": "000000480000:1"},
        {"text": "未", "start": 32.5, "end": 32.5, "provenance": "000000480000:2"},
        {"text": "後", "start": 34.0, "end": 35.0, "provenance": "old:2"},
    ]
    assert candidate["humanReviewed"] is False
    assert candidate["speechAccuracyVerified"] is False
    assert candidate["deliveryComplete"] is False
    assert record(candidate["refinement"])["clockOffsetApplications"] == 1
    assert candidate["unresolvedPoints"] == 1
    assert candidate["adjacentOverlaps"] == 1
    assert output.stat().st_mode & 0o777 == 0o600


def test_refuses_overwrite(config: Path, tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ReplayError, match="new candidate"):
        adapter.run(config, output)
    assert output.read_text(encoding="utf-8") == "existing"


def test_refuses_candidate_tamper(config: Path, tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    adapter.run(config, output)
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="differs from its pinned plan"):
        adapter.verify(config, output)


def test_source_tamper_is_rejected(config: Path, tmp_path: Path) -> None:
    (tmp_path / "whole.wav").write_bytes(b"changed")
    with pytest.raises(StageError, match="SHA256"):
        adapter.plan(config)


@pytest.mark.parametrize("value", [False, True, 2])
def test_config_version_refused(config: Path, value: object) -> None:
    data = record(json.loads(config.read_text(encoding="utf-8")))
    data["version"] = value
    save(config, data)
    with pytest.raises(ReplayError, match="Unsupported"):
        adapter.plan(config)


def test_missing_selection_refused(config: Path) -> None:
    data = record(json.loads(config.read_text(encoding="utf-8")))
    data["stage"] = ""
    save(config, data)
    with pytest.raises(ReplayError, match="Missing local ASR stage"):
        adapter.plan(config)


def test_extra_fake_cpu_manifest_refused(config: Path) -> None:
    data = record(json.loads(config.read_text(encoding="utf-8")))
    record(data["artifacts"])["fakeCPUComplete"] = {"path": "fake", "sha256": "0" * 64}
    save(config, data)
    with pytest.raises(ReplayError, match="Missing genuine"):
        adapter.plan(config)


def test_claimed_review_refused(config: Path, tmp_path: Path) -> None:
    document = record(json.loads((tmp_path / "labels.json").read_text(encoding="utf-8")))
    document["humanReviewed"] = True
    rewrite_artifact(config, "labels", document)
    with pytest.raises(ReplayError, match="unreviewed"):
        adapter.plan(config)


def test_failed_job_refused(config: Path) -> None:
    rewrite_artifact(
        config, "end", {"exitCode": 1, "completed": False, "reservationReleased": True}
    )
    with pytest.raises(ReplayError, match="terminal envelope"):
        adapter.plan(config)


def test_failed_status_refused(config: Path, tmp_path: Path) -> None:
    data = record(json.loads(config.read_text(encoding="utf-8")))
    (tmp_path / "job.status").write_text("1\n", encoding="ascii")
    record(data["artifacts"])["status"] = {
        "path": str(tmp_path / "job.status"),
        "sha256": file_digest(tmp_path / "job.status"),
    }
    save(config, data)
    with pytest.raises(ReplayError, match="did not finish"):
        adapter.plan(config)


def test_historical_source_change_refused(config: Path) -> None:
    rewrite_artifact(config, "sourceAfter", {"hashes": {}, "differences": {"changed": True}})
    with pytest.raises(ReplayError, match="changed during recognition"):
        adapter.plan(config)


def test_running_checkpoint_refused(config: Path, tmp_path: Path) -> None:
    document = record(json.loads((tmp_path / "state/recognize.json").read_text(encoding="utf-8")))
    document["status"] = "running"
    rewrite_artifact(config, "checkpoint", document)
    with pytest.raises(ReplayError, match="not complete"):
        adapter.plan(config)


def test_output_hash_not_matching_actual_receipt_refused(config: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "state/recognize.json"
    document = record(json.loads(checkpoint.read_text(encoding="utf-8")))
    record(document["outputs"])[str(tmp_path / "asr/raw-transcript.json")] = "0" * 64
    rewrite_artifact(config, "checkpoint", document)
    with pytest.raises(ReplayError, match="complete stage receipt"):
        adapter.plan(config)


def test_old_source_cannot_be_repinned_as_current(config: Path, tmp_path: Path) -> None:
    document = record(json.loads((tmp_path / "before.json").read_text(encoding="utf-8")))
    document[str(Path(adapter.__file__).with_name("local_asr.py").resolve())] = "0" * 64
    rewrite_artifact(config, "sourceBefore", document)
    rewrite_artifact(config, "sourceAfter", {"hashes": document, "differences": {}})
    with pytest.raises(ReplayError, match="differs from current producer"):
        adapter.plan(config)


def test_wrong_model_refused(config: Path, tmp_path: Path) -> None:
    document = record(json.loads((tmp_path / "asr/inputs.json").read_text(encoding="utf-8")))
    document["weights_sha256"] = "0" * 64
    rewrite_artifact(config, "inputs", document)
    with pytest.raises(ReplayError, match="non-turbo"):
        adapter.plan(config)


def test_extraction_outside_source_refused(config: Path, tmp_path: Path) -> None:
    document = record(json.loads((tmp_path / "window.json").read_text(encoding="utf-8")))
    document["endSample"] = 800000
    rewrite_artifact(config, "window", document)
    with pytest.raises(ValueError, match="extent"):
        adapter.plan(config)


def test_target_boundary_outside_extraction_refused(config: Path, tmp_path: Path) -> None:
    document = record(json.loads((tmp_path / "labels.json").read_text(encoding="utf-8")))
    labels = document["labels"]
    assert isinstance(labels, list)
    record(labels[1])["start"] = 29.99
    rewrite_artifact(config, "labels", document)
    with pytest.raises(ValueError, match="exceeds bounded verified extraction"):
        adapter.plan(config)


def test_cli_candidate_and_verification(
    config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "candidate.json"
    assert adapter.main([str(config), str(output)]) == 0
    assert adapter.main([str(config), str(output), "--verify"]) == 0
    assert capsys.readouterr().out.count('"candidateOnly": true') == 2
