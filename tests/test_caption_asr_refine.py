"""Synthetic, offline repair candidates; no native tools or private recordings."""

import json
import wave
from pathlib import Path

import pytest

from replay.caption_asr_refine import main, refine
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_MODEL, DEFAULT_WEIGHTS_SHA256
from replay.stages import file_digest


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def save(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return pin(path)


@pytest.fixture
def config(tmp_path: Path) -> dict[str, object]:
    audio = tmp_path / "window.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 96000)
    tokens = save(
        tmp_path / "original.json",
        {
            "version": 1,
            "clock": "output",
            "tokens": [
                {"text": "前", "start": 0, "end": 0.5, "utterance": "old:0"},
                {"text": "旧", "start": 1, "end": 1.5, "utterance": "old:1"},
                {"text": "語", "start": 2, "end": 3, "utterance": "old:2"},
                {"text": "後", "start": 5, "end": 6, "utterance": "old:3"},
            ],
        },
    )
    return {
        "version": 1,
        "audio": pin(audio),
        "window": save(
            tmp_path / "window.json",
            {
                "version": 1,
                "source": str(tmp_path / "source.wav"),
                "sourceSHA256": "a" * 64,
                "startSample": 160000,
                "endSample": 256000,
                "sampleRate": 16000,
                "windowSHA256": file_digest(audio),
                "clock": "continuous-input-pcm",
                "sourceClockProjected": False,
            },
        ),
        "inputs": save(
            tmp_path / "inputs.json",
            {
                "version": 1,
                "source": str(audio),
                "source_sha256": file_digest(audio),
                "model": str(tmp_path / "model"),
                "config_sha256": DEFAULT_CONFIG_SHA256,
                "weights_sha256": DEFAULT_WEIGHTS_SHA256,
                "sample_rate": 16000,
                "samples": 96000,
                "clock": "continuous-input-pcm",
                "human_reviewed": False,
            },
        ),
        "recognition": save(
            tmp_path / "raw.json",
            {
                "language": "ja",
                "text": "新除去次",
                "segments": [
                    {
                        "start": 0,
                        "end": 6,
                        "text": "新除去次",
                        "words": [
                            {"word": "新", "start": 1, "end": 1.5, "probability": 0.9},
                            {"word": "除去", "start": 3, "end": 3.5, "probability": 0.9},
                            {"word": "次", "start": 4, "end": 4.5, "probability": 0.9},
                        ],
                    }
                ],
            },
        ),
        "clockEvidence": save(
            tmp_path / "clock.json",
            {
                "clock": "native-source-presentation-already-projected",
                "noSecondClockProjection": True,
            },
        ),
        "sourceSHA256": "a" * 64,
        "cuts": save(
            tmp_path / "cuts.json",
            {
                "cuts": [
                    {"source_start_frame": 0, "source_end_frame": 360, "output_start_frame": 0},
                    {"source_start_frame": 420, "source_end_frame": 600, "output_start_frame": 360},
                ]
            },
        ),
        "manifest": save(
            tmp_path / "manifest.json",
            {
                "version": 1,
                "tokens": tokens,
                "spans": [
                    {
                        "sourceStart": {"value": 300, "timescale": 30},
                        "outputStart": {"value": 0, "timescale": 30},
                        "duration": {"value": 60, "timescale": 30},
                    },
                    {
                        "sourceStart": {"value": 420, "timescale": 30},
                        "outputStart": {"value": 60, "timescale": 30},
                        "duration": {"value": 180, "timescale": 30},
                    },
                ],
                "foreground": [],
                "native": {},
                "calibration": {},
                "outputFrameRate": 30,
                "region": [0, 0, 1080, 1920],
                "centerX": 540,
                "clearance": 24,
            },
        ),
        "measurementDirectory": str(tmp_path / "measurement"),
        "segmentOffsetFrame": 300,
        "replacement": {"startSample": 176000, "endSample": 240000},
    }


def run(config: dict[str, object], tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    save(path, config)
    return refine(path, tmp_path / "candidate", config_sha256=file_digest(path))


@pytest.fixture
def batch_config(config: dict[str, object], tmp_path: Path) -> dict[str, object]:
    config["version"] = 2
    model = DEFAULT_MODEL.resolve()
    config["inputs"] = save(
        tmp_path / "batch.json",
        {
            "version": 1,
            "inputs": {
                str(tmp_path / "window.wav"): file_digest(tmp_path / "window.wav"),
                str(model / "config.json"): DEFAULT_CONFIG_SHA256,
                str(model / "weights.npz"): DEFAULT_WEIGHTS_SHA256,
            },
            "samples": 96000,
            "sampleRate": 16000,
            "windowSamples": 960000,
            "strideSamples": 480000,
            "clock": "input-pcm-not-projected",
            "device": "cpu",
            "dtypeContract": "provider-default-fp16-unchanged",
        },
    )
    raw_hash = file_digest(tmp_path / "raw.json")
    config["recognitionClock"] = save(
        tmp_path / "batch-clock.json",
        {
            "startSample": 0,
            "endSample": 96000,
            "sampleRate": 16000,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": raw_hash,
        },
    )
    config["recognitionComplete"] = save(
        tmp_path / "batch-complete.json",
        {"rawSHA256": [raw_hash], "windows": 1, "deliveryComplete": False},
    )
    config["clockEvidence"] = save(
        tmp_path / "native-plan.json",
        {
            "clock": "native-source-presentation-already-projected",
            "source": str(tmp_path / "source.wav"),
            "sourceSHA256": "a" * 64,
            "samples": 320000,
            "sampleRate": 16000,
            "humanReviewed": False,
        },
    )
    return config


def test_batch_candidate_retains_native_clock_and_provenance(
    batch_config: dict[str, object], tmp_path: Path
) -> None:
    target = run(batch_config, tmp_path)
    data = json.loads(target.read_text())
    assert [(t["text"], t["start"], t["end"]) for t in data["tokens"]] == [
        ("前", 0, 0.5),
        ("新", 1, 1.5),
        ("次", 2, 2.5),
        ("後", 5, 6),
    ]
    receipt = json.loads((target.parent / "receipt.json").read_text())
    assert receipt["recognitionInputKind"] == "batch-asr-single-window"
    assert receipt["sourceClockProjection"] == "identity-native-presentation-no-second-projection"
    assert receipt["candidateOnly"] is True
    assert str(tmp_path / "batch-complete.json") in receipt["inputs"]


@pytest.mark.parametrize(
    "artifact,field,value",
    [
        ("inputs", "version", 2),
        ("inputs", "samples", 96001),
        ("inputs", "sampleRate", 8000),
        ("inputs", "windowSamples", 320000),
        ("inputs", "strideSamples", 160000),
        ("inputs", "clock", "source"),
        ("inputs", "device", "gpu"),
        ("inputs", "dtypeContract", "fp32"),
        ("inputs", "inputs", {}),
        ("recognitionClock", "startSample", 1),
        ("recognitionClock", "endSample", 96001),
        ("recognitionClock", "sampleRate", 8000),
        ("recognitionClock", "humanReviewed", True),
        ("recognitionClock", "emptyRecognitionIsNotSilence", False),
        ("recognitionClock", "rawSHA256", "a" * 64),
        ("recognitionComplete", "rawSHA256", []),
        ("recognitionComplete", "windows", 2),
        ("recognitionComplete", "deliveryComplete", True),
        ("clockEvidence", "sourceSHA256", "b" * 64),
        ("clockEvidence", "source", "/different.wav"),
        ("clockEvidence", "samples", 256000 - 1),
        ("clockEvidence", "sampleRate", 8000),
        ("clockEvidence", "humanReviewed", True),
    ],
)
def test_batch_rejects_unbound_or_ambiguous_evidence(
    batch_config: dict[str, object], tmp_path: Path, artifact: str, field: str, value: object
) -> None:
    reference = batch_config[artifact]
    assert isinstance(reference, dict)
    path = Path(reference["path"])
    data = json.loads(path.read_text())
    data[field] = value
    batch_config[artifact] = save(path, data)
    with pytest.raises(ValueError):
        run(batch_config, tmp_path)
    assert not (tmp_path / "candidate").exists()


def test_mapping_across_cuts_offset_and_preserved_tokens(
    config: dict[str, object], tmp_path: Path
) -> None:
    original = (tmp_path / "original.json").read_bytes()
    result = run(config, tmp_path)
    data = json.loads(result.read_text(encoding="utf-8"))
    assert [(t["text"], t["start"], t["end"]) for t in data["tokens"]] == [
        ("前", 0, 0.5),
        ("新", 1.0, 1.5),
        ("次", 2.0, 2.5),
        ("後", 5, 6),
    ]
    assert data["tokens"][0] == {"text": "前", "start": 0, "end": 0.5, "utterance": "old:0"}
    assert data["tokens"][-1] == {"text": "後", "start": 5, "end": 6, "utterance": "old:3"}
    assert (tmp_path / "original.json").read_bytes() == original
    receipt = json.loads((result.parent / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["removedTokenIndices"] == [1, 2]
    assert [(t["text"], t["start"], t["end"]) for t in receipt["omittedByCuts"]] == [
        ("除去", 13.0, 13.5)
    ]
    assert receipt["humanReviewed"] is False
    assert receipt["deliveryComplete"] is False
    assert not (result.parent / "manifest.json").exists()


@pytest.mark.parametrize(
    "field", ["window", "inputs", "recognition", "cuts", "manifest", "audio", "clockEvidence"]
)
def test_mismatched_pin(config: dict[str, object], tmp_path: Path, field: str) -> None:
    reference = config[field]
    assert isinstance(reference, dict)
    reference["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Pinned candidate input changed"):
        run(config, tmp_path)
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("sourceSHA256", "b" * 64),
        ("segmentOffsetFrame", 301),
        ("version", True),
        ("replacement", {"startSample": 176001, "endSample": 240000}),
        ("replacement", {"startSample": 0, "endSample": 240000}),
        ("replacement", {"startSample": 176000, "endSample": 176000}),
        ("replacement", {"startSample": 192000, "endSample": 224000}),
        ("unexpected", 1),
    ],
)
def test_invalid_contract(
    config: dict[str, object], tmp_path: Path, field: str, value: object
) -> None:
    config[field] = value
    with pytest.raises(ValueError):
        run(config, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "b" * 64),
        ("source", "/nonexistent.wav"),
        ("samples", 96001),
        ("weights_sha256", "b" * 64),
        ("sample_rate", True),
        ("clock", "output"),
        ("human_reviewed", True),
    ],
)
def test_bad_recognition_inputs(
    config: dict[str, object], tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "inputs.json"
    inputs = json.loads(path.read_text(encoding="utf-8"))
    inputs[field] = value
    config["inputs"] = save(path, inputs)
    with pytest.raises(ValueError):
        run(config, tmp_path)


@pytest.mark.parametrize(
    "words",
    [
        [{"word": "語", "start": 1, "end": 1, "probability": 0.9}],
        [
            {"word": "語", "start": 1, "end": 2, "probability": 0.9},
            {"word": "語", "start": 1.5, "end": 2.5, "probability": 0.9},
        ],
        [{"word": "語", "start": 1.5, "end": 2.5, "probability": 0.9}],
        [{"word": "語", "start": 0.5, "end": 1.5, "probability": 0.9}],
        [],
    ],
)
def test_zero_overlap_cut_and_selection_boundaries(
    config: dict[str, object], tmp_path: Path, words: list[dict[str, object]]
) -> None:
    text = "語" * len(words)
    config["recognition"] = save(
        tmp_path / "raw.json",
        {
            "language": "ja",
            "text": text,
            "segments": [{"start": 0, "end": 6, "text": text, "words": words}],
        },
    )
    with pytest.raises(ValueError):
        run(config, tmp_path)


def test_same_onset_zero_piece_uses_existing_mlx_union(
    config: dict[str, object], tmp_path: Path
) -> None:
    config["recognition"] = save(
        tmp_path / "raw.json",
        {
            "language": "ja",
            "text": "新語",
            "segments": [
                {
                    "start": 0,
                    "end": 6,
                    "text": "新語",
                    "words": [
                        {"word": "新", "start": 1, "end": 1, "probability": 0.9},
                        {"word": "語", "start": 1, "end": 1.5, "probability": 0.9},
                    ],
                }
            ],
        },
    )
    data = json.loads(run(config, tmp_path).read_text(encoding="utf-8"))
    assert [(t["text"], t["start"], t["end"]) for t in data["tokens"]] == [
        ("前", 0, 0.5),
        ("新語", 1.0, 1.5),
        ("後", 5, 6),
    ]


def test_existing_output_is_preserved(config: dict[str, object], tmp_path: Path) -> None:
    target = run(config, tmp_path)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        run(config, tmp_path)
    assert target.read_bytes() == before


def test_completed_job_is_refused(config: dict[str, object], tmp_path: Path) -> None:
    measurement = tmp_path / "measurement"
    measurement.mkdir()
    save(measurement / "candidate.json", {"untouched": True})
    with pytest.raises(ValueError, match="Completed measurement"):
        run(config, tmp_path)
    assert (measurement / "candidate.json").read_text(encoding="utf-8") == '{"untouched": true}'


def test_ambiguous_clock_evidence(config: dict[str, object], tmp_path: Path) -> None:
    config["clockEvidence"] = save(
        tmp_path / "clock.json", {"clock": "continuous-input-pcm", "noSecondClockProjection": False}
    )
    with pytest.raises(ValueError, match="identity source clock mismatch"):
        run(config, tmp_path)


def test_config_hash_and_duplicate_keys(config: dict[str, object], tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save(path, config)
    with pytest.raises(ValueError, match="Pinned candidate input changed"):
        refine(path, tmp_path / "candidate", config_sha256="0" * 64)
    path.write_text('{"version": 1, "version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        main([str(path), str(tmp_path / "candidate")])


def test_cli(config: dict[str, object], tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save(path, config)
    assert main([str(path), str(tmp_path / "candidate")]) == 0


@pytest.mark.parametrize("value", [-1, 0, 120000])
def test_invalid_offset_or_job_extent(
    config: dict[str, object], tmp_path: Path, value: int
) -> None:
    config["segmentOffsetFrame"] = value
    with pytest.raises(ValueError):
        run(config, tmp_path)


@pytest.mark.parametrize("field,value", [("version", 2), ("clock", "source")])
def test_old_token_schema(
    config: dict[str, object], tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "original.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokens"] = save(path, data)
    config["manifest"] = save(manifest_path, manifest)
    with pytest.raises(ValueError, match="segment-local output tokens"):
        run(config, tmp_path)


@pytest.mark.parametrize(
    "spans",
    [
        [],
        [
            {
                "sourceStart": {"value": 300, "timescale": 30},
                "outputStart": {"value": 0, "timescale": 30},
                "duration": {"value": 90, "timescale": 30},
            },
        ],
    ],
)
def test_missing_or_unsupported_spans(
    config: dict[str, object], tmp_path: Path, spans: list[object]
) -> None:
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["spans"] = spans
    config["manifest"] = save(path, manifest)
    with pytest.raises(ValueError, match="[Ss]pan"):
        run(config, tmp_path)


def test_old_token_boundary_not_silently_removed(config: dict[str, object], tmp_path: Path) -> None:
    path = tmp_path / "original.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tokens"][1]["start"] = 0.9
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokens"] = save(path, data)
    config["manifest"] = save(manifest_path, manifest)
    with pytest.raises(ValueError, match="straddles replacement boundary"):
        run(config, tmp_path)


def test_output_outside_work_is_refused(config: dict[str, object], tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save(path, config)
    with pytest.raises(ValueError, match="under edits/work"):
        refine(path, Path.cwd() / "not-a-work-output", config_sha256=file_digest(path))


def test_linked_output_rejected(config: dict[str, object], tmp_path: Path) -> None:
    (tmp_path / "candidate").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="Linked tracking output"):
        run(config, tmp_path)
