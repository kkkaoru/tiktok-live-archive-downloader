"""Independent PCM recognition candidates preserve source and neighbor clocks."""

import json
import wave
from pathlib import Path

import pytest

from replay import asr_window_refine
from replay.audio_windows import AudioWindow
from replay.caption_repair import ClockLabel
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import StageError, file_digest


def save(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": file_digest(path)}


def pcm(path: Path, samples: int, *, channels: int = 1, value: bytes = b"\x01\x00") -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(value * samples * channels)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    model_config = save(model / "config.json", {"synthetic": True})
    (model / "weights.npz").write_bytes(b"synthetic weights")
    weight_hash = file_digest(model / "weights.npz")
    monkeypatch.setattr(asr_window_refine, "DEFAULT_MODEL", model)
    monkeypatch.setattr(asr_window_refine, "DEFAULT_CONFIG_SHA256", model_config["sha256"])
    monkeypatch.setattr(asr_window_refine, "DEFAULT_WEIGHTS_SHA256", weight_hash)
    source, audio = tmp_path / "whole.wav", tmp_path / "window.wav"
    pcm(source, 640000)
    pcm(audio, 160000)
    common = {
        "version": 1,
        "sampleRate": 16000,
        "windowSamples": 960000,
        "strideSamples": 480000,
        "clock": "input-pcm-not-projected",
    }
    original_inputs = save(
        tmp_path / "original-inputs.json",
        {**common, "samples": 640000, "inputs": {str(source): file_digest(source)}},
    )
    inputs = save(
        tmp_path / "inputs.json",
        {
            **common,
            "samples": 160000,
            "device": "cpu",
            "dtypeContract": "provider-default-fp16-unchanged",
            "inputs": {
                str(audio): file_digest(audio),
                model_config["path"]: model_config["sha256"],
                str(model / "weights.npz"): weight_hash,
            },
        },
    )
    raw = save(
        tmp_path / "raw.json",
        {
            "language": "ja",
            "text": "新語",
            "segments": [
                {
                    "start": 1,
                    "end": 3,
                    "text": "新語",
                    "words": [
                        {"word": "新", "start": 1, "end": 2, "probability": 0.9},
                        {"word": "語", "start": 2, "end": 3, "probability": 0.9},
                    ],
                }
            ],
        },
    )
    clock = save(
        tmp_path / "clock.json",
        {
            "startSample": 0,
            "endSample": 160000,
            "sampleRate": 16000,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": raw["sha256"],
        },
    )
    complete = save(
        tmp_path / "complete.json",
        {"rawSHA256": [raw["sha256"]], "windows": 1, "deliveryComplete": False},
    )
    labels = save(
        tmp_path / "labels.json",
        {
            "version": 1,
            "clock": "input-pcm-not-projected",
            "labels": [
                {"text": "前", "start": 29, "end": 30, "provenance": "old:0"},
                {"text": "粗", "start": 31, "end": 33, "provenance": "old:1"},
                {"text": "後", "start": 34, "end": 35, "provenance": "old:2"},
            ],
            "parentSHA256": {original_inputs["path"]: original_inputs["sha256"]},
            "humanReviewed": False,
            "speechAccuracyVerified": False,
            "deliveryComplete": False,
        },
    )
    window = save(
        tmp_path / "window.json",
        {
            "version": 1,
            "source": str(source),
            "sourceSHA256": file_digest(source),
            "startSample": 480000,
            "endSample": 640000,
            "sampleRate": 16000,
            "windowSHA256": file_digest(audio),
            "clock": "continuous-input-pcm",
            "sourceClockProjected": False,
        },
    )
    path = tmp_path / "refine.json"
    save(
        path,
        {
            "version": 1,
            "artifacts": {
                "labels": labels,
                "originalInputs": original_inputs,
                "inputs": inputs,
                "raw": raw,
                "clock": clock,
                "complete": complete,
                "window": window,
                "audio": {"path": str(audio), "sha256": file_digest(audio)},
            },
            "replacement": {
                "firstLabel": 1,
                "stopLabel": 2,
                "startSample": 496000,
                "endSample": 528000,
            },
        },
    )
    return path


def mutate_artifact(config: Path, artifact: str, field: str, value: object) -> None:
    document = record(json.loads(config.read_text()))
    artifacts = record(document["artifacts"])
    binding = record(artifacts[artifact])
    path = Path(str(binding["path"]))
    item = record(json.loads(path.read_text()))
    item[field] = value
    artifacts[artifact] = save(path, item)
    save(config, document)


def test_candidate_keeps_original_neighbors_and_applies_offset_once(config: Path) -> None:
    before = config.with_name("labels.json").read_bytes()
    output = config.with_name("candidate.json")
    assert asr_window_refine.run(config, output) == 4
    result = record(json.loads(output.read_text()))
    labels = result["labels"]
    assert isinstance(labels, list)
    assert [(record(row)["start"], record(row)["end"]) for row in labels] == [
        (29, 30),
        (31, 32),
        (32, 33),
        (34, 35),
    ]
    assert "".join(str(record(row)["text"]) for row in labels) == "前新語後"
    assert config.with_name("labels.json").read_bytes() == before
    assert result["speechAccuracyVerified"] is False
    assert result["deliveryComplete"] is False
    refinement = record(result["refinement"])
    assert refinement["clockOffsetApplications"] == 1
    assert refinement["sourceProjectionApplications"] == 0
    assert refinement["candidateOnly"] is True
    assert asr_window_refine.main([str(config), str(config.with_name("cli.json"))]) == 0
    assert asr_window_refine.verify(config, output) == 4
    assert asr_window_refine.main([str(config), str(output), "--verify"]) == 0
    with pytest.raises(ValueError, match="absent output"):
        asr_window_refine.run(config, output)
    result["speechAccuracyVerified"] = True
    save(output, result)
    with pytest.raises(ValueError, match="differs from its pinned plan"):
        asr_window_refine.verify(config, output)


@pytest.mark.parametrize(
    ("artifact", "field", "value", "message"),
    [
        ("window", "startSample", True, "sample-exact"),
        ("window", "endSample", 2000000, "60 seconds"),
        ("window", "sourceClockProjected", True, "Extraction must bind"),
        ("window", "source", "whole.wav", "Extraction must bind"),
        ("window", "windowSHA256", "a" * 64, "Extraction must bind"),
        ("window", "endSample", 624000, "sample extent"),
        ("labels", "parentSHA256", {}, "Original labels"),
        ("labels", "clock", "source-presentation", "unprojected"),
        ("labels", "humanReviewed", True, "Original labels"),
        ("inputs", "device", "auto", "default-model"),
        ("inputs", "dtypeContract", "fp32", "default-model"),
        ("inputs", "samples", 176000, "default-model"),
        ("clock", "startSample", 1, "default-model"),
        ("complete", "windows", 2, "default-model"),
        ("inputs", "inputs", {}, "ASR dependencies"),
        ("originalInputs", "inputs", {}, "ASR dependencies"),
    ],
)
def test_invalid_evidence(
    config: Path, artifact: str, field: str, value: object, message: str
) -> None:
    mutate_artifact(config, artifact, field, value)
    with pytest.raises((ValueError, ReplayError), match=message):
        asr_window_refine.run(config, config.with_name("rejected.json"))


@pytest.mark.parametrize(
    "replacement",
    [
        {"firstLabel": True},
        {"firstLabel": 1, "stopLabel": 2, "startSample": 400000, "endSample": 528000},
        {"firstLabel": 1, "stopLabel": 2, "startSample": 496001, "endSample": 528000},
        {"firstLabel": 1, "stopLabel": 2, "startSample": 496000, "endSample": 520000},
    ],
)
def test_invalid_replacement(config: Path, replacement: object) -> None:
    document = record(json.loads(config.read_text()))
    document["replacement"] = replacement
    save(config, document)
    with pytest.raises(ValueError, match="integer|exceeds|splits"):
        asr_window_refine.run(config, config.with_name("rejected.json"))


def test_unrelated_or_corrupted_pcm_is_rejected(config: Path) -> None:
    audio = config.with_name("window.wav")
    pcm(audio, 160000, value=b"\x02\x00")
    document = record(json.loads(config.read_text()))
    artifacts = record(document["artifacts"])
    artifacts["audio"] = {"path": str(audio), "sha256": file_digest(audio)}
    save(config, document)
    mutate_artifact(config, "window", "windowSHA256", file_digest(audio))
    inputs = record(json.loads(config.with_name("inputs.json").read_text()))
    dependencies = record(inputs["inputs"])
    dependencies[str(audio)] = file_digest(audio)
    mutate_artifact(config, "inputs", "inputs", dependencies)
    with pytest.raises(ValueError, match="samples differ"):
        asr_window_refine.run(config, config.with_name("rejected.json"))


def test_changed_pin_and_bad_config_rejected(config: Path) -> None:
    with pytest.raises(ValueError, match="canonical directory"):
        asr_window_refine.run(config, config.parent / "missing" / "out.json")
    document = record(json.loads(config.read_text()))
    document["version"] = 2
    save(config, document)
    with pytest.raises(ValueError, match="Unsupported"):
        asr_window_refine.run(config, config.with_name("rejected.json"))
    document["version"] = 1
    artifacts = record(document["artifacts"])
    artifacts.pop("raw")
    save(config, document)
    with pytest.raises(ValueError, match="bindings"):
        asr_window_refine.run(config, config.with_name("rejected.json"))
    artifacts["raw"] = {"path": str(config.with_name("raw.json")), "sha256": "a" * 64}
    save(config, document)
    with pytest.raises(StageError, match="SHA256"):
        asr_window_refine.run(config, config.with_name("rejected.json"))


def test_exact_provenance_planner_needs_no_manual_label_indices(config: Path) -> None:
    document = record(json.loads(config.read_text()))
    document.pop("replacement")
    document["targetProvenance"] = "old:1"
    save(config, document)
    before = config.with_name("labels.json").read_bytes()
    output = config.with_name("planned-candidate.json")
    assert asr_window_refine.run(config, output) == 4
    result = record(json.loads(output.read_text()))
    refinement = record(result["refinement"])
    assert refinement["targetProvenance"] == "old:1"
    assert refinement["replacement"] == {
        "firstLabel": 1,
        "stopLabel": 2,
        "startSample": 496000,
        "endSample": 528000,
    }
    assert config.with_name("labels.json").read_bytes() == before
    assert asr_window_refine.verify(config, output) == 4
    document["targetProvenance"] = 1
    save(config, document)
    with pytest.raises(ValueError, match="target provenance"):
        asr_window_refine.run(config, config.with_name("invalid-target.json"))


def test_wholeword_planner_expands_observed_edges_and_neighbors() -> None:
    original = tuple(
        ClockLabel(text, start, end, f"000000000000:{index}")
        for index, (text, start, end) in enumerate(
            (
                ("前", 25.0, 26.0),
                ("左", 29.0, 31.0),
                ("粗", 31.0, 33.0),
                ("右", 33.0, 34.0),
                ("後", 35.0, 36.0),
            )
        )
    )
    alternative = (
        ClockLabel("新左", 29.5, 31.5, "000000320000:0"),
        ClockLabel("新右", 31.5, 33.5, "000000320000:1"),
    )
    result = asr_window_refine.derive_replacement(
        original,
        alternative,
        window=AudioWindow(320000, 320000),
        target_provenance="000000000000:2",
    )
    assert result == {"firstLabel": 1, "stopLabel": 4, "startSample": 464000, "endSample": 544000}
    assert alternative[0].start == 29.5 and alternative[-1].end == 33.5


@pytest.mark.parametrize(
    ("target", "fragments", "expected"),
    [
        (
            (31.1, 31.35),
            (
                ("フ", 31, 31.1),
                ("ォロ", 31.1, 31.2),
                ("ワ", 31.2, 31.3),
                ("ー", 31.3, 31.4),
                ("さん", 31.4, 31.5),
                ("次", 32, 32.2),
            ),
            {"firstLabel": 1, "stopLabel": 2, "startSample": 496000, "endSample": 504000},
        ),
        (
            (31.5, 31.65),
            (
                ("ワ", 31, 31.1),
                ("ール", 31.1, 31.3),
                ("ド", 31.3, 31.4),
                ("ワ", 31.4, 31.5),
                ("ール", 31.5, 31.7),
                ("ド", 31.7, 31.8),
                ("の", 31.8, 31.9),
                ("出", 32.5, 32.6),
            ),
            {"firstLabel": 1, "stopLabel": 2, "startSample": 502400, "endSample": 510400},
        ),
    ],
)
def test_planner_keeps_loanword_suffix_and_repeated_wholeword(
    target: tuple[float, float],
    fragments: tuple[tuple[str, float, float], ...],
    expected: dict[str, int],
) -> None:
    original = (
        ClockLabel("前", 29, 30, "000000000000:0"),
        ClockLabel("粗", *target, "000000000000:1"),
        ClockLabel("後", 33, 34, "000000000000:2"),
    )
    alternative = tuple(
        ClockLabel(text, start, end, f"000000480000:{index}")
        for index, (text, start, end) in enumerate(fragments)
    )
    assert (
        asr_window_refine.derive_replacement(
            original,
            alternative,
            window=AudioWindow(480000, 160000),
            target_provenance="000000000000:1",
        )
        == expected
    )


def test_planner_does_not_join_word_fragments_across_real_gap() -> None:
    original = (ClockLabel("粗", 31, 31.1, "000000000000:0"),)
    alternative = (
        ClockLabel("ワ", 31, 31.1, "000000480000:0"),
        ClockLabel("ールド", 32, 32.2, "000000480000:1"),
    )
    assert asr_window_refine.derive_replacement(
        original,
        alternative,
        window=AudioWindow(480000, 160000),
        target_provenance="000000000000:0",
    ) == {"firstLabel": 0, "stopLabel": 1, "startSample": 496000, "endSample": 497600}


def test_planner_rejects_alternative_scene_boundary() -> None:
    original = (ClockLabel("粗", 31, 31.4, "scene1:000000000000:0"),)
    alternative = (
        ClockLabel("ワ", 31, 31.1, "scene1:000000480000:0"),
        ClockLabel("ールド", 31.1, 31.4, "scene2:000000480000:1"),
    )
    with pytest.raises(ValueError, match="alternative recognition boundary"):
        asr_window_refine.derive_replacement(
            original,
            alternative,
            window=AudioWindow(480000, 160000),
            target_provenance="scene1:000000000000:0",
        )


def test_planner_preserves_zero_duration_native_record() -> None:
    original = (ClockLabel("粗", 31, 31, "000000000000:0"),)
    alternative = (ClockLabel("点", 31, 31, "000000480000:0"),)
    with pytest.raises(ValueError, match="bounded verified"):
        asr_window_refine.derive_replacement(
            original,
            alternative,
            window=AudioWindow(480000, 160000),
            target_provenance="000000000000:0",
        )


@pytest.mark.parametrize("context", ["source2:", "scene2:", "speaker2:", "cut-2:", "unknown"])
def test_planner_preserves_explicit_context_and_unknown_boundaries(context: str) -> None:
    original = (
        ClockLabel("左", 29.0, 31.0, context + "000000000000:0"),
        ClockLabel("粗", 31.0, 33.0, "source1:000000000000:1"),
    )
    alternative = (ClockLabel("新", 30.0, 33.0, "000000320000:0"),)
    with pytest.raises(ValueError, match="boundary"):
        asr_window_refine.derive_replacement(
            original,
            alternative,
            window=AudioWindow(320000, 320000),
            target_provenance="source1:000000000000:1",
        )


@pytest.mark.parametrize("target", ["", "missing", "duplicate"])
def test_planner_rejects_missing_or_ambiguous_provenance(target: str) -> None:
    original = (ClockLabel("粗", 1.0, 3.0, "duplicate"), ClockLabel("次", 3.0, 4.0, "duplicate"))
    with pytest.raises(ValueError, match="exact original"):
        asr_window_refine.derive_replacement(
            original, original, window=AudioWindow(0, 160000), target_provenance=target
        )


def test_planner_requires_recognized_bounded_support() -> None:
    original = (ClockLabel("粗", 31.0, 33.0, "000000000000:0"),)
    with pytest.raises(ValueError, match="No recognized"):
        asr_window_refine.derive_replacement(
            original, (), window=AudioWindow(480000, 160000), target_provenance="000000000000:0"
        )
    with pytest.raises(ValueError, match="bounded verified"):
        asr_window_refine.derive_replacement(
            original,
            (ClockLabel("外", 29.0, 33.0, "000000000000:1"),),
            window=AudioWindow(480000, 160000),
            target_provenance="000000000000:0",
        )
    dense = tuple(ClockLabel("密", 31.0, 33.0, f"000000000000:{i}") for i in range(513))
    with pytest.raises(ValueError, match="bounded verified"):
        asr_window_refine.derive_replacement(
            dense, original, window=AudioWindow(480000, 160000), target_provenance="000000000000:1"
        )
    with pytest.raises(ValueError, match="bounded verified"):
        asr_window_refine.derive_replacement(
            (ClockLabel("長", 0.0, 61.0, "000000000000:0"),),
            original,
            window=AudioWindow(0, 1600000),
            target_provenance="000000000000:0",
        )


def test_sample_envelope_does_not_expand_on_binary_float_roundoff() -> None:
    assert asr_window_refine._sample_boundary(1518.6400000000001, upper=True) == 24298240
    assert asr_window_refine._sample_boundary(1.00001) == 16000
    assert asr_window_refine._sample_boundary(1.00001, upper=True) == 16001


@pytest.mark.parametrize("failure", [None, "clock", "parent"])
def test_original_completed_overlap_reuses_native_offset_once(
    config: Path, failure: str | None
) -> None:
    document = record(json.loads(config.read_text()))
    artifacts = record(document["artifacts"])
    source, clip = config.with_name("whole.wav"), config.with_name("window.wav")
    pcm(source, 1120000)
    pcm(clip, 640000)
    original_inputs = record(json.loads(config.with_name("original-inputs.json").read_text()))
    new_inputs = record(json.loads(config.with_name("inputs.json").read_text()))
    dependencies = record(new_inputs["inputs"])
    dependencies.pop(str(clip))
    dependencies[str(source)] = file_digest(source)
    original_inputs.update(
        samples=1120000,
        inputs=dependencies,
        device="cpu",
        dtypeContract="provider-default-fp16-unchanged",
    )
    artifacts["originalInputs"] = save(config.with_name("original-inputs.json"), original_inputs)
    artifacts["inputs"] = artifacts["originalInputs"]
    raw_hash = str(record(artifacts["raw"])["sha256"])
    artifacts["complete"] = save(
        config.with_name("complete.json"),
        {"rawSHA256": ["a" * 64, raw_hash], "windows": 2, "deliveryComplete": False},
    )
    artifacts["clock"] = save(
        config.with_name("clock.json"),
        {
            "startSample": 480000,
            "endSample": 1120000 if failure != "clock" else 1120001,
            "sampleRate": 16000,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": raw_hash,
        },
    )
    artifacts["audio"] = {"path": str(clip), "sha256": file_digest(clip)}
    window = record(json.loads(config.with_name("window.json").read_text()))
    window.update(
        endSample=1120000, windowSHA256=file_digest(clip), sourceSHA256=file_digest(source)
    )
    artifacts["window"] = save(config.with_name("window.json"), window)
    assembled = record(json.loads(config.with_name("labels.json").read_text()))
    assembled["parentSHA256"] = {
        str(record(artifacts[name])["path"]): str(record(artifacts[name])["sha256"])
        for name in ("originalInputs", "complete", "raw", "clock")
        if name != "raw" or failure != "parent"
    }
    artifacts["labels"] = save(config.with_name("labels.json"), assembled)
    document.pop("replacement")
    document["targetProvenance"] = "old:1"
    save(config, document)
    output = config.with_name("overlap-candidate.json")
    if failure:
        with pytest.raises(ValueError, match="extraction clock|bound by the original"):
            asr_window_refine.run(config, output)
    else:
        assert asr_window_refine.run(config, output) == 4
        candidate = record(json.loads(output.read_text()))
        assert record(candidate["refinement"])["recognitionWindowKind"] == "original-overlap"
        labels = candidate["labels"]
        assert isinstance(labels, list)
        assert record(labels[1])["start"] == 31 and record(labels[2])["end"] == 33
        assert asr_window_refine.verify(config, output) == 4
