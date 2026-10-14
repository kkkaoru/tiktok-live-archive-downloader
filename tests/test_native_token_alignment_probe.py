import importlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import asr_adoption
from replay import native_token_alignment_probe as probe
from replay.core import ReplayError
from replay.local_asr import fingerprint


class FakeBackend:
    def align(
        self, samples: NDArray[np.float32], model: Path, windows: tuple[probe.TokenWindow, ...]
    ) -> tuple[probe.NativeWord, ...]:
        return (probe.NativeWord("ター", (30736,), 0.1, 0.9, 0.5),)


@pytest.fixture
def probe_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    work = tmp_path / "edits/work/probe"
    work.mkdir(parents=True)
    model = tmp_path / "edits/models/model"
    model.mkdir(parents=True)
    source = work / "window.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)
    config, weights = model / "config.json", model / "weights.npz"
    config.write_text('{"n_mels":128}', encoding="utf-8")
    weights.write_bytes(b"synthetic")
    inputs = work / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": fingerprint(source),
                "model": str(model),
                "config_sha256": fingerprint(config),
                "weights_sha256": fingerprint(weights),
            }
        ),
        encoding="utf-8",
    )
    raw = work / "raw-transcript.json"
    raw.write_text(
        json.dumps(
            {
                "language": "ja",
                "text": "ター",
                "segments": [
                    {
                        "seek": 0,
                        "start": 0.1,
                        "end": 0.9,
                        "text": "ター",
                        "tokens": [50365, 30736, 50415],
                        "words": [{"word": "ター", "start": 0.1, "end": 0.9, "probability": 0.5}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifacts = {
        "source": source,
        "raw": raw,
        "inputs": inputs,
        "config": config,
        "weights": weights,
    }
    extra = {
        "validation": {},
        "window": {
            "startSample": 16000,
            "endSample": 32000,
            "sampleRate": 16000,
            "windowSHA256": fingerprint(source),
            "clock": "continuous-input-pcm",
        },
        "workflow": {},
        "checkpoint": {
            "status": "complete",
            "outputs": {str(raw): fingerprint(raw), str(inputs): fingerprint(inputs)},
        },
        "pins": {},
        "end": {"exitCode": 0, "completed": True},
        "before": {"synthetic": "unchanged"},
        "after": {"hashes": {"synthetic": "unchanged"}, "differences": {}},
    }
    locations = {
        name: work / ("recognize.json" if name == "checkpoint" else name + ".json")
        for name in extra
    }
    tuple(
        locations[name].write_text(json.dumps(value), encoding="utf-8")
        for name, value in extra.items()
    )
    artifacts.update(locations)
    checkpoint = locations["checkpoint"]
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    data["outputs"][str(locations["validation"])] = fingerprint(locations["validation"])
    checkpoint.write_text(json.dumps(data), encoding="utf-8")
    own = Path(probe.__file__).resolve()
    document = {
        "version": 1,
        "kind": "native-token-alignment-probe-request",
        "artifacts": {
            name: {"path": str(path), "sha256": fingerprint(path)}
            for name, path in artifacts.items()
        },
        "runtime": [{"path": str(own), "sha256": fingerprint(own)}],
        "model": str(model),
        "stage": "recognize",
    }
    path = work / "request.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(asr_adoption, "verify", lambda *args: None)
    return path


def test_run_verify_conserves_single_bpe(probe_request: Path) -> None:
    output = probe_request.parent / "result.json"
    probe.run(probe_request, output, backend=FakeBackend())
    probe.verify(probe_request, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["words"] == [
        {
            "text": "ター",
            "tokens": [30736],
            "start": 0.1,
            "end": 0.9,
            "probability": 0.5,
            "sourceStart": 1.1,
            "sourceEnd": 1.9,
        }
    ]
    assert result["autoregressiveDecoding"] is False
    assert result["semanticApproved"] is False


@pytest.mark.parametrize("value", [True, -1, 1.5, "0"])
def test_integer_rejects_malformed(value: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        probe._integer(value)


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf")])
def test_number_rejects_malformed(value: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        probe._number(value)


@pytest.mark.parametrize("value", [None, "", "\0"])
def test_text_rejects_malformed(value: object) -> None:
    with pytest.raises(ReplayError, match="text"):
        probe._text(value)


def test_request_input_tamper(probe_request: Path) -> None:
    (probe_request.parent / "raw-transcript.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="Reference changed"):
        probe.load_request(probe_request)


def test_missing_artifact(probe_request: Path) -> None:
    data = json.loads(probe_request.read_text(encoding="utf-8"))
    del data["artifacts"]["window"]
    probe_request.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="producer references"):
        probe.load_request(probe_request)


def test_escaping_output(probe_request: Path) -> None:
    with pytest.raises(ReplayError, match="checkout-local"):
        probe.run(
            probe_request,
            probe_request.parent.parent.parent.parent / "bad.json",
            backend=FakeBackend(),
        )


def test_no_overwrite(probe_request: Path) -> None:
    output = probe_request.parent / "result.json"
    probe.run(probe_request, output, backend=FakeBackend())
    with pytest.raises(ReplayError, match="new checkout-local"):
        probe.run(probe_request, output, backend=FakeBackend())


def test_window_clock_tamper(probe_request: Path) -> None:
    data = json.loads(probe_request.read_text(encoding="utf-8"))
    window = probe_request.parent / "window.json"
    value = json.loads(window.read_text(encoding="utf-8"))
    value["endSample"] = 31999
    window.write_text(json.dumps(value), encoding="utf-8")
    data["artifacts"]["window"]["sha256"] = fingerprint(window)
    probe_request.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="Window clock"):
        probe.load_request(probe_request)


def test_output_clock_tamper(probe_request: Path) -> None:
    output = probe_request.parent / "result.json"
    probe.run(probe_request, output, backend=FakeBackend())
    result = json.loads(output.read_text(encoding="utf-8"))
    result["words"][0]["sourceStart"] = 1.11
    output.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ReplayError, match="adjusted"):
        probe.verify(probe_request, output)


def test_output_binding_tamper(probe_request: Path) -> None:
    output = probe_request.parent / "result.json"
    probe.run(probe_request, output, backend=FakeBackend())
    result = json.loads(output.read_text(encoding="utf-8"))
    result["request"]["sha256"] = "0" * 64
    output.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ReplayError, match="binding"):
        probe.verify(probe_request, output)


def test_backend_clock_outside_pcm(probe_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        FakeBackend, "align", lambda *args: (probe.NativeWord("ター", (30736,), 0.1, 1.01, 0.5),)
    )
    with pytest.raises(ReplayError, match="outside PCM"):
        probe.run(probe_request, probe_request.parent / "result.json", backend=FakeBackend())


def test_backend_cannot_subdivide_authentic_token(
    probe_request: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        FakeBackend,
        "align",
        lambda *args: (
            probe.NativeWord("タ", (100,), 0.1, 0.5, 0.5),
            probe.NativeWord("ー", (101,), 0.5, 0.9, 0.5),
        ),
    )
    with pytest.raises(ReplayError, match="token identity"):
        probe.run(probe_request, probe_request.parent / "result.json", backend=FakeBackend())


def test_backend_cannot_change_text(probe_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        FakeBackend, "align", lambda *args: (probe.NativeWord("wrong", (30736,), 0.1, 0.9, 0.5),)
    )
    with pytest.raises(ReplayError, match="text identity"):
        probe.run(probe_request, probe_request.parent / "result.json", backend=FakeBackend())


def test_quota(probe_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "MAX_BYTES", 1)
    with pytest.raises(ReplayError, match="quota"):
        probe.run(probe_request, probe_request.parent / "result.json", backend=FakeBackend())


def test_installed_forward_only(probe_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tensor = SimpleNamespace(astype=lambda *args: object(), __getitem__=lambda *args: object())
    native = SimpleNamespace(
        is_multilingual=True, num_languages=100, dims=SimpleNamespace(n_mels=128)
    )
    word = SimpleNamespace(word="ター", tokens=[30736], start=0.1, end=0.9, probability=0.5)
    loader = SimpleNamespace(snapshot_download=object(), load_model=lambda *args, **kwargs: native)
    modules = {
        "mlx.core": SimpleNamespace(float16=object()),
        "mlx_whisper.load_models": loader,
        "mlx_whisper.audio": SimpleNamespace(
            log_mel_spectrogram=lambda *args, **kwargs: tensor,
            pad_or_trim=lambda *args, **kwargs: tensor,
        ),
        "mlx_whisper.timing": SimpleNamespace(find_alignment=lambda *args, **kwargs: [word]),
        "mlx_whisper.tokenizer": SimpleNamespace(
            get_tokenizer=lambda *args, **kwargs: SimpleNamespace(decode=lambda *args: "ター")
        ),
    }
    monkeypatch.setattr(importlib, "import_module", lambda name: modules[name])
    probe.run(probe_request, probe_request.parent / "result.json")
    assert json.loads((probe_request.parent / "result.json").read_text(encoding="utf-8"))["words"][
        0
    ]["tokens"] == [30736]


def test_invocation_reject() -> None:
    with pytest.raises(ReplayError, match="not callable"):
        probe._invoke(None)


def test_offline_required(probe_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE")
    with pytest.raises(ReplayError, match="offline"):
        probe.run(probe_request, probe_request.parent / "result.json")


def test_native_word_bounds() -> None:
    with pytest.raises(ReplayError, match="bounds"):
        probe._native_word(
            SimpleNamespace(word="x", tokens=[1], start=0, end=1.1, probability=0.5),
            offset=0,
            duration=1,
        )


def test_native_numpy_probability_matches_installed_api() -> None:
    word = probe._native_word(
        SimpleNamespace(
            word="ター",
            tokens=[30736],
            start=np.float64(0.1),
            end=np.float64(0.9),
            probability=np.float32(0.5),
        ),
        offset=0,
        duration=1,
    )
    assert word == probe.NativeWord("ター", (30736,), 0.1, 0.9, 0.5)


def test_numpy_nonfinite_probability_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        probe._number(np.float32("nan"))


def test_numpy_integer_scalar() -> None:
    assert probe._number(np.int32(1)) == 1.0


@pytest.mark.parametrize("value", [np.bool_(True), np.array([1.0]), complex(1, 0), object()])
def test_nonscalar_numeric_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        probe._number(value)


def test_cli_verify(probe_request: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = probe_request.parent / "result.json"
    probe.run(probe_request, output, backend=FakeBackend())
    probe.main([str(probe_request), str(output), "--verify"])
    assert capsys.readouterr().out == ""


def test_cli_run(probe_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "InstalledAlignment", FakeBackend)
    output = probe_request.parent / "result.json"
    probe.main([str(probe_request), str(output)])
    assert output.is_file()


@pytest.fixture
def retained_request(probe_request: Path) -> Path:
    data = json.loads(probe_request.read_text(encoding="utf-8"))
    work = probe_request.parent
    original = work / "original.wav"
    with wave.open(str(original), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 960000)
    crop = Path(data["artifacts"]["source"]["path"])
    with wave.open(str(crop), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 480000)
    raw_path = Path(data["artifacts"]["raw"]["path"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["segments"][0]["seek"] = 100
    raw["segments"][0]["words"][0].update(start=1.1, end=1.9)
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    inputs = Path(data["artifacts"]["inputs"]["path"])
    inputs.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "input-pcm-not-projected",
                "sampleRate": 16000,
                "samples": 960000,
                "windowSamples": 960000,
                "strideSamples": 480000,
                "inputs": {
                    str(original): fingerprint(original),
                    data["artifacts"]["config"]["path"]: data["artifacts"]["config"]["sha256"],
                    data["artifacts"]["weights"]["path"]: data["artifacts"]["weights"]["sha256"],
                },
            }
        ),
        encoding="utf-8",
    )
    complete, clock = work / "complete.json", work / "clock.json"
    complete.write_text(
        json.dumps({"windows": 1, "deliveryComplete": False, "rawSHA256": [fingerprint(raw_path)]}),
        encoding="utf-8",
    )
    clock.write_text(
        json.dumps(
            {
                "startSample": 0,
                "endSample": 960000,
                "sampleRate": 16000,
                "humanReviewed": False,
                "emptyRecognitionIsNotSilence": True,
                "rawSHA256": fingerprint(raw_path),
            }
        ),
        encoding="utf-8",
    )
    window = Path(data["artifacts"]["window"]["path"])
    window.write_text(
        json.dumps(
            {
                "startSample": 16000,
                "endSample": 496000,
                "sampleRate": 16000,
                "source": str(original),
                "sourceSHA256": fingerprint(original),
                "windowSHA256": fingerprint(crop),
                "clock": "continuous-input-pcm",
                "sourceClockProjected": False,
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "original": original,
        "complete": complete,
        "clock": clock,
        "source": crop,
        "raw": raw_path,
        "inputs": inputs,
        "window": window,
        "config": Path(data["artifacts"]["config"]["path"]),
        "weights": Path(data["artifacts"]["weights"]["path"]),
    }
    data = {
        "version": 2,
        "kind": "native-token-alignment-probe-request",
        "artifacts": {
            name: {"path": str(path), "sha256": fingerprint(path)} for name, path in paths.items()
        },
        "runtime": data["runtime"],
        "model": data["model"],
        "origin": {
            "kind": "retained-clocked-collection",
            "segmentIndices": [0],
            "historicalDtypeObserved": False,
            "computeDtype": "float16-provider-default",
        },
    }
    probe_request.write_text(json.dumps(data), encoding="utf-8")
    return probe_request


def test_retained_actual_seek_crop_and_missing_dtype(retained_request: Path) -> None:
    output = retained_request.parent / "retained-result.json"
    probe.run(retained_request, output, backend=FakeBackend())
    probe.verify(retained_request, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["sourceStartSample"] == 16000
    assert result["samples"] == 480000
    assert result["origin"] == {
        "kind": "retained-clocked-collection",
        "segmentIndices": [0],
        "historicalDtypeObserved": False,
        "computeDtype": "float16-provider-default",
        "originalSeek": 100,
        "ownerStartSample": 0,
        "ownerEndSample": 960000,
        "contextChanged": True,
        "contextUncertainty": "Crop log-mel FFT boundaries may differ from original mel slice",
        "historicalDeclaredDtypeContract": None,
    }
    assert result["words"][0]["sourceStart"] == 1.1


@pytest.mark.parametrize(
    "field,value",
    [
        ("historicalDtypeObserved", True),
        ("computeDtype", "float32"),
        ("segmentIndices", []),
        ("segmentIndices", [1]),
        ("segmentIndices", [0, 0]),
        ("kind", "local-asr-completed"),
    ],
)
def test_retained_origin_rejects_false_history(
    retained_request: Path, field: str, value: object
) -> None:
    data = json.loads(retained_request.read_text(encoding="utf-8"))
    data["origin"][field] = value
    retained_request.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        probe.load_request(retained_request)


@pytest.mark.parametrize(
    "artifact,field,value",
    [
        ("window", "startSample", 0),
        ("window", "endSample", 496001),
        ("window", "endSample", 16000),
        ("window", "endSample", 976000),
        ("window", "sourceClockProjected", True),
        ("window", "sampleRate", 8000),
        ("complete", "windows", 2),
        ("complete", "rawSHA256", []),
        ("complete", "deliveryComplete", True),
        ("clock", "rawSHA256", "0" * 64),
        ("inputs", "inputs", {}),
        ("raw", "segments", []),
    ],
)
def test_retained_rejects_rebound_bad_evidence(
    retained_request: Path,
    artifact: str,
    field: str,
    value: object,
) -> None:
    data = json.loads(retained_request.read_text(encoding="utf-8"))
    path = Path(data["artifacts"][artifact]["path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    document[field] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    data["artifacts"][artifact]["sha256"] = fingerprint(path)
    retained_request.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        probe.load_request(retained_request)


def test_retained_pcm_bytes_must_match(retained_request: Path) -> None:
    data = json.loads(retained_request.read_text(encoding="utf-8"))
    crop = Path(data["artifacts"]["source"]["path"])
    content = bytearray(crop.read_bytes())
    content[-1] = 1
    crop.write_bytes(content)
    data["artifacts"]["source"]["sha256"] = fingerprint(crop)
    window = Path(data["artifacts"]["window"]["path"])
    receipt = json.loads(window.read_text(encoding="utf-8"))
    receipt["windowSHA256"] = fingerprint(crop)
    window.write_text(json.dumps(receipt), encoding="utf-8")
    data["artifacts"]["window"]["sha256"] = fingerprint(window)
    retained_request.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="Crop bytes"):
        probe.load_request(retained_request)


def test_retained_latter_context_uses_actual_seek(retained_request: Path) -> None:
    data = json.loads(retained_request.read_text(encoding="utf-8"))
    raw_path = Path(data["artifacts"]["raw"]["path"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["segments"][0]["seek"] = 2734
    raw["segments"][0]["words"][0].update(start=36.5, end=40.54)
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    complete = Path(data["artifacts"]["complete"]["path"])
    complete.write_text(
        json.dumps({"windows": 1, "deliveryComplete": False, "rawSHA256": [fingerprint(raw_path)]}),
        encoding="utf-8",
    )
    clock = Path(data["artifacts"]["clock"]["path"])
    owner = json.loads(clock.read_text(encoding="utf-8"))
    owner["rawSHA256"] = fingerprint(raw_path)
    clock.write_text(json.dumps(owner), encoding="utf-8")
    window = Path(data["artifacts"]["window"]["path"])
    receipt = json.loads(window.read_text(encoding="utf-8"))
    receipt.update(startSample=437440, endSample=917440)
    window.write_text(json.dumps(receipt), encoding="utf-8")
    data["artifacts"]["raw"]["sha256"] = fingerprint(raw_path)
    data["artifacts"]["complete"]["sha256"] = fingerprint(complete)
    data["artifacts"]["clock"]["sha256"] = fingerprint(clock)
    data["artifacts"]["window"]["sha256"] = fingerprint(window)
    retained_request.write_text(json.dumps(data), encoding="utf-8")
    parsed = probe.load_request(retained_request)
    assert parsed.start_sample == 437440
    assert parsed.windows == (probe.TokenWindow(0, (30736,), "ター"),)
    assert parsed.origin is not None
    assert parsed.origin["originalSeek"] == 2734
