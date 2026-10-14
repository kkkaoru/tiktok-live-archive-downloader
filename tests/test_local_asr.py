import json
import sys
import wave
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pytest import MonkeyPatch

from replay import local_asr
from replay.asr_runtime import ASRDevice, WhisperRuntime
from replay.core import ReplayError


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: MonkeyPatch) -> tuple[Path, Path, Path]:
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "weights.safetensors").write_bytes(b"synthetic")
    module = ModuleType("mlx_whisper")

    def recognize(audio: object, **kwargs: object) -> object:
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (16000,)
        assert kwargs["word_timestamps"] is True
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["language"] == "ja"
        return {
            "language": "ja",
            "text": "テスト",
            "segments": [
                {
                    "text": "テスト",
                    "start": 0.1,
                    "end": 0.8,
                    "words": [{"word": "テスト", "start": 0.1, "end": 0.8, "probability": 0.9}],
                }
            ],
        }

    monkeypatch.setattr(module, "transcribe", recognize, raising=False)
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    return source, model, tmp_path / "evidence"


def test_recognition_preserves_raw_and_unreviewed_clock(inputs: tuple[Path, Path, Path]) -> None:
    source, model, output = inputs
    result = local_asr.transcribe_file(
        source,
        model=model,
        weights_sha256=local_asr.fingerprint(model / "weights.safetensors"),
        config_sha256=local_asr.fingerprint(model / "config.json"),
        output=output,
    )
    assert result.name == "raw-transcript.json"
    assert json.loads(result.read_text(encoding="utf-8"))["text"] == "テスト"
    assert json.loads((output / "validation.json").read_text(encoding="utf-8")) == {
        "word_groups": 1,
        "unresolved_points": 0,
        "human_reviewed": False,
        "source_clock_projected": False,
    }
    assert json.loads((output / "inputs.json").read_text(encoding="utf-8"))["samples"] == 16000
    with pytest.raises(FileExistsError):
        local_asr.transcribe_file(
            source,
            model=model,
            weights_sha256=local_asr.fingerprint(model / "weights.safetensors"),
            config_sha256=local_asr.fingerprint(model / "config.json"),
            output=output,
        )


def test_mismatched_hash_stops_before_output(inputs: tuple[Path, Path, Path]) -> None:
    source, model, output = inputs
    with pytest.raises(ReplayError, match="fingerprint mismatch"):
        local_asr.transcribe_file(
            source, model=model, weights_sha256="0" * 64, config_sha256="0" * 64, output=output
        )
    assert not output.exists()


def test_remote_or_missing_model_rejected(inputs: tuple[Path, Path, Path]) -> None:
    source, model, output = inputs
    with pytest.raises(ReplayError, match="local model directory"):
        local_asr.transcribe_file(
            source,
            model=model / "missing",
            weights_sha256="0" * 64,
            config_sha256="0" * 64,
            output=output,
        )


def test_ambiguous_weights_rejected(inputs: tuple[Path, Path, Path]) -> None:
    source, model, output = inputs
    (model / "weights.npz").write_bytes(b"other")
    with pytest.raises(ReplayError, match="exactly one"):
        local_asr.transcribe_file(
            source, model=model, weights_sha256="0" * 64, config_sha256="0" * 64, output=output
        )


def test_symlink_input_rejected(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"test")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(ReplayError, match="regular local"):
        local_asr.fingerprint(link)


@pytest.mark.parametrize("channels,rate,count", [(2, 16000, 16), (1, 8000, 16), (1, 16000, 0)])
def test_invalid_audio_contract(tmp_path: Path, channels: int, rate: int, count: int) -> None:
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\0\0" * count)
    with pytest.raises(ReplayError, match="PCM16/16kHz"):
        local_asr.pcm_audio(source)


def test_truncated_pcm_rejected(inputs: tuple[Path, Path, Path]) -> None:
    source, model, output = inputs
    source.write_bytes(source.read_bytes()[:-2])
    with pytest.raises(ReplayError, match="Truncated"):
        local_asr.pcm_audio(source)


def test_empty_asr_retained_not_accepted(
    inputs: tuple[Path, Path, Path], monkeypatch: MonkeyPatch
) -> None:
    source, model, output = inputs
    monkeypatch.setattr(
        sys.modules["mlx_whisper"],
        "transcribe",
        lambda *args, **kwargs: {"language": "ja", "text": "", "segments": []},
    )
    with pytest.raises(ReplayError, match="absent speech"):
        local_asr.transcribe_file(
            source,
            model=model,
            weights_sha256=local_asr.fingerprint(model / "weights.safetensors"),
            config_sha256=local_asr.fingerprint(model / "config.json"),
            output=output,
        )
    assert (output / "raw-transcript.json").is_file()
    assert not (output / "validation.json").exists()


def test_uncallable_recognizer(inputs: tuple[Path, Path, Path], monkeypatch: MonkeyPatch) -> None:
    source, model, output = inputs
    monkeypatch.setattr(sys.modules["mlx_whisper"], "transcribe", None)
    with pytest.raises(ReplayError, match="not callable"):
        local_asr.transcribe_file(
            source,
            model=model,
            weights_sha256=local_asr.fingerprint(model / "weights.safetensors"),
            config_sha256=local_asr.fingerprint(model / "config.json"),
            output=output,
        )


def test_source_change_retains_evidence(
    inputs: tuple[Path, Path, Path], monkeypatch: MonkeyPatch
) -> None:
    source, model, output = inputs

    def mutate(audio: object, **kwargs: object) -> object:
        source.write_bytes(b"changed")
        return {"language": "ja", "text": "", "segments": []}

    monkeypatch.setattr(sys.modules["mlx_whisper"], "transcribe", mutate)
    with pytest.raises(ReplayError, match="input changed"):
        local_asr.transcribe_file(
            source,
            model=model,
            weights_sha256=local_asr.fingerprint(model / "weights.safetensors"),
            config_sha256=local_asr.fingerprint(model / "config.json"),
            output=output,
        )
    assert (output / "raw-transcript.json").is_file()
    assert not (output / "validation.json").exists()


def test_cli_defaults_to_pinned_full_large_v3(monkeypatch: MonkeyPatch) -> None:
    def transcribe(
        source: Path,
        *,
        model: Path,
        weights_sha256: str,
        config_sha256: str,
        output: Path,
        device: ASRDevice,
    ) -> Path:
        assert device == ASRDevice.AUTO
        assert source == Path("input.wav")
        assert output == Path("new-evidence")
        assert model == Path(
            "edits/models/mlx-whisper-large-v3/49e6aa286ad60c14352c404340ded53710378a11"
        )
        assert weights_sha256 == "05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d"
        assert config_sha256 == "34982ce6ae286095000f82ae9583b3431639e8b092bf60c961f203745e6500e3"
        return output / "raw-transcript.json"

    monkeypatch.setattr(local_asr, "transcribe_file", transcribe)
    assert local_asr.main(["input.wav", "new-evidence"]) == 0


def test_cli_default_rejects_unverified_weights(inputs: tuple[Path, Path, Path]) -> None:
    source, model, output = inputs
    with pytest.raises(ReplayError, match="fingerprint mismatch"):
        local_asr.main([str(source), str(output), "--model", str(model)])
    assert not output.exists()


def test_cpu_cli_preserves_unreviewed_contract(
    inputs: tuple[Path, Path, Path], monkeypatch: MonkeyPatch
) -> None:
    source, model, output = inputs

    def predict(self: WhisperRuntime, samples: object, model: Path, *, evidence: Path) -> object:
        assert self.device == ASRDevice.CPU
        assert evidence.name == "cpu-runtime"
        return {
            "language": "ja",
            "text": "合成",
            "segments": [
                {
                    "start": 0.1,
                    "end": 0.8,
                    "text": "合成",
                    "words": [{"word": "合成", "start": 0.1, "end": 0.8, "probability": 0.9}],
                }
            ],
        }

    monkeypatch.setattr(WhisperRuntime, "predict", predict)
    assert (
        local_asr.main(
            [
                str(source),
                str(output),
                "--model",
                str(model),
                "--device",
                "cpu",
                "--weights-sha256",
                local_asr.fingerprint(model / "weights.safetensors"),
                "--config-sha256",
                local_asr.fingerprint(model / "config.json"),
            ]
        )
        == 0
    )
    saved = json.loads((output / "inputs.json").read_text(encoding="utf-8"))
    assert saved["device"] == "cpu"
    assert saved["dtype_contract"] == "provider-default-fp16-unchanged"
    assert saved["human_reviewed"] is False


def test_invalid_device_cli_never_runs_model(monkeypatch: MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> Path:
        pytest.fail("Invalid CLI must not execute inference")

    monkeypatch.setattr(local_asr, "transcribe_file", forbidden)
    with pytest.raises(SystemExit, match="2"):
        local_asr.main(["input.wav", "out", "--device", "cuda"])


def test_cli(inputs: tuple[Path, Path, Path]) -> None:
    source, model, output = inputs
    assert (
        local_asr.main(
            [
                str(source),
                str(output),
                "--model",
                str(model),
                "--weights-sha256",
                local_asr.fingerprint(model / "weights.safetensors"),
                "--config-sha256",
                local_asr.fingerprint(model / "config.json"),
            ]
        )
        == 0
    )
