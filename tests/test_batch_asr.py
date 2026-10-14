import fcntl
import json
import sys
import wave
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pytest import MonkeyPatch

from replay import batch_asr
from replay.asr_runtime import ASRDevice, WhisperRuntime
from replay.core import ReplayError
from replay.local_asr import fingerprint


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: MonkeyPatch) -> tuple[Path, Path, Path, list[int]]:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x01\x00" * (61 * 16000))
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.npz").write_bytes(b"synthetic weights")
    (model / "config.json").write_text("{}", encoding="utf-8")
    calls: list[int] = []
    module = ModuleType("mlx_whisper")

    def predict(samples: object, **kwargs: object) -> object:
        assert isinstance(samples, np.ndarray)
        calls.append(len(samples))
        assert kwargs["temperature"] == 0.0
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["word_timestamps"] is True
        assert kwargs["language"] == "ja"
        return {"segments": [], "text": ""}

    monkeypatch.setattr(module, "transcribe", predict, raising=False)
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    return source, model, tmp_path / "out", calls


def run(source: Path, model: Path, output: Path) -> Path:
    return batch_asr.transcribe_windows(
        source,
        output,
        model=model,
        weights_sha256=fingerprint(model / "weights.npz"),
        config_sha256=fingerprint(model / "config.json"),
    )


def test_exact_overlap_clocks_and_resume(setup: tuple[Path, Path, Path, list[int]]) -> None:
    source, model, output, calls = setup
    result = run(source, model, output)
    assert calls == [960000, 496000]
    saved = json.loads((output / "000000480000.clock.json").read_text())
    assert saved["startSample"] == 480000
    assert saved["endSample"] == 976000
    assert saved["emptyRecognitionIsNotSilence"] is True
    assert saved["humanReviewed"] is False
    assert json.loads(result.read_text())["deliveryComplete"] is False
    run(source, model, output)
    assert calls == [960000, 496000]


def test_partial_resume_without_duplicate_inference(
    setup: tuple[Path, Path, Path, list[int]], monkeypatch: MonkeyPatch
) -> None:
    source, model, output, calls = setup
    original = batch_asr._predict
    failed = False

    def interrupt(samples: np.ndarray[tuple[int], np.dtype[np.float32]], model: Path) -> object:
        nonlocal failed
        if len(samples) < 960000 and not failed:
            failed = True
            raise RuntimeError("interrupted")
        return original(samples, model)

    monkeypatch.setattr(batch_asr, "_predict", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(source, model, output)
    run(source, model, output)
    assert calls == [960000, 496000]


@pytest.mark.parametrize("mode", ["changed", "missing", "clock", "complete", "manifest"])
def test_tampered_completed_run_never_reinfer(
    setup: tuple[Path, Path, Path, list[int]], mode: str
) -> None:
    source, model, output, calls = setup
    run(source, model, output)
    paths = {
        "changed": "000000000000.raw.json",
        "missing": "000000000000.raw.json",
        "clock": "000000000000.clock.json",
        "complete": "complete.json",
        "manifest": "inputs.json",
    }
    path = output / paths[mode]
    if mode == "missing":
        path.unlink()
    else:
        path.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError):
        run(source, model, output)
    assert calls == [960000, 496000]


def test_missing_model_and_wrong_hash(setup: tuple[Path, Path, Path, list[int]]) -> None:
    source, model, output, calls = setup
    with pytest.raises(ReplayError, match="local model"):
        batch_asr.transcribe_windows(source, output, model=model / "absent")
    with pytest.raises(ReplayError, match="fingerprint"):
        batch_asr.transcribe_windows(source, output, model=model)
    assert calls == []
    assert not output.exists()


def test_ambiguous_weights(setup: tuple[Path, Path, Path, list[int]]) -> None:
    source, model, output, _ = setup
    (model / "weights.safetensors").write_bytes(b"other")
    with pytest.raises(ReplayError, match="exactly one"):
        run(source, model, output)


def test_invalid_audio(setup: tuple[Path, Path, Path, list[int]]) -> None:
    source, model, output, _ = setup
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0" * 64)
    with pytest.raises(ReplayError, match="bounded mono"):
        run(source, model, output)


def test_truncated_pcm_preserved(setup: tuple[Path, Path, Path, list[int]]) -> None:
    source, model, output, _ = setup
    source.write_bytes(source.read_bytes()[:100])
    with pytest.raises(ReplayError, match="Truncated"):
        run(source, model, output)
    assert not (output / "complete.json").exists()


def test_uncallable_provider(
    setup: tuple[Path, Path, Path, list[int]], monkeypatch: MonkeyPatch
) -> None:
    source, model, output, _ = setup
    monkeypatch.setattr(sys.modules["mlx_whisper"], "transcribe", None)
    with pytest.raises(ReplayError, match="not callable"):
        run(source, model, output)


def test_malformed_provider_retains_raw(
    setup: tuple[Path, Path, Path, list[int]], monkeypatch: MonkeyPatch
) -> None:
    source, model, output, _ = setup
    monkeypatch.setattr(batch_asr, "_predict", lambda *args: {"segments": None})
    with pytest.raises(ReplayError, match="segment array"):
        run(source, model, output)
    assert (output / "000000000000.raw.json").exists()
    with pytest.raises(ReplayError, match="bounded regular"):
        run(source, model, output)


def test_input_mutation_stops_completion(
    setup: tuple[Path, Path, Path, list[int]], monkeypatch: MonkeyPatch
) -> None:
    source, model, output, _ = setup

    def mutate(*args: object) -> object:
        (model / "config.json").write_text('{"changed":true}', encoding="utf-8")
        return {"segments": []}

    monkeypatch.setattr(batch_asr, "_predict", mutate)
    with pytest.raises(ReplayError, match="during inference"):
        run(source, model, output)
    assert not (output / "complete.json").exists()


def test_existing_unbound_and_linked_lock_refused(
    setup: tuple[Path, Path, Path, list[int]],
) -> None:
    source, model, output, _ = setup
    output.mkdir()
    with pytest.raises(ReplayError, match="no input receipt"):
        run(source, model, output)
    (output / ".run.lock").unlink()
    (output / ".run.lock").symlink_to(source)
    with pytest.raises(ReplayError, match="Linked ASR lock"):
        run(source, model, output)


def test_active_lock_refused(setup: tuple[Path, Path, Path, list[int]]) -> None:
    source, model, output, _ = setup
    output.mkdir()
    with (output / ".run.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run(source, model, output)


def test_read_bound_and_links(tmp_path: Path) -> None:
    huge = tmp_path / "huge"
    with huge.open("wb") as stream:
        stream.truncate(8 * 1024 * 1024 + 1)
    with pytest.raises(ReplayError, match="bounded regular"):
        batch_asr._read(huge)
    link = tmp_path / "link"
    link.symlink_to(huge)
    with pytest.raises(ReplayError, match="bounded regular"):
        batch_asr._read(link)


def test_cpu_windows_share_runtime_and_resume_without_inference(
    setup: tuple[Path, Path, Path, list[int]], monkeypatch: MonkeyPatch
) -> None:
    source, model, output, calls = setup
    instances: list[WhisperRuntime] = []

    def predict(self: WhisperRuntime, samples: object, model: Path, *, evidence: Path) -> object:
        assert self.device == ASRDevice.CPU
        instances.append(self)
        return {"segments": []}

    monkeypatch.setattr(WhisperRuntime, "predict", predict)
    batch_asr.transcribe_windows(
        source,
        output,
        model=model,
        device=ASRDevice.CPU,
        weights_sha256=fingerprint(model / "weights.npz"),
        config_sha256=fingerprint(model / "config.json"),
    )
    assert len(instances) == 2
    assert instances[0] is instances[1]
    saved = json.loads((output / "inputs.json").read_text(encoding="utf-8"))
    assert saved["device"] == "cpu"
    assert saved["dtypeContract"] == "provider-default-fp16-unchanged"
    batch_asr.transcribe_windows(
        source,
        output,
        model=model,
        device=ASRDevice.CPU,
        weights_sha256=fingerprint(model / "weights.npz"),
        config_sha256=fingerprint(model / "config.json"),
    )
    assert len(instances) == 2
    with pytest.raises(ReplayError, match="inputs changed"):
        run(source, model, output)
    assert calls == []


def test_cpu_cli(monkeypatch: MonkeyPatch) -> None:
    def capture(source: Path, output: Path, *, device: ASRDevice) -> Path:
        assert device == ASRDevice.CPU
        return output

    monkeypatch.setattr(batch_asr, "transcribe_windows", capture)
    assert batch_asr.main(["source.wav", "out", "--device", "cpu"]) == 0
    with pytest.raises(SystemExit, match="2"):
        batch_asr.main(["source.wav", "out", "--device", "cuda"])


def test_cli(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[Path, Path]] = []

    def capture(source: Path, output: Path, *, device: ASRDevice) -> Path:
        assert device == ASRDevice.AUTO
        calls.append((source, output))
        return output

    monkeypatch.setattr(batch_asr, "transcribe_windows", capture)
    assert batch_asr.main(["source.wav", "new-output"]) == 0
    assert calls == [(Path("source.wav"), Path("new-output"))]
