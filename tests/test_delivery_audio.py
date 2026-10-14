import json
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import delivery_audio as audio
from replay.core import ReplayError


def test_clip_safe_gain_lowers_only_hot_audio() -> None:
    assert audio.clip_safe_gain("[Parsed] max_volume: 1.5 dB") == -1.6
    assert audio.clip_safe_gain("max_volume: -6.0 dB") == 0.0


def test_clip_safe_gain_requires_measurement() -> None:
    with pytest.raises(ReplayError, match="max_volume"):
        audio.clip_safe_gain("no measurement")


def test_clip_safe_gain_rejects_silence() -> None:
    with pytest.raises(ReplayError, match="silent"):
        audio.clip_safe_gain("max_volume: -inf dB")


def test_input_command_converts_on_continuous_clock(tmp_path: Path) -> None:
    command = audio.input_command(tmp_path / "in.mkv", tmp_path / "out.wav", gain=-1.5, ffmpeg="ff")
    assert command[command.index("-af") + 1] == "asetpts=N/SR/TB,aresample=44100,volume=-1.500000dB"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"


def test_separate_writes_receipt(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(arguments: Sequence[str]) -> str:
        calls.append(list(arguments))
        return "max_volume: 0.5 dB"

    def separator(model_input: Path, vocals: Path) -> str:
        vocals.write_bytes(model_input.name.encode())
        return "SeparationResult(realtime_factor=6.5)"

    receipt = audio.separate(
        tmp_path / "source.mkv", tmp_path / "out", runner=runner, separator=separator
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["gainDB"] == -0.6
    assert data["separation"] == "'SeparationResult(realtime_factor=6.5)'"
    assert (tmp_path / "out" / "vocals.wav").read_text(encoding="utf-8") == "model-input.wav"
    assert calls[0][-2:] == ["null", "-"]


def test_separate_refuses_existing_directory(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        audio.separate(tmp_path / "source.mkv", tmp_path / "out")


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert audio.main([str(tmp_path / "missing.mkv"), str(tmp_path / "out")]) == 1
    assert "DELIVERY_AUDIO_ERROR" in capsys.readouterr().err


@pytest.mark.parametrize("damage", ["none", "length", "truncated", "receipt", "format"])
def test_material_verifier(tmp_path: Path, damage: str) -> None:
    source = tmp_path / "source.mov"
    directory = tmp_path / "separation"
    directory.mkdir()
    model_input, vocals = directory / "model-input.wav", directory / "vocals.wav"
    with wave.open(str(model_input), "wb") as stream:
        stream.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(400))
    with wave.open(str(vocals), "wb") as stream:
        stream.setparams(
            (2, 2, 16000 if damage == "format" else 44100, 0, "NONE", "not compressed")
        )
        stream.writeframes(bytes(404 if damage == "length" else 400))
    if damage == "truncated":
        vocals.write_bytes(vocals.read_bytes()[:-4])
    (directory / "result.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "modelInput": str(model_input),
                "vocals": str(vocals),
                "timestampCorrection": damage == "receipt",
                "deliveryComplete": False,
            }
        ),
        encoding="utf-8",
    )
    assert audio.main([str(source), str(directory), "--verify"]) == (0 if damage == "none" else 1)


def test_default_separator_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from replay import bgm_separation

    monkeypatch.setattr(bgm_separation, "separate_vocals", lambda *args: "synthetic")
    receipt = audio.separate(
        tmp_path / "source", tmp_path / "result", runner=lambda arguments: "max_volume: -2 dB"
    )
    monkeypatch.setattr(audio, "separate", lambda *args, **kwargs: receipt)
    assert audio.main([str(tmp_path / "source"), str(tmp_path / "result")]) == 0


def test_default_runner_returns_stderr() -> None:
    assert audio._run(["/bin/sh", "-c", "echo measured >&2"]) == "measured\n"


def test_prepared_cpu_reuses_input_without_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from replay import audio_cpu_probe, bgm_separation

    monkeypatch.setattr(audio_cpu_probe, "audit_descriptors", lambda *args: None)
    monkeypatch.setattr(bgm_separation, "_import", lambda name: None)
    source = tmp_path / "prepared.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(400))
    original = source.read_bytes()

    def separator(model_input: Path, vocals: Path, config: bgm_separation.SeparationConfig) -> str:
        assert model_input == source
        assert config.device == "cpu"
        vocals.write_bytes(model_input.read_bytes())
        return "synthetic CPU"

    def no_decode(arguments: Sequence[str]) -> str:
        pytest.fail("Prepared input must not launch a decoder")

    monkeypatch.setattr(bgm_separation, "separate_vocals", separator)
    receipt = audio.separate(
        source, tmp_path / "out", prepared_input=True, device="cpu", runner=no_decode
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["preparedInput"] is True
    assert data["device"] == "cpu"
    assert data["gainDB"] == 0.0
    assert source.read_bytes() == original
    assert not (tmp_path / "out/model-input.wav").exists()
    audio.verify_material(source, tmp_path / "out")


@pytest.mark.parametrize("rate", [16000, 48000])
def test_prepared_input_refuses_wrong_clock(tmp_path: Path, rate: int) -> None:
    source = tmp_path / "prepared.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(400))
    with pytest.raises(ReplayError, match="Prepared input"):
        audio.separate(source, tmp_path / "out", prepared_input=True, device="cpu")
    assert not (tmp_path / "out").exists()
