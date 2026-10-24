import importlib
import json
import math
import subprocess
import types
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from replay import speech_onsets
from replay.core import ReplayError
from replay.speech_onsets import Piece, Token

VOCABULARY = {"<pad>": 0, "あ": 1, "い": 2, "あい": 3, "う": 4}
STRONG, FAINT = math.log(0.98), math.log(0.01)


def emission_rows(frames: int, peaks: dict[int, int]) -> npt.NDArray[np.float64]:
    """Blank everywhere except each peak frame, which emits its vocabulary entry."""
    rows = np.full((frames, 5), FAINT)
    rows[:, 0] = STRONG
    rows[list(peaks), 0] = FAINT
    rows[list(peaks), list(peaks.values())] = STRONG
    return rows


def document(*tokens: object) -> dict[str, object]:
    return {"clock": "output", "tokens": list(tokens)}


def test_tokens_are_read_from_an_output_clock_document() -> None:
    read = speech_onsets.read_tokens(
        document({"text": "あ", "start": 1, "end": 1.5, "utterance": "u:0"})
    )
    assert read == [Token("あ", 1.0, 1.5, "u:0")]


def test_a_source_clock_document_is_refused() -> None:
    with pytest.raises(ReplayError, match="output-clock"):
        speech_onsets.read_tokens({"clock": "source", "tokens": []})


def test_an_empty_token_list_is_refused() -> None:
    with pytest.raises(ReplayError, match="nonempty"):
        speech_onsets.read_tokens(document())


def test_a_token_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(ReplayError, match="Invalid speech token"):
        speech_onsets.read_tokens(document("あ"))


def test_a_token_without_text_is_refused() -> None:
    with pytest.raises(ReplayError, match="Invalid speech token"):
        speech_onsets.read_tokens(document({"text": "", "start": 0, "end": 1, "utterance": "u"}))


def test_a_boolean_token_time_is_refused() -> None:
    with pytest.raises(ReplayError, match="Invalid speech token"):
        speech_onsets.read_tokens(
            document({"text": "あ", "start": True, "end": 1, "utterance": "u"})
        )


def test_a_token_ending_before_it_starts_is_refused() -> None:
    with pytest.raises(ReplayError, match="interval"):
        speech_onsets.read_tokens(document({"text": "あ", "start": 2, "end": 1, "utterance": "u"}))


def test_unordered_tokens_are_refused() -> None:
    with pytest.raises(ReplayError, match="ordered"):
        speech_onsets.read_tokens(
            document(
                {"text": "あ", "start": 2, "end": 3, "utterance": "u"},
                {"text": "い", "start": 1, "end": 3, "utterance": "u"},
            )
        )


def test_windows_split_at_long_pauses_and_long_spans() -> None:
    tokens = [
        Token("あ", 0.0, 0.5, "u"),
        Token("い", 0.6, 1.0, "u"),
        Token("う", 3.0, 3.5, "u"),
        Token("あ", 3.6, 24.0, "u"),
    ]
    assert speech_onsets.windows(tokens) == [range(0, 2), range(2, 3), range(3, 4)]


def test_pieces_take_the_longest_entry_and_skip_unvoiced_and_unknown_text() -> None:
    assert speech_onsets.pieces("あい ー?えう", VOCABULARY) == [Piece(0, 2, 3), Piece(6, 7, 4)]


def test_forced_alignment_finds_each_targets_first_frame() -> None:
    rows = emission_rows(8, {1: 1, 2: 1, 5: 2})
    assert speech_onsets.forced_align(rows, [1, 2]) == [(1, STRONG), (5, STRONG)]


def test_repeated_targets_are_separated_by_a_blank() -> None:
    rows = emission_rows(6, {1: 1, 4: 1})
    assert speech_onsets.forced_align(rows, [1, 1]) == [(1, STRONG), (4, STRONG)]


def test_audio_too_short_for_the_text_is_refused() -> None:
    with pytest.raises(ValueError, match="too short"):
        speech_onsets.forced_align(emission_rows(2, {}), [1, 1])


def test_an_alignment_without_any_path_is_refused() -> None:
    rows = np.full((3, 5), -1e31)
    with pytest.raises(ValueError, match="No CTC path"):
        speech_onsets.forced_align(rows, [1])


def test_measured_onsets_keep_strong_pieces_only() -> None:
    tokens = [Token("あい", 1.0, 1.4, "u"), Token("う", 1.4, 1.6, "u")]
    rows = emission_rows(40, {10: 3, 30: 4})
    rows[30, 4] = math.log(0.001)  # う is barely supported
    found = speech_onsets.measured(tokens, 0.5, rows, VOCABULARY)
    assert found == {0: pytest.approx(0.7), 1: pytest.approx(0.7)}


def test_text_without_vocabulary_pieces_measures_nothing() -> None:
    tokens = [Token("ー", 1.0, 1.4, "u")]
    assert speech_onsets.measured(tokens, 0.5, emission_rows(40, {}), VOCABULARY) == {}


def test_a_window_without_an_alignment_measures_nothing() -> None:
    tokens = [Token("あいう", 1.0, 1.4, "u")]
    assert speech_onsets.measured(tokens, 0.5, emission_rows(1, {}), VOCABULARY) == {}


def test_characters_take_measured_starts_and_never_go_back() -> None:
    tokens = [Token("あい", 1.0, 1.4, "u:0"), Token("う", 1.4, 1.6, "u:1")]
    result = speech_onsets.characters(tokens, {0: 1.1, 1: 1.5})
    assert result == [
        {"text": "あ", "start": 1.1, "end": 1.5, "anchor": 1.0, "utterance": "u:0",
         "timing": "ctc"},
        {"text": "い", "start": 1.5, "end": 1.5, "anchor": 1.0, "utterance": "u:0",
         "timing": "ctc"},
        {"text": "う", "start": 1.5, "end": 1.6, "anchor": 1.4, "utterance": "u:1",
         "timing": "asr"},
    ]  # fmt: skip


def test_window_bounds_reach_past_the_speech_but_not_into_the_next_window() -> None:
    tokens = [Token("あ", 1.0, 2.0, "u"), Token("い", 2.8, 3.0, "u")]
    assert speech_onsets.bounds(tokens, range(0, 1), 10.0) == (0.5, 2.8)
    assert speech_onsets.bounds(tokens, range(1, 2), 3.2) == (2.3, 3.2)


def test_bucketed_audio_is_normalized_and_padded_to_a_whole_bucket() -> None:
    samples = np.array([1.0, -1.0], dtype=np.float32)
    padded = speech_onsets.bucketed(samples)
    assert padded.size == 32_000
    assert padded[:3].tolist() == [pytest.approx(1.0), pytest.approx(-1.0), 0.0]


def test_align_times_each_character_of_the_recording() -> None:
    tokens = [Token("あ", 1.0, 1.4, "u:0"), Token("え", 1.4, 1.6, "u:0")]

    def emissions(samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        return emission_rows(samples.size // 320, {35: 1})

    audio = np.zeros(3 * 16_000, dtype=np.float32)
    result, counts = speech_onsets.align(tokens, audio, emissions, VOCABULARY)
    assert [item["start"] for item in result] == [pytest.approx(1.2), pytest.approx(1.4)]
    assert [item["timing"] for item in result] == ["ctc", "asr"]
    assert counts == {"windows": 1, "unalignedWindows": 0, "characters": 2,
                      "measuredCharacters": 1}  # fmt: skip


def model_directory(tmp_path: Path, *, config: str = "{}") -> Path:
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "vocab.json").write_text(json.dumps(VOCABULARY), encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"weights")
    files = {
        "config.json": {"sha256": speech_onsets.digest(directory / "config.json")},
        "vocab.json": {"sha256": speech_onsets.digest(directory / "vocab.json")},
        "model.safetensors": {"sha256": speech_onsets.digest(directory / "model.safetensors")},
    }
    (directory / "config.json").write_text(config, encoding="utf-8")
    record = {"files": files}
    (directory / "ACQUISITION-RECORD.json").write_text(json.dumps(record), encoding="utf-8")
    return directory


def test_the_model_files_match_their_acquisition_record(tmp_path: Path) -> None:
    directory = model_directory(tmp_path)
    files = speech_onsets.verified_model(directory)
    assert sorted(files) == ["config.json", "model.safetensors", "vocab.json"]
    assert files["model.safetensors"] == (
        "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"
    )


def test_a_changed_model_file_is_refused(tmp_path: Path) -> None:
    directory = model_directory(tmp_path, config='{"changed": true}')
    with pytest.raises(ReplayError, match="config.json does not match"):
        speech_onsets.verified_model(directory)


def test_a_model_without_an_acquisition_record_is_refused(tmp_path: Path) -> None:
    directory = model_directory(tmp_path)
    (directory / "ACQUISITION-RECORD.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ReplayError, match="acquisition record"):
        speech_onsets.verified_model(directory)


class FakeModel:
    def __init__(self) -> None:
        self.device = ""

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        return self

    def __call__(self, batch: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(logits=FakeTensor(np.zeros((1, 2, 3))))


class FakeTensor:
    def __init__(self, values: npt.NDArray[np.float64]) -> None:
        self.values = values

    def __getitem__(self, index: int | None) -> "FakeTensor":
        return FakeTensor(self.values[index])

    def to(self, _device: str) -> "FakeTensor":
        return self

    def float(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> npt.NDArray[np.float64]:
        return self.values


class NoGradient:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


def fake_libraries(monkeypatch: pytest.MonkeyPatch, *, mps: bool) -> dict[str, object]:
    loaded: dict[str, object] = {}

    def from_pretrained(path: str, **options: object) -> FakeModel:
        loaded.update(path=path, options=options)
        return FakeModel()

    torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        no_grad=NoGradient,
        from_numpy=FakeTensor,
        log_softmax=lambda tensor, dim: FakeTensor(np.log(np.full(tensor.values.shape, 1 / 3))),
    )
    transformers = types.SimpleNamespace(
        Wav2Vec2ForCTC=types.SimpleNamespace(from_pretrained=from_pretrained)
    )
    modules = {"torch": torch, "transformers": transformers}
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    return loaded


def test_the_model_emits_frame_log_probabilities_from_local_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = fake_libraries(monkeypatch, mps=True)
    emissions = speech_onsets.load_emissions(tmp_path, "mps")
    values = emissions(np.zeros(640, dtype=np.float32))
    assert values.shape == (2, 3)
    assert values[0, 0] == pytest.approx(math.log(1 / 3))
    assert loaded == {"path": str(tmp_path),
                      "options": {"local_files_only": True, "use_safetensors": True}}  # fmt: skip


def test_an_unavailable_mps_device_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_libraries(monkeypatch, mps=False)
    with pytest.raises(ReplayError, match="MPS device is unavailable"):
        speech_onsets.load_emissions(tmp_path, "mps")


def test_the_voice_track_decodes_to_mono_16_khz_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_options: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        data = np.array([0.5, -0.5], dtype=np.float32).tobytes()
        return subprocess.CompletedProcess(command, 0, stdout=data)

    monkeypatch.setattr(subprocess, "run", run)
    assert speech_onsets.decode(Path("v.wav")).tolist() == [0.5, -0.5]
    assert calls == [["ffmpeg", "-nostdin", "-v", "error", "-i", "v.wav", "-vn", "-ac", "1",
                      "-ar", "16000", "-f", "f32le", "-"]]  # fmt: skip


def test_a_voice_track_without_audio_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command: list[str], **_options: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, stdout=b"")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ReplayError, match="no audio"):
        speech_onsets.decode(Path("v.wav"))


def test_the_cli_writes_per_character_tokens_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = model_directory(tmp_path)
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps(document(
        {"text": "あ", "start": 1.0, "end": 1.4, "utterance": "u:0"}
    )), encoding="utf-8")  # fmt: skip
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"pcm")
    monkeypatch.setattr(speech_onsets, "decode", lambda _path: np.zeros(48_000, np.float32))
    monkeypatch.setattr(
        speech_onsets, "load_emissions",
        lambda _directory, _device: lambda samples: emission_rows(samples.size // 320, {35: 1}),
    )  # fmt: skip
    output = tmp_path / "onsets.json"
    arguments = [str(tokens), str(audio), str(output), "--model", str(directory)]
    assert speech_onsets.main(arguments) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["clock"] == "output"
    assert written["counts"] == {"windows": 1, "unalignedWindows": 0, "characters": 1,
                                 "measuredCharacters": 1}  # fmt: skip
    assert written["tokens"][0]["start"] == pytest.approx(1.2)
    assert written["inputs"]["audio"] == (
        "0747302d3e634b7e8e70be88b368c7350f5a863b79c11dfeab93c0bc5b01116c"
    )
    assert capsys.readouterr().out.startswith('{"windows": 1')
    assert not (tmp_path / ".onsets.json.partial").exists()
    with pytest.raises(ReplayError, match="overwrite"):
        speech_onsets.main(arguments)
