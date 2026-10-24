"""Whisper close alternatives from replayed chunk records, with a scripted recognizer."""

import importlib
import json
import wave
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import asr_whisper_alternatives as alternatives
from replay.core import ReplayError

EOT, SOT_PREV, SOT, TS = 10, 11, 12, 13  # text tokens are 0-9, spelled "abcdefghij"
VOCABULARY = 20


class Scripted:
    """Recorded tokens win by 10 nats except where ``close`` names an alternative 1 nat behind;
    greedy steps follow ``after`` (the next token for each forced or generated one)."""

    eot: int = EOT
    timestamp_begin: int = TS
    sot_prev: int = SOT_PREV
    sot_sequence: tuple[int, ...] = (SOT,)
    max_prompt: int = 3

    def __init__(self, close: dict[int, int], after: dict[int, int], wrong: int = -1) -> None:
        self.close, self.after, self.wrong = close, after, wrong
        self.forwarded: list[list[int]] = []
        self.seeks: list[int] = []
        self.released = 0

    def encode(self, text: str) -> list[int]:
        return [ord(character) % 10 for character in text.strip()]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join("abcdefghij"[token] for token in tokens)

    def features(self, audio: NDArray[np.float32], seek: int) -> object:
        self.seeks.append(seek)
        return ("features", seek)

    def forward(
        self, tokens: Sequence[int], features: object
    ) -> tuple[NDArray[np.float64], object]:
        self.forwarded.append(list(tokens))
        table = np.full((len(tokens), VOCABULARY), -20.0)
        for position, token in enumerate(tokens[1:], 1):
            table[position - 1, token] = 0.0
            alternative = self.close.get(position)
            if alternative is not None:
                table[position - 1, alternative] = -1.0
            if position == self.wrong:
                table[position - 1, (token + 1) % 10] = 1.0
        return table, len(tokens)

    def step(
        self, token: int, features: object, cache: object
    ) -> tuple[NDArray[np.float64], object]:
        row = np.full(VOCABULARY, -5.0)
        row[self.after.get(token, EOT)] = -0.5
        return row, cache

    def truncated(self, cache: object, length: int) -> object:
        return length

    def release(self) -> None:
        self.released += 1


def chunk(segments: list[dict[str, object]], **extra: object) -> dict[str, object]:
    return {"chunk": {"start": 16000, "end": 32000}, "segments": segments, **extra}


def segment(seek: int, tokens: list[int], start: float = 0.5, index: int = 0) -> dict[str, object]:
    return {"id": index, "seek": seek, "start": start, "end": start + 1.0, "tokens": tokens}


SAMPLES = np.zeros(48000, dtype=np.int16)


def test_close_alternative_that_rejoins_is_a_candidate() -> None:
    # tokens: ts b c d e ts ; position 2 (c) has f close; f -> d -> e re-joins the record.
    recognizer = Scripted(close={3: 5}, after={5: 3, 3: 4})
    found, stats = alternatives.chunk_segments(
        recognizer, "000000016000", chunk([segment(0, [TS, 1, 2, 3, 4, TS])]), SAMPLES
    )
    assert found == [
        alternatives.Segment(
            "000000016000",
            0,
            1.5,
            2.5,
            "bcde",
            (alternatives.Candidate(2, 3, (5,), "bfde", 1.0, 0.0, -2.0),),
        )
    ]
    assert stats == {"positions": 4, "agreements": 4, "skippedWindows": 0}
    assert recognizer.forwarded == [[SOT, TS, 1, 2, 3, 4, TS]]


def test_alternative_that_never_rejoins_or_ends_is_dropped() -> None:
    looping = Scripted(close={3: 5}, after={5: 6, 6: 6})
    ending = Scripted(close={3: 5}, after={})
    record = chunk([segment(0, [TS, 1, 2, 3, 4, TS])])
    assert alternatives.chunk_segments(looping, "c", record, SAMPLES)[0][0].candidates == ()
    assert alternatives.chunk_segments(ending, "c", record, SAMPLES)[0][0].candidates == ()


def test_alternative_crossing_a_timestamp_is_dropped() -> None:
    # c -> f, then the path skips the closing timestamp and re-joins in the next segment.
    recognizer = Scripted(close={3: 5}, after={5: 1, 1: 2})
    record = chunk([segment(0, [TS, 1, 2, TS]), segment(0, [TS, 1, 2, TS], 1.5, 1)])
    found, _ = alternatives.chunk_segments(recognizer, "c", record, SAMPLES)
    assert [item.candidates for item in found] == [(), ()]


def test_first_window_gets_the_kept_prompt_and_later_windows_none() -> None:
    recognizer = Scripted(close={}, after={})
    record = chunk([segment(0, [TS, 1, TS]), segment(3000, [TS, 2, TS], 1.0, 1)], prompt="abcd")
    alternatives.chunk_segments(recognizer, "c", record, SAMPLES)
    assert recognizer.forwarded == [[SOT_PREV, 8, 9, 0, SOT, TS, 1, TS], [SOT, TS, 2, TS]]
    assert recognizer.seeks == [0, 3000]


def test_guarded_retry_dropped_the_prompt() -> None:
    recognizer = Scripted(close={}, after={})
    record = chunk([segment(0, [TS, 1, TS])], prompt="ab", guard={"retry": "prompt-echo"})
    alternatives.chunk_segments(recognizer, "c", record, SAMPLES)
    assert recognizer.forwarded == [[SOT, TS, 1, TS]]


def test_window_whose_replay_disagrees_yields_no_candidates() -> None:
    recognizer = Scripted(close={3: 5}, after={5: 3, 3: 4}, wrong=2)
    found, stats = alternatives.chunk_segments(
        recognizer, "c", chunk([segment(0, [TS, 1, 2, 3, 4, TS])]), SAMPLES
    )
    assert found[0].candidates == ()
    assert stats == {"positions": 4, "agreements": 3, "skippedWindows": 1}


def test_repaired_chunk_yields_no_candidates() -> None:
    recognizer = Scripted(close={3: 5}, after={5: 3, 3: 4})
    record = chunk([segment(0, [TS, 1, 2, 3, 4, TS])], repair={"method": "window"})
    found, stats = alternatives.chunk_segments(recognizer, "c", record, SAMPLES)
    assert (found[0].candidates, stats["skippedWindows"]) == ((), 1)


@pytest.mark.parametrize(
    "record",
    [
        {"chunk": {"start": 0, "end": 1}},
        chunk([{"seek": "0", "tokens": [1]}]),
        chunk([{"seek": 0, "tokens": ["1"]}]),
    ],
)
def test_malformed_records_are_rejected(record: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        alternatives.chunk_segments(Scripted({}, {}), "c", record, SAMPLES)


def write_mono(path: Path) -> Path:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.zeros(48000, dtype="<i2").tobytes())
    return path


def test_replay_reads_every_numbered_record_in_order(tmp_path: Path) -> None:
    store = tmp_path / "chunks"
    store.mkdir()
    (store / "000000016000.json").write_text(json.dumps(chunk([segment(0, [TS, 1, TS])])))
    (store / "000000000000.json").write_text(
        json.dumps({**chunk([segment(0, [TS, 2, TS])]), "chunk": {"start": 0, "end": 16000}})
    )
    (store / "previous-inputs.manifest").write_text("x")
    recognizer = Scripted({}, {})
    found, totals = alternatives.replay(recognizer, store, write_mono(tmp_path / "mono.wav"))
    assert [(item.chunk, item.text, item.start) for item in found] == [
        ("000000000000", "c", 0.5),
        ("000000016000", "b", 1.5),
    ]
    assert totals == {"positions": 2, "agreements": 2, "skippedWindows": 0, "chunks": 2}
    assert recognizer.released == 2


class FakeModel:
    num_languages = 100
    dims = SimpleNamespace(n_text_ctx=448, n_mels=128)

    def embed_audio(self, mel: NDArray[np.float32]) -> NDArray[np.float32]:
        return mel

    def decoder(
        self, tokens: NDArray[np.int64], features: object, kv_cache: object = None
    ) -> tuple[NDArray[np.float32], object, None]:
        length = tokens.shape[1]
        logits = np.zeros((1, length, 4), dtype=np.float32)
        logits[..., 1] = np.log(3.0)
        cache = [((np.ones((1, length, 2)), np.ones((1, length, 2))), "cross")]
        return logits, cache, None


def fake_modules() -> dict[str, object]:
    tokenizer = SimpleNamespace(
        eot=10,
        timestamp_begin=13,
        sot_prev=11,
        sot_sequence=[12, 5, 6],
        encode=lambda text: [1, 2],
        decode=lambda tokens: "x" * len(tokens),
    )
    return {
        "mlx.core": SimpleNamespace(
            float16=np.float16,
            float32=np.float32,
            array=np.array,
            logsumexp=lambda values, axis, keepdims: np.log(
                np.sum(np.exp(values), axis=axis, keepdims=keepdims)
            ),
            clear_cache=lambda: None,
        ),
        "mlx_whisper.load_models": SimpleNamespace(load_model=lambda path, dtype: FakeModel()),
        "mlx_whisper.audio": SimpleNamespace(
            N_SAMPLES=8,
            N_FRAMES=4,
            log_mel_spectrogram=lambda audio, n_mels, padding: np.zeros((len(audio) + padding, 2)),
            pad_or_trim=lambda window, frames, axis: np.zeros((frames, 2)),
        ),
        "mlx_whisper.tokenizer": SimpleNamespace(get_tokenizer=lambda *args, **kwargs: tokenizer),
    }


def test_mlx_recognizer_adapts_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = fake_modules()
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    recognizer = alternatives.MlxRecognizer(Path("model"))
    assert (
        recognizer.eot,
        recognizer.timestamp_begin,
        recognizer.sot_prev,
        recognizer.sot_sequence,
        recognizer.max_prompt,
    ) == (10, 13, 11, (12, 5, 6), 223)
    assert (recognizer.encode("ab"), recognizer.decode([1, 2, 3])) == ([1, 2], "xxx")
    features = recognizer.features(np.zeros(6, dtype=np.float32), 0)
    assert features.shape == (1, 4, 2)
    table, cache = recognizer.forward([1, 2, 3], features)
    assert table.shape == (3, 4)
    assert table[0] == pytest.approx(np.log([1 / 6, 3 / 6, 1 / 6, 1 / 6]))
    row, _ = recognizer.step(1, features, recognizer.truncated(cache, 2))
    assert row == pytest.approx(np.log([1 / 6, 3 / 6, 1 / 6, 1 / 6]))
    assert recognizer.truncated(cache, 2)[0][0][0].shape == (1, 2, 2)
    recognizer.release()
