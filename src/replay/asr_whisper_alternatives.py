"""Whisper's close second choices, replayed from the stored context-stream chunk records.

Each chunk record holds the tokens large-v3 decoded for its audio slice. Teacher-forcing those
tokens through the same model, audio window and prompt (``asr_context_streams.worker``: the
initial prompt only on the first 30 s window, timestamps on, fp16) reproduces the decoder's
per-position log-probabilities: 99.96% of 71,975 text positions on three recordings had the
recorded token as the argmax (2026-10-24). Where an alternative text token lies within ``MARGIN``
nats of the recorded one, the alternative is forced and decoding continues greedily until it
reproduces ``RESYNC`` recorded tokens; the replaced span is the candidate, with both paths'
log-probabilities. Windows whose replay disagrees with the record (a repaired or re-decoded
chunk) yield no candidates.

Whisper's output is never changed here; ``replay.asr_kanji_fix`` decides what, if anything, to
edit.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from replay.asr_context_streams import read_pcm
from replay.core import ReplayError
from replay.recordings import record

MARGIN = 3.0  # nats between the recorded token and an alternative
TOP = 4  # alternatives examined per position (the recorded token included)
MAX_STEPS = 8
RESYNC = 2  # recorded tokens the alternative path must reproduce to count as re-joined
MAX_JOIN = 14  # tokens after the alternative within which it must re-join
MIN_AGREEMENT = 0.97  # replayed argmax == recorded token, per window, to trust its log-probs
RATE = 16000
# mlx has no type stubs; the recognizer's arrays and caches are opaque to this module.
Opaque = Any
Rows = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Candidate:
    start: int  # token positions in the segment's tokens
    end: int
    tokens: tuple[int, ...]
    text: str  # the whole segment's text with the alternative
    margin: float
    original_log_probability: float  # recorded span plus the re-join tokens
    alternative_log_probability: float


@dataclass(frozen=True, slots=True)
class Segment:
    chunk: str
    index: int
    start: float  # seconds on the input clock
    end: float
    text: str
    candidates: tuple[Candidate, ...]


class Recognizer(Protocol):
    eot: int
    timestamp_begin: int
    sot_prev: int
    sot_sequence: tuple[int, ...]
    max_prompt: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: Sequence[int]) -> str: ...
    def features(self, audio: NDArray[np.float32], seek: int) -> Opaque: ...
    def forward(self, tokens: Sequence[int], features: Opaque) -> tuple[Rows, Opaque]: ...
    def step(self, token: int, features: Opaque, cache: Opaque) -> tuple[Rows, Opaque]: ...
    def truncated(self, cache: Opaque, length: int) -> Opaque: ...


class MlxRecognizer:
    """large-v3 on MLX (deferred imports: the orchestrator and tests skip MLX)."""

    def __init__(self, model: Path) -> None:
        self.mx = importlib.import_module("mlx.core")
        loader = importlib.import_module("mlx_whisper.load_models")
        self.audio = importlib.import_module("mlx_whisper.audio")
        tokenizers = importlib.import_module("mlx_whisper.tokenizer")
        self.model = loader.load_model(str(model), dtype=self.mx.float16)
        self.tokenizer = tokenizers.get_tokenizer(
            True, num_languages=self.model.num_languages, language="ja", task="transcribe"
        )
        self.eot = int(self.tokenizer.eot)
        self.timestamp_begin = int(self.tokenizer.timestamp_begin)
        self.sot_prev = int(self.tokenizer.sot_prev)
        self.sot_sequence = tuple(int(token) for token in self.tokenizer.sot_sequence)
        self.max_prompt = int(self.model.dims.n_text_ctx) // 2 - 1

    def encode(self, text: str) -> list[int]:
        return [int(token) for token in self.tokenizer.encode(text)]

    def decode(self, tokens: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(tokens)))

    def features(self, audio: NDArray[np.float32], seek: int) -> Opaque:
        padding = int(self.audio.N_SAMPLES)
        frames = int(self.audio.N_FRAMES)
        mel = self.audio.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels, padding=padding)
        content = mel.shape[-2] - frames
        window = mel[seek : seek + min(frames, content - seek)]
        return self.model.embed_audio(
            self.audio.pad_or_trim(window, frames, axis=-2).astype(self.mx.float16)[None]
        )

    def _rows(self, logits: Opaque) -> Rows:
        values = logits.astype(self.mx.float32)
        table = values - self.mx.logsumexp(values, axis=-1, keepdims=True)
        return np.asarray(table, dtype=np.float64)

    def forward(self, tokens: Sequence[int], features: Opaque) -> tuple[Rows, Opaque]:
        logits, cache, _ = self.model.decoder(self.mx.array([list(tokens)]), features)
        return self._rows(logits[0]), cache

    def step(self, token: int, features: Opaque, cache: Opaque) -> tuple[Rows, Opaque]:
        logits, cache, _ = self.model.decoder(self.mx.array([[token]]), features, kv_cache=cache)
        return self._rows(logits[0, -1]), cache

    def truncated(self, cache: Opaque, length: int) -> Opaque:
        """The self-attention cache of the first ``length`` positions (causal, so exact)."""
        return [((keys[:, :length], values[:, :length]), cross) for (keys, values), cross in cache]

    def release(self) -> None:
        self.mx.clear_cache()


def _windows(result: Mapping[str, object]) -> list[tuple[int, list[dict[str, object]]]]:
    """The record's segments grouped by the 30 s window (``seek``) they were decoded in."""
    groups: list[tuple[int, list[dict[str, object]]]] = []
    segments = result.get("segments")
    if not isinstance(segments, list):
        raise ReplayError("A chunk record needs a segments array")
    for segment in segments:
        row = record(segment)
        seek = row.get("seek")
        if not isinstance(seek, int):
            raise ReplayError("A chunk segment needs its seek")
        _tokens(row)
        if not groups or groups[-1][0] != seek:
            groups.append((seek, []))
        groups[-1][1].append(row)
    return groups


def _tokens(row: Mapping[str, object]) -> list[int]:
    tokens = row.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise ReplayError("A chunk segment's tokens must be integers")
    return [int(token) for token in tokens]


def _prompt(result: Mapping[str, object]) -> str:
    """The initial prompt of the decode the record kept (a guarded retry dropped it)."""
    guard = result.get("guard")
    retried = isinstance(guard, dict) and guard.get("retry")
    prompt = result.get("prompt")
    return "" if retried or not isinstance(prompt, str) else prompt


def _rejoined(
    recognizer: Recognizer,
    features: Opaque,
    cache: Opaque,
    alternative: int,
    tokens: Sequence[int],
    position: int,
    first: float,
) -> tuple[list[int], list[float], int] | None:
    """Greedy continuation after a forced alternative until ``RESYNC`` recorded tokens recur."""
    generated, probabilities, token = [alternative], [first], alternative
    for _ in range(MAX_STEPS):
        row, cache = recognizer.step(token, features, cache)
        row[recognizer.eot + 1 : recognizer.timestamp_begin] = -np.inf  # no other specials
        token = int(np.argmax(row))
        generated.append(token)
        probabilities.append(float(row[token]))
        if len(generated) > RESYNC:
            tail = generated[-RESYNC:]
            for joined in range(position + 1, min(position + MAX_JOIN, len(tokens) - RESYNC + 1)):
                if list(tokens[joined : joined + RESYNC]) == tail:
                    return generated[:-RESYNC], probabilities, joined
        if token == recognizer.eot:
            return None
    return None


def _segment_candidates(
    recognizer: Recognizer,
    features: Opaque,
    table: Rows,
    cache: Opaque,
    base: int,
    tokens: Sequence[int],
    span: range,
) -> tuple[list[Candidate], int, int]:
    """Candidates inside one segment, plus (text positions, argmax agreements)."""
    found: list[Candidate] = []
    positions = agreements = 0
    recorded = [float(table[base + at - 1, token]) for at, token in enumerate(tokens)]
    for position in span:
        token = tokens[position]
        if token >= recognizer.eot:
            continue
        row = table[base + position - 1, : recognizer.eot]
        order = [int(value) for value in np.argsort(-row)[:TOP]]
        positions += 1
        agreements += order[0] == token
        for alternative in order:
            if alternative == token or row[alternative] < row[token] - MARGIN:
                continue
            prefix = recognizer.truncated(cache, base + position)
            first = float(table[base + position - 1, alternative])
            joined = _rejoined(recognizer, features, prefix, alternative, tokens, position, first)
            if joined is None:
                continue
            replacement, probabilities, rejoin = joined
            if rejoin > span.stop or any(
                value >= recognizer.eot for value in (*replacement, *tokens[position:rejoin])
            ):
                continue  # timestamps change, or the span leaves the segment
            lo, hi = position - span.start, rejoin - span.start
            local = list(tokens[span.start : span.stop])
            changed = local[:lo] + replacement + local[hi:]
            found.append(
                Candidate(
                    lo,
                    hi,
                    tuple(replacement),
                    recognizer.decode([value for value in changed if value < recognizer.eot]),
                    float(row[token] - row[alternative]),
                    sum(recorded[position : rejoin + RESYNC]),
                    sum(probabilities),
                )
            )
    return found, positions, agreements


def chunk_segments(
    recognizer: Recognizer, name: str, result: Mapping[str, object], samples: NDArray[np.int16]
) -> tuple[list[Segment], dict[str, int]]:
    """Every segment of one chunk record with its candidates, and the replay's agreement."""
    span = record(result.get("chunk"))
    start, end = int(str(span["start"])), int(str(span["end"]))
    audio = samples[start:end].astype(np.float32) / 32768.0
    trusted = "repair" not in result
    segments: list[Segment] = []
    stats = {"positions": 0, "agreements": 0, "skippedWindows": 0}
    for window, (seek, rows) in enumerate(_windows(result)):
        features = recognizer.features(audio, seek)
        initial = list(recognizer.sot_sequence)
        prompt = _prompt(result)
        if window == 0 and prompt:
            encoded = recognizer.encode(" " + prompt.strip())
            initial = [recognizer.sot_prev, *encoded[-recognizer.max_prompt :], *initial]
        tokens = [token for row in rows for token in _tokens(row)]
        table, cache = recognizer.forward([*initial, *tokens], features)
        cursor, positions, agreements = 0, 0, 0
        window_segments: list[Segment] = []
        for row in rows:
            count = len(_tokens(row))
            found, seen, agreed = _segment_candidates(
                recognizer,
                features,
                table,
                cache,
                len(initial),
                tokens,
                range(cursor, cursor + count),
            )
            positions, agreements = positions + seen, agreements + agreed
            text = recognizer.decode(
                [token for token in tokens[cursor : cursor + count] if token < recognizer.eot]
            )
            window_segments.append(
                Segment(
                    name,
                    int(str(row.get("id", len(segments) + len(window_segments)))),
                    start / RATE + float(str(row["start"])),
                    start / RATE + float(str(row["end"])),
                    text,
                    tuple(found),
                )
            )
            cursor += count
        stats["positions"] += positions
        stats["agreements"] += agreements
        if not trusted or (positions and agreements / positions < MIN_AGREEMENT):
            stats["skippedWindows"] += 1
            window_segments = [
                Segment(item.chunk, item.index, item.start, item.end, item.text, ())
                for item in window_segments
            ]
        segments.extend(window_segments)
    return segments, stats


def records(store: Path) -> Iterator[tuple[str, dict[str, object]]]:
    """The store's chunk records in input order (file stems are zero-padded sample offsets)."""
    for path in sorted(store.glob("[0-9]*.json")):
        yield path.stem, record(json.loads(path.read_text(encoding="utf-8")))


def replay(recognizer: Recognizer, store: Path, mono: Path) -> tuple[list[Segment], dict[str, int]]:
    """All segments of a recording with their candidates."""
    samples = read_pcm(mono)
    segments: list[Segment] = []
    totals = {"positions": 0, "agreements": 0, "skippedWindows": 0, "chunks": 0}
    for name, result in records(store):
        found, stats = chunk_segments(recognizer, name, result, samples)
        segments.extend(found)
        totals["chunks"] += 1
        for key, value in stats.items():
            totals[key] += value
        release = getattr(recognizer, "release", None)
        if callable(release):
            release()
    return segments, totals
