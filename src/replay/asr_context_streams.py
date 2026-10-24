"""Context-aware interleaved recognition (after arXiv 2608.31170) with the pinned MLX Whisper.

Instead of fixed overlapping windows decoded one by one without context, the recognition input
is cut at pauses into speech chunks of at most ``MAX_CHUNK_SECONDS``, split into ``streams``
contiguous streams, and decoded in rounds: round m decodes the m-th chunk of every stream at
once, one persistent worker (model loaded once) per stream. Each stream carries its own decoded
text as the next chunk's previous-text prompt (MLX Whisper keeps its last 223 tokens). The first
chunk of every stream but the first is decoded again at the end with the preceding stream's
final text, so no chunk after the very first lacks context.

Guards, all recorded per chunk: a chunk without voice energy is not decoded (Whisper would echo
the prompt); a prompted result whose segments repeat (compression ratio above 2.4), are
improbable (average log-probability below -1.0) or restate the prompt's end is decoded again
without the prompt; a result still failing those checks never becomes context.

    python -m replay.asr_context_streams run MONO.wav MODEL_DIR OUTPUT [--streams B] [--no-context]
    python -m replay.asr_context_streams worker MONO.wav MODEL_DIR     (one JSON line per chunk)
    python -m replay.asr_context_streams publish MONO.wav MODEL_DIR WORK LABELS MANIFEST
    python -m replay.asr_context_streams mono VOCALS.wav FFMPEG MONO.wav RECEIPT

``mono`` decodes the separated vocals to the recognition input exactly as the windowed
recognizer's mono stage does (``replay_transcription_plan.mono_command``: no stretching, padding
or trimming), checks that the duration is unchanged and writes a completion receipt binding it.

``publish`` runs with context on one stream and writes the recognition labels and the input
manifest the clock projection checks (clock ``input-pcm-not-projected``, the PCM digest, its
sample count and rate), as the windowed recognizer's assembly does.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
import wave
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay import asr_decode_patches, caption_io, caption_linguistics
from replay.core import ReplayError
from replay.recordings import record
from replay.replay_transcription_plan import mono_command
from replay.stages import file_digest

RATE = 16000
FRAME = 480  # 30 ms energy frames
MAX_CHUNK_SECONDS = 30.0
MIN_PAUSE_SECONDS = 0.4  # a pause at least this long may end a chunk
PAD_SECONDS = 0.15
SPEECH_ABOVE_FLOOR_DB = 15.0  # speech frames rise this far above the recording's noise floor
FLOOR_PERCENTILE = 10.0
DIGITAL_SILENCE_DB = -100.0  # separated vocals contain exact zeros; they are not the noise floor
COMPRESSION_LIMIT = 2.4  # Whisper's own repetition threshold
LOGPROB_LIMIT = -1.0
ECHO_CHARACTERS = 8
# Whisper closes a chunk with a phrase it never heard (ご視聴ありがとうございました): a short last
# segment it itself rates as probably silent. Such a tail is dropped; a rare real short word at a
# chunk end may go with it.
TAIL_SECONDS = 1.0
# A word clock longer than this is clamped to its loudest voiced run (Whisper's word alignment
# can stretch a word over the following pause); clocks only shrink, never move outside the word.
LONG_WORD_SECONDS = 2.0
# Whisper places word onsets about 0.2 s before the voice (measured on 10-14b: median -0.23 s
# against the audio). A word that starts before any voice starts at its first voiced frame; the
# softer onset margin keeps quiet consonants. Clocks only move forward inside the word.
ONSET_ABOVE_FLOOR_DB = 10.0
# A moved start keeps one caption display frame before the next word's start, so two words that
# precede the same voice onset never collapse onto one frame (captions show at 30 fps).
ONSET_SPACING_SECONDS = 1 / 30
TAIL_NO_SPEECH = 0.5
PATCHED_MODULES = ("decoding", "timing", "whisper")  # output-identical decode speedups
# Whisper token clocks may leave a pause inside one morphological word (ス | マホ, ゆ | っくり):
# the earlier fragment sits at an earlier, unrelated voice. Captions reset at any pause longer
# than their hold, so such a word would split across pages. Fragments before the last pause in a
# word move up against the rest of it (never earlier). A shorter gap (often one onset snapping
# opened) closes by extending the earlier fragment: moving it would open a pause before it.
WORD_MOVE_PAUSE_SECONDS = 0.25  # caption_timing.CaptionPolicy.maximum_gap
FRAGMENT_SECONDS_PER_CHARACTER = 0.15  # a moved fragment keeps at most this spoken length
# A pause that is mostly voice holds an unrecognized word; one this long separates utterances
# whatever the analysis of the unpunctuated chunk text joins (おいしょ, 12 s, おとぎ → しょお).
WORD_PAUSE_VOICED = 0.5
WORD_PAUSE_LIMIT_SECONDS = 8.0
# A chunk still looping after the unprompted retry is decoded again from the segment where its
# text turns repetitive, then sampled at rising temperatures (seeded before every call, so a
# re-run repeats it), and otherwise loses that text as unrecognized. It keeps resetting the
# stream context, so the chunks after it are decoded exactly as before the repair.
REPAIR_TEMPERATURES = (0.2, 0.4, 0.6, 0.8, 1.0)
REPAIR_SEED = 0
MIN_REPAIR_SAMPLES = RATE // 10  # a window shorter than this holds no word to recover
STORE_KEY = "inputs.key"  # what the stored chunks were decoded from (not a chunk: no .json)
PREVIOUS_MANIFEST = "previous-inputs.manifest"  # a published manifest kept for legacy stores
DECODE_OPTIONS: dict[str, object] = {
    "language": "ja", "verbose": False, "condition_on_previous_text": False,
    "word_timestamps": True, "temperature": 0.0, "fp16": True,
}  # fmt: skip

Samples = NDArray[np.int16]


@dataclass(frozen=True, slots=True)
class Chunk:
    start: int  # samples
    end: int

    @property
    def name(self) -> str:
        return f"{self.start:012d}"


def read_pcm(path: Path) -> Samples:
    """Mono PCM16 at 16 kHz, the recognition input contract."""
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (RATE, 1, 2):
            raise ReplayError("Recognition input must be mono PCM16 at 16 kHz")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)


def frame_levels(samples: Samples) -> NDArray[np.float64]:
    """Each 30 ms frame's RMS level in dBFS."""
    count = len(samples) // FRAME
    frames = samples[: count * FRAME].astype(np.float64).reshape(count, FRAME)
    rms = np.sqrt(np.mean(frames * frames, axis=1)) / 32768.0
    return np.asarray(20 * np.log10(np.maximum(rms, 1e-7)))


def speech_frames(
    levels: NDArray[np.float64], margin: float = SPEECH_ABOVE_FLOOR_DB
) -> NDArray[np.bool_]:
    """Frames above the recording's own noise floor by ``margin`` dB.

    The floor is a low percentile of the frames that are not digital silence.
    """
    audible = levels[levels > DIGITAL_SILENCE_DB]
    if not len(audible):
        return np.zeros(len(levels), dtype=bool)
    floor = float(np.percentile(audible, FLOOR_PERCENTILE))
    return np.asarray(levels > floor + margin)


def regions(speech: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Speech frame runs, joined across pauses shorter than MIN_PAUSE_SECONDS (frame indices)."""
    edges = np.diff(np.concatenate(([0], speech.astype(np.int8), [0])))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    joined: list[tuple[int, int]] = []
    pause = round(MIN_PAUSE_SECONDS * RATE / FRAME)
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if joined and start - joined[-1][1] < pause:
            joined[-1] = (joined[-1][0], end)
        else:
            joined.append((start, end))
    return joined


def split_long(start: int, end: int, levels: NDArray[np.float64]) -> list[tuple[int, int]]:
    """A region longer than a chunk, cut at its quietest frame in each chunk's second half."""
    limit = int(MAX_CHUNK_SECONDS * RATE / FRAME)
    parts = []
    while end - start > limit:
        low, high = start + limit // 2, start + limit
        cut = low + int(np.argmin(levels[low:high]))
        parts.append((start, cut))
        start = cut
    parts.append((start, end))
    return parts


def speech_chunks(samples: Samples, levels: NDArray[np.float64] | None = None) -> list[Chunk]:
    """Speech regions packed into chunks of at most MAX_CHUNK_SECONDS, cut only at pauses."""
    levels = frame_levels(samples) if levels is None else levels
    limit = int(MAX_CHUNK_SECONDS * RATE / FRAME)
    pad = round(PAD_SECONDS * RATE / FRAME)
    packed: list[tuple[int, int]] = []
    for start, end in regions(speech_frames(levels)):
        for part in split_long(start, end, levels):
            if packed and part[1] - packed[-1][0] <= limit - 2 * pad:
                packed[-1] = (packed[-1][0], part[1])
            else:
                packed.append(part)
    chunks = []
    for index, (start, end) in enumerate(packed):
        low = max(start - pad, packed[index - 1][1] if index else 0)
        high = min(end + pad, packed[index + 1][0] if index + 1 < len(packed) else len(levels))
        chunks.append(Chunk(low * FRAME, min(high * FRAME, len(samples))))
    return chunks


def streams_of(chunks: Sequence[Chunk], count: int) -> list[list[Chunk]]:
    """Contiguous streams of near-equal length (the paper's stream partition)."""
    size = math.ceil(len(chunks) / max(1, count)) if chunks else 0
    return [list(chunks[i : i + size]) for i in range(0, len(chunks), size)] if size else []


def text_of(result: Mapping[str, object]) -> str:
    return str(result.get("text", "")).strip()


def failing(result: Mapping[str, object], prompt: str) -> str | None:
    """Why a decoded chunk must not be trusted, or None."""
    segments = result.get("segments")
    for value in segments if isinstance(segments, list) else []:
        segment = record(value)
        ratio, logprob = segment.get("compression_ratio"), segment.get("avg_logprob")
        if isinstance(ratio, (int, float)) and ratio > COMPRESSION_LIMIT:
            return "repetition"
        if isinstance(logprob, (int, float)) and logprob < LOGPROB_LIMIT:
            return "improbable"
    tail = prompt.strip()[-ECHO_CHARACTERS:]
    if len(tail) == ECHO_CHARACTERS and tail in text_of(result):
        return "prompt-echo"
    return None


def trimmed(result: Mapping[str, object]) -> tuple[dict[str, object], str | None]:
    """The result without a phantom last segment (short and rated silent), and what was cut."""
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return dict(result), None
    last = record(segments[-1])
    start, end, silent = last.get("start"), last.get("end"), last.get("no_speech_prob")
    numbers = all(isinstance(v, (int, float)) for v in (start, end, silent))
    if not numbers or float(str(end)) - float(str(start)) >= TAIL_SECONDS:
        return dict(result), None
    if float(str(silent)) < TAIL_NO_SPEECH:
        return dict(result), None
    kept = segments[:-1]
    text = "".join(str(record(segment).get("text", "")) for segment in kept)
    return {**result, "segments": kept, "text": text}, str(last.get("text", ""))


Decode = Callable[[Chunk, str], dict[str, object]]
Redecode = Callable[[Chunk, str, float], dict[str, object]]


def guarded(
    chunk: Chunk, prompt: str, first: Mapping[str, object], decode: Decode
) -> dict[str, object]:
    """A decoded chunk with its guard record: retried without the prompt when untrusted."""
    reason = failing(first, prompt) if prompt else None
    result = decode(chunk, "") if reason is not None else first
    result, tail = trimmed(result)
    guard = {"prompted": bool(prompt), "retry": reason, "droppedTail": tail,
             "trusted": failing(result, "") is None}  # fmt: skip
    return {**result, "guard": guard, "prompt": prompt,
            "chunk": {"start": chunk.start, "end": chunk.end}}  # fmt: skip


def compression_ratio(text: str) -> float:
    """Whisper's repetition measure: UTF-8 length over its zlib length."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def segments_of(result: Mapping[str, object]) -> list[dict[str, object]]:
    segments = result.get("segments")
    return [record(segment) for segment in segments] if isinstance(segments, list) else []


def looping(result: Mapping[str, object]) -> bool:
    """Whether any decoded segment is repetitive (an improbable-only chunk is not)."""
    ratios = (segment.get("compression_ratio") for segment in segments_of(result))
    return any(isinstance(r, (int, float)) and r > COMPRESSION_LIMIT for r in ratios)


def loop_start(segments: Sequence[Mapping[str, object]]) -> int:
    """The first segment whose text makes the chunk's text so far repetitive."""
    text = ""
    for index, segment in enumerate(segments):
        text += str(segment.get("text", ""))
        if compression_ratio(text) > COMPRESSION_LIMIT:
            return index
    return len(segments)


def shifted(segment: Mapping[str, object], seconds: float) -> dict[str, object]:
    """A segment of a later window on the chunk clock."""
    moved = {**segment, "start": float(str(segment["start"])) + seconds,
             "end": float(str(segment["end"])) + seconds}  # fmt: skip
    words = segment.get("words")
    if isinstance(words, list):
        moved["words"] = [
            {
                **record(word),
                "start": float(str(record(word)["start"])) + seconds,
                "end": float(str(record(word)["end"])) + seconds,
            }  # fmt: skip
            for word in words
        ]
    return moved


def repaired(result: Mapping[str, object], chunk: Chunk, decode: Redecode) -> dict[str, object]:
    """``result`` with a still-repetitive tail re-decoded or dropped; others unchanged."""
    guard = record(result.get("guard"))
    if guard.get("trusted") or "repair" in result or not looping(result):
        return dict(result)
    segments = segments_of(result)
    index = loop_start(segments) % len(segments)  # no repetitive prefix: the whole chunk
    since = float(str(segments[index]["start"]))
    window = Chunk(min(chunk.end, chunk.start + round(since * RATE)), chunk.end)
    # The whole chunk at temperature 0 without a prompt is the guard's own retry.
    attempts = ((0.0,) if index else ()) + REPAIR_TEMPERATURES
    method, temperature, tail, added = "dropped", None, None, []
    for value in attempts if window.end - window.start >= MIN_REPAIR_SAMPLES else ():
        candidate, cut = trimmed(decode(window, "", value))
        if not looping(candidate):
            method = "subwindow" if value == 0.0 else "temperature"
            temperature, tail = value, cut
            offset = (window.start - chunk.start) / RATE
            added = [shifted(segment, offset) for segment in segments_of(candidate)]
            break
    kept = segments[:index]
    # The kept segments' decode measured its whole text; re-measure the text that stays.
    ratio = compression_ratio("".join(str(segment.get("text", "")) for segment in kept))
    kept = [{**segment, "compression_ratio": ratio} for segment in kept]
    final = kept + added
    repair = {"method": method, "fromSeconds": since, "temperature": temperature,
              "replaced": "".join(str(segment.get("text", "")) for segment in segments[index:]),
              "droppedTail": tail, "seed": REPAIR_SEED}  # fmt: skip
    text = "".join(str(segment.get("text", "")) for segment in final)
    return {**result, "segments": final, "text": text, "repair": repair}


def store_key(audio: Path, model: Path) -> dict[str, object]:
    """What a stored chunk result depends on: input PCM, weights, options and the provider."""
    weights = sorted(path for path in model.iterdir() if path.is_file()) if model.is_dir() else []
    provider = {name: metadata.version(name) for name in ("mlx", "mlx-whisper")}
    return {"audio": file_digest(audio), "model": {p.name: file_digest(p) for p in weights},
            "options": DECODE_OPTIONS, "provider": provider,
            "patches": file_digest(Path(asr_decode_patches.__file__))}  # fmt: skip


def adopted(store: Path, key: Mapping[str, object], audio: Path, model: Path) -> bool:
    """Whether ``store`` holds results of ``key``'s inputs: keyed, or a legacy store whose
    published manifest bound the same PCM and weights. An unbound store is never reused."""
    keyed, previous = store / STORE_KEY, store / PREVIOUS_MANIFEST
    if keyed.exists():
        return bool(json.loads(keyed.read_text(encoding="utf-8")) == key)
    if not previous.exists():
        return False
    inputs = record(record(json.loads(previous.read_text(encoding="utf-8"))).get("inputs"))
    weights = record(key["model"])
    return (
        inputs.get(str(audio)) == key["audio"]
        and bool(weights)
        and all(inputs.get(str(model / name)) == digest for name, digest in weights.items())
    )


def carried(context: str, result: Mapping[str, object], *, enabled: bool) -> str:
    """The stream's context after a chunk: its text appended, or reset by an untrusted result."""
    if not enabled:
        return ""
    if not record(result.get("guard")).get("trusted"):
        return ""
    return (context + text_of(result))[-600:]  # MLX keeps the last 223 tokens of it


def interleaved(
    streams: Sequence[Sequence[Chunk]],
    decode_round: Callable[[list[tuple[int, Chunk, str]]], list[dict[str, object]]],
    *,
    context: bool,
    done: Callable[[Chunk], Mapping[str, object] | None] = lambda _chunk: None,
) -> dict[str, dict[str, object]]:
    """Every chunk's result, decoding round m's chunks together and carrying stream contexts."""
    texts = ["" for _ in streams]
    results: dict[str, dict[str, object]] = {}
    for m in range(max((len(stream) for stream in streams), default=0)):
        todo: list[tuple[int, Chunk, str]] = []
        for index, stream in enumerate(streams):
            if m >= len(stream):
                continue
            chunk = stream[m]
            previous = done(chunk)
            if previous is not None:
                results[chunk.name] = dict(previous)
                texts[index] = carried(texts[index], previous, enabled=context)
            else:
                todo.append((index, chunk, texts[index] if context else ""))
        for (index, chunk, _prompt), result in zip(todo, decode_round(todo), strict=True):
            results[chunk.name] = result
            texts[index] = carried(texts[index], result, enabled=context)
    if context:  # cold start: stream i's first chunk again, with stream i-1's final context
        todo = [(i, s[0], texts[i - 1]) for i, s in enumerate(streams) if i and s and texts[i - 1]]
        for (_index, chunk, _prompt), result in zip(todo, decode_round(todo), strict=True):
            results[chunk.name] = {**result, "coldStart": True}
    return results


class Worker:
    """A persistent recognition process for one stream: the model loads once."""

    def __init__(self, audio: Path, model: Path) -> None:
        command = [sys.executable, "-m", "replay.asr_context_streams", "worker", str(audio),
                   str(model)]  # fmt: skip
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8"
        )

    def send(self, chunk: Chunk, prompt: str, temperature: float = 0.0) -> None:
        if self.process.stdin is None:
            raise ReplayError("The recognition worker has no input pipe")
        line = {
            "start": chunk.start,
            "end": chunk.end,
            "prompt": prompt,
            "temperature": temperature,
        }
        self.process.stdin.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def decode(self, chunk: Chunk, prompt: str, temperature: float = 0.0) -> dict[str, object]:
        self.send(chunk, prompt, temperature)
        return self.receive()

    def receive(self) -> dict[str, object]:
        if self.process.stdout is None:
            raise ReplayError("The recognition worker has no output pipe")
        line = self.process.stdout.readline()
        if not line:
            raise ReplayError("The recognition worker stopped")
        return record(json.loads(line))

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.wait(timeout=120) != 0:
            raise ReplayError("The recognition worker failed")


def worker(audio: Path, model: Path) -> None:
    """Decode chunks from stdin (JSON lines) with the pinned options; one result line each."""
    provider = importlib.import_module("mlx_whisper")  # deferred: orchestrator and tests skip MLX
    mlx = importlib.import_module("mlx.core")
    stock = [importlib.import_module(f"mlx_whisper.{name}") for name in PATCHED_MODULES]
    asr_decode_patches.install(mlx, *stock, version=metadata.version("mlx-whisper"))
    samples = read_pcm(audio)
    for line in sys.stdin:
        request = record(json.loads(line))
        start, end = int(str(request["start"])), int(str(request["end"]))
        prompt = str(request.get("prompt") or "")
        temperature = float(str(request.get("temperature") or 0.0))
        audio_chunk = samples[start:end].astype(np.float32) / 32768.0
        options = {**DECODE_OPTIONS, "initial_prompt": prompt or None}
        if temperature:  # seeded per call: the sample does not depend on earlier requests
            mlx.random.seed(REPAIR_SEED)
            options["temperature"] = temperature
        result = provider.transcribe(audio_chunk, path_or_hf_repo=str(model), **options)
        # MLX keeps freed buffers cached per chunk shape: measured +0.35 GiB per chunk, 32 GB
        # after a long recording. Releasing them each chunk keeps the worker at its model size.
        mlx.clear_cache()
        sys.stdout.write(json.dumps(result, ensure_ascii=False, default=float) + "\n")
        sys.stdout.flush()


def labels(
    results: Mapping[str, Mapping[str, object]], rate: int = RATE
) -> list[dict[str, object]]:
    """Recognition labels on the input PCM clock, provenance ``<chunk start>:<group>``."""
    rows: list[dict[str, object]] = []
    for name in sorted(results):
        result = results[name]
        span = record(result.get("chunk"))
        start, end = int(str(span["start"])), int(str(span["end"]))
        if not text_of(result):
            continue
        groups = caption_io.mlx_groups(result, duration=(end - start) / rate)
        offset = start / rate
        for index, group in enumerate(groups):
            rows.append({"text": group.text, "start": offset + group.start,
                         "end": offset + group.end, "provenance": f"{name}:{index}"})  # fmt: skip
    return rows


def onset_snapped(
    rows: Sequence[dict[str, object]], voice: NDArray[np.bool_]
) -> tuple[list[dict[str, object]], int]:
    """Labels whose start precedes the voice begin at their first voiced frame (inside the word)."""
    result, count = [], 0
    for index, row in enumerate(rows):
        start, end = float(str(row["start"])), float(str(row["end"]))
        first, last = int(start * RATE / FRAME), math.ceil(end * RATE / FRAME)
        silent_start = first < len(voice) and not voice[first]
        voiced = np.nonzero(voice[first:last])[0] if silent_start else np.zeros(0, dtype=np.intp)
        onset = (first + int(voiced[0])) * FRAME / RATE if len(voiced) else start
        if index + 1 < len(rows):
            onset = min(onset, float(str(rows[index + 1]["start"])) - ONSET_SPACING_SECONDS)
        if onset <= start or onset >= end:
            result.append(row)
            continue
        result.append({**row, "start": onset, "onsetSnap": {"originalStart": start}})
        count += 1
    return result, count


def tightened(
    rows: Sequence[dict[str, object]], speech: NDArray[np.bool_]
) -> tuple[list[dict[str, object]], int]:
    """Labels whose long clocks are clamped to the voiced run inside them with the most speech."""
    result, count = [], 0
    for row in rows:
        start, end = float(str(row["start"])), float(str(row["end"]))
        first, last = int(start * RATE / FRAME), math.ceil(end * RATE / FRAME)
        voiced = regions(speech[first:last]) if end - start > LONG_WORD_SECONDS else []
        if not voiced:
            result.append(row)
            continue
        low, high = max(voiced, key=lambda run: run[1] - run[0])
        clamped = (
            max(start, (first + low) * FRAME / RATE),
            min(end, (first + high) * FRAME / RATE),
        )
        result.append({**row, "start": clamped[0], "end": clamped[1],
                       "voicedClamp": {"originalStart": start, "originalEnd": end}})  # fmt: skip
        count += 1
    return result, count


def words_joined(
    rows: Sequence[dict[str, object]], voice: NDArray[np.bool_]
) -> tuple[list[dict[str, object]], int]:
    """Labels with no pause inside a morphological word of their chunk; the count of moved."""
    result = [dict(row) for row in rows]
    count, first = 0, 0
    while first < len(result):
        last = first + 1
        while last < len(result) and _chunk_of(result[last]) == _chunk_of(result[first]):
            last += 1
        count += _joined_chunk(result[first:last], voice)
        first = last
    return result, count


def _chunk_of(row: Mapping[str, object]) -> str:
    return str(row["provenance"]).split(":")[0]


def _clock(row: Mapping[str, object], key: str) -> float:
    return float(str(row[key]))


def _bridgeable(
    left: Mapping[str, object], right: Mapping[str, object], voice: NDArray[np.bool_]
) -> bool:
    """A pause one word may span: short enough, and quiet (voice there is an unrecognized word)."""
    end, start = _clock(left, "end"), _clock(right, "start")
    first, last = int(end * RATE / FRAME), math.ceil(start * RATE / FRAME)
    quiet = last <= first or float(voice[first:last].mean()) <= WORD_PAUSE_VOICED
    return start - end <= WORD_PAUSE_LIMIT_SECONDS and quiet


def _joined_chunk(rows: list[dict[str, object]], voice: NDArray[np.bool_]) -> int:
    ends = frozenset(caption_linguistics.boundaries("".join(str(row["text"]) for row in rows))[1])
    offset, count, word = 0, 0, [rows[0]]
    for left, right in itertools.pairwise(rows):
        offset += len(str(left["text"]))
        if offset not in ends and _bridgeable(left, right, voice):
            word.append(right)
            continue
        count += _joined_word(word)
        word = [right]
    return count + _joined_word(word)


def _joined_word(word: list[dict[str, object]]) -> int:
    pauses = [
        index
        for index in range(1, len(word))
        if _clock(word[index], "start") - _clock(word[index - 1], "end") > WORD_MOVE_PAUSE_SECONDS
    ]
    block = word[: pauses[-1]] if pauses else []
    if block:
        spoken = sum(_clock(row, "end") - _clock(row, "start") for row in block)
        characters = sum(len(str(row["text"])) for row in block)
        scale = min(1.0, characters * FRAGMENT_SECONDS_PER_CHARACTER / spoken) if spoken else 0.0
        # Every fragment moves later: the block ended before the pause.
        cursor = _clock(word[pauses[-1]], "start") - spoken * scale
        for row in block:
            length = (_clock(row, "end") - _clock(row, "start")) * scale
            row["wordJoin"] = {"originalStart": row["start"], "originalEnd": row["end"]}
            row["start"], row["end"] = cursor, cursor + length
            cursor += length
    for left, right in itertools.pairwise(word):
        left["end"] = max(_clock(left, "end"), _clock(right, "start"))
    return len(block)


def run(
    audio: Path, model: Path, output: Path, *, streams: int, context: bool
) -> dict[str, object]:
    """Recognize ``audio`` into ``output`` (resumable per chunk); returns the run summary."""
    samples = read_pcm(audio)
    levels = frame_levels(samples)  # once: chunking and the clock corrections share it
    chunks = speech_chunks(samples, levels)
    planned = streams_of(chunks, streams)
    store = output / "chunks"
    key = store_key(audio, model)
    if store.exists() and not adopted(store, key, audio, model):
        shutil.rmtree(store)  # results of other inputs (or unbound ones) are never reused
    store.mkdir(parents=True, exist_ok=True)
    (store / STORE_KEY).write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    workers = [Worker(audio, model) for _ in planned]

    def kept(chunk: Chunk, result: dict[str, object]) -> dict[str, object]:
        path = store / f"{chunk.name}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
        return result

    def done(chunk: Chunk) -> dict[str, object] | None:
        path = store / f"{chunk.name}.json"
        if not path.exists():
            return None
        stored = record(json.loads(path.read_text(encoding="utf-8")))
        fixed = repaired(stored, chunk, workers[0].decode)  # stored before the repair existed
        return kept(chunk, fixed) if fixed != stored else stored

    started = time.monotonic()

    def decode_round(todo: list[tuple[int, Chunk, str]]) -> list[dict[str, object]]:
        for index, chunk, prompt in todo:  # every stream decodes its chunk at the same time
            workers[index].send(chunk, prompt)
        first = [workers[index].receive() for index, _chunk, _prompt in todo]
        results = []
        for (index, chunk, prompt), result in zip(todo, first, strict=True):
            final = guarded(chunk, prompt, result, workers[index].decode)
            results.append(kept(chunk, repaired(final, chunk, workers[index].decode)))
        return results

    try:
        results = interleaved(planned, decode_round, context=context, done=done)
    finally:
        for item in workers:
            item.close()
    rows, clamped = tightened(labels(results), speech_frames(levels))
    rows, snapped = onset_snapped(rows, speech_frames(levels, ONSET_ABOVE_FLOOR_DB))
    rows, joined = words_joined(rows, speech_frames(levels, ONSET_ABOVE_FLOOR_DB))
    guards = [record(result.get("guard")) for result in results.values()]
    repairs = [record(r["repair"]) for r in results.values() if "repair" in r]
    summary: dict[str, object] = {
        "version": 1,
        "method": "context-aware-interleaved-streams (arXiv 2608.31170)",
        "context": context,
        "streams": len(planned),
        "chunks": len(chunks),
        "speechSeconds": sum(c.end - c.start for c in chunks) / RATE,
        "audioSeconds": len(samples) / RATE,
        "wallSeconds": round(time.monotonic() - started, 1),
        "retries": sum(1 for guard in guards if guard.get("retry")),
        "untrusted": sum(1 for guard in guards if not guard.get("trusted")),
        "droppedTails": sum(1 for guard in guards if guard.get("droppedTail")),
        "repairs": {
            method: sum(1 for repair in repairs if repair.get("method") == method)
            for method in ("subwindow", "temperature", "dropped")
        },
        "voicedClamps": clamped,
        "onsetSnaps": snapped,
        "wordJoins": joined,
        "options": DECODE_OPTIONS,
    }
    (output / "labels.json").write_text(
        json.dumps({"summary": summary, "labels": rows}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return summary


def prepare_mono(vocals: Path, ffmpeg: Path, mono: Path, receipt: Path) -> dict[str, object]:
    """The recognition input decoded from the vocals, with its completion receipt."""
    if receipt.exists():
        return record(json.loads(receipt.read_text(encoding="utf-8")))
    mono.unlink(missing_ok=True)  # a decode without its receipt was interrupted
    command = list(mono_command(vocals, mono, ffmpeg))
    subprocess.run(command, check=True)
    with wave.open(str(vocals), "rb") as source:
        seconds = source.getnframes() / source.getframerate()
    samples = len(read_pcm(mono))
    if abs(seconds - samples / RATE) > 1 / RATE:
        raise ReplayError("Mono decoding changed the continuous sample duration")
    document: dict[str, object] = {
        "version": 1, "status": "complete", "producer": "replay.asr_context_streams mono",
        "command": command,
        "inputs": {str(vocals): file_digest(vocals), str(ffmpeg): file_digest(ffmpeg)},
        "outputs": {str(mono): file_digest(mono)}, "samples": samples, "sampleRate": RATE,
        "clock": "input-pcm-not-projected",
    }  # fmt: skip
    receipt.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return document


def publish(
    audio: Path, model: Path, work: Path, labels_path: Path, manifest: Path
) -> dict[str, object]:
    """Recognize ``audio`` (context, one stream) and write its manifest and bound labels."""
    if labels_path.exists() or manifest.exists():
        raise ReplayError("Refusing to overwrite recognition labels or their manifest")
    summary = run(audio, model, work, streams=1, context=True)
    recognized = record(json.loads((work / "labels.json").read_text(encoding="utf-8")))
    code = [Path(__file__), Path(caption_io.__file__), Path(caption_linguistics.__file__),
            Path(asr_decode_patches.__file__)]  # fmt: skip
    weights = sorted(path for path in model.iterdir() if path.is_file())
    with wave.open(str(audio), "rb") as stream:
        samples = stream.getnframes()
    inputs = {str(path): file_digest(path) for path in (audio, *weights, *code)}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "version": 1, "clock": "input-pcm-not-projected", "inputs": inputs,
        "samples": samples, "sampleRate": RATE, "providerOptions": DECODE_OPTIONS,
        "method": summary["method"],
    }, indent=1) + "\n", encoding="utf-8")  # fmt: skip
    labels_path.write_text(json.dumps({
        "version": 1, "clock": "input-pcm-not-projected", "labels": recognized["labels"],
        "parentSHA256": {str(manifest): file_digest(manifest)}, "recognition": summary,
        "humanReviewed": False, "speechAccuracyVerified": False, "deliveryComplete": False,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")  # fmt: skip
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker", "publish", "mono"))
    parser.add_argument("audio", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("labels", type=Path, nargs="?")
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "worker":
        worker(args.audio, args.model)
        return 0
    if args.output is None:
        raise ReplayError("run needs an output directory")
    if args.mode == "mono":  # audio=vocals, model=ffmpeg, output=mono, labels=receipt
        if args.labels is None:
            raise ReplayError("mono needs a receipt path")
        prepare_mono(args.audio, args.model, args.output, args.labels)
        return 0
    if args.mode == "publish":
        if args.labels is None or args.manifest is None:
            raise ReplayError("publish needs labels and manifest paths")
        summary = publish(args.audio, args.model, args.output, args.labels, args.manifest)
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    summary = run(args.audio, args.model, args.output, streams=args.streams,
                  context=not args.no_context)  # fmt: skip
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
