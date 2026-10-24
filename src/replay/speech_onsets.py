"""Time each recognized character by when it is voiced, on the output clock.

Recognizer token times are coarse: a token usually starts where the previous one ended, before
its sound, and a token of several characters gives no time for its later characters. This aligns
the known token text to the delivered voice track with the pinned Japanese wav2vec2 CTC model
(Viterbi forced alignment over its 20 ms frames) and writes one token per character:

- ``start``: the first CTC frame of the character's vocabulary piece when that piece has strong
  acoustic support (mean log-probability at least ``WEAK_LOG_PROBABILITY``), labelled
  ``"timing": "ctc"``; otherwise the recognizer's time (the token's interval split evenly),
  labelled ``"timing": "asr"``;
- ``anchor``: the recognizer's start of the token the character came from, unchanged, so a
  consumer keeps assigning characters to the captions paging built from those tokens.

Starts never decrease. Unvoiced symbols, spaces, characters outside the vocabulary and windows
without a valid alignment keep the recognizer's time. Nothing here changes text or claims speech
accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from replay.core import ReplayError
from replay.mps_graph_cache import released_mps_graphs

SAMPLE_RATE = 16_000
FRAME_SECONDS = 320 / SAMPLE_RATE  # the model's convolution stride: 5 * 2**6 samples
BLANK = 0
WEAK_LOG_PROBABILITY = -4.0
# No acoustic event: never aligned (mirrors asr_ctc_label_refine).
UNVOICED = frozenset("。、，．！？!?…・「」『』（）()〜~ー")
# A window ends at a pause this long or at this span, and reaches this far around its speech.
PAUSE_SECONDS, WINDOW_SECONDS, PAD_SECONDS = 1.5, 20.0, 0.5
# A late voice can sit this far past its recognized end; the window reaches that far when free.
LATE_SECONDS = 1.5
# Audio lengths are padded up to a multiple of this so the device compiles few input shapes.
BUCKET_SECONDS = 2.0
NEGATIVE = -1e30
CHUNK_BYTES = 1 << 20

Emissions = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: float
    end: float
    utterance: str


@dataclass(frozen=True, slots=True)
class Piece:
    """A vocabulary entry covering ``text[first:last]`` of a window's joined text."""

    first: int
    last: int
    identifier: int


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError("Invalid speech token")
    return float(value)


def _token(item: object) -> Token:
    if not isinstance(item, dict):
        raise ReplayError("Invalid speech token")
    text, utterance = item.get("text"), item.get("utterance")
    if not isinstance(text, str) or not text or not isinstance(utterance, str):
        raise ReplayError("Invalid speech token")
    start, end = _number(item.get("start")), _number(item.get("end"))
    if not 0 <= start <= end:
        raise ReplayError("Invalid speech token interval")
    return Token(text, start, end, utterance)


def read_tokens(document: object) -> list[Token]:
    if not isinstance(document, dict) or document.get("clock") != "output":
        raise ReplayError("Expected output-clock speech tokens")
    entries = document.get("tokens")
    if not isinstance(entries, list) or not entries:
        raise ReplayError("The speech tokens are not a nonempty list")
    tokens = [_token(item) for item in entries]
    if any(
        following.start < token.start for token, following in zip(tokens, tokens[1:], strict=False)
    ):
        raise ReplayError("Speech tokens must be ordered by start")
    return tokens


def windows(tokens: Sequence[Token]) -> list[range]:
    """Consecutive token ranges split at long pauses and bounded in span."""
    result: list[range] = []
    first = 0
    for index in range(1, len(tokens)):
        pause = tokens[index].start - tokens[index - 1].end
        if pause >= PAUSE_SECONDS or tokens[index].end - tokens[first].start > WINDOW_SECONDS:
            result.append(range(first, index))
            first = index
    result.append(range(first, len(tokens)))
    return result


def pieces(text: str, vocabulary: dict[str, int]) -> list[Piece]:
    """Greedy longest vocabulary match; unvoiced and unknown characters are left out."""
    longest = max(len(entry) for entry in vocabulary)
    result: list[Piece] = []
    index = 0
    while index < len(text):
        if text[index].isspace() or text[index] in UNVOICED:
            index += 1
            continue
        for size in range(min(longest, len(text) - index), 0, -1):
            identifier = vocabulary.get(text[index : index + size])
            if (
                identifier is not None
                and identifier != BLANK
                and not UNVOICED & set(text[index : index + size])
            ):
                result.append(Piece(index, index + size, identifier))
                index += size
                break
        else:
            index += 1
    return result


def forced_align(
    log_probabilities: npt.NDArray[np.float64], targets: Sequence[int]
) -> list[tuple[int, float]]:
    """CTC Viterbi alignment: each target's first frame and mean log-probability."""
    frames = log_probabilities.shape[0]
    repeats = sum(1 for a, b in zip(targets, targets[1:], strict=False) if a == b)
    if not targets or frames < len(targets) + repeats:
        raise ValueError("The audio is too short for the text")
    labels = np.full(2 * len(targets) + 1, BLANK, dtype=np.int64)
    labels[1::2] = targets
    states = labels.size
    skippable = np.zeros(states, dtype=bool)
    skippable[3::2] = labels[3::2] != labels[1:-2:2]
    score = np.full(states, NEGATIVE)
    score[:2] = log_probabilities[0, labels[:2]]
    back = np.zeros((frames, states), dtype=np.int8)
    for frame in range(1, frames):
        step = np.concatenate(([NEGATIVE], score[:-1]))
        skip = np.where(skippable, np.concatenate(([NEGATIVE, NEGATIVE], score[:-2])), NEGATIVE)
        stacked = np.stack((score, step, skip))
        back[frame] = np.argmax(stacked, axis=0)
        score = stacked[back[frame], np.arange(states)] + log_probabilities[frame, labels]
    state = states - 1 if score[-1] >= score[-2] else states - 2
    if score[state] <= NEGATIVE / 2:
        raise ValueError("No CTC path covers the text")
    path = np.zeros(frames, dtype=np.int64)
    for frame in range(frames - 1, -1, -1):
        path[frame] = state
        state -= int(back[frame, state])
    result = []
    for index, target in enumerate(targets):
        spoken = np.flatnonzero(path == 2 * index + 1)
        result.append((int(spoken[0]), float(np.mean(log_probabilities[spoken, target]))))
    return result


def recognized_times(tokens: Sequence[Token]) -> list[float]:
    """Each character's recognizer time: its token's interval split evenly."""
    return [
        token.start + (token.end - token.start) * index / len(token.text)
        for token in tokens
        for index in range(len(token.text))
    ]


def measured(
    tokens: Sequence[Token],
    begin: float,
    log_probabilities: npt.NDArray[np.float64],
    vocabulary: dict[str, int],
) -> dict[int, float]:
    """Strongly supported character onsets of one window, keyed by character index."""
    text = "".join(token.text for token in tokens)
    found = pieces(text, vocabulary)
    if not found:
        return {}
    try:
        spans = forced_align(log_probabilities, [piece.identifier for piece in found])
    except ValueError:
        return {}
    return {
        index: begin + frame * FRAME_SECONDS
        for piece, (frame, mean) in zip(found, spans, strict=True)
        if mean >= WEAK_LOG_PROBABILITY
        for index in range(piece.first, piece.last)
    }


def characters(tokens: Sequence[Token], onsets: dict[int, float]) -> list[dict[str, object]]:
    """One output token per character; ``onsets`` are measured times by character index."""
    times = recognized_times(tokens)
    starts: list[float] = []
    for index, time in enumerate(times):
        starts.append(max(onsets.get(index, time), starts[-1] if starts else 0.0))
    result: list[dict[str, object]] = []
    index = 0
    for token in tokens:
        for offset, character in enumerate(token.text):
            last = offset + 1 == len(token.text)
            end = token.end if last else starts[index + 1]
            result.append({
                "text": character, "start": starts[index], "end": max(end, starts[index]),
                "anchor": token.start, "utterance": token.utterance,
                "timing": "ctc" if index in onsets else "asr",
            })  # fmt: skip
            index += 1
    return result


def bounds(tokens: Sequence[Token], window: range, duration: float) -> tuple[float, float]:
    """The audio a window aligns against: its speech, padded, never into the next window."""
    begin = max(0.0, tokens[window.start].start - PAD_SECONDS)
    following = tokens[window.stop].start if window.stop < len(tokens) else duration
    last = tokens[window.stop - 1].end
    end = min(duration, max(last + PAD_SECONDS, min(following, last + LATE_SECONDS)))
    return begin, max(end, begin + FRAME_SECONDS)


def bucketed(samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normalized samples, zero-padded up to the next bucket length."""
    normalized = (samples - samples.mean()) / np.sqrt(samples.var() + 1e-7)
    bucket = int(BUCKET_SECONDS * SAMPLE_RATE)
    padded = np.zeros(-(-samples.size // bucket) * bucket, dtype=np.float32)
    padded[: samples.size] = normalized
    return padded


def align(
    tokens: Sequence[Token],
    audio: npt.NDArray[np.float32],
    emissions: Emissions,
    vocabulary: dict[str, int],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Per-character tokens for the whole recording, and alignment counts."""
    duration = audio.size / SAMPLE_RATE
    onsets: dict[int, float] = {}
    offset, unaligned = 0, 0
    for window in windows(tokens):
        group = tokens[window.start : window.stop]
        begin, end = bounds(tokens, window, duration)
        samples = audio[int(begin * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        frames = int(samples.size / SAMPLE_RATE / FRAME_SECONDS)
        found = measured(group, begin, emissions(bucketed(samples))[:frames], vocabulary)
        unaligned += not found
        onsets.update({offset + index: time for index, time in found.items()})
        offset += sum(len(token.text) for token in group)
    result = characters(tokens, onsets)
    counts = {
        "windows": len(windows(tokens)),
        "unalignedWindows": unaligned,
        "characters": len(result),
        "measuredCharacters": len(onsets),
    }
    return result, counts


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def verified_model(directory: Path) -> dict[str, str]:
    """The model files' digests, checked against the acquisition record."""
    record = json.loads((directory / "ACQUISITION-RECORD.json").read_text(encoding="utf-8"))
    files = record.get("files") if isinstance(record, dict) else None
    if not isinstance(files, dict):
        raise ReplayError("The CTC model has no acquisition record")
    result = {}
    for name in ("config.json", "vocab.json", "model.safetensors"):
        expected = files.get(name)
        actual = digest(directory / name)
        if not isinstance(expected, dict) or expected.get("sha256") != actual:
            raise ReplayError(f"The CTC model file {name} does not match its acquisition record")
        result[name] = actual
    return result


def load_emissions(directory: Path, device: str) -> Emissions:
    """The pinned model's frame log-probabilities, from local files only."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ReplayError("The MPS device is unavailable; no automatic fallback is allowed")
    model = transformers.Wav2Vec2ForCTC.from_pretrained(
        str(directory), local_files_only=True, use_safetensors=True
    )
    model = model.to(device).eval()

    def emissions(samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        with torch.no_grad():
            logits = model(torch.from_numpy(samples)[None].to(device)).logits[0]
            values = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        return np.asarray(values, dtype=np.float64)

    return emissions


def decode(path: Path) -> npt.NDArray[np.float32]:
    """The whole voice track as 16 kHz mono float samples."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
               "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]  # fmt: skip
    data = subprocess.run(command, capture_output=True, check=True).stdout
    if not data:
        raise ReplayError("The voice track decoded to no audio")
    return np.frombuffer(data, dtype=np.float32)


def write_new(path: Path, value: object) -> None:
    """Write ``value`` beside ``path`` and move it into place, so a present file is complete."""
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)
    with partial.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    os.link(partial, path)
    partial.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time recognized characters by CTC alignment")
    parser.add_argument("tokens", type=Path, help="output-clock speech tokens")
    parser.add_argument("audio", type=Path, help="the output-clock voice track")
    parser.add_argument("output", type=Path, help="new per-character tokens")
    parser.add_argument("--model", type=Path, required=True, help="pinned wav2vec2 CTC model")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        raise ReplayError(f"Refusing to overwrite {arguments.output}")
    tokens = read_tokens(json.loads(arguments.tokens.read_text(encoding="utf-8")))
    files = verified_model(arguments.model)
    vocabulary = json.loads((arguments.model / "vocab.json").read_text(encoding="utf-8"))
    audio = decode(arguments.audio)
    with released_mps_graphs():
        emissions = load_emissions(arguments.model, arguments.device)
        result, counts = align(tokens, audio, emissions, vocabulary)
    write_new(arguments.output, {
        "version": 1, "clock": "output", "timing": "ctc-first-frame",
        "model": {"path": str(arguments.model), "sha256": files},
        "inputs": {"tokens": digest(arguments.tokens), "audio": digest(arguments.audio)},
        "counts": counts, "tokens": result,
    })  # fmt: skip
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
