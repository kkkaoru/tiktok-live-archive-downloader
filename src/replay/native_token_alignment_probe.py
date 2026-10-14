"""Forward-only native Whisper alignment of immutable, already-recognized tokens.

This diagnostic never generates text, re-tokenizes it, or subdivides a token's
clock. Native pre-heuristic intervals are evidence, not approved captions.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from replay import asr_adoption
from replay.asr_assembly import _windows
from replay.caption_io import mlx_groups
from replay.core import ReplayError
from replay.local_asr import fingerprint, pcm_audio
from replay.native_invocation import canonical, read, require, rows
from replay.recordings import record

MAX_BYTES = 16 * 1024**2
MAX_TOKENS = 4096
SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_FRAMES = 3000
N_SAMPLES = 480000
EOT = 50257


@dataclass(frozen=True)
class Reference:
    path: Path
    sha256: str


@dataclass(frozen=True)
class TokenWindow:
    seek: int
    tokens: tuple[int, ...]
    text: str


@dataclass(frozen=True)
class NativeWord:
    text: str
    tokens: tuple[int, ...]
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class Request:
    reference: Reference
    artifacts: dict[str, Reference]
    runtime: tuple[Reference, ...]
    model: Path
    stage: str
    start_sample: int
    sample_count: int
    windows: tuple[TokenWindow, ...]
    origin: dict[str, object] | None = None


class AlignmentBackend(Protocol):
    def align(
        self, samples: NDArray[np.float32], model: Path, windows: tuple[TokenWindow, ...]
    ) -> tuple[NativeWord, ...]: ...


def _text(value: object) -> str:
    require(isinstance(value, str) and bool(value) and "\0" not in value, "Invalid text")
    return str(value)


def _integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Expected nonnegative integer")
    return value


def _number(value: object) -> float:
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected finite number")
    return float(value)


def _reference(value: object) -> Reference:
    item = record(value)
    require(set(item) == {"path", "sha256"}, "Invalid reference fields")
    path, digest = canonical(item.get("path")), _text(item.get("sha256"))
    require(len(digest) == 64 and fingerprint(path) == digest, "Reference changed")
    return Reference(path, digest)


def _token_windows(raw: dict[str, object], *, samples: int) -> tuple[TokenWindow, ...]:
    mlx_groups(raw, duration=samples / SAMPLE_RATE)
    grouped: dict[int, tuple[list[int], list[str]]] = {}
    for value in rows(raw.get("segments")):
        segment = record(value)
        seek = _integer(segment.get("seek"))
        require(seek * HOP_LENGTH < samples, "Token seek exceeds actual PCM")
        tokens = tuple(_integer(value) for value in rows(segment.get("tokens")))
        require(len(tokens) <= MAX_TOKENS and all(token < 51866 for token in tokens), "Token quota")
        text_tokens = [token for token in tokens if token < EOT]
        require(bool(text_tokens), "Missing authentic text tokens")
        target, texts = grouped.setdefault(seek, ([], []))
        target.extend(text_tokens)
        texts.append(_text(segment.get("text")))
    require(0 < len(grouped) <= 60, "Invalid token window count")
    require(sum(len(tokens) for tokens, texts in grouped.values()) <= MAX_TOKENS, "Token quota")
    return tuple(
        TokenWindow(seek, tuple(tokens), "".join(texts))
        for seek, (tokens, texts) in grouped.items()
    )


def load_request(path: Path) -> Request:
    """Validate genuine producer/source/model evidence before importing any MLX code."""
    canonical(str(path))
    root = Path.cwd().resolve()
    require(path.is_relative_to(root / "edits/work"), "Request must be checkout-local")
    data = read(path)
    if data.get("version") == 2:
        return _load_retained_request(path, data)
    require(
        set(data) == {"version", "kind", "artifacts", "runtime", "model", "stage"}
        and data.get("version") == 1
        and data.get("kind") == "native-token-alignment-probe-request",
        "Invalid probe request",
    )
    refs = record(data.get("artifacts"))
    require(
        set(refs)
        == {
            "source",
            "raw",
            "inputs",
            "validation",
            "window",
            "workflow",
            "checkpoint",
            "pins",
            "end",
            "before",
            "after",
            "config",
            "weights",
        },
        "Missing genuine producer references",
    )
    artifacts = {name: _reference(value) for name, value in refs.items()}
    require(
        all(ref.path.is_relative_to(root) for ref in artifacts.values()),
        "Artifacts must be checkout-local",
    )
    runtime = tuple(_reference(value) for value in rows(data.get("runtime")))
    require(1 <= len(runtime) <= 64, "Invalid runtime closure")
    require(len({ref.path for ref in runtime}) == len(runtime), "Duplicate runtime pin")
    require(Path(__file__).resolve() in {ref.path for ref in runtime}, "Probe code must be pinned")
    model = canonical(data.get("model"))
    require(model.is_dir() and not model.is_symlink(), "Model must already exist locally")
    inputs = read(artifacts["inputs"].path)
    require(
        inputs.get("source") == str(artifacts["source"].path)
        and inputs.get("model") == str(model)
        and inputs.get("source_sha256") == artifacts["source"].sha256
        and inputs.get("config_sha256") == artifacts["config"].sha256
        and inputs.get("weights_sha256") == artifacts["weights"].sha256
        and artifacts["config"].path == model / "config.json"
        and artifacts["weights"].path.parent == model
        and artifacts["weights"].path.name in {"weights.npz", "weights.safetensors"},
        "ASR model/source pins differ",
    )
    before, after = read(artifacts["before"].path), read(artifacts["after"].path)
    require(
        after.get("hashes") == before and after.get("differences") == {}, "Producer source changed"
    )
    end = read(artifacts["end"].path)
    require(end.get("exitCode") == 0 and end.get("completed") is True, "Producer did not complete")
    stage = _text(data.get("stage"))
    require(artifacts["checkpoint"].path.name == stage + ".json", "Checkpoint stage mismatch")
    checkpoint = read(artifacts["checkpoint"].path)
    saved = record(checkpoint.get("outputs"))
    require(
        checkpoint.get("status") == "complete"
        and all(
            saved.get(str(artifacts[name].path)) == artifacts[name].sha256
            for name in ("raw", "inputs", "validation")
        ),
        "Completed producer output binding differs",
    )
    asr_adoption.verify(
        artifacts["workflow"].path,
        artifacts["checkpoint"].path.parent,
        stage,
        artifacts["raw"].path.parent,
        artifacts["pins"].path,
    )
    count = len(pcm_audio(artifacts["source"].path))
    window = read(artifacts["window"].path)
    start, stop = _integer(window.get("startSample")), _integer(window.get("endSample"))
    require(
        stop - start == count
        and window.get("sampleRate") == SAMPLE_RATE
        and window.get("windowSHA256") == artifacts["source"].sha256
        and window.get("clock") == "continuous-input-pcm",
        "Window clock differs from actual PCM",
    )
    return Request(
        Reference(path, fingerprint(path)),
        artifacts,
        runtime,
        model,
        stage,
        start,
        count,
        _token_windows(read(artifacts["raw"].path), samples=count),
    )


def _load_retained_request(path: Path, data: dict[str, object]) -> Request:
    """Bind a retained collection without inventing a historical local receipt."""
    require(
        set(data) == {"version", "kind", "artifacts", "runtime", "model", "origin"}
        and data.get("kind") == "native-token-alignment-probe-request",
        "Invalid retained probe request",
    )
    origin = record(data.get("origin"))
    require(
        set(origin) == {"kind", "segmentIndices", "historicalDtypeObserved", "computeDtype"}
        and origin.get("kind") == "retained-clocked-collection"
        and origin.get("historicalDtypeObserved") is False
        and origin.get("computeDtype") == "float16-provider-default",
        "Historical dtype must remain unobserved",
    )
    refs = record(data.get("artifacts"))
    require(
        set(refs)
        == {
            "source",
            "original",
            "raw",
            "inputs",
            "complete",
            "clock",
            "window",
            "config",
            "weights",
        },
        "Missing retained collection references",
    )
    artifacts = {name: _reference(value) for name, value in refs.items()}
    root = Path.cwd().resolve()
    require(
        all(ref.path.is_relative_to(root) for ref in artifacts.values()),
        "Artifacts must be checkout-local",
    )
    runtime = tuple(_reference(value) for value in rows(data.get("runtime")))
    require(
        1 <= len(runtime) <= 64
        and len({ref.path for ref in runtime}) == len(runtime)
        and Path(__file__).resolve() in {ref.path for ref in runtime},
        "Invalid retained runtime closure",
    )
    model = canonical(data.get("model"))
    require(
        model.is_dir()
        and not model.is_symlink()
        and artifacts["config"].path == model / "config.json"
        and artifacts["weights"].path.parent == model
        and artifacts["weights"].path.name in {"weights.npz", "weights.safetensors"},
        "Retained local model identity differs",
    )
    inputs = read(artifacts["inputs"].path)
    saved_inputs = record(inputs.get("inputs"))
    require(
        all(
            saved_inputs.get(str(artifacts[name].path)) == artifacts[name].sha256
            for name in ("original", "config", "weights")
        ),
        "Retained model/PCM pins differ",
    )
    owner_clock = read(artifacts["clock"].path)
    owners = _windows(inputs)
    matches = [owner for owner in owners if owner.start_sample == owner_clock.get("startSample")]
    complete = read(artifacts["complete"].path)
    hashes = rows(complete.get("rawSHA256"))
    require(
        len(matches) == 1
        and len(hashes) == len(owners)
        and complete.get("windows") == len(owners)
        and complete.get("deliveryComplete") is False,
        "Retained collection is incomplete",
    )
    owner = matches[0]
    require(
        hashes[owners.index(owner)] == artifacts["raw"].sha256
        and owner_clock
        == {
            "startSample": owner.start_sample,
            "endSample": owner.end_sample,
            "sampleRate": SAMPLE_RATE,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": artifacts["raw"].sha256,
        },
        "Retained raw owner clock differs",
    )
    raw = read(artifacts["raw"].path)
    mlx_groups(raw, duration=owner.sample_count / SAMPLE_RATE)
    segments = rows(raw.get("segments"))
    indices = tuple(_integer(value) for value in rows(origin.get("segmentIndices")))
    require(
        bool(indices)
        and len(set(indices)) == len(indices)
        and list(indices) == sorted(indices)
        and all(index < len(segments) for index in indices),
        "Invalid saved segment selection",
    )
    selected = [record(segments[index]) for index in indices]
    seeks = {_integer(segment.get("seek")) for segment in selected}
    require(len(seeks) == 1, "Selected segments require one authentic context")
    seek = next(iter(seeks))
    start = owner.start_sample + seek * HOP_LENGTH
    count = min(N_SAMPLES, owner.end_sample - start)
    window = read(artifacts["window"].path)
    require(
        0 < count <= N_SAMPLES
        and window.get("startSample") == start
        and window.get("endSample") == start + count
        and window.get("sampleRate") == SAMPLE_RATE
        and window.get("source") == str(artifacts["original"].path)
        and window.get("sourceSHA256") == artifacts["original"].sha256
        and window.get("windowSHA256") == artifacts["source"].sha256
        and window.get("clock") == "continuous-input-pcm"
        and window.get("sourceClockProjected") is False,
        "Crop must preserve the actual saved seek context",
    )
    with (
        wave.open(str(artifacts["original"].path), "rb") as original,
        wave.open(str(artifacts["source"].path), "rb") as crop,
    ):
        require(
            original.getparams()[:3] == crop.getparams()[:3] == (1, 2, SAMPLE_RATE)
            and original.getcomptype() == crop.getcomptype() == "NONE"
            and original.getnframes() == inputs.get("samples")
            and crop.getnframes() == count,
            "Retained PCM format differs",
        )
        original.setpos(start)
        require(
            original.readframes(count) == crop.readframes(count),
            "Crop bytes differ from original PCM",
        )
    require(
        all(
            len(rows(segment.get("tokens"))) <= MAX_TOKENS
            and all(_integer(value) < 51866 for value in rows(segment.get("tokens")))
            for segment in selected
        ),
        "Retained token quota",
    )
    windows = tuple(
        TokenWindow(
            0,
            tuple(
                _integer(value) for value in rows(segment.get("tokens")) if _integer(value) < EOT
            ),
            _text(segment.get("text")),
        )
        for segment in selected
    )
    require(
        all(window.tokens and len(window.tokens) <= MAX_TOKENS for window in windows)
        and sum(len(window.tokens) for window in windows) <= MAX_TOKENS,
        "Retained token quota",
    )
    tokens = tuple(token for window in windows for token in window.tokens)
    combined = TokenWindow(0, tokens, "".join(window.text for window in windows))
    proof = {
        **origin,
        "originalSeek": seek,
        "ownerStartSample": owner.start_sample,
        "ownerEndSample": owner.end_sample,
        "contextChanged": True,
        "contextUncertainty": "Crop log-mel FFT boundaries may differ from original mel slice",
        "historicalDeclaredDtypeContract": inputs.get("dtypeContract"),
    }
    return Request(
        Reference(path, fingerprint(path)),
        artifacts,
        runtime,
        model,
        "retained-collection",
        start,
        count,
        (combined,),
        proof,
    )


def _invoke(function: object, *args: object, **kwargs: object) -> object:
    if not callable(function):
        raise ReplayError("Installed API is not callable")
    result: object = function(*args, **kwargs)
    return result


def _attribute(value: object, name: str) -> object:
    result: object = getattr(value, name, None)
    return result


def _native_word(value: object, *, offset: float, duration: float) -> NativeWord:
    text = _text(getattr(value, "word", None))
    tokens = tuple(_integer(item) for item in rows(getattr(value, "tokens", None)))
    start = _number(getattr(value, "start", None)) + offset
    end = _number(getattr(value, "end", None)) + offset
    probability = _number(getattr(value, "probability", None))
    require(
        bool(tokens) and 0 <= start <= end <= duration and 0 <= probability <= 1,
        "Native alignment bounds",
    )
    return NativeWord(text, tokens, start, end, probability)


def _validate_words(words: tuple[NativeWord, ...], request: Request) -> None:
    require(bool(words), "Empty alignment is not speech absence proof")
    duration = request.sample_count / SAMPLE_RATE
    require(
        all(
            math.isfinite(word.start)
            and math.isfinite(word.end)
            and 0 <= word.start <= word.end <= duration
            and math.isfinite(word.probability)
            and 0 <= word.probability <= 1
            for word in words
        ),
        "Alignment clock/probability outside PCM",
    )
    require(
        all(right.start >= left.start for left, right in zip(words, words[1:], strict=False)),
        "Native alignment onset order regressed",
    )
    require(
        tuple(token for word in words for token in word.tokens)
        == tuple(token for window in request.windows for token in window.tokens),
        "Alignment token identity differs",
    )
    require(
        "".join(word.text for word in words) == "".join(window.text for window in request.windows),
        "Alignment text identity differs",
    )


class InstalledAlignment:
    """Use the installed fp16 loader and unmodified native DTW, never transcribe."""

    def align(
        self, samples: NDArray[np.float32], model: Path, windows: tuple[TokenWindow, ...]
    ) -> tuple[NativeWord, ...]:
        require(os.environ.get("HF_HUB_OFFLINE") == "1", "Alignment requires offline model loading")
        require(model.is_dir() and canonical(str(model)) == model, "Pinned local model disappeared")
        core = importlib.import_module("mlx.core")
        loader = importlib.import_module("mlx_whisper.load_models")
        audio = importlib.import_module("mlx_whisper.audio")
        timing = importlib.import_module("mlx_whisper.timing")
        tokenizer_module = importlib.import_module("mlx_whisper.tokenizer")
        native = _invoke(loader.load_model, str(model), dtype=core.float16)
        tokenizer = _invoke(
            tokenizer_module.get_tokenizer,
            _attribute(native, "is_multilingual"),
            num_languages=_attribute(native, "num_languages"),
            language="ja",
            task="transcribe",
        )
        mels = _integer(_attribute(_attribute(native, "dims"), "n_mels"))
        mel = _invoke(audio.log_mel_spectrogram, samples, n_mels=mels, padding=N_SAMPLES)
        result: list[NativeWord] = []
        for window in windows:
            decoded = _invoke(_attribute(tokenizer, "decode"), list(window.tokens))
            require(decoded == window.text, "Authentic token decode differs from raw text")
            frames = min(N_FRAMES, len(samples) // HOP_LENGTH - window.seek)
            require(frames > 0, "No native alignment frames")
            piece = _invoke(
                _attribute(mel, "__getitem__"), slice(window.seek, window.seek + frames)
            )
            padded = _invoke(audio.pad_or_trim, piece, N_FRAMES, axis=-2)
            typed = _invoke(_attribute(padded, "astype"), core.float16)
            aligned = _invoke(
                timing.find_alignment, native, tokenizer, list(window.tokens), typed, frames
            )
            require(isinstance(aligned, list), "Invalid native alignment response")
            if not isinstance(aligned, list):
                raise TypeError("Invalid native alignment response")
            words = tuple(
                _native_word(value, offset=window.seek / 100, duration=len(samples) / SAMPLE_RATE)
                for value in aligned
            )
            require(
                tuple(token for word in words for token in word.tokens) == window.tokens,
                "Alignment changed authentic tokens",
            )
            result.extend(words)
        return tuple(result)


def run(request: Path, output: Path, *, backend: AlignmentBackend | None = None) -> Path:
    """Publish one bounded diagnostic with immutable before/after provenance."""
    root = Path.cwd().resolve()
    canonical(str(output))
    require(
        output.is_relative_to(root / "edits/work")
        and output.parent.is_dir()
        and not output.exists(),
        "Expected new checkout-local evidence output",
    )
    parsed = load_request(request)
    references = (parsed.reference, *parsed.artifacts.values(), *parsed.runtime)
    samples = pcm_audio(parsed.artifacts["source"].path)
    selected = InstalledAlignment() if backend is None else backend
    words = selected.align(samples, parsed.model, parsed.windows)
    _validate_words(words, parsed)
    require(
        all(fingerprint(ref.path) == ref.sha256 for ref in references),
        "Probe input/runtime changed",
    )
    record_value = {
        "version": 1,
        "kind": "native-token-alignment-probe",
        "request": {"path": str(request), "sha256": parsed.reference.sha256},
        "inputs": {str(ref.path): ref.sha256 for ref in references},
        "samples": parsed.sample_count,
        "sourceStartSample": parsed.start_sample,
        "sampleRate": SAMPLE_RATE,
        "dtype": "float16-provider-default",
        "operation": "forward_with_cross_qk/native-find_alignment-before-word-heuristics",
        "autoregressiveDecoding": False,
        "retokenized": False,
        "words": [
            {
                **asdict(word),
                "sourceStart": parsed.start_sample / SAMPLE_RATE + word.start,
                "sourceEnd": parsed.start_sample / SAMPLE_RATE + word.end,
            }
            for word in words
        ],
        "semanticApproved": False,
        "deliveryComplete": False,
    }
    if parsed.origin is not None:
        record_value["origin"] = parsed.origin
    payload = json.dumps(record_value, ensure_ascii=False, allow_nan=False).encode()
    require(len(payload) < MAX_BYTES, "Probe evidence quota")
    with output.open("xb") as stream:
        stream.write(payload + b"\n")
    return output


def verify(request: Path, output: Path) -> None:
    """Independently validate immutable references and native-clock conservation."""
    parsed = load_request(request)
    document = read(output)
    refs = (parsed.reference, *parsed.artifacts.values(), *parsed.runtime)
    require(
        document.get("request") == {"path": str(request), "sha256": parsed.reference.sha256}
        and document.get("inputs") == {str(ref.path): ref.sha256 for ref in refs},
        "Probe binding changed",
    )
    require(document.get("origin") == parsed.origin, "Diagnostic origin changed")
    require(
        document.get("kind") == "native-token-alignment-probe"
        and document.get("version") == 1
        and document.get("autoregressiveDecoding") is False
        and document.get("retokenized") is False
        and document.get("semanticApproved") is False
        and document.get("deliveryComplete") is False
        and document.get("samples") == parsed.sample_count
        and document.get("sourceStartSample") == parsed.start_sample
        and document.get("sampleRate") == SAMPLE_RATE
        and document.get("dtype") == "float16-provider-default"
        and document.get("operation")
        == "forward_with_cross_qk/native-find_alignment-before-word-heuristics",
        "Invalid diagnostic contract",
    )
    words: list[NativeWord] = []
    for value in rows(document.get("words")):
        item = record(value)
        require(
            set(item)
            == {"text", "tokens", "start", "end", "probability", "sourceStart", "sourceEnd"},
            "Invalid native word fields",
        )
        word = NativeWord(
            _text(item.get("text")),
            tuple(_integer(token) for token in rows(item.get("tokens"))),
            _number(item.get("start")),
            _number(item.get("end")),
            _number(item.get("probability")),
        )
        require(
            item.get("sourceStart") == parsed.start_sample / SAMPLE_RATE + word.start
            and item.get("sourceEnd") == parsed.start_sample / SAMPLE_RATE + word.end,
            "Source clock was adjusted",
        )
        words.append(word)
    _validate_words(tuple(words), parsed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        verify(args.request, args.output)
    else:
        print(run(args.request, args.output))


if __name__ == "__main__":
    main()
