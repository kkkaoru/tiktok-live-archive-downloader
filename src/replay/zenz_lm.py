"""Score kana-to-kanji candidates with the pinned zenz-v2.5 model (GPT-2), without transformers.

zenz (azooKey's Zenzai model) is a GPT-2 conditioned on a katakana reading and an optional left
context. The prompt is the v2-family ``candidateEvaluationPrompt`` of AzooKeyKanaKanjiConverter
(commit 04ce2ec): ``<s>\\uEE00<katakana>\\uEE02<left context, last 40>\\uEE01``, followed by the
candidate surface and ``</s>``. ASCII spaces become U+3000 before tokenizing, as Zenzai's
``preprocessText`` does. A candidate's score is its teacher-forced log-probability; two surfaces
of one reading compare fairly because the prompt is identical.

The tokenizer is GPT-2 byte-level BPE read from the model's ``vocab.json``/``merges.txt``; the
forward pass is written against a numpy-like array module, so production runs it on MLX and the
tests run a tiny random model on numpy.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import regex

from replay.core import ReplayError

INPUT, OUTPUT, CONTEXT = "", "", ""
MAX_CONTEXT = 40  # ZenzPromptBuilder trims the left context to its last 40 characters
LAYER_NORM_EPSILON = 1e-5
BOS, EOS, UNKNOWN = "<s>", "</s>", "[UNK]"
# GPT-2's pre-tokenizer (transformers GPT2Tokenizer), which zenz's tokenizer config selects.
PRETOKENIZER = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
MAX_TOKENS = 1024  # n_positions
# mlx has no type stubs and the same code runs on numpy arrays, hence ``Any`` for arrays and ``xp``.
Array = Any


@lru_cache(maxsize=1)
def _byte_symbols() -> dict[int, str]:
    """GPT-2's reversible byte -> printable character table."""
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    table = {byte: chr(byte) for byte in printable}
    extra = 0
    for byte in range(256):
        if byte not in table:
            table[byte] = chr(256 + extra)
            extra += 1
    return table


class Tokenizer:
    """GPT-2 byte-level BPE; ids match transformers' GPT2Tokenizer for zenz (measured on 1,353
    prompts and surfaces). Bytes outside the vocabulary map to ``[UNK]`` as there."""

    def __init__(self, vocabulary: Mapping[str, int], merges: Sequence[tuple[str, str]]) -> None:
        self.vocabulary = dict(vocabulary)
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        for name in (BOS, EOS, UNKNOWN):
            if name not in self.vocabulary:
                raise ReplayError(f"zenz vocabulary lacks {name}")
        self.bos, self.eos = self.vocabulary[BOS], self.vocabulary[EOS]
        self.unknown = self.vocabulary[UNKNOWN]
        self._cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def load(cls, directory: Path) -> Tokenizer:
        vocabulary = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        if not isinstance(vocabulary, dict) or not all(
            isinstance(key, str) and isinstance(value, int) for key, value in vocabulary.items()
        ):
            raise ReplayError("zenz vocab.json must map tokens to ids")
        lines = (directory / "merges.txt").read_text(encoding="utf-8").splitlines()
        merges = []
        for line in lines[1:] if lines and lines[0].startswith("#version") else lines:
            parts = line.split(" ")
            if len(parts) != 2:
                raise ReplayError("zenz merges.txt needs two symbols per line")
            merges.append((parts[0], parts[1]))
        return cls(vocabulary, merges)

    def _word(self, piece: str) -> tuple[int, ...]:
        if piece in self._cache:
            return self._cache[piece]
        symbols = [_byte_symbols()[byte] for byte in piece.encode("utf-8")]
        while len(symbols) > 1:
            pairs = zip(symbols, symbols[1:], strict=False)
            rank, index = min((self.ranks.get(pair, math.inf), at) for at, pair in enumerate(pairs))
            if rank == math.inf:
                break
            symbols[index : index + 2] = [symbols[index] + symbols[index + 1]]
        ids = tuple(self.vocabulary.get(symbol, self.unknown) for symbol in symbols)
        self._cache[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [token for piece in PRETOKENIZER.findall(text) for token in self._word(piece)]


def prompt(reading: str, context: str) -> str:
    """The v2-family Zenzai evaluation prompt for a katakana reading and its left context."""
    tail = context[-MAX_CONTEXT:]
    return INPUT + reading + (CONTEXT + tail if tail else "") + OUTPUT


def normalized(text: str) -> str:
    """Zenzai's ``preprocessText``: ASCII spaces become U+3000, newlines are dropped."""
    return text.replace(" ", "　").replace("\n", "")


class Model:
    """GPT-2 forward pass over ``xp`` (``mlx.core`` in production, numpy in tests)."""

    def __init__(self, weights: Mapping[str, Array], xp: Any, heads: int) -> None:
        self.xp, self.heads = xp, heads
        self.w = {name.removeprefix("transformer."): value for name, value in weights.items()}
        self.layers = sum(1 for name in self.w if name.endswith(".ln_1.weight"))
        if not self.layers or "wte.weight" not in self.w or "wpe.weight" not in self.w:
            raise ReplayError("zenz weights are not a GPT-2 checkpoint")

    def _norm(self, x: Array, name: str) -> Array:
        xp = self.xp
        mean = xp.mean(x, axis=-1, keepdims=True)
        variance = xp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        scaled = (x - mean) / xp.sqrt(variance + LAYER_NORM_EPSILON)
        return scaled * self.w[f"{name}.weight"] + self.w[f"{name}.bias"]

    def log_probabilities(self, ids: Array) -> Array:
        """Next-token log-probabilities for a (batch, length) id array."""
        xp, w = self.xp, self.w
        batch, length = ids.shape
        x = w["wte.weight"][ids] + w["wpe.weight"][:length]
        mask = xp.triu(xp.full((length, length), -1e9, dtype=x.dtype), k=1)
        width = x.shape[-1] // self.heads
        for layer in range(self.layers):
            prefix = f"h.{layer}"
            qkv = self._norm(x, f"{prefix}.ln_1") @ w[f"{prefix}.attn.c_attn.weight"]
            qkv = qkv + w[f"{prefix}.attn.c_attn.bias"]
            q, k, v = (
                qkv[..., part * x.shape[-1] : (part + 1) * x.shape[-1]]
                .reshape(batch, length, self.heads, width)
                .transpose(0, 2, 1, 3)
                for part in range(3)
            )
            scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(width) + mask
            scores = xp.exp(scores - xp.max(scores, axis=-1, keepdims=True))
            attention = scores / xp.sum(scores, axis=-1, keepdims=True)
            mixed = (attention @ v).transpose(0, 2, 1, 3).reshape(batch, length, x.shape[-1])
            x = x + mixed @ w[f"{prefix}.attn.c_proj.weight"] + w[f"{prefix}.attn.c_proj.bias"]
            hidden = self._norm(x, f"{prefix}.ln_2") @ w[f"{prefix}.mlp.c_fc.weight"]
            hidden = hidden + w[f"{prefix}.mlp.c_fc.bias"]
            # gelu_new: the tanh approximation GPT-2 was trained with
            hidden = (
                0.5 * hidden * (1 + xp.tanh(0.7978845608028654 * (hidden + 0.044715 * hidden**3)))
            )
            x = x + hidden @ w[f"{prefix}.mlp.c_proj.weight"] + w[f"{prefix}.mlp.c_proj.bias"]
        logits = self._norm(x, "ln_f") @ w["wte.weight"].T
        peak = xp.max(logits, axis=-1, keepdims=True)
        return logits - peak - xp.log(xp.sum(xp.exp(logits - peak), axis=-1, keepdims=True))


class Scorer:
    """Teacher-forced log P(surface ``</s>`` | prompt) for (reading, context, surface) items."""

    def __init__(self, tokenizer: Tokenizer, model: Model, batch: int = 16) -> None:
        self.tokenizer, self.model, self.batch = tokenizer, model, batch

    def score(self, items: Sequence[tuple[str, str, str]]) -> list[float]:
        xp, results = self.model.xp, []
        encoded = [
            (
                [self.tokenizer.bos, *self.tokenizer.encode(normalized(prompt(reading, context)))],
                [*self.tokenizer.encode(normalized(surface)), self.tokenizer.eos],
            )
            for reading, context, surface in items
        ]
        for start in range(0, len(encoded), self.batch):
            group = encoded[start : start + self.batch]
            width = max(len(head) + len(tail) for head, tail in group)
            if width > MAX_TOKENS:
                raise ReplayError("zenz input exceeds the model's 1,024 positions")
            # Right padding is invisible to earlier positions under the causal mask.
            rows = [
                [*head, *tail] + [self.tokenizer.eos] * (width - len(head) - len(tail))
                for head, tail in group
            ]
            # One host copy per batch; per-element reads would synchronize the GPU each time.
            table = np.asarray(self.model.log_probabilities(xp.array(rows)), dtype=np.float64)
            for row, (head, tail) in enumerate(group):
                positions = np.arange(len(head) - 1, len(head) + len(tail) - 1)
                results.append(float(table[row, positions, tail].sum()))
        return results


def load(directory: Path) -> Scorer:
    """The pinned checkpoint on MLX (deferred import: tests and the orchestrator skip MLX)."""
    mlx = importlib.import_module("mlx.core")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model_type") != "gpt2":
        raise ReplayError("zenz config.json must describe a GPT-2 model")
    stored = mlx.load(str(directory / "model.safetensors"))
    weights = {name: value.astype(mlx.float32) for name, value in stored.items()}
    return Scorer(Tokenizer.load(directory), Model(weights, mlx, int(config["n_head"])))
