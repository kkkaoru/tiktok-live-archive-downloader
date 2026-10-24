"""Output-identical MLX Whisper decode patches, exercised with numpy standing in for mlx.core."""

import types
from typing import Any

import numpy as np
import pytest

from replay import asr_decode_patches as patches
from replay.core import ReplayError


def logsumexp(x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    top = x.max(axis=axis, keepdims=True)
    return np.asarray(top + np.log(np.exp(x - top).sum(axis=axis, keepdims=True)))


def softmax(x: np.ndarray, axis: int, precise: bool) -> np.ndarray:
    assert precise
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return np.asarray(e / e.sum(axis=axis, keepdims=True))


MX = types.SimpleNamespace(
    float32=np.float32, inf=np.inf, arange=np.arange, zeros=np.zeros, where=np.where,
    logsumexp=logsumexp, softmax=softmax,
)  # fmt: skip
TEXT = np.array([[5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # a text token clearly beats every timestamp
STAMP = np.array([[0.0, 0, 0, 0, 0, 0, 0, 5, 5, 5]])  # timestamps together beat any text token


def rules(no_timestamps: int | None = 6, initial: int | None = 1) -> Any:
    tokenizer = types.SimpleNamespace(timestamp_begin=7, eot=5, no_timestamps=no_timestamps)
    return types.SimpleNamespace(
        tokenizer=tokenizer, sample_begin=2, max_initial_timestamp_index=initial
    )


def allowed(logits: np.ndarray, tokens: list[int], rule: Any = None) -> list[int]:
    apply = patches.timestamp_rules(MX)
    out = apply(rule or rules(), logits, np.array([tokens]))
    return [index for index, value in enumerate(out[0]) if np.isfinite(value)]


def test_the_first_sampled_token_is_an_early_timestamp() -> None:
    assert allowed(TEXT, [1, 2]) == [7, 8]
    assert allowed(TEXT, [1, 2], rules(no_timestamps=None, initial=None)) == [7, 8, 9]


def test_timestamps_come_in_pairs() -> None:
    assert allowed(TEXT, [1, 2, 3]) == [0, 1, 2, 3, 4, 5, 7, 8, 9]
    assert allowed(TEXT, [1, 2, 8]) == [0, 1, 2, 3, 4, 5]
    assert allowed(TEXT, [1, 2, 3, 8]) == [5, 7, 8, 9]
    assert allowed(TEXT, [1, 2, 8, 9]) == [0, 1, 2, 3, 4, 5]


def test_timestamps_win_when_their_probability_beats_every_text_token() -> None:
    assert allowed(STAMP, [1, 2, 3]) == [7, 8, 9]


def test_the_stock_monotonic_rule_stays_empty() -> None:
    # Stock bans timestamps below the last one's *position* + 1, never a timestamp token.
    assert allowed(STAMP, [1, 2, 9, 3]) == [7, 8, 9]


class Whisper:
    def decode(self, mel: np.ndarray) -> Any:
        return types.SimpleNamespace(audio_features=np.full((2,), 7.0))

    def forward_with_cross_qk(self, mel: np.ndarray, tokens: list[int]) -> tuple[str, object]:
        return "encoded again", mel.shape

    def decoder(self, tokens: list[int], features: np.ndarray) -> tuple[object, str, str]:
        return ("reused", features.tolist()), "cache", "qk"


def find_alignment(
    model: Whisper, tokenizer: object, text_tokens: list[int], mel: np.ndarray, frames: int
) -> object:
    return model.forward_with_cross_qk(mel[None], text_tokens)


def test_the_alignment_reuses_the_encoding_of_the_decoded_mel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timing = types.SimpleNamespace(find_alignment=find_alignment)
    whisper = types.SimpleNamespace(Whisper=Whisper)
    decode, align, forward = patches.reused_features(timing, whisper)
    monkeypatch.setattr(Whisper, "decode", decode)
    monkeypatch.setattr(Whisper, "forward_with_cross_qk", forward)
    model, mel, other, batch = Whisper(), np.zeros((3, 4)), np.zeros((3, 4)), np.zeros((1, 3, 4))
    model.decode(mel)
    assert align(model, None, [1], mel, 3) == (("reused", [[7.0, 7.0]]), "qk")
    assert align(model, None, [1], other, 3) == ("encoded again", (1, 3, 4))
    assert model.forward_with_cross_qk(mel, [1]) == ("encoded again", (3, 4))
    model.decode(batch)  # a batched decode is not transcribe's window: nothing is kept
    assert align(model, None, [1], batch, 3) == ("encoded again", (1, 1, 3, 4))
    assert align(model, None, [1], mel, 3) == (("reused", [[7.0, 7.0]]), "qk")


class Attention:
    n_head = 2

    def query(self, x: np.ndarray) -> np.ndarray:
        return x * 2

    def key(self, x: np.ndarray) -> np.ndarray:
        return x + 1

    def value(self, x: np.ndarray) -> np.ndarray:
        return x - 1

    def out(self, x: np.ndarray) -> np.ndarray:
        return x * 3

    def __call__(
        self, x: np.ndarray, xa: Any = None, mask: Any = None, kv_cache: Any = None
    ) -> Any:
        """mlx-whisper 0.4.3's attention for a cross-attention call (or a self-attention marker)."""
        if xa is None:
            return "self-attention"
        q = self.query(x)
        k, v = (self.key(xa), self.value(xa)) if kv_cache is None else kv_cache
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        keys = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        values = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)
        qk = q @ keys
        out = (softmax(qk, -1, True) @ values).transpose(0, 2, 1, 3).reshape(n_batch, n_ctx, -1)
        return self.out(out), (k, v), qk


def test_cross_attention_scales_the_cached_keys_once() -> None:
    attend = patches.scaled_cross_attention(MX, Attention.__call__)
    layer = Attention()
    x = np.array([[[0.5, -1.0, 0.25, 2.0]]])
    xa = np.array([[[1.0, 0.0, -2.0, 0.5], [0.0, 3.0, 1.0, -1.0], [2.0, 2.0, 0.0, 1.0]]])
    out, (k, v), qk = attend(layer, x, xa)
    stock_out, (stock_k, stock_v), stock_qk = layer(x, xa)
    assert np.array_equal(out, stock_out) and np.array_equal(qk, stock_qk)
    assert np.array_equal(v, stock_v)
    assert np.array_equal(k, stock_k * 2**-0.25)  # head size 2: scale 2^-0.25, applied once
    cached, cache, cached_qk = attend(layer, x, xa, kv_cache=(k, v))
    assert np.array_equal(cached, out) and cache == (k, v) and np.array_equal(cached_qk, qk)
    assert attend(layer, x) == "self-attention"


def test_install_replaces_the_stock_functions_of_the_reproduced_release_only() -> None:
    decoding = types.SimpleNamespace(ApplyTimestampRules=type("Rules", (), {}))
    timing = types.SimpleNamespace(find_alignment=find_alignment)
    whisper = types.SimpleNamespace(
        Whisper=type("Model", (Whisper,), {}), MultiHeadAttention=type("Heads", (Attention,), {})
    )
    with pytest.raises(ReplayError, match="0.4.3 only"):
        patches.install(MX, decoding, timing, whisper, version="0.4.4")
    assert timing.find_alignment is find_alignment
    patches.install(MX, decoding, timing, whisper, version="0.4.3")
    assert decoding.ApplyTimestampRules.apply.__name__ == "apply"
    assert timing.find_alignment.__name__ == "align_kept"
    assert whisper.Whisper.decode.__name__ == "decode_kept"
    assert whisper.Whisper.forward_with_cross_qk.__name__ == "forward_kept"
    assert whisper.MultiHeadAttention.__call__.__name__ == "attend"
    assert whisper.MultiHeadAttention()(np.zeros((1, 1, 4))) == "self-attention"
