"""Output-identical speedups for the pinned MLX Whisper decode, installed once per worker.

Each replaces a stock mlx-whisper 0.4.3 function with the same arithmetic done with less work;
on 16 chunks of a recording the transcribe JSON stayed byte-identical and recognition took 20%
less time (measured 2026-10-22):

- timestamp rules: the stock filter copies the tokens to the host at every step (``tolist``),
  which stalls the decode loop's ``async_eval`` look-ahead; the same mask is built with mx ops.
- word alignment: the stock alignment encodes the window's mel again; it reuses the encoder
  output of the decode of that same mel.
- cross-attention: the stock attention scales the cached audio keys in every layer at every
  token; they are scaled once, when the cache is built (the same fp16 multiply).

``mx`` and the mlx-whisper modules are passed in (mlx has no type stubs, hence ``Any``), so the
orchestrator and tests never import MLX.
"""

from collections.abc import Callable
from typing import Any

from replay.core import ReplayError

Function = Callable[..., Any]
# The patches reproduce this release's arithmetic, including its empty monotonic-timestamp mask.
PATCHED_VERSION = "0.4.3"


def timestamp_rules(mx: Any) -> Function:
    """``ApplyTimestampRules.apply`` without the per-step host copy of the tokens."""

    def apply(self: Any, logits: Any, tokens: Any) -> Any:
        tokenizer = self.tokenizer
        begin, eot = tokenizer.timestamp_begin, tokenizer.eot
        sampled = tokens.shape[1] - self.sample_begin  # a shape, known without a sync
        column = mx.arange(logits.shape[-1])[None]
        mask = mx.zeros(logits.shape, mx.float32)
        if tokenizer.no_timestamps is not None:
            mask = mx.where(column == tokenizer.no_timestamps, -mx.inf, mask)
        if sampled >= 1:  # timestamps come in pairs: text after a pair, else timestamp or EOT
            last = tokens[:, -1:] >= begin
            penultimate = tokens[:, -2:-1] >= begin if sampled >= 2 else True
            banned = mx.where(penultimate, column >= begin, column < eot)
            mask = mx.where(last & banned, -mx.inf, mask)
        # Stock's "timestamps never decrease" rule collects token positions, not token values, so
        # its mask [timestamp_begin:position + 1] is always empty; omitting it mirrors stock.
        if sampled == 0:
            mask = mx.where(column < begin, -mx.inf, mask)
            if self.max_initial_timestamp_index is not None:
                last_allowed = begin + self.max_initial_timestamp_index
                mask = mx.where(column > last_allowed, -mx.inf, mask)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        timestamp_logprob = mx.logsumexp(logprobs[:, begin:], axis=-1, keepdims=True)
        max_text_token_logprob = logprobs[:, :begin].max(axis=-1, keepdims=True)
        mask[:, :begin] = mx.where(
            timestamp_logprob > max_text_token_logprob, -mx.inf, mask[:, :begin]
        )
        return logits + mask

    return apply


def reused_features(timing: Any, whisper: Any) -> tuple[Function, Function, Function]:
    """``Whisper.decode``, ``find_alignment`` and ``Whisper.forward_with_cross_qk`` sharing the
    decode's encoder output with the alignment of the same mel object."""
    decode, align = whisper.Whisper.decode, timing.find_alignment
    forward = whisper.Whisper.forward_with_cross_qk
    kept: dict[str, Any] = {"reuse": False}

    def decode_kept(self: Any, mel: Any, *args: Any, **kwargs: Any) -> Any:
        result = decode(self, mel, *args, **kwargs)
        if mel.ndim == 2:  # one window: transcribe's call; the mel stays referenced, so `is` holds
            kept.update(mel=mel, features=result.audio_features)
        return result

    def align_kept(
        model: Any, tokenizer: Any, text_tokens: Any, mel: Any, *args: Any, **kwargs: Any
    ) -> Any:
        kept["reuse"] = kept.get("mel") is mel
        try:
            return align(model, tokenizer, text_tokens, mel, *args, **kwargs)
        finally:
            kept["reuse"] = False

    def forward_kept(self: Any, mel: Any, tokens: Any) -> Any:
        if not kept["reuse"]:
            return forward(self, mel, tokens)
        logits, _cache, cross_qk = self.decoder(tokens, kept["features"][None])
        return logits, cross_qk

    return decode_kept, align_kept, forward_kept


def scaled_cross_attention(mx: Any, stock: Function) -> Function:
    """``MultiHeadAttention.__call__`` whose cross-attention cache holds pre-scaled keys."""

    def attend(self: Any, x: Any, xa: Any = None, mask: Any = None, kv_cache: Any = None) -> Any:
        if xa is None:
            return stock(self, x, xa, mask, kv_cache)
        q = self.query(x)
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        k, v = (self.key(xa) * scale, self.value(xa)) if kv_cache is None else kv_cache
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        keys = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1)
        values = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)
        qk = q @ keys
        weights = mx.softmax(qk, axis=-1, precise=True)
        out = (weights @ values).transpose(0, 2, 1, 3).reshape(n_batch, n_ctx, n_state)
        return self.out(out), (k, v), qk

    return attend


def install(mx: Any, decoding: Any, timing: Any, whisper: Any, *, version: str) -> None:
    """Replace the stock functions in this process; only for the release they reproduce."""
    if version != PATCHED_VERSION:
        raise ReplayError(f"The decode patches reproduce mlx-whisper {PATCHED_VERSION} only")
    decoding.ApplyTimestampRules.apply = timestamp_rules(mx)
    decode, align, forward = reused_features(timing, whisper)
    whisper.Whisper.decode, timing.find_alignment = decode, align
    whisper.Whisper.forward_with_cross_qk = forward
    attention = whisper.MultiHeadAttention
    attention.__call__ = scaled_cross_attention(mx, attention.__call__)
