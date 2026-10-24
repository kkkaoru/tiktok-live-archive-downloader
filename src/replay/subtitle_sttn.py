"""Refill repaired caption pixels with the pinned STTN video inpainting model.

Per-frame spatial inpainting (Telea) streaks and flickers on textured, moving backgrounds. STTN
(Spatial-Temporal Transformer Networks, the checkpoint of video-subtitle-remover pinned under
``edits/models/sttn-det``) fills each frame's masked pixels from neighbouring and reference
frames of the same chunk, so the fill follows the scene over time.

The masks are the exact pixels the caption repair changed (before encoding, unchanged pixels
equal the source byte for byte). Only those pixels are replaced; every other pixel is the source.
The band of rows holding all masks is processed as overlapping 432x240 tiles (the model's
geometry) with linear blending across tile overlaps, in temporal windows with references.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import threading
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.subtitle_colour_residual import SAME

Image = NDArray[np.uint8]
Mask = NDArray[np.bool_]

TILE_WIDTH, TILE_HEIGHT = 432, 240
# Measured on chunk 111 of 10-14b (with two tiles): ±10 neighbours without distant reference
# frames take 192 s per chunk against 291 s with references every 40 frames, for the same look
# (12 vs 11 of 899 frames above 0.02 instability). References: None disables them.
NEIGHBORS = 10
REFERENCE_STEP: int | None = None
WINDOW = 240
ENCODE_FRAMES = 24  # frames per encoder pass inside a window (memory only; the same features)
# Compute reductions, recorded in chunk pins through compute(). Only the skip is on by default (it
# leaves every read pixel as before); the others wait for the calibration in
# edits/work/pending-fixes/metal-memory/quality/ (accept: <= 0.1 dB mean, no block below -0.5 dB).
# A frame is decoded in a tile only where its masks come within EDGE scaled pixels of the tile's
# columns (the composite's cubic upscale reads under three pixels around a mask); a group of
# neighbours holding no such frame is not run at all.
SKIP_UNNEEDED = True
EDGE = 4
# Feature rows (a quarter of the tile's) decoded around the band's rows: the decoder's receptive
# field is under three; the margin keeps the surroundings of the band in the decoded region.
# None decodes all 60 rows. Ground truth (edits/work/pending-fixes/metal-memory/quality/, 28
# cases): 2 rows +0.17 dB mean, worst block -0.04 dB, 0.98 of the fill time.
ROW_MARGIN: int | None = 2
# Share of ring pixels (RING_PIXELS around a repeated mask) that may change beyond SAME while a
# frame still reuses the earlier fill; 0 reuses only when none changed.
REUSE_CHANGED = 0.0
# A frame may reuse an earlier fill although its repaired pixels differ (YOLO boxes move a few
# pixels between frames, so equal masks reused only 3% of a real chunk's frames): its pixels must
# lie inside the computed frame's (a newly covered pixel may hold the text the mask grew for and
# is filled for real), and the surroundings of both must be still. Recorded in chunk pins (compute).
# Real chunk 10-06 #100: 125 instead of 27 of 900 frames reused, 88 instead of 98 STTN runs, the
# OCR audit and colour gate clean, flicker inside the masks unchanged (66 of 899 frames above
# 0.02 either way); edits/work/metal-opt-20261024/acct/.
REUSE_ANY_MASK = True
# Pixels the computed frame's mask may be grown by before the reusing frame's must lie inside it:
# the boxes jitter outward by a few pixels every frame (a strict subset reused 4.4% of a real
# chunk's frames, 3 pixels 14%), while a new character extends a caption by ~20 pixels or more.
REUSE_SLACK = 3
# Model precision on MPS: "bfloat16" only. float16 overflows on real content: against bfloat16
# on the 7-source c60 gate, 11 of 140 blocks fell below -0.5 dB, the worst -16.6 dB
# (edits/work/int8-sttn-20261024/gate-fp16-noise.jsonl, 2026-10-24).
DTYPE = "bfloat16"
# Frames shared by consecutive windows, blended linearly; recorded in chunk pins (compute). Ground
# truth (28 cases, real c60 masks): 6 against 12 is -0.03 dB mean on the frames around the seam,
# seam flicker unchanged, 1.9% fewer tile frames; 2 raised the flicker.
OVERLAP = 6
# Two tiles across the band (scale ~0.70 instead of ~0.85): about half the GPU work per chunk;
# chunk 111 of 10-14b: 11 of 899 frames above 0.02 instability (5 with three), same look.
MAX_TILES = 2
# Recorded in chunk pins. Each caption (frames between caption-free frames or caption changes) is
# filled with one tile instead of MAX_TILES where that costs nothing visible: (a) its masks fit
# one tile cropped around them at a scale no coarser than the whole band's (ground truth: equal
# PSNR on average, half the time), or (b) their surroundings are plain (see FLAT).
TILES = "adaptive-v1"
CONTEXT = 48  # band pixels kept beside a cropped run's masks for the model to read
# texture() below which one tile across the band is used. Ground truth (real backgrounds of 7
# recordings at 4 heights under the real 10/21 caption masks, 140 blocks of 60 frames): one tile
# was as good or better only where the score was 0 (a plain screen); from about 1 it lost 0-1.6 dB
# against two tiles with no score separating the losses, so only plain surroundings use one.
FLAT = 0.5
SAMPLES = 8  # frames of a caption whose surroundings texture() measures
CHANGE = 0.6  # overlap (intersection over union) of masks below which a new caption starts
SHORTEST = 30  # frames: a caption is not split again before this many frames
MASK_GROWTH = 3
# Surroundings checked before reusing a fill: about one smallest attention patch (9x5 features,
# 36x20 tile pixels) at the band's ~0.7 tile scale.
RING_PIXELS = 32  # pixels around the changed support, so edges are refilled too
CHECKPOINT = "sttn.pth"
ARCHITECTURE = "architecture.py"
# Recorded in chunk pins. Metal kernels replace torch MPS's 7-D permute copies of the patch
# tokens (about 7 GB/s; slower than the attention matmuls on the finest level): attention 1.5x,
# a whole 300-frame fill 1.2-1.4x at equal PSNR/flicker (10/17 chunk 30, A/B under load).
# v2: fused LeakyReLU/residual kernels and CPU work overlapped with the GPU (same output).
ATTENTION = "metal-fused-v2"
# Levels with fewer tokens than this keep torch's copies and split their long reduction instead
# (level 0: 21 tokens of 414720 features, 4.7 -> 0.4 ms).
SPLIT = 64
METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;
// p: T, H, W, ph, pw, dk, cbase, cstep, st, sc, sh, sw (element strides of the feature map).
// Token features are ordered (y, x, channel); attention only needs q, k, v and y to share it.
kernel void gather(device const ushort* z, device ushort* tok, constant uint* p,
                   uint e [[thread_position_in_grid]]) {
    uint T = p[0], H = p[1], W = p[2], ph = p[3], pw = p[4], dk = p[5], cbase = p[6], cstep = p[7];
    uint st = p[8], sc = p[9], sh = p[10], sw = p[11];
    uint oh = H / ph, ow = W / pw, N = T * oh * ow, px = ph * pw;
    uint s = e / (N * px), r = e % (N * px), n = r / px, yx = r % px, y = yx / pw, x = yx % pw;
    uint t = n / (oh * ow), ij = n % (oh * ow), i = ij / ow, j = ij % ow;
    device const ushort* src = z + t * st + (cbase + s * cstep) * sc + (i * ph + y) * sh
                               + (j * pw + x) * sw;
    device ushort* dst = tok + ((s * N + n) * px + yx) * dk;
    for (uint c = 0; c < dk; ++c) dst[c] = src[c * sc];
}
kernel void scatter(device const ushort* tok, device ushort* out, constant uint* p,
                    uint e [[thread_position_in_grid]]) {
    uint T = p[0], H = p[1], W = p[2], ph = p[3], pw = p[4], dk = p[5], cbase = p[6];
    uint st = p[8], sc = p[9], sh = p[10], sw = p[11];
    uint oh = H / ph, ow = W / pw, px = ph * pw;
    uint n = e / px, yx = e % px, y = yx / pw, x = yx % pw;
    uint t = n / (oh * ow), ij = n % (oh * ow), i = ij / ow, j = ij % ow;
    device ushort* dst = out + t * st + cbase * sc + (i * ph + y) * sh + (j * pw + x) * sw;
    device const ushort* src = tok + (n * px + yx) * dk;
    for (uint c = 0; c < dk; ++c) dst[c * sc] = src[c];
}
// LeakyReLU in place, and out = x + LeakyReLU(y): one pass instead of torch's two or three.
// The slope arrives rounded to bfloat16 like torch's, so both equal torch bit for bit.
kernel void leaky(device bfloat4* x, constant float& s, uint i [[thread_position_in_grid]]) {
    float4 v = float4(x[i]);
    x[i] = bfloat4(select(v * s, v, v > 0));
}
kernel void add_leaky(device const bfloat4* x, device const bfloat4* y, device bfloat4* out,
                      constant float& s, uint i [[thread_position_in_grid]]) {
    float4 v = float4(y[i]);
    out[i] = bfloat4(float4(x[i]) + float4(bfloat4(select(v * s, v, v > 0))));
}
kernel void leaky_half(device half4* x, constant float& s, uint i [[thread_position_in_grid]]) {
    float4 v = float4(x[i]);
    x[i] = half4(select(v * s, v, v > 0));
}
kernel void add_leaky_half(device const half4* x, device const half4* y, device half4* out,
                           constant float& s, uint i [[thread_position_in_grid]]) {
    float4 v = float4(y[i]);
    out[i] = half4(float4(x[i]) + float4(half4(select(v * s, v, v > 0))));
}
// uint8 NHWC frames and their uint8 masks -> the masked bfloat16 NCHW encoder input, one pass.
// The table holds torch's own result for every (mask byte, pixel byte) pair (input_table).
kernel void prepare(device const uchar* frames, device const uchar* masks,
                    device const ushort* table, device ushort* x, constant uint& hw,
                    uint e [[thread_position_in_grid]]) {
    uint n = e / (3 * hw), r = e % (3 * hw), c = r / hw, pixel = n * hw + r % hw;
    x[e] = table[uint(masks[pixel]) * 256 + frames[pixel * 3 + c]];
}
// out += the 0..255 pixel value of each bfloat16 tanh output, read from torch's own result for
// every bfloat16 bit pattern (output_table): one pass instead of four temporaries.
kernel void accumulate(device const ushort* t, device const float* table, device float* out,
                       uint i [[thread_position_in_grid]]) {
    out[i] += table[t[i]];
}
"""
_metal: list[Any] = []  # the compiled library, built on first use in this process
# Kernel layout tensors by device, geometry and strides: a block needed six host-to-GPU copies.
_layouts: dict[tuple[str | int, ...], Any] = {}


class Tiler(Protocol):
    def tile(
        self,
        frames: NDArray[np.uint8],
        masks: NDArray[np.uint8],
        rows: slice,
        needed: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]: ...


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def patch_attention(self: Any, x: Any, m: Any, b: int, c: int) -> Any:
    """``MultiHeadedAttention.forward`` of the pinned STTN, computed with fewer GPU copies.

    Bit-identical to the pinned code (checked on MPS in float32 and bfloat16): the pinned code
    builds a token-by-token mask it never applies (``masked_fill`` without assignment), so it is
    not built here; the three 1x1 query/key/value projections run as one convolution.
    """
    import torch
    import torch.nn.functional as functional

    frames_total, _, h, w = x.size()
    t = frames_total // b
    d_k = c // len(self.patchsize)
    q_all, k_all, v_all = functional.conv2d(x, *_projection(self)).chunk(3, dim=1)
    output = []
    for level, (width, height) in enumerate(self.patchsize):
        out_w, out_h = w // width, h // height

        shape = (b, t, d_k, out_h, height, out_w, width)
        q, k, v = (
            z[:, level * d_k : (level + 1) * d_k]
            .reshape(shape)
            .permute(0, 1, 3, 5, 2, 4, 6)
            .reshape(b, t * out_h * out_w, -1)
            for z in (q_all, k_all, v_all)
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        y = torch.matmul(functional.softmax(scores, dim=-1), v)
        y = y.view(b, t, out_h, out_w, d_k, height, width).permute(0, 1, 4, 2, 5, 3, 6)
        output.append(y.reshape(frames_total, d_k, h, w))
    return self.output_linear(torch.cat(output, 1))


def _library() -> Any:
    import torch

    if not _metal:
        _metal.append(torch.mps.compile_shader(METAL_SOURCE))
    return _metal[0]


def _layout(z: Any, *values: int) -> Any:
    """Kernel parameters: the level's geometry, then the feature map's element strides."""
    import torch

    parameters = [*values, *z.stride()]
    key = (str(z.device), *parameters)
    if key not in _layouts:  # never written after creation, so kernels may share it
        _layouts[key] = torch.tensor(parameters, dtype=torch.int32, device=z.device)
    return _layouts[key]


def _projection(self: Any) -> tuple[Any, Any]:
    """The query, key and value 1x1 weights and biases as one projection.

    ``Inpainter`` stores them once its weights are final (``freeze_projections``); otherwise
    they are concatenated per call.
    """
    import torch

    frozen: tuple[Any, Any] | None = getattr(self, "frozen_projection", None)
    if frozen is not None:
        return frozen
    weight = torch.cat([self.query_embedding.weight, self.key_embedding.weight,
                        self.value_embedding.weight])  # fmt: skip
    bias = torch.cat([self.query_embedding.bias, self.key_embedding.bias,
                      self.value_embedding.bias])  # fmt: skip
    return weight, bias


def freeze_projections(network: Any) -> None:
    """Store each attention's merged projection; call only after the last weight change."""
    for module in network.modules():
        if hasattr(module, "query_embedding"):
            module.frozen_projection = None  # recomputed from the current weights
            module.frozen_projection = _projection(module)


def _metal_ready(x: Any, b: int) -> bool:
    import torch

    return bool(x.device.type == "mps" and x.dtype in (torch.bfloat16, torch.float16) and b == 1)


def metal_patch_attention(self: Any, x: Any, m: Any, b: int, c: int) -> Any:
    """``patch_attention`` with Metal token gather/scatter; same weights and math (bf16 MPS).

    Other devices, dtypes or batches fall back to ``patch_attention`` unchanged.
    """
    if not _metal_ready(x, b):
        return patch_attention(self, x, m, b, c)
    return self.output_linear(_metal_merged(self, x, c))


def _metal_merged(self: Any, x: Any, c: int, rows: tuple[int, int] | None = None) -> Any:
    """The attention's per-level results before ``output_linear`` (Metal path only).

    With ``rows``, only the queries of patches touching those feature rows are computed on the
    gathered levels (keys and values stay whole, so those rows are the same bits); the other
    rows of the result are left unwritten.
    """
    import torch
    import torch.nn.functional as functional

    frames_total, _, h, w = x.size()
    d_k = c // len(self.patchsize)
    qkv = functional.conv2d(x, *_projection(self))
    q_all, k_all, v_all = qkv.chunk(3, dim=1)
    merged = torch.empty_like(q_all, memory_format=torch.preserve_format)
    library = _library()
    for level, (width, height) in enumerate(self.patchsize):
        out_h, out_w = h // height, w // width
        n, size = frames_total * out_h * out_w, d_k * height * width
        if n < SPLIT and size % SPLIT == 0:
            shape = (frames_total, d_k, out_h, height, out_w, width)
            q, k, v = (
                z[:, level * d_k : (level + 1) * d_k]
                .reshape(shape)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(n, SPLIT, -1)
                .transpose(0, 1)
                for z in (q_all, k_all, v_all)
            )
            scores = (q @ k.transpose(1, 2)).sum(0) / math.sqrt(size)
            y = functional.softmax(scores, dim=-1) @ v.transpose(0, 1).reshape(n, size)
            y = y.view(frames_total, out_h, out_w, d_k, height, width).permute(0, 3, 1, 4, 2, 5)
            merged[:, level * d_k : (level + 1) * d_k] = y.reshape(frames_total, d_k, h, w)
            continue
        tokens = torch.empty((3, n, size), device=x.device, dtype=x.dtype)
        geometry = (frames_total, h, w, height, width, d_k, level * d_k)
        library.gather(qkv.view(torch.int16), tokens.view(torch.int16),
                       _layout(qkv, *geometry, c), threads=3 * n * height * width)  # fmt: skip
        q, k, v = tokens
        first, last = (0, out_h) if rows is None else (rows[0] // height, -(-rows[1] // height))
        if last - first < out_h:
            y = torch.empty_like(q)
            patches = q.view(frames_total, out_h, out_w, size)[:, first:last].reshape(-1, size)
            scores = patches @ k.transpose(0, 1) / math.sqrt(size)
            part = (functional.softmax(scores, dim=-1) @ v).view(frames_total, -1, out_w, size)
            y.view(frames_total, out_h, out_w, size)[:, first:last] = part
        else:
            scores = q @ k.transpose(0, 1) / math.sqrt(size)
            y = functional.softmax(scores, dim=-1) @ v
        library.scatter(y.view(torch.int16), merged.view(torch.int16),
                        _layout(merged, *geometry, 0), threads=n * height * width)  # fmt: skip
    return merged


def _slope(activation: Any, dtype: Any) -> float:
    """The LeakyReLU slope as torch applies it to ``dtype`` (0.2 becomes 0.2001953125 in bf16)."""
    import torch

    return float(torch.tensor(activation.negative_slope, dtype=dtype))


def _fusable(*tensors: Any) -> bool:
    return all(t.is_contiguous() and t.numel() % 4 == 0 for t in tensors)


def metal_block_forward(self: Any, state: dict[str, Any]) -> dict[str, Any]:
    """``TransformerBlock.forward`` with each LeakyReLU and residual add as one Metal pass.

    Bit-identical to the pinned block on bf16 MPS; anything else runs the pinned arithmetic.
    """
    x, m, b, c = state["x"], state["m"], state["b"], state["c"]
    linear, ff = self.attention.output_linear, self.feed_forward.conv
    if not (_metal_ready(x, b) and _fusable(x)):
        x = x + self.attention(x, m, b, c)
        return {"x": x + self.feed_forward(x), "m": m, "b": b, "c": c}
    library = _library()
    import torch

    half = x.dtype == torch.float16
    fused_add = library.add_leaky_half if half else library.add_leaky
    fused = library.leaky_half if half else library.leaky

    def add_leaky(base: Any, value: Any, activation: Any) -> Any:
        if not _fusable(value):
            return base + activation(value)
        out = value.new_empty(value.shape)
        fused_add(base, value, out, _slope(activation, x.dtype), threads=out.numel() // 4)
        return out

    rows = state.get("rows")
    if rows is not None:
        return {"x": _block_rows(self, x, c, rows, add_leaky, fused), "m": m, "b": b, "c": c}
    x = add_leaky(x, linear[0](_metal_merged(self.attention, x, c)), linear[1])
    hidden = ff[0](x)
    if _fusable(hidden):
        fused(hidden, _slope(ff[1], x.dtype), threads=hidden.numel() // 4)
    else:
        hidden = ff[1](hidden)
    x = add_leaky(x, ff[2](hidden), ff[3])
    return {"x": x, "m": m, "b": b, "c": c}


def _block_rows(
    self: Any,
    x: Any,
    c: int,
    rows: tuple[int, int],
    add_leaky: Callable[[Any, Any, Any], Any],
    fused: Any,
) -> Any:
    """The fused block's output on feature ``rows`` only: the same bits as those rows of the
    whole block, computing each layer only on the rows the next one reads.

    Working back from ``rows``: the second FFN conv reads one row around, the dilated first two,
    ``output_linear`` one, and the attention queries of those rows (keys and values whole). A 3x3
    convolution of a row crop equals the whole convolution on rows whose window lies inside the
    crop or at the frame's edge (checked bit for bit on MPS, edits/work/metal-opt-20261024/).
    """
    linear, ff = self.attention.output_linear, self.feed_forward.conv
    height = x.shape[2]

    def around(span: tuple[int, int], reach: int) -> tuple[int, int]:
        return max(0, span[0] - reach), min(height, span[1] + reach)

    def crop(tensor: Any, span: tuple[int, int], within: tuple[int, int]) -> Any:
        return tensor[:, :, span[0] - within[0] : span[1] - within[0]].contiguous()

    wanted = (rows[0], rows[1])
    hidden_rows = around(wanted, 1)
    first_rows = around(hidden_rows, 2)
    merged_rows = around(first_rows, 1)
    merged = _metal_merged(self.attention, x, c, merged_rows)
    projected = linear[0](merged[:, :, merged_rows[0] : merged_rows[1]])
    first = add_leaky(
        crop(x, first_rows, (0, height)), crop(projected, first_rows, merged_rows), linear[1]
    )
    hidden = crop(ff[0](first), hidden_rows, first_rows)
    if _fusable(hidden):
        fused(hidden, _slope(ff[1], x.dtype), threads=hidden.numel() // 4)
    else:
        hidden = ff[1](hidden)
    return add_leaky(
        crop(first, wanted, first_rows), crop(ff[2](hidden), wanted, hidden_rows), ff[3]
    )


def input_table(device: Any, dtype: Any) -> Any:
    """The masked encoder input of every (mask byte, pixel byte) pair, as torch computes it.

    Row ``mask`` holds ``(byte / 255 * 2 - 1) * (1 - mask)`` with torch's own operations and
    roundings, so a lookup equals torch's per-element arithmetic bit for bit for any input.
    """
    import torch

    byte = torch.arange(256, device=device, dtype=torch.uint8)
    value = byte.float().div_(255).mul_(2).sub_(1).to(dtype)
    return (value[None, :] * (1 - byte.to(dtype)[:, None])).contiguous()


def output_table(device: Any, dtype: Any = None) -> Any:
    """``(v.float() + 1) / 2 * 255`` of every 16-bit ``v`` (bfloat16 by default), by its bits."""
    import torch

    every = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(dtype or torch.bfloat16)
    return ((every.to(device).float() + 1) / 2 * 255).contiguous()


def compute() -> str:
    """The compute choices that change the fill, for chunk pins (old and new never mix)."""
    rows = "all" if ROW_MARGIN is None else f"m{ROW_MARGIN}"
    skip = f"skip{EDGE}" if SKIP_UNNEEDED else "noskip"
    masks = f"-any{REUSE_SLACK}" if REUSE_ANY_MASK else ""
    overlap = "" if OVERLAP == 12 else f"-o{OVERLAP}"
    choices = f"rows{rows}-reuse{REUSE_CHANGED:g}{masks}{overlap}"
    return f"v1-{DTYPE}-n{NEIGHBORS}-{skip}-{choices}"


def load_architecture(directory: Path, sha: str) -> ModuleType:
    path = directory / ARCHITECTURE
    if sha256(path) != sha:
        raise ValueError("The STTN architecture differs from its pin")
    spec = importlib.util.spec_from_file_location("replay_sttn_architecture", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the STTN architecture")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    attention = getattr(loaded, "MultiHeadedAttention", None)
    if attention is not None:
        attention.forward = metal_patch_attention
    block = getattr(loaded, "TransformerBlock", None)
    if block is not None:
        block.forward = metal_block_forward
    return loaded


def _decoded_rows(rows: slice) -> tuple[int, int]:
    """Tile rows the decoder computes for ``rows``: whole feature rows, ``ROW_MARGIN`` beyond."""
    start, stop = rows.indices(TILE_HEIGHT)[:2]
    if ROW_MARGIN is None:
        return 0, TILE_HEIGHT
    top = max(0, start // 4 - ROW_MARGIN) * 4
    bottom = min(TILE_HEIGHT // 4, -(-stop // 4) + ROW_MARGIN) * 4
    return top, bottom


class Inpainter:
    """The pinned STTN generator on Apple MPS (weights only, verified by digest)."""

    def __init__(self, directory: Path, *, checkpoint_sha: str, architecture_sha: str) -> None:
        import torch

        if sha256(directory / CHECKPOINT) != checkpoint_sha:
            raise ValueError("The STTN checkpoint differs from its pin")
        architecture = load_architecture(directory, architecture_sha)
        self._torch = torch
        self._device = torch.device("mps")
        network: Any = architecture.InpaintGenerator()
        state = torch.load(directory / CHECKPOINT, map_location="cpu", weights_only=True)
        network.load_state_dict(state.get("netG", state))
        if DTYPE != "bfloat16":
            raise ValueError("STTN runs in bfloat16 only: float16 overflows on real content")
        self._dtype = torch.bfloat16
        self._network = network.to(self._device, dtype=self._dtype).eval()
        freeze_projections(self._network)
        self._table = input_table(self._device, self._dtype).view(torch.int16)
        self._unit = output_table(self._device, self._dtype)
        # The model's masks: the pinned attention never applies them (see patch_attention), so
        # one blank frame, expanded without a copy, replaces converting every window's masks.
        self._blank = torch.zeros((1, 1, TILE_HEIGHT, TILE_WIDTH), device=self._device,
                                  dtype=self._dtype)  # fmt: skip

    def tile(
        self,
        frames: NDArray[np.uint8],
        masks: NDArray[np.uint8],
        rows: slice,
        needed: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]:
        """Inpaint one 432x240 tile sequence; returns the predicted pixels (0..255) of ``rows``.

        Frames not ``needed`` (all are by default) are left 0 when ``SKIP_UNNEEDED`` is set.
        """
        result = self._predict(frames, masks, rows, needed)
        # torch keeps freed MPS buffers cached: 5.05 GiB were held after a chunk's fill, 2.04 GiB
        # with this, at the same speed and with identical output (measured 2026-10-20). Emptied
        # once the call's tensors are released: inside the call they kept 1.9-2.6 GB cached
        # through the audit and encode (measured 2026-10-22).
        self._torch.mps.empty_cache()
        return result

    def _predict(
        self,
        frames: NDArray[np.uint8],
        masks: NDArray[np.uint8],
        rows: slice,
        needed: NDArray[np.bool_] | None,
    ) -> NDArray[np.float32]:
        torch = self._torch
        count = len(frames)
        # Frames travel as uint8 and are converted on the GPU; predictions are summed there and
        # copied back once, so each window costs one transfer each way.
        pixels = torch.from_numpy(frames).to(self._device)
        marks = torch.from_numpy(masks).to(self._device)
        # Only the rows the caller reads are summed and copied back.
        out = torch.zeros(
            (count, 3, len(range(TILE_HEIGHT)[rows]), TILE_WIDTH), device=self._device
        )
        weight = torch.zeros((count, 1, 1, 1), device=self._device)
        library = _library()
        wanted = np.ones(count, bool) if needed is None or not SKIP_UNNEEDED else needed
        top, bottom = _decoded_rows(rows)
        start, stop = rows.indices(TILE_HEIGHT)[:2]
        step = REFERENCE_STEP
        with torch.no_grad():
            # Without references a run reads only its own frames, so each run is encoded alone:
            # the window's features (796 MB for 240 frames) are never held and a run that is
            # skipped is never encoded (the encoder treats frames apart: the same values).
            features = None if step is None else self._encode(pixels, marks)
            for centre in range(NEIGHBORS // 2, count + NEIGHBORS, 2 * NEIGHBORS):
                near = slice(max(0, centre - NEIGHBORS), min(count, centre + NEIGHBORS + 1))
                decode = np.flatnonzero(wanted[near])
                if not len(decode):
                    continue  # no frame of this run is read from this tile
                references = (
                    []
                    if step is None
                    else [i for i in range(0, count, step) if not near.start <= i < near.stop]
                )
                # The neighbours are one run of frames: sliced, they are views, where indexing
                # with a list copied their features and masks (14-20 ms per run, 2026-10-22).
                chosen = [*range(near.start, near.stop), *references] if references else near
                x = (
                    self._encode(pixels[near], marks[near])
                    if features is None
                    else features[chosen]
                )
                predicted = self._infer(x, (top // 4, bottom // 4))
                every = len(decode) == near.stop - near.start
                part = predicted[: near.stop - near.start]
                if not every:
                    part = part[torch.from_numpy(decode).to(self._device)]
                decoded = self._network.decoder(part)
                within = slice(start - top, stop - top)
                pixel = torch.tanh(decoded[:, :, within]).contiguous()
                if every:
                    library.accumulate(pixel.view(torch.int16), self._unit, out[near],
                                       threads=pixel.numel())  # fmt: skip
                    weight[near] += 1
                else:
                    at = torch.from_numpy(decode + near.start).to(self._device)
                    sums = out[at]
                    library.accumulate(pixel.view(torch.int16), self._unit, sums,
                                       threads=pixel.numel())  # fmt: skip
                    out[at] = sums
                    weight[at] += 1
            del pixels, marks, features
            result = (out / weight.clamp(min=1)).permute(0, 2, 3, 1).cpu().numpy()
        return np.asarray(result, dtype=np.float32)

    def _infer(self, features: Any, rows: tuple[int, int]) -> Any:
        """The transformer's output on the feature ``rows`` the decoder reads.

        The last block computes only those rows (``_block_rows``, the same bits); every earlier
        block is whole, since attention reads all of its rows.
        """
        network, blank = self._network, self._blank.expand(len(features), -1, -1, -1)
        transformer = getattr(network, "transformer", None)
        if rows == (0, TILE_HEIGHT // 4):
            return network.infer(features, blank)
        if transformer is None:  # a network without the pinned blocks: every row, then the rows
            return network.infer(features, blank)[:, :, rows[0] : rows[1]]
        functional = self._torch.nn.functional
        state = {"x": features, "m": functional.interpolate(blank, scale_factor=1.0 / 4),
                 "b": 1, "c": features.shape[1]}  # fmt: skip
        *blocks, last = transformer
        for block in blocks:
            state = block(state)
        out = last({**state, "rows": rows})["x"]
        # A block that is not the fused bfloat16 one (another device) returns every row.
        return out if out.shape[2] == rows[1] - rows[0] else out[:, :, rows[0] : rows[1]]

    def _encode(self, pixels: Any, marks: Any) -> Any:
        """Encoder features of every frame, computed ``ENCODE_FRAMES`` at a time.

        The same values as one pass over the window (the encoder treats frames apart; checked
        bit for bit on MPS), without the window's full-resolution activations: a 96-frame pass
        held 2090 MB of MPS driver memory, 24-frame passes 1066 MB (measured 2026-10-22).
        The masked input is built by one Metal pass from the uint8 frames and masks instead of
        torch's permute copy, float32 steps and mask product (about 3 GB of GPU reads and writes
        per 240 frames): 120 ms against 25 ms per 240 frames, bit for bit (2026-10-22).
        """
        torch = self._torch
        library = _library()
        features: Any = None
        for first in range(0, len(pixels), ENCODE_FRAMES):
            part = slice(first, first + ENCODE_FRAMES)
            count, height, width = pixels[part].shape[:3]
            # Contiguous NCHW: on the channels-last view attention took about 110 ms against 62 ms.
            x = torch.empty((count, 3, height, width), device=self._device, dtype=self._dtype)
            library.prepare(pixels[part], marks[part], self._table, x.view(torch.int16),
                            height * width, threads=x.numel())  # fmt: skip
            encoded = self._network.encoder(x)
            if features is None:
                features = encoded.new_empty((len(pixels), *encoded.shape[1:]))
            features[part] = encoded
        return features


def starts(width: int) -> list[int]:
    """Tile offsets covering ``width`` with overlaps of at least a quarter tile."""
    if width <= TILE_WIDTH:
        return [0]
    count = int(np.ceil((width - TILE_WIDTH) / (TILE_WIDTH * 3 / 4))) + 1
    return [round(i * (width - TILE_WIDTH) / (count - 1)) for i in range(count)]


def ramp(offsets: Sequence[int]) -> list[NDArray[np.float32]]:
    """Per-tile column weights that blend linearly across every overlap."""
    weights = []
    for index, offset in enumerate(offsets):
        w = np.ones(TILE_WIDTH, np.float32)
        if index > 0:
            overlap = offsets[index - 1] + TILE_WIDTH - offset
            w[:overlap] = np.linspace(0, 1, overlap + 2, dtype=np.float32)[1:-1]
        if index < len(offsets) - 1:
            overlap = offset + TILE_WIDTH - offsets[index + 1]
            w[TILE_WIDTH - overlap :] = np.linspace(1, 0, overlap + 2, dtype=np.float32)[1:-1]
        weights.append(w)
    return weights


def changed(current: NDArray[np.uint8], previous: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Pixels whose largest channel difference exceeds the shared ``SAME`` tolerance."""
    result = np.zeros(current.shape[:2], dtype=bool)
    if current is previous:  # an unrepaired frame is passed through as the source itself
        return result
    # Only rows that differ at all can change; a repair touches a band of a full frame, and the
    # int16 difference of every row cost 15.5 ms per 1080x1920 frame against 1.8 ms.
    rows = np.flatnonzero((current != previous).any(axis=(1, 2)))
    if rows.size:
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        band = current[top:bottom].astype(np.int16) - previous[top:bottom].astype(np.int16)
        result[top:bottom] = np.abs(band).max(axis=2) > SAME
    return result


def plan_reuse(
    bands: NDArray[np.uint8], grown: Sequence[NDArray[np.uint8]] | NDArray[np.uint8]
) -> list[int | None]:
    """For each frame, the frame whose STTN fill it uses; ``None`` when it has nothing to fill.

    A frame without repaired pixels (no caption detected) needs no fill. A frame whose repaired
    pixels are exactly those of the previous frame (with ``REUSE_ANY_MASK``, any inside the
    computed frame's), and
    whose surroundings (``RING_PIXELS`` around its and the computed frame's pixels) have no pixel
    changed beyond ``SAME`` since the frame that was computed, reuses that frame's result instead
    of running the model again. ``REUSE_CHANGED`` above 0 tolerates that share of changed
    surrounding pixels.
    """
    ring = np.ones((2 * RING_PIXELS + 1,) * 2, np.uint8)
    slack = np.ones((2 * REUSE_SLACK + 1,) * 2, np.uint8)
    source: list[int | None] = []
    cover: list[NDArray[np.uint8]] = []  # the last computed frame's mask, grown by REUSE_SLACK
    for index, mask in enumerate(grown):
        if not mask.any():
            source.append(None)
            continue
        earlier = source[-1] if source else None
        inside = REUSE_ANY_MASK and earlier is not None and not (mask & ~cover[0]).any()
        if earlier is not None and (inside or np.array_equal(mask, grown[index - 1])):
            both = mask | grown[earlier]
            around = np.asarray(cv2.dilate(both, ring), dtype=bool) & ~both.astype(bool)
            moved = changed(bands[index], bands[earlier])[around]
            if not moved.any() or moved.mean() <= REUSE_CHANGED:
                source.append(earlier)
                continue
        source.append(index)
        if REUSE_ANY_MASK:
            cover[:] = [np.asarray(cv2.dilate(mask, slack), dtype=np.uint8)]
    return source


def _scale(height: int, width: int, tiles: int) -> float:
    """The scale fitting a ``height`` x ``width`` band into ``tiles`` overlapping tiles."""
    widest = TILE_WIDTH + (tiles - 1) * TILE_WIDTH * 3 // 4
    return min(TILE_HEIGHT / height, widest / width)


def texture(bands: NDArray[np.uint8], grown: NDArray[np.uint8], frames: Sequence[int]) -> float:
    """Detail one tile across the band would lose around the masks, over ``SAMPLES`` frames.

    The largest mean absolute difference between a frame and its round trip through the
    one-tile scale, on a ring ``RING_PIXELS`` wide outside the masks (whose own spread is
    left out), so only the background the fill continues counts.
    """
    height, width = grown.shape[1:]
    scale = _scale(height, width, 1)
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    # The round trip spreads the caption's own pixels about two small pixels outwards.
    near = np.ones((2 * math.ceil(2 / scale) + 1,) * 2, np.uint8)
    far = np.ones((2 * RING_PIXELS + 1,) * 2, np.uint8)
    score = 0.0
    for frame in sorted({frames[round(i)] for i in np.linspace(0, len(frames) - 1, SAMPLES)}):
        ring = (cv2.dilate(grown[frame], far) > 0) & (cv2.dilate(grown[frame], near) == 0)
        if ring.any():
            band = bands[frame]
            trip = cv2.resize(
                cv2.resize(band, small, interpolation=cv2.INTER_AREA), (width, height),
                interpolation=cv2.INTER_CUBIC,
            )  # fmt: skip
            lost = cv2.absdiff(band, trip).max(axis=2)
            score = max(score, float(lost[ring].mean()))
    return score


def captions(grown: NDArray[np.uint8], source: Sequence[int | None]) -> list[list[int]]:
    """Frame ranges ``[first, last)`` of one caption: split where nothing is filled, or where the
    masks overlap the previous frame's by less than ``CHANGE`` after at least ``SHORTEST`` frames.

    A reused fill has the previous frame's exact masks, so it never starts a range.
    """
    ranges: list[list[int]] = []
    for index, frame in enumerate(source):
        if frame is None:
            continue
        if ranges and ranges[-1][1] == index:
            now, before = grown[index].view(bool), grown[index - 1].view(bool)
            same = np.count_nonzero(now & before) >= CHANGE * np.count_nonzero(now | before)
            if index - ranges[-1][0] < SHORTEST or same:
                ranges[-1][1] = index + 1
                continue
        ranges.append([index, index + 1])
    return ranges


def tile_plan(
    bands: NDArray[np.uint8], grown: NDArray[np.uint8], source: Sequence[int | None]
) -> list[tuple[int, int, int, int, int]]:
    """Fill groups ``(first, last, left, right, tiles)``: frames, band columns and tile count.

    Each caption is planned apart; consecutive captions with the same geometry form one group, so
    a chunk planned entirely at ``MAX_TILES`` across the band is filled exactly as one call. A
    geometry changes only where the caption does, where its fill changes shape anyway.
    """
    height, width = grown.shape[1:]
    whole = _scale(height, width, MAX_TILES)
    # The narrowest crop one tile covers without scaling the band up or past the tile's height.
    narrowest = min(width, max(TILE_WIDTH, math.ceil(TILE_WIDTH * height / TILE_HEIGHT)))
    groups: list[tuple[int, int, int, int, int]] = []
    for first, last in captions(grown, source):
        columns = np.flatnonzero(grown[first:last].any(axis=(0, 1)))
        left = max(0, int(columns[0]) - CONTEXT)
        right = min(width, int(columns[-1]) + 1 + CONTEXT)
        spare = max(0, narrowest - (right - left))
        left = max(0, min(left - spare // 2, width - max(narrowest, right - left)))
        right = max(right, left + narrowest)
        if right - left < width and _scale(height, right - left, 1) >= whole:
            geometry = (left, right, 1)
        elif texture(bands, grown, range(first, last)) < FLAT:
            geometry = (0, width, 1)
        else:
            geometry = (0, width, MAX_TILES)
        if groups and groups[-1][2:] == geometry:
            groups[-1] = (groups[-1][0], last, *geometry)
        else:
            groups.append((first, last, *geometry))
    return groups


def fill_band(
    inpaint: Tiler,
    bands: NDArray[np.uint8],
    masks: NDArray[np.bool_],
    *,
    in_place: bool = False,
    progress: Callable[[int], None] | None = None,
) -> NDArray[np.uint8]:
    """The bands with every masked pixel replaced by the model's temporally coherent fill.

    ``in_place`` writes the fill into ``bands`` and grows ``masks`` in place (for a caller that
    owns both), so neither is copied. ``progress`` receives, as the fill advances, how many
    leading frames of the result are final (never written again).
    """
    kernel = np.ones((2 * MASK_GROWTH + 1,) * 2, np.uint8)
    if in_place:
        grown = masks.view(np.uint8)
        for m in grown:
            cv2.dilate(m, kernel, dst=m)  # OpenCV dilates in place exactly
    else:
        grown = np.stack([cv2.dilate(m.view(np.uint8), kernel) for m in masks])
    source = plan_reuse(bands, grown)
    result = bands if in_place else bands.copy()
    report = progress if progress is not None else _ignore
    for group in tile_plan(bands, grown, source):
        report(group[0])  # groups follow each other: every earlier frame is final
        _fill(inpaint, bands, grown, source, result, group, report)
    report(len(result))
    return result


def _ignore(_: int) -> None:
    return None


def _fill(
    inpaint: Tiler,
    bands: NDArray[np.uint8],
    grown: NDArray[np.uint8],
    source: Sequence[int | None],
    result: NDArray[np.uint8],
    group: tuple[int, int, int, int, int],
    progress: Callable[[int], None],
) -> None:
    """Fill one group's frames within its columns into ``result`` (see ``tile_plan``)."""
    begin, end, left, right, tiles = group
    height, width = grown.shape[1], right - left
    scale = _scale(height, width, tiles)
    scaled_width = max(TILE_WIDTH, round(width * scale))
    scaled_height = min(TILE_HEIGHT, round(height * scale))
    top = (TILE_HEIGHT - scaled_height) // 2
    bottom = TILE_HEIGHT - scaled_height - top
    cut = slice(left, right)

    def fit(
        image: NDArray[np.uint8], interpolation: int, border: int, out: NDArray[np.uint8]
    ) -> None:
        small = cv2.resize(
            np.ascontiguousarray(image[:, cut]), (scaled_width, scaled_height),
            interpolation=interpolation,
        )  # fmt: skip
        cv2.copyMakeBorder(small, top, bottom, 0, 0, border, dst=out)  # straight into the stack

    computed = sorted({frame for frame in source[begin:end] if frame is not None})
    position = {frame: index for index, frame in enumerate(computed)}
    count = len(computed)  # the model runs only on these frames, in order
    small = np.empty((count, TILE_HEIGHT, scaled_width, 3), np.uint8)
    small_masks = np.empty((count, TILE_HEIGHT, scaled_width), np.uint8)

    def fit_frame(k: int, i: int) -> None:
        fit(bands[i], cv2.INTER_AREA, cv2.BORDER_REFLECT_101, small[k])
        fit(grown[i], cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, small_masks[k])

    # OpenCV releases the GIL, so frames are fitted on every core: on one, 900 frames took
    # 1.1 s (0.12 s on the pool) while the GPU waited for its first tile (2026-10-23).
    with ThreadPoolExecutor() as fitting:
        list(fitting.map(fit_frame, range(count), computed))
    offsets = starts(scaled_width)
    weights = ramp(offsets)
    # Only the band's rows of the tile are read back, so only they are accumulated (the padding
    # rows held ~45% of this float32 buffer, about 1 GB per 900-frame chunk).
    rows = slice(top, top + scaled_height)
    # Sums of the frames of the current window only (positions from ``base``): a frame is final
    # once the window after it starts, so the overlap is carried and the rest is composited.
    # A whole-chunk float32 buffer held about 1.7 GB for 900 frames; this holds one window.
    total = np.zeros((min(WINDOW, count), scaled_height, scaled_width, 3), np.float32)
    cover = np.zeros((min(WINDOW, count), 1, 1, 1), np.float32)
    base = 0
    spans = [(first, min(count, first + WINDOW)) for first in range(0, count, WINDOW - OVERLAP)]
    spans = spans[: next(i for i, (_, last) in enumerate(spans) if last == count) + 1]
    jobs = [
        (first, last, offset, w)
        for first, last in spans
        for offset, w in zip(offsets, weights, strict=True)
        if small_masks[first:last, :, offset : offset + TILE_WIDTH].any()
    ]
    done = begin  # frames before this one are composited

    def write(index: int) -> None:
        frame = source[index]
        if frame is None:
            return
        keep = grown[index][:, cut, None].view(bool)
        if frame != index:
            # The computed frame's fill where it was repaired, its source pixels elsewhere: the
            # surroundings of both are still (plan_reuse), so either is this frame's background.
            np.copyto(result[index][:, cut], result[frame][:, cut], where=keep)
            return
        at = position[frame] - base
        band = total[at] / np.maximum(cover[at], 1e-6)
        up = cv2.resize(band, (width, height), interpolation=cv2.INTER_CUBIC)
        np.clip(up, 0, 255, out=up)
        # A masked copy instead of gathering and scattering the masked pixels: 1.3 against
        # 2.4 ms per 1080-wide frame, the same bytes (2026-10-23).
        np.copyto(result[index][:, cut], up.astype(np.uint8), where=keep)

    def composite(final: int) -> None:
        """Write every frame whose fill (a computed position below ``final``) is complete."""
        nonlocal done
        last = done
        while last < end:
            frame = source[last]
            if frame is not None and position[frame] >= final:
                break
            last += 1
        # The frames are independent, so they are written in parallel (the last window's frames
        # wait for nothing but this once the GPU's last tile is back); a reused frame copies its
        # computed frame's result, so the computed frames are written first.
        frames = range(done, last)
        list(host.map(write, [index for index in frames if source[index] == index]))
        list(host.map(write, [index for index in frames if source[index] != index]))
        done = last
        progress(done)

    def predict(job: tuple[int, int, int, NDArray[np.float32]]) -> NDArray[np.float32]:
        first, last, offset, _ = job
        columns = slice(offset, offset + TILE_WIDTH)
        near = slice(max(0, offset - EDGE), offset + TILE_WIDTH + EDGE)
        needed = small_masks[first:last, :, near].any(axis=(1, 2))
        return inpaint.tile(
            small[first:last, :, columns], small_masks[first:last, :, columns], rows, needed
        )

    # The model runs one tile ahead on a worker thread, so blending and compositing (about a
    # fifth of a fill's time) overlap the GPU instead of following it.
    buffer = np.empty((min(WINDOW, count), scaled_height, scaled_width, 3), np.float32)
    with ThreadPoolExecutor(max_workers=1) as worker, ThreadPoolExecutor() as host:
        pending = deque(worker.submit(predict, job) for job in jobs[:2])
        queued = len(pending)
        for index, (first, last) in enumerate(spans):
            window = buffer[: last - first]  # one buffer reused by every window
            window.fill(0)
            across = np.zeros((1, 1, scaled_width, 1), np.float32)
            while pending and jobs[queued - len(pending)][0] == first:
                _, _, offset, w = jobs[queued - len(pending)]
                pred = pending.popleft().result()
                if queued < len(jobs):
                    pending.append(worker.submit(predict, jobs[queued]))
                    queued += 1
                part = pred  # the band's rows only
                part *= w[None, None, :, None]
                window[:, :, offset : offset + TILE_WIDTH] += part
                across[:, :, offset : offset + TILE_WIDTH] += w[None, None, :, None]
            frames_weight = np.ones((last - first, 1, 1, 1), np.float32)
            if first > 0:
                ramp_in = min(OVERLAP, last - first)
                frames_weight[:ramp_in, 0, 0, 0] = np.linspace(0, 1, ramp_in + 2)[1:-1]
            if last < count:
                frames_weight[-OVERLAP:, 0, 0, 0] = np.minimum(
                    frames_weight[-OVERLAP:, 0, 0, 0], np.linspace(1, 0, OVERLAP + 2)[1:-1]
                )
            window /= np.maximum(across, 1e-6)
            window *= frames_weight
            total[: last - first] += window
            cover[: last - first] += frames_weight
            final = spans[index + 1][0] if index + 1 < len(spans) else count
            composite(final)
            carried = last - final  # frames the next window adds to
            total[:carried] = total[final - first : last - first].copy()
            cover[:carried] = cover[final - first : last - first].copy()
            total[carried:] = 0
            cover[carried:] = 0
            base = final


def refill_bands(
    bands: Iterator[Image], masks: Sequence[Mask] | NDArray[np.bool_], inpaint: Tiler
) -> NDArray[np.uint8]:
    """Every frame's source band with its masked pixels refilled (bands and masks in order).

    Bands are copied one by one into a single array as they arrive, so the source blocks are
    released while stacking instead of all being held beside the stacked copy. A mask array is
    consumed (grown in place); a list of masks is stacked first.
    """
    return fill_band(inpaint, *_stacked(bands, masks), in_place=True)


def refill_stream(
    bands: Iterator[Image], masks: Sequence[Mask] | NDArray[np.bool_], inpaint: Tiler
) -> Iterator[Image]:
    """``refill_bands`` handing each frame's band over as soon as its fill is final.

    The fill runs on a worker thread and a band is yielded once no later write touches it, so
    the caller encodes while the model still fills later frames instead of after it (the
    decode/encode pass of a 900-frame 1080x1920 chunk took 6.9-8.5 s alone, 2026-10-22). The
    bands and their order are those of ``refill_bands``.
    """
    collected, stacked = _stacked(bands, masks)
    ready = threading.Condition()
    final = [0]

    def progress(done: int) -> None:
        with ready:
            final[0] = done
            ready.notify_all()

    def wake(_: object) -> None:
        with ready:
            ready.notify_all()

    # shortcut: a consumer that stops early waits for the rest of the fill; add a stop flag if
    # an aborted encode must fail faster.
    with ThreadPoolExecutor(max_workers=1) as worker:
        filled = worker.submit(
            fill_band, inpaint, collected, stacked, in_place=True, progress=progress
        )
        filled.add_done_callback(wake)
        for index in range(len(collected)):
            with ready:
                while final[0] <= index and not filled.done():
                    ready.wait()
            if final[0] <= index:
                filled.result()  # the fill's own error
                raise RuntimeError("The STTN fill ended before every frame was final")
            yield collected[index]
        filled.result()


def _stacked(
    bands: Iterator[Image], masks: Sequence[Mask] | NDArray[np.bool_]
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    stacked = np.asarray(masks, dtype=bool)
    collected: NDArray[np.uint8] | None = None
    count = 0
    for band in bands:
        if collected is None:
            collected = np.empty((len(stacked), *band.shape), dtype=np.uint8)
        if count == len(stacked) or band.shape != collected.shape[1:]:
            raise ValueError("Source bands and repair masks differ")
        collected[count] = band
        count += 1
    if collected is None or count != len(stacked) or collected.shape[:3] != stacked.shape:
        raise ValueError("Source bands and repair masks differ")
    return collected, stacked
