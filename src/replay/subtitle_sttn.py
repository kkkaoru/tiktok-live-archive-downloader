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
from collections import deque
from collections.abc import Iterator, Sequence
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
OVERLAP = 12  # frames shared by consecutive windows, blended linearly
# Two tiles across the band (scale ~0.70 instead of ~0.85): about half the GPU work per chunk;
# chunk 111 of 10-14b: 11 of 899 frames above 0.02 instability (5 with three), same look.
MAX_TILES = 2
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
"""
_metal: list[Any] = []  # the compiled library, built on first use in this process


class Tiler(Protocol):
    def tile(self, frames: NDArray[np.uint8], masks: NDArray[np.uint8]) -> NDArray[np.float32]: ...


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
    weight = torch.cat([self.query_embedding.weight, self.key_embedding.weight,
                        self.value_embedding.weight])  # fmt: skip
    bias = torch.cat([self.query_embedding.bias, self.key_embedding.bias,
                      self.value_embedding.bias])  # fmt: skip
    q_all, k_all, v_all = functional.conv2d(x, weight, bias).chunk(3, dim=1)
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

    return torch.tensor([*values, *z.stride()], dtype=torch.int32, device=z.device)


def _metal_ready(x: Any, b: int) -> bool:
    import torch

    return bool(x.device.type == "mps" and x.dtype == torch.bfloat16 and b == 1)


def metal_patch_attention(self: Any, x: Any, m: Any, b: int, c: int) -> Any:
    """``patch_attention`` with Metal token gather/scatter; same weights and math (bf16 MPS).

    Other devices, dtypes or batches fall back to ``patch_attention`` unchanged.
    """
    if not _metal_ready(x, b):
        return patch_attention(self, x, m, b, c)
    return self.output_linear(_metal_merged(self, x, c))


def _metal_merged(self: Any, x: Any, c: int) -> Any:
    """The attention's per-level results before ``output_linear`` (Metal path only)."""
    import torch
    import torch.nn.functional as functional

    frames_total, _, h, w = x.size()
    d_k = c // len(self.patchsize)
    weight = torch.cat([self.query_embedding.weight, self.key_embedding.weight,
                        self.value_embedding.weight])  # fmt: skip
    bias = torch.cat([self.query_embedding.bias, self.key_embedding.bias,
                      self.value_embedding.bias])  # fmt: skip
    qkv = functional.conv2d(x, weight, bias)
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
        scores = q @ k.transpose(0, 1) / math.sqrt(size)
        y = functional.softmax(scores, dim=-1) @ v
        library.scatter(y.view(torch.int16), merged.view(torch.int16),
                        _layout(merged, *geometry, 0), threads=n * height * width)  # fmt: skip
    return merged


def _slope(activation: Any) -> float:
    """The LeakyReLU slope as torch applies it to bfloat16 (0.2 becomes 0.2001953125)."""
    import torch

    return float(torch.tensor(activation.negative_slope, dtype=torch.bfloat16))


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

    def add_leaky(base: Any, value: Any, activation: Any) -> Any:
        if not _fusable(value):
            return base + activation(value)
        out = value.new_empty(value.shape)
        library.add_leaky(base, value, out, _slope(activation), threads=out.numel() // 4)
        return out

    x = add_leaky(x, linear[0](_metal_merged(self.attention, x, c)), linear[1])
    hidden = ff[0](x)
    if _fusable(hidden):
        library.leaky(hidden, _slope(ff[1]), threads=hidden.numel() // 4)
    else:
        hidden = ff[1](hidden)
    x = add_leaky(x, ff[2](hidden), ff[3])
    return {"x": x, "m": m, "b": b, "c": c}


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
        # bfloat16 keeps the model's -1e9 attention mask representable (float16 does not).
        self._dtype = torch.bfloat16
        self._network = network.to(self._device, dtype=self._dtype).eval()

    def tile(self, frames: NDArray[np.uint8], masks: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Inpaint one 432x240 tile sequence; returns the predicted pixels (0..255)."""
        torch = self._torch
        count = len(frames)
        # Frames travel as uint8 and are converted on the GPU; predictions are summed there and
        # copied back once, so each window costs one transfer each way.
        x = torch.from_numpy(frames).to(self._device).float().div(255).mul(2).sub(1)
        # Contiguous NCHW: on the channels-last view attention took about 110 ms against 62 ms.
        x = x.permute(0, 3, 1, 2).contiguous().to(self._dtype)
        m = torch.from_numpy(masks).to(self._device).unsqueeze(1).to(self._dtype)
        out = torch.zeros((count, 3, TILE_HEIGHT, TILE_WIDTH), device=self._device)
        weight = torch.zeros((count, 1, 1, 1), device=self._device)
        with torch.no_grad():
            features = self._network.encoder(x * (1 - m))
            for centre in range(NEIGHBORS // 2, count + NEIGHBORS, 2 * NEIGHBORS):
                near = list(range(max(0, centre - NEIGHBORS), min(count, centre + NEIGHBORS + 1)))
                step = REFERENCE_STEP
                references = (
                    [] if step is None else [i for i in range(0, count, step) if i not in near]
                )
                predicted = self._network.infer(features[near + references], m[near + references])
                decoded = torch.tanh(self._network.decoder(predicted[: len(near)])).float()
                out[near] += (decoded + 1) / 2 * 255
                weight[near] += 1
            result = (out / weight.clamp(min=1)).permute(0, 2, 3, 1).cpu().numpy()
        # torch keeps freed MPS buffers cached: 5.05 GiB were held after a chunk's fill, 2.04 GiB
        # with this, at the same speed and with identical output (measured 2026-10-20).
        torch.mps.empty_cache()
        return np.asarray(result, dtype=np.float32)


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
    pixels are exactly those of the previous frame, and whose surroundings (``RING_PIXELS``
    around them) have no pixel changed beyond ``SAME`` since the frame that was computed, reuses
    that frame's fill instead of running the model again.
    """
    ring = np.ones((2 * RING_PIXELS + 1,) * 2, np.uint8)
    source: list[int | None] = []
    for index, mask in enumerate(grown):
        if not mask.any():
            source.append(None)
            continue
        earlier = source[-1] if source else None
        if earlier is not None and np.array_equal(mask, grown[index - 1]):
            around = np.asarray(cv2.dilate(mask, ring), dtype=bool) & ~mask.astype(bool)
            if not changed(bands[index], bands[earlier])[around].any():
                source.append(earlier)
                continue
        source.append(index)
    return source


def fill_band(
    inpaint: Tiler, bands: NDArray[np.uint8], masks: NDArray[np.bool_], *, in_place: bool = False
) -> NDArray[np.uint8]:
    """The bands with every masked pixel replaced by the model's temporally coherent fill.

    ``in_place`` writes the fill into ``bands`` and grows ``masks`` in place (for a caller that
    owns both), so neither is copied.
    """
    count, height, width = masks.shape
    widest = TILE_WIDTH + (MAX_TILES - 1) * TILE_WIDTH * 3 // 4  # still covered by MAX_TILES
    scale = min(TILE_HEIGHT / height, widest / width)
    scaled_width = max(TILE_WIDTH, round(width * scale))
    scaled_height = min(TILE_HEIGHT, round(height * scale))
    top = (TILE_HEIGHT - scaled_height) // 2
    bottom = TILE_HEIGHT - scaled_height - top

    def fit(
        image: NDArray[np.uint8], interpolation: int, border: int, out: NDArray[np.uint8]
    ) -> None:
        small = cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)
        cv2.copyMakeBorder(small, top, bottom, 0, 0, border, dst=out)  # straight into the stack

    kernel = np.ones((2 * MASK_GROWTH + 1,) * 2, np.uint8)
    if in_place:
        grown = masks.view(np.uint8)
        for m in grown:
            cv2.dilate(m, kernel, dst=m)  # OpenCV dilates in place exactly
    else:
        grown = np.stack([cv2.dilate(m.view(np.uint8), kernel) for m in masks])
    source = plan_reuse(bands, grown)
    computed = sorted({frame for frame in source if frame is not None})
    if not computed:
        return bands if in_place else bands.copy()
    position = {frame: index for index, frame in enumerate(computed)}
    count = len(computed)  # the model runs only on these frames, in order
    small = np.empty((count, TILE_HEIGHT, scaled_width, 3), np.uint8)
    small_masks = np.empty((count, TILE_HEIGHT, scaled_width), np.uint8)
    for k, i in enumerate(computed):
        fit(bands[i], cv2.INTER_AREA, cv2.BORDER_REFLECT_101, small[k])
        fit(grown[i], cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, small_masks[k])
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
    result = bands if in_place else bands.copy()
    done = 0  # frames before this one are composited

    def composite(final: int) -> None:
        """Write every frame whose fill (a computed position below ``final``) is complete."""
        nonlocal done
        while done < len(source):
            frame = source[done]
            if frame is not None:
                if position[frame] >= final:
                    return
                keep = grown[done].view(bool)
                at = position[frame] - base
                band = total[at] / np.maximum(cover[at], 1e-6)
                up = cv2.resize(band, (width, height), interpolation=cv2.INTER_CUBIC)
                result[done][keep] = np.clip(up[keep], 0, 255).astype(np.uint8)
            done += 1

    def predict(job: tuple[int, int, int, NDArray[np.float32]]) -> NDArray[np.float32]:
        first, last, offset, _ = job
        columns = slice(offset, offset + TILE_WIDTH)
        return inpaint.tile(small[first:last, :, columns], small_masks[first:last, :, columns])

    # The model runs one tile ahead on a worker thread, so blending and compositing (about a
    # fifth of a fill's time) overlap the GPU instead of following it.
    buffer = np.empty((min(WINDOW, count), scaled_height, scaled_width, 3), np.float32)
    with ThreadPoolExecutor(max_workers=1) as worker:
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
                part = pred[:, rows]
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
    return result


def refill_bands(
    bands: Iterator[Image], masks: Sequence[Mask] | NDArray[np.bool_], inpaint: Tiler
) -> NDArray[np.uint8]:
    """Every frame's source band with its masked pixels refilled (bands and masks in order).

    Bands are copied one by one into a single array as they arrive, so the source blocks are
    released while stacking instead of all being held beside the stacked copy. A mask array is
    consumed (grown in place); a list of masks is stacked first.
    """
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
    return fill_band(inpaint, collected, stacked, in_place=True)
