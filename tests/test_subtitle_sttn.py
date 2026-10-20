"""STTN refill: tile layout, blending, masked-only replacement and the pinned model loader."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from replay import subtitle_sttn as sttn

TINY_ARCHITECTURE = """
import torch
import torch.nn as nn


class InpaintGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 3, 1)
        self.decoder = nn.Identity()

    def infer(self, feat, masks):
        return feat
"""


class Flat:
    """A fake inpainter predicting one grey value everywhere."""

    def __init__(self) -> None:
        self.calls = 0
        self.frames = 0

    def tile(self, frames: NDArray[np.uint8], masks: NDArray[np.uint8]) -> NDArray[np.float32]:
        self.calls += 1
        self.frames += len(frames)
        assert frames.shape[1:3] == (sttn.TILE_HEIGHT, sttn.TILE_WIDTH)
        return np.full((*frames.shape[:3], 3), 200, dtype=np.float32)


def test_tiles_cover_the_width_with_overlaps() -> None:
    assert sttn.starts(300) == [0]
    offsets = sttn.starts(913)
    assert offsets[0] == 0
    assert offsets[-1] == 913 - sttn.TILE_WIDTH
    assert all(b - a < sttn.TILE_WIDTH for a, b in zip(offsets, offsets[1:], strict=False))


def test_blend_weights_sum_to_one_across_every_overlap() -> None:
    offsets = sttn.starts(913)
    total = np.zeros(913, np.float32)
    for offset, weight in zip(offsets, sttn.ramp(offsets), strict=True):
        total[offset : offset + sttn.TILE_WIDTH] += weight
    assert np.allclose(total, 1, atol=1e-5)


def test_only_masked_pixels_are_replaced() -> None:
    bands = np.full((70, 120, 400, 3), 10, dtype=np.uint8)
    masks = np.zeros((70, 120, 400), dtype=bool)
    masks[:, 50:60, 100:200] = True
    painter = Flat()
    result = sttn.refill_bands(iter(bands), list(masks), painter)
    assert painter.calls > 0
    assert (result[:, 55, 150] == 200).all()
    assert (result[:, 5, 5] == 10).all()


def test_tiles_without_masks_are_skipped() -> None:
    bands = np.zeros((10, 120, 2000, 3), dtype=np.uint8)
    masks = np.zeros((10, 120, 2000), dtype=bool)
    masks[:, 50:60, 10:30] = True
    painter = Flat()
    sttn.refill_bands(iter(bands), list(masks), painter)
    assert painter.calls == 1


def test_bands_and_masks_must_match() -> None:
    with pytest.raises(ValueError, match="differ"):
        sttn.refill_bands(
            iter(np.zeros((2, 10, 10, 3), np.uint8)), [np.zeros((10, 10), bool)], Flat()
        )


def write_model(tmp_path: Path) -> tuple[str, str]:
    import torch

    (tmp_path / sttn.ARCHITECTURE).write_text(TINY_ARCHITECTURE, encoding="utf-8")
    module = sttn.load_architecture(tmp_path, sttn.sha256(tmp_path / sttn.ARCHITECTURE))
    torch.save({"netG": module.InpaintGenerator().state_dict()}, tmp_path / sttn.CHECKPOINT)
    return sttn.sha256(tmp_path / sttn.CHECKPOINT), sttn.sha256(tmp_path / sttn.ARCHITECTURE)


def test_the_pinned_model_inpaints_a_tile_sequence(tmp_path: Path) -> None:
    checkpoint, architecture = write_model(tmp_path)
    painter = sttn.Inpainter(tmp_path, checkpoint_sha=checkpoint, architecture_sha=architecture)
    frames = np.zeros((25, sttn.TILE_HEIGHT, sttn.TILE_WIDTH, 3), np.uint8)
    masks = np.ones((25, sttn.TILE_HEIGHT, sttn.TILE_WIDTH), np.uint8)
    out = painter.tile(frames, masks)
    assert out.shape == (25, sttn.TILE_HEIGHT, sttn.TILE_WIDTH, 3)
    assert out.dtype == np.float32


def test_a_changed_checkpoint_or_architecture_is_refused(tmp_path: Path) -> None:
    checkpoint, architecture = write_model(tmp_path)
    with pytest.raises(ValueError, match="checkpoint differs"):
        sttn.Inpainter(tmp_path, checkpoint_sha="0" * 64, architecture_sha=architecture)
    with pytest.raises(ValueError, match="architecture differs"):
        sttn.Inpainter(tmp_path, checkpoint_sha=checkpoint, architecture_sha="0" * 64)


def test_sha256_is_the_content_digest(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"abc")
    assert sttn.sha256(tmp_path / "a") == hashlib.sha256(b"abc").hexdigest()


ORIGINAL_ATTENTION = """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def forward(self, query, key, value, m):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        scores.masked_fill(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1), nn.LeakyReLU(0.2))
        self.attention = Attention()

    def forward(self, x, m, b, c):
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        _query, _key, _value = (self.query_embedding(x), self.key_embedding(x),
                                self.value_embedding(x))
        for (width, height), query, key, value in zip(
                self.patchsize, torch.chunk(_query, len(self.patchsize), dim=1),
                torch.chunk(_key, len(self.patchsize), dim=1),
                torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height
            mm = m.view(b, t, 1, out_h, height, out_w, width)
            mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b, t*out_h*out_w, height*width)
            mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, t*out_h*out_w, 1)
            def tokens(z):
                z = z.view(b, t, d_k, out_h, height, out_w, width)
                return z.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                    b, t*out_h*out_w, d_k*height*width)
            y, _ = self.attention(tokens(query), tokens(key), tokens(value), mm)
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            output.append(y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w))
        return self.output_linear(torch.cat(output, 1))


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, m, b, c = x['x'], x['m'], x['b'], x['c']
        x = x + self.attention(x, m, b, c)
        x = x + self.feed_forward(x)
        return {'x': x, 'm': m, 'b': b, 'c': c}
"""


def test_the_patched_attention_equals_the_pinned_code(tmp_path: Path) -> None:
    import torch

    (tmp_path / sttn.ARCHITECTURE).write_text(ORIGINAL_ATTENTION, encoding="utf-8")
    sha = sttn.sha256(tmp_path / sttn.ARCHITECTURE)
    plain_spec = importlib.util.spec_from_file_location(
        "plain_attention", tmp_path / "architecture.py"
    )
    assert plain_spec is not None and plain_spec.loader is not None
    plain = importlib.util.module_from_spec(plain_spec)
    plain_spec.loader.exec_module(plain)
    patched = sttn.load_architecture(tmp_path, sha)
    torch.manual_seed(0)
    reference = plain.MultiHeadedAttention([(4, 2), (2, 1)], d_model=8)
    module = patched.MultiHeadedAttention([(4, 2), (2, 1)], d_model=8)
    module.load_state_dict(reference.state_dict())
    x = torch.randn(6, 8, 4, 8)
    m = (torch.rand(6, 1, 4, 8) > 0.5).float()
    with torch.no_grad():
        # Same math; CPU may round the merged projection differently (MPS matched bit for bit).
        assert torch.allclose(module(x, m, 2, 8), reference(x, m, 2, 8), atol=1e-6, rtol=0)


def test_frames_without_captions_are_not_computed() -> None:
    bands = np.zeros((4, 60, 120, 3), np.uint8)
    grown = [np.zeros((60, 120), np.uint8) for _ in range(4)]
    grown[2][20:30, 40:80] = 1
    bands[2] = 50  # moving, so the captioned frame is computed
    assert sttn.plan_reuse(bands, grown) == [None, None, 2, None]


def test_an_unchanged_caption_over_still_surroundings_reuses_the_fill() -> None:
    bands = np.zeros((4, 60, 120, 3), np.uint8)
    grown = [np.zeros((60, 120), np.uint8) for _ in range(4)]
    for mask in grown:
        mask[20:30, 40:80] = 1
    bands[3, 10, 45] = 200  # motion right beside the caption: computed again
    assert sttn.plan_reuse(bands, grown) == [0, 0, 0, 3]


def test_a_changed_caption_is_computed_again() -> None:
    bands = np.zeros((3, 60, 120, 3), np.uint8)
    grown = [np.zeros((60, 120), np.uint8) for _ in range(3)]
    for index, mask in enumerate(grown):
        mask[20:30, 40 : 60 + index] = 1
    assert sttn.plan_reuse(bands, grown) == [0, 1, 2]


def test_a_reused_fill_is_the_computed_frames_fill() -> None:
    bands = np.full((3, 120, 400, 3), 10, dtype=np.uint8)
    masks = np.zeros((3, 120, 400), dtype=bool)
    masks[:, 50:60, 100:200] = True
    painter = Flat()
    result = sttn.refill_bands(iter(bands), list(masks), painter)
    assert painter.frames == painter.calls  # one computed frame per tile
    assert (result[:, 55, 150] == 200).all()


def test_bands_without_any_caption_are_returned_unchanged() -> None:
    bands = np.full((2, 120, 400, 3), 10, dtype=np.uint8)
    painter = Flat()
    result = sttn.refill_bands(iter(bands), list(np.zeros((2, 120, 400), bool)), painter)
    assert painter.calls == 0
    assert np.array_equal(result, bands)


def test_changed_marks_only_differences_beyond_the_tolerance() -> None:
    previous = np.zeros((6, 4, 3), dtype=np.uint8)
    current = previous.copy()
    current[1, 2] = (0, 17, 0)
    current[4, 0] = (16, 0, 0)
    assert sttn.changed(current, previous).tolist() == [
        [False, False, False, False],
        [False, False, True, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
    assert not sttn.changed(previous, previous).any()


def _attention_pair(tmp_path: Path, device: str) -> tuple["torch.nn.Module", "torch.Tensor"]:
    import torch

    (tmp_path / sttn.ARCHITECTURE).write_text(ORIGINAL_ATTENTION, encoding="utf-8")
    patched = sttn.load_architecture(tmp_path, sttn.sha256(tmp_path / sttn.ARCHITECTURE))
    torch.manual_seed(0)
    # Level (16, 12) has 21 tokens (split reduction); level (4, 3) has 336 (Metal gather).
    module = patched.MultiHeadedAttention([(16, 12), (4, 3)], d_model=4)
    return module.to(device, dtype=torch.bfloat16).eval(), torch.randn(21, 4, 12, 16)


def test_metal_attention_matches_the_patched_attention_on_mps(tmp_path: Path) -> None:
    import torch

    module, x = _attention_pair(tmp_path, "mps")
    x = x.to("mps", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = sttn.patch_attention(module, x, None, 1, 4).float()
        metal = sttn.metal_patch_attention(module, x, None, 1, 4).float()
        layout = sttn.metal_patch_attention(
            module, x.to(memory_format=torch.channels_last), None, 1, 4
        ).float()
    assert torch.allclose(metal, expected, atol=0.05, rtol=0)
    assert torch.allclose(layout, expected, atol=0.05, rtol=0)


def test_metal_attention_falls_back_off_mps(tmp_path: Path) -> None:
    import torch

    module, x = _attention_pair(tmp_path, "cpu")
    x = x.to(torch.bfloat16)
    with torch.no_grad():
        expected = sttn.patch_attention(module, x, None, 1, 4)
        assert torch.equal(sttn.metal_patch_attention(module, x, None, 1, 4), expected)


def _block_pair(
    tmp_path: Path, device: str
) -> tuple["torch.nn.Module", "torch.Tensor", "torch.nn.Module", "torch.Tensor"]:
    import torch

    (tmp_path / sttn.ARCHITECTURE).write_text(ORIGINAL_ATTENTION, encoding="utf-8")
    plain_spec = importlib.util.spec_from_file_location("plain_block", tmp_path / sttn.ARCHITECTURE)
    assert plain_spec is not None and plain_spec.loader is not None
    plain = importlib.util.module_from_spec(plain_spec)
    plain_spec.loader.exec_module(plain)
    patched = sttn.load_architecture(tmp_path, sttn.sha256(tmp_path / sttn.ARCHITECTURE))
    torch.manual_seed(0)
    block = patched.TransformerBlock([(16, 12), (4, 3)], hidden=4)
    reference = plain.TransformerBlock([(16, 12), (4, 3)], hidden=4)
    reference.load_state_dict(block.state_dict())
    x = torch.randn(21, 4, 12, 16) * 3
    return (
        block.to(device, dtype=torch.bfloat16).eval(),
        x.to(device, dtype=torch.bfloat16),
        reference.to(device, dtype=torch.bfloat16).eval(),
        torch.zeros(21, 1, 12, 16, device=device, dtype=torch.bfloat16),
    )


def test_fused_block_equals_the_pinned_block_on_mps(tmp_path: Path) -> None:
    import torch

    block, x, reference, m = _block_pair(tmp_path, "mps")
    state = {"x": x, "m": m, "b": 1, "c": 4}
    before = x.clone()
    with torch.no_grad():
        fused = block(state)["x"]
        # The pinned attention with Metal gather/scatter, then the pinned block arithmetic.
        attended = x + sttn.metal_patch_attention(block.get_submodule("attention"), x, m, 1, 4)
        expected = attended + block.get_submodule("feed_forward")(attended)
        pinned = reference(state)["x"]
    assert torch.equal(fused, expected)
    assert torch.allclose(fused.float(), pinned.float(), atol=0.1, rtol=0)
    assert torch.equal(x, before)  # the caller's features are not written


def test_fused_block_falls_back_off_mps(tmp_path: Path) -> None:
    import torch

    block, x, reference, m = _block_pair(tmp_path, "cpu")
    state = {"x": x, "m": m, "b": 1, "c": 4}
    with torch.no_grad():
        assert torch.equal(block(state)["x"], reference(state)["x"])


def test_the_activation_kernels_equal_torch_bit_for_bit() -> None:
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(0)
    x = torch.randn(4, 8, 6, 6, device="mps", dtype=torch.bfloat16) * 3
    y = torch.randn_like(x) * 3
    slope = sttn._slope(torch.nn.LeakyReLU(0.2))
    leaky = y.clone()
    sttn._library().leaky(leaky, slope, threads=leaky.numel() // 4)
    out = torch.empty_like(x)
    sttn._library().add_leaky(x, y, out, slope, threads=out.numel() // 4)
    assert slope == 0.2001953125
    assert torch.equal(leaky, functional.leaky_relu(y, 0.2))
    assert torch.equal(out, x + functional.leaky_relu(y, 0.2))


def test_every_frame_is_filled_across_windows_gaps_and_reuse() -> None:
    count = sttn.WINDOW + 100  # 300 computed frames: two overlapping windows
    bands = np.zeros((count, 60, 300, 3), dtype=np.uint8)
    bands[:] = (np.arange(count) * 37 % 200).astype(np.uint8)[:, None, None, None]
    masks = np.zeros((count, 60, 300), dtype=bool)
    masks[:, 20:30, 40:120] = True
    masks[30:50] = False  # frames without captions are not computed
    bands[150:170] = bands[149]  # an unchanged caption over still surroundings reuses its fill
    painter = Flat()
    result = sttn.refill_bands(iter(bands), list(masks), painter)
    assert painter.calls == 2
    assert (result[:30, 25, 80] == 200).all()
    assert (result[30:50, 25, 80] == bands[30:50, 25, 80]).all()
    assert (result[50:, 25, 80] == 200).all()
    assert (result[:, 5, 5] == bands[:, 5, 5]).all()


def test_a_frame_compared_with_itself_has_no_change() -> None:
    frame = np.full((4, 5, 3), 9, dtype=np.uint8)
    assert sttn.changed(frame, frame).tolist() == [[False] * 5] * 4


def test_refill_writes_into_the_callers_bands_when_asked() -> None:
    bands = np.full((3, 60, 300, 3), 10, dtype=np.uint8)
    masks = np.zeros((3, 60, 300), dtype=bool)
    masks[:, 20:30, 40:120] = True
    result = sttn.fill_band(Flat(), bands, masks, in_place=True)
    assert result is bands
    assert (bands[:, 25, 80] == 200).all()
    assert (bands[:, 5, 5] == 10).all()


class Varying:
    """A fake inpainter whose prediction depends on the frames, masks and window position."""

    def tile(self, frames: NDArray[np.uint8], masks: NDArray[np.uint8]) -> NDArray[np.float32]:
        order = np.arange(len(frames), dtype=np.float32)[:, None, None, None] % 7
        return np.asarray(frames * 0.7 + masks[..., None] * 31.0 + order, dtype=np.float32)


def test_refill_over_three_windows_equals_the_plain_fill() -> None:
    count = 2 * sttn.WINDOW + 30
    rng = np.random.default_rng(0)
    bands = rng.integers(0, 255, (count, 40, 300, 3), dtype=np.uint8)
    masks = np.zeros((count, 40, 300), dtype=bool)
    masks[:, 10:20, 30:150] = True
    masks[60:90] = False
    bands[200:230] = bands[199]
    expected = sttn.fill_band(Varying(), bands, masks)
    result = sttn.refill_bands(iter(list(bands)), masks.copy(), Varying())
    assert np.array_equal(result, expected)


def test_refill_rejects_bands_that_do_not_match_the_masks() -> None:
    masks = np.zeros((2, 4, 5), dtype=bool)
    with pytest.raises(ValueError, match="differ"):
        sttn.refill_bands(iter([np.zeros((4, 5, 3), np.uint8)] * 3), masks, Flat())
    with pytest.raises(ValueError, match="differ"):
        sttn.refill_bands(
            iter([np.zeros((4, 5, 3), np.uint8), np.zeros((3, 5, 3), np.uint8)]), masks, Flat()
        )
    with pytest.raises(ValueError, match="differ"):
        sttn.refill_bands(iter([]), masks, Flat())
