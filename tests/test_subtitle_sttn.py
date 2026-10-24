"""STTN refill: tile layout, blending, masked-only replacement and the pinned model loader."""

import hashlib
import importlib.util
import threading
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

    def tile(
        self,
        frames: NDArray[np.uint8],
        masks: NDArray[np.uint8],
        rows: slice,
        needed: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]:
        self.calls += 1
        self.frames += len(frames)
        assert frames.shape[1:3] == (sttn.TILE_HEIGHT, sttn.TILE_WIDTH)
        return np.full((*frames.shape[:3], 3), 200, dtype=np.float32)[:, rows]


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
    out = painter.tile(frames, masks, slice(None))
    assert out.shape == (25, sttn.TILE_HEIGHT, sttn.TILE_WIDTH, 3)
    assert out.dtype == np.float32


ENCODER_ARCHITECTURE = """
import torch.nn as nn


class InpaintGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                                     nn.Conv2d(8, 3, 3, 2, 1))
        self.decoder = nn.Upsample(scale_factor=4)  # features at a quarter, as in the pinned STTN

    def infer(self, feat, masks):
        return feat * 0.5
"""


def _encoder_painter(tmp_path: Path) -> sttn.Inpainter:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / sttn.ARCHITECTURE).write_text(ENCODER_ARCHITECTURE, encoding="utf-8")
    module = sttn.load_architecture(tmp_path, sttn.sha256(tmp_path / sttn.ARCHITECTURE))
    torch.manual_seed(0)
    torch.save({"netG": module.InpaintGenerator().state_dict()}, tmp_path / sttn.CHECKPOINT)
    return sttn.Inpainter(
        tmp_path,
        checkpoint_sha=sttn.sha256(tmp_path / sttn.CHECKPOINT),
        architecture_sha=sttn.sha256(tmp_path / sttn.ARCHITECTURE),
    )


def _tile_inputs() -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 256, (27, sttn.TILE_HEIGHT, sttn.TILE_WIDTH, 3), dtype=np.uint8)
    masks = (rng.random((27, sttn.TILE_HEIGHT, sttn.TILE_WIDTH)) > 0.7).astype(np.uint8)
    return frames, masks


def test_the_encoder_in_frame_passes_equals_one_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    painter = _encoder_painter(tmp_path)
    frames, masks = _tile_inputs()
    whole = painter.tile(frames, masks, slice(None))
    monkeypatch.setattr(sttn, "ENCODE_FRAMES", 4)  # 27 frames: six full passes and a partial one
    assert np.array_equal(painter.tile(frames, masks, slice(None)), whole)
    rows = slice(30, 211)
    assert np.array_equal(painter.tile(frames, masks, rows), whole[:, rows])


def test_reference_frames_follow_the_neighbours_they_do_not_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    painter = _encoder_painter(tmp_path)  # its transformer treats frames apart
    frames, masks = _tile_inputs()
    plain = painter.tile(frames, masks, slice(None))
    monkeypatch.setattr(sttn, "REFERENCE_STEP", 4)
    assert np.array_equal(painter.tile(frames, masks, slice(None)), plain)


def test_each_run_is_encoded_alone_and_a_skipped_run_is_not_encoded(tmp_path: Path) -> None:
    painter = _encoder_painter(tmp_path)
    frames, masks = _tile_inputs()  # 27 frames: runs 0-15 and 15-26
    needed = np.zeros(len(frames), bool)
    needed[[3, 4]] = True
    encoded: list[int] = []
    painter._network.encoder.register_forward_hook(
        lambda _module, inputs, _output: encoded.append(len(inputs[0]))
    )
    painter.tile(frames, masks, slice(None), needed)
    assert encoded == [16]


def test_the_metal_input_pass_equals_torch_arithmetic_for_any_bytes(tmp_path: Path) -> None:
    painter = _encoder_painter(tmp_path)
    rng = np.random.default_rng(5)
    frames = rng.integers(0, 256, (9, 16, 24, 3), dtype=np.uint8)
    masks = rng.integers(0, 256, (9, 16, 24), dtype=np.uint8)  # 0/1 in use; any byte holds
    masks[:4] = masks[:4] > 128
    pixels, marks = torch.from_numpy(frames).to("mps"), torch.from_numpy(masks).to("mps")
    with torch.no_grad():
        x = pixels.permute(0, 3, 1, 2).contiguous().float().div_(255).mul_(2).sub_(1)
        m = marks.unsqueeze(1).to(torch.bfloat16)
        expected = painter._network.encoder(x.to(torch.bfloat16) * (1 - m))
        assert torch.equal(painter._encode(pixels, marks), expected)


def test_the_metal_accumulation_equals_torch_arithmetic_for_every_bfloat16() -> None:
    every = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    every = every.to("mps").reshape(4, 128, 128)
    sums = torch.rand((4, 128, 128)).mul(5000).to("mps")
    expected = sums + (every.float() + 1) / 2 * 255
    sttn._library().accumulate(
        every.view(torch.int16), sttn.output_table("mps"), sums, threads=sums.numel()
    )
    assert torch.equal(sums.nan_to_num(nan=-1.0), expected.nan_to_num(nan=-1.0))
    assert torch.equal(sums.isnan(), expected.isnan())


def test_the_mps_cache_is_emptied_after_the_tile_tensors_are_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    painter = _encoder_painter(tmp_path)
    frames, masks = _tile_inputs()
    before = torch.mps.current_allocated_memory()
    live: list[int] = []
    monkeypatch.setattr(
        torch.mps, "empty_cache", lambda: live.append(torch.mps.current_allocated_memory())
    )
    painter.tile(frames, masks, slice(None))
    assert len(live) == 1 and live[0] <= before  # no tensor of the window is left


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


def test_a_changed_caption_is_computed_again(monkeypatch: pytest.MonkeyPatch) -> None:
    bands = np.zeros((3, 60, 120, 3), np.uint8)
    grown = [np.zeros((60, 120), np.uint8) for _ in range(3)]
    for index, mask in enumerate(grown):
        mask[20:30, 40 : 60 + index] = 1
    monkeypatch.setattr(sttn, "REUSE_ANY_MASK", False)
    assert sttn.plan_reuse(bands, grown) == [0, 1, 2]


def test_a_shrinking_caption_over_still_surroundings_reuses_the_earlier_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sttn, "REUSE_ANY_MASK", True)
    bands = np.zeros((4, 60, 120, 3), np.uint8)
    grown = [np.zeros((60, 120), np.uint8) for _ in range(4)]
    for index, mask in enumerate(grown):
        mask[20:30, 40 : 80 - index] = 1
    bands[3, 10, 45] = 200  # motion beside the caption: computed again
    assert sttn.plan_reuse(bands, grown) == [0, 0, 0, 3]


def test_a_mask_jittering_within_the_slack_reuses_the_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sttn, "REUSE_ANY_MASK", True)
    bands = np.zeros((3, 60, 120, 3), np.uint8)
    grown = np.zeros((3, 60, 120), np.uint8)
    grown[0, 20:30, 40:80] = 1
    grown[1, 18:32, 38:82] = 1  # boxes grown by 2 pixels: inside the slack
    grown[2, 20:30, 40 : 80 + sttn.REUSE_SLACK + 1] = 1  # beyond it: computed again
    assert sttn.plan_reuse(bands, grown) == [0, 0, 2]


def test_scrolling_text_beside_a_jittering_mask_is_filled_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sttn, "REUSE_ANY_MASK", True)
    bands = np.zeros((2, 80, 160, 3), np.uint8)
    for row in range(5, 75, 6):  # lines of text above and below the caption
        bands[0, row : row + 2, ::5] = 255
        bands[1, row + 3 : row + 5, ::5] = 255  # scrolled by 3 pixels
    grown = np.zeros((2, 80, 160), np.uint8)
    grown[0, 30:45, 40:120] = 1
    grown[1, 31:46, 40:120] = 1  # the mask follows within the slack
    assert sttn.plan_reuse(bands, grown) == [0, 1]


def test_a_mask_growing_over_new_text_is_filled_for_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sttn, "REUSE_ANY_MASK", True)
    bands = np.full((3, 120, 400, 3), 10, dtype=np.uint8)
    masks = np.zeros((3, 120, 400), dtype=bool)
    masks[:, 50:60, 100:200] = True
    masks[1:, 50:60, 200:220] = True  # text appears beside the caption: the mask grows over it
    bands[1:, 55, 210] = 255  # in the computed frame that pixel is visible background
    masks[2, 50:60, 100:110] = False  # then shrinks inside frame 1's mask
    assert sttn.plan_reuse(bands, masks.view(np.uint8)) == [0, 1, 1]
    painter = Flat()
    result = sttn.refill_bands(iter(bands), list(masks), painter)
    assert painter.frames == 2 * painter.calls  # frames 0 and 1 computed, frame 2 reused
    assert (result[1:, 55, 210] == 200).all()  # filled, never the earlier frame's background
    assert (result[:, 55, 150] == 200).all()


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


def test_frozen_projections_and_cached_layouts_equal_the_per_call_path_on_mps(
    tmp_path: Path,
) -> None:
    import torch

    module, x = _attention_pair(tmp_path, "mps")
    x = x.to("mps", dtype=torch.bfloat16)
    sttn._layouts.clear()
    with torch.no_grad():
        per_call = sttn.metal_patch_attention(module, x, None, 1, 4)
        cached_layouts = len(sttn._layouts)
        sttn.freeze_projections(module)
        frozen = sttn.metal_patch_attention(module, x, None, 1, 4)
    assert torch.equal(frozen, per_call)
    assert cached_layouts == 2  # the gather level's token and merged layouts
    assert len(sttn._layouts) == 2  # reused, not rebuilt
    assert sttn._projection(module) is vars(module)["frozen_projection"]


def test_a_layout_is_cached_per_geometry_and_strides() -> None:
    import torch

    sttn._layouts.clear()
    z = torch.zeros(2, 3, 4)
    first = sttn._layout(z, 5, 6)
    assert sttn._layout(z, 5, 6) is first
    assert first.tolist() == [5, 6, 12, 4, 1]
    assert sttn._layout(z, 5, 7).tolist() == [5, 7, 12, 4, 1]
    assert sttn._layout(z.transpose(1, 2), 5, 6).tolist() == [5, 6, 12, 1, 4]
    assert len(sttn._layouts) == 3


def test_freezing_follows_the_current_weights(tmp_path: Path) -> None:
    import torch

    module, x = _attention_pair(tmp_path, "cpu")
    x = x.to(torch.bfloat16)
    sttn.freeze_projections(module)
    with torch.no_grad():
        module.get_parameter("query_embedding.weight").mul_(2)
        stale = sttn.patch_attention(module, x, None, 1, 4)
        sttn.freeze_projections(module)
        refrozen = sttn.patch_attention(module, x, None, 1, 4)
        vars(module)["frozen_projection"] = None
        per_call = sttn.patch_attention(module, x, None, 1, 4)
    assert torch.equal(refrozen, per_call)
    assert not torch.equal(stale, per_call)


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


@pytest.mark.parametrize("rows", [(0, 12), (0, 5), (4, 9), (7, 12), (6, 7)])
def test_the_last_block_on_some_rows_equals_those_rows_of_the_whole_block(
    tmp_path: Path, rows: tuple[int, int]
) -> None:
    import torch

    block, x, _, m = _block_pair(tmp_path, "mps")
    state = {"x": x, "m": m, "b": 1, "c": 4}
    with torch.no_grad():
        whole = block(state)["x"]
        part = block({**state, "rows": rows})["x"]
    assert torch.equal(part, whole[:, :, rows[0] : rows[1]])


def test_the_transformer_runs_its_last_block_on_the_decoded_rows_only(tmp_path: Path) -> None:
    import torch

    block, x, _, _ = _block_pair(tmp_path, "mps")

    class Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = torch.nn.Sequential(block, block)

        def infer(self, feat: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
            m = torch.nn.functional.interpolate(masks, scale_factor=1.0 / 4)
            out: torch.Tensor = self.transformer({"x": feat, "m": m, "b": 1, "c": 4})["x"]
            return out

    host = sttn.Inpainter.__new__(sttn.Inpainter)  # the network only, without the pinned files
    host._network, host._torch = Network(), torch
    host._blank = torch.zeros((1, 1, 48, 64), device="mps", dtype=torch.bfloat16)
    with torch.no_grad():
        whole = host._network.infer(x, host._blank.expand(len(x), -1, -1, -1))
        part = host._infer(x, (3, 10))
        full = host._infer(x, (0, sttn.TILE_HEIGHT // 4))
    assert torch.equal(part, whole[:, :, 3:10])
    assert torch.equal(full, whole)


def test_fused_block_falls_back_off_mps(tmp_path: Path) -> None:
    import torch

    block, x, reference, m = _block_pair(tmp_path, "cpu")
    state = {"x": x, "m": m, "b": 1, "c": 4}
    with torch.no_grad():
        assert torch.equal(block(state)["x"], reference(state)["x"])


@pytest.mark.parametrize(
    ("dtype", "suffix", "rounded"),
    [(torch.bfloat16, "", 0.2001953125), (torch.float16, "_half", 0.199951171875)],
)
def test_the_activation_kernels_equal_torch_bit_for_bit(
    dtype: torch.dtype, suffix: str, rounded: float
) -> None:
    import torch.nn.functional as functional

    torch.manual_seed(0)
    x = torch.randn(4, 8, 6, 6, device="mps", dtype=dtype) * 3
    y = torch.randn_like(x) * 3
    slope = sttn._slope(torch.nn.LeakyReLU(0.2), dtype)
    leaky = y.clone()
    getattr(sttn._library(), "leaky" + suffix)(leaky, slope, threads=leaky.numel() // 4)
    out = torch.empty_like(x)
    getattr(sttn._library(), "add_leaky" + suffix)(x, y, out, slope, threads=out.numel() // 4)
    assert slope == rounded
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

    def tile(
        self,
        frames: NDArray[np.uint8],
        masks: NDArray[np.uint8],
        rows: slice,
        needed: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]:
        order = np.arange(len(frames), dtype=np.float32)[:, None, None, None] % 7
        full = frames * 0.7 + masks[..., None] * 31.0 + order
        return np.asarray(full[:, rows], dtype=np.float32)


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


def _streamed_case() -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Three windows, caption-free gaps between groups, reused fills and a crop group."""
    count = 2 * sttn.WINDOW + 30
    rng = np.random.default_rng(4)
    bands = rng.integers(0, 255, (count, 40, 300, 3), dtype=np.uint8)
    masks = np.zeros((count, 40, 300), dtype=bool)
    masks[:, 10:20, 30:150] = True
    masks[60:90] = False
    masks[300:400, 10:20, 30:150] = False
    masks[300:400, 5:30, 200:290] = True
    bands[200:230] = bands[199]
    return bands, masks


def test_streamed_refill_hands_over_the_refilled_bands_once_final() -> None:
    bands, masks = _streamed_case()
    expected = sttn.refill_bands(iter(list(bands)), masks.copy(), Varying())
    # Each band is copied when handed over: equal copies prove no later write changed it.
    streamed = [band.copy() for band in sttn.refill_stream(iter(list(bands)), masks, Varying())]
    assert np.array_equal(np.stack(streamed), expected)


class Gated(Varying):
    """Holds every tile until the consumer has received a band."""

    def __init__(self) -> None:
        self.finished = 0
        self.open = threading.Event()

    def tile(
        self,
        frames: NDArray[np.uint8],
        masks: NDArray[np.uint8],
        rows: slice,
        needed: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]:
        assert self.open.wait(10), "no band was handed over before the fill ended"
        self.finished += 1
        return super().tile(frames, masks, rows)


def test_streamed_refill_hands_over_bands_while_the_fill_continues() -> None:
    bands, masks = _streamed_case()
    masks[:30] = False  # caption-free leading frames are final before any tile is filled
    painter = Gated()
    stream = sttn.refill_stream(iter(list(bands)), masks, painter)
    first = next(stream).copy()
    assert painter.finished == 0
    painter.open.set()
    assert np.array_equal(first, bands[0])
    assert len(list(stream)) == len(bands) - 1
    assert painter.finished > 0


class Failing(Varying):
    def tile(
        self,
        frames: NDArray[np.uint8],
        masks: NDArray[np.uint8],
        rows: slice,
        needed: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]:
        raise RuntimeError("model failed")


def test_streamed_refill_raises_the_fill_error() -> None:
    bands, masks = _streamed_case()
    with pytest.raises(RuntimeError, match="model failed"):
        list(sttn.refill_stream(iter(list(bands)), masks, Failing()))


def test_streamed_refill_refuses_a_fill_that_ends_without_final_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sttn, "fill_band", lambda *args, **kwargs: args[1])
    bands, masks = _streamed_case()
    with pytest.raises(RuntimeError, match="before every frame was final"):
        list(sttn.refill_stream(iter(list(bands)), masks, Varying()))


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


def caption_masks(count: int, columns: slice) -> NDArray[np.uint8]:
    masks = np.zeros((count, 60, 1200), np.uint8)
    masks[:, 20:40, columns] = 1
    return masks


def test_captions_split_where_nothing_is_filled_or_the_caption_changes() -> None:
    grown = caption_masks(100, slice(100, 400))
    grown[50:] = caption_masks(50, slice(700, 1000))  # another caption elsewhere
    grown[10] = caption_masks(1, slice(700, 1000))[0]  # a change too soon after the start
    source: list[int | None] = list(range(100))
    source[80:85] = [None] * 5
    assert sttn.captions(grown, source) == [[0, 50], [50, 80], [85, 100]]


def test_a_narrow_caption_is_cropped_to_one_tile_and_a_wide_one_follows_the_texture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flat = np.full((90, 60, 1200, 3), 90, np.uint8)
    grown = caption_masks(90, slice(500, 561))
    grown[30:] = caption_masks(60, slice(100, 1101))
    grown[60:] = caption_masks(30, slice(150, 1101))
    source: list[int | None] = list(range(90))
    monkeypatch.setattr(sttn, "FLAT", 1.0)
    assert sttn.tile_plan(flat, grown, source) == [(0, 30, 315, 747, 1), (30, 90, 0, 1200, 1)]
    monkeypatch.setattr(sttn, "FLAT", 0.0)
    assert sttn.tile_plan(flat, grown, source) == [(0, 30, 315, 747, 1), (30, 90, 0, 1200, 2)]


def test_a_crop_stays_inside_the_band_at_its_edges() -> None:
    bands = np.zeros((5, 60, 1200, 3), np.uint8)
    assert sttn.tile_plan(bands, caption_masks(5, slice(1150, 1200)), list(range(5))) == [
        (0, 5, 768, 1200, 1)
    ]
    assert sttn.tile_plan(bands, caption_masks(5, slice(0, 30)), list(range(5))) == [
        (0, 5, 0, 432, 1)
    ]


def test_texture_measures_the_surroundings_not_the_caption() -> None:
    grown = caption_masks(4, slice(400, 800))
    flat = np.full((4, 60, 1200, 3), 120, np.uint8)
    flat[grown.astype(bool)] = 255  # the caption itself does not count
    striped = flat.copy()
    striped[:, :, ::8] = 0
    striped[grown.astype(bool)] = 255
    assert sttn.texture(flat, grown, range(4)) == 0.0
    assert sttn.texture(striped, grown, range(4)) > 20
    assert sttn.texture(flat, np.ones_like(grown), range(4)) == 0.0  # no surroundings at all


def test_a_cropped_fill_replaces_only_masked_pixels_with_one_tile() -> None:
    bands = np.full((6, 60, 1200, 3), 10, np.uint8)
    masks = caption_masks(6, slice(500, 561)).astype(bool)
    painter = Flat()
    result = sttn.fill_band(painter, bands, masks)
    assert painter.calls == 1
    assert (result[:, 30, 530] == 200).all()
    assert (result[:, 30, 300] == 10).all()
    assert (result[:, 5, 530] == 10).all()


def test_the_compute_pin_names_every_choice_that_changes_the_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sttn, "DTYPE", "float16")
    monkeypatch.setattr(sttn, "NEIGHBORS", 7)
    monkeypatch.setattr(sttn, "SKIP_UNNEEDED", False)
    monkeypatch.setattr(sttn, "ROW_MARGIN", None)
    monkeypatch.setattr(sttn, "REUSE_CHANGED", 0.02)
    monkeypatch.setattr(sttn, "REUSE_ANY_MASK", False)
    monkeypatch.setattr(sttn, "OVERLAP", 12)
    assert sttn.compute() == "v1-float16-n7-noskip-rowsall-reuse0.02"
    monkeypatch.setattr(sttn, "OVERLAP", 6)
    assert sttn.compute() == "v1-float16-n7-noskip-rowsall-reuse0.02-o6"
    monkeypatch.setattr(sttn, "REUSE_ANY_MASK", True)
    monkeypatch.setattr(sttn, "SKIP_UNNEEDED", True)
    monkeypatch.setattr(sttn, "ROW_MARGIN", 4)
    assert (
        sttn.compute() == f"v1-float16-n7-skip{sttn.EDGE}-rowsm4-reuse0.02-any{sttn.REUSE_SLACK}-o6"
    )


def test_decoded_rows_are_whole_feature_rows_with_the_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sttn, "ROW_MARGIN", 2)
    assert sttn._decoded_rows(slice(35, 205)) == (24, 216)
    assert sttn._decoded_rows(slice(None)) == (0, sttn.TILE_HEIGHT)
    monkeypatch.setattr(sttn, "ROW_MARGIN", None)
    assert sttn._decoded_rows(slice(35, 205)) == (0, sttn.TILE_HEIGHT)


def test_unneeded_frames_are_not_decoded_and_needed_ones_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    painter = _encoder_painter(tmp_path)  # its transformer and decoder treat frames apart
    frames, masks = _tile_inputs()
    whole = painter.tile(frames, masks, slice(None))
    needed = np.zeros(len(frames), bool)
    needed[[3, 4, 25]] = True  # frames 11-21 form a run with nothing needed: it is not run
    calls: list[int] = []
    infer = painter._network.infer

    def counted(f: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        calls.append(len(f))
        result: torch.Tensor = infer(f, m)
        return result

    monkeypatch.setattr(painter._network, "infer", counted)
    skipped = painter.tile(frames, masks, slice(None), needed)
    assert np.array_equal(skipped[needed], whole[needed])
    assert not skipped[~needed].any()
    assert len(calls) == 2
    monkeypatch.setattr(sttn, "SKIP_UNNEEDED", False)
    assert np.array_equal(painter.tile(frames, masks, slice(None), needed), whole)


class Needy(Varying):
    """Leaves unneeded frames 0, as the model does when it skips them."""

    skipped = False

    def tile(
        self,
        frames: NDArray[np.uint8],
        masks: NDArray[np.uint8],
        rows: slice,
        needed: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float32]:
        result = super().tile(frames, masks, rows)
        if needed is not None:
            result[~needed] = 0
            self.skipped = self.skipped or not needed.all()
        return result


def test_frames_left_out_of_a_tile_are_never_read_from_it() -> None:
    count = 120
    rng = np.random.default_rng(6)
    bands = rng.integers(0, 255, (count, 60, 1080, 3), dtype=np.uint8)
    masks = np.zeros((count, 60, 1080), dtype=bool)
    masks[:, 20:40, 60:1000] = True  # a caption across both tiles
    masks[1::2, :, :800] = False  # every other frame only inside the right tile
    painter = Needy()
    expected = sttn.fill_band(Varying(), bands, masks)
    assert np.array_equal(sttn.fill_band(painter, bands, masks), expected)
    assert painter.skipped


def test_a_reuse_tolerates_the_configured_share_of_changed_surroundings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands = np.zeros((2, 100, 100, 3), np.uint8)
    bands[1, 10:12, 10:12] = 200  # 4 of the ring's 6656 pixels move
    grown = np.zeros((2, 100, 100), np.uint8)
    grown[:, 40:60, 40:60] = 1
    assert sttn.plan_reuse(bands, grown) == [0, 1]
    monkeypatch.setattr(sttn, "REUSE_CHANGED", 0.01)
    assert sttn.plan_reuse(bands, grown) == [0, 0]


def test_float16_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sttn, "DTYPE", "float16")
    with pytest.raises(ValueError, match="bfloat16 only: float16 overflows"):
        _encoder_painter(tmp_path)
