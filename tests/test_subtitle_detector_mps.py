"""The batched MPS detector decodes injected maps with OpenCV and reuses the production mask."""

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from replay import subtitle_detector_mps as mps
from replay.local_asr import fingerprint

ROI = np.zeros((196, 1077, 3), dtype=np.uint8)
TALL = np.zeros((300, 100, 3), dtype=np.uint8)


def _bar(batch: NDArray[np.float32]) -> NDArray[np.float32]:
    count, _, height, width = batch.shape
    maps = np.zeros((count, 1, height, width), dtype=np.float32)
    maps[:, :, height // 4 : height // 2, width // 8 : width // 2] = 0.95
    return maps


def _blank(batch: NDArray[np.float32]) -> NDArray[np.float32]:
    count, _, height, width = batch.shape
    return np.zeros((count, 1, height, width), dtype=np.float32)


def test_detect_masks_the_decoded_text_bar_on_the_canvas() -> None:
    detector = mps.MpsTextDetector(_bar)
    (mask,) = detector.detect([ROI])
    assert mask.shape == (196, 1077)
    assert mask.dtype == np.uint8
    assert int(np.count_nonzero(mask)) > 5000
    assert detector.grid_sizes == [(736, 128)]
    assert detector.square_context_regions == [1]
    assert detector.observed_regions == [2]


def test_detect_returns_empty_masks_for_a_blank_network() -> None:
    detector = mps.MpsTextDetector(_blank)
    (mask,) = detector.detect([ROI])
    assert int(np.count_nonzero(mask)) == 0
    assert detector.observed_regions == [0]


def test_detect_groups_different_grids_and_keeps_input_order() -> None:
    detector = mps.MpsTextDetector(_bar)
    first, second = detector.detect([ROI, TALL])
    assert first.shape == (196, 1077)
    assert second.shape == (300, 100)
    assert detector.grid_sizes == [(736, 128), (256, 736)]


def test_detect_rejects_empty_and_oversized_batches() -> None:
    detector = mps.MpsTextDetector(_bar)
    with pytest.raises(ValueError, match="1–8"):
        detector.detect([])
    with pytest.raises(ValueError, match="1–8"):
        detector.detect([ROI] * 9)


def test_detect_rejects_a_combined_region_budget_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps, "MAX_REGIONS", 1)
    detector = mps.MpsTextDetector(_bar)
    with pytest.raises(ValueError, match="budget"):
        detector.detect([ROI])


def test_embedded_decode_network_must_match_its_checksum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps, "DECODE_NET_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        mps.MpsTextDetector(_bar)


def test_load_network_runs_a_pinned_exported_program(tmp_path: Path) -> None:
    class Head(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :1] * 0.5

    path = tmp_path / "head.pt2"
    batch_dim = torch.export.Dim("batch", min=1, max=8)
    exported = torch.export.export(
        Head(), (torch.zeros(2, 3, 4, 6),), dynamic_shapes=({0: batch_dim},)
    )
    torch.export.save(exported, str(path))
    run = mps.load_network(path, sha256=fingerprint(path), device="cpu")
    batch = np.ones((3, 3, 4, 6), dtype=np.float32)
    assert run(batch).tolist() == np.full((3, 1, 4, 6), 0.5, dtype=np.float32).tolist()


def test_load_network_rejects_a_wrong_checksum_and_missing_files(tmp_path: Path) -> None:
    folder = tmp_path
    model = folder / "model.pt2"
    model.write_bytes(b"not an exported program")
    with pytest.raises(ValueError, match="hash-verified"):
        mps.load_network(model, sha256="0" * 64, device="cpu")
    with pytest.raises(ValueError, match="hash-verified"):
        mps.load_network(folder / "absent.pt", sha256="0" * 64, device="cpu")


def test_loading_onto_mps_registers_the_graph_release_at_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.pt2"
    model.write_bytes(b"x")
    calls: list[str] = []

    class Program:
        def module(self) -> "Program":
            return self

        def to(self, device: str) -> "Program":
            calls.append(f"to {device}")
            return self

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.export, "load", lambda path: Program())
    monkeypatch.setattr(mps, "release_mps_graphs_at_exit", lambda: calls.append("release"))
    mps.load_network(model, sha256=fingerprint(model), device="mps")
    assert calls == ["release", "to mps"]


def test_load_network_refuses_an_unavailable_mps_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        mps.load_network(model, sha256=fingerprint(model), device="mps")


def test_a_larger_batch_limit_accepts_more_rois_in_one_call() -> None:
    detector = mps.MpsTextDetector(_bar, max_batch=12)
    masks = detector.detect([ROI] * 12)
    assert len(masks) == 12
    assert detector.grid_sizes == [(736, 128)] * 12


def test_the_batch_limit_must_stay_in_range() -> None:
    with pytest.raises(ValueError, match="batch limit"):
        mps.MpsTextDetector(_bar, max_batch=0)
    with pytest.raises(ValueError, match="batch limit"):
        mps.MpsTextDetector(_bar, max_batch=65)


def test_cpu_thread_count_must_be_within_bounds() -> None:
    with pytest.raises(ValueError, match="CPU thread"):
        mps.MpsTextDetector(_bar, cpu_threads=0)
    with pytest.raises(ValueError, match="CPU thread"):
        mps.MpsTextDetector(_bar, cpu_threads=17)


def test_threaded_decode_matches_single_thread_masks() -> None:
    single = mps.MpsTextDetector(_bar, cpu_threads=1).detect([ROI, ROI, ROI])
    many = mps.MpsTextDetector(_bar, cpu_threads=4).detect([ROI, ROI, ROI])
    assert [mask.tobytes() for mask in many] == [mask.tobytes() for mask in single]


def test_a_lazy_network_checks_now_and_loads_once_on_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.pt2"
    model.write_bytes(b"program")
    with pytest.raises(ValueError, match="hash-verified"):
        mps.lazy_network(model, sha256="0" * 64, device="cpu")
    loads: list[Path] = []

    def load(path: Path, *, sha256: str, device: str) -> mps.Network:
        loads.append(path)
        return lambda batch: batch * 2

    monkeypatch.setattr(mps, "load_network", load)
    run = mps.lazy_network(model, sha256=fingerprint(model), device="cpu")
    assert loads == []
    assert run(np.ones(2, dtype=np.float32)).tolist() == [2.0, 2.0]
    assert run(np.ones(1, dtype=np.float32)).tolist() == [2.0]
    assert loads == [model]
