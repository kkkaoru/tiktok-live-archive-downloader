"""Batched PP-OCRv3 DB text detection: network on Apple MPS, OpenCV's own DB decoder.

The network is a checksum-pinned ``torch.export`` program converted from the pinned ONNX
detector, dynamic over batch and 32-multiple grids. Its probability maps are decoded by
``cv2.dnn.TextDetectionModel_DB`` itself (through a tiny embedded channel-0 network) with the
production thresholds, so contour, score and unclip behaviour is OpenCV's, and the production
``polygon_mask`` projection is reused unchanged. No download, no automatic fallback, no
executable model code beyond the pinned exported program.
"""

import base64
import hashlib
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .local_asr import fingerprint
from .mps_graph_cache import release_mps_graphs_at_exit
from .subtitle_detector import DETECTOR_SIZE, polygon_mask
from .subtitle_inpaint import Image, validate_roi

Network = Callable[[NDArray[np.float32]], NDArray[np.float32]]

# Channel-0 + Squeeze ONNX network with dynamic spatial dimensions (input 1x3xHxW -> HxW).
DECODE_NET_SHA256 = "cad6aa6a6f14648e8d6bb261ed71d30746b55499542c4a1e73b48a9b1137786d"
DECODE_NET_BASE64 = (
    "CAg6nQEKFgoBeAoBcwoBZQoBYRIBYyIFU2xpY2UKEgoBYwoBcRIBeSIHU3F1ZWV6ZRIIY2hhbm5l"
    "bDAqCggBEAc6AQBCAXMqCggBEAc6AQFCAWUqCggBEAc6AQFCAWEqCwgCEAc6AgABQgFxWh0KAXgS"
    "GAoWCAESEgoCCAEKAggDCgMSAWgKAxIBd2IVCgF5EhAKDggBEgoKAxIBaAoDEgF3QgQKABAP"
)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SCALE = np.array([1 / 255 / 0.229, 1 / 255 / 0.224, 1 / 255 / 0.225], dtype=np.float32)
BINARY_THRESHOLD = 0.3
POLYGON_THRESHOLD = 0.5
MAX_CANDIDATES = 200
UNCLIP_RATIO = 2.0
MAX_REGIONS = 64
MAX_BATCH = 8
MAX_LOOKAHEAD_BATCH = 64
MAX_CPU_THREADS = 16
DEFAULT_CPU_THREADS = 4
MAX_MODEL_BYTES = 64 * 1024 * 1024


def decoder_model() -> Any:
    raw = base64.b64decode(DECODE_NET_BASE64)
    if hashlib.sha256(raw).hexdigest() != DECODE_NET_SHA256:
        raise ValueError("Embedded decode network differs from its pinned checksum")
    model = cv2.dnn.TextDetectionModel_DB(cv2.dnn.readNetFromONNX(np.frombuffer(raw, np.uint8)))
    model.setBinaryThreshold(BINARY_THRESHOLD)
    model.setPolygonThreshold(POLYGON_THRESHOLD)
    model.setMaxCandidates(MAX_CANDIDATES)
    model.setUnclipRatio(UNCLIP_RATIO)
    model.setInputMean((0.0, 0.0, 0.0))
    model.setInputScale(1.0)
    return model


def lazy_network(model: Path, *, sha256: str, device: str = "mps") -> Network:
    """The pinned detector, hash-checked now but loaded onto the device only on first use.

    A chunk with YOLO admission and masks never calls it, so it no longer pays the load.
    """
    if model.is_symlink() or not model.is_file() or fingerprint(model) != sha256:
        raise ValueError("Expected a bounded, hash-verified local exported detector program")
    loaded: list[Network] = []

    def run(batch: NDArray[np.float32]) -> NDArray[np.float32]:
        if not loaded:
            loaded.append(load_network(model, sha256=sha256, device=device))
        return loaded[0](batch)

    return run


def load_network(model: Path, *, sha256: str, device: str = "mps") -> Network:
    """Load the pinned exported detector program onto an explicit device; never fall back."""
    if (
        model.is_symlink()
        or not model.is_file()
        or not 0 < model.stat().st_size <= MAX_MODEL_BYTES
        or fingerprint(model) != sha256
    ):
        raise ValueError("Expected a bounded, hash-verified local exported detector program")
    import torch  # noqa: PLC0415 - optional heavy dependency, only for the real device run

    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("The MPS device is unavailable; no automatic fallback is allowed")
    if device == "mps":
        release_mps_graphs_at_exit()
    # The program loads on the CPU and is exported in inference mode, so it only moves devices.
    program: Any = torch.export.load(str(model))
    module = program.module().to(device)

    def run(batch: NDArray[np.float32]) -> NDArray[np.float32]:
        with torch.no_grad():
            output = module(torch.from_numpy(batch).to(device)).float().cpu().numpy()
        return np.asarray(output, dtype=np.float32)

    return run


class MpsTextDetector:
    """Same ``detect`` contract and receipts as ``LearnedTextDetector``, batched network."""

    def __init__(
        self,
        network: Network,
        *,
        max_batch: int = MAX_BATCH,
        cpu_threads: int = DEFAULT_CPU_THREADS,
    ) -> None:
        if not 1 <= max_batch <= MAX_LOOKAHEAD_BATCH:
            raise ValueError("Invalid detector batch limit")
        if not 1 <= cpu_threads <= MAX_CPU_THREADS:
            raise ValueError("Invalid detector CPU thread count")
        self._network = network
        self._max_batch = max_batch
        # OpenCV releases the GIL; each worker thread owns its stateful DB decoder.
        self._pool = ThreadPoolExecutor(max_workers=cpu_threads)
        self._local = threading.local()
        decoder_model()  # fail fast on a corrupted embedded decode network
        self.grid_sizes: list[tuple[int, int]] = []
        self.square_context_regions: list[int] = []
        self.observed_regions: list[int] = []
        self.intersected_regions: list[int] = []

    def _thread_decoder(self) -> Any:
        decoder = getattr(self._local, "decoder", None)
        if decoder is None:
            decoder = decoder_model()
            self._local.decoder = decoder
        return decoder

    def _decoded(
        self, image: Image, prob: NDArray[np.float32], grid: tuple[int, int]
    ) -> tuple[Image, int, int]:
        decoder = self._thread_decoder()
        decoder.setInputSize(grid)
        polygons, confidences = decoder.detect(np.repeat(prob[..., None], 3, axis=2))
        mask = polygon_mask(
            polygons,
            np.asarray(confidences, dtype=np.float64),
            width=image.shape[1],
            height=image.shape[0],
            grid=grid,
        )
        bounds = np.array(grid, dtype=np.int32)
        crossing = sum(
            1
            for polygon in polygons
            if np.any(np.asarray(polygon, dtype=np.int32) < 0)
            or np.any(np.asarray(polygon, dtype=np.int32) > bounds)
        )
        return mask, len(polygons), crossing

    def _views(
        self, images: Sequence[Image], grid: tuple[int, int]
    ) -> list[tuple[Image, int, int]]:
        def blob(image: Image) -> NDArray[np.float32]:
            return ((cv2.resize(image, grid).astype(np.float32) - MEAN) * SCALE).transpose(2, 0, 1)

        blobs = np.stack(list(self._pool.map(blob, images)))
        probs = self._network(blobs).reshape(len(images), grid[1], grid[0])
        decode = partial(self._decoded, grid=grid)
        return list(self._pool.map(decode, images, list(probs)))

    def _grouped(
        self, images: Sequence[Image], grids: Sequence[tuple[int, int]]
    ) -> dict[int, tuple[Image, int, int]]:
        found: dict[int, tuple[Image, int, int]] = {}
        for grid in sorted(set(grids)):
            members = [index for index, wanted in enumerate(grids) if wanted == grid]
            views = self._views([images[index] for index in members], grid)
            found.update(zip(members, views, strict=True))
        return found

    def detect(self, images: Sequence[Image]) -> tuple[Image, ...]:
        """Detect 1..max_batch ROIs; each gets its aspect view plus the square view."""
        if not 1 <= len(images) <= self._max_batch:
            raise ValueError(f"Supply 1–{self._max_batch} detector ROIs")
        grids: list[tuple[int, int]] = []
        for image in images:
            validate_roi(image)
            height, width = image.shape[:2]
            scale = DETECTOR_SIZE / max(width, height)
            grids.append(
                (
                    max(32, round(width * scale / 32) * 32),
                    max(32, round(height * scale / 32) * 32),
                )
            )
        square = (DETECTOR_SIZE, DETECTOR_SIZE)
        first = self._grouped(images, grids)
        extra = [index for index, grid in enumerate(grids) if grid != square]
        second = dict(
            zip(
                extra,
                self._grouped([images[index] for index in extra], [square] * len(extra)).values(),
                strict=True,
            )
        )
        masks: list[Image] = []
        for index, grid in enumerate(grids):
            mask, regions, crossings = first[index]
            square_regions = 0
            if index in second:
                other, square_regions, other_crossings = second[index]
                if regions + square_regions > MAX_REGIONS:
                    raise ValueError("Combined detector region budget exceeded")
                mask = np.maximum(mask, other)
                regions += square_regions
                crossings += other_crossings
            masks.append(mask)
            self.grid_sizes.append(grid)
            self.square_context_regions.append(square_regions)
            self.observed_regions.append(regions)
            self.intersected_regions.append(crossings)
        return tuple(masks)
