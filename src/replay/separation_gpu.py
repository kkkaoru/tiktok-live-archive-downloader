"""Vocals separation on the GPU with a per-window silence gate and a receipt.

The pinned Mel-RoFormer runs 8-second half-overlapped windows. On the GPU each window is a fresh
call, so a window the accelerator fails to compute (all-zero output for audible input) is detected
and retried instead of silently producing missing speech. The overlap-add and the 600 s / 5 s
chunk assembly are the original ones (`reduce_windows`, `separate_vocals`), so the result is the
original path's result with the same pinned weights. The module reads the input and writes one new
WAV plus a JSON receipt; it never overwrites and never marks quality as approved.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.bgm_separation import (
    MODEL_SAMPLE_RATE,
    SeparationBackend,
    SeparationConfig,
    _MlxBackend,
    separate_vocals,
)
from replay.core import ReplayError
from replay.separation_window_budget import WINDOW, Window, reduce_windows

MAX_RETRIES = 2
DEVICE = "mps"
VOCALS = "vocals"


class GatedBackend:
    """Run each 8-second window through `inner`, refusing silent output for audible input."""

    def __init__(self, inner: SeparationBackend) -> None:
        self._inner = inner
        self.windows = 0
        self.retries = 0

    @property
    def sources(self) -> tuple[str, ...]:
        return self._inner.sources

    def _window(self, chunk: NDArray[np.float32], window: Window) -> NDArray[np.float32]:
        padded = np.zeros((2, WINDOW), dtype=np.float32)
        padded[:, : window.count] = chunk[:, window.start : window.start + window.count]
        audible = bool(np.any(padded))
        for _ in range(MAX_RETRIES + 1):
            predicted = np.asarray(self._inner.extract_vocals(padded), dtype=np.float32)
            usable = (
                predicted.shape == (2, WINDOW)
                and bool(np.isfinite(predicted).all())
                and (not audible or bool(np.any(predicted)))
            )
            if usable:
                self.windows += 1
                return predicted
            self.retries += 1
        raise ReplayError(f"Accelerator returned no usable voice for window {window.index}")

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return reduce_windows(int(chunk.shape[1]), lambda window: self._window(chunk, window))

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return chunk - self.extract_vocals(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def separate(
    source: Path, output: Path, receipt: Path, backend: SeparationBackend
) -> dict[str, object]:
    """Separate `source` into `output` and write the receipt; neither may exist."""
    if output.exists() or output.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise ReplayError("Refusing to overwrite the output or receipt")
    gated = GatedBackend(backend)
    result = separate_vocals(source, output, SeparationConfig(device=DEVICE), gated)
    document: dict[str, object] = {
        "device": DEVICE,
        "source": str(source),
        "sourceSHA256": sha256(source),
        "vocals": str(output),
        "vocalsSHA256": sha256(output),
        "sampleRate": MODEL_SAMPLE_RATE,
        "audioSeconds": result.audio_seconds,
        "wallSeconds": result.wall_seconds,
        "realtimeFactor": result.realtime_factor,
        "windows": gated.windows,
        "retries": gated.retries,
        "qualityApproved": False,
    }
    receipt.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(separate(args.source, args.output, args.receipt, _MlxBackend(DEVICE))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
