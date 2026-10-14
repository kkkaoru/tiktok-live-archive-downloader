"""Explicit Whisper CPU requests; operator support still needs a real bounded probe.

CPU callers must use a fresh, single-threaded process. Keep the provider's FP16
contract: no dtype conversion, device fallback or model substitution is performed.
"""

import importlib
import os
import sys
from contextlib import AbstractContextManager
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from replay.audio_cpu_probe import audit_descriptors
from replay.core import ReplayError
from replay.native_invocation import canonical, require


class ASRDevice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"


class Recognizer(Protocol):
    def __call__(self, samples: NDArray[np.float32], **kwargs: object) -> object: ...


def validate_cpu_paths(device: ASRDevice, *paths: Path) -> None:
    """Reject escaping input/output parents before CPU evidence writes begin."""
    if device == ASRDevice.CPU:
        root = Path.cwd().resolve()
        for path in paths:
            checked = canonical(str(path.absolute()))
            require(checked.is_relative_to(root), "CPU paths must stay inside checkout")


def _recognizer() -> Recognizer:
    candidate: object = importlib.import_module("mlx_whisper").transcribe
    if not callable(candidate):
        raise ReplayError("Installed recognizer is not callable")
    return candidate


def _predict(recognizer: Recognizer, samples: NDArray[np.float32], model: Path) -> object:
    return recognizer(
        samples,
        path_or_hf_repo=str(model.resolve()),
        language="ja",
        verbose=False,
        condition_on_previous_text=False,
        word_timestamps=True,
        temperature=0.0,
    )


class WhisperRuntime:
    """Reuse only this instance's CPU import/cache across sequential batch windows."""

    def __init__(self, device: ASRDevice = ASRDevice.AUTO) -> None:
        self.device = ASRDevice(device)
        self._cpu_recognizer: Recognizer | None = None

    def predict(self, samples: NDArray[np.float32], model: Path, *, evidence: Path) -> object:
        if self.device == ASRDevice.AUTO:
            return _predict(_recognizer(), samples, model)
        root = Path.cwd().resolve()
        evidence = canonical(str(evidence.absolute()))
        require(evidence.is_relative_to(root / "edits/work"), "CPU evidence must stay local")
        # Check configured destinations before importing any runtime. FD observations
        # below are additional evidence, not a syscall trace or a filesystem sandbox.
        for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
            path = canonical(os.environ.get(name))
            require(path.is_relative_to(root) and path.is_dir(), f"{name} must stay local")
        if self._cpu_recognizer is None and "mlx_whisper" in sys.modules:
            raise ReplayError("CPU ASR requires a fresh process; provider cache device is unknown")
        evidence.mkdir(mode=0o700)
        audit_descriptors(evidence, "before-core-import")
        mx = importlib.import_module("mlx.core")
        audit_descriptors(evidence, "after-core-import")
        stream: object = getattr(mx, "stream", None)
        synchronize: object = getattr(mx, "synchronize", None)
        cpu: object = getattr(mx, "cpu", None)
        if not callable(stream) or not callable(synchronize) or cpu is None:
            raise ReplayError("Missing MLX CPU API")
        context: object = stream(cpu)
        if not isinstance(context, AbstractContextManager):
            raise ReplayError("MLX stream must provide a context manager")
        with context:
            if self._cpu_recognizer is None:
                self._cpu_recognizer = _recognizer()
            audit_descriptors(evidence, "after-whisper-import")
            # transcribe owns load_model/mx.eval, decode async_eval/eval and alignment
            # materialization. All are inside this device AND stream context.
            result = _predict(self._cpu_recognizer, samples, model)
            synchronize(cpu)
            audit_descriptors(evidence, "after-inference-eval")
            return result
