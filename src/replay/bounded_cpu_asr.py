"""Explicit allocator policy around unchanged, resumable CPU recognition."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType

from replay.asr_runtime import ASRDevice, validate_cpu_paths
from replay.audio_cpu_probe import audit_descriptors
from replay.batch_asr import transcribe_windows
from replay.core import ReplayError
from replay.local_asr import DEFAULT_MODEL, fingerprint
from replay.native_invocation import canonical, require


def _invoke(module: ModuleType, name: str, *args: object) -> object:
    function: object = getattr(module, name, None)
    if not callable(function):
        raise ReplayError(f"Missing MLX allocator API: {name}")
    return function(*args)


def _counter(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ReplayError("Invalid allocator counter")
    return value


def _memory(module: ModuleType) -> dict[str, int]:
    return {
        name: _counter(_invoke(module, f"get_{name}_memory"))
        for name in ("active", "cache", "peak")
    }


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, indent=2, allow_nan=False)


def run(source: Path, output: Path, *, evidence: Path, cache_limit: int) -> Path:
    """No runtime patching, model substitution, clock changes or fallback."""
    require(
        type(cache_limit) is int and 0 <= cache_limit <= 2 * 1024**3,
        "Cache limit must be explicit and bounded",
    )
    require("mlx_whisper" not in sys.modules, "CPU policy requires a fresh provider process")
    root = Path.cwd().resolve()
    validate_cpu_paths(ASRDevice.CPU, source, output, DEFAULT_MODEL, evidence)
    evidence = canonical(str(evidence.absolute()))
    require(
        evidence.is_relative_to(root / "edits/work"), "Policy evidence must stay in local scratch"
    )
    require(
        not evidence.is_relative_to(output.resolve())
        and not output.resolve().is_relative_to(evidence),
        "Policy evidence and recognition output must be independent",
    )
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        path = canonical(os.environ.get(name))
        require(
            path.is_relative_to(root / "edits/work") and path.is_dir(),
            "CPU scratch must stay local",
        )
    pins = {str(Path(__file__).resolve()): fingerprint(Path(__file__))}
    evidence.mkdir(mode=0o700)
    audit_descriptors(evidence, "before-core-import")
    module = importlib.import_module("mlx.core")
    audit_descriptors(evidence, "after-core-import")
    cpu: object = getattr(module, "cpu", None)
    require(cpu is not None, "Missing explicit CPU device")
    previous = _counter(_invoke(module, "set_cache_limit", cache_limit))
    try:
        require(
            _counter(_invoke(module, "set_cache_limit", cache_limit)) == cache_limit,
            "Allocator policy was not installed",
        )
        _write(
            evidence / "started.json",
            {
                "version": 1,
                "inputs": pins,
                "device": "cpu",
                "cacheLimitBytes": cache_limit,
                "previousCacheLimitBytes": previous,
                "memory": _memory(module),
                "source": str(source.resolve()),
                "output": str(output.resolve()),
                "dtypeContract": "provider-default-fp16-unchanged",
            },
        )
        complete = transcribe_windows(source, output, device=ASRDevice.CPU)
        _invoke(module, "synchronize", cpu)
        require(
            _counter(_invoke(module, "set_cache_limit", cache_limit)) == cache_limit,
            "Allocator policy changed during recognition",
        )
        require(
            {path: fingerprint(Path(path)) for path in pins} == pins,
            "CPU policy source changed during recognition",
        )
        audit_descriptors(evidence, "after-recognition")
        _write(
            evidence / "complete.json",
            {
                "version": 1,
                "inputs": pins,
                "cacheLimitBytes": cache_limit,
                "memory": _memory(module),
                "recognitionCompletion": {
                    "path": str(complete.resolve()),
                    "sha256": fingerprint(complete),
                },
                "deliveryComplete": False,
            },
        )
        return complete
    finally:
        _invoke(module, "set_cache_limit", previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--cache-limit-bytes", type=int, required=True)
    args = parser.parse_args()
    print(run(args.source, args.output, evidence=args.evidence, cache_limit=args.cache_limit_bytes))


if __name__ == "__main__":
    main()
