"""Bounded model-free CPU allocator observations; never ASR or GPU fallback."""

import argparse
import gc
import importlib
import json
import os
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType

from replay.audio_cpu_probe import audit_descriptors
from replay.core import ReplayError
from replay.native_invocation import canonical, require


def _call(module: ModuleType, name: str, *args: object, **kwargs: object) -> object:
    function: object = getattr(module, name, None)
    if not callable(function):
        raise ReplayError(f"Missing MLX allocator API: {name}")
    return function(*args, **kwargs)


def _integer(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ReplayError("Invalid MLX memory counter")
    return value


def _snapshot(module: ModuleType) -> dict[str, int]:
    return {
        name: _integer(_call(module, f"get_{name}_memory")) for name in ("active", "cache", "peak")
    }


def observe(
    module: ModuleType, *, elements: int, iterations: int, cache_limit: int
) -> dict[str, object]:
    """Bound allocations to at most three16MiB arrays; release policy is explicit."""
    if (
        type(elements) is not int
        or not 1 <= elements <= 4_194_304
        or type(iterations) is not int
        or not 1 <= iterations <= 16
        or type(cache_limit) is not int
        or not 0 <= cache_limit <= 128 * 1024**2
    ):
        raise ReplayError("CPU allocator probe exceeds its bounded resource policy")
    cpu: object = getattr(module, "cpu", None)
    dtype: object = getattr(module, "float32", None)
    if cpu is None or dtype is None:
        raise ReplayError("Missing explicit CPU or float32 API")
    context = _call(module, "stream", cpu)
    if not isinstance(context, AbstractContextManager):
        raise ReplayError("MLX CPU stream must be a context manager")
    before = _snapshot(module)
    previous_limit = _integer(_call(module, "set_cache_limit", cache_limit))
    observations: list[dict[str, object]] = []
    try:
        with context:
            for index in range(iterations):
                count = elements - index * elements // (2 * iterations)
                left = _call(module, "ones", (count,), dtype=dtype, stream=cpu)
                right = _call(module, "ones", (count,), dtype=dtype, stream=cpu)
                result = _call(module, "add", left, right, stream=cpu)
                _call(module, "eval", result)
                _call(module, "synchronize", cpu)
                allocated = _snapshot(module)
                del result, right, left
                gc.collect()
                _call(module, "synchronize", cpu)
                released = _snapshot(module)
                _call(module, "clear_cache")
                cleared = _snapshot(module)
                if cleared["cache"] != 0:
                    raise ReplayError("MLX clear_cache did not release its reported cache")
                observations.append(
                    {
                        "elements": count,
                        "allocated": allocated,
                        "released": released,
                        "cleared": cleared,
                    }
                )
    finally:
        _call(module, "set_cache_limit", previous_limit)
    return {
        "version": 1,
        "device": "cpu",
        "dtype": "float32",
        "modelLoaded": False,
        "cacheLimitBytes": cache_limit,
        "previousCacheLimitBytes": previous_limit,
        "before": before,
        "iterations": observations,
        "after": _snapshot(module),
        "ASRImpactVerified": False,
    }


def run(output: Path, *, elements: int, iterations: int, cache_limit: int) -> Path:
    root = Path.cwd().resolve()
    output = canonical(str(output.absolute()))
    require(output.is_relative_to(root / "edits/work"), "Allocator probe output must stay local")
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
        path = canonical(os.environ.get(name))
        require(
            path.is_relative_to(root / "edits/work") and path.is_dir(),
            "Probe scratch must stay local",
        )
    output.mkdir(mode=0o700)
    audit_descriptors(output, "before-core-import")
    module = importlib.import_module("mlx.core")
    result = observe(module, elements=elements, iterations=iterations, cache_limit=cache_limit)
    audit_descriptors(output, "after-cpu-allocator-probe")
    receipt = output / "receipt.json"
    with receipt.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(result, stream, indent=2, allow_nan=False)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--elements", type=int, default=1_048_576)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--cache-limit-bytes", type=int, default=32 * 1024**2)
    args = parser.parse_args()
    print(
        run(
            args.output,
            elements=args.elements,
            iterations=args.iterations,
            cache_limit=args.cache_limit_bytes,
        )
    )


if __name__ == "__main__":
    main()
