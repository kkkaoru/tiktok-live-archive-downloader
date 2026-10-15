"""Finish missing CPU ASR windows from the end of a collection in parallel.

``isolated_cpu_asr.run`` stays the only writer of the run policy and the completion
receipt and is receipt-pinned, so it cannot change. This module only produces the same
immutable per-window evidence (``raw``/``clock`` files and the ``cpu-runtime`` directory)
through the same isolated runtime, for windows the serial run has not reached yet. The
serial run reuses every finished window after verifying it, so completion receipts and
their verifiers see identical window evidence. While the serial run is live, no window within
``SAFETY_MARGIN`` of its frontier is claimed, which keeps the two writers apart. When the serial
run is not running, ``exclusive`` takes its ``.run`` lease so every missing window can be
claimed; the child deadline used here is not part of any receipt.
"""

import argparse
import threading
import wave
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from replay.asr_runtime import ASRDevice, validate_cpu_paths
from replay.audio_windows import AudioWindow, plan_audio_windows
from replay.batch_asr import RATE, _read, _window
from replay.core import ReplayError
from replay.isolated_cpu_asr import Executor, IsolatedRuntime, execute_child
from replay.local_asr import DEFAULT_MODEL, fingerprint
from replay.native_invocation import require
from replay.recordings import record
from replay.stages import execution_lock

SAFETY_MARGIN = 3
MAX_WORKERS = 8
WINDOW_SAMPLES = 60 * RATE
STRIDE_SAMPLES = 30 * RATE


def _finished(output: Path, window: AudioWindow) -> bool:
    return (output / f"{window.start_sample:012d}.clock.json").is_file()


def frontier(output: Path, windows: Sequence[AudioWindow]) -> int:
    """Index of the first window without a clock receipt (the serial run's next window)."""
    for index, window in enumerate(windows):
        if not _finished(output, window):
            return index
    return len(windows)


def _require_same_inputs(source: Path, output: Path, model: Path) -> None:
    """The serial run recorded fingerprints of its source and pinned model; match them."""
    recorded = record(_read(output / "inputs.json").get("inputs"))
    weights = [p for p in (model / "weights.npz", model / "weights.safetensors") if p.exists()]
    require(len(weights) == 1, "Expected one pinned model weight file")
    for path in (source, model / "config.json", weights[0]):
        require(
            recorded.get(str(path.resolve())) == fingerprint(path),
            "Prewarm inputs differ from the serial ASR run",
        )


class _Claims:
    """Hand each missing window, last first, to exactly one worker at a time."""

    def __init__(self, count: int) -> None:
        self._next = count - 1
        self._lock = threading.Lock()

    def take(self) -> int:
        with self._lock:
            index = self._next
            self._next -= 1
            return index


def _prewarm_worker(
    source: Path,
    output: Path,
    windows: Sequence[AudioWindow],
    *,
    claims: _Claims,
    model: Path,
    runtime: IsolatedRuntime,
    margin: int,
) -> int:
    created = 0
    with wave.open(str(source), "rb") as audio:
        while (index := claims.take()) >= 0:
            if index < frontier(output, windows) + margin:
                break
            if _finished(output, windows[index]):
                continue
            _window(audio, windows[index], output, model, runtime)
            created += 1
    return created


def _run_workers(
    source: Path,
    output: Path,
    windows: Sequence[AudioWindow],
    *,
    workers: int,
    model: Path,
    runtime: IsolatedRuntime,
    margin: int,
) -> int:
    claims = _Claims(len(windows))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _prewarm_worker,
                source,
                output,
                windows,
                claims=claims,
                model=model,
                runtime=runtime,
                margin=margin,
            )
            for _ in range(workers)
        ]
        return sum(future.result() for future in futures)


def prewarm(
    source: Path,
    output: Path,
    *,
    workers: int,
    timeout: int = 300,
    model: Path = DEFAULT_MODEL,
    execute: Executor = execute_child,
    exclusive: bool = False,
) -> int:
    """Create missing windows ahead of the serial run; returns how many were created."""
    require(type(workers) is int and 1 <= workers <= MAX_WORKERS, "Invalid prewarm worker count")
    require(type(timeout) is int and 1 <= timeout <= 3600, "Invalid child deadline")
    validate_cpu_paths(ASRDevice.CPU, source, output, model)
    require(
        (output / "inputs.json").is_file() and not (output / "complete.json").exists(),
        "Prewarm needs an in-progress collection started by the serial run",
    )
    _require_same_inputs(source, output, model)
    with wave.open(str(source), "rb") as audio:
        require(
            (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, RATE),
            "Expected mono PCM16/16kHz",
        )
        windows = plan_audio_windows(
            audio.getnframes(), window_samples=WINDOW_SAMPLES, stride_samples=STRIDE_SAMPLES
        )
    runtime = IsolatedRuntime(execute, timeout)
    if not exclusive:
        return _run_workers(
            source,
            output,
            windows,
            workers=workers,
            model=model,
            runtime=runtime,
            margin=SAFETY_MARGIN,
        )
    with execution_lock(output, ".run") as acquired:
        require(acquired, "ASR output already has an owner")
        return _run_workers(
            source, output, windows, workers=workers, model=model, runtime=runtime, margin=0
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--child-timeout", type=int, default=300)
    parser.add_argument("--exclusive", action="store_true")
    args = parser.parse_args()
    try:
        created = prewarm(
            args.source,
            args.output,
            workers=args.workers,
            timeout=args.child_timeout,
            model=args.model,
            exclusive=args.exclusive,
        )
    except ReplayError as error:
        raise SystemExit(f"ASR_PREWARM_ERROR {error}") from error
    print(f"ASR_PREWARM_CREATED {created}", flush=True)


if __name__ == "__main__":
    main()
