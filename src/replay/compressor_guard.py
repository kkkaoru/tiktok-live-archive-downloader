"""Caller-side robustness for the caption-render Compressor batch (``compressor_batch`` is pinned).

- Every part's recorded output is verified after the batch: it exists and holds exactly the
  planned frames. A part Compressor reported as successful without such an output ("success
  without output") has its job state (and any partial output) moved aside, so the batch submits it
  once more; a part still unverified after that resubmission fails the step.
- The verified outputs' sha256 are recorded in ``verified-outputs.json`` beside the job states.
- A circuit breaker for "job controller is down": the step stops submitting, restarts Compressor
  (a task-related app restart: quit, force it after ``QUIT_SECONDS`` if it does not exit, open it
  in the background) and waits until its job controller runs again before it resumes the batch.
  Compressor renders one job at a time machine-wide behind the caption-render lease, so only this
  recording's jobs are affected.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record

UNTRACKED = re.compile(r"Refusing to overwrite an untracked output: (/\S+)")
CONTROLLER_DOWN = re.compile(r"ジョブコントローラがダウンしています|job controller is down")
QUIT_SECONDS = 30.0
READY_SECONDS = 120.0
POLL_SECONDS = 2.0
CONTROLLER = "JobControllerService"
TAIL_BYTES = 64 * 1024

Call = Callable[..., subprocess.CompletedProcess[str]]
Packets = Callable[[Path], int]


def controller_down(log: Path, since: int) -> bool:
    """True when the batch's part of ``log`` (from byte ``since``) reports the controller down."""
    if not log.is_file():
        return False
    size = log.stat().st_size
    with log.open("rb") as stream:
        stream.seek(max(since, size - TAIL_BYTES) if size >= since else 0)
        return CONTROLLER_DOWN.search(stream.read().decode("utf-8", errors="replace")) is not None


def _running(call: Call, pattern: str) -> bool:
    return call(["/usr/bin/pgrep", "-f", pattern], capture_output=True, text=True).returncode == 0


def restart(
    app: Path,
    *,
    call: Call = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Quit Compressor (forcing it if it hangs), reopen it and wait for its job controller."""
    executable = str(app / "Contents" / "MacOS" / "Compressor")
    call(["/usr/bin/osascript", "-e", f'tell application "{app}" to quit'], capture_output=True,
         text=True, timeout=QUIT_SECONDS)  # fmt: skip
    deadline = clock() + QUIT_SECONDS
    while _running(call, executable) and clock() < deadline:
        sleep(POLL_SECONDS)
    if _running(call, executable):
        call(["/usr/bin/pkill", "-KILL", "-f", executable], capture_output=True, text=True)
    call(["/usr/bin/open", "-g", "-a", str(app)], capture_output=True, text=True, check=True)
    deadline = clock() + READY_SECONDS
    while not _running(call, CONTROLLER):
        if clock() >= deadline:
            raise ReplayError("Compressor's job controller did not start after the restart")
        sleep(POLL_SECONDS)


def unverified(parts: Path, states: Path, packets: Packets) -> list[str]:
    """Parts whose recorded successful output is missing or lacks its planned frames."""
    found = []
    for row in json.loads(parts.read_text(encoding="utf-8")):
        item = record(row)
        name, frames = str(item["name"]), int(str(item["frames"]))
        state = states / f"{name}.json"
        saved = record(json.loads(state.read_text(encoding="utf-8"))) if state.is_file() else {}
        output = Path(str(saved.get("output", "")))
        if saved.get("status") == "Successful" and (
            not output.is_file() or packets(output) != frames
        ):
            found.append(name)
    return found


def untracked(log: Path, since: int, media: Path) -> Path | None:
    """The output the batch refused as untracked in its part of ``log`` (a Failed job's partial
    output whose state was retired), when it lies under ``media``; else None."""
    if not log.is_file():
        return None
    with log.open("rb") as stream:
        stream.seek(since)
        found = UNTRACKED.findall(stream.read().decode("utf-8", errors="replace"))
    if not found:
        return None
    output = Path(found[-1])
    return output if output.is_relative_to(media) and output.is_file() else None


def set_aside(output: Path, stamp: int) -> Path:
    """Move a partial output one level up: compressor_batch refuses any "<stem>.*" beside it."""
    aside = output.parent.parent / f"{output.stem}.unverified-{stamp}{output.suffix}"
    output.rename(aside)
    return aside


def retire(states: Path, names: list[str]) -> None:
    """Move the named parts' job states (and partial outputs) aside so they are submitted again."""
    stamp = time.time_ns()
    for name in names:
        state = states / f"{name}.json"
        saved = record(json.loads(state.read_text(encoding="utf-8")))
        # Only a terminal Successful state records "output"; a submitted or failed one has it in
        # its job configuration.
        output = Path(
            str(saved.get("output") or record(saved.get("jobConfig", {})).get("output", ""))
        )
        if output.is_file():
            set_aside(output, stamp)
        state.rename(states / f"{name}.unverified-{stamp}.json")


def record_hashes(parts: Path, states: Path, output: Path) -> None:
    """Write each part's verified output path and sha256."""
    rows = {}
    for row in json.loads(parts.read_text(encoding="utf-8")):
        name = str(record(row)["name"])
        movie = Path(
            str(record(json.loads((states / f"{name}.json").read_text("utf-8")))["output"])
        )
        with movie.open("rb") as stream:
            rows[name] = {
                "path": str(movie),
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
            }
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staged = output.with_name(f".{output.name}.partial")
    staged.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    staged.replace(output)
