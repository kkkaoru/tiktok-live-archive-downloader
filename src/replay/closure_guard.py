"""Fail a running step fast when a code file its stage pinned is edited under it.

A paging stage pins its whole code closure (``CODE-CLOSURE*.json``: path to sha256) when it is
created and refuses to finish if any of those files changed meanwhile; the failure came minutes
later as "Workflow stopped (StageError)" (10 failures, failure survey 2026-10-24). While a step
runs, ``watching`` re-reads the closures written since the step started (their hashes are the
files' content when the stage began) and re-hashes the files every ``INTERVAL_SECONDS``. Only a
content change counts: a touch or a byte-identical rewrite hashes the same. On the first change
it calls ``on_change`` once (the pipeline
then marks the step log ``EDITED_MID_RUN`` and raises a notice; the step is never stopped and
finishes, and if its stage then refuses the changed code, ``step_retry`` classifies that as
deterministic and relaunchable at once). Steps run from a code snapshot (``code_snapshot``) pin
immutable files and are not watched. It only detects: it cannot prevent the edit.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

from replay import failure_notify

INTERVAL_SECONDS = 30.0
MARKER = "EDITED_MID_RUN"
MAX_CLOSURE_BYTES = 4 * 1024 * 1024


def closures(roots: tuple[Path, ...], since: float) -> dict[Path, str]:
    """The files pinned by closures written at or after ``since`` (path to sha256)."""
    pins: dict[Path, str] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("CODE-CLOSURE*.json"):
            if path.stat().st_mtime < since or path.stat().st_size > MAX_CLOSURE_BYTES:
                continue
            inputs = json.loads(path.read_text(encoding="utf-8")).get("inputs")
            if isinstance(inputs, dict):
                pins.update({Path(str(name)): str(digest) for name, digest in inputs.items()})
    return pins


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def changed(pins: dict[Path, str]) -> list[str]:
    return sorted(str(path) for path, digest in pins.items() if _sha256(path) != digest)


@contextmanager
def watching(
    roots: tuple[Path, ...],
    since: float,
    *,
    on_change: Callable[[list[str]], None],
    interval: float = INTERVAL_SECONDS,
) -> Generator[None]:
    """Watch the step's pinned closures until the block ends."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            edited = changed(closures(roots, since))
            if edited:
                on_change(edited)
                return

    thread = threading.Thread(target=loop, name="closure-guard", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def note_edit(recording: str, step: str, log: Path, edited: list[str]) -> None:
    """Mark the step's log and raise a notice; the step keeps running."""
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"{MARKER} {' '.join(edited[:5])}: a pinned file changed under the step\n")
    failure_notify.notify(
        recording, step, cls="edited-mid-run", cause=f"{len(edited)} pinned file(s) changed",
        action="the step finishes; relaunch it if its stage refuses the change",
    )  # fmt: skip
