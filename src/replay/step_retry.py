"""Supervised, classified, progress-gated retry of a failed recording pipeline step.

A step's run log ends with ``Producer failed (N); inspect <log>``; only the part of that producer
log (and of the run log) written during the failed attempt is classified, so an older failure
further up an appended log never counts. ``SIGNATURES`` maps the failure signatures measured in
the failure survey (bottleneck-survey REPORT5, 153 attempts) to a class:

- ``transient``: Compressor's controller, submit or empty-output failures, an upload part or the
  uploader deploy, a compose ``media_edit`` call, MPS running out of memory. These are rerun
  with backoff (``BACKOFF_SECONDS``), at most ``MAX_RETRIES`` times within ``WINDOW_SECONDS``
  per step, and after the first retry only while the previous attempt made progress (more
  finished stage, part or chunk receipts): a retry that adds neither evidence nor progress stops.
- ``volume``: a device error (EIO, ENXIO, ENODEV), Compressor's "result folder not accessible"
  (a handle to a volume that detached and came back), or a missing file while a volume detached
  during the attempt. Rerun at most once, after ``volume_events.recover`` saw the volume back
  with its UUID, restarted Compressor after a detach and re-checked the receipts written just
  before the failure.
- ``stale``: a preflight refusal of partial or interrupted state that ``clear_interrupted`` and
  ``clear_partial_chunks`` remove when the step starts again: rerun once at once (logged).
- ``deterministic``: audits and gates, changed or missing inputs, stale state, bugs; never rerun.
  ``edited-mid-run`` (a pinned code or input file changed under the running step; see
  ``closure_guard``) belongs here and may be relaunched at once by hand.
- ``unknown``: everything else, never rerun. An operator stop (SIGTERM "Step stopped") is
  deliberate and never rerun either.

Each failed attempt leaves ``<step>-<ns>.failure.json`` in the step logs: class, subclass, the
matching line, whether it was retried and why not. A failure that is not retried also raises an
immediate local notice (``failure_notify``: a macOS notification and ``failures.jsonl``). A
rerun runs the same command, so it resumes exactly like a relaunch: finished work is reused.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from replay import failure_notify, volume_events
from replay.stages import StageError

SIGNATURES: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
    (cls, sub, re.compile(pattern))
    for cls, sub, pattern in (
        ("deterministic", "edited-mid-run", r"EDITED_MID_RUN"),
        ("operator", "step stopped (SIGTERM)", r"Step stopped; its finished work is reused"),
        ("stale", "preflight refusal of partial state",
         r"Partial outputs require a read-only provenance verifier"
         r"|Unregistered outputs require explicit verified adoption"
         r"|interrupted attempt cannot resume"),
        ("transient", "Compressor job controller down",
         r"ジョブコントローラがダウンしています|job controller is down"),
        ("transient", "Compressor submit or empty output",
         r"success without output|Compressor submit|unknown submission outcome"),
        ("transient", "upload network", r"Part [0-9]+ of \S+ failed|Uploader deploy failed"),
        ("transient", "compose media_edit call",
         r"(?s)CalledProcessError.{0,400}media_edit|media_edit.{0,400}CalledProcessError"),
        ("transient", "MPS out of memory", r"MPS backend out of memory"),
        ("volume", "device I/O error",
         r"\[Errno (?:5|6|19)\]|Input/output error|Device not configured"
         r"|Operation not supported by device"),
        ("volume", "Compressor result folder not accessible",
         r"結果フォルダ.{0,600}アクセスできません|result folder.{0,300}not accessible"),
        ("deterministic", "audit or gate",
         r"Audit failed after escalation|No repaired pixel to refill|OutputLimitError"
         r"|insufficient_compression|Rendered caption gradient is missing"
         r"|No matching loaded videos|completion gate did not pass"),
        ("deterministic", "inputs or code changed",
         r"changed its immutable inputs|input changed|code closure|receipt predates"
         r"|belongs to a different original|inputs changed; refusing to overwrite"),
        ("deterministic", "missing input or stale state",
         r"FileNotFoundError|FileExistsError|has not passed|Only integer frame rates"
         r"|receipt must list|has no frames"),
    )
)  # fmt: skip
PRODUCER_LOG = re.compile(r"Producer failed \(-?[0-9]+\); inspect (/\S+)")
TAIL_BYTES = 64 * 1024
BACKOFF_SECONDS = (30.0, 60.0, 120.0)
MAX_RETRIES = 3
WINDOW_SECONDS = 3600.0
RECEIPTS = ("chunk-receipt.json", "state/*.json")

Execute = Callable[[Path], None]
Recover = Callable[..., str | None]


@dataclass(frozen=True, slots=True)
class Failure:
    cls: str
    sub: str
    cause: str


def log_sizes(directory: Path) -> dict[Path, int]:
    """Current sizes of the step logs, the start of what the next attempt appends."""
    return {path: path.stat().st_size for path in directory.glob("*.log") if path.is_file()}


def _tail(path: Path, start: int) -> str:
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(start, size - TAIL_BYTES) if size >= start else 0)
        return stream.read().decode("utf-8", errors="replace")


def classify(run_log: Path, before: dict[Path, int]) -> Failure:
    """The class of the attempt's failure, from what it wrote to its run and producer logs."""
    texts = [_tail(run_log, 0)] if run_log.is_file() else []
    producers = PRODUCER_LOG.findall(texts[0]) if texts else []
    if producers and Path(producers[-1]).is_file():
        texts.insert(0, _tail(Path(producers[-1]), before.get(Path(producers[-1]), 0)))
    for text in texts:
        for cls, sub, pattern in SIGNATURES:
            found = pattern.search(text)
            if found:
                line = text[max(0, found.start() - 200) : found.end()].splitlines()[-1]
                return Failure(cls, sub, line.strip()[:400])
    last = next((line for line in reversed("".join(texts).splitlines()) if line.strip()), "")
    return Failure("unknown", "unclassified", last.strip()[:400])


def transient(run_log: Path, before: dict[Path, int]) -> bool:
    return classify(run_log, before).cls == "transient"


def receipts(roots: tuple[Path, ...]) -> int:
    """Finished stage, part and chunk receipts under the step's work roots (its progress)."""
    return sum(
        1 for root in roots if root.is_dir() for pattern in RECEIPTS for _ in root.rglob(pattern)
    )


def _record(logs: Path, step: str, row: dict[str, object]) -> None:
    path = logs / f"{step}-{time.time_ns()}.failure.json"
    path.write_text(json.dumps({"step": step, **row}, indent=1) + "\n", encoding="utf-8")


def run_with_retry(
    execute: Execute,
    run_log: Callable[[], Path],
    *,
    logs: Path,
    step: str,
    progress: Callable[[], int] = lambda: 0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    recording: str = "",
    notice: Callable[..., None] | None = None,
    volumes: Callable[[], volume_events.Snapshot] = dict,
    recover: Recover = volume_events.recover,
    receipt_roots: tuple[Path, ...] = (),
) -> None:
    """Run the step; rerun a transient failure under backoff, intensity and progress limits.

    ``volumes`` snapshots the mounted volumes at each attempt's start (none by default), so a
    missing file while one detached counts as ``volume``.
    """
    retries: list[float] = []
    previous_progress: int | None = None
    cleared = recovered = False
    while True:
        before, started, mounted = log_sizes(logs), progress(), volumes()
        attempt = run_log()
        try:
            execute(attempt)
        except StageError:
            failure = classify(attempt, before)
            if failure.sub == "missing input or stale state" and volume_events.detached(
                mounted, volumes()
            ):
                failure = Failure("volume", "volume detached during the step", failure.cause)
            made = progress() - started
            recent = [moment for moment in retries if clock() - moment < WINDOW_SECONDS]
            reason = (
                None
                if failure.cls == "transient"
                or (failure.cls == "stale" and not cleared)
                or (failure.cls == "volume" and not recovered)
                else f"{failure.cls} failures are not retried"
            )
            transient = failure.cls == "transient"
            if reason is None and transient and len(recent) >= MAX_RETRIES:
                reason = f"{MAX_RETRIES} retries within {WINDOW_SECONDS:.0f} s"
            if reason is None and transient and previous_progress is not None and made <= 0:
                reason = "the previous attempt made no progress"
            if reason is None and failure.cls == "volume":
                recovered = True  # one retry: a second detach needs a person
                reason = recover(mounted, receipt_roots, failed_at=time.time())
            _record(logs, step, {
                "attempt": len(retries) + 1, "runLog": str(attempt), "class": failure.cls,
                "subclass": failure.sub, "cause": failure.cause, "progress": made,
                "retried": reason is None, "stopReason": reason,
            })  # fmt: skip
            if reason is not None:
                if failure.cls != "operator":
                    action = "relaunch now" if failure.sub == "edited-mid-run" else "needs a person"
                    tell = failure_notify.notify if notice is None else notice
                    tell(recording, step, cls=failure.cls, cause=failure.cause, action=action)
                raise
            if failure.cls == "volume":
                print(f"{step}: {failure.sub}; the volume is back, relaunching once", flush=True)
                continue
            if failure.cls == "stale":
                cleared = True  # its next start clears the partial state; once is enough
                print(f"{step}: {failure.sub}; relaunching at once to clear it", flush=True)
                continue
            delay = BACKOFF_SECONDS[min(len(recent), len(BACKOFF_SECONDS) - 1)]
            print(f"{step}: {failure.sub} (see {attempt}); retry in {delay:.0f} s", flush=True)
            retries.append(clock())
            previous_progress = made
            sleep(delay)
            continue
        return
