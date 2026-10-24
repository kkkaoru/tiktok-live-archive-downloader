"""Immediate local notice of a failure that needs a person: a macOS notification and a record.

Deterministic failures waited a median 17.5 min (p90 7 h) for a relaunch (failure survey,
2026-10-24) because nobody saw them. ``notify`` appends one JSON line to ``failures.jsonl``
beside the recording pipelines and shows a notification through ``osascript`` (local only, no
network). Neither may stop the pipeline: a failed notification is printed and ignored.
"""

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

Run = Callable[..., object]
OSASCRIPT = "/usr/bin/osascript"
TITLE = "Recording pipeline"
MAX_TEXT = 200


def record_path() -> Path:
    return Path(__file__).resolve().parents[2] / "edits/work/recording-pipeline/failures.jsonl"


def _quoted(text: str) -> str:
    return '"' + text[:MAX_TEXT].replace("\\", "\\\\").replace('"', '\\"') + '"'


def notify(
    recording: str,
    step: str,
    *,
    cls: str,
    cause: str,
    action: str,
    record: Path | None = None,
    run: Run = subprocess.run,
) -> None:
    """Record the failure and show a local notification; never raises."""
    row = {"t": round(time.time(), 1), "recording": recording, "step": step, "class": cls,
           "cause": cause, "action": action}  # fmt: skip
    path = record_path() if record is None else record
    script = (
        f"display notification {_quoted(f'{step}: {cause}')} with title {_quoted(TITLE)} "
        f"subtitle {_quoted(f'{recording} — {cls}, {action}')}"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        run([OSASCRIPT, "-e", script], check=False, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Failure notice not delivered: {error}", file=sys.stderr)
