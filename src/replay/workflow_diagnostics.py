"""Run a workflow and keep a private, redacted failure record instead of one opaque line.

``replay.workflow`` deliberately prints only the exception class. This wrapper runs the same
plan and, on failure, writes ``<state>/last-failure.json`` with the exception chain, the
repository-authored message of ``StageError``/``ReplayError`` (redacted and truncated), the
errno of OS errors and the innermost source location of every link. Plan contents, command
arguments, timeout arguments and locals are never recorded.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

from replay import workflow
from replay.core import ReplayError
from replay.stages import StageError

FAILURE_FILE = "last-failure.json"
MAX_MESSAGE = 500
MAX_CHAIN = 6
SECRET = re.compile(
    r"(?i)((?:session|token|secret|password|passwd|cookie|authorization|api[_-]?key)[\w-]*"
    r"\s*[=:]\s*)\S+"
)
EXPECTED = (StageError, ReplayError)


def _message(error: BaseException) -> str:
    if isinstance(error, EXPECTED):
        return SECRET.sub(r"\1[redacted]", str(error))[:MAX_MESSAGE]
    if isinstance(error, OSError):
        return f"errno {error.errno}: {error.strerror}"
    return ""


def _where(error: BaseException) -> str:
    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        return ""
    last = frames[-1]
    return f"{Path(last.filename).name}:{last.name}:{last.lineno}"


def chain(error: BaseException) -> list[dict[str, str]]:
    """The exception and its causes/contexts, safest fields only."""
    links: list[dict[str, str]] = []
    current: BaseException | None = error
    while current is not None and len(links) < MAX_CHAIN:
        links.append(
            {
                "type": type(current).__qualname__,
                "message": _message(current),
                "where": _where(current),
            }
        )
        current = current.__cause__ or current.__context__
    return links


def record(state: Path, error: BaseException, *, plan: Path, targets: Sequence[str]) -> Path:
    """Atomically write the private failure record next to the stage state."""
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = state / FAILURE_FILE
    temporary = state / f".{FAILURE_FILE}.{os.getpid()}"
    payload = {
        "version": 1,
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plan": plan.name,
        "targets": list(targets),
        "chain": chain(error),
    }
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    temporary.replace(destination)
    return destination


def _cancel(_signum: int, _frame: FrameType | None) -> None:
    raise StageError("Workflow cancelled; partial outputs remain unapproved")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Trusted local JSON command manifest")
    parser.add_argument("--state", required=True, type=Path, help="Private checkpoint directory")
    parser.add_argument("--check", action="store_true", help="Print dependency graph; run nothing")
    parser.add_argument("--target", action="append", default=[])
    arguments = parser.parse_args(argv)
    if arguments.check:
        return workflow.main(argv)
    previous = signal.signal(signal.SIGTERM, _cancel)
    try:
        workflow.run_plan(
            arguments.plan, state_directory=arguments.state, targets=tuple(arguments.target)
        )
    except subprocess.TimeoutExpired as error:
        path = record(arguments.state, error, plan=arguments.plan, targets=arguments.target)
        print(f"Workflow deadline exceeded; details: {path}", file=sys.stderr)
        return 1
    except (StageError, OSError, ValueError) as error:
        path = record(arguments.state, error, plan=arguments.plan, targets=arguments.target)
        links = chain(error)
        head = f"{links[0]['type']}: {links[0]['message']}".rstrip(": ")
        print(f"Workflow stopped ({head}); details: {path}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
