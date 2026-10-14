"""Bounded, read-only macOS process evidence; never invokes Compressor.

ps identities have one-second resolution, and snapshots are not atomic. Neither
PPID nor temporal overlap proves why a helper launched or caused a failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

FIELDS = "pid=,ppid=,uid=,etime=,pcpu=,lstart=,comm="
LIMITATIONS = (
    "Sampling misses short calls; first-seen is not creation time. "
    "ps start identity has second resolution; snapshots and argument reads race. "
    "Arguments are flattened by ps: categories are flag hints, not parsed argv. "
    "PPID 1 (including LaunchServices) cannot establish ownership. "
    "Parent links and timing do not prove launch cause or hang causality."
)


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    uid: int
    elapsed: str
    cpu: float | None
    start: str
    executable: str

    @property
    def identity(self) -> tuple[int, int, str, str]:
        return self.pid, self.uid, self.start, self.executable


@dataclass(frozen=True)
class Scope:
    executable: str
    helpers: tuple[str, ...]


def selected_scope(executable: Path) -> Scope:
    executable = executable.resolve(strict=True)
    if executable.parts[-3:] != ("Contents", "MacOS", "Compressor"):
        raise ValueError("Select an app's Contents/MacOS/Compressor executable")
    app = executable.parents[2]
    if app.suffix != ".app" or not executable.is_file():
        raise ValueError("Select an app's Contents/MacOS/Compressor executable")
    helpers = tuple(
        sorted(
            {
                str(path.resolve(strict=True))
                for path in (app / "Contents").rglob("RegisterProExtension")
                if path.is_file() and path.resolve(strict=True).is_relative_to(app)
            }
        )
    )
    if not helpers:
        raise ValueError("Selected app contains no RegisterProExtension executable")
    return Scope(str(executable), helpers)


def parse_processes(text: str) -> list[Process]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=10)
        if len(fields) != 11:
            raise ValueError("Malformed ps metadata row")
        pid, ppid, uid = map(int, fields[:3])
        cpu = None if fields[4] == "-" else float(fields[4])
        if pid <= 0 or ppid < 0 or (cpu is not None and not math.isfinite(cpu)):
            raise ValueError("Invalid ps metadata")
        rows.append(Process(pid, ppid, uid, fields[3], cpu, " ".join(fields[5:10]), fields[10]))
    return rows


def ps(arguments: list[str], *, deadline: float, scratch: Path) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Observation deadline reached")
    result = subprocess.run(
        ["/bin/ps", "-ww", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=min(2.0, remaining),
        env={
            **os.environ,
            "LC_ALL": "C",
            "TZ": "UTC",
            "TMPDIR": str(scratch),
            "TMP": str(scratch),
            "TEMP": str(scratch),
        },
    )
    # macOS ps returns 1 with no output when a selected PID has disappeared.
    if (
        result.returncode == 1
        and not result.stdout.strip()
        and not result.stderr.strip()
        and "-p" in arguments
    ):
        return ""
    if result.returncode or result.stderr.strip():
        raise RuntimeError(f"ps failed (exit {result.returncode}); observation incomplete")
    return result.stdout


def call_kind(arguments: str) -> str:
    tokens = arguments.split()
    monitor, submit = "-monitor" in tokens, "-jobpath" in tokens
    if monitor and submit:
        return "ambiguous"
    if monitor:
        return "monitor"
    return "submission" if submit else "other"


def sample(
    scope: Scope, *, deadline: float, scratch: Path
) -> tuple[list[Process], list[dict[str, object]]]:
    rows = parse_processes(ps(["-axo", FIELDS], deadline=deadline, scratch=scratch))
    selected = [p for p in rows if p.executable in (scope.executable, *scope.helpers)]
    calls: list[dict[str, object]] = []
    for process in selected:
        if process.executable != scope.executable:
            continue
        selector = ["-p", str(process.pid), "-o", FIELDS]
        before = parse_processes(ps(selector, deadline=deadline, scratch=scratch))
        if not before or before[0].identity != process.identity:
            calls.append({"process": asdict(process), "kind": "disappeared_or_changed"})
            continue
        # Only previously exact-matched CLI PIDs are queried, never helper/other argv.
        arguments = ps(["-p", str(process.pid), "-o", "args="], deadline=deadline, scratch=scratch)
        after = parse_processes(ps(selector, deadline=deadline, scratch=scratch))
        stable = bool(after and after[0].identity == process.identity and arguments.strip())
        calls.append(
            {
                "process": asdict(process),
                "kind": call_kind(arguments) if stable else "disappeared_or_changed",
            }
        )
    return selected, calls


def helper_events(
    rows: list[Process],
    *,
    scope: Scope,
    seen: set[tuple[int, int, str, str]],
    baseline: bool,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    calls = [p for p in rows if p.executable == scope.executable]
    for helper in rows:
        if helper.executable not in scope.helpers or helper.identity in seen:
            continue
        seen.add(helper.identity)
        parents = [p for p in calls if p.pid == helper.ppid and p.uid == helper.uid]
        events.append(
            {
                "helper": asdict(helper),
                "status": "baseline" if baseline else "first_seen",
                "observed_parent": asdict(parents[0]) if parents else None,
                "concurrent_cli_identities": [asdict(p) for p in calls],
                "confidence": "snapshot_parent_link" if parents else "temporal_only_or_unknown",
                "causal_attribution": False,
            }
        )
    return events


def observe(executable: Path, output: Path, *, duration: float, interval: float) -> int:
    if not (1 <= duration <= 120 and 0.25 <= interval <= 10):
        raise ValueError("Require duration 1..120 seconds and interval 0.25..10 seconds")
    root = Path(__file__).resolve().parents[2]
    work = root / "edits" / "work"
    parent = output.absolute().parent.resolve(strict=True)
    if not work.resolve(strict=True).is_relative_to(root) or not parent.is_relative_to(
        work.resolve()
    ):
        raise ValueError("Evidence must be under this checkout's edits/work")
    scope = selected_scope(executable)
    target = parent / output.name
    # Exclusive, private output; no existing evidence is replaced, including symlinks.
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        start = time.monotonic()
        deadline = start + duration
        samples: list[dict[str, object]] = []
        seen: set[tuple[int, int, str, str]] = set()
        error: str | None = None
        try:
            while time.monotonic() < deadline:
                elapsed = time.monotonic() - start
                rows, calls = sample(scope, deadline=deadline, scratch=parent)
                samples.append(
                    {
                        "processes": [asdict(p) for p in rows],
                        "calls": calls,
                        "events": helper_events(rows, scope=scope, seen=seen, baseline=not samples),
                        "elapsed_seconds": elapsed,
                        "utc_epoch_seconds": time.time(),
                    }
                )
                time.sleep(min(interval, max(0, deadline - time.monotonic())))
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"
        json.dump(
            {
                "version": 1,
                "scope": asdict(scope),
                "duration_seconds": duration,
                "interval_seconds": interval,
                "limitations": LIMITATIONS,
                "complete": error is None,
                "error": error,
                "samples": samples,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return 0 if error is None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    return observe(args.executable, args.output, duration=args.duration, interval=args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
