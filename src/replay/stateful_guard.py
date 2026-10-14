"""External sampled watchdog, NOT an OS RSS cap or filesystem quota.

Run only as the session leader created by replay.workflow. The native child stays
in that owned process group; existing workflow timeout/SIGTERM cleanup therefore
also covers framework waits. SIGKILL of the workflow itself cannot run cleanup.
"""

import argparse
import os
import resource
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from replay.core import ReplayError
from replay.native_invocation import canonical, durable, read, require
from replay.recordings import record
from replay.stateful_audio import integer

POLL_SECONDS = 0.05
PROBE_TIMEOUT = 2


def saved_bytes(directory: Path) -> int:
    if not directory.exists():
        require(not directory.is_symlink(), "Linked output")
        return 0
    info = directory.lstat()
    require(stat.S_ISDIR(info.st_mode), "Output is not a directory")
    total = 0
    for path in directory.iterdir():
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), "Linked owned output")
        if stat.S_ISDIR(info.st_mode):
            total += saved_bytes(path)
        else:
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Invalid owned output")
            total += info.st_size
    return total


def group_rss(group: int) -> int:
    result = subprocess.run(
        ["/bin/ps", "-axo", "pgid=,rss="],
        check=True,
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT,
    )
    return sum(
        int(fields[1]) * 1024
        for line in result.stdout.splitlines()
        if len(fields := line.split()) == 2 and int(fields[0]) == group
    )


def supervise(*, request: Path, native: Path, report: Path) -> None:
    require(os.getpid() == os.getpgrp(), "Watchdog requires workflow-owned session")
    request, native, report = (canonical(str(path)) for path in (request, native, report))
    data = read(request)
    directory = canonical(data.get("outputDirectory"))
    require(
        directory.is_relative_to(Path.cwd() / "edits/work")
        and report.is_relative_to(Path.cwd() / "edits/work"),
        "Watchdog paths must stay local",
    )
    budget = record(record(data.get("statefulSelection")).get("budget"))
    rss_limit = integer(budget.get("maxRSSBytes"), 1)
    disk_limit = integer(budget.get("maxSavedBytes"), 1)
    wall = budget.get("maxWallSeconds")
    require(type(wall) in (int, float) and 0 < float(str(wall)) <= 3600, "Invalid deadline")
    require(
        not directory.exists() and not directory.is_symlink() and not report.exists(),
        "Guard never resumes partial native state",
    )
    durable(report.with_suffix(".started.json"), {"status": "started", "hardCap": False})
    # No new session: descendants remain inside the workflow's owned group.
    with subprocess.Popen([str(native), str(request)], stdin=subprocess.DEVNULL) as child:
        try:
            result = monitor(
                child,
                directory=directory,
                rss_limit=rss_limit,
                disk_limit=disk_limit,
                wall=float(str(wall)),
            )
            durable(report, result)
        except (OSError, ValueError, ReplayError, subprocess.SubprocessError) as error:
            # Stop before Popen.__exit__ can wait on a blocked native framework.
            try:
                if not report.exists():
                    durable(
                        report,
                        {"status": "failed", "reason": type(error).__name__, "hardCap": False},
                    )
            finally:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            raise


def monitor(
    child: subprocess.Popen[bytes], *, directory: Path, rss_limit: int, disk_limit: int, wall: float
) -> dict[str, object]:
    started = last = time.monotonic()
    peak_rss = peak_saved = 0
    max_gap = 0.0
    while True:
        rss, saved = group_rss(os.getpgrp()), saved_bytes(directory)
        now = time.monotonic()
        max_gap = max(max_gap, now - last)
        last = now
        peak_rss, peak_saved = max(peak_rss, rss), max(peak_saved, saved)
        require(
            rss <= rss_limit and saved <= disk_limit and now - started <= wall,
            "Sampled resource threshold or deadline exceeded",
        )
        code = child.poll()
        if code is not None:
            kernel_peak = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            if sys.platform != "darwin":
                kernel_peak *= 1024
            require(
                code == 0 and kernel_peak <= rss_limit,
                "Native failure or kernel child RSS peak exceeded",
            )
            break
        time.sleep(POLL_SECONDS)
    final_saved = saved_bytes(directory)
    require(final_saved <= disk_limit, "Final saved bytes exceeded")
    return {
        "status": "complete",
        "hardCap": False,
        "sampledGroupPeakRSSBytes": peak_rss,
        "kernelChildPeakRSSBytes": kernel_peak,
        "sampledPeakSavedBytes": peak_saved,
        "finalSavedBytes": final_saved,
        "pollSeconds": POLL_SECONDS,
        "maximumObservedSampleGapSeconds": max_gap,
        "overshootBoundBytes": None,
        "betweenSamplePeaksExcluded": False,
        "scope": "owned process group RSS; native output logical bytes; report excluded",
    }


def verify_guard(request: Path, report: Path) -> None:
    budget = record(record(read(request).get("statefulSelection")).get("budget"))
    result = read(report)
    require(
        result.get("status") == "complete" and result.get("hardCap") is False,
        "Watchdog did not complete",
    )
    require(
        max(
            integer(result.get("sampledGroupPeakRSSBytes"), 1),
            integer(result.get("kernelChildPeakRSSBytes"), 1),
        )
        <= integer(budget.get("maxRSSBytes"), 1),
        "RSS observation exceeded threshold",
    )
    directory = canonical(read(request).get("outputDirectory"))
    require(
        saved_bytes(directory) == result.get("finalSavedBytes")
        and max(
            integer(result.get("finalSavedBytes")), integer(result.get("sampledPeakSavedBytes"))
        )
        <= integer(budget.get("maxSavedBytes"), 1),
        "Saved bytes changed/exceeded threshold",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    supervise(request=args.request, native=args.native, report=args.report)


if __name__ == "__main__":
    main()
