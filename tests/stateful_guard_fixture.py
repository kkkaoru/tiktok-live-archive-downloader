#!/usr/bin/env python3
"""Bounded fake native child for process-group watchdog tests, not media processing."""

import os
import sys
import time
from pathlib import Path

from replay.native_invocation import read


def main() -> int:
    request = read(Path(sys.argv[1]))
    output = request.get("outputDirectory")
    if not isinstance(output, str):
        raise ValueError("Missing test output directory")
    directory = Path(output)
    directory.mkdir()
    (directory / "pid").write_text(str(os.getpid()), encoding="utf-8")
    (directory / "stateful.json").write_text("synthetic completion marker", encoding="utf-8")
    if request.get("testLink") is True:
        (directory / "link").symlink_to(directory / "pid")
    if request.get("testFailure") is True:
        return 1
    time.sleep(10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
