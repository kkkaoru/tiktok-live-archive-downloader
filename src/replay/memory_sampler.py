"""A background sampler of machine memory and swap while a recording pipeline runs.

Every ``INTERVAL_SECONDS`` one JSON line goes to ``memory.jsonl`` in the pipeline's step logs:
``kern.memorystatus_level`` (the free-memory level the admission gates read), selected
``vm_stat`` counters in pages (swap-ins and swap-outs are cumulative since boot), the swap in use
and the largest resident processes. Watchdog resets and swap thrash could not be tied to a step
without such a series. Reading costs three short commands per minute; a failed read leaves that
field ``null`` and never stops the pipeline.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

INTERVAL_SECONDS = 60.0
TOP_PROCESSES = 5
COMMAND_SECONDS = 10
VM_FIELDS = {
    "Pages free": "free",
    "Pages active": "active",
    "Pages inactive": "inactive",
    "Pages wired down": "wired",
    "Pages occupied by compressor": "compressor",
    "Swapins": "swapins",
    "Swapouts": "swapouts",
}
VM_LINE = re.compile(r'^"?([^":]+)"?:\s+([0-9]+)\.$')
PAGE_SIZE = re.compile(r"page size of ([0-9]+) bytes")
SWAP_USED = re.compile(r"used = ([0-9.]+)M")

Read = Callable[[list[str]], str | None]


def read(command: list[str]) -> str | None:
    """A command's output, or None when it fails or hangs."""
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=COMMAND_SECONDS
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def sample(run: Read = read) -> dict[str, object]:
    """One reading; fields that could not be read are None."""
    level = run(["sysctl", "-n", "kern.memorystatus_level"])
    vm = run(["vm_stat"])
    swap = run(["sysctl", "-n", "vm.swapusage"])
    ps = run(["ps", "-axo", "rss=,comm="])
    pages: dict[str, int] | None = None
    page_bytes: int | None = None
    if vm is not None:
        size = PAGE_SIZE.search(vm)
        page_bytes = int(size.group(1)) if size else None
        found = (VM_LINE.match(line.strip()) for line in vm.splitlines())
        pages = {
            VM_FIELDS[m.group(1)]: int(m.group(2)) for m in found if m and m.group(1) in VM_FIELDS
        }
    used = SWAP_USED.search(swap) if swap is not None else None
    top: list[dict[str, object]] | None = None
    if ps is not None:
        rows = (line.strip().split(None, 1) for line in ps.splitlines())
        sized = sorted(
            ((int(row[0]), row[1]) for row in rows if len(row) == 2 and row[0].isdigit()),
            reverse=True,
        )
        top = [{"rssKiB": rss, "command": name[-80:]} for rss, name in sized[:TOP_PROCESSES]]
    return {
        "t": round(time.time(), 1),
        "memoryFreePercent": int(level) if level and level.strip().isdigit() else None,
        "pageBytes": page_bytes,
        "pages": pages,
        "swapUsedMiB": float(used.group(1)) if used else None,
        "top": top,
    }


@contextmanager
def sampling(
    output: Path, *, interval: float = INTERVAL_SECONDS, run: Read = read
) -> Generator[None]:
    """Append a sample to ``output`` now and every ``interval`` until the block ends."""
    stop = threading.Event()

    def loop() -> None:
        while True:
            line = json.dumps(sample(run)) + "\n"
            try:
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(line)
            except OSError:
                return  # the log directory went away: sampling only ever observes
            if stop.wait(interval):
                return

    thread = threading.Thread(target=loop, name="memory-sampler", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
