"""Choose a worker count from measured headroom instead of a fixed number.

Measured on the 2 h Oct6 render: the native workers use ~55 MB each and CPU/GPU stayed
50-88% idle at 3 workers, while swap on this Mac sat at 11.7 of 12.3 GB because other
resident apps hold the memory. Memory, not CPU, is the limit, so more workers are allowed
only while swap has room and the machine is not already busy; otherwise the base count stays.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

SWAP_USED_LIMIT = 0.85
LOAD_LIMIT = 0.5
SWAP_PATTERN = re.compile(r"total = ([0-9.]+)M\s+used = ([0-9.]+)M")


@dataclass(frozen=True, slots=True)
class Resources:
    cpus: int
    load: float
    swap_total_mb: float
    swap_used_mb: float


def read_resources() -> Resources:
    """Current CPU count, 1-minute load and swap usage (swap unknown counts as full)."""
    match = SWAP_PATTERN.search(
        subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
        ).stdout
    )
    total, used = (float(match.group(1)), float(match.group(2))) if match else (1.0, 1.0)
    return Resources(os.cpu_count() or 1, os.getloadavg()[0], total, used)


def workers(
    base: int, *, maximum: int | None = None, read: Callable[[], Resources] = read_resources
) -> int:
    """`base` workers, doubled (up to `maximum`) when swap and load leave headroom."""
    if base < 1:
        raise ValueError("base worker count must be positive")
    ceiling = maximum if maximum is not None else base * 2
    resources = read()
    swap_used = resources.swap_used_mb / max(resources.swap_total_mb, 1.0)
    if swap_used >= SWAP_USED_LIMIT or resources.load >= LOAD_LIMIT * resources.cpus:
        return base
    return max(base, min(ceiling, resources.cpus // 2))
