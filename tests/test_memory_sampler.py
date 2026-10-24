"""The memory sampler parses the machine's counters and appends one line per interval."""

import json
import subprocess
from pathlib import Path

import pytest

from replay import memory_sampler

VM_STAT = (
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
    "Pages free:                                    38314.\n"
    "Pages active:                                 835555.\n"
    "Pages wired down:                             511383.\n"
    '"Translation faults":                     2266877548.\n'
    "Pages occupied by compressor:                 743880.\n"
    "Swapins:                                     6939841.\n"
    "Swapouts:                                    8307254.\n"
)
OUTPUTS = {
    "kern.memorystatus_level": "59\n",
    "vm_stat": VM_STAT,
    "vm.swapusage": "total = 10240.00M  used = 8902.75M  free = 1337.25M  (encrypted)\n",
    "ps": " 6380432 .venv/bin/python\n  1347568 /System/mds_stores\nbad row\n",
}


def fake(command: list[str]) -> str | None:
    return OUTPUTS.get(command[-1]) or OUTPUTS.get(command[0])


def test_a_sample_reads_level_pages_swap_and_the_largest_processes() -> None:
    reading = memory_sampler.sample(fake)
    reading.pop("t")
    assert reading == {
        "memoryFreePercent": 59,
        "pageBytes": 16384,
        "pages": {
            "free": 38314,
            "active": 835555,
            "wired": 511383,
            "compressor": 743880,
            "swapins": 6939841,
            "swapouts": 8307254,
        },
        "swapUsedMiB": 8902.75,
        "top": [
            {"rssKiB": 6380432, "command": ".venv/bin/python"},
            {"rssKiB": 1347568, "command": "/System/mds_stores"},
        ],
    }


def test_unreadable_counters_are_null() -> None:
    reading = memory_sampler.sample(lambda _command: None)
    reading.pop("t")
    assert reading == {
        "memoryFreePercent": None,
        "pageBytes": None,
        "pages": None,
        "swapUsedMiB": None,
        "top": None,
    }


def test_sampling_appends_until_the_block_ends(tmp_path: Path) -> None:
    output = tmp_path / "memory.jsonl"
    with memory_sampler.sampling(output, interval=60.0, run=fake):
        pass
    rows = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["memoryFreePercent"] == 59


def test_sampling_stops_quietly_when_its_log_is_gone(tmp_path: Path) -> None:
    with memory_sampler.sampling(tmp_path / "missing/memory.jsonl", run=fake):
        pass
    assert not (tmp_path / "missing").exists()


def test_a_failing_or_hanging_command_reads_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert memory_sampler.read(["false"]) is None
    assert memory_sampler.read(["/nonexistent/command"]) is None

    def hang(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("vm_stat", 10)

    monkeypatch.setattr(subprocess, "run", hang)
    assert memory_sampler.read(["vm_stat"]) is None


def test_a_command_output_is_returned() -> None:
    assert memory_sampler.read(["echo", "59"]) == "59\n"
