"""Caption-render's Compressor guard: verified outputs, one resubmit, and the controller breaker."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from replay import compressor_guard as guard
from replay.core import ReplayError


def layout(tmp_path: Path, outputs: dict[str, bytes | None]) -> tuple[Path, Path]:
    states = tmp_path / "state"
    states.mkdir()
    rows = []
    for name, data in outputs.items():
        movie = tmp_path / name / "motion-alpha.mov"
        movie.parent.mkdir()
        if data is not None:
            movie.write_bytes(data)
        (states / f"{name}.json").write_text(
            json.dumps({"status": "Successful", "output": str(movie)}), "utf-8"
        )
        rows.append({"name": name, "frames": 3})
    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps(rows), "utf-8")
    return parts, states


def test_success_without_output_or_with_short_output_is_unverified(tmp_path: Path) -> None:
    parts, states = layout(tmp_path, {"p0": b"abc", "p1": None, "p2": b"ab"})
    assert guard.unverified(parts, states, lambda movie: len(movie.read_bytes())) == ["p1", "p2"]
    (states / "p0.json").unlink()  # not submitted yet: nothing to verify
    assert guard.unverified(parts, states, lambda movie: 3) == ["p1"]


def test_retiring_moves_the_state_and_partial_output_aside(tmp_path: Path) -> None:
    parts, states = layout(tmp_path, {"p1": None, "p2": b"ab"})
    guard.retire(states, ["p1", "p2"])
    assert sorted(path.name.split(".")[0] for path in states.iterdir()) == ["p1", "p2"]
    assert not (states / "p1.json").exists() and not (states / "p2.json").exists()
    assert list((tmp_path / "p2").iterdir()) == []  # nothing left for the batch to refuse
    assert [
        path.name.startswith("motion-alpha.unverified-") for path in tmp_path.glob("motion-*")
    ] == [True]


def test_retiring_a_failed_part_moves_the_output_named_by_its_job(tmp_path: Path) -> None:
    states = tmp_path / "state"
    states.mkdir()
    movie = tmp_path / "p3" / "motion-alpha.mov"
    movie.parent.mkdir()
    movie.write_bytes(b"")
    state = {"status": "submitted", "jobConfig": {"name": "p3", "output": str(movie)}}
    (states / "p3.json").write_text(json.dumps(state), "utf-8")
    guard.retire(states, ["p3"])
    assert list(movie.parent.iterdir()) == []
    assert [
        path.name.startswith("motion-alpha.unverified-") for path in tmp_path.glob("motion-*")
    ] == [True]
    assert [path.name.startswith("p3.unverified-") for path in states.iterdir()] == [True]


def test_verified_outputs_are_hashed(tmp_path: Path) -> None:
    parts, states = layout(tmp_path, {"p0": b"abc"})
    guard.record_hashes(parts, states, tmp_path / "out" / "verified-outputs.json")
    assert json.loads((tmp_path / "out" / "verified-outputs.json").read_text("utf-8")) == {
        "p0": {
            "path": str(tmp_path / "p0/motion-alpha.mov"),
            "sha256": hashlib.sha256(b"abc").hexdigest(),
        }
    }


def test_the_controller_down_signal_is_read_from_the_batchs_own_lines(tmp_path: Path) -> None:
    log = tmp_path / "caption-render.log"
    log.write_text("COMPRESSOR_BATCH_ERROR … ジョブコントローラがダウンしています\n", "utf-8")
    since = log.stat().st_size
    assert guard.controller_down(log, 0) is True
    assert guard.controller_down(log, since) is False  # an older batch's failure
    assert guard.controller_down(tmp_path / "none.log", 0) is False


class Machine:
    """A fake Compressor: running until quit (or killed); its controller starts after ``boot``."""

    def __init__(self, *, hangs: bool, boot: int) -> None:
        self.running, self.hangs, self.boot = True, hangs, boot
        self.calls: list[str] = []
        self.controller_checks = 0

    def __call__(self, command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(command[0].rsplit("/", 1)[-1] + " " + command[1])
        if command[0].endswith("osascript") and not self.hangs:
            self.running = False
        if command[0].endswith("pkill"):
            self.running = False
        if command[0].endswith("open"):
            self.running = True
        code = 1
        if command[0].endswith("pgrep") and command[-1] == guard.CONTROLLER:
            self.controller_checks += 1
            code = 0 if self.controller_checks > self.boot else 1
        elif command[0].endswith("pgrep"):
            code = 0 if self.running else 1
        return subprocess.CompletedProcess(command, code, "", "")


def test_a_restart_quits_reopens_and_waits_for_the_controller(tmp_path: Path) -> None:
    machine = Machine(hangs=False, boot=2)
    ticks = iter(range(10_000))
    guard.restart(tmp_path / "Compressor.app", call=machine, sleep=lambda _s: None,
                  clock=lambda: float(next(ticks)))  # fmt: skip
    assert machine.calls == [
        "osascript -e", "pgrep -f", "pgrep -f", "open -g", "pgrep -f", "pgrep -f", "pgrep -f",
    ]  # fmt: skip


def test_a_hanging_compressor_is_force_quit(tmp_path: Path) -> None:
    machine = Machine(hangs=True, boot=0)
    ticks = iter(range(0, 10_000, 10))
    guard.restart(tmp_path / "Compressor.app", call=machine, sleep=lambda _s: None,
                  clock=lambda: float(next(ticks)))  # fmt: skip
    assert "pkill -KILL" in machine.calls


def test_a_controller_that_never_starts_fails_the_restart(tmp_path: Path) -> None:
    machine = Machine(hangs=False, boot=10_000)
    ticks = iter(range(0, 100_000, 10))
    with pytest.raises(ReplayError, match="job controller did not start"):
        guard.restart(tmp_path / "Compressor.app", call=machine, sleep=lambda _s: None,
                      clock=lambda: float(next(ticks)))  # fmt: skip
