"""The default pytest scratch sweep removes only finished, old runs and protects live ones."""

import fcntl
import hashlib
import os
import time
from pathlib import Path

from replay import pytest_scratch

OLD = time.time() - 7200
NEW = time.time() - 60


def _lock_name(run: Path) -> str:
    return hashlib.sha256(os.fsencode(run / "pytest")).hexdigest()


def test_sweep_removes_old_finished_runs_beyond_the_newest(tmp_path: Path) -> None:
    newest = tmp_path / ("run-" + "a" * 32)
    middle = tmp_path / ("run-" + "b" * 32)
    oldest = tmp_path / ("run-" + "c" * 32)
    newest.mkdir()
    middle.mkdir()
    oldest.mkdir()
    os.utime(newest, (OLD + 20, OLD + 20))
    os.utime(middle, (OLD + 10, OLD + 10))
    os.utime(oldest, (OLD, OLD))
    removed = pytest_scratch.sweep(tmp_path, keep=1, min_age=3600)
    assert removed == [middle, oldest]
    assert newest.is_dir()
    assert not middle.exists()
    assert not oldest.exists()


def test_sweep_keeps_runs_younger_than_the_minimum_age(tmp_path: Path) -> None:
    young = tmp_path / ("run-" + "d" * 32)
    young.mkdir()
    os.utime(young, (NEW, NEW))
    assert pytest_scratch.sweep(tmp_path, keep=0, min_age=3600) == []
    assert young.is_dir()


def test_sweep_protects_a_run_whose_lock_is_held(tmp_path: Path) -> None:
    live = tmp_path / ("run-" + "e" * 32)
    live.mkdir()
    os.utime(live, (OLD, OLD))
    locks = tmp_path / "locks"
    locks.mkdir()
    lock = locks / _lock_name(live)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    try:
        assert pytest_scratch.sweep(tmp_path, keep=0, min_age=3600) == []
    finally:
        os.close(descriptor)
    assert live.is_dir()


def test_sweep_removes_the_lock_file_of_a_removed_run(tmp_path: Path) -> None:
    done = tmp_path / ("run-" + "f" * 32)
    done.mkdir()
    os.utime(done, (OLD, OLD))
    locks = tmp_path / "locks"
    locks.mkdir()
    lock = locks / _lock_name(done)
    lock.write_text("", encoding="utf-8")
    assert pytest_scratch.sweep(tmp_path, keep=0, min_age=3600) == [done]
    assert not lock.exists()


def test_sweep_ignores_symlinks_and_unrelated_names(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / ("run-" + "1" * 32)
    link.symlink_to(target, target_is_directory=True)
    other = tmp_path / "run-not-a-uuid"
    other.mkdir()
    os.utime(other, (OLD, OLD))
    assert pytest_scratch.sweep(tmp_path, keep=0, min_age=0) == []
    assert target.is_dir()
    assert other.is_dir()
