import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from replay import intermediate_offload, leaf_retirement, r2_worker, work_offload
from replay.core import ReplayError
from replay.r2_worker import RemoteObject

UPLOADED = datetime(2026, 10, 17, 3, 0, tzinfo=UTC)
LATER = datetime.now(UTC) + timedelta(days=1)
EARLIER = datetime.now(UTC) - timedelta(days=1)


class FakeStore:
    """In-memory R2 double; `fail` makes every upload raise."""

    def __init__(self, *, fail: bool = False) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.fail = fail

    def head(self, key: str) -> RemoteObject | None:
        stored = self.objects.get(key)
        if stored is None:
            return None
        return RemoteObject(len(stored[0]), UPLOADED, dict(stored[1]))

    def put(self, key: str, path: Path, metadata: Mapping[str, str]) -> None:
        if self.fail:
            raise ReplayError("synthetic upload failure")
        self.objects[key] = (path.read_bytes(), dict(metadata))

    def get(self, key: str, destination: Path) -> int:
        data = self.objects[key][0]
        destination.write_bytes(data)
        return len(data)


def closed(paths: Sequence[str]) -> set[str]:
    return set()


@pytest.fixture(autouse=True)
def small_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work_offload, "MIN_BYTES", 1)


def media(path: Path, data: bytes = b"synthetic scratch") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def work_root(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    return Path(os.path.realpath(root))


def test_scan_takes_only_quiet_single_link_regular_files(tmp_path: Path) -> None:
    unit = work_root(tmp_path) / "pilot"
    kept = media(unit / "render" / "frames.mov")
    media(unit / ".git" / "objects" / "pack.pack")
    media(unit / "evidence" / intermediate_offload.MANIFEST_NAME)
    media(unit / "evidence" / intermediate_offload.LOCK_NAME)
    media(unit / "evidence" / f".{intermediate_offload.MANIFEST_NAME}.tmp")
    media(unit / f"restore.mov{intermediate_offload.PARTIAL_SUFFIX}")
    linked = media(unit / "linked.mov")
    os.link(linked, unit / "linked-again.mov")
    (unit / "symlink.mov").symlink_to(kept)
    rows = work_offload.scan(unit, int(LATER.timestamp() * 1e9))
    assert [row.path for row in rows] == [str(kept)]


def test_scan_skips_recent_and_sparse_files(tmp_path: Path) -> None:
    unit = work_root(tmp_path) / "pilot"
    media(unit / "fresh.mov")
    with (unit / "sparse.mp4").open("wb") as stream:
        stream.truncate(64 * 1024 * 1024)
    assert work_offload.scan(unit, int(EARLIER.timestamp() * 1e9)) == []
    assert [
        Path(row.path).name for row in work_offload.scan(unit, int(LATER.timestamp() * 1e9))
    ] == ["fresh.mov"]


def test_scan_ignores_a_file_that_vanished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unit = work_root(tmp_path) / "pilot"
    gone = media(unit / "gone.mov")
    lstat = os.lstat

    def vanished(path: str) -> os.stat_result:
        if path == str(gone):
            raise FileNotFoundError(path)
        return lstat(path)

    monkeypatch.setattr(os, "lstat", vanished)
    assert work_offload.scan(unit, int(LATER.timestamp() * 1e9)) == []


def test_units_skip_own_directories_files_and_symlinks(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    (root / "pilot").mkdir()
    (root / r2_worker.PROJECT_ROOT.name).mkdir()
    (root / work_offload.LOG_DIRECTORY).mkdir()
    media(root / "notes.json")
    (root / "alias").symlink_to(root / "pilot")
    assert [unit.name for unit in work_offload.units(root)] == ["pilot"]


def test_units_leave_skipped_names_alone(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    (root / "active").mkdir()
    (root / "pilot").mkdir()
    assert [unit.name for unit in work_offload.units(root, frozenset({"active"}))] == ["pilot"]


def test_main_skips_named_units(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = work_root(tmp_path)
    media(root / "active" / "frames.mov")
    media(root / "pilot" / "frames.mov")
    arguments = ["--work-root", str(root), "--quiet-days", "-1", "--skip", "active"]
    assert work_offload.main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["candidates"] == 1


def test_dry_run_plans_without_touching_anything(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    source = media(root / "pilot" / "frames.mov")
    (root / "empty").mkdir()
    results = work_offload.sweep(root, cutoff=LATER, store=None, log=tmp_path / "log.jsonl")
    assert [(result.name, result.candidates) for result in results] == [("pilot", 1)]
    assert source.exists()
    assert not (root / "pilot" / "evidence").exists()


def test_apply_uploads_retires_and_restores(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    source = media(root / "pilot" / "render" / "frames.mov")
    os.chmod(source, 0o640)
    log = tmp_path / "log.jsonl"
    store = FakeStore()
    [result] = work_offload.sweep(root, cutoff=LATER, store=store, log=log, probe=closed)
    assert (result.uploaded, result.retired, result.error) == (1, 1, None)
    assert result.retired_bytes == len(b"synthetic scratch")
    assert not source.exists()
    assert intermediate_offload.status(root / "pilot", recording="pilot")["retired"] == 1
    [key] = store.objects
    assert key.startswith(f"{intermediate_offload.KEY_PREFIX}/pilot/")
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[0])["removed"] is True
    intermediate_offload.restore(root / "pilot", recording="pilot", store=store)
    assert source.read_bytes() == b"synthetic scratch"
    assert source.stat().st_mode & 0o777 == 0o640


def test_rerun_after_restore_reuses_the_object(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    media(root / "pilot" / "frames.mov")
    store = FakeStore()
    work_offload.sweep(root, cutoff=LATER, store=store, log=tmp_path / "log", probe=closed)
    intermediate_offload.restore(root / "pilot", recording="pilot", store=store)
    [result] = work_offload.sweep(
        root, cutoff=LATER, store=store, log=tmp_path / "log", probe=closed
    )
    assert (result.uploaded, result.reused, result.retired) == (0, 1, 1)


def test_open_file_is_uploaded_but_held(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    source = media(root / "pilot" / "frames.mov")
    [result] = work_offload.sweep(
        root, cutoff=LATER, store=FakeStore(), log=tmp_path / "log", probe=lambda paths: set(paths)
    )
    assert (result.uploaded, result.retired, result.held) == (1, 0, {"open handle": 1})
    assert source.exists()
    assert intermediate_offload.status(root / "pilot", recording="pilot")["uploaded"] == 1


def test_changed_row_is_skipped_and_never_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = work_root(tmp_path)
    source = media(root / "pilot" / "frames.mov")
    scan = work_offload.scan

    def stale(unit: Path, cutoff_ns: int) -> list[leaf_retirement.Row]:
        rows = scan(unit, cutoff_ns)
        source.write_bytes(b"rewritten while the sweep ran")
        return rows

    monkeypatch.setattr(work_offload, "scan", stale)
    store = FakeStore()
    [result] = work_offload.sweep(
        root, cutoff=LATER, store=store, log=tmp_path / "log", probe=closed
    )
    assert (result.uploaded, result.retired, result.held) == (0, 0, {"identity changed": 1})
    assert source.read_bytes() == b"rewritten while the sweep ran"
    assert store.objects == {}


def test_failed_unit_deletes_nothing_and_the_sweep_continues(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    first = media(root / "a-pilot" / "frames.mov")
    second = media(root / "b-pilot" / "frames.mov", b"other scratch")
    calls: list[str] = []

    class FirstFails(FakeStore):
        def put(self, key: str, path: Path, metadata: Mapping[str, str]) -> None:
            calls.append(key)
            if len(calls) == 1:
                raise ReplayError("synthetic upload failure")
            super().put(key, path, metadata)

    results = work_offload.sweep(
        root, cutoff=LATER, store=FirstFails(), log=tmp_path / "log", probe=closed
    )
    assert [(result.name, result.retired, result.error) for result in results] == [
        ("a-pilot", 0, "synthetic upload failure"),
        ("b-pilot", 1, None),
    ]
    assert first.exists()
    assert not second.exists()


def test_report_totals_and_failures() -> None:
    results = [
        work_offload.UnitResult("a", 2, 10, uploaded=1, reused=1, retired=2, retired_bytes=10),
        work_offload.UnitResult("b", 1, 5, error="boom"),
    ]
    summary = work_offload.report(results, cutoff=UPLOADED, applied=True)
    assert summary == {
        "mode": "apply",
        "cutoff": "2026-10-17T12:00:00+09:00",
        "units": 2,
        "candidates": 3,
        "candidateBytes": 15,
        "uploaded": 1,
        "reused": 1,
        "retired": 2,
        "retiredBytes": 10,
        "failed": {"b": "boom"},
    }


def test_main_dry_run_prints_the_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = work_root(tmp_path)
    media(root / "pilot" / "frames.mov")
    assert work_offload.main(["--work-root", str(root), "--quiet-days", "-1"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed["mode"], printed["candidates"]) == ("dry-run", 1)
    assert (root / work_offload.LOG_DIRECTORY).is_dir()


def test_main_apply_runs_inside_one_worker_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = work_root(tmp_path)
    source = media(root / "pilot" / "frames.mov")
    buckets: list[str] = []
    store = FakeStore()

    @contextmanager
    def session(bucket: str) -> Iterator[FakeStore]:
        buckets.append(bucket)
        yield store

    monkeypatch.setattr(r2_worker, "session", session)
    assert work_offload.main(["--work-root", str(root), "--quiet-days", "-1", "--apply"]) == 0
    assert buckets == [intermediate_offload.BUCKET]
    assert json.loads(capsys.readouterr().out)["retired"] == 1
    assert not source.exists()


def test_main_reports_a_failed_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = work_root(tmp_path)
    media(root / "pilot" / "frames.mov")

    @contextmanager
    def session(bucket: str) -> Iterator[FakeStore]:
        yield FakeStore(fail=True)

    monkeypatch.setattr(r2_worker, "session", session)
    assert work_offload.main(["--work-root", str(root), "--quiet-days", "-1", "--apply"]) == 1
    assert json.loads(capsys.readouterr().out)["failed"] == {"pilot": "synthetic upload failure"}


def test_main_reports_a_session_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = work_root(tmp_path)

    @contextmanager
    def session(bucket: str) -> Iterator[FakeStore]:
        raise ReplayError("R2 Worker deploy failed")
        yield FakeStore()

    monkeypatch.setattr(r2_worker, "session", session)
    assert work_offload.main(["--work-root", str(root), "--apply"]) == 1
    assert "WORK_OFFLOAD_ERROR R2 Worker deploy failed" in capsys.readouterr().err
