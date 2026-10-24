"""Code snapshots: a running step keeps its code while src/ is edited, and is recorded."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from replay import code_snapshot

LATE_IMPORT = "import time; time.sleep(0.5); import replay.probe as p; print(p.VALUE)"


def package(tmp_path: Path, value: int) -> Path:
    replay = tmp_path / "src" / "replay"
    replay.mkdir(parents=True, exist_ok=True)
    (replay / "__init__.py").write_text("", "utf-8")
    (replay / "probe.py").write_text(f"VALUE = {value}\n", "utf-8")
    return replay


def test_a_running_step_keeps_its_code_while_src_is_edited(tmp_path: Path) -> None:
    source = package(tmp_path, 1)
    sha, snapshot = code_snapshot.materialise(source, tmp_path / "snapshots")
    step = subprocess.Popen(
        [sys.executable, "-c", LATE_IMPORT],
        env={**os.environ, "PYTHONPATH": str(snapshot)}, stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    (source / "probe.py").write_text("VALUE = 2\n", "utf-8")  # the edit, mid-run (a lazy import)
    out, _ = step.communicate(timeout=20)
    assert out.strip() == "1"
    assert code_snapshot.content_sha(source) != sha
    assert (snapshot / "replay" / "probe.py").read_text("utf-8") == "VALUE = 1\n"


def test_snapshots_are_deduplicated_and_read_only(tmp_path: Path) -> None:
    source = package(tmp_path, 1)
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "probe.pyc").write_bytes(b"x")
    first = code_snapshot.materialise(source, tmp_path / "snapshots")
    assert code_snapshot.materialise(source, tmp_path / "snapshots") == first
    files = sorted(path.name for path in (first[1] / "replay").iterdir())
    assert files == ["__init__.py", "probe.py"]
    with pytest.raises(PermissionError):
        (first[1] / "replay" / "probe.py").write_text("VALUE = 3\n", "utf-8")


def test_prune_keeps_snapshots_in_use_or_named_by_receipts(tmp_path: Path) -> None:
    snapshots, recordings = tmp_path / "snapshots", tmp_path / "recordings"
    old_sha, old = code_snapshot.materialise(package(tmp_path / "a", 1), snapshots)
    used_sha, used = code_snapshot.materialise(package(tmp_path / "b", 2), snapshots)
    named_sha, named = code_snapshot.materialise(package(tmp_path / "c", 3), snapshots)
    for path in (old, used, named):
        os.utime(path, (1, 1))
    (recordings / "replay-1" / "steps").mkdir(parents=True)
    (recordings / "replay-1" / "steps" / "paging.json").write_text(
        json.dumps({"codeSnapshot": named_sha}), "utf-8"
    )
    (recordings / "replay-1" / "steps" / "bad.json").write_text("[", "utf-8")
    with code_snapshot.in_use(used):
        assert code_snapshot.prune(snapshots, recordings) == [old_sha]
        assert code_snapshot.prune(snapshots, recordings, dry_run=False) == [old_sha]
    assert sorted(path.name for path in snapshots.iterdir()) == sorted([used_sha, named_sha])
    assert code_snapshot.prune(snapshots, recordings, keep_hours=0.0) == [used_sha]


def test_snapshots_are_off_until_enabled(tmp_path: Path) -> None:
    assert code_snapshot.enabled(tmp_path) is False
    (tmp_path / "settings.json").write_text('{"enabled": true}', "utf-8")
    assert code_snapshot.enabled(tmp_path) is True


def test_a_change_during_the_copy_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = package(tmp_path, 1)
    shas = iter(["first", "second"])
    monkeypatch.setattr(code_snapshot, "content_sha", lambda _p: next(shas))
    with pytest.raises(ValueError, match="changed while its snapshot was taken"):
        code_snapshot.materialise(source, tmp_path / "snapshots")


def test_the_cli_lists_unused_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(code_snapshot, "root", lambda: tmp_path / "snapshots")
    (tmp_path / "snapshots").mkdir()
    assert code_snapshot.main(["prune", "--recordings", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"unused": [], "deleted": False}
