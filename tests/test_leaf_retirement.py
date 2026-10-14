from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import leaf_retirement as retirement
from replay.core import ReplayError


def _closed(_path: str) -> bool | None:
    return False


def _none_open(_paths: Sequence[str]) -> set[str] | None:
    return set()


def _row(path: Path, classification: str = "eligible") -> dict[str, object]:
    info = os.lstat(path)
    row: dict[str, object] = {
        "path": str(path),
        "kind": "symlink" if path.is_symlink() else "file",
        "dev": info.st_dev,
        "inode": info.st_ino,
        "nlink": info.st_nlink,
        "bytes": info.st_size,
        "mtimeNs": info.st_mtime_ns,
        "ctimeNs": info.st_ctime_ns,
        "classification": classification,
    }
    if path.is_symlink():
        row["linkTarget"] = os.readlink(path)
    return row


def _signed(directory: Path, rows: list[dict[str, object]]) -> tuple[Path, str]:
    path = directory / "dispositions.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path, retirement.sha256_of(path)


def _file(path: Path, data: bytes = b"media") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _real(path: Path) -> Path:
    return path.resolve()


def test_load_rejects_a_changed_disposition_file(tmp_path: Path) -> None:
    path, _digest = _signed(tmp_path, [])
    with pytest.raises(ReplayError, match="differs from the pinned value"):
        retirement.load_eligible(path, "0" * 64)


def test_load_requires_a_rows_array(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ReplayError, match="needs a rows array"):
        retirement.load_eligible(path, retirement.sha256_of(path))


def test_load_returns_only_eligible_rows(tmp_path: Path) -> None:
    keep = _file(tmp_path / "keep.mov")
    drop = _file(tmp_path / "drop.mov")
    path, digest = _signed(tmp_path, [_row(keep, "keep"), _row(drop)])
    assert [row.path for row in retirement.load_eligible(path, digest)] == [str(drop)]


def test_load_rejects_a_row_that_is_not_an_object(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"rows": [{"classification": "eligible"}]}), encoding="utf-8")
    with pytest.raises(ReplayError, match="needs a path and a file or symlink kind"):
        retirement.load_eligible(path, retirement.sha256_of(path))


def test_row_rejects_a_non_integer_field(tmp_path: Path) -> None:
    row = _row(_file(tmp_path / "a.mov"))
    row["inode"] = "7"
    with pytest.raises(ReplayError, match="inode must be an integer"):
        retirement._row(row)


def test_row_rejects_a_non_object() -> None:
    with pytest.raises(ReplayError, match="rows must be objects"):
        retirement._row("path")


def test_symlink_row_requires_its_target(tmp_path: Path) -> None:
    target = _file(tmp_path / "t.mov")
    (tmp_path / "l.mov").symlink_to(target)
    row = _row(tmp_path / "l.mov")
    del row["linkTarget"]
    with pytest.raises(ReplayError, match="exact target"):
        retirement._row(row)


def test_check_accepts_an_untouched_file(tmp_path: Path) -> None:
    row = retirement._row(_row(_file(tmp_path / "a.mov")))
    assert retirement.check(row, [_real(tmp_path)], _closed) is None


def test_check_holds_a_path_outside_the_roots(tmp_path: Path) -> None:
    row = retirement._row(_row(_file(tmp_path / "a.mov")))
    assert retirement.check(row, [Path("/nowhere")], _closed) == "outside the allowed roots"


def test_check_holds_a_path_with_a_symlinked_ancestor(tmp_path: Path) -> None:
    real = _file(tmp_path / "real" / "a.mov")
    (tmp_path / "alias").symlink_to(tmp_path / "real")
    row = retirement._row(_row(real) | {"path": str(tmp_path / "alias" / "a.mov")})
    assert retirement.check(row, [tmp_path], _closed) == "ancestor is not canonical"


def test_check_holds_an_absent_file(tmp_path: Path) -> None:
    path = _file(tmp_path / "a.mov")
    row = retirement._row(_row(path))
    path.unlink()
    assert retirement.check(row, [_real(tmp_path)], _closed) == "already absent"


def test_check_holds_a_file_that_changed(tmp_path: Path) -> None:
    path = _file(tmp_path / "a.mov")
    row = retirement._row(_row(path))
    path.write_bytes(b"changed media")
    assert retirement.check(row, [_real(tmp_path)], _closed) == "identity changed"


def test_check_holds_a_file_whose_size_differs(tmp_path: Path) -> None:
    path = _file(tmp_path / "a.mov")
    row = retirement._row(_row(path) | {"bytes": 999})
    assert retirement.check(row, [_real(tmp_path)], _closed) == "not the recorded regular file"


def test_check_holds_a_symlink_whose_target_changed(tmp_path: Path) -> None:
    target = _file(tmp_path / "t.mov")
    (tmp_path / "l.mov").symlink_to(target)
    row = retirement._row(_row(tmp_path / "l.mov") | {"linkTarget": "/elsewhere"})
    assert retirement.check(row, [_real(tmp_path)], _closed) == "symlink changed"


def test_check_holds_a_symlink_row_whose_path_is_a_regular_file(tmp_path: Path) -> None:
    path = _file(tmp_path / "a.mov")
    row = retirement._row(_row(path) | {"kind": "symlink", "linkTarget": "x"})
    assert retirement.check(row, [_real(tmp_path)], _closed) == "symlink changed"


def test_check_holds_a_file_with_an_open_handle(tmp_path: Path) -> None:
    path = _file(tmp_path / "a.mov")
    row = retirement._row(_row(path))
    assert retirement.check(row, [_real(tmp_path)], lambda _p: True) == "open handle"


def test_dry_run_unlinks_nothing(tmp_path: Path) -> None:
    path = _file(tmp_path / "a.mov")
    dispositions, digest = _signed(tmp_path, [_row(path)])
    summary = retirement.retire(
        dispositions,
        digest,
        [_real(tmp_path)],
        tmp_path / "log.jsonl",
        apply=False,
        probe=_none_open,
    )
    assert (summary.retired, summary.logical_bytes, path.exists()) == (1, 5, True)


def test_apply_unlinks_files_and_symlink_aliases_without_following_targets(tmp_path: Path) -> None:
    target = _file(tmp_path / "keep" / "target.mov")
    alias = tmp_path / "alias.mov"
    alias.symlink_to(target)
    victim = _file(tmp_path / "a.mov")
    dispositions, digest = _signed(tmp_path, [_row(alias), _row(victim)])
    summary = retirement.retire(
        dispositions,
        digest,
        [_real(tmp_path)],
        tmp_path / "log.jsonl",
        apply=True,
        probe=_none_open,
    )
    assert (summary.retired, summary.logical_bytes) == (2, 5)
    assert (alias.exists(), alias.is_symlink(), victim.exists(), target.exists()) == (
        False,
        False,
        False,
        True,
    )


def test_apply_logs_removed_and_held_rows(tmp_path: Path) -> None:
    kept = _file(tmp_path / "kept.mov")
    gone = _file(tmp_path / "gone.mov")
    dispositions, digest = _signed(tmp_path, [_row(kept), _row(gone)])
    gone.write_bytes(b"changed")
    log = tmp_path / "log.jsonl"
    summary = retirement.retire(
        dispositions, digest, [_real(tmp_path)], log, apply=True, probe=_none_open
    )
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert summary.held == {"identity changed": 1}
    assert lines == [
        {"path": str(gone), "held": "identity changed"},
        {"path": str(kept), "removed": True, "kind": "file", "bytes": 5},
    ]


def test_main_prints_a_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _file(tmp_path / "a.mov")
    dispositions, digest = _signed(tmp_path, [_row(path)])
    monkeypatch.setattr(retirement, "lsof_is_open", _closed)
    code = retirement.main(
        [
            str(dispositions),
            "--sha256",
            digest,
            "--root",
            str(_real(tmp_path)),
            "--log",
            str(tmp_path / "l.jsonl"),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "mode": "dry-run",
        "eligible": 1,
        "retired": 1,
        "logicalBytes": 5,
        "held": {},
    }


def test_check_holds_when_the_open_handle_probe_fails(tmp_path: Path) -> None:
    row = retirement._row(_row(_file(tmp_path / "a.mov")))
    assert retirement.check(row, [_real(tmp_path)], lambda _p: None) == "open-handle check failed"


def test_roots_match_on_directory_boundaries_not_string_prefixes(tmp_path: Path) -> None:
    sibling = _file(tmp_path / "work-old" / "a.mov")
    row = retirement._row(_row(sibling))
    assert retirement.check(row, [_real(tmp_path / "work")], _closed) == "outside the allowed roots"


def test_the_open_handle_probe_runs_once_per_chunk(tmp_path: Path) -> None:
    paths = [_file(tmp_path / f"{name}.mov") for name in ("a", "b", "c")]
    dispositions, digest = _signed(tmp_path, [_row(path) for path in paths])
    probed: list[list[str]] = []

    def probe(batch: Sequence[str]) -> set[str] | None:
        probed.append(list(batch))
        return set()

    retirement.retire(
        dispositions,
        digest,
        [_real(tmp_path)],
        tmp_path / "log.jsonl",
        apply=False,
        probe=probe,
        chunk_size=2,
    )
    assert probed == [[str(paths[0]), str(paths[1])], [str(paths[2])]]


def test_a_failed_probe_holds_the_whole_chunk_and_unlinks_nothing(tmp_path: Path) -> None:
    first = _file(tmp_path / "a.mov")
    second = _file(tmp_path / "b.mov")
    dispositions, digest = _signed(tmp_path, [_row(first), _row(second)])
    summary = retirement.retire(
        dispositions,
        digest,
        [_real(tmp_path)],
        tmp_path / "log.jsonl",
        apply=True,
        probe=lambda _paths: None,
    )
    assert (summary.retired, summary.held, first.exists(), second.exists()) == (
        0,
        {"open-handle check failed": 2},
        True,
        True,
    )


def test_only_the_open_path_in_a_chunk_is_held(tmp_path: Path) -> None:
    busy = _file(tmp_path / "a.mov")
    idle = _file(tmp_path / "b.mov")
    dispositions, digest = _signed(tmp_path, [_row(busy), _row(idle)])
    summary = retirement.retire(
        dispositions,
        digest,
        [_real(tmp_path)],
        tmp_path / "log.jsonl",
        apply=True,
        probe=lambda _paths: {str(busy)},
    )
    assert (summary.retired, summary.held, busy.exists(), idle.exists()) == (
        1,
        {"open handle": 1},
        True,
        False,
    )


def test_a_row_that_changes_after_the_probe_is_held_not_unlinked(tmp_path: Path) -> None:
    path = _file(tmp_path / "a.mov")
    dispositions, digest = _signed(tmp_path, [_row(path)])

    def probe(_paths: Sequence[str]) -> set[str] | None:
        path.write_bytes(b"rewritten after the probe")
        return set()

    summary = retirement.retire(
        dispositions, digest, [_real(tmp_path)], tmp_path / "log.jsonl", apply=True, probe=probe
    )
    assert (summary.retired, summary.held, path.exists()) == (0, {"identity changed": 1}, True)


def test_apply_holds_an_open_file_and_keeps_it(tmp_path: Path) -> None:
    path = _file(tmp_path / "a.mov")
    dispositions, digest = _signed(tmp_path, [_row(path)])
    summary = retirement.retire(
        dispositions,
        digest,
        [_real(tmp_path)],
        tmp_path / "log.jsonl",
        apply=True,
        probe=lambda paths: set(paths),
    )
    assert (summary.retired, summary.held, path.exists()) == (0, {"open handle": 1}, True)


def test_lsof_sees_a_file_this_process_holds_open(tmp_path: Path) -> None:
    path = _file(tmp_path / "held.mov")
    with path.open("rb"):
        assert retirement.lsof_is_open(str(path)) is True


def test_lsof_reports_a_closed_file_as_not_open(tmp_path: Path) -> None:
    assert retirement.lsof_is_open(str(_file(tmp_path / "idle.mov"))) is False


def test_lsof_failure_is_not_read_as_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failed = subprocess.CompletedProcess(["lsof"], 2, stdout="", stderr="lsof: broken")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: failed)
    assert retirement.lsof_is_open(str(tmp_path / "x")) is None


def test_lsof_timeout_is_not_read_as_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def hang(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["lsof"], 1)

    monkeypatch.setattr(subprocess, "run", hang)
    assert retirement.lsof_is_open(str(tmp_path / "x")) is None


def test_batch_lsof_finds_only_the_open_path(tmp_path: Path) -> None:
    held = _file(tmp_path / "held.mov")
    idle = _file(tmp_path / "idle.mov")
    with held.open("rb"):
        assert retirement.lsof_open_among([str(held), str(idle)]) == {str(held)}


def test_batch_lsof_reports_an_empty_set_when_nothing_is_open(tmp_path: Path) -> None:
    paths = [str(_file(tmp_path / "a.mov")), str(_file(tmp_path / "b.mov"))]
    assert retirement.lsof_open_among(paths) == set()


def test_batch_lsof_failure_is_not_read_as_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failed = subprocess.CompletedProcess(["lsof"], 2, stdout="", stderr="lsof: broken")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: failed)
    assert retirement.lsof_open_among([str(tmp_path / "x")]) is None


def test_batch_lsof_output_naming_another_path_is_unusable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    odd = subprocess.CompletedProcess(["lsof"], 0, stdout="n/somewhere/else\n", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: odd)
    assert retirement.lsof_open_among([str(tmp_path / "x")]) is None


def test_batch_lsof_timeout_is_not_read_as_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def hang(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["lsof"], 1)

    monkeypatch.setattr(subprocess, "run", hang)
    assert retirement.lsof_open_among([str(tmp_path / "x")]) is None
