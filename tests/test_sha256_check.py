from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from replay import sha256_check


def manifest(tmp_path: Path, content: bytes, digest: str | None = None) -> Path:
    target = tmp_path / "media.bin"
    target.write_bytes(content)
    line = f"{digest or hashlib.sha256(content).hexdigest()}  {target}\n"
    path = tmp_path / "closure.sha256"
    path.write_text(line, encoding="utf-8")
    return path


def arguments(path: Path) -> list[str]:
    return [
        str(path),
        "--manifest-sha256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "--scope",
        str(path.parent),
        "--workers",
        "4",
        "--memory-budget-bytes",
        "268435456",
        "--baseline-rss-bytes",
        "33554432",
        "--worker-rss-bytes",
        "33554432",
    ]


def test_matching_manifest_passes(tmp_path: Path) -> None:
    assert sha256_check.main(arguments(manifest(tmp_path, b"data"))) == 0


def test_tampered_input_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = manifest(tmp_path, b"data", digest="0" * 64)
    assert sha256_check.main(arguments(path)) == 1
    assert "FAILED" in capsys.readouterr().err


def test_missing_file_fails(tmp_path: Path) -> None:
    path = manifest(tmp_path, b"data")
    (tmp_path / "media.bin").unlink()
    assert sha256_check.main(arguments(path)) == 1


def test_malformed_line_and_usage(tmp_path: Path) -> None:
    bad = tmp_path / "bad.sha256"
    bad.write_text("not a digest\n", encoding="utf-8")
    assert sha256_check.main(arguments(bad)) == 2
    assert sha256_check.main([]) == 2
    assert sha256_check.main([str(tmp_path / "absent")]) == 2


@pytest.mark.parametrize("text", ["", "\n", "x" * (1024 * 1024 + 1), "0" * 64 + "  relative"])
def test_invalid_manifest(text: str) -> None:
    with pytest.raises(ValueError):
        sha256_check.parse(text)


@pytest.mark.parametrize(
    "workers,budget,baseline,worker",
    [
        (0, 100, 1, 1),
        (5, 100, 1, 1),
        (True, 100, 1, 1),
        (1, 100, 0, 1),
        (1, 100, 1, -1),
        (1, 100, 101, 1),
    ],
)
def test_invalid_budget(workers: int, budget: int, baseline: int, worker: int) -> None:
    with pytest.raises(ValueError):
        sha256_check.DigestLimits(workers, budget, baseline, worker)


@pytest.mark.parametrize("budget,expected", [(50331648, 1), (67108864, 2), (100663296, 4)])
def test_memory_admission(budget: int, expected: int) -> None:
    assert sha256_check.DigestLimits(4, budget, 33554432, 1).effective_workers(4) == expected


def test_invalid_entry_count() -> None:
    with pytest.raises(ValueError):
        sha256_check.DigestLimits(4, 268435456, 33554432, 33554432).effective_workers(0)


def test_empty_regular_file_supported(tmp_path: Path) -> None:
    assert sha256_check.main(arguments(manifest(tmp_path, b""))) == 0


def test_duplicate_entries(tmp_path: Path) -> None:
    path = manifest(tmp_path, b"data")
    path.write_text(path.read_text() * 2)
    assert sha256_check.main(arguments(path)) == 2


def test_excess_entries(tmp_path: Path) -> None:
    text = "".join(f"{'0' * 64}  {tmp_path / str(index)}\n" for index in range(257))
    with pytest.raises(ValueError):
        sha256_check.parse(text)


def test_nul_path() -> None:
    with pytest.raises(ValueError):
        sha256_check.parse("0" * 64 + "  /bad\0path")


def test_pinned_manifest_tamper(tmp_path: Path) -> None:
    path = manifest(tmp_path, b"data")
    args = arguments(path)
    path.write_text(path.read_text().replace("media.bin", "other.bin"))
    assert sha256_check.main(args) == 2


def test_scope_escape(tmp_path: Path) -> None:
    path = manifest(tmp_path, b"data")
    inside = tmp_path / "inside"
    inside.mkdir()
    args = arguments(path)
    args[4] = str(inside)
    assert sha256_check.main(args) == 2


@pytest.mark.parametrize("parent", [False, True])
def test_symlink_rejected(tmp_path: Path, parent: bool) -> None:
    path = manifest(tmp_path, b"data")
    link = tmp_path / "link"
    link.symlink_to(tmp_path if parent else tmp_path / "media.bin")
    target = link / "media.bin" if parent else link
    path.write_text(f"{'0' * 64}  {target}\n")
    assert sha256_check.main(arguments(path)) == 2


def test_nonregular_input(tmp_path: Path) -> None:
    path = manifest(tmp_path, b"data")
    path.write_text(f"{'0' * 64}  {tmp_path}\n")
    assert sha256_check.main(arguments(path)) == 1


def test_manifest_oversize(tmp_path: Path) -> None:
    path = manifest(tmp_path, b"data")
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    assert sha256_check.main(arguments(path)) == 2


def test_mutation_during_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "media.bin"
    target.write_bytes(b"original")
    real_fstat = os.fstat
    calls = 0

    def changed(fd: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write_bytes(b"changed")
        return real_fstat(fd)

    monkeypatch.setattr(os, "fstat", changed)
    with pytest.raises(ValueError, match="during"):
        sha256_check.digest(target)


def test_substitution_before_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "media.bin"
    target.write_bytes(b"original")
    real_open = os.open

    def replaced(path: Path, flags: int) -> int:
        target.unlink()
        target.write_bytes(b"substitute")
        return real_open(path, flags)

    monkeypatch.setattr(os, "open", replaced)
    with pytest.raises(ValueError, match="before"):
        sha256_check.digest(target)


def test_manifest_changed_during_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = manifest(tmp_path, b"data")
    args = arguments(path)

    def change(entries: list[tuple[str, Path]], *, limits: sha256_check.DigestLimits) -> list[Path]:
        path.write_text("changed")
        return []

    monkeypatch.setattr(sha256_check, "mismatches", change)
    assert sha256_check.main(args) == 2


def test_worker_order_and_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[int] = []

    def executor(*, max_workers: int) -> ThreadPoolExecutor:
        observed.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    monkeypatch.setattr(sha256_check, "ThreadPoolExecutor", executor)
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    failed = sha256_check.mismatches(
        [("0" * 64, first), ("0" * 64, second)],
        limits=sha256_check.DigestLimits(4, 50331648, 33554432, 1),
    )
    assert [path.name for path in failed] == ["first", "second"]
    assert observed == [1]


def test_unexpected_failure_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(path: Path) -> str:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(sha256_check, "digest", broken)
    with pytest.raises(RuntimeError, match="unexpected"):
        sha256_check.mismatches(
            [("0" * 64, tmp_path / "file")],
            limits=sha256_check.DigestLimits(1, 67108864, 33554432, 1),
        )


def test_concurrent_completion_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    second_finished = threading.Event()

    def reversed_digest(path: Path) -> str:
        if path.name == "first":
            if not second_finished.wait(timeout=2):
                raise RuntimeError("Second worker did not run concurrently")
        else:
            second_finished.set()
        return "f" * 64

    monkeypatch.setattr(sha256_check, "digest", reversed_digest)
    result = sha256_check.mismatches(
        [("0" * 64, first), ("0" * 64, second)],
        limits=sha256_check.DigestLimits(2, 100663296, 33554432, 33554432),
    )
    assert [path.name for path in result] == ["first", "second"]


def test_early_input_mutates_before_group_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    original_digest = sha256_check.digest

    def mutation(path: Path) -> str:
        value = original_digest(path)
        if path == second:
            first.write_bytes(b"changed after its successful hash")
        return value

    monkeypatch.setattr(sha256_check, "digest", mutation)
    result = sha256_check.mismatches(
        [(hashlib.sha256(b"one").hexdigest(), first), (hashlib.sha256(b"two").hexdigest(), second)],
        limits=sha256_check.DigestLimits(1, 100663296, 33554432, 33554432),
    )
    assert [path.name for path in result] == ["first"]


@pytest.mark.parametrize("text", ["/a/./file", "/a//file", "/a/../file"])
def test_noncanonical_spelling(text: str) -> None:
    with pytest.raises(ValueError):
        sha256_check.parse("0" * 64 + "  " + text)


def test_empty_scope_and_invalid_digest(tmp_path: Path) -> None:
    path = manifest(tmp_path, b"data")
    with pytest.raises(ValueError, match="SHA256"):
        sha256_check.verify(
            path,
            manifest_sha256="bad",
            scopes=(),
            limits=sha256_check.DigestLimits(1, 100663296, 33554432, 33554432),
        )
