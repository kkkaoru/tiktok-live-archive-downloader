import os
import sys
from pathlib import Path

import pytest

from replay import digest_memo


def large(tmp_path: Path) -> Path:
    path = tmp_path / "large.bin"
    with path.open("wb") as stream:
        stream.truncate(digest_memo.MEMO_MINIMUM_BYTES)
    return path


def test_large_digest_is_reused_while_identity_is_unchanged(tmp_path: Path) -> None:
    calls: list[Path] = []

    def digest(path: Path) -> str:
        calls.append(path)
        return f"digest-{len(calls)}"

    wrapped = digest_memo.memoized(digest)
    path = large(tmp_path)
    assert wrapped(path) == "digest-1"
    assert wrapped(path) == "digest-1"
    os.utime(path, ns=(1, 1))
    assert wrapped(path) == "digest-2"


def test_small_files_are_always_rehashed(tmp_path: Path) -> None:
    calls: list[Path] = []

    def digest(path: Path) -> str:
        calls.append(path)
        return "small"

    path = tmp_path / "small.bin"
    path.write_bytes(b"x")
    wrapped = digest_memo.memoized(digest)
    assert wrapped(path) == "small"
    assert wrapped(path) == "small"
    assert calls == [path, path]


def test_install_patches_module_references() -> None:
    from replay import caption_measured_job, stages

    original = stages.file_digest
    references = {
        module: module.__dict__["file_digest"]
        for name, module in list(sys.modules.items())
        if name.startswith("replay.") and hasattr(module, "file_digest")
    }
    try:
        patched = digest_memo.install(original)
        assert "replay.caption_measured_job" in patched
        assert caption_measured_job.__dict__["file_digest"] is not original
    finally:
        restored = [
            module.__dict__.update(file_digest=value) for module, value in references.items()
        ]
        assert len(restored) == len(references)


def test_main_rejects_other_modules(capsys: pytest.CaptureFixture[str]) -> None:
    assert digest_memo.main(["replay.cli"]) == 2
    assert "usage" in capsys.readouterr().err


def test_main_runs_the_wrapped_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    from replay import foreground_collection, stages

    installed: list[digest_memo.Digest] = []
    monkeypatch.setattr(digest_memo, "install", lambda digest: installed.append(digest))
    seen: list[list[str]] = []
    monkeypatch.setattr(foreground_collection, "main", lambda: seen.append(sys.argv[1:]))
    monkeypatch.setattr(sys, "argv", ["digest_memo"])
    assert digest_memo.main(["replay.foreground_collection", "verify", "a", "b"]) == 0
    assert installed == [stages.file_digest]
    assert seen == [["verify", "a", "b"]]
