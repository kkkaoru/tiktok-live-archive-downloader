"""Allocator entry preserves the existing CPU batch contract."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from replay import bounded_cpu_asr as entry
from replay.asr_runtime import ASRDevice
from replay.core import ReplayError


class Memory(ModuleType):
    def __init__(self) -> None:
        super().__init__("fake_core")
        self.cpu = "cpu"
        self.limit = 123

    def set_cache_limit(self, value: int) -> int:
        previous = self.limit
        self.limit = value
        return previous

    def get_active_memory(self) -> int:
        return 1

    def get_cache_memory(self) -> int:
        return 0

    def get_peak_memory(self) -> int:
        return 2

    def synchronize(self, cpu: object) -> None:
        assert cpu == self.cpu


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Memory]:
    scratch = tmp_path / "edits/work"
    scratch.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    # The entry requires a fresh provider process; an earlier test in this one may have imported it.
    for name in [key for key in sys.modules if key.split(".")[0] == "mlx_whisper"]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        monkeypatch.setenv(name, str(scratch))
    monkeypatch.setattr(entry, "DEFAULT_MODEL", tmp_path / "model")
    monkeypatch.setattr(entry, "audit_descriptors", lambda path, phase: None)
    module = Memory()
    monkeypatch.setattr("replay.bounded_cpu_asr.importlib.import_module", lambda name: module)
    return scratch, module


@pytest.mark.parametrize("limit", [0, 512 * 1024**2, 2 * 1024**3])
def test_entry_preserves_batch_contract(
    setup: tuple[Path, Memory],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    limit: int,
) -> None:
    scratch, module = setup
    source, output, evidence = scratch / "mono.wav", scratch / "asr", scratch / "policy"
    calls: list[tuple[Path, Path, ASRDevice]] = []

    def transcribe(first: Path, second: Path, *, device: ASRDevice) -> Path:
        assert module.limit == limit
        calls.append((first, second, device))
        second.mkdir()
        complete = second / "complete.json"
        complete.write_text("{}")
        return complete

    monkeypatch.setattr(entry, "transcribe_windows", transcribe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "entry",
            str(source),
            str(output),
            "--evidence",
            str(evidence),
            "--cache-limit-bytes",
            str(limit),
        ],
    )
    entry.main()
    assert calls == [(source, output, ASRDevice.CPU)]
    assert module.limit == 123
    assert str(output / "complete.json") in capsys.readouterr().out
    assert json.loads((evidence / "complete.json").read_text())["deliveryComplete"] is False


def test_failure_restores_policy(
    setup: tuple[Path, Memory], monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch, module = setup

    def fail(source: Path, output: Path, *, device: ASRDevice) -> Path:
        raise ReplayError("inference failed")

    monkeypatch.setattr(entry, "transcribe_windows", fail)
    with pytest.raises(ReplayError, match="inference failed"):
        entry.run(
            scratch / "source", scratch / "output", evidence=scratch / "evidence", cache_limit=0
        )
    assert module.limit == 123
    assert (scratch / "evidence/started.json").is_file()
    assert not (scratch / "evidence/complete.json").exists()


@pytest.mark.parametrize("limit", [-1, True, 2 * 1024**3 + 1])
def test_limit_rejected(setup: tuple[Path, Memory], limit: int) -> None:
    scratch, _ = setup
    with pytest.raises(ReplayError, match="bounded"):
        entry.run(scratch / "s", scratch / "o", evidence=scratch / "e", cache_limit=limit)


def test_external_scratch_and_overlap_rejected(
    setup: tuple[Path, Memory], monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch, _ = setup
    with pytest.raises(ReplayError, match="independent"):
        entry.run(scratch / "s", scratch / "o", evidence=scratch / "o/e", cache_limit=0)
    monkeypatch.setenv("TMP", str(scratch.parent))
    with pytest.raises(ReplayError, match="scratch"):
        entry.run(scratch / "s", scratch / "o", evidence=scratch / "e", cache_limit=0)


def test_typed_allocator_errors() -> None:
    with pytest.raises(ReplayError, match="counter"):
        entry._counter(True)
    with pytest.raises(ReplayError, match="Missing MLX"):
        entry._invoke(ModuleType("empty"), "missing")


def test_provider_cache_is_rejected(
    setup: tuple[Path, Memory], monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch, _ = setup
    monkeypatch.setitem(sys.modules, "mlx_whisper", ModuleType("mlx_whisper"))
    with pytest.raises(ReplayError, match="fresh provider"):
        entry.run(scratch / "s", scratch / "o", evidence=scratch / "e", cache_limit=0)


def test_policy_mutation_rejected(
    setup: tuple[Path, Memory], monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch, module = setup

    def transcribe(source: Path, output: Path, *, device: ASRDevice) -> Path:
        module.limit = 1000
        return output / "complete.json"

    monkeypatch.setattr(entry, "transcribe_windows", transcribe)
    with pytest.raises(ReplayError, match="changed during recognition"):
        entry.run(scratch / "s", scratch / "o", evidence=scratch / "e", cache_limit=0)
    assert module.limit == 123
    assert not (scratch / "e/complete.json").exists()
