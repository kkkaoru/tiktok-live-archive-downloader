"""Synthetic allocator contracts; no model or real MLX import."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType

import pytest

from replay import mlx_cache_probe as probe
from replay.core import ReplayError


class Allocator(ModuleType):
    def __init__(self) -> None:
        super().__init__("synthetic_mlx")
        self.cpu = "cpu"
        self.float32 = "float32"
        self.limit = 100
        self.cache = 20
        self.fail_clear = False

    def stream(self, device: object) -> nullcontext[None]:
        assert device == self.cpu
        return nullcontext()

    def get_active_memory(self) -> int:
        return 12

    def get_cache_memory(self) -> int:
        return self.cache

    def get_peak_memory(self) -> int:
        return 32

    def set_cache_limit(self, limit: int) -> int:
        old = self.limit
        self.limit = limit
        return old

    def ones(self, shape: object, *, dtype: object, stream: object) -> object:
        assert dtype == self.float32 and stream == self.cpu
        return shape

    def add(self, left: object, right: object, *, stream: object) -> object:
        assert left == right and stream == self.cpu
        self.cache = 20
        return left

    def eval(self, result: object) -> None:
        assert result is not None

    def synchronize(self, device: object) -> None:
        assert device == self.cpu

    def clear_cache(self) -> None:
        if not self.fail_clear:
            self.cache = 0


def test_observe_restores_limit_and_records_real_counters() -> None:
    module = Allocator()
    result = probe.observe(module, elements=100, iterations=2, cache_limit=40)
    assert module.limit == 100
    assert result["modelLoaded"] is False
    assert result["ASRImpactVerified"] is False
    assert result["iterations"] == [
        {
            "elements": count,
            "allocated": {"active": 12, "cache": 20, "peak": 32},
            "released": {"active": 12, "cache": 20, "peak": 32},
            "cleared": {"active": 12, "cache": 0, "peak": 32},
        }
        for count in (100, 75)
    ]


@pytest.mark.parametrize(
    "elements,iterations,limit", [(0, 1, 0), (True, 1, 0), (1, 17, 0), (1, 1, -1), (4194305, 1, 0)]
)
def test_resource_bounds(elements: int, iterations: int, limit: int) -> None:
    with pytest.raises(ReplayError, match="bounded"):
        probe.observe(Allocator(), elements=elements, iterations=iterations, cache_limit=limit)


@pytest.mark.parametrize("value", [True, -1, 1.2, 2**63])
def test_counter_validation(value: object) -> None:
    with pytest.raises(ReplayError, match="counter"):
        probe._integer(value)


def test_clear_failure_restores_limit() -> None:
    module = Allocator()
    module.fail_clear = True
    with pytest.raises(ReplayError, match="clear_cache"):
        probe.observe(module, elements=1, iterations=1, cache_limit=0)
    assert module.limit == 100


def test_missing_apis() -> None:
    with pytest.raises(ReplayError, match="Missing MLX"):
        probe._call(ModuleType("empty"), "eval")
    with pytest.raises(ReplayError, match="explicit CPU"):
        probe.observe(ModuleType("empty"), elements=1, iterations=1, cache_limit=0)


def test_bad_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    module = Allocator()
    monkeypatch.setattr(module, "stream", lambda _: object())
    with pytest.raises(ReplayError, match="context manager"):
        probe.observe(module, elements=1, iterations=1, cache_limit=0)


def test_local_run_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scratch = tmp_path / "edits/work"
    scratch.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, str(scratch))
    monkeypatch.setattr("replay.mlx_cache_probe.importlib.import_module", lambda _: Allocator())
    audits: list[str] = []
    monkeypatch.setattr(probe, "audit_descriptors", lambda path, phase: audits.append(phase))
    monkeypatch.setattr(
        sys, "argv", ["probe", str(scratch / "result"), "--elements", "1", "--iterations", "1"]
    )
    probe.main()
    receipt = scratch / "result/receipt.json"
    assert receipt.is_file() and receipt.stat().st_mode & 0o777 == 0o600
    assert str(receipt) in capsys.readouterr().out
    assert len(audits) == 2
    with pytest.raises(ReplayError, match="stay local"):
        probe.run(tmp_path / "escaped", elements=1, iterations=1, cache_limit=0)
    monkeypatch.setenv("TMP", str(tmp_path))
    with pytest.raises(ReplayError, match="scratch"):
        probe.run(scratch / "bad", elements=1, iterations=1, cache_limit=0)
