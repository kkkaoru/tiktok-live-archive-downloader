"""Cached MPS graphs are released only when torch is loaded and MPS is available."""

import atexit
import os
import signal
import sys
import threading
from types import SimpleNamespace

import pytest

from replay import mps_graph_cache as cache


def fake_torch(*, available: bool, calls: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: available)),
        mps=SimpleNamespace(empty_cache=lambda: calls.append("empty_cache")),
    )


def test_the_cache_is_released_when_torch_is_loaded_and_mps_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=True, calls=calls))
    cache.release_mps_graphs()
    assert calls == ["empty_cache"]


def test_nothing_is_released_without_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=False, calls=calls))
    cache.release_mps_graphs()
    assert calls == []


def test_torch_is_not_imported_by_a_process_that_never_loaded_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    cache.release_mps_graphs()
    assert "torch" not in sys.modules


def test_the_block_releases_the_cache_when_it_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=True, calls=calls))
    with cache.released_mps_graphs():
        calls.append("work")
    assert calls == ["work", "empty_cache"]


def test_the_block_releases_the_cache_when_it_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=True, calls=calls))
    with pytest.raises(RuntimeError, match="failed"), cache.released_mps_graphs():
        raise RuntimeError("failed")
    assert calls == ["empty_cache"]


def test_sigterm_inside_the_block_exits_after_releasing_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=True, calls=calls))
    with pytest.raises(SystemExit) as stopped, cache.released_mps_graphs():
        os.kill(os.getpid(), signal.SIGTERM)
    assert stopped.value.code == 143
    assert calls == ["empty_cache"]


def test_the_previous_sigterm_handler_is_restored_after_the_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=True, calls=[]))
    before = signal.getsignal(signal.SIGTERM)
    with cache.released_mps_graphs():
        assert signal.getsignal(signal.SIGTERM) is cache.exit_on_sigterm
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_worker_thread_leaves_signal_handling_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    seen: list[object] = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=True, calls=calls))
    before = signal.getsignal(signal.SIGTERM)

    def work() -> None:
        with cache.released_mps_graphs():
            seen.append(signal.getsignal(signal.SIGTERM))

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert seen == [before]
    assert calls == ["empty_cache"]


def test_the_exit_release_is_registered_once_and_turns_a_default_sigterm_into_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[object] = []
    monkeypatch.setattr(atexit, "register", registered.append)
    cache.release_mps_graphs_at_exit.cache_clear()
    before = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        cache.release_mps_graphs_at_exit()
        cache.release_mps_graphs_at_exit()
        installed = signal.getsignal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, before)
        cache.release_mps_graphs_at_exit.cache_clear()
    assert registered == [cache.release_mps_graphs]
    assert installed is cache.exit_on_sigterm


def test_the_exit_release_keeps_an_application_sigterm_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atexit, "register", lambda function: None)
    cache.release_mps_graphs_at_exit.cache_clear()

    def application(signum: int, frame: object) -> None:
        return None

    before = signal.signal(signal.SIGTERM, application)
    try:
        cache.release_mps_graphs_at_exit()
        kept = signal.getsignal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, before)
        cache.release_mps_graphs_at_exit.cache_clear()
    assert kept is application


def test_the_exit_release_from_a_worker_thread_leaves_signals_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[object] = []
    monkeypatch.setattr(atexit, "register", registered.append)
    cache.release_mps_graphs_at_exit.cache_clear()
    before = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        worker = threading.Thread(target=cache.release_mps_graphs_at_exit)
        worker.start()
        worker.join()
        after = signal.getsignal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, before)
        cache.release_mps_graphs_at_exit.cache_clear()
    assert registered == [cache.release_mps_graphs]
    assert after is signal.SIG_DFL
