from __future__ import annotations

import json
from pathlib import Path

import pytest

from replay import compressor_batch as batch
from replay import compressor_repin
from replay.core import ReplayError


def receipt(tmp_path: Path, name: str, controller: Path) -> tuple[Path, str]:
    executable = tmp_path / "compressor"
    executable.write_bytes(b"cli")
    paths = list(batch.runtime_paths(executable))
    paths[2] = controller
    pins = [batch._pin(path) for path in paths]
    saved = {"submissionVersion": 2, "command": [str(executable)], "runtimeInputs": pins}
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    (state / f"{name}.json").write_text(json.dumps(saved), encoding="utf-8")
    return state, str(pins[2]["sha256"])


def test_repin_replaces_only_the_edited_controller_and_keeps_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = tmp_path / "compressor_batch.py"
    controller.write_text("one", encoding="utf-8")
    original = batch.runtime_paths
    monkeypatch.setattr(
        batch, "runtime_paths", lambda tool: (*original(tool)[:2], controller, *original(tool)[3:])
    )
    state, old = receipt(tmp_path, "alpha-0000", controller)
    controller.write_text("two", encoding="utf-8")
    assert compressor_repin.repin(state, old, apply=False) == ["alpha-0000.json"]
    assert "runtimePinHistory" not in json.loads((state / "alpha-0000.json").read_text())
    assert compressor_repin.repin(state, old, apply=True) == ["alpha-0000.json"]
    saved = json.loads((state / "alpha-0000.json").read_text())
    assert saved["runtimePinHistory"][0]["replaced"]["sha256"] == old
    assert compressor_repin.repin(state, old, apply=True) == []
    assert batch.verify_runtime(saved, rehash=True) is True


def test_repin_refuses_other_digests_and_reports_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    controller = tmp_path / "compressor_batch.py"
    controller.write_text("one", encoding="utf-8")
    original = batch.runtime_paths
    monkeypatch.setattr(
        batch, "runtime_paths", lambda tool: (*original(tool)[:2], controller, *original(tool)[3:])
    )
    state, _ = receipt(tmp_path, "alpha-0000", controller)
    controller.write_text("two", encoding="utf-8")
    with pytest.raises(ReplayError, match="controller digest"):
        compressor_repin.repin(state, "0" * 64, apply=True)
    assert compressor_repin.main([str(state), "0" * 64]) == 1
    assert "COMPRESSOR_REPIN_ERROR" in capsys.readouterr().err
    (state / "alpha-0001.json").write_text('{"status": "Successful"}', encoding="utf-8")
    assert compressor_repin.main([str(state), "1" * 64]) == 1


def test_main_reports_dry_run_and_skips_unrelated_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    controller = tmp_path / "compressor_batch.py"
    controller.write_text("one", encoding="utf-8")
    original = batch.runtime_paths
    monkeypatch.setattr(
        batch, "runtime_paths", lambda tool: (*original(tool)[:2], controller, *original(tool)[3:])
    )
    state, old = receipt(tmp_path, "alpha-0000", controller)
    (state / "alpha-0001.json").write_text('{"status": "Successful"}', encoding="utf-8")
    assert compressor_repin.main([str(state), old]) == 0
    assert "would repin 0 receipts" in capsys.readouterr().out
    controller.write_text("two", encoding="utf-8")
    assert compressor_repin.main([str(state), old, "--apply"]) == 0
    assert "repinned 1 receipts" in capsys.readouterr().out


def test_unexpected_pin_changes_and_bad_commands_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = tmp_path / "compressor_batch.py"
    controller.write_text("one", encoding="utf-8")
    original = batch.runtime_paths
    monkeypatch.setattr(
        batch, "runtime_paths", lambda tool: (*original(tool)[:2], controller, *original(tool)[3:])
    )
    state, old = receipt(tmp_path, "alpha-0000", controller)
    (tmp_path / "compressor").write_bytes(b"different")
    with pytest.raises(ReplayError, match="unexpected changed"):
        compressor_repin.repin(state, old, apply=True)
    bad = {"submissionVersion": 2, "runtimeInputs": [], "command": []}
    (state / "alpha-0002.json").write_text(json.dumps(bad), encoding="utf-8")
    (state / "alpha-0000.json").unlink()
    with pytest.raises(ReplayError, match="Invalid command"):
        compressor_repin.repin(state, old, apply=True)
