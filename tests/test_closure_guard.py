"""The closure guard sees an edit to a pinned file under a running step and stops the step."""

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from replay import closure_guard, failure_notify


def closure(tmp_path: Path, files: dict[Path, bytes], *, mtime: float | None = None) -> Path:
    for path, data in files.items():
        path.write_bytes(data)
    pins = {str(path): hashlib.sha256(data).hexdigest() for path, data in files.items()}
    path = tmp_path / "paging-v1" / "CODE-CLOSURE.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"inputs": pins}), "utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_only_closures_written_since_the_step_started_are_watched(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    closure(tmp_path, {module: b"x = 1\n"}, mtime=100.0)
    assert closure_guard.closures((tmp_path, tmp_path / "none"), since=200.0) == {}
    assert list(closure_guard.closures((tmp_path,), since=50.0)) == [module]


def test_a_changed_or_missing_pinned_file_is_reported(tmp_path: Path) -> None:
    kept, edited, gone = tmp_path / "a.py", tmp_path / "b.py", tmp_path / "c.py"
    closure(tmp_path, {kept: b"a", edited: b"b", gone: b"c"})
    edited.write_bytes(b"B")
    gone.unlink()
    pins = closure_guard.closures((tmp_path,), since=0.0)
    assert closure_guard.changed(pins) == [str(edited), str(gone)]


def test_watching_calls_back_once_on_the_first_edit(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    closure(tmp_path, {module: b"x = 1\n"})
    seen: list[list[str]] = []
    fired = threading.Event()

    def on_change(edited: list[str]) -> None:
        seen.append(edited)
        fired.set()

    with closure_guard.watching((tmp_path,), 0.0, on_change=on_change, interval=0.01):
        module.write_bytes(b"x = 2\n")
        assert fired.wait(5)
    assert seen == [[str(module)]]


def test_an_edit_is_noticed_and_the_step_keeps_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        failure_notify, "notify", lambda *args, **kw: sent.append((*args, kw["cls"]))
    )
    log = tmp_path / "paging.log"
    closure_guard.note_edit("9", "paging", log, ["/src/replay/caption_native_paging.py"])
    assert log.read_text("utf-8") == (
        "EDITED_MID_RUN /src/replay/caption_native_paging.py: "
        "a pinned file changed under the step\n"
    )
    assert sent == [("9", "paging", "edited-mid-run")]


def test_a_touch_or_identical_rewrite_is_not_an_edit(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    closure(tmp_path, {module: b"x = 1\n"})
    os.utime(module, (1, 1))
    module.write_bytes(b"x = 1\n")
    assert closure_guard.changed(closure_guard.closures((tmp_path,), since=0.0)) == []
