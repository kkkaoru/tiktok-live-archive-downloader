import fcntl
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay.core import ReplayError
from replay.jobs import (
    already_complete,
    download_once,
    record_path,
    register_completed,
    write_record,
)


def test_invalid_id(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="digits"):
        record_path(tmp_path, "../123")


def test_download_and_skip(tmp_path: Path) -> None:
    output = tmp_path / "video.mp4"
    action = Mock(side_effect=lambda: output.write_bytes(b"video"))
    assert download_once(root=tmp_path, replay_id="123", output=output, download=action)
    assert not download_once(
        root=tmp_path, replay_id="123", output=tmp_path / "other.mp4", download=action
    )
    assert action.call_count == 1
    assert json.loads((tmp_path / "123.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_busy_skips(tmp_path: Path) -> None:
    action = Mock()
    with (tmp_path / "123.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not download_once(
            root=tmp_path, replay_id="123", output=tmp_path / "v.mp4", download=action
        )
    action.assert_not_called()


def test_failure_then_retry(tmp_path: Path) -> None:
    output = tmp_path / "v.mp4"
    with pytest.raises(RuntimeError, match="network"):
        download_once(
            root=tmp_path,
            replay_id="123",
            output=output,
            download=Mock(side_effect=RuntimeError("network")),
        )
    assert json.loads((tmp_path / "123.json").read_text(encoding="utf-8"))["status"] == "failed"

    def action() -> None:
        output.write_bytes(b"ok")

    assert download_once(root=tmp_path, replay_id="123", output=output, download=action)


def test_missing_publication(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="nonempty"):
        download_once(
            root=tmp_path, replay_id="123", output=tmp_path / "v.mp4", download=lambda: None
        )


def test_existing_unregistered(tmp_path: Path) -> None:
    output = tmp_path / "v.mp4"
    output.write_bytes(b"ok")
    with pytest.raises(ReplayError, match="not registered"):
        download_once(root=tmp_path, replay_id="123", output=output, download=lambda: None)


@pytest.mark.parametrize("data", [[], {}, {"status": "bad", "output": "/tmp/x"}])
def test_bad_record(tmp_path: Path, data: object) -> None:
    path = tmp_path / "123.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="Invalid"):
        already_complete(path)


def test_completed_file_removed(tmp_path: Path) -> None:
    output = tmp_path / "v.mp4"
    output.write_bytes(b"ok")
    register_completed(root=tmp_path, replay_id="123", output=output)
    output.unlink()
    with pytest.raises(ReplayError, match="missing or changed"):
        already_complete(tmp_path / "123.json")


def test_interrupted_output_requires_verification(tmp_path: Path) -> None:
    output = tmp_path / "v.mp4"
    output.write_bytes(b"ok")
    write_record(tmp_path / "123.json", output, "running")
    with pytest.raises(ReplayError, match="Interrupted"):
        already_complete(tmp_path / "123.json")
    register_completed(root=tmp_path, replay_id="123", output=output)
    assert already_complete(tmp_path / "123.json")


def test_register_missing(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="missing or empty"):
        register_completed(root=tmp_path, replay_id="123", output=tmp_path / "v.mp4")
