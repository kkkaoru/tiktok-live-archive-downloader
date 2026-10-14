"""Only exact named executable pins cross the caption tool boundary."""

from pathlib import Path

import pytest
from test_delivery_completion_v2 import Fixture, fingerprint
from test_delivery_completion_v2 import delivery as delivery

from replay import delivery_completion_v2 as completion
from replay.core import ReplayError


def test_exact_external_tool_pin_not_parent_directory(delivery: Fixture, tmp_path: Path) -> None:
    tool = tmp_path / "ffmpeg"
    tool.write_bytes(b"synthetic binary")
    tool.chmod(0o700)
    store = completion.Evidence(delivery.root, delivery.source)
    completion._caption_tools(
        store,
        {
            "measurementTools": {"ffmpeg": fingerprint(tool)},
            "measurements": {"parts": [{}]},
        },
    )
    assert store.digest(tool) == fingerprint(tool)["sha256"]
    other = tmp_path / "unregistered"
    other.write_bytes(b"not a tool")
    with pytest.raises(ReplayError, match="authorized storage"):
        store.digest(other)
    tool.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="changed"):
        store.verify_unchanged()


@pytest.mark.parametrize("fault", ["name", "mode", "sha", "private", "role", "directory"])
def test_external_tool_registration_fails_closed(
    delivery: Fixture, tmp_path: Path, fault: str
) -> None:
    directory = tmp_path / "private" if fault == "private" else tmp_path
    directory.mkdir(exist_ok=True)
    tool = directory / ("wrong" if fault == "name" else "ffmpeg")
    tool.write_bytes(b"synthetic binary")
    tool.chmod(0o600 if fault == "mode" else 0o700)
    pin = fingerprint(tool)
    if fault == "sha":
        pin["sha256"] = "0" * 64
    elif fault == "directory":
        pin["path"] = str(tmp_path)
    role = "unknown" if fault == "role" else "ffmpeg"
    with pytest.raises(ReplayError):
        completion._caption_tools(
            completion.Evidence(delivery.root, delivery.source),
            {
                "measurementTools": {role: pin},
                "measurements": {"parts": [{}]},
            },
        )
