"""Each production CLI must import independently of pytest collection order."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "replay.fcp_head_pipeline",
        "replay.fcp_part_assembly",
        "replay.fcp_transformed_source",
        "replay.delivery_render",
        "replay.delivery_pipeline",
        "replay.delivery_render_resume",
        "replay.native_render",
        "replay.part_render",
    ],
)
def test_help_imports_without_app_or_media_operations(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
