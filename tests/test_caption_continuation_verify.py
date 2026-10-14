import json
import shutil
from pathlib import Path

import pytest
from test_asr_clock_projection import config as projection_config
from test_continuous_quiet_plan import config as quiet_config

from replay import asr_clock_projection, continuous_quiet_plan
from replay.caption_continuation_verify import main, verify
from replay.core import ReplayError

__all__ = ["projection_config", "quiet_config"]


def change(path: Path, key: str, value: object) -> None:
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))


def test_projection_readonly(projection_config: Path) -> None:
    output = projection_config.parent / "projected.json"
    asr_clock_projection.run(projection_config, output)
    before = output.read_bytes()
    assert main(["projection", str(projection_config), str(output)]) == 0
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "key,value", [("labels", []), ("sourceStart", "0"), ("unresolvedPoints", 4)]
)
def test_projection_domain_reject(projection_config: Path, key: str, value: object) -> None:
    output = projection_config.parent / "projected.json"
    asr_clock_projection.run(projection_config, output)
    change(output, key, value)
    with pytest.raises(ReplayError, match="original clock/text"):
        verify("projection", projection_config, output)


@pytest.mark.parametrize(
    "key", ["parentSHA256", "configSHA256", "projectionSHA256", "clock", "deliveryComplete"]
)
def test_identity_reject(projection_config: Path, key: str) -> None:
    output = projection_config.parent / "projected.json"
    asr_clock_projection.run(projection_config, output)
    change(output, key, "wrong")
    with pytest.raises(ReplayError, match="identity"):
        verify("projection", projection_config, output)


def test_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="Unknown"):
        verify("other", tmp_path, tmp_path)


def test_quiet_readonly_and_corruptions(quiet_config: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    root = quiet_config.parent / "quiet"
    continuous_quiet_plan.run(quiet_config, root, ffmpeg=ffmpeg)
    output = root / "cuts.json"
    original = output.read_bytes()
    verify("quiet", quiet_config, output)
    assert output.read_bytes() == original
    for key, value in [("cuts", []), ("sourceFrames", 0), ("sourceQuiet", []), ("fps", 60)]:
        output.write_bytes(original)
        change(output, key, value)
        with pytest.raises(ReplayError, match="source-clock support"):
            verify("quiet", quiet_config, output)
    output.write_bytes(original)
    change(output, "peakSHA256", "wrong")
    with pytest.raises(ReplayError, match="measurement identity"):
        verify("quiet", quiet_config, output)
