"""Preparation dependency boundary, without applications or native helpers."""

import sys
from pathlib import Path

from replay.motion_dependencies import preparation_inputs


def test_preparation_inputs_are_exact_unique_and_include_shared_code(tmp_path: Path) -> None:
    paths = preparation_inputs(
        config=tmp_path / "config.json",
        template=tmp_path / "template.moti",
        plan=tmp_path / "plan.json",
        calibration=tmp_path / "calibration.json",
    )
    assert len(paths) == len(set(paths)) == 14
    assert paths[:4] == (
        tmp_path / "config.json",
        tmp_path / "template.moti",
        tmp_path / "plan.json",
        tmp_path / "calibration.json",
    )
    assert paths[4] == Path(sys.executable).resolve(strict=True)
    assert {path.name for path in paths[5:]} == {
        "motion_dependencies.py",
        "motion_stages.py",
        "motion_finish.py",
        "motion_caption.py",
        "caption_style_defaults.py",
        "caption_font_identity.py",
        "caption_contrast.py",
        "core.py",
        "recordings.py",
    }
