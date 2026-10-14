"""Explicit code/data inputs for the bounded Motion stages/finish export path.

These pins establish reproducible source preparation at submission, not a
hermetic Apple rendering environment or retrospective preparation timestamps.
"""

import sys
from pathlib import Path

from replay import (
    caption_contrast,
    caption_font_identity,
    caption_style_defaults,
    core,
    motion_caption,
    motion_finish,
    motion_stages,
    recordings,
)


def preparation_inputs(
    *, config: Path, template: Path, plan: Path, calibration: Path
) -> tuple[Path, ...]:
    """Use as producer dependencies and as the consumer's required minimum."""
    return tuple(
        dict.fromkeys(
            (
                config.absolute(),
                template.absolute(),
                plan.absolute(),
                calibration.absolute(),
                Path(sys.executable).resolve(strict=True),
                Path(__file__).absolute(),
                Path(motion_stages.__file__).absolute(),
                Path(motion_finish.__file__).absolute(),
                Path(motion_caption.__file__).absolute(),
                Path(caption_style_defaults.__file__).absolute(),
                Path(caption_font_identity.__file__).absolute(),
                Path(caption_contrast.__file__).absolute(),
                Path(core.__file__).absolute(),
                Path(recordings.__file__).absolute(),
            )
        )
    )
