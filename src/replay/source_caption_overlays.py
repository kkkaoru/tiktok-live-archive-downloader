"""Allow only previously audited Motion caption alpha above cleaned sources."""

from fractions import Fraction
from math import isfinite
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record
from replay.source_caption_binding import _json, _pin


def _approved(config: dict[str, object], clean: Path) -> dict[Path, Fraction]:
    references = config.get("captionOverlays", [])
    if not isinstance(references, list):
        raise ReplayError("Caption overlay evidence must be a list")
    result: dict[Path, Fraction] = {}
    for raw in references:
        reference = record(raw)
        audit = _json(_pin(reference.get("audit")))
        if (
            reference.get("role") != "motion-caption-alpha"
            or audit.get("kind") != "delivery-caption-audit"
            or audit.get("alphaOnlyAccepted") is not True
        ):
            raise ReplayError("Overlay is not an accepted Motion caption alpha audit")
        measured = record(audit.get("measurements"))
        rate, parts = measured.get("rate"), measured.get("parts")
        if type(rate) is not int or rate <= 0 or not isinstance(parts, list) or not parts:
            raise ReplayError("Caption alpha audit lacks a measured clock")
        for raw_part in parts:
            part = record(raw_part)
            alpha = _pin(part.get("alpha"))
            source = _pin(part.get("source"))
            plan = _json(_pin(part.get("plan")))
            _pin(part.get("producerState"))
            frames = part.get("frames")
            measurement = record(part.get("measurement"))
            clock = record(measurement.get("clock"))
            checks = record(part.get("checks"))
            if (
                alpha == clean
                or source.suffix != ".moti"
                or type(frames) is not int
                or frames <= 0
                or plan.get("frames") != frames
                or plan.get("rate") != rate
                or part.get("historicalSubmissionBound") is not True
                or part.get("motionSourcePlanStyleVerified") is not True
                or checks.get("boundsAndGapsVerified") is not True
                or checks.get("gapsVerified") is not True
                or measurement.get("fullDecodeVerified") is not True
                or measurement.get("decodedFrames") != frames
                or clock
                != {
                    "frames": frames,
                    "start": "0",
                    "end": str(Fraction(frames, rate)),
                    "exactCFR": True,
                }
            ):
                raise ReplayError("Caption alpha producer/hash/frame-clock evidence is incomplete")
            result[alpha] = Fraction(frames, rate)
    return result


def validate_overlays(rows: list[object], *, config: dict[str, object], clean: Path) -> None:
    approved = _approved(config, clean)
    for raw in rows:
        row = record(raw)
        path = row.get("sourcePath")
        selection = record(row.get("selection"))
        start, duration = selection.get("startSeconds"), selection.get("durationSeconds")
        if (
            not isinstance(path, str)
            or Path(path) not in approved
            or type(start) not in (int, float)
            or type(duration) not in (int, float)
            or not isfinite(float(str(start)))
            or not isfinite(float(str(duration)))
            or float(str(start)) < 0
            or float(str(duration)) <= 0
            or float(str(start)) + float(str(duration)) > float(approved[Path(path)])
            or selection.get("rate") != 1
        ):
            raise ReplayError("Production overlay is outside verified caption alpha media/clock")
