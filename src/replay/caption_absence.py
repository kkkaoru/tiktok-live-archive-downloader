"""Measured absence of burned-in source captions, for a source whose cleanup profile finds none.

The cleanup bootstrap refuses a blanket mask when its profile OCR sees no variable Japanese
subtitle band. That sparse profile alone is not proof. Here the same profile OCR and the same
verifier (``replay_cleanup_plan.verify_profile``) run on a copy of the bootstrap request that
samples at least ``MINIMUM_FRAMES`` evenly spaced frames; only that verifier's own no-band refusal
yields a summary. The summary binds the request and the OCR outputs by hash. ``verify`` checks
those hashes and reads the pinned observations again through the current band learner (not the
producer code pins, which later unrelated edits change), so the gate never trusts the verdict.
"""

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path

from replay import replay_cleanup_plan as plan
from replay import subtitle_cleanup_workflow as cleanup
from replay.core import ReplayError
from replay.edit_completion import NO_CAPTION_BAND
from replay.recordings import record
from replay.subtitle_profile import learn_subtitle_bands

MINIMUM_FRAMES = 120
SURVEY_SAMPLES = 240
NO_BANDS = "No variable Japanese subtitle bands"
# The reason the delivery gate accepts for an uncaptioned source.
REASON = NO_CAPTION_BAND


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def no_band(request: Path, output: Path) -> bool:
    """True only when the verified profile at ``output`` is refused for want of a caption band."""
    try:
        plan.verify_profile(request, output)
    except (OSError, ValueError) as error:
        return str(error).startswith(NO_BANDS)
    return False


def band_free(request: Path, output: Path) -> bool:
    """The surveyed observations, read again on their own clock, hold no caption band."""
    job = plan.load(request)
    profile = output / "profile"
    frames, _ = cleanup.collector_frames(
        profile / "inputs.json", profile / "complete.json", job.data
    )
    if tuple(frame.requested_time for frame in frames) != job.times:
        return False
    try:
        learn_subtitle_bands(frames)
    except ValueError as error:
        return str(error).startswith(NO_BANDS)
    return False


def survey_request(request: Path, destination: Path, samples: int = SURVEY_SAMPLES) -> Path:
    """Copy the bootstrap request with ``samples`` profile frames (never fewer than it had)."""
    data = record(json.loads(request.read_text(encoding="utf-8")))
    data["profileSamples"] = max(samples, int(str(data.get("profileSamples", 0))))
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return destination


def summary(
    request: Path, output: Path, *, verified: Callable[[Path, Path], bool] = no_band
) -> dict[str, object]:
    """The absence summary of a surveyed profile; refused unless it is thick and band-free."""
    job = plan.load(request)
    frames = len(job.times)
    if frames < MINIMUM_FRAMES:
        raise ReplayError(f"The caption absence survey has {frames} frames, fewer than 120")
    if not verified(request, output):
        raise ReplayError("The survey did not verify a source without caption bands")
    profile = output / "profile"
    return {
        "version": 1,
        "reason": REASON,
        "sourceSHA256": job.source.sha256,
        "totalFrames": frames,
        "request": pin(request),
        "survey": str(output),
        "profileInputs": pin(profile / "inputs.json"),
        "profileComplete": pin(profile / "complete.json"),
    }


def verify(pinned: Mapping[str, object]) -> str | None:
    """Why a pinned absence summary no longer holds, or ``None`` when the full check passes."""
    path = Path(str(pinned.get("path")))
    if not path.is_file() or pin(path) != pinned:
        return "the caption absence summary is missing or changed"
    recorded = record(json.loads(path.read_text(encoding="utf-8")))
    request = Path(str(record(recorded.get("request")).get("path")))
    try:
        measured = summary(request, Path(str(recorded.get("survey"))), verified=band_free)
    except (OSError, ValueError, ReplayError) as error:
        return f"caption absence not verified ({error})"
    if measured != recorded:
        return "the caption absence summary differs from its survey"
    return None
