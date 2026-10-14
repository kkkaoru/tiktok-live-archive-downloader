"""Validate phrase-ASS evidence separately from the Motion-alpha backend."""

import argparse
import json
import math
import re
from pathlib import Path

from replay.caption_motion_bundle import read_output_tokens
from replay.caption_timing import CaptionPolicy, phrase_units
from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import SpeakerCue, render_srt
from replay.stages import file_digest

MAX_BYTES = 16 * 1024 * 1024


def read_document(path: Path) -> dict[str, object]:
    """Read a bounded, nonlinked evidence document."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ReplayError("ASS evidence must be a bounded regular document")
    return record(json.loads(path.read_text(encoding="utf-8")))


def bound_path(value: object) -> Path:
    """Resolve a fingerprinted artifact without silently following a new alias."""
    ref = record(value)
    raw, digest = ref.get("path"), ref.get("sha256")
    if not isinstance(raw, str) or not isinstance(digest, str):
        raise ReplayError("ASS evidence reference lacks path/hash")
    path = Path(raw)
    if not path.is_absolute() or path.resolve() != path or file_digest(path) != digest:
        raise ReplayError("ASS evidence artifact identity differs")
    return path


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ReplayError("ASS measurement must be finite numeric data")
    return float(value)


def rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 100_000:
        raise ReplayError("ASS measurement rows are missing or unbounded")
    return [record(item) for item in value]


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def transcription(plan: Path, cues: Path, cuts: Path) -> tuple[str | None, dict[str, object]]:
    """Compare complete sanitized recognized text, rather than word/cue counts."""
    tokens, _, _ = read_output_tokens(cues, cuts)
    units = phrase_units(
        tokens,
        policy=CaptionPolicy(
            linguistic_boundaries=True,
            complete_phrases=True,
            maximum_lines=3,
            maximum_characters=80,
            measured_layout=lambda text: text,
        ),
    )
    expected = compact("".join(unit.text for unit in units))
    pages = rows(read_document(plan).get("captions"))
    text = ""
    for page in pages:
        item = page.get("text")
        if not isinstance(item, str):
            raise ReplayError("ASS phrase text is missing")
        text += compact(item)
    return (None if text == expected else "phrase text differs from sanitized recognition"), {
        "recognizedCharacters": len(expected),
        "displayedCharacters": len(text),
        "phraseCues": len(pages),
        "semanticAlignmentVerified": False,
    }


def assess(
    evidence: Path,
    movie: Path,
    transcript: Path,
) -> tuple[str | None, dict[str, object], dict[str, object]]:
    """Require separate actual color, stability, geometry and timing measurements.

    Missing measurements fail explicitly. Agent assertions and successful graph
    execution never substitute for the measured rows or bound rendered media.
    """
    proof = read_document(evidence)
    if proof.get("backend") != "libass-pos-phrase":
        raise ReplayError("Wrong ASS completion backend")
    refs = record(proof.get("artifacts"))
    bound = {name: bound_path(value) for name, value in refs.items()}
    if bound.get("movie") != movie.resolve() or bound.get("transcript") != transcript.resolve():
        raise ReplayError("ASS evidence is for a different delivery")
    required = {"movie", "transcript", "plan", "cues", "cuts", "ass", "energy"}
    if not required.issubset(bound):
        raise ReplayError("ASS completion artifact set is incomplete")
    movie_hash = record(refs["movie"]).get("sha256")
    plan = read_document(bound["plan"])
    pages = rows(plan.get("captions"))
    failure, facts = transcription(bound["plan"], bound["cues"], bound["cuts"])
    facts["transcriptionFailure"] = failure
    violations: list[str] = []
    if failure:
        violations.append(failure)
    if plan.get("animation") != "none":
        violations.append("phrase animation contract differs")
    expected_srt = render_srt(
        tuple(
            SpeakerCue(
                number(page.get("start")),
                number(page.get("end")),
                "unassigned",
                str(page.get("text")),
            )
            for page in pages
        )
    )
    if transcript.read_text(encoding="utf-8") != expected_srt:
        violations.append("delivered SRT differs from the rendered phrase clock/text")
    energy = read_document(bound["energy"])
    if energy.get("planSHA256") != file_digest(bound["plan"]):
        violations.append("energy measurements are for a different phrase plan")
    if (
        energy.get("allCuesMeasured") is not True
        or number(energy.get("medianAbsoluteLagSeconds")) > 0.08
        or number(energy.get("p90AbsoluteLagSeconds")) > 0.25
    ):
        violations.append("all-cue vocal onset lag exceeds its measured budget")
    for page in pages[:-1]:
        start = math.ceil(round(number(page.get("start")), 2) * 30 - 1e-8)
        end = math.ceil(round(number(page.get("end")), 2) * 30 - 1e-8)
        if end - start < 6:
            violations.append("phrase lasts fewer than six displayed frames")
            break
    measurements = proof.get("measurements")
    if not isinstance(measurements, dict):
        measurements = {}
    checks = {
        "color": ("deltaE76", 6.0),
        "stability": ("existingGlyphMovementPx", 0.0),
        "appearance": ("relativeReferenceError", 0.1),
    }
    for name, (field, ceiling) in checks.items():
        raw = measurements.get(name)
        if raw is None:
            violations.append(f"missing measured {name} evidence")
            continue
        result = read_document(bound_path(raw))
        if result.get("movieSHA256") != movie_hash:
            violations.append(f"{name} measurement is for different media")
        measured_rows = rows(result.get("samples"))
        if any(not 0 <= number(row.get(field)) <= ceiling for row in measured_rows):
            violations.append(f"measured {name} exceeds its budget")
        if name == "color" and (
            {row.get("endpoint") for row in measured_rows} != {"top", "bottom"}
            or {row.get("lineCount") for row in measured_rows} != {1, 2, 3}
        ):
            violations.append("color proof lacks endpoints across one/two/three lines")
    raw_geometry = measurements.get("geometry")
    if raw_geometry is None:
        violations.append("missing measured safety evidence")
    else:
        geometry = read_document(bound_path(raw_geometry))
        if geometry.get("movieSHA256") != movie_hash:
            violations.append("geometry measurement is for different media")
        for sample in rows(geometry.get("samples")):
            if (
                number(sample.get("left")) < number(geometry.get("safeLeft"))
                or number(sample.get("right")) > number(geometry.get("safeRight"))
                or number(sample.get("bottom")) > number(geometry.get("safeBottom"))
            ):
                violations.append("rendered caption escapes the measured safe region")
                break
    facts["violations"] = violations
    facts["backend"] = "libass-pos-phrase"
    facts["evidence"] = str(evidence)
    return "; ".join(violations) if violations else None, facts, {"violations": violations}


def main(argv: list[str] | None = None) -> int:
    """Pin actual artifacts; absent quality measurements stay explicitly absent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    for name in ("movie", "transcript", "plan", "cues", "cuts", "ass", "energy"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("color", "stability", "appearance", "geometry"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to replace ASS completion evidence")
    artifacts: dict[str, object] = {}
    measurements: dict[str, object] = {}
    for name, value in vars(args).items():
        if name == "output" or value is None:
            continue
        if not isinstance(value, Path):
            raise ReplayError("Evidence CLI expects paths")
        path = value.resolve(strict=True)
        ref = {"path": str(path), "sha256": file_digest(path)}
        if name in ("color", "stability", "appearance", "geometry"):
            measurements[name] = ref
        else:
            artifacts[name] = ref
    args.output.write_text(
        json.dumps(
            {
                "version": 1,
                "backend": "libass-pos-phrase",
                "artifacts": artifacts,
                "measurements": measurements,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
