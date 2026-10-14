"""Explicit candidate-only visibility exclusions; native timing guards stay intact.

Original projected recognition remains pinned and every hidden lexical unit is
recorded. This module is never selected by the ordinary caption workflow.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from replay.artifact_json import _path, _pin
from replay.caption_io import _array, _number, _string
from replay.caption_linguistics import atomic_tokens
from replay.caption_measured_job import _json, _object, _token
from replay.caption_timing import CaptionPolicy, SpokenToken
from replay.foreground_evidence import _bound_bytes
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 64 * 1024 * 1024
MAX_TOKENS = 100_000
MAX_TARGETS = 16
MAX_HIDDEN_TOKENS = 512
MAX_REGION_SECONDS = 60
KIND = "explicit-caption-visibility-candidate"
LIMITATIONS = (
    "Uncertain recognized lexical units are hidden only in this explicit candidate.",
    "Original recognition is preserved; complete captions and speech accuracy are unverified.",
)


@dataclass(frozen=True, slots=True)
class VisibilityRegion:
    first: int
    stop: int
    start: float
    end: float


def _read(path: Path) -> dict[str, object]:
    return record(_json(_bound_bytes(path, file_digest(path), MAX_BYTES)))


def _output(path: Path) -> Path:
    canonical = _path(str(path.absolute()))
    if (
        not canonical.is_relative_to(ROOT / "edits" / "work")
        or canonical.exists()
        or canonical.is_symlink()
        or not canonical.parent.is_dir()
    ):
        raise ValueError("Candidate output must be absent under existing edits/work")
    return canonical


def lexical_regions(
    tokens: tuple[SpokenToken, ...], targets: tuple[str, ...]
) -> tuple[VisibilityRegion, ...]:
    """Select intact lexical units using existing recognition boundaries only."""
    if not tokens or not 1 <= len(targets) <= MAX_TARGETS or len(set(targets)) != len(targets):
        raise ValueError("Expected bounded unique explicit exclusion targets")
    positions: set[int] = set()
    for target in targets:
        matches = [index for index, token in enumerate(tokens) if token.utterance == target]
        if not target or len(matches) != 1:
            raise ValueError("Expected one exact projected recognition target")
        positions.add(matches[0])
    units = atomic_tokens(list(tokens), maximum_gap=CaptionPolicy().maximum_gap)
    regions: list[VisibilityRegion] = []
    first = 0
    for unit in units:
        stop, text = first, ""
        while stop < len(tokens) and len(text) < len(unit.text):
            text += tokens[stop].text
            stop += 1
        if text != unit.text or stop == first:
            raise ValueError("Lexical planner changed original recognized text")
        if any(first <= position < stop for position in positions):
            start, end = tokens[first].start, max(token.end for token in tokens[first:stop])
            if stop - first > MAX_HIDDEN_TOKENS or end - start > MAX_REGION_SECONDS:
                raise ValueError("Exclusion exceeds bounded wholeword support")
            regions.append(VisibilityRegion(first, stop, start, end))
        first = stop
    if first != len(tokens):
        raise ValueError("Lexical planner dropped original recognition")
    if any(right.start < left.end for left, right in zip(regions, regions[1:], strict=False)):
        raise ValueError("Exclusion regions overlap; preserve unresolved recognition")
    return tuple(regions)


def visible_tokens(
    tokens: tuple[SpokenToken, ...], regions: tuple[VisibilityRegion, ...]
) -> list[dict[str, object]]:
    """Retain text and recognized times, separating context across hidden units."""
    result: list[dict[str, object]] = []
    region_index = 0
    for index, token in enumerate(tokens):
        while region_index < len(regions) and index >= regions[region_index].stop:
            region_index += 1
        if region_index < len(regions) and regions[region_index].first <= index:
            continue
        result.append({**asdict(token), "utterance": f"region:{region_index}:{token.utterance}"})
    if not result:
        raise ValueError("Candidate cannot hide all recognized text")
    return result


def _overlay(config: dict[str, object]) -> dict[str, object]:
    raw = _object(config, "version operation projectedTokens targets")
    if type(raw["version"]) is not int or raw["version"] != 1 or raw["operation"] != "overlay":
        raise ValueError("Expected explicit version1 overlay operation")
    projected_path = _pin(raw["projectedTokens"])
    projected = record(
        _json(
            _bound_bytes(
                projected_path, _string(record(raw["projectedTokens"])["sha256"]), MAX_BYTES
            )
        )
    )
    if (
        projected.get("version") != 1
        or projected.get("clock") != "output"
        or projected.get("boundaryConflicts") != []
        or "candidateVisibility" in projected
    ):
        raise ValueError("Expected original conflict-free projected output-clock tokens")
    dependencies = [_pin(projected.get(name)) for name in ("labels", "cuts")]
    snapshot = InputHashSnapshot.capture(
        [(path, file_digest(path)) for path in (projected_path, *dependencies)]
    )
    tokens = tuple(_token(row) for row in _array(projected.get("tokens"), MAX_TOKENS))
    target_rows = tuple(
        _object(row, "utterance reason") for row in _array(raw["targets"], MAX_TARGETS)
    )
    targets = tuple(_string(row["utterance"]) for row in target_rows)
    if any(not _string(row["reason"]).strip() for row in target_rows):
        raise ValueError("Explicit exclusions require a reason")
    regions = lexical_regions(tokens, targets)
    snapshot.finish()
    return {
        **projected,
        "tokens": visible_tokens(tokens, regions),
        "candidateOnly": True,
        "candidateVisibility": {
            "kind": KIND,
            "original": raw["projectedTokens"],
            "dependencies": {str(path): file_digest(path) for path in dependencies},
            "targets": list(target_rows),
            "regions": [
                {
                    **asdict(region),
                    "tokens": [asdict(token) for token in tokens[region.first : region.stop]],
                }
                for region in regions
            ],
            "limitations": list(LIMITATIONS),
            "humanReviewed": False,
            "speechAccuracyVerified": False,
            "completeCaptionsVerified": False,
            "deliveryComplete": False,
        },
    }


def clip_holds(
    captions: list[object], regions: tuple[tuple[float, float], ...]
) -> list[dict[str, object]]:
    """Clear prior holds at observed exclusion onset; never invent a new onset."""
    result: list[dict[str, object]] = []
    for value in captions:
        cue = record(value)
        text, start, end = (
            _string(cue.get("text")),
            _number(cue.get("start")),
            _number(cue.get("end")),
        )
        if not text.strip() or not 0 <= start < end:
            raise ValueError("Invalid measured caption interval")
        for left, right in regions:
            if left <= start < right:
                raise ValueError("Visible cue starts inside explicitly hidden recognition")
            if start < left < end:
                end = left
        result.append({**cue, "end": end})
    return result


def _clip(config: dict[str, object]) -> dict[str, object]:
    raw = _object(config, "version operation overlayConfig overlay reflow")
    if type(raw["version"]) is not int or raw["version"] != 1 or raw["operation"] != "clip":
        raise ValueError("Expected explicit version1 clip operation")
    overlay_config, overlay_path, reflow_path = (
        _pin(raw[name]) for name in ("overlayConfig", "overlay", "reflow")
    )
    snapshot = InputHashSnapshot.capture(
        [(path, file_digest(path)) for path in (overlay_config, overlay_path, reflow_path)]
    )
    overlay = _read(overlay_path)
    if overlay != plan(overlay_config):
        raise ValueError("Candidate overlay differs from its original recognition proof")
    visibility = record(overlay.get("candidateVisibility"))
    reflow = _read(reflow_path)
    inputs = record(reflow.get("inputs"))
    if (
        reflow.get("version") != 1
        or reflow.get("candidateOnly") is not True
        or inputs.get(str(overlay_path)) != file_digest(overlay_path)
    ):
        raise ValueError("Measured reflow must pin this exact visible candidate input")
    region_rows = _array(visibility.get("regions"), MAX_TARGETS)
    regions = tuple(
        (_number(record(row).get("start")), _number(record(row).get("end"))) for row in region_rows
    )
    captions = clip_holds(_array(reflow.get("captions"), MAX_TOKENS), regions)
    snapshot.finish()
    return {
        **reflow,
        "captions": captions,
        "inputs": {
            **inputs,
            str(overlay_config): file_digest(overlay_config),
            str(reflow_path): file_digest(reflow_path),
        },
        "candidateVisibility": visibility,
        "limitations": list(LIMITATIONS),
        "completeCaptionsVerified": False,
        "deliveryComplete": False,
    }


def plan(config: Path) -> dict[str, object]:
    """Recompute the explicit candidate without modifying original artifacts."""
    config = _path(str(config.absolute()))
    snapshot = InputHashSnapshot.capture(
        [(config, file_digest(config)), (Path(__file__), file_digest(Path(__file__)))]
    )
    raw = _read(config)
    operations = {"overlay": _overlay, "clip": _clip}
    operation = raw.get("operation")
    if not isinstance(operation, str) or operation not in operations:
        raise ValueError("Expected explicit overlay or clip operation")
    result = operations[operation](raw)
    snapshot.finish()
    return {
        **result,
        "visibilityProducer": {"path": str(Path(__file__)), "sha256": file_digest(Path(__file__))},
        "visibilityConfig": {"path": str(config), "sha256": file_digest(config)},
    }


def run(config: Path, output: Path, *, verify: bool = False) -> int:
    """Write or independently compare a candidate; neither approves delivery."""
    expected = plan(config)
    if verify:
        digest = file_digest(output)
        if _read(output) != json.loads(json.dumps(expected, allow_nan=False)):
            raise ValueError("Visibility candidate differs from its pinned plan")
        if file_digest(output) != digest:
            raise ValueError("Visibility candidate changed during verification")
    else:
        with _output(output).open("x", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(expected, stream, ensure_ascii=False, allow_nan=False, indent=2)
    values = expected.get("tokens", expected.get("captions"))
    return len(_array(values, MAX_TOKENS))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    count = run(arguments.config, arguments.output, verify=arguments.verify)
    print(f"Explicit candidate visibility verified for {count} records; delivery unapproved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
