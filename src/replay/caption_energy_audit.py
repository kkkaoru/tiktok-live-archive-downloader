"""Operational caption-to-vocal-energy onset measurements, never invented word clocks."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.caption_motion_bundle import read_retained_cuts
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest

WINDOW_SECONDS = 0.01
SEARCH_SECONDS = 0.75
RELATIVE_THRESHOLD = 0.2
MINIMUM_ACTIVE_WINDOWS = 3


def rms_windows(path: Path) -> NDArray[np.float64]:
    """Read bounded PCM16 vocal audio without retaining full decoded samples."""
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() not in (1, 2):
            raise ReplayError("Energy measurement requires mono/stereo PCM16")
        step = round(audio.getframerate() * WINDOW_SECONDS)
        if step <= 0 or audio.getnframes() / audio.getframerate() > 86400:
            raise ReplayError("Audio exceeds energy measurement budget")
        result: list[float] = []
        while payload := audio.readframes(step):
            samples = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768
            result.append(float(np.sqrt(np.mean(samples * samples))))
    return np.asarray(result, dtype=np.float64)


def onset_lag(
    energy: NDArray[np.float64], start: float, *, search_seconds: float = SEARCH_SECONDS
) -> float | None:
    """Nearest sustained threshold crossing in a bounded local vocal window.

    This is a stated energy-onset definition, not semantic forced alignment.
    Continuous voice without a local rising boundary remains unsupported.
    """
    if not math.isfinite(start) or start < 0 or not 0 < search_seconds <= 1.5:
        raise ReplayError("Invalid caption onset")
    first = max(0, round((start - search_seconds) / WINDOW_SECONDS))
    last = min(len(energy), round((start + search_seconds) / WINDOW_SECONDS))
    local = energy[first:last]
    if len(local) < MINIMUM_ACTIVE_WINDOWS + 1 or float(np.max(local)) < 1e-4:
        return None
    floor, peak = float(np.quantile(local, 0.1)), float(np.quantile(local, 0.95))
    if peak - floor < 1e-4:
        return None
    threshold = floor + (peak - floor) * RELATIVE_THRESHOLD
    active = local >= threshold
    candidates = [
        first + index
        for index in range(1, len(active) - MINIMUM_ACTIVE_WINDOWS + 1)
        if not active[index - 1] and bool(np.all(active[index : index + MINIMUM_ACTIVE_WINDOWS]))
    ]
    if not candidates:
        return None
    chosen = min(candidates, key=lambda index: abs(index * WINDOW_SECONDS - start))
    return start - chosen * WINDOW_SECONDS


def audit(plan: dict[str, object], energy: NDArray[np.float64]) -> dict[str, object]:
    rows = plan.get("captions")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100_000:
        raise ReplayError("Expected bounded phrase captions")
    results: list[dict[str, object]] = []
    absolute_lags: list[float] = []
    for raw in rows:
        cue = record(raw)
        start = cue.get("start")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise ReplayError("Caption onset must be numeric")
        lag = onset_lag(energy, float(start))
        expanded = lag is None
        if expanded:
            lag = onset_lag(energy, float(start), search_seconds=1.5)
        results.append(
            {
                "start": start,
                "lagSeconds": lag,
                "supported": lag is not None,
                "expandedSearch": expanded,
            }
        )
        if lag is not None:
            absolute_lags.append(abs(lag))
    absolute = np.asarray(absolute_lags)
    median = float(np.median(absolute)) if len(absolute) else None
    p90 = float(np.quantile(absolute, 0.9)) if len(absolute) else None
    supported = len(absolute)
    return {
        "version": 1,
        "method": "nearest-sustained-local-energy-rise",
        "windowSeconds": WINDOW_SECONDS,
        "searchSeconds": SEARCH_SECONDS,
        "relativeThreshold": RELATIVE_THRESHOLD,
        "thresholdPolicy": "local-p10-plus-20-percent-dynamic-range",
        "missingOnsetSearchLimitSeconds": 1.5,
        "semanticAlignmentVerified": False,
        "cueCount": len(rows),
        "supportedCues": supported,
        "unsupportedCues": len(rows) - supported,
        "medianAbsoluteLagSeconds": median,
        "p90AbsoluteLagSeconds": p90,
        "thresholdsPassed": median is not None
        and median <= 0.08
        and p90 is not None
        and p90 <= 0.25,
        "allCuesMeasured": supported == len(rows),
        "records": results,
    }


def refine(
    plan: dict[str, object],
    measurement: dict[str, object],
    cuts: tuple[tuple[int, int, int], ...],
) -> dict[str, object]:
    """Use observed energy onsets inside original cut/order/lifetime constraints.

    Recognition records remain in the source plan. This display-clock correction
    is not semantic forced alignment; failed/ambiguous anchors stay explicit.
    """
    rows, records = plan.get("captions"), measurement.get("records")
    if not isinstance(rows, list) or not isinstance(records, list) or len(rows) != len(records):
        raise ReplayError("Onset measurements do not bind every phrase")
    captions = [record(raw) for raw in rows]
    original: list[float] = []
    for cue in captions:
        start = cue.get("start")
        if isinstance(start, bool) or not isinstance(start, (float, int)):
            raise ReplayError("Invalid phrase onset metadata")
        original.append(float(start))
    result: list[dict[str, object]] = []
    unchanged: list[dict[str, object]] = []
    changed = 0
    for index, (cue, raw) in enumerate(zip(captions, records, strict=True)):
        measured = record(raw)
        start, end = original[index], cue.get("end")
        if measured.get("start") != start or not isinstance(end, (int, float)):
            raise ReplayError("Onset measurement clock differs from the phrase")
        lag = measured.get("lagSeconds")
        selected = start
        if isinstance(lag, (float, int)) and not isinstance(lag, bool):
            candidate = start - float(lag)
            owner = next((span for span in cuts if span[0] / 30 <= start < span[1] / 30), None)
            if owner is None:
                raise ReplayError("Phrase onset is outside retained cuts")
            lower = max(owner[0] / 30, (original[index - 1] + start) / 2 + 0.1 if index else 0)
            upper = min(
                owner[1] / 30,
                float(end) - 0.2,
                (start + original[index + 1]) / 2 - 0.1
                if index + 1 < len(original)
                else float(end),
            )
            if lower <= candidate <= upper:
                selected = candidate
                changed += abs(selected - start) > 1e-9
            else:
                unchanged.append(
                    {
                        "index": index,
                        "reason": "cut/order/minimum-duration boundary",
                        "candidate": candidate,
                    }
                )
        else:
            unchanged.append({"index": index, "reason": "no supported energy onset"})
        result.append(
            {
                **cue,
                "start": selected,
                "recognizedStart": start,
                "energyOnsetSelected": selected != start,
            }
        )
    for index, cue in enumerate(result[:-1]):
        end, next_start = cue["end"], result[index + 1]["start"]
        if not isinstance(end, (float, int)) or not isinstance(next_start, (float, int)):
            raise ReplayError("Invalid corrected phrase boundaries")
        cue["end"] = min(end, next_start)
    return {
        **plan,
        "captions": result,
        "onsetRefinement": {
            "changedCues": changed,
            "unchangedAnchors": unchanged,
            "recognitionClocksPreserved": True,
            "semanticAlignmentVerified": False,
            "evaluationUsesSameEnergyDetector": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("vocals", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuts", type=Path)
    parser.add_argument("--refine-plan", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite energy audit")
    plan = record(json.loads(args.plan.read_text(encoding="utf-8")))
    result = audit(plan, rms_windows(args.vocals))
    result["planSHA256"] = file_digest(args.plan)
    result["vocalSHA256"] = file_digest(args.vocals)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.refine_plan is not None:
        if args.cuts is None or args.refine_plan.exists() or args.refine_plan.is_symlink():
            raise ReplayError("Refinement needs retained cuts and a new output")
        corrected = refine(plan, result, read_retained_cuts(args.cuts))
        corrected["sourcePlanSHA256"] = result["planSHA256"]
        corrected["energyAuditSHA256"] = file_digest(args.output)
        args.refine_plan.write_text(
            json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
