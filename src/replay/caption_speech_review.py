"""Select low-energy phrase windows for semantic review without deleting text."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.caption_ass_completion import number, read_document, rows
from replay.caption_energy_audit import WINDOW_SECONDS, rms_windows
from replay.core import ReplayError
from replay.stages import file_digest

MINIMUM_FLOOR_DB = -70.0
FLOOR_BELOW_P90_DB = 25.0
REVIEW_LIMIT = 12


def select(plan: dict[str, object], energy: NDArray[np.float64]) -> dict[str, object]:
    """Rank cue-local speech support; voiced laughter still needs semantic review."""
    if len(energy) == 0 or not np.all(np.isfinite(energy)) or np.any(energy < 0):
        raise ReplayError("Invalid phrase review energy")
    db = 20 * np.log10(energy + 1e-12)
    floor = max(MINIMUM_FLOOR_DB, float(np.quantile(db, 0.9)) - FLOOR_BELOW_P90_DB)
    measured: list[dict[str, object]] = []
    for index, cue in enumerate(rows(plan.get("captions"))):
        start, end = number(cue.get("start")), number(cue.get("end"))
        text = cue.get("text")
        if not isinstance(text, str) or not 0 <= start < end <= len(energy) * WINDOW_SECONDS:
            raise ReplayError("Phrase review must use the matching output audio clock")
        first, last = math.floor(start / WINDOW_SECONDS), math.ceil(end / WINDOW_SECONDS)
        local = db[first:last]
        measured.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "text": text,
                "activeFraction": float(np.mean(local >= floor)),
                "peakDb": float(np.max(local)),
            }
        )
    chosen = sorted(measured, key=review_key)[:REVIEW_LIMIT]
    return {
        "version": 1,
        "method": "cue-local-rms-lowest-activity-review",
        "speechFloorDb": floor,
        "measuredCues": len(measured),
        "zeroActivityCues": sum(row["activeFraction"] == 0 for row in measured),
        "samples": chosen,
        "semanticAlignmentVerified": False,
        "limitation": "Energy cannot distinguish speech from laughter or prove words; "
        "no cues changed.",
    }


def review_key(row: dict[str, object]) -> tuple[float, float]:
    return number(row["activeFraction"]), number(row["start"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("vocals", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite phrase speech review")
    result = select(read_document(args.plan), rms_windows(args.vocals))
    result["planSHA256"], result["vocalSHA256"] = file_digest(args.plan), file_digest(args.vocals)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
