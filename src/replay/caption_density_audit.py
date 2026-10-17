"""Read-only audit of token timing that native caption paging cannot display.

Native paging refuses focused captions for tokens that last longer than the focused limit and
for bursts of replacements that start faster than one per frame. This lists those places with
their output and source clocks, the original token text, and padded recognition windows of at
most ``MAX_WINDOW_SECONDS`` for re-measurement. It only reads pinned data: no timing is moved,
split or invented, and it approves nothing.
"""

import json
import math
from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from replay.caption_scale import FOCUSED_TIMING_SECONDS
from replay.delivery_captions import Cut, read_cuts
from replay.recordings import record

RATE = 30
BURST_FRAMES = 3  # onsets within this many frames form one burst
BURST_TOKENS = 4  # tokens needed inside the burst span to call it dense
ONSET_EPSILON = 1e-6
CHAIN_SECONDS = 2.5  # a density chain starts a few cues before the delayed replacement
COUNT_KEYS = ("coarseTokens", "denseBursts", "identicalOnsets", "displayDensity")
PADDING_SECONDS = 2.0
MAX_WINDOW_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: float
    end: float
    utterance: str


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str  # "coarse-token" or "dense-burst"
    start: float
    end: float
    text: str
    detail: str


@dataclass(frozen=True, slots=True)
class Window:
    start: float
    end: float
    kinds: tuple[str, ...]
    text: str


def read_tokens(path: Path) -> tuple[Token, ...]:
    rows = record(json.loads(path.read_text(encoding="utf-8"))).get("tokens")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Projected tokens are missing")
    tokens: list[Token] = []
    for raw in rows:
        row = record(raw)
        text, start, end, owner = (row.get(key) for key in ("text", "start", "end", "utterance"))
        if (
            not isinstance(text, str)
            or not isinstance(owner, str)
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise ValueError("Invalid projected token")
        tokens.append(Token(text, float(start), float(end), owner))
    return tuple(tokens)


def coarse_tokens(
    tokens: Sequence[Token], *, limit: float = FOCUSED_TIMING_SECONDS
) -> list[Finding]:
    """Tokens whose own duration exceeds the focused-timing limit."""
    return [
        Finding("coarse-token", t.start, t.end, t.text, f"duration {t.end - t.start:.3f}s")
        for t in tokens
        if t.end - t.start > limit
    ]


def dense_bursts(tokens: Sequence[Token]) -> list[Finding]:
    """Maximal runs of at least ``BURST_TOKENS`` onsets inside ``BURST_FRAMES`` frames."""
    starts = [t.start for t in tokens]
    findings: list[Finding] = []
    span = BURST_FRAMES / RATE
    index = 0
    while index < len(tokens):
        last = bisect_right(starts, starts[index] + span) - 1
        if last - index + 1 >= BURST_TOKENS:
            group = tokens[index : last + 1]
            findings.append(
                Finding(
                    "dense-burst",
                    group[0].start,
                    max(t.end for t in group),
                    "".join(t.text for t in group),
                    f"{len(group)} onsets within {starts[last] - starts[index]:.3f}s",
                )
            )
            index = last + 1
        else:
            index += 1
    return findings


def identical_onsets(tokens: Sequence[Token]) -> list[Finding]:
    """Neighbouring tokens that begin at the same instant (paging refuses unequal pairs)."""
    return [
        Finding(
            "identical-onset",
            first.start,
            max(first.end, second.end),
            first.text + "|" + second.text,
            f"durations {first.end - first.start:.3f}s and {second.end - second.start:.3f}s",
        )
        for first, second in zip(tokens, tokens[1:], strict=False)
        if abs(first.start - second.start) < ONSET_EPSILON
    ]


def probe_findings(path: Path, *, offset: float) -> list[Finding]:
    """Density violations logged by a paging probe (one JSON object per line).

    The probe logs segment-local cue times; ``offset`` is the segment's output start.
    """
    findings: list[Finding] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = record(json.loads(line))
        start, text, previous = row.get("cueStart"), row.get("text"), row.get("previous")
        delay = row.get("delay")
        if (
            row.get("kind") != "density"
            or isinstance(start, bool)
            or not isinstance(start, (int, float))
            or not isinstance(text, str)
            or isinstance(delay, bool)
            or not isinstance(delay, (int, float))
        ):
            raise ValueError("Invalid paging probe record")
        findings.append(
            Finding(
                "display-density",
                max(0.0, offset + float(start) - CHAIN_SECONDS),
                offset + float(start) + 1 / RATE,
                f"{previous or ''} / {text}",
                f"replacement delayed {float(delay):.4f}s (> one frame)",
            )
        )
    return findings


def frame_histogram(tokens: Sequence[Token]) -> dict[int, int]:
    """How many frames have 1, 2, 3, ... token onsets (onset frame = ceil(start*rate))."""
    counts = Counter(math.ceil(t.start * RATE) for t in tokens)
    return dict(sorted(Counter(counts.values()).items()))


def source_seconds(cuts: Sequence[Cut], seconds: float) -> float | None:
    """Source-presentation seconds of an output time, or None outside every retained cut."""
    frame = seconds * RATE
    for cut in cuts:
        if cut.output_start <= frame < cut.output_end:
            return (cut.source_start + frame - cut.output_start) / RATE
    return None


def windows(
    findings: Sequence[Finding], tokens: Sequence[Token], *, duration: float
) -> list[Window]:
    """Pad findings, merge overlaps and split anything longer than ``MAX_WINDOW_SECONDS``."""
    spans: list[tuple[float, float, set[str]]] = []
    for item in sorted(findings, key=lambda f: f.start):
        start, end = (
            max(0.0, item.start - PADDING_SECONDS),
            min(duration, item.end + PADDING_SECONDS),
        )
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end), spans[-1][2] | {item.kind})
        else:
            spans.append((start, end, {item.kind}))
    result: list[Window] = []
    for start, end, kinds in spans:
        count = max(1, math.ceil((end - start) / MAX_WINDOW_SECONDS))
        step = (end - start) / count
        for number in range(count):
            low, high = start + number * step, start + (number + 1) * step
            text = "".join(t.text for t in tokens if low <= t.start < high)
            result.append(Window(low, high, tuple(sorted(kinds)), text))
    return result


def _segment_rows(path: Path) -> list[tuple[int, float, float]]:
    rows = record(json.loads(path.read_text(encoding="utf-8"))).get("segments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Recognition segment index is missing")
    result: list[tuple[int, float, float]] = []
    for raw in rows:
        row = record(raw)
        number, offset, frames = row.get("segment"), row.get("offsetFrame"), row.get("frames")
        if type(number) is not int or type(offset) is not int or type(frames) is not int:
            raise ValueError("Invalid recognition segment row")
        result.append((number, offset / RATE, (offset + frames) / RATE))
    return result


def _count(findings: Sequence[Finding], kind: str, start: float, end: float) -> int:
    return sum(1 for f in findings if f.kind == kind and start <= f.start < end)


def _located(item: Finding | Window, cuts: Sequence[Cut]) -> dict[str, object]:
    return {
        "outputStart": round(item.start, 3),
        "outputEnd": round(item.end, 3),
        "sourceStart": source_seconds(cuts, item.start),
        "sourceEnd": source_seconds(cuts, max(item.start, item.end - 1 / RATE)),
    }


def report(
    *,
    tokens: Path,
    cuts: Path,
    segments: Path,
    probe: tuple[Path, int] | None = None,
) -> dict[str, object]:
    """Findings per recognition segment plus padded re-measurement windows."""
    spoken = read_tokens(tokens)
    plan = read_cuts(cuts)
    rows = _segment_rows(segments)
    duration = plan[-1].output_end / RATE
    findings = coarse_tokens(spoken) + dense_bursts(spoken) + identical_onsets(spoken)
    if probe is not None:
        origin = {number: start for number, start, _ in rows}
        if probe[1] not in origin:
            raise ValueError("Paging probe names an unknown recognition segment")
        findings += probe_findings(probe[0], offset=origin[probe[1]])
    per_segment = [
        {
            "segment": number,
            "outputStart": round(start, 3),
            "outputEnd": round(end, 3),
            "coarseTokens": _count(findings, "coarse-token", start, end),
            "denseBursts": _count(findings, "dense-burst", start, end),
            "identicalOnsets": _count(findings, "identical-onset", start, end),
            "displayDensity": _count(findings, "display-density", start, end),
        }
        for number, start, end in rows
    ]
    return {
        "version": 1,
        "kind": "caption-density-audit",
        "rate": RATE,
        "focusedLimitSeconds": FOCUSED_TIMING_SECONDS,
        "burst": {"frames": BURST_FRAMES, "tokens": BURST_TOKENS},
        "tokens": len(spoken),
        "onsetsPerFrameHistogram": frame_histogram(spoken),
        "findings": [
            {"kind": f.kind, "text": f.text, "detail": f.detail, **_located(f, plan)}
            for f in findings
        ],
        "segments": [row for row in per_segment if any(row[key] for key in COUNT_KEYS)],
        "windows": [
            {"kinds": list(w.kinds), "originalAsrText": w.text, **_located(w, plan)}
            for w in windows(findings, spoken, duration=duration)
        ],
        "candidateOnly": True,
        "timingChanged": False,
    }
