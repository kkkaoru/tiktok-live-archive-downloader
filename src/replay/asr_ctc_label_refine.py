"""Replace a supported run of assembled ASR labels with CTC-aligned onsets (candidate only).

A pinned CTC candidate (one entry per word, onset = strongest 20 ms emission frame, no end
time) is matched to an explicit block of assembled labels by exact text after both sides drop
whitespace and the aligner's unvoiced symbols (labels made only of such symbols stay as they
are). Only the longest contiguous run of tokens without ``weakSupport`` is replaced; weak
spans keep their original, unreviewed ASR labels. Each replaced label starts at the CTC onset
and ends at the next aligned onset (a derived value, recorded as such); the run's last label
keeps the original end when it shows at least one frame, else lasts one frame, and never
reaches the next original label. Declared duplicate-onset labels may be removed, with or
without a block. With ``mergeWeak`` a run of weak units lying between two supported units is
folded into the preceding supported label: the merged text is the exact concatenation, only
the strong first token times it and the weak tokens are cited as unused. The old assembly is
untouched and nothing here approves speech accuracy.
"""

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import accumulate, pairwise
from pathlib import Path

from replay.asr_clock_projection import _binding, _labels
from replay.caption_measured_job import _json
from replay.caption_repair import ClockLabel
from replay.caption_scale import FOCUSED_TIMING_SECONDS
from replay.recordings import record
from replay.stages import file_digest

MAX_BYTES = 128 * 1024 * 1024
ONSET = "ctc-peak"
END_NEXT = "derived-next-onset"
END_ORIGINAL = "original-asr-end"
END_FRAME = "onset-plus-one-frame"
# Mirrors the aligner: symbols with no acoustic event are neither aligned nor timed.
UNVOICED_SYMBOLS = frozenset("。、，．！？!?…・「」『』（）()〜~ー")


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    onset: float
    mean_log_probability: float
    weak: bool


@dataclass(frozen=True, slots=True)
class Outcome:
    labels: list[dict[str, object]]
    removed: list[ClockLabel]
    selected: list[dict[str, object]]
    deleted: list[ClockLabel]
    report: dict[str, object]


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number")
    return float(value)


def read_json(path: Path) -> dict[str, object]:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Refinement JSON exceeds byte budget")
    return record(_json(path.read_bytes()))


def voiced(text: str) -> str:
    """Text without whitespace and unvoiced symbols (what the aligner actually aligns)."""
    return "".join(c for c in text if not c.isspace() and c not in UNVOICED_SYMBOLS)


def tokens(candidate: dict[str, object]) -> list[Token]:
    rows = candidate.get("tokens")
    if not isinstance(rows, list) or not rows:
        raise ValueError("CTC candidate has no tokens")
    result: list[Token] = []
    for raw in rows:
        row = record(raw)
        text, weak = row.get("token"), row.get("weakSupport")
        if not isinstance(text, str) or not text or not isinstance(weak, bool):
            raise ValueError("Invalid CTC token")
        result.append(
            Token(
                text,
                _number(row.get("peakStart")),
                _number(row.get("meanLogProbability")),
                weak,
            )
        )
    return result


@dataclass(frozen=True, slots=True)
class Unit:
    """Labels [first, stop) of the block and CTC tokens [begin, end) that cover the same text."""

    first: int
    stop: int
    begin: int
    end: int
    support_end: int  # tokens [begin, support_end) time the label; the rest are weak and unused


def units(block: tuple[ClockLabel, ...], aligned: list[Token]) -> list[Unit | None]:
    """Chunks where token edges and label edges coincide; None when a chunk cannot be mapped.

    One token over several labels merges them into one word; several tokens over one label
    time that label by the first token. Crossing edges, or an unvoiced-only label inside a
    chunk, make the chunk unmappable (a recognized label is never cut).
    """
    texts = [voiced(label.text) for label in block]
    token_ends = list(accumulate(len(voiced(token.text)) for token in aligned))
    if "".join(texts) != "".join(voiced(token.text) for token in aligned):
        raise ValueError("CTC text differs from the declared label block")
    voiced_labels = [i for i, text in enumerate(texts) if text]
    label_ends = list(accumulate(len(texts[i]) for i in voiced_labels))
    shared = sorted(set(token_ends) & set(label_ends))
    result: list[Unit | None] = []
    token_cursor = label_cursor = 0
    for boundary in shared:
        tokens_here = [j for j in range(token_cursor, len(aligned)) if token_ends[j] <= boundary]
        labels_here = [
            k for k in range(label_cursor, len(voiced_labels)) if label_ends[k] <= boundary
        ]
        first, stop = voiced_labels[labels_here[0]], voiced_labels[labels_here[-1]] + 1
        silent = any(not texts[i] for i in range(first, stop))
        mappable = not silent and (len(tokens_here) == 1 or len(labels_here) == 1)
        unit = Unit(first, stop, tokens_here[0], tokens_here[-1] + 1, tokens_here[-1] + 1)
        result.append(unit if mappable else None)
        token_cursor, label_cursor = tokens_here[-1] + 1, labels_here[-1] + 1
    return result


def _weak(aligned: list[Token], unit: Unit) -> bool:
    return any(aligned[j].weak for j in range(unit.begin, unit.support_end))


def _all_weak(aligned: list[Token], unit: Unit) -> bool:
    """A unit made only of weak tokens; one with a supported token cannot be cited as unused."""
    return all(aligned[j].weak for j in range(unit.begin, unit.end))


def _fold_group(
    mapped: list[Unit | None], aligned: list[Token], index: int
) -> tuple[Unit, int] | None:
    """The supported unit at ``index`` merged with the weak units it bridges, and the next index.

    None unless at least one weak unit follows contiguously and a supported unit continues
    the label sequence without a gap right after them.
    """
    merged = _unit(mapped, index)
    end = index + 1
    while end < len(mapped):
        weak = mapped[end]
        if weak is None or not _all_weak(aligned, weak) or weak.first != merged.stop:
            break
        merged = Unit(merged.first, weak.stop, merged.begin, weak.end, merged.support_end)
        end += 1
    after = mapped[end] if end < len(mapped) else None
    if end == index + 1 or after is None or _weak(aligned, after) or after.first != merged.stop:
        return None
    return merged, end


def fold_weak(mapped: list[Unit | None], aligned: list[Token]) -> list[Unit | None]:
    """Fold weak units that sit between two supported, contiguous units into the left one.

    The left unit must itself be supported (its first token is strong); the folded labels keep
    their text and cite their weak tokens as unused. Gaps holding unvoiced-only labels, and
    weak units that do not touch a supported unit on both sides, are left exactly as they are.
    """
    result: list[Unit | None] = []
    index = 0
    while index < len(mapped):
        unit = mapped[index]
        folded = (
            None if unit is None or _weak(aligned, unit) else _fold_group(mapped, aligned, index)
        )
        if folded is None:
            result.append(unit)
            index += 1
        else:
            result.append(folded[0])
            index = folded[1]
    return result


def _attempt(
    labels: tuple[ClockLabel, ...],
    aligned: list[Token],
    mapped: list[Unit | None],
    times: dict[int, float],
    run: tuple[int, int],
    *,
    first: int,
    frame: Fraction,
    fold_unvoiced: bool,
) -> tuple[int, int, list[ClockLabel], list[dict[str, object]]] | str:
    """The replacement of one unit run, or the name of the rule that refused it."""
    low = first + _unit(mapped, run[0]).first
    high = first + _unit(mapped, run[1] - 1).stop
    removed = list(labels[low:high])
    try:
        selected = selected_labels(
            labels,
            aligned,
            mapped,
            times,
            run,
            first=first,
            frame=frame,
            limit=high,
            fold_unvoiced=fold_unvoiced,
        )
    except ValueError as error:
        return str(error)
    if low > 0 and labels[low - 1].end > times[run[0]]:
        return "Aligned run overlaps the unchanged left neighbour"
    return low, high, removed, selected


def _unit(mapped: list[Unit | None], index: int) -> Unit:
    unit = mapped[index]
    if unit is None:
        raise ValueError("Chunk does not align with whole labels")
    return unit


def _best_run(
    labels: tuple[ClockLabel, ...],
    aligned: list[Token],
    mapped: list[Unit | None],
    times: dict[int, float],
    run: tuple[int, int],
    *,
    first: int,
    frame: Fraction,
    fold_unvoiced: bool,
) -> tuple[tuple[int, int], int, int, list[ClockLabel], list[dict[str, object]]] | Counter[str]:
    """Longest sub-run (earliest on ties) that shows every label a frame and fits its neighbours."""
    failures: Counter[str] = Counter()
    for length in range(run[1] - run[0], 0, -1):
        for begin in range(run[0], run[1] - length + 1):
            candidate = (begin, begin + length)
            outcome = _attempt(
                labels,
                aligned,
                mapped,
                times,
                candidate,
                first=first,
                frame=frame,
                fold_unvoiced=fold_unvoiced,
            )
            if isinstance(outcome, str):
                failures[outcome] += 1
            else:
                return candidate, outcome[0], outcome[1], outcome[2], outcome[3]
    return failures


def longest_supported_run(
    aligned: list[Token], mapped: list[Unit | None]
) -> tuple[int, int] | None:
    """Longest contiguous run of supported, label-aligned units (first one on ties)."""
    best: tuple[int, int] | None = None
    start: int | None = None
    for index in range(len(mapped) + 1):
        unit = mapped[index] if index < len(mapped) else None
        supported = unit is not None and not _weak(aligned, unit)
        if supported and start is None:
            start = index
        if not supported and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    return best


def _label(item: ClockLabel, **extra: object) -> dict[str, object]:
    return {**asdict(item), **extra}


def refine(
    labels: tuple[ClockLabel, ...],
    aligned: list[Token],
    *,
    first: int,
    stop: int,
    offset: Fraction,
    frame: Fraction,
    delete: tuple[int, ...] = (),
    merge_weak: bool = False,
    fold_unvoiced: bool = False,
) -> Outcome:
    """Pure planning step; unchanged labels keep their exact original values."""
    if not 0 <= first < stop <= len(labels) or frame <= 0:
        raise ValueError("Expected a bounded label block and a positive frame period")
    mapped = units(labels[first:stop], aligned)
    if merge_weak:
        mapped = fold_weak(mapped, aligned)
    report: dict[str, object] = {
        "block": [first, stop],
        "tokens": len(aligned),
        "chunks": len(mapped),
        "weakTokens": sum(1 for token in aligned if token.weak),
        "unmappedChunks": sum(1 for unit in mapped if unit is None),
        "replaced": False,
    }
    selected: list[dict[str, object]] = []
    removed: list[ClockLabel] = []
    low = high = first
    run = longest_supported_run(aligned, mapped)
    if run is None:
        report["skipped"] = "no token is both acoustically supported and aligned to whole labels"
    else:
        times = {
            u: float(
                Fraction(aligned[_unit(mapped, u).begin].onset).limit_denominator(10**9) - offset
            )
            for u in range(*run)
        }
        found = _best_run(
            labels,
            aligned,
            mapped,
            times,
            run,
            first=first,
            frame=frame,
            fold_unvoiced=fold_unvoiced,
        )
        if isinstance(found, Counter):
            report["skipped"] = "no sub-run satisfies the frame, order and neighbour rules"
            report["failures"] = dict(found)
        else:
            run, low, high, removed, selected = found
            report.update(replaced=True, run=list(run))
    return _assemble(labels, delete, (low, high), selected, removed, report)


def refine_deletions(labels: tuple[ClockLabel, ...], delete: tuple[int, ...]) -> Outcome:
    """Declared duplicate-onset removals without any CTC replacement."""
    report: dict[str, object] = {"block": None, "replaced": False}
    return _assemble(labels, delete, (0, 0), [], [], report)


def _assemble(
    labels: tuple[ClockLabel, ...],
    delete: tuple[int, ...],
    region: tuple[int, int],
    selected: list[dict[str, object]],
    removed: list[ClockLabel],
    report: dict[str, object],
) -> Outcome:
    low, high = region
    deleted = _deletions(labels, delete, low, high)
    gone = set(delete)
    result: list[dict[str, object]] = []
    starts: list[float] = []
    for index in range(len(labels)):
        if index in gone:
            continue
        if low <= index < high:
            if index == low:
                result.extend(selected)
                starts.extend(_number(item["start"]) for item in selected)
            continue
        result.append(_label(labels[index]))
        starts.append(labels[index].start)
    if any(b < a for a, b in pairwise(starts)):
        raise ValueError("Refined labels reverse word onsets")
    return Outcome(result, removed, selected, deleted, report)


def selected_labels(
    labels: tuple[ClockLabel, ...],
    aligned: list[Token],
    mapped: list[Unit | None],
    times: dict[int, float],
    run: tuple[int, int],
    *,
    first: int,
    frame: Fraction,
    limit: int,
    fold_unvoiced: bool = False,
) -> list[dict[str, object]]:
    """CTC labels for the run plus the unvoiced-only labels between its units, unchanged.

    With ``fold_unvoiced`` those labels are instead appended, text and provenance intact and listed
    under ``foldedUnvoiced``, to the preceding timed label: their own onsets could precede the
    measured onset of the label they follow, and no time is invented for a symbol the aligner skips.
    """
    period = float(frame)
    indices = list(range(*run))
    chosen: list[dict[str, object]] = []
    for position, index in enumerate(indices):
        unit = _unit(mapped, index)
        members = labels[first + unit.first : first + unit.stop]
        start = times[index]
        if position + 1 < len(indices):
            end, how = times[indices[position + 1]], END_NEXT
        else:
            original = members[-1].end
            end, how = (
                (original, END_ORIGINAL)
                if original - start >= period
                else (start + period, END_FRAME)
            )
            if limit < len(labels):
                end = min(end, labels[limit].start)
        if end - start < period * (1 - 1e-9):
            raise ValueError("Aligned token would display for less than one frame")
        if unit.support_end < unit.end and end - start > FOCUSED_TIMING_SECONDS:
            raise ValueError("Merged label would be too coarse for focused captions")
        evidence = aligned[unit.begin : unit.end]
        chosen.append(
            {
                "text": "".join(member.text for member in members),
                "start": start,
                "end": end,
                "provenance": members[0].provenance,
                "timing": {"onset": ONSET, "end": how},
                "ctc": {
                    "tokenIndices": list(range(unit.begin, unit.end)),
                    "meanLogProbabilities": [token.mean_log_probability for token in evidence],
                    "weakSupport": False,
                    **(
                        {"unusedWeak": list(range(unit.support_end, unit.end))}
                        if unit.support_end < unit.end
                        else {}
                    ),
                },
            }
        )
        if position + 1 < len(indices):
            following = first + _unit(mapped, indices[position + 1]).first
            between = range(first + unit.stop, following)
            if fold_unvoiced and between:
                chosen[-1]["text"] = str(chosen[-1]["text"]) + "".join(
                    labels[k].text for k in between
                )
                chosen[-1]["foldedUnvoiced"] = [_label(labels[k]) for k in between]
            else:
                chosen.extend(_label(labels[k], carried=True) for k in between)
    return chosen


def _deletions(
    labels: tuple[ClockLabel, ...], delete: tuple[int, ...], low: int, high: int
) -> list[ClockLabel]:
    deleted: list[ClockLabel] = []
    for index in delete:
        if not 0 < index < len(labels) or low <= index < high:
            raise ValueError("Declared deletion is out of range or inside the replaced run")
        if labels[index].start != labels[index - 1].start:
            raise ValueError(
                "Only a label with the identical onset of its predecessor may be removed"
            )
        deleted.append(labels[index])
    return deleted


def run(config: Path, output: Path) -> int:
    """Write the candidate label set; returns its label count."""
    if output.exists() or output.is_symlink() or output.parent != output.parent.resolve():
        raise ValueError("Expected a new output under a canonical directory")
    digest = file_digest(config)
    data = read_json(config)
    keys = {"version", "artifacts", "clock", "block", "deleteLabels"}
    if set(data) != keys or data["version"] != 1:
        raise ValueError("Unsupported CTC refinement configuration")
    artifacts = record(data["artifacts"])
    block = data["block"]
    wanted = {"labels"} if block is None else {"labels", "candidate"}
    if set(artifacts) != wanted:
        raise ValueError(
            "Expected pinned labels, and a CTC candidate exactly when a block is given"
        )
    bound = {name: _binding(value) for name, value in artifacts.items()}
    for path, pin in bound.values():
        if file_digest(path) != pin:
            raise ValueError("A pinned refinement input changed")
    original = read_json(bound["labels"][0])
    clock = record(data["clock"])
    offset, frame = Fraction(str(clock.get("sourceAudioStart"))), Fraction(str(clock.get("frame")))
    deletes = data["deleteLabels"]
    if not isinstance(deletes, list) or any(type(item) is not int for item in deletes):
        raise ValueError("Expected exact integer deletion coordinates")
    declared = tuple(int(str(item)) for item in deletes)
    labels = _labels(original)
    candidate: dict[str, object] = {}
    merge = fold = False
    if block is None:
        if not declared:
            raise ValueError("A refinement without a block must declare deletions")
        outcome = refine_deletions(labels, declared)
    else:
        window = record(block)
        first, stop = window.get("firstLabel"), window.get("stopLabel")
        if type(first) is not int or type(stop) is not int:
            raise ValueError("Expected exact integer block coordinates")
        if set(window) - {"firstLabel", "stopLabel", "mergeWeak", "foldUnvoiced"}:
            raise ValueError("Unsupported block option")
        option, fold_option = window.get("mergeWeak", False), window.get("foldUnvoiced", False)
        if type(option) is not bool or type(fold_option) is not bool:
            raise ValueError("mergeWeak and foldUnvoiced must be booleans")
        merge, fold = option, fold_option
        candidate = read_json(bound["candidate"][0])
        outcome = refine(
            labels,
            tokens(candidate),
            first=first,
            stop=stop,
            offset=offset,
            frame=frame,
            delete=declared,
            merge_weak=merge,
            fold_unvoiced=fold,
        )
    pins = {name: {"path": str(path), "sha256": pin} for name, (path, pin) in bound.items()}
    document = {
        **original,
        "labels": outcome.labels,
        "parentSHA256": {
            **record(original["parentSHA256"]),
            **{str(path): pin for path, pin in bound.values()},
        },
        "ctcRefinementSHA256": file_digest(Path(__file__)),
        "ctcRefinement": {
            "configSHA256": digest,
            "labels": pins["labels"],
            "candidate": pins.get("candidate"),
            "clock": {"sourceAudioStart": str(offset), "frame": str(frame)},
            "model": candidate.get("model"),
            "mergeWeak": merge,
            "foldUnvoiced": fold,
            "removed": [asdict(label) for label in outcome.removed],
            "selected": outcome.selected,
            "deleted": [asdict(label) for label in outcome.deleted],
            "deletedIndices": list(declared),
            "report": outcome.report,
            "candidateOnly": True,
        },
        "humanReviewed": False,
        "speechAccuracyVerified": False,
        "qualityApproved": False,
        "deliveryComplete": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return len(outcome.labels)
