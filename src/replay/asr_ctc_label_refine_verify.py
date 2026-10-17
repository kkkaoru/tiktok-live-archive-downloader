"""Read-only proof that a CTC-refined label set differs from its original only as declared.

Checks (a) only the declared replaced block and declared duplicate removals differ from the
pinned original assembly, (b) starts are monotonic and the changed region overlaps no label,
(c) every replaced label shows for at least one frame, and (d) each replaced label cites a
supported (``weakSupport`` false) token of the pinned CTC candidate with its measured
``meanLogProbability`` and onset. Under ``mergeWeak`` a label may also list trailing weak
tokens it did not use for timing. It proves consistency of the evidence, never speech accuracy.
"""

from fractions import Fraction
from pathlib import Path

from replay.asr_clock_projection import _labels
from replay.asr_ctc_label_refine import ONSET, Token, read_json, tokens, voiced
from replay.caption_repair import ClockLabel
from replay.recordings import record
from replay.stages import file_digest

TOLERANCE = 1e-9
ALLOWED_CHANGES = frozenset(
    {
        "labels",
        "parentSHA256",
        "ctcRefinementSHA256",
        "ctcRefinement",
        "humanReviewed",
        "speechAccuracyVerified",
        "qualityApproved",
        "deliveryComplete",
    }
)


def _pinned(value: object) -> Path:
    ref = record(value)
    path = Path(str(ref.get("path")))
    if file_digest(path) != ref.get("sha256"):
        raise ValueError("A pinned verification input changed")
    return path


def _removed_index(original: tuple[ClockLabel, ...], removed: tuple[ClockLabel, ...]) -> int:
    if not removed:
        raise ValueError("The refinement declares no replaced block")
    for index in range(len(original) - len(removed) + 1):
        if original[index : index + len(removed)] == removed:
            return index
    raise ValueError("Declared removed labels are not a contiguous original block")


def _clock_label(item: object) -> ClockLabel:
    row = record(item)
    text, provenance = row.get("text"), row.get("provenance")
    if not isinstance(text, str) or not isinstance(provenance, str):
        raise ValueError("Invalid declared label")
    return ClockLabel(text, _number(row.get("start")), _number(row.get("end")), provenance)


def _same(label: ClockLabel, item: object) -> bool:
    row = record(item)
    return (
        row.get("text") == label.text
        and row.get("start") == label.start
        and row.get("end") == label.end
        and row.get("provenance") == label.provenance
    )


def verify(refined: Path) -> dict[str, object]:
    """Return a proof summary or raise ``ValueError`` naming the failed property."""
    document = read_json(refined)
    meta = record(document.get("ctcRefinement"))
    original_doc = read_json(_pinned(meta.get("labels")))
    original = _labels(original_doc)
    changed = {
        k for k in set(document) | set(original_doc) if document.get(k) != original_doc.get(k)
    }
    if not changed <= ALLOWED_CHANGES:
        raise ValueError(f"Undeclared document fields differ: {sorted(changed - ALLOWED_CHANGES)}")
    removed = tuple(_clock_label(item) for item in _rows(meta.get("removed")))
    deleted = tuple(_clock_label(item) for item in _rows(meta.get("deleted")))
    selected = _rows(meta.get("selected"))
    indices = meta.get("deletedIndices")
    if not isinstance(indices, list) or [original[i] for i in indices if type(i) is int] != list(
        deleted
    ):
        raise ValueError("Declared deletions differ from the original labels")
    gone = {i for i in indices if type(i) is int}
    low = _removed_index(original, removed) if removed else 0
    high = low + len(removed)
    _duplicates(original, gone, (low, high))
    _carried(original, removed, selected, fold=meta.get("foldUnvoiced") is True)
    expected_keep = [i for i in range(len(original)) if i not in gone]
    rows = _rows(document.get("labels"))
    if len(rows) != len(original) - len(deleted) - len(removed) + len(selected):
        raise ValueError("Refined label count differs from the declared edits")
    position = 0
    for index in expected_keep:
        if removed and low <= index < high:
            if index == low:
                if rows[position : position + len(selected)] != selected:
                    raise ValueError("Replaced labels differ from the declared selection")
                position += len(selected)
            continue
        if not _same(original[index], rows[position]):
            raise ValueError(f"Label {index} differs outside the declared edits")
        position += 1
    _monotonic(rows)
    frame = Fraction(str(record(meta.get("clock")).get("frame")))
    offset = Fraction(str(record(meta.get("clock")).get("sourceAudioStart")))
    aligned_rows = [row for row in selected if "timing" in row]
    if aligned_rows:
        candidate = read_json(_pinned(meta.get("candidate")))
        _cited(
            aligned_rows,
            tokens(candidate),
            frame=frame,
            offset=offset,
            merge_weak=meta.get("mergeWeak") is True,
        )
    return {
        "labels": len(rows),
        "replaced": len(aligned_rows),
        "carried": len(selected) - len(aligned_rows),
        "deleted": len(deleted),
        "frame": str(frame),
    }


def _duplicates(original: tuple[ClockLabel, ...], gone: set[int], region: tuple[int, int]) -> None:
    for index in gone:
        if not 0 < index < len(original) or region[0] <= index < region[1]:
            raise ValueError("A declared deletion is out of range or inside the replaced block")
        if original[index].start != original[index - 1].start:
            raise ValueError("A deleted label does not share its predecessor's onset")


def _folded(row: dict[str, object], *, fold: bool) -> list[dict[str, object]]:
    """Unvoiced-only labels appended to a timed label; only valid when foldUnvoiced is declared."""
    folded = row.get("foldedUnvoiced", [])
    if not isinstance(folded, list) or (folded and not fold):
        raise ValueError("A label folds unvoiced labels without foldUnvoiced")
    members = [record(item) for item in folded]
    suffix = "".join(str(member.get("text")) for member in members)
    if not str(row.get("text")).endswith(suffix):
        raise ValueError("A folded label does not end with its folded unvoiced text")
    return members


def _carried(
    original: tuple[ClockLabel, ...],
    removed: tuple[ClockLabel, ...],
    selected: list[dict[str, object]],
    *,
    fold: bool,
) -> None:
    """Unvoiced-only labels inside the block are carried over exactly or folded; text is kept."""
    kept: list[dict[str, object]] = []
    pieces: list[str] = []
    for row in selected:
        members = _folded(row, fold=fold) if "timing" in row else []
        kept.extend(members if members else [row] if row.get("carried") is True else [])
        suffix = "".join(str(member.get("text")) for member in members)
        if "timing" in row:
            pieces.append(str(row.get("text"))[: len(str(row.get("text"))) - len(suffix)])
    silent = [label for label in removed if not voiced(label.text)]
    if len(kept) != len(silent) or not all(_same(a, b) for a, b in zip(silent, kept, strict=True)):
        raise ValueError("Unvoiced-only labels were not carried over unchanged")
    before = "".join(label.text for label in removed if voiced(label.text))
    if before != "".join(pieces):
        raise ValueError("The replaced block changed its recognized text")


def _rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("Expected a list of label records")
    return [record(item) for item in value]


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number")
    return float(value)


def _monotonic(rows: list[dict[str, object]]) -> None:
    for before, after in zip(rows, rows[1:], strict=False):
        if _number(after["start"]) < _number(before["start"]):
            raise ValueError("Label onsets are not monotonic")
        if "timing" in before and _number(before["end"]) > _number(after["start"]) + TOLERANCE:
            raise ValueError("A replaced label overlaps its successor")
        if "timing" in after and _number(before["end"]) > _number(after["start"]) + TOLERANCE:
            raise ValueError("A replaced label overlaps its predecessor")


def _timed(
    ctc: dict[str, object], indices: list[int], aligned: list[Token], *, merge_weak: bool
) -> list[Token]:
    """Tokens that time the label; weak ones may only trail them, and only under mergeWeak."""
    unused = ctc.get("unusedWeak", [])
    if not isinstance(unused, list) or any(type(i) is not int for i in unused):
        raise ValueError("A replaced label cites invalid unused weak tokens")
    if unused and not (merge_weak and unused == indices[len(indices) - len(unused) :]):
        raise ValueError("A replaced label cites unused weak tokens without mergeWeak")
    if any(not aligned[i].weak for i in unused if type(i) is int):
        raise ValueError("A replaced label marks a supported token as unused")
    return [aligned[i] for i in indices[: len(indices) - len(unused)]]


def _cited(
    selected: list[dict[str, object]],
    aligned: list[Token],
    *,
    frame: Fraction,
    offset: Fraction,
    merge_weak: bool,
) -> None:
    for row in selected:
        ctc = record(row.get("ctc"))
        timing = record(row.get("timing"))
        indices = ctc.get("tokenIndices")
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(i) is not int or not 0 <= i < len(aligned) for i in indices)
        ):
            raise ValueError("A replaced label cites no CTC token")
        cited = [aligned[i] for i in indices if type(i) is int]
        timed = _timed(ctc, [i for i in indices if type(i) is int], aligned, merge_weak=merge_weak)
        first = cited[0]
        if (
            any(token.weak for token in timed)
            or ctc.get("weakSupport") is not False
            or ctc.get("meanLogProbabilities") != [t.mean_log_probability for t in cited]
            or voiced(str(row.get("text"))) != "".join(voiced(token.text) for token in cited)
            or timing.get("onset") != ONSET
            or abs(
                _number(row.get("start"))
                - float(Fraction(first.onset).limit_denominator(10**9) - offset)
            )
            > TOLERANCE
        ):
            raise ValueError("A replaced label does not match its supported CTC token")
        if _number(row.get("end")) - _number(row.get("start")) < float(frame) * (1 - TOLERANCE):
            raise ValueError("A replaced label shows for less than one frame")
