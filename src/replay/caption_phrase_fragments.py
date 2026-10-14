"""Hash-bound display grouping of intact dependent recognition fragments."""

import argparse
import hashlib
import sys
from fractions import Fraction
from math import isfinite
from pathlib import Path

from replay.caption_linguistics import (
    DEPENDENT_PARTS,
    DICTIONARY_PATH,
    MAX_TEXT_LENGTH,
    _tokenizer,
    recognition_continuity,
)
from replay.caption_measured_job import _token
from replay.caption_native_inputs import segment_binding
from replay.caption_native_paging import read_tokens
from replay.caption_onset_reconcile import Reader, _job_rows, _path, _reference, _values
from replay.caption_timing import CaptionPolicy, display_time
from replay.caption_token_sources import load_sources
from replay.core import ReplayError
from replay.delivery_captions import Token, _local_token
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.sequence_journal import encode, publish
from replay.stages import file_digest


def group_fragments(
    values: list[object], *, maximum_gap: float = CaptionPolicy().maximum_gap
) -> tuple[list[object], list[dict[str, object]]]:
    """Repair subframe dependent fragments; retain ordinary fine-ASR fit evidence."""
    if not isfinite(maximum_gap) or not 0 <= maximum_gap <= 1 or len(values) > 100_000:
        raise ReplayError("Invalid fragment grouping budget")
    tokens = tuple(_token(value) for value in values)
    if any(right.start < left.start for left, right in zip(tokens, tokens[1:], strict=False)):
        raise ReplayError("Recognition onset order regressed")
    normalized: list[object] = []
    decisions: list[dict[str, object]] = []
    first = 0
    while first < len(tokens):
        stop = first + 1
        length = len(tokens[first].text)
        while stop < len(tokens):
            before, after = tokens[stop - 1], tokens[stop]
            if (
                recognition_continuity(before.utterance) != recognition_continuity(after.utterance)
                or after.start - before.end > maximum_gap
                or length + len(after.text) > MAX_TEXT_LENGTH
            ):
                break
            length += len(after.text)
            stop += 1
        text = "".join(token.text for token in tokens[first:stop])
        items = tuple(_tokenizer().tokenize(text))
        if "".join(item.surface() for item in items) != text:
            raise ReplayError("Morphology changed original recognition text")
        features = {item.start(): item.feature().split(",") for item in items}
        punctuation = {
            point
            for item in items
            if item.feature().split(",")[0] == "記号"
            for point in range(item.start(), item.end() + 1)
        }
        starts = [first]
        position = 0
        for index in range(first + 1, stop):
            position += len(tokens[index - 1].text)
            feature = features.get(position)
            if feature is not None and len(feature) < 2:
                raise ReplayError("Malformed morphology feature")
            dependent = feature is not None and (
                feature[0] in DEPENDENT_PARTS or feature[1] in {"接尾", "非自立"}
            )
            short = any(
                display_time(token.end) - display_time(token.start) < Fraction(1, 30)
                for token in (tokens[index - 1], tokens[index])
            )
            can_join = (
                position not in punctuation
                and (dependent or feature is None)
                and short
                and tokens[index].end - tokens[starts[-1]].start
                <= CaptionPolicy().maximum_context_seconds
            )
            if not can_join:
                starts.append(index)
        starts.append(stop)
        for left, right in zip(starts, starts[1:], strict=False):
            originals = values[left:right]
            if right - left == 1:
                normalized.append(values[left])
                continue
            grouped = {
                **record(values[left]),
                "text": "".join(token.text for token in tokens[left:right]),
                "end": max(token.end for token in tokens[left:right]),
            }
            normalized.append(grouped)
            decisions.append(
                {
                    "originalIndices": list(range(left, right)),
                    "originalRows": originals,
                    "displayRow": grouped,
                    "firstActualAnchor": tokens[left].start,
                    "reason": "subframe-dependent-or-intra-morpheme-fragment",
                }
            )
        first = stop
    if "".join(_token(row).text for row in normalized) != "".join(t.text for t in tokens):
        raise ReplayError("Grouping did not conserve original text")
    return normalized, decisions


def _artifacts(
    directory: Path, output: Path, selected: tuple[int, ...]
) -> tuple[dict[Path, bytes], list[object]]:
    _path(directory)
    _path(output)
    if not output.is_relative_to(Path.cwd() / "edits" / "work"):
        raise ReplayError("Fragment artifacts require checkout work storage")
    reader = Reader()
    receipt = reader.read(directory / "receipt.json")
    index_path, index = reader.ref(receipt.get("segments"))
    _, config = reader.ref(receipt.get("config"))
    _, projected = reader.ref(config.get("projectedTokens"))
    _, labels = reader.ref(projected.get("labels"))
    if (
        index_path != directory / "segments.json"
        or index.get("rate") != 30
        or projected.get("clock") != "output"
        or projected.get("boundaryConflicts") != []
        or projected.get("cuts") != config.get("cuts")
        or labels.get("clock") != "source-presentation"
    ):
        raise ReplayError("Original physical recognition provenance differs")
    reader.ref(config.get("cuts"))
    entries = _values(index, "segments")
    if len(entries) > 512 or len(set(selected)) != len(selected):
        raise ReplayError("Invalid segment selection")
    projected_rows = _values(projected, "tokens")
    by_owner: dict[str, list[object]] = {}
    for value in projected_rows:
        by_owner.setdefault(_token(value).utterance, []).append(value)
    code = Path(__file__).resolve()
    reader.inputs[str(code)] = file_digest(code)
    dictionary = DICTIONARY_PATH.resolve()
    reader.inputs[str(dictionary)] = file_digest(dictionary)
    files: dict[Path, bytes] = {}
    replacements: list[object] = []
    audit: list[object] = []
    seen: set[int] = set()
    for value in entries:
        row = record(value)
        number, offset = row.get("segment"), row.get("offsetFrame")
        if type(number) is not int or type(offset) is not int or number in seen or offset < 0:
            raise ReplayError("Invalid original segment coordinates")
        seen.add(number)
        combined = segment_binding(directory, row)
        binding = reader.read(combined)
        bindings = record(binding.get("inputs"))
        original_inputs = record(receipt.get("inputs"))
        if any(bindings.get(path) != digest for path, digest in original_inputs.items()):
            raise ReplayError("Recognition producer input contract changed")
        read_tokens(directory, row, bindings)
        affected: list[dict[str, object]] = []
        for job in _values(row, "jobs"):
            if type(job) is not int:
                raise ReplayError("Invalid original job")
            original = directory / f"tokens-{job:04d}.json"
            document = reader.read(original, bindings.get(str(original)))
            values = _job_rows(document, "tokens")
            for local in values:
                token = _token(local)
                if not any(
                    _local_token(
                        Token(
                            _token(parent).text,
                            _token(parent).start,
                            _token(parent).end,
                            _token(parent).utterance,
                        ),
                        offset / 30,
                    )
                    == local
                    for parent in by_owner.get(token.utterance, [])
                ):
                    raise ReplayError("Fragment lost its original projected anchor")
            grouped, decisions = group_fragments(values)
            audit.append(
                {
                    "segment": number,
                    "job": job,
                    "originalCount": len(values),
                    "displayCount": len(grouped),
                    "groups": len(decisions),
                    "originalSubframeSpans": sum(
                        _token(item).end - _token(item).start < 1 / 30 for item in values
                    ),
                    "displaySubframeSpans": sum(
                        _token(item).end - _token(item).start < 1 / 30 for item in grouped
                    ),
                    "shortIndependentRows": [
                        item for item in grouped if _token(item).end - _token(item).start < 1 / 30
                    ],
                }
            )
            if not decisions or number not in selected:
                continue
            tokens_path = output / f"tokens-{number:04d}-{job:04d}.json"
            files[tokens_path] = encode({**document, "tokens": grouped})
            manifest = output / f"adopted-{number:04d}-{job:04d}.json"
            files[manifest] = encode(
                {
                    "version": 1,
                    "tokens": _reference(tokens_path, files[tokens_path]),
                    "originalTokens": {
                        "path": str(original),
                        "sha256": reader.inputs[str(original)],
                    },
                    "originalRows": values,
                    "decisions": decisions,
                    "inputs": dict(reader.inputs),
                    "originalAnchorsUnchanged": True,
                    "displayGroupingChanged": True,
                    "qualityApproved": False,
                    "speechAccuracyVerified": False,
                }
            )
            affected.append(
                {
                    "segment": number,
                    "job": job,
                    "tokens": _reference(tokens_path, files[tokens_path]),
                    "adoptedManifest": _reference(manifest, files[manifest]),
                }
            )
        if affected:
            additions = {str(combined): reader.inputs[str(combined)]}
            for replacement in affected:
                for key in ("tokens", "adoptedManifest"):
                    ref = record(replacement[key])
                    additions[str(ref["path"])] = str(ref["sha256"])
            overlay = output / f"combined-{number:04d}.json"
            files[overlay] = encode({**binding, "inputs": {**bindings, **additions}})
            for replacement in affected:
                replacement.update(
                    {
                        "combined": _reference(overlay, files[overlay]),
                        "originalCombined": {
                            "path": str(combined),
                            "sha256": reader.inputs[str(combined)],
                        },
                    }
                )
            replacements.extend(affected)
    if not set(selected) <= seen:
        raise ReplayError("Selected segment does not exist")
    files[output / "token-sources.json"] = encode({"version": 1, "replacements": replacements})
    files[output / "audit.json"] = encode(
        {
            "version": 1,
            "inputs": reader.inputs,
            "selected": list(selected),
            "segments": audit,
            "originalAnchorsUnchanged": True,
            "speechAccuracyVerified": False,
            "qualityApproved": False,
        }
    )
    reader.verify()
    return files, entries


def verify(directory: Path, output: Path, selected: tuple[int, ...]) -> None:
    files, entries = _artifacts(directory, output, selected)
    for path, content in files.items():
        if _bound_bytes(path, hashlib.sha256(content).hexdigest(), 64 << 20) != content:
            raise ReplayError("Phrase-fragment artifacts changed")
    sources, _ = load_sources(output / "token-sources.json", entries)
    for entry in entries:
        row = record(entry)
        binding = Reader().read(segment_binding(directory, row, sources))
        read_tokens(directory, row, record(binding.get("inputs")), sources)


def prepare(directory: Path, output: Path, selected: tuple[int, ...]) -> Path:
    if output.exists() or not selected:
        raise ReplayError("Expected fresh output and explicit affected segments")
    files, _ = _artifacts(directory, output, selected)
    output.mkdir(mode=0o700)
    for path, content in files.items():
        publish(path, content)
    verify(directory, output, selected)
    return output / "token-sources.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--segment", type=int, action="append", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            verify(args.directory, args.output, tuple(args.segment))
        else:
            prepare(args.directory, args.output, tuple(args.segment))
    except (OSError, ValueError, ReplayError) as error:
        print(f"CAPTION_PHRASE_FRAGMENTS_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
