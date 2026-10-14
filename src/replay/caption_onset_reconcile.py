"""Reconcile proven overlapping-window onset duplicates without changing token clocks."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from replay.caption_linguistics import ASSEMBLED_GROUP_OWNER, RAW_CHUNK_NAME, recognition_continuity
from replay.caption_measured_job import _pairs, _token
from replay.caption_native_inputs import segment_binding
from replay.caption_native_paging import read_tokens
from replay.caption_token_sources import load_sources
from replay.core import ReplayError
from replay.delivery_captions import Token, _local_token
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.sequence_journal import encode, publish
from replay.stages import file_digest

REPOSITORY = Path(__file__).resolve().parents[2]
MAX_BYTES = 64 << 20
MAX_TOTAL_BYTES = 256 << 20
MAX_TOKENS = 100_000


def _path(path: Path) -> Path:
    if not path.is_absolute() or path != path.resolve() or not path.is_relative_to(REPOSITORY):
        raise ReplayError("Expected canonical checkout-local path")
    return path


class Reader:
    def __init__(self) -> None:
        self.inputs: dict[str, str] = {}
        self.total = 0

    def read(self, path: Path, expected: object = None) -> dict[str, object]:
        _path(path)
        size = path.stat().st_size
        self.total += size
        if not 0 < size <= MAX_BYTES or self.total > MAX_TOTAL_BYTES:
            raise ReplayError("Reconciliation metadata budget exceeded")
        digest = file_digest(path)
        if expected is not None and expected != digest:
            raise ReplayError("Original metadata binding changed")
        self.inputs[str(path)] = digest
        return record(json.loads(_bound_bytes(path, digest, MAX_BYTES), object_pairs_hook=_pairs))

    def ref(self, value: object) -> tuple[Path, dict[str, object]]:
        ref = record(value)
        if (
            set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str)
            or not isinstance(ref["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None
        ):
            raise ReplayError("Expected exact original reference")
        path = Path(ref["path"])
        return path, self.read(path, ref["sha256"])

    def verify(self) -> None:
        if any(file_digest(_path(Path(path))) != digest for path, digest in self.inputs.items()):
            raise ReplayError("Original inputs changed during reconciliation")


def _window(owner: str) -> tuple[str, str]:
    original, separator, cut = owner.rpartition(":cut-")
    if not separator or re.fullmatch(r"[0-9]+", cut) is None:
        raise ReplayError("Unknown recognition owner or cut")
    assembled = ASSEMBLED_GROUP_OWNER.fullmatch(original)
    if assembled is not None:
        return original, "assembled:" + assembled.group("window")
    path = Path(original)
    if RAW_CHUNK_NAME.fullmatch(path.name):
        return original, str(path)
    raise ReplayError("Unknown recognition window")


def reconcile_tokens(
    values: list[object], *, known_owners: frozenset[str]
) -> tuple[list[object], list[dict[str, object]]]:
    """Choose an intact longest original row; ties choose the earliest original index."""
    if not 1 <= len(values) <= MAX_TOKENS:
        raise ReplayError("Invalid token count")
    tokens = tuple(_token(value) for value in values)
    selected: list[object] = []
    decisions: list[dict[str, object]] = []
    position = 0
    while position < len(tokens):
        end = position + 1
        while end < len(tokens) and tokens[end].start == tokens[position].start:
            end += 1
        if position and tokens[position].start < tokens[position - 1].start:
            raise ReplayError("Token onset order regressed")
        candidates = tokens[position:end]
        if len(candidates) > 1:
            identities = [_window(token.utterance) for token in candidates]
            if (
                len({token.text for token in candidates}) != 1
                or len({recognition_continuity(token.utterance) for token in candidates}) != 1
                or any(owner not in known_owners for owner, _ in identities)
                or len({window for _, window in identities}) != len(candidates)
                or max(token.start for token in candidates)
                >= min(token.end for token in candidates)
            ):
                raise ReplayError("Ambiguous equal-onset recognition tokens")
            chosen = max(range(position, end), key=lambda index: (tokens[index].end, -index))
            decisions.append(
                {
                    "candidateIndices": list(range(position, end)),
                    "candidates": values[position:end],
                    "chosenIndex": chosen,
                }
            )
        else:
            chosen = position
        selected.append(values[chosen])
        position = end
    return selected, decisions


def _values(document: dict[str, object], key: str) -> list[object]:
    values = document.get(key)
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_TOKENS:
        raise ReplayError("Expected bounded metadata rows")
    return values


def _job_rows(document: dict[str, object], key: str) -> list[object]:
    """A recognized window may legitimately own zero tokens; bounds still apply."""
    values = document.get(key)
    if not isinstance(values, list) or len(values) > MAX_TOKENS:
        raise ReplayError("Expected bounded metadata rows")
    return values


def _reference(path: Path, content: bytes) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def _source_decisions(decisions: list[dict[str, object]], labels: list[object]) -> None:
    for decision in decisions:
        originals: list[dict[str, object]] = []
        for candidate in _values(decision, "candidates"):
            token = _token(candidate)
            owner, _ = _window(token.utterance)
            matches = [
                record(row)
                for row in labels
                if record(row).get("provenance") == owner and record(row).get("text") == token.text
            ]
            if len(matches) != 1:
                raise ReplayError("Ambiguous original source onset")
            source = matches[0]
            _token(
                {
                    "text": source["text"],
                    "start": source.get("start"),
                    "end": source.get("end"),
                    "utterance": token.utterance,
                }
            )
            originals.append(source)
        if any(row["start"] != originals[0]["start"] for row in originals):
            raise ReplayError("Source onset differs before cut projection")
        decision["sourceCandidates"] = originals


def _artifacts(directory: Path, output: Path) -> tuple[dict[Path, bytes], list[object]]:
    _path(directory)
    _path(output)
    reader = Reader()
    receipt = reader.read(directory / "receipt.json")
    index_path, index = reader.ref(receipt.get("segments"))
    if (
        index_path != directory / "segments.json"
        or index.get("version") != 1
        or index.get("rate") != 30
    ):
        raise ReplayError("Invalid original segment index")
    _, config = reader.ref(receipt.get("config"))
    projected_path, projected = reader.ref(config.get("projectedTokens"))
    _, cuts = reader.ref(config.get("cuts"))
    labels_path, labels = reader.ref(projected.get("labels"))
    original_inputs = record(receipt.get("inputs"))
    if (
        projected.get("version") != 1
        or projected.get("clock") != "output"
        or projected.get("cuts") != config.get("cuts")
        or projected.get("boundaryConflicts") != []
        or labels.get("clock") != "source-presentation"
        or not cuts
        or any(
            original_inputs.get(str(path)) != reader.inputs[str(path)]
            for path in (projected_path, labels_path)
        )
    ):
        raise ReplayError("Original recognition provenance differs")
    owners = frozenset(str(record(row).get("provenance")) for row in _values(labels, "labels"))
    projected_values = _values(projected, "tokens")
    by_owner: dict[str, list[dict[str, object]]] = {}
    for value in projected_values:
        token = _token(value)
        by_owner.setdefault(token.utterance, []).append(record(value))
    entries = _values(index, "segments")
    if len(entries) > 512:
        raise ReplayError("Segment count exceeded")
    files: dict[Path, bytes] = {}
    replacements: list[dict[str, object]] = []
    all_decisions: list[dict[str, object]] = []
    code = Path(__file__).resolve()
    reader.inputs[str(code)] = file_digest(code)
    seen: set[tuple[int, int]] = set()
    for entry in entries:
        row = record(entry)
        number, offset = row.get("segment"), row.get("offsetFrame")
        if type(number) is not int or type(offset) is not int or offset < 0:
            raise ReplayError("Invalid segment coordinates")
        combined = segment_binding(directory, row)
        binding = reader.read(combined)
        bindings = record(binding.get("inputs"))
        if any(bindings.get(path) != digest for path, digest in original_inputs.items()):
            raise ReplayError("Recognition binding lost original source or context pins")
        read_tokens(directory, row, bindings)
        affected: list[dict[str, object]] = []
        for job in _values(row, "jobs"):
            if type(job) is not int or (number, job) in seen:
                raise ReplayError("Duplicate or invalid job membership")
            seen.add((number, job))
            original = directory / f"tokens-{job:04d}.json"
            document = reader.read(original, bindings.get(str(original)))
            values = _job_rows(document, "tokens")
            if not values:
                # A recognized window without owned tokens needs no overlay.
                continue
            for value in values:
                token = _token(value)
                candidates = by_owner.get(token.utterance, [])
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
                    == value
                    for parent in candidates
                ):
                    raise ReplayError("Local token differs from original projected recognition")
            normalized, decisions = reconcile_tokens(values, known_owners=owners)
            if not decisions:
                continue
            _source_decisions(decisions, _values(labels, "labels"))
            tokens_path = output / f"tokens-{number:04d}-{job:04d}.json"
            files[tokens_path] = encode({**document, "tokens": normalized})
            manifest_path = output / f"adopted-{number:04d}-{job:04d}.json"
            files[manifest_path] = encode(
                {
                    "version": 1,
                    "tokens": _reference(tokens_path, files[tokens_path]),
                    "originalTokens": {
                        "path": str(original),
                        "sha256": reader.inputs[str(original)],
                    },
                    "decisions": decisions,
                    "inputs": dict(reader.inputs),
                    "qualityApproved": False,
                    "speechAccuracyVerified": False,
                    "timestampsChanged": False,
                }
            )
            replacement: dict[str, object] = {
                "segment": number,
                "job": job,
                "tokens": _reference(tokens_path, files[tokens_path]),
                "adoptedManifest": _reference(manifest_path, files[manifest_path]),
            }
            affected.append(replacement)
            all_decisions.append({"segment": number, "job": job, "decisions": decisions})
        if affected:
            additions = {str(combined): reader.inputs[str(combined)]}
            for replacement in affected:
                for key in ("tokens", "adoptedManifest"):
                    ref = record(replacement[key])
                    additions[str(ref["path"])] = str(ref["sha256"])
            overlay = output / f"combined-{number:04d}.json"
            files[overlay] = encode({**binding, "inputs": {**bindings, **additions}})
            for replacement in affected:
                replacement["combined"] = _reference(overlay, files[overlay])
                replacement["originalCombined"] = {
                    "path": str(combined),
                    "sha256": reader.inputs[str(combined)],
                }
            replacements.extend(affected)
    files[output / "token-sources.json"] = encode({"version": 1, "replacements": replacements})
    files[output / "reconciliation.json"] = encode(
        {
            "version": 1,
            "inputs": reader.inputs,
            "decisions": all_decisions,
            "qualityApproved": False,
            "speechAccuracyVerified": False,
            "timestampsChanged": False,
        }
    )
    reader.verify()
    return files, entries


def verify(directory: Path, output: Path) -> None:
    files, entries = _artifacts(directory, output)
    for path, expected in files.items():
        if _bound_bytes(path, hashlib.sha256(expected).hexdigest(), MAX_BYTES) != expected:
            raise ReplayError("Reconciliation output changed")
    mapping = output / "token-sources.json"
    if not record(json.loads(files[mapping]))["replacements"]:
        return
    sources, _ = load_sources(mapping, entries)
    for entry in entries:
        row = record(entry)
        combined = segment_binding(directory, row, sources)
        bindings = record(Reader().read(combined).get("inputs"))
        read_tokens(directory, row, bindings, sources)


def prepare(directory: Path, output: Path) -> Path:
    if output.exists():
        raise ReplayError("Reconciliation requires a new output directory")
    files, _ = _artifacts(directory, output)
    output.mkdir(mode=0o700)
    for path, content in files.items():
        publish(path, content)
    verify(directory, output)
    return output / "token-sources.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            verify(args.directory, args.output)
        else:
            prepare(args.directory, args.output)
    except (OSError, ValueError, ReplayError) as error:
        print(f"CAPTION_ONSET_RECONCILE_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
