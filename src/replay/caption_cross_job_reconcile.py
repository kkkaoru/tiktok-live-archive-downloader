"""Reconcile proven overlap duplicates across recognition jobs within one segment."""

import argparse
import hashlib
import sys
from dataclasses import asdict
from pathlib import Path

from replay import asr_assembly, caption_onset_reconcile
from replay.asr_assembly import _windows
from replay.caption_io import mlx_groups
from replay.caption_linguistics import ASSEMBLED_GROUP_OWNER
from replay.caption_measured_job import _token
from replay.caption_native_inputs import segment_binding
from replay.caption_native_paging import read_tokens
from replay.caption_onset_reconcile import (
    Reader,
    _job_rows,
    _path,
    _reference,
    _source_decisions,
    _values,
    reconcile_tokens,
)
from replay.caption_token_sources import load_sources
from replay.core import ReplayError
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.sequence_journal import encode, publish
from replay.stages import file_digest


def ownership_proof(
    decision: dict[str, object], *, reader: Reader, collection: Path, labels: Path
) -> None:
    """Record actual midpoint ownership; never splice one owner's extent into another."""
    original = reader.read(labels)
    manifest_path, complete_path = collection / "inputs.json", collection / "complete.json"
    manifest, complete = reader.read(manifest_path), reader.read(complete_path)
    parents = record(original.get("parentSHA256"))
    if original.get("clock") != "input-pcm-not-projected" or any(
        parents.get(str(path)) != reader.inputs[str(path)]
        for path in (manifest_path, complete_path)
    ):
        raise ReplayError("Original PCM assembly does not bind this collection")
    windows = _windows(manifest)
    hashes = complete.get("rawSHA256")
    if (
        not isinstance(hashes, list)
        or len(hashes) != len(windows)
        or complete.get("windows") != len(windows)
        or complete.get("deliveryComplete") is not False
    ):
        raise ReplayError("Original completed window coverage differs")
    proofs: list[object] = []
    for candidate, source in zip(
        _values(decision, "candidates"), _values(decision, "sourceCandidates"), strict=True
    ):
        token, label = _token(candidate), record(source)
        owner = str(label.get("provenance"))
        matched = ASSEMBLED_GROUP_OWNER.fullmatch(owner)
        if matched is None:
            raise ReplayError("Cross-job reconciliation requires assembled raw-window ownership")
        start = int(matched.group("window"))
        matches = [index for index, window in enumerate(windows) if window.start_sample == start]
        if len(matches) != 1:
            raise ReplayError("Original raw owner is outside the collection")
        index = matches[0]
        window = windows[index]
        raw_path = collection / f"{start:012d}.raw.json"
        clock_path = collection / f"{start:012d}.clock.json"
        raw = reader.read(raw_path, hashes[index])
        clock = reader.read(clock_path, parents.get(str(clock_path)))
        if parents.get(str(raw_path)) != reader.inputs[str(raw_path)] or clock != {
            "startSample": start,
            "endSample": window.end_sample,
            "sampleRate": 16000,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": reader.inputs[str(raw_path)],
        }:
            raise ReplayError("Original raw/clock identity differs")
        groups = mlx_groups(raw, duration=window.sample_count / 16000)
        group_index = int(owner.rsplit(":", 1)[1])
        if group_index >= len(groups):
            raise ReplayError("Original raw group is absent")
        group = groups[group_index]
        if (
            label.get("text") != group.text
            or label.get("start") != group.start + start / 16000
            or label.get("end") != group.end + start / 16000
            or token.text != group.text
        ):
            raise ReplayError("Candidate differs from its original native raw group")
        left = (windows[index - 1].end_sample + start) / 32000 if index else 0.0
        right = (
            (window.end_sample + windows[index + 1].start_sample) / 32000
            if index + 1 < len(windows)
            else window.end_sample / 16000
        )
        midpoint = (float(str(label["start"])) + float(str(label["end"]))) / 2
        if not left <= midpoint < right:
            raise ReplayError("Original label violates its actual midpoint ownership")
        proofs.append(
            {
                "owner": owner,
                "raw": {"path": str(raw_path), "sha256": reader.inputs[str(raw_path)]},
                "clock": {"path": str(clock_path), "sha256": reader.inputs[str(clock_path)]},
                "rawGroup": asdict(group),
                "windowStartSample": start,
                "windowEndSample": window.end_sample,
                "ownershipStart": left,
                "ownershipEnd": right,
                "originalMidpoint": midpoint,
            }
        )
    decision["ownershipProofs"] = proofs
    chosen = decision.get("chosenIndex")
    indices = decision.get("candidateIndices")
    if type(chosen) is not int or not isinstance(indices, list) or chosen not in indices:
        raise ReplayError("Invalid existing reconciliation selection")
    selected = record(_values(decision, "candidates")[indices.index(chosen)])
    decision["selectedCanonicalOwner"] = selected["utterance"]
    decision["selectedActualRow"] = selected
    decision["policy"] = "existing-longest-intact-native-row; ties-earliest-original-index"
    decision["semanticConfidenceRankingUsed"] = False
    decision["extentOrOwnerSpliced"] = False


def _artifacts(
    directory: Path, mapping: Path, collection: Path, labels: Path, output: Path
) -> tuple[dict[Path, bytes], list[object]]:
    _path(output)
    if not output.is_relative_to(Path.cwd() / "edits" / "work"):
        raise ReplayError("Reconciliation output requires checkout work storage")
    reader = Reader()
    for code in (
        Path(__file__).resolve(),
        Path(asr_assembly.__file__).resolve(),
        Path(caption_onset_reconcile.__file__).resolve(),
    ):
        reader.inputs[str(code)] = file_digest(code)
    original = reader.read(labels)
    source_labels = _values(original, "labels")
    owners = frozenset(str(record(row).get("provenance")) for row in source_labels)
    receipt = reader.read(directory / "receipt.json")
    _, index = reader.ref(receipt.get("segments"))
    entries = _values(index, "segments")
    if len(entries) > 512 or index.get("rate") != 30:
        raise ReplayError("Invalid original segment clock")
    _, config = reader.ref(receipt.get("config"))
    _, projected = reader.ref(config.get("projectedTokens"))
    _, source = reader.ref(projected.get("labels"))
    if record(source.get("parentSHA256")).get(str(labels)) != reader.inputs[str(labels)]:
        raise ReplayError("Projected recognition does not bind the original PCM labels")
    previous = reader.read(mapping)
    sources, mapping_pins = load_sources(mapping, entries)
    reader.inputs.update(mapping_pins)
    replacements = [dict(record(value)) for value in _values(previous, "replacements")]
    files: dict[Path, bytes] = {}
    audit: list[object] = []
    for entry in entries:
        row = record(entry)
        number = row.get("segment")
        if type(number) is not int:
            raise ReplayError("Invalid segment number")
        current = reader.read(segment_binding(directory, row, sources))
        bindings = record(current.get("inputs"))
        tokens, token_pins = read_tokens(directory, row, bindings, sources)
        reader.inputs.update(token_pins)
        values: list[object] = [asdict(token) for token in tokens]
        normalized, decisions = reconcile_tokens(values, known_owners=owners)
        audit.append(
            {
                "segment": number,
                "originalTokens": len(values),
                "retainedTokens": len(normalized),
                "decisions": decisions,
            }
        )
        if not decisions:
            continue
        _source_decisions(decisions, source_labels)
        for decision in decisions:
            ownership_proof(decision, reader=reader, collection=collection, labels=labels)
        removed: set[int] = set()
        for decision in decisions:
            indices = _values(decision, "candidateIndices")
            removed.update(int(str(index)) for index in indices if index != decision["chosenIndex"])
        position = 0
        for job in _values(row, "jobs"):
            if type(job) is not int:
                raise ReplayError("Invalid original job")
            path = directory / f"tokens-{job:04d}.json"
            for token_source in sources:
                if token_source.segment == number and token_source.job == job:
                    path = token_source.tokens
            document = reader.read(path, bindings.get(str(path)))
            members = _job_rows(document, "tokens")
            kept = [value for index, value in enumerate(members, position) if index not in removed]
            position += len(members)
            if kept == members:
                continue
            target = output / f"tokens-{number:04d}-{job:04d}.json"
            files[target] = encode({**document, "tokens": kept})
            manifest = output / f"adopted-{number:04d}-{job:04d}.json"
            files[manifest] = encode(
                {
                    "version": 1,
                    "tokens": _reference(target, files[target]),
                    "originalTokens": {"path": str(path), "sha256": reader.inputs[str(path)]},
                    "originalRows": members,
                    "decisions": decisions,
                    "inputs": dict(reader.inputs),
                    "timestampsChanged": False,
                    "qualityApproved": False,
                    "speechAccuracyVerified": False,
                }
            )
            replacement = {
                "segment": number,
                "job": job,
                "tokens": _reference(target, files[target]),
                "adoptedManifest": _reference(manifest, files[manifest]),
            }
            replacements = [
                value
                for value in replacements
                if (value.get("segment"), value.get("job")) != (number, job)
            ]
            replacements.append(replacement)
        baseline = segment_binding(directory, row)
        base = reader.read(baseline)
        additions = {str(baseline): reader.inputs[str(baseline)]}
        segment_members = [value for value in replacements if value.get("segment") == number]
        for value in segment_members:
            for key in ("tokens", "adoptedManifest"):
                ref = record(value[key])
                additions[str(ref["path"])] = str(ref["sha256"])
        combined = output / f"combined-{number:04d}.json"
        files[combined] = encode({**base, "inputs": {**record(base.get("inputs")), **additions}})
        for value in segment_members:
            value["combined"] = _reference(combined, files[combined])
            value["originalCombined"] = {
                "path": str(baseline),
                "sha256": reader.inputs[str(baseline)],
            }
    files[output / "token-sources.json"] = encode({"version": 1, "replacements": replacements})
    files[output / "audit.json"] = encode(
        {
            "version": 1,
            "inputs": reader.inputs,
            "segments": audit,
            "timestampsChanged": False,
            "speechAccuracyVerified": False,
        }
    )
    reader.verify()
    return files, entries


def verify(directory: Path, mapping: Path, collection: Path, labels: Path, output: Path) -> None:
    files, entries = _artifacts(directory, mapping, collection, labels, output)
    for path, content in files.items():
        if _bound_bytes(path, hashlib.sha256(content).hexdigest(), 64 << 20) != content:
            raise ReplayError("Cross-job reconciliation artifacts changed")
    sources, _ = load_sources(output / "token-sources.json", entries)
    for entry in entries:
        row = record(entry)
        combined = segment_binding(directory, row, sources)
        read_tokens(directory, row, record(Reader().read(combined).get("inputs")), sources)


def prepare(directory: Path, mapping: Path, collection: Path, labels: Path, output: Path) -> Path:
    if output.exists():
        raise ReplayError("Cross-job reconciliation requires a fresh output namespace")
    files, _ = _artifacts(directory, mapping, collection, labels, output)
    output.mkdir(mode=0o700)
    for path, content in files.items():
        publish(path, content)
    verify(directory, mapping, collection, labels, output)
    return output / "token-sources.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "mapping", "collection", "labels", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        function = verify if args.verify else prepare
        function(args.directory, args.mapping, args.collection, args.labels, args.output)
    except (OSError, ValueError, ReplayError) as error:
        print(f"CAPTION_CROSS_JOB_RECONCILE_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
