"""Prepare recognition-only inputs for native paging without invented measurements."""

import argparse
from pathlib import Path

from replay import delivery_captions
from replay.artifact_json import _json, _pin
from replay.caption_token_sources import TokenSource, combined_path
from replay.core import ReplayError
from replay.delivery_captions import (
    _local_token,
    _write_new,
    partition,
    read_cuts,
    read_tokens,
)
from replay.fcp_part_assembly import _reference
from replay.recordings import record
from replay.stages import file_digest

KIND = "recognition-inputs"
MAX_RECOGNITION_TOKENS = 4096


def segment_binding(
    directory: Path, row: dict[str, object], sources: tuple[TokenSource, ...] = ()
) -> Path:
    """Resolve either an explicit fresh recognition input or a historical combine."""
    number = row.get("segment")
    if type(number) is not int or number < 0:
        raise ReplayError("Invalid recognition segment")
    reference = row.get("recognitionInput")
    path = directory / f"combined-{number:04d}.json"
    if reference is not None:
        path = _pin(reference)
        document = _json(path)
        if (
            document.get("kind") != KIND
            or document.get("version") != 1
            or document.get("segment") != number
            or document.get("measurementsPerformed") is not False
            or document.get("jobs") != row.get("jobs")
        ):
            raise ReplayError("Fresh recognition binding differs from segment index")
    return combined_path(path, number, sources)


def recognition_pins(
    document: dict[str, object], *, source: str, source_digest: str
) -> dict[str, str]:
    """Verify fresh provenance while reusing the caller's already hashed source."""
    if document.get("kind") != KIND:
        return {}
    inputs = record(document.get("inputs"))
    if inputs.get(source) != source_digest:
        raise ReplayError("Fresh recognition input source differs")
    result: dict[str, str] = {}
    for name, digest in inputs.items():
        if not isinstance(digest, str):
            raise ReplayError("Invalid fresh recognition input hash")
        if name == source:
            continue
        _pin({"path": name, "sha256": digest})
        result[name] = digest
    return result


def prepare(config_path: Path, output: Path) -> Path:
    config_path = config_path.resolve()
    config = _json(config_path)
    if config.get("version") != 1 or output.exists() or output != output.resolve():
        raise ReplayError("Recognition preparation requires version1 and a new canonical directory")
    paths = {name: _pin(config.get(name)) for name in ("projectedTokens", "cuts", "source")}
    projected = _json(paths["projectedTokens"])
    if (
        projected.get("version") != 1
        or projected.get("clock") != "output"
        or projected.get("cuts") != _reference(paths["cuts"])
        or projected.get("boundaryConflicts") != []
    ):
        raise ReplayError("Expected conflict-free projected recognition on the pinned cuts")
    labels = _pin(projected.get("labels"))
    tokens = read_tokens(paths["projectedTokens"])
    cuts = read_cuts(paths["cuts"])
    segments = partition(
        cuts,
        tokens,
        (),
        rate=30,
        require_foreground=False,
        maximum_job_tokens=MAX_RECOGNITION_TOKENS,
    )
    if not 1 <= len(segments) <= 512:
        raise ReplayError("Recognition segment budget exceeded")
    output.mkdir(mode=0o700)
    pins = {str(path): file_digest(path) for path in (*paths.values(), labels, config_path)}
    pins[str(Path(__file__).resolve())] = file_digest(Path(__file__))
    helper = Path(delivery_captions.__file__).resolve()
    pins[str(helper)] = file_digest(helper)
    index: list[dict[str, object]] = []
    for segment in segments:
        if sum(len(job.tokens) for job in segment.jobs) > MAX_RECOGNITION_TOKENS:
            raise ReplayError("Recognition segment token budget exceeded")
        inputs = dict(pins)
        for job in segment.jobs:
            path = output / f"tokens-{job.index:04d}.json"
            _write_new(
                path,
                {
                    "version": 1,
                    "clock": "output",
                    "tokens": [_local_token(token, segment.offset / 30) for token in job.tokens],
                },
            )
            inputs[str(path)] = file_digest(path)
        jobs = [job.index for job in segment.jobs]
        binding = output / f"recognition-{segment.index:04d}.json"
        _write_new(
            binding,
            {
                "version": 1,
                "kind": KIND,
                "segment": segment.index,
                "jobs": jobs,
                "inputs": inputs,
                "measurementsPerformed": False,
                "recognitionClockChanged": False,
            },
        )
        index.append(
            {
                "segment": segment.index,
                "offsetFrame": segment.offset,
                "frames": segment.frames,
                "jobs": jobs,
                "recognitionInput": _reference(binding),
            }
        )
    _write_new(output / "segments.json", {"version": 1, "rate": 30, "segments": index})
    _write_new(
        output / "receipt.json",
        {
            "version": 1,
            "candidateOnly": True,
            "config": _reference(config_path),
            "segments": _reference(output / "segments.json"),
            "inputs": pins,
            "positiveTokens": len(tokens),
            "measurementsPerformed": False,
            "zeroLengthLabels": projected.get("zeroLengthLabels"),
            "omitted": projected.get("omitted"),
            "recognitionClockChanged": False,
        },
    )
    verify(output)
    return output / "segments.json"


def verify(output: Path) -> None:
    receipt = _json(output / "receipt.json")
    index = _json(_pin(receipt.get("segments")))
    for name, digest in record(receipt.get("inputs")).items():
        _pin({"path": name, "sha256": digest})
    entries = index.get("segments")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 512:
        raise ReplayError("Invalid recognition segment index")
    for raw in entries:
        row = record(raw)
        path = segment_binding(output, row)
        for name, digest in record(_json(path).get("inputs")).items():
            # Media and shared inputs were checked once above, not once per segment.
            if name not in record(receipt.get("inputs")):
                _pin({"path": name, "sha256": digest})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if _json(args.output / "receipt.json").get("config") != _reference(args.config):
            raise ReplayError("Recognition preparation config changed")
        verify(args.output)
    else:
        print(prepare(args.config, args.output))


if __name__ == "__main__":
    main()
