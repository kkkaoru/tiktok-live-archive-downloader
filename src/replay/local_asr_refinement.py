"""Seal unreviewed label candidates from genuine completed local ASR windows.

The local producer retains its own input/raw/validation contract. This adapter
never constructs CPU batch manifests, changes recognized clocks or runs a model.
Changed historical producer code is refused by the existing adoption verifier;
captured history is retained, never relabelled as a current-code generation.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from replay.asr_adoption import verify as verify_adoption
from replay.asr_assembly import _labels as native_labels
from replay.asr_clock_projection import _binding, _labels
from replay.asr_label_refine import replace_block
from replay.asr_window_refine import _dependencies, _pcm, derive_replacement
from replay.code_dependency_inventory import capture
from replay.input_hash_snapshot import InputHashSnapshot
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_MODEL, DEFAULT_WEIGHTS_SHA256
from replay.native_invocation import read, require
from replay.recordings import record
from replay.stages import file_digest
from replay.workflow import load_stages

ARTIFACTS = frozenset(
    {
        "labels",
        "originalInputs",
        "window",
        "audio",
        "inputs",
        "raw",
        "validation",
        "plan",
        "checkpoint",
        "status",
        "end",
        "sourceBefore",
        "sourceAfter",
        "pins",
    }
)
MAX_CONFIG_BYTES = 64 * 1024
MAX_STATUS_BYTES = 32
RATE = 16000


@dataclass(frozen=True, slots=True)
class Refinement:
    """Validated selection and explicitly pinned external artifacts."""

    stage: str
    target: str
    bindings: dict[str, tuple[Path, str]]


def _config(path: Path) -> Refinement:
    require(path.stat().st_size <= MAX_CONFIG_BYTES, "Oversized local refinement config")
    data = read(path)
    require(
        set(data) == {"version", "artifacts", "stage", "targetProvenance"}
        and type(data["version"]) is int
        and data["version"] == 1,
        "Unsupported local refinement configuration",
    )
    stage, target = data["stage"], data["targetProvenance"]
    require(isinstance(stage, str) and bool(stage), "Missing local ASR stage")
    require(isinstance(target, str) and bool(target), "Missing target provenance")
    if not isinstance(stage, str) or not isinstance(target, str):
        raise ValueError("Expected textual selection")
    artifacts = record(data["artifacts"])
    require(set(artifacts) == ARTIFACTS, "Missing genuine local ASR artifacts")
    return Refinement(stage, target, {name: _binding(value) for name, value in artifacts.items()})


def _source_history(paths: dict[str, Path]) -> dict[str, object]:
    before, after = read(paths["sourceBefore"]), read(paths["sourceAfter"])
    require(
        bool(before) and after.get("hashes") == before and after.get("differences") == {},
        "Historical producer sources changed during recognition",
    )
    for name, digest in before.items():
        require(isinstance(digest, str), "Invalid historical source pin")
        require(
            file_digest(Path(name)) == digest, "Historical source differs from current producer"
        )
    return before


def _producer(selection: Refinement, paths: dict[str, Path]) -> dict[str, object]:
    stage = next((s for s in load_stages(paths["plan"]) if s.name == selection.stage), None)
    require(stage is not None, "Missing completed local ASR stage")
    if stage is None:
        raise ValueError("Missing producer")
    require(
        "replay.local_asr" in stage.command
        and "--device" in stage.command
        and stage.command.index("--device") + 1 < len(stage.command)
        and stage.command[stage.command.index("--device") + 1] in {"auto", "cpu"},
        "Expected an explicit existing local ASR device",
    )
    require(
        paths["checkpoint"].name == f"{selection.stage}.json"
        and paths["inputs"].parent == paths["raw"].parent == paths["validation"].parent
        and paths["raw"].name == "raw-transcript.json"
        and paths["inputs"].name == "inputs.json"
        and paths["validation"].name == "validation.json"
        and {paths[name] for name in ("inputs", "raw", "validation")} <= set(stage.outputs),
        "Local raw artifacts differ from their actual producer",
    )
    with paths["status"].open("rb") as stream:
        status = stream.read(MAX_STATUS_BYTES + 1)
    require(status == b"0\n", "Local ASR job did not finish successfully")
    end = read(paths["end"])
    require(
        type(end.get("exitCode")) is int
        and end.get("exitCode") == 0
        and end.get("completed") is True
        and end.get("reservationReleased") is True,
        "Local ASR admission lacks a completed terminal envelope",
    )
    checkpoint = read(paths["checkpoint"])
    require(checkpoint.get("status") == "complete", "Local ASR stage is not complete")
    outputs = record(checkpoint.get("outputs"))
    require(
        set(outputs) == {str(path) for path in stage.outputs}
        and all(
            outputs.get(str(paths[name])) == selection.bindings[name][1]
            for name in ("inputs", "raw", "validation")
        ),
        "Local ASR raw outputs differ from their complete stage receipt",
    )
    before = _source_history(paths)
    require(
        {Path(name) for name in before} <= set(stage.inputs),
        "Historical source inventory is outside the actual producer",
    )
    verify_adoption(
        paths["plan"],
        paths["checkpoint"].parent,
        selection.stage,
        paths["inputs"].parent,
        paths["pins"],
    )
    return before


def plan(config: Path) -> dict[str, object]:
    """Purely rederive a candidate after validating actual generation and PCM."""
    selection = _config(config)
    paths = {name: binding[0] for name, binding in selection.bindings.items()}
    pins = {name: binding[1] for name, binding in selection.bindings.items()}
    original, original_inputs, inputs = (
        read(paths[name]) for name in ("labels", "originalInputs", "inputs")
    )
    model = DEFAULT_MODEL.resolve(strict=True)
    require(
        inputs.get("model") == str(model)
        and inputs.get("source") == str(paths["audio"])
        and inputs.get("config_sha256") == DEFAULT_CONFIG_SHA256
        and inputs.get("weights_sha256") == DEFAULT_WEIGHTS_SHA256,
        "Local ASR must retain the pinned non-turbo model and exact WAV",
    )
    code = {str(source.path): source.sha256 for source in capture(("local_asr_refinement",))}
    dependencies = _dependencies(original_inputs)
    snapshot = InputHashSnapshot.capture(
        [
            (config, file_digest(config)),
            *selection.bindings.values(),
            *dependencies,
            (model / "config.json", DEFAULT_CONFIG_SHA256),
            (model / "weights.npz", DEFAULT_WEIGHTS_SHA256),
            *((Path(name), digest) for name, digest in code.items()),
        ]
    )
    require(
        record(original.get("parentSHA256")).get(str(paths["originalInputs"]))
        == pins["originalInputs"]
        and original.get("humanReviewed") is False
        and original.get("speechAccuracyVerified") is False
        and original.get("deliveryComplete") is False,
        "Original labels must bind an unreviewed unprojected assembly",
    )
    historical_sources = _producer(selection, paths)
    window = _pcm(read(paths["window"]), paths["audio"], original_inputs)
    alternative = tuple(
        native_labels(
            read(paths["raw"]),
            window=window,
            owner_start=window.start_sample / RATE,
            owner_end=window.end_sample / RATE,
        )
    )
    labels = _labels(original)
    interval = derive_replacement(
        labels, alternative, window=window, target_provenance=selection.target
    )
    first, stop = interval["firstLabel"], interval["stopLabel"]
    result, selected = replace_block(
        labels,
        alternative,
        first=first,
        stop=stop,
        start_sample=interval["startSample"],
        end_sample=interval["endSample"],
    )
    snapshot.finish()
    return {
        **original,
        "labels": [asdict(label) for label in result],
        "parentSHA256": {
            **record(original.get("parentSHA256")),
            **{str(path): digest for path, digest in selection.bindings.values()},
            **code,
        },
        "refinementSHA256": file_digest(Path(__file__)),
        "refinement": {
            "configSHA256": file_digest(config),
            "replacement": interval,
            "targetProvenance": selection.target,
            "recognitionWindowKind": "genuine-local-asr",
            "extractionOffsetSample": window.start_sample,
            "removed": [asdict(label) for label in labels[first:stop]],
            "selected": [asdict(label) for label in selected],
            "historicalProducerSourceSHA256": historical_sources,
            "historicalProducerStageFingerprint": read(paths["checkpoint"])["fingerprint"],
            "clockOffsetApplications": 1,
            "sourceProjectionApplications": 0,
            "candidateOnly": True,
        },
        "onsetReversals": 0,
        "adjacentOverlaps": sum(b.start < a.end for a, b in pairwise(result)),
        "unresolvedPoints": sum(label.start == label.end for label in result),
        "humanReviewed": False,
        "speechAccuracyVerified": False,
        "deliveryComplete": False,
    }


def run(config: Path, output: Path) -> int:
    """Publish one new candidate, retaining every original external label."""
    require(
        not output.exists()
        and not output.is_symlink()
        and output.parent == output.parent.resolve(strict=True),
        "Expected a new candidate in an existing canonical directory",
    )
    document = plan(config)
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return len(_labels(document))


def verify(config: Path, output: Path) -> int:
    """Require full candidate equality, without semantic or delivery approval."""
    snapshot = InputHashSnapshot.capture([(output, file_digest(output))])
    actual = read(output)
    require(actual == plan(config), "Local refinement candidate differs from its pinned plan")
    snapshot.finish()
    return len(_labels(actual))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    count = verify(args.config, args.output) if args.verify else run(args.config, args.output)
    print(json.dumps({"labels": count, "candidateOnly": True, "speechAccuracyVerified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
