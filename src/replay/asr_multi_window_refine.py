"""Seal candidate labels from independently completed sample-exact CPU windows."""

import argparse
import json
import wave
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

from replay import asr_assembly, asr_label_refine, asr_window_refine, caption_io
from replay.asr_clock_projection import _binding, _labels
from replay.asr_label_refine import _read, replace_block
from replay.asr_window_refine import _alternative, _dependencies
from replay.caption_repair import ClockLabel
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.sequence_journal import encode, publish
from replay.stages import file_digest


def _refs(value: object) -> dict[str, tuple[Path, str]]:
    return {name: _binding(ref) for name, ref in record(value).items()}


def _plan(config: Path) -> dict[str, object]:
    if config != config.resolve(strict=True) or not config.is_relative_to(
        Path.cwd() / "edits" / "work"
    ):
        raise ValueError("Expected canonical checkout-work configuration")
    config_digest = file_digest(config)
    code = Path(__file__).resolve()
    code_digest = file_digest(code)
    code_bindings = [(code, code_digest)]
    for module in (asr_assembly, asr_label_refine, asr_window_refine, caption_io):
        source_name = module.__file__
        if not isinstance(source_name, str):
            raise ValueError("Missing shared refinement source identity")
        source_path = Path(source_name).resolve()
        code_bindings.append((source_path, file_digest(source_path)))
    snapshots = [InputHashSnapshot.capture([(config, config_digest), *code_bindings])]
    definition = _read(config)
    if (
        set(definition) != {"version", "artifacts", "windows", "replacement", "priorObservations"}
        or definition["version"] != 1
    ):
        raise ValueError("Expected explicit multi-window candidate refinement")
    bases = _refs(definition["artifacts"])
    if set(bases) != {"labels", "originalInputs"}:
        raise ValueError("Expected original unprojected labels and PCM manifest")
    bindings = [(config, config_digest), *code_bindings, *bases.values()]
    snapshots.append(InputHashSnapshot.capture(list(bases.values())))
    original, original_inputs = (_read(bases[name][0]) for name in ("labels", "originalInputs"))
    if (
        record(original.get("parentSHA256")).get(str(bases["originalInputs"][0]))
        != bases["originalInputs"][1]
    ):
        raise ValueError("Original labels lost their original collection binding")
    original_dependencies = _dependencies(original_inputs)
    bindings.extend(original_dependencies)
    snapshots.append(InputHashSnapshot.capture(original_dependencies))
    prior = definition["priorObservations"]
    windows = definition["windows"]
    if (
        not isinstance(prior, list)
        or len(prior) > 32
        or not isinstance(windows, list)
        or not 2 <= len(windows) <= 32
    ):
        raise ValueError("Expected bounded independent observations")
    prior_bindings = [_binding(ref) for ref in prior]
    bindings.extend(prior_bindings)
    if prior_bindings:
        snapshots.append(InputHashSnapshot.capture(prior_bindings))
    alternatives: list[ClockLabel] = []
    observations: list[object] = []
    intervals: list[tuple[int, int]] = []
    source: Path | None = None
    uncertain: list[object] = []
    for value in windows:
        refs = _refs(value)
        if set(refs) != {"inputs", "complete", "raw", "clock", "window", "audio"}:
            raise ValueError("Expected exact completed native CPU window bindings")
        bindings.extend(refs.values())
        snapshots.append(InputHashSnapshot.capture(list(refs.values())))
        paths = {name: ref[0] for name, ref in refs.items()} | {
            "originalInputs": bases["originalInputs"][0]
        }
        pins = {name: ref[1] for name, ref in refs.items()}
        inputs = _read(paths["inputs"])
        dependencies = _dependencies(inputs)
        bindings.extend(dependencies)
        snapshots.append(InputHashSnapshot.capture(dependencies))
        window, labels = _alternative(paths, pins, original_inputs, original)
        window_doc = _read(paths["window"])
        actual_source = window_doc.get("source")
        if not isinstance(actual_source, str) or (
            source is not None and source != Path(actual_source)
        ):
            raise ValueError("Independent windows must use one unchanged original PCM")
        source = Path(actual_source)
        intervals.append((window.start_sample, window.end_sample))
        alternatives.extend(labels)
        observations.append(
            {
                "artifacts": value,
                "nativeSampleOffset": window.start_sample,
                "offsetApplications": 1,
                "labels": [asdict(label) for label in labels],
            }
        )
        if labels and labels[0].start == window.start_sample / 16000:
            uncertain.append(
                {
                    "provenance": labels[0].provenance,
                    "start": labels[0].start,
                    "reason": (
                        "native first onset coincides with extraction boundary; "
                        "not reviewed speech onset"
                    ),
                }
            )
    if source is None or source != source.resolve(strict=True):
        raise ValueError("Expected canonical original PCM source")
    gaps: list[object] = []
    with wave.open(str(source), "rb") as audio:
        if (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
            audio.getcomptype(),
        ) != (1, 2, 16000, "NONE"):
            raise ValueError("Expected original mono PCM16/16kHz")
        for left, right in pairwise(intervals):
            if left[1] > right[0]:
                raise ValueError("Independent extraction windows overlap or reverse")
            count = right[0] - left[1]
            if count > 60 * 16000:
                raise ValueError("Unrecognized gap exceeds sample verification budget")
            if count:
                audio.setpos(left[1])
                payload = audio.readframes(count)
                if len(payload) != count * 2 or any(payload):
                    raise ValueError("Unrecognized gap is not exact original PCM digital zero")
                gaps.append(
                    {
                        "startSample": left[1],
                        "endSample": right[0],
                        "samples": count,
                        "exactDigitalZero": True,
                        "ASRObservedThisGap": False,
                    }
                )
    replacement = record(definition["replacement"])
    names = ("firstLabel", "stopLabel", "startSample", "endSample")
    if set(replacement) != set(names) or any(type(replacement[name]) is not int for name in names):
        raise ValueError("Expected exact whole-label replacement coordinates")
    first, stop, start, end = (int(str(replacement[name])) for name in names)
    if start != intervals[0][0] or end != intervals[-1][1] or end - start > 60 * 16000:
        raise ValueError("Replacement must bind the entire observed extraction envelope")
    old_labels = _labels(original)
    result, selected = replace_block(
        old_labels, tuple(alternatives), first=first, stop=stop, start_sample=start, end_sample=end
    )
    if any(
        label.start < start / 16000 or label.end > end / 16000 for label in old_labels[first:stop]
    ):
        raise ValueError("Replacement splits an original recognized label")
    for snapshot in snapshots:
        snapshot.finish()
    return {
        **original,
        "labels": [asdict(label) for label in result],
        "parentSHA256": {
            **record(original.get("parentSHA256")),
            **{str(path): digest for path, digest in bindings},
        },
        "refinementSHA256": code_digest,
        "refinement": {
            "kind": "independently-completed-cpu-windows-with-zero-gap-proof",
            "configSHA256": config_digest,
            "replacement": replacement,
            "removed": [asdict(label) for label in old_labels[first:stop]],
            "selected": [asdict(label) for label in selected],
            "observations": observations,
            "unrecognizedGaps": gaps,
            "uncertainOnsets": uncertain,
            "sourceProjectionApplications": 0,
            "nativeClocksAdjusted": False,
            "priorObservations": prior,
            "candidateOnly": True,
        },
        "onsetReversals": 0,
        "adjacentOverlaps": sum(b.start < a.end for a, b in pairwise(result)),
        "unresolvedPoints": sum(label.start == label.end for label in result),
        "candidateOnly": True,
        "humanReviewed": False,
        "speechAccuracyVerified": False,
        "deliveryComplete": False,
    }


def run(config: Path, output: Path) -> int:
    if (
        output.exists()
        or output.is_symlink()
        or output.parent != output.parent.resolve(strict=True)
        or not output.is_relative_to(Path.cwd() / "edits" / "work")
    ):
        raise ValueError("Expected a fresh candidate path under a canonical directory")
    document = _plan(config)
    publish(output, encode(document))
    return len(_labels(document))


def verify(config: Path, output: Path) -> int:
    digest = file_digest(output)
    snapshot = InputHashSnapshot.capture([(output, digest)])
    actual, expected = _read(output), _plan(config)
    if actual != expected:
        raise ValueError("Multi-window candidate differs from its actual pinned observations")
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
