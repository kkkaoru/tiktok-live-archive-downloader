"""Bind completed future artifacts to existing caption consumer schemas."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from replay import cut_audio_support, delivery_captions
from replay.audio_clock import AudioFrameClock
from replay.caption_projection import CutSpan, project_tokens
from replay.continuous_quiet_plan import _labels
from replay.core import ReplayError
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.stages import file_digest

MAX_JSON_BYTES = 128 * 1024 * 1024
ROLES = {
    "cut-audio-support": frozenset({"cuts", "labels", "audioClock"}),
    "native-inputs": frozenset({"projectedTokens", "cuts", "source"}),
}


def _path(value: object) -> Path:
    if not isinstance(value, str):
        raise ReplayError("Expected explicit artifact path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=True) or not path.is_file():
        raise ReplayError("Expected canonical regular artifact")
    return path


def _read(path: Path) -> dict[str, object]:
    if not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ReplayError("Consumer binding JSON exceeds its byte budget")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def _document(template: Path) -> tuple[dict[str, object], dict[Path, str]]:
    data = _read(template)
    kind = data.get("kind")
    if (
        set(data) != {"version", "kind", "artifactPaths", "expectedInputs"}
        or data.get("version") != 1
        or not isinstance(kind, str)
        or kind not in ROLES
    ):
        raise ReplayError("Unknown consumer binding schema")
    roles = record(data.get("artifactPaths"))
    expected = record(data.get("expectedInputs"))
    required_fixed = {"audioClock"} if kind == "cut-audio-support" else {"source"}
    if set(roles) != ROLES[kind] or not required_fixed <= set(expected) <= set(roles):
        raise ReplayError("Consumer roles or fixed identity constraints differ")
    paths = {name: _path(value) for name, value in roles.items()}
    refs = {name: _ref(path) for name, path in paths.items()}
    if any(refs[name] != record(value) for name, value in expected.items()):
        raise ReplayError("Consumer expected source/clock identity changed")
    pins = {path: refs[name]["sha256"] for name, path in paths.items()}
    pins[template] = file_digest(template)
    pins[Path(__file__).resolve()] = file_digest(Path(__file__))
    cuts = _read(paths["cuts"])
    if cuts.get("version") != 1 or cuts.get("clock") != "source-presentation":
        raise ReplayError("Consumer cuts must use source presentation clock")
    if kind == "cut-audio-support":
        labels = _read(paths["labels"])
        AudioFrameClock.from_probe(_read(paths["audioClock"]))
        if (
            labels.get("clock") != "source-presentation"
            or record(labels.get("parentSHA256")).get(str(paths["audioClock"]))
            != refs["audioClock"]["sha256"]
            or record(cuts.get("parentSHA256")).get(str(paths["audioClock"]))
            != refs["audioClock"]["sha256"]
            or record(cuts.get("parentSHA256")).get(str(paths["labels"]))
            != refs["labels"]["sha256"]
        ):
            raise ReplayError("Consumer labels/cuts do not share the pinned clock")
        return {"version": 1, "artifacts": refs}, pins
    projected = _read(paths["projectedTokens"])
    if (
        projected.get("version") != 1
        or projected.get("clock") != "output"
        or projected.get("cuts") != refs["cuts"]
        or projected.get("boundaryConflicts") != []
    ):
        raise ReplayError("Consumer requires conflict-free recognition on these exact cuts")
    _projected(projected, paths["cuts"], pins)
    return {"version": 1, **refs}, pins


def _projected(document: dict[str, object], cuts: Path, pins: dict[Path, str]) -> None:
    label_ref = record(document.get("labels"))
    labels = _path(label_ref.get("path"))
    if label_ref != _ref(labels) or _read(labels).get("clock") != "source-presentation":
        raise ReplayError("Projected recognition label identity/clock changed")
    tokens, points = delivery_captions.read_labels(labels)
    spans = tuple(
        CutSpan(cut.source_start, cut.source_end, cut.output_start)
        for cut in delivery_captions.read_cuts(cuts)
    )
    projected = project_tokens(tokens, spans, fps=30)
    if (
        document.get("tokens") != [delivery_captions._spoken(token) for token in projected.tokens]
        or document.get("omitted")
        != [delivery_captions._spoken(token) for token in projected.omitted]
        or document.get("boundaryConflicts")
        != [delivery_captions._spoken(token) for token in projected.boundary_conflicts]
        or document.get("zeroLengthLabels") != points
    ):
        raise ReplayError("Projected recognition text/clock differs from source labels")
    pins[labels] = file_digest(labels)


def verify_support(config: Path, output: Path) -> None:
    """Check existing audio-support output through its pure intersection function."""
    config, output = _path(str(config.absolute())), _path(str(output.absolute()))
    data, actual = _read(config), _read(output)
    raw = record(data.get("artifacts"))
    if data.get("version") != 1 or set(raw) != ROLES["cut-audio-support"]:
        raise ReplayError("Invalid audio-support result configuration")
    paths = {name: _path(record(value).get("path")) for name, value in raw.items()}
    if any(record(raw[name]) != _ref(path) for name, path in paths.items()):
        raise ReplayError("Audio-support result inputs changed")
    cuts, labels = _read(paths["cuts"]), _read(paths["labels"])
    fps = cuts.get("fps")
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ReplayError("Invalid audio-support frame rate")
    result = cut_audio_support.intersect(
        cut_audio_support._cuts(cuts),
        _labels(labels),
        AudioFrameClock.from_probe(_read(paths["audioClock"])),
        fps=fps,
    )
    parents = record(actual.get("parentSHA256"))
    original_parents = record(cuts.get("parentSHA256"))
    required = {**original_parents, str(config): file_digest(config)}
    required.update({str(path): file_digest(path) for path in paths.values()})
    producer = Path(cut_audio_support.__file__).resolve()
    required.update({str(producer): file_digest(producer)})
    if not all(isinstance(digest, str) for digest in parents.values()):
        raise ReplayError("Invalid audio-support result pins")
    if any(parents.get(name) != digest for name, digest in required.items()):
        raise ReplayError("Audio-support result lineage changed")
    snapshot = InputHashSnapshot.capture(
        (_path(name), digest) for name, digest in parents.items() if isinstance(digest, str)
    )
    expected = dict(cuts)
    expected.update(
        {
            "cuts": [asdict(cut) for cut in result.cuts],
            "parentSHA256": parents,
            "audioSupport": {
                "policy": "outer-envelope-whole-frames-no-recognition-removal",
                "firstFrame": result.first_frame,
                "endFrame": result.end_frame,
                "removedFrames": [list(span) for span in result.removed],
                "recognizedSpeechRemoved": False,
                "internalGapsVerified": False,
            },
            "nonSpeechRemovalVerified": False,
            "humanReviewed": False,
            "deliveryComplete": False,
        }
    )
    if actual != expected:
        raise ReplayError("Audio-support result differs from original clock intersection")
    snapshot.finish()


def _output(path: Path) -> Path:
    path = path.absolute()
    if (
        path != path.resolve()
        or not path.is_relative_to(Path.cwd() / "edits/work")
        or not path.parent.is_dir()
    ):
        raise ReplayError("Expected canonical repository work output")
    return path


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def prepare(template: Path, output: Path) -> None:
    """Publish only existing consumer configuration, plus binding provenance."""
    template, output = _path(str(template.absolute())), _output(output)
    receipt = output.with_suffix(".binding.json")
    if output.exists() or receipt.exists() or output == receipt:
        raise ReplayError("Consumer binding outputs must be new")
    document, pins = _document(template)
    snapshot = InputHashSnapshot.capture(pins.items())
    snapshot.finish()
    _write(output, document)
    _write(
        receipt,
        {
            "version": 1,
            "template": _ref(template),
            "output": _ref(output),
            "inputs": {str(path): digest for path, digest in pins.items()},
            "measurementsPerformed": False,
            "deliveryComplete": False,
        },
    )
    verify(template, output)


def verify(template: Path, output: Path) -> None:
    """Read-only current identity and existing-schema equivalence check."""
    template, output = _path(str(template.absolute())), _path(str(output.absolute()))
    expected, pins = _document(template)
    receipt = _read(output.with_suffix(".binding.json"))
    snapshot = InputHashSnapshot.capture(pins.items())
    if (
        _read(output) != expected
        or receipt.get("version") != 1
        or receipt.get("template") != _ref(template)
        or receipt.get("output") != _ref(output)
        or receipt.get("inputs") != {str(path): digest for path, digest in pins.items()}
        or receipt.get("measurementsPerformed") is not False
        or receipt.get("deliveryComplete") is not False
    ):
        raise ReplayError("Consumer binding provenance changed")
    snapshot.finish()


def verify_projection(template: Path, output: Path) -> None:
    """Validate a completed projection before the native config exists."""
    template, output = _path(str(template.absolute())), _path(str(output.absolute()))
    document, pins = _document(template)
    if document.get("projectedTokens") != _ref(output):
        raise ReplayError("Projection verifier output differs from native template")
    snapshot = InputHashSnapshot.capture(pins.items())
    snapshot.finish()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("prepare", "verify", "verify-support", "verify-projection")
    )
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    operation = {
        "prepare": prepare,
        "verify": verify,
        "verify-support": verify_support,
        "verify-projection": verify_projection,
    }[args.operation]
    operation(args.template, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
