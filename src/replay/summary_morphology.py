"""Pinned Japanese morphology for exact canonical summary previews.

The frozen preview consumer supplies the text and section clocks. Vibrato only
adds POS evidence and Swift Character ranges; it never invents a speech clock.
"""

import argparse
import hashlib
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import regex
from vibrato import Vibrato

from replay import caption_linguistics as linguistics
from replay import summary_caption_inputs as canonical
from replay.caption_measured_job import _json as strict_json
from replay.core import ReplayError
from replay.delivery_captions import _write_new
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
MAX_PREVIEW_BYTES = 4 << 20
MAX_TOKEN_BYTES = 128 << 20
MAX_INVENTORY_BYTES = 16 << 20
MAX_SECTION_CODEPOINTS = 200000
MAX_SECTION_TOKENS = 100000
MAX_SECTIONS = 512
MAX_SECTION_CLOCK = 2 * canonical.MAX_FRAMES + 600
PREVIEW_TIMEOUT_SECONDS = 120
VIBRATO_VERSION = "0.2.3"
PARTS = frozenset(
    {
        "名詞",
        "動詞",
        "形容詞",
        "助詞",
        "助動詞",
        "記号",
        "感動詞",
        "副詞",
        "接続詞",
        "連体詞",
        "接頭詞",
        "フィラー",
        "その他",
    }
)
CONFIG_FIELDS = frozenset(
    {
        "version",
        "preview",
        "canonicalManifest",
        "canonicalConfig",
        "previewExecutable",
        "dictionary",
    }
)
CODE_FILES = (
    "summary_morphology.py",
    "caption_linguistics.py",
    "caption_measured_job.py",
    "foreground_evidence.py",
    "recordings.py",
    "stages.py",
    "input_hash_cache.py",
    "delivery_captions.py",
    "tracking_observations.py",
    "caption_io.py",
    "caption_timing.py",
    "core.py",
    *canonical.CODE_FILES,
)


@dataclass(frozen=True)
class Inputs:
    preview: bytes
    sections: tuple[dict[str, object], ...]
    references: dict[str, object]
    pins: dict[str, str]
    executable: Path
    manifest: Path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ReplayError("Morphology requires explicit canonical input paths")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve()
        or not path.is_relative_to(ROOT)
        or not path.is_file()
        or any(part in {"private", "downloads"} for part in path.parts)
    ):
        raise ReplayError("Morphology input must be a regular canonical checkout file")
    return path


def _pin(value: object, pins: dict[str, str]) -> Path:
    row = record(value)
    if set(row) != {"path", "sha256"}:
        raise ReplayError("Morphology input requires an exact path and SHA256 pin")
    path = _path(row["path"])
    digest = row["sha256"]
    if not isinstance(digest, str) or not regex.fullmatch("[0-9a-f]{64}", digest):
        raise ReplayError("Morphology input SHA256 is invalid")
    if file_digest(path) != digest:
        raise ReplayError("Pinned morphology input changed")
    pins[str(path)] = digest
    return path


def _clock(value: object) -> float:
    if type(value) not in (int, float) or not isinstance(value, int | float):
        raise ReplayError("Summary section clocks must be finite numbers")
    if value < 0 or value > MAX_SECTION_CLOCK or not math.isfinite(value):
        raise ReplayError("Summary section clocks must be finite nonnegative numbers")
    return float(value)


def _sections(data: bytes) -> tuple[dict[str, object], ...]:
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise ReplayError("Exact preview must retain its single terminal LF")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReplayError("Exact preview must be UTF8") from error
    raw = record(strict_json(data))
    values = raw.get("sections")
    if (
        set(raw) != {"sections"}
        or not isinstance(values, list)
        or not 1 <= len(values) <= MAX_SECTIONS
    ):
        raise ReplayError("Expected a bounded exact source summary preview")
    result: list[dict[str, object]] = []
    previous = -1.0
    for value in values:
        row = record(value)
        if set(row) != {"startSeconds", "endSeconds", "text"}:
            raise ReplayError("Summary section must preserve exact clocks and text")
        start, end = _clock(row["startSeconds"]), _clock(row["endSeconds"])
        text = row["text"]
        if (
            start <= previous
            or end <= start
            or not isinstance(text, str)
            or not text.strip()
            or "\0" in text
            or len(text) > MAX_SECTION_CODEPOINTS
        ):
            raise ReplayError("Summary section text or clock exceeds its contract")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ReplayError("Summary section text must be valid UTF8") from error
        result.append(row)
        previous = start
    return tuple(result)


def _manifest_matches(raw: dict[str, object], expected: dict[str, object]) -> None:
    if "version" in raw and type(raw["version"]) is not int:
        raise ReplayError("Canonical morphology manifest requires an integer version")
    jobs = raw.get("jobs")
    if isinstance(jobs, list) and any(type(record(job).get("job")) is not int for job in jobs):
        raise ReplayError("Canonical morphology jobs require integer membership")
    selected = raw.get("sectionStarts")
    comparison = {key: value for key, value in raw.items() if key != "sectionStarts"}
    for flag in ("candidateOnly", "recognitionAccuracyVerified", "renderedDeliveryVerified"):
        if flag in comparison and type(comparison[flag]) is not bool:
            raise ReplayError("Canonical morphology approval flags must be booleans")
        if flag in expected and flag not in comparison:
            comparison[flag] = expected[flag]
    if comparison != expected:
        raise ReplayError("Canonical morphology manifest differs from its selected source config")
    if "sectionStarts" in raw and (
        not isinstance(selected, list)
        or not selected
        or any(type(item) is not int or item < 0 or item % 600 for item in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ReplayError("Canonical section selection requires unique ten-minute start clocks")


def load(config: Path) -> Inputs:
    """Validate scope before any content read and bind all canonical source tokens."""
    config = _path(str(config))
    pins = {str(config): file_digest(config)}
    raw = record(strict_json(_bound_bytes(config, pins[str(config)], MAX_PREVIEW_BYTES)))
    if set(raw) != CONFIG_FIELDS or type(raw.get("version")) is not int or raw["version"] != 1:
        raise ReplayError("Expected version1 canonical morphology configuration")
    if version("vibrato") != VIBRATO_VERSION:
        raise ReplayError("Morphology requires the locked Vibrato package version")
    # Validate every selected path before reading any of their document bytes.
    for key in CONFIG_FIELDS - {"version"}:
        _path(record(raw[key]).get("path"))
    paths = {key: _pin(raw[key], pins) for key in CONFIG_FIELDS - {"version"}}
    dictionary = (ROOT / linguistics.DICTIONARY_PATH).resolve()
    if paths["dictionary"] != dictionary or pins[str(dictionary)] != linguistics.DICTIONARY_SHA256:
        raise ReplayError("Morphology requires the unchanged pinned IPADIC dictionary")
    _bound_bytes(dictionary, pins[str(dictionary)], linguistics.MAX_DICTIONARY_BYTES)
    loaded = canonical.load(paths["canonicalConfig"])
    manifest = record(
        strict_json(
            _bound_bytes(
                paths["canonicalManifest"],
                pins[str(paths["canonicalManifest"])],
                canonical.MAX_MANIFEST_BYTES,
            )
        )
    )
    _manifest_matches(manifest, loaded.manifest)
    pins.update(loaded.direct | loaded.tokens)
    if sum(Path(name).stat().st_size for name in loaded.tokens) > MAX_TOKEN_BYTES:
        raise ReplayError("Canonical token bytes exceed the bounded preview budget")
    data = _bound_bytes(paths["preview"], pins[str(paths["preview"])], MAX_PREVIEW_BYTES)
    sections = _sections(data)
    code = {
        str(Path(__file__).with_name(name)): file_digest(Path(__file__).with_name(name))
        for name in CODE_FILES
    }
    pins.update(code)
    for name in (".python-version", "uv.lock", "pyproject.toml"):
        path = ROOT / name
        pins[str(_path(str(path)))] = file_digest(path)
    return Inputs(
        data,
        sections,
        {key: raw[key] for key in paths},
        pins,
        paths["previewExecutable"],
        paths["canonicalManifest"],
    )


def _verify_preview(inputs: Inputs, directory: Path) -> None:
    # Capturing to an explicit work file avoids Foundation/stdio scratch defaults.
    # Total pinned token JSON is bounded before execution; read output is bounded too.
    capture = directory / "preview-check.log"
    environment = {
        **os.environ,
        "TMPDIR": str(directory),
        "TMP": str(directory),
        "TEMP": str(directory),
        "LLVM_PROFILE_FILE": str(directory / "preview-%p-%m.profraw"),
    }
    with capture.open("xb") as stream, (directory / "preview-errors.log").open("xb") as errors:
        subprocess.run(
            [str(inputs.executable), "preview", "--caption-tokens", str(inputs.manifest)],
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=errors,
            timeout=PREVIEW_TIMEOUT_SECONDS,
            check=True,
        )
    observed = _bound_bytes(capture, file_digest(capture), MAX_PREVIEW_BYTES)
    if observed != inputs.preview:
        raise ReplayError("Pinned preview differs from the exact canonical source consumer")


def tokens(text: str, *, tokenizer: Vibrato | None = None) -> list[dict[str, object]]:
    """Retain every tokenizer surface and reject codepoint splits inside a grapheme."""
    if not text or "\0" in text or len(text) > MAX_SECTION_CODEPOINTS:
        raise ReplayError("Morphology omitted source text or exceeded its bound")
    boundaries = {0: 0}
    for index, match in enumerate(regex.finditer(r"\X", text), 1):
        boundaries[match.end()] = index
    result: list[dict[str, object]] = []
    previous = 0
    tokenizer = (
        tokenizer if tokenizer is not None else linguistics._load_tokenizer(max_grouping_len=1)
    )
    for token in tokenizer.tokenize(text):
        start, end, surface = token.start(), token.end(), token.surface()
        features = token.feature().split(",")
        if (
            start != previous
            or end <= start
            or end > len(text)
            or surface != text[start:end]
            or start not in boundaries
            or end not in boundaries
            or not 2 <= len(features) <= 16
            or features[0] not in PARTS
            or any(len(feature.encode("utf-8")) > 16384 for feature in features)
            or len(result) >= MAX_SECTION_TOKENS
        ):
            raise ReplayError("Morphology must preserve complete valid source/grapheme coverage")
        result.append(
            {
                "text": surface,
                "lowerCharacter": boundaries[start],
                "upperCharacter": boundaries[end],
                "partOfSpeech": features[0],
                "features": features,
            }
        )
        previous = end
    if not result or previous != len(text):
        raise ReplayError("Morphology omitted source text or whitespace")
    return result


def _document(inputs: Inputs) -> dict[str, object]:
    sections: list[dict[str, object]] = []
    tokenizer = linguistics._load_tokenizer(max_grouping_len=1)
    for section in inputs.sections:
        text = section["text"]
        if not isinstance(text, str):
            raise ReplayError("Expected validated summary text")
        sections.append(
            {
                "startSeconds": section["startSeconds"],
                "endSeconds": section["endSeconds"],
                "textSHA256": _sha(text.encode("utf-8")),
                "tokens": tokens(text, tokenizer=tokenizer),
            }
        )
    return {
        "version": 1,
        "method": "vibrato-ipadic-v1",
        "transcriptFingerprint": _sha(inputs.preview[:-1]),
        **inputs.references,
        "producerPins": inputs.pins,
        "sections": sections,
        "candidateOnly": True,
        "summaryQualityVerified": False,
        "recognitionAccuracyVerified": False,
        "recognitionClocksChanged": False,
        "tokenizerVersion": VIBRATO_VERSION,
    }


def produce(
    config: Path, output: Path, *, existing: bool = False, inventory: Path | None = None
) -> Path:
    """Publish immutable morphology after canonical preview and terminal pin checks."""
    output = output.absolute()
    if not output.is_relative_to(ROOT / "edits" / "work"):
        raise ReplayError("Morphology output must be in canonical repository work storage")
    if existing:
        _path(str(output))
    else:
        canonical._new_work_file(output)
    inputs = load(config)
    if inventory is not None:
        inventory = _path(str(inventory))
        digest = file_digest(inventory)
        declared = record(strict_json(_bound_bytes(inventory, digest, MAX_INVENTORY_BYTES)))
        if (
            set(declared) != {"version", "inputs"}
            or type(declared.get("version")) is not int
            or declared["version"] != 1
            or declared["inputs"] != inputs.pins
        ):
            raise ReplayError("Morphology inventory differs from its full transitive dependencies")
        inputs.pins[str(inventory)] = digest
    # Repeated read-only verification keeps each audit without replacing evidence.
    audits = output.parent / (output.stem + "-audits")
    if (
        audits.is_symlink()
        or audits != audits.resolve()
        or (audits.exists() and not audits.is_dir())
    ):
        raise ReplayError("Morphology audit scratch must be canonical work storage")
    audits.mkdir(mode=0o700, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="check-", dir=audits))
    _verify_preview(inputs, directory)
    document = _document(inputs)
    for name, digest in inputs.pins.items():
        _bound_bytes(_path(name), digest, max(MAX_TOKEN_BYTES, linguistics.MAX_DICTIONARY_BYTES))
    if existing:
        saved = record(strict_json(_bound_bytes(output, file_digest(output), MAX_TOKEN_BYTES)))
        if saved != document:
            raise ReplayError("Saved morphology differs from the current pinned source")
    else:
        _write_new(output, document)
    return output


def plan(config: Path, directory: Path) -> Path:
    """Declare immutable transitive dependencies through one bounded inventory."""
    directory = canonical._output(directory, new=True)
    inputs = load(config)
    directory.mkdir(mode=0o700)
    inventory = directory / "input-inventory.json"
    _write_new(inventory, {"version": 1, "inputs": inputs.pins})
    direct: set[str] = set()
    direct.update(str(record(reference)["path"]) for reference in inputs.references.values())
    direct.update(str(Path(__file__).with_name(name)) for name in CODE_FILES)
    direct.update({str(config), str(inventory)})
    direct.update(str(ROOT / name) for name in (".python-version", "uv.lock", "pyproject.toml"))
    output = directory / "morphology.json"
    command = [
        sys.executable,
        "-m",
        "replay.summary_morphology",
        "produce",
        str(config),
        str(output),
        "--inventory",
        str(inventory),
    ]
    graph = directory / "workflow.json"
    _write_new(
        graph,
        {
            "version": 1,
            "stages": [
                {
                    "name": "canonical-summary-morphology",
                    "command": command,
                    "inputs": sorted(direct),
                    "outputs": [str(output)],
                    "timeout_seconds": 600,
                    "verify_command": [*command, "--verify"],
                }
            ],
        },
    )
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    producer = commands.add_parser("produce")
    producer.add_argument("config", type=Path)
    producer.add_argument("output", type=Path)
    producer.add_argument("--verify", action="store_true")
    producer.add_argument("--inventory", type=Path)
    planner = commands.add_parser("plan")
    planner.add_argument("config", type=Path)
    planner.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.operation == "plan":
        print(plan(args.config, args.directory))
    else:
        print(produce(args.config, args.output, existing=args.verify, inventory=args.inventory))


if __name__ == "__main__":
    main()
