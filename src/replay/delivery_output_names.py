"""Canonical broadcast-date names and pinned whole-recording Compressor requests.

Names inherit the validated original's JST timestamp and stable recording ID.
Preparing a request neither submits an encode nor verifies the encoded container.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import ParseError

from replay.caption_measured_job import _json as strict_json
from replay.compressor_preset import read as read_preset
from replay.core import ReplayError
from replay.edit_completion import CONFIG_NAME, load_config
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
MAX_CONFIG_BYTES = 65536
RECORDING = re.compile(r"(?:replay-[0-9]{1,32}|youtube-[A-Za-z0-9_-]{11})")
STAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
REQUEST_FIELDS = frozenset({"version", "editConfig", "wholeMaster", "preset", "outputDirectory"})


def delivery_stem(source: Path, recording: str) -> str:
    """Validate the original's canonical datetime and exact stable ID."""
    if RECORDING.fullmatch(recording) is None:
        raise ReplayError("Expected a stable recording identifier")
    suffix = f"_JST_{recording}"
    if not source.stem.endswith(suffix):
        raise ReplayError(
            "Original filename lacks its validated source date and identifier (JST start)"
        )
    stamp = source.stem[: -len(suffix)]
    try:
        date = datetime.strptime(stamp, STAMP_FORMAT)
    except ValueError as error:
        raise ReplayError("Original source date is invalid") from error
    if date.strftime(STAMP_FORMAT) != stamp:
        raise ReplayError("Original source date is not canonical")
    return f"{source.stem}-review"


def preset_extension(preset: Path) -> str:
    """Preserve the container requested by a bounded native encoding preset."""
    root = read_preset(preset)
    encoder = root.find("encoder")
    extension = root.findtext("encoder/file-extension")
    if encoder is None or (encoder.get("name"), extension) not in {
        ("MPEG4", "mp4"),
        ("H264iPod", "m4v"),
    }:
        raise ReplayError("Unsupported or inconsistent native preset container extension")
    return f".{extension}"


def encoded_name(source: Path, recording: str, preset: Path) -> str:
    """Use the same dated stem for the actual Compressor file and public alias."""
    return delivery_stem(source, recording) + preset_extension(preset)


def recorded_stem_matches(path: Path, source: Path, recording: str, *, historical: bool) -> bool:
    """Accept a legacy stem only in a caller's explicitly pinned history audit.

    This compares a recorded output identity; it never selects or finds a file.
    Callers must first verify the immutable job and state SHA256/output bindings.
    """
    canonical = delivery_stem(source, recording)
    return path.stem == canonical or (historical and path.stem == "final-encode")


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ReplayError("Expected an explicit canonical output naming path")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve()
        or any(part in {"private", "downloads"} for part in path.parts)
    ):
        raise ReplayError("Output naming paths must be canonical and outside private/downloads")
    return path


def _pin(value: object, inputs: dict[str, str]) -> Path:
    row = record(value)
    if set(row) != {"path", "sha256"}:
        raise ReplayError("Expected an exact path/SHA256 request input pin")
    path = _path(row["path"])
    if not path.is_file() or path.stat().st_size <= 0:
        raise ReplayError("Request inputs must be nonempty regular files")
    digest = file_digest(path)
    if row["sha256"] != digest:
        raise ReplayError("Compressor request input hash changed")
    inputs[str(path)] = digest
    return path


def _new_file(path: Path) -> Path:
    path = _path(str(path))
    if (
        not path.is_relative_to(ROOT / "edits" / "work")
        or not path.parent.is_dir()
        or path.exists()
        or path.is_symlink()
    ):
        raise ReplayError(
            "Request outputs must be absent repository work files with existing parents"
        )
    return path


def _write(path: Path, data: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    path.chmod(0o600)


def prepare_request(config: Path, output: Path, receipt: Path) -> Path:
    """Prepare one H264/AAC MP4 encode request for an explicitly pinned whole master."""
    config = _path(str(config))
    if not config.is_file() or not 0 < config.stat().st_size <= MAX_CONFIG_BYTES:
        raise ReplayError("Expected a bounded whole Compressor request configuration")
    raw = record(strict_json(config.read_bytes()))
    if set(raw) != REQUEST_FIELDS or type(raw.get("version")) is not int or raw["version"] != 1:
        raise ReplayError("Unknown or missing whole Compressor request configuration fields")
    output, receipt = _new_file(output), _new_file(receipt)
    if len({config, output, receipt}) != 3:
        raise ReplayError("Compressor request configuration and outputs must be distinct")
    inputs = {str(config): file_digest(config)}
    edit_path = _pin(raw["editConfig"], inputs)
    if edit_path.name != CONFIG_NAME or not edit_path.is_relative_to(ROOT):
        raise ReplayError("Expected the repository recording's pinned edit-config.json")
    edit = load_config(edit_path.parent)
    original = edit.source
    if original.stat().st_size != edit.source_size:
        raise ReplayError("Original source size changed from its validated edit config")
    recording = edit_path.parent.name
    stem = delivery_stem(original, recording)
    master = _pin(raw["wholeMaster"], inputs)
    preset = _pin(raw["preset"], inputs)
    directory = _path(raw["outputDirectory"])
    if not directory.is_dir() or not (
        directory.is_relative_to(ROOT / "edits" / "work")
        or (edit.storage_root is not None and directory.is_relative_to(edit.storage_root))
    ):
        raise ReplayError("Compressor output directory must be existing recording storage or work")
    if master == original:
        raise ReplayError("Whole Compressor input must be an edited master, not the original")
    name = encoded_name(original, recording, preset)
    setting = read_preset(preset)
    if (
        not name.endswith(".mp4")
        or setting.findtext("encoder/video-encode/codec-type") != "avc1"
        or setting.findtext("encoder/audio-encode/codec-type") != "aac "
        or setting.find("encoder/video-encode[@isEnabled='yes']") is None
        or setting.find("encoder/audio-encode[@isEnabled='yes']") is None
    ):
        raise ReplayError("Whole MP4 request requires native enabled H264/AAC MPEG4 settings")
    if any(directory.glob(f"{stem}.*")):
        raise ReplayError("Refusing an existing or ambiguous dated Compressor output")
    if any(file_digest(Path(path)) != digest for path, digest in inputs.items()):
        raise ReplayError("Compressor request inputs changed during preparation")
    request: dict[str, object] = {
        "batchName": stem,
        "bundleID": "com.apple.CompressorApp",
        "outputDirectory": str(directory),
        "outputName": name,
        "presetPath": str(preset),
        "sourcePath": str(master),
    }
    _write(output, request)
    _write(
        receipt,
        {
            "version": 1,
            "kind": "whole-compressor-request",
            "recording": recording,
            "inputs": inputs,
            "request": {"path": str(output), "sha256": file_digest(output)},
            "original": {"path": str(original), "bytes": edit.source_size},
            "timestampSource": "validated-original-filename-not-new-api-observation",
            "output": str(directory / name),
            "wholeMaster": str(master),
            "presetContainer": "mp4",
            "presetVideoCodec": "avc1",
            "presetAudioCodec": "aac",
            "encodeSubmitted": False,
            "actualContainerVerified": False,
            "wholeRenderVerified": False,
            "deliveryApproved": False,
        },
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        print(prepare_request(args.config, args.request, args.receipt))
    except (ReplayError, OSError, ValueError, ParseError) as error:
        print(f"DELIVERY_OUTPUT_NAMES_ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
