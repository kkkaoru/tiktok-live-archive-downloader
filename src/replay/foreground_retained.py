"""Revalidate retained observations with read-only historical source-code archives.

Archives attest producer bytes only: they are never imported or executed. Metadata
preflight deliberately does not verify source media bytes and cannot authorize reuse.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path

from replay import foreground_collection as collection
from replay import tracking_observations as storage
from replay.foreground_evidence import ForegroundWindow, _bound_bytes
from replay.foreground_observations import MAX_SAMPLES, ForegroundFrame, foreground_frames
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROLES = frozenset(
    {
        "tracking_observations.py",
        "head_observations.py",
        "subtitle_clock.py",
        "vertical_captions.py",
        "caption_io.py",
        "recordings.py",
        "core.py",
        "stages.py",
        "foreground_observations.py",
        "foreground_collection.py",
    }
)
READER_FILES = (*collection.CODE_FILES, "foreground_evidence.py", "foreground_retained.py")


@dataclass(frozen=True)
class Verification:
    window: ForegroundWindow
    inputs: dict[str, str]
    producer_code: dict[str, str]
    media_bytes_verified: bool

    def receipt(self) -> dict[str, object]:
        return {
            "version": 1,
            "protocol": "foreground-collection-v1",
            "verifiedAt": datetime.now(UTC).isoformat(),
            "inputs": self.inputs,
            "historicalProducerCode": self.producer_code,
            "currentReaderCode": {
                str(Path(__file__).with_name(name)): self.inputs[
                    str(Path(__file__).with_name(name))
                ]
                for name in READER_FILES
            },
            "provenanceSHA256": self.window.provenance_sha256,
            "frames": len(self.window.frames),
            "mediaBytesVerified": self.media_bytes_verified,
            "historicalExecutionReissued": False,
            "captionPlacementApproved": False,
            "deliveryComplete": False,
        }


def _archive_paths(descriptor: Path, digest: str) -> dict[str, Path]:
    raw = record(json.loads(_bound_bytes(descriptor, digest, storage.CONFIG_BYTES)))
    if (
        set(raw) != {"version", "protocol", "archives"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or raw["protocol"] != "foreground-collection-v1"
    ):
        raise ValueError("Unsupported retained foreground protocol")
    archives = raw["archives"]
    if not isinstance(archives, list) or not 1 <= len(archives) <= len(ARCHIVE_ROLES):
        raise ValueError("Invalid historical code archives")
    result: dict[str, Path] = {}
    for value in archives:
        item = record(value)
        role = item.get("role")
        if set(item) != {"role", "path", "sha256"} or not isinstance(role, str):
            raise ValueError("Invalid historical code role")
        if role not in ARCHIVE_ROLES or role in result:
            raise ValueError("Only unique allowlisted source-code roles may be archived")
        if not isinstance(item["path"], str) or not Path(item["path"]).is_absolute():
            raise ValueError("Historical archive paths must be absolute")
        path = storage._checked_output(Path(item["path"]))
        if not path.is_relative_to(ROOT / "edits" / "work"):
            raise ValueError("Historical archives must stay under edits/work")
        _bound_bytes(path, storage._digest(item["sha256"]), storage.CONFIG_BYTES)
        result[role] = path
    return result


def verify(
    config: Path,
    output: Path,
    *,
    config_sha256: str,
    provenance_sha256: str,
    descriptor: Path,
    descriptor_sha256: str,
    metadata_only: bool = False,
) -> Verification:
    """Independently check current schema, exact clocks, batches and protected pins.

    Explicit metadata-only mode hashes native/code/config/response but merely stats
    the original media. Full mode additionally verifies the unchanged media bytes.
    Neither mode rewrites the producer or asserts historical execution by this reader.
    """
    archives = _archive_paths(descriptor, descriptor_sha256)
    job = collection.load_job(config)
    if job.config_hash != storage._digest(config_sha256):
        raise ValueError("Pinned foreground configuration changed")
    output = storage._checked_output(output)
    provenance = output / "provenance.json"
    raw = record(json.loads(_bound_bytes(provenance, provenance_sha256, storage.CONFIG_BYTES)))
    original = {name: storage._digest(value) for name, value in record(raw.get("inputs")).items()}
    code = {str(Path(__file__).with_name(name)): name for name in collection.CODE_FILES}
    expected = {str(job.config), str(job.source), str(job.native), *code}
    if len(expected) != 3 + len(code) or set(original) != expected:
        raise ValueError("Retained foreground input roles differ")
    inputs = {str(descriptor): descriptor_sha256, str(provenance): provenance_sha256}
    producer_code: dict[str, str] = {}
    for name, old in original.items():
        path = Path(name)
        role = code.get(name)
        if role is not None:
            producer_code[name] = old
            path = archives.get(role, path)
        if metadata_only and name == str(job.source):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Missing retained source media")
            continue
        if file_digest(path) != old:
            raise ValueError(f"Retained foreground binding changed: {name}")
        inputs[str(path)] = old
    if original[str(job.source)] != job.source_hash or original[str(job.native)] != job.native_hash:
        raise ValueError("Protected foreground source/native pins differ")
    for name in READER_FILES:
        path = Path(__file__).with_name(name)
        inputs[str(path)] = file_digest(path)
    stored = json.loads(storage._read(output / "inputs.json", storage.CONFIG_BYTES))
    # The current producer parser checks request clocks, diagnostic emptiness,
    # response schema and every batch digest against the original receipt.
    if stored != original or raw != collection._receipt(job, output, original):
        raise ValueError("Retained foreground provenance changed")
    inputs[str(output / "inputs.json")] = file_digest(output / "inputs.json")
    frames: list[ForegroundFrame] = []
    for index, ticks in enumerate(batched(job.clock.pts, MAX_SAMPLES)):
        batch = output / f"batch-{index:04d}"
        for filename in ("request.json", "response.json"):
            path = batch / filename
            inputs[str(path)] = file_digest(path)
        frames.extend(
            foreground_frames(
                json.loads(storage._read(batch / "response.json", collection.RESPONSE_BYTES)),
                times=tuple(tick * job.clock.time_base for tick in ticks),
                width=job.width,
                height=job.height,
            )
        )
    if raw != collection._receipt(job, output, original):
        raise ValueError("Retained foreground responses changed during reading")
    for name, digest in inputs.items():
        if file_digest(Path(name)) != digest:
            raise ValueError("Retained verification input changed during reading")
    return Verification(
        ForegroundWindow(job, tuple(frames), provenance_sha256),
        inputs,
        producer_code,
        not metadata_only,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--provenance-sha256", required=True)
    parser.add_argument("--descriptor-sha256", required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    result = verify(
        args.config,
        args.output,
        config_sha256=args.config_sha256,
        provenance_sha256=args.provenance_sha256,
        descriptor=args.descriptor,
        descriptor_sha256=args.descriptor_sha256,
        metadata_only=args.metadata_only,
    )
    receipt = storage._checked_output(args.receipt.absolute())
    if not receipt.is_relative_to(ROOT / "edits" / "work"):
        raise ValueError("Verification receipts must stay under edits/work")
    storage._write(receipt, result.receipt())


if __name__ == "__main__":
    main()
