"""Record verified native part completion without repeating media decoding."""

import argparse
import hashlib
import json
from pathlib import Path

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError, write_private_text
from replay.fcp_native_parts import _idle, _observation
from replay.fcp_part_assembly import verify
from replay.part_master_publication import verify_published
from replay.recordings import record
from replay.stages import file_digest


def update(config_path: Path) -> None:
    """Compare pinned progress, verify existing proof, and atomically update one row."""
    config = _json(config_path)
    part = config.get("part")
    if config.get("version") != 1 or type(part) is not int or part < 1:
        raise ReplayError("Expected a positive native part number")
    progress_path = _pin(config.get("progress"))
    assembly = _pin(config.get("assembly"))
    original_digest = file_digest(progress_path)
    progress = _json(progress_path)
    rows = progress.get("parts")
    if progress.get("version") != 1 or not isinstance(rows, list):
        raise ReplayError("Expected native part progress")
    members = [record(item) for item in rows]
    selected = [row for row in members if row.get("part") == part]
    if len(selected) != 1:
        raise ReplayError("Native part progress identity is absent or ambiguous")
    proof_path = verify(assembly, part=part)
    proof = _json(proof_path)
    movie = _pin(proof.get("movie"))
    destination = _path(progress.get("libraryAndMasterRoot")) / str(selected[0].get("master"))
    if destination != movie:
        raise ReplayError("Progress movie differs from the verified master")
    completion_paths: tuple[Path, ...]
    if "retirement" in config and "publication" not in config:
        raise ReplayError("Staging retirement requires its publication")
    if "publication" in config:
        publication_path = _pin(config["publication"])
        retirement = _pin(config["retirement"]) if "retirement" in config else None
        completion_path, intent_path = verify_published(
            publication_path, proof_path, part=part, retirement=retirement
        )
        completion_paths = (publication_path, completion_path, intent_path) + (
            () if retirement is None else (retirement,)
        )
        decode_evidence = "Pinned original full AV completion and identical published bytes"
    else:
        status = _pin(config.get("decodeStatus"))
        if status.read_text(encoding="utf-8").strip() != "0":
            raise ReplayError("Full AV decode has not succeeded")
        background_path, background = _observation(config.get("background"), historical=True)
        _idle(background)
        if min(status.stat().st_mtime, background_path.stat().st_mtime) < movie.stat().st_mtime:
            raise ReplayError("Completion observations predate the final movie write")
        completion_paths = (status, background_path)
        decode_evidence = "Explicit pinned status supplied for this verified movie"
    observed = record(proof.get("observed"))
    frames = observed.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ReplayError("Verified master has no frame evidence")
    selected[0].update(
        {
            "state": "complete",
            "sizeBytes": movie.stat().st_size,
            "videoFrames": len(frames),
            "fullAVDecodeStatus": 0,
            "shareObservedIdle": True,
            "qualityApproved": False,
            "frameProof": {"path": str(proof_path), "sha256": file_digest(proof_path)},
        }
    )
    progress["parts"] = members
    content = json.dumps(progress, ensure_ascii=False, indent=2) + "\n"
    receipt_path = _path(config.get("receiptPath"))
    receipt = {
        "version": 1,
        "part": part,
        "progress": str(progress_path),
        "previousSHA256": original_digest,
        "updatedSHA256": hashlib.sha256(content.encode()).hexdigest(),
        "previousDocument": _json(progress_path),
        "inputs": {
            str(path): file_digest(path)
            for path in (
                config_path,
                assembly,
                proof_path,
                *completion_paths,
            )
        },
        "movie": proof.get("movie"),
        "qualityApproved": False,
        "decodeEvidence": decode_evidence,
    }
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ReplayError("Progress update receipt already exists")
    if file_digest(progress_path) != original_digest:
        raise ReplayError("Progress changed during verification; rebind before updating")
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
    write_private_text(progress_path, content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    update(parser.parse_args().config)


if __name__ == "__main__":
    main()
