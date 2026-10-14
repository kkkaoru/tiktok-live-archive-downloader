"""Hash-verified exclusive publication of completed native masters across volumes."""

import argparse
import os
import stat
from pathlib import Path

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_part_assembly import verify as verify_part
from replay.native_export_receipt import save_new
from replay.recordings import record
from replay.stages import file_digest

COPY_BLOCK_BYTES = 8 * 1024 * 1024
MINIMUM_RESERVE_BYTES = 30 * 1024**3


def _identity(path: Path) -> tuple[int, int, int, int]:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise ReplayError("Expected a nonempty regular part master")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _completed(path: Path, *, replacement: Path | None = None) -> tuple[dict[str, object], Path]:
    receipt = _json(path)
    if (
        receipt.get("version") != 1
        or receipt.get("kind") != "verified-native-part"
        or receipt.get("completionVerified") is not True
        or receipt.get("shareObservedIdle") is not True
        or type(receipt.get("fullAVDecodeExitCode")) is not int
        or receipt.get("fullAVDecodeExitCode") != 0
    ):
        raise ReplayError("Native part has no independent completion proof")
    original_movie = record(receipt.get("movie"))
    original_path = _path(original_movie.get("path"))
    movie = _pin(receipt.get("movie")) if replacement is None else replacement
    if replacement is not None and (
        movie == original_path or file_digest(movie) != original_movie.get("sha256")
    ):
        raise ReplayError("Retired staging replacement differs from completed bytes")
    inputs = record(receipt.get("inputs"))
    if not inputs or any(
        file_digest(
            movie if replacement is not None and _path(name) == original_path else _path(name)
        )
        != digest
        for name, digest in inputs.items()
    ):
        raise ReplayError("Native completion inputs changed")
    proof = _json(_pin(receipt.get("frameProof")))
    if proof.get("movie") != receipt.get("movie") or proof.get("part") != receipt.get("part"):
        raise ReplayError("Completion and frame proof identify different media")
    return receipt, movie


def publish(config_path: Path) -> Path:
    """Keep the source; retain intent/partial on failure instead of silently retrying."""
    config_path = _path(str(config_path))
    config = _json(config_path)
    if config.get("version") != 1:
        raise ReplayError("Expected version-one publication configuration")
    completion_path = _pin(config.get("completion"))
    receipt, source = _completed(completion_path)
    destination = _path(config.get("destination"))
    output = _path(config.get("receiptOutput"))
    proof_output = _path(config.get("frameProofOutput"))
    intent = output.with_suffix(".intent.json")
    partial = destination.with_name(f".{destination.name}.publication-partial")
    outputs = (destination, output, proof_output, intent, partial)
    if len(set(outputs + (source, config_path, completion_path))) != 8 or any(
        path.exists() or path.is_symlink() or not path.parent.is_dir() for path in outputs
    ):
        raise ReplayError(
            "Publication outputs must be distinct, absent and in existing directories"
        )
    identity = _identity(source)
    space = os.statvfs(destination.parent)
    reserve = max(MINIMUM_RESERVE_BYTES, space.f_blocks * space.f_frsize // 20)
    if space.f_bavail * space.f_frsize < identity[2] + reserve:
        raise ReplayError("Insufficient publication space including reserve")
    config_hash = file_digest(config_path)
    completion_hash = file_digest(completion_path)
    source_proof = _pin(receipt.get("frameProof"))
    frame_proof = _json(source_proof)
    save_new(
        intent,
        {
            "version": 1,
            "kind": "native-part-publication-intent",
            "source": receipt["movie"],
            "destination": str(destination),
            "sourceIdentity": list(identity),
            "completion": config["completion"],
            "config": {"path": str(config_path), "sha256": config_hash},
            "sourcePreserved": True,
        },
    )
    with (
        os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), "rb") as reader,
        partial.open("xb") as writer,
    ):
        opened = os.fstat(reader.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ReplayError("Source changed before copying")
        os.fchmod(writer.fileno(), 0o600)
        while block := reader.read(COPY_BLOCK_BYTES):
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    expected = record(receipt["movie"])["sha256"]
    if (
        _identity(source) != identity
        or file_digest(source) != expected
        or file_digest(partial) != expected
        or file_digest(config_path) != config_hash
        or file_digest(completion_path) != completion_hash
    ):
        raise ReplayError("Publication source, copy or binding changed")
    _path(str(destination))
    os.link(partial, destination)
    published = {"path": str(destination), "sha256": expected}
    if file_digest(destination) != expected:
        raise ReplayError("Published part digest differs")
    frame_proof.update(
        movie=published,
        sourceFrameProof=receipt["frameProof"],
        publicationIntent={"path": str(intent), "sha256": file_digest(intent)},
    )
    save_new(proof_output, frame_proof)
    save_new(
        output,
        {
            "version": 1,
            "kind": "published-native-part",
            "source": receipt["movie"],
            "movie": published,
            "completion": config["completion"],
            "frameProof": {"path": str(proof_output), "sha256": file_digest(proof_output)},
            "intent": {"path": str(intent), "sha256": file_digest(intent)},
            "sourcePreserved": True,
            "qualityApproved": False,
        },
    )
    partial.unlink()
    return output


def verify_published(
    path: Path, proof_path: Path, *, part: int, retirement: Path | None = None
) -> tuple[Path, Path]:
    """Bind copied bytes to original completion without treating copy time as export time."""
    published = _json(path)
    if (
        published.get("version") != 1
        or published.get("kind") != "published-native-part"
        or published.get("sourcePreserved") is not True
        or _pin(published.get("frameProof")) != proof_path
    ):
        raise ReplayError("Invalid native publication receipt")
    completion_path = _pin(published.get("completion"))
    replacement = None if retirement is None else _retired_replacement(retirement, path, published)
    completed, _ = _completed(completion_path, replacement=replacement)
    movie = _pin(published.get("movie"))
    proof = _json(proof_path)
    original = _json(_pin(completed.get("frameProof")))
    intent_path = _pin(published.get("intent"))
    intent = _json(intent_path)
    expected = dict(original)
    expected.update(
        movie=published.get("movie"),
        sourceFrameProof=completed.get("frameProof"),
        publicationIntent=published.get("intent"),
    )
    if (
        completed.get("part") != part
        or published.get("source") != completed.get("movie")
        or record(published.get("movie")).get("sha256")
        != record(completed.get("movie")).get("sha256")
        or proof != expected
        or intent.get("source") != completed.get("movie")
        or intent.get("destination") != str(movie)
        or intent.get("completion") != published.get("completion")
    ):
        raise ReplayError("Published master does not preserve its original completion binding")
    return completion_path, intent_path


def _retired_replacement(path: Path, publication_path: Path, published: dict[str, object]) -> Path:
    retired = _json(path)
    if (
        retired.get("version") != 1
        or retired.get("kind") != "verified-staging-retirement"
        or retired.get("stagingRetirementAuthorized") is not True
        or retired.get("deletionPerformed") is not False
        or _pin(retired.get("publication")) != publication_path
        or retired.get("source") != published.get("source")
        or retired.get("movie") != published.get("movie")
        or retired.get("completion") != published.get("completion")
        or retired.get("frameProof") != published.get("frameProof")
        or retired.get("intent") != published.get("intent")
    ):
        raise ReplayError("Staging retirement does not bind this publication")
    config = _json(_pin(retired.get("config")))
    if (
        config.get("version") != 1
        or config.get("retireStaging") is not True
        or config.get("publication") != retired.get("publication")
        or config.get("assembly") != retired.get("assembly")
        or _path(config.get("receiptOutput")) != path
    ):
        raise ReplayError("Staging retirement intent changed")
    part = retired.get("part")
    if (
        type(part) is not int
        or part < 1
        or verify_part(_pin(retired.get("assembly")), part=part)
        != _pin(published.get("frameProof"))
    ):
        raise ReplayError("Retirement assembly differs from publication proof")
    source = _path(record(retired.get("source")).get("path"))
    movie = _pin(retired.get("movie"))
    identity = retired.get("sourceIdentity")
    if (
        source == movie
        or not isinstance(identity, list)
        or len(identity) != 4
        or any(type(value) is not int or value < 0 for value in identity)
        or retired.get("destinationIdentity") != list(_identity(movie))
        or identity[:2] == list(_identity(movie))[:2]
    ):
        raise ReplayError("Invalid staging retirement identities")
    if source.exists() and (
        list(_identity(source)) != identity
        or file_digest(source) != record(retired.get("source")).get("sha256")
    ):
        raise ReplayError("Staging source changed after retirement authorization")
    return movie


def prepare_retirement(config_path: Path) -> Path:
    """Authorize retiring one verified duplicate; never delete media or rewrite history."""
    config_path = _path(str(config_path))
    config_hash = file_digest(config_path)
    config = _json(config_path)
    if config.get("version") != 1 or config.get("retireStaging") is not True:
        raise ReplayError("Explicit staging retirement intent is required")
    publication_path = _pin(config.get("publication"))
    published = _json(publication_path)
    proof_path = _pin(published.get("frameProof"))
    proof = _json(proof_path)
    part = proof.get("part")
    if type(part) is not int or part < 1:
        raise ReplayError("Expected a positive retired part number")
    verify_published(publication_path, proof_path, part=part)
    assembly = _pin(config.get("assembly"))
    if verify_part(assembly, part=part) != proof_path:
        raise ReplayError("Retirement assembly differs from publication proof")
    source = _pin(published.get("source"))
    movie = _pin(published.get("movie"))
    source_identity, destination_identity = _identity(source), _identity(movie)
    output = _path(config.get("receiptOutput"))
    if (
        output in {source, movie, config_path, publication_path, proof_path}
        or output.exists()
        or output.is_symlink()
        or source_identity[:2] == destination_identity[:2]
    ):
        raise ReplayError("Retirement requires distinct copies and an absent receipt")
    retirement = {
        "version": 1,
        "kind": "verified-staging-retirement",
        "stagingRetirementAuthorized": True,
        "deletionPerformed": False,
        "part": part,
        "assembly": config["assembly"],
        "source": published["source"],
        "movie": published["movie"],
        "sourceIdentity": list(source_identity),
        "destinationIdentity": list(destination_identity),
        "publication": config["publication"],
        "completion": published["completion"],
        "frameProof": published["frameProof"],
        "intent": published["intent"],
        "config": {"path": str(config_path), "sha256": config_hash},
    }
    verify_published(publication_path, proof_path, part=part)
    if (
        _identity(source) != source_identity
        or _identity(movie) != destination_identity
        or file_digest(config_path) != config_hash
        or file_digest(publication_path) != record(config["publication"]).get("sha256")
        or file_digest(assembly) != record(config["assembly"]).get("sha256")
    ):
        raise ReplayError("Retirement inputs changed during verification")
    save_new(output, retirement)
    return output


def verify_retirement(config_path: Path) -> None:
    """Revalidate immutable completion and the surviving copy with explicit retirement."""
    config = _json(_path(str(config_path)))
    if config.get("version") != 1:
        raise ReplayError("Expected version-one retirement verification")
    retirement = _pin(config.get("retirement"))
    retired = _json(retirement)
    publication_path = _pin(retired.get("publication"))
    proof_path = _pin(retired.get("frameProof"))
    part = _json(proof_path).get("part")
    if type(part) is not int or part < 1:
        raise ReplayError("Expected a positive retired part number")
    verify_published(publication_path, proof_path, part=part, retirement=retirement)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-retirement", action="store_true")
    mode.add_argument("--verify-retirement", action="store_true")
    args = parser.parse_args()
    if args.prepare_retirement:
        prepare_retirement(args.config)
    elif args.verify_retirement:
        verify_retirement(args.config)
    else:
        publish(args.config)


if __name__ == "__main__":
    main()
