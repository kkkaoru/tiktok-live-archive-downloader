"""Read-only verification of original cleanup admission artifacts; never allocate RGB."""

import argparse
import json
import sys
from pathlib import Path

from replay import cleanup_lane_successor as successor
from replay import subtitle_cleanup_workflow as cleanup
from replay.caption_masks import MaskPolicy
from replay.local_asr import fingerprint
from replay.recordings import record
from replay.sequence_journal import encode, publish
from replay.subtitle_cleanup_plan import admission, search_profile
from replay.subtitle_eligibility import _reference

ARTIFACTS = ("ocr-batches.json", "admission.json", "eligibility.json", "candidate-config.json")


def _compare(path: Path, expected: object) -> None:
    cleanup._read(path)
    if path.read_text(encoding="utf-8") != json.dumps(expected, indent=2, allow_nan=False) + "\n":
        raise ValueError(f"Original admission artifact differs: {path.name}")


def _verify_artifacts(chunk: Path) -> dict[str, str]:
    """Reuse original pure admission algorithms; mirror only its serialized schema."""
    cleanup.check_ocr(chunk)
    data, settings, clock = cleanup._chunk(chunk)
    workers = data.get("maximumActiveChunks", 1)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Invalid original active chunk capacity budget")
    directory = chunk.parent
    ocr = cleanup._local(Path(str(data.get("ocrDirectory"))))
    frames, batches = cleanup.collector_frames(ocr / "inputs.json", ocr / "complete.json", settings)
    seeds, _ = cleanup.collector_frames(
        _reference(settings.get("profileInputs"), {}),
        _reference(settings.get("profileComplete"), {}),
        settings,
    )
    plan = admission(frames, clock, search_profile(seeds), policy=MaskPolicy())
    _compare(directory / "ocr-batches.json", batches)
    _compare(directory / "admission.json", plan)
    source, native, model = (record(settings.get(key)) for key in ("source", "native", "model"))
    _compare(
        directory / "eligibility.json",
        {
            "version": 1,
            "kind": "source-caption-eligibility",
            "sourceSHA256": source["sha256"],
            "canvas": [frames[0].width, frames[0].height],
            "timeBase": str(clock.time_base),
            "decoder": settings.get("decoder"),
            "codec": settings["sourceClock"],
            "classificationEvidence": settings["classificationEvidence"],
            "ocrProvenance": cleanup._pin(ocr / "inputs.json"),
            "ocr": cleanup._pin(directory / "ocr-batches.json"),
            "frames": plan["frames"],
        },
    )
    active = (
        [
            index
            for index, row in enumerate(plan["frames"])
            if record(row).get("classification") == "overlay-subtitle"
        ]
        if isinstance(plan["frames"], list)
        else []
    )
    candidate: dict[str, object] = {
        "source": source["path"],
        "sourceSHA256": source["sha256"],
        "nativeExecutable": native["path"],
        "nativeSHA256": native["sha256"],
        "outputDirectory": str(directory / "repair"),
        "workDirectory": str(directory / "scratch"),
        "width": frames[0].width,
        "height": frames[0].height,
        "clock": data["clock"],
        "masks": plan["masks"],
        "visionTextRegions": True,
        "prepareVisionContext": False,
        "learnedTextDetector": {
            "backend": "ppocrv3-db",
            "model": model["path"],
            "sha256": model["sha256"],
        },
        "sourceEligibility": cleanup._pin(directory / "eligibility.json"),
        "hardwareEncode": False,
    }
    if active:
        candidate["evidenceFrames"] = sorted({active[0], active[-1]})
    _compare(directory / "candidate-config.json", candidate)
    return {str(directory / name): fingerprint(directory / name) for name in ARTIFACTS}


def verify_admission(chunk: Path, *, verify_repair: bool = False) -> dict[str, object]:
    """Verify existing artifacts and endpoints; do not complete an original Stage."""
    cleanup._local(chunk)
    initial = {str(chunk): fingerprint(chunk)}
    # Require all four artifacts before any semantic verification. No partial adoption.
    for name in ARTIFACTS:
        path = chunk.parent / name
        cleanup._read(path)
        initial[str(path)] = fingerprint(path)
    _, settings, _ = cleanup._chunk(chunk)
    source = record(settings["source"])
    source_path = Path(str(source["path"]))
    if fingerprint(source_path) != source["sha256"]:
        raise ValueError("Original source changed before admission verification")
    outputs = _verify_artifacts(chunk)
    if verify_repair:
        cleanup.check_repair(chunk)
    if _verify_artifacts(chunk) != outputs or any(
        fingerprint(Path(name)) != sha for name, sha in initial.items()
    ):
        raise ValueError("Admission inputs or outputs changed during verification")
    if fingerprint(source_path) != source["sha256"]:
        raise ValueError("Original source changed after admission verification")
    return {
        "version": 1,
        "kind": "readonly-original-cleanup-admission",
        "chunk": {"path": str(chunk), "sha256": initial[str(chunk)]},
        "outputs": outputs,
        "repairVerified": verify_repair,
        "originalStageCompleted": False,
        "qualityApproved": False,
    }


def verify_successor(plan: Path, index: int, *, verify_repair: bool = False) -> dict[str, object]:
    """Retain the existing owner/lease/input guards for a new verification Stage."""
    with successor._owned(plan, index) as (_, row):
        result = verify_admission(successor._path(row["chunk"]), verify_repair=verify_repair)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunk", type=Path)
    parser.add_argument("--successor", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--verify-repair", action="store_true")
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--verify-proof", action="store_true")
    args = parser.parse_args(argv)
    try:
        if (args.successor is None) != (args.index is None):
            raise ValueError("Successor and index must be supplied together")
        if args.successor is not None:
            with successor._owned(args.successor, args.index) as (_, row):
                if successor._path(row["chunk"]) != args.chunk:
                    raise ValueError("Selected successor chunk differs")
                result = verify_admission(args.chunk, verify_repair=args.verify_repair)
        else:
            result = verify_admission(args.chunk, verify_repair=args.verify_repair)
        if args.verify_proof and args.proof is None:
            raise ValueError("Proof path is required for verification")
        if args.proof is not None:
            proof = cleanup._local(args.proof)
            if args.verify_proof:
                if cleanup._read(proof) != result:
                    raise ValueError("Recorded admission proof differs")
            else:
                publish(proof, encode(result))
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        print(f"READONLY_ADMISSION_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
