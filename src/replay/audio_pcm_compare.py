"""Compare hash-bound, already aligned PCM16 stems without a quality tolerance."""

import argparse
import sys
from pathlib import Path

import numpy as np

from replay.audio_cpu_probe import sha256
from replay.bgm_separation import MODEL_SAMPLE_RATE, _read_wav
from replay.core import ReplayError
from replay.native_invocation import canonical, durable, require


def compare(
    reference: Path,
    candidate: Path,
    output: Path,
    *,
    reference_sha256: str,
    candidate_sha256: str,
) -> None:
    reference, candidate, output = (canonical(str(path)) for path in (reference, candidate, output))
    require(output.is_relative_to(Path.cwd() / "edits/work"), "Comparison output must stay local")
    require(
        sha256(reference) == reference_sha256 and sha256(candidate) == candidate_sha256,
        "Comparison input hash changed",
    )
    left, channels = _read_wav(reference)
    right, right_channels = _read_wav(candidate)
    require(
        channels == right_channels and left.shape == right.shape,
        "Comparison requires aligned identical sample counts/channels",
    )
    require(left.shape[1] <= 60 * MODEL_SAMPLE_RATE, "Representative comparison exceeds 60 seconds")
    error = (right.astype(np.float64) - left.astype(np.float64)) * 32768.0
    require(
        sha256(reference) == reference_sha256 and sha256(candidate) == candidate_sha256,
        "Comparison input changed during measurement",
    )
    maximum = float(np.max(np.abs(error)))
    rms = float(np.sqrt(np.mean(error**2)))
    durable(
        output,
        {
            "reference": str(reference),
            "referenceSHA256": reference_sha256,
            "candidate": str(candidate),
            "candidateSHA256": candidate_sha256,
            "samplesPerChannel": int(left.shape[1]),
            "channels": channels,
            "sampleRate": MODEL_SAMPLE_RATE,
            "alignmentSampleOffset": 0,
            "alignmentSearched": False,
            "inputAndModelProvenanceRequiresSeparateEvidence": True,
            "maximumAbsoluteErrorPCM16": maximum,
            "rmsErrorPCM16": rms,
            "maximumAbsoluteErrorFullScale": maximum / 32768.0,
            "rmsErrorFullScale": rms / 32768.0,
            "differentChannelSamples": int(np.count_nonzero(error)),
            "maximumAbsoluteErrorPCM16PerChannel": np.max(np.abs(error), axis=1).tolist(),
            "rmsErrorPCM16PerChannel": np.sqrt(np.mean(error**2, axis=1)).tolist(),
            "qualityTolerance": None,
            "musicRemovalApproved": False,
            "speechPreservationApproved": False,
            "deliveryApproved": False,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        compare(
            args.reference,
            args.candidate,
            args.output,
            reference_sha256=args.reference_sha256,
            candidate_sha256=args.candidate_sha256,
        )
    except (ReplayError, OSError, ValueError) as error:
        print(f"PCM_COMPARE_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
