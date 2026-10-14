"""Verify a complete measured caption artifact without rendering or publication."""

import argparse
from fractions import Fraction
from pathlib import Path

from replay.core import ReplayError
from replay.fcp_caption_retarget import RetainedSpan, read_reflow
from replay.stages import file_digest


def verify(
    path: Path,
    *,
    source: Path,
    frames: int,
    rate: int,
    producer_evidence: Path | None = None,
    start_frame: int = 0,
) -> int:
    """Verify all current inputs and complete output-clock coverage; return cue count."""
    if type(frames) is not int or not 1 <= frames <= 10_000_000:
        raise ReplayError("Expected bounded output frame count")
    if type(rate) is not int or not 1 <= rate <= 60:
        raise ReplayError("Expected bounded output frame rate")
    if type(start_frame) is not int or not 0 <= start_frame <= 10_000_000 - frames:
        raise ReplayError("Expected bounded output start frame")
    source = source.absolute()
    cues = read_reflow(
        path,
        (
            RetainedSpan(
                Fraction(0), Fraction(start_frame, rate), Fraction(0), Fraction(frames, rate)
            ),
        ),
        source=source,
        source_sha256=file_digest(source),
        producer_evidence=producer_evidence,
    )
    return len(cues)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reflow", type=Path)
    parser.add_argument("--source-media", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--rate", type=int, default=30)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--producer-evidence", type=Path)
    args = parser.parse_args()
    count = verify(
        args.reflow,
        source=args.source_media,
        frames=args.frames,
        rate=args.rate,
        producer_evidence=args.producer_evidence,
        start_frame=args.start_frame,
    )
    print(f"Verified {count} measured caption stages; native rendered delivery remains unverified")


if __name__ == "__main__":
    main()
