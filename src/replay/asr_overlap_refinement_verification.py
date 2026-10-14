"""Read-only exact verification through the existing native overlap producer."""

import argparse
from pathlib import Path

from replay import asr_label_refine
from replay.asr_assembly_verification import _candidate, _work
from replay.input_hash_snapshot import InputHashSnapshot
from replay.native_invocation import require
from replay.sequence_journal import bounded_read, decode
from replay.stages import file_digest


def verify(config: Path, labels: Path) -> int:
    """Preserve originals and compare exact producer bytes in checkout-local scratch."""
    config, labels = _work(config), _work(labels)
    producer = Path(asr_label_refine.__file__).resolve()
    snapshot = InputHashSnapshot.capture(
        [(path, file_digest(path)) for path in (config, labels, producer, Path(__file__).resolve())]
    )
    actual = bounded_read(labels)
    decode(actual)
    with _candidate() as expected:
        count = asr_label_refine.run(config, expected)
        require(
            actual == bounded_read(expected),
            "Overlap candidate differs from exact native observations",
        )
        snapshot.finish()
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args(argv)
    verify(args.config, args.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
