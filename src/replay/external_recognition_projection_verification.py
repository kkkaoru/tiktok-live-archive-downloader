"""Verify direct-producer recognition with exact existing projection semantics.

No factory PCM receipt is invented. Current caller-declared PCM/ASR/clock hashes,
exact projected bytes and unchanged file identities are checked using the existing
producer. Scratch is bounded and removed; originals, word clocks and approvals
remain unchanged. This is neither speech-quality nor delivery acceptance.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from replay import asr_assembly_verification as scratch
from replay import external_recognition_projection as projection
from replay.native_invocation import identity, require
from replay.recordings import record
from replay.sequence_journal import bounded_read, decode

ROLES = frozenset({"labels", "pcm", "audioClock", "asrInputs"})


def verify(config: Path, output: Path) -> int:
    """Return the exact conserved label count; regenerate metadata only, never ASR."""
    config, output = scratch._work(config), scratch._work(output)
    artifacts = record(decode(bounded_read(config)).get("artifacts"))
    require(set(artifacts) == ROLES, "Expected complete direct-producer projection bindings")
    paths = [config, output, Path(projection.__file__).resolve()]
    paths.extend(
        Path(projection.__file__).with_name(name)
        for name in ("asr_clock_projection.py", "audio_clock.py")
    )
    for value in artifacts.values():
        ref = record(value)
        name = ref.get("path")
        require(isinstance(name, str), "Missing projection input path")
        paths.append(scratch._input(str(name)))
    before = {path: identity(path) for path in paths}
    actual = bounded_read(output)
    decode(actual)
    with scratch._candidate() as candidate:
        count = projection.run(config, candidate)
        expected = bounded_read(candidate)
        require(
            actual == expected, "Direct projected labels differ from exact native-clock document"
        )
        require(
            all(identity(path) == observed for path, observed in before.items()),
            "Direct projection inputs or output changed during verification",
        )
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    verify(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
