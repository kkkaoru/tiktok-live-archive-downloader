"""Verify exact assembler bytes and original PCM clocks without changing originals.

The existing assembler regenerates its expected document in bounded checkout-local
scratch. Verification preserves unresolved clocks and ownership selection exactly;
it does not certify speech accuracy, silence, human review or delivery completion.
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from replay import asr_assembly
from replay.native_invocation import canonical, identity, require
from replay.recordings import record
from replay.sequence_journal import bounded_read, decode

SCRATCH_ROOT = Path("edits/work/asr-assembly-verification")


def _work(path: Path) -> Path:
    checked = canonical(str(path))
    require(
        checked.is_relative_to(Path.cwd().resolve() / "edits/work"),
        "Assembly verification requires checkout-work paths",
    )
    return checked


def _input(name: str) -> Path:
    path = canonical(name)
    root = Path.cwd().resolve()
    require(
        path.is_relative_to(root)
        and not path.is_relative_to(root / "private")
        and not path.is_relative_to(root / "downloads"),
        "Assembly input escapes permitted checkout data",
    )
    return path


def _identities(source: Path, labels: Path) -> dict[Path, dict[str, int]]:
    manifest = source / "inputs.json"
    complete = source / "complete.json"
    recipe = decode(bounded_read(manifest))
    decode(bounded_read(complete))
    windows = asr_assembly._windows(recipe)
    inputs = record(recipe.get("inputs"))
    require(1 <= len(inputs) <= 16, "Invalid assembly inputs")
    paths = [manifest, complete, labels, *(_input(name) for name in inputs)]
    for window in windows:
        raw = source / f"{window.start_sample:012d}.raw.json"
        clock = source / f"{window.start_sample:012d}.clock.json"
        decode(bounded_read(raw))
        decode(bounded_read(clock))
        paths.extend((raw, clock))
    return {path: identity(path) for path in paths}


@contextmanager
def _candidate() -> Iterator[Path]:
    root = _work(Path.cwd().resolve() / SCRATCH_ROOT)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    child = root / f"verification-{uuid.uuid4().hex}"
    child.mkdir(mode=0o700)
    owned = child.stat()
    candidate = child / "expected-labels.json"
    try:
        yield candidate
    finally:
        require(
            child == child.resolve()
            and child.stat().st_ino == owned.st_ino
            and child.stat().st_dev == owned.st_dev,
            "Verifier scratch identity changed",
        )
        require(not candidate.is_symlink(), "Verifier candidate became linked")
        if candidate.exists():
            identity(candidate)
            candidate.unlink()
        child.rmdir()


def verify(source: Path, labels: Path) -> int:
    """Return the conserved label count after exact-byte and immutable-input checks.

    Scratch contains only one generated bounded JSON document and is removed on
    success or mismatch. Every original retains its identity and contents.
    """
    source, labels = _work(source), _work(labels)
    before = _identities(source, labels)
    actual = bounded_read(labels)
    decode(actual)
    with _candidate() as candidate:
        count = asr_assembly.assemble(source, candidate)
        expected = bounded_read(candidate)
        require(actual == expected, "Assembled labels differ from exact native-clock document")
        require(
            all(identity(path) == observed for path, observed in before.items()),
            "Assembly inputs or labels changed during verification",
        )
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_collection", type=Path)
    parser.add_argument("labels_path", type=Path)
    args = parser.parse_args(argv)
    verify(args.source_collection, args.labels_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
