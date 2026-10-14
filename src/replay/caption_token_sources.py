"""Explicit, hash-bound selection of already-adopted recognition replacements."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from replay.caption_measured_job import _token
from replay.core import ReplayError
from replay.foreground_evidence import _bound_bytes
from replay.motion_stages import _integer
from replay.recordings import record
from replay.stages import file_digest


@dataclass(frozen=True)
class TokenSource:
    segment: int
    job: int
    tokens: Path
    token_digest: str
    manifest: Path
    manifest_digest: str
    combined: tuple[Path, str] | None = None
    original_combined: tuple[Path, str] | None = None

    def verify(self, bindings: dict[str, object]) -> dict[str, str]:
        pins = {str(self.tokens): self.token_digest, str(self.manifest): self.manifest_digest}
        if any(bindings.get(path) != digest for path, digest in pins.items()):
            raise ReplayError("Adopted token source and manifest must both be candidate-bound")
        manifest = record(json.loads(_bound_bytes(self.manifest, self.manifest_digest, 4 << 20)))
        if manifest.get("version") != 1 or manifest.get("tokens") != {
            "path": str(self.tokens),
            "sha256": self.token_digest,
        }:
            raise ReplayError("Adopted manifest does not identify the selected token source")
        _bound_bytes(self.tokens, self.token_digest, 64 << 20)
        return pins


def combined_path(default: Path, segment: int, sources: tuple[TokenSource, ...]) -> Path:
    """Select a minimal lineage overlay; old captions and clocks stay byte-equivalent data.

    Only additional input pins are permitted. Every selected replacement remains
    subject to TokenSource.verify; this does not approve recognition accuracy.
    """
    members = tuple(source for source in sources if source.segment == segment)
    overlays = tuple(source for source in members if source.combined is not None)
    if not overlays:
        return default
    first = overlays[0]
    if first.combined is None or first.original_combined is None:
        raise ReplayError("Combined overlay requires original evidence")
    original_path, original_digest = first.original_combined
    path, digest = first.combined
    if (
        original_path != default.resolve()
        or path == original_path
        or any(
            source.combined != first.combined or source.original_combined != first.original_combined
            for source in overlays
        )
    ):
        raise ReplayError("Combined overlay source or segment conflict")
    old = record(json.loads(_bound_bytes(original_path, original_digest, 64 << 20)))
    current = record(json.loads(_bound_bytes(path, digest, 64 << 20)))
    expected = dict(old)
    inputs = dict(record(old.get("inputs")))
    additions = {str(original_path): original_digest}
    for source in members:
        additions.update(
            {str(source.tokens): source.token_digest, str(source.manifest): source.manifest_digest}
        )
    if any(key in inputs and inputs[key] != value for key, value in additions.items()):
        raise ReplayError("Combined overlay cannot replace existing evidence")
    inputs.update(additions)
    expected["inputs"] = inputs
    if current != expected:
        raise ReplayError("Combined overlay changed captions, clocks or unrelated evidence")
    return path


def _reference(value: object) -> tuple[Path, str]:
    item = record(value)
    path, digest = item.get("path"), item.get("sha256")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or str(Path(path).resolve()) != path
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ReplayError("Expected canonical absolute token reference and SHA256")
    return Path(path), digest


def load_sources(
    path: Path, entries: list[object]
) -> tuple[tuple[TokenSource, ...], dict[str, str]]:
    """Every replacement belongs to an existing segment/job; no implicit discovery."""
    digest = file_digest(path)
    raw = record(json.loads(_bound_bytes(path, digest, 1 << 20)))
    replacements = raw.get("replacements")
    if (
        raw.get("version") != 1
        or not isinstance(replacements, list)
        or not 1 <= len(replacements) <= 512
    ):
        raise ReplayError("Expected bounded version1 token source map")
    membership: set[tuple[int, int]] = set()
    for value in entries:
        row = record(value)
        jobs = row.get("jobs")
        if not isinstance(jobs, list):
            raise ReplayError("Expected segment job membership")
        membership.update((_integer(row.get("segment")), _integer(job)) for job in jobs)
    result: list[TokenSource] = []
    seen: set[tuple[int, int]] = set()
    for value in replacements:
        row = record(value)
        key = (_integer(row.get("segment")), _integer(row.get("job")))
        if key in seen or key not in membership:
            raise ReplayError("Token replacement is duplicated or outside segment job membership")
        seen.add(key)
        tokens, token_digest = _reference(row.get("tokens"))
        manifest, manifest_digest = _reference(row.get("adoptedManifest"))
        combined = _reference(row["combined"]) if "combined" in row else None
        original = _reference(row["originalCombined"]) if "originalCombined" in row else None
        if (combined is None) != (original is None):
            raise ReplayError("Combined overlay requires paired original evidence")
        result.append(
            TokenSource(*key, tokens, token_digest, manifest, manifest_digest, combined, original)
        )
    bindings = {
        str(path.resolve()): digest,
        str(Path(__file__).resolve()): file_digest(Path(__file__)),
    }
    for source in result:
        for reference in (source.combined, source.original_combined):
            if reference is not None:
                name, value = str(reference[0]), reference[1]
                if name in bindings and bindings[name] != value:
                    raise ReplayError("Conflicting combined overlay pins")
                bindings[name] = value
    return tuple(result), bindings


def main() -> None:
    """Read-only adoption preflight; no native measurement or artifact writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    index_path = args.directory / "segments.json"
    index = record(json.loads(_bound_bytes(index_path, file_digest(index_path), 4 << 20)))
    entries = index.get("segments")
    if not isinstance(entries, list):
        raise ReplayError("Expected segment index")
    sources, _ = load_sources(args.mapping, entries)
    for source in sources:
        combined = combined_path(
            args.directory / f"combined-{source.segment:04d}.json", source.segment, sources
        )
        raw = record(json.loads(_bound_bytes(combined, file_digest(combined), 64 << 20)))
        source.verify(record(raw.get("inputs")))
        tokens = record(json.loads(_bound_bytes(source.tokens, source.token_digest, 64 << 20)))
        values = tokens.get("tokens")
        if (
            tokens.get("version") != 1
            or tokens.get("clock") != "output"
            or not isinstance(values, list)
            or not 1 <= len(values) <= 4096
        ):
            raise ReplayError("Expected bounded recognized output-clock tokens")
        tuple(_token(value) for value in values)
    print(json.dumps({"verifiedReplacements": len(sources), "readOnly": True}))


if __name__ == "__main__":
    main()
