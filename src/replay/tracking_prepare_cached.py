"""Prepare unchanged tracking workflows with invocation-local verified input reuse."""

import argparse
import json
from pathlib import Path

from replay import tracking_batch, tracking_observations
from replay.code_dependency_inventory import capture
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest


def prepare(manifest: Path, output: Path, *, unseeded: bool = False) -> Path:
    """Keep normal producer checks; freshly hash inputs at both operation boundaries."""
    manifest = manifest.absolute()
    output = tracking_observations._checked_output(output)
    if output.exists():
        raise FileExistsError(output)
    raw = record(json.loads(tracking_observations._read(manifest, tracking_batch.MAX_INPUT_BYTES)))
    paths = {manifest}
    for name in ("source", "nativeExecutable"):
        value = raw.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("Explicit source/native path required")
        paths.add(Path(value))
    for name in ("clock", "seeds", "eof"):
        value = record(raw.get(name)).get("path")
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("Explicit clock/seed/EOF path required")
        paths.add(Path(value))
    paths.update(item.path for item in capture(("tracking_prepare_cached",)))
    cache = InputHashCache(sorted(paths))
    cache.prime()
    with input_hash_context(cache):
        if (
            record(
                json.loads(tracking_observations._read(manifest, tracking_batch.MAX_INPUT_BYTES))
            )
            != raw
        ):
            raise ValueError("Manifest changed before preparation")
        workflow = tracking_batch.prepare(manifest, output, unseeded=unseeded)
        inputs = cache.verify()
    tracking_observations._write(
        output / "cache-provenance.json",
        {
            "version": 1,
            "protocol": "tracking-prepare-verified-cache-v1",
            "inputs": inputs,
            "outputs": {
                str(path): file_digest(path) for path in (workflow, output / "binding.json")
            },
            "hashReads": cache.hash_reads,
            "bytesRead": cache.bytes_read,
            "cacheHits": cache.cache_hits,
            "terminalFreshHashesVerified": True,
            "nativeExecuted": False,
        },
    )
    return workflow


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--unseeded", action="store_true")
    args = parser.parse_args(argv)
    print(prepare(args.manifest, args.output, unseeded=args.unseeded))


if __name__ == "__main__":
    main()
