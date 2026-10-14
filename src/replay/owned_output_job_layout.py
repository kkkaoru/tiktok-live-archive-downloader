"""Read-only pinned-metadata guard separating owned model output from job runtimes.

This validates declared roots only, not undisclosed process scratch or job readiness.
No directory, socket, model, media, or completion record is created or read.
"""

import argparse
import hashlib
import json
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
MAX_METADATA_BYTES = 1024 * 1024
MAX_ROOTS = 64
OWNED_FIELDS = ("outputDirectory", "workDirectory", "work")
RUNTIME_FIELDS = ("workDirectory", "scratchDirectory", "cacheDirectory")


@dataclass(frozen=True)
class MetadataPin:
    path: Path
    sha256: str


def _canonical(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts or path.resolve() != path:
        raise ValueError("Expected canonical non-symlink path")
    if not path.is_relative_to(REPOSITORY):
        raise ValueError("Path is outside this checkout")
    return path


def _directory(value: object) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("Expected explicit directory text")
    path = _canonical(Path(value))
    if path.exists() and not path.is_dir():
        raise ValueError("Declared root is not a directory")
    return path


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate metadata key")
        result[key] = value
    return result


def _read(pin: MetadataPin) -> dict[str, object]:
    path = _canonical(pin.path)
    if re.fullmatch(r"[0-9a-f]{64}", pin.sha256) is None:
        raise ValueError("Invalid metadata SHA256")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_METADATA_BYTES:
        raise ValueError("Expected bounded regular metadata")
    with path.open("rb") as stream:
        content = stream.read(MAX_METADATA_BYTES + 1)
    after = path.stat()
    if (
        before != after
        or _canonical(path) != path
        or len(content) != before.st_size
        or hashlib.sha256(content).hexdigest() != pin.sha256
    ):
        raise ValueError("Pinned metadata changed")
    value: object = json.loads(content, object_pairs_hook=_unique)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Expected metadata object")
    return {key: item for key, item in value.items()}


def validate_layout(
    *,
    config: MetadataPin,
    request: MetadataPin,
    owned_field: str,
    orchestration_roots: tuple[Path, ...],
    reject_reciprocal: bool = False,
) -> dict[str, object]:
    """Check pinned known fields and all declared external runtime roots without writes."""
    if owned_field not in OWNED_FIELDS or not 1 <= len(orchestration_roots) <= MAX_ROOTS:
        raise ValueError("Expected supported owned field and bounded orchestration roots")
    configured, admission = _read(config), _read(request)
    if admission.get("version") != 1 or admission.get("kind") != "measured-job-admission":
        raise ValueError("Unsupported admission request")
    owned = _directory(configured.get(owned_field))
    roots = tuple(_directory(admission.get(key)) for key in RUNTIME_FIELDS)
    roots += tuple(_directory(str(path)) for path in orchestration_roots)
    for root in roots:
        if root.is_relative_to(owned):
            raise ValueError("Orchestration root overlaps owned output")
        if reject_reciprocal and owned.is_relative_to(root):
            raise ValueError("Owned output overlaps orchestration root")
    # Re-read only small pinned metadata and re-resolve roots at the endpoint.
    if _read(config) != configured or _read(request) != admission:
        raise ValueError("Metadata changed during layout validation")
    for root in (owned, *roots):
        _directory(str(root))
    return {
        "version": 1,
        "kind": "owned-output-job-layout",
        "layoutVerified": True,
        "config": {"path": str(config.path), "sha256": config.sha256},
        "request": {"path": str(request.path), "sha256": request.sha256},
        "ownedField": owned_field,
        "ownedOutput": str(owned),
        "orchestrationRoots": [str(path) for path in dict.fromkeys(roots)],
        "rejectReciprocal": reject_reciprocal,
        "metadataOnly": True,
        "runtimeObserved": False,
        "productionAdmitted": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--owned-field", choices=OWNED_FIELDS, required=True)
    parser.add_argument("--orchestration-root", type=Path, action="append", required=True)
    parser.add_argument("--reject-reciprocal", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate_layout(
            config=MetadataPin(args.config, args.config_sha256),
            request=MetadataPin(args.request, args.request_sha256),
            owned_field=args.owned_field,
            orchestration_roots=tuple(args.orchestration_root),
            reject_reciprocal=args.reject_reciprocal,
        )
    except (OSError, ValueError) as error:
        print(f"OWNED_OUTPUT_LAYOUT_ERROR {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
