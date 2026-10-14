"""Safely apply one verified audio-restoration revision to current deliveries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from replay.audio_restore import apply_listening_approved, restore_delivery
from replay.bgm_separation import MODEL_NAME, MODEL_REVISION, MODEL_SHA256
from replay.core import ReplayError
from replay.delivery import delivery_stem

DEFAULT_REVISION_NAME = "audio-restore-mel-roformer-mlx"
LISTENING_REVISION_NAME = "audio-mel-roformer-listening-approved"


def _revision_root(source: Path) -> Path:
    root = next((parent for parent in source.parents if parent.name == "revisions"), None)
    if root is None:
        raise ReplayError(f"Delivery is outside a revisions tree: {source}")
    return root


def _already_complete(source: Path, revision: Path, repository: Path) -> bool:
    try:
        revision.resolve(strict=True).relative_to((repository / "edits").resolve(strict=True))
    except (OSError, ValueError):
        return False
    report_path = revision / "metadata" / "verification.json"
    try:
        report: object = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(report, dict):
        return False
    if report.get("source") != str(source.relative_to(repository)):
        return False
    status = report.get("status")
    delivery = revision / "delivery" / f"{delivery_stem(_revision_root(source).parent)}.mp4"
    legacy = revision / "delivery" / source.name
    if report.get("delivery") == str(legacy.relative_to(repository)):
        delivery = legacy
    if status == "no-instrumental-content":
        return not delivery.exists() and not legacy.exists()
    if status not in ("verified", "listening-approved") or report.get("delivery") != str(
        delivery.relative_to(repository)
    ):
        return False
    if status == "listening-approved":
        runtime = report.get("processing_minutes_per_audio_hour")
        if (
            report.get("model") != f"{MODEL_NAME}@{MODEL_REVISION}"
            or report.get("model_sha256") != MODEL_SHA256
            or report.get("full_audio_video_decode") != "passed"
            or isinstance(runtime, bool)
            or not isinstance(runtime, (int, float))
            or not 0.0 <= runtime < 30.0
        ):
            return False
    try:
        delivery.resolve(strict=True).relative_to((repository / "edits").resolve(strict=True))
    except (OSError, ValueError):
        return False
    return delivery.is_file()


def discover_current_deliveries(
    repository: Path,
    revision_name: str = DEFAULT_REVISION_NAME,
) -> tuple[tuple[Path, Path], ...]:
    """Require one source per replay before scheduling any restoration."""
    if (
        not revision_name
        or Path(revision_name).name != revision_name
        or revision_name in (".", "..")
    ):
        raise ValueError("Revision name must be one path component")
    root = repository.resolve(strict=True)
    edits = (root / "edits").resolve(strict=True)
    targets: list[tuple[Path, Path]] = []
    sources: dict[str, Path] = {}
    for source in sorted(edits.glob("replay-*/revisions/**/delivery/full-review.mp4")):
        lexical_relative = source.relative_to(edits)
        stem = delivery_stem(edits / lexical_relative.parts[0])
        own_names = (revision_name, f"{stem}-{revision_name}")
        if any(
            lexical_relative.parts[2] == name or lexical_relative.parts[2].startswith(name + "-")
            for name in own_names
        ):
            continue
        try:
            source = source.resolve(strict=True)
            relative = source.relative_to(edits)
        except (OSError, ValueError) as error:
            raise ReplayError("Current delivery must remain inside edits") from error
        replay = relative.parts[0]
        if replay in sources:
            raise ReplayError(
                f"Multiple deliveries for one source require reconciliation: {replay}"
            )
        sources[replay] = source

    for source in sources.values():
        relative = source.relative_to(edits)
        identifier = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()[:12]
        revisions = _revision_root(source)
        stem = delivery_stem(revisions.parent)
        legacy_revision = revisions / f"{revision_name}-{identifier}"
        if legacy_revision.exists() or legacy_revision.is_symlink():
            if _already_complete(source, legacy_revision, root):
                continue
            raise ReplayError(
                f"Refusing existing incomplete restoration revision: {legacy_revision}"
            )
        revision = revisions / f"{stem}-{revision_name}-{identifier}"
        if revision.exists() or revision.is_symlink():
            if _already_complete(source, revision, root):
                continue
            raise ReplayError(f"Refusing existing incomplete restoration revision: {revision}")
        if not source.with_suffix(".srt").is_file():
            raise ReplayError(f"Matching transcript is missing: {source.with_suffix('.srt')}")
        targets.append((source, revision))
    return tuple(targets)


def restore_current_deliveries(
    repository: Path,
    revision_name: str = DEFAULT_REVISION_NAME,
    device: str = "auto",
    listening_approved: bool = False,
) -> tuple[dict[str, object], ...]:
    """Process discovered deliveries sequentially and stop at the first failure."""
    outcomes: list[dict[str, object]] = []
    processor = apply_listening_approved if listening_approved else restore_delivery
    for source, revision in discover_current_deliveries(repository, revision_name):
        outcomes.append(processor(source, revision, device))
    return tuple(outcomes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-name")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--listening-approved",
        action="store_true",
        help="use separate listening-approved revisions",
    )
    arguments = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    name = arguments.revision_name or (
        LISTENING_REVISION_NAME if arguments.listening_approved else DEFAULT_REVISION_NAME
    )
    try:
        outcomes = restore_current_deliveries(
            repository,
            revision_name=name,
            device=arguments.device,
            listening_approved=arguments.listening_approved,
        )
    except (OSError, ValueError, ReplayError) as error:
        print(f"audio restore batch stopped safely: {error}", file=sys.stderr)
        return 2
    print(json.dumps(outcomes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
