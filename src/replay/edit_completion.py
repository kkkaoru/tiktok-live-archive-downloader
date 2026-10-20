"""Per-recording edit configuration and the all-stages completion gate.

Every processed replay must remove background music, cut non-speech, hide the
source's burned-in captions, transcribe speech and burn dynamic captions. A
delivery that skipped any stage is reported as incomplete instead of being
treated as finished, and is superseded (never deleted) by a complete revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from replay import edit_config
from replay.core import ReplayError
from replay.edit_config import CONFIG_KEYS as CONFIG_KEYS
from replay.edit_config import CONFIG_NAME as CONFIG_NAME
from replay.edit_config import MAX_JSON_BYTES as MAX_JSON_BYTES
from replay.edit_config import MEDIA_SUFFIXES as MEDIA_SUFFIXES
from replay.edit_config import REQUIRED_STAGES as REQUIRED_STAGES
from replay.edit_config import STORAGE_KEY as STORAGE_KEY
from replay.edit_config import YOUTUBE_CONFIG_KEYS as YOUTUBE_CONFIG_KEYS
from replay.edit_config import EditConfig as EditConfig
from replay.edit_config import _replay_id as _replay_id
from replay.edit_config import _storage as _storage
from replay.edit_config import native_media_path as native_media_path
from replay.edit_config import storage_root as storage_root
from replay.jobs import already_complete
from replay.recordings import record

BGM_STATUSES = frozenset({"verified", "listening-approved"})
# Coarse OCR completed and found no repeatable caption band: nothing to hide.
NO_CAPTION_BAND = "no-repeatable-caption-band-observed"
RENDER_EVIDENCE = (
    "summary.json",
    "cuts.json",
    "captions.json",
    "render-plan.json",
    "metadata/complete.json",
    "metadata/mask-frame-audit.json",
)
NOT_SHARED_RENDER = "not produced by the shared cut/mask/caption render"


def _read_json(path: Path) -> object:
    return edit_config._read_json(path, maximum_bytes=MAX_JSON_BYTES)


def bind_source(root: Path, source: Path) -> Path:
    """Point an existing config at a derived copy of its source, preserving the old config bytes.

    The copy keeps the source's file name (delivery names derive from it) and lives elsewhere;
    the previous file is kept as `edit-config.<sha256-prefix>.json`, as for a storage root.
    """
    config = load_config(root)
    media = native_media_path(source)
    if media.name != config.source.name or media == config.source:
        raise ReplayError("A derived source keeps the source's name in another directory")
    path = root / CONFIG_NAME
    data = record(_read_json(path))
    data.update(source=str(media), source_size=media.stat().st_size)
    _replace_config(root, path, data)
    return path


def _replace_config(root: Path, path: Path, data: dict[str, object]) -> None:
    os.link(path, root / f"edit-config.{_digest(path)[:16]}.json")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.chmod(stream.name, 0o600)
    os.replace(stream.name, path)
    load_config(root)


def bind_storage(root: Path, directory: Path) -> Path:
    """Add a storage root to an existing config, preserving the old config bytes.

    The previous file is kept as `edit-config.<sha256-prefix>.json` so evidence
    bound to its hash remains verifiable; a config with a storage root is final.
    """
    config = load_config(root)
    if config.storage_root is not None:
        raise ReplayError("Edit config already names a storage root")
    path = root / CONFIG_NAME
    data = record(_read_json(path))
    data[STORAGE_KEY] = str(storage_root(str(directory), root=root, source=config.source))
    _replace_config(root, path, data)
    return path


def create_config(root: Path, *, jobs: Path, engine: Path, executor_scope: Path) -> Path:
    """Write the per-recording config from its completed job; never overwrite one."""
    replay_id = _replay_id(root)
    job = jobs / f"{replay_id}.json"
    if not already_complete(job):
        raise ReplayError("A completed download job is required before editing")
    data = record(_read_json(job))
    output = data.get("output")
    if not isinstance(output, str):
        raise ReplayError("Invalid replay job record.")
    source = native_media_path(Path(output))
    if not engine.is_dir() or not executor_scope.is_dir():
        raise ReplayError("Edit engine and Executor scope directories must exist")
    config = {
        "version": 1,
        "replay_id": replay_id,
        "source": str(source),
        "source_size": source.stat().st_size,
        "engine": str(engine.resolve()),
        "executor_scope": str(executor_scope.resolve()),
        "stages": dict.fromkeys(REQUIRED_STAGES, True),
    }
    return _write_config(root, config)


def _write_config(root: Path, config: dict[str, object]) -> Path:
    """Publish a private config atomically without replacing an existing revision."""
    path = root / CONFIG_NAME
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root) as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.chmod(stream.name, 0o600)
        os.link(stream.name, path)
    return path


def _youtube_inputs(root: Path, source: Path, publication: Path) -> tuple[str, Path]:
    return edit_config._youtube_inputs(root, source, publication, reader=_read_json)


def create_youtube_config(
    root: Path, *, source: Path, publication: Path, engine: Path, executor_scope: Path
) -> Path:
    """Bind an existing YouTube original; this neither downloads nor approves a delivery."""
    video_id, media = _youtube_inputs(root, source, publication)
    if not engine.is_dir() or not executor_scope.is_dir():
        raise ReplayError("Edit engine and Executor scope directories must exist")
    data: dict[str, object] = {
        "version": 2,
        "video_id": video_id,
        "source": str(media),
        "source_size": media.stat().st_size,
        "source_sha256": _digest(media),
        "publication": str(publication.resolve()),
        "publication_sha256": _digest(publication),
        "engine": str(engine.resolve()),
        "executor_scope": str(executor_scope.resolve()),
        "stages": dict.fromkeys(REQUIRED_STAGES, True),
    }
    _load_youtube_config(root, data)
    return _write_config(root, data)


def _load_youtube_config(root: Path, data: dict[str, object]) -> EditConfig:
    return edit_config._load_youtube_config(root, data, reader=_read_json, digest=_digest)


def load_config(root: Path) -> EditConfig:
    """Keep historical evidence-reader and digest hooks for existing callers."""
    return edit_config.load_config(root, reader=_read_json, digest=_digest)


def _burned_captions(render: Path, burn: dict[str, object]) -> list[object] | None:
    """Captions a part burns with libass, or None when the ASS is not the planned file."""
    name, digest, captions = burn.get("ass"), burn.get("assSha256"), burn.get("captions")
    if (
        not isinstance(name, str)
        or not isinstance(digest, str)
        or not isinstance(captions, list)
        or not isinstance(burn.get("style"), dict)
    ):
        return None
    path = render / name
    if not path.is_file() or path.parent.parent != render or _digest(path) != digest:
        return None
    return captions


def _plan_totals(render: Path) -> tuple[int, int, bool]:
    """Count burned captions and masks, and whether every part styled its captions.

    Captions are either native recipe captions with a caption style, or a part-level
    libass burn whose ASS file still has the planned SHA-256.
    """
    plan = _read_json(render / "render-plan.json")
    if not isinstance(plan, list) or not plan:
        raise ReplayError("Render plan is empty")
    captions = masks = 0
    styled = True
    for part in plan:
        fields = record(part)
        video = record(record(record(fields.get("request")).get("recipe")).get("video"))
        part_masks = video.get("masks")
        if not isinstance(part_masks, list):
            raise ReplayError("Render plan part lacks a mask list")
        masks += len(part_masks)
        if "captionBurn" in fields:
            burned = _burned_captions(render, record(fields["captionBurn"]))
            styled = styled and burned is not None
            captions += len(burned or [])
            continue
        part_captions = video.get("captions")
        if not isinstance(part_captions, list):
            raise ReplayError("Render plan part lacks a caption list")
        captions += len(part_captions)
        styled = styled and isinstance(video.get("captionStyle"), dict)
    return captions, masks, styled


def _cut_failure(summary: dict[str, object], cuts: dict[str, object]) -> str | None:
    segments = cuts.get("segments")
    frames = summary.get("outputFrames")
    if not isinstance(segments, list) or not segments or type(frames) is not int:
        return "cut plan is missing"
    last = record(segments[-1]).get("sourceEndFrame")
    if cuts.get("cursor") != frames or type(last) is not int or frames >= last:
        return "no non-speech interval was removed"
    return None


def _mask_failure(root: Path, masks: object, planned: int, audit: dict[str, object]) -> str | None:
    if type(masks) is not int or masks < 0 or (masks > 0) != (planned > 0):
        return "mask count is missing or differs from the render plan"
    if audit.get("missingMaskFrames") != 0 or audit.get("extraMaskFrames") != 0:
        return "rendered mask frames differ from the plan"
    if masks > 0:
        return "fixed source-caption blur is not detected local repair"
    evidence = root / "evidence" / "source-masks-none" / "summary.json"
    summary = record(_read_json(evidence)) if evidence.is_file() else {}
    reason = summary.get("reason")
    # Earlier batches recorded an all-empty OCR survey with a different reason; it is
    # accepted only when it states that no sampled frame contained any text.
    no_text = summary.get("noSourceSubtitlesDetected") is True and (
        summary.get("observedTextFrames") == 0
    )
    if reason == NO_CAPTION_BAND or no_text:
        return None
    return f"source captions left visible ({reason})"


def _render_failures(render: Path, root: Path) -> dict[str, str]:
    summary = record(_read_json(render / "summary.json"))
    complete = record(_read_json(render / "metadata" / "complete.json"))
    audit = record(_read_json(render / "metadata" / "mask-frame-audit.json"))
    planned_captions, planned_masks, styled = _plan_totals(render)
    failures: dict[str, str] = {}
    cut = _cut_failure(summary, record(_read_json(render / "cuts.json")))
    if cut is not None:
        failures["silence_cut"] = cut
    mask = _mask_failure(root, summary.get("masks"), planned_masks, audit)
    if mask is not None:
        failures["source_caption_mask"] = mask
    cues = summary.get("captionCues")
    if type(cues) is not int or cues <= 0:
        failures["transcription"] = "no transcript cues"
    captions = _read_json(render / "captions.json")
    if (
        not isinstance(captions, list)
        or not captions
        or len(captions) != cues
        or planned_captions < len(captions)
        or not styled
        or complete.get("fullAVDecodePassed") is not True
        or complete.get("videoFrames") != summary.get("outputFrames")
    ):
        failures["dynamic_captions"] = "captions were not burned by a verified full render"
    return failures


def native_delivery_assessment(delivery: Path, *, root: Path) -> dict[str, object] | None:
    """Return explicit v3 detail without confusing recorded stages with strict success."""
    index = delivery.parent.parent / "metadata" / "native-evidence.json"
    if not index.exists() and not index.is_symlink():
        return None
    data = record(_read_json(index))
    if data.get("version") != 3 and data.get("backend") != "native-fcp-partitioned-whole":
        return None
    from replay import native_delivery_completion
    from replay.stages import StageError

    try:
        load_config(root)
        return native_delivery_completion.assess(
            {"path": str(index.absolute()), "sha256": _digest(index)},
            root=root.absolute(),
            delivery=delivery,
            verify_content=True,
        )
    except (ReplayError, StageError, ValueError, OSError) as error:
        return {
            "backend": "native-fcp-partitioned-whole",
            "candidateAvailable": False,
            "strictComplete": False,
            "strictFailures": dict.fromkeys(REQUIRED_STAGES, [f"Invalid native evidence: {error}"]),
        }


def native_stage_failures(details: dict[str, object]) -> dict[str, str]:
    failures = record(details.get("strictFailures"))
    if not failures and details.get("strictComplete") is not True:
        return dict.fromkeys(REQUIRED_STAGES, "Native strict verification incomplete")
    return {
        stage: "; ".join(str(item) for item in value) if isinstance(value, list) else str(value)
        for stage, value in failures.items()
    }


def assess_delivery(delivery: Path, *, root: Path) -> dict[str, str]:
    """Return failed stages for one dated delivery; an empty mapping means complete."""
    pipeline_index = delivery.parent.parent / "metadata" / "pipeline-evidence.json"
    if pipeline_index.is_file():
        # Recording-pipeline deliveries carry their own step evidence (replay.pipeline_completion).
        from replay import pipeline_completion

        return pipeline_completion.assess(pipeline_index, delivery=delivery, root=root)
    native = native_delivery_assessment(delivery, root=root)
    if native is not None:
        return native_stage_failures(native)
    revision = delivery.parent.parent
    native_index = revision / "metadata" / "native-evidence.json"
    if native_index.exists() or native_index.is_symlink():
        # Keep optional editing dependencies out of legacy configuration/discovery.
        from replay import native_completion

        try:
            config = load_config(root)
            assessor = native_completion.assess
            if record(_read_json(native_index)).get("version") == 2:
                from replay import delivery_completion_v2

                assessor = delivery_completion_v2.assess
            return assessor(
                native_index,
                delivery=delivery,
                root=root,
                source=config.source,
                storage=config.storage_root,
            )
        except (ReplayError, ValueError, OSError) as error:
            return dict.fromkeys(REQUIRED_STAGES, f"Invalid native evidence: {error}")
    failures: dict[str, str] = {}
    report_path = revision / "metadata" / "verification.json"
    report = record(_read_json(report_path)) if report_path.is_file() else {}
    source = report.get("source")
    render = revision
    if not isinstance(source, str):
        failures["bgm_removal"] = "no music-removal verification"
    else:
        if (
            report.get("status") not in BGM_STATUSES
            or report.get("full_audio_video_decode") != "passed"
        ):
            failures["bgm_removal"] = f"music removal not verified ({report.get('status')})"
        render = (root.parents[1] / source).parent.parent
    if not all((render / name).is_file() for name in RENDER_EVIDENCE):
        failures.update(dict.fromkeys(REQUIRED_STAGES[1:], NOT_SHARED_RENDER))
    else:
        failures.update(_render_failures(render, root))
    srt = delivery.with_suffix(".srt")
    if not srt.is_file() or "-->" not in srt.read_text(encoding="utf-8", errors="replace"):
        failures.setdefault("transcription", "delivery has no transcript track file")
    return failures


def _stored_delivery_link(path: Path, *, root: Path, storage: Path | None) -> bool:
    if storage is None or not path.is_relative_to(root):
        return False
    expected = storage / path.relative_to(root)
    # Only an exact, unlinked mirror in explicitly authorized storage is accepted.
    return path.resolve() == expected and not expected.is_symlink() and expected.is_file()


def dated_deliveries(root: Path, storage: Path | None = None) -> list[Path]:
    """Find supported dated media; discovery does not certify its required stages.

    A local link may name its exact mirror in the configured storage root. Prefer
    that local identity so existing supersession records and sidecars stay valid.
    """
    suffix = f"_JST_{root.name}-review"
    bases = (root,) if storage is None else (root, storage)
    deliveries: dict[Path, Path] = {}
    for base in bases:
        for path in (base / "revisions").glob("**/delivery/*"):
            if (
                path.suffix.lower() not in MEDIA_SUFFIXES
                or not path.stem.endswith(suffix)
                or not path.is_file()
            ):
                continue
            if base == storage and path.resolve() != path:
                continue
            if path.is_symlink() and not _stored_delivery_link(path, root=root, storage=storage):
                continue
            deliveries.setdefault(path.resolve(), path)
    if storage is not None:
        selected = current_candidate_delivery(root, storage)
        if selected is not None:
            # Preserve the selected revision identity for its native evidence.
            deliveries[selected.resolve()] = selected
    return sorted(deliveries.values())


def current_candidate_delivery(root: Path, storage: Path) -> Path | None:
    """Admit only the publisher's current, immutable-copy-bound candidate alias."""
    from replay.candidate_publication import MAX_SELECTION_BYTES, SELECTION_NAME
    from replay.delivery_output_names import delivery_stem

    selection = storage / "evidence" / SELECTION_NAME
    if not selection.exists() and not selection.is_symlink():
        return None
    config = load_config(root)
    if config.storage_root != storage or selection.resolve() != selection:
        raise ReplayError("Current candidate storage binding differs")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ReplayError("Duplicate candidate header key")
            result[key] = value
        return result

    if not selection.is_file() or not 0 < selection.stat().st_size <= MAX_SELECTION_BYTES:
        raise ReplayError("Current candidate header must be bounded regular metadata")
    before = selection.stat()
    content = selection.read_bytes()
    data = record(json.loads(content, object_pairs_hook=pairs))
    fields = {
        "version",
        "status",
        "movie",
        "movieSize",
        "movieTarget",
        "transcript",
        "transcriptSha256",
        "fcpxml",
        "fcpxmlSha256",
        "editConfigSha256",
        "movieMtimeNanoseconds",
        "renderConsumptionVerified",
        "requiredStagesVerified",
    }
    if (
        set(data) != fields
        or type(data["version"]) is not int
        or data["version"] != 1
        or data["status"] != "unverified-candidate"
        or data["renderConsumptionVerified"] is not False
        or data["requiredStagesVerified"] is not False
    ):
        raise ReplayError("Unsupported current candidate header")

    def relative(key: str) -> Path:
        value = data[key]
        if not isinstance(value, str) or not value or "\0" in value:
            raise ReplayError("Invalid candidate relative path")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or str(path) != value:
            raise ReplayError("Candidate path escapes its recording storage")
        return storage / path

    named, target, transcript = relative("movie"), relative("movieTarget"), relative("transcript")
    stem = delivery_stem(config.source, root.name)
    revision = named.relative_to(storage).parts
    if (
        len(revision) != 4
        or revision[0] != "revisions"
        or revision[2] != "delivery"
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", revision[1]) is None
        or named.stem != stem
        or named.suffix not in MEDIA_SUFFIXES
        or transcript != named.with_suffix(".srt")
        or named.parent.resolve() != named.parent
        or target.resolve(strict=True) != target
        or not target.is_file()
        or target == config.source
        or target.suffix not in MEDIA_SUFFIXES
        or named.resolve(strict=True) != target
        or transcript.resolve(strict=True) != transcript
        or not transcript.is_file()
    ):
        raise ReplayError("Current candidate media/transcript identity differs")
    media_before = target.stat()
    transcript_before = transcript.stat()
    for key in ("transcriptSha256", "fcpxmlSha256", "editConfigSha256"):
        digest = data[key]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ReplayError("Invalid candidate digest")
    xml_value = data["fcpxml"]
    if not isinstance(xml_value, str):
        raise ReplayError("Invalid candidate semantic path")
    xml = Path(xml_value)
    if (
        not xml.is_absolute()
        or xml.resolve(strict=True) != xml
        or not xml.is_file()
        or xml.suffix != ".fcpxml"
        or not (xml.is_relative_to(root.parent) or xml.is_relative_to(storage))
    ):
        raise ReplayError("Invalid candidate semantic path")
    immutable = named.parent / "candidate.json"
    if (
        type(data["movieSize"]) is not int
        or data["movieSize"] <= 0
        or type(data["movieMtimeNanoseconds"]) is not int
        or media_before.st_size != data["movieSize"]
        or media_before.st_mtime_ns != data["movieMtimeNanoseconds"]
        or transcript.stat().st_size > MAX_JSON_BYTES
        or _digest(transcript) != data["transcriptSha256"]
        or _digest(root / CONFIG_NAME) != data["editConfigSha256"]
        or xml.stat().st_size > MAX_JSON_BYTES
        or _digest(xml) != data["fcpxmlSha256"]
        or immutable.resolve(strict=True) != immutable
        or not immutable.is_file()
        or immutable.stat().st_size != len(content)
        or immutable.read_bytes() != content
    ):
        raise ReplayError("Current candidate publisher evidence differs")
    if (
        selection.stat() != before
        or selection.read_bytes() != content
        or target.stat() != media_before
        or transcript.stat() != transcript_before
    ):
        raise ReplayError("Current candidate changed during discovery")
    return named


def delivery_label(root: Path, delivery: Path) -> str:
    """Stable identity of a delivery relative to its recording or storage root."""
    if delivery.is_relative_to(root):
        return str(delivery.relative_to(root))
    storage = load_config(root).storage_root
    if storage is None or not delivery.is_relative_to(storage):
        raise ReplayError("Delivery lies outside the recording and its storage root")
    return f"storage:{delivery.relative_to(storage)}"


def _deliveries(root: Path) -> list[Path]:
    return dated_deliveries(root, load_config(root).storage_root)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def superseded_records(root: Path) -> list[dict[str, object]]:
    directory = root / "evidence" / "superseded"
    if not directory.is_dir():
        return []
    return [record(_read_json(path)) for path in sorted(directory.glob("*.json"))]


def _record_home(root: Path, delivery: Path) -> tuple[Path, str]:
    """Where a delivery's supersession record lives, and its label there.

    A storage-root delivery is recorded in that store's own ``evidence/superseded`` with a path
    relative to the store, which is what the Viewer library of that store reads.
    """
    if delivery.is_relative_to(root):
        return root / "evidence" / "superseded", str(delivery.relative_to(root))
    storage = load_config(root).storage_root
    if storage is None or not delivery.is_relative_to(storage):
        raise ReplayError("Delivery lies outside the recording and its storage root")
    return storage / "evidence" / "superseded", str(delivery.relative_to(storage))


def is_superseded(root: Path, delivery: Path) -> bool:
    """A delivery is superseded only while its recorded identity is unchanged."""
    home, label = _record_home(root, delivery)
    # Older records of storage deliveries were kept under the recording as "storage:<path>".
    places = [(home.parent.parent, label), (root, delivery_label(root, delivery))]
    for place, relative in places:
        for item in superseded_records(place):
            if item.get("delivery") == relative and item.get("size") == delivery.stat().st_size:
                return item.get("sha256") == _digest(delivery)
    return False


def supersede_incomplete(root: Path) -> list[Path]:
    """Record incomplete dated deliveries so a complete revision can be published.

    Nothing is deleted or renamed; a complete delivery is never superseded.
    """
    recorded: list[Path] = []
    for delivery in _deliveries(root):
        failures = assess_delivery(delivery, root=root)
        if not failures or is_superseded(root, delivery):
            continue
        digest = _digest(delivery)
        directory, label = _record_home(root, delivery)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        entry = {
            "delivery": label,
            "sha256": digest,
            "size": delivery.stat().st_size,
            "missingStages": failures,
            "status": "superseded-incomplete-required-stages",
        }
        with (directory / f"{digest[:16]}.json").open("x", encoding="utf-8") as stream:
            json.dump(entry, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        recorded.append(delivery)
    return recorded


def supersede_for_regeneration(root: Path, reason: str) -> list[Path]:
    """Record complete dated deliveries that an explicit request asks to rebuild.

    Nothing is deleted or renamed. The recorded deliveries stop counting as complete,
    so the shared workflow renders a successor; they are removed only after it passes.
    """
    if not reason.strip():
        raise ReplayError("A regeneration needs a stated reason")
    recorded: list[Path] = []
    for delivery in _deliveries(root):
        if assess_delivery(delivery, root=root) or is_superseded(root, delivery):
            continue
        digest = _digest(delivery)
        directory, label = _record_home(root, delivery)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        entry = {
            "delivery": label,
            "sha256": digest,
            "size": delivery.stat().st_size,
            "reason": reason,
            "status": "superseded-for-regeneration",
        }
        with (directory / f"{digest[:16]}.json").open("x", encoding="utf-8") as stream:
            json.dump(entry, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        recorded.append(delivery)
    return recorded


def completion_report(root: Path) -> dict[str, object]:
    """Summarize all dated deliveries; complete when one current delivery passes every stage."""
    load_config(root)
    rows: list[dict[str, object]] = []
    for delivery in _deliveries(root):
        native = native_delivery_assessment(delivery, root=root)
        failures = (
            native_stage_failures(native)
            if native is not None
            else assess_delivery(delivery, root=root)
        )
        superseded = is_superseded(root, delivery)
        rows.append(
            {
                "delivery": delivery_label(root, delivery),
                "complete": not failures and not superseded,
                "superseded": superseded,
                "missingStages": failures,
                **({"nativeAssessment": native} if native is not None else {}),
            }
        )
    return {
        "replay": root.name,
        "complete": any(row["complete"] for row in rows),
        "deliveries": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-recording edit config and completion gate")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Write edit-config.json from the completed job")
    init.add_argument("root", type=Path)
    init.add_argument("--jobs", type=Path, default=Path("private/jobs"))
    init.add_argument("--engine", type=Path, required=True)
    init.add_argument("--executor-scope", type=Path, required=True)
    youtube = commands.add_parser("init-youtube", help="Bind an existing dated YouTube original")
    youtube.add_argument("root", type=Path)
    youtube.add_argument("--source", type=Path, required=True)
    youtube.add_argument("--publication", type=Path, required=True)
    youtube.add_argument("--engine", type=Path, required=True)
    youtube.add_argument("--executor-scope", type=Path, required=True)
    for name in ("check", "supersede", "source", "scope"):
        commands.add_parser(name).add_argument("root", type=Path)
    regenerate = commands.add_parser(
        "regenerate", help="Supersede complete deliveries an explicit request asks to rebuild"
    )
    regenerate.add_argument("root", type=Path)
    regenerate.add_argument("--reason", required=True)
    bind = commands.add_parser("bind-storage", help="Name an external storage root once")
    bind.add_argument("root", type=Path)
    bind.add_argument("storage", type=Path)
    args = parser.parse_args(argv)
    root = Path(args.root).absolute()
    try:
        if args.command == "bind-storage":
            print(bind_storage(root, Path(args.storage).absolute()))
            return 0
        if args.command == "init":
            print(
                create_config(
                    root, jobs=args.jobs, engine=args.engine, executor_scope=args.executor_scope
                )
            )
            return 0
        if args.command == "init-youtube":
            print(
                create_youtube_config(
                    root,
                    source=args.source,
                    publication=args.publication,
                    engine=args.engine,
                    executor_scope=args.executor_scope,
                )
            )
            return 0
        if args.command == "source":
            print(load_config(root).source)
            return 0
        if args.command == "scope":
            print(load_config(root).executor_scope)
            return 0
        if args.command == "regenerate":
            load_config(root)
            for path in supersede_for_regeneration(root, args.reason):
                print(f"SUPERSEDED_FOR_REGENERATION {delivery_label(root, path)}")
            return 0
        if args.command == "supersede":
            load_config(root)
            for path in supersede_incomplete(root):
                print(f"SUPERSEDED {delivery_label(root, path)}")
            return 0
        report = completion_report(root)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["complete"] else 2
    except (ReplayError, OSError, ValueError) as error:
        print(f"EDIT_GATE_ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
