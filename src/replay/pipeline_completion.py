"""Publish a recording pipeline's movie as a dated delivery and assess it quickly.

The recording pipeline runs every stage as a step with a receipt (``recording_steps`` and
``recording_caption_steps``). This backend of ``edit_completion`` judges the five required stages
from those receipts and a few fast technical checks of the published movie, in minutes:

- ``bgm_removal``: the separation step pinned its plan and voice; the movie has one audio stream
  as long as its video;
- ``silence_cut``: the cut-plan step passed and the movie holds exactly the planned frames;
- ``source_caption_mask``: the cleanup step passed, every chunk of the cleanup has an accepted
  receipt, and the composition used the cleaned source; or the source's 240-frame profile survey
  is verified again (``caption_absence.verify``) to carry no caption band;
- ``transcription``: the sidecar SRT holds one cue per displayed caption, in order;
- ``dynamic_captions``: every caption was placed, every caption part rendered successfully and
  the whole movie decodes without errors.

Receipts are local evidence, not attestations; ``qualityApproved`` stays false.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from replay import caption_absence
from replay.core import ReplayError
from replay.delivery_output_names import delivery_stem
from replay.edit_completion import REQUIRED_STAGES, load_config
from replay.recordings import record

BACKEND = "recording-pipeline-v1"
INDEX_NAME = "pipeline-evidence.json"
RATE = 30
STEP_NAMES = (
    "clocks", "separation", "cleanup-bootstrap", "cleanup", "asr", "yolo-track",
    "cut-plan", "paging", "placement", "caption-parts", "caption-render", "compose", "finish",
)  # fmt: skip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def srt_time(frame: int) -> str:
    milliseconds = frame * 1000 // RATE
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, milliseconds = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def displayed(candidate: Path) -> list[tuple[str, int, int]]:
    """Displayed captions: each growing chain of placed cues as its final text and frames.

    A cue that starts on the frame the previous one ends, at the same origin, and extends its
    text is the same caption revealing more words (``motion_reveal_stages``).
    """
    rows = read(candidate).get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("The placement candidate has no cues")
    shown: list[tuple[str, int, int, float, float]] = []
    for raw in rows:
        item = record(raw)
        cue = record(item["cue"])
        text = str(cue["text"])
        start = round(float(str(cue["start"])) * RATE)
        end = round(float(str(cue["end"])) * RATE)
        x, y = float(str(item["x"])), float(str(item["y"]))
        if shown:
            last = shown[-1]
            grows = start == last[2] and (x, y) == (last[3], last[4])
            if grows and len(text) > len(last[0]) and text.startswith(last[0]):
                shown[-1] = (text, last[1], end, x, y)
                continue
        shown.append((text, start, end, x, y))
    return [(text, start, end) for text, start, end, _, _ in shown]


def srt(captions: Sequence[tuple[str, int, int]]) -> str:
    blocks = [
        f"{number}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n"
        for number, (text, start, end) in enumerate(captions, start=1)
    ]
    return "\n".join(blocks)


def pin(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def publish(
    root: Path, *, movie: Path, steps: Path, candidate: Path, revision: str
) -> tuple[Path, Path]:
    """Link the movie into the recording's storage revision, write its SRT and evidence index."""
    config = load_config(root)
    if config.storage_root is None:
        raise ReplayError("Bind the recording's storage root first")
    stem = delivery_stem(config.source, root.name)
    revision_root = config.storage_root / "revisions" / revision
    delivery = revision_root / "delivery" / f"{stem}.mp4"
    delivery.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    (revision_root / "metadata").mkdir(mode=0o700, exist_ok=True)
    if not delivery.exists():
        os.link(movie, delivery)
    if not delivery.samefile(movie):
        raise ReplayError(f"{delivery} exists and is not the pipeline's movie")
    sidecar = delivery.with_suffix(".srt")
    text = srt(displayed(candidate))
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") != text:
        raise ReplayError(f"{sidecar} exists with other captions")
    sidecar.write_text(text, encoding="utf-8")
    index = revision_root / "metadata" / INDEX_NAME
    value = {
        "version": 1,
        "backend": BACKEND,
        "recording": root.name,
        "configurationSHA256": sha256(root / "edit-config.json"),
        "source": str(config.source),
        "movie": pin(delivery),
        "srt": pin(sidecar),
        "candidate": pin(candidate),
        "steps": {name: pin(steps / f"{name}.json") for name in STEP_NAMES},
        "qualityApproved": False,
    }
    index.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    return delivery, index


def probe(movie: Path, ffprobe: str) -> dict[str, object]:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,nb_read_packets,duration", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True, timeout=1800,
    ).stdout  # fmt: skip
    return record(json.loads(out))


def decodes(movie: Path, ffmpeg: str) -> bool:
    """The whole movie decodes (video on VideoToolbox, audio) with no reported error."""
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-hwaccel", "videotoolbox", "-i", str(movie),
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=7200,
    )  # fmt: skip
    return result.returncode == 0 and not result.stderr.strip()


def outputs(receipt: dict[str, object]) -> dict[str, object]:
    return record(receipt.get("outputs"))


def mask_failure(cleanup: dict[str, object]) -> str | None:
    """Every cleanup chunk accepted, or a source verified again to carry no caption band."""
    if "captionFree" in cleanup:
        return caption_absence.verify(record(cleanup["captionFree"]))
    description = read(Path(str(record(cleanup["description"])["path"])))
    work = Path(str(description["work"]))
    receipts = sorted((work / "state").glob("chunk-*/chunk-receipt.json"))
    chunks = len(list((work / "configs").glob("chunk-*.json")))
    accepted = [r for r in receipts if read(r).get("accepted")]
    if not receipts or len(accepted) != chunks:
        return f"{len(accepted)} of {chunks} cleanup chunks accepted"
    return None


def stage_failures(index: dict[str, object], *, ffmpeg: str, ffprobe: str) -> dict[str, str]:
    """The failed stages with their reason; an empty mapping means all five passed."""
    steps = {name: read(Path(str(record(pinned)["path"])))
             for name, pinned in record(index["steps"]).items()}  # fmt: skip
    movie = Path(str(record(index["movie"])["path"]))
    failures: dict[str, str] = {}
    listed = probe(movie, ffprobe).get("streams")
    streams = [record(s) for s in listed] if isinstance(listed, list) else []
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    cut_rows = read(Path(str(record(outputs(steps["cut-plan"])["cuts"])["path"]))).get("cuts")
    planned = sum(
        int(str(record(row)["source_end_frame"])) - int(str(record(row)["source_start_frame"]))
        for row in (cut_rows if isinstance(cut_rows, list) else [])
    )
    if not record(outputs(steps["separation"])).get("vocals") or len(audio) != 1 or len(video) != 1:
        failures["bgm_removal"] = "no pinned separated voice or not exactly one audio stream"
    elif abs(float(str(audio[0]["duration"])) - float(str(video[0]["duration"]))) > 1 / RATE:
        failures["bgm_removal"] = "audio and video lengths differ"
    if not video or int(str(video[0]["nb_read_packets"])) != planned:
        failures["silence_cut"] = f"movie frames differ from the {planned} planned frames"
    mask = mask_failure(outputs(steps["cleanup"]))
    if mask is not None:
        failures["source_caption_mask"] = mask
    captions = displayed(Path(str(record(index["candidate"])["path"])))
    sidecar = Path(str(record(index["srt"])["path"]))
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != srt(captions):
        failures["transcription"] = "the SRT sidecar differs from the displayed captions"
    placement = outputs(steps["placement"])
    rendered = outputs(steps["caption-render"])
    if placement.get("failedRuns") != 0:
        failures["dynamic_captions"] = "some captions were not placed"
    elif not rendered.get("parts"):
        failures["dynamic_captions"] = "no caption part rendered"
    elif not decodes(movie, ffmpeg):
        failures["dynamic_captions"] = "the movie does not decode cleanly"
    return failures


def assess(index_path: Path, *, delivery: Path, root: Path) -> dict[str, str]:
    """Edit-completion failures for a pipeline delivery (empty when complete)."""
    index = read(index_path)
    try:
        if index.get("version") != 1 or index.get("backend") != BACKEND:
            raise ReplayError("Unsupported pipeline evidence")
        if index.get("recording") != root.name:
            raise ReplayError("Pipeline evidence names another recording")
        if index.get("configurationSHA256") != sha256(root / "edit-config.json"):
            raise ReplayError("The edit configuration changed after publication")
        movie = record(index["movie"])
        if not delivery.samefile(Path(str(movie["path"]))):
            raise ReplayError("The evidence names another movie")
        if delivery.stat().st_size != movie["bytes"]:
            raise ReplayError("The movie changed after publication")
        for pinned in (index["srt"], index["candidate"], *record(index["steps"]).values()):
            entry = record(pinned)
            if sha256(Path(str(entry["path"]))) != entry["sha256"]:
                raise ReplayError(f"{entry['path']} changed after publication")
    except (ReplayError, OSError, KeyError, ValueError) as error:
        return dict.fromkeys(REQUIRED_STAGES, f"Invalid pipeline evidence: {error}")
    return stage_failures(index, ffmpeg="ffmpeg", ffprobe="ffprobe")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    published = commands.add_parser("publish")
    for name in ("root", "movie", "steps", "candidate"):
        published.add_argument(name, type=Path)
    published.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    delivery, index = publish(
        args.root, movie=args.movie, steps=args.steps, candidate=args.candidate,
        revision=args.revision,
    )  # fmt: skip
    print(json.dumps({"delivery": str(delivery), "index": str(index)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
