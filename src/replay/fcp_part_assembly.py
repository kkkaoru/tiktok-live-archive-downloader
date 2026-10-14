"""Assemble frame-verified burned masters with one exact-clock processed voice."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json as _json
from replay.artifact_json import _path as _path
from replay.artifact_json import _pin as _pin
from replay.core import ReplayError
from replay.fcp_caption_srt import Cue, _time, caption_cues, srt_text
from replay.fcp_partition import _merged, _required, _stamp
from replay.recordings import record
from replay.stages import file_digest

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_PROBE_DECODER_THREADS = 64


@dataclass(frozen=True)
class FrameProbeTool:
    """The observed executable and explicit decoder policy for one new probe."""

    path: Path
    sha256: str
    decoder_threads: int = 0

    def __post_init__(self) -> None:
        _path(str(self.path))
        if (
            re.fullmatch(r"[a-f0-9]{64}", self.sha256) is None
            or type(self.decoder_threads) is not int
            or not 0 <= self.decoder_threads <= MAX_PROBE_DECODER_THREADS
        ):
            raise ReplayError("Frame probe requires SHA256 and bounded decoder threads")

    @classmethod
    def from_reference(cls, value: object, *, decoder_threads: object = 0) -> "FrameProbeTool":
        row = record(value)
        digest = row.get("sha256")
        if not isinstance(digest, str) or type(decoder_threads) is not int:
            raise ReplayError("Frame probe tool requires a pinned executable and integer threads")
        return cls(_path(row.get("path")), digest, decoder_threads)

    def verify(self) -> None:
        if not self.path.is_file() or not os.access(self.path, os.X_OK):
            raise ReplayError("Frame probe tool must be an executable regular file")
        if file_digest(self.path) != self.sha256:
            raise ReplayError("Frame probe executable hash changed")

    def provenance(self) -> dict[str, object]:
        return {
            "version": 1,
            "executable": {"path": str(self.path), "sha256": self.sha256},
            "decoderThreads": self.decoder_threads,
        }


def _probe_provenance(proof: dict[str, object]) -> FrameProbeTool | None:
    """Missing historical tool evidence is unknown, never inferred or rewritten."""
    if "probeProvenance" not in proof:
        return None
    row = record(proof["probeProvenance"])
    if type(row.get("version")) is not int or row.get("version") != 1:
        raise ReplayError("Unsupported frame probe provenance")
    return FrameProbeTool.from_reference(
        row.get("executable"), decoder_threads=row.get("decoderThreads")
    )


@dataclass(frozen=True)
class Timeline:
    manifest: Path
    semantic: Path
    format: ET.Element
    frame: Fraction
    duration: Fraction
    parts: tuple[dict[str, object], ...]


def _reference(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_digest(path)}


def _save(path: Path, document: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def timeline(path: Path) -> Timeline:
    raw = _json(path)
    semantic = _pin(raw.get("source"))
    rows = raw.get("parts")
    if raw.get("version") != 1 or not isinstance(rows, list) or not 1 <= len(rows) <= 1024:
        raise ReplayError("Expected bounded partition manifest")
    expected = caption_cues(semantic)
    root = ET.parse(semantic).getroot()
    sequence = _required(root, "./library/event/project/sequence")
    formats = [
        item
        for item in root.findall("./resources/format")
        if item.get("id") == sequence.get("format")
    ]
    if len(formats) != 1:
        raise ReplayError("Expected one assembly format")
    frame = _time(formats[0].attrib, "frameDuration")
    duration = _time(sequence.attrib, "duration")
    if frame <= 0 or _time({"duration": str(raw.get("duration"))}, "duration") != duration:
        raise ReplayError("Partition duration differs from semantic timeline")
    cursor = Fraction(0)
    cues: list[Cue] = []
    parts = tuple(record(row) for row in rows)
    for row in parts:
        part = _pin(row)
        start = _time({"offset": str(row.get("globalStart"))}, "offset")
        length = _time({"duration": str(row.get("duration"))}, "duration")
        if start != cursor or length <= 0 or (length / frame).denominator != 1:
            raise ReplayError("Partition frames are discontinuous")
        part_root = ET.parse(part).getroot()
        part_sequence = _required(part_root, "./library/event/project/sequence")
        if _time(part_sequence.attrib, "duration") != length:
            raise ReplayError("Part sequence duration differs")
        if row.get("titles") != 0:
            cues.extend(
                (left + start, right + start, text) for left, right, text in caption_cues(part)
            )
        cursor += length
    if cursor != duration or _merged(cues) != expected:
        raise ReplayError("Partition does not reproduce semantic caption clock")
    return Timeline(path, semantic, formats[0], frame, duration, parts)


def prepare(
    manifest: Path,
    media_directory: Path,
    voice: Path,
    output: Path,
    *,
    voice_receipt: Path | None = None,
) -> None:
    plan = timeline(manifest)
    media_directory = _path(str(media_directory))
    _path(str(voice))
    _voice_info(voice, duration=plan.duration)
    _save(
        output,
        {
            "version": 1,
            "partition": _reference(manifest),
            "voice": _reference(voice),
            "voiceReceipt": _reference(voice_receipt) if voice_receipt is not None else None,
            "parts": [
                {
                    "movie": str(media_directory / f"part-{index + 1:04d}-master.mov"),
                    "proof": str(
                        output.resolve().parent / f"assembly-part-{index + 1:04d}-proof.json"
                    ),
                }
                for index in range(len(plan.parts))
            ],
        },
    )


def _config(path: Path) -> tuple[Timeline, dict[str, object], list[object]]:
    raw = _json(path)
    plan = timeline(_pin(raw.get("partition")))
    rows = raw.get("parts")
    if raw.get("version") != 1 or not isinstance(rows, list) or len(rows) != len(plan.parts):
        raise ReplayError("Assembly must bind every ordered part")
    if raw.get("voiceReceipt") is not None:
        receipt = _json(_pin(raw["voiceReceipt"]))
        voice = _pin(raw.get("voice"))
        rate, channels = _voice_info(voice, duration=plan.duration)
        if (
            receipt.get("outputSHA256") != file_digest(voice)
            or receipt.get("sampleRate") != rate
            or receipt.get("channels") != channels
            or receipt.get("outputSamples") != plan.duration * rate
        ):
            raise ReplayError("Whole voice receipt differs from selected audio")
    return plan, raw, rows


def _check_video(raw: dict[str, object], plan: Timeline, index: int) -> None:
    streams = raw.get("streams")
    if not isinstance(streams, list):
        raise ReplayError("Expected video stream observations")
    videos = [record(row) for row in streams if record(row).get("codec_type") == "video"]
    length = _time({"duration": str(plan.parts[index].get("duration"))}, "duration")
    if len(videos) != 1:
        raise ReplayError("Expected one burned video stream")
    video = videos[0]
    try:
        valid = (
            video.get("width") == int(plan.format.get("width", "0"))
            and video.get("height") == int(plan.format.get("height", "0"))
            and video.get("start_pts") == 0
            and Fraction(str(video.get("avg_frame_rate"))) == 1 / plan.frame
            and Fraction(str(video.get("duration_ts"))) * Fraction(str(video.get("time_base")))
            == length
            and video.get("nb_read_frames") == str((length / plan.frame).numerator)
        )
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Invalid observed video clock") from error
    if not valid:
        raise ReplayError("Burned master frame clock or geometry differs")
    frames = raw.get("frames")
    if not isinstance(frames, list) or len(frames) != (length / plan.frame).numerator:
        raise ReplayError("Expected every decoded frame timestamp")
    base = Fraction(str(video.get("time_base")))
    for index, value in enumerate(frames):
        timestamp = record(value).get("best_effort_timestamp")
        if type(timestamp) is not int or timestamp * base != index * plan.frame:
            raise ReplayError("Decoded master frame timestamps are not the exact output clock")


def probe(config: Path, *, part: int, tool: FrameProbeTool) -> Path:
    plan, raw, rows = _config(config)
    if not 1 <= part <= len(rows):
        raise ReplayError("Part index outside assembly")
    row = record(rows[part - 1])
    movie, output = _path(row.get("movie")), _path(row.get("proof"))
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ReplayError("Frame proof output must be absent in an existing directory")
    tool.verify()
    media = _reference(movie)
    result = subprocess.run(
        [
            str(tool.path),
            "-v",
            "error",
            "-threads",
            str(tool.decoder_threads),
            "-count_frames",
            "-show_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type,width,height,start_pts,avg_frame_rate,duration_ts,time_base,nb_read_frames:frame=best_effort_timestamp",
            "-of",
            "json",
            str(movie),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if result.returncode != 0 or len(result.stdout.encode("utf-8")) > MAX_JSON_BYTES:
        raise ReplayError("Burned master frame probe failed")
    try:
        observed = record(json.loads(result.stdout))
    except json.JSONDecodeError as error:
        raise ReplayError("Burned master frame probe returned invalid JSON") from error
    _check_video(observed, plan, part - 1)
    tool.verify()
    if _reference(movie) != media:
        raise ReplayError("Burned master changed during probe")
    _save(
        output,
        {
            "version": 1,
            "movie": media,
            "partition": raw["partition"],
            "part": part,
            "observed": observed,
            "audioIgnored": True,
            "probeProvenance": tool.provenance(),
        },
    )
    return output


def _voice_info(voice: Path, *, duration: Fraction) -> tuple[int, int]:
    with wave.open(str(voice), "rb") as audio:
        samples, rate, channels = audio.getnframes(), audio.getframerate(), audio.getnchannels()
    if not 1 <= channels <= 64 or rate <= 0 or Fraction(samples, rate) != duration:
        raise ReplayError("Whole processed voice does not match exact output clock")
    return rate, channels


def _master(
    plan: Timeline, config: dict[str, object], row: dict[str, object], *, index: int
) -> tuple[Path, Path]:
    proof_path = _path(row.get("proof"))
    proof = _json(proof_path)
    _probe_provenance(proof)
    if "crossRevisionReuse" in proof:
        from replay.fcp_part_reuse import verify_proof

        verify_proof(proof_path, partition=config["partition"], part=index + 1)
    movie = _pin(proof.get("movie"))
    if (
        proof.get("version") != 1
        or proof.get("part") != index + 1
        or proof.get("partition") != config["partition"]
        or str(movie) != row.get("movie")
    ):
        raise ReplayError("Master proof does not bind its ordered partition")
    _check_video(record(proof.get("observed")), plan, index)
    return movie, proof_path


def verify(config: Path, *, part: int) -> Path:
    """Verify an existing frame proof without probing or changing any artifact."""
    plan, raw, rows = _config(config)
    if not 1 <= part <= len(rows):
        raise ReplayError("Part index outside assembly")
    return _master(plan, raw, record(rows[part - 1]), index=part - 1)[1]


def assemble(config: Path, output: Path, *, name: str) -> Path:
    plan, raw, rows = _config(config)
    voice = _pin(raw.get("voice"))
    rate, channels = _voice_info(voice, duration=plan.duration)
    root = ET.Element("fcpxml", version="1.11")
    resources = ET.SubElement(root, "resources")
    format_node = ET.fromstring(ET.tostring(plan.format))
    format_node.set("id", "format")
    resources.append(format_node)
    project = ET.SubElement(
        ET.SubElement(ET.SubElement(root, "library"), "event", name=name), "project", name=name
    )
    sequence = ET.SubElement(
        project,
        "sequence",
        format="format",
        duration=_stamp(plan.duration),
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = ET.SubElement(sequence, "spine")
    proofs: list[dict[str, str]] = []
    for index, value in enumerate(rows):
        row = record(value)
        movie, proof_path = _master(plan, raw, row, index=index)
        proofs.append(_reference(proof_path))
        duration = str(plan.parts[index]["duration"])
        identifier = f"master{index + 1}"
        asset = ET.SubElement(
            resources,
            "asset",
            id=identifier,
            name=movie.name,
            start="0s",
            duration=duration,
            hasVideo="1",
            hasAudio="0",
            format="format",
        )
        ET.SubElement(asset, "media-rep", kind="original-media", src=movie.as_uri())
        ET.SubElement(
            spine,
            "asset-clip",
            ref=identifier,
            name=movie.stem,
            offset=str(plan.parts[index]["globalStart"]),
            start="0s",
            duration=duration,
            srcEnable="video",
        )
    asset = ET.SubElement(
        resources,
        "asset",
        id="voice",
        name="Processed voice",
        start="0s",
        duration=_stamp(plan.duration),
        hasVideo="0",
        hasAudio="1",
        audioSources="1",
        audioChannels=str(channels),
        audioRate=str(rate),
    )
    ET.SubElement(asset, "media-rep", kind="original-media", src=voice.as_uri())
    ET.SubElement(
        spine[0],
        "asset-clip",
        ref="voice",
        lane="-1",
        name="Processed voice",
        offset="0s",
        start="0s",
        duration=_stamp(plan.duration),
        srcEnable="audio",
    )
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with output.open("xb") as handle:
        handle.write(content)
    receipt = output.with_suffix(".lineage.json")
    _save(
        receipt,
        {
            "version": 1,
            "candidateOnly": True,
            "assemblyXML": _reference(output),
            "partition": raw["partition"],
            "semanticXML": _reference(plan.semantic),
            "semanticSRTSHA256": hashlib.sha256(srt_text(plan.semantic).encode()).hexdigest(),
            "masterProofs": proofs,
            "voice": raw["voice"],
            "voiceReceipt": raw.get("voiceReceipt"),
            "duration": _stamp(plan.duration),
            "voiceOutputStart": "0s",
            "partAudioDisabled": True,
            "nativeAssemblyRenderVerified": False,
        },
    )
    return receipt


def verify_assembly(config: Path, output: Path) -> None:
    """Check the assembled XML and immutable caption/media lineage without rendering."""
    plan, raw, rows = _config(config)
    voice = _pin(raw.get("voice"))
    rate, channels = _voice_info(voice, duration=plan.duration)
    receipt = _json(output.with_suffix(".lineage.json"))
    if (
        receipt.get("assemblyXML") != _reference(output)
        or receipt.get("partition") != raw["partition"]
        or receipt.get("semanticXML") != _reference(plan.semantic)
        or receipt.get("voice") != raw["voice"]
        or receipt.get("voiceReceipt") != raw.get("voiceReceipt")
        or receipt.get("semanticSRTSHA256")
        != hashlib.sha256(srt_text(plan.semantic).encode()).hexdigest()
        or receipt.get("partAudioDisabled") is not True
        or receipt.get("voiceOutputStart") != "0s"
    ):
        raise ReplayError("Assembly lineage changed")
    root = ET.parse(output).getroot()
    sequence = _required(root, "./library/event/project/sequence")
    spine = _required(sequence, "./spine")
    formats = root.findall("./resources/format")
    expected_format = dict(plan.format.attrib) | {"id": "format"}
    if (
        _time(sequence.attrib, "duration") != plan.duration
        or sequence.get("tcStart") != "0s"
        or sequence.get("format") != "format"
        or len(formats) != 1
        or formats[0].attrib != expected_format
        or len(spine) != len(rows)
        or root.findall(".//title")
    ):
        raise ReplayError("Assembly timeline differs from burned part plan")
    assets = {item.get("id"): item for item in root.findall("./resources/asset")}
    if len(assets) != len(rows) + 1:
        raise ReplayError("Unexpected assembly media resources")
    proofs: list[dict[str, str]] = []
    for index, value in enumerate(rows):
        movie, proof = _master(plan, raw, record(value), index=index)
        proofs.append(_reference(proof))
        clip = spine[index]
        asset = assets.get(clip.get("ref"))
        if (
            clip.tag != "asset-clip"
            or clip.get("start") != "0s"
            or clip.get("offset") != plan.parts[index]["globalStart"]
            or clip.get("duration") != plan.parts[index]["duration"]
            or clip.get("srcEnable") != "video"
            or len(clip) != (1 if index == 0 else 0)
            or asset is None
            or asset.get("hasAudio") != "0"
            or asset.get("hasVideo") != "1"
            or asset.get("format") != "format"
            or asset.get("start") != "0s"
            or asset.get("duration") != plan.parts[index]["duration"]
            or _required(asset, "media-rep").get("src") != movie.as_uri()
        ):
            raise ReplayError("Assembly clip frame/source clock or disabled audio changed")
    audio = _required(spine[0], "asset-clip")
    voice_asset = assets.get(audio.get("ref"))
    if (
        audio.get("offset") != "0s"
        or audio.get("start") != "0s"
        or audio.get("duration") != _stamp(plan.duration)
        or audio.get("srcEnable") != "audio"
        or audio.get("lane") != "-1"
        or voice_asset is None
        or voice_asset.get("hasAudio") != "1"
        or voice_asset.get("hasVideo") != "0"
        or voice_asset.get("start") != "0s"
        or voice_asset.get("duration") != _stamp(plan.duration)
        or voice_asset.get("audioRate") != str(rate)
        or voice_asset.get("audioChannels") != str(channels)
        or _required(voice_asset, "media-rep").get("src") != voice.as_uri()
        or receipt.get("masterProofs") != proofs
    ):
        raise ReplayError("Assembly whole-voice clock or master proof lineage changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("manifest", type=Path)
    prepare_parser.add_argument("media_directory", type=Path)
    prepare_parser.add_argument("voice", type=Path)
    prepare_parser.add_argument("output", type=Path)
    prepare_parser.add_argument("--voice-receipt", type=Path)
    probe_parser = sub.add_parser("probe")
    probe_parser.add_argument("config", type=Path)
    probe_parser.add_argument("--part", type=int, required=True)
    probe_parser.add_argument("--ffprobe", type=Path, required=True)
    probe_parser.add_argument("--ffprobe-sha256", required=True)
    probe_parser.add_argument("--decoder-threads", type=int, default=0)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("config", type=Path)
    verify_parser.add_argument("--part", type=int, required=True)
    assembly_parser = sub.add_parser("assemble")
    assembly_parser.add_argument("config", type=Path)
    assembly_parser.add_argument("output", type=Path)
    assembly_parser.add_argument("--name", required=True)
    final_verify = sub.add_parser("verify-assembly")
    final_verify.add_argument("config", type=Path)
    final_verify.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.manifest,
            args.media_directory,
            args.voice,
            args.output,
            voice_receipt=args.voice_receipt,
        )
    elif args.command == "probe":
        tool = FrameProbeTool(args.ffprobe, args.ffprobe_sha256, args.decoder_threads)
        print(probe(args.config, part=args.part, tool=tool))
    elif args.command == "verify":
        print(verify(args.config, part=args.part))
    elif args.command == "verify-assembly":
        verify_assembly(args.config, args.output)
    else:
        print(assemble(args.config, args.output, name=args.name))


if __name__ == "__main__":
    main()
