"""Reuse verified uncut cleanup media without changing semantic title clocks."""

import argparse
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.core import ReplayError
from replay.delivery_source_assembly import load_chunks
from replay.fcp_caption_srt import _time, caption_cues
from replay.fcp_part_assembly import _json, _pin, _reference, _save, _voice_info
from replay.fcp_partition import _required, _stamp
from replay.fcp_project import GAUSSIAN_UID
from replay.fcp_proxy_binding import bind as bind_proxy
from replay.frame_cell_extent import terminal_cell_receipt, validate_terminal_cell
from replay.recordings import record
from replay.stages import file_digest


def clean_source(config: dict[str, object]) -> tuple[Path, Path, Fraction]:
    """Validate the saved exact-clock assembly and historical producer correspondence."""
    manifest = _pin(config.get("assemblyManifest"))
    receipt_path = _pin(config.get("assemblyReceipt"))
    index_path = _pin(config.get("cleanIndex"))
    receipt = _json(receipt_path)
    if (
        receipt.get("exactPresentationClockVerified") is not True
        or receipt.get("compressedPacketsIdentical") is not True
        or receipt.get("indexSHA256") != file_digest(index_path)
    ):
        raise ReplayError("Cleanup assembly lacks exact source-clock evidence")
    bindings: dict[str, str] = {}
    chunks = load_chunks(manifest, sha256=file_digest(manifest), bindings=bindings)
    recorded = record(receipt.get("inputs"))
    if not chunks or any(recorded.get(path) != digest for path, digest in bindings.items()):
        raise ReplayError("Cleanup assembly differs from pinned producer inputs")
    evidence = record(receipt.get("probeEvidence"))
    if not evidence:
        raise ReplayError("Cleanup assembly has no frame/packet evidence")
    for path, digest in evidence.items():
        _pin({"path": path, "sha256": digest})
    clean = _pin(receipt.get("output"))
    rows = _json(index_path).get("ranges")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ReplayError("Require one uncut cleanup source")
    row = record(rows[0])
    first, last = chunks[0].clock, chunks[-1].clock
    if (
        row.get("media") != str(clean)
        or row.get("sha256") != record(receipt["output"]).get("sha256")
        or row.get("sourceFirstPTS") != first.pts[0]
        or row.get("sourceEndPTS") != last.pts[-1] + last.last_duration
        or row.get("frames") != sum(len(chunk.clock.pts) for chunk in chunks)
        or receipt.get("frames") != row.get("frames")
        or row.get("timeBase") != str(first.time_base)
        or first.pts[0] != 0
    ):
        raise ReplayError("Cleanup index is not the complete uncut source clock")
    original = Path(chunks[0].source[0])
    # Historical producer bytes are evidence, never an executable backend.
    return original, clean, (last.pts[-1] + last.last_duration) * last.time_base


def replace_media(
    root: ET.Element,
    *,
    original: Path,
    clean: Path,
    duration: Fraction,
    voice: Path,
    remove_source_masks: bool,
    allow_terminal_cell: bool = False,
) -> int:
    """Replace only the original video asset and explicitly recognized blur overlays."""
    resources = _required(root, "./resources")
    assets = resources.findall("asset")
    sources = [a for a in assets if a.find(f"media-rep[@src='{original.as_uri()}']") is not None]
    if len(sources) != 1:
        raise ReplayError("Semantic XML must reference the cleanup producer's one original")
    source = sources[0]
    source_id = source.get("id")
    if source_id is None or source.get("start", "0s") != "0s":
        raise ReplayError("Unsupported original resource clock")
    source.set("hasAudio", "0")
    for key in ("audioSources", "audioChannels", "audioRate"):
        source.attrib.pop(key, None)
    source.set("duration", _stamp(duration))
    _required(source, "media-rep").set("src", clean.as_uri())
    gaussian = {e.get("id") for e in resources.findall("effect") if e.get("uid") == GAUSSIAN_UID}
    spine = _required(root, "./library/event/project/sequence/spine")
    sequence = _required(root, "./library/event/project/sequence")
    formats = [
        item for item in resources.findall("format") if item.get("id") == sequence.get("format")
    ]
    if len(formats) != 1:
        raise ReplayError("Expected one display frame format")
    frame_duration = _time(formats[0].attrib, "frameDuration")
    voice_ids: set[str] = set()
    removed = 0
    for clip in spine:
        if clip.tag != "asset-clip" or clip.get("ref") != source_id:
            raise ReplayError("Require a direct single-source primary timeline")
        validate_terminal_cell(
            start=_time(clip.attrib, "start"),
            end=_time(clip.attrib, "start") + _time(clip.attrib, "duration"),
            physical_end=duration,
            frame_duration=frame_duration,
            allow_terminal_cell=allow_terminal_cell,
        )
        clip.set("srcEnable", "video")
        for child in list(clip):
            if child.tag == "audio-channel-source" and child.get("active") != "0":
                raise ReplayError("Active original source audio is unsupported")
            if child.tag != "asset-clip":
                continue
            if child.get("srcEnable") == "audio":
                identifier = child.get("ref")
                if identifier is None:
                    raise ReplayError("Missing processed voice reference")
                voice_ids.add(identifier)
                continue
            filters = child.findall("filter-video")
            if (
                not remove_source_masks
                or child.get("ref") != source_id
                or child.get("srcEnable") != "video"
                or child.get("lane") != "1"
                or child.find("adjust-crop/trim-rect") is None
                or len(filters) != 1
                or filters[0].get("ref") not in gaussian
                or any(
                    e.tag not in {"adjust-crop", "filter-video"}
                    and not (e.tag == "audio-channel-source" and e.get("active") == "0")
                    for e in child
                )
            ):
                raise ReplayError("Unrecognized source overlay; refusing silent removal")
            clip.remove(child)
            removed += 1
    voices = [a for a in assets if a.get("id") in voice_ids]
    if len(voice_ids) != 1 or len(voices) != 1 or voices[0].get("hasVideo") != "0":
        raise ReplayError("Require one existing processed voice resource")
    _required(voices[0], "media-rep").set("src", voice.as_uri())
    return removed


def transform(config_path: Path, output: Path, *, name: str) -> Path:
    config = _json(config_path)
    if config.get("version") != 1 or config.get("removeSourceMasks") is not True:
        raise ReplayError("Require explicit source-mask replacement intent")
    terminal_policy = config.get("terminalFrameCell", False)
    if not isinstance(terminal_policy, bool):
        raise ReplayError("Terminal display-cell policy must be an explicit boolean")
    semantic = _pin(config.get("semanticXML"))
    original_cues = caption_cues(semantic)
    original, clean, duration = clean_source(config)
    voice = _pin(config.get("voice"))
    voice_receipt_path = _pin(config.get("voiceReceipt"))
    root = ET.parse(semantic).getroot()
    proxy_correspondence = None
    if "proxyBinding" in config:
        original, proxy_correspondence = bind_proxy(root, config, clean=clean)
    sequence = _required(root, "./library/event/project/sequence")
    total = _time(sequence.attrib, "duration")
    rate, channels = _voice_info(voice, duration=total)
    audio_receipt = _json(voice_receipt_path)
    if (
        audio_receipt.get("outputSHA256") != record(config["voice"]).get("sha256")
        or audio_receipt.get("sampleRate") != rate
        or audio_receipt.get("channels") != channels
        or audio_receipt.get("outputSamples") != total * rate
    ):
        raise ReplayError("Processed voice receipt does not match semantic output clock")
    removed = replace_media(
        root,
        original=original,
        clean=clean,
        duration=duration,
        voice=voice,
        remove_source_masks=True,
        allow_terminal_cell=terminal_policy,
    )
    expected = original_cues
    selected = config.get("outputWindow")
    if selected is not None:
        if (
            not isinstance(selected, list)
            or len(selected) != 2
            or not all(isinstance(v, str) for v in selected)
        ):
            raise ReplayError("Expected rational output window")
        start, end = (_time({"time": str(value)}, "time") for value in selected)
        from replay.fcp_output_window import window

        root = window(root, start, end)
        expected = tuple(
            (max(a, start) - start, min(b, end) - start, text)
            for a, b, text in original_cues
            if a < end and b > start
        )
    project = _required(root, "./library/event/project")
    project.set("name", name)
    project.attrib.pop("uid", None)
    with output.open("xb") as handle:
        handle.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    if caption_cues(output) != expected:
        raise ReplayError("Transformed timeline changed semantic caption text or clock")
    receipt_path = output.with_suffix(".source-lineage.json")
    _save(
        receipt_path,
        {
            "version": 1,
            "candidateOnly": True,
            "config": _reference(config_path),
            "semanticXML": config["semanticXML"],
            "outputXML": _reference(output),
            "cleanSource": _reference(clean),
            "voice": config["voice"],
            "voiceReceipt": config["voiceReceipt"],
            "removedSourceMasks": removed,
            "captionClockEqual": True,
            "outputWindow": selected,
            "sourceCaptionQualityAccepted": False,
            "nativeRenderVerified": False,
            "proxyCorrespondence": proxy_correspondence,
            "terminalFrameCell": None
            if not terminal_policy
            else terminal_cell_receipt(
                physical_end=duration,
                frame_duration=_time(
                    _required(root, f"./resources/format[@id='{sequence.get('format')}']").attrib,
                    "frameDuration",
                ),
            ),
        },
    )
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.config, args.output)
    else:
        print(transform(args.config, args.output, name=args.name))


def verify(config_path: Path, output: Path) -> None:
    """Verify immutable generated XML and its clean-media/voice bindings."""
    receipt = _json(output.with_suffix(".source-lineage.json"))
    config = _json(config_path)
    if (
        receipt.get("config") != _reference(config_path)
        or receipt.get("outputXML") != _reference(output)
        or receipt.get("semanticXML") != config.get("semanticXML")
        or receipt.get("voice") != config.get("voice")
        or receipt.get("voiceReceipt") != config.get("voiceReceipt")
        or receipt.get("captionClockEqual") is not True
    ):
        raise ReplayError("Transformed source lineage changed")
    for key in ("semanticXML", "voice", "voiceReceipt", "cleanSource"):
        _pin(receipt.get(key))
    if "proxyBinding" in config:
        binding = record(config["proxyBinding"])
        correspondence = record(receipt.get("proxyCorrespondence"))
        if correspondence.get("binding") != binding:
            raise ReplayError("Proxy correspondence lineage changed")
        for key in ("original", "proxy", "proxyClock", "sourceConfig"):
            _pin(binding.get(key))
    clean_source(config)
    caption_cues(output)


if __name__ == "__main__":
    main()
