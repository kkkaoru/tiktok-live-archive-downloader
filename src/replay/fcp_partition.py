"""Partition immutable direct-spine FCPXML at existing clip boundaries only."""

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.core import ReplayError
from replay.fcp_caption_srt import Cue, _time, caption_cues
from replay.fcpxml_chapters import MAX_XML_BYTES, MAX_XML_ELEMENTS
from replay.stages import file_digest

MAX_PARTS = 1024


@dataclass(frozen=True)
class Part:
    start: Fraction
    duration: Fraction
    clips: int
    titles: int
    content: bytes


def _stamp(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}s"


def _required(root: ET.Element, path: str) -> ET.Element:
    value = root.find(path)
    if value is None:
        raise ReplayError(f"Missing FCP structure: {path}")
    return value


def _merged(cues: list[Cue]) -> tuple[Cue, ...]:
    result: list[Cue] = []
    for start, end, text in cues:
        if result and result[-1][1] == start and result[-1][2] == text:
            result[-1] = (result[-1][0], end, text)
        else:
            result.append((start, end, text))
    return tuple(result)


def _clip_check(clip: ET.Element, *, frame: Fraction) -> None:
    duration = _time(clip.attrib, "duration")
    start = _time(clip.attrib, "start")
    if clip.tag != "asset-clip" or any(
        (value / frame).denominator != 1 for value in (_time(clip.attrib, "offset"), duration)
    ):
        raise ReplayError("Partition requires direct frame-aligned source clips")
    for child in clip:
        if child.tag in {"asset-clip", "title", "audio", "video"}:
            offset = _time(child.attrib, "offset")
            end = offset + _time(child.attrib, "duration")
            if not start <= offset < end <= start + duration:
                raise ReplayError("Connected media extends outside its preserved source clip")


def _part(
    base: ET.Element,
    clips: list[ET.Element],
    *,
    number: int,
    library_directory: Path | None = None,
    library_name_prefix: str = "part",
) -> Part:
    root = copy.deepcopy(base)
    if library_directory is not None:
        _required(root, "./library").set(
            "location",
            (library_directory / f"{library_name_prefix}-{number:04d}.fcpbundle").as_uri(),
        )
    project = _required(root, "./library/event/project")
    project.set("name", f"{project.get('name', 'Project')} Part {number:04d}")
    project.attrib.pop("uid", None)
    sequence = _required(project, "./sequence")
    spine = _required(sequence, "./spine")
    start = _time(clips[0].attrib, "offset")
    end = _time(clips[-1].attrib, "offset") + _time(clips[-1].attrib, "duration")
    for clip in clips:
        member = copy.deepcopy(clip)
        member.set("offset", _stamp(_time(clip.attrib, "offset") - start))
        spine.append(member)
    sequence.set("duration", _stamp(end - start))
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if len(content) > MAX_XML_BYTES or sum(1 for _ in root.iter()) > MAX_XML_ELEMENTS:
        raise ReplayError("Partition exceeds native XML budget; reduce clip group size")
    return Part(start, end - start, len(clips), len(spine.findall("./asset-clip/title")), content)


def partition(
    path: Path,
    *,
    target_duration: Fraction,
    maximum_titles: int,
    library_directory: Path | None = None,
    library_name_prefix: str = "part",
) -> tuple[Part, ...]:
    """Keep every clip subtree intact; only top offsets/project metadata change."""
    if not 0 < target_duration <= 3600 or not 1 <= maximum_titles <= 10000:
        raise ReplayError("Expected bounded partition duration and title count")
    if not 1 <= len(library_name_prefix) <= 100 or any(
        value in library_name_prefix for value in ("/", "\\", "\n", "\r", "\x00")
    ):
        raise ReplayError("Expected bounded library name prefix")
    if library_directory is not None and (
        not library_directory.is_absolute()
        or library_directory.resolve() != library_directory
        or not library_directory.is_dir()
    ):
        raise ReplayError("Library directory must be an existing canonical absolute directory")
    caption_cues(path)  # Bounded XML, entity, complete clock and title-domain validation.
    root = ET.parse(path).getroot()
    sequence = _required(root, "./library/event/project/sequence")
    format_id = sequence.get("format")
    formats = root.findall("./resources/format")
    matching = [item for item in formats if item.get("id") == format_id]
    if len(matching) != 1:
        raise ReplayError("Expected one referenced sequence format")
    frame = _time(matching[0].attrib, "frameDuration")
    if frame <= 0:
        raise ReplayError("Expected positive frame duration")
    spine = _required(sequence, "./spine")
    clips = list(spine)
    if not clips:
        raise ReplayError("Expected source clips")
    groups: list[list[ET.Element]] = []
    current: list[ET.Element] = []
    duration = Fraction(0)
    titles = 0
    for clip in clips:
        _clip_check(clip, frame=frame)
        count = len(clip.findall("./title"))
        length = _time(clip.attrib, "duration")
        if count > maximum_titles:
            raise ReplayError("A source clip alone exceeds the title budget")
        if current and (duration + length > target_duration or titles + count > maximum_titles):
            groups.append(current)
            current, duration, titles = [], Fraction(0), 0
        current.append(clip)
        duration += length
        titles += count
    groups.append(current)
    if len(groups) > MAX_PARTS:
        raise ReplayError("Partition count exceeds bounded output")
    for clip in clips:
        spine.remove(clip)
    return tuple(
        _part(
            root,
            group,
            number=index + 1,
            library_directory=library_directory,
            library_name_prefix=library_name_prefix,
        )
        for index, group in enumerate(groups)
    )


def write_parts(
    path: Path,
    output: Path,
    *,
    target_duration: Fraction,
    maximum_titles: int,
    library_directory: Path | None = None,
    library_name_prefix: str = "part",
) -> Path:
    """Write new projects and prove their global display cues equal the original."""
    if output.absolute() != output.resolve() or output.exists():
        raise ReplayError("Partition output must be a new canonical directory")
    digest = file_digest(path)
    parts = partition(
        path,
        target_duration=target_duration,
        maximum_titles=maximum_titles,
        library_directory=library_directory,
        library_name_prefix=library_name_prefix,
    )
    expected = caption_cues(path)
    output.mkdir(mode=0o700)
    rows: list[dict[str, object]] = []
    reconstructed: list[Cue] = []
    for index, part in enumerate(parts):
        target = output / f"part-{index + 1:04d}.fcpxml"
        with target.open("xb") as handle:
            handle.write(part.content)
        cues = caption_cues(target) if part.titles else ()
        reconstructed.extend(
            (left + part.start, right + part.start, text) for left, right, text in cues
        )
        rows.append(
            {
                "path": str(target.resolve()),
                "sha256": hashlib.sha256(part.content).hexdigest(),
                "globalStart": _stamp(part.start),
                "duration": _stamp(part.duration),
                "clips": part.clips,
                "titles": part.titles,
            }
        )
    if _merged(reconstructed) != expected or file_digest(path) != digest:
        raise ReplayError("Partition changed global captions or original XML")
    manifest = output / "manifest.json"
    document = {
        "version": 1,
        "candidateOnly": True,
        "source": {"path": str(path.resolve()), "sha256": digest},
        "duration": _stamp(sum((part.duration for part in parts), Fraction(0))),
        "parts": rows,
        "globalCaptionCuesEqual": True,
        "clipSubtreesPreservedExceptTopOffset": True,
        "nativeRenderVerified": False,
        "libraryDirectory": None if library_directory is None else str(library_directory),
        "libraryCreationVerified": False,
        "libraryNamePrefix": library_name_prefix,
    }
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-duration", type=Fraction, required=True)
    parser.add_argument("--maximum-titles", type=int, required=True)
    parser.add_argument("--library-directory", type=Path)
    parser.add_argument("--library-name-prefix", default="part")
    args = parser.parse_args()
    result = write_parts(
        args.source,
        args.output,
        target_duration=args.target_duration,
        maximum_titles=args.maximum_titles,
        library_directory=args.library_directory,
        library_name_prefix=args.library_name_prefix,
    )
    print(result)


if __name__ == "__main__":
    main()
