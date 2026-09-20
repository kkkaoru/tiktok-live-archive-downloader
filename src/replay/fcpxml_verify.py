"""Mechanical FCP media/timeline checks, separate from DTD and import fidelity."""

from dataclasses import dataclass
from fractions import Fraction
from math import isclose, isfinite, log10
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

from .fcpxml_chapters import MAX_XML_BYTES, MAX_XML_ELEMENTS


@dataclass(frozen=True, slots=True)
class AssetReference:
    identifier: str
    path: Path
    duration: Fraction
    proxy: Path | None = None


@dataclass(frozen=True, slots=True)
class FCPReport:
    first_frame: int
    frames: int
    clips: int
    titles: int


def clock(value: str) -> Fraction:
    if not value.endswith("s"):
        raise ValueError("Invalid FCP clock")
    return Fraction(value[:-1])


def _media_url(value: str, expected: Path) -> None:
    url = urlsplit(value)
    if url.scheme != "file" or url.netloc or url.query or url.fragment:
        raise ValueError("Unexpected media URL")
    path = Path(unquote(url.path))
    if not path.is_file() or path.resolve() != expected.resolve():
        raise ValueError("Missing or changed media reference")


def _resources(root: ET.Element, assets: tuple[AssetReference, ...], effect: Path) -> None:
    actual = root.findall("resources/asset")
    if {asset.get("id") for asset in actual} != {asset.identifier for asset in assets} or len(
        actual
    ) != len(assets):
        raise ValueError("Resource identities changed")
    for expected in assets:
        element = next(asset for asset in actual if asset.get("id") == expected.identifier)
        if (
            clock(element.attrib["duration"]) != expected.duration
            or clock(element.attrib["start"]) != 0
        ):
            raise ValueError("Resource clock changed")
        references = {
            item.attrib["kind"]: item.attrib["src"] for item in element.findall("media-rep")
        }
        wanted = {"original-media": expected.path}
        if expected.proxy is not None:
            wanted["proxy-media"] = expected.proxy
        if references.keys() != wanted.keys() or len(element.findall("media-rep")) != len(wanted):
            raise ValueError("Resource representations changed")
        for kind, path in wanted.items():
            _media_url(references[kind], path)
    title = root.find("resources/effect")
    if title is None or title.get("id") != "r5":
        raise ValueError("Missing title effect")
    _media_url(title.attrib["src"], effect)


def _attachments(
    parent: ET.Element, *, gain: float, start: Fraction, duration: Fraction, output: Fraction
) -> None:
    voices = 0
    for child in parent.findall("asset-clip"):
        if clock(child.attrib["offset"]) != start or clock(child.attrib["duration"]) != duration:
            raise ValueError("Attached media alignment changed")
        reference = child.get("ref")
        if reference == "r3":
            voices += 1
            adjustment = child.find("adjust-volume")
            if (
                adjustment is None
                or clock(child.attrib["start"]) != start
                or child.get("srcEnable") != "audio"
            ):
                raise ValueError("Voice selection changed")
            amount = float(adjustment.attrib["amount"].removesuffix("dB"))
            if not isclose(amount, 20 * log10(gain), abs_tol=1e-9):
                raise ValueError("Voice gain differs from encoded output")
        elif reference == "r4":
            if clock(child.attrib["start"]) != output or child.get("srcEnable") != "video":
                raise ValueError("Rendered-band selection changed")
        else:
            raise ValueError("Unexpected attached media")
    if voices != 1:
        raise ValueError("Expected exactly one separated voice")


def verify_fcpxml(
    content: bytes,
    *,
    assets: tuple[AssetReference, ...],
    effect: Path,
    total_frames: int,
    gain: float,
) -> FCPReport:
    """Verify existing references and precise selection clocks; never claim import."""
    if len(content) > MAX_XML_BYTES or b"<!DOCTYPE" in content or b"<!ENTITY" in content:
        raise ValueError("Unsafe or oversized FCPXML")
    if not isfinite(gain) or gain <= 0 or total_frames <= 0:
        raise ValueError("Invalid output settings")
    root = ET.fromstring(content.decode("utf-8"))
    if root.tag != "fcpxml" or len(list(root.iter())) > MAX_XML_ELEMENTS:
        raise ValueError("Invalid FCPXML structure")
    _resources(root, assets, effect)
    durations = {asset.identifier: asset.duration for asset in assets}
    sequences = root.findall(".//sequence")
    if len(sequences) != 1:
        raise ValueError("Expected one sequence")
    sequence = sequences[0]
    length, first = clock(sequence.attrib["duration"]), clock(sequence.attrib["tcStart"])
    if (
        min(length, first) < 0
        or (length * 30).denominator != 1
        or (first * 30).denominator != 1
        or first + length > Fraction(total_frames, 30)
    ):
        raise ValueError("Invalid chapter clock")
    cursor = Fraction(0)
    title_count = 0
    clips = sequence.findall("spine/asset-clip")
    for clip in clips:
        start, duration = clock(clip.attrib["start"]), clock(clip.attrib["duration"])
        if (
            clip.get("ref") != "r2"
            or clip.get("srcEnable") != "video"
            or clock(clip.attrib["offset"]) != cursor
            or start < 0
            or duration <= 0
            or start + duration > min(durations["r2"], durations["r3"])
        ):
            raise ValueError("Invalid primary selection")
        _attachments(clip, gain=gain, start=start, duration=duration, output=cursor + first)
        for title in clip.findall("title"):
            title_start, title_duration = (
                clock(title.attrib["offset"]),
                clock(title.attrib["duration"]),
            )
            if (
                title.get("ref") != "r5"
                or title_start < start
                or title_duration <= 0
                or title_start + title_duration > start + duration
            ):
                raise ValueError("Title exceeds parent selection")
            title_count += 1
        cursor += duration
    if cursor != length or not clips or durations["r4"] != Fraction(total_frames, 30):
        raise ValueError("Sequence coverage changed")
    return FCPReport(int(first * 30), int(length * 30), len(clips), title_count)
