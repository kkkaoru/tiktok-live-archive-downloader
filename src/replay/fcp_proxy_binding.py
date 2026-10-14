"""Bound explicit CFR proxy correspondence to cleanup, retaining candidate limits."""

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _pin
from replay.core import ReplayError
from replay.delivery_media import FrameClock, read_clock
from replay.delivery_source_assembly import load_chunks
from replay.fcp_caption_srt import _time
from replay.fcp_partition import _required
from replay.proxy_clock_correspondence import verify_clocks
from replay.recordings import record
from replay.stages import file_digest

SAMPLE_WIDTH = 180
SAMPLE_HEIGHT = 320
SAMPLE_COUNT = 5
MINIMUM_CORRELATION = 0.97
MAXIMUM_MEAN_ERROR = 8.0
PROBE_TIMEOUT = 120


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int


def geometry(path: Path) -> Geometry:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,sample_aspect_ratio",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=PROBE_TIMEOUT,
    )
    data = record(json.loads(result.stdout))
    streams = data.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ReplayError("Proxy geometry requires one video stream")
    row = record(streams[0])
    width, height = row.get("width"), row.get("height")
    if (
        type(width) is not int
        or type(height) is not int
        or min(width, height) <= 0
        or max(width, height) > 16384
        or row.get("sample_aspect_ratio", "1:1") not in {"1:1", "N/A"}
    ):
        raise ReplayError("Proxy binding requires bounded square-pixel geometry")
    return Geometry(width, height)


def sample(path: Path, time: Fraction) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{float(time):.9f}",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-vf",
            f"scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT}:flags=area,format=gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=PROBE_TIMEOUT,
    )
    if len(result.stdout) != SAMPLE_WIDTH * SAMPLE_HEIGHT:
        raise ReplayError("Proxy correspondence sample has unexpected dimensions")
    return result.stdout


def compare_samples(original: bytes, proxy: bytes) -> dict[str, object]:
    if len(original) != SAMPLE_WIDTH * SAMPLE_HEIGHT or len(proxy) != len(original):
        raise ReplayError("Proxy sample sizes differ")
    count = len(original)
    mean_a, mean_b = sum(original) / count, sum(proxy) / count
    variance_a = sum((v - mean_a) ** 2 for v in original)
    variance_b = sum((v - mean_b) ** 2 for v in proxy)
    if min(variance_a, variance_b) < count:
        raise ReplayError("Flat samples cannot establish proxy correspondence")
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(original, proxy, strict=True))
    correlation = covariance / math.sqrt(variance_a * variance_b)
    error = sum(abs(a - b) for a, b in zip(original, proxy, strict=True)) / count
    if correlation < MINIMUM_CORRELATION or error > MAXIMUM_MEAN_ERROR:
        raise ReplayError("Proxy visual correspondence sample differs")
    return {
        "originalSampleSHA256": hashlib.sha256(original).hexdigest(),
        "proxySampleSHA256": hashlib.sha256(proxy).hexdigest(),
        "correlation": correlation,
        "meanAbsoluteLumaError": error,
    }


def bind(
    root: ET.Element, config: dict[str, object], *, clean: Path
) -> tuple[Path, dict[str, object]]:
    """Validate and retarget only an explicitly pinned, sampled proxy candidate."""
    reference = record(config.get("proxyBinding"))
    if reference.get("policy") != "aspect-fit-cfr-candidate":
        raise ReplayError("Explicit proxy candidate correspondence policy is required")
    original, proxy = _pin(reference.get("original")), _pin(reference.get("proxy"))
    clock_path = _pin(reference.get("proxyClock"))
    source_config = _pin(reference.get("sourceConfig"))
    if _json(source_config).get("source") != str(original):
        raise ReplayError("Historical source configuration does not name the bound original")
    manifest = _pin(config.get("assemblyManifest"))
    chunks = load_chunks(manifest, sha256=file_digest(manifest), bindings={})
    if chunks[0].source[1] != record(reference["original"]).get("sha256"):
        raise ReplayError("Proxy original hash differs from cleanup producer")
    clock = FrameClock(
        tuple(value for chunk in chunks for value in chunk.clock.pts),
        chunks[0].clock.time_base,
        chunks[-1].clock.last_duration,
    )
    proxy_clock = read_clock(clock_path)
    correspondence = verify_clocks(clock, proxy_clock)
    source_geometry, proxy_geometry, clean_geometry = (
        geometry(original),
        geometry(proxy),
        geometry(clean),
    )
    if (
        source_geometry != clean_geometry
        or source_geometry.width * proxy_geometry.height
        != proxy_geometry.width * source_geometry.height
    ):
        raise ReplayError("Proxy binding requires unchanged aspect ratio and clean geometry")
    resources = _required(root, "./resources")
    sequence = _required(root, "./library/event/project/sequence")
    display = _required(resources, f"format[@id='{sequence.get('format')}']")
    interval = (proxy_clock.pts[1] - proxy_clock.pts[0]) * proxy_clock.time_base
    if (
        display.get("width") != str(proxy_geometry.width)
        or display.get("height") != str(proxy_geometry.height)
        or _time(display.attrib, "frameDuration") != interval
    ):
        raise ReplayError("Proxy canvas or clock differs from semantic sequence")
    assets = [
        a
        for a in resources.findall("asset")
        if a.find(f"media-rep[@src='{proxy.as_uri()}']") is not None
    ]
    if len(assets) != 1 or assets[0].get("start", "0s") != "0s":
        raise ReplayError("Proxy binding requires one zero-origin semantic asset")
    samples: list[dict[str, object]] = []
    for index in range(SAMPLE_COUNT):
        frame = (len(clock.pts) - 1) * (index + 1) // (SAMPLE_COUNT + 1)
        time = proxy_clock.pts[frame] * proxy_clock.time_base + interval / 2
        observation = compare_samples(sample(original, time), sample(proxy, time))
        samples.append({"frame": frame, "sampleTime": str(time), **observation})
    identifier = "proxyBoundCleanFormat"
    if resources.find(f"*[@id='{identifier}']") is not None:
        raise ReplayError("Proxy clean format identifier already exists")
    attributes = dict(display.attrib)
    attributes.update(
        id=identifier, width=str(clean_geometry.width), height=str(clean_geometry.height)
    )
    attributes.pop("name", None)
    resources.insert(0, ET.Element("format", attributes))
    assets[0].set("format", identifier)
    # replace_media continues to enforce its exact URI guard using this verified proxy.
    return proxy, {
        "candidateOnly": True,
        "binding": reference,
        "clock": correspondence,
        "samples": samples,
        "sampledVisualCorrespondence": True,
        "fullVideoPixelIdentityVerified": False,
        "proxyProducerReceiptAvailable": False,
        "captionClockChanged": False,
        "geometryRecovered": False,
        "cleanWidth": clean_geometry.width,
        "cleanHeight": clean_geometry.height,
    }
