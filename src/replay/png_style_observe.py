"""Observe an explicitly reviewed PNG canvas through its embedded ICC profile.

Uses installed LittleCMS transicc, never inferred transfer curves. Chromatic
support/erosion is a sampling heuristic, not recovered glyph alpha or approval.
"""

import argparse
import hashlib
import json
import struct
import subprocess
import zlib
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.caption_pixels import MAX_PIXELS, _bounds
from replay.motion_calibration_verify import Inputs
from replay.recordings import record
from replay.stages import file_digest

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_BYTES = 64 * 1024 * 1024
MAX_PROFILE_BYTES = 4 * 1024 * 1024
MAX_COLORS = 65536
CHROMA_MIN = 32


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def embedded_profile(data: bytes) -> tuple[str, bytes]:
    """Validate bounded RGB8 PNG chunks and extract, never assign, the ICC profile."""
    if not data.startswith(PNG_SIGNATURE) or len(data) > MAX_BYTES:
        raise ValueError("Expected bounded PNG")
    offset = len(PNG_SIGNATURE)
    profile: tuple[str, bytes] | None = None
    first = True
    ended = False
    while offset + 12 <= len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + size + 12
        if end > len(data):
            raise ValueError("Truncated PNG chunk")
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : end - 4]
        if data[offset:end] != chunk(kind, body):
            raise ValueError("PNG CRC mismatch")
        if first:
            if kind != b"IHDR" or len(body) != 13:
                raise ValueError("Missing PNG header")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if (
                not 0 < width * height <= MAX_PIXELS
                or depth != 8
                or color not in (2, 6)
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                raise ValueError("Expected bounded noninterlaced RGB8/RGBA8 PNG")
            first = False
        if kind == b"iCCP":
            name, separator, compressed = body.partition(b"\0")
            if profile is not None or not separator or not 1 <= len(name) <= 79:
                raise ValueError("Invalid or duplicate embedded profile")
            if not compressed.startswith(b"\0"):
                raise ValueError("Unsupported ICC compression")
            decoder = zlib.decompressobj()
            icc = decoder.decompress(compressed[1:], MAX_PROFILE_BYTES + 1)
            if len(icc) > MAX_PROFILE_BYTES or not decoder.eof or decoder.unused_data:
                raise ValueError("Invalid or oversized ICC stream")
            if len(icc) < 128 or icc[36:40] != b"acsp" or icc[16:20] != b"RGB ":
                raise ValueError("Expected embedded RGB ICC profile")
            if struct.unpack_from(">I", icc)[0] != len(icc):
                raise ValueError("ICC size mismatch")
            profile = name.decode("latin-1"), icc
        offset = end
        if kind == b"IEND":
            ended = True
            break
    if not ended or offset != len(data) or profile is None:
        raise ValueError("Incomplete PNG or missing embedded ICC; no sRGB assumption")
    return profile


def transform_palette(
    colors: NDArray[np.uint8], *, executable: Path, profile: Path, output: Path
) -> tuple[NDArray[np.float64], list[str]]:
    if colors.ndim != 2 or colors.shape[1] != 3 or not 0 < len(colors) <= MAX_COLORS:
        raise ValueError("Invalid or excessive palette")
    command = [str(executable), "-n", f"-i{profile}", "-o*sRGB", "-t1", "-c0"]
    text = "".join(f"{r} {g} {b}\n" for r, g, b in colors)
    (output / "transicc-input.txt").write_text(text, encoding="ascii")
    result = subprocess.run(command, input=text, capture_output=True, text=True, timeout=30)
    (output / "transicc-stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / "transicc-stderr.txt").write_text(result.stderr, encoding="utf-8")
    result.check_returncode()
    rows = [line.split() for line in result.stdout.splitlines()]
    if len(rows) != len(colors) or any(len(row) != 3 for row in rows):
        raise ValueError("Unexpected transicc result dimensions")
    mapped = np.array(rows, dtype=np.float64)
    if not np.isfinite(mapped).all():
        raise ValueError("Nonfinite colour conversion")
    return mapped, command


def distribution(rgb: NDArray[np.float64], mask: NDArray[np.bool_]) -> dict[str, object]:
    values = rgb[mask]
    if not len(values):
        return {"pixels": 0, "bounds": None}
    colors, counts = np.unique(values, axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return {
        "pixels": len(values),
        "bounds": _bounds(mask),
        "quantiles0_5_50_95_100": np.percentile(values, [0, 5, 50, 95, 100], axis=0).tolist(),
        "histogram": [{"srgb255": colors[i].tolist(), "count": int(counts[i])} for i in order],
    }


def support_observations(rgb: NDArray[np.float64]) -> list[dict[str, object]]:
    """Retain full distributions and every supported row, without target-colour selection."""
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    seed = (chroma >= CHROMA_MIN).astype(np.uint8)
    reports: list[dict[str, object]] = []
    for radius in (0, 1, 2):
        mask = cv2.erode(
            seed,
            np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.bool_)
        report = distribution(rgb, mask)
        report["erosionRadiusPixels"] = radius
        report["rows"] = [
            {"y": int(y), **distribution(rgb[y : y + 1], mask[y : y + 1])}
            for y in np.flatnonzero(np.any(mask, axis=1))
        ]
        reports.append(report)
    return reports


def observe(
    source: Path, output: Path, *, canvas: tuple[int, int, int, int], executable: Path
) -> dict[str, object]:
    inputs = Inputs()
    source = inputs.path(str(source))
    output = inputs.path(str(output))
    inputs.bind(source)
    inputs.bind(Path(__file__))
    if source.stat().st_size > MAX_BYTES:
        raise ValueError("PNG exceeds byte budget")
    tool = executable.resolve(strict=True)
    tool_hash = file_digest(tool)
    data = source.read_bytes()
    name, profile = embedded_profile(data)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("PNG decoding failed")
    left, top, right, bottom = canvas
    if not 0 <= left < right <= image.shape[1] or not 0 <= top < bottom <= image.shape[0]:
        raise ValueError("Reviewed canvas is outside PNG")
    if image.shape[2] == 4 and np.any(image[top:bottom, left:right, 3] != 255):
        raise ValueError("Transparent canvas: glyph opacity cannot be assumed")
    crop = np.array(image[top:bottom, left:right, :3][:, :, ::-1], dtype=np.uint8)
    colors, inverse = np.unique(crop.reshape(-1, 3), axis=0, return_inverse=True)
    output.mkdir(mode=0o700)
    profile_path = output / "source.icc"
    profile_path.write_bytes(profile)
    mapped, command = transform_palette(
        colors, executable=tool, profile=profile_path, output=output
    )
    rgb = mapped[inverse].reshape(crop.shape)
    encoded = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    success, png = cv2.imencode(".png", encoded[:, :, ::-1])
    if not success:
        raise ValueError("Managed canvas encoding failed")
    # OpenCV writes no colour profile: explicitly tag the managed new PNG as sRGB.
    png_bytes = png.tobytes()
    (output / "canvas-srgb.png").write_bytes(
        png_bytes[:33] + chunk(b"sRGB", b"\x01") + png_bytes[33:]
    )
    report: dict[str, object] = {
        "version": 1,
        "inputs": inputs.hashes,
        "sourceDimensions": [image.shape[1], image.shape[0]],
        "canvasXYXY": canvas,
        "canvasSelection": "caller-reviewed screenshot pixels; not calibrated project coordinates",
        "profileName": name,
        "profileSHA256": hashlib.sha256(profile).hexdigest(),
        "profileHeaderHex": profile[:128].hex(),
        "executable": str(tool),
        "executableSHA256": tool_hash,
        "command": command,
        "mapping": "embedded ICC to LittleCMS built-in sRGB, relative colorimetric, no BPC, c0",
        "opencvVersion": cv2.__version__,
        "numpyVersion": np.__version__,
        "palette": [
            {"sourceRGB8": color.tolist(), "srgb255": converted.tolist()}
            for color, converted in zip(colors, mapped, strict=True)
        ],
        "outOfRangeChannelsBeforePreviewClip": int(np.count_nonzero((rgb < 0) | (rgb > 255))),
        "sampling": {
            "chromaMinimum": CHROMA_MIN,
            "method": "max-min channel spread, square erosion radius 0/1/2; no intent matching",
            "limitation": (
                "screenshot opaque, glyph alpha unknown; erosion may retain blended pixels"
            ),
            "coordinates": "crop-local half-open bounds and zero-based rows",
        },
        "support": support_observations(rgb),
        "captionQualityApproved": False,
        "renderCalibrationIssued": False,
    }
    inputs.verify()
    if file_digest(tool) != tool_hash:
        raise ValueError("Colour executable changed")
    report["outputs"] = {
        str(path): file_digest(path) for path in sorted(output.iterdir()) if path.is_file()
    }
    with (output / "observation.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def verify(output: Path) -> None:
    """Recheck bound bytes and explicit nonapproval, not perceptual acceptance."""
    inputs = Inputs()
    report = inputs.json(output / "observation.json")
    if (
        report.get("version") != 1
        or report.get("captionQualityApproved") is not False
        or report.get("renderCalibrationIssued") is not False
    ):
        raise ValueError("Invalid observation scope")
    for group in ("inputs", "outputs"):
        for name, digest in record(report.get(group)).items():
            if inputs.bind(inputs.path(name)) != digest:
                raise ValueError("Observation binding changed")
    executable = report.get("executable")
    if not isinstance(executable, str) or file_digest(Path(executable)) != report.get(
        "executableSHA256"
    ):
        raise ValueError("Colour executable binding changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--canvas", type=int, nargs=4)
    parser.add_argument("--transicc", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.source)
        return
    if args.output is None or args.canvas is None or args.transicc is None:
        parser.error("observation requires output, --canvas and --transicc")
    left, top, right, bottom = args.canvas
    observe(args.source, args.output, canvas=(left, top, right, bottom), executable=args.transicc)


if __name__ == "__main__":
    main()
