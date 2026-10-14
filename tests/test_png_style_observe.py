import json
import runpy
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from replay import png_style_observe as observer
from replay.motion_calibration_verify import Inputs


@pytest.fixture
def icc() -> bytes:
    data = bytearray(128)
    data[:4] = struct.pack(">I", 128)
    data[16:20] = b"RGB "
    data[36:40] = b"acsp"
    return bytes(data)


@pytest.fixture
def png(icc: bytes) -> bytes:
    image = np.zeros((12, 16, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    image[0, 0, 3] = 0
    image[2:10, 4:12, :3] = (220, 140, 60)
    success, encoded = cv2.imencode(".png", image)
    assert success
    data = encoded.tobytes()
    return data[:33] + observer.chunk(b"iCCP", b"test\0\0" + zlib.compress(icc)) + data[33:]


def test_profile(png: bytes, icc: bytes) -> None:
    assert observer.embedded_profile(png) == ("test", icc)


@pytest.mark.parametrize("data", [b"", b"not PNG", observer.PNG_SIGNATURE])
def test_invalid_png(data: bytes) -> None:
    with pytest.raises(ValueError):
        observer.embedded_profile(data)


@pytest.mark.parametrize("mutation", ["crc", "truncated", "missing", "trailing", "duplicate"])
def test_bad_chunks(png: bytes, mutation: str) -> None:
    changed = {
        "crc": png[:32] + b"x" + png[33:],
        "truncated": png[:-6],
        "missing": png[:33] + observer.chunk(b"IEND", b""),
        "trailing": png + b"extra",
        "duplicate": png[:33] + png[33:-12] + png[33:],
    }[mutation]
    with pytest.raises(ValueError):
        observer.embedded_profile(changed)


@pytest.mark.parametrize(
    "body",
    [b"\0\0", b"name\0\1", b"name\0\0" + zlib.compress(b"invalid")],
)
def test_bad_profile(png: bytes, body: bytes) -> None:
    with pytest.raises(ValueError):
        observer.embedded_profile(png[:33] + observer.chunk(b"iCCP", body) + png[-12:])


@pytest.mark.parametrize("mode", ["header", "size", "stream", "budget", "rgb"])
def test_profile_guards(png: bytes, icc: bytes, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    data = bytearray(icc)
    if mode == "size":
        data[:4] = struct.pack(">I", 127)
    if mode == "rgb":
        data[16:20] = b"CMYK"
    if mode == "budget":
        monkeypatch.setattr(observer, "MAX_PROFILE_BYTES", 120)
    compressed = zlib.compress(data)
    if mode == "stream":
        compressed += b"junk"
    header = png[:33] if mode != "header" else observer.PNG_SIGNATURE
    value = header + observer.chunk(b"iCCP", b"test\0\0" + compressed) + png[-12:]
    with pytest.raises(ValueError):
        observer.embedded_profile(value)


def test_header_depth(png: bytes) -> None:
    header = bytearray(png[16:29])
    header[8] = 16
    with pytest.raises(ValueError, match="RGB8"):
        observer.embedded_profile(
            observer.PNG_SIGNATURE + observer.chunk(b"IHDR", bytes(header)) + png[33:]
        )


def test_support() -> None:
    rgb = np.zeros((9, 9, 3), dtype=np.float64)
    rgb[1:8, 1:8] = (60, 140, 220)
    reports = observer.support_observations(rgb)
    assert reports[0]["pixels"] == 49
    assert reports[1]["pixels"] == 25
    assert reports[2]["pixels"] == 9
    assert reports[2]["bounds"] == [3, 3, 6, 6]
    assert reports[2]["histogram"] == [{"srgb255": [60.0, 140.0, 220.0], "count": 9}]
    assert observer.distribution(rgb, np.zeros((9, 9), dtype=np.bool_)) == {
        "pixels": 0,
        "bounds": None,
    }


@pytest.mark.parametrize("stdout", ["1 2\n", "nan 2 3\n", "bad 2 3\n"])
def test_transform_bad_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, stdout, ""),
    )
    with pytest.raises(ValueError):
        observer.transform_palette(
            np.array([[0, 0, 0]], dtype=np.uint8),
            executable=Path("tool"),
            profile=Path("profile"),
            output=tmp_path,
        )


def test_transform_invalid_palette(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="palette"):
        observer.transform_palette(
            np.zeros((1, 4), dtype=np.uint8),
            executable=Path("tool"),
            profile=Path("profile"),
            output=tmp_path,
        )


def test_observe_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, png: bytes) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.png"
    source.write_bytes(png)
    tool = tmp_path / "tool"
    tool.write_bytes(b"synthetic executable")
    output = tmp_path / "result"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, "0 0 0\n60 140 220\n", "fixture"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "observe",
            str(source),
            str(output),
            "--canvas",
            "1",
            "1",
            "16",
            "12",
            "--transicc",
            str(tool),
        ],
    )
    # Inputs also binds the module implementation in this checkout.
    monkeypatch.setattr(Inputs, "path", lambda self, value: Path(value).absolute())
    runpy.run_path(observer.__file__, run_name="__main__")
    result = json.loads((output / "observation.json").read_text(encoding="utf-8"))
    assert result["profileName"] == "test"
    assert result["support"][1]["pixels"] == 36
    assert result["captionQualityApproved"] is False
    assert result["outOfRangeChannelsBeforePreviewClip"] == 0
    assert b"sRGB\x01" in (output / "canvas-srgb.png").read_bytes()
    assert source.read_bytes() == png
    monkeypatch.setattr(sys, "argv", ["observe", str(output), "--verify"])
    observer.main()
    with pytest.raises(FileExistsError):
        observer.observe(source, output, canvas=(1, 1, 16, 12), executable=tool)
    tool.write_bytes(b"changed tool")
    with pytest.raises(ValueError, match="executable binding"):
        observer.verify(output)
    source.write_bytes(b"changed capture")
    with pytest.raises(ValueError, match="binding changed"):
        observer.verify(output)
    result["captionQualityApproved"] = True
    (output / "observation.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="scope"):
        observer.verify(output)


def test_cli_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["observe", "source.png"])
    with pytest.raises(SystemExit):
        observer.main()


@pytest.mark.parametrize("mode", ["canvas", "size", "transparent", "decode"])
def test_observe_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, png: bytes, mode: str
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(png)
    monkeypatch.setattr(Inputs, "path", lambda self, value: Path(value).absolute())
    if mode == "size":
        monkeypatch.setattr(observer, "MAX_BYTES", 1)
    if mode == "transparent":
        monkeypatch.setattr(cv2, "imdecode", lambda *args: np.zeros((12, 16, 4), np.uint8))
    if mode == "decode":
        monkeypatch.setattr(cv2, "imdecode", lambda *args: None)
    canvas = (-1, 0, 16, 12) if mode == "canvas" else (1, 1, 16, 12)
    with pytest.raises(ValueError):
        observer.observe(source, tmp_path / "output", canvas=canvas, executable=source)
