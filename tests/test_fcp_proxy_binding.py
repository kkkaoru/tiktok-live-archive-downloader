import json
import subprocess
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_proxy_binding as binding
from replay.artifact_json import _pin
from replay.core import ReplayError
from replay.delivery_media import FrameClock
from replay.delivery_source_assembly import Chunk
from replay.fcp_part_assembly import _reference
from replay.fcp_partition import _required
from replay.subtitle_clock import VideoFrameClock


def test_samples_require_content_not_just_same_clock() -> None:
    pixels = bytes(range(256)) * 225
    result = binding.compare_samples(pixels, pixels)
    assert result["correlation"] == 1
    assert result["meanAbsoluteLumaError"] == 0
    with pytest.raises(ReplayError, match="differs"):
        binding.compare_samples(pixels, pixels[::-1])


def test_flat_and_wrong_length_samples_are_not_evidence() -> None:
    with pytest.raises(ReplayError, match="sizes"):
        binding.compare_samples(b"x", b"x")
    with pytest.raises(ReplayError, match="Flat"):
        binding.compare_samples(bytes(57600), bytes(57600))


def test_probe_reads_square_pixel_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 0, b'{"streams":[{"width":720,"height":1280,"sample_aspect_ratio":"1:1"}]}'
        ),
    )
    assert binding.geometry(Path("input.mov")) == binding.Geometry(720, 1280)


@pytest.mark.parametrize("document", [{"streams": []}, {"streams": [{"width": 0, "height": 8}]}])
def test_invalid_probe_rejected(
    document: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, json.dumps(document).encode()),
    )
    with pytest.raises(ReplayError, match="geometry|stream"):
        binding.geometry(Path("input.mov"))


def test_sample_has_fixed_bounded_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, bytes(57600))
    )
    assert len(binding.sample(Path("input.mov"), Fraction(1, 2))) == 57600
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, b""))
    with pytest.raises(ReplayError, match="dimensions"):
        binding.sample(Path("input.mov"), Fraction(1, 2))


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    original, proxy, clean = (
        tmp_path / "original.mkv",
        tmp_path / "proxy.mov",
        tmp_path / "clean.mov",
    )
    original.write_bytes(b"original")
    proxy.write_bytes(b"proxy")
    clean.write_bytes(b"clean")
    source_config = tmp_path / "source.json"
    source_config.write_text(json.dumps({"source": str(original)}))
    manifest = tmp_path / "assembly.json"
    manifest.write_text("{}")
    clock_path = tmp_path / "clock.json"
    clock_path.write_text("{}")
    chunk = Chunk(
        clean,
        VideoFrameClock((0, 33, 67), Fraction(1, 1000), 33),
        (str(original), str(_reference(original)["sha256"])),
    )
    monkeypatch.setattr(binding, "load_chunks", lambda *a, **k: (chunk,))
    monkeypatch.setattr(
        binding, "read_clock", lambda path: FrameClock((0, 1, 2), Fraction(1, 30), 1)
    )
    monkeypatch.setattr(
        binding,
        "geometry",
        lambda path: binding.Geometry(1080, 1920) if path == proxy else binding.Geometry(720, 1280),
    )
    monkeypatch.setattr(binding, "sample", lambda path, time: bytes(range(256)) * 225)
    return {
        "assemblyManifest": _reference(manifest),
        "proxyBinding": {
            "policy": "aspect-fit-cfr-candidate",
            "original": _reference(original),
            "proxy": _reference(proxy),
            "proxyClock": _reference(clock_path),
            "sourceConfig": _reference(source_config),
        },
    }


def root_for(config: dict[str, object]) -> ET.Element:
    row = config["proxyBinding"]
    assert isinstance(row, dict)
    proxy = _pin(row["proxy"])
    return ET.fromstring(
        '<fcpxml><resources><format id="f" width="1080" height="1920" '
        'frameDuration="1/30s"/><asset id="s" start="0s">'
        f'<media-rep src="{proxy.as_uri()}"/></asset></resources>'
        '<library><event><project><sequence format="f"/>'
        "</project></event></library></fcpxml>"
    )


def test_binding_changes_only_asset_format_and_preserves_candidate_limits(
    config: dict[str, object], tmp_path: Path
) -> None:
    root = root_for(config)
    proxy, evidence = binding.bind(root, config, clean=tmp_path / "clean.mov")
    assert proxy.name == "proxy.mov"
    assert _required(root, "./resources/asset").attrib["format"] == "proxyBoundCleanFormat"
    assert _required(root, "./resources/format").attrib["width"] == "720"
    assert evidence["fullVideoPixelIdentityVerified"] is False
    assert evidence["proxyProducerReceiptAvailable"] is False
    assert evidence["sampledVisualCorrespondence"] is True
    samples = evidence["samples"]
    assert isinstance(samples, list) and len(samples) == 5


@pytest.mark.parametrize(
    "fault", ["policy", "source", "hash", "aspect", "sequence", "asset", "duplicate"]
)
def test_binding_rejects_mismatches(
    config: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    root = root_for(config)
    row = config["proxyBinding"]
    assert isinstance(row, dict)
    if fault == "policy":
        row["policy"] = "guess"
    elif fault == "source":
        path = _pin(row["sourceConfig"])
        path.write_text('{"source":"wrong"}')
        row["sourceConfig"] = _reference(path)
    elif fault == "hash":
        path = tmp_path / "other.mov"
        path.write_bytes(b"different")
        row["original"] = _reference(path)
        config_path = _pin(row["sourceConfig"])
        config_path.write_text(json.dumps({"source": str(path)}))
        row["sourceConfig"] = _reference(config_path)
    elif fault == "aspect":
        monkeypatch.setattr(
            binding,
            "geometry",
            lambda path: (
                binding.Geometry(100, 100)
                if path.name == "proxy.mov"
                else binding.Geometry(720, 1280)
            ),
        )
    elif fault == "sequence":
        _required(root, "./resources/format").set("width", "100")
    elif fault == "asset":
        _required(root, "./resources/asset").set("start", "1s")
    else:
        ET.SubElement(_required(root, "./resources"), "format", id="proxyBoundCleanFormat")
    with pytest.raises(ReplayError):
        binding.bind(root, config, clean=tmp_path / "clean.mov")
