"""Synthetic native schema edits, never an encode or a GUI operation."""

import base64
import json
import plistlib
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import compressor_preset as preset
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference


def sample(path: Path, *, nested: bool = False) -> Path:
    props: dict[str, object] = {
        "H264EntropyMode": {"objectValue": "CABAC"},
        "ProfileLevel": {"objectValue": "H264_High_AutoLevel"},
    }
    if nested:
        props = {"codec-properties": props, "encoder-id": "avc1"}
    payload = base64.b64encode(plistlib.dumps(props)).decode("ascii")
    path.write_text(
        '<setting name="native"><encoder name="MPEG4"><file-extension>mp4</file-extension>'
        "<audio-encode><audio-encoding-bitrate>128000</audio-encoding-bitrate></audio-encode>"
        '<video-encode><bounds width="1080" height="1920"/><encoder-properties>'
        f"{payload}</encoder-properties></video-encode></encoder><filter-set/></setting>"
    )
    return path


def config(tmp_path: Path, source: Path) -> dict[str, object]:
    return {
        "version": 1,
        "source": _reference(source),
        "name": "Reusable portrait",
        "edits": [
            {
                "path": "encoder/audio-encode/audio-encoding-bitrate",
                "before": {"text": "128000", "attributes": {}},
                "after": {"text": "384000", "attributes": {}},
            }
        ],
    }


def derive(tmp_path: Path, value: dict[str, object]) -> dict[str, object]:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return preset.derive(path, tmp_path / "output.compressorsetting")


def test_derive_preserves_native_structure_and_requires_later_native_readback(
    tmp_path: Path,
) -> None:
    source = sample(tmp_path / "source.setting")
    original = source.read_bytes()
    report = derive(tmp_path, config(tmp_path, source))
    assert report["nativeApplicationReadback"] is False
    assert report["encodedArtifactVerified"] is False
    root = preset.read(tmp_path / "output.compressorsetting")
    assert root.findtext("encoder/audio-encode/audio-encoding-bitrate") == "384000"
    assert root.find("encoder/video-encode/bounds") is not None
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        derive(tmp_path, config(tmp_path, source))


@pytest.mark.parametrize("nested", [False, True])
def test_profile_can_come_from_official_nested_qt_donor(tmp_path: Path, nested: bool) -> None:
    source = sample(tmp_path / "source.setting", nested=nested)
    donor = sample(tmp_path / "donor.setting", nested=True)
    value = config(tmp_path, source)
    value["profileSource"] = _reference(donor)
    report = derive(tmp_path, value)
    assert report["codecProperties"] == preset.properties(preset.read(donor))
    envelope = preset.property_envelope(preset.read(tmp_path / "output.compressorsetting"))
    assert ("codec-properties" in envelope) is nested
    if nested:
        assert envelope["encoder-id"] == "avc1"


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "name",
        "edits",
        "source-hash",
        "before",
        "unknown",
        "duplicate",
        "missing",
        "not-text",
    ],
)
def test_invalid_edits_fail_before_output(tmp_path: Path, mutation: str) -> None:
    source = sample(tmp_path / "source.setting")
    value = config(tmp_path, source)
    if mutation in ("version", "name", "edits"):
        value[mutation] = None
    elif mutation == "source-hash":
        source.write_text("changed")
    else:
        edit: dict[str, object] = {
            "path": "encoder/audio-encode/audio-encoding-bitrate",
            "before": {"text": "128000", "attributes": {}},
            "after": {"text": "384000", "attributes": {}},
        }
        if mutation == "before":
            edit["before"] = {}
        elif mutation == "unknown":
            edit["path"] = "job-action"
        elif mutation == "missing":
            edit["path"] = "encoder/optimize-for-network"
        elif mutation == "not-text":
            edit["after"] = {"text": 3, "attributes": {}}
        value["edits"] = [edit, edit] if mutation == "duplicate" else [edit]
    with pytest.raises((ReplayError, ValueError)):
        derive(tmp_path, value)
    assert not (tmp_path / "output.compressorsetting").exists()


@pytest.mark.parametrize(
    "mode",
    [
        "wrong-root",
        "action",
        "filters",
        "no-filter",
        "codec",
        "extension",
        "no-properties",
        "empty-properties",
        "missing-donor-fields",
    ],
)
def test_untrusted_or_incompatible_native_schema_is_rejected(tmp_path: Path, mode: str) -> None:
    source = sample(tmp_path / "source.setting")
    root = ET.parse(source).getroot()
    encoder = root.find("encoder")
    assert encoder is not None
    if mode == "wrong-root":
        root.tag = "job"
    elif mode == "action":
        ET.SubElement(root, "job-action")
    elif mode in ("filters", "no-filter"):
        filters = root.find("filter-set")
        assert filters is not None
        if mode == "filters":
            ET.SubElement(filters, "filter")
        else:
            root.remove(filters)
    elif mode == "codec":
        encoder.set("name", "QT")
    elif mode == "extension":
        extension = encoder.find("file-extension")
        assert extension is not None
        extension.text = "mov"
    else:
        video = encoder.find("video-encode")
        assert video is not None
        properties = video.find("encoder-properties")
        assert properties is not None
        if mode == "no-properties":
            video.remove(properties)
        elif mode == "empty-properties":
            properties.text = ""
        else:
            properties.text = base64.b64encode(plistlib.dumps({})).decode("ascii")
    source.write_bytes(ET.tostring(root))
    value = config(tmp_path, source)
    if mode in ("no-properties", "empty-properties", "missing-donor-fields"):
        value["profileSource"] = _reference(source)
    with pytest.raises(ReplayError):
        derive(tmp_path, value)


def test_symlink_input_is_not_a_native_sample(tmp_path: Path) -> None:
    source = sample(tmp_path / "source.setting")
    link = tmp_path / "link.setting"
    link.symlink_to(source)
    with pytest.raises(ReplayError):
        preset.read(link)


def test_cli_inspect_and_derive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = sample(tmp_path / "source.setting")
    monkeypatch.setattr("sys.argv", ["preset", "inspect", str(source)])
    preset.main()
    assert json.loads(capsys.readouterr().out)["name"] == "native"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config(tmp_path, source)))
    monkeypatch.setattr(
        "sys.argv", ["preset", "derive", str(path), str(tmp_path / "output.setting")]
    )
    preset.main()
    assert json.loads(capsys.readouterr().out)["name"] == "Reusable portrait"
