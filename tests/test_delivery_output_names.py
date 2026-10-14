import json
from pathlib import Path

import pytest

from replay import delivery_output_names as names
from replay.core import ReplayError
from replay.stages import file_digest


def native_preset(path: Path, *, encoder: str = "MPEG4", extension: str = "mp4") -> Path:
    path.write_text(
        f'<setting><encoder name="{encoder}"><file-extension>{extension}</file-extension>'
        '<video-encode isEnabled="yes"><codec-type>avc1</codec-type></video-encode>'
        '<audio-encode isEnabled="yes"><codec-type>aac </codec-type></audio-encode>'
        "</encoder><filter-set/></setting>",
        encoding="utf-8",
    )
    return path


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(names, "ROOT", tmp_path)
    root = tmp_path / "edits/replay-123"
    root.mkdir(parents=True)
    work = tmp_path / "edits/work"
    work.mkdir()
    original = tmp_path / "2024-02-29_09-30-59_JST_replay-123.mp4"
    original.write_bytes(b"original")
    edit = root / "edit-config.json"
    edit.write_text(
        json.dumps(
            {
                "version": 1,
                "replay_id": "123",
                "source": str(original),
                "source_size": 8,
                "engine": str(tmp_path),
                "executor_scope": str(tmp_path),
                "stages": dict.fromkeys(
                    (
                        "bgm_removal",
                        "silence_cut",
                        "source_caption_mask",
                        "transcription",
                        "dynamic_captions",
                    ),
                    True,
                ),
            }
        ),
        encoding="utf-8",
    )
    master = work / "whole.mov"
    master.write_bytes(b"whole master")
    preset = native_preset(work / "native.setting")
    directory = work / "delivery"
    directory.mkdir()
    config = work / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "editConfig": pin(edit),
                "wholeMaster": pin(master),
                "preset": pin(preset),
                "outputDirectory": str(directory),
            }
        ),
        encoding="utf-8",
    )
    return config, work / "request.json", work / "receipt.json"


@pytest.mark.parametrize(
    "recording",
    ["replay-999999999", "youtube-AbCd_123-45"],
)
def test_unknown_recordings_keep_original_datetime(recording: str) -> None:
    assert names.delivery_stem(Path(f"2028-02-29_23-59-59_JST_{recording}.mkv"), recording) == (
        f"2028-02-29_23-59-59_JST_{recording}-review"
    )


@pytest.mark.parametrize(
    "source,recording,message",
    [
        ("2024-02-30_00-00-00_JST_replay-1.mp4", "replay-1", "invalid"),
        ("2024-02-29_24-00-00_JST_replay-1.mp4", "replay-1", "invalid"),
        ("2024-2-29_00-00-00_JST_replay-1.mp4", "replay-1", "canonical"),
        ("2024-02-29_00-00-00_JST_replay-2.mp4", "replay-1", "identifier"),
        ("undated.mp4", "replay-1", "identifier"),
        ("x", "replay-abc", "stable"),
        ("x", "youtube-short", "stable"),
    ],
)
def test_rejects_invalid_clock_or_identifier(source: str, recording: str, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        names.delivery_stem(Path(source), recording)


@pytest.mark.parametrize("encoder,extension", [("MPEG4", "mp4"), ("H264iPod", "m4v")])
def test_actual_preset_extension_is_preserved(tmp_path: Path, encoder: str, extension: str) -> None:
    preset = native_preset(tmp_path / "native.setting", encoder=encoder, extension=extension)
    assert names.encoded_name(Path("2024-01-02_03-04-05_JST_replay-1.mp4"), "replay-1", preset) == (
        f"2024-01-02_03-04-05_JST_replay-1-review.{extension}"
    )


@pytest.mark.parametrize(
    "encoder,extension", [("H264iPod", "mp4"), ("MPEG4", "m4v"), ("QT", "mov")]
)
def test_inconsistent_native_container_is_rejected(
    tmp_path: Path, encoder: str, extension: str
) -> None:
    preset = native_preset(tmp_path / "native.setting", encoder=encoder, extension=extension)
    with pytest.raises(ReplayError, match="container"):
        names.preset_extension(preset)


def test_history_identity_never_selects_a_legacy_fallback() -> None:
    source = Path("2024-01-02_03-04-05_JST_replay-1.mp4")
    canonical = Path("2024-01-02_03-04-05_JST_replay-1-review.m4v")
    assert names.recorded_stem_matches(canonical, source, "replay-1", historical=False)
    assert names.recorded_stem_matches(
        Path("final-encode.m4v"), source, "replay-1", historical=True
    )
    assert not names.recorded_stem_matches(
        Path("final-encode.m4v"), source, "replay-1", historical=False
    )
    assert not names.recorded_stem_matches(Path("other.mp4"), source, "replay-1", historical=True)


def test_whole_request_keeps_native_schema_and_honest_evidence(
    prepared: tuple[Path, Path, Path],
) -> None:
    config, output, receipt = prepared
    assert names.prepare_request(config, output, receipt) == output
    request = json.loads(output.read_text())
    assert request == {
        "batchName": "2024-02-29_09-30-59_JST_replay-123-review",
        "bundleID": "com.apple.CompressorApp",
        "outputDirectory": str(config.parent / "delivery"),
        "outputName": "2024-02-29_09-30-59_JST_replay-123-review.mp4",
        "presetPath": str(config.parent / "native.setting"),
        "sourcePath": str(config.parent / "whole.mov"),
    }
    evidence = json.loads(receipt.read_text())
    assert evidence["request"] == pin(output)
    assert evidence["encodeSubmitted"] is False
    assert evidence["actualContainerVerified"] is False
    assert evidence["wholeRenderVerified"] is False
    assert evidence["timestampSource"] == "validated-original-filename-not-new-api-observation"
    assert (output.stat().st_mode & 0o777) == 0o600
    with pytest.raises(ReplayError, match="absent"):
        names.prepare_request(config, output, receipt)


@pytest.mark.parametrize("stem_extension", [".mp4", ".m4v", ".json"])
def test_existing_same_stem_is_never_resubmitted(
    prepared: tuple[Path, Path, Path], stem_extension: str
) -> None:
    config, output, receipt = prepared
    existing = (
        config.parent / "delivery" / f"2024-02-29_09-30-59_JST_replay-123-review{stem_extension}"
    )
    existing.write_bytes(b"existing")
    with pytest.raises(ReplayError, match="existing or ambiguous"):
        names.prepare_request(config, output, receipt)
    assert not output.exists()
    assert existing.read_bytes() == b"existing"


@pytest.mark.parametrize("field", ["editConfig", "wholeMaster", "preset"])
def test_changed_pin_prevents_request(prepared: tuple[Path, Path, Path], field: str) -> None:
    config, output, receipt = prepared
    data = json.loads(config.read_text())
    data[field]["sha256"] = "0" * 64
    config.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="hash changed"):
        names.prepare_request(config, output, receipt)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "extra", "version"])
def test_request_configuration_is_strict(prepared: tuple[Path, Path, Path], kind: str) -> None:
    config, output, receipt = prepared
    data = json.loads(config.read_text())
    if kind == "duplicate":
        config.write_text('{"version":1,"version":1}')
    elif kind == "nonfinite":
        config.write_text('{"version":NaN}')
    else:
        data["unexpected" if kind == "extra" else "version"] = True
        config.write_text(json.dumps(data))
    with pytest.raises((ReplayError, ValueError), match="Duplicate|Non-finite|fields"):
        names.prepare_request(config, output, receipt)


@pytest.mark.parametrize("field", ["outputDirectory", "wholeMaster", "editConfig"])
def test_request_paths_cannot_escape_or_use_symlinks(
    prepared: tuple[Path, Path, Path], field: str
) -> None:
    config, output, receipt = prepared
    data = json.loads(config.read_text())
    if field == "outputDirectory":
        data[field] = str(config.parent.parent.parent)
    else:
        target = Path(data[field]["path"])
        linked = config.parent / "linked"
        linked.symlink_to(target)
        data[field]["path"] = str(linked)
    config.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="canonical|existing recording storage"):
        names.prepare_request(config, output, receipt)


@pytest.mark.parametrize("change", ["codec", "disabled", "container"])
def test_whole_request_rejects_non_mp4_h264_aac(
    prepared: tuple[Path, Path, Path], change: str
) -> None:
    config, output, receipt = prepared
    data = json.loads(config.read_text())
    preset = Path(data["preset"]["path"])
    text = preset.read_text()
    if change == "codec":
        text = text.replace("avc1", "hvc1")
    elif change == "disabled":
        text = text.replace('isEnabled="yes"', 'isEnabled="no"')
    else:
        text = text.replace('name="MPEG4"', 'name="H264iPod"').replace("mp4", "m4v")
    preset.write_text(text)
    data["preset"] = pin(preset)
    config.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="H264/AAC"):
        names.prepare_request(config, output, receipt)


def test_cli_success_and_failure(
    prepared: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config, output, receipt = prepared
    assert names.main([str(config), str(output), str(receipt)]) == 0
    assert names.main([str(config), str(output), str(receipt)]) == 1
    assert "DELIVERY_OUTPUT_NAMES_ERROR" in capsys.readouterr().err
