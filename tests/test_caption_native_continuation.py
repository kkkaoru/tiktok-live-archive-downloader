import json
from pathlib import Path

import pytest
from test_caption_consumer_binding import native as native_template
from test_caption_consumer_binding import ref, root, write

from replay import caption_native_continuation as continuation
from replay import caption_native_inputs, caption_reflow_verify, clock_contract
from replay import caption_style_defaults as defaults
from replay import workflow as workflow_cli
from replay.caption_consumer_binding import _read
from replay.caption_consumer_binding import prepare as bind
from replay.core import ReplayError
from replay.stages import file_digest

__all__ = ["native_template", "root"]


@pytest.fixture
def config(native_template: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = native_template.parent
    clock = write(root / "clock.json", {"synthetic": True})
    cuts = root / "cuts.json"
    data = json.loads(cuts.read_text())
    data.update(
        {
            "fps": 30,
            "audioSupport": {"recognizedSpeechRemoved": False},
            "parentSHA256": {str(clock): file_digest(clock)},
        }
    )
    write(cuts, data)
    tokens = root / "tokens.json"
    data = json.loads(tokens.read_text())
    data["cuts"] = ref(cuts)
    write(tokens, data)
    native_config = root / "native-config.json"
    bind(native_template, native_config)
    caption_native_inputs.prepare(native_config, root / "recognition-inputs")
    tool = root / "tool"
    tool.write_bytes(b"synthetic tool, never executed")
    font = root / defaults.FONT_FILE_NAME
    font.write_bytes(b"synthetic font, never loaded")
    font_receipt = write(
        root / "font.json",
        {
            "primary": {
                "postScriptName": defaults.FONT_POSTSCRIPT_NAME,
                "family": defaults.FONT_FAMILY,
                "face": defaults.FONT_FACE,
                "fontURL": font.as_uri(),
                "pointSize": 12,
                "variations": [{"identifier": defaults.FONT_WEIGHT_AXIS, "value": 700}],
            },
            "fonts": [ref(font)],
        },
    )
    metrics = write(
        root / "metrics.json",
        {
            "style": {
                "font_name": defaults.FONT_POSTSCRIPT_NAME,
                "ass_font_size": 12,
                "border": 1,
                "rim": 1,
                "motion_line_advance": 12,
            },
            "expectedPointSize": 12,
            "width": 900,
            "inputs": {str(tool): file_digest(tool), str(font_receipt): file_digest(font_receipt)},
        },
    )
    template = root / "template.moti"
    template.write_text("synthetic Motion", encoding="utf-8")
    monkeypatch.setattr(continuation, "group_controls", lambda document: ())
    closure = write(
        root / "closure.json",
        {
            "inputs": {
                str(Path(caption_native_inputs.__file__).resolve()): file_digest(
                    Path(caption_native_inputs.__file__)
                )
            }
        },
    )
    return write(
        root / "continuation.json",
        {
            "version": 1,
            "inputs": {
                "source": ref(root / "source.bin"),
                "stereoPCM": ref(tool),
                "audioClock": ref(clock),
                "nativeExecutable": ref(tool),
                "longExecutable": ref(tool),
                "python": ref(tool),
                "ffmpeg": ref(tool),
                "fontReceipt": ref(font_receipt),
                "measurementReference": ref(metrics),
                "template": ref(template),
                "codeClosure": ref(closure),
            },
            "futurePaths": {"cuts": str(cuts), "nativeConfig": str(native_config)},
            "evidence": {"font": ref(font_receipt)},
        },
    )


def test_prepare_derived_frames_no_execution(config: Path) -> None:
    output = config.parent / "next"
    assert continuation.main([str(config), str(output)]) == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["frames"] == 30
    assert receipt["measurementsPerformed"] is False
    selection = json.loads((output / "tempo-selection.json").read_text())
    assert selection["intervals"] == [
        {"start": {"value": 0, "timescale": 30}, "end": {"value": 30, "timescale": 30}}
    ]
    workflow = json.loads((output / "workflow.json").read_text())
    assert workflow["stages"][-1]["verify_command"][-2:] == ["--frames", "30"]
    assert not (output / "paging").exists()
    assert not (output / "tempo").exists()
    assert (
        workflow_cli.main(
            [str(output / "workflow.json"), "--state", str(output / "state"), "--check"]
        )
        == 0
    )


@pytest.mark.parametrize("field,value", [("version", 2), ("futurePaths", {}), ("inputs", {})])
def test_invalid_schema(config: Path, field: str, value: object) -> None:
    data = json.loads(config.read_text())
    data[field] = value
    write(config, data)
    with pytest.raises(ReplayError):
        continuation.prepare(config, config.parent / "next")


def test_existing_output(config: Path) -> None:
    with pytest.raises(ReplayError, match="new canonical"):
        continuation.prepare(config, config.parent)


def test_changed_pin(config: Path) -> None:
    (config.parent / "tool").write_bytes(b"changed")
    with pytest.raises(ReplayError, match="pin changed"):
        continuation.prepare(config, config.parent / "next")


def test_wrong_source(config: Path) -> None:
    path = config.parent / "native-config.json"
    data = json.loads(path.read_text())
    data["source"] = ref(config.parent / "tool")
    write(path, data)
    with pytest.raises(ReplayError, match="selected source/cuts"):
        continuation.prepare(config, config.parent / "next")


@pytest.mark.parametrize("value", [True, "12", float("inf")])
def test_invalid_number(value: object) -> None:
    with pytest.raises(ReplayError, match="finite"):
        continuation._number(value)


@pytest.mark.parametrize(
    "field,value", [("font_name", "wrong"), ("font_name", None), ("border", 20)]
)
def test_bad_style(config: Path, field: str, value: object) -> None:
    path = config.parent / "metrics.json"
    data = json.loads(path.read_text())
    data["style"][field] = value
    write(path, data)
    paths = {
        name: Path(row["path"]) for name, row in json.loads(config.read_text())["inputs"].items()
    }
    with pytest.raises(ReplayError):
        continuation._measurement(paths)


@pytest.mark.parametrize("field,value", [("width", 0), ("expectedPointSize", 13), ("inputs", {})])
def test_bad_measurement(config: Path, field: str, value: object) -> None:
    path = config.parent / "metrics.json"
    data = json.loads(path.read_text())
    data[field] = value
    write(path, data)
    paths = {
        name: Path(row["path"]) for name, row in json.loads(config.read_text())["inputs"].items()
    }
    with pytest.raises(ReplayError):
        continuation._measurement(paths)


def test_missing_font_assets(config: Path) -> None:
    path = config.parent / "font.json"
    data = json.loads(path.read_text())
    data["fonts"] = []
    write(path, data)
    paths = {
        name: Path(row["path"]) for name, row in json.loads(config.read_text())["inputs"].items()
    }
    with pytest.raises(ReplayError, match="asset pins"):
        continuation._measurement(paths)


def test_tempo_contract_uses_ordinary_verifier(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = config.parent / "next"
    continuation.prepare(config, output)
    (output / "tempo").mkdir()
    (output / "tempo/candidate.wav").write_bytes(b"synthetic output")
    write(output / "tempo/receipt.json", {"synthetic": True})
    calls: list[Path] = []

    def verify(path: Path) -> dict[str, object]:
        calls.append(path)
        return {"synthetic": True}

    monkeypatch.setattr(clock_contract, "verify", verify)
    assert continuation.main([str(config), str(output), "--tempo-contract"]) == 0
    assert len(calls) == 1
    data = json.loads((output / "tempo-clock-config.json").read_text())
    assert data["previousApplications"] == 0
    assert data["outputClock"] == "edited-cut-pcm"


@pytest.fixture
def head_ready(config: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    output = config.parent / "next"
    continuation.prepare(config, output)
    for relative in ("paging", "paging/metrics-0000", "tempo"):
        (output / relative).mkdir(exist_ok=True)
    for relative in (
        "paging/reflow.json",
        "paging/metrics-0000/inputs.json",
        "tempo/receipt.json",
        "tempo-clock-config.json",
    ):
        write(output / relative, {"synthetic": True})
    (output / "tempo/candidate.wav").write_bytes(b"synthetic, never decoded")
    clean = config.parent / "clean.bin"
    clean.write_bytes(b"synthetic, never decoded")
    roles = {
        name: str(write(config.parent / f"{name}.json", {"synthetic": True}))
        for name in ("assemblyManifest", "assemblyReceipt", "cleanIndex")
    }
    template = write(
        config.parent / "head-template.json",
        {
            "version": 1,
            "name": "Synthetic head",
            "faceIndex": ref(write(config.parent / "face.json", {})),
            "cleanPaths": roles,
        },
    )
    monkeypatch.setattr(
        continuation, "clean_source", lambda value: (config.parent / "source.bin", clean, 1)
    )
    monkeypatch.setattr(clock_contract, "verify", lambda path: {"outputFrames": 30})
    monkeypatch.setattr(caption_reflow_verify, "verify", lambda *args, **kwargs: None)
    return config, output, template


def test_head_uses_completed_clean_and_ordinary_producers(
    head_ready: tuple[Path, Path, Path],
) -> None:
    config, output, template = head_ready
    assert continuation.main([str(config), str(output), "--head-template", str(template)]) == 0
    graph = json.loads((output / "head-workflow.json").read_text())
    head = json.loads((output / "head-config.json").read_text())
    assert head["headReferencePolicy"] == "automatic-face-relative"
    assert "referenceHeadTop" not in head
    assert graph["stages"][1]["verify_command"][-1] == "--verify"
    assert graph["sourceCaptionProduction"]["bindings"]["clean-semantic"].endswith(
        "clean-transform-binding.json"
    )
    assert not (output / "head").exists()
    assert (
        workflow_cli.main(
            [str(output / "head-workflow.json"), "--state", str(output / "head-state"), "--check"]
        )
        == 0
    )


@pytest.mark.parametrize("field,value", [("version", 2), ("cleanPaths", {}), ("name", "")])
def test_head_rejects_invalid_template(
    head_ready: tuple[Path, Path, Path], field: str, value: object
) -> None:
    config, output, template = head_ready
    data = json.loads(template.read_text())
    data[field] = value
    write(template, data)
    with pytest.raises(ReplayError):
        continuation.prepare_head(config, output, template=template)


def test_head_rejects_wrong_media(
    head_ready: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, template = head_ready
    monkeypatch.setattr(
        continuation,
        "clean_source",
        lambda value: (config.parent / "wrong", config.parent / "clean.bin", 1),
    )
    with pytest.raises(ReplayError, match="different source"):
        continuation.prepare_head(config, output, template=template)


def test_head_rejects_wrong_clock(
    head_ready: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, template = head_ready
    monkeypatch.setattr(clock_contract, "verify", lambda path: {"outputFrames": 31})
    with pytest.raises(ReplayError, match="frame counts"):
        continuation.prepare_head(config, output, template=template)


def test_full_labels_above_small_metadata_budget(root: Path) -> None:
    path = write(
        root / "large-labels.json",
        {
            "clock": "source-presentation",
            "labels": [{"text": "発話", "start": 1.0, "end": 1.1, "provenance": "recognition"}]
            * 140000,
        },
    )
    assert path.stat().st_size > 8 * 1024 * 1024
    labels = _read(path).get("labels")
    assert isinstance(labels, list)
    assert len(labels) == 140000
