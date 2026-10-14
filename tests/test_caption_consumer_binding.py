import json
from pathlib import Path

import pytest

from replay import cut_audio_support, delivery_captions
from replay.caption_consumer_binding import main, prepare, verify
from replay.core import ReplayError
from replay.stages import file_digest


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "edits/work"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def native(root: Path) -> Path:
    source = root / "source.bin"
    source.write_bytes(b"synthetic source")
    cuts = write(
        root / "cuts.json",
        {
            "version": 1,
            "clock": "source-presentation",
            "cuts": [{"source_start_frame": 0, "source_end_frame": 30, "output_start_frame": 0}],
        },
    )
    labels = write(
        root / "labels.json",
        {
            "clock": "source-presentation",
            "labels": [{"text": "声", "start": 0.2, "end": 0.4, "provenance": "native:0"}],
        },
    )
    projected = delivery_captions.project(labels, cuts, root / "tokens.json", rate=30)
    return write(
        root / "template.json",
        {
            "version": 1,
            "kind": "native-inputs",
            "artifactPaths": {
                "source": str(source),
                "cuts": str(cuts),
                "projectedTokens": str(projected),
            },
            "expectedInputs": {"source": ref(source)},
        },
    )


@pytest.fixture
def support(root: Path) -> Path:
    clock = write(
        root / "clock.json",
        {
            "streams": [{"sample_rate": "16000", "time_base": "1/16000"}],
            "frames": [{"pts": 1000, "nb_samples": 16000}],
        },
    )
    labels = write(
        root / "labels.json",
        {
            "clock": "source-presentation",
            "parentSHA256": {str(clock): file_digest(clock)},
            "labels": [{"text": "声", "start": 0.2, "end": 0.4, "provenance": "native:0"}],
        },
    )
    cuts = write(
        root / "cuts.json",
        {
            "version": 1,
            "clock": "source-presentation",
            "fps": 30,
            "sourceFrames": 33,
            "cuts": [{"source_start_frame": 0, "source_end_frame": 33, "output_start_frame": 0}],
            "parentSHA256": {str(clock): file_digest(clock), str(labels): file_digest(labels)},
        },
    )
    return write(
        root / "template.json",
        {
            "version": 1,
            "kind": "cut-audio-support",
            "artifactPaths": {"cuts": str(cuts), "labels": str(labels), "audioClock": str(clock)},
            "expectedInputs": {"audioClock": ref(clock)},
        },
    )


def test_native_prepare_verify(native: Path) -> None:
    output = native.parent / "config.json"
    assert main(["prepare", str(native), str(output)]) == 0
    assert main(["verify", str(native), str(output)]) == 0
    assert set(json.loads(output.read_text())) == {"version", "source", "cuts", "projectedTokens"}
    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(output.with_suffix(".binding.json").read_text())["deliveryComplete"] is False


def test_support_prepare(support: Path) -> None:
    output = support.parent / "config.json"
    prepare(support, output)
    verify(support, output)
    assert set(json.loads(output.read_text())["artifacts"]) == {"cuts", "labels", "audioClock"}


@pytest.mark.parametrize(
    "field,value", [("kind", "other"), ("version", 2), ("style", 20), ("font", "old")]
)
def test_unknown_schema(native: Path, field: str, value: object) -> None:
    data = json.loads(native.read_text())
    data[field] = value
    write(native, data)
    with pytest.raises(ReplayError, match="schema"):
        prepare(native, native.parent / "config.json")


def test_missing_fixed_source(native: Path) -> None:
    data = json.loads(native.read_text())
    data["expectedInputs"] = {}
    write(native, data)
    with pytest.raises(ReplayError, match="roles"):
        prepare(native, native.parent / "config.json")


def test_changed_source(native: Path) -> None:
    (native.parent / "source.bin").write_bytes(b"different source")
    with pytest.raises(ReplayError, match="identity changed"):
        prepare(native, native.parent / "config.json")


def test_wrong_projected_clock(native: Path) -> None:
    data = json.loads((native.parent / "tokens.json").read_text())
    data["clock"] = "source-presentation"
    write(native.parent / "tokens.json", data)
    with pytest.raises(ReplayError, match="conflict-free"):
        prepare(native, native.parent / "config.json")


def test_wrong_cuts_clock(native: Path) -> None:
    write(native.parent / "cuts.json", {"version": 1, "clock": "output"})
    with pytest.raises(ReplayError, match="presentation clock"):
        prepare(native, native.parent / "config.json")


def test_support_wrong_label_clock(support: Path) -> None:
    write(support.parent / "labels.json", {"clock": "output", "parentSHA256": {}})
    with pytest.raises(ReplayError, match="pinned clock"):
        prepare(support, support.parent / "config.json")


def test_no_overwrite(native: Path) -> None:
    output = native.parent / "config.json"
    prepare(native, output)
    with pytest.raises(ReplayError, match="must be new"):
        prepare(native, output)


def test_output_tampering(native: Path) -> None:
    output = native.parent / "config.json"
    prepare(native, output)
    write(output, {"version": 2})
    with pytest.raises(ReplayError, match="provenance changed"):
        verify(native, output)


def test_receipt_tampering(native: Path) -> None:
    output = native.parent / "config.json"
    prepare(native, output)
    write(output.with_suffix(".binding.json"), {})
    with pytest.raises(ReplayError, match="provenance changed"):
        verify(native, output)


def test_output_scope(native: Path, tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="repository work"):
        prepare(native, tmp_path / "outside.json")


def test_artifact_symlink(native: Path) -> None:
    alias = native.parent / "alias.bin"
    alias.symlink_to(native.parent / "source.bin")
    data = json.loads(native.read_text())
    data["artifactPaths"]["source"] = str(alias)
    write(native, data)
    with pytest.raises(ReplayError, match="canonical"):
        prepare(native, native.parent / "config.json")


def test_bad_path_type(native: Path) -> None:
    data = json.loads(native.read_text())
    data["artifactPaths"]["source"] = 1
    write(native, data)
    with pytest.raises(ReplayError, match="explicit artifact"):
        prepare(native, native.parent / "config.json")


def test_empty_metadata(native: Path) -> None:
    native.write_bytes(b"")
    with pytest.raises(ReplayError, match="byte budget"):
        prepare(native, native.parent / "config.json")


def test_support_result(support: Path) -> None:
    config, output = support.parent / "config.json", support.parent / "supported.json"
    prepare(support, config)
    cut_audio_support.run(config, output)
    assert main(["verify-support", str(config), str(output)]) == 0
    data = json.loads(output.read_text())
    data["audioSupport"]["recognizedSpeechRemoved"] = True
    write(output, data)
    with pytest.raises(ReplayError, match="intersection"):
        main(["verify-support", str(config), str(output)])


def test_projected_text_tampering(native: Path) -> None:
    path = native.parent / "tokens.json"
    data = json.loads(path.read_text())
    data["tokens"][0]["text"] = "invented"
    write(path, data)
    with pytest.raises(ReplayError, match="text/clock"):
        prepare(native, native.parent / "config.json")


def test_projected_labels_changed(native: Path) -> None:
    write(native.parent / "labels.json", {"clock": "output"})
    with pytest.raises(ReplayError, match="label identity"):
        prepare(native, native.parent / "config.json")


def test_projection_verifier(native: Path) -> None:
    assert main(["verify-projection", str(native), str(native.parent / "tokens.json")]) == 0
    with pytest.raises(ReplayError, match="output differs"):
        main(["verify-projection", str(native), str(native.parent / "source.bin")])


@pytest.mark.parametrize("mutation", ["config", "pin", "fps", "parent", "pin-type"])
def test_support_result_guards(support: Path, mutation: str) -> None:
    config, output = support.parent / "config.json", support.parent / "supported.json"
    prepare(support, config)
    cut_audio_support.run(config, output)
    if mutation == "config":
        write(config, {"version": 2, "artifacts": {}})
    elif mutation == "pin":
        write(support.parent / "labels.json", {})
    elif mutation == "fps":
        cuts = support.parent / "cuts.json"
        data = json.loads(cuts.read_text())
        data["fps"] = 0
        write(cuts, data)
        data = json.loads(config.read_text())
        data["artifacts"]["cuts"] = ref(cuts)
        write(config, data)
    else:
        data = json.loads(output.read_text())
        if mutation == "parent":
            data["parentSHA256"] = {}
        else:
            data["parentSHA256"]["invalid"] = 1
        write(output, data)
    with pytest.raises(ReplayError):
        main(["verify-support", str(config), str(output)])
