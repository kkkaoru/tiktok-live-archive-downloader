import json
import sys
from pathlib import Path

import pytest

from replay import caption_native_inputs as inputs
from replay.caption_native_paging import read_tokens
from replay.core import ReplayError
from replay.delivery_captions import Cut, Token, partition
from replay.fcp_part_assembly import _reference
from replay.recordings import record


@pytest.fixture
def config(tmp_path: Path) -> Path:
    source = tmp_path / "source.mov"
    source.write_bytes(b"synthetic source")
    labels = tmp_path / "labels.json"
    labels.write_text('{"labels":[]}')
    cuts = tmp_path / "cuts.json"
    cuts.write_text(
        json.dumps(
            {"cuts": [{"source_start_frame": 0, "source_end_frame": 150, "output_start_frame": 0}]}
        )
    )
    tokens = tmp_path / "projected.json"
    tokens.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "output",
                "cuts": _reference(cuts),
                "labels": _reference(labels),
                "boundaryConflicts": [],
                "omitted": [],
                "zeroLengthLabels": [{"text": "点", "start": 4.5, "end": 4.5}],
                "tokens": [
                    {
                        "text": "認識された長い発話",
                        "start": 0.2,
                        "end": 4.1,
                        "utterance": "000000000000:3:cut-0",
                    }
                ],
            }
        )
    )
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": _reference(source),
                "cuts": _reference(cuts),
                "projectedTokens": _reference(tokens),
            }
        )
    )
    return path


def test_fresh_inputs_preserve_coarse_clock_without_fake_measurements(config: Path) -> None:
    output = config.parent / "prepared"
    index_path = inputs.prepare(config, output)
    index = json.loads(index_path.read_text())
    row = index["segments"][0]
    binding = inputs.segment_binding(output, row)
    document = json.loads(binding.read_text())
    assert document["measurementsPerformed"] is False
    assert document["kind"] == "recognition-inputs"
    assert not (output / "combined-0000.json").exists()
    tokens, _ = read_tokens(output, row, document["inputs"])
    assert tokens[0].text == "認識された長い発話"
    assert tokens[0].start == 0.2
    assert tokens[0].end == 4.1
    assert tokens[0].utterance == "000000000000:3:cut-0"
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["zeroLengthLabels"] == [{"text": "点", "start": 4.5, "end": 4.5}]
    assert receipt["positiveTokens"] == 1
    source_ref = json.loads(config.read_text())["source"]
    pins = inputs.recognition_pins(
        document, source=source_ref["path"], source_digest=source_ref["sha256"]
    )
    assert str(config) in pins
    assert source_ref["path"] not in pins
    (output / "tokens-0000.json").write_text("changed")
    with pytest.raises(ReplayError, match="hash"):
        inputs.verify(output)


def test_legacy_foreground_requirement_remains_default() -> None:
    with pytest.raises(ReplayError, match="foreground"):
        partition((Cut(0, 30, 0),), (Token("a", 0.1, 0.2, "u"),), (), rate=30)


def test_cli_verify_and_config_binding(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = config.parent / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare", str(config), str(output)])
    inputs.main()
    monkeypatch.setattr(sys, "argv", ["prepare", str(config), str(output), "--verify"])
    inputs.main()
    config.write_text("{}")
    with pytest.raises(ReplayError, match="config changed"):
        inputs.main()


def test_prepare_rejects_conflicts_and_existing_output(config: Path) -> None:
    raw = json.loads(config.read_text())
    tokens = Path(raw["projectedTokens"]["path"])
    data = json.loads(tokens.read_text())
    data["boundaryConflicts"] = [{"text": "cross-cut"}]
    tokens.write_text(json.dumps(data))
    raw["projectedTokens"] = _reference(tokens)
    config.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="conflict-free"):
        inputs.prepare(config, config.parent / "prepared")
    with pytest.raises(ReplayError, match="new canonical"):
        inputs.prepare(config, config.parent)


def test_binding_rejects_measurement_claim_and_wrong_segment(config: Path) -> None:
    output = config.parent / "prepared"
    index = json.loads(inputs.prepare(config, output).read_text())
    row = index["segments"][0]
    path = inputs.segment_binding(output, row)
    data = json.loads(path.read_text())
    data["measurementsPerformed"] = True
    path.write_text(json.dumps(data))
    row["recognitionInput"] = _reference(path)
    with pytest.raises(ReplayError, match="differs"):
        inputs.segment_binding(output, row)
    with pytest.raises(ReplayError, match="Invalid"):
        inputs.segment_binding(output, {"segment": True})
    assert inputs.segment_binding(output, {"segment": 2}).name == "combined-0002.json"


def test_fresh_provenance_rejects_wrong_hash_type_and_source(config: Path) -> None:
    assert inputs.recognition_pins({}, source="x", source_digest="h") == {}
    with pytest.raises(ReplayError, match="source differs"):
        inputs.recognition_pins(
            {"kind": "recognition-inputs", "inputs": {}}, source="x", source_digest="h"
        )
    with pytest.raises(ReplayError, match="hash"):
        inputs.recognition_pins(
            {"kind": "recognition-inputs", "inputs": {"x": "h", "bad": 3}},
            source="x",
            source_digest="h",
        )
    output = config.parent / "prepared"
    index_path = inputs.prepare(config, output)
    index_path.write_text('{"segments":[]}')
    receipt_path = output / "receipt.json"
    receipt = record(json.loads(receipt_path.read_text()))
    receipt["segments"] = _reference(index_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ReplayError, match="segment index"):
        inputs.verify(output)
