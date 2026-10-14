"""Synthetic edited clocks; no recordings, apps, media processing or legacy config."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from replay import delivery_clock_contract as clock
from replay.core import ReplayError
from replay.fcp_caption_srt import Cue, caption_cues
from replay.stages import file_digest


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(clock, "ROOT", tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    root = tmp_path / "edits/replay-123"
    root.mkdir()
    original = tmp_path / "originals/2024-02-29_09-30-59_JST_replay-123.mp4"
    original.parent.mkdir()
    original.write_bytes(b"original")
    storage = tmp_path / "library/replay-123"
    storage.mkdir(parents=True)
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
                "storage_root": str(storage),
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
    cuts = work / "cuts.json"
    cuts.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "source-presentation",
                "fps": 30,
                "cuts": [
                    {"source_start_frame": 100, "source_end_frame": 115, "output_start_frame": 0},
                    {"source_start_frame": 300, "source_end_frame": 315, "output_start_frame": 15},
                ],
            }
        ),
        encoding="utf-8",
    )
    xml = work / "semantic.fcpxml"
    xml.write_text(
        '<!DOCTYPE fcpxml><fcpxml><resources><format id="r1" width="640" height="960" '
        'frameDuration="1/30s"/></resources><library><event><project>'
        '<sequence format="r1" duration="1s" tcStart="0s"><spine>'
        '<asset-clip offset="0s" start="10/3s" duration="1/2s">'
        '<title offset="10/3s" duration="1/2s"><text>First</text></title></asset-clip>'
        '<asset-clip offset="1/2s" start="10s" duration="1/2s">'
        '<title offset="10s" duration="1/2s"><text>Second</text></title></asset-clip>'
        "</spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    path = work / "clock-expectation.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "delivery-clock-expectation",
                "editConfig": pin(edit),
                "cuts": pin(cuts),
                "semanticXML": pin(xml),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_source_jumps_and_complete_input_binding(contract: Path) -> None:
    result = clock.load_clock_contract(contract)
    assert (result.rate, result.width, result.height, result.expected_frames) == (30, 640, 960, 30)
    assert result.delivery_root == contract.parents[2] / "library/replay-123"
    assert result.path == contract
    assert result.cuts == contract.parent / "cuts.json"
    assert len(result.inputs) == 4
    assert result.inputs[0] == clock.BoundInput(contract, file_digest(contract))
    assert result.inputs[1].path.name == "edit-config.json"
    assert result.inputs[2].sha256 == file_digest(contract.parent / "cuts.json")
    assert result.inputs[3].sha256 == file_digest(contract.parent / "semantic.fcpxml")
    with pytest.raises(FrozenInstanceError):
        result.__setattr__("rate", 25)


def test_other_rate_geometry_and_three_retained_spans(contract: Path) -> None:
    cuts = contract.parent / "cuts.json"
    cuts.write_text(
        '{"clock":"source-presentation","fps":25,"cuts":['
        '{"source_start_frame":0,"source_end_frame":5,"output_start_frame":0},'
        '{"source_start_frame":20,"source_end_frame":30,"output_start_frame":5},'
        '{"source_start_frame":100,"source_end_frame":110,"output_start_frame":15}]}',
        encoding="utf-8",
    )
    xml = contract.parent / "semantic.fcpxml"
    xml.write_text(
        '<fcpxml><resources><format id="r9" width="1920" height="1080" '
        'frameDuration="1/25s"/><format id="unused"/></resources><library><event><project>'
        '<sequence format="r9" duration="1s"><spine>'
        '<asset-clip offset="0s" start="0s" duration="1/5s">'
        '<title offset="0s" duration="1/5s"><text>A</text></title></asset-clip>'
        '<asset-clip offset="1/5s" start="4/5s" duration="2/5s">'
        '<title offset="4/5s" duration="2/5s"><text>B</text></title></asset-clip>'
        '<asset-clip offset="3/5s" start="4s" duration="2/5s">'
        '<title offset="4s" duration="2/5s"><text>C</text></title></asset-clip>'
        "</spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    raw = json.loads(contract.read_text(encoding="utf-8"))
    raw["cuts"] = pin(cuts)
    raw["semanticXML"] = pin(xml)
    contract.write_text(json.dumps(raw), encoding="utf-8")
    result = clock.load_clock_contract(contract)
    assert (result.rate, result.width, result.height, result.expected_frames) == (
        25,
        1920,
        1080,
        25,
    )
    assert len(result.inputs) == 4


@pytest.mark.parametrize("field", ["editConfig", "cuts", "semanticXML"])
def test_changed_pinned_input_is_rejected(contract: Path, field: str) -> None:
    raw = json.loads(contract.read_text(encoding="utf-8"))
    path = Path(raw[field]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Pinned"):
        clock.load_clock_contract(contract)


@pytest.mark.parametrize(
    "change", [{"version": True}, {"version": 2}, {"kind": "legacy"}, {"rate": 30}]
)
def test_contract_schema_is_exact(contract: Path, change: dict[str, object]) -> None:
    raw = json.loads(contract.read_text(encoding="utf-8"))
    raw.update(change)
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="schema"):
        clock.load_clock_contract(contract)


@pytest.mark.parametrize("text", ['{"version":1,"version":1}', '{"x":NaN}', "[]", "{"])
def test_json_boundary_is_strict(contract: Path, text: str) -> None:
    contract.write_text(text, encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        clock.load_clock_contract(contract)


@pytest.mark.parametrize("field", ["editConfig", "cuts", "semanticXML"])
def test_input_ref_rejects_extra_fields(contract: Path, field: str) -> None:
    raw = json.loads(contract.read_text(encoding="utf-8"))
    raw[field]["accepted"] = True
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="exactly"):
        clock.load_clock_contract(contract)


@pytest.mark.parametrize("digest", [None, 123, "x" * 64, "A" * 64])
def test_invalid_sha_is_not_a_pin(contract: Path, digest: object) -> None:
    raw = json.loads(contract.read_text(encoding="utf-8"))
    raw["cuts"]["sha256"] = digest
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="exact SHA"):
        clock.load_clock_contract(contract)


@pytest.mark.parametrize("path", [None, "", "\x00", "relative", "/outside/input.json"])
def test_outside_or_invalid_ref_is_rejected_before_open(contract: Path, path: object) -> None:
    raw = json.loads(contract.read_text(encoding="utf-8"))
    raw["cuts"]["path"] = path
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="paths|storage"):
        clock.load_clock_contract(contract)


@pytest.mark.parametrize("folder", ["private", "downloads"])
def test_account_storage_is_never_opened(contract: Path, folder: str) -> None:
    raw = json.loads(contract.read_text(encoding="utf-8"))
    raw["cuts"]["path"] = str(contract.parent / folder / "do-not-open")
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="storage"):
        clock.load_clock_contract(contract)


def test_symlinked_pin_is_rejected(contract: Path) -> None:
    linked = contract.parent / "linked.json"
    linked.symlink_to(contract.parent / "cuts.json")
    raw = json.loads(contract.read_text(encoding="utf-8"))
    raw["cuts"]["path"] = str(linked)
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="unlinked"):
        clock.load_clock_contract(contract)


def test_contract_must_be_work_local(contract: Path) -> None:
    outside = contract.parents[1] / "other.json"
    outside.write_bytes(contract.read_bytes())
    with pytest.raises(ReplayError, match="storage"):
        clock.load_clock_contract(outside)


@pytest.mark.parametrize(
    "change", [{"clock": "source-physical"}, {"fps": 25}, {"fps": 30.0}, {"cuts": []}, {"cuts": {}}]
)
def test_cut_clock_domain_and_nonempty_rows(contract: Path, change: dict[str, object]) -> None:
    cuts = contract.parent / "cuts.json"
    raw = json.loads(cuts.read_text(encoding="utf-8"))
    raw.update(change)
    cuts.write_text(json.dumps(raw), encoding="utf-8")
    control = json.loads(contract.read_text(encoding="utf-8"))
    control["cuts"] = pin(cuts)
    contract.write_text(json.dumps(control), encoding="utf-8")
    with pytest.raises(ReplayError, match="clock|rows"):
        clock.load_clock_contract(contract)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("output_start_frame", 16, "contiguous"),
        ("output_start_frame", True, "contiguous"),
        ("output_start_frame", None, "contiguous"),
        ("source_start_frame", 110, "ordered"),
        ("source_end_frame", 300, "positive"),
    ],
)
def test_cut_topology_guards(contract: Path, field: str, value: object, message: str) -> None:
    cuts = contract.parent / "cuts.json"
    raw = json.loads(cuts.read_text(encoding="utf-8"))
    raw["cuts"][1][field] = value
    cuts.write_text(json.dumps(raw), encoding="utf-8")
    control = json.loads(contract.read_text(encoding="utf-8"))
    control["cuts"] = pin(cuts)
    contract.write_text(json.dumps(control), encoding="utf-8")
    with pytest.raises(ReplayError, match=message):
        clock.load_clock_contract(contract)


@pytest.mark.parametrize(
    "old,new,message",
    [
        ('width="640"', 'width="0"', "dimensions"),
        ('height="960"', 'height="16385"', "bounded"),
        ('frameDuration="1/30s"', 'frameDuration="0s"', "rate"),
        ('frameDuration="1/30s"', 'frameDuration="1001/30000s"', "rate"),
        ('frameDuration="1/30s"', 'frameDuration="1/241s"', "bounded"),
        ('format="r1" duration=', 'format="missing" duration=', "referenced"),
        ('duration="1s" tcStart=', 'duration="2s" tcStart=', "coverage"),
        ('tcStart="0s"', 'tcStart="1s"', "zero"),
        ('id="r1"', 'id=""', "identifiers"),
        ("</resources>", '<format id="r1"/></resources>', "identifiers"),
        ("<sequence format=", "<sequence/><sequence format=", "exactly one"),
        ("<sequence format=", "<gap><sequence format=", "exactly one"),
        ("<fcpxml>", "<wrong>", "root"),
        ("</fcpxml>", "", "invalid"),
        ("<!DOCTYPE fcpxml>", '<!DOCTYPE fcpxml [<!ENTITY a "x">]>', "declarations"),
        ("<!DOCTYPE fcpxml>", '<!DOCTYPE fcpxml SYSTEM "file:///not-read">', "declarations"),
    ],
)
def test_semantic_geometry_and_safe_xml(contract: Path, old: str, new: str, message: str) -> None:
    xml = contract.parent / "semantic.fcpxml"
    xml.write_text(xml.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    raw = json.loads(contract.read_text(encoding="utf-8"))
    raw["semanticXML"] = pin(xml)
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match=message):
        clock.load_clock_contract(contract)


@pytest.mark.parametrize(
    "budget,value", [("MAX_CONTRACT_BYTES", 1), ("MAX_XML_ELEMENTS", 1), ("MAX_XML_DEPTH", 1)]
)
def test_input_and_structure_budgets(
    contract: Path, monkeypatch: pytest.MonkeyPatch, budget: str, value: int
) -> None:
    monkeypatch.setattr(clock, budget, value)
    with pytest.raises((ReplayError, ValueError), match="budget"):
        clock.load_clock_contract(contract)


@pytest.mark.parametrize("budget,value", [("MAX_FRAMES", 29), ("MAX_SECONDS", 0)])
def test_edited_duration_decoder_budget(
    contract: Path, monkeypatch: pytest.MonkeyPatch, budget: str, value: int
) -> None:
    monkeypatch.setattr(clock, budget, value)
    with pytest.raises(ReplayError, match="budget|rows"):
        clock.load_clock_contract(contract)


def test_missing_authorized_storage(contract: Path) -> None:
    raw = json.loads(contract.read_text(encoding="utf-8"))
    edit = Path(raw["editConfig"]["path"])
    data = json.loads(edit.read_text(encoding="utf-8"))
    del data["storage_root"]
    edit.write_text(json.dumps(data), encoding="utf-8")
    raw["editConfig"] = pin(edit)
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="authorized"):
        clock.load_clock_contract(contract)


def test_same_input_role_is_rejected(contract: Path) -> None:
    raw = json.loads(contract.read_text(encoding="utf-8"))
    raw["cuts"] = raw["editConfig"]
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="distinct"):
        clock.load_clock_contract(contract)


def test_wrong_edit_config_basename(contract: Path) -> None:
    raw = json.loads(contract.read_text(encoding="utf-8"))
    other = contract.parent / "other.json"
    other.write_bytes(Path(raw["editConfig"]["path"]).read_bytes())
    raw["editConfig"] = pin(other)
    contract.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="named edit-config"):
        clock.load_clock_contract(contract)


def test_xml_output_duration_must_equal_cuts(contract: Path) -> None:
    cuts = contract.parent / "cuts.json"
    raw = json.loads(cuts.read_text(encoding="utf-8"))
    raw["cuts"][1]["source_end_frame"] = 330
    cuts.write_text(json.dumps(raw), encoding="utf-8")
    control = json.loads(contract.read_text(encoding="utf-8"))
    control["cuts"] = pin(cuts)
    contract.write_text(json.dumps(control), encoding="utf-8")
    with pytest.raises(ReplayError, match="duration differs"):
        clock.load_clock_contract(contract)


def test_terminal_input_binding_rejects_mutation(
    contract: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(path: Path) -> tuple[Cue, ...]:
        caption_cues(path)
        path.write_bytes(path.read_bytes() + b" ")
        return ()

    monkeypatch.setattr(clock, "caption_cues", mutate)
    with pytest.raises(ValueError, match="Pinned"):
        clock.load_clock_contract(contract)
