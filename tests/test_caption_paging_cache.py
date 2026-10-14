import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

from replay import caption_native_layout
from replay import caption_paging_cache as cache
from replay.caption_native_layout import NativeCaptionLayout, NativeCaptionStyle
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest


def native(argv: tuple[str, ...], output: Path, timeout: int) -> None:
    assert timeout == 60
    request = record(json.loads(Path(argv[-1]).read_text()))
    texts = request["texts"]
    assert isinstance(texts, list)
    output.write_text(
        json.dumps(
            {
                "motionRenderVerified": False,
                "font": {"postScriptName": "Synthetic", "lineAdvance": 12, "pointSize": 10},
                "bounds": [
                    {"text": text, "width": 20, "height": 12, "lineWidths": [20]} for text in texts
                ],
            }
        )
    )


@dataclass
class Fixture:
    root: Path
    part: Path
    current: NativeCaptionLayout
    evidence: tuple[Path, ...]
    asset: Path


@pytest.fixture
def case(tmp_path: Path) -> Fixture:
    executable, asset = tmp_path / "native", tmp_path / "font.ttf"
    executable.write_text("synthetic native")
    asset.write_bytes(b"synthetic font")
    proof, calibration, template = (
        tmp_path / "font.json",
        tmp_path / "cal.json",
        tmp_path / "t.moti",
    )
    proof.write_text(
        json.dumps(
            {
                "primary": {
                    "postScriptName": "Synthetic",
                    "pointSize": 10,
                    "variations": [{"identifier": 1, "value": 700}],
                },
                "fonts": [{"path": str(asset), "sha256": file_digest(asset)}],
            }
        )
    )
    calibration.write_text("{}")
    template.write_text("synthetic template")
    evidence = (proof, calibration, template)
    style = NativeCaptionStyle("Synthetic", 12, 1, 1, 14)
    root = tmp_path / "old"
    root.mkdir()
    old = NativeCaptionLayout(
        executable=executable,
        directory=root / "metrics-0000",
        style=style,
        width=80,
        height=4096,
        evidence=evidence,
        expected_point_size=10,
        execute=native,
    )
    old.prepare_lines(("old", "word"))
    part = root / "part-0000.json"
    part.write_text(
        json.dumps(
            {
                "settings": {"style": asdict(style), "pointSize": 10, "width": 80},
                "inputs": old.verify_inputs(),
                "captions": "NEVER READ",
            }
        )
    )
    current = NativeCaptionLayout(
        executable=executable,
        directory=tmp_path / "fresh",
        style=style,
        width=80,
        height=4096,
        evidence=evidence,
        expected_point_size=10,
        execute=native,
    )
    return Fixture(root, part, current, evidence, asset)


def load(case: Fixture) -> cache.PagingMeasurementCache:
    return cache.reuse_measurements(case.current, root=case.root, segment=0, evidence=case.evidence)


@pytest.mark.parametrize("relative", [False, True])
def test_reuse_only_geometry_and_measure_misses(
    case: Fixture, monkeypatch: pytest.MonkeyPatch, relative: bool
) -> None:
    before = case.part.read_bytes()
    monkeypatch.chdir(case.root.parent)
    measured = cache.reuse_measurements(
        case.current,
        root=case.root,
        segment=0,
        evidence=tuple(path.relative_to(case.root.parent) for path in case.evidence)
        if relative
        else case.evidence,
    )
    assert measured.measure("old\nword") == (20, 26)
    assert not tuple(case.current.directory.glob("*.response.json"))
    measured.prepare_lines(("old", "new", "new"))
    assert len(tuple(case.current.directory.glob("*.response.json"))) == 1
    request = json.loads((case.current.directory / "batch-000000.request.json").read_text())
    assert request["texts"] == ["new"]
    pins = measured.verify_inputs()
    assert pins[str(case.part)] == file_digest(case.part)
    assert pins[str(case.asset)] == file_digest(case.asset)
    assert case.part.read_bytes() == before
    case.asset.write_bytes(b"changed font")
    with pytest.raises(ReplayError, match="identity changed"):
        measured.verify_inputs()


@pytest.mark.parametrize(
    "field", ["style", "pointSize", "width", "producer", "spec", "evidence", "hash", "empty"]
)
def test_mismatched_identity_rejected(case: Fixture, field: str) -> None:
    data = record(json.loads(case.part.read_text()))
    settings, inputs = record(data["settings"]), record(data["inputs"])
    if field == "style":
        record(settings["style"])["border"] = True
    elif field in {"pointSize", "width"}:
        settings[field] = 999
    elif field == "producer":
        inputs[str(case.current.executable)] = "0" * 64
    elif field == "spec":
        inputs[str(Path(caption_native_layout.__file__).resolve())] = "0" * 64
    elif field == "evidence":
        del inputs[str(case.evidence[0])]
    elif field == "hash":
        inputs[str(case.evidence[0])] = "bad"
    else:
        data["inputs"] = {}
    case.part.write_text(json.dumps(data))
    with pytest.raises((ReplayError, ValueError)):
        load(case)


@pytest.mark.parametrize(
    "change", ["fonts", "primary", "axis", "duplicate", "value", "asset", "linked", "absent"]
)
def test_font_proof_rejected(case: Fixture, change: str) -> None:
    path = case.evidence[0]
    data = record(json.loads(path.read_text()))
    primary = record(data["primary"])
    if change == "fonts":
        data["fonts"] = []
    elif change == "primary":
        primary["pointSize"] = 11
    elif change in {"axis", "duplicate", "value"}:
        primary["variations"] = [
            {
                "identifier": True if change == "axis" else 1,
                "value": "bad" if change == "value" else 700,
            }
        ]
        if change == "duplicate":
            primary["variations"] = [{"identifier": 1, "value": 700}] * 2
    elif change == "asset":
        case.asset.write_text("changed")
    elif change == "linked":
        link = case.asset.with_name("link.ttf")
        link.symlink_to(case.asset)
        data["fonts"] = [{"path": str(link), "sha256": file_digest(case.asset)}]
    else:
        data = {}
    path.write_text(json.dumps(data))
    # A current run built using this new proof still cannot use invalid data.
    case.current._inputs[str(path)] = file_digest(path)
    part = record(json.loads(case.part.read_text()))
    record(part["inputs"])[str(path)] = file_digest(path)
    case.part.write_text(json.dumps(part))
    with pytest.raises((ReplayError, ValueError)):
        load(case)


@pytest.mark.parametrize(
    "change", ["missing", "rows", "widths", "bool", "request", "duplicatejson"]
)
def test_invalid_measurement_json(case: Fixture, change: str) -> None:
    path = case.root / "metrics-0000/batch-000000.response.json"
    part = record(json.loads(case.part.read_text()))
    inputs = record(part["inputs"])
    if change == "missing":
        del inputs[str(path)]
    else:
        data = record(json.loads(path.read_text()))
        if change == "request":
            path = path.with_name("batch-000000.request.json")
            data = record(json.loads(path.read_text()))
            data["rim"] = True
        elif change == "rows":
            data["bounds"] = []
        elif change in {"widths", "bool"}:
            data["bounds"] = [
                {
                    "text": "old",
                    "width": 20,
                    "height": 12,
                    "lineWidths": [] if change == "widths" else [True],
                }
            ]
        path.write_text(
            '{"bounds":[],"bounds":[]}' if change == "duplicatejson" else json.dumps(data)
        )
        inputs[str(path)] = file_digest(path)
    case.part.write_text(json.dumps(part))
    with pytest.raises((ReplayError, ValueError)):
        load(case)


def test_scope_and_count_guards(case: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    for segment, evidence in [(True, case.evidence), (-1, case.evidence), (0, ())]:
        with pytest.raises(ReplayError):
            cache.reuse_measurements(
                case.current, root=case.root, segment=segment, evidence=evidence
            )
    with pytest.raises(ReplayError, match="root"):
        cache.reuse_measurements(
            case.current, root=case.root / "absent", segment=0, evidence=case.evidence
        )
    monkeypatch.setattr(cache, "MAX_PINS", 1)
    with pytest.raises(ReplayError, match="count"):
        load(case)


def test_linked_part_rejected(case: Fixture) -> None:
    moved = case.part.with_suffix(".bytes")
    case.part.rename(moved)
    case.part.symlink_to(moved)
    with pytest.raises(ValueError, match="Linked"):
        load(case)


@pytest.mark.parametrize("valid", [True, False])
def test_failed_planning_reuses_only_verified_measurements(case: Fixture, valid: bool) -> None:
    data = record(json.loads(case.part.read_text()))
    data.update(
        {
            "version": 1,
            "segment": 0,
            "candidateOnly": valid,
            "error": "Native token exceeds display budget; example",
        }
    )
    failure = case.part.with_name("failure-0000.json")
    case.part.rename(case.part.with_suffix(".retained"))
    failure.write_text(json.dumps(data))
    if valid:
        measured = load(case)
        assert measured.measure("old")[0] == 20
        assert not tuple(case.current.directory.glob("*.response.json"))
        assert str(failure) in measured.verify_inputs()
    else:
        with pytest.raises(ReplayError, match="failed-planning"):
            load(case)


def test_cache_reuses_transitively_pinned_measurement_directories(case: Fixture) -> None:
    data = record(json.loads(case.part.read_text()))
    inputs = record(data["inputs"])
    older = case.root / "previous-measurements"
    older.mkdir()
    for path in (case.root / "metrics-0000").glob("*.json"):
        if path.name == "inputs.json":
            continue
        moved = older / path.name
        path.rename(moved)
        inputs[str(moved)] = inputs.pop(str(path))
    case.part.write_text(json.dumps(data))
    measured = load(case)
    assert measured.measure("old")[0] == 20
    assert not tuple(case.current.directory.glob("*.response.json"))
