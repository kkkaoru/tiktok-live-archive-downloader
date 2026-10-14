import json
from pathlib import Path

import pytest

from replay import candidate_publication, delivery_verification, workflow
from replay import youtube_delivery_validation as artifact
from replay import youtube_delivery_workflow as delivery
from replay.artifact_json import _json
from replay.core import ReplayError
from replay.delivery_render import load_config
from replay.stages import file_digest


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def config(tmp_path: Path) -> Path:
    library, edit = tmp_path / "library", tmp_path / "edits/replay-test"
    paths = {key: tmp_path / f"{key}.fixture" for key in delivery.PINS}
    paths["movie"] = library / edit.name / "final.mp4"
    paths["editConfig"] = edit / "edit-config.json"
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic fixture")
    write(
        paths["cuts"],
        {"cuts": [{"source_start_frame": 0, "source_end_frame": 60, "output_start_frame": 0}]},
    )
    render: dict[str, object] = {
        "version": 1,
        "recording": edit.name,
        "rate": 30,
        "width": 1080,
        "height": 1920,
        "cuts": str(paths["cuts"]),
        "source": str(tmp_path / "raw.mp4"),
        "deliveryRoot": str(library / edit.name),
        "motion": {"template": str(paths["motionTemplate"]), "expectedText": "Title", "finish": []},
        "encoding": {},
        "audio": {"source": str(paths["master"]), "clock": str(paths["cuts"])},
    }
    for key in (
        "captions",
        "work",
        "storage",
        "sourceChunks",
        "alphaPreset",
        "encodePreset",
        "native",
        "finalPreset",
        "calibration",
        "viewerRoot",
    ):
        render[key] = str(tmp_path / key)
    write(paths["renderConfig"], render)
    paths["semanticXML"].write_text(
        '<fcpxml><resources><effect id="title" uid="template-uid" '
        f'src="{paths["motionTemplate"].as_uri()}"/></resources>'
        '<library><event><project><sequence duration="2s"><spine>'
        '<asset-clip ref="video" offset="0s" start="0s" duration="2s"><title ref="title"/>'
        "</asset-clip>"
        "</spine></sequence></project></event></library></fcpxml>"
    )
    write(
        paths["assemblyLineage"],
        {
            "semanticXML": ref(paths["semanticXML"]),
            "assemblyXML": ref(paths["assemblyXML"]),
            "duration": "2s",
        },
    )
    return write(
        tmp_path / "config.json",
        {
            "version": 1,
            **{key: ref(path) for key, path in paths.items()},
            "templateUID": "template-uid",
            "evidenceRoot": str(tmp_path / "evidence"),
            "editRoot": str(edit),
            "library": str(library),
            "revision": "new-caption",
            "previousSelectionSHA256": "a" * 64,
        },
    )


def loaded(config: Path) -> delivery.Config:
    return delivery.load(config, sha256=file_digest(config))


def accepted(config: delivery.Config) -> None:
    first, second, _ = delivery.reports(config)
    write(
        first,
        {
            "technicalPropertiesAccepted": True,
            "movie": str(config.pins["movie"]),
            "master": str(config.pins["master"]),
            "inputStatIdentity": [
                delivery._identity(config.pins[key]) for key in ("movie", "master")
            ],
        },
    )
    inputs = [
        delivery_verification._fingerprint(config.pins[key])
        for key in ("renderConfig", "cuts", "movie")
    ]
    write(
        second,
        {
            "technicalAccepted": True,
            "fullDecode": True,
            "expectedVideoContract": "h264",
            "inputs": inputs,
            "inputsAfter": inputs,
            "expected": {"frames": 60, "rate": 30, "width": 1080, "height": 1920},
            "videoClock": {"frames": 60, "start": "0", "end": "2", "exactCFR": True},
            "audioDecodedClock": {
                "count": 1,
                "sampleCount": 96000,
                "start": "0",
                "end": "2",
                "span": "2",
            },
            "audioPacketClock": {
                "count": 2,
                "durationTicksSum": 96000,
                "start": "0",
                "end": "2",
                "span": "2",
            },
        },
    )


def test_build_graph_checks_real_dependencies_without_execution(config: Path) -> None:
    item = loaded(config)
    plan = delivery.build(config, item.evidence)
    assert workflow.main([str(plan), "--state", str(item.evidence / "state"), "--check"]) == 0
    stages = workflow.parse_stages(_json(plan))
    assert [stage.name for stage in stages] == list(delivery.STAGES)
    for stage in stages:
        assert set(item.pins.values()).issubset(stage.inputs)
        assert item.pins["semanticXML"] in stage.inputs
        assert item.pins["motionTemplate"] in stage.inputs
    assert set(delivery.reports(item)).issubset(stages[-1].inputs)
    assert not delivery.reports(item)[0].exists()
    with pytest.raises(ReplayError, match="fresh"):
        delivery.build(config, item.evidence)


@pytest.mark.parametrize("key", ["semanticXML", "motionTemplate", "movie", "master", "cuts"])
def test_mutated_pin_stops_before_any_validation(
    config: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    item = loaded(config)
    item.pins[key].write_bytes(b"changed")
    monkeypatch.setattr(
        artifact, "validate", lambda *a, **k: pytest.fail("Must not inspect stale media")
    )
    with pytest.raises(ReplayError, match="hash changed"):
        delivery.run(config, sha256=file_digest(config), stage="artifact")


@pytest.mark.parametrize(
    "fault",
    [
        "lineage",
        "template",
        "duration",
        "raw",
        "master",
        "cuts",
        "edit",
        "revision",
        "cas",
        "schema",
        "outside",
    ],
)
def test_configuration_refuses_wrong_provenance(config: Path, fault: str) -> None:
    item, raw = loaded(config), _json(config)
    if fault == "lineage":
        write(item.pins["assemblyLineage"], {"semanticXML": raw["assemblyXML"]})
        raw["assemblyLineage"] = ref(item.pins["assemblyLineage"])
    elif fault in {"template", "duration"}:
        xml = item.pins["semanticXML"]
        xml.write_text(
            xml.read_text().replace(
                "template-uid" if fault == "template" else 'duration="2s"',
                "wrong-uid" if fault == "template" else 'duration="3s"',
            )
        )
        raw["semanticXML"] = ref(xml)
        lineage = _json(item.pins["assemblyLineage"])
        write(item.pins["assemblyLineage"], {**lineage, "semanticXML": raw["semanticXML"]})
        raw["assemblyLineage"] = ref(item.pins["assemblyLineage"])
    elif fault == "raw":
        render = _json(item.pins["renderConfig"])
        write(item.pins["renderConfig"], {**render, "source": str(item.pins["movie"])})
        raw["renderConfig"] = ref(item.pins["renderConfig"])
    elif fault == "master":
        raw["movie"] = raw["master"]
    elif fault == "cuts":
        raw["cuts"] = raw["master"]
    elif fault == "edit":
        raw["editConfig"] = raw["master"]
    elif fault == "revision":
        raw["revision"] = "../escape"
    elif fault == "cas":
        raw["previousSelectionSHA256"] = "not-a-hash"
    elif fault == "schema":
        raw["unknown"] = True
    else:
        raw["evidenceRoot"] = str(delivery.REPOSITORY.parent / "outside")
    write(config, raw)
    with pytest.raises(ReplayError):
        loaded(config)


@pytest.mark.parametrize(
    "report,key,value",
    [
        (0, "technicalPropertiesAccepted", False),
        (0, "movie", "/wrong.mp4"),
        (0, "inputStatIdentity", []),
        (1, "technicalAccepted", False),
        (1, "expectedVideoContract", "hevc"),
        (1, "fullDecode", False),
        (1, "inputs", []),
        (1, "inputsAfter", []),
        (1, "expected", {}),
        (1, "videoClock", {}),
        (1, "audioDecodedClock", {}),
        (1, "audioPacketClock", {"count": 0}),
        (
            1,
            "audioPacketClock",
            {"count": 1, "durationTicksSum": 1, "start": "0", "end": "1", "span": "2"},
        ),
    ],
)
def test_gate_rejects_failed_or_wrong_movie_report(
    config: Path, report: int, key: str, value: object
) -> None:
    item = loaded(config)
    accepted(item)
    path = delivery.reports(item)[report]
    write(path, {**_json(path), key: value})
    with pytest.raises(ReplayError):
        delivery.gate(item)


def test_run_report_gate_and_explicit_cas_publication(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = loaded(config)
    item.evidence.mkdir()
    calls: list[str] = []

    def inspect(*args: object, **kwargs: object) -> bool:
        calls.append("artifact")
        return True

    def decode(*args: object, **kwargs: object) -> bool:
        assert kwargs["video_contract"] == "h264"
        calls.append("decode")
        accepted(item)
        return True

    def publish(options: candidate_publication.Publication) -> Path:
        assert options.previous_selection_sha256 == "a" * 64
        assert options.fcpxml == item.pins["semanticXML"]
        assert options.movie == item.pins["movie"]
        calls.append("publish")
        return write(item.evidence / "selection-fixture.json", {"status": "unverified-candidate"})

    monkeypatch.setattr(artifact, "validate", inspect)
    monkeypatch.setattr(delivery_verification, "verify", decode)
    monkeypatch.setattr(candidate_publication, "publish", publish)
    for stage in delivery.STAGES:
        delivery.run(config, sha256=file_digest(config), stage=stage)
    assert calls == ["artifact", "decode", "publish"]
    assert _json(item.evidence / "publication.json")["deliveryApproved"] is False
    write(delivery.reports(item)[2], {})
    with pytest.raises(ReplayError, match="gate changed"):
        delivery.run(config, sha256=file_digest(config), stage="publish")


@pytest.mark.parametrize("stage", ["artifact", "decode", "unknown"])
def test_failed_runtime_never_advances(
    config: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    monkeypatch.setattr(artifact, "validate", lambda *a, **k: False)
    monkeypatch.setattr(delivery_verification, "verify", lambda *a, **k: False)
    with pytest.raises(ReplayError):
        delivery.run(config, sha256=file_digest(config), stage=stage)


def test_cli_build_and_errors(config: Path) -> None:
    assert delivery.main(["build", str(config), str(loaded(config).evidence)]) == 0
    assert delivery.main(["run", str(config), "--sha256", "0" * 64, "--stage", "gate"]) == 1


def test_lineage_duration_cannot_disagree_with_semantic(config: Path) -> None:
    item, raw = loaded(config), _json(config)
    path = item.pins["assemblyLineage"]
    write(path, {**_json(path), "duration": "3s"})
    write(config, {**raw, "assemblyLineage": ref(path)})
    with pytest.raises(ReplayError, match="duration"):
        loaded(config)


@pytest.mark.parametrize(
    "change", ["source", "offset", "duration", "nested", "retime", "refclip", "gap", "absent"]
)
def test_semantic_cut_clock_rejects_same_duration_wrong_source_or_unsupported_structure(
    config: Path, change: str
) -> None:
    item, raw = loaded(config), _json(config)
    path = item.pins["semanticXML"]
    text = path.read_text()
    changes = {
        "source": ('start="0s"', 'start="3s"'),
        "offset": ('offset="0s"', 'offset="1s"'),
        "duration": ('duration="2s"><title', 'duration="1s"><title'),
        "nested": ('<title ref="title"/>', '<asset-clip ref="video"/><title ref="title"/>'),
        "retime": ('<title ref="title"/>', '<timeMap/><title ref="title"/>'),
        "refclip": ('<title ref="title"/>', '<ref-clip/><title ref="title"/>'),
        "gap": ('<asset-clip ref="video"', '<gap ref="video"'),
        "absent": ("<spine>", "<other>"),
    }
    before, after = changes[change]
    text = text.replace(before, after)
    if change == "gap":
        text = text.replace("</asset-clip>", "</gap>")
    if change == "absent":
        text = text.replace("</spine>", "</other>")
    path.write_text(text)
    raw["semanticXML"] = ref(path)
    lineage = item.pins["assemblyLineage"]
    write(lineage, {**_json(lineage), "semanticXML": ref(path)})
    raw["assemblyLineage"] = ref(lineage)
    write(config, raw)
    with pytest.raises(ReplayError):
        loaded(config)


@pytest.mark.parametrize("second_ref", ["video", "other-video"])
def test_ordered_source_cuts_and_connected_processed_audio(config: Path, second_ref: str) -> None:
    item = loaded(config)
    write(
        item.pins["cuts"],
        {
            "cuts": [
                {"source_start_frame": 0, "source_end_frame": 30},
                {"source_start_frame": 90, "source_end_frame": 120},
            ]
        },
    )
    from xml.etree import ElementTree as ET

    semantic = ET.fromstring(
        '<fcpxml><sequence><spine><asset-clip ref="video" offset="0s" start="0s" duration="1s">'
        '<asset-clip ref="voice" srcEnable="audio"/></asset-clip>'
        f'<asset-clip ref="{second_ref}" offset="1s" start="3s" duration="1s"/>'
        "</spine></sequence></fcpxml>"
    )
    sequence = semantic.find("sequence")
    assert sequence is not None
    render = load_config(item.pins["renderConfig"])
    if second_ref == "video":
        delivery._cut_clock(semantic, sequence, render)
    else:
        with pytest.raises(ReplayError, match="uncut source clock"):
            delivery._cut_clock(semantic, sequence, render)
