import json
from pathlib import Path
from typing import Never

import pytest

from replay import summary_caption_inputs as summary
from replay.caption_token_sources import TokenSource
from replay.core import ReplayError
from replay.stages import file_digest
from replay.workflow import load_stages, stage_graph


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    (tmp_path / "edits" / "work").mkdir(parents=True)
    captions = tmp_path / "captions"
    captions.mkdir()
    (captions / "tokens-0000.json").write_text(
        '{"version":1,"clock":"output","tokens":[{"text":"やばい","start":0.2,'
        '"end":0.4,"utterance":"first"},{"text":"やばい","start":0.4,"end":0.6,'
        '"utterance":"second"}]}',
        encoding="utf-8",
    )
    (captions / "tokens-0001.json").write_text(
        '{"version":1,"clock":"output","tokens":[{"text":"次です","start":0.1,'
        '"end":0.4,"utterance":"next"}]}',
        encoding="utf-8",
    )
    index = captions / "segments.json"
    index.write_text(
        '{"version":1,"rate":30,"segments":[{"segment":0,"offsetFrame":0,'
        '"frames":30,"jobs":[0]},{"segment":1,"offsetFrame":30,"frames":30,'
        '"jobs":[1]}]}',
        encoding="utf-8",
    )
    reflow = tmp_path / "reflow.json"
    reflow.write_text(
        json.dumps(
            {
                "version": 1,
                "inputs": {
                    str(index): file_digest(index),
                    str(captions / "tokens-0000.json"): file_digest(captions / "tokens-0000.json"),
                    str(captions / "tokens-0001.json"): file_digest(captions / "tokens-0001.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    xml = tmp_path / "display.fcpxml"
    xml.write_text(
        '<fcpxml><library><event><project><sequence duration="2s"><spine>'
        '<asset-clip offset="0s" start="10s" duration="2s">'
        '<title offset="10s" duration="1s"><text>表示</text></title>'
        "</asset-clip></spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    srt = tmp_path / "display.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n表示\n", encoding="utf-8")
    media = tmp_path / "actual.mov"
    media.write_bytes(b"actual regular media identity only")
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "captions": str(captions),
                "media": str(media),
                "segmentIndex": {"path": str(index), "sha256": file_digest(index)},
                "reflow": {"path": str(reflow), "sha256": file_digest(reflow)},
                "displayXML": {"path": str(xml), "sha256": file_digest(xml)},
                "displaySRT": {"path": str(srt), "sha256": file_digest(srt)},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_preserves_repetition_local_clock_and_distinct_display(config: Path) -> None:
    loaded = summary.load(config)
    assert loaded.token_count == 3
    assert loaded.manifest["clockDomain"] == "edited-output"
    assert loaded.manifest["candidateOnly"] is True
    assert loaded.manifest["recognitionAccuracyVerified"] is False
    assert loaded.manifest["renderedDeliveryVerified"] is False
    jobs = loaded.manifest["jobs"]
    assert isinstance(jobs, list)
    assert len(jobs) == 2
    tokens = json.loads((config.parent / "captions" / "tokens-0000.json").read_text())
    assert tokens["tokens"] == [
        {"text": "やばい", "start": 0.2, "end": 0.4, "utterance": "first"},
        {"text": "やばい", "start": 0.4, "end": 0.6, "utterance": "second"},
    ]


def _refresh_document_pins(config: Path, paths: list[Path]) -> None:
    reflow = config.parent / "reflow.json"
    bindings = json.loads(reflow.read_text())
    for path in paths:
        bindings["inputs"][str(path)] = file_digest(path)
    reflow.write_text(json.dumps(bindings), encoding="utf-8")
    settings = json.loads(config.read_text())
    settings["segmentIndex"]["sha256"] = file_digest(config.parent / "captions" / "segments.json")
    settings["reflow"]["sha256"] = file_digest(reflow)
    config.write_text(json.dumps(settings), encoding="utf-8")


def _add_empty_jobs(config: Path, count: int) -> list[Path]:
    captions = config.parent / "captions"
    paths = [captions / f"tokens-{job:04d}.json" for job in range(2, 2 + count)]
    for path in paths:
        path.write_text('{"version":1,"clock":"output","tokens":[]}', encoding="utf-8")
    index = captions / "segments.json"
    raw = json.loads(index.read_text())
    raw["segments"][0]["jobs"] = [0, *range(2, 2 + count)]
    index.write_text(json.dumps(raw), encoding="utf-8")
    _refresh_document_pins(config, [index, *paths])
    return paths


@pytest.mark.parametrize("count", [1, 2])
def test_empty_jobs_keep_membership_pins_and_nonempty_recognition(config: Path, count: int) -> None:
    ordinary = config.parent / "captions" / "tokens-0000.json"
    before = ordinary.read_bytes()
    empty = _add_empty_jobs(config, count)
    loaded = summary.load(config)
    assert loaded.token_count == 3
    assert loaded.manifest["jobs"] == [
        {"job": 0, "tokens": {"path": str(ordinary), "sha256": file_digest(ordinary)}},
        *[
            {"job": job, "tokens": {"path": str(path), "sha256": file_digest(path)}}
            for job, path in enumerate(empty, 2)
        ],
        {
            "job": 1,
            "tokens": {
                "path": str(config.parent / "captions" / "tokens-0001.json"),
                "sha256": file_digest(config.parent / "captions" / "tokens-0001.json"),
            },
        },
    ]
    assert {str(path): file_digest(path) for path in empty}.items() <= loaded.tokens.items()
    assert ordinary.read_bytes() == before
    directory = config.parent / "edits" / "work" / "empty-preserved"
    summary.plan(config, directory)
    summary.verify_batch(
        directory / "batches" / "batch-0000.json", directory / "batches" / "batch-0000.proof.json"
    )
    summary.produce(config, directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["recognizedTokens"] == 3
    assert {str(path): file_digest(path) for path in empty}.items() <= receipt["inputs"].items()
    assert receipt["recognitionClocksChanged"] is False
    assert receipt["displaySRTMatchesXML"] is True


def test_all_empty_segment_keeps_native_nonempty_guard(config: Path) -> None:
    empty = _add_empty_jobs(config, 2)
    index = config.parent / "captions" / "segments.json"
    raw = json.loads(index.read_text())
    raw["segments"][0]["jobs"] = [2, 3]
    index.write_text(json.dumps(raw), encoding="utf-8")
    _refresh_document_pins(config, [index, *empty])
    with pytest.raises(ReplayError, match="Segment token count"):
        summary.load(config)


@pytest.mark.parametrize(
    "field,value", [("version", True), ("version", 2), ("clock", "source"), ("tokens", None)]
)
def test_empty_job_still_requires_valid_version_clock_array(
    config: Path, field: str, value: object
) -> None:
    empty = _add_empty_jobs(config, 1)[0]
    raw = json.loads(empty.read_text())
    raw[field] = value
    empty.write_text(json.dumps(raw), encoding="utf-8")
    _refresh_document_pins(config, [empty])
    with pytest.raises(ReplayError):
        summary.load(config)


def test_empty_job_hash_is_required_and_transitively_rechecked(config: Path) -> None:
    empty = _add_empty_jobs(config, 1)[0]
    directory = config.parent / "edits" / "work" / "changed-empty"
    summary.plan(config, directory)
    summary.verify_batch(
        directory / "batches" / "batch-0000.json", directory / "batches" / "batch-0000.proof.json"
    )
    empty.write_text('{"version":1,"clock":"output","tokens":[],"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        summary.produce(config, directory)


def test_adopted_empty_job_retains_manifest_binding(config: Path) -> None:
    empty = _add_empty_jobs(config, 1)[0]
    manifest = config.parent / "empty-adopted.json"
    manifest.write_text(
        json.dumps({"version": 1, "tokens": {"path": str(empty), "sha256": file_digest(empty)}})
    )
    source = TokenSource(0, 2, empty, file_digest(empty), manifest, file_digest(manifest))
    bindings: dict[str, object] = {
        str(empty): file_digest(empty),
        str(manifest): file_digest(manifest),
    }
    selected, pins, count = summary._selected_job(
        config.parent / "captions",
        {"segment": 0},
        job=2,
        bindings=bindings,
        sources=(source,),
        duration=1,
    )
    assert selected == {"job": 2, "tokens": {"path": str(empty), "sha256": file_digest(empty)}}
    assert count == 0
    assert pins == bindings
    with pytest.raises(ReplayError, match="both be candidate-bound"):
        summary._selected_job(
            config.parent / "captions",
            {"segment": 0},
            job=2,
            bindings={str(empty): file_digest(empty)},
            sources=(source,),
            duration=1,
        )


def test_full_segment_preserves_combined_native_token_budget(config: Path) -> None:
    captions = config.parent / "captions"
    index = captions / "segments.json"
    raw_index = json.loads(index.read_text())
    raw_index["segments"] = [{"segment": 0, "offsetFrame": 0, "frames": 30, "jobs": [0, 1]}]
    index.write_text(json.dumps(raw_index), encoding="utf-8")
    raw_tokens = json.loads((captions / "tokens-0000.json").read_text())
    raw_tokens["tokens"] = [raw_tokens["tokens"][0]] * 2049
    (captions / "tokens-0000.json").write_text(json.dumps(raw_tokens), encoding="utf-8")
    (captions / "tokens-0001.json").write_text(json.dumps(raw_tokens), encoding="utf-8")
    _refresh_document_pins(
        config, [index, captions / "tokens-0000.json", captions / "tokens-0001.json"]
    )
    with pytest.raises(ReplayError, match="Segment token count"):
        summary.load(config)


@pytest.mark.parametrize(
    "selection", ["default-link", "adopted-link", "adopted-private", "adopted-downloads"]
)
def test_all_selected_paths_are_rejected_before_native_reads(
    config: Path, selection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = _add_empty_jobs(config, 1)[0]
    sources: tuple[TokenSource, ...] = ()
    if selection.endswith("link"):
        target = config.parent / "actual-empty.json"
        target.write_bytes(default.read_bytes())
        selected = default if selection == "default-link" else config.parent / "adopted-link.json"
        if selected == default:
            selected.unlink()
        selected.symlink_to(target)
    else:
        directory = config.parent / selection.removeprefix("adopted-")
        directory.mkdir()
        selected = directory / "empty.json"
        selected.write_bytes(default.read_bytes())
    if selection.startswith("adopted-"):
        sources = (TokenSource(0, 2, selected, "0" * 64, config, file_digest(config)),)

    def forbidden_reader(
        directory: Path,
        row: dict[str, object],
        bindings: dict[str, object],
        sources: tuple[TokenSource, ...],
    ) -> Never:
        raise AssertionError("Native token reader ran before every selected path was validated")

    monkeypatch.setattr(summary, "read_tokens", forbidden_reader)
    with pytest.raises(ReplayError, match="regular canonical"):
        summary._segment_jobs(
            config.parent / "captions",
            {"segment": 0},
            members=[0, 2],
            bindings={},
            sources=sources,
            duration=1,
        )


def test_graph_batch_produce_verify_and_nonoverwrite(config: Path) -> None:
    directory = config.parent / "edits" / "work" / "prepared"
    graph = summary.plan(config, directory)
    stages = load_stages(graph)
    assert stage_graph(stages) == {"batch-0000": (), "summary-input-manifest": ("batch-0000",)}
    assert len(stages[0].inputs) == 5
    summary.verify_batch(
        directory / "batches" / "batch-0000.json", directory / "batches" / "batch-0000.proof.json"
    )
    summary.produce(config, directory)
    summary.produce(config, directory, existing=True)
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["recognizedTokens"] == 3
    assert receipt["segmentOffsetApplications"] == 1
    assert receipt["recognitionClocksChanged"] is False
    assert receipt["displaySRTMatchesXML"] is True
    assert receipt["mediaBinding"] == "identity-only-not-render-verification"
    assert receipt["summaryQualityVerified"] is False
    with pytest.raises(ReplayError, match="new immutable"):
        summary.produce(config, directory)
    with pytest.raises(ReplayError, match="work directory"):
        summary.plan(config, directory)


def test_multiple_batches_retain_all_transitive_dependencies(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(summary, "MAX_BATCH_PINS", 1)
    directory = config.parent / "edits" / "work" / "split"
    stages = load_stages(summary.plan(config, directory))
    assert stage_graph(stages) == {
        "batch-0000": (),
        "batch-0001": (),
        "summary-input-manifest": ("batch-0000", "batch-0001"),
    }
    assert len(stages[0].inputs) == 4
    assert len(stages[1].inputs) == 4
    summary.verify_batch(
        directory / "batches" / "batch-0000.json", directory / "batches" / "batch-0000.proof.json"
    )
    summary.verify_batch(
        directory / "batches" / "batch-0001.json", directory / "batches" / "batch-0001.proof.json"
    )
    summary.produce(config, directory)
    assert json.loads((directory / "manifest.json").read_text())["clockDomain"] == "edited-output"


def test_adopted_sources_override_exact_job_and_keep_originals(config: Path) -> None:
    replacement = config.parent / "replacement.json"
    replacement.write_text(
        '{"version":1,"clock":"output","tokens":[{"text":"新語",'
        '"start":0.2,"end":0.6,"utterance":"recognized"}]}',
        encoding="utf-8",
    )
    manifest = config.parent / "adopted.json"
    manifest.write_text(
        json.dumps(
            {"version": 1, "tokens": {"path": str(replacement), "sha256": file_digest(replacement)}}
        ),
        encoding="utf-8",
    )
    mapping = config.parent / "sources.json"
    mapping.write_text(
        json.dumps(
            {
                "version": 1,
                "replacements": [
                    {
                        "segment": 0,
                        "job": 0,
                        "tokens": {"path": str(replacement), "sha256": file_digest(replacement)},
                        "adoptedManifest": {"path": str(manifest), "sha256": file_digest(manifest)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reflow = config.parent / "reflow.json"
    raw = json.loads(reflow.read_text())
    raw["inputs"][str(replacement)] = file_digest(replacement)
    raw["inputs"][str(manifest)] = file_digest(manifest)
    reflow.write_text(json.dumps(raw), encoding="utf-8")
    settings = json.loads(config.read_text())
    settings["tokenSources"] = {"path": str(mapping), "sha256": file_digest(mapping)}
    settings["reflow"]["sha256"] = file_digest(reflow)
    config.write_text(json.dumps(settings), encoding="utf-8")
    loaded = summary.load(config)
    assert loaded.token_count == 2
    jobs = loaded.manifest["jobs"]
    assert isinstance(jobs, list)
    assert jobs[0]["tokens"]["path"].endswith("/replacement.json")
    assert (config.parent / "captions" / "tokens-0000.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("extra", "unexpected"),
        ("captions", "/wrong"),
        ("media", "/missing"),
    ],
)
def test_rejects_invalid_configuration(config: Path, field: str, value: object) -> None:
    raw = json.loads(config.read_text())
    raw[field] = value
    config.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        summary.load(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("rate", 0),
        ("segments", []),
    ],
)
def test_rejects_invalid_index(config: Path, field: str, value: object) -> None:
    index = config.parent / "captions" / "segments.json"
    raw = json.loads(index.read_text())
    raw[field] = value
    index.write_text(json.dumps(raw), encoding="utf-8")
    settings = json.loads(config.read_text())
    settings["segmentIndex"]["sha256"] = file_digest(index)
    config.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ReplayError, match="Reflow"):
        summary.load(config)


@pytest.mark.parametrize(
    "index",
    [
        {"version": 1, "rate": 0, "segments": []},
        {"version": True, "rate": 30, "segments": [{}]},
        {"version": 1, "rate": 30, "segments": []},
    ],
)
def test_index_structure_guard(index: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        summary._entries(index)


@pytest.mark.parametrize(
    "field,value",
    [("segment", -1), ("offsetFrame", -1), ("frames", 0), ("jobs", []), ("jobs", [0, 0])],
)
def test_rejects_invalid_segment_membership(config: Path, field: str, value: object) -> None:
    index = config.parent / "captions" / "segments.json"
    raw = json.loads(index.read_text())
    raw["segments"][0][field] = value
    index.write_text(json.dumps(raw), encoding="utf-8")
    reflow = config.parent / "reflow.json"
    bindings = json.loads(reflow.read_text())
    bindings["inputs"][str(index)] = file_digest(index)
    reflow.write_text(json.dumps(bindings), encoding="utf-8")
    settings = json.loads(config.read_text())
    settings["segmentIndex"]["sha256"] = file_digest(index)
    settings["reflow"]["sha256"] = file_digest(reflow)
    config.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ReplayError):
        summary.load(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("start", -1),
        ("end", 4),
        ("end", 0.1),
        ("text", " "),
        ("utterance", ""),
        ("utterance", "u" * 513),
    ],
)
def test_rejects_invalid_recognized_clocks_and_text(
    config: Path, field: str, value: object
) -> None:
    tokens = config.parent / "captions" / "tokens-0000.json"
    raw = json.loads(tokens.read_text())
    raw["tokens"][0][field] = value
    tokens.write_text(json.dumps(raw), encoding="utf-8")
    reflow = config.parent / "reflow.json"
    bindings = json.loads(reflow.read_text())
    bindings["inputs"][str(tokens)] = file_digest(tokens)
    reflow.write_text(json.dumps(bindings), encoding="utf-8")
    settings = json.loads(config.read_text())
    settings["reflow"]["sha256"] = file_digest(reflow)
    config.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        summary.load(config)


def test_different_srt_is_not_silently_used_for_recognition(config: Path) -> None:
    srt = config.parent / "display.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n別\n", encoding="utf-8")
    raw = json.loads(config.read_text())
    raw["displaySRT"]["sha256"] = file_digest(srt)
    config.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="SRT differs"):
        summary.load(config)


def test_changed_transitive_token_is_rejected_after_batch(config: Path) -> None:
    directory = config.parent / "edits" / "work" / "changed"
    summary.plan(config, directory)
    summary.verify_batch(
        directory / "batches" / "batch-0000.json", directory / "batches" / "batch-0000.proof.json"
    )
    (config.parent / "captions" / "tokens-0001.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        summary.produce(config, directory)


def test_missing_batch_proof_cannot_reach_consumer(config: Path) -> None:
    directory = config.parent / "edits" / "work" / "missing-proof"
    summary.plan(config, directory)
    with pytest.raises(ReplayError, match="regular canonical"):
        summary.produce(config, directory)


def test_batch_descriptor_and_proof_cannot_change(config: Path) -> None:
    directory = config.parent / "edits" / "work" / "invalid-proof"
    summary.plan(config, directory)
    descriptor = directory / "batches" / "batch-0000.json"
    proof = directory / "batches" / "batch-0000.proof.json"
    summary.verify_batch(descriptor, proof)
    raw = json.loads(proof.read_text())
    raw["candidateOnly"] = False
    proof.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="batch proof differs"):
        summary.produce(config, directory)
    batch = json.loads(descriptor.read_text())
    batch["inputs"].pop(str(config.parent / "captions" / "tokens-0001.json"))
    descriptor.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(ReplayError, match="cover"):
        summary.produce(config, directory)


def test_path_scope_and_invalid_pins_are_rejected(config: Path) -> None:
    with pytest.raises(ReplayError):
        summary._path(None)
    with pytest.raises(ReplayError):
        summary._path("relative")
    link = config.parent / "link.json"
    link.symlink_to(config)
    with pytest.raises(ReplayError):
        summary.load(link)
    with pytest.raises(ReplayError, match="path and SHA256"):
        summary._pin({"path": str(config), "sha256": None}, {})
    with pytest.raises(ReplayError, match="hash changed"):
        summary._pin({"path": str(config), "sha256": "wrong"}, {})
    with pytest.raises(ReplayError, match="work directory"):
        summary.plan(config, config.parent / "outside")


def test_batch_shape_and_output_scope(config: Path) -> None:
    bad = config.parent / "bad-batch.json"
    bad.write_text('{"version":1,"kind":"wrong","inputs":{}}', encoding="utf-8")
    with pytest.raises(ReplayError, match="bounded summary"):
        summary.verify_batch(bad, config.parent / "proof.json")
    directory = config.parent / "edits" / "work" / "scope"
    summary.plan(config, directory)
    with pytest.raises(ReplayError, match="repository work file"):
        summary.verify_batch(
            directory / "batches" / "batch-0000.json", config.parent / "proof.json"
        )


def test_budget_guards(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(summary, "MAX_TOKENS", 2)
    with pytest.raises(ReplayError, match="budget"):
        summary.load(config)


def test_changed_saved_manifest_is_rejected(config: Path) -> None:
    directory = config.parent / "edits" / "work" / "changed-manifest"
    summary.plan(config, directory)
    summary.verify_batch(
        directory / "batches" / "batch-0000.json", directory / "batches" / "batch-0000.proof.json"
    )
    summary.produce(config, directory)
    output = directory / "manifest.json"
    raw = json.loads(output.read_text())
    raw["candidateOnly"] = False
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="manifest or receipt differs"):
        summary.produce(config, directory, existing=True)


def test_cli_plan_and_batch_then_produce_and_verify(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = config.parent / "edits" / "work" / "cli"
    monkeypatch.setattr("sys.argv", ["summary", "plan", str(config), str(directory)])
    summary.main()
    monkeypatch.setattr(
        "sys.argv",
        [
            "summary",
            "batch",
            str(directory / "batches" / "batch-0000.json"),
            str(directory / "batches" / "batch-0000.proof.json"),
        ],
    )
    summary.main()
    monkeypatch.setattr("sys.argv", ["summary", "produce", str(config), str(directory)])
    summary.main()
    monkeypatch.setattr("sys.argv", ["summary", "verify", str(config), str(directory)])
    summary.main()
    assert (directory / "manifest.json").is_file()


def test_configure_exports_exact_display_and_cli(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = config.parent / "edits" / "work" / "new-config.json"
    srt = config.parent / "edits" / "work" / "new-display.srt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summary",
            "configure",
            str(output),
            "--captions",
            str(config.parent / "captions"),
            "--reflow",
            str(config.parent / "reflow.json"),
            "--xml",
            str(config.parent / "display.fcpxml"),
            "--media",
            str(config.parent / "actual.mov"),
            "--srt",
            str(srt),
        ],
    )
    summary.main()
    assert srt.read_text() == "1\n00:00:00,000 --> 00:00:01,000\n表示\n"
    assert summary.load(output).token_count == 3
    with pytest.raises(ReplayError, match="new canonical"):
        summary.configure(
            output,
            summary.Preparation(
                config.parent / "captions",
                config.parent / "reflow.json",
                config.parent / "display.fcpxml",
                config.parent / "actual.mov",
                srt,
            ),
        )
