"""Synthetic receipt integration only; no real-media quality is asserted."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

from replay import delivery_completion_v2 as completion
from replay.bgm_separation import SeparationPolicy
from replay.core import ReplayError
from replay.edit_completion import assess_delivery
from replay.recordings import record


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def fingerprint(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def ref(path: Path) -> dict[str, object]:
    item = fingerprint(path)
    del item["bytes"]
    return item


@dataclass
class Fixture:
    root: Path
    movie: Path
    source: Path
    index: Path
    render: Path
    receipts: dict[str, Path]

    def check(self) -> dict[str, str]:
        return completion.assess(
            self.index, delivery=self.movie, root=self.root, source=self.source
        )

    def change(self, key: str, values: dict[str, object]) -> None:
        path = self.receipts[key]
        data = record(json.loads(path.read_text(encoding="utf-8")))
        data.update(values)
        write(path, data)
        index = record(json.loads(self.index.read_text(encoding="utf-8")))
        index[key] = ref(path)
        write(self.index, index)


@pytest.fixture
def delivery(tmp_path: Path) -> Fixture:
    root = tmp_path / "repo/edits/replay-7"
    root.mkdir(parents=True)
    source = write(root / "2024-01-02_03-04-05_JST_replay-7.mp4", "synthetic original")
    movie = write(root / "revisions/v2/delivery/candidate.mp4", "synthetic final")
    srt = write(movie.with_suffix(".srt"), "synthetic canonical")
    canonical = write(root / "canonical.srt", "synthetic canonical")
    voice = write(root / "voice.wav", "synthetic voice")
    cuts = write(root / "cuts.json", {})
    storage = root / "render"
    compressor = write(storage / "final/final-encode.m4v", "synthetic encoded")
    master = write(storage / "av-master.mov", "synthetic PCM master")
    render = write(
        root / "render-config.json",
        {
            "storage": str(storage),
            "work": str(storage),
            "source": str(source),
            "recording": root.name,
            "cuts": str(cuts),
            "audio": {"source": str(voice)},
            "rate": 30,
            "width": 1080,
            "height": 1920,
            "finalPreset": "synthetic.setting",
        },
    )
    write(
        root / "edit-config.json",
        {
            "version": 1,
            "replay_id": "7",
            "source": str(source),
            "source_size": source.stat().st_size,
            "engine": str(root),
            "executor_scope": str(root),
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
        },
    )
    jobs = write(
        storage / "final-job.json",
        {
            "jobs": [
                {
                    "name": "final",
                    "source": str(storage / "av-master.mov"),
                    "preset": "synthetic.setting",
                    "output": str(compressor),
                }
            ]
        },
    )
    state = write(
        storage / "final-job-state/final.json",
        {
            "status": "Successful",
            "jobID": "11111111-2222-3333-4444-555555555555",
            "output": str(compressor),
            "bytes": compressor.stat().st_size,
        },
    )
    xml = write(root / "timeline.fcpxml", "synthetic XML")
    speech = {"level_delta_db_p50": 0, "level_delta_db_p05": 0, "correlation": 1}
    audio = {
        "version": 1,
        "kind": "delivery-audio-quality",
        "inputs": [fingerprint(render), fingerprint(movie), fingerprint(voice)],
        "checks": {
            "selectedPCMReceiptBound": True,
            "nativeTimesPreserved": True,
            "aacSampleTimeMapping": True,
            "speechGatePassed": True,
            "musicGatePassed": True,
            "inputsStable": True,
        },
        "measurements": {
            "mapping": {
                "selectedSamples": 132300,
                "nativeTimesPreserved": True,
                "maximumAbsoluteDrift": "0",
            },
            "aac": {"decodedSamples": 132300, "audioPCMIdentical": False},
            "quality": {
                "policy": asdict(SeparationPolicy()),
                "minimumSpeechCorrelation": 0.995,
                "musicGatePassed": True,
                "speechGatePassed": True,
                "protectedSamples": 1000,
                "windows": [
                    {
                        "startSample": 0,
                        "endSample": 132300,
                        "protectedSamples": 1000,
                        "speech": speech,
                        "speechGatePassed": True,
                        "beforeInstrumentalDBFS": -20,
                        "afterInstrumentalDBFS": -30,
                        "music": {"passed": True},
                    }
                ],
            },
        },
        "accepted": True,
        "limitations": [],
    }
    mask = {
        "version": 1,
        "kind": "delivery-mask-quality",
        "inputs": [fingerprint(render), fingerprint(source)],
        "checks": {
            "recordedMaskExecutionMatchesPlan": True,
            "executionRowsMatchProtectedCuts": True,
            "actualCompositorProvenanceVerified": True,
            "selectedEmptyZero": True,
            "qualityComplete": True,
            "losslessEvidenceVerified": True,
        },
        "measurements": {
            "selectedFrames": 90,
            "selectedMaskFrames": 30,
            "measuredFrames": 30,
            "missingMaskFrames": 0,
            "extraMaskFrames": 0,
            "selectedEmpty": 0,
            "technicallyInvalidFrames": 0,
            "readableFrames": 0,
            "maximumResidualFraction": 0.005,
            "maximumUnstableFraction": 0.02,
        },
        "accepted": True,
        "limitations": [],
    }
    caption = {
        "version": 1,
        "kind": "delivery-caption-audit",
        "inputs": [
            fingerprint(render),
            fingerprint(movie),
            fingerprint(canonical),
            fingerprint(srt),
        ],
        "checks": {
            "measuredPlanCorrespondence": True,
            "orderedFrameTiming": True,
            "canonicalSRTFrameEquivalent": True,
            "canonicalSRTActualDisplayVerified": True,
            "historicalToolCodeBound": True,
            "finalCompositionVerified": True,
            "candidateSRTExactMatch": True,
            "fullAlphaCoverage": True,
            "boundsAndGapsVerified": True,
            "historicalSubmissionBound": True,
            "recognitionCutProjectionVerified": True,
        },
        "measurements": {
            "wholeVideoFrames": 90,
            "expectedAlphaParts": 1,
            "auditedAlphaParts": 1,
            "canonicalSRT": ref(canonical),
            "parts": [
                {
                    "frames": 90,
                    "backend": "direct-compressor-cli",
                    "historicalSubmissionBound": True,
                    "checks": {
                        "fullCaptionStates": 2,
                        "sampleCount": 60,
                        "constraintFrames": 60,
                        "measuredFrames": 90,
                        "boundsAndGapsVerified": True,
                        "gapsVerified": True,
                        "visibleStatesVerified": True,
                        "fullStateConstraintCoverage": True,
                    },
                }
            ],
        },
        "accepted": True,
        "limitations": [],
    }
    receipts = {
        "renderConfig": render,
        "technical": write(
            root / "technical.json",
            {
                "inputs": [fingerprint(render), fingerprint(movie), fingerprint(cuts)],
                "technicalAccepted": True,
                "fullDecode": True,
                "expected": {"frames": 90, "rate": 30, "width": 1080, "height": 1920},
                "videoClock": {"frames": 90, "exactCFR": True},
            },
        ),
        "remux": write(
            root / "remux.json",
            {
                "inputs": [
                    fingerprint(render),
                    fingerprint(movie),
                    fingerprint(compressor),
                    fingerprint(jobs),
                    fingerprint(state),
                    fingerprint(master),
                ],
                "remuxIdentityAccepted": True,
                "nativeStatus": "Successful",
                "nativeJobID": "11111111-2222-3333-4444-555555555555",
                "source": str(compressor),
                "candidate": str(movie),
                "streams": [
                    {
                        "type": "video",
                        "packets": 90,
                        "orderedPayloadsEqual": True,
                        "rationalClocksEqual": True,
                    },
                    {
                        "type": "audio",
                        "packets": 130,
                        "orderedPayloadsEqual": True,
                        "rationalClocksEqual": True,
                    },
                ],
            },
        ),
        "audioQuality": write(root / "audio.json", audio),
        "maskQuality": write(root / "mask.json", mask),
        "captionAudit": write(root / "caption.json", caption),
        "cutExecution": write(
            root / "cut-execution.json",
            {
                "inputs": [
                    fingerprint(render),
                    fingerprint(source),
                    fingerprint(movie),
                    fingerprint(cuts),
                ],
                "selectionPlanMatchesProtectedCuts": True,
                "recordedExecutionMatchesPlan": True,
                "selectedFrames": 90,
                "sourceFrames": 100,
                "removedFrames": 10,
            },
        ),
        "fcp": write(
            root / "fcp.json",
            {
                "inputs": [fingerprint(render), fingerprint(movie), fingerprint(xml)],
                "xml": str(xml),
                "xmlSHA256": fingerprint(xml)["sha256"],
                "importObserved": True,
                "noModalErrorObserved": True,
            },
        ),
    }
    index = movie.parent.parent / "metadata/native-evidence.json"
    completion.assemble(root, movie, index, receipts)
    return Fixture(root, movie, source, index, render, receipts)


def test_synthetic_end_to_end_and_dispatch(delivery: Fixture) -> None:
    assert delivery.check() == {}
    assert assess_delivery(delivery.movie, root=delivery.root) == {}


@pytest.mark.parametrize(
    "key,stage",
    [
        ("technical", "bgm_removal"),
        ("remux", "silence_cut"),
        ("audioQuality", "bgm_removal"),
        ("maskQuality", "source_caption_mask"),
        ("captionAudit", "transcription"),
        ("cutExecution", "silence_cut"),
        ("fcp", "dynamic_captions"),
    ],
)
def test_each_missing_receipt_fails(delivery: Fixture, key: str, stage: str) -> None:
    delivery.receipts[key].unlink()
    assert key in delivery.check()[stage]


@pytest.mark.parametrize(
    "key,stage",
    [
        ("audioQuality", "bgm_removal"),
        ("maskQuality", "source_caption_mask"),
        ("captionAudit", "transcription"),
    ],
)
def test_false_quality_never_passes(delivery: Fixture, key: str, stage: str) -> None:
    delivery.change(key, {"accepted": False})
    assert "unverified" in delivery.check()[stage]


@pytest.mark.parametrize(
    "key,values,stage",
    [
        ("remux", {"source": "relative.mov"}, "bgm_removal"),
        ("technical", {"fullDecode": 1}, "bgm_removal"),
        ("remux", {"streams": [{"type": "audio"}]}, "bgm_removal"),
        (
            "remux",
            {
                "streams": [
                    {
                        "type": "video",
                        "packets": 90,
                        "orderedPayloadsEqual": False,
                        "rationalClocksEqual": True,
                    },
                    {
                        "type": "audio",
                        "packets": 130,
                        "orderedPayloadsEqual": True,
                        "rationalClocksEqual": True,
                    },
                ]
            },
            "bgm_removal",
        ),
        ("audioQuality", {"checks": {}}, "bgm_removal"),
        ("audioQuality", {"inputsAfter": []}, "bgm_removal"),
        ("maskQuality", {"kind": "unknown"}, "source_caption_mask"),
        ("captionAudit", {"measurements": {}}, "transcription"),
        ("fcp", {"importObserved": False}, "dynamic_captions"),
        ("cutExecution", {"selectedFrames": 89}, "silence_cut"),
    ],
)
def test_invalid_contracts(
    delivery: Fixture, key: str, values: dict[str, object], stage: str
) -> None:
    delivery.change(key, values)
    assert stage in delivery.check()


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximumResidualFraction", 0.006),
        ("maximumUnstableFraction", 0.021),
        ("readableFrames", 1),
        ("selectedEmpty", 1),
        ("measuredFrames", 29),
    ],
)
def test_mask_thresholds_and_coverage(delivery: Fixture, field: str, value: object) -> None:
    data = record(json.loads(delivery.receipts["maskQuality"].read_text(encoding="utf-8")))
    measured = record(data["measurements"])
    measured[field] = value
    delivery.change("maskQuality", {"measurements": measured})
    assert "source_caption_mask" in delivery.check()


@pytest.mark.parametrize(
    "field,value",
    [
        ("afterInstrumentalDBFS", -29),
        ("endSample", 132299),
        ("protectedSamples", 0),
        ("speech", {"level_delta_db_p50": -1.51, "level_delta_db_p05": 0, "correlation": 1}),
        ("speech", {"level_delta_db_p50": 0, "level_delta_db_p05": 0, "correlation": 0.994}),
    ],
)
def test_audio_policy_and_full_coverage(delivery: Fixture, field: str, value: object) -> None:
    data = record(json.loads(delivery.receipts["audioQuality"].read_text(encoding="utf-8")))
    measured = record(data["measurements"])
    quality = record(measured["quality"])
    windows = quality["windows"]
    assert isinstance(windows, list)
    window = record(windows[0])
    window[field] = value
    quality["windows"] = [window]
    measured["quality"] = quality
    delivery.change("audioQuality", {"measurements": measured})
    assert "bgm_removal" in delivery.check()


@pytest.mark.parametrize(
    "field",
    ["canonicalSRTActualDisplayVerified", "historicalToolCodeBound", "finalCompositionVerified"],
)
def test_alpha_only_cannot_complete(delivery: Fixture, field: str) -> None:
    path = delivery.receipts["captionAudit"]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["checks"][field] = False
    delivery.change("captionAudit", {"checks": data["checks"]})
    assert "dynamic_captions" in delivery.check()


def test_changed_pinned_input(delivery: Fixture) -> None:
    (delivery.root / "voice.wav").write_bytes(b"changed voice")
    assert "Processing input changed" in delivery.check()["bgm_removal"]


def test_wrong_remux_chain(delivery: Fixture) -> None:
    delivery.change("remux", {"source": str(delivery.movie)})
    assert "actual Compressor" in delivery.check()["bgm_removal"]


@pytest.mark.parametrize("correct_recording", [True, False])
def test_pinned_dated_compressor_output_matches_recording(
    delivery: Fixture, correct_recording: bool
) -> None:
    remux = record(json.loads(delivery.receipts["remux"].read_text()))
    old = Path(str(remux["source"]))
    basename = (
        "2024-01-02_03-04-05_JST_replay-7-review.m4v" if correct_recording else "unrelated.m4v"
    )
    output = old.with_name(basename)
    old.rename(output)
    render = record(json.loads(delivery.render.read_text()))
    work = Path(str(render["work"]))
    jobs_path = work / "final-job.json"
    jobs = record(json.loads(jobs_path.read_text()))
    rows = jobs.get("jobs")
    assert isinstance(rows, list)
    row = record(rows[0])
    row["output"] = str(output)
    jobs["jobs"] = [row]
    write(jobs_path, jobs)
    state_path = work / "final-job-state/final.json"
    state = record(json.loads(state_path.read_text()))
    state["output"] = str(output)
    write(state_path, state)
    rows = remux.get("inputs")
    assert isinstance(rows, list)
    paths = [Path(str(record(item)["path"])) for item in rows]
    inputs = [fingerprint(output if path == old else path) for path in paths]
    delivery.change("remux", {"source": str(output), "inputs": inputs})
    result = delivery.check()
    if correct_recording:
        assert result == {}
    else:
        assert "pinned recording identity" in result["bgm_removal"]


def test_changed_movie_and_config_fail(delivery: Fixture) -> None:
    delivery.movie.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="current recording"):
        delivery.check()


def test_changed_receipt_and_changed_bound_input(delivery: Fixture) -> None:
    delivery.receipts["audioQuality"].write_bytes(b"{}")
    (delivery.root / "voice.wav").write_bytes(b"changed")
    assert "Receipt SHA256 changed" in delivery.check()["bgm_removal"]


def test_empty_shell_reports_specific_stages(delivery: Fixture) -> None:
    output = delivery.root / "shell.json"
    completion.assemble(delivery.root, delivery.movie, output, {"renderConfig": delivery.render})
    failures = completion.assess(
        output, delivery=delivery.movie, root=delivery.root, source=delivery.source
    )
    assert len(failures) == 5
    assert "audioQuality" in failures["bgm_removal"]
    assert "fcp" in failures["dynamic_captions"]


def test_assemble_does_not_overwrite(delivery: Fixture) -> None:
    with pytest.raises(FileExistsError):
        completion.assemble(delivery.root, delivery.movie, delivery.index, delivery.receipts)


def test_cli_success_and_failure(delivery: Fixture, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        completion.main(["check", str(delivery.root), str(delivery.movie), str(delivery.index)])
        == 0
    )
    assert '"summaries": "not assessed"' in capsys.readouterr().out
    assert (
        completion.main(
            [
                "assemble",
                str(delivery.root),
                str(delivery.movie),
                str(delivery.root / "shell.json"),
                "--renderConfig",
                str(delivery.render),
            ]
        )
        == 1
    )
    assert '"accepted": false' in capsys.readouterr().out
    assert (
        completion.main(
            ["check", str(delivery.root), str(delivery.movie), str(delivery.root / "absent.json")]
        )
        == 1
    )
    assert '"accepted": false' in capsys.readouterr().err


def test_external_sibling_scope(tmp_path: Path) -> None:
    root = tmp_path / "repo/edits/replay-7"
    root.mkdir(parents=True)
    source = write(root / "source.mp4", "source")
    storage = tmp_path / "external/edited-deliveries/replay-7"
    storage.mkdir(parents=True)
    artifact = storage.parent / "work/replay-7/render"
    file = write(artifact / "alpha.mov", "alpha")
    render = write(
        root / "render.json",
        {"source": str(source), "recording": root.name, "storage": str(artifact)},
    )
    store = completion.Evidence(root, source, storage)
    completion._configure(store, {"renderConfig": ref(render)})
    assert store.digest(file) == fingerprint(file)["sha256"]
    other = write(storage.parent / "work/replay-8/render/alpha.mov", "other")
    with pytest.raises(ReplayError, match="authorized storage"):
        store.digest(other)
    linked = artifact / "escape.mov"
    linked.symlink_to(other)
    with pytest.raises(ReplayError, match="Linked evidence"):
        store.digest(linked)


@pytest.mark.parametrize("suffix", ["", "work/replay-8/render", "work/replay-7"])
def test_unsafe_external_roots(tmp_path: Path, suffix: str) -> None:
    root = tmp_path / "repo/edits/replay-7"
    root.mkdir(parents=True)
    source = write(root / "source.mp4", "source")
    storage = tmp_path / "external/edited-deliveries/replay-7"
    storage.mkdir(parents=True)
    render = write(
        root / "render.json",
        {"source": str(source), "recording": root.name, "storage": str(storage.parent / suffix)},
    )
    with pytest.raises(ReplayError, match="exact sibling"):
        completion._configure(
            completion.Evidence(root, source, storage), {"renderConfig": ref(render)}
        )


def test_detect_mutation_during_check(delivery: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    original = completion._fcp

    def mutate(
        store: completion.Evidence, data: dict[str, object], *, render_path: Path, movie: Path
    ) -> None:
        original(store, data, render_path=render_path, movie=movie)
        movie.write_bytes(b"changed during verification")

    monkeypatch.setattr(completion, "_fcp", mutate)
    with pytest.raises(ReplayError, match="changed during verification"):
        delivery.check()


@pytest.mark.parametrize("kind", ["cutExecution", "audioQuality", "maskQuality"])
def test_native_numeric_measurement_reuses_original_rules(delivery: Fixture, kind: str) -> None:
    render_data = record(json.loads(delivery.render.read_text()))
    receipt = record(json.loads(delivery.receipts[kind].read_text()))
    index = record(json.loads(delivery.index.read_text()))
    technical = record(json.loads(delivery.receipts["technical"].read_text()))
    frames = record(technical["expected"])["frames"]
    context: dict[str, object] = {
        "frames": frames,
        "fps": render_data["rate"],
        "cuts": {"path": render_data["cuts"]},
        "voice": {"path": record(render_data["audio"])["source"]},
    }
    store = completion.Evidence(delivery.root, delivery.source)
    completion.validate_native_measurement(
        kind,
        store=store,
        receipt=receipt,
        descriptor=delivery.render,
        context=context,
        movie=delivery.movie,
    )
    assert index["version"] == 2


def test_native_measurement_rejects_unknown_contract(delivery: Fixture) -> None:
    with pytest.raises(ReplayError, match="Unsupported"):
        completion.validate_native_measurement(
            "alpha",
            store=completion.Evidence(delivery.root, delivery.source),
            receipt={},
            descriptor=delivery.render,
            context={"frames": 60},
            movie=delivery.movie,
        )
    with pytest.raises(ReplayError, match="positive"):
        completion.validate_native_measurement(
            "cutExecution",
            store=completion.Evidence(delivery.root, delivery.source),
            receipt={},
            descriptor=delivery.render,
            context={"frames": 0},
            movie=delivery.movie,
        )
