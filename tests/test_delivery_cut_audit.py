"""Synthetic recorded-execution chain; never launches media tools."""

import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import pytest

from replay import (
    compressor_batch,
    mechanical_join,
    mechanical_plan_provenance,
    native_invocation,
)
from replay import delivery_completion_v2 as adapter
from replay import delivery_cut_audit as audit
from replay import delivery_render as render
from replay import workflow as workflow_runner
from replay.core import ReplayError
from replay.delivery_render_resume import fingerprint
from replay.native_edit_parts import partition
from replay.recordings import record
from replay.stages import StageError
from replay.workflow import parse_stages


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def ref(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": binding(path)["sha256"]}


def update(path: Path, values: dict[str, object]) -> None:
    write(path, {**record(json.loads(path.read_text(encoding="utf-8"))), **values})


def cleanup_evidence(config_path: Path, *, pts: tuple[int, ...], last_duration: int) -> None:
    """Pin synthetic producer, candidate and exact-clock assembly without replacing guards."""
    config = render.load_config(config_path)
    root = config.path.parent / "cleanup"
    candidate = write(root / "chunk/candidate.mp4", "synthetic locally repaired frames")
    producer = write(
        root / "producer.json",
        {
            "source": str(config.source),
            "sourceSHA256": binding(config.source)["sha256"],
            "outputDirectory": str(candidate.parent),
            "clock": {"pts": list(pts), "timeBase": "1/30", "lastDuration": last_duration},
        },
    )
    receipt = write(
        root / "candidate.json",
        {
            "videoSHA256": binding(candidate)["sha256"],
            "inputs": {
                str(producer): binding(producer)["sha256"],
                str(config.source): binding(config.source)["sha256"],
            },
        },
    )
    clean = write(config.storage / "continuous.mov", "synthetic assembled source")
    row = {
        "media": str(clean),
        "sourceFirstPTS": pts[0],
        "sourceEndPTS": pts[-1] + last_duration,
        "timeBase": "1/30",
        "frames": len(pts),
        "sha256": binding(clean)["sha256"],
    }
    write(config.chunks, {"ranges": [row]})
    chunk_index = write(
        root / "chunks.json",
        {"ranges": [{**row, "media": str(candidate), "sha256": binding(candidate)["sha256"]}]},
    )
    manifest = write(
        root / "manifest.json",
        {
            "version": 1,
            "index": ref(chunk_index),
            "chunks": [{"candidate": ref(receipt), "producer": ref(producer)}],
        },
    )
    clock = write(root / "clock.json", {"pts": list(pts), "lastDuration": last_duration})
    assembly = write(
        root / "assembly.json",
        {
            "exactPresentationClockVerified": True,
            "compressedPacketsIdentical": True,
            "indexSHA256": binding(config.chunks)["sha256"],
            "output": ref(clean),
            "frames": len(pts),
            "probeEvidence": {str(clock): binding(clock)["sha256"]},
            "inputs": {
                str(path): binding(path)["sha256"]
                for path in (
                    manifest,
                    chunk_index,
                    receipt,
                    producer,
                    candidate,
                )
            },
        },
    )
    cleanup = write(
        root / "cleanup.json",
        {
            "version": 1,
            "assemblyManifest": ref(manifest),
            "assemblyReceipt": ref(assembly),
            "cleanIndex": ref(config.chunks),
        },
    )
    update(config_path, {"sourceCleanup": ref(cleanup)})


def seal(plan: Path, stage_raw: dict[str, object]) -> None:
    stage = parse_stages({"stages": [stage_raw], "version": 1})[0]
    inputs = {str(path): str(binding(path)["sha256"]) for path in stage.inputs}
    write(
        plan.with_name(f"{plan.stem}-state") / f"{stage.name}.json",
        {
            "status": "complete",
            "fingerprint": fingerprint(stage, inputs),
            "outputs": {str(path): binding(path)["sha256"] for path in stage.outputs},
        },
    )


@dataclass
class Fixture:
    config: Path
    movie: Path
    technical: Path
    paths: audit.EvidencePaths
    root: Path

    def run(self) -> dict[str, object]:
        return audit.audit(
            self.config, self.movie, self.technical, self.root / "report.json", paths=self.paths
        )

    def store(self) -> audit.Bindings:
        config = render.load_config(self.config)
        return audit.Bindings(
            config.viewer_root, config.source, config.delivery_root, artifact_root=config.storage
        )


def native_part(
    config: render.Config, source_request: Path, index: int
) -> tuple[dict[str, object], dict[str, object], Path]:
    offset, frames, request = partition(
        record(json.loads(source_request.read_text(encoding="utf-8")))
    )[index]
    part = config.composition / "parts-0000" / f"part-{index:04d}" / "request.json"
    write(part, request)
    return (
        {**ref(part), "offsetFrame": offset, "frames": frames},
        {},
        config.storage / f"native-{index}.mp4",
    )


@pytest.fixture
def chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "edits/replay-7"
    source = write(root / "2024-01-02_03-04-05_JST_replay-7.mp4", "synthetic original")
    movie = write(root / "delivery/final.mp4", "synthetic final")
    native = root / "native"
    native.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        "request = Path(sys.argv[2])\n"
        "data = json.loads(request.read_text())\n"
        "number = int(request.parent.name.split('-')[-1])\n"
        "rate = data['recipe']['video']['frameRate']\n"
        "clips = data['recipe']['clips']\n"
        "frames = round(sum(c['selection']['durationSeconds'] for c in clips) * rate)\n"
        "movie = Path(data['outputDirectory']) / f'native-{number}.mp4'\n"
        "movie.write_bytes(b'synthetic native output')\n"
        "print(json.dumps({'outputPath': str(movie), 'frameCount': frames, "
        "'expectedDurationSeconds': frames / rate, 'actualDurationSeconds': frames / rate}))\n",
        encoding="utf-8",
    )
    native.chmod(0o700)
    asr = write(root / "speech-and-quiet.json", "synthetic pre-existing evidence")
    cuts = write(
        root / "cuts.json",
        {
            "sourceFrames": 30,
            "normalizationVerified": False,
            "inputs": {str(asr): binding(asr)["sha256"]},
            "cuts": [
                {"source_start_frame": 1, "source_end_frame": 3, "output_start_frame": 0},
                {"source_start_frame": 6, "source_end_frame": 10, "output_start_frame": 2},
            ],
        },
    )
    media = write(root / "storage/continuous.mov", "synthetic assembled source")
    index = write(
        root / "index.json",
        {
            "ranges": [
                {
                    "media": str(media),
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": 30,
                    "timeBase": "1/30",
                    "frames": 3,
                    "sha256": binding(media)["sha256"],
                }
            ]
        },
    )
    config_path = write(
        root / "config.json",
        {
            "version": 1,
            "recording": root.name,
            "captions": str(root / "captions"),
            "work": str(root / "work"),
            "storage": str(root / "storage"),
            "rate": 30,
            "width": 64,
            "height": 64,
            "cuts": str(cuts),
            "sourceChunks": str(index),
            "motion": {"template": str(root / "template"), "expectedText": "Title", "finish": []},
            "alphaPreset": str(root / "alpha"),
            "encodePreset": str(root / "encode"),
            "native": str(native),
            "encoding": {"codec": "hevc"},
            "audio": {"source": str(root / "voice"), "clock": str(root / "audio-clock")},
            "source": str(source),
            "deliveryRoot": str(root / "delivery"),
            "finalPreset": str(root / "preset"),
            "calibration": str(root / "calibration"),
            "viewerRoot": str(root),
        },
    )
    cleanup_evidence(config_path, pts=(0, 7, 18), last_duration=12)
    config = render.load_config(config_path)
    helper = root / "native_invocation.py"
    helper.write_bytes(Path(native_invocation.__file__).read_bytes())
    monkeypatch.setattr(native_invocation, "__file__", str(helper))
    clock = write(
        root / "clock.json",
        {
            "streams": [{"time_base": "1/30"}],
            "frames": [
                {"pts": 0, "duration": 7},
                {"pts": 7, "duration": 11},
                {"pts": 18, "duration": 12},
            ],
        },
    )
    clock_receipt = write(
        root / "clock-receipt.json", {"inputs": [binding(source), binding(clock)]}
    )
    proof = write(
        root / "protection.json",
        {
            "version": 1,
            "coordinateSpace": "source-grid-slots",
            "rate": 30,
            "inputs": [binding(cuts), binding(asr)],
            "protectedSpeechWithASRGuard": [[1, 3], [6, 9]],
            "removedNonSpeech": [[0, 1], [3, 6], [10, 30]],
            "allSpeechAndASRGuardsIncluded": True,
            "removedIntervalsVerifiedNonSpeech": True,
        },
    )
    clock_data = record(json.loads(clock.read_text(encoding="utf-8")))
    probe = write(root / "output-frames.json", {"frames": clock_data["frames"]})
    streams = write(root / "output-streams.json", {"streams": clock_data["streams"]})
    assembly = write(
        root / "assembly.json",
        {
            "indexSHA256": binding(index)["sha256"],
            "exactPresentationClockVerified": True,
            "compressedPacketsIdentical": True,
            "output": ref(media),
            "probeEvidence": {
                str(probe): binding(probe)["sha256"],
                str(streams): binding(streams)["sha256"],
            },
        },
    )
    request = write(
        config.composition / "request-0000.json",
        {
            "outputDirectory": str(config.storage),
            "outputName": "candidate.mp4",
            "recipe": {
                "video": {
                    "width": 64,
                    "height": 64,
                    "frameRate": 30,
                    "encoding": {"codec": "hevc"},
                },
                "clips": [
                    render._clip_json(clip, 30)
                    for clip in render.clips(render.read_cuts(cuts), render.read_chunks(index), 30)
                ],
            },
        },
    )
    parts = [native_part(config, request, index) for index in range(2)]
    write(
        config.composition / "parts-0000/manifest.json",
        {"sourceRequest": ref(request), "parts": [item[0] for item in parts]},
    )
    [workflow] = render.render(config, workers=1)

    state = workflow.with_name(f"{workflow.stem}-state")
    assert workflow_runner.main([str(workflow), "--state", str(state), "--check"]) == 0
    workflow_runner.run_plan(workflow, state_directory=state)
    write(config.storage / "joined-video.mp4", "synthetic joined")
    write(config.work / "audio-selected/selected.wav", "synthetic voice")
    master = write(config.storage / "av-master.mov", "synthetic master")
    [join_plan] = render.join(config)
    seal(join_plan, record(render._list(record(json.loads(join_plan.read_text()))["stages"])[0]))
    [mux_plan] = render.final_mux(config)
    seal(mux_plan, record(render._list(record(json.loads(mux_plan.read_text()))["stages"])[0]))
    encoded = config.storage / "final/final-encode.m4v"
    jobs = write(
        config.work / "final-job.json",
        {
            "jobs": [
                {
                    "name": "final",
                    "source": str(master),
                    "preset": str(config.final_preset),
                    "output": str(encoded),
                }
            ]
        },
    )
    write(config.final_preset, "synthetic preset")

    def compressor(arguments: Sequence[str]) -> str:
        if "-jobpath" in arguments:
            return '{"batch":{"jobs":[{"jobID":"11111111-2222-3333-4444-555555555555"}]}}'
        write(encoded, "synthetic compressed")
        return (
            '[{"jobStatus":{"jobid":"11111111-2222-3333-4444-555555555555","status":"Successful"}}]'
        )

    compressor_batch.run(
        compressor_batch.read_jobs(jobs),
        compressor_batch.Batch(
            config.work / "final-job-state", config.native, "synthetic", compressor, lambda _s: None
        ),
        in_flight=1,
    )
    state = config.work / "final-job-state/final.json"
    remux = write(
        root / "remux.json",
        {
            "inputs": [
                binding(path) for path in (config_path, movie, encoded, jobs, state, master)
            ],
            "source": str(encoded),
            "candidate": str(movie),
            "remuxIdentityAccepted": True,
            "nativeStatus": "Successful",
            "nativeJobID": "11111111-2222-3333-4444-555555555555",
            "streams": [
                {
                    "type": "video",
                    "orderedPayloadsEqual": True,
                    "rationalClocksEqual": True,
                    "packets": 6,
                },
                {
                    "type": "audio",
                    "orderedPayloadsEqual": True,
                    "rationalClocksEqual": True,
                    "packets": 2,
                },
            ],
        },
    )
    technical = write(
        root / "technical.json",
        {
            "inputs": [binding(path) for path in (config_path, cuts, movie)],
            "technicalAccepted": True,
            "fullDecode": True,
            "expected": {"rate": 30, "width": 64, "height": 64, "frames": 6},
            "videoClock": {"frames": 6, "exactCFR": True},
        },
    )
    return Fixture(
        config_path,
        movie,
        technical,
        audit.EvidencePaths(clock, clock_receipt, assembly, proof, remux),
        root,
    )


def test_complete_chain_calls_real_adapter(chain: Fixture) -> None:
    result = chain.run()
    assert result["accepted"] is True
    assert result["sourceFrames"] == 30
    assert result["decodedSourceFrames"] == 3
    assert result["selectedFrames"] == 6
    assert result["removedFrames"] == 24
    assert result["failures"] == {}
    assert result["inputsAfter"] == result["inputs"]
    assert record(result["checks"])["completionV2Adapter"] is True
    assert len(render._list(result["actualNativeReceipts"])) == 2
    config = render.load_config(chain.config)
    adapter._cuts(
        chain.store(),
        result,
        render_path=chain.config,
        render=record(json.loads(chain.config.read_text())),
        movie=chain.movie,
        frames=6,
    )
    assert record(json.loads(config.cuts.read_text()))["normalizationVerified"] is False
    receipt = record(render._list(result["actualNativeReceipts"])[0])
    facts = native_invocation.read(Path(str(receipt["runtimeBinding"])))
    assert receipt["executionContract"] == "native-invocation-v1"
    assert receipt["movieSHA256Recorded"] == record(facts["output"])["sha256"]
    assert receipt["receiptSHA256"] == record(facts["response"])["sha256"]
    assert facts["inputs"] == facts["inputsAfter"]


@pytest.mark.parametrize(
    "field", ["source_clock", "clock_receipt", "assembly_receipt", "protection", "remux_receipt"]
)
def test_missing_input_is_reported_and_other_checks_run(chain: Fixture, field: str) -> None:
    chain.paths = replace(chain.paths, **{field: None})
    result = chain.run()
    assert result["accepted"] is False
    assert record(result["checks"])["nativeSelectionPlan"] is True
    assert record(result["checks"])["inputsStable"] is True


def test_partial_terminal_not_extended(chain: Fixture) -> None:
    clock = chain.root / "clock.json"
    write(clock, {"streams": [{"time_base": "1/1000"}], "frames": [{"pts": 0, "duration": 272}]})
    write(
        chain.root / "clock-receipt.json",
        {"inputs": [binding(render.load_config(chain.config).source), binding(clock)]},
    )
    result = chain.run()
    assert result["sourceFrames"] == 8
    assert result["terminalPartialSeconds"] == "2/375"
    assert result["accepted"] is False
    assert record(result["checks"])["sourceGridContract"] is False


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [{"pts": 1, "duration": 30}],
        [{"pts": 0, "duration": 7}, {"pts": 8, "duration": 22}],
        [{"pts": 0, "duration": 0}],
    ],
)
def test_invalid_original_clock(chain: Fixture, frames: list[dict[str, int]]) -> None:
    clock = chain.root / "clock.json"
    update(clock, {"frames": frames})
    write(
        chain.root / "clock-receipt.json",
        {"inputs": [binding(render.load_config(chain.config).source), binding(clock)]},
    )
    assert record(chain.run()["checks"])["originalSourceClock"] is False


@pytest.mark.parametrize(
    "values",
    [
        {"protectedSpeechWithASRGuard": [[2, 7]]},
        {"protectedSpeechWithASRGuard": []},
        {"protectedSpeechWithASRGuard": [[1]]},
        {"removedNonSpeech": []},
        {"allSpeechAndASRGuardsIncluded": False},
        {"rate": 60},
    ],
)
def test_protection_rejections(chain: Fixture, values: dict[str, object]) -> None:
    update(chain.root / "protection.json", values)
    result = chain.run()
    assert result["selectionPlanMatchesProtectedCuts"] is False
    assert record(result["checks"])["protectedSpeechAndRemovedNonSpeech"] is False


def test_recorded_request_hash_changed(chain: Fixture) -> None:
    part = chain.root / "work/compose/parts-0000/part-0000/request.json"
    update(part, {"outputName": "wrong.mp4"})
    result = chain.run()
    assert record(result["checks"])["nativeSelectionPlan"] is False
    assert result["recordedExecutionMatchesPlan"] is False


def test_legacy_response_only_cannot_gain_audit_time_movie_proof(chain: Fixture) -> None:
    config = render.load_config(chain.config)
    plan = config.composition / "render-0-0000.json"
    raw = record(json.loads(plan.read_text()))
    stages = [record(row) for row in render._list(raw["stages"])]
    part = config.composition / "parts-0000/part-0000/request.json"
    response = part.with_name("render.response.json")
    stages[0] = {
        **stages[0],
        "command": [
            "/bin/bash",
            "-c",
            'exec "$1" edit-media "$2" > "$3"',
            "native-render",
            str(config.native),
            str(part),
            str(response),
        ],
        "verify_command": [],
        "outputs": [str(response)],
    }
    write(plan, {"version": 1, "stages": stages})
    seal(plan, stages[0])
    result = chain.run()
    assert result["accepted"] is False
    assert "lacks actual native movie SHA256" in str(
        record(result["failures"])["completedNativeExecution"]
    )
    assert record(render._list(result["actualNativeReceipts"])[0])["movieSHA256Recorded"] is None


def test_actual_movie_not_bound_in_completed_record(chain: Fixture) -> None:
    plan = chain.root / "work/compose/render-0-0000.json"
    raw = record(json.loads(plan.read_text()))
    stages = [record(item) for item in render._list(raw["stages"])]
    stages[0]["outputs"] = [
        str(chain.root / "work/compose/parts-0000/part-0000/render.response.json")
    ]
    write(plan, {"version": 1, "stages": stages})
    seal(plan, stages[0])
    result = chain.run()
    assert "Runtime native wrapper contract differs" in str(
        record(result["failures"])["completedNativeExecution"]
    )
    assert result["recordedExecutionMatchesPlan"] is False
    assert len(render._list(result["actualNativeReceipts"])) == 1
    assert record(result["checks"])["compressorAndRemux"] is True


@pytest.mark.parametrize(
    "relative",
    [
        "native",
        "storage/native-0.mp4",
        "work/compose/parts-0000/part-0000/render.response.json",
        "storage/joined-video.mp4",
        "storage/av-master.mov",
        "2024-01-02_03-04-05_JST_replay-7.mp4",
        "delivery/final.mp4",
    ],
)
def test_mutation_rejected(chain: Fixture, relative: str) -> None:
    (chain.root / relative).write_text("changed", encoding="utf-8")
    assert chain.run()["accepted"] is False


@pytest.mark.parametrize(
    "values",
    [{"indexSHA256": "0" * 64}, {"compressedPacketsIdentical": False}, {"probeEvidence": {}}],
)
def test_assembly_rejections(chain: Fixture, values: dict[str, object]) -> None:
    update(chain.root / "assembly.json", values)
    assert record(chain.run()["checks"])["sourceAssembly"] is False


def test_sha_rehash_not_stat_reuse(chain: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    store = chain.store()
    store.digest(chain.movie)
    chain.movie.write_text("changed with cached identity", encoding="utf-8")
    monkeypatch.setattr(store, "stat_identity", lambda path: store.identities[path])
    with pytest.raises(ReplayError, match="SHA256 changed"):
        store.after()


def test_cached_identity_and_symlink_refused(chain: Fixture) -> None:
    store = chain.store()
    store.digest(chain.movie)
    chain.movie.write_text("changed", encoding="utf-8")
    with pytest.raises(ReplayError, match="mutated"):
        store.digest(chain.movie)
    link = chain.root / "linked"
    link.symlink_to(chain.movie)
    with pytest.raises(ReplayError, match="canonical"):
        store.digest(link)


def test_cli_and_nonoverwrite(chain: Fixture) -> None:
    args = [
        str(chain.config),
        str(chain.movie),
        str(chain.technical),
        str(chain.root / "report.json"),
    ]
    assert audit.main(args) == 1
    assert audit.main(args) == 1


def test_cli_positive(chain: Fixture) -> None:
    assert (
        audit.main(
            [
                str(chain.config),
                str(chain.movie),
                str(chain.technical),
                str(chain.root / "report.json"),
                "--source-clock",
                str(chain.paths.source_clock),
                "--clock-receipt",
                str(chain.paths.clock_receipt),
                "--assembly-receipt",
                str(chain.paths.assembly_receipt),
                "--protection",
                str(chain.paths.protection),
                "--remux-receipt",
                str(chain.paths.remux_receipt),
            ]
        )
        == 0
    )


def rewrite_origin(chain: Fixture, scale: int) -> render.Config:
    config = render.load_config(chain.config)
    raw = record(json.loads(config.chunks.read_text(encoding="utf-8")))
    row = record(render._list(raw["ranges"])[0])
    row.update(sourceFirstPTS=1, sourceEndPTS=scale + 1, timeBase=f"1/{scale}")
    write(config.chunks, {"ranges": [row]})
    source = config.composition / "request-0000.json"
    request = record(json.loads(source.read_text(encoding="utf-8")))
    recipe = record(request["recipe"])
    recipe["clips"] = [
        render._clip_json(clip, config.rate)
        for clip in render.clips(
            render.read_cuts(config.cuts), render.read_chunks(config.chunks), config.rate
        )
    ]
    request["recipe"] = recipe
    write(source, request)
    parts = [native_part(config, source, index) for index in range(2)]
    write(
        config.composition / "parts-0000/manifest.json",
        {"sourceRequest": ref(source), "parts": [item[0] for item in parts]},
    )
    return config


def test_exact_fractional_source_mapping(chain: Fixture) -> None:
    config = rewrite_origin(chain, 60_000)
    assert len(audit.selection_plan(chain.store(), config)) == 2
    start = render.clips(render.read_cuts(config.cuts), render.read_chunks(config.chunks), 30)[
        0
    ].start
    assert start == Fraction(1999, 60_000)
    # This is a recipe-mapping test, not full original-source acceptance.
    assert Fraction(str(float(start))) != start


def test_unrepresentable_native_origin_not_rounded(chain: Fixture) -> None:
    config = rewrite_origin(chain, 90_000)
    with pytest.raises(ReplayError, match="not representable"):
        audit.selection_plan(chain.store(), config)


def test_unsupported_terminal_28_milliseconds(chain: Fixture) -> None:
    config = render.load_config(chain.config)
    write(config.cuts, {"cuts": [{"source_start_frame": 0, "source_end_frame": 9}]})
    write(
        config.chunks,
        {
            "ranges": [
                {
                    "media": str(config.storage / "continuous.mov"),
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": 272,
                    "timeBase": "1/1000",
                }
            ]
        },
    )
    with pytest.raises(ReplayError, match="boundary|outside"):
        audit.selection_plan(chain.store(), config)


def test_retained_same_total_wrong_order(chain: Fixture) -> None:
    config = render.load_config(chain.config)
    write(
        config.cuts,
        {
            "cuts": [
                {"source_start_frame": 2, "source_end_frame": 4},
                {"source_start_frame": 6, "source_end_frame": 10},
            ]
        },
    )
    with pytest.raises(ReplayError, match="rational cut-plan mapping"):
        audit.selection_plan(chain.store(), config)


def test_no_removed_count_fabricated(chain: Fixture) -> None:
    config = render.load_config(chain.config)
    write(
        config.cuts,
        {"sourceFrames": 30, "cuts": [{"source_start_frame": 0, "source_end_frame": 30}]},
    )
    result = chain.run()
    assert result["removedFrames"] == 0
    assert result["accepted"] is False
    with pytest.raises(ReplayError, match="accounting"):
        adapter._cuts(
            chain.store(),
            {
                **result,
                "selectionPlanMatchesProtectedCuts": True,
                "recordedExecutionMatchesPlan": True,
            },
            render_path=config.path,
            render=record(json.loads(config.path.read_text())),
            movie=chain.movie,
            frames=30,
        )


def test_missing_join_media_dependency_and_independent_mux(chain: Fixture) -> None:
    config = render.load_config(chain.config)
    plan = config.work / "join-workflow.json"
    raw = record(json.loads(plan.read_text()))
    stage = record(render._list(raw["stages"])[0])
    stage["inputs"] = [str(config.work / "join-plan.json")]
    write(plan, {"version": 1, "stages": [stage]})
    seal(plan, stage)
    result = chain.run()
    assert record(result["checks"])["joinExecution"] is False
    assert record(result["checks"])["muxExecution"] is True
    assert record(result["checks"])["compressorAndRemux"] is True


def test_join_order_not_just_total(chain: Fixture) -> None:
    config = render.load_config(chain.config)
    plan = config.work / "join-plan.json"
    write(plan, list(reversed(render._list(render._json(plan)))))
    workflow = config.work / "join-workflow.json"
    seal(workflow, record(render._list(record(render._json(workflow))["stages"])[0]))
    result = chain.run()
    assert record(result["checks"])["joinExecution"] is False


def test_binding_mapping_schema(chain: Fixture) -> None:
    write(
        chain.root / "clock-receipt.json",
        {
            "inputs": {
                str(render.load_config(chain.config).source): binding(
                    render.load_config(chain.config).source
                )["sha256"],
                str(chain.root / "clock.json"): binding(chain.root / "clock.json")["sha256"],
            }
        },
    )
    assert chain.run()["accepted"] is True


def test_changed_during_audit_flags_fail_closed(
    chain: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = audit.Bindings.after

    def mutated(store: audit.Bindings) -> list[dict[str, object]]:
        chain.movie.write_text("changed during audit", encoding="utf-8")
        return original(store)

    monkeypatch.setattr(audit.Bindings, "after", mutated)
    result = chain.run()
    assert result["accepted"] is False
    assert result["recordedExecutionMatchesPlan"] is False
    assert result["selectionPlanMatchesProtectedCuts"] is False
    assert record(result["checks"])["inputsStable"] is False


@pytest.mark.parametrize(
    "fault",
    ["legacy", "missing-master", "changed-master", "changed-preset", "submit-id", "terminal"],
)
def test_new_compressor_producer_binding_rejects_missing_or_changed_proof(
    chain: Fixture, fault: str
) -> None:
    config = render.load_config(chain.config)
    path = config.work / "final-job-state/final.json"
    state = record(render._json(path))
    if fault == "legacy":
        state.pop("submissionVersion")
    elif fault == "missing-master":
        state["inputs"] = render._list(state["inputs"])[1:]
    elif fault == "changed-master":
        (config.storage / "av-master.mov").write_bytes(b"changed master")
    elif fault == "changed-preset":
        config.final_preset.write_bytes(b"changed preset")
    elif fault == "submit-id":
        state["submitResponse"] = '{"batch":{"jobs":[{"jobID":"different"}]}}'
    else:
        state["terminalInputsVerified"] = False
    write(path, state)
    result = chain.run()
    assert result["accepted"] is False
    assert record(result["checks"])["compressorInputBinding"] is False


def test_actual_source_frames_are_not_nominal_grid_slots(chain: Fixture) -> None:
    config = render.load_config(chain.config)
    update(config.cuts, {"actualSourceFrames": 30})
    result = chain.run()
    assert result["decodedSourceFrames"] == 3
    assert result["sourceFrames"] == 30
    assert record(result["checks"])["sourceGridContract"] is False


def recorded_join(
    tmp_path: Path, *, part_count: int, evidence_count: int, manifest: bool | None = None
) -> tuple[render.Config, audit.Bindings]:
    """Build only small synthetic recorded inputs; no native or media command runs."""
    root = tmp_path / "edits/replay-7"
    source = write(root / "source.mp4", "synthetic original")
    cuts = write(root / "cuts.json", {"cuts": []})
    chunks = write(root / "chunks.json", {"ranges": []})
    config = render.Config(
        path=root / "config.json",
        recording="replay-7",
        captions=root / "captions",
        work=root / "work",
        storage=root / "storage",
        rate=30,
        width=64,
        height=64,
        cuts=cuts,
        chunks=chunks,
        template=root / "template",
        expected_text="",
        finish=(),
        alpha_preset=root / "alpha",
        encode_preset=root / "encode",
        native=root / "native",
        encoding={},
        audio_source=root / "voice",
        audio_clock=root / "clock",
        source=source,
        delivery_root=root / "delivery",
        final_preset=root / "preset",
        calibration=root / "calibration",
        viewer_root=root,
        cut_evidence=tuple(
            write(root / f"cut-evidence-{index}.json", {"syntheticEvidence": index})
            for index in range(evidence_count)
        ),
    )
    requests = [
        config.composition / "parts-0000" / f"part-{index:04d}" / "request.json"
        for index in range(part_count)
    ]
    movies = [
        write(config.storage / f"native-{index:04d}.mp4", {"syntheticPart": index})
        for index in range(part_count)
    ]
    responses = [
        write(
            request.with_name("render.response.json"),
            {"outputPath": str(movie), "frameCount": index + 1},
        )
        for index, (request, movie) in enumerate(zip(requests, movies, strict=True))
    ]
    write(
        config.composition / "parts-0000/manifest.json",
        {
            "parts": [
                {"path": str(request), "frames": index + 1}
                for index, request in enumerate(requests)
            ]
        },
    )
    write(config.storage / "joined-video.mp4", "synthetic joined")
    assert len(responses) == part_count
    use_manifest = (
        mechanical_join.needs_manifest(part_count, evidence_count) if manifest is None else manifest
    )
    if use_manifest:
        plan = mechanical_join.generate_join(render.rendered_parts(config), config)
    else:
        [plan] = render.join(config)
    seal(plan, record(render._list(record(render._json(plan))["stages"])[0]))
    store = audit.Bindings(root, source, config.delivery_root, artifact_root=config.storage)
    store.native_movies.extend(movies)
    return config, store


@pytest.mark.parametrize(
    ("part_count", "evidence_count", "manifest"),
    [(253, 0, False), (254, 0, True), (252, 1, False), (253, 1, True), (250, 4, True)],
)
def test_join_audit_accepts_real_generator_at_conditional_array_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    part_count: int,
    evidence_count: int,
    manifest: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    config, store = recorded_join(tmp_path, part_count=part_count, evidence_count=evidence_count)
    stage = parse_stages(store.read(config.work / "join-workflow.json"))[0]
    assert (stage.command[2] == "replay.mechanical_join") is manifest
    assert len(stage.inputs) <= 256
    audit.downstream(store, config, join=True)
    assert store.after() == store.rows()


def test_manifest_join_916_parts_are_individually_pinned_and_after_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config, store = recorded_join(tmp_path, part_count=916, evidence_count=4)
    audit.downstream(store, config, join=True)
    assert len(store.native_movies) == 916
    assert len(parse_stages(store.read(config.work / "join-workflow.json"))[0].inputs) == 7
    assert all(movie in store.hashes for movie in store.native_movies)
    assert store.after() == store.rows()
    store.native_movies[-1].write_bytes(b"changed after audit")
    with pytest.raises(ReplayError, match="Input changed"):
        store.after()


@pytest.fixture
def manifest_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[render.Config, audit.Bindings]:
    monkeypatch.chdir(tmp_path)
    return recorded_join(tmp_path, part_count=2, evidence_count=1, manifest=True)


def reseal_join(config: render.Config, stage: dict[str, object] | None = None) -> None:
    plan = config.work / "join-workflow.json"
    raw_stage = stage or record(render._list(record(render._json(plan))["stages"])[0])
    write(plan, {"version": 1, "stages": [raw_stage]})
    seal(plan, raw_stage)


def test_manifest_path_and_output_are_rejected_before_indirect_part_pinning(
    manifest_case: tuple[render.Config, audit.Bindings], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = manifest_case
    plan = config.work / "join-workflow.json"
    stage = record(render._list(record(render._json(plan))["stages"])[0])
    command = render._list(stage["command"])
    command[4] = str(config.work / "other-manifest.json")
    stage["command"] = command
    reseal_join(config, stage)

    def forbidden_pin(_path: Path, _sha: object) -> Path:
        pytest.fail("Invalid command must not reach indirect part pinning")

    monkeypatch.setattr(store, "pin", forbidden_pin)
    with pytest.raises(ReplayError, match="manifest join command/input/output"):
        audit.downstream(store, config, join=True)


@pytest.mark.parametrize(
    ("index", "replacement"),
    [(0, "/usr/bin/false"), (3, "join-files"), (5, "--audio"), (8, "60")],
)
def test_manifest_join_changed_command_is_rejected(
    manifest_case: tuple[render.Config, audit.Bindings], index: int, replacement: str
) -> None:
    config, store = manifest_case
    stage = record(
        render._list(record(render._json(config.work / "join-workflow.json"))["stages"])[0]
    )
    command = render._list(stage["command"])
    command[index] = replacement
    stage["command"] = command
    reseal_join(config, stage)
    with pytest.raises(ReplayError, match="command|upstream"):
        audit.downstream(store, config, join=True)


def test_manifest_join_changed_output_is_rejected(
    manifest_case: tuple[render.Config, audit.Bindings],
) -> None:
    config, store = manifest_case
    output = write(config.storage / "other-joined.mp4", "synthetic other join")
    stage = record(
        render._list(record(render._json(config.work / "join-workflow.json"))["stages"])[0]
    )
    command = render._list(stage["command"])
    command[6] = str(output)
    stage.update(command=command, outputs=[str(output)])
    reseal_join(config, stage)
    with pytest.raises(ReplayError, match="command/input/output"):
        audit.downstream(store, config, join=True)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_manifest_join_incomplete_stage_input_pins_are_rejected(
    manifest_case: tuple[render.Config, audit.Bindings], index: int
) -> None:
    config, store = manifest_case
    stage = record(
        render._list(record(render._json(config.work / "join-workflow.json"))["stages"])[0]
    )
    inputs = render._list(stage["inputs"])
    inputs.pop(index)
    stage["inputs"] = inputs
    reseal_join(config, stage)
    with pytest.raises(ReplayError, match="command/input/output"):
        audit.downstream(store, config, join=True)


def test_manifest_join_duplicate_stage_input_is_rejected(
    manifest_case: tuple[render.Config, audit.Bindings],
) -> None:
    config, store = manifest_case
    stage = record(
        render._list(record(render._json(config.work / "join-workflow.json"))["stages"])[0]
    )
    stage["inputs"] = [*render._list(stage["inputs"]), str(config.cuts)]
    write(config.work / "join-workflow.json", {"version": 1, "stages": [stage]})
    with pytest.raises(StageError, match="distinct"):
        audit.downstream(store, config, join=True)


def test_manifest_join_completed_fingerprint_is_required_before_manifest_parsing(
    manifest_case: tuple[render.Config, audit.Bindings], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, store = manifest_case
    update(config.work / "join-workflow-state/join.json", {"fingerprint": "0" * 64})

    def forbidden_parse(_data: object) -> tuple[mechanical_join.ManifestPart, ...]:
        pytest.fail("Incomplete original stage must not reach manifest parsing")

    monkeypatch.setattr(mechanical_join, "parse_manifest", forbidden_parse)
    with pytest.raises(ReplayError, match="Completed record input/output"):
        audit.downstream(store, config, join=True)


def test_manifest_join_modified_manifest_does_not_gain_a_new_fingerprint(
    manifest_case: tuple[render.Config, audit.Bindings],
) -> None:
    config, store = manifest_case
    update(config.work / "join-manifest.json", {"version": 2})
    with pytest.raises(ReplayError, match="Completed record input/output"):
        audit.downstream(store, config, join=True)


def test_manifest_join_missing_manifest_is_rejected(
    manifest_case: tuple[render.Config, audit.Bindings],
) -> None:
    config, store = manifest_case
    (config.work / "join-manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        audit.downstream(store, config, join=True)


@pytest.mark.parametrize("fault", ["swapped", "frames", "digest", "missing", "duplicate"])
def test_manifest_join_rejects_resealed_wrong_inventory(
    manifest_case: tuple[render.Config, audit.Bindings], fault: str
) -> None:
    config, store = manifest_case
    manifest = config.work / "join-manifest.json"
    parts = [record(row) for row in render._list(record(render._json(manifest))["parts"])]
    if fault == "swapped":
        parts.reverse()
    elif fault == "frames":
        parts[0]["durationFrames"] = 2
    elif fault == "digest":
        parts[0]["sha256"] = "0" * 64
    elif fault == "missing":
        parts.pop()
    else:
        parts[1] = parts[0]
    update(manifest, {"parts": parts})
    reseal_join(config)
    with pytest.raises(ReplayError, match="order/path/frame|SHA256 binding|part count|repeats"):
        audit.downstream(store, config, join=True)


def test_manifest_join_rejects_modified_part_without_stage_media_inputs(
    manifest_case: tuple[render.Config, audit.Bindings],
) -> None:
    config, store = manifest_case
    store.native_movies[0].write_bytes(b"changed native part")
    with pytest.raises(ReplayError, match="SHA256 binding differs"):
        audit.downstream(store, config, join=True)


def test_manifest_join_rejects_missing_part(
    manifest_case: tuple[render.Config, audit.Bindings],
) -> None:
    config, store = manifest_case
    store.native_movies[0].unlink()
    with pytest.raises(FileNotFoundError):
        audit.downstream(store, config, join=True)


@pytest.mark.parametrize("fault", ["symlink", "noncanonical"])
def test_manifest_join_rejects_noncanonical_part_even_when_both_inventories_agree(
    manifest_case: tuple[render.Config, audit.Bindings], fault: str
) -> None:
    config, store = manifest_case
    movie = store.native_movies[0]
    if fault == "symlink":
        alias = config.storage / "native-alias.mp4"
        alias.symlink_to(movie)
    else:
        alias = config.storage / ".." / "storage" / movie.name
    store.native_movies[0] = alias
    manifest = config.work / "join-manifest.json"
    parts = [record(row) for row in render._list(record(render._json(manifest))["parts"])]
    parts[0]["sourcePath"] = str(alias)
    update(manifest, {"parts": parts})
    reseal_join(config)
    with pytest.raises(ReplayError, match="noncanonical"):
        audit.downstream(store, config, join=True)


def test_manifest_join_rejects_linked_manifest(
    manifest_case: tuple[render.Config, audit.Bindings],
) -> None:
    config, store = manifest_case
    manifest = config.work / "join-manifest.json"
    saved = config.work / "saved-manifest.json"
    manifest.rename(saved)
    manifest.symlink_to(saved)
    with pytest.raises(ReplayError, match="noncanonical"):
        audit.downstream(store, config, join=True)


def test_complete_chain_retains_all_checks_with_manifest_join(chain: Fixture) -> None:
    config = render.load_config(chain.config)
    rendered = render.rendered_parts(config)
    movies = [
        Path(str(record(render._json(request.with_name("render.response.json")))["outputPath"]))
        for request, _frames in rendered
    ]
    manifest = mechanical_join.write_manifest(
        config.work / "join-manifest.json",
        (
            mechanical_join.pin_part(movie, frames)
            for (_request, frames), movie in zip(rendered, movies, strict=True)
        ),
    )
    stage = mechanical_join.join_stage(
        manifest,
        config.storage / "joined-video.mp4",
        config.rate,
        [sys.executable, "-m", "replay.mechanical_join"],
        [config.cuts, config.chunks, *config.cut_evidence],
    )
    stage["timeout_seconds"] = render.JOIN_TIMEOUT_SECONDS
    reseal_join(config, stage)
    result = chain.run()
    assert result["accepted"] is True
    assert result["recordedExecutionMatchesPlan"] is True
    assert record(result["checks"])["joinExecution"] is True
    assert record(result["checks"])["muxExecution"] is True
    assert result["inputsAfter"] == result["inputs"]


def test_manifest_join_rejects_incomplete_completed_native_inventory(
    manifest_case: tuple[render.Config, audit.Bindings],
) -> None:
    config, store = manifest_case
    store.native_movies.pop()
    with pytest.raises(ReplayError, match="part count"):
        audit.downstream(store, config, join=True)


def test_manifest_join_rejects_frames_differing_from_original_rendered_parts(
    manifest_case: tuple[render.Config, audit.Bindings],
) -> None:
    config, store = manifest_case
    partition_manifest = config.composition / "parts-0000/manifest.json"
    parts = [record(row) for row in render._list(record(render._json(partition_manifest))["parts"])]
    parts[0]["frames"] = 3
    update(partition_manifest, {"parts": parts})
    with pytest.raises(ReplayError, match="order/path/frame"):
        audit.downstream(store, config, join=True)


@pytest.fixture(params=[False, True])
def sealed_join_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[render.Config, audit.Bindings, Path]:
    monkeypatch.chdir(tmp_path)
    tools = tmp_path.parent / f"{tmp_path.name}-tools"
    python = write(tools / "python", "synthetic Python executable")
    write(tools / "ffprobe", "synthetic ffprobe executable")
    write(tools / "ffmpeg", "synthetic ffmpeg executable")
    code = write(tmp_path / "helper.py", "synthetic helper code")
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(mechanical_plan_provenance, "executable", lambda name: tools / name)
    monkeypatch.setattr(mechanical_plan_provenance, "code_paths", lambda: (code,))
    config, store = recorded_join(tmp_path, part_count=2, evidence_count=1, manifest=request.param)
    stage = record(
        render._list(record(render._json(config.work / "join-workflow.json"))["stages"])[0]
    )
    mechanical_plan_provenance.bind_verifier(stage)
    reseal_join(config, stage)
    return config, store, tools


def test_sealed_join_establishes_only_exact_tool_scope_and_pins_all_parts(
    sealed_join_case: tuple[render.Config, audit.Bindings, Path],
) -> None:
    config, store, tools = sealed_join_case
    with pytest.raises(ReplayError, match="authorized storage"):
        store.digest(tools / "ffmpeg")
    audit.downstream(store, config, join=True)
    assert set(store.caption_resources) == {tools / "python", tools / "ffprobe", tools / "ffmpeg"}
    assert all(path in store.hashes for path in store.native_movies)
    assert store.after() == store.rows()
    (tools / "ffmpeg").write_bytes(b"changed after acceptance")
    with pytest.raises(ReplayError, match="Input changed"):
        store.after()


@pytest.mark.parametrize("fault", ["python", "ffprobe", "ffmpeg", "helper", "contract", "part"])
def test_sealed_join_changed_provenance_is_rejected(
    sealed_join_case: tuple[render.Config, audit.Bindings, Path], fault: str
) -> None:
    config, store, tools = sealed_join_case
    stage = parse_stages(store.read(config.work / "join-workflow.json"))[0]
    if fault == "helper":
        path = config.viewer_root.parents[1] / "helper.py"
    elif fault == "contract":
        path = Path(stage.verify_command[3])
    elif fault == "part":
        path = store.native_movies[0]
    else:
        path = tools / fault
    path.write_bytes(b"modified original producer input")
    with pytest.raises(ReplayError, match="provenance differs|contract changed"):
        audit.downstream(store, config, join=True)
    assert store.caption_resources == {}


def test_sealed_join_missing_extra_pin_is_rejected(
    sealed_join_case: tuple[render.Config, audit.Bindings, Path],
) -> None:
    config, store, tools = sealed_join_case
    stage = record(
        render._list(record(render._json(config.work / "join-workflow.json"))["stages"])[0]
    )
    stage["inputs"] = [
        path for path in render._list(stage["inputs"]) if path != str(tools / "ffmpeg")
    ]
    reseal_join(config, stage)
    with pytest.raises(ReplayError, match="sealed consumer contract"):
        audit.downstream(store, config, join=True)


def test_sealed_join_rejects_unknown_extra_even_in_genuine_resealed_contract(
    sealed_join_case: tuple[render.Config, audit.Bindings, Path],
) -> None:
    config, store, _tools = sealed_join_case
    plan = config.work / "join-workflow.json"
    stage = parse_stages(render._json(plan))[0]
    original = record(mechanical_plan_provenance.read(Path(stage.verify_command[3]))["stage"])
    extra = write(config.work / "unrelated.json", "unrelated extra input")
    original["inputs"] = [*render._list(original["inputs"]), str(extra)]
    mechanical_plan_provenance.bind_verifier(original)
    reseal_join(config, original)
    with pytest.raises(ReplayError, match="command/input/output|ordered join"):
        audit.downstream(store, config, join=True)


def test_sealed_join_does_not_replace_original_completed_fingerprint(
    sealed_join_case: tuple[render.Config, audit.Bindings, Path],
) -> None:
    config, store, _tools = sealed_join_case
    update(config.work / "join-workflow-state/join.json", {"fingerprint": "0" * 64})
    with pytest.raises(ReplayError, match="Completed record input/output"):
        audit.downstream(store, config, join=True)
