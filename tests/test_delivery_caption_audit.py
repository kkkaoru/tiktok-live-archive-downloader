"""Synthetic frame/clock/layout/receipt tests; no apps or recordings."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_motion_caption import sequence_template as sequence_template
from test_motion_caption import stage_plan as stage_plan
from test_motion_caption import text_template as text_template

from replay import delivery_caption_audit as audit
from replay import delivery_verification as verification
from replay import motion_finish, motion_stages
from replay.compressor_batch import Batch, Job
from replay.core import ReplayError
from replay.delivery_render import Config
from replay.recordings import record


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, object]:
    return {
        key: value for key, value in audit.Inputs().bind(path).items() if key in {"path", "sha256"}
    }


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setattr(verification, "REPOSITORY", tmp_path)
    captions, work = tmp_path / "captions", tmp_path / "render"
    work.mkdir()
    plan = work / "motion/segment-0000/plans/part-0000.json"
    write_json(
        plan,
        {
            "frames": 5,
            "width": 4,
            "height": 4,
            "rate": 30,
            "placements": [
                {"text": "あ", "startFrame": 0, "endFrame": 2, "x": 0, "y": 0},
                {"text": "あい", "startFrame": 3, "endFrame": 5, "x": 0, "y": 0},
            ],
        },
    )
    font = {"renderedAscent": 1}
    calibration = write_json(
        tmp_path / "calibration.json",
        {
            "nativeFont": font,
            "style": {
                "motion_line_advance": 1,
                "font_name": "Test",
                "ass_font_size": 2,
                "border": 1,
                "rim": 1,
            },
            "padding": 0,
        },
    )
    response = write_json(
        tmp_path / "batch-000000.response.json",
        {
            "font": font,
            "bounds": [
                {"text": "あ", "width": 2, "height": 2},
                {"text": "あい", "width": 2, "height": 2},
            ],
        },
    )
    request = write_json(
        tmp_path / "batch-000000.request.json",
        {
            "fontName": "Test",
            "assFontSize": 2,
            "border": 1,
            "rim": 1,
            "texts": ["あ", "あい"],
        },
    )
    combined = write_json(
        captions / "combined-0000.json",
        {
            "version": 1,
            "candidateOnly": True,
            "coordinateSystem": "top-left canvas; calibrated text origin",
            "canvas": [4, 4],
            "renderCalibration": ref(calibration),
            "policy": {"frame_rate": 30},
            "inputs": {
                str(response): ref(response)["sha256"],
                str(request): ref(request)["sha256"],
            },
            "candidates": [
                {"cue": {"text": "あ", "start": 1 / 30, "end": 3 / 30}, "x": 2, "y": 2},
                {"cue": {"text": "あい", "start": 4 / 30, "end": 6 / 30}, "x": 2, "y": 2},
            ],
        },
    )
    write_json(
        work / "motion/segment-0000/plans/manifest.json",
        {
            "frames": 7,
            "rate": 30,
            "candidate": ref(combined),
            "renderCalibration": ref(calibration),
            "parts": [{**ref(plan), "offsetFrame": 1, "frames": 5, "stages": 2}],
        },
    )
    write_json(
        captions / "segments.json", {"segments": [{"segment": 0, "offsetFrame": 0, "frames": 7}]}
    )
    cuts = write_json(
        tmp_path / "cuts.json",
        {"cuts": [{"source_start_frame": 30, "source_end_frame": 37, "output_start_frame": 0}]},
    )
    dummy = tmp_path / "dummy"
    dummy.write_bytes(b"fixture")
    source = work / "motion/segment-0000/part-0000.moti"
    source.write_bytes(b"synthetic motion document, never submitted")
    movie = work / "alpha/segment-0000-part-0000.mov"
    movie.parent.mkdir()
    movie.write_bytes(b"fake alpha only for mocked decoder tests")
    write_json(
        work / "alpha-jobs.json",
        {
            "jobs": [
                {
                    "name": "alpha-0000-0000",
                    "source": str(source),
                    "preset": str(dummy),
                    "output": str(movie),
                }
            ]
        },
    )
    write_json(
        work / "alpha-jobs-state/alpha-0000-0000.json",
        {
            "status": "Successful",
            "jobID": "job-1",
            "output": str(movie),
            "bytes": movie.stat().st_size,
        },
    )
    cfg = Config(
        path=dummy,
        recording="fixture",
        captions=captions,
        work=work,
        storage=work,
        rate=30,
        width=4,
        height=4,
        cuts=cuts,
        chunks=dummy,
        template=dummy,
        expected_text="Title",
        finish=(),
        alpha_preset=dummy,
        encode_preset=dummy,
        native=dummy,
        encoding={},
        audio_source=dummy,
        audio_clock=dummy,
        source=dummy,
        delivery_root=tmp_path,
        final_preset=dummy,
        calibration=calibration,
        viewer_root=tmp_path,
    )
    monkeypatch.setattr(audit, "load_config", lambda _: cfg)
    return cfg


@pytest.fixture
def part(config: Config) -> audit.Part:
    return audit.load_parts(config, audit.Inputs())[0][0]


@pytest.fixture
def measurement() -> dict[str, object]:
    return {
        "rasterRuns": [
            {"startFrame": 0, "endFrame": 2, "bounds": [1, 1, 3, 3]},
            {"startFrame": 2, "endFrame": 3, "bounds": None},
            {"startFrame": 3, "endFrame": 5, "bounds": [1, 1, 3, 3]},
        ],
        "decodedFrames": 5,
        "fullDecodeVerified": True,
    }


@pytest.fixture
def constraints() -> list[dict[str, object]]:
    return [
        {
            "startFrame": 0,
            "endFrame": 2,
            "text": "あ",
            "envelope": [1, 1, 3, 3],
            "region": [0, 0, 4, 4],
            "centerX": 2,
            "centerTolerancePixels": 0,
            "clearance": 1,
            "foregroundTop": 4,
            "faceTop": 4,
            "headTop": 4,
        },
        {
            "startFrame": 3,
            "endFrame": 5,
            "text": "あい",
            "envelope": [1, 1, 3, 3],
            "region": [0, 0, 4, 4],
            "centerX": 2,
            "centerTolerancePixels": 0,
            "clearance": 1,
            "foregroundTop": 4,
            "faceTop": 4,
            "headTop": 4,
        },
    ]


def test_plans_and_exact_srt(config: Config, part: audit.Part) -> None:
    assert part.offset == 1
    assert part.envelopes == ((1.0, 1.0, 3.0, 3.0), (1.0, 1.0, 3.0, 3.0))
    assert audit.canonical_srt([part], config.rate) == (
        "1\n00:00:00,033 --> 00:00:00,100\nあ\n\n2\n00:00:00,133 --> 00:00:00,200\nあい\n\n"
    )
    assert audit.timestamp(108031, 30) == "01:00:01,033"
    assert audit.timestamp(2, 30) == "00:00:00,066"


@pytest.mark.parametrize("frame", [1, 2, 29, 31, 107999])
def test_floor_is_frame_equivalent(frame: int) -> None:
    ms = frame * 1000 // 30
    assert (frame - 1) * 1000 < ms * 30 <= frame * 1000


def test_srt_overlap_rate_text_and_order(part: audit.Part) -> None:
    with pytest.raises(ReplayError, match="Overlapping"):
        audit.canonical_srt([part, part], 30)
    with pytest.raises(ReplayError, match="frame rate"):
        audit.canonical_srt([part], 0)
    with pytest.raises(ReplayError, match="clock/text"):
        audit.cue({"startFrame": 0, "endFrame": 0, "text": "a"}, 5)
    with pytest.raises(ReplayError, match="SRT text"):
        audit.cue({"startFrame": 0, "endFrame": 1, "text": "a\n\nb"}, 5)


def test_raster_nonzero16_and_rgba() -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.uint16)
    rgba[1:3, 1:3, 3] = 65535
    rgba[0, 3, 3] = 1
    assert audit.raster_bounds(rgba[:, :, 3]) == [1, 0, 4, 3]
    assert audit.raster_bounds(np.zeros((2, 2), dtype=np.uint16)) is None
    with pytest.raises(ReplayError, match="alpha16"):
        audit.raster_bounds(rgba)
    runs: list[dict[str, object]] = []
    audit.append_run(runs, 0, None)
    audit.append_run(runs, 1, None)
    audit.append_run(runs, 2, [0, 0, 1, 1])
    assert runs == [
        {"startFrame": 0, "endFrame": 2, "bounds": None},
        {"startFrame": 2, "endFrame": 3, "bounds": [0, 0, 1, 1]},
    ]


def test_every_full_state_and_gap(
    part: audit.Part, measurement: dict[str, object], constraints: list[dict[str, object]]
) -> None:
    assert audit.evaluate(part, measurement, constraints) == {
        "fullCaptionStates": 2,
        "sampleCount": 4,
        "measuredFrames": 5,
        "gapFrames": 1,
        "gapsVerified": True,
        "visibleStatesVerified": True,
        "constraintFrames": 4,
        "fullStateConstraintCoverage": True,
        "measuredEnvelopeFrames": 4,
        "measuredEnvelopeVerified": True,
        "boundsAndGapsVerified": True,
    }
    assert audit.evaluate(part, measurement, constraints[:1])["boundsAndGapsVerified"] is False
    assert (
        audit.evaluate(replace(part, envelopes=()), measurement, constraints)[
            "boundsAndGapsVerified"
        ]
        is False
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("headTop", 3),
        ("faceTop", 3),
        ("foregroundTop", 3),
        ("centerX", 1),
        ("envelope", [1, 1, 2, 3]),
        ("region", [0, 0, 2, 4]),
    ],
)
def test_constraint_failure(
    part: audit.Part,
    measurement: dict[str, object],
    constraints: list[dict[str, object]],
    key: str,
    value: object,
) -> None:
    constraints[1][key] = value
    assert audit.evaluate(part, measurement, constraints)["boundsAndGapsVerified"] is False


@pytest.mark.parametrize(
    "key,value",
    [("clearance", -1), ("centerTolerancePixels", -1), ("text", "changed"), ("startFrame", 4)],
)
def test_invalid_constraint(
    part: audit.Part,
    measurement: dict[str, object],
    constraints: list[dict[str, object]],
    key: str,
    value: object,
) -> None:
    constraints[1][key] = value
    with pytest.raises(ReplayError):
        audit.evaluate(part, measurement, constraints)


def test_gaps_empty_states_and_cache_integrity(
    part: audit.Part, constraints: list[dict[str, object]]
) -> None:
    filled: dict[str, object] = {
        "rasterRuns": [{"startFrame": 0, "endFrame": 5, "bounds": [1, 1, 3, 3]}]
    }
    assert audit.evaluate(part, filled, constraints)["gapsVerified"] is False
    empty: dict[str, object] = {"rasterRuns": [{"startFrame": 0, "endFrame": 5, "bounds": None}]}
    assert audit.evaluate(part, empty, constraints)["visibleStatesVerified"] is False
    with pytest.raises(ReplayError, match="every alpha"):
        audit.evaluate(part, {"rasterRuns": [{"startFrame": 1, "endFrame": 5}]}, constraints)
    with pytest.raises(ReplayError, match="Incomplete"):
        audit.evaluate(part, {"rasterRuns": []}, constraints)
    outside = replace(part, envelopes=((1, 1, 2, 3), (1, 1, 2, 3)))
    assert audit.evaluate(outside, filled, constraints)["measuredEnvelopeVerified"] is False


@pytest.mark.parametrize(
    "call,value",
    [
        (audit.absolute, "relative"),
        (audit.rows, {}),
        (audit.integer, True),
        (audit.number, float("nan")),
        (audit.box, [0, 0, 0, 1]),
        (audit.box, []),
        (audit.number, "1"),
    ],
)
def test_bad_primitives(call: object, value: object) -> None:
    if not callable(call):
        raise AssertionError("test requires callable")
    with pytest.raises(ReplayError):
        call(value)


def test_inputs_path_and_changed_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verification, "REPOSITORY", tmp_path)
    path = tmp_path / "input"
    path.write_bytes(b"first")
    inputs = audit.Inputs()
    inputs.bind(path)
    assert inputs.rows()[0]["bytes"] == 5
    with pytest.raises(ReplayError, match="fingerprint"):
        inputs.bind(path, "a" * 64)
    with pytest.raises(ReplayError, match="SHA256"):
        inputs.reference({"path": str(path)})
    with pytest.raises(ReplayError, match="checkout"):
        audit.local(tmp_path.parent)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ReplayError, match="symlinks"):
        inputs.bind(link)
    path.write_bytes(b"other")
    with pytest.raises(ReplayError, match="changed"):
        inputs.verify()


def test_json_budget_and_duplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "json"
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError):
        audit.document(path, audit.Inputs())
    monkeypatch.setattr(audit, "MAX_JSON", 1)
    with pytest.raises(ReplayError, match="budget"):
        audit.document(path, audit.Inputs())


@pytest.mark.parametrize(
    "change", ["clock", "candidate", "extent", "states", "text", "empty", "coverage", "segment"]
)
def test_reject_changed_plans(config: Config, change: str) -> None:
    path = config.work / "motion/segment-0000/plans/manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if change == "clock":
        data["rate"] = 25
    elif change == "candidate":
        other = config.captions / "other.json"
        other.write_bytes((config.captions / "combined-0000.json").read_bytes())
        data["candidate"] = ref(other)
    elif change == "extent":
        data["parts"][0]["offsetFrame"] = 4
    elif change == "states":
        data["parts"][0]["stages"] = 1
    elif change == "text":
        plan = Path(data["parts"][0]["path"])
        raw = json.loads(plan.read_text(encoding="utf-8"))
        raw["placements"][0]["text"] = "changed"
        write_json(plan, raw)
        data["parts"][0].update(ref(plan))
    elif change == "empty":
        data["parts"] = []
    elif change == "coverage":
        write_json(config.cuts, {"cuts": [{"source_start_frame": 30, "source_end_frame": 38}]})
    else:
        write_json(
            config.captions / "segments.json",
            {"segments": [{"segment": 0, "offsetFrame": 1, "frames": 7}]},
        )
    write_json(path, data)
    with pytest.raises(ReplayError):
        audit.load_parts(config, audit.Inputs())


def test_stale_historical_metrics_remain_missing(config: Config) -> None:
    metric = config.path.parent / "batch-000000.response.json"
    metric.write_bytes(b"{}")
    inputs = audit.Inputs()
    parts, _ = audit.load_parts(config, inputs)
    assert len(inputs.missing) == 2
    assert parts[0].envelopes == ()


def test_constraint_document(
    config: Config, part: audit.Part, constraints: list[dict[str, object]]
) -> None:
    path = write_json(
        config.work / "constraints.json",
        {
            "version": 1,
            "kind": "delivery-caption-constraints",
            "parts": {
                part.name: {
                    "plan": ref(part.plan),
                    "inputs": [ref(config.calibration)],
                    "states": constraints,
                }
            },
        },
    )
    assert len(audit.constraints_for(path, part, audit.Inputs())) == 2
    assert audit.constraints_for(None, part, audit.Inputs()) == []
    assert audit.constraints_for(path, replace(part, name="unknown"), audit.Inputs()) == []
    data = json.loads(path.read_text(encoding="utf-8"))
    data["parts"][part.name]["inputs"] = []
    write_json(path, data)
    with pytest.raises(ReplayError, match="provenance"):
        audit.constraints_for(path, part, audit.Inputs())


def test_completed_job_identity(config: Config, part: audit.Part) -> None:
    job = Job(
        part.name,
        part.source,
        config.alpha_preset,
        config.work / "alpha/segment-0000-part-0000.mov",
    )
    result = audit.completed(job, config.work / "alpha-jobs-state", audit.Inputs())
    assert result[1] == "job-1"
    assert (
        audit.historical_binding(
            None, job=job, part=part, movie=result[0], job_id=result[1], inputs=audit.Inputs()
        )
        is False
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "Running"),
        ("jobID", "bad id"),
        ("bytes", 1),
        ("output", "/unknown/movie.mov"),
        ("requested_output", "/elsewhere/other.mov"),
    ],
)
def test_completed_rejects_changed_receipt(
    config: Config, part: audit.Part, key: str, value: object
) -> None:
    state = config.work / "alpha-jobs-state/alpha-0000-0000.json"
    data = json.loads(state.read_text(encoding="utf-8"))
    data[key] = value
    write_json(state, data)
    job = Job(
        part.name,
        part.source,
        config.alpha_preset,
        config.work / "alpha/segment-0000-part-0000.mov",
    )
    with pytest.raises(ReplayError):
        audit.completed(job, state.parent, audit.Inputs())


@pytest.fixture
def producer_state(config: Config, part: audit.Part) -> tuple[Job, Path]:
    job = Job(part.name, part.source, config.alpha_preset, config.work / "alpha/new-alpha.mov")
    state = config.work / "new-state"
    state.mkdir()

    def runner(command: Sequence[str]) -> str:
        if "-monitor" in command:
            return '[{"jobStatus":{"jobID":"job-1","status":"Successful"}}]'
        assert "-jobpath" in command
        return '{"batch":{"jobs":[{"jobID":"job-1"}]}}'

    batch = Batch(state, config.native, "synthetic", runner, lambda _: None)
    job_id = batch.submit(job)
    job.output.write_bytes(b"synthetic alpha; no native job")
    batch.snapshot()
    batch.finish(job, job_id)
    return job, state / f"{job.name}.json"


def test_same_state_producer_binding(producer_state: tuple[Job, Path]) -> None:
    job, path = producer_state
    inputs = audit.Inputs()
    assert (
        audit.state_binding(path, job=job, movie=job.output, job_id="job-1", inputs=inputs) is True
    )
    inputs.verify()
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="changed during audit"):
        inputs.verify()


@pytest.mark.parametrize(
    "key,value",
    [
        ("terminalInputsVerified", False),
        ("jobID", "job-other"),
        ("status", "Failed"),
        ("submitResponse", '{"batch":{"jobs":[{"jobID":"other"}]}}'),
        ("jobConfigSHA256", "0" * 64),
        ("inputs", []),
        ("monitorResponse", {}),
        ("requested_output", "/unrelated/output.mov"),
    ],
)
def test_same_state_rejects_changed_fields(
    producer_state: tuple[Job, Path], key: str, value: object
) -> None:
    job, path = producer_state
    data = json.loads(path.read_text(encoding="utf-8"))
    data[key] = value
    write_json(path, data)
    with pytest.raises(ReplayError):
        audit.state_binding(path, job=job, movie=job.output, job_id="job-1", inputs=audit.Inputs())


def test_same_state_wrong_input_sha(producer_state: tuple[Job, Path]) -> None:
    job, path = producer_state
    data = json.loads(path.read_text(encoding="utf-8"))
    data["inputs"][0]["sha256"] = "0" * 64
    write_json(path, data)
    with pytest.raises(ReplayError, match="SHA256"):
        audit.state_binding(path, job=job, movie=job.output, job_id="job-1", inputs=audit.Inputs())


def test_same_state_changed_source(producer_state: tuple[Job, Path]) -> None:
    job, path = producer_state
    job.source.write_bytes(b"changed source")
    with pytest.raises(ReplayError, match="input changed"):
        audit.state_binding(path, job=job, movie=job.output, job_id="job-1", inputs=audit.Inputs())


def test_same_state_failed_raw_monitor(producer_state: tuple[Job, Path]) -> None:
    job, path = producer_state
    data = json.loads(path.read_text(encoding="utf-8"))
    monitor = write_json(
        path.parent / "failure.json", [{"jobStatus": {"jobID": "job-1", "status": "Failed"}}]
    )
    data["monitorResponse"] = ref(monitor)
    write_json(path, data)
    with pytest.raises(ReplayError, match="contradict"):
        audit.state_binding(path, job=job, movie=job.output, job_id="job-1", inputs=audit.Inputs())


def test_same_state_legacy_is_not_upgraded(config: Config, part: audit.Part) -> None:
    job = Job(
        part.name,
        part.source,
        config.alpha_preset,
        config.work / "alpha/segment-0000-part-0000.mov",
    )
    path = config.work / "alpha-jobs-state" / f"{part.name}.json"
    original = path.read_bytes()
    assert (
        audit.state_binding(path, job=job, movie=job.output, job_id="job-1", inputs=audit.Inputs())
        is False
    )
    assert path.read_bytes() == original


def test_truthful_history(config: Config, part: audit.Part) -> None:
    job = Job(
        part.name,
        part.source,
        config.alpha_preset,
        config.work / "alpha/segment-0000-part-0000.mov",
    )
    submit = write_json(config.work / "submit.json", {"batch": {"jobs": [{"jobID": "job-1"}]}})
    monitor = write_json(
        config.work / "monitor.json", [{"jobStatus": {"jobID": "job-1", "status": "Successful"}}]
    )
    entry: dict[str, object] = {
        "fingerprintMoment": "before-submit",
        "jobID": "job-1",
        "source": ref(job.source),
        "preset": ref(job.preset),
        "output": ref(job.output),
        "plan": ref(part.plan),
        "submitResponse": ref(submit),
        "monitorResponse": ref(monitor),
        "command": [
            "Compressor",
            "-jobpath",
            str(job.source),
            "-settingpath",
            str(job.preset),
            "-locationpath",
            str(job.output),
        ],
    }
    path = write_json(
        config.work / "history.json",
        {"version": 1, "kind": "compressor-direct-cli-history", "jobs": {job.name: entry}},
    )
    assert (
        audit.historical_binding(
            path, job=job, part=part, movie=job.output, job_id="job-1", inputs=audit.Inputs()
        )
        is True
    )
    entry["fingerprintMoment"] = "audit-now"
    write_json(
        path, {"version": 1, "kind": "compressor-direct-cli-history", "jobs": {job.name: entry}}
    )
    with pytest.raises(ReplayError, match="submission-time"):
        audit.historical_binding(
            path, job=job, part=part, movie=job.output, job_id="job-1", inputs=audit.Inputs()
        )


def test_transcript_cut_projection(config: Config, part: audit.Part) -> None:
    source = write_json(
        config.work / "source-tokens.json",
        {
            "clock": "source-presentation",
            "tokens": [
                {"text": "removed", "start": 0.0, "end": 0.5},
                {"text": "あ", "start": 31 / 30, "end": 33 / 30},
                {"text": "あい", "start": 34 / 30, "end": 36 / 30},
                {"text": "removed", "start": 2, "end": 3},
            ],
        },
    )
    path = write_json(
        config.work / "transcript.json",
        {
            "version": 1,
            "kind": "caption-source-tokens",
            "tokens": ref(source),
            "cuts": ref(config.cuts),
            "inputs": [ref(config.path)],
        },
    )
    result = audit.transcript_check(path, config=config, parts=[part], inputs=audit.Inputs())
    assert result["verified"] is True
    assert result["sourceCharactersRetained"] == 3
    assert (
        audit.transcript_check(None, config=config, parts=[part], inputs=audit.Inputs())["verified"]
        is False
    )
    changed = replace(part, cues=(replace(part.cues[0], text="う"), part.cues[1]))
    assert (
        audit.transcript_check(path, config=config, parts=[changed], inputs=audit.Inputs())[
            "textConserved"
        ]
        is False
    )
    changed_time = replace(part, cues=(replace(part.cues[0], start=1), part.cues[1]))
    assert (
        audit.transcript_check(path, config=config, parts=[changed_time], inputs=audit.Inputs())[
            "orderedOnsetFramesEqual"
        ]
        is False
    )


def test_transcript_growth_and_cut_crossing(config: Config, part: audit.Part) -> None:
    source = write_json(
        config.work / "source-tokens.json",
        {
            "clock": "source-presentation",
            "tokens": [
                {"text": "あ", "start": 31 / 30, "end": 33 / 30},
                {"text": "い", "start": 33 / 30, "end": 36 / 30},
            ],
        },
    )
    path = write_json(
        config.work / "transcript.json",
        {
            "version": 1,
            "kind": "caption-source-tokens",
            "tokens": ref(source),
            "cuts": ref(config.cuts),
            "inputs": [ref(config.path)],
        },
    )
    contiguous = replace(part, cues=(part.cues[0], replace(part.cues[1], start=2)))
    assert (
        audit.transcript_check(path, config=config, parts=[contiguous], inputs=audit.Inputs())[
            "verified"
        ]
        is True
    )
    write_json(
        source,
        {"clock": "source-presentation", "tokens": [{"text": "あ", "start": 0.9, "end": 1.1}]},
    )
    write_json(
        path,
        {
            "version": 1,
            "kind": "caption-source-tokens",
            "tokens": ref(source),
            "cuts": ref(config.cuts),
            "inputs": [ref(config.path)],
        },
    )
    assert (
        audit.transcript_check(path, config=config, parts=[part], inputs=audit.Inputs())["reason"]
        == "source token crosses cut; allocation is unverified"
    )


@pytest.mark.parametrize("complete", [True, False])
def test_producer_state_to_caption_audit(
    config: Config,
    part: audit.Part,
    producer_state: tuple[Job, Path],
    measurement: dict[str, object],
    constraints: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
) -> None:
    job, path = producer_state
    (config.work / "alpha-jobs-state" / f"{job.name}.json").write_bytes(path.read_bytes())
    write_json(
        config.work / "alpha-jobs.json",
        {
            "jobs": [
                {
                    "name": job.name,
                    "source": str(job.source),
                    "preset": str(job.preset),
                    "output": str(job.output),
                }
            ]
        },
    )
    constraint_path = write_json(
        config.work / "constraints.json",
        {
            "version": 1,
            "kind": "delivery-caption-constraints",
            "parts": {
                part.name: {
                    "plan": ref(part.plan),
                    "inputs": [ref(config.calibration)],
                    "states": constraints if complete else constraints[:1],
                }
            },
        },
    )
    srt = config.work / "candidate.srt"
    srt.write_text(audit.canonical_srt([part], config.rate), encoding="utf-8")
    monkeypatch.setattr(audit, "verify_motion_source", lambda *args: None)
    monkeypatch.setattr(audit, "cached_measurement", lambda *args, **kwargs: measurement)
    result = audit.audit(
        config.path,
        srt,
        config.work / "audit-state",
        phase="audit",
        cache=config.work / "cache",
        ffmpeg=config.path,
        ffprobe=config.path,
        constraints=constraint_path,
    )
    assert record(result["checks"])["historicalSubmissionBound"] is True
    assert result["alphaOnlyAccepted"] is complete
    assert result["wholeDeliveryAccepted"] is False
    assert result["accepted"] is False


def test_retained_labels_projection(config: Config, part: audit.Part) -> None:
    labels = write_json(
        config.work / "labels.json",
        {
            "labels": [
                {"text": "あ", "start": 31 / 30, "end": 33 / 30, "provenance": "retained-asr"},
                {"text": "あい", "start": 34 / 30, "end": 36 / 30, "provenance": "retained-asr"},
            ]
        },
    )
    evidence = {
        "version": 1,
        "kind": "caption-source-labels",
        "labels": ref(labels),
        "cuts": ref(config.cuts),
        "inputs": [ref(config.source)],
    }
    path = write_json(config.work / "labels-evidence.json", evidence)
    assert (
        audit.transcript_check(path, config=config, parts=[part], inputs=audit.Inputs())["verified"]
        is True
    )
    write_json(labels, {"labels": [{"text": "あ", "start": 1, "end": 1}]})
    evidence["labels"] = ref(labels)
    write_json(path, evidence)
    with pytest.raises(ReplayError, match="zero-length"):
        audit.transcript_check(path, config=config, parts=[part], inputs=audit.Inputs())


@pytest.mark.parametrize("complete", [True, False])
def test_standalone_alpha_probe(
    config: Config,
    part: audit.Part,
    measurement: dict[str, object],
    constraints: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
) -> None:
    path = write_json(
        config.work / "probe-constraints.json",
        {
            "version": 1,
            "kind": "delivery-caption-constraints",
            "parts": {
                part.name: {
                    "plan": ref(part.plan),
                    "inputs": [ref(config.calibration)],
                    "states": constraints if complete else constraints[:1],
                }
            },
        },
    )
    monkeypatch.setattr(audit, "cached_measurement", lambda *args, **kwargs: measurement)
    result = audit.probe_alpha(
        config.path,
        part.plan,
        config.source,
        config.work / "probe.json",
        name=part.name,
        constraints=path,
        cache=config.work / "probe-cache",
        ffmpeg=config.path,
        ffprobe=config.path,
    )
    assert result["alphaOnlyAccepted"] is complete
    assert result["wholeDeliveryAccepted"] is False
    assert result["accepted"] is False
    assert audit.main(
        [
            "probe-alpha",
            str(config.path),
            str(config.source),
            str(config.work / "probe-cli.json"),
            "--plan",
            str(part.plan),
            "--name",
            part.name,
            "--constraints",
            str(path),
            "--cache",
            str(config.work / "probe-cache"),
            "--ffmpeg",
            str(config.path),
            "--ffprobe",
            str(config.path),
        ]
    ) == (0 if complete else 1)
    assert (
        audit.main(
            [
                "probe-alpha",
                str(config.path),
                str(config.source),
                str(config.work / "missing.json"),
                "--cache",
                str(config.work / "probe-cache"),
            ]
        )
        == 2
    )


@pytest.fixture
def composition_fixture(
    config: Config,
    part: audit.Part,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, list[dict[str, object]]]:
    alpha = config.work / "alpha/segment-0000-part-0000.mov"
    final = config.work / "final.mov"
    final.write_bytes(b"synthetic final composition")
    request = write_json(
        config.work / "part/request.json",
        {
            "recipe": {
                "additionalVideo": [
                    {
                        "sourcePath": str(alpha),
                        "selection": {"startSeconds": 0.0, "durationSeconds": 5 / 30, "rate": 1},
                        "offsetSeconds": 1 / 30,
                        "opacity": 1,
                    }
                ]
            }
        },
    )
    response = write_json(request.with_name("render.response.json"), {"outputPath": str(final)})
    native = {
        "actualNativeReceipt": str(response),
        "receiptSHA256": ref(response)["sha256"],
        "movie": str(final),
        "movieSHA256Recorded": ref(final)["sha256"],
        "movieSHA256AtAudit": ref(final)["sha256"],
    }
    bindings = [
        audit.Inputs().bind(p) for p in (config.path, config.cuts, final, alpha, request, response)
    ]
    path = write_json(
        config.work / "cut-execution.json",
        {
            "version": 1,
            "kind": "delivery-cut-execution",
            "accepted": True,
            "selectionPlanMatchesProtectedCuts": True,
            "recordedExecutionMatchesPlan": True,
            "checks": {"completedNativeExecution": True},
            "inputs": bindings,
            "inputsAfter": bindings,
            "actualNativeReceipts": [native],
            "selectedFrames": 7,
        },
    )
    monkeypatch.setattr(audit, "rendered_parts", lambda _: [(request, 7)])
    return path, final, [{"alpha": ref(alpha)}]


def test_actual_composition_binding(
    config: Config,
    part: audit.Part,
    composition_fixture: tuple[Path, Path, list[dict[str, object]]],
) -> None:
    path, final, results = composition_fixture
    assert (
        audit.composition_check(
            path, final, config=config, parts=[part], results=results, inputs=audit.Inputs()
        )["verified"]
        is True
    )
    final.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="fingerprint"):
        audit.composition_check(
            path, final, config=config, parts=[part], results=results, inputs=audit.Inputs()
        )


@pytest.mark.parametrize(
    "change",
    [
        "rejected",
        "missing-input",
        "missing-alpha",
        "missing-native",
        "native-id",
        "placement",
        "clock",
    ],
)
def test_composition_rejects_incomplete_or_changed(
    config: Config,
    part: audit.Part,
    composition_fixture: tuple[Path, Path, list[dict[str, object]]],
    change: str,
) -> None:
    path, final, results = composition_fixture
    data = json.loads(path.read_text(encoding="utf-8"))
    if change == "rejected":
        data["accepted"] = False
    elif change == "missing-input":
        data["inputs"] = data["inputs"][1:]
    elif change == "missing-alpha":
        results = []
    elif change == "missing-native":
        data["actualNativeReceipts"] = []
    elif change == "native-id":
        data["actualNativeReceipts"][0]["receiptSHA256"] = "0" * 64
    elif change == "placement":
        part = replace(part, offset=2)
    else:
        data["selectedFrames"] = 8
    data["inputsAfter"] = data["inputs"]
    write_json(path, data)
    with pytest.raises(ReplayError):
        audit.composition_check(
            path, final, config=config, parts=[part], results=results, inputs=audit.Inputs()
        )


def test_canonical_phase_keeps_old_srt(config: Config, tmp_path: Path) -> None:
    old = tmp_path / "old.srt"
    old.write_text("old centisecond/float candidate", encoding="utf-8")
    result = audit.audit(
        config.path,
        old,
        tmp_path / "new",
        phase="canonical-srt",
        cache=tmp_path / "cache",
        ffmpeg=config.path,
        ffprobe=config.path,
    )
    assert result["accepted"] is False
    checks = record(result["checks"])
    assert checks["canonicalSRTFrameEquivalent"] is True
    assert checks["candidateSRTExactMatch"] is False
    assert old.read_text(encoding="utf-8") == "old centisecond/float candidate"
    assert (tmp_path / "new/canonical.srt").exists()
    with pytest.raises(FileExistsError):
        audit.audit(
            config.path,
            old,
            tmp_path / "new",
            phase="canonical-srt",
            cache=tmp_path / "cache",
            ffmpeg=config.path,
            ffprobe=config.path,
        )


def test_audit_reuses_measurements_but_never_approves_partial(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measurement: dict[str, object]
) -> None:
    def measure(*args: object, **kwargs: object) -> dict[str, object]:
        return measurement

    monkeypatch.setattr(audit, "measure_alpha", measure)
    monkeypatch.setattr(audit, "verify_motion_source", lambda *args: None)
    old = tmp_path / "old.srt"
    old.write_text("old", encoding="utf-8")
    result = audit.audit(
        config.path,
        old,
        tmp_path / "first",
        phase="audit",
        cache=tmp_path / "cache",
        ffmpeg=config.path,
        ffprobe=config.path,
    )
    assert result["accepted"] is False
    assert record(result["checks"])["fullAlphaCoverage"] is True

    def no_decode(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("unchanged alpha must reuse raster cache")

    monkeypatch.setattr(audit, "measure_alpha", no_decode)
    again = audit.audit(
        config.path,
        old,
        tmp_path / "second",
        phase="audit",
        cache=tmp_path / "cache",
        ffmpeg=config.path,
        ffprobe=config.path,
    )
    assert again["accepted"] is False
    partial = audit.audit(
        config.path,
        old,
        tmp_path / "partial",
        phase="audit",
        cache=tmp_path / "cache",
        ffmpeg=config.path,
        ffprobe=config.path,
        start=1,
        limit=1,
    )
    assert record(partial["checks"])["fullAlphaCoverage"] is False
    assert partial["accepted"] is False


def test_same_size_alpha_change_invalidates_raster_cache(
    config: Config,
    part: audit.Part,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    measurement: dict[str, object],
) -> None:
    calls: list[int] = []

    def measure(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(1)
        return measurement

    monkeypatch.setattr(audit, "measure_alpha", measure)
    movie = config.work / "alpha/segment-0000-part-0000.mov"
    audit.cached_measurement(
        movie,
        config=config,
        part=part,
        cache=tmp_path / "cache",
        inputs=audit.Inputs(),
        ffmpeg=config.path,
        ffprobe=config.path,
        timeout=10,
    )
    movie.write_bytes(b"x" * movie.stat().st_size)
    audit.cached_measurement(
        movie,
        config=config,
        part=part,
        cache=tmp_path / "cache",
        inputs=audit.Inputs(),
        ffmpeg=config.path,
        ffprobe=config.path,
        timeout=10,
    )
    assert len(calls) == 2


def test_missing_alpha_is_receipt_not_render(config: Config, tmp_path: Path) -> None:
    old = tmp_path / "old.srt"
    old.write_text("old", encoding="utf-8")
    state = config.work / "alpha-jobs-state/alpha-0000-0000.json"
    write_json(state, {"status": "Pending", "jobID": "job-1"})
    result = audit.audit(
        config.path,
        old,
        tmp_path / "failed",
        phase="audit",
        cache=tmp_path / "cache",
        ffmpeg=config.path,
        ffprobe=config.path,
    )
    assert result["accepted"] is False
    assert record(result["checks"])["fullAlphaCoverage"] is False


def test_cli_exit_codes(config: Config, tmp_path: Path) -> None:
    old = tmp_path / "old.srt"
    old.write_text("old", encoding="utf-8")
    assert (
        audit.main(
            [
                "canonical-srt",
                str(config.path),
                str(old),
                str(tmp_path / "cli"),
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    assert (
        audit.main(
            [
                "audit",
                str(config.path),
                str(old),
                str(tmp_path / "partial"),
                "--cache",
                str(tmp_path / "cache"),
                "--start",
                "1",
            ]
        )
        == 1
    )
    assert (
        audit.main(
            [
                "audit",
                str(config.path),
                str(old),
                str(tmp_path / "invalid"),
                "--cache",
                str(tmp_path / "cache"),
                "--limit",
                "0",
            ]
        )
        == 2
    )


def test_real_synthetic_alpha_clocks_decode(
    config: Config, part: audit.Part, tmp_path: Path
) -> None:
    executable = shutil.which("ffmpeg")
    probe = shutil.which("ffprobe")
    if executable is None or probe is None:
        pytest.fail("FFmpeg and ffprobe are required")
    rgba = np.zeros((5, 4, 4, 4), dtype=np.uint8)
    rgba[:2, 1:3, 1:3, :] = 255
    rgba[3:, 1:3, 1:3, :] = 255
    raw = tmp_path / "fixture.rgba"
    raw.write_bytes(rgba.tobytes())
    movie = tmp_path / "fixture.mov"
    env = dict(os.environ, TMPDIR=str(tmp_path), TMP=str(tmp_path), TEMP=str(tmp_path))
    env.pop("FFREPORT", None)
    subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            "4x4",
            "-framerate",
            "30",
            "-i",
            str(raw),
            "-c:v",
            "png",
            "-pix_fmt",
            "rgba",
            str(movie),
        ],
        check=True,
        timeout=20,
        env=env,
        capture_output=True,
    )
    directory = tmp_path / "measure"
    directory.mkdir()
    result = audit.measure_alpha(
        movie,
        config=config,
        part=part,
        directory=directory,
        ffmpeg=Path(executable).resolve(),
        ffprobe=Path(probe).resolve(),
        timeout=20,
    )
    assert record(result["clock"]) == {
        "frames": 5,
        "start": "0",
        "end": "1/6",
        "exactCFR": True,
    }
    assert result["rasterRuns"] == [
        {"startFrame": 0, "endFrame": 2, "bounds": [1, 1, 3, 3]},
        {"startFrame": 2, "endFrame": 3, "bounds": None},
        {"startFrame": 3, "endFrame": 5, "bounds": [1, 1, 3, 3]},
    ]
    assert (directory / "alpha.process.json").exists()
    bad = tmp_path / "wrong-clock"
    bad.mkdir()
    with pytest.raises(ReplayError, match="duration/endpoint"):
        audit.measure_alpha(
            movie,
            config=replace(config, rate=25),
            part=part,
            directory=bad,
            ffmpeg=Path(executable).resolve(),
            ffprobe=Path(probe).resolve(),
            timeout=20,
        )


@pytest.mark.parametrize(
    "body,match",
    [
        ("pass", "truncated"),
        ("import sys; sys.stdout.buffer.write(bytes(192))", "Extra"),
        ("import sys; sys.stderr.write('error')", "truncated"),
        ("import time; time.sleep(5)", "deadline"),
    ],
)
def test_decoder_failure_retains_process(
    config: Config, part: audit.Part, tmp_path: Path, body: str, match: str
) -> None:
    script = tmp_path / "decoder"
    script.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    script.chmod(0o700)
    directory = tmp_path / "decode"
    directory.mkdir()
    with pytest.raises(ReplayError, match=match):
        audit.decode_alpha(
            config.path, tool=script, config=config, part=part, directory=directory, timeout=1
        )
    assert (directory / "alpha.process.json").exists()


def test_actual_motion_xml_plan_style_binding(
    config: Config, part: audit.Part, sequence_template: str, stage_plan: dict[str, object]
) -> None:
    template = config.work / "template.moti"
    template.write_text(sequence_template, encoding="utf-8")
    write_json(part.plan, stage_plan)
    style = motion_finish.selected_finish_style(
        top=(0.1, 0.2, 0.3),
        bottom=(0.3, 0.4, 0.5),
        font_name="NotoSansJP-Thin_Bold",
        font_size=40.0,
        stroke_scale=0.5,
        shadow_scale=1.0,
    )
    generated = motion_finish.finish_sequence(
        motion_stages.prepare(sequence_template, stage_plan), style=style
    )
    part.source.write_text(generated, encoding="utf-8")
    configured = replace(
        config,
        template=template,
        finish=(
            "--top",
            "0.1",
            "0.2",
            "0.3",
            "--bottom",
            "0.3",
            "0.4",
            "0.5",
            "--font-name",
            "NotoSansJP-Thin_Bold",
            "--font-size",
            "40",
            "--stroke-scale",
            "0.5",
            "--shadow-scale",
            "1",
        ),
    )
    audit.verify_motion_source(configured, part, audit.Inputs())
    part.source.write_text(generated + "\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="Actual Motion source differs"):
        audit.verify_motion_source(configured, part, audit.Inputs())


@pytest.mark.parametrize(
    "finish",
    [(), ("--top", "1"), ("--override", "1"), ("--top", "1", "1", "1", "--top", "1", "1", "1")],
)
def test_source_style_not_guessed(
    config: Config, part: audit.Part, finish: tuple[str, ...]
) -> None:
    with pytest.raises(ReplayError, match="finish arguments"):
        audit.verify_motion_source(replace(config, finish=finish), part, audit.Inputs())
