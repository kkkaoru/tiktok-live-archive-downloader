"""Synthetic original multi-lane receipts; no media or native execution."""

import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import pytest

from replay import tracking_face_finalize as final
from replay import tracking_observations as storage
from replay.recordings import record
from replay.stages import Stage, _fingerprint, execution_lock, file_digest
from replay.subtitle_clock import VideoFrameClock


def test_full_clock_exceeds_window_budget_without_changing_window_guard() -> None:
    ticks = list(range(0, 180003, 3))
    clock = final._clock({"pts": ticks, "timeBase": "1/90", "lastDuration": 2})
    assert len(clock.pts) == 60001
    assert clock.pts == tuple(ticks)
    assert (clock.pts[-1] + clock.last_duration) * clock.time_base == Fraction(90001, 45)
    with pytest.raises(ValueError, match="presentation frame"):
        VideoFrameClock(clock.pts, clock.time_base, clock.last_duration)


@pytest.mark.parametrize(
    ("ticks", "base", "duration"),
    [
        ([-1, 0], "1/30", 1),
        ([0, 0], "1/30", 1),
        ([2, 1], "1/30", 1),
        ([0], "0/30", 1),
        ([0], "-1/30", 1),
        ([0], "1/1000000001", 1),
        ([0], "1/30", 0),
        ([0], "1/30", -1),
        ([0], "1/30", True),
        ([86400], "1/1", 1),
        ([0], "invalid", 1),
        ([0], "1/0", 1),
    ],
)
def test_full_clock_rejects_invalid_support(ticks: list[int], base: str, duration: int) -> None:
    with pytest.raises((ValueError, ZeroDivisionError)):
        final._clock({"pts": ticks, "timeBase": base, "lastDuration": duration})


def test_full_clock_frame_budget() -> None:
    with pytest.raises(ValueError, match="exceeds budget"):
        final._clock({"pts": [0] * 1000001, "timeBase": "1/30", "lastDuration": 1})


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def case(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, Path]:
    source, native = tmp_path / "source", tmp_path / "native"
    source.write_bytes(b"synthetic source")
    native.write_bytes(b"synthetic native")
    native.chmod(0o700)
    stages, rows = [], []
    for number in range(2):
        config, evidence, state = (
            tmp_path / f"window{number}.json",
            tmp_path / f"out{number}",
            tmp_path / f"state{number}",
        )
        evidence.mkdir()
        (evidence / "stderr.log").touch()
        state.mkdir()
        next_policy = getattr(request, "param", "reject") == "next"
        pts = [number, number + 1] if next_policy else [number * 2, number * 2 + 1]
        write(
            config,
            {
                "version": 1,
                "source": str(source),
                "sourceSHA256": file_digest(source),
                "nativeExecutable": str(native),
                "nativeSHA256": file_digest(native),
                "width": 16,
                "height": 16,
                "clock": {"pts": pts, "timeBase": "1/30", "lastDuration": 1},
            },
        )
        job = storage.load_job(config)
        write(evidence / "request.json", job.request())
        write(
            evidence / "response.json",
            [
                {
                    "presentationValue": point,
                    "presentationTimescale": 30,
                    "width": 16,
                    "height": 16,
                    "method": "detection"
                    if getattr(request, "param", "reject") == "detected"
                    else "missing",
                    "detectionAttempted": True,
                    "faces": [
                        {"confidence": 0.9, "region": {"x": 2, "y": 3, "width": 4, "height": 5}}
                    ]
                    if getattr(request, "param", "reject") == "detected"
                    else [],
                }
                for point in pts
            ],
        )
        write(
            evidence / "provenance.json",
            {
                "inputs": storage._bindings(job),
                "requestSHA256": file_digest(evidence / "request.json"),
                "responseSHA256": file_digest(evidence / "response.json"),
                "frames": 2,
                "nativeExitCode": 0,
                "captionPlacementApproved": False,
            },
        )
        stage = Stage(
            name=f"tracking-{number}",
            command=("synthetic-never-run", str(config)),
            inputs=(config, source, native),
            outputs=tuple(
                evidence / name for name in ("request.json", "response.json", "provenance.json")
            ),
            timeout_seconds=60,
        )
        write(
            state / f"{stage.name}.json",
            {
                "status": "complete",
                "fingerprint": _fingerprint(stage),
                "outputs": {str(path): file_digest(path) for path in stage.outputs},
            },
        )
        stages.append(
            {
                **asdict(stage),
                "inputs": [str(path) for path in stage.inputs],
                "outputs": [str(path) for path in stage.outputs],
            }
        )
        rows.append(
            {
                "index": number,
                "name": stage.name,
                "config": str(config),
                "evidence": str(evidence),
                "originalState": str(state),
                "originalCheckpoint": str(state / f"{stage.name}.json"),
            }
        )
    workflow, clock, index = (
        tmp_path / "workflow.json",
        tmp_path / "clock.json",
        tmp_path / "index.json",
    )
    write(workflow, {"version": 1, "stages": stages})
    write(
        clock,
        {"pts": [0, 1, 2] if next_policy else [0, 1, 2, 3], "timeBase": "1/30", "lastDuration": 1},
    )
    write(index, {"version": 1, "windows": rows})
    config = tmp_path / "finalize.json"
    write(
        config,
        {
            "version": 1,
            "seedOverlap": "next" if next_policy else "reject",
            **{
                name: {"path": str(path), "sha256": file_digest(path)}
                for name, path in (("workflow", workflow), ("clock", clock), ("index", index))
            },
        },
    )
    return config, tmp_path / "result.json"


def test_original_lanes_union_and_readonly(
    case: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = case
    result = final.finalize(config, output)
    assert result["frames"] == 4
    assert result["missing"] == 4
    assert result["methods"] == {"missing": 4}
    assert result["confidence"] == {"count": 0, "minimum": None, "maximum": None}
    assert result["geometryDerived"] is False
    before = output.read_bytes()
    final.finalize(config, output, verify=True)
    monkeypatch.setattr("sys.argv", ["final", "verify", str(config), str(output)])
    final.main()
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "path", ["out0/response.json", "state0/tracking-0.json", "source", "native"]
)
def test_tampered_original_refused(case: tuple[Path, Path], path: str) -> None:
    config, output = case
    (config.parent / path).write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        final.finalize(config, output)
    assert not output.exists()


def test_receipt_tamper_and_occupied(case: tuple[Path, Path]) -> None:
    config, output = case
    final.finalize(config, output)
    with pytest.raises(ValueError, match="occupied"):
        final.finalize(config, output)
    write(output, {})
    with pytest.raises(ValueError, match="receipt"):
        final.finalize(config, output, verify=True)


@pytest.mark.parametrize("field,value", [("version", True), ("seedOverlap", "hold"), ("index", {})])
def test_bad_config(case: tuple[Path, Path], field: str, value: object) -> None:
    config, output = case
    raw = record(final._json(config))
    raw[field] = value
    write(config, raw)
    with pytest.raises(ValueError):
        final.finalize(config, output)


@pytest.mark.parametrize("value", [None, "relative", "/tmp/out"])
def test_scope(value: object) -> None:
    with pytest.raises(ValueError):
        final._local(value)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"pts": [True], "timeBase": "1/30", "lastDuration": 1},
        {"pts": [], "timeBase": "1/30", "lastDuration": 1},
    ],
)
def test_bad_clock(value: object) -> None:
    with pytest.raises(ValueError):
        final._clock(value)


def test_actual_union_not_merely_frame_count(case: tuple[Path, Path]) -> None:
    config, output = case
    clock = config.parent / "clock.json"
    write(clock, {"pts": [0, 1, 2, 4], "timeBase": "1/30", "lastDuration": 1})
    raw = record(final._json(config))
    raw["clock"] = {"path": str(clock), "sha256": file_digest(clock)}
    write(config, raw)
    with pytest.raises(ValueError, match="union"):
        final.finalize(config, output)


@pytest.mark.parametrize("name", [".workflow", "tracking-0"])
def test_active_owner_blocks(case: tuple[Path, Path], name: str) -> None:
    config, output = case
    with execution_lock(config.parent / "state0", name), pytest.raises(ValueError, match="owned"):
        final.finalize(config, output)


def test_metadata_pin_changed(case: tuple[Path, Path]) -> None:
    config, output = case
    write(config.parent / "index.json", {})
    with pytest.raises(ValueError, match="pin"):
        final.finalize(config, output)


@pytest.mark.parametrize("case", ["next"], indirect=True)
def test_explicit_next_seed_overlap(case: tuple[Path, Path]) -> None:
    config, output = case
    result = final.finalize(config, output)
    assert result["frames"] == 3
    assert result["methods"] == {"missing": 3}
    assert result["seedOverlap"] == "next"


@pytest.mark.parametrize("case", ["detected"], indirect=True)
def test_confidence_is_observation_not_approval(case: tuple[Path, Path]) -> None:
    config, output = case
    result = final.finalize(config, output)
    assert result["methods"] == {"detection": 4}
    assert result["missing"] == 0
    assert result["confidence"] == {"count": 4, "minimum": 0.9, "maximum": 0.9}
    assert result["headNearVerified"] is False


@pytest.mark.parametrize("change", ["order", "checkpoint", "omit", "empty"])
def test_index_lineage_failures(case: tuple[Path, Path], change: str) -> None:
    config, output = case
    path = config.parent / "index.json"
    document = record(final._json(path))
    rows = document["windows"]
    assert isinstance(rows, list)
    if change == "order":
        record(rows[0])["index"] = 1
    elif change == "checkpoint":
        record(rows[0])["originalCheckpoint"] = str(config.parent / "wrong.json")
    elif change == "omit":
        rows.pop()
    else:
        rows.clear()
    write(path, document)
    raw = record(final._json(config))
    raw["index"] = {"path": str(path), "sha256": file_digest(path)}
    write(config, raw)
    with pytest.raises(ValueError):
        final.finalize(config, output)
