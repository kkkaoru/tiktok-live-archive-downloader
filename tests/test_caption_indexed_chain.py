"""Derive ordinary caption workflows from actual completed native indices."""

import fcntl
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import caption_indexed_chain as chain
from replay.caption_style_defaults import MAX_CAPTION_LINES
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import Stage, _fingerprint, file_digest, run_stage
from replay.workflow import load_stages


def write(path: Path, value: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": file_digest(path)}


@dataclass(frozen=True)
class Case:
    request: Path
    output: Path
    directory: Path
    cache: Path
    source: Path
    native: Path


@pytest.fixture
def case(tmp_path: Path) -> Case:
    directory, cache = tmp_path / "recognition", tmp_path / "cache"
    directory.mkdir()
    cache.mkdir()
    source, native = tmp_path / "source", tmp_path / "native"
    source.write_bytes(b"synthetic source")
    native.write_bytes(b"synthetic native")
    rows: list[dict[str, object]] = []
    cursor = 0
    for n, frames in enumerate((30, 60, 90)):
        tokens = directory / f"tokens-{n:04d}.json"
        token_ref = write(tokens, {"version": 1, "clock": "output", "tokens": []})
        binding = write(
            directory / f"recognition-{n:04d}.json",
            {
                "version": 1,
                "kind": "recognition-inputs",
                "segment": n,
                "jobs": [n],
                "measurementsPerformed": False,
                "inputs": {str(source): file_digest(source), str(tokens): token_ref["sha256"]},
            },
        )
        rows.append(
            {
                "segment": n,
                "offsetFrame": cursor,
                "frames": frames,
                "jobs": [n],
                "recognitionInput": binding,
            }
        )
        cursor += frames
    index = write(directory / "segments.json", {"version": 1, "rate": 30, "segments": rows})
    write(
        directory / "receipt.json",
        {"segments": index, "inputs": {str(source): file_digest(source)}},
    )
    for n in (0, 2):
        write(
            cache / f"part-{n:04d}.json",
            {
                "inputs": {
                    str(tmp_path / "old.request.json"): "a" * 64,
                    "historical-reader.py": "b" * 64,
                }
            },
        )
    request = tmp_path / "request.json"
    write(
        request,
        {
            "version": 1,
            "kind": chain.KIND,
            "recognitionDirectory": str(directory),
            "source": {"path": str(source), "sha256": file_digest(source)},
            "pagingCommandTemplate": [
                sys.executable,
                "-B",
                "-m",
                "replay.caption_native_paging",
                "{recognition}",
                "{output}",
                str(native),
                "--source-media",
                str(source),
                "--evidence",
                str(native),
                "--long-executable",
                str(native),
            ],
            "selectionGroups": [[0, 1], [2]],
            "cacheRoot": str(cache),
            "cacheSegments": [0, 2],
            "inputPins": {str(native): file_digest(native)},
        },
    )
    return Case(request, tmp_path / "child", directory, cache, source, native)


def data(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text()))


def mutate(case: Case, field: str, value: object) -> None:
    request = data(case.request)
    request[field] = value
    write(case.request, request)


def materialize(case: Case, stage: Stage) -> None:
    parts = stage.outputs[1:]
    span = int(stage.name.rsplit("-", 1)[1])
    plan = chain.verify_plan(case.request, case.output)
    start, frames = plan.spans[span]
    for part in parts:
        write(part, {"synthetic": True})
    write(
        stage.outputs[0],
        {
            "version": 1,
            "candidateOnly": True,
            "inputs": {str(case.source): file_digest(case.source)},
            "coverage": {"start": start / 30, "end": (start + frames) / 30},
            "captions": [{"text": "synthetic", "start": start / 30, "end": (start + frames) / 30}],
            "layout": {
                "version": 1,
                "scope": "horizontal-phrase-layout-only",
                "recognitionClocksChanged": False,
                "originsRequireRetargeting": True,
                "maximumLines": MAX_CAPTION_LINES,
                "pointSize": 80,
                "paragraphWidth": 972,
                "widthMeasurement": "native-point-size-checked-glyph-bounds-including-border-rim",
                "boundaries": "pinned-vibrato-ipadic-phrase-boundaries",
            },
        },
    )


def complete(case: Case, state: Path) -> None:
    for stage in load_stages(case.output / "child-workflow.json"):

        def execute(
            command: tuple[str, ...], log: Path, timeout: int, stage: Stage = stage
        ) -> None:
            if "verify-span" in command:
                chain.verify_span(case.request, case.output, int(command[-1]))
            else:
                materialize(case, stage)
            log.write_text("synthetic normal stage\n")

        assert run_stage(stage, state_directory=state, execute=execute) == "completed"


def test_normal_current_child_and_cli(case: Case, tmp_path: Path) -> None:
    chain.main(["prepare", str(case.request), str(case.output)])
    chain.main(["verify-plan", str(case.request), str(case.output)])
    state = tmp_path / "state"
    complete(case, state)
    chain.main(["verify-span", str(case.request), str(case.output), "--span", "0"])
    chain.main(["verify-child", str(case.request), str(case.output), "--state", str(state)])
    plan = chain.verify_plan(case.request, case.output)
    assert plan.spans == ((0, 90), (90, 90))
    assert plan.receipt["fullSRTVerified"] is False
    with pytest.raises(ValueError, match="Occupied"):
        chain.prepare(case.request, case.output)
    for args in (["verify-span"], ["verify-child"]):
        with pytest.raises(SystemExit):
            chain.main([*args, str(case.request), str(case.output)])
    with pytest.raises(ValueError, match="Unknown indexed span"):
        chain.verify_span(case.request, case.output, True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("kind", "wrong"),
        ("recognitionDirectory", "relative"),
        ("source", {}),
        ("inputPins", {}),
        ("inputPins", {"relative": "a" * 64}),
        ("inputPins", {"/missing": "bad"}),
        ("selectionGroups", []),
        ("selectionGroups", [[0, 2]]),
        ("selectionGroups", [[0], [0]]),
        ("selectionGroups", [[3]]),
        ("selectionGroups", [[True]]),
        ("cacheSegments", [0, 0]),
        ("cacheSegments", [3]),
        ("pagingCommandTemplate", []),
        ("pagingCommandTemplate", [True]),
        ("pagingCommandTemplate", ["{recognition}", "{output}"]),
        ("pagingCommandTemplate", ["no placeholders"]),
    ],
)
def test_invalid_requests(case: Case, field: str, value: object) -> None:
    mutate(case, field, value)
    with pytest.raises((ValueError, ReplayError)):
        chain.prepare(case.request, case.output)
    assert not case.output.exists()


@pytest.mark.parametrize(
    "change", ["source", "native", "tokens", "cache", "index", "workflow", "receipt"]
)
def test_changed_inputs_or_prepared_artifacts(case: Case, change: str) -> None:
    chain.prepare(case.request, case.output)
    paths = {
        "source": case.source,
        "native": case.native,
        "tokens": case.directory / "tokens-0000.json",
        "cache": case.cache / "part-0000.json",
        "index": case.directory / "segments.json",
        "workflow": case.output / "child-workflow.json",
        "receipt": case.output / "prepared.json",
    }
    paths[change].write_bytes(b"{}")
    with pytest.raises((ValueError, ReplayError, OSError)):
        chain.verify_plan(case.request, case.output)


def test_empty_tokens_and_explicit_exclusion(case: Case) -> None:
    mutate(case, "selectionGroups", [[0], [2]])
    mutate(case, "cacheSegments", [])
    chain.prepare(case.request, case.output)
    assert chain.verify_plan(case.request, case.output).receipt["excludedSegments"] == [1]
    stages = load_stages(case.output / "child-workflow.json")
    assert "--measurement-cache" not in stages[0].command
    materialize(case, stages[0])
    chain.verify_span(case.request, case.output, 0)


def test_failure_cache_requires_real_requests(case: Case) -> None:
    (case.cache / "part-0000.json").rename(case.cache / "failure-0000.json")
    chain.prepare(case.request, case.output)
    write(case.cache / "failure-0000.json", {"inputs": {}})
    with pytest.raises(ValueError, match="no pinned glyph"):
        chain.verify_plan(case.request, case.output)


def test_actual_variable_index_clocks(case: Case) -> None:
    document = data(case.directory / "segments.json")
    rows = document["segments"]
    assert isinstance(rows, list)
    cursor = 0
    for row, frames in zip(rows, (17, 33, 20), strict=True):
        member = record(row)
        member["offsetFrame"], member["frames"] = cursor, frames
        cursor += frames
    reference = write(case.directory / "segments.json", document)
    receipt = data(case.directory / "receipt.json")
    receipt["segments"] = reference
    write(case.directory / "receipt.json", receipt)
    chain.prepare(case.request, case.output)
    assert chain.verify_plan(case.request, case.output).spans == ((0, 50), (50, 20))


@pytest.mark.parametrize(("origin", "difference"), [(5549, 0), (0, -3), (17, 2), (0, -2)])
def test_preserves_origins_and_partitions_packed_boundaries(
    case: Case, origin: int, difference: int
) -> None:
    index = data(case.directory / "segments.json")
    rows = index["segments"]
    assert isinstance(rows, list)
    offsets = (origin, origin + 30 + difference, origin + 90 + difference)
    for value, offset in zip(rows, offsets, strict=True):
        record(value)["offsetFrame"] = offset
    reference = write(case.directory / "segments.json", index)
    receipt = data(case.directory / "receipt.json")
    receipt["segments"] = reference
    write(case.directory / "receipt.json", receipt)
    mutate(case, "selectionGroups", [[0, 1, 2]])
    chain.prepare(case.request, case.output)
    plan = chain.verify_plan(case.request, case.output)
    expected = (
        ((origin, 180),) if difference == 0 else ((origin, 30), (origin + 30 + difference, 150))
    )
    assert plan.spans == expected
    assert plan.receipt["derivedGroups"] == ([[0, 1, 2]] if difference == 0 else [[0], [1, 2]])
    assert plan.receipt["excludedSegments"] == []
    for span, stage in enumerate(load_stages(case.output / "child-workflow.json")):
        materialize(case, stage)
        chain.verify_span(case.request, case.output, span)


def test_sixty_three_rows_derive_seven_exact_runs(case: Case) -> None:
    rows: list[dict[str, object]] = []
    cursor = 0
    for number in range(63):
        cursor -= {4: 3, 5: 2, 9: 2, 10: 2}.get(number, 0)
        token = case.directory / f"tokens-{number:04d}.json"
        write(token, {"version": 1, "clock": "output", "tokens": []})
        ref = write(
            case.directory / f"recognition-{number:04d}.json",
            {
                "version": 1,
                "kind": "recognition-inputs",
                "segment": number,
                "jobs": [number],
                "measurementsPerformed": False,
                "inputs": {
                    str(case.source): file_digest(case.source),
                    str(token): file_digest(token),
                },
            },
        )
        rows.append(
            {
                "segment": number,
                "offsetFrame": cursor,
                "frames": 30,
                "jobs": [number],
                "recognitionInput": ref,
            }
        )
        cursor += 30
    ref = write(case.directory / "segments.json", {"version": 1, "rate": 30, "segments": rows})
    receipt = data(case.directory / "receipt.json")
    receipt["segments"] = ref
    write(case.directory / "receipt.json", receipt)
    mutate(case, "selectionGroups", [list(range(12)), [13, 14, 15], list(range(17, 63))])
    chain.prepare(case.request, case.output)
    plan = chain.verify_plan(case.request, case.output)
    assert plan.receipt["derivedGroups"] == [
        list(range(4)),
        [4],
        list(range(5, 9)),
        [9],
        [10, 11],
        [13, 14, 15],
        list(range(17, 63)),
    ]
    assert plan.receipt["excludedSegments"] == [12, 16]
    assert (
        sum(len(stage.outputs) - 1 for stage in load_stages(case.output / "child-workflow.json"))
        == 61
    )


@pytest.mark.parametrize("fault", ["selection", "source", "executable", "evidence", "position"])
def test_command_contract_rejections(case: Case, fault: str) -> None:
    request = data(case.request)
    command = request["pagingCommandTemplate"]
    assert isinstance(command, list)
    if fault == "selection":
        command.extend(["--segment", "0"])
    elif fault == "source":
        command[command.index("--source-media") + 1] = str(case.native)
    elif fault == "executable":
        command[command.index("replay.caption_native_paging") + 3] = str(case.source)
    elif fault == "evidence":
        command[command.index("--evidence") + 1] = str(case.source)
    else:
        first, last = command.index("{recognition}"), command.index("{output}")
        command[first], command[last] = command[last], command[first]
    write(case.request, request)
    with pytest.raises(ValueError):
        chain.prepare(case.request, case.output)


def test_partial_child_failed_receipt_and_held_lease(case: Case, tmp_path: Path) -> None:
    chain.prepare(case.request, case.output)
    stages = load_stages(case.output / "child-workflow.json")
    state = tmp_path / "state"
    state.mkdir()
    materialize(case, stages[0])
    write(
        state / f"{stages[0].name}.json",
        {"status": "running", "fingerprint": _fingerprint(stages[0])},
    )
    with pytest.raises(ValueError, match="genuine current completion"):
        chain.verify_child(case.request, case.output, state)
    assert stages[0].outputs[0].exists()
    for part in stages[0].outputs:
        part.unlink()
    for receipt in state.glob("*.json"):
        receipt.unlink()
    complete(case, state)
    with (state / ".workflow.lock").open("wb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active"):
            chain.verify_child(case.request, case.output, state)
    write(stages[1].outputs[0], {"broken": True})
    with pytest.raises(ValueError, match="genuine current completion"):
        chain.verify_child(case.request, case.output, state)
