"""Synthetic ownership records; no real user fixtures or cleanup."""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from replay import pytest_lifecycle as module
from replay.native_sequence import ContractError
from replay.pytest_scratch import STATE, Scratch
from replay.sequence_journal import decode, encode


@pytest.fixture
def state(tmp_path: Path) -> module.Lifecycle:
    root = tmp_path / "checkout"
    work = root / "edits/work"
    evidence = work / "evidence"
    evidence.mkdir(parents=True)
    python = work / "python"
    python.mkdir()
    base = work / "base"
    locks = work / "pytest-default/locks"
    locks.mkdir(parents=True)
    (locks / hashlib.sha256(os.fsencode(base)).hexdigest()).write_bytes(b"")
    data = b"{}"
    (evidence / "run-start.json").write_bytes(data)
    return module.Lifecycle(Scratch(root, base, python), evidence, hashlib.sha256(data).hexdigest())


def test_registered_success_and_failed_fixture_remain_distinct(state: module.Lifecycle) -> None:
    first = state.scratch.python / "first"
    first.mkdir()
    module.register(state, "test::first", first)
    module.outcome(state, node="test::first", phase="setup", result="passed")
    module.outcome(state, node="test::first", phase="call", result="failed")
    module.outcome(state, node="test::first", phase="teardown", result="passed")
    result = module.load_finish(module.finish(state, 1))
    assert result["exitStatus"] == 1
    assert result["deletionPerformed"] is False
    assert result["processTerminationVerified"] is False
    assert first.is_dir()
    with pytest.raises(FileExistsError):
        module.finish(state, 1)


@pytest.mark.parametrize("phase,result", [("unknown", "passed"), ("call", "unknown")])
def test_invalid_report_refused(state: module.Lifecycle, phase: str, result: str) -> None:
    with pytest.raises(ContractError):
        module.outcome(state, node="a", phase=phase, result=result)


def test_repeated_phase_and_record_limit_refused(state: module.Lifecycle) -> None:
    module.outcome(state, node="a", phase="setup", result="passed")
    with pytest.raises(ContractError):
        module.outcome(state, node="a", phase="setup", result="passed")
    with patch.object(module, "MAX_RECORDS", 1), pytest.raises(ContractError):
        module.outcome(state, node="b", phase="call", result="passed")


def test_escaping_link_and_root_ownership_refused(state: module.Lifecycle) -> None:
    outside = state.evidence / "outside"
    outside.mkdir()
    link = state.scratch.python / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContractError):
        module.register(state, "a", link)
    with pytest.raises(ContractError):
        module.register(state, "a", state.scratch.python)
    with pytest.raises(ContractError):
        module.register(state, "a", outside)
    child = state.scratch.python / "child"
    child.mkdir()
    module.register(state, "a", child)
    with pytest.raises(ContractError):
        module.register(state, "b", child)
    assert outside.is_dir()


def test_changed_start_preserved_and_refused(state: module.Lifecycle) -> None:
    (state.evidence / "run-start.json").write_bytes(b"changed")
    with pytest.raises(ContractError):
        module.finish(state, 0)
    assert not (state.evidence / "session-finish.json").exists()


@pytest.mark.parametrize("attribute", ["fixtures"])
def test_ancestor_and_descendant_registration_are_both_rejected(
    state: module.Lifecycle, attribute: str
) -> None:
    parent = state.scratch.python / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    module.register(state, "child-owner", child)
    with pytest.raises(ContractError, match="overlaps"):
        module.register(state, "parent-owner", parent)
    nested = child / "nested"
    nested.mkdir()
    with pytest.raises(ContractError, match="overlaps"):
        module.register(state, "nested-owner", nested)
    sibling = parent / "sibling"
    sibling.mkdir()
    module.register(state, "sibling-owner", sibling)
    assert len(state.fixtures) == 2
    with pytest.raises(AttributeError):
        setattr(state, attribute, {})
    assert not isinstance(state.fixtures, dict)


@pytest.mark.parametrize("count", [1000, 4000])
def test_sibling_registration_scale_with_disjoint_paths(
    state: module.Lifecycle, count: int
) -> None:
    paths = tuple(state.scratch.python / f"synthetic-{index}" for index in range(count))
    tuple(map(Path.mkdir, paths))
    started = time.monotonic()
    tuple(map(lambda path: module.register(state, str(path), path), paths))
    current_seconds = time.monotonic() - started
    assert len(state.fixtures) == count
    print(f"registration-scale count={count} new={current_seconds:.6f}")


def test_finish_size_for_twelve_thousand_synthetic_nodes_stays_bounded() -> None:
    checkout = Path(__file__).resolve().parents[1]
    base = (
        checkout
        / "edits/work/deliveries-v2/full-verification-current-inputs-v31/pytest-gate/basetemp"
    )
    names = tuple(
        f"tests/test_generic_contract.py::test_synthetic_fixture_case[{i}]" for i in range(12000)
    )
    payload = {
        "version": 1,
        "kind": "pytest-lifecycle-finish",
        "startSHA256": "0" * 64,
        "exitStatus": 0,
        "fixtures": {
            str(base / f"test_synthetic_fixture_case_{i}"): name for i, name in enumerate(names)
        },
        "outcomes": {
            name: {"setup": "passed", "call": "passed", "teardown": "passed"} for name in names
        },
        "processTerminationVerified": False,
        "deletionPerformed": False,
    }
    encoded = encode(payload)
    assert len(encoded) < 8 * 1024 * 1024
    print(f"synthetic-finish-size nodes=12000 fixtures=12000 bytes={len(encoded)} bound=8388608")


def test_large_parameter_node_is_preserved_across_bounded_fragments(
    state: module.Lifecycle,
) -> None:
    node = "test::large[" + "x" * (9 * 1024 * 1024) + "]"
    child = state.scratch.python / "large"
    child.mkdir()
    module.register(state, node, child)
    module.outcome(state, node=node, phase="call", result="passed")
    path = module.finish(state, 0)
    manifest = decode(path.read_bytes())
    assert manifest["version"] == 2
    assert manifest["fixtureCount"] == 1
    assert manifest["outcomeCount"] == 1
    assert path.stat().st_size < 8 * 1024 * 1024
    result = module.load_finish(path)
    assert result["fixtures"] == {str(child): node}
    assert result["outcomes"] == {node: {"call": "passed"}}
    assert len(list(state.evidence.glob("*.fragment"))) == 19


@pytest.mark.parametrize("mutation", ["order", "missing", "hash", "count", "payload", "aggregate"])
def test_fragment_manifest_tampering_is_rejected(state: module.Lifecycle, mutation: str) -> None:
    path = module.finish(state, 0)
    manifest = decode(path.read_bytes())
    fragment = path.with_name("session-finish-0000.fragment")
    rows = manifest["fragments"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    if mutation == "order":
        row["name"] = "../alien.fragment"
    elif mutation == "missing":
        fragment.unlink()
    elif mutation == "hash":
        row["sha256"] = "0" * 64
    elif mutation == "count":
        manifest["fixtureCount"] = 1
    elif mutation == "payload":
        fragment.write_bytes(b"{}")
    else:
        manifest["sha256"] = "0" * 64
    path.write_bytes(encode(manifest))
    with pytest.raises((ContractError, FileNotFoundError)):
        module.load_finish(path)


def test_finish_aggregate_and_fragment_limits_preserve_partial_evidence(
    state: module.Lifecycle,
) -> None:
    with (
        patch.object(module, "MAX_FINISH_BYTES", 8),
        pytest.raises(ContractError, match="aggregate"),
    ):
        module.finish(state, 0)
    assert not (state.evidence / "session-finish.json").exists()
    with (
        patch.object(module, "MAX_FINISH_FRAGMENTS", 0),
        pytest.raises(ContractError, match="fragment count"),
    ):
        module.finish(state, 0)
    assert not (state.evidence / "session-finish.json").exists()


@pytest.mark.parametrize(
    "change",
    [{"version": 9}, {"bytes": 0}, {"fragments": []}, {"fixtureCount": False}, {"unknown": True}],
)
def test_invalid_manifest_boundaries_fail_before_reads(
    state: module.Lifecycle, change: dict[str, object]
) -> None:
    path = module.finish(state, 0)
    manifest = decode(path.read_bytes())
    manifest.update(change)
    path.write_bytes(encode(manifest))
    with pytest.raises(ContractError, match="Invalid bounded"):
        module.load_finish(path)


def test_start_actual_paths_and_hooks(state: module.Lifecycle) -> None:
    (state.evidence / "run-start.json").unlink()
    request = state.evidence.parent / "request.json"
    request.write_bytes(
        encode(
            {
                "version": 1,
                "kind": "pytest-lifecycle-request",
                "evidenceDirectory": str(state.evidence),
            }
        )
    )
    config = Mock(spec=pytest.Config)
    config.stash = pytest.Stash()
    config.stash[STATE] = state.scratch
    config.known_args_namespace = Mock(basetemp=str(state.scratch.base))
    config.getoption.return_value = str(request)
    parser = Mock(spec=pytest.Parser)
    module.pytest_addoption(parser)
    parser.addoption.assert_called_once()
    with (
        patch.object(module, "verify_bootstrap"),
        patch("replay.pytest_lifecycle.SystemHost.birth", return_value="actual-birth"),
    ):
        module.pytest_configure(config)
    observed = decode((state.evidence / "run-start.json").read_bytes())
    assert observed["kind"] == "pytest-lifecycle-start"
    assert observed["base"] == str(state.scratch.base)
    active = config.stash[module.LIFECYCLE]
    child = state.scratch.python / "synthetic"
    child.mkdir()
    fixture = Mock(spec=pytest.FixtureRequest)
    fixture.config = config
    fixture.node = Mock(nodeid="example")
    module.disposable_callback(fixture)(child)
    item = Mock(spec=pytest.Item)
    item.config = config
    hook = module.pytest_runtest_makereport(item, Mock())
    next(hook)
    report = Mock(spec=pytest.TestReport, nodeid="example", when="call", outcome="passed")
    with pytest.raises(StopIteration):
        hook.send(report)
    session = Mock(spec=pytest.Session)
    session.config = config
    module.pytest_sessionfinish(session, 0)
    assert active.outcomes == {"example": {"call": "passed"}}


def test_start_rejects_occupied_base_and_proof_inside_scratch(state: module.Lifecycle) -> None:
    request = state.evidence.parent / "request.json"
    config = Mock(spec=pytest.Config)
    config.stash = pytest.Stash()
    config.stash[STATE] = state.scratch
    config.known_args_namespace = Mock(basetemp=str(state.scratch.base))
    request.write_bytes(
        encode(
            {
                "version": 1,
                "kind": "pytest-lifecycle-request",
                "evidenceDirectory": str(state.scratch.python),
            }
        )
    )
    with patch.object(module, "verify_bootstrap"), pytest.raises(ContractError):
        module.start(config, request)
    state.scratch.base.mkdir()
    request.write_bytes(
        encode(
            {
                "version": 1,
                "kind": "pytest-lifecycle-request",
                "evidenceDirectory": str(state.evidence),
            }
        )
    )
    with patch.object(module, "verify_bootstrap"), pytest.raises(ContractError):
        module.start(config, request)
    request.write_bytes(b"{}")
    with patch.object(module, "verify_bootstrap"), pytest.raises(ContractError):
        module.start(config, request)
    config.getoption.return_value = None
    with pytest.raises(pytest.UsageError):
        module.pytest_configure(config)


def test_real_plugin_observes_two_actual_scratch_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    work = project / "edits/work"
    work.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    evidence = work / "external-evidence"
    evidence.mkdir()
    request = work / "request.json"
    request.write_bytes(
        encode(
            {"version": 1, "kind": "pytest-lifecycle-request", "evidenceDirectory": str(evidence)}
        )
    )
    test = project / "test_synthetic.py"
    test.write_text(
        "def test_synthetic(tmp_path, lifecycle_disposable):\n"
        "    lifecycle_disposable(tmp_path)\n"
        "    (tmp_path / 'fixture.txt').write_text('synthetic', encoding='utf-8')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-p",
            "replay.pytest_scratch",
            "-p",
            "replay.pytest_lifecycle",
            "--lifecycle-request",
            str(request),
            "--basetemp",
            str(work / "child-base"),
            "-q",
            str(test),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        start_new_session=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    start = decode((evidence / "run-start.json").read_bytes())
    finish = module.load_finish(evidence / "session-finish.json")
    assert start["base"] != start["pythonTmp"]
    assert finish["exitStatus"] == 0
    assert finish["processTerminationVerified"] is False
    fixtures = finish["fixtures"]
    assert isinstance(fixtures, dict)
    assert len(fixtures) == 1
