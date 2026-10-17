"""Real isolated synthetic pytest children and mocked unsafe boundary failures."""

import hashlib
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from replay import leaf_retirement, pytest_scratch_disposition
from replay import pytest_gate_lifecycle as module
from replay import pytest_lifecycle as lifecycle
from replay.native_sequence import ContractError
from replay.sequence_journal import decode, encode


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass
class Gate:
    checkout: Path
    root: Path
    request: Path
    test: Path
    approval: Path


@pytest.fixture
def gate(tmp_path: Path) -> Gate:
    project = tmp_path / "project"
    work = project / "edits/work"
    root = work / "gate"
    root.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (project / "src").mkdir()
    (project / "tests").mkdir()
    source = project / "src/sample.py"
    source.write_text("def answer():\n    return 42\n", encoding="utf-8")
    test = project / "tests/test_sample.py"
    test.write_text(
        "from src.sample import answer\ndef test_sample(tmp_path):\n"
        "    (tmp_path / 'synthetic.txt').write_text('fixture', encoding='utf-8')\n"
        "    assert answer() == 42\n",
        encoding="utf-8",
    )
    approval = work / "approval.json"
    approval.write_bytes(
        encode(
            {
                "kind": "pytest-generated-fixture-owner-policy",
                "generatedTmpPath": "synthetic-disposable-after-complete-success",
                "cleanupAuthorized": True,
            }
        )
    )
    request = work / "request.json"
    request.write_bytes(
        encode(
            {
                "version": 1,
                "kind": "pytest-owned-gate",
                "root": str(root),
                "python": str(Path(sys.executable).absolute()),
                "tests": [str(test)],
                "coverageSources": [str(source.parent)],
                "sourcePins": [ref(source), ref(test)],
                "approval": ref(approval),
                "apply": True,
            }
        )
    )
    return Gate(project, root, request, test, approval)


def test_real_success_records_and_retires_only_generated_fixture(gate: Gate) -> None:
    # The original apply implementation is used; only OS FD observation is synthetic.
    original = leaf_retirement.retire
    with patch.object(
        leaf_retirement,
        "retire",
        side_effect=lambda *args, **kwargs: original(*args, **kwargs, probe=lambda paths: set()),
    ):
        result = module.run(gate.request, checkout=gate.checkout)
    raw = decode(result.read_bytes())
    assert raw["pytestExitStatus"] == 0
    assert raw["coveragePassed"] is True
    assert raw["retiredLeaves"] == 1
    assert not (gate.root / "basetemp/test_sample0/synthetic.txt").exists()
    assert (gate.root / "evidence/coverage.json").is_file()
    command = decode((gate.root / "evidence/dispatch.json").read_bytes())["command"]
    assert isinstance(command, list)
    assert command.count("pytest") == 1
    assert command.count("--cov-report=json:" + str(gate.root / "evidence/coverage.json")) == 1
    assert command.count("cache_dir=" + str(gate.root / "cache/pytest")) == 1
    assert (gate.root / "cache/pytest/v/cache/nodeids").is_file()
    assert not (gate.checkout / ".pytest_cache").exists()
    assert module.run(gate.request, checkout=gate.checkout) == result


def test_real_failure_never_calls_apply_and_preserves_fixture(gate: Gate) -> None:
    gate.test.write_text(
        "from src.sample import answer\ndef test_sample(tmp_path):\n"
        "    (tmp_path / 'synthetic.txt').write_text('reproduction', encoding='utf-8')\n"
        "    assert answer() == 99\n",
        encoding="utf-8",
    )
    raw = decode(gate.request.read_bytes())
    raw["sourcePins"] = [ref(gate.checkout / "src/sample.py"), ref(gate.test)]
    gate.request.write_bytes(encode(raw))
    with patch.object(leaf_retirement, "retire") as retire:
        result = module.run(gate.request, checkout=gate.checkout)
    retire.assert_not_called()
    assert decode(result.read_bytes())["pytestExitStatus"] == 1
    assert (gate.root / "basetemp/test_sample0/synthetic.txt").read_text(
        encoding="utf-8"
    ) == "reproduction"


def test_open_handle_uncertainty_holds_real_success_fixture(gate: Gate) -> None:
    original = leaf_retirement.retire
    with patch.object(
        leaf_retirement,
        "retire",
        side_effect=lambda *args, **kwargs: original(*args, **kwargs, probe=lambda paths: None),
    ):
        result = module.run(gate.request, checkout=gate.checkout)
    assert decode(result.read_bytes())["retiredLeaves"] == 0
    assert (gate.root / "basetemp/test_sample0/synthetic.txt").exists()


def test_unfinished_intent_is_not_relaunched(gate: Gate) -> None:
    (gate.root / "intent.json").write_bytes(b"{}")
    with (
        patch.object(subprocess, "Popen") as launch,
        pytest.raises(ContractError, match="reconciled"),
    ):
        module.run(gate.request, checkout=gate.checkout)
    launch.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("apply", "yes"),
        ("sourcePins", []),
        ("tests", []),
        ("tests", ["/outside/test.py"]),
        ("coverageSources", ["relative"]),
        ("approval", {"path": "missing"}),
    ],
)
def test_invalid_inputs_refused_before_process(gate: Gate, field: str, value: object) -> None:
    raw = decode(gate.request.read_bytes())
    raw[field] = value
    gate.request.write_bytes(encode(raw))
    with patch.object(subprocess, "Popen") as launch, pytest.raises(ContractError):
        module.run(gate.request, checkout=gate.checkout)
    launch.assert_not_called()


def test_explicit_policy_and_occupied_output_are_required(gate: Gate) -> None:
    gate.approval.write_bytes(encode({"cleanupAuthorized": False}))
    raw = decode(gate.request.read_bytes())
    raw["approval"] = ref(gate.approval)
    gate.request.write_bytes(encode(raw))
    with pytest.raises(ContractError, match="ownership"):
        module.run(gate.request, checkout=gate.checkout)


def test_hook_requires_actual_path_and_parent_pin(gate: Gate) -> None:
    parser = Mock(spec=pytest.Parser)
    module.pytest_addoption(parser)
    parser.addoption.assert_called_once()
    config = Mock(spec=pytest.Config)
    config.getoption.return_value = None
    with pytest.raises(ContractError):
        module.pytest_configure(config)
    fixture = Mock(spec=pytest.FixtureDef, argname="tmp_path")
    request = Mock(spec=pytest.FixtureRequest)
    request.config = config
    hook = module.pytest_fixture_setup(fixture, request)
    next(hook)
    with pytest.raises(ContractError):
        hook.send("not-a-path")


def test_cli_reports_failed_gate_without_masking_status(gate: Gate) -> None:
    result = gate.root / "synthetic-result.json"
    result.write_bytes(encode({"pytestExitStatus": 1, "coveragePassed": False}))
    with patch.object(module, "run", return_value=result):
        assert module.main([str(gate.request)]) == 1


def test_makefile_coverage_argument_checks_exact_owned_output(gate: Gate) -> None:
    result = gate.root / "synthetic-result.json"
    result.write_bytes(encode({"pytestExitStatus": 0, "coveragePassed": True}))
    argument = "--cov-report=json:" + str(gate.root / "evidence/coverage.json")
    with (
        patch.object(module, "CHECKOUT", gate.checkout),
        patch.object(module, "run", return_value=result) as run,
    ):
        assert module.main([str(gate.request), argument]) == 0
    run.assert_called_once_with(gate.request)


@pytest.mark.parametrize("value", ["term", "json:relative.json", "json:/unowned/coverage.json"])
def test_makefile_coverage_mismatch_never_starts_pytest(gate: Gate, value: str) -> None:
    with (
        patch.object(module, "CHECKOUT", gate.checkout),
        patch.object(module, "run") as run,
        pytest.raises(ContractError, match="coverage output"),
    ):
        module.main([str(gate.request), "--cov-report=" + value])
    run.assert_not_called()


def test_configured_hook_uses_actual_returned_fixture(gate: Gate) -> None:
    config = Mock(spec=pytest.Config)
    config.rootpath = gate.checkout
    config.stash = pytest.Stash()
    config.getoption.return_value = str(gate.request)
    module.pytest_configure(config)
    config.stash[lifecycle.LIFECYCLE] = Mock(spec=lifecycle.Lifecycle)
    fixture = Mock(spec=pytest.FixtureDef, argname="tmp_path")
    request = Mock(spec=pytest.FixtureRequest)
    request.config = config
    request.node = Mock(nodeid="tests/test_sample.py::test_sample")
    hook = module.pytest_fixture_setup(fixture, request)
    next(hook)
    with patch.object(lifecycle, "register") as register, pytest.raises(StopIteration):
        hook.send(gate.root / "actual-fixture")
    register.assert_called_once_with(
        config.stash[lifecycle.LIFECYCLE],
        "tests/test_sample.py::test_sample",
        gate.root / "actual-fixture",
    )


def test_occupied_base_and_missing_python_never_launch(gate: Gate) -> None:
    (gate.root / "basetemp").mkdir()
    with (
        patch.object(subprocess, "Popen") as launch,
        pytest.raises(ContractError, match="occupied"),
    ):
        module.run(gate.request, checkout=gate.checkout)
    launch.assert_not_called()


def test_original_source_mutation_and_untrusted_extension_refused(gate: Gate) -> None:
    raw = decode(gate.request.read_bytes())
    alien = gate.checkout / "sample.mov"
    alien.write_bytes(b"synthetic")
    raw["sourcePins"] = [ref(alien)]
    gate.request.write_bytes(encode(raw))
    with pytest.raises(ContractError, match="code scope"):
        module.run(gate.request, checkout=gate.checkout)


def test_coverage_failure_does_not_apply_successful_test_fixture(gate: Gate) -> None:
    source = gate.checkout / "src/sample.py"
    source.write_text(
        "def answer():\n    return 42\ndef unused():\n    return 7\n", encoding="utf-8"
    )
    raw = decode(gate.request.read_bytes())
    raw["sourcePins"] = [ref(source), ref(gate.test)]
    gate.request.write_bytes(encode(raw))
    with patch.object(leaf_retirement, "retire") as retire:
        result = module.run(gate.request, checkout=gate.checkout)
    retire.assert_not_called()
    assert decode(result.read_bytes())["coveragePassed"] is False
    assert (gate.root / "basetemp/test_sample0/synthetic.txt").exists()


def test_large_disposition_reference_and_reentry_keep_small_proofs_bounded(gate: Gate) -> None:
    leaves = gate.root / "large-disposition.json"
    leaves.write_bytes(
        b'{"version":1,"kind":"owner-policy-leaf-disposition",'
        b'"mediaContentsRead":false,"rows":[]}' + b" " * (18 * 1024 * 1024)
    )
    reference = module._leaf_ref(leaves, gate.checkout)
    module._verify_proofs({"disposition": reference}, gate.checkout)
    with pytest.raises(ContractError, match="bound"):
        module._ref(leaves)
    with pytest.raises(ContractError, match="exact metadata"):
        module._verify_proofs({"disposition": {"path": str(leaves)}}, gate.checkout)


def test_disposition_over_domain_budget_is_refused(gate: Gate) -> None:
    leaves = gate.root / "oversized.json"
    with leaves.open("wb") as stream:
        stream.truncate(128 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="bounded"):
        module._leaf_ref(leaves, gate.checkout)


@pytest.mark.parametrize("packages", [[], ["bad/name"], ["not_declared"], ["src", "src"], [7]])
def test_coverage_packages_refuse_unbound_names(gate: Gate, packages: object) -> None:
    (gate.checkout / "src/__init__.py").write_bytes(b"")
    raw = decode(gate.request.read_bytes())
    raw["coveragePackages"] = packages
    gate.request.write_bytes(encode(raw))
    with pytest.raises(ContractError, match="[Cc]overage"):
        module._load(gate.request, gate.checkout)


def test_package_coverage_includes_relocated_file_and_real_subprocess(gate: Gate) -> None:
    init = gate.checkout / "src/__init__.py"
    init.write_bytes(b"")
    child = gate.checkout / "src/child.py"
    child.write_text("def answer():\n    return 7\nanswer()\n", encoding="utf-8")
    gate.test.write_text(
        "import os, subprocess, sys\nfrom pathlib import Path\nfrom coverage import Coverage\n"
        "from src import sample\n"
        "def test_sample(tmp_path):\n"
        "    sample.__file__ = str(tmp_path / 'sample.py')\n"
        "    assert sample.answer() == 42\n"
        "    active = Coverage.current()\n    assert active is not None\n"
        "    env = dict(os.environ, COVERAGE_FILE=str(Path(active.config.data_file).absolute()))\n"
        "    subprocess.run([sys.executable, '-m', 'coverage', 'run', '--parallel-mode',"
        " '--source=src', '-m', 'src.child'], env=env, check=True)\n",
        encoding="utf-8",
    )
    raw = decode(gate.request.read_bytes())
    raw["coveragePackages"] = ["src"]
    raw["sourcePins"] = [
        ref(init),
        ref(child),
        ref(gate.checkout / "src/sample.py"),
        ref(gate.test),
    ]
    raw["apply"] = False
    gate.request.write_bytes(encode(raw))
    result = decode(module.run(gate.request, checkout=gate.checkout).read_bytes())
    assert result["pytestExitStatus"] == 0
    assert result["coveragePassed"] is True
    coverage = decode((gate.root / "evidence/coverage.json").read_bytes())
    files = coverage["files"]
    assert isinstance(files, dict)
    sample = files["src/sample.py"]
    child_result = files["src/child.py"]
    assert isinstance(sample, dict) and isinstance(child_result, dict)
    assert sample["missing_lines"] == []
    assert child_result["missing_lines"] == []


def test_lazy_plugin_cold_import_definitions_are_measured(gate: Gate) -> None:
    # This gate runs pytest in the real checkout, whose scratch plugin refuses an explicit
    # basetemp inside its default tree; tmp_path lives there, so the gate root must not.
    root = module.CHECKOUT / "edits/work/pytest-gate-cold-import" / uuid.uuid4().hex / "gate"
    root.mkdir(parents=True)
    try:
        gate.test.write_text("def test_cold_import():\n    assert True\n", encoding="utf-8")
        raw = decode(gate.request.read_bytes())
        raw["root"] = str(root)
        raw["coverageSources"] = [str(module.CHECKOUT / "src")]
        raw["coveragePackages"] = ["replay"]
        raw["sourcePins"] = [ref(gate.test)]
        raw["apply"] = False
        gate.request.write_bytes(encode(raw))
        result = decode(module.run(gate.request).read_bytes())
        coverage = decode((root / "evidence/coverage.json").read_bytes())
    finally:
        shutil.rmtree(root.parent)
    assert result["pytestExitStatus"] == 0
    files = coverage["files"]
    assert isinstance(files, dict)
    constants = files["src/replay/caption_style_defaults.py"]
    assert isinstance(constants, dict)
    assert constants["missing_lines"] == []


@pytest.mark.parametrize("missing", [False, True])
def test_existing_result_rejects_changed_finish_fragment(gate: Gate, missing: bool) -> None:
    raw = decode(gate.request.read_bytes())
    raw["apply"] = False
    gate.request.write_bytes(encode(raw))
    result = module.run(gate.request, checkout=gate.checkout)
    fragment = gate.root / "evidence/session-finish-0000.fragment"
    if missing:
        fragment.unlink()
    else:
        fragment.write_bytes(b"changed")
    with (
        patch.object(subprocess, "Popen") as launch,
        patch.object(leaf_retirement, "retire") as retire,
        pytest.raises((ContractError, FileNotFoundError)),
    ):
        module.run(gate.request, checkout=gate.checkout)
    launch.assert_not_called()
    retire.assert_not_called()
    assert result.is_file()
    assert (gate.root / "basetemp/test_sample0/synthetic.txt").is_file()


def test_fragment_change_after_prepare_refuses_apply(gate: Gate) -> None:
    original = pytest_scratch_disposition.prepare

    def corrupt_after_prepare(request: Path, output: Path, *, checkout: Path) -> Path:
        result = original(request, output, checkout=checkout)
        (gate.root / "evidence/session-finish-0000.fragment").write_bytes(
            b"changed after preparation"
        )
        return result

    with (
        patch.object(pytest_scratch_disposition, "prepare", side_effect=corrupt_after_prepare),
        patch.object(leaf_retirement, "retire") as retire,
        pytest.raises(ContractError, match="fragment identity"),
    ):
        module.run(gate.request, checkout=gate.checkout)
    retire.assert_not_called()
    assert not (gate.root / "result.json").exists()
    assert (gate.root / "basetemp/test_sample0/synthetic.txt").is_file()
