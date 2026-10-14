"""Cold-import instrumentation and original hook delegation contracts."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from coverage import Coverage

from replay import pytest_gate_lifecycle as controller
from replay import pytest_gate_plugin as module
from replay import pytest_lifecycle as lifecycle


def test_options_and_active_coverage_precede_original_configuration() -> None:
    parser = Mock(spec=pytest.Parser)
    module.pytest_addoption(parser)
    assert parser.addoption.call_count == 2
    session = Mock(spec=pytest.Session)
    session.config = Mock(spec=pytest.Config)
    with (
        patch.object(Coverage, "current", return_value=Mock(spec=Coverage)),
        patch.object(lifecycle, "pytest_configure") as first,
        patch.object(controller, "pytest_configure") as second,
    ):
        module.pytest_sessionstart(session)
    first.assert_called_once_with(session.config)
    second.assert_called_once_with(session.config)


def test_missing_coverage_refuses_owned_run() -> None:
    with patch.object(Coverage, "current", return_value=None), pytest.raises(pytest.UsageError):
        module.pytest_sessionstart(Mock(spec=pytest.Session))


def test_fixture_and_report_wrappers_delegate_actual_results() -> None:
    request = Mock(spec=pytest.FixtureRequest)
    request.config = Mock(spec=pytest.Config)
    request.config.getoption.return_value = None
    fixture = Mock(spec=pytest.FixtureDef, argname="unowned")
    wrapper = module.pytest_fixture_setup(fixture, request)
    next(wrapper)
    with pytest.raises(StopIteration) as finished:
        wrapper.send("actual value")
    assert finished.value.value == "actual value"
    item = Mock(spec=pytest.Item)
    item.config = Mock(spec=pytest.Config)
    item.config.stash = pytest.Stash()
    item.config.stash[lifecycle.LIFECYCLE] = Mock(spec=lifecycle.Lifecycle)
    report = Mock(spec=pytest.TestReport, nodeid="node", when="call", outcome="passed")
    hook = module.pytest_runtest_makereport(item, Mock(spec=pytest.CallInfo))
    next(hook)
    with patch.object(lifecycle, "outcome") as outcome, pytest.raises(StopIteration):
        hook.send(report)
    outcome.assert_called_once_with(
        item.config.stash[lifecycle.LIFECYCLE], node="node", phase="call", result="passed"
    )


def test_finish_delegates_only_initialized_state() -> None:
    session = Mock(spec=pytest.Session)
    session.config = Mock(spec=pytest.Config)
    session.config.stash = pytest.Stash()
    with patch.object(lifecycle, "pytest_sessionfinish") as finish:
        module.pytest_sessionfinish(session, 1)
        finish.assert_not_called()
        session.config.stash[lifecycle.LIFECYCLE] = Mock(spec=lifecycle.Lifecycle)
        module.pytest_sessionfinish(session, 1)
    finish.assert_called_once_with(session, 1)


def test_explicit_disposable_fixture_keeps_original_contract() -> None:
    request = Mock(spec=pytest.FixtureRequest)
    request.config = Mock(spec=pytest.Config)
    request.config.stash = pytest.Stash()
    request.config.stash[lifecycle.LIFECYCLE] = Mock(spec=lifecycle.Lifecycle)
    request.node = Mock(nodeid="node")
    with patch.object(lifecycle, "disposable_callback") as callback:
        module.disposable_callback(request)
    callback.assert_called_once_with(request)


def test_cold_plugin_import_is_lightweight_and_measured_in_subprocess(tmp_path: Path) -> None:
    probe = tmp_path / "cold_import_probe.py"
    probe.write_text(
        "import sys\nimport replay.pytest_gate_plugin\n"
        "assert 'replay.fcp_native_queue' not in sys.modules\n"
        "assert 'replay.pytest_gate_lifecycle' not in sys.modules\n"
        "assert 'replay.pytest_lifecycle' not in sys.modules\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    active = Coverage.current()
    if active is not None:
        environment["COVERAGE_FILE"] = str(Path(active.config.data_file).absolute())
        command = [
            sys.executable,
            "-B",
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--source=replay.pytest_gate_plugin",
            str(probe),
        ]
    else:
        command = [sys.executable, "-B", str(probe)]
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
