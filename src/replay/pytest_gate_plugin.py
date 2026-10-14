"""Load owned gate instrumentation only after pytest-cov starts measuring imports."""

from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from coverage import Coverage


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--lifecycle-request", default=None, help="Pinned owner lifecycle request")
    parser.addoption(
        "--owned-gate-request", default=None, help="Pinned generated-fixture owner policy"
    )


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    if Coverage.current() is None:
        raise pytest.UsageError("Owned gate plugin requires active coverage before replay imports")
    # Required lazy boundary: these chains import production modules whose definitions
    # must be measured, unlike eager pytest -p loading before pytest-cov initialization.
    from replay import pytest_gate_lifecycle, pytest_lifecycle

    pytest_lifecycle.pytest_configure(session.config)
    pytest_gate_lifecycle.pytest_configure(session.config)


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(
    fixturedef: pytest.FixtureDef[object], request: pytest.FixtureRequest
) -> Generator[None, object, object]:
    from replay import pytest_gate_lifecycle

    return (yield from pytest_gate_lifecycle.pytest_fixture_setup(fixturedef, request))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    from replay import pytest_lifecycle

    return (yield from pytest_lifecycle.pytest_runtest_makereport(item, call))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    from replay import pytest_lifecycle

    if pytest_lifecycle.LIFECYCLE in session.config.stash:
        pytest_lifecycle.pytest_sessionfinish(session, exitstatus)


def disposable_callback(request: pytest.FixtureRequest) -> Callable[[Path], None]:
    from replay import pytest_lifecycle

    return pytest_lifecycle.disposable_callback(request)


lifecycle_disposable = pytest.fixture(disposable_callback, name="lifecycle_disposable")
