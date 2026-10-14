"""Require the early scratch bootstrap before collecting repository tests."""

import pytest

from replay.pytest_scratch import verify_bootstrap


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    verify_bootstrap(config)
