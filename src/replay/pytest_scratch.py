"""Set repository-local pytest scratch before capture or conftest import.

Load explicitly with ``-p replay.pytest_scratch``. Importing this module has no
filesystem or environment side effects; this is a development-only plugin.
"""

import fcntl
import hashlib
import os
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[2]
TEMP_ENVIRONMENT = ("TMPDIR", "TMP", "TEMP", "PYTEST_DEBUG_TEMPROOT")


@dataclass(frozen=True)
class Scratch:
    root: Path
    base: Path
    python: Path


STATE = pytest.StashKey[Scratch]()


def _checked(path: Path, *, root: Path) -> Path:
    work = root / "edits" / "work"
    default = work / "pytest-default"
    if (
        ".." in path.parts
        or not path.is_absolute()
        or not path.is_relative_to(work)
        or path in (work, default)
        or any(
            part.is_symlink() or (part.exists() and not part.is_dir())
            for part in (path, *path.parents)
        )
        or path != path.resolve()
    ):
        raise pytest.UsageError("pytest scratch must be an unlinked descendant of edits/work")
    return path


def _root_directory(root: Path) -> Path:
    default = root / "edits" / "work" / "pytest-default"
    if default.resolve() != default or any(
        part.is_symlink() or (part.exists() and not part.is_dir())
        for part in (default, *default.parents)
    ):
        raise pytest.UsageError("pytest-default must be a real repository work directory")
    default.mkdir(parents=True, exist_ok=True, mode=0o700)
    return default


def _lock(base: Path, *, root: Path, default: Path) -> int:
    locks = _checked(default / "locks", root=root)
    locks.mkdir(exist_ok=True, mode=0o700)
    name = hashlib.sha256(os.fsencode(base)).hexdigest()
    descriptor = os.open(locks / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise pytest.UsageError("pytest scratch is already owned by another test run") from error
    return descriptor


def _base(value: object, *, root: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or any(char in value for char in "\0\r\n"):
        raise pytest.UsageError("Expected a nonempty pytest base path")
    return _checked(Path(value).absolute(), root=root)


def _bootstrap(config: pytest.Config) -> None:
    root = config.rootpath
    if root.resolve() != root or not root.is_relative_to(CHECKOUT):
        raise pytest.UsageError("pytest root must remain inside this checkout")
    explicit = _base(config.known_args_namespace.basetemp, root=root)
    if explicit is not None and explicit.is_relative_to(root / "edits" / "work" / "pytest-default"):
        raise pytest.UsageError("Explicit pytest bases cannot occupy the default scratch tree")
    default = _root_directory(root)
    run = _checked(default / f"run-{uuid.uuid4().hex}", root=root)
    base = _checked(run / "pytest", root=root) if explicit is None else explicit
    descriptor = _lock(base, root=root, default=default)
    config.add_cleanup(lambda: os.close(descriptor))
    run.mkdir(mode=0o700)
    scratch = _checked(run / "python-tmp", root=root)
    scratch.mkdir(mode=0o700)
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    for key in TEMP_ENVIRONMENT:
        patch.setenv(key, str(scratch))
    # An imported plugin may already have cached an OS tempfile path.
    patch.setattr(tempfile, "tempdir", str(scratch))
    config.stash[STATE] = Scratch(root, base, scratch)


def verify_bootstrap(config: pytest.Config) -> None:
    """Check the early hook state before tmp_path can delete/create anything."""
    state = config.stash.get(STATE, None)
    if state is None:
        raise pytest.UsageError(
            "Repository tests require the early -p replay.pytest_scratch plugin"
        )
    configured = _base(config.getoption("basetemp"), root=state.root)
    if configured is not None and configured != state.base:
        raise pytest.UsageError("pytest base changed after scratch bootstrap")
    _checked(state.base, root=state.root)
    _checked(state.python, root=state.root)
    if tempfile.tempdir != str(state.python) or any(
        os.environ.get(key) != str(state.python) for key in TEMP_ENVIRONMENT
    ):
        raise pytest.UsageError("Python temporary paths changed after scratch bootstrap")
    config.option.basetemp = str(state.base)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_load_initial_conftests(early_config: pytest.Config) -> Iterator[None]:
    _bootstrap(early_config)
    yield


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    verify_bootstrap(config)
