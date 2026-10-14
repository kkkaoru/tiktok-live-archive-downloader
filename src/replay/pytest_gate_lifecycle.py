"""Finite isolated pytest gate plus exact synthetic scratch disposition and retirement."""

import argparse
import fcntl
import hashlib
import os
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest

from replay import leaf_disposition, leaf_retirement
from replay import pytest_lifecycle as lifecycle
from replay import pytest_scratch_disposition as disposition
from replay.native_sequence import ContractError
from replay.recordings import record
from replay.sequence_journal import Pin, bounded_read, decode, encode, publish
from replay.stages import execution_lock

CHECKOUT = Path(__file__).resolve().parents[2]
POLICY = pytest.StashKey[Pin]()


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(bounded_read(path)).hexdigest()}


def _leaf_ref(path: Path, checkout: Path) -> dict[str, str]:
    reference = leaf_disposition.metadata_reference(path, checkout=checkout)
    return {"path": str(reference.path), "sha256": reference.sha256}


def _verify_proofs(proofs: dict[str, object], checkout: Path) -> None:
    for name, value in proofs.items():
        if name != "disposition":
            reference = _pin(value)
            if name == "finish":
                lifecycle.load_finish(reference.path)
            continue
        raw = record(value)
        path, digest = raw.get("path"), raw.get("sha256")
        if (
            set(raw) != {"path", "sha256"}
            or not isinstance(path, str)
            or not isinstance(digest, str)
        ):
            raise ContractError("Disposition requires an exact metadata reference")
        leaf_disposition.verify_reference(
            leaf_disposition.Reference(Path(path), digest), checkout=checkout
        )


def _pin(value: object) -> Pin:
    raw = record(value)
    path, digest = raw.get("path"), raw.get("sha256")
    if set(raw) != {"path", "sha256"} or not isinstance(path, str) or not isinstance(digest, str):
        raise ContractError("An exact pinned input is required")
    candidate = Path(path)
    if (
        not candidate.is_relative_to(CHECKOUT)
        or {"private", "downloads", ".git", "models"}.intersection(
            candidate.relative_to(CHECKOUT).parts
        )
        or candidate.suffix
        not in {
            ".py",
            ".toml",
            ".lock",
            ".json",
            ".jsonl",
            ".log",
            ".status",
            ".sha256",
            ".txt",
            ".sh",
        }
    ):
        raise ContractError("Pin must reference unprotected code scope or metadata")
    result = Pin(candidate, digest)
    result.verify()
    return result


def _load(request: Path, checkout: Path) -> dict[str, object]:
    lifecycle.canonical(request, checkout / "edits/work")
    raw = decode(bounded_read(request))
    if (
        set(raw) - {"coveragePackages"}
        != {
            "version",
            "kind",
            "root",
            "python",
            "tests",
            "coverageSources",
            "sourcePins",
            "approval",
            "apply",
        }
        or raw.get("version") != 1
        or raw.get("kind") != "pytest-owned-gate"
        or type(raw.get("apply")) is not bool
    ):
        raise ContractError("Invalid owned pytest gate request")
    policy = decode(bounded_read(_pin(raw["approval"]).path))
    if (
        policy.get("kind") != "pytest-generated-fixture-owner-policy"
        or policy.get("generatedTmpPath") != "synthetic-disposable-after-complete-success"
        or policy.get("cleanupAuthorized") is not True
    ):
        raise ContractError(
            "Explicit synthetic fixture ownership and cleanup authorization required"
        )
    pins = raw["sourcePins"]
    if not isinstance(pins, list) or not pins or len(pins) > 4096:
        raise ContractError("Pinned original source inventory required")
    for value in pins:
        candidate = record(value).get("path")
        if (
            not isinstance(candidate, str)
            or not Path(candidate).is_relative_to(checkout)
            or Path(candidate).suffix
            not in {
                ".py",
                ".toml",
                ".lock",
                ".json",
                ".sh",
            }
        ):
            raise ContractError("Source pin is outside declared code scope")
        _pin(value)
    for key in ("tests", "coverageSources"):
        values = raw[key]
        if (
            not isinstance(values, list)
            or not values
            or len(values) > 4096
            or any(
                not isinstance(value, str)
                or not Path(value).is_absolute()
                or Path(value) != Path(value).resolve()
                or not Path(value).is_relative_to(checkout)
                or not Path(value).exists()
                for value in values
            )
        ):
            raise ContractError("Tests and coverage must name actual canonical checkout inputs")
    _coverage_sources(raw)
    return raw


def _coverage_sources(raw: dict[str, object]) -> list[str]:
    paths = raw["coverageSources"]
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ContractError("Coverage paths must be strings")
    packages = raw.get("coveragePackages")
    if packages is None:
        return [str(path) for path in paths]
    if not isinstance(packages, list) or not 1 <= len(packages) <= 4096:
        raise ContractError("Explicit coverage packages require a bounded nonempty array")
    result: list[str] = []
    for package in packages:
        if not isinstance(package, str) or not all(
            part.isidentifier() for part in package.split(".")
        ):
            raise ContractError("Coverage package must be a Python module name")
        parts = package.split(".")
        candidates = [
            Path(str(path)).joinpath(
                *(parts[1:] if Path(str(path)).name == parts[0] else parts), "__init__.py"
            )
            for path in paths
        ]
        if package in result or not any(
            path.is_file() and path == path.resolve() for path in candidates
        ):
            raise ContractError("Coverage package is not uniquely declared inside source paths")
        result.append(package)
    return result


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--owned-gate-request", default=None)


def pytest_configure(config: pytest.Config) -> None:
    name = config.getoption("--owned-gate-request")
    if not isinstance(name, str):
        raise ContractError("Pinned gate request required")
    raw = _load(Path(name), config.rootpath)
    config.stash[POLICY] = _pin(raw["approval"])


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(
    fixturedef: pytest.FixtureDef[object], request: pytest.FixtureRequest
) -> Generator[None, object, object]:
    value = yield
    if fixturedef.argname == "tmp_path":
        name = request.config.getoption("--owned-gate-request")
        if not isinstance(name, str) or not isinstance(value, Path):
            raise ContractError("Observed tmp_path ownership is absent")
        request.config.stash[POLICY].verify()
        lifecycle.register(request.config.stash[lifecycle.LIFECYCLE], request.node.nodeid, value)
    return value


def run(request: Path, *, checkout: Path = CHECKOUT) -> Path:
    raw = _load(request, checkout)
    root = lifecycle.canonical(Path(str(raw["root"])), checkout / "edits/work")
    if not root.is_dir():
        raise ContractError("Existing owned gate root required")
    request_ref = _ref(request)
    with execution_lock(root, "pytest-gate") as acquired:
        if not acquired:
            raise ContractError("Another controller owns this gate")
        result_path = root / "result.json"
        if result_path.exists():
            result = decode(bounded_read(result_path))
            if result.get("request") != request_ref:
                raise ContractError("Completed gate belongs to another request")
            _verify_proofs(record(result.get("proofs")), checkout)
            return result_path
        intent = root / "intent.json"
        if intent.exists() or intent.is_symlink():
            raise ContractError("Unfinished gate must be reconciled; do not duplicate pytest")
        base, evidence = root / "basetemp", root / "evidence"
        cache = root / "cache"
        if any(path.exists() or path.is_symlink() for path in (base, evidence, cache)):
            raise ContractError("Gate output namespace is occupied")
        python = Path(str(raw["python"]))
        if (
            python.parent != python.parent.resolve()
            or not python.is_file()
            or not os.access(python, os.X_OK)
        ):
            raise ContractError("Pinned environment Python executable required")
        evidence.mkdir(mode=0o700)
        cache.mkdir(mode=0o700)
        publish(intent, encode({"request": request_ref, "sourcePins": raw["sourcePins"]}))
        plugin_request = evidence / "lifecycle-request.json"
        publish(
            plugin_request,
            encode(
                {
                    "version": 1,
                    "kind": "pytest-lifecycle-request",
                    "evidenceDirectory": str(evidence),
                }
            ),
        )
        coverage = evidence / "coverage.json"
        sources, tests = raw["coverageSources"], raw["tests"]
        if not isinstance(sources, list) or not isinstance(tests, list):
            raise ContractError("Validated gate arrays changed")
        command = [
            str(python),
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-o",
            f"cache_dir={cache / 'pytest'}",
            "--rootdir",
            str(checkout),
            "-p",
            "replay.pytest_scratch",
            "-p",
            "replay.pytest_gate_plugin",
            "--lifecycle-request",
            str(plugin_request),
            "--owned-gate-request",
            str(request),
            "--basetemp",
            str(base),
            "--cov-report=term-missing",
            f"--cov-report=json:{coverage}",
            *[f"--cov={value}" for value in _coverage_sources(raw)],
            *[str(value) for value in tests],
        ]
        log = evidence / "pytest.log"
        environment = dict(os.environ)
        environment["COVERAGE_FILE"] = str(evidence / ".coverage")
        with (
            log.open("xb") as stream,
            subprocess.Popen(
                command,
                cwd=checkout,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                umask=0o022,
            ) as process,
        ):
            publish(
                evidence / "dispatch.json",
                encode(
                    {
                        "pid": process.pid,
                        "pgid": process.pid,
                        "command": command,
                        "request": request_ref,
                    }
                ),
            )
            status = process.wait()
        start_path, finish_path = evidence / "run-start.json", evidence / "session-finish.json"
        start = decode(bounded_read(start_path))
        owner = record(start.get("owner"))
        dispatched = decode(bounded_read(evidence / "dispatch.json"))
        if owner.get("pid") != dispatched.get("pid") or owner.get("pgid") != dispatched.get("pgid"):
            raise ContractError("Actual pytest process does not match isolated dispatch")
        _load(request, checkout)
        if _ref(request) != request_ref:
            raise ContractError("Gate request changed during pytest")
        terminal = evidence / "terminal.json"
        publish(
            terminal,
            encode(
                {
                    "kind": "pytest-process-terminal",
                    "owner": owner,
                    "waitStatusObserved": True,
                    "startSHA256": _ref(start_path)["sha256"],
                    "exitStatus": status,
                    "sourcePins": raw["sourcePins"],
                }
            ),
        )
        finish = lifecycle.load_finish(finish_path)
        covered = decode(bounded_read(coverage))
        files = record(covered.get("files"))
        coverage_passed = bool(files) and all(
            type(record(record(value).get("summary")).get("percent_covered")) in (int, float)
            and float(str(record(record(value).get("summary"))["percent_covered"])) >= 90
            for value in files.values()
        )
        if finish.get("exitStatus") != status:
            raise ContractError("Parent wait and pytest finish differ")
        proofs: dict[str, object] = {
            "start": _ref(start_path),
            "finish": _ref(finish_path),
            "terminal": _ref(terminal),
            "coverage": _ref(coverage),
            "log": _ref(log),
        }
        retired = 0
        if status == 0 and coverage_passed:
            approval = evidence / "approval.json"
            publish(
                approval,
                encode(
                    {
                        "kind": "pytest-synthetic-retirement-approval",
                        "startSHA256": _ref(start_path)["sha256"],
                        "originalOwnerPolicy": raw["approval"],
                        "disposablePaths": [],
                    }
                ),
            )
            prepare_request = evidence / "prepare-request.json"
            publish(
                prepare_request,
                encode(
                    {
                        "version": 1,
                        "kind": "pytest-scratch-finalization",
                        **{
                            name: proofs[name]
                            for name in ("start", "finish", "terminal", "coverage")
                        },
                        "approval": _ref(approval),
                        "proofs": [raw["approval"], _ref(log)],
                        "sourcePins": raw["sourcePins"],
                    }
                ),
            )
            output = root / "disposition"
            output.mkdir(mode=0o700)
            leaves = disposition.prepare(prepare_request, output, checkout=checkout)
            proofs["prepared"] = _ref(output / "prepared.json")
            proofs["disposition"] = _leaf_ref(leaves, checkout)
            if raw["apply"] is True:
                roots = [Path(str(start["base"])), Path(str(start["pythonTmp"]))]
                if not disposition.SystemOwnerProbe().inactive(
                    int(str(owner["pid"])), int(str(owner["pgid"]))
                ):
                    raise ContractError("Owner became active before retirement")
                _load(request, checkout)
                lock_record = record(start["baseLock"])
                lock_path = lifecycle.canonical(
                    Path(str(lock_record["path"])), checkout / "edits/work"
                )
                descriptor = os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    locked = os.fstat(descriptor)
                    if (locked.st_dev, locked.st_ino) != (
                        lock_record.get("dev"),
                        lock_record.get("inode"),
                    ):
                        raise ContractError("Original base lock changed before retirement")
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _verify_proofs(dict(proofs), checkout)
                    summary = leaf_retirement.retire(
                        leaves,
                        _leaf_ref(leaves, checkout)["sha256"],
                        roots,
                        evidence / "retirement.jsonl",
                        apply=True,
                    )
                finally:
                    os.close(descriptor)
                retired = summary.retired
                proofs["retirement"] = _ref(evidence / "retirement.jsonl")
        publish(
            result_path,
            encode(
                {
                    "version": 1,
                    "kind": "pytest-owned-gate-result",
                    "request": request_ref,
                    "pytestExitStatus": status,
                    "coveragePassed": coverage_passed,
                    "retiredLeaves": retired,
                    "proofs": proofs,
                    "failedFixturesPreserved": status != 0,
                }
            ),
        )
        return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--cov-report")
    args = parser.parse_args(argv)
    if args.cov_report is not None:
        raw = _load(args.request, CHECKOUT)
        expected = "json:" + str(Path(str(raw["root"])) / "evidence/coverage.json")
        if args.cov_report != expected:
            raise ContractError("Makefile coverage output must equal the owned gate coverage path")
    result = decode(bounded_read(run(args.request)))
    print(encode(result).decode(), end="")
    return 0 if result["pytestExitStatus"] == 0 and result["coveragePassed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
