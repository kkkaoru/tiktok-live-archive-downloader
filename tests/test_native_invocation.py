import hashlib
import json
import subprocess
from io import BufferedReader
from pathlib import Path
from typing import BinaryIO

import pytest

from replay import native_invocation as invocation
from replay import stage_recovery
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import Stage, StageError, run_stage


@pytest.fixture
def native_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    native = tmp_path / "native"
    native.write_bytes(b"synthetic executable")
    source = tmp_path / "source.mov"
    source.write_bytes(b"synthetic source")
    alpha = tmp_path / "alpha.mov"
    alpha.write_bytes(b"synthetic alpha")
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "outputDirectory": str(tmp_path / "outputs"),
                "recipe": {
                    "clips": [{"sourcePath": str(source)}],
                    "additionalVideo": [{"sourcePath": str(alpha)}],
                },
            }
        )
    )

    def native_call(
        argv: list[str],
        *,
        stdin: int,
        stdout: BinaryIO,
        stderr: BinaryIO,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        assert argv == [str(native), "edit-media", str(path)]
        assert (
            invocation.read(path.with_name("render.invocation.json"))["fingerprintMoment"]
            == "before-native-call"
        )
        movie = tmp_path / "outputs/edit-dynamic/movie.mp4"
        movie.parent.mkdir()
        movie.write_bytes(b"synthetic rendered movie")
        stdout.write(
            json.dumps(
                {
                    "outputPath": str(movie),
                    "frameCount": 3,
                    "expectedDurationSeconds": 0.1,
                    "actualDurationSeconds": 0.1,
                }
            ).encode()
        )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", native_call)
    return path


def run(request: Path) -> Path:
    return invocation.run(request.parent / "native", request, request.parent / "outputs")


def verify(request: Path) -> dict[str, object]:
    return invocation.verify(request.parent / "native", request, request.parent / "outputs")


def test_real_wrapper_success_and_reuse_via_stage(native_request: Path) -> None:
    request = native_request
    stage = Stage(
        name="render",
        command=invocation.command(request.parent / "native", request, request.parent / "outputs"),
        verify_command=invocation.command(
            request.parent / "native", request, request.parent / "outputs", verify=True
        ),
        inputs=invocation.inputs(request.parent / "native", request),
        outputs=invocation.paths(request),
        timeout_seconds=1800,
    )
    calls: list[str] = []

    def execute(argv: tuple[str, ...], log: Path, timeout: int) -> None:
        calls.append(argv[3])
        assert invocation.main(list(argv[3:])) == 0

    assert (
        run_stage(stage, state_directory=request.parent / "state", execute=execute) == "completed"
    )
    assert run_stage(stage, state_directory=request.parent / "state", execute=execute) == "reused"
    assert calls == ["run", "verify", "verify"]
    saved = verify(request)
    assert saved["inputs"] == saved["inputsAfter"]
    assert (
        record(saved["response"])["sha256"]
        == hashlib.sha256(request.with_name("render.response.json").read_bytes()).hexdigest()
    )
    assert record(saved["output"])["bytes"] == 24
    with pytest.raises(ReplayError, match="never resubmit"):
        run(request)
    movie = request.parent / "outputs/edit-dynamic/movie.mp4"
    movie.write_bytes(b"mutated")
    with pytest.raises(AssertionError):
        run_stage(stage, state_directory=request.parent / "state", execute=execute)
    assert calls == ["run", "verify", "verify", "verify"]


@pytest.mark.parametrize(
    "mode",
    [
        "nonzero",
        "partial",
        "missing-fields",
        "timeout",
        "exception",
        "source",
        "alpha",
        "native",
        "request",
        "invocation",
        "escape",
        "symlink",
        "hardlink",
        "rename",
    ],
)
def test_failed_native_preserves_evidence_no_duplicate(
    native_request: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    request = native_request
    real_mock = subprocess.run

    def fail(
        argv: list[str],
        *,
        stdin: int,
        stdout: BinaryIO,
        stderr: BinaryIO,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        result = real_mock(
            argv, stdin=stdin, stdout=stdout, stderr=stderr, timeout=timeout, check=check
        )
        movie = request.parent / "outputs/edit-dynamic/movie.mp4"
        if mode in {"partial", "missing-fields"}:
            stdout.seek(0)
            stdout.truncate()
            stdout.write(
                b"{" if mode == "partial" else json.dumps({"outputPath": str(movie)}).encode()
            )
        elif mode == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        elif mode == "exception":
            raise OSError("unknown launch outcome")
        elif mode in {"source", "alpha", "native", "request"}:
            target = (
                request
                if mode == "request"
                else request.parent / (mode if mode == "native" else f"{mode}.mov")
            )
            target.write_bytes(b"mutated input")
        elif mode == "invocation":
            request.with_name("render.invocation.json").write_bytes(b"changed pre-call record")
        elif mode == "escape":
            stdout.seek(0)
            stdout.truncate()
            stdout.write(
                json.dumps(
                    {
                        "outputPath": str(request.parent / "source.mov"),
                        "frameCount": 3,
                        "expectedDurationSeconds": 0.1,
                        "actualDurationSeconds": 0.1,
                    }
                ).encode()
            )
        elif mode == "symlink":
            movie.rename(movie.with_suffix(".saved"))
            movie.symlink_to(movie.with_suffix(".saved"))
        elif mode == "hardlink":
            movie.with_suffix(".alias").hardlink_to(movie)
        elif mode == "rename":
            movie.rename(movie.with_suffix(".saved"))
        stderr.write(b"synthetic diagnostic")
        return subprocess.CompletedProcess(argv, 1) if mode == "nonzero" else result

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises((ReplayError, OSError, ValueError, subprocess.SubprocessError)):
        run(request)
    assert request.with_name("render.invocation.json").is_file()
    assert request.with_name("render.response.json").is_file()
    assert not request.with_name("render.binding.json").exists()
    assert (request.parent / "outputs/edit-dynamic").is_dir()
    with pytest.raises(ReplayError, match="never resubmit"):
        run(request)
    with pytest.raises(FileNotFoundError):
        verify(request)


@pytest.mark.parametrize(
    "name",
    ["source.mov", "alpha.mov", "native", "render.response.json", "outputs/edit-dynamic/movie.mp4"],
)
def test_mutation_during_hash_is_rejected(
    native_request: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    request = native_request
    digest = hashlib.file_digest

    def mutate(stream: BufferedReader, algorithm: str) -> object:
        result = digest(stream, algorithm)
        if Path(str(stream.name)).name == Path(name).name:
            with Path(stream.name).open("ab") as target:
                target.write(b"mutation")
        return result

    monkeypatch.setattr(hashlib, "file_digest", mutate)
    with pytest.raises(ReplayError, match="changed during hashing"):
        run(request)
    assert not request.with_name("render.binding.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "source.mov",
        "alpha.mov",
        "render.response.json",
        "render.invocation.json",
        "outputs/edit-dynamic/movie.mp4",
    ],
)
def test_reuse_rejects_replacement_and_rename(native_request: Path, name: str) -> None:
    request = native_request
    run(request)
    target = request.parent / name
    content = target.read_bytes()
    target.rename(target.with_suffix(".saved"))
    target.write_bytes(content)
    with pytest.raises(ReplayError):
        verify(request)


def test_output_escape_before_call_and_invalid_paths(native_request: Path) -> None:
    request = native_request
    with pytest.raises(ReplayError, match="escapes authorization"):
        invocation.run(request.parent / "native", request, request.parent / "unrelated")
    assert not request.with_name("render.invocation.json").exists()
    with pytest.raises(ReplayError, match="Noncanonical"):
        invocation.canonical("relative")
    with pytest.raises(ReplayError, match="Missing"):
        invocation.canonical(None)
    with pytest.raises(ReplayError, match="list"):
        invocation.rows({})
    request.write_bytes(b"")
    with pytest.raises(ReplayError, match="Empty"):
        invocation.pin(request)
    with pytest.raises(ReplayError, match="JSON size"):
        invocation.read(request)


def test_input_changed_after_pre_call_record(
    native_request: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = native_request
    durable = invocation.durable

    def change(path: Path, value: object) -> None:
        durable(path, value)
        (request.parent / "source.mov").write_bytes(b"changed before call")

    monkeypatch.setattr(invocation, "durable", change)
    with pytest.raises(ReplayError, match="before native call"):
        run(request)
    with pytest.raises(ReplayError, match="never resubmit"):
        run(request)


def test_stage_running_without_binding_cannot_render_again(native_request: Path) -> None:
    request = native_request
    stage = Stage(
        name="render",
        command=invocation.command(request.parent / "native", request, request.parent / "outputs"),
        inputs=invocation.inputs(request.parent / "native", request),
        outputs=invocation.paths(request),
        timeout_seconds=1800,
    )
    invocation.durable(request.with_name("render.invocation.json"), {"status": "unknown"})
    with pytest.raises(StageError, match="provenance verifier"):
        run_stage(stage, state_directory=request.parent / "state")
    assert not (request.parent / "outputs").exists()


def test_recovery_cannot_erase_no_resubmit_marker(
    native_request: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = native_request
    native, directory = request.parent / "native", request.parent / "outputs"
    stage = Stage(
        name="render",
        command=invocation.command(native, request, directory),
        inputs=invocation.inputs(native, request),
        outputs=invocation.paths(request),
        timeout_seconds=1800,
    )
    workflow = request.parent / "workflow.json"
    workflow.write_text(
        json.dumps(
            {
                "version": 1,
                "stages": [
                    {
                        "name": "render",
                        "command": list(stage.command),
                        "inputs": list(map(str, stage.inputs)),
                        "outputs": list(map(str, stage.outputs)),
                        "timeout_seconds": 1800,
                    }
                ],
            }
        )
    )
    state = request.parent / "state"
    durable = invocation.durable

    def interrupted(path: Path, value: object) -> None:
        durable(path, value)
        raise OSError("interruption before native launch")

    def execute(argv: tuple[str, ...], log: Path, timeout: int) -> None:
        run(request)

    monkeypatch.setattr(invocation, "durable", interrupted)
    with pytest.raises(OSError, match="interruption"):
        run_stage(stage, state_directory=state, execute=execute)
    assert len(stage_recovery.recover(workflow, state)) == 2
    assert not request.with_name("render.invocation.json").exists()
    assert not request.with_name("render.stderr").exists()
    with pytest.raises(ReplayError, match="never resubmit"):
        run_stage(stage, state_directory=state, execute=execute)
    assert not directory.exists()


def test_cli_error_retains_outcome(native_request: Path) -> None:
    request = native_request
    assert (
        invocation.main(
            ["run", str(request.parent / "native"), str(request), str(request.parent / "outputs")]
        )
        == 0
    )
    assert (
        invocation.main(
            ["run", str(request.parent / "native"), str(request), str(request.parent / "outputs")]
        )
        == 1
    )
