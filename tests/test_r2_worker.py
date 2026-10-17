import hashlib
import json
import secrets
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from replay import r2_worker
from replay.core import ReplayError
from replay.r2_worker import Part, Project, RemoteObject, WorkerStore

URL = "https://replay-offload-test.example.workers.dev"


class FakeWorker:
    """The token-gated upload Worker's HTTP contract, held in memory."""

    def __init__(self, *, bad_etag: bool = False, failing: frozenset[str] = frozenset()) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.parts: dict[int, bytes] = {}
        self.paths: list[str] = []
        self.bad_etag = bad_etag
        self.failing = failing

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path in self.failing:
            return httpx.Response(500)
        if request.headers.get("X-Upload-Token") != "token":
            return httpx.Response(403)
        key = request.url.params.get("key", "")
        if path == "/ready":
            return httpx.Response(200, json={"ready": True})
        if path == "/head":
            if key not in self.objects:
                return httpx.Response(404)
            body = {
                "size": len(self.objects[key]),
                "uploaded": "2026-10-17T03:00:00.000Z",
                "metadata": self.metadata[key],
            }
            return httpx.Response(200, json=body)
        if path == "/get":
            if key not in self.objects:
                return httpx.Response(404)
            return httpx.Response(200, content=self.objects[key])
        if path == "/part":
            number = int(request.url.params["n"])
            self.parts[number] = request.content
            etag = "0" * 32 if self.bad_etag else hashlib.md5(request.content).hexdigest()
            return httpx.Response(200, json={"partNumber": number, "etag": etag})
        body = json.loads(request.content)
        if path == "/create":
            self.metadata[body["key"]] = body["metadata"]
            return httpx.Response(200, json={"uploadId": "upload-1"})
        if path == "/complete":
            self.objects[body["key"]] = b"".join(self.parts[n] for n in sorted(self.parts))
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json={"ok": True})


def store(worker: FakeWorker) -> WorkerStore:
    client = httpx.Client(transport=httpx.MockTransport(worker))
    return WorkerStore(client, URL, "token", lambda _seconds: None)


def completed(code: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["cf"], code, stdout, stderr)


def test_config_text_binds_the_bucket_as_media() -> None:
    project = Project(Path("/w/replay-offload-1"), "replay-offload-1", "replay-intermediates")
    assert r2_worker.config_text(project) == (
        'import { defineConfig, bindings } from "cf/config";\n'
        "\n"
        "export default defineConfig({\n"
        "  worker: {\n"
        '    name: "replay-offload-1",\n'
        '    entrypoint: "worker.js",\n'
        '    compatibilityDate: "2025-06-01",\n'
        "    workersDev: true,\n"
        '    env: { MEDIA: bindings.r2({ name: "replay-intermediates" }) },\n'
        "  },\n"
        "});\n"
    )


def test_package_text_pins_the_deploy_tools() -> None:
    assert json.loads(r2_worker.package_text("replay-offload-1")) == {
        "name": "replay-offload-1",
        "private": True,
        "type": "module",
        "devDependencies": {"cf": "1.0.0-beta.16", "wrangler": "4.150.0"},
    }


def test_parts_cover_the_file_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r2_worker, "CHUNK_BYTES", 4)
    assert r2_worker.parts(10) == [Part(1, 0, 4), Part(2, 4, 4), Part(3, 8, 2)]
    assert r2_worker.parts(0) == [Part(1, 0, 0)]


def test_put_uploads_parts_and_records_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r2_worker, "CHUNK_BYTES", 4)
    source = tmp_path / "part.mov"
    source.write_bytes(b"abcdefghij")
    worker = FakeWorker()
    store(worker).put("intermediates/replay-1/x", source, {"sha256": "x"})
    assert worker.objects == {"intermediates/replay-1/x": b"abcdefghij"}
    assert worker.metadata == {"intermediates/replay-1/x": {"sha256": "x"}}


def test_put_aborts_when_a_part_never_matches_its_md5(tmp_path: Path) -> None:
    source = tmp_path / "part.mov"
    source.write_bytes(b"abcdefghij")
    worker = FakeWorker(bad_etag=True)
    with pytest.raises(ReplayError, match="Part 1 of part.mov failed or did not match its MD5"):
        store(worker).put("k", source, {})
    assert worker.paths == ["/create", "/part", "/part", "/part", "/part", "/abort"]


def test_put_reports_a_failed_abort(tmp_path: Path) -> None:
    source = tmp_path / "part.mov"
    source.write_bytes(b"abcdefghij")
    worker = FakeWorker(bad_etag=True, failing=frozenset({"/abort"}))
    with pytest.raises(ReplayError, match="its abort failed"):
        store(worker).put("k", source, {})


def test_put_needs_an_upload_id(tmp_path: Path) -> None:
    source = tmp_path / "part.mov"
    source.write_bytes(b"abcdefghij")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"uploadId": ""})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ReplayError, match="no upload ID"):
        WorkerStore(client, URL, "token", lambda _seconds: None).put("k", source, {})


def test_post_raises_on_an_http_error(tmp_path: Path) -> None:
    source = tmp_path / "part.mov"
    source.write_bytes(b"abcdefghij")
    with pytest.raises(ReplayError, match="/create failed with HTTP 500"):
        store(FakeWorker(failing=frozenset({"/create"}))).put("k", source, {})


def test_send_rejects_a_short_read(tmp_path: Path) -> None:
    source = tmp_path / "part.mov"
    source.write_bytes(b"ab")
    with pytest.raises(ReplayError, match="Short read of part 1 of part.mov"):
        store(FakeWorker())._send("k", "upload-1", Part(1, 0, 4), source)


def test_head_returns_size_upload_time_and_metadata() -> None:
    worker = FakeWorker()
    worker.objects["k"] = b"abc"
    worker.metadata["k"] = {"sha256": "s"}
    assert store(worker).head("k") == RemoteObject(
        3, datetime(2026, 10, 17, 3, 0, tzinfo=UTC), {"sha256": "s"}
    )
    assert store(worker).head("missing") is None


def test_head_rejects_errors_and_malformed_answers() -> None:
    with pytest.raises(ReplayError, match="/head failed with HTTP 500"):
        store(FakeWorker(failing=frozenset({"/head"}))).head("k")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"size": "3", "uploaded": "x", "metadata": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ReplayError, match="malformed object metadata"):
        WorkerStore(client, URL, "token", lambda _seconds: None).head("k")


def test_get_streams_into_a_new_file(tmp_path: Path) -> None:
    worker = FakeWorker()
    worker.objects["k"] = b"abcdefghij"
    destination = tmp_path / "restored.mov"
    assert store(worker).get("k", destination) == 10
    assert destination.read_bytes() == b"abcdefghij"
    with pytest.raises(FileExistsError):
        store(worker).get("k", destination)


def test_get_reports_a_missing_object(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="/get failed with HTTP 404"):
        store(FakeWorker()).get("missing", tmp_path / "restored.mov")
    assert not (tmp_path / "restored.mov").exists()


def test_wait_ready_retries_until_the_worker_answers() -> None:
    answers = iter(
        [
            httpx.Response(404),
            httpx.Response(200, text="<html>edge</html>"),
            httpx.Response(200, json={"ready": True}),
        ]
    )
    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: next(answers)))
    WorkerStore(client, URL, "token", sleeps.append).wait_ready()
    assert sleeps == [3.0, 3.0]


def test_wait_ready_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r2_worker, "READY_ATTEMPTS", 2)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ReplayError, match="did not become ready"):
        WorkerStore(client, URL, "token", lambda _seconds: None).wait_ready()


def test_session_deploys_yields_and_always_deletes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r2_worker, "NODE_MODULES", tmp_path)
    monkeypatch.setattr(secrets, "token_hex", lambda _size: "abcd1234")
    monkeypatch.setattr(secrets, "token_urlsafe", lambda _size: "token")
    calls: list[tuple[str, ...]] = []

    def runner(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append((*args, cwd.name))
        return completed(0, stdout=f"Deployed to {URL}\n")

    client = httpx.Client(transport=httpx.MockTransport(FakeWorker()))
    with r2_worker.session(
        "replay-intermediates", runner=runner, client=client, root=tmp_path / "w"
    ) as worker:
        project = tmp_path / "w" / "replay-offload-abcd1234"
        assert json.loads((project / "secrets.json").read_text(encoding="utf-8")) == {
            "UPLOAD_TOKEN": "token"
        }
        assert oct((project / "secrets.json").stat().st_mode & 0o777) == "0o600"
        assert (project / "node_modules").is_symlink()
        assert worker.url == URL
    assert calls == [
        ("deploy", "--secrets-file", "secrets.json", "replay-offload-abcd1234"),
        (
            "workers",
            "delete",
            "replay-offload-abcd1234",
            "--force",
            "tiktok-live-archive-downloader",
        ),
    ]
    assert list((tmp_path / "w").iterdir()) == []


def test_session_deletes_after_a_failed_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(r2_worker, "NODE_MODULES", tmp_path)
    calls: list[str] = []

    def runner(args: Sequence[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append(args[0])
        return completed(1, stderr="authentication required")

    with (
        pytest.raises(ReplayError, match="deploy failed: authentication required"),
        r2_worker.session("b", runner=runner, client=httpx.Client(), root=tmp_path / "w"),
    ):
        pytest.fail("the session must not yield after a failed deploy")
    assert calls == ["deploy", "workers"]
    assert list((tmp_path / "w").iterdir()) == []


def test_prepare_needs_the_worker_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r2_worker, "WORKER_SOURCE", tmp_path / "missing.js")
    with pytest.raises(ReplayError, match="needs web-viewer/src/upload-worker.js"):
        r2_worker.prepare(Project(tmp_path / "p", "p", "b"), "token")


def test_run_cf_invokes_the_cli_without_a_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        seen.append((command, options["cwd"], options["timeout"], options["check"]))
        return completed(0)

    monkeypatch.setattr(subprocess, "run", run)
    assert r2_worker.run_cf(["workers", "list"], tmp_path).returncode == 0
    assert seen == [(["cf", "workers", "list"], tmp_path, 600, False)]
