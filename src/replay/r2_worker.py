"""Temporary token-gated R2 Worker for objects above the 300 MB `cf r2 objects put` limit.

The Worker is web-viewer's `upload-worker.js`; no long-lived S3 key exists on this Mac. A
session deploys it under a random name with a fresh random token from a project directory
inside ignored `edits/work/r2-worker/`, and always deletes the Worker and that directory
afterwards. Uploads are multipart; each part is verified against the MD5 ETag R2 returns.
Whole-object integrity (SHA-256) belongs to the caller, which stores it as custom metadata
and rehashes downloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from replay.core import ReplayError
from replay.recordings import record

REPOSITORY = Path(__file__).resolve().parents[2]
WORKER_SOURCE = REPOSITORY / "web-viewer" / "src" / "upload-worker.js"
NODE_MODULES = REPOSITORY / "web-viewer" / "node_modules"
PROJECT_ROOT = REPOSITORY / "edits" / "work" / "r2-worker"
CHUNK_BYTES = 64 * 1024 * 1024
PARALLEL_PARTS = 4
PART_ATTEMPTS = 4
READY_ATTEMPTS = 30
RETRY_DELAY_SECONDS = 3.0
CF_TIMEOUT_SECONDS = 600
HTTP_TIMEOUT = httpx.Timeout(600.0, connect=30.0)
HTTP_OK_LIMIT = 300
HTTP_NOT_FOUND = 404
DOWNLOAD_BLOCK_BYTES = 1 << 20
# Cloudflare blocks the default HTTP client signature at the edge.
USER_AGENT = "curl/8.7.1"
WORKERS_DEV_URL = re.compile(r"https://[\w.-]+\.workers\.dev")
COMPATIBILITY_DATE = "2025-06-01"
DEV_DEPENDENCIES: Mapping[str, str] = {"cf": "1.0.0-beta.16", "wrangler": "4.150.0"}
OBJECT_TYPE = "application/octet-stream"
NAME_PREFIX = "replay-offload-"

Runner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]
Sleep = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class RemoteObject:
    size: int
    uploaded: datetime
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class Project:
    directory: Path
    name: str
    bucket: str


@dataclass(frozen=True, slots=True)
class Part:
    number: int
    offset: int
    length: int


def run_cf(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """The authenticated `cf` CLI; its output never carries the upload token."""
    return subprocess.run(
        ["cf", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=CF_TIMEOUT_SECONDS,
        check=False,
    )


def config_text(project: Project) -> str:
    return "\n".join(
        [
            'import { defineConfig, bindings } from "cf/config";',
            "",
            "export default defineConfig({",
            "  worker: {",
            f"    name: {json.dumps(project.name)},",
            '    entrypoint: "worker.js",',
            f"    compatibilityDate: {json.dumps(COMPATIBILITY_DATE)},",
            "    workersDev: true,",
            f"    env: {{ MEDIA: bindings.r2({{ name: {json.dumps(project.bucket)} }}) }},",
            "  },",
            "});",
            "",
        ]
    )


def package_text(name: str) -> str:
    manifest = {
        "name": name,
        "private": True,
        "type": "module",
        "devDependencies": dict(DEV_DEPENDENCIES),
    }
    return json.dumps(manifest) + "\n"


def prepare(project: Project, token: str) -> None:
    """Write the Worker project; `cf deploy` needs wrangler inside the project directory."""
    if not WORKER_SOURCE.is_file() or not NODE_MODULES.is_dir():
        raise ReplayError("The R2 Worker needs web-viewer/src/upload-worker.js and node_modules")
    project.directory.mkdir(mode=0o700, parents=True)
    shutil.copyfile(WORKER_SOURCE, project.directory / "worker.js")
    (project.directory / "cloudflare.config.ts").write_text(config_text(project), encoding="utf-8")
    (project.directory / "package.json").write_text(package_text(project.name), encoding="utf-8")
    descriptor = os.open(
        project.directory / "secrets.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"UPLOAD_TOKEN": token}, stream)
    (project.directory / "node_modules").symlink_to(NODE_MODULES, target_is_directory=True)


def deploy(project: Project, runner: Runner) -> str:
    """Deploy the prepared project and return its workers.dev URL."""
    done = runner(["deploy", "--secrets-file", "secrets.json"], project.directory)
    match = WORKERS_DEV_URL.search(f"{done.stdout}\n{done.stderr}")
    if done.returncode != 0 or match is None:
        raise ReplayError(f"R2 Worker deploy failed: {done.stderr[-400:]}")
    return match.group(0)


def remove(project: Project, runner: Runner) -> None:
    """Delete the Worker, then its project directory, even after a failed deploy."""
    runner(["workers", "delete", project.name, "--force"], REPOSITORY)
    shutil.rmtree(project.directory, ignore_errors=True)


def parts(size: int) -> list[Part]:
    count = max(1, -(-size // CHUNK_BYTES))
    return [
        Part(
            number, (number - 1) * CHUNK_BYTES, min(CHUNK_BYTES, size - (number - 1) * CHUNK_BYTES)
        )
        for number in range(1, count + 1)
    ]


def _remote(raw: object) -> RemoteObject:
    data = record(raw)
    size, uploaded, metadata = data.get("size"), data.get("uploaded"), data.get("metadata")
    if type(size) is not int or not isinstance(uploaded, str) or not isinstance(metadata, dict):
        raise ReplayError("R2 Worker returned malformed object metadata")
    values = {str(key): value for key, value in metadata.items() if isinstance(value, str)}
    return RemoteObject(size, datetime.fromisoformat(uploaded), values)


def _is_ready(response: httpx.Response) -> bool:
    if response.status_code != httpx.codes.OK:
        return False
    try:
        return record(response.json()).get("ready") is True
    except (ValueError, ReplayError):
        return False


class WorkerStore:
    """Object operations through a deployed Worker; part failures retry, then abort."""

    def __init__(self, client: httpx.Client, url: str, token: str, sleep: Sleep) -> None:
        self.client = client
        self.url = url
        self.token = token
        self.sleep = sleep

    def _headers(self) -> dict[str, str]:
        return {"X-Upload-Token": self.token, "User-Agent": USER_AGENT}

    def _post(self, path: str, body: Mapping[str, object]) -> dict[str, object]:
        response = self.client.post(f"{self.url}{path}", headers=self._headers(), json=body)
        if response.status_code >= HTTP_OK_LIMIT:
            raise ReplayError(f"R2 Worker {path} failed with HTTP {response.status_code}")
        return record(response.json())

    def wait_ready(self) -> None:
        """Only the Worker's own answer counts: the edge returns generic 404s before it is live."""
        for _attempt in range(READY_ATTEMPTS):
            try:
                response = self.client.get(f"{self.url}/ready", headers=self._headers())
            except httpx.HTTPError:
                response = None
            if response is not None and _is_ready(response):
                return
            self.sleep(RETRY_DELAY_SECONDS)
        raise ReplayError("R2 Worker did not become ready")

    def head(self, key: str) -> RemoteObject | None:
        response = self.client.get(f"{self.url}/head", params={"key": key}, headers=self._headers())
        if response.status_code == HTTP_NOT_FOUND:
            return None
        if response.status_code >= HTTP_OK_LIMIT:
            raise ReplayError(f"R2 Worker /head failed with HTTP {response.status_code}")
        return _remote(response.json())

    def _send(self, key: str, upload_id: str, part: Part, path: Path) -> dict[str, object]:
        with path.open("rb") as stream:
            chunk = os.pread(stream.fileno(), part.length, part.offset)
        if len(chunk) != part.length:
            raise ReplayError(f"Short read of part {part.number} of {path.name}")
        expected = hashlib.md5(chunk, usedforsecurity=False).hexdigest()
        params = {"key": key, "uploadId": upload_id, "n": str(part.number)}
        for _attempt in range(PART_ATTEMPTS):
            try:
                response = self.client.put(
                    f"{self.url}/part", params=params, headers=self._headers(), content=chunk
                )
            except httpx.HTTPError:
                response = None
            if response is not None and response.status_code < HTTP_OK_LIMIT:
                sent = record(response.json())
                if str(sent.get("etag", "")).strip('"') == expected:
                    return {"partNumber": part.number, "etag": sent["etag"]}
            self.sleep(RETRY_DELAY_SECONDS)
        raise ReplayError(f"Part {part.number} of {path.name} failed or did not match its MD5")

    def put(self, key: str, path: Path, metadata: Mapping[str, str]) -> None:
        body = {"key": key, "type": OBJECT_TYPE, "metadata": dict(metadata)}
        upload_id = self._post("/create", body).get("uploadId")
        if not isinstance(upload_id, str) or not upload_id:
            raise ReplayError("R2 Worker returned no upload ID")
        try:
            with ThreadPoolExecutor(max_workers=PARALLEL_PARTS) as pool:
                sent = list(
                    pool.map(
                        lambda part: self._send(key, upload_id, part, path),
                        parts(path.stat().st_size),
                    )
                )
            self._post("/complete", {"key": key, "uploadId": upload_id, "parts": sent})
        except (ReplayError, httpx.HTTPError) as error:
            try:
                self._post("/abort", {"key": key, "uploadId": upload_id})
            except (ReplayError, httpx.HTTPError) as aborted:
                raise ReplayError(
                    f"Upload of {path.name} failed and its abort failed: {aborted}"
                ) from error
            raise

    def get(self, key: str, destination: Path) -> int:
        """Stream an object into a new file and return its byte count; never overwrites."""
        written = 0
        url = f"{self.url}/get"
        with self.client.stream(
            "GET", url, params={"key": key}, headers=self._headers()
        ) as response:
            if response.status_code >= HTTP_OK_LIMIT:
                raise ReplayError(f"R2 Worker /get failed with HTTP {response.status_code}")
            with destination.open("xb") as stream:
                for block in response.iter_bytes(DOWNLOAD_BLOCK_BYTES):
                    stream.write(block)
                    written += len(block)
                stream.flush()
                os.fsync(stream.fileno())
        return written


@contextmanager
def session(
    bucket: str,
    *,
    runner: Runner = run_cf,
    client: httpx.Client | None = None,
    sleep: Sleep = time.sleep,
    root: Path = PROJECT_ROOT,
) -> Iterator[WorkerStore]:
    """A ready Worker bound to `bucket`; deleted on every exit path."""
    name = f"{NAME_PREFIX}{secrets.token_hex(4)}"
    project = Project(root / name, name, bucket)
    token = secrets.token_urlsafe(32)
    owned = client is None
    http = client if client is not None else httpx.Client(timeout=HTTP_TIMEOUT, trust_env=False)
    try:
        prepare(project, token)
        store = WorkerStore(http, deploy(project, runner), token, sleep)
        store.wait_ready()
        yield store
    finally:
        remove(project, runner)
        if owned:
            http.close()
