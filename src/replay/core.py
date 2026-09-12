"""Private candidate storage and explicit network scope."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

SAFE_HEADERS = frozenset({"user-agent", "referer", "origin", "cookie", "authorization"})


class ReplayError(Exception):
    """An actionable failure safe to print without exposing signed URLs."""


@dataclass(frozen=True)
class Candidate:
    url: str = field(repr=False)
    headers: dict[str, str] = field(repr=False)
    kind: str
    source: str = "request"

    @property
    def id(self) -> str:
        return hashlib.sha256(self.url.encode()).hexdigest()[:16]

    @property
    def label(self) -> str:
        return (
            f"{self.id}  {self.kind}  {urlsplit(self.url).hostname}  {self.source}  [URL redacted]"
        )


@dataclass(frozen=True)
class Scope:
    hosts: tuple[str, ...]

    def check(self, url: str) -> None:
        parts = urlsplit(url)
        if (
            parts.scheme != "https"
            or parts.username is not None
            or parts.password is not None
            or parts.hostname not in self.hosts
            or parts.port not in (None, 443)
        ):
            raise ReplayError("URL outside explicit HTTPS host scope; review --host settings.")


def media_kind(url: str, content_type: str) -> str | None:
    path = urlsplit(url).path.lower()
    mime = content_type.partition(";")[0].strip().lower()
    if path.endswith(".m3u8") or mime in (
        "application/vnd.apple.mpegurl",
        "application/x-mpegurl",
        "audio/mpegurl",
    ):
        return "hls"
    if mime in ("video/mp4", "video/x-flv", "video/quicktime"):
        return "video"
    if path.endswith((".mp4", ".flv", ".mov")):
        return "video"
    return None


def candidate_from_request(
    *,
    url: str,
    headers: dict[str, str],
    content_type: str,
    status: int,
    method: str,
    scope: Scope,
) -> Candidate | None:
    try:
        scope.check(url)
    except (ReplayError, ValueError):
        return None
    kind = media_kind(url, content_type)
    if method != "GET" or status not in (200, 206) or kind is None:
        return None
    kept = {k.lower(): v for k, v in headers.items() if k.lower() in SAFE_HEADERS}
    return Candidate(url, kept, kind)


def private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def write_private_text(path: Path, text: str) -> None:
    """Atomically replace a private file; never truncate an existing credential."""
    private_directory(path.parent)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_candidate(root: Path, candidate: Candidate) -> bool:
    private_directory(root)
    target = root / f"{candidate.id}.json"
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(asdict(candidate), stream, ensure_ascii=False)
    return True


def load_candidate(path: Path) -> Candidate:
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReplayError("Invalid candidate file.")
    url, headers, kind = data.get("url"), data.get("headers"), data.get("kind")
    if (
        not isinstance(url, str)
        or kind not in ("video", "hls", "replay-link")
        or not isinstance(headers, dict)
    ):
        raise ReplayError("Invalid candidate fields.")
    checked: dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ReplayError("Invalid candidate headers.")
        if key.lower() in SAFE_HEADERS:
            if "\r" in value or "\n" in value:
                raise ReplayError("Invalid header value.")
            checked[key.lower()] = value
    source = data.get("source", "request")
    if source not in ("request", "api-advertised"):
        raise ReplayError("Invalid candidate source.")
    return Candidate(url, checked, str(kind), str(source))


def import_har(path: Path, *, root: Path, scope: Scope) -> int:
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("log"), dict):
        raise ReplayError("Invalid HAR log.")
    entries: object = data["log"].get("entries")
    if not isinstance(entries, list):
        raise ReplayError("Invalid HAR entries.")
    return sum(_import_entry(entry, root=root, scope=scope) for entry in entries)


def _import_entry(entry: object, *, root: Path, scope: Scope) -> int:
    if not isinstance(entry, dict):
        return 0
    request, response = entry.get("request"), entry.get("response")
    if not isinstance(request, dict) or not isinstance(response, dict):
        return 0
    url, method, status = request.get("url"), request.get("method"), response.get("status")
    if not isinstance(url, str) or not isinstance(method, str) or not isinstance(status, int):
        return 0
    headers: dict[str, str] = {}
    raw_headers = request.get("headers", [])
    if isinstance(raw_headers, list):
        for header in raw_headers:
            if isinstance(header, dict):
                key, value = header.get("name"), header.get("value")
                if isinstance(key, str) and isinstance(value, str):
                    headers[key] = value
    content = response.get("content", {})
    mime = content.get("mimeType", "") if isinstance(content, dict) else ""
    candidate = candidate_from_request(
        url=url,
        method=method,
        status=status,
        headers=headers,
        content_type=mime if isinstance(mime, str) else "",
        scope=scope,
    )
    return int(candidate is not None and save_candidate(root, candidate))
