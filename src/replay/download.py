"""Scoped streaming downloads; HLS is localized before invoking ffmpeg."""

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from urllib.parse import urljoin, urlsplit

import httpx

from replay.core import Candidate, ReplayError, Scope
from replay.media import quicktime_video_tags

MAX_PLAYLIST_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5


class Downloader:
    def __init__(
        self,
        *,
        client: httpx.Client,
        scope: Scope,
        candidate: Candidate,
        max_bytes: int = 20 * 1024**3,
        workers: int = 6,
    ) -> None:
        if max_bytes <= 0:
            raise ReplayError("max_bytes must be positive.")
        if not 1 <= workers <= 16:
            raise ReplayError("workers must be between 1 and 16.")
        self.client = client
        self.scope = scope
        self.candidate = candidate
        self.remaining = max_bytes
        self.workers = workers
        self.budget_lock = Lock()

    @contextmanager
    def response(self, url: str) -> Iterator[httpx.Response]:
        for _ in range(MAX_REDIRECTS + 1):
            self.scope.check(url)
            same_origin = urlsplit(url).netloc == urlsplit(self.candidate.url).netloc
            headers = (
                self.candidate.headers
                if same_origin
                else {k: v for k, v in self.candidate.headers.items() if k == "user-agent"}
            )
            # Build directly to avoid client cookie-jar credentials crossing origins.
            request = httpx.Request("GET", url, headers=headers)
            response = self.client.send(request, stream=True, follow_redirects=False)
            try:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if location is None:
                        raise ReplayError("Redirect without a destination.")
                    url = urljoin(url, location)
                    continue
                if response.status_code in (401, 403, 410):
                    raise ReplayError(
                        "Access denied or URL expired; replay in the app and recapture."
                    )
                if response.status_code != 200:
                    raise ReplayError(
                        f"Expected full HTTP 200 response, got {response.status_code}."
                    )
                yield response
                return
            finally:
                response.close()
        raise ReplayError("Too many redirects.")

    def save_stream(self, url: str, path: Path, *, direct: bool = False) -> None:
        with self.response(url) as response:
            if direct:
                mime = response.headers.get("content-type", "").lower()
                if "text/" in mime or "json" in mime or "mpegurl" in mime:
                    raise ReplayError(
                        "Server returned non-video content; capture the replay again."
                    )
            with path.open("xb") as stream:
                path.chmod(0o600)
                for chunk in response.iter_bytes():
                    with self.budget_lock:
                        self.remaining -= len(chunk)
                        if self.remaining < 0:
                            raise ReplayError("Download exceeded --max-gib limit.")
                    stream.write(chunk)
            if path.stat().st_size == 0:
                raise ReplayError("Server returned an empty video.")

    def playlist(self, url: str) -> tuple[str, str]:
        with self.response(url) as response:
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > MAX_PLAYLIST_BYTES:
                    raise ReplayError("Playlist too large.")
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError as error:
                raise ReplayError("Playlist is not UTF-8.") from error
            if not text.startswith("#EXTM3U"):
                raise ReplayError("Invalid HLS playlist.")
            return text, str(response.url)

    def fetch_segments(self, jobs: list[tuple[str, Path]]) -> None:
        for job in jobs:
            self.scope.check(job[0])
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self.save_stream, url, path) for url, path in jobs]
            try:
                for future in as_completed(futures):
                    future.result()
            finally:
                for future in futures:
                    future.cancel()

    def hls(self, url: str, directory: Path, output: Path) -> None:
        if shutil.which("ffmpeg") is None:
            raise ReplayError("ffmpeg is required for HLS; install it first.")
        text, base = self.playlist(url)
        if "#EXT-X-STREAM-INF:" in text:
            if re.search(r"#EXT-X-MEDIA:.*TYPE=(AUDIO|VIDEO)", text):
                raise ReplayError("Separate HLS renditions are not supported in this version.")
            variants = re.findall(r"#EXT-X-STREAM-INF:([^\n]+)\n([^#\s][^\n]*)", text)
            if not variants:
                raise ReplayError("Invalid HLS master playlist.")
            _, uri = max(variants, key=variant_bandwidth)
            text, base = self.playlist(urljoin(base, uri.strip()))
        if "#EXT-X-ENDLIST" not in text:
            raise ReplayError("Not a completed VOD playlist; refusing an open-ended LIVE download.")
        unsupported = (
            "#EXT-X-BYTERANGE",
            "#EXT-X-KEY",
            "#EXT-X-SESSION-KEY",
            "#EXT-X-PART",
            "#EXT-X-STREAM-INF",
            "#EXT-X-GAP",
            "#EXT-X-DEFINE",
        )
        if any(tag in text for tag in unsupported):
            raise ReplayError("Encrypted, ranged, nested or incomplete HLS is not supported.")
        local_lines: list[str] = []
        jobs: list[tuple[str, Path]] = []
        segments = 0
        segment_extension = "ts"
        for index, line in enumerate(text.splitlines()):
            line = line.strip()
            if line.startswith("#EXT-X-MAP:"):
                match = re.search(r'URI="([^"]+)"', line)
                if match is None or "BYTERANGE" in line:
                    raise ReplayError("Unsupported HLS initialization segment.")
                segment_extension = "m4s"
                name = f"init-{index}.mp4"
                jobs.append((urljoin(base, match[1]), directory / name))
                local_lines.append(f'#EXT-X-MAP:URI="{name}"')
            elif line and not line.startswith("#"):
                name = f"segment-{index}.{segment_extension}"
                jobs.append((urljoin(base, line), directory / name))
                local_lines.append(name)
                segments += 1
            elif line.startswith(
                (
                    "#EXTM3U",
                    "#EXTINF:",
                    "#EXT-X-TARGETDURATION:",
                    "#EXT-X-MEDIA-SEQUENCE:",
                    "#EXT-X-VERSION:",
                    "#EXT-X-DISCONTINUITY",
                    "#EXT-X-ENDLIST",
                )
            ):
                local_lines.append(line)
        if segments == 0:
            raise ReplayError("HLS contains no segments.")
        self.fetch_segments(jobs)
        playlist = directory / "local.m3u8"
        playlist.write_text("\n".join(local_lines) + "\n", encoding="utf-8")
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-protocol_whitelist",
            "file,crypto,data",
            "-allowed_extensions",
            "ALL",
            "-i",
            str(playlist),
            "-map",
            "0:v?",
            "-map",
            "0:a?",
            "-c",
            "copy",
            *quicktime_video_tags(playlist),
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=1800)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ReplayError(
                "Local ffmpeg remux failed or timed out; output was not published."
            ) from error

    def download(self, output: Path) -> None:
        if self.candidate.kind == "replay-link":
            raise ReplayError("This is an application/page navigation link, not a video URL.")
        if output.exists():
            raise ReplayError("Output already exists; choose a new filename.")
        if not output.parent.is_dir():
            raise ReplayError("Output directory does not exist.")
        self.scope.check(self.candidate.url)
        with tempfile.TemporaryDirectory(prefix=".replay-", dir=output.parent) as temporary:
            directory = Path(temporary)
            result = directory / "result"
            if self.candidate.kind == "hls":
                self.hls(self.candidate.url, directory, result)
            else:
                self.save_stream(self.candidate.url, result, direct=True)
            if not result.exists() or result.stat().st_size == 0:
                raise ReplayError("No video was produced.")
            result.chmod(0o600)
            # A hard link atomically publishes the completed file without overwriting.
            os.link(result, output)


def variant_bandwidth(variant: tuple[str, str]) -> int:
    match = re.search(r"(?:^|,)BANDWIDTH=(\d+)", variant[0])
    return int(match[1]) if match is not None else 0
