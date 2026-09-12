import subprocess
from collections.abc import Iterator
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import httpx
import pytest

from replay.core import Candidate, ReplayError, Scope
from replay.download import MAX_PLAYLIST_BYTES, Downloader, variant_bandwidth


@pytest.fixture
def client() -> Iterator[httpx.Client]:
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"video"))
    ) as session:
        yield session


def test_navigation_link(tmp_path: Path, client: httpx.Client) -> None:
    downloader = Downloader(
        client=client,
        scope=Scope(("cdn.test",)),
        candidate=Candidate("sslocal://replay?id=1", {}, "replay-link"),
    )
    with pytest.raises(ReplayError, match="navigation link"):
        downloader.download(tmp_path / "out.mp4")


def test_direct(tmp_path: Path, client: httpx.Client) -> None:
    downloader = Downloader(
        client=client,
        scope=Scope(("cdn.test",)),
        candidate=Candidate("https://cdn.test/a", {}, "video"),
    )
    downloader.download(tmp_path / "out.mp4")
    assert (tmp_path / "out.mp4").read_bytes() == b"video"
    assert (tmp_path / "out.mp4").stat().st_mode & 0o777 == 0o600
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ReplayError, match="already exists"):
        downloader.download(tmp_path / "out.mp4")
    with pytest.raises(ReplayError, match="directory"):
        downloader.download(tmp_path / "missing" / "out.mp4")


def test_limit(tmp_path: Path, client: httpx.Client) -> None:
    downloader = Downloader(
        client=client,
        scope=Scope(("cdn.test",)),
        candidate=Candidate("https://cdn.test/a", {}, "video"),
        max_bytes=2,
    )
    with pytest.raises(ReplayError, match="limit"):
        downloader.download(tmp_path / "out.mp4")
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ReplayError, match="positive"):
        Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/a", {}, "video"),
            max_bytes=0,
        )


@pytest.mark.parametrize("status", [206, 401, 403, 410, 500])
def test_http_errors(tmp_path: Path, status: int) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status))) as client:
        downloader = Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/a?secret=x", {}, "video"),
        )
        with pytest.raises(ReplayError) as error:
            downloader.download(tmp_path / "out.mp4")
        assert "secret" not in str(error.value)
        assert not (tmp_path / "out.mp4").exists()


def test_redirect_credentials(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "cdn.test":
            return httpx.Response(302, headers={"location": "https://second.test/b"})
        return httpx.Response(200, content=b"ok")

    with httpx.Client(transport=httpx.MockTransport(handle), cookies={"bad": "jar"}) as client:
        Downloader(
            client=client,
            scope=Scope(("cdn.test", "second.test")),
            candidate=Candidate(
                "https://cdn.test/a",
                {
                    "cookie": "secret",
                    "authorization": "Bearer secret",
                    "referer": "https://cdn.test/private",
                    "user-agent": "test",
                },
                "video",
            ),
        ).download(tmp_path / "out")
    assert requests[0].headers["cookie"] == "secret"
    assert requests[1].headers.get("cookie") is None
    assert requests[1].headers.get("authorization") is None
    assert requests[1].headers.get("referer") is None
    assert requests[1].headers["user-agent"] == "test"


@pytest.mark.parametrize(
    ("location", "message"),
    [("https://evil.test/a", "scope"), ("https://cdn.test/a", "Too many"), (None, "destination")],
)
def test_bad_redirect(tmp_path: Path, location: str | None, message: str) -> None:
    headers = {} if location is None else {"location": location}
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(302, headers=headers))
    ) as client:
        downloader = Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/a", {}, "video"),
        )
        with pytest.raises(ReplayError, match=message):
            downloader.download(tmp_path / "out")


@pytest.mark.parametrize(
    ("body", "mime", "message"), [(b"", "video/mp4", "empty"), (b"login", "text/html", "non-video")]
)
def test_bad_direct(tmp_path: Path, body: bytes, mime: str, message: str) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=body, headers={"content-type": mime})
        )
    ) as client:
        downloader = Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/a", {}, "video"),
        )
        with pytest.raises(ReplayError, match=message):
            downloader.download(tmp_path / "out")


@pytest.mark.parametrize(
    ("body", "message"),
    [(b"invalid", "Invalid"), (b"\xff", "UTF-8"), (b"x" * (MAX_PLAYLIST_BYTES + 1), "too large")],
)
def test_bad_playlist(body: bytes, message: str) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    ) as client:
        downloader = Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/a", {}, "hls"),
        )
        with pytest.raises(ReplayError, match=message):
            downloader.playlist("https://cdn.test/a")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("#EXTM3U\n#EXTINF:1,\na.ts", "completed VOD"),
        ("#EXTM3U\n#EXT-X-KEY:METHOD=AES-128\n#EXT-X-ENDLIST", "Encrypted"),
        ("#EXTM3U\n#EXT-X-MAP:BYTERANGE=1\n#EXT-X-ENDLIST", "initialization"),
        ("#EXTM3U\n#EXT-X-ENDLIST", "no segments"),
        ('#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,URI="a"\n#EXT-X-STREAM-INF:BANDWIDTH=1\nb', "Separate"),
        ("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\n#bad", "master"),
    ],
)
def test_unsupported_hls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, message: str
) -> None:
    monkeypatch.setattr("replay.download.shutil.which", lambda name: "/usr/bin/ffmpeg")
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=text))
    ) as client:
        downloader = Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/a", {}, "hls"),
        )
        with pytest.raises(ReplayError, match=message):
            downloader.download(tmp_path / "out")
    assert list(tmp_path.iterdir()) == []


def test_hls_localization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        data = {
            "/master": (
                "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nlow\n"
                "#EXT-X-STREAM-INF:BANDWIDTH=2\nsub/high\n"
            ),
            "/sub/high": (
                '#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-TARGETDURATION:1\n#EXT-X-MAP:URI="init"\n'
                "#EXTINF:1,\na.m4s\n#EXT-X-DISCONTINUITY\n#EXTINF:1,\nb.m4s\n#EXT-X-ENDLIST\n"
            ),
        }
        return httpx.Response(200, content=data.get(request.url.path, "bytes").encode())

    def remux(command: list[str], **kwargs: object) -> None:
        local = Path(command[command.index("-i") + 1])
        assert "https://" not in local.read_text(encoding="utf-8")
        assert (local.parent / "init-3.mp4").read_bytes() == b"bytes"
        assert kwargs == {"check": True, "capture_output": True, "timeout": 1800}
        assert command[command.index("-tag:v:0") + 1] == "hvc1"
        Path(command[-1]).write_bytes(b"mp4")

    monkeypatch.setattr("replay.download.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("replay.download.subprocess.run", remux)
    monkeypatch.setattr("replay.download.quicktime_video_tags", lambda path: ["-tag:v:0", "hvc1"])
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/master", {}, "hls"),
        ).download(tmp_path / "out.mp4")
    assert (tmp_path / "out.mp4").read_bytes() == b"mp4"
    assert requests[:2] == ["https://cdn.test/master", "https://cdn.test/sub/high"]
    assert sorted(requests[2:]) == [
        "https://cdn.test/sub/a.m4s",
        "https://cdn.test/sub/b.m4s",
        "https://cdn.test/sub/init",
    ]
    assert variant_bandwidth(("CODECS=abc", "x")) == 0


@pytest.mark.parametrize(
    "failure",
    [subprocess.CalledProcessError(1, "ffmpeg"), subprocess.TimeoutExpired("ffmpeg", 1800)],
)
def test_ffmpeg_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    monkeypatch.setattr("replay.download.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("replay.download.subprocess.run", Mock(side_effect=failure))
    monkeypatch.setattr("replay.download.quicktime_video_tags", lambda path: [])
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text="#EXTM3U\n#EXTINF:1,\na.ts\n#EXT-X-ENDLIST")
        )
    ) as client:
        downloader = Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/a", {}, "hls"),
        )
        with pytest.raises(ReplayError, match="remux failed"):
            downloader.download(tmp_path / "out")


def test_missing_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: httpx.Client
) -> None:
    monkeypatch.setattr("replay.download.shutil.which", lambda name: None)
    downloader = Downloader(
        client=client,
        scope=Scope(("cdn.test",)),
        candidate=Candidate("https://cdn.test/a", {}, "hls"),
    )
    with pytest.raises(ReplayError, match="ffmpeg is required"):
        downloader.download(tmp_path / "out")


def test_parallel_segments(tmp_path: Path) -> None:
    barrier = Barrier(2)

    def respond(request: httpx.Request) -> httpx.Response:
        barrier.wait(timeout=3)
        return httpx.Response(200, content=request.url.path.encode())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        downloader = Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/list", {}, "hls"),
            workers=2,
        )
        downloader.fetch_segments(
            [("https://cdn.test/a", tmp_path / "a"), ("https://cdn.test/b", tmp_path / "b")]
        )
    assert (tmp_path / "a").read_bytes() == b"/a"
    assert (tmp_path / "b").read_bytes() == b"/b"


@pytest.mark.parametrize("workers", [0, 17])
def test_bad_workers(client: httpx.Client, workers: int) -> None:
    with pytest.raises(ReplayError, match="workers"):
        Downloader(
            client=client,
            scope=Scope(("cdn.test",)),
            candidate=Candidate("https://cdn.test/list", {}, "hls"),
            workers=workers,
        )


def test_no_remux_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: httpx.Client
) -> None:
    monkeypatch.setattr(Downloader, "hls", lambda *args: None)
    downloader = Downloader(
        client=client,
        scope=Scope(("cdn.test",)),
        candidate=Candidate("https://cdn.test/a", {}, "hls"),
    )
    with pytest.raises(ReplayError, match="No video"):
        downloader.download(tmp_path / "out")
