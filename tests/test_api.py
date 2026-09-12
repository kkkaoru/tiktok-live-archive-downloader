import gzip
import json
from pathlib import Path

import pytest

from replay.api import MAX_API_BYTES, advertised_media, api_stream
from replay.core import load_candidate


def test_embedded_urls() -> None:
    result = advertised_media(
        {
            "payload": '{"action":"snssdk1233://replay?url=https%3A%2F%2Fcdn.test%2Fa.mp4%3Ftoken%3Dx"}'
        }
    )
    assert len(result) == 2
    assert result[0].kind == "replay-link"
    assert result[1].url == "https://cdn.test/a.mp4?token=x"


def test_advertised() -> None:
    result = advertised_media(
        {
            "data": [
                "https://cdn.test/a.mp4?token=x",
                "https://cdn.test/a.mp4?token=x",
                "http://cdn.test/a.mp4",
                "https://cdn.test:bad/a.mp4",
                "https://cdn.test/image.jpg",
                "text",
                "sslocal://unrelated",
            ]
        }
    )
    assert len(result) == 1
    assert result[0].source == "api-advertised"
    assert result[0].headers == {}


@pytest.mark.parametrize("encoding", ["", "gzip"])
def test_stream(tmp_path: Path, encoding: str) -> None:
    messages: list[str] = []
    body = json.dumps({"play": "https://cdn.test/a.mp4?token=x"}).encode()
    if encoding == "gzip":
        body = gzip.compress(body)
    stream = api_stream(root=tmp_path, encoding=encoding, report=messages.append)
    assert stream(body) == body
    assert stream(b"") == b""
    candidate = load_candidate(next(tmp_path.glob("*.json")))
    assert candidate.url == "https://cdn.test/a.mp4?token=x"
    assert "token" not in " ".join(messages)
    stream(body)
    stream(b"")
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize(
    ("body", "encoding", "message"),
    [
        (b"x" * (MAX_API_BYTES + 1), "", "exceeded"),
        (b"invalid", "", "not readable"),
        (b"invalid", "br", "unsupported"),
        (b"invalid", "gzip", "not readable"),
        (gzip.compress(b"x" * (MAX_API_BYTES + 1)), "gzip", "decompression limit"),
    ],
)
def test_reject(tmp_path: Path, body: bytes, encoding: str, message: str) -> None:
    messages: list[str] = []
    stream = api_stream(root=tmp_path, encoding=encoding, report=messages.append)
    stream(body)
    stream(b"")
    assert message in messages[0]
    assert list(tmp_path.iterdir()) == []
