import json
from pathlib import Path

import pytest

from replay.core import (
    Candidate,
    ReplayError,
    Scope,
    candidate_from_request,
    import_har,
    load_candidate,
    media_kind,
    save_candidate,
    write_private_text,
)


def test_candidate_repr_redacts_credentials_and_signed_urls() -> None:
    candidate = Candidate(
        "https://cdn.test/video?signature=private", {"authorization": "Bearer private"}, "hls"
    )
    assert repr(candidate) == "Candidate(kind='hls', source='request')"


def test_atomic_text_is_private(tmp_path: Path) -> None:
    output = tmp_path / "token"
    write_private_text(output, "credential")
    assert output.read_text(encoding="utf-8") == "credential"
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.test/a",
        "https://other.test/a",
        "https://u:p@cdn.test/a",
        "https://cdn.test:444/a",
    ],
)
def test_scope_rejects(url: str) -> None:
    with pytest.raises(ReplayError):
        Scope(("cdn.test",)).check(url)


def test_scope_accepts() -> None:
    Scope(("cdn.test",)).check("https://cdn.test/a?secret=x")


@pytest.mark.parametrize(
    ("url", "mime", "expected"),
    [
        ("https://cdn.test/a.m3u8", "", "hls"),
        ("https://cdn.test/a", "application/vnd.apple.mpegurl", "hls"),
        ("https://cdn.test/a", "video/mp4; codecs=x", "video"),
        ("https://cdn.test/a.flv", "", "video"),
        ("https://cdn.test/a.ts", "video/mp2t", None),
    ],
)
def test_media_kind(url: str, mime: str, expected: str | None) -> None:
    assert media_kind(url, mime) == expected


def test_private_roundtrip(tmp_path: Path) -> None:
    candidate = Candidate("https://cdn.test/a?secret=x", {"cookie": "session=x"}, "video")
    root = tmp_path / "private"
    assert save_candidate(root, candidate)
    assert not save_candidate(root, candidate)
    path = next(root.glob("*.json"))
    assert path.stat().st_mode & 0o777 == 0o600
    assert root.stat().st_mode & 0o777 == 0o700
    assert load_candidate(path) == candidate
    assert "secret" not in candidate.label
    assert "session" not in candidate.label
    assert "cdn.test" in candidate.label


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"url": "https://cdn.test/a", "kind": "video", "headers": {"x": 2}},
        {"url": "https://cdn.test/a", "kind": "video", "headers": {"cookie": "a\r\nx:b"}},
    ],
)
def test_invalid_candidate(tmp_path: Path, data: object) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        load_candidate(path)


def test_invalid_source(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text(
        '{"url":"https://cdn.test/a","kind":"video","headers":{},"source":"bad"}',
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="source"):
        load_candidate(path)


def test_drop_unneeded_header(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text(
        '{"url":"https://cdn.test/a","kind":"video","headers":{"X-Ignore":"secret","Cookie":"ok"}}',
        encoding="utf-8",
    )
    assert load_candidate(path).headers == {"cookie": "ok"}


@pytest.mark.parametrize(
    ("url", "status", "method"),
    [
        ("https://other.test/a.mp4", 200, "GET"),
        ("https://cdn.test:invalid/a.mp4", 200, "GET"),
        ("https://cdn.test/a.mp4", 403, "GET"),
        ("https://cdn.test/a.mp4", 200, "POST"),
        ("https://cdn.test/api", 200, "GET"),
    ],
)
def test_candidate_filter(url: str, status: int, method: str) -> None:
    assert (
        candidate_from_request(
            url=url,
            headers={},
            content_type="",
            status=status,
            method=method,
            scope=Scope(("cdn.test",)),
        )
        is None
    )


def test_import(tmp_path: Path) -> None:
    path = tmp_path / "source.har"
    path.write_text(
        json.dumps(
            {
                "log": {
                    "entries": [
                        {
                            "request": {
                                "url": "https://cdn.test/a.mp4",
                                "method": "GET",
                                "headers": [
                                    {"name": "Cookie", "value": "secret"},
                                    {"name": "Range", "value": "bytes=0-9"},
                                    {"name": 7},
                                    None,
                                ],
                            },
                            "response": {"status": 206, "content": {"mimeType": "video/mp4"}},
                        },
                        None,
                        {},
                        {"request": {}, "response": {}},
                        {
                            "request": {
                                "url": "https://cdn.test/b.mp4",
                                "method": "GET",
                                "headers": None,
                            },
                            "response": {"status": 200, "content": []},
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert import_har(path, root=tmp_path / "store", scope=Scope(("cdn.test",))) == 2
    assert import_har(path, root=tmp_path / "store", scope=Scope(("cdn.test",))) == 0


@pytest.mark.parametrize("data", [[], {}, {"log": {}}, {"log": {"entries": 1}}])
def test_invalid_har(tmp_path: Path, data: object) -> None:
    path = tmp_path / "bad.har"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        import_har(path, root=tmp_path / "store", scope=Scope(("cdn.test",)))
