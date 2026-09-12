from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mitmproxy import http
from mitmproxy.test import tflow

from replay.addon import ReplayCapture
from replay.core import load_candidate


def test_tls_failure(capsys: pytest.CaptureFixture[str]) -> None:
    data = Mock()
    data.conn.sni = "api.test"
    ReplayCapture().tls_failed_client(data)
    assert capsys.readouterr().out == "Client TLS failed: api.test; check certificate trust.\n"


def test_load() -> None:
    loader = Mock()
    ReplayCapture().load(loader)
    assert loader.add_option.call_count == 3


def test_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "replay.addon.ctx.options",
        SimpleNamespace(
            replay_hosts="cdn.test", replay_store=str(tmp_path / "store"), replay_api_hosts=""
        ),
        raising=False,
    )
    flow = tflow.tflow(resp=True)
    flow.request = http.Request.make(
        "GET",
        "https://cdn.test/replay.mp4?secret=x",
        headers=http.Headers(cookie="secret", range="bytes=0-9"),
    )
    flow.response = http.Response.make(206, headers={"Content-Type": "video/mp4"})
    addon = ReplayCapture()
    addon.responseheaders(flow)
    addon.responseheaders(flow)
    addon.http_connect(flow)
    assert flow.response.stream is True
    assert len(list((tmp_path / "store").iterdir())) == 1
    candidate = load_candidate(next((tmp_path / "store").glob("*.json")))
    assert candidate.headers == {"cookie": "secret"}
    output = capsys.readouterr().out
    assert output.count("Replay candidate:") == 1
    assert "secret" not in output
    flow.response = None
    addon.responseheaders(flow)


def test_discovery(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        "replay.addon.ctx.options",
        SimpleNamespace(replay_hosts="", replay_api_hosts=""),
        raising=False,
    )
    flow = tflow.tflow()
    flow.request = http.Request.make("CONNECT", "https://cdn.test:443")
    ReplayCapture().http_connect(flow)
    assert capsys.readouterr().out == "CONNECT host: cdn.test\n"


@pytest.mark.parametrize("path", ["replay", "notice/system"])
def test_api_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setattr(
        "replay.addon.ctx.options",
        SimpleNamespace(replay_hosts="", replay_api_hosts="api.test", replay_store=str(tmp_path)),
        raising=False,
    )
    flow = tflow.tflow(resp=True)
    flow.request = http.Request.make("GET", f"https://api.test/{path}/?secret=x")
    flow.response = http.Response.make(200, headers={"content-type": "application/json"})
    ReplayCapture().requestheaders(flow)
    assert flow.request.headers["accept-encoding"] == "identity"
    ReplayCapture().responseheaders(flow)
    assert callable(flow.response.stream)
    flow.response.stream(b'{"url":"https://cdn.test/a.mp4"}')
    flow.response.stream(b"")
    assert len(list(tmp_path.glob("*.json"))) == 1
