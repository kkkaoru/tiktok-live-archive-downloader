import json
import sys
import urllib.error
import urllib.request

import pytest

from replay import youtube_channels
from replay.core import ReplayError
from replay.recording_viewer_steps import Fetch

KASUMI = "UCEmlNV5-edGhNWYDMeGDgWw"


@pytest.fixture(autouse=True)
def viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(youtube_channels, "_viewer", lambda: ("viewer.test", "t" * 40))


def _recording() -> tuple[list[urllib.request.Request], Fetch]:
    seen: list[urllib.request.Request] = []

    def fetch(request: urllib.request.Request) -> dict[str, object]:
        seen.append(request)
        return {"version": 1, "channels": {}}

    return seen, fetch


def test_list_reads_with_the_bearer() -> None:
    seen, fetch = _recording()
    assert youtube_channels.request(["list"], fetch=fetch) == {"version": 1, "channels": {}}
    assert [seen[0].get_method(), seen[0].full_url, seen[0].data] == [
        "GET",
        "https://viewer.test/api/channels.json",
        None,
    ]
    assert seen[0].get_header("Authorization") == "Bearer " + "t" * 40
    assert seen[0].get_header("User-agent") == "replay-recording-pipeline/1"


def test_set_and_remove_send_one_channel() -> None:
    seen, fetch = _recording()
    youtube_channels.request(["set", KASUMI, "熊童子かすみ"], fetch=fetch)
    youtube_channels.request(["remove", KASUMI], fetch=fetch)
    assert [request.get_method() for request in seen] == ["PUT", "DELETE"]
    assert (
        seen[0].data
        == json.dumps({"channelId": KASUMI, "label": "熊童子かすみ"}, ensure_ascii=False).encode()
    )
    assert seen[1].data == b'{"channelId": "UCEmlNV5-edGhNWYDMeGDgWw"}'
    assert seen[1].get_header("Content-type") == "application/json"


def test_unknown_actions_stop_before_any_request() -> None:
    seen, fetch = _recording()
    with pytest.raises(ReplayError, match="Usage"):
        youtube_channels.request(["set", KASUMI], fetch=fetch)
    assert seen == []


def test_main_prints_the_record(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        youtube_channels,
        "_fetch",
        lambda request: {"version": 1, "channels": {KASUMI: {"label": "熊童子かすみ"}}},
    )
    monkeypatch.setattr(sys, "argv", ["youtube_channels", "list"])
    youtube_channels.main()
    assert json.loads(capsys.readouterr().out) == {
        "version": 1,
        "channels": {KASUMI: {"label": "熊童子かすみ"}},
    }


def test_main_reports_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: urllib.request.Request) -> dict[str, object]:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(youtube_channels, "_fetch", refuse)
    monkeypatch.setattr(sys, "argv", ["youtube_channels", "list"])
    with pytest.raises(SystemExit, match="refused"):
        youtube_channels.main()
