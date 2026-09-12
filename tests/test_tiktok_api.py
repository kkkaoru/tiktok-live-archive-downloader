import json
from unittest.mock import Mock

import httpx
import pytest

from replay.core import ReplayError
from replay.recordings import Notice
from replay.tiktok_api import Credentials, TikTokAPI


def test_validate_session_reads_one_page_without_resolving() -> None:
    handler = Mock(
        return_value=httpx.Response(
            200,
            json={
                "status_code": 0,
                "notice_list": {
                    "notice_list": [],
                    "has_more": 1,
                    "max_time": 10,
                },
            },
        )
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        TikTokAPI(client=client, credentials=Credentials("a" * 32)).validate_session()
    assert handler.call_count == 1
    assert handler.call_args.args[0].url.path == "/tiktok/notice/system_notice_box/v1/"


def test_environment_credentials_are_private() -> None:
    credentials = Credentials.from_environment(
        {"TIKTOK_SESSIONID": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
    )
    assert repr(credentials) == "Credentials()"


@pytest.mark.parametrize("value", ["", "short", "a" * 31 + "\n"])
def test_bad_credentials(value: str) -> None:
    with pytest.raises(ReplayError, match="TIKTOK_SESSIONID"):
        Credentials.from_environment({"TIKTOK_SESSIONID": value})


def test_notification_pagination_and_creator_filter() -> None:
    handler = Mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "status_code": 0,
                    "notice_list": {
                        "notice_list": [
                            {
                                "template_notice": {
                                    "schema_url": "sslocal://webcast_replay_video?roomId=1&anchor_id=99&user_type=1"
                                }
                            },
                            {
                                "template_notice": {
                                    "schema_url": "sslocal://webcast_replay_video?roomId=3&anchor_id=77&user_type=1"
                                }
                            },
                        ],
                        "has_more": 1,
                        "max_time": 10,
                    },
                },
            ),
            httpx.Response(
                200,
                json={
                    "status_code": 0,
                    "notice_list": {
                        "notice_list": [
                            {
                                "template_notice": {
                                    "schema_url": "sslocal://webcast_replay_video?roomId=1&anchor_id=99&user_type=1"
                                }
                            },
                            {
                                "template_notice": {
                                    "schema_url": "sslocal://webcast_replay_video?roomId=2&anchor_id=99&user_type=1"
                                }
                            },
                        ],
                        "has_more": 0,
                    },
                },
            ),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        notices = list(api.notices(anchor_id="99"))
    assert notices == [
        Notice(replay_id="1", anchor_id="99", user_type="1"),
        Notice(replay_id="2", anchor_id="99", user_type="1"),
    ]
    request = handler.call_args_list[1].args[0]
    assert request.url.params["version_code"] == "460942"
    assert json.loads(request.url.params["group"])["max_time"] == 10
    assert json.loads(request.url.params["group"])["is_mark_read"] == 0
    assert request.headers["cookie"] == "sessionid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_empty_page() -> None:
    handler = Mock(
        return_value=httpx.Response(
            200, json={"status_code": 0, "notice_list": {"notice_list": None, "has_more": 0}}
        )
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        assert list(api.notices(anchor_id="99")) == []


@pytest.mark.parametrize(
    "page",
    [{"notice_list": [], "has_more": "yes"}, {"notice_list": [], "has_more": 1, "max_time": 0}],
)
def test_invalid_pagination(page: object) -> None:
    handler = Mock(return_value=httpx.Response(200, json={"status_code": 0, "notice_list": page}))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        with pytest.raises(ReplayError):
            list(api.notices(anchor_id="99"))


def test_repeated_cursor() -> None:
    handler = Mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "status_code": 0,
                    "notice_list": {"notice_list": [], "has_more": 1, "max_time": 10},
                },
            ),
            httpx.Response(
                200,
                json={
                    "status_code": 0,
                    "notice_list": {"notice_list": [], "has_more": 1, "max_time": 10},
                },
            ),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        with pytest.raises(ReplayError, match="did not advance"):
            list(api.notices(anchor_id="99"))


def test_page_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("replay.tiktok_api.MAX_NOTICE_PAGES", 1)
    handler = Mock(
        return_value=httpx.Response(
            200,
            json={
                "status_code": 0,
                "notice_list": {"notice_list": [], "has_more": 1, "max_time": 10},
            },
        )
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        with pytest.raises(ReplayError, match="page limit"):
            list(api.notices(anchor_id="99"))


def test_resolve() -> None:
    handler = Mock(
        return_value=httpx.Response(200, json={"status_code": 0, "data": {"replays": []}})
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        result = api.resolve(Notice(replay_id="1", anchor_id="99", user_type="1"))
    assert result.available is False
    assert dict(handler.call_args.args[0].url.params) == {
        "aid": "1233",
        "room_ids": "1",
        "user_type": "1",
        "need_share": "false",
    }


@pytest.mark.parametrize(
    "body", [b"bad json", b"\xff", b"{}", b'{"status_code":20003}', b'{"status_code":1}']
)
def test_api_error_data(body: bytes) -> None:
    handler = Mock(return_value=httpx.Response(200, content=body))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        with pytest.raises(ReplayError):
            list(api.notices(anchor_id="99"))


def test_redirect_never_receives_credentials() -> None:
    handler = Mock(return_value=httpx.Response(302, headers={"Location": "https://evil.test/"}))
    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        with pytest.raises(ReplayError, match="redirects are not followed"):
            list(api.notices(anchor_id="99"))
    assert handler.call_count == 1


def test_network_error_is_redacted() -> None:
    handler = Mock(side_effect=httpx.ConnectError("secret URL"))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        with pytest.raises(ReplayError, match="network error") as caught:
            list(api.notices(anchor_id="99"))
    assert "secret URL" not in str(caught.value)


def test_bounded_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("replay.tiktok_api.MAX_API_BYTES", 4)
    handler = Mock(return_value=httpx.Response(200, content=b"12345"))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        with pytest.raises(ReplayError, match="size limit"):
            list(api.notices(anchor_id="99"))


def test_api_host_is_fixed() -> None:
    with httpx.Client() as client:
        api = TikTokAPI(client=client, credentials=Credentials("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        with pytest.raises(ReplayError, match="not allowlisted"):
            api._get("https://evil.test/", {})
