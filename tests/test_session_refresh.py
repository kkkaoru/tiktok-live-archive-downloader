import fcntl
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from replay.core import ReplayError, write_private_text
from replay.session_refresh import (
    Account,
    RefreshResult,
    account_info,
    browser_credentials,
    cookie_grant,
    fingerprint,
    load_snapshot,
    refresh,
    refresh_session,
    snapshot_path,
)
from replay.tiktok_api import Credentials


@pytest.fixture
def headers() -> dict[str, str]:
    return {
        "date": "Wed, 01 Jan 2025 00:00:00 GMT",
        "set-cookie": (
            "sessionid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; Domain=tiktok.com; Path=/; Max-Age=3600"
        ),
    }


def test_server_reissuance_is_not_rotation_or_extension(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    handler = Mock(
        side_effect=[
            httpx.Response(
                200, headers=headers, json={"message": "success", "data": {"user_id_str": "42"}}
            ),
            httpx.Response(
                200, headers=headers, json={"message": "success", "data": {"user_id_str": "42"}}
            ),
            httpx.Response(
                200, json={"status_code": 0, "notice_list": {"notice_list": [], "has_more": 0}}
            ),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = refresh(
            client=client, credentials=Credentials("a" * 32), output=tmp_path / "session"
        )
    assert result.report() == {
        "status": "reissued",
        "web_login": False,
        "expiry_extended": None,
        "cookie_expires_at": "2025-01-01T01:00:00+00:00",
        "api_verified": True,
    }
    assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in repr(result)
    assert (tmp_path / "session").read_text(
        encoding="utf-8"
    ) == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
    assert (tmp_path / "session").stat().st_mode & 0o777 == 0o600
    assert snapshot_path(tmp_path / "session").stat().st_mode & 0o777 == 0o600
    assert handler.call_args_list[0].args[0].headers["x-tt-passport-force-refresh-cookie"] == "1"
    assert "x-tt-passport-force-refresh-cookie" not in handler.call_args_list[1].args[0].headers
    assert handler.call_args_list[0].args[0].url.params["aid"] == "1459"
    assert (
        handler.call_args_list[2].args[0].headers["cookie"]
        == "sessionid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )


@pytest.mark.parametrize("prior_expiry,expected", [(1735693200, False), (1735689600, True)])
def test_fingerprint_bound_expiry_comparison(
    tmp_path: Path, headers: dict[str, str], prior_expiry: int, expected: bool
) -> None:
    output = tmp_path / "session"
    snapshot_path(output).write_text(
        json.dumps(
            {
                "user_id": "42",
                "fingerprint": fingerprint(Credentials("a" * 32)),
                "expires_at": prior_expiry,
            }
        ),
        encoding="utf-8",
    )
    handler = Mock(
        side_effect=[
            httpx.Response(
                200, headers=headers, json={"message": "success", "data": {"user_id_str": "42"}}
            ),
            httpx.Response(
                200, headers=headers, json={"message": "success", "data": {"user_id_str": "42"}}
            ),
            httpx.Response(200, json={"status_code": 0, "notice_list": {"has_more": 0}}),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = refresh(client=client, credentials=Credentials("a" * 32), output=output)
    assert result.expiry_extended is expected


def test_rotated_cookie_is_verified_before_publication(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    output = tmp_path / "session"
    output.write_text("old", encoding="utf-8")
    handler = Mock(
        side_effect=[
            httpx.Response(
                200, headers=headers, json={"message": "success", "data": {"user_id_str": "42"}}
            ),
            httpx.Response(
                200, headers=headers, json={"message": "success", "data": {"user_id_str": "42"}}
            ),
            httpx.Response(200, json={"status_code": 0, "notice_list": {"has_more": 0}}),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = refresh(client=client, credentials=Credentials("b" * 32), output=output)
    assert result.rotated is True
    assert (
        handler.call_args_list[1].args[0].headers["cookie"]
        == "sessionid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    assert (
        handler.call_args_list[2].args[0].headers["cookie"]
        == "sessionid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )


@pytest.mark.parametrize("verify_user,prior_user", [("99", None), ("42", "99")])
def test_account_mismatch_preserves_token(
    tmp_path: Path, headers: dict[str, str], verify_user: str, prior_user: str | None
) -> None:
    output = tmp_path / "session"
    output.write_text("old", encoding="utf-8")
    if prior_user is not None:
        snapshot_path(output).write_text(
            json.dumps(
                {
                    "user_id": prior_user,
                    "fingerprint": fingerprint(Credentials("a" * 32)),
                    "expires_at": 1735693200,
                }
            ),
            encoding="utf-8",
        )
    handler = Mock(
        side_effect=[
            httpx.Response(
                200, headers=headers, json={"message": "success", "data": {"user_id_str": "42"}}
            ),
            httpx.Response(
                200,
                headers=headers,
                json={"message": "success", "data": {"user_id_str": verify_user}},
            ),
        ]
    )
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ReplayError, match="Account mismatch"),
    ):
        refresh(client=client, credentials=Credentials("a" * 32), output=output)
    assert output.read_text(encoding="utf-8") == "old"
    assert handler.call_count == 2


def test_replay_rejection_preserves_token(tmp_path: Path, headers: dict[str, str]) -> None:
    output = tmp_path / "session"
    output.write_text("old", encoding="utf-8")
    handler = Mock(
        side_effect=[
            httpx.Response(
                200, headers=headers, json={"message": "success", "data": {"user_id_str": "42"}}
            ),
            httpx.Response(
                200, headers=headers, json={"message": "success", "data": {"user_id_str": "42"}}
            ),
            httpx.Response(200, json={"status_code": 20003}),
        ]
    )
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ReplayError, match="expired"),
    ):
        refresh(client=client, credentials=Credentials("a" * 32), output=output)
    assert output.read_text(encoding="utf-8") == "old"
    assert not snapshot_path(output).exists()


@pytest.mark.parametrize(
    "cookie",
    [
        "",
        "other=value",
        "sessionid=bad; Path=/; Max-Age=3600",
        "sessionid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; Path=/; Domain=evil.test; Max-Age=3600",
        "sessionid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; Path=/other; Max-Age=3600",
        "sessionid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; Path=/; Max-Age=0",
        "sessionid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; Path=/; Max-Age=bad",
        "sessionid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; Path=/; Max-Age=9999999999",
    ],
)
def test_invalid_cookie(cookie: str, headers: dict[str, str]) -> None:
    with pytest.raises(ReplayError):
        cookie_grant(httpx.Headers(headers | {"set-cookie": cookie}))


@pytest.mark.parametrize(
    "date", ["bad", "Wed, 01 Jan 2025 00:00:00", "Fri, 31 Dec 9999 23:59:59 GMT"]
)
def test_invalid_date(date: str, headers: dict[str, str]) -> None:
    with pytest.raises(ReplayError):
        cookie_grant(httpx.Headers(headers | {"date": date}))


def test_conflicting_cookie(headers: dict[str, str]) -> None:
    values = httpx.Headers(headers)
    values["set-cookie"] = headers["set-cookie"]
    conflicting = httpx.Headers(
        [
            *values.multi_items(),
            ("set-cookie", "sessionid=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; Path=/; Max-Age=3600"),
        ]
    )
    with pytest.raises(ReplayError, match="unambiguous"):
        cookie_grant(conflicting)


@pytest.mark.parametrize(
    "response,message",
    [
        (httpx.Response(302, headers={"location": "https://evil.test"}), "redirects refused"),
        (httpx.Response(200, content=b"broken"), "invalid JSON"),
        (httpx.Response(200, content=b"\xff"), "invalid JSON"),
        (
            httpx.Response(200, json={"message": "error", "data": {"secret": "hidden"}}),
            "Mac reauthentication",
        ),
        (httpx.Response(200, content=b"x" * (2 * 1024 * 1024 + 1)), "size limit"),
    ],
)
def test_account_failures_are_redacted(response: httpx.Response, message: str) -> None:
    handler = Mock(return_value=response)
    with (
        httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client,
        pytest.raises(ReplayError, match=message) as caught,
    ):
        account_info(client, credentials=Credentials("a" * 32), force_refresh=True)
    assert "hidden" not in str(caught.value)
    assert handler.call_count == 1


def test_network_failure_redacted() -> None:
    handler = Mock(side_effect=httpx.ConnectError("private token"))
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ReplayError, match="network error") as caught,
    ):
        account_info(client, credentials=Credentials("a" * 32), force_refresh=False)
    assert "private token" not in str(caught.value)


@pytest.mark.parametrize(
    "expiry,digest",
    [(False, "a" * 64), (float("nan"), "a" * 64), (1, "bad"), (float("inf"), "a" * 64)],
)
def test_bad_metadata(tmp_path: Path, expiry: float | bool, digest: str) -> None:
    output = tmp_path / "token"
    snapshot_path(output).write_text(
        json.dumps({"user_id": "42", "fingerprint": digest, "expires_at": expiry}), encoding="utf-8"
    )
    with pytest.raises(ReplayError, match="metadata"):
        load_snapshot(output)


def test_browser_cookie_extraction_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_login(*, session: Path, timeout: int) -> None:
        assert timeout == 42
        session.write_text(
            json.dumps(
                {
                    "cookies": [
                        {
                            "name": "sessionid",
                            "domain": ".tiktok.com",
                            "path": "/",
                            "value": "a" * 32,
                        },
                        {
                            "name": "sessionid",
                            "domain": ".evil.test",
                            "path": "/",
                            "value": "b" * 32,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("replay.session_refresh.login", fake_login)
    assert (
        browser_credentials(directory=tmp_path, timeout=42).session_id
        == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "cookies", [None, [], [{"name": "sessionid", "domain": ".tiktok.com", "path": "/", "value": 1}]]
)
def test_bad_browser_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cookies: object
) -> None:
    def fake_login(*, session: Path, timeout: int) -> None:
        session.write_text(json.dumps({"cookies": cookies}), encoding="utf-8")

    monkeypatch.setattr("replay.session_refresh.login", fake_login)
    with pytest.raises(ReplayError):
        browser_credentials(directory=tmp_path, timeout=1)
    assert list(tmp_path.iterdir()) == []


def test_refresh_lock_and_missing_credentials(tmp_path: Path) -> None:
    output = tmp_path / "session"
    with pytest.raises(ReplayError, match="Set TIKTOK_SESSIONID"):
        refresh_session(output=output, credentials=None)
    with output.with_name("session.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ReplayError, match="already running"):
            refresh_session(output=output, credentials=Credentials("a" * 32))


def test_explicit_web_login_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    browser = Mock(return_value=Credentials("a" * 32))
    operation = Mock(
        return_value=RefreshResult(Account("42", Credentials("a" * 32), 1735693200), False, None)
    )
    monkeypatch.setattr("replay.session_refresh.browser_credentials", browser)
    monkeypatch.setattr("replay.session_refresh.refresh", operation)
    result = refresh_session(
        output=tmp_path / "token", credentials=Credentials("b" * 32), web_login=True
    )
    assert result.web_login is True
    assert result.rotated is True
    assert operation.call_args.kwargs["client"].is_closed is True


def test_http_wrapper_never_opens_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    browser = Mock(side_effect=AssertionError("must not open"))
    operation = Mock(
        return_value=RefreshResult(Account("42", Credentials("a" * 32), 1735693200), False, None)
    )
    monkeypatch.setattr("replay.session_refresh.browser_credentials", browser)
    monkeypatch.setattr("replay.session_refresh.refresh", operation)
    result = refresh_session(output=tmp_path / "token", credentials=Credentials("a" * 32))
    assert result.web_login is False
    browser.assert_not_called()


def test_web_login_uses_pinned_account_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "token"
    snapshot_path(output).write_text(
        json.dumps(
            {
                "user_id": "42",
                "fingerprint": fingerprint(Credentials("b" * 32)),
                "expires_at": 1735693200,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "replay.session_refresh.browser_credentials", Mock(return_value=Credentials("a" * 32))
    )
    monkeypatch.setattr(
        "replay.session_refresh.refresh",
        Mock(
            return_value=RefreshResult(
                Account("42", Credentials("a" * 32), 1735693200), False, None
            )
        ),
    )
    result = refresh_session(output=output, credentials=None, web_login=True)
    assert result.rotated is True
    assert result.web_login is True


def test_failed_token_publication_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "token"
    output.write_text("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n", encoding="utf-8")
    monkeypatch.setattr(
        "replay.session_refresh.account_info",
        Mock(return_value=Account("42", Credentials("a" * 32), 1735693200)),
    )
    monkeypatch.setattr("replay.session_refresh.TikTokAPI.validate_session", Mock())

    def failing_publication(path: Path, text: str) -> None:
        if path == output:
            raise OSError("full disk")
        write_private_text(path, text)

    monkeypatch.setattr("replay.session_refresh.write_private_text", failing_publication)
    with httpx.Client() as client, pytest.raises(OSError, match="full disk"):
        refresh(client=client, credentials=Credentials("b" * 32), output=output)
    assert output.read_text(encoding="utf-8") == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
    snapshot = load_snapshot(output)
    assert snapshot is not None
    assert snapshot.expires_at == 1735693200


def test_private_text_replacement(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "token"
    write_private_text(output, "old")
    write_private_text(output, "new")
    assert output.read_text(encoding="utf-8") == "new"
    assert output.stat().st_mode & 0o777 == 0o600
    assert len(list(output.parent.iterdir())) == 1
