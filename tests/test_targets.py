import sys
from unittest.mock import MagicMock, Mock

import httpx
import pytest
from playwright.sync_api import Error

from replay.core import ReplayError
from replay.targets import (
    block_media,
    browser_profile,
    profile_id,
    resolve_user,
    target_id,
    username,
)


@pytest.mark.parametrize(
    "selector",
    [
        "creator",
        "@Creator",
        " https://www.tiktok.com/@Creator/?lang=ja ",
        "https://tiktok.com/@creator",
    ],
)
def test_username(selector: str) -> None:
    assert username(selector) == "creator"


@pytest.mark.parametrize(
    "selector",
    [
        "",
        "表示名",
        "a/b",
        "creator.",
        "a" * 25,
        "https://evil.test/@creator",
        "http://www.tiktok.com/@creator",
        "https://user:pass@www.tiktok.com/@creator",
        "https://www.tiktok.com:444/@creator",
        "https://www.tiktok.com/@creator/video/123",
        "https://www.tiktok.com/video/123",
        "https://[bad/@creator",
        "https://www.tiktok.com/@a%2Fb",
    ],
)
def test_invalid_selector(selector: str) -> None:
    with pytest.raises(ReplayError):
        username(selector)


def test_leading_period_in_username() -> None:
    assert username("@.creator") == ".creator"


def test_verified_profile_identity() -> None:
    assert (
        profile_id(
            '<html><script id="other">irrelevant</script>'
            '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">'
            '{"__DEFAULT_SCOPE__":{"webapp.user-detail":{"userInfo":{"user":{"id":"123","uniqueId":"Creator"}}}}}'
            "</script></html>",
            expected_username="creator",
        )
        == "123"
    )


@pytest.mark.parametrize(
    "html",
    [
        "<html>challenge</html>",
        '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">{}</script>'
        '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">{}</script>',
        '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">bad</script>',
        '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">{}</script>',
        '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">'
        '{"__DEFAULT_SCOPE__":{"webapp.user-detail":{"userInfo":'
        '{"user":{"id":"123","uniqueId":"other"}}}}}</script>',
    ],
)
def test_invalid_profile(html: str) -> None:
    with pytest.raises(ReplayError):
        profile_id(html, expected_username="creator")


def test_profile_lookup_never_sends_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIKTOK_SESSIONID", "a" * 32)
    handler = Mock(
        return_value=httpx.Response(
            200,
            text=(
                '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">'
                '{"__DEFAULT_SCOPE__":{"webapp.user-detail":{"userInfo":{"user":{"id":"123","uniqueId":"creator"}}}}}'
                "</script>"
            ),
        )
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    factory = Mock(return_value=client)
    monkeypatch.setattr("replay.targets.httpx.Client", factory)
    assert resolve_user("https://www.tiktok.com/@Creator?secret=not-forwarded") == "123"
    assert str(handler.call_args.args[0].url) == "https://www.tiktok.com/@creator"
    assert "cookie" not in handler.call_args.args[0].headers
    assert "authorization" not in handler.call_args.args[0].headers
    assert factory.call_args.kwargs["trust_env"] is False
    assert client.is_closed is True


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"location": "https://evil.test"}),
        httpx.Response(200, content=b"x" * (2 * 1024 * 1024 + 1)),
        httpx.Response(200, content=b"\xff"),
    ],
)
def test_lookup_failures(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> None:
    handler = Mock(return_value=response)
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    monkeypatch.setattr("replay.targets.httpx.Client", Mock(return_value=client))
    with pytest.raises(ReplayError):
        resolve_user("creator")
    assert handler.call_count == 1
    assert client.is_closed is True


def test_network_error_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(Mock(side_effect=httpx.ConnectError("secret")))
    )
    monkeypatch.setattr("replay.targets.httpx.Client", Mock(return_value=client))
    with pytest.raises(ReplayError, match="network error") as caught:
        resolve_user("creator")
    assert "secret" not in str(caught.value)


def test_cli_id_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = Mock(side_effect=AssertionError("no lookup"))
    monkeypatch.setattr("replay.targets.resolve_user", resolver)
    assert target_id(anchor_id="123", user=None, environ={"TIKTOK_TARGET_USER": "other"}) == "123"
    resolver.assert_not_called()


def test_cli_username_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = Mock(return_value="123")
    monkeypatch.setattr("replay.targets.resolve_user", resolver)
    assert (
        target_id(anchor_id=None, user="@creator", environ={"TIKTOK_TARGET_USER": "other"}) == "123"
    )
    resolver.assert_called_once_with("@creator")


def test_environment_target(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = Mock(return_value="456")
    monkeypatch.setattr("replay.targets.resolve_user", resolver)
    assert target_id(anchor_id=None, user=None, environ={"TIKTOK_TARGET_USER": "@second"}) == "456"
    resolver.assert_called_once_with("@second")


@pytest.mark.parametrize("environment", [{}, {"TIKTOK_TARGET_USER": " "}])
def test_missing_target(environment: dict[str, str]) -> None:
    with pytest.raises(ReplayError, match="Specify"):
        target_id(anchor_id=None, user=None, environ=environment)


def test_empty_http_uses_isolated_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    client = httpx.Client(transport=httpx.MockTransport(Mock(return_value=httpx.Response(200))))
    monkeypatch.setattr("replay.targets.httpx.Client", Mock(return_value=client))
    browser = Mock(return_value="123")
    monkeypatch.setattr("replay.targets.browser_profile", browser)
    assert resolve_user("@creator") == "123"
    browser.assert_called_once_with("creator")


def test_browser_profile_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = MagicMock()
    launch = factory.return_value.__enter__.return_value.chromium.launch
    browser = launch.return_value
    page = browser.new_context.return_value.new_page.return_value
    page.url = "https://www.tiktok.com/@creator"
    page.content.return_value = (
        '<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__">'
        '{"__DEFAULT_SCOPE__":{"webapp.user-detail":{"userInfo":'
        '{"user":{"id":"123","uniqueId":"creator"}}}}}</script>'
    )
    monkeypatch.setattr("playwright.sync_api.sync_playwright", factory)
    assert browser_profile("creator") == "123"
    launch.assert_called_once_with(channel="chrome", headless=True)
    browser.close.assert_called_once_with()


@pytest.mark.parametrize(
    "url,html",
    [
        ("https://www.tiktok.com/login", ""),
        ("https://evil.test/@creator", ""),
        ("https://www.tiktok.com/@creator", "x" * (2 * 1024 * 1024 + 1)),
    ],
)
def test_browser_invalid_page(monkeypatch: pytest.MonkeyPatch, url: str, html: str) -> None:
    factory = MagicMock()
    browser = factory.return_value.__enter__.return_value.chromium.launch.return_value
    page = browser.new_context.return_value.new_page.return_value
    page.url = url
    page.content.return_value = html
    monkeypatch.setattr("playwright.sync_api.sync_playwright", factory)
    with pytest.raises(ReplayError):
        browser_profile("creator")
    browser.close.assert_called_once_with()


def test_missing_browser_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(ReplayError, match="web support"):
        browser_profile("creator")


def test_browser_error_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = MagicMock()
    factory.return_value.__enter__.side_effect = Error("secret URL")
    monkeypatch.setattr("playwright.sync_api.sync_playwright", factory)
    with pytest.raises(ReplayError, match="unavailable") as caught:
        browser_profile("creator")
    assert "secret URL" not in str(caught.value)


@pytest.mark.parametrize(
    "resource,url", [("media", "https://www.tiktok.com/video"), ("document", "https://evil.test/")]
)
def test_browser_blocks_media_and_external_navigation(resource: str, url: str) -> None:
    route = Mock()
    route.request.resource_type = resource
    route.request.url = url
    block_media(route)
    route.abort.assert_called_once_with()
    route.continue_.assert_not_called()


@pytest.mark.parametrize("resource", ["script", "document"])
def test_browser_allows_profile_resources(resource: str) -> None:
    route = Mock()
    route.request.resource_type = resource
    route.request.url = "https://www.tiktok.com/@creator"
    block_media(route)
    route.continue_.assert_called_once_with()


def test_conflicting_targets() -> None:
    with pytest.raises(ReplayError, match="not both"):
        target_id(anchor_id="123", user="creator", environ={})
