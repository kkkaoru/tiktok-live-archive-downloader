"""Resolve runtime creator selectors without sharing account credentials."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

import httpx

from replay.core import ReplayError, Scope
from replay.recordings import numeric_id, record

if TYPE_CHECKING:
    from playwright.sync_api import Route

MAX_PROFILE_BYTES = 2 * 1024 * 1024
PROFILE_CHUNK_BYTES = 65536
HYDRATION_ID = "__UNIVERSAL_DATA_FOR_REHYDRATION__"


class ProfileUnavailable(ReplayError):
    """An ordinary browser may be needed to obtain the public page state."""


class ProfileState(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.collecting = False
        self.count = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and dict(attrs).get("id") == HYDRATION_ID:
            self.count += 1
            self.collecting = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.collecting = False

    def handle_data(self, data: str) -> None:
        if self.collecting:
            self.parts.append(data)


def username(selector: str) -> str:
    value = selector.strip()
    if "://" in value:
        try:
            Scope(("www.tiktok.com", "tiktok.com")).check(value)
            path = unquote(urlsplit(value).path)
        except (ReplayError, ValueError) as error:
            raise ReplayError("Use a canonical HTTPS TikTok profile URL.") from error
        if not path.startswith("/@"):
            raise ReplayError("Use a TikTok profile URL, not a video or shortened link.")
        value = path[2:].removesuffix("/")
    else:
        value = value.removeprefix("@")
    if re.fullmatch(r"[a-zA-Z0-9_.]{1,24}", value) is None or value.endswith("."):
        raise ReplayError("Invalid TikTok username; display names are not supported.")
    return value.lower()


def profile_id(html: str, *, expected_username: str) -> str:
    parser = ProfileState()
    parser.feed(html)
    parser.close()
    if parser.count == 0:
        raise ProfileUnavailable("Profile identity unavailable; use a verified --anchor-id.")
    if parser.count != 1:
        raise ReplayError("Profile identity unavailable or ambiguous; use a verified --anchor-id.")
    try:
        state = record(json.loads("".join(parser.parts)))
    except json.JSONDecodeError as error:
        raise ReplayError("Profile returned invalid identity data.") from error
    scope = record(state.get("__DEFAULT_SCOPE__"))
    detail = record(scope.get("webapp.user-detail"))
    user = record(record(detail.get("userInfo")).get("user"))
    handle = user.get("uniqueId")
    if not isinstance(handle, str) or handle.lower() != expected_username:
        raise ReplayError("Profile username mismatch; refusing to target another account.")
    return numeric_id(user.get("id"))


def block_media(route: Route) -> None:
    if route.request.resource_type == "document":
        try:
            Scope(("www.tiktok.com",)).check(route.request.url)
        except (ReplayError, ValueError):
            route.abort()
            return
    if route.request.resource_type == "media":
        route.abort()
    else:
        route.continue_()


def browser_profile(handle: str) -> str:
    """Read a public page in an isolated headless browser; never solve challenges."""
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError as error:
        raise ReplayError(
            "Profile requires web support: uv sync --extra web; or use --anchor-id."
        ) from error
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
            try:
                context = browser.new_context()
                context.route("**/*", block_media)
                page = context.new_page()
                page.goto(
                    f"https://www.tiktok.com/@{handle}",
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
                if urlsplit(page.url).path.removesuffix("/") != f"/@{handle}":
                    raise ReplayError("Profile navigation changed identity; use --anchor-id.")
                Scope(("www.tiktok.com",)).check(page.url)
                page.wait_for_selector(f"script#{HYDRATION_ID}", state="attached", timeout=15000)
                html = page.content()
                if len(html.encode("utf-8")) > MAX_PROFILE_BYTES:
                    raise ReplayError("Browser profile exceeds the size limit.")
                return profile_id(html, expected_username=handle)
            finally:
                browser.close()
    except Error as error:
        raise ReplayError(
            "Browser profile unavailable or challenged; use a verified --anchor-id."
        ) from error


def resolve_user(selector: str) -> str:
    handle = username(selector)
    body = bytearray()
    try:
        # This client never receives TIKTOK_SESSIONID or any authenticated API client state.
        with (
            httpx.Client(timeout=httpx.Timeout(30, connect=15), trust_env=False) as client,
            client.stream(
                "GET", f"https://www.tiktok.com/@{handle}", follow_redirects=False
            ) as response,
        ):
            if response.status_code != 200:
                raise ReplayError(
                    "Profile lookup failed; redirects are not followed. Use --anchor-id."
                )
            for chunk in response.iter_bytes(chunk_size=PROFILE_CHUNK_BYTES):
                body.extend(chunk)
                if len(body) > MAX_PROFILE_BYTES:
                    raise ReplayError("Profile response exceeds the size limit.")
    except httpx.HTTPError as error:
        raise ReplayError("Profile network error; account credentials were not sent.") from error
    try:
        html = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReplayError("Profile returned invalid text.") from error
    try:
        return profile_id(html, expected_username=handle)
    except ProfileUnavailable:
        return browser_profile(handle)


def target_id(*, anchor_id: str | None, user: str | None, environ: Mapping[str, str]) -> str:
    if anchor_id is not None and user is not None:
        raise ReplayError("Choose either --anchor-id or --user, not both.")
    if anchor_id is not None:
        return numeric_id(anchor_id)
    selector = user if user is not None else environ.get("TIKTOK_TARGET_USER")
    if selector is None or not selector.strip():
        raise ReplayError("Specify --anchor-id, --user, or TIKTOK_TARGET_USER.")
    return resolve_user(selector)
