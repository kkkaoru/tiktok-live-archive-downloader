"""User-driven web authentication isolated from existing browser profiles."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from replay.core import ReplayError, Scope, write_private_text

if TYPE_CHECKING:
    from playwright.sync_api import BrowserContext, StorageState

LOGIN_URL = "https://www.tiktok.com/login"
AUTH_COOKIE_NAMES = frozenset({"sessionid", "sessionid_ss", "sid_tt"})
MAX_LOGIN_SECONDS = 1800
LOGIN_POLL_MS = 500


def tiktok_domain(domain: str) -> bool:
    host = domain.removeprefix(".").lower()
    return host == "tiktok.com" or host.endswith(".tiktok.com")


def tiktok_origin(origin: str) -> bool:
    try:
        host = urlsplit(origin).hostname
        if host is None or not tiktok_domain(host):
            return False
        Scope((host,)).check(origin)
    except (ValueError, ReplayError):
        return False
    return True


def scoped_state(state: StorageState) -> StorageState:
    """Never persist third-party identity-provider cookies or storage."""
    return {
        "cookies": [
            cookie for cookie in state.get("cookies", []) if tiktok_domain(cookie.get("domain", ""))
        ],
        "origins": [
            origin for origin in state.get("origins", []) if tiktok_origin(origin["origin"])
        ],
    }


def save_state(path: Path, state: StorageState) -> None:
    write_private_text(path, json.dumps(scoped_state(state), ensure_ascii=False))


def _wait_for_login(context: BrowserContext, *, timeout: int) -> StorageState:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cookies = context.cookies(["https://www.tiktok.com/"])
        if any(
            cookie.get("name") in AUTH_COOKIE_NAMES and cookie.get("value") for cookie in cookies
        ):
            return context.storage_state()
        if not context.pages:
            raise ReplayError("Login browser was closed; no session was saved.")
        context.pages[0].wait_for_timeout(LOGIN_POLL_MS)
    raise ReplayError("Login timed out; no saved session was replaced.")


def login(*, session: Path, timeout: int = 600) -> None:
    """Open Chrome only when explicitly invoked by the user for authentication."""
    if not 1 <= timeout <= MAX_LOGIN_SECONDS:
        raise ReplayError("Login timeout must be between 1 and 1800 seconds.")
    # Keep the optional browser dependency out of ordinary download commands.
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError as error:
        raise ReplayError("Install web support: uv sync --extra web") from error

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome", headless=False)
            try:
                context = browser.new_context()
                page = context.new_page()
                page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
                print(
                    "Complete TikTok login in the dedicated browser. Credentials are not logged.",
                    flush=True,
                )
                state = _wait_for_login(context, timeout=timeout)
                save_state(session, state)
            finally:
                browser.close()
    except Error as error:
        raise ReplayError(
            "Web login failed or the browser was closed. No browser errors were logged."
        ) from error
