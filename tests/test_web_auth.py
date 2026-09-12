import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from playwright.sync_api import Error

from replay.core import ReplayError
from replay.web_auth import login, save_state, scoped_state, tiktok_domain, tiktok_origin


@pytest.mark.parametrize("domain", ["tiktok.com", ".tiktok.com", "WWW.TIKTOK.COM"])
def test_tiktok_domains(domain: str) -> None:
    assert tiktok_domain(domain)


@pytest.mark.parametrize(
    "origin",
    [
        "http://www.tiktok.com",
        "https://tiktok.com.evil.test",
        "https://evil-tiktok.com",
        "https://u:p@www.tiktok.com",
        "https://www.tiktok.com:444",
        "file:///tmp/a",
        "https://[bad/",
    ],
)
def test_untrusted_origins(origin: str) -> None:
    assert not tiktok_origin(origin)


def test_scope_filters_identity_provider_state() -> None:
    assert scoped_state(
        {
            "cookies": [
                {"domain": ".tiktok.com", "name": "sessionid", "value": "private"},
                {"domain": ".google.com", "name": "session", "value": "excluded"},
                {"name": "missing-domain", "value": "excluded"},
            ],
            "origins": [
                {
                    "origin": "https://www.tiktok.com",
                    "localStorage": [{"name": "pref", "value": "ok"}],
                },
                {"origin": "https://accounts.google.com", "localStorage": []},
            ],
        }
    ) == {
        "cookies": [{"domain": ".tiktok.com", "name": "sessionid", "value": "private"}],
        "origins": [
            {"origin": "https://www.tiktok.com", "localStorage": [{"name": "pref", "value": "ok"}]}
        ],
    }


def test_atomic_private_state(tmp_path: Path) -> None:
    path = tmp_path / "private" / "web.json"
    save_state(path, {})
    assert json.loads(path.read_text(encoding="utf-8")) == {"cookies": [], "origins": []}
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert len(list(path.parent.iterdir())) == 1
    save_state(path, {"cookies": []})
    assert json.loads(path.read_text(encoding="utf-8")) == {"cookies": [], "origins": []}
    assert path.stat().st_mode & 0o777 == 0o600


def test_failed_replace_keeps_existing_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "web.json"
    path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(os, "replace", Mock(side_effect=OSError("full disk")))
    with pytest.raises(OSError, match="full disk"):
        save_state(path, {})
    assert path.read_text(encoding="utf-8") == "existing"
    assert len(list(tmp_path.iterdir())) == 1


def test_login_waits_then_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = MagicMock()
    browser = factory.return_value.__enter__.return_value.chromium.launch.return_value
    context = browser.new_context.return_value
    context.pages = [Mock()]
    context.cookies.side_effect = [[], [{"name": "sessionid", "value": "private"}]]
    context.storage_state.return_value = {"cookies": [], "origins": []}
    monkeypatch.setattr("playwright.sync_api.sync_playwright", factory)
    login(session=tmp_path / "web.json")
    context.pages[0].wait_for_timeout.assert_called_once_with(500)
    browser.close.assert_called_once_with()
    assert (tmp_path / "web.json").exists()


def test_login_timeout_keeps_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "web.json"
    path.write_text("existing", encoding="utf-8")
    factory = MagicMock()
    browser = factory.return_value.__enter__.return_value.chromium.launch.return_value
    monkeypatch.setattr("playwright.sync_api.sync_playwright", factory)
    monkeypatch.setattr("replay.web_auth.time.monotonic", Mock(side_effect=[0, 2]))
    with pytest.raises(ReplayError, match="timed out"):
        login(session=path, timeout=1)
    assert path.read_text(encoding="utf-8") == "existing"
    browser.close.assert_called_once_with()


def test_closed_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = MagicMock()
    browser = factory.return_value.__enter__.return_value.chromium.launch.return_value
    context = browser.new_context.return_value
    context.cookies.return_value = []
    context.pages = []
    monkeypatch.setattr("playwright.sync_api.sync_playwright", factory)
    with pytest.raises(ReplayError, match="browser was closed"):
        login(session=tmp_path / "web.json")
    assert not (tmp_path / "web.json").exists()


def test_browser_error_is_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = MagicMock()
    factory.return_value.__enter__.side_effect = Error("secret URL")
    monkeypatch.setattr("playwright.sync_api.sync_playwright", factory)
    with pytest.raises(ReplayError, match="Web login failed") as caught:
        login(session=tmp_path / "web.json")
    assert "secret URL" not in str(caught.value)


def test_missing_optional_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(ReplayError, match="Install web support"):
        login(session=tmp_path / "web.json")


@pytest.mark.parametrize("timeout", [0, 1801])
def test_bad_timeout(tmp_path: Path, timeout: int) -> None:
    with pytest.raises(ReplayError, match="timeout must"):
        login(session=tmp_path / "web.json", timeout=timeout)
