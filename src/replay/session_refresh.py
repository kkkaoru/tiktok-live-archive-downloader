"""Official web cookie reissuance, with explicit and verified Mac reauthentication."""

import fcntl
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from replay.core import ReplayError, private_directory, write_private_text
from replay.recordings import numeric_id, record
from replay.tiktok_api import MAX_API_BYTES, Credentials, TikTokAPI
from replay.web_auth import login

ACCOUNT_ENDPOINT = "https://www.tiktok.com/passport/web/account/info/"
COOKIE_DOMAINS = frozenset({"", "tiktok.com", ".tiktok.com", "www.tiktok.com", ".www.tiktok.com"})
EXPIRY_TOLERANCE = 2


@dataclass(frozen=True)
class Account:
    user_id: str
    credentials: Credentials
    expires_at: float


@dataclass(frozen=True)
class Snapshot:
    user_id: str
    fingerprint: str
    expires_at: float


@dataclass(frozen=True)
class RefreshResult:
    account: Account
    rotated: bool
    expiry_extended: bool | None
    web_login: bool = False

    def report(self) -> dict[str, object]:
        return {
            "status": "rotated" if self.rotated else "reissued",
            "web_login": self.web_login,
            "expiry_extended": self.expiry_extended,
            "cookie_expires_at": datetime.fromtimestamp(self.account.expires_at, UTC).isoformat(),
            "api_verified": True,
        }


def fingerprint(credentials: Credentials) -> str:
    return hashlib.sha256(credentials.session_id.encode()).hexdigest()


def snapshot_path(output: Path) -> Path:
    return output.with_name(output.name + ".metadata.json")


def load_snapshot(output: Path) -> Snapshot | None:
    path = snapshot_path(output)
    if not path.exists():
        return None
    value = record(json.loads(path.read_text(encoding="utf-8")))
    user_id = numeric_id(value.get("user_id"))
    digest, expiry = value.get("fingerprint"), value.get("expires_at")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or type(expiry) not in (float, int)
        or not isinstance(expiry, (float, int))
        or not 0 < expiry < 253402300799
    ):
        raise ReplayError("Invalid session metadata; no credential was replaced.")
    return Snapshot(user_id, digest, float(expiry))


def cookie_grant(headers: httpx.Headers) -> tuple[Credentials, float]:
    grants: set[tuple[str, float]] = set()
    try:
        for header in headers.get_list("set-cookie"):
            cookies = SimpleCookie()
            cookies.load(header)
            if "sessionid" not in cookies:
                continue
            cookie = cookies["sessionid"]
            if cookie["domain"].lower() not in COOKIE_DOMAINS or cookie["path"] != "/":
                raise ReplayError("Session cookie has an unexpected scope.")
            age = int(cookie["max-age"])
            date = parsedate_to_datetime(headers.get("date", ""))
            if date.tzinfo is None or not 0 < age <= 366 * 86400:
                raise ReplayError("Session cookie has an invalid server lifetime.")
            expiry = date.timestamp() + age
            if not math.isfinite(expiry) or not 0 < expiry < 253402300799:
                raise ReplayError("Session cookie expiry is invalid.")
            grants.add((Credentials(cookie.value).session_id, expiry))
    except (CookieError, ValueError, OverflowError) as error:
        raise ReplayError(
            "Malformed server session cookie; credentials were not logged."
        ) from error
    if len(grants) != 1:
        raise ReplayError(
            "Server did not issue one unambiguous session cookie; no refresh claimed."
        )
    token, expires = grants.pop()
    return Credentials(token), expires


def account_info(client: httpx.Client, *, credentials: Credentials, force_refresh: bool) -> Account:
    headers = {"Cookie": "sessionid=" + credentials.session_id}
    if force_refresh:
        headers["x-tt-passport-force-refresh-cookie"] = "1"
    body = bytearray()
    try:
        with client.stream(
            "GET",
            ACCOUNT_ENDPOINT,
            params={"aid": "1459", "locale": "ja-JP", "app_language": "ja"},
            headers=headers,
            follow_redirects=False,
        ) as response:
            if response.status_code != 200:
                raise ReplayError(f"Session API HTTP {response.status_code}; redirects refused.")
            for chunk in response.iter_bytes(chunk_size=65536):
                body.extend(chunk)
                if len(body) > MAX_API_BYTES:
                    raise ReplayError("Session API response exceeds the size limit.")
            response_headers = response.headers
    except httpx.HTTPError as error:
        raise ReplayError("Session API network error; credentials were not logged.") from error
    try:
        payload = record(json.loads(body))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ReplayError("Session API returned invalid JSON.") from error
    if payload.get("message") != "success":
        raise ReplayError("Session rejected. Mac reauthentication: replay refresh-session --login")
    user_id = numeric_id(record(payload.get("data")).get("user_id_str"))
    issued, expires_at = cookie_grant(response_headers)
    return Account(user_id, issued, expires_at)


def browser_credentials(*, directory: Path, timeout: int) -> Credentials:
    with TemporaryDirectory(dir=directory) as temporary:
        state = Path(temporary) / "web.json"
        login(session=state, timeout=timeout)
        cookies = record(json.loads(state.read_text(encoding="utf-8"))).get("cookies")
        if not isinstance(cookies, list):
            raise ReplayError("Web login did not provide cookies.")
        tokens: set[str] = set()
        for item in cookies:
            cookie = record(item)
            domain = cookie.get("domain")
            if (
                cookie.get("name") == "sessionid"
                and isinstance(domain, str)
                and domain in COOKIE_DOMAINS - {""}
                and cookie.get("path") == "/"
            ):
                value = cookie.get("value")
                if not isinstance(value, str):
                    raise ReplayError("Web login returned an invalid session cookie.")
                tokens.add(Credentials(value).session_id)
        if len(tokens) != 1:
            raise ReplayError("Web login did not provide one unambiguous TikTok session.")
        return Credentials(tokens.pop())


def refresh(*, client: httpx.Client, credentials: Credentials, output: Path) -> RefreshResult:
    """Save only server-issued credentials verified for the same account and API."""
    previous = load_snapshot(output)
    issued = account_info(client, credentials=credentials, force_refresh=True)
    verified = account_info(client, credentials=issued.credentials, force_refresh=False)
    if issued.user_id != verified.user_id or (
        previous is not None and previous.user_id != issued.user_id
    ):
        raise ReplayError("Account mismatch; existing credentials were not replaced.")
    # Keep the latest cookie from the verifying response; its identity is server-attested.
    TikTokAPI(client=client, credentials=verified.credentials).validate_session()
    extended = None
    if previous is not None and previous.fingerprint == fingerprint(credentials):
        extended = verified.expires_at > previous.expires_at + EXPIRY_TOLERANCE
    result = RefreshResult(verified, verified.credentials != credentials, extended)
    snapshot = Snapshot(verified.user_id, fingerprint(verified.credentials), verified.expires_at)
    # Metadata is fingerprint-bound: a failed token publication cannot give the old token
    # the new token's expiry. The previous plaintext credential remains untouched on failure.
    write_private_text(snapshot_path(output), json.dumps(asdict(snapshot)))
    write_private_text(output, verified.credentials.session_id + "\n")
    return result


def refresh_session(
    *, output: Path, credentials: Credentials | None, web_login: bool = False, timeout: int = 600
) -> RefreshResult:
    private_directory(output.parent)
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReplayError("Another session refresh is already running.") from error
        previous = load_snapshot(output)
        original = credentials
        if web_login:
            credentials = browser_credentials(directory=output.parent, timeout=timeout)
        if credentials is None:
            raise ReplayError("Set TIKTOK_SESSIONID, or use refresh-session --login on this Mac.")
        with httpx.Client(timeout=httpx.Timeout(30, connect=15), trust_env=False) as client:
            result = refresh(client=client, credentials=credentials, output=output)
        if web_login:
            old_digest = fingerprint(original) if original is not None else None
            if previous is not None:
                old_digest = previous.fingerprint
            result = replace(
                result,
                web_login=True,
                rotated=old_digest is not None
                and old_digest != fingerprint(result.account.credentials),
            )
        return result
