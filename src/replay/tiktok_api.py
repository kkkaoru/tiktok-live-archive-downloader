"""Environment-authenticated API access without Android or browser dependencies."""

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import chain
from typing import Self

import httpx

from replay.core import ReplayError
from replay.recordings import (
    Notice,
    Recording,
    notice_from_json,
    numeric_id,
    record,
    recording_from_json,
)

NOTICE_ENDPOINT = "https://api19-normal-c-alisg.tiktokv.com/tiktok/notice/system_notice_box/v1/"
REPLAY_ENDPOINT = "https://webcast-va.tiktokv.com/webcast/room/replay/info/"
MAX_API_BYTES = 2 * 1024 * 1024
MAX_NOTICE_PAGES = 100
CLIENT_VERSION_CODE = "460942"
CLIENT_VERSION_NAME = "46.9.42"


@dataclass(frozen=True)
class Credentials:
    session_id: str = field(repr=False)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-fA-F]{32}", self.session_id) is None:
            raise ReplayError("Set TIKTOK_SESSIONID to a valid 32-character session token.")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> Self:
        return cls(session_id=environ.get("TIKTOK_SESSIONID", ""))


class TikTokAPI:
    def __init__(self, *, client: httpx.Client, credentials: Credentials) -> None:
        self.client = client
        self.credentials = credentials

    def _get(self, endpoint: str, params: dict[str, str]) -> dict[str, object]:
        if endpoint not in {NOTICE_ENDPOINT, REPLAY_ENDPOINT}:
            raise ReplayError("API endpoint is not allowlisted.")
        body = bytearray()
        try:
            with self.client.stream(
                "GET",
                endpoint,
                params=params,
                headers={"Cookie": "sessionid=" + self.credentials.session_id},
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise ReplayError(
                        f"TikTok API HTTP error {response.status_code}; redirects are not followed."
                    )
                for chunk in response.iter_bytes(chunk_size=65536):
                    body.extend(chunk)
                    if len(body) > MAX_API_BYTES:
                        raise ReplayError("TikTok API response exceeds the size limit.")
        except httpx.HTTPError as error:
            raise ReplayError(
                "TikTok API network error; credentials and URLs were not logged."
            ) from error
        try:
            result = record(json.loads(body))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ReplayError("TikTok API returned invalid JSON.") from error
        code = result.get("status_code")
        if type(code) is not int:
            raise ReplayError("TikTok API omitted its status code.")
        if code == 20003:
            raise ReplayError("TikTok session expired or was rejected. Refresh TIKTOK_SESSIONID.")
        if code != 0:
            raise ReplayError(f"TikTok API rejected the request (code {code}).")
        return result

    def _notice_page(self, cursor: int) -> dict[str, object]:
        group = {
            "group": 540,
            "count": 20,
            "cursor": "",
            "max_time": cursor,
            "min_time": 0,
            "is_mark_read": 0,
            "first_read_time_in_session": 0,
            "first_read_time_in_session_for_friend": 0,
            "ab_settings": {},
        }
        response = self._get(
            NOTICE_ENDPOINT,
            {
                "aid": "1233",
                "app_name": "musical_ly",
                "version_code": CLIENT_VERSION_CODE,
                "version_name": CLIENT_VERSION_NAME,
                "group": json.dumps(group, separators=(",", ":")),
            },
        )
        return record(response.get("notice_list"))

    def _notice_pages(self) -> Iterator[list[object]]:
        cursor = 0
        page_count = 0
        while page_count < MAX_NOTICE_PAGES:
            page_count += 1
            page = self._notice_page(cursor)
            items = page.get("notice_list")
            more = page.get("has_more")
            if items is None and more == 0:
                items = []
            if not isinstance(items, list) or type(more) not in (int, bool) or more not in (0, 1):
                raise ReplayError("Malformed notification page; catalog completeness is unknown.")
            yield items
            if not more:
                return
            next_cursor = page.get("max_time")
            if (
                type(next_cursor) is not int
                or next_cursor <= 0
                or (cursor != 0 and next_cursor >= cursor)
            ):
                raise ReplayError(
                    "Notification pagination did not advance; refusing a partial catalog."
                )
            cursor = next_cursor
        raise ReplayError("Notification page limit reached; catalog is incomplete.")

    def validate_session(self) -> None:
        """Verify authentication and one notification page without resolving media."""
        next(self._notice_pages())

    def notices(self, *, anchor_id: str) -> Iterator[Notice]:
        numeric_id(anchor_id)
        seen: set[str] = set()
        for item in chain.from_iterable(self._notice_pages()):
            notice = notice_from_json(item)
            if (
                notice is not None
                and notice.anchor_id == anchor_id
                and notice.replay_id not in seen
            ):
                seen.add(notice.replay_id)
                yield notice

    def resolve(self, notice: Notice) -> Recording:
        response = self._get(
            REPLAY_ENDPOINT,
            {
                "aid": "1233",
                "room_ids": numeric_id(notice.replay_id),
                "user_type": notice.user_type,
                "need_share": "false",
            },
        )
        return recording_from_json(response.get("data"), replay_id=notice.replay_id)
