"""YouTube channel labels stored by the web viewer's Worker (``/api/channels.json``).

Both viewers read the same R2 record; this CLI lists, sets or removes one channel's label with
the CLIPS_API_TOKEN bearer from ``web-viewer/secrets.json`` (never printed).
"""

import json
import sys
import urllib.error
import urllib.request

from replay.core import ReplayError
from replay.recording_viewer_steps import USER_AGENT, Fetch, _fetch, _viewer

USAGE = "Usage: python -m replay.youtube_channels list | set CHANNEL_ID LABEL | remove CHANNEL_ID"


def request(action: list[str], *, fetch: Fetch = _fetch) -> dict[str, object]:
    """Send one list/set/remove call and return the stored ``{version, channels}`` record."""
    match action:
        case ["list"]:
            method, body = "GET", None
        case ["set", channel_id, label]:
            method, body = "PUT", {"channelId": channel_id, "label": label}
        case ["remove", channel_id]:
            method, body = "DELETE", {"channelId": channel_id}
        case _:
            raise ReplayError(USAGE)
    domain, token = _viewer()
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False).encode()
    return fetch(
        urllib.request.Request(
            f"https://{domain}/api/channels.json", data=data, headers=headers, method=method
        )
    )


def main() -> None:
    try:
        answer = request(sys.argv[1:], fetch=_fetch)
    except (ReplayError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(answer, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
