import pytest

from replay.core import ReplayError
from replay.recordings import (
    Notice,
    notice_from_json,
    numeric_id,
    record,
    recording_from_json,
)


def test_notice_identity() -> None:
    assert notice_from_json(
        {
            "template_notice": {
                "schema_url": "sslocal://webcast_replay_video?roomId=1&anchor_id=2&user_type=1"
            }
        }
    ) == Notice(replay_id="1", anchor_id="2", user_type="1")


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"template_notice": {}},
        {"template_notice": {"schema_url": "https://www.tiktok.com/other"}},
    ],
)
def test_unrelated_notice(value: object) -> None:
    assert notice_from_json(value) is None


@pytest.mark.parametrize(
    "schema",
    [
        "https://[bad/",
        "sslocal://webcast_replay_video?roomId=1",
        "sslocal://webcast_replay_video?roomId=1&anchor_id=2&user_type=2",
        "sslocal://webcast_replay_video?roomId=one&anchor_id=2&user_type=1",
        "sslocal://webcast_replay_video?roomId=1&roomId=2&anchor_id=2&user_type=1",
    ],
)
def test_bad_notice(schema: str) -> None:
    with pytest.raises(ReplayError):
        notice_from_json({"template_notice": {"schema_url": schema}})


@pytest.mark.parametrize("value", [None, [], {1: "bad"}])
def test_bad_object(value: object) -> None:
    with pytest.raises(ReplayError, match="Malformed"):
        record(value)


@pytest.mark.parametrize("value", ["", "a", "1" * 33, 123])
def test_bad_id(value: object) -> None:
    with pytest.raises(ReplayError, match="Invalid numeric"):
        numeric_id(value)


def test_available_recording() -> None:
    result = recording_from_json(
        {
            "replays": [
                {
                    "id": "1",
                    "title": "Example",
                    "available": True,
                    "m3u8_url": "https://v16m.tiktokcdn.com/a.m3u8?secret=1",
                    "hls_video_meta_info": {"duration": 120.5},
                    "start_time": 1704067200,
                }
            ]
        },
        replay_id="1",
    )
    assert result.title == "Example"
    assert result.duration == 120.5
    assert result.start_time == 1704067200
    assert result.available is True
    assert "secret" not in repr(result)


@pytest.mark.parametrize(
    "value", [{"replays": []}, {"replays": [{"id": "1", "title": "Expired", "available": False}]}]
)
def test_unavailable_recording(value: object) -> None:
    result = recording_from_json(value, replay_id="1")
    assert result.available is False
    assert result.media_url is None


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"replays": [{}, {}]},
        {"replays": [{"id": "2"}]},
        {"replays": [{"id": "1", "title": 1, "available": True}]},
    ],
)
def test_invalid_replay_data(value: object) -> None:
    with pytest.raises(ReplayError):
        recording_from_json(value, replay_id="1")


@pytest.mark.parametrize(
    "url",
    [None, "", "http://v16m.tiktokcdn.com/a.m3u8", "https://evil.test/a.m3u8", "https://[bad/"],
)
def test_bad_media_url(url: object) -> None:
    with pytest.raises(ReplayError):
        recording_from_json(
            {
                "replays": [
                    {
                        "id": "1",
                        "title": "Example",
                        "available": True,
                        "m3u8_url": url,
                        "hls_video_meta_info": {"duration": 1},
                    }
                ]
            },
            replay_id="1",
        )


@pytest.mark.parametrize("duration", [True, "1", 0, -1, float("nan"), float("inf"), 10**400])
def test_bad_duration(duration: object) -> None:
    with pytest.raises(ReplayError, match="duration"):
        recording_from_json(
            {
                "replays": [
                    {
                        "id": "1",
                        "title": "Example",
                        "available": True,
                        "m3u8_url": "https://v16m.tiktokcdn.com/a.m3u8",
                        "hls_video_meta_info": {"duration": duration},
                    }
                ]
            },
            replay_id="1",
        )
