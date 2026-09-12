import json

from replay.android_log import replay_candidates


def test_pair_and_update() -> None:
    result = replay_candidates(
        [
            '{"payload":{"type":"url","url":"https://webcast.tiktokv.com/webcast/room/replay/info/?room_ids=123"}}',
            '{"payload":{"type":"url","url":"https://v16m.tiktokcdn.com/first.m3u8"}}',
            '{"payload":{"type":"url","url":"https://v16m.tiktokcdn.com/new.m3u8"}}',
        ]
    )
    assert list(result) == ["123"]
    assert result["123"].url == "https://v16m.tiktokcdn.com/new.m3u8"
    assert result["123"].headers == {}
    assert result["123"].source == "api-advertised"


def test_unrelated_and_invalid_lines() -> None:
    assert (
        replay_candidates(
            [
                "x" * 65537,
                "not json",
                "[]",
                '{"payload":"ready"}',
                '{"payload":{"type":"url","url":5}}',
                '{"payload":{"type":"url","url":"/relative.m3u8"}}',
                '{"payload":{"type":"url","url":"http://cdn.test/a.m3u8"}}',
                '{"payload":{"type":"url","url":"https://[invalid/"}}',
                '{"payload":{"type":"url","url":"https://v16m.tiktokcdn.com/no-room.m3u8"}}',
            ]
        )
        == {}
    )


def test_ambiguous_rooms_clear_association() -> None:
    assert (
        replay_candidates(
            [
                '{"payload":{"type":"url","url":"https://webcast.tiktokv.com/webcast/room/replay/info/?room_ids=123"}}',
                '{"payload":{"type":"url","url":"https://webcast.tiktokv.com/webcast/room/replay/info/?room_ids=123,456"}}',
                '{"payload":{"type":"url","url":"https://v16m.tiktokcdn.com/a.m3u8"}}',
            ]
        )
        == {}
    )


def test_untrusted_api_host() -> None:
    assert (
        replay_candidates(
            [
                json.dumps(
                    {
                        "payload": {
                            "type": "url",
                            "url": "https://example.com/webcast/room/replay/info/?room_ids=123",
                        }
                    }
                ),
                '{"payload":{"type":"url","url":"https://v16m.tiktokcdn.com/a.m3u8"}}',
            ]
        )
        == {}
    )
