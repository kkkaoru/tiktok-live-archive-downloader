import json
from pathlib import Path

import pytest

from replay.caption_timing import SpokenToken
from replay.core import ReplayError
from replay.speaker_subtitles import SpeakerCue
from replay.youtube_caption_rebuild import load_voice_tokens, main, rebuild


def test_words_grow_then_switch_instead_of_sliding() -> None:
    tokens = [
        ("middle", SpokenToken("みんな", 1.0, 1.2, "w1")),
        ("middle", SpokenToken("の", 1.2, 1.3, "w1")),
        ("middle", SpokenToken("元に届いてる", 1.3, 1.9, "w1")),
        ("middle", SpokenToken("かな?", 1.9, 2.2, "w1")),
    ]
    reviewed = (
        SpeakerCue(1.0, 1.2, "middle", "みんな"),
        SpeakerCue(1.2, 1.3, "middle", "みんなの"),
        SpeakerCue(1.3, 1.9, "middle", "みんなの元に届いてる"),
        SpeakerCue(1.9, 2.2, "middle", "元に届いてるかな?"),
    )
    assert [cue.text for cue in rebuild(tokens, reviewed)] == [
        "みんな",
        "みんなの",
        "みんなの元に届いてる",
        "みんなの元に届いてるかな?",
    ]


def test_reviewed_speaker_and_correction_are_kept_and_unassigned_is_carried() -> None:
    tokens = [("right", SpokenToken("結末", 5.0, 5.4, "w2"))]
    reviewed = (
        SpeakerCue(5.0, 5.4, "left", "今月の月末"),
        SpeakerCue(9.0, 10.0, "unassigned", "忙しい"),
    )
    assert rebuild(tokens, reviewed) == (
        SpeakerCue(5.0, 5.48, "left", "月末"),
        SpeakerCue(9.0, 10.0, "unassigned", "忙しい"),
    )


def test_unmatched_tokens_use_their_voice_lane_and_shared_onsets_merge() -> None:
    tokens = [
        ("left", SpokenToken("あ", 1.0, 1.2, "w")),
        ("left", SpokenToken("い", 1.0, 1.4, "w")),
    ]
    assert rebuild(tokens, ()) == (SpeakerCue(1.0, 1.48, "left", "あい"),)


def test_same_lane_reviewed_slot_supplies_the_corrected_text() -> None:
    tokens = [("left", SpokenToken("ね", 1.0, 1.2, "w"))]
    reviewed = (
        SpeakerCue(1.0, 1.2, "right", "違う"),
        SpeakerCue(1.0, 1.2, "left", "別"),
    )
    assert rebuild(tokens, reviewed) == (SpeakerCue(1.0, 1.28, "left", "別"),)


def test_candidate_validation(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"voices": {}}))
    with pytest.raises(ReplayError, match="no voice list"):
        load_voice_tokens(path, {1: "left"})
    path.write_text(json.dumps({"voices": [{"voice": 2, "projected_tokens": []}]}))
    with pytest.raises(ReplayError, match="not mapped"):
        load_voice_tokens(path, {1: "left"})
    path.write_text(json.dumps({"voices": [{"voice": 1, "projected_tokens": [{"text": 1}]}]}))
    with pytest.raises(ReplayError, match="Invalid projected token"):
        load_voice_tokens(path, {1: "left"})


def test_cli_writes_rebuilt_cues(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    candidate = tmp_path / "c.json"
    candidate.write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "voice": 1,
                        "projected_tokens": [
                            {"text": " こんばんは", "start": 1, "end": 2, "utterance": "u"}
                        ],
                    }
                ]
            }
        )
    )
    reviewed = tmp_path / "r.csv"
    reviewed.write_text("start,end,speaker,text\n1,2,left,こんばんは\n", encoding="utf-8")
    output = tmp_path / "o.csv"
    arguments = [str(candidate), str(reviewed), str(output), "--lane", "1=left"]
    assert main(arguments) == 0
    assert output.read_text(encoding="utf-8") == (
        "start,end,speaker,text\n1.000,2.080,left,こんばんは\n"
    )
    assert json.loads(capsys.readouterr().out) == {"cues": 1}
    with pytest.raises(ReplayError, match="overwrite"):
        main(arguments)
    output.unlink()
    with pytest.raises(ReplayError, match="VOICE=lane"):
        main([*arguments[:3], "--lane", "x=left"])


def test_sub_millisecond_captions_are_not_emitted() -> None:
    tokens = [
        ("left", SpokenToken("あ", 1.0, 1.2, "w")),
        ("left", SpokenToken("い", 1.0004, 1.4, "w")),
    ]
    assert rebuild(tokens, ()) == (SpeakerCue(1.0004, 1.48, "left", "あい"),)
