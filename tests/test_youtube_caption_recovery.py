import json
import struct
import wave
from array import array
from pathlib import Path

import pytest

from replay.caption_timing import SpokenToken
from replay.core import ReplayError
from replay.speaker_subtitles import SpeakerCue
from replay.youtube_caption_recovery import main, recover, segment_tokens

RATE = 16000


def voice(seconds: float, loud: list[tuple[float, float]]) -> array[int]:
    samples = array("h")
    for index in range(round(seconds * RATE)):
        moment = index / RATE
        voiced = any(start <= moment < end for start, end in loud)
        samples.append((8000 if voiced else 20) * (1 if index % 2 else -1))
    return samples


def test_segment_tokens_keep_only_clocked_words() -> None:
    segment = {
        "words": [
            {"word": " こん", "start": 1.0, "end": 1.4},
            {"word": "ば", "start": 1.4, "end": 1.4},
            {"word": "", "start": 1.5, "end": 1.6},
            {"word": "んは", "start": 1.6, "end": 2.0},
        ]
    }
    assert segment_tokens(segment, "u") == (
        SpokenToken("こん", 1.0, 1.4, "u"),
        SpokenToken("んは", 1.6, 2.0, "u"),
    )
    assert segment_tokens({"words": None}, "u") == ()


def test_audible_span_without_cue_gets_labelled_growing_captions() -> None:
    cues = (SpeakerCue(1.0, 3.0, "middle", "最初"),)
    calls: list[tuple[float, float]] = []

    def transcribe(start: float, end: float) -> object:
        calls.append((start, end))
        return {
            "segments": [
                {
                    "words": [
                        {"word": "こんばん", "start": 1.5, "end": 2.0},
                        {"word": "は", "start": 2.0, "end": 2.3},
                    ]
                },
                {"words": [{"word": "x", "start": 2.4, "end": 2.4}]},
            ]
        }

    result, provenance = recover(
        cues, voice(12.0, [(1, 3), (5, 8)]), transcribe, lambda start, end: "right"
    )
    assert calls == [(3.5, 9.5)]
    assert result == (
        SpeakerCue(1.0, 3.0, "middle", "最初"),
        SpeakerCue(5.0, 5.5, "right", "こんばん"),
        SpeakerCue(5.5, 5.88, "right", "こんばんは"),
    )
    assert provenance == {
        "holes": [
            {
                "start": 5.0,
                "end": 8.0,
                "owner_start": 4.5,
                "owner_end": 8.5,
                "recoveredText": "こんばんは",
            }
        ],
        "recoveredCues": 2,
    }


def write_wav(path: Path, samples: array[int]) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes(samples.tobytes())


def write_stem(path: Path, loud: tuple[float, float]) -> str:
    rate = 100
    values = array(
        "f", (0.5 if loud[0] <= index / rate < loud[1] else 0.01 for index in range(1200))
    )
    data = values.tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data))
        + b"WAVEfmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return str(path)


def test_cli_labels_only_when_diarization_and_stem_agree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_wav(tmp_path / "vocals.wav", voice(12.0, [(1, 3), (5, 8)]))
    (tmp_path / "cues.csv").write_text(
        "start,end,speaker,text\n1,3,middle,最初\n", encoding="utf-8"
    )
    (tmp_path / "d.json").write_text(
        json.dumps({"segments": [{"start": 4, "end": 9, "speaker": 2}]})
    )
    (tmp_path / "cuts.json").write_text("[]")
    arguments = [
        str(tmp_path / "cues.csv"),
        str(tmp_path / "vocals.wav"),
        str(tmp_path / "d.json"),
        str(tmp_path / "cuts.json"),
        str(tmp_path / "out.csv"),
        str(tmp_path / "provenance.json"),
        "--lane",
        "2=left",
        "--stem",
        f"left={write_stem(tmp_path / 'l.wav', (5, 8))}",
        "--stem",
        f"right={write_stem(tmp_path / 'r.wav', (0, 1))}",
    ]

    def transcribe(path: Path) -> object:
        return {"segments": [{"words": [{"word": "続き", "start": 2.0, "end": 2.5}]}]}

    assert main(arguments, transcribe=transcribe) == 0
    assert json.loads(capsys.readouterr().out) == {"cues": 2, "recovered": 1}
    assert (tmp_path / "out.csv").read_text(encoding="utf-8") == (
        "start,end,speaker,text\n1.000,3.000,middle,最初\n5.500,6.080,left,続き\n"
    )
    with pytest.raises(ReplayError, match="overwrite"):
        main(arguments, transcribe=transcribe)
    (tmp_path / "out.csv").unlink()
    (tmp_path / "provenance.json").unlink()
    (tmp_path / "d.json").write_text(
        json.dumps({"segments": [{"start": 4, "end": 9, "speaker": 9}]})
    )
    assert main(arguments, transcribe=transcribe) == 0
    assert "5.500,6.080,unassigned,続き" in (tmp_path / "out.csv").read_text(encoding="utf-8")
    capsys.readouterr()
    with pytest.raises(ReplayError, match="pinned recognizer"):
        main([*arguments[:4], str(tmp_path / "o2.csv"), str(tmp_path / "p2.json")])
