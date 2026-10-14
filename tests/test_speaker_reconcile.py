import json
import struct
from array import array
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.speaker_reconcile import (
    DiarizedSegment,
    Float32Stem,
    dominant_speaker,
    load_diarization,
    main,
    reconcile,
    source_seconds,
)
from replay.speaker_subtitles import SpeakerCue


def write_stem(path: Path, loud: tuple[float, float], *, seconds: int = 10) -> Path:
    rate = 100
    values = array(
        "f",
        (0.5 if loud[0] <= index / rate < loud[1] else 0.01 for index in range(seconds * rate)),
    )
    data = values.tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt) + 8 + 8 + len(data))
        + b"WAVE"
        + b"fact"
        + struct.pack("<I", 4)
        + struct.pack("<I", len(values))
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return path


def test_two_agreeing_estimators_relabel_and_disagreement_keeps_label(tmp_path: Path) -> None:
    stems = {
        "left": Float32Stem(write_stem(tmp_path / "l.wav", (1, 2))),
        "middle": Float32Stem(write_stem(tmp_path / "m.wav", (3, 5))),
    }
    segments = (DiarizedSegment(1, 2, 7), DiarizedSegment(3, 5, 7))
    cues = (
        SpeakerCue(1.1, 1.9, "middle", "左の声"),
        SpeakerCue(3.1, 3.9, "left", "中央の声"),
        SpeakerCue(6, 7, "left", "判定なし"),
        SpeakerCue(1.2, 1.8, "unassigned", "未確定"),
    )
    result, changes = reconcile(
        cues, segments=segments, diarized_lanes={7: "left"}, stems=stems, cuts=()
    )
    assert [cue.speaker for cue in result] == ["left", "left", "left", "unassigned"]
    assert changes == {("middle", "left"): 1}


def test_overlapping_or_sparse_diarization_is_not_dominant() -> None:
    segments = (DiarizedSegment(0, 1, 1), DiarizedSegment(0.5, 1, 2))
    assert dominant_speaker(0, 1, segments) is None
    assert dominant_speaker(0, 1, (DiarizedSegment(0, 0.5, 1),)) is None
    assert dominant_speaker(0, 1, (DiarizedSegment(0, 0.9, 1),)) == 1
    assert dominant_speaker(2, 3, segments) is None


def test_output_times_map_back_over_removed_frames() -> None:
    assert source_seconds(0.5, ((30, 59),)) == 0.5
    assert source_seconds(2.0, ((30, 59),)) == 3.0
    assert source_seconds(2.0, ((30, 59), (90, 119))) == 4.0


def test_stem_rms_and_format_validation(tmp_path: Path) -> None:
    stem = Float32Stem(write_stem(tmp_path / "v.wav", (1, 2)))
    assert round(stem.rms(1, 2), 3) == 0.5
    assert stem.rms(20, 30) == 0.0
    (tmp_path / "bad.wav").write_bytes(b"RIFX0000WAVE")
    with pytest.raises(ReplayError, match="RIFF/WAVE"):
        Float32Stem(tmp_path / "bad.wav")
    pcm = struct.pack("<HHIIHH", 1, 1, 100, 200, 2, 16)
    (tmp_path / "pcm.wav").write_bytes(
        b"RIFF" + struct.pack("<I", 28) + b"WAVEfmt " + struct.pack("<I", 16) + pcm
    )
    with pytest.raises(ReplayError, match="32-bit float"):
        Float32Stem(tmp_path / "pcm.wav")
    (tmp_path / "empty.wav").write_bytes(b"RIFF" + struct.pack("<I", 4) + b"WAVE")
    with pytest.raises(ReplayError, match="no format or data"):
        Float32Stem(tmp_path / "empty.wav")


def test_diarization_validation(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "speaker": 1}]}))
    assert load_diarization(path) == (DiarizedSegment(0.0, 1.0, 1),)
    path.write_text(json.dumps({"segments": [{"start": 1, "end": 1, "speaker": 1}]}))
    with pytest.raises(ReplayError, match="Invalid diarization"):
        load_diarization(path)
    path.write_text(json.dumps({"segments": {}}))
    with pytest.raises(ReplayError, match="bounded segment list"):
        load_diarization(path)


def test_unknown_lanes_are_rejected(tmp_path: Path) -> None:
    stems = {"left": Float32Stem(write_stem(tmp_path / "l.wav", (1, 2)))}
    with pytest.raises(ReplayError, match="identified speaker lanes"):
        reconcile((), segments=(), diarized_lanes={1: "unassigned"}, stems=stems, cuts=())


def test_cli_writes_new_cues_and_reports_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "cues.csv").write_text(
        'start,end,speaker,text\n1.1,1.9,middle,"左, の声"\n', encoding="utf-8"
    )
    (tmp_path / "d.json").write_text(
        json.dumps({"segments": [{"start": 1, "end": 2, "speaker": 2}]})
    )
    (tmp_path / "cuts.json").write_text("[]")
    left = write_stem(tmp_path / "l.wav", (1, 2))
    middle = write_stem(tmp_path / "m.wav", (5, 6))
    output = tmp_path / "out.csv"
    arguments = [
        str(tmp_path / "cues.csv"),
        str(tmp_path / "d.json"),
        str(tmp_path / "cuts.json"),
        str(output),
        "--lane",
        "2=left",
        "--stem",
        f"left={left}",
        "--stem",
        f"middle={middle}",
    ]
    assert main(arguments) == 0
    assert output.read_text(encoding="utf-8") == 'start,end,speaker,text\n1.1,1.9,left,"左, の声"\n'
    assert json.loads(capsys.readouterr().out) == {"middle->left": 1}
    with pytest.raises(ReplayError, match="overwrite"):
        main(arguments)
    output.unlink()
    with pytest.raises(ReplayError, match="KEY=VALUE"):
        main([*arguments[:4], "--lane", "broken"])
    (tmp_path / "cuts.json").write_text("{}")
    with pytest.raises(ReplayError, match="Cut frames"):
        main(arguments)
