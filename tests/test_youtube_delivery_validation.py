import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from replay import youtube_delivery_validation as validation
from replay.core import ReplayError


@pytest.fixture
def probe() -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "profile": "High",
                "width": 32,
                "height": 48,
                "avg_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
                "field_order": "progressive",
                "color_primaries": "bt709",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "bit_rate": "5234000",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "profile": "LC",
                "channels": 2,
                "channel_layout": "stereo",
                "sample_rate": "48000",
                "bit_rate": "376000",
            },
        ]
    }


@pytest.fixture
def movie(tmp_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    output = tmp_path / "fixture.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=32x48:rate=30:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-x264-params",
            "colorprim=bt709:transfer=bt709:colormatrix=bt709",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-b:a",
            "384k",
            "-movflags",
            "+faststart",
            "-use_editlist",
            "0",
            str(output),
        ],
        check=True,
        timeout=30,
        capture_output=True,
    )
    return output.resolve()


def test_real_fixture_verifies(
    tmp_path: Path, movie: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = shutil.which("ffprobe")
    assert executable is not None
    monkeypatch.setattr(validation, "REPOSITORY", tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    output = tmp_path / "edits/work/result"
    assert (
        validation.validate(movie, master=movie, output=output, ffprobe=Path(executable).resolve())
        is True
    )
    result = json.loads((output / "result.json").read_text())
    assert result["technicalPropertiesAccepted"] is True
    assert result["deliveryApproved"] is False
    assert result["container"]["editListCount"] == 0
    assert result["container"]["fastStart"] is True


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"1234",
        struct.pack(">I4s", 7, b"free"),
        struct.pack(">I4s", 100, b"moov"),
        struct.pack(">I4s", 1, b"mdat") + b"x",
        struct.pack(">I4sQ", 1, b"mdat", 2**64 - 1),
        struct.pack(">I4sQ", 1, b"mdat", 15),
        struct.pack(">I4s", 16, b"moov") + struct.pack(">I4s", 12, b"trak"),
    ],
)
def test_malformed_boxes_rejected(tmp_path: Path, payload: bytes) -> None:
    movie = tmp_path / "bad.mp4"
    movie.write_bytes(payload)
    with pytest.raises(ReplayError):
        validation.inspect_container(movie)


def test_extended_sparse_media_is_skipped(tmp_path: Path) -> None:
    movie = tmp_path / "sparse.mp4"
    with movie.open("wb") as stream:
        stream.write(b"\x00\x00\x00\x14ftypisom\x00\x00\x00\x00mp42")
        stream.write(b"\x00\x00\x00\x08moov")
        stream.write(struct.pack(">I4sQ", 1, b"mdat", 2**32))
        stream.seek(28 + 2**32 - 1)
        stream.write(b"\x00")
    assert validation.inspect_container(movie) == {
        "topLevelBoxes": ["ftyp", "moov", "mdat"],
        "boxCount": 3,
        "editListCount": 0,
        "fastStart": True,
        "failures": [],
    }


def test_zero_size_and_nested_editlist(tmp_path: Path) -> None:
    movie = tmp_path / "edit.mp4"
    movie.write_bytes(
        b"\x00\x00\x00\x14ftypisom\x00\x00\x00\x00mp42"
        b"\x00\x00\x00\x20moov\x00\x00\x00\x18trak"
        b"\x00\x00\x00\x10edts\x00\x00\x00\x08elst"
        b"\x00\x00\x00\x00mdatpayload"
    )
    assert validation.inspect_container(movie)["failures"] == [
        "Edit list present: preserve priming/sync and return to producer"
    ]
    assert movie.read_bytes().endswith(b"mdatpayload")


def test_tail_moov_and_fragments_fail(tmp_path: Path) -> None:
    movie = tmp_path / "tail.mp4"
    movie.write_bytes(
        b"\x00\x00\x00\x14ftypisom\x00\x00\x00\x00mp42"
        b"\x00\x00\x00\x08mdat\x00\x00\x00\x08moov"
        b"\x00\x00\x00\x08moof"
    )
    assert validation.inspect_container(movie)["failures"] == [
        "moov follows media data: Fast Start is absent",
        "Fragmented MP4 is outside this delivery contract",
    ]


@pytest.mark.parametrize(
    "brand",
    [
        b"\x00\x00\x00\x08ftyp",
        b"\x00\x00\x00\x11ftypisom0000x",
        b"\x00\x00\x00\x10ftypqt  0000",
        b"\x00\x00\x00\x10ftypzzzz0000",
    ],
)
def test_invalid_brands(tmp_path: Path, brand: bytes) -> None:
    movie = tmp_path / "brand.mp4"
    movie.write_bytes(brand + b"\x00\x00\x00\x08moov\x00\x00\x00\x08mdat")
    with pytest.raises(ReplayError):
        validation.inspect_container(movie)


@pytest.mark.parametrize("limit,value", [("MAX_DEPTH", -1), ("MAX_BOXES", 0)])
def test_parser_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str, value: int
) -> None:
    movie = tmp_path / "limit.mp4"
    movie.write_bytes(b"\x00\x00\x00\x08free")
    monkeypatch.setattr(validation, limit, value)
    with pytest.raises(ReplayError):
        validation.inspect_container(movie)


def test_nonexact_vbr_does_not_fail(probe: dict[str, object]) -> None:
    assert validation.check_streams(probe, probe)["failures"] == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("profile", "Baseline"),
        ("pix_fmt", "yuv422p"),
        ("field_order", "tt"),
        ("color_transfer", "unknown"),
        ("width", 64),
        ("avg_frame_rate", "25/1"),
    ],
)
def test_video_properties_fail(probe: dict[str, object], key: str, value: object) -> None:
    master = json.loads(json.dumps(probe))
    candidate = json.loads(json.dumps(probe))
    candidate["streams"][0][key] = value
    assert validation.check_streams(candidate, master)["failures"] != []


@pytest.mark.parametrize(
    "key,value",
    [("profile", "HE-AAC"), ("sample_rate", "44100"), ("channels", 1), ("channel_layout", "mono")],
)
def test_audio_properties_fail(probe: dict[str, object], key: str, value: object) -> None:
    candidate = json.loads(json.dumps(probe))
    candidate["streams"][1][key] = value
    assert validation.check_streams(candidate, probe)["failures"] != []


@pytest.mark.parametrize("bad", [{}, {"streams": "wrong"}, {"streams": []}])
def test_invalid_stream_shape(probe: dict[str, object], bad: object) -> None:
    with pytest.raises(ReplayError):
        validation.check_streams(bad, probe)


@pytest.mark.parametrize("rate", [None, "0/0", "0/1", "invalid"])
def test_invalid_rate(probe: dict[str, object], rate: object) -> None:
    candidate = json.loads(json.dumps(probe))
    candidate["streams"][0]["avg_frame_rate"] = rate
    with pytest.raises(ReplayError):
        validation.check_streams(candidate, probe)


def test_invalid_master_dimensions(probe: dict[str, object]) -> None:
    master = json.loads(json.dumps(probe))
    master["streams"][0]["width"] = None
    with pytest.raises(ReplayError, match="dimensions"):
        validation.check_streams(probe, master)


def test_outside_evidence_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="repository"):
        validation.validate(tmp_path, master=tmp_path, output=Path("relative"), ffprobe=tmp_path)


def test_linked_evidence_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, "REPOSITORY", tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    (tmp_path / "edits/work/link").symlink_to(tmp_path / "edits/work", target_is_directory=True)
    with pytest.raises(ReplayError, match="symlinks"):
        validation.validate(
            tmp_path, master=tmp_path, output=tmp_path / "edits/work/link/output", ffprobe=tmp_path
        )


def test_invalid_inputs_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, "REPOSITORY", tmp_path)
    with pytest.raises(ReplayError, match="canonical"):
        validation.validate(
            tmp_path, master=tmp_path, output=tmp_path / "edits/work/output", ffprobe=tmp_path
        )


def test_cli_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["validation", "movie", "master", "out", "--ffprobe", "ffprobe"]
    )
    assert validation.main() == 1


def test_media_mutation_is_rejected(
    tmp_path: Path, movie: Path, probe: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "REPOSITORY", tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)

    def changing_probe(movie: Path, *, executable: Path, output: Path) -> object:
        with movie.open("ab") as stream:
            stream.write(b"changed")
        return probe

    monkeypatch.setattr(validation, "_probe", changing_probe)
    with pytest.raises(ReplayError, match="changed"):
        validation.validate(
            movie, master=movie, output=tmp_path / "edits/work/result", ffprobe=movie
        )
    assert not (tmp_path / "edits/work/result/result.json").exists()


def test_probe_metadata_bound(tmp_path: Path, movie: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = shutil.which("ffprobe")
    assert executable is not None
    monkeypatch.setattr(validation, "REPOSITORY", tmp_path)
    monkeypatch.setattr(validation, "MAX_PROBE_BYTES", 1)
    (tmp_path / "edits/work").mkdir(parents=True)
    with pytest.raises(ReplayError, match="metadata"):
        validation.validate(
            movie,
            master=movie,
            output=tmp_path / "edits/work/result",
            ffprobe=Path(executable).resolve(),
        )


def test_real_edit_lists_fail_without_modifying_movie(tmp_path: Path, movie: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    output = tmp_path / "with-edits.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(movie),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-use_editlist",
            "1",
            str(output),
        ],
        check=True,
        timeout=30,
        capture_output=True,
    )
    original = output.read_bytes()
    assert validation.inspect_container(output)["failures"] == [
        "Edit list present: preserve priming/sync and return to producer"
    ]
    assert output.read_bytes() == original
