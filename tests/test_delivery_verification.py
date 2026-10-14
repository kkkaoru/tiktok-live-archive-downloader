"""Synthetic final-media receipts; never use recordings or live jobs."""

import json
import os
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from replay import delivery_clock_contract as clock
from replay import delivery_verification as verification
from replay.core import ReplayError
from replay.delivery_render import CONFIG_KEYS, load_config
from replay.recordings import record
from replay.stages import Stage, StageError, run_stage


@pytest.fixture
def config(tmp_path: Path) -> Path:
    cuts = tmp_path / "cuts.json"
    cuts.write_text(
        json.dumps(
            {"cuts": [{"source_start_frame": 0, "source_end_frame": 6, "output_start_frame": 0}]}
        ),
        encoding="utf-8",
    )
    raw: dict[str, object] = dict.fromkeys(CONFIG_KEYS, str(tmp_path / "unused"))
    raw.update(
        version=1,
        recording="synthetic",
        rate=30,
        width=64,
        height=64,
        cuts=str(cuts),
        deliveryRoot=str(tmp_path),
        encoding={"codec": "hevc"},
        motion={"template": str(tmp_path / "template"), "expectedText": "Title", "finish": []},
        audio={"source": str(tmp_path / "voice"), "clock": str(tmp_path / "clock")},
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.fixture
def fake_tools(tmp_path: Path) -> tuple[Path, Path]:
    """Trusted synthetic executables exercise real pipe capture and JSON boundaries."""
    tool = tmp_path / "tool"
    tool.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, os\n"
        "from pathlib import Path\n"
        "assert os.environ['TMPDIR'] == os.getcwd()\n"
        "assert os.environ['TEMP'] == os.getcwd()\n"
        "if '-show_streams' in sys.argv:\n"
        " print(json.dumps({'streams': ["
        "{'index':0,'codec_type':'video','codec_name':'hevc','codec_tag_string':'hvc1',"
        "'width':64,'height':64,'avg_frame_rate':'30/1','time_base':'1/30'},"
        "{'index':1,'codec_type':'audio','codec_name':'aac','sample_rate':'48000','time_base':'1/48000'}]}))\n"
        "elif '-show_frames' in sys.argv:\n"
        " print(json.dumps({'frames': [dict(media_type='video',pts=n,duration=1) for n in range(6)]"
        " + [dict(media_type='audio',pts=0,nb_samples=9600)]}))\n"
        "elif '-show_packets' in sys.argv:\n"
        " print(json.dumps({'packets': [dict(stream_index=0,pts=n,duration=1) for n in range(6)]"
        " + [dict(stream_index=1,pts=-1024,duration=1024),"
        "dict(stream_index=1,pts=0,duration=9600)],"
        " 'streams':[dict(index=0,nb_read_packets='6'),dict(index=1,nb_read_packets='2')]}))\n",
        encoding="utf-8",
    )
    tool.chmod(0o700)
    return tool, tool


def test_success(config: Path, fake_tools: tuple[Path, Path]) -> None:
    movie = config.parent / "movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = config.parent / "evidence"
    assert verification.verify(config, movie, evidence, ffmpeg=fake_tools[0], ffprobe=fake_tools[1])
    receipt = record(json.loads((evidence / "receipt.json").read_text(encoding="utf-8")))
    assert receipt["technicalAccepted"] is True
    assert receipt["fullDecode"] is True
    assert receipt["videoPackets"] == 6
    assert receipt["videoClock"] == {"frames": 6, "start": "0", "end": "1/5", "exactCFR": True}
    assert receipt["audioPacketClock"] == {
        "count": 2,
        "start": "-8/375",
        "end": "1/5",
        "span": "83/375",
        "durationTicksSum": 10624,
    }
    assert receipt["audioEndMinusVideoEnd"] == "0"
    assert receipt["approvals"] == dict(
        captions=False,
        sourceCaptions=False,
        musicRemoval=False,
        transcription=False,
        FCP=False,
        quality=False,
        sync=False,
    )
    with pytest.raises(FileExistsError):
        verification.verify(config, movie, evidence, ffmpeg=fake_tools[0], ffprobe=fake_tools[1])


def test_decode_retained_when_probe_fails(config: Path, fake_tools: tuple[Path, Path]) -> None:
    movie = config.parent / "movie.mp4"
    movie.write_bytes(b"synthetic")
    fail = config.parent / "fail"
    fail.write_text(
        f"#!{sys.executable}\nimport sys\nprint('probe failed',file=sys.stderr)\nsys.exit(7)\n"
    )
    fail.chmod(0o700)
    evidence = config.parent / "evidence"
    assert not verification.verify(config, movie, evidence, ffmpeg=fake_tools[0], ffprobe=fail)
    receipt = record(json.loads((evidence / "receipt.json").read_text()))
    assert receipt["fullDecode"] is True
    assert receipt["technicalAccepted"] is False
    assert (evidence / "streams.stderr").read_text() == "probe failed\n"
    assert record(json.loads((evidence / "streams.process.json").read_text()))["returncode"] == 7
    assert record(json.loads((evidence / "decode.process.json").read_text()))["returncode"] == 0


@pytest.mark.parametrize("change", ["movie", "cuts", "config"])
def test_mutation(config: Path, fake_tools: tuple[Path, Path], change: str) -> None:
    movie = config.parent / "movie.mp4"
    movie.write_bytes(b"synthetic")
    target = {"movie": movie, "cuts": config.parent / "cuts.json", "config": config}[change]
    mutate = config.parent / "mutate"
    mutate.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\np=Path({str(target)!r})\n"
        "p.write_bytes(p.read_bytes()+b' ')\n"
    )
    mutate.chmod(0o700)
    evidence = config.parent / "evidence"
    assert not verification.verify(config, movie, evidence, ffmpeg=mutate, ffprobe=fake_tools[1])
    receipt = record(json.loads((evidence / "receipt.json").read_text()))
    assert receipt["bindingError"] == "Bound input changed during verification"
    assert receipt["technicalAccepted"] is False


@pytest.mark.parametrize("mode", ["symlink", "hardlink", "outside", "missing"])
def test_candidate_paths(config: Path, fake_tools: tuple[Path, Path], mode: str) -> None:
    movie = config.parent / "movie.mp4"
    original = config.parent / "original.mp4"
    original.write_bytes(b"synthetic")
    if mode == "symlink":
        movie.symlink_to(original)
    elif mode == "hardlink":
        os.link(original, movie)
    elif mode == "outside":
        movie = config.parent.parent / "not-in-delivery.mp4"
    evidence = config.parent / "evidence"
    assert not verification.verify(
        config, movie, evidence, ffmpeg=fake_tools[0], ffprobe=fake_tools[1]
    )
    assert record(json.loads((evidence / "receipt.json").read_text()))["technicalAccepted"] is False
    assert not (evidence / "decode.stdout").exists()


def test_evidence_escape(config: Path, fake_tools: tuple[Path, Path]) -> None:
    link = config.parent / "link"
    link.symlink_to(config.parent, target_is_directory=True)
    with pytest.raises(ReplayError, match="canonical"):
        verification.verify(
            config, config, link / "new", ffmpeg=fake_tools[0], ffprobe=fake_tools[1]
        )
    with pytest.raises(ReplayError, match="inside"):
        verification.verify(
            config,
            config,
            verification.REPOSITORY.parent / "never-create",
            ffmpeg=fake_tools[0],
            ffprobe=fake_tools[1],
        )


@pytest.mark.parametrize(
    "rows,message",
    [
        ([{"pts": 0, "duration": 1}], "count"),
        ([{"pts": 1, "duration": 1}, {"pts": 2, "duration": 1}], "PTS"),
        ([{"pts": 0, "duration": 1}, {"pts": 0, "duration": 1}], "PTS"),
        ([{"pts": 0, "duration": 1}, {"pts": 1, "duration": 2}], "duration"),
        ([{"pts": 0, "duration": 1}, {"pts": 1}], "integer"),
    ],
)
def test_bad_video_clock(rows: list[dict[str, object]], message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        verification._video(rows, scale=Fraction(1, 30), rate=30, frames=2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("codec_name", "h264"),
        ("codec_tag_string", "hev1"),
        ("width", 32),
        ("avg_frame_rate", "25/1"),
    ],
)
def test_bad_streams(config: Path, field: str, value: object) -> None:
    video: dict[str, object] = dict(
        codec_type="video",
        codec_name="hevc",
        codec_tag_string="hvc1",
        width=64,
        height=64,
        avg_frame_rate="30/1",
    )
    video[field] = value
    with pytest.raises(ReplayError):
        verification._streams(
            {
                "streams": [
                    video,
                    {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
                ]
            },
            load_config(config),
        )


def test_h264_requires_explicit_contract_and_exact_avc1_tag(
    config: Path, fake_tools: tuple[Path, Path]
) -> None:
    tool = fake_tools[0]
    movie = config.parent / "movie.mp4"
    movie.write_bytes(b"synthetic")
    assert not verification.verify(
        config,
        movie,
        config.parent / "hevc-not-h264",
        ffmpeg=tool,
        ffprobe=tool,
        video_contract="h264",
    )
    tool.write_text(tool.read_text().replace("'hevc'", "'h264'").replace("'hvc1'", "'avc1'"))
    assert not verification.verify(
        config, movie, config.parent / "default-rejected", ffmpeg=tool, ffprobe=tool
    )
    evidence = config.parent / "h264-accepted"
    assert verification.verify(
        config, movie, evidence, ffmpeg=tool, ffprobe=tool, video_contract="h264"
    )
    receipt = record(json.loads((evidence / "receipt.json").read_text()))
    assert receipt["expectedVideoContract"] == "h264"
    assert receipt["fullDecode"] is True
    tool.write_text(tool.read_text().replace("'avc1'", "'hvc1'"))
    assert not verification.verify(
        config,
        movie,
        config.parent / "wrong-tag",
        ffmpeg=tool,
        ffprobe=tool,
        video_contract="h264",
    )


@pytest.mark.parametrize(
    "streams",
    [
        [],
        [{"codec_type": "video"}],
        [{"codec_type": "video"}, {"codec_type": "video"}, {"codec_type": "audio"}],
    ],
)
def test_missing_extra_streams(config: Path, streams: list[dict[str, object]]) -> None:
    with pytest.raises(ReplayError):
        verification._streams({"streams": streams}, load_config(config))


@pytest.mark.parametrize(
    "script,timeout,message",
    [
        ("import time; time.sleep(3)", 1, "runtime"),
        ("import sys; print('bad',file=sys.stderr)", 3, "child failed"),
        ("import sys; sys.exit(4)", 3, "child failed"),
        ("print('x'*10000)", 3, "byte limit"),
    ],
)
def test_process_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str, timeout: int, message: str
) -> None:
    monkeypatch.setattr(verification, "MAX_BYTES", 1000)
    with pytest.raises(ReplayError, match=message):
        verification._run(
            [sys.executable, "-c", script], directory=tmp_path, name="run", timeout=timeout
        )
    assert record(json.loads((tmp_path / "run.process.json").read_text()))["returncode"] is not None


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("range(6)", "range(5)", "EOF"),
        ("nb_read_packets='6'", "nb_read_packets='5'", "packet count"),
        ("'codec_name':'aac'", "'codec_name':None", "audio codec"),
        ("'avg_frame_rate':'30/1'", "'avg_frame_rate':'25/1'", "rate mismatch"),
        ("duration=1)", "duration=2)", "duration/endpoint"),
    ],
)
def test_probe_validation_receipt(
    config: Path,
    fake_tools: tuple[Path, Path],
    old: str,
    new: str,
    message: str,
) -> None:
    tool = fake_tools[0]
    tool.write_text(tool.read_text().replace(old, new), encoding="utf-8")
    movie = config.parent / "movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = config.parent / "evidence"
    assert not verification.verify(config, movie, evidence, ffmpeg=tool, ffprobe=tool)
    receipt = record(json.loads((evidence / "receipt.json").read_text()))
    assert message in str(receipt["error"])
    assert receipt["fullDecode"] is True
    assert receipt["technicalAccepted"] is False


def test_bad_probe_json(config: Path, fake_tools: tuple[Path, Path]) -> None:
    probe = config.parent / "probe"
    probe.write_text(f"#!{sys.executable}\nprint('not json')\n", encoding="utf-8")
    probe.chmod(0o700)
    movie = config.parent / "movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = config.parent / "evidence"
    assert not verification.verify(config, movie, evidence, ffmpeg=fake_tools[0], ffprobe=probe)
    assert (evidence / "streams.stdout").read_text() == "not json\n"
    assert record(json.loads((evidence / "receipt.json").read_text()))["technicalAccepted"] is False


def test_decode_failed_receipt(config: Path, fake_tools: tuple[Path, Path]) -> None:
    tool = fake_tools[0]
    tool.write_text(
        f"#!{sys.executable}\nimport sys\nprint('invalid EOF', file=sys.stderr)\nsys.exit(1)\n"
    )
    movie = config.parent / "movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = config.parent / "evidence"
    assert not verification.verify(config, movie, evidence, ffmpeg=tool, ffprobe=tool)
    receipt = record(json.loads((evidence / "receipt.json").read_text()))
    assert receipt.get("fullDecode", False) is False
    assert (evidence / "decode.stderr").read_text() == "invalid EOF\n"


def test_process_missing(tmp_path: Path) -> None:
    with pytest.raises(ReplayError):
        verification._run([str(tmp_path / "missing")], directory=tmp_path, name="run", timeout=1)
    assert record(json.loads((tmp_path / "run.process.json").read_text()))["returncode"] is None


@pytest.mark.parametrize("value", [True, 1.1, None, "N/A"])
def test_invalid_integer(value: object) -> None:
    with pytest.raises(ReplayError):
        verification._int(value)


@pytest.mark.parametrize("value", [None, "0/1", "1/1000000001", "30"])
def test_invalid_scale(value: object) -> None:
    with pytest.raises(ReplayError):
        verification._scale(value)


@pytest.mark.parametrize(
    "cuts",
    [
        [{"source_start_frame": 1, "source_end_frame": 1}],
        [{"source_start_frame": 0, "source_end_frame": 5, "output_start_frame": 1}],
        [
            {"source_start_frame": 0, "source_end_frame": 5},
            {"source_start_frame": 3, "source_end_frame": 6},
        ],
        [{"source_start_frame": 0, "source_end_frame": 4000001}],
    ],
)
def test_bad_cuts(config: Path, cuts: list[dict[str, object]]) -> None:
    (config.parent / "cuts.json").write_text(json.dumps({"cuts": cuts}))
    with pytest.raises(ReplayError):
        verification._expected(load_config(config))


def test_json_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "large.json"
    path.write_text("{}")
    monkeypatch.setattr(verification, "MAX_BYTES", 1)
    with pytest.raises(ReplayError, match="byte"):
        verification._json(path)


@pytest.mark.parametrize("rows", [[], [{"pts": 0, "nb_samples": 0}]])
def test_audio_invalid(rows: list[dict[str, object]]) -> None:
    with pytest.raises(ReplayError):
        verification._audio(rows, scale=Fraction(1, 48000), sample_rate=48000)


def test_cli(config: Path, fake_tools: tuple[Path, Path]) -> None:
    movie = config.parent / "movie.mp4"
    movie.write_bytes(b"synthetic")
    args = [
        str(config),
        str(movie),
        str(config.parent / "evidence"),
        "--ffmpeg",
        str(fake_tools[0]),
        "--ffprobe",
        str(fake_tools[1]),
    ]
    assert verification.main(args) == 0
    assert verification.main(args) == 1
    args[2] = str(config.parent / "bad-timeout")
    assert verification.main(args + ["--timeout", "0"]) == 1
    args[2] = str(config.parent / "bad-tool")
    args[4] = "missing"
    assert verification.main(args) == 1


def test_real_hevc_aac(config: Path) -> None:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    movie = config.parent / "fixture.mp4"
    verification._run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=64x64:r=30:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.2",
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=error:pools=1",
            "-tag:v",
            "hvc1",
            "-c:a",
            "aac",
            str(movie),
        ],
        directory=config.parent,
        name="fixture",
        timeout=30,
    )
    evidence = config.parent / "real-evidence"
    assert verification.verify(config, movie, evidence, ffmpeg=Path(ffmpeg), ffprobe=Path(ffprobe))
    receipt = record(json.loads((evidence / "receipt.json").read_text()))
    assert receipt["technicalAccepted"] is True
    assert receipt["videoPackets"] == 6
    assert receipt["fullDecode"] is True


@pytest.mark.parametrize(
    "name,options",
    [
        ("streams", ["-show_streams", "-show_format"]),
        (
            "frames",
            [
                "-show_frames",
                "-show_entries",
                "frame=media_type,pts,duration,pkt_duration,nb_samples",
            ],
        ),
        (
            "packets",
            [
                "-show_packets",
                "-count_packets",
                "-show_entries",
                "packet=stream_index,pts,dts,duration:stream=index,nb_read_packets",
            ],
        ),
    ],
)
def test_h264_aac_probe_threads_preserve_every_selected_entry(
    config: Path, name: str, options: list[str]
) -> None:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    movie = config.parent / "h264-aac.mp4"
    verification._run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=64x64:r=30:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.2",
            "-c:v",
            "libx264",
            "-tag:v",
            "avc1",
            "-c:a",
            "aac",
            str(movie),
        ],
        directory=config.parent,
        name="h264-fixture",
        timeout=30,
    )
    single = verification._run(
        [
            ffprobe,
            "-v",
            "error",
            "-threads",
            "1",
            "-protocol_whitelist",
            "file,pipe",
            "-of",
            "json",
            *options,
            str(movie),
        ],
        directory=config.parent,
        name=f"{name}-single",
        timeout=30,
    )
    automatic = verification._run(
        [
            ffprobe,
            "-v",
            "error",
            "-threads",
            "0",
            "-protocol_whitelist",
            "file,pipe",
            "-of",
            "json",
            *options,
            str(movie),
        ],
        directory=config.parent,
        name=f"{name}-automatic",
        timeout=30,
    )
    if name == "frames":
        # Parallel decoders can interleave audio/video reports differently.
        # Preserve every selected field and the original order within each stream.
        single_frames = verification._rows(
            record(json.loads(single.read_text(encoding="utf-8"))), "frames"
        )
        auto_frames = verification._rows(
            record(json.loads(automatic.read_text(encoding="utf-8"))), "frames"
        )
        assert {row["media_type"] for row in single_frames} == {"video", "audio"}
        assert {row["media_type"] for row in auto_frames} == {"video", "audio"}
        assert [row for row in single_frames if row["media_type"] == "video"] == [
            row for row in auto_frames if row["media_type"] == "video"
        ]
        assert [row for row in single_frames if row["media_type"] == "audio"] == [
            row for row in auto_frames if row["media_type"] == "audio"
        ]
    else:
        assert single.read_bytes() == automatic.read_bytes()
    evidence = config.parent / "automatic-verification"
    assert verification.verify(
        config,
        movie,
        evidence,
        ffmpeg=Path(ffmpeg),
        ffprobe=Path(ffprobe),
        video_contract="h264",
    )
    if name == "frames":
        verified_frames = verification._rows(
            record(json.loads((evidence / "frames.stdout").read_text(encoding="utf-8"))), "frames"
        )
        assert [row for row in verified_frames if row["media_type"] == "video"] == [
            row for row in auto_frames if row["media_type"] == "video"
        ]
        assert [row for row in verified_frames if row["media_type"] == "audio"] == [
            row for row in auto_frames if row["media_type"] == "audio"
        ]
    else:
        assert (evidence / f"{name}.stdout").read_bytes() == automatic.read_bytes()
    process = record(json.loads((evidence / f"{name}.process.json").read_text(encoding="utf-8")))
    command = process["command"]
    assert isinstance(command, list)
    assert command[1:9] == [
        "-v",
        "error",
        "-threads",
        "0",
        "-protocol_whitelist",
        "file,pipe",
        "-of",
        "json",
    ]
    receipt = record(json.loads((evidence / "receipt.json").read_text(encoding="utf-8")))
    assert receipt["videoClock"] == {"frames": 6, "start": "0", "end": "1/5", "exactCFR": True}
    assert receipt["videoPackets"] == 6
    assert receipt["fullDecode"] is True
    assert receipt["technicalAccepted"] is True
    packet_clock = record(receipt["audioPacketClock"])
    assert packet_clock["start"] == "-8/375"
    assert receipt["approvals"] == dict(
        captions=False,
        sourceCaptions=False,
        musicRemoval=False,
        transcription=False,
        FCP=False,
        quality=False,
        sync=False,
    )


@pytest.mark.parametrize("changed", ["movie", "probe"])
def test_probe_stage_refuses_changed_input_or_tool_on_resume(
    config: Path, fake_tools: tuple[Path, Path], changed: str
) -> None:
    movie = config.parent / "movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = config.parent / "stage-evidence"
    stage = Stage(
        name="verify-media",
        command=("verify-immutable-synthetic-media",),
        inputs=(config, config.parent / "cuts.json", movie, fake_tools[1]),
        outputs=(evidence / "receipt.json",),
        timeout_seconds=30,
    )
    calls: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        calls.append(command)
        assert verification.verify(
            config, movie, evidence, ffmpeg=fake_tools[0], ffprobe=fake_tools[1]
        )

    state = config.parent / "stage-state"
    assert run_stage(stage, state_directory=state, execute=execute) == "completed"
    retained = (evidence / "receipt.json").read_bytes()
    target = {"movie": movie, "probe": fake_tools[1]}[changed]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(StageError, match="Completed stage or its inputs changed"):
        run_stage(stage, state_directory=state, execute=execute)
    assert calls == [("verify-immutable-synthetic-media",)]
    assert (evidence / "receipt.json").read_bytes() == retained


def test_probe_tool_mutation_cannot_finish_or_retry_stage(
    config: Path, fake_tools: tuple[Path, Path]
) -> None:
    movie = config.parent / "movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = config.parent / "stage-evidence"
    stage = Stage(
        name="verify-media",
        command=("verify-immutable-synthetic-media",),
        inputs=(config, config.parent / "cuts.json", movie, fake_tools[1]),
        outputs=(evidence / "receipt.json",),
        timeout_seconds=30,
    )
    calls: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        calls.append(command)
        assert verification.verify(
            config, movie, evidence, ffmpeg=fake_tools[0], ffprobe=fake_tools[1]
        )
        probe = fake_tools[1]
        probe.write_bytes(probe.read_bytes() + b"\n")

    state = config.parent / "stage-state"
    with pytest.raises(StageError, match="immutable inputs"):
        run_stage(stage, state_directory=state, execute=execute)
    retained = (evidence / "receipt.json").read_bytes()
    with pytest.raises(StageError, match="Completed stage or its inputs changed"):
        run_stage(stage, state_directory=state, execute=execute)
    assert calls == [("verify-immutable-synthetic-media",)]
    assert (evidence / "receipt.json").read_bytes() == retained


@pytest.fixture
def clock_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(clock, "ROOT", tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    edit_root = tmp_path / "edits/replay-123"
    edit_root.mkdir()
    source = tmp_path / "originals/2024-02-29_09-30-59_JST_replay-123.mp4"
    source.parent.mkdir()
    source.write_bytes(b"original")
    storage = tmp_path / "library/replay-123"
    storage.mkdir(parents=True)
    edit = edit_root / "edit-config.json"
    edit.write_text(
        json.dumps(
            {
                "version": 1,
                "replay_id": "123",
                "source": str(source),
                "source_size": 8,
                "engine": str(tmp_path),
                "executor_scope": str(tmp_path),
                "storage_root": str(storage),
                "stages": dict.fromkeys(
                    (
                        "bgm_removal",
                        "silence_cut",
                        "source_caption_mask",
                        "transcription",
                        "dynamic_captions",
                    ),
                    True,
                ),
            }
        ),
        encoding="utf-8",
    )
    cuts = work / "cuts.json"
    cuts.write_text(
        '{"clock":"source-presentation","fps":30,"cuts":['
        '{"source_start_frame":100,"source_end_frame":103,"output_start_frame":0},'
        '{"source_start_frame":300,"source_end_frame":303,"output_start_frame":3}]}',
        encoding="utf-8",
    )
    xml = work / "semantic.fcpxml"
    xml.write_text(
        '<fcpxml><resources><format id="r1" width="64" height="64" '
        'frameDuration="1/30s"/></resources><library><event><project>'
        '<sequence format="r1" duration="1/5s"><spine>'
        '<asset-clip offset="0s" start="10/3s" duration="1/10s">'
        '<title offset="10/3s" duration="1/10s"><text>First</text></title></asset-clip>'
        '<asset-clip offset="1/10s" start="10s" duration="1/10s">'
        '<title offset="10s" duration="1/10s"><text>Second</text></title></asset-clip>'
        "</spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    path = work / "expectation.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "delivery-clock-expectation",
                "editConfig": {
                    "path": str(edit),
                    "sha256": verification._fingerprint(edit)["sha256"],
                },
                "cuts": {"path": str(cuts), "sha256": verification._fingerprint(cuts)["sha256"]},
                "semanticXML": {
                    "path": str(xml),
                    "sha256": verification._fingerprint(xml)["sha256"],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_clock_contract_has_no_legacy_config_and_preserves_raw_audio_clock(
    clock_contract: Path, fake_tools: tuple[Path, Path]
) -> None:
    movie = clock_contract.parents[2] / "library/replay-123/movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = clock_contract.parent / "verified"
    assert not (clock_contract.parent / "render-config.json").exists()
    assert verification.verify(
        clock_contract,
        movie,
        evidence,
        ffmpeg=fake_tools[0],
        ffprobe=fake_tools[1],
        clock_contract=True,
    )
    receipt = record(json.loads((evidence / "receipt.json").read_text()))
    assert receipt["expected"] == {"frames": 6, "rate": 30, "width": 64, "height": 64}
    assert receipt["expectationKind"] == "pinned-edited-clock-contract"
    inputs = receipt["inputs"]
    assert isinstance(inputs, list)
    assert len(inputs) == 5
    assert receipt["inputsAfter"] == receipt["inputs"]
    assert receipt["audioPacketClock"] == {
        "count": 2,
        "start": "-8/375",
        "end": "1/5",
        "span": "83/375",
        "durationTicksSum": 10624,
    }
    assert receipt["audioEndMinusVideoEnd"] == "0"
    assert receipt["approvals"] == dict(
        captions=False,
        sourceCaptions=False,
        musicRemoval=False,
        transcription=False,
        FCP=False,
        quality=False,
        sync=False,
    )
    assert not verification.verify(
        clock_contract,
        movie,
        clock_contract.parent / "legacy-default",
        ffmpeg=fake_tools[0],
        ffprobe=fake_tools[1],
    )


@pytest.mark.parametrize("field", ["editConfig", "cuts", "semanticXML", "contract"])
def test_clock_contract_closure_mutation_is_retained(
    clock_contract: Path, fake_tools: tuple[Path, Path], field: str
) -> None:
    raw = record(json.loads(clock_contract.read_text(encoding="utf-8")))
    target = clock_contract if field == "contract" else Path(str(record(raw[field])["path"]))
    mutate = clock_contract.parent / "mutate"
    mutate.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\np=Path({str(target)!r})\n"
        "p.write_bytes(p.read_bytes()+b' ')\n",
        encoding="utf-8",
    )
    mutate.chmod(0o700)
    movie = clock_contract.parents[2] / "library/replay-123/movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = clock_contract.parent / "mutated"
    assert not verification.verify(
        clock_contract,
        movie,
        evidence,
        ffmpeg=mutate,
        ffprobe=fake_tools[1],
        clock_contract=True,
    )
    receipt = record(json.loads((evidence / "receipt.json").read_text()))
    assert receipt["bindingError"] == "Bound input changed during verification"
    assert receipt["technicalAccepted"] is False
    with pytest.raises(FileExistsError):
        verification.verify(
            clock_contract,
            movie,
            evidence,
            ffmpeg=mutate,
            ffprobe=fake_tools[1],
            clock_contract=True,
        )


@pytest.mark.parametrize("change", ["source-clock", "output-offset", "duration", "hash"])
def test_clock_contract_rejects_double_projection_and_stale_inputs(
    clock_contract: Path, fake_tools: tuple[Path, Path], change: str
) -> None:
    cuts = clock_contract.parent / "cuts.json"
    xml = clock_contract.parent / "semantic.fcpxml"
    if change == "source-clock":
        cuts.write_text(
            cuts.read_text(encoding="utf-8").replace("source-presentation", "edited-output"),
            encoding="utf-8",
        )
    elif change == "output-offset":
        cuts.write_text(
            cuts.read_text(encoding="utf-8").replace(
                '"output_start_frame":3', '"output_start_frame":0'
            ),
            encoding="utf-8",
        )
    elif change == "duration":
        xml.write_text(
            xml.read_text(encoding="utf-8").replace('duration="1/5s"', 'duration="1/4s"'),
            encoding="utf-8",
        )
    else:
        xml.write_bytes(xml.read_bytes() + b" ")
    if change != "hash":
        raw = record(json.loads(clock_contract.read_text(encoding="utf-8")))
        raw["cuts"] = {"path": str(cuts), "sha256": verification._fingerprint(cuts)["sha256"]}
        raw["semanticXML"] = {"path": str(xml), "sha256": verification._fingerprint(xml)["sha256"]}
        clock_contract.write_text(json.dumps(raw), encoding="utf-8")
    movie = clock_contract.parents[2] / "library/replay-123/movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = clock_contract.parent / "invalid"
    assert not verification.verify(
        clock_contract,
        movie,
        evidence,
        ffmpeg=fake_tools[0],
        ffprobe=fake_tools[1],
        clock_contract=True,
    )
    receipt = record(json.loads((evidence / "receipt.json").read_text()))
    assert receipt["technicalAccepted"] is False
    assert receipt["fullDecode"] is False
    assert not (evidence / "decode.stdout").exists()


def test_clock_contract_explicit_cli_mode(
    clock_contract: Path, fake_tools: tuple[Path, Path]
) -> None:
    movie = clock_contract.parents[2] / "library/replay-123/movie.mp4"
    movie.write_bytes(b"synthetic")
    assert (
        verification.main(
            [
                str(clock_contract),
                str(movie),
                str(clock_contract.parent / "cli"),
                "--ffmpeg",
                str(fake_tools[0]),
                "--ffprobe",
                str(fake_tools[1]),
                "--clock-contract",
            ]
        )
        == 0
    )


def test_clock_contract_uses_distinct_geometry_and_three_retained_spans(
    clock_contract: Path, fake_tools: tuple[Path, Path]
) -> None:
    cuts = clock_contract.parent / "cuts.json"
    xml = clock_contract.parent / "semantic.fcpxml"
    cuts.write_text(
        '{"clock":"source-presentation","fps":25,"cuts":['
        '{"source_start_frame":50,"source_end_frame":51,"output_start_frame":0},'
        '{"source_start_frame":100,"source_end_frame":102,"output_start_frame":1},'
        '{"source_start_frame":150,"source_end_frame":152,"output_start_frame":3}]}',
        encoding="utf-8",
    )
    xml.write_text(
        '<fcpxml><resources><format id="r1" width="96" height="128" '
        'frameDuration="1/25s"/></resources><library><event><project>'
        '<sequence format="r1" duration="1/5s"><spine>'
        '<asset-clip offset="0s" start="2s" duration="1/25s">'
        '<title offset="2s" duration="1/25s"><text>First</text></title></asset-clip>'
        '<asset-clip offset="1/25s" start="4s" duration="2/25s">'
        '<title offset="4s" duration="2/25s"><text>Second</text></title></asset-clip>'
        '<asset-clip offset="3/25s" start="6s" duration="2/25s">'
        '<title offset="6s" duration="2/25s"><text>Third</text></title></asset-clip>'
        "</spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    raw = record(json.loads(clock_contract.read_text(encoding="utf-8")))
    raw["cuts"] = {"path": str(cuts), "sha256": verification._fingerprint(cuts)["sha256"]}
    raw["semanticXML"] = {"path": str(xml), "sha256": verification._fingerprint(xml)["sha256"]}
    clock_contract.write_text(json.dumps(raw), encoding="utf-8")
    tool = fake_tools[0]
    tool.write_text(
        tool.read_text(encoding="utf-8")
        .replace("'width':64,'height':64", "'width':96,'height':128")
        .replace("30/1", "25/1")
        .replace("1/30", "1/25")
        .replace("range(6)", "range(5)")
        .replace("nb_read_packets='6'", "nb_read_packets='5'"),
        encoding="utf-8",
    )
    movie = clock_contract.parents[2] / "library/replay-123/movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = clock_contract.parent / "three-spans"
    assert verification.verify(
        clock_contract,
        movie,
        evidence,
        ffmpeg=tool,
        ffprobe=tool,
        clock_contract=True,
    )
    receipt = record(json.loads((evidence / "receipt.json").read_text(encoding="utf-8")))
    assert receipt["expected"] == {"frames": 5, "rate": 25, "width": 96, "height": 128}
    assert receipt["videoClock"] == {"frames": 5, "start": "0", "end": "1/5", "exactCFR": True}


@pytest.mark.parametrize("change", ["contract", "semanticXML"])
def test_clock_contract_rejects_mutation_after_loader_before_decode(
    clock_contract: Path,
    fake_tools: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    original = clock.load_clock_contract

    def mutated_load(path: Path) -> clock.ClockExpectation:
        expectation = original(path)
        target = path if change == "contract" else path.parent / "semantic.fcpxml"
        target.write_bytes(target.read_bytes() + b" ")
        return expectation

    monkeypatch.setattr(verification, "load_clock_contract", mutated_load)
    movie = clock_contract.parents[2] / "library/replay-123/movie.mp4"
    movie.write_bytes(b"synthetic")
    evidence = clock_contract.parent / "loader-race"
    assert not verification.verify(
        clock_contract,
        movie,
        evidence,
        ffmpeg=fake_tools[0],
        ffprobe=fake_tools[1],
        clock_contract=True,
    )
    receipt = record(json.loads((evidence / "receipt.json").read_text(encoding="utf-8")))
    assert receipt["technicalAccepted"] is False
    assert receipt["fullDecode"] is False
    assert not (evidence / "decode.stdout").exists()
