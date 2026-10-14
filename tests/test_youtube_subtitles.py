import hashlib
import importlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from replay import youtube_subtitle_finalize as finalizer
from replay import youtube_subtitle_repair as repair
from replay import youtube_subtitles as yt
from replay.core import ReplayError
from replay.speaker_subtitles import SpeakerCue


@pytest.fixture
def config(tmp_path: Path) -> Path:
    source = tmp_path / "source.mkv"
    source.touch()
    publication = tmp_path / "publication.json"
    publication.write_text(
        json.dumps({"id": "abcdefghijk", "timestamp": 1700000000}), encoding="utf-8"
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.npz").write_bytes(b"weights")
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "video_id": "abcdefghijk",
                "source": str(source),
                "publication": str(publication),
                "output_root": str(tmp_path / "output"),
                "keep_bgm": True,
                "audio_policy": "original",
                "subtitle_mode": "burn_and_embed",
                "speakers": 1,
                "model": str(model),
                "weights_sha256": hashlib.sha256(b"weights").hexdigest(),
                "ffmpeg": "ffmpeg",
                "ffprobe": "ffprobe",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_config_keeps_bgm_and_burns_subtitles(config: Path) -> None:
    job = yt.load_job(config)
    assert job.keep_bgm is True
    assert job.audio_policy == "original"
    assert job.subtitle_mode == "burn_and_embed"
    assert job.speakers == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("audio_policy", "existing_reduced"),
        ("subtitle_mode", "sidecar_only"),
        ("speakers", 2),
        ("keep_bgm", "true"),
        ("video_id", "wrong"),
        ("weights_sha256", "bad"),
        ("model", ""),
    ],
)
def test_bad_config_is_not_silently_defaulted(config: Path, field: str, value: object) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    data[field] = value
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        yt.load_job(config)


def test_existing_reduced_input_does_not_invent_separation(config: Path) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    data.update(keep_bgm=False, audio_policy="existing_reduced", speakers=3)
    config.write_text(json.dumps(data), encoding="utf-8")
    assert yt.load_job(config).audio_policy == "existing_reduced"


def test_nonfile_input_is_rejected(config: Path) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    data["source"] = str(config.parent)
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="existing source"):
        yt.load_job(config)


def test_native_segment_is_not_interpolated() -> None:
    assert yt.segment_cues(
        {
            "language": "ja",
            "segments": [
                {"start": 1.2, "end": 3.4, "text": " 発話 "},
                {"start": 3.4, "end": 3.4, "text": ""},
            ],
        },
        offset=60,
        duration=5,
    ) == (SpeakerCue(61.2, 63.4, "unassigned", "発話"),)


@pytest.mark.parametrize(
    "value",
    [
        {"language": "en", "segments": []},
        {"language": "ja", "segments": None},
        {"language": "ja", "segments": [{"start": 1, "end": 1, "text": "語"}]},
        {"language": "ja", "segments": [{"start": 1, "end": 1.001, "text": "語"}]},
        {"language": "ja", "segments": [{"start": True, "end": 2, "text": "語"}]},
        {"language": "ja", "segments": [{"start": 1, "end": 2, "text": 1}]},
        {"language": "ja", "segments": [{"start": 1, "end": 6, "text": "語"}]},
        {
            "language": "ja",
            "segments": [
                {"start": 2, "end": 3, "text": "語"},
                {"start": 1, "end": 2, "text": "次"},
            ],
        },
    ],
)
def test_bad_native_clocks_are_recovered_not_stretched(value: object) -> None:
    with pytest.raises(ReplayError):
        yt.segment_cues(value, offset=0, duration=5)


def test_invalid_audio_extent_rejected() -> None:
    with pytest.raises(ReplayError, match="extent"):
        yt.segment_cues({"language": "ja", "segments": []}, offset=-1, duration=5)


def _recognize(_path: str, **_kwargs: object) -> object:
    return {"language": "ja", "segments": [{"start": 1, "end": 3, "text": "自動字幕"}]}


def _fake_run(command: list[str], *, cwd: Path | None = None) -> str:
    if command[0] == "ffprobe":
        return '{"format":{"duration":"26"}}'
    if "-vf" in command:
        Path(command[-1]).touch()
    return ""


def test_pipeline_burns_embeds_preserves_audio_and_never_overwrites(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(transcribe=_recognize)
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(yt, "_run", _fake_run)
    first = yt.run_job(yt.load_job(config))
    second = yt.run_job(yt.load_job(config))
    assert first != second
    assert (first / f"{first.name}.mp4").is_file()
    assert "自動字幕" in (first / f"{first.name}.srt").read_text(encoding="utf-8")
    result = json.loads((first / "metadata/result.json").read_text(encoding="utf-8"))
    assert result["keep_bgm"] is True
    assert result["subtitle_mode"] == "burn_and_embed"
    assert result["full_decode"] == "passed"
    assert result["video"] == "H264 burn-in from ASS; no embedded subtitle track"
    assert (first / f"{first.name}.ass").is_file()
    assert not (first / "metadata/embedded-display-states.srt").exists()


def test_renderer_burns_ass_once_without_an_embedded_track(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def capture(command: list[str], *, cwd: Path | None = None) -> str:
        commands.append(command)
        return _fake_run(command, cwd=cwd)

    monkeypatch.setattr(yt, "_run", capture)
    output = config.parent / "rendered"
    (output / "metadata").mkdir(parents=True)
    yt.render_subtitles(
        yt.load_job(config),
        output=output,
        cues=(
            SpeakerCue(1, 3, "right", "右"),
            SpeakerCue(1, 2, "left", "左"),
        ),
    )
    assert (output / "rendered.srt").read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:02,000\n[左] 左\n\n2\n00:00:01,000 --> 00:00:03,000\n[右] 右\n"
    )
    assert "[左]" not in (output / "rendered.ass").read_text(encoding="utf-8")
    assert commands[0][commands[0].index("-vf") + 1] == "ass=rendered.ass"
    assert "-sn" in commands[0]
    assert "mov_text" not in commands[0]
    assert commands[0].count("-i") == 1


def test_edited_video_cannot_be_a_subtitle_source(config: Path) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    edited = config.parent / "youtube-edits" / "x-edit-abc.mp4"
    edited.parent.mkdir()
    edited.write_bytes(b"video")
    config.write_text(json.dumps({**data, "source": str(edited)}), encoding="utf-8")
    with pytest.raises(ReplayError, match="must not be an edited video"):
        yt.load_job(config)


def test_changed_model_stops_before_output(config: Path) -> None:
    job = yt.load_job(config)
    (job.model / "weights.npz").write_bytes(b"changed")
    with pytest.raises(ReplayError, match="checksum changed"):
        yt.run_job(job)
    assert not job.output_root.exists()


def test_bad_duration_stops(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yt, "_run", lambda _command: '{"format":{"duration":"nan"}}')
    with pytest.raises(ReplayError, match="duration"):
        yt.run_job(yt.load_job(config))


def test_missing_callable_stops(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(transcribe=None)
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(yt, "_run", _fake_run)
    with pytest.raises(ReplayError, match="not callable"):
        yt.run_job(yt.load_job(config))


def test_empty_recognition_is_not_silence(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: {"language": "ja", "segments": []}
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(yt, "_run", _fake_run)
    with pytest.raises(ReplayError, match="No native captions"):
        yt.run_job(yt.load_job(config))


def test_main_uses_config(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["youtube_subtitles", str(config)])
    monkeypatch.setattr(yt, "run_job", lambda _job, **_kwargs: config.parent)
    yt.main()


@pytest.fixture
def repair_config(config: Path) -> Path:
    job = yt.load_job(config)
    source_job = job.output_root / "source-job"
    (source_job / "metadata").mkdir(parents=True)
    (source_job / "metadata/job.json").write_text(json.dumps(yt._job_data(job)), encoding="utf-8")
    evidence = source_job / "metadata/audit.json"
    evidence.write_text("{}", encoding="utf-8")
    path = config.parent / "repair.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_config": str(config),
                "source_job": str(source_job),
                "output_name": "native-repairs-v1",
                "evidence": str(evidence),
                "windows": [
                    {
                        "start": 1,
                        "end": 5,
                        "target_start": 2,
                        "target_end": 4,
                        "reason": "independent speech coverage hypothesis",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _repair_run(command: list[str], *, cwd: Path | None = None) -> str:
    if command[0] == "ffprobe":
        return _fake_run(command, cwd=cwd)
    with wave.open(command[-1], "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(2 * 16000 * 4))
    return ""


def test_config_defined_repair_reuses_actual_recognition(
    repair_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib, "import_module", lambda _name: SimpleNamespace(transcribe=_recognize)
    )
    monkeypatch.setattr(repair, "_run", _repair_run)
    monkeypatch.setattr(repair, "_duration", lambda _job: 26.0)
    output = repair.run_repairs(repair_config)
    record = json.loads((output / "clip-00000.json").read_text(encoding="utf-8"))
    assert record["source_offset"] == 1
    assert record["measured_duration"] == 4
    assert record["all_segments_renderable"] is True
    assert record["status"] == "native-ASR-hypothesis-awaiting-reconciliation"
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: pytest.fail("cached inference repeated"),
        ),
    )
    assert repair.run_repairs(repair_config) == output


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("unknown", True),
        ("output_name", "../outside"),
        ("windows", []),
        ("windows", [{"start": 1}]),
        ("windows", [{"start": -1, "end": 5, "target_start": 2, "target_end": 4, "reason": "bad"}]),
    ],
)
def test_repair_boundary_validation(
    repair_config: Path,
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repair, "_duration", lambda _job: 26.0)
    data = json.loads(repair_config.read_text(encoding="utf-8"))
    data[field] = value
    repair_config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        repair.run_repairs(repair_config)


def test_repair_invalid_native_clocks_are_retained_not_rendered(
    repair_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: {
                "language": "ja",
                "segments": [
                    {"start": 1, "end": 9, "text": "範囲外"},
                ],
            },
        ),
    )
    monkeypatch.setattr(repair, "_run", _repair_run)
    monkeypatch.setattr(repair, "_duration", lambda _job: 26.0)
    output = repair.run_repairs(repair_config)
    record = json.loads((output / "clip-00000.json").read_text(encoding="utf-8"))
    assert record["all_segments_renderable"] is False
    assert record["recognition"]["segments"][0]["end"] == 9


def test_repair_changed_plan_refuses_to_overwrite(
    repair_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib, "import_module", lambda _name: SimpleNamespace(transcribe=_recognize)
    )
    monkeypatch.setattr(repair, "_run", _repair_run)
    monkeypatch.setattr(repair, "_duration", lambda _job: 26.0)
    repair.run_repairs(repair_config)
    data = json.loads(repair_config.read_text(encoding="utf-8"))
    data["windows"][0]["reason"] = "different hypothesis"
    repair_config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="plan changed"):
        repair.run_repairs(repair_config)


def test_repair_source_binding_is_required(repair_config: Path) -> None:
    data = json.loads(repair_config.read_text(encoding="utf-8"))
    (Path(data["source_job"]) / "metadata/job.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="configuration mismatch"):
        repair.run_repairs(repair_config)


def test_repair_evidence_cannot_escape_job(repair_config: Path) -> None:
    data = json.loads(repair_config.read_text(encoding="utf-8"))
    data["evidence"] = str(repair_config)
    repair_config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="evidence outside"):
        repair.run_repairs(repair_config)


def test_repair_changed_model_refused(
    repair_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = json.loads(repair_config.read_text(encoding="utf-8"))
    job = yt.load_job(Path(data["source_config"]))
    (job.model / "weights.npz").write_bytes(b"changed")
    monkeypatch.setattr(repair, "_duration", lambda _job: 26.0)
    with pytest.raises(ReplayError, match="checksum changed"):
        repair.run_repairs(repair_config)


def test_repair_not_callable_refused(
    repair_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib, "import_module", lambda _name: SimpleNamespace(transcribe=None))
    monkeypatch.setattr(repair, "_duration", lambda _job: 26.0)
    with pytest.raises(ReplayError, match="not callable"):
        repair.run_repairs(repair_config)


def test_repair_main(repair_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["repair", str(repair_config)])
    monkeypatch.setattr(repair, "run_repairs", lambda _path: repair_config.parent)
    repair.main()


@pytest.fixture
def selection_config(repair_config: Path) -> Path:
    plan = json.loads(repair_config.read_text(encoding="utf-8"))
    source_job = Path(plan["source_job"])
    metadata = source_job / "metadata"
    recognition = metadata / "selected.json"
    recognition.write_text(
        json.dumps(
            {
                "source_offset": 1,
                "recognition": _recognize("fixture"),
            }
        ),
        encoding="utf-8",
    )
    with wave.open(str(recognition.with_suffix(".wav")), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(2 * 16000 * 4))
    path = repair_config.parent / "selection.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_config": plan["source_config"],
                "source_job": plan["source_job"],
                "review": plan["evidence"],
                "selection": [{"record": "metadata/selected.json", "segment": 0}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_finalizer_renders_exact_native_provenance(
    selection_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalizer, "_duration", lambda _job: 26.0)
    monkeypatch.setattr(yt, "_run", _fake_run)
    output = finalizer.finalize(selection_config)
    assert "00:00:02,000 --> 00:00:04,000" in (output / f"{output.name}.srt").read_text(
        encoding="utf-8"
    )
    assert (output / "metadata/native-selection.json").is_file()
    assert (output / f"{output.name}.mp4").is_file()
    assert finalizer.finalize(selection_config) != output


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("unknown", True),
        ("selection", []),
        ("selection", [{"record": "metadata/selected.json", "segment": True}]),
        ("selection", [{"record": "metadata/selected.json", "segment": 10}]),
        ("selection", [{"record": "metadata/selected.json", "segment": 0, "text": "捏造"}]),
        (
            "selection",
            [
                {"record": "metadata/selected.json", "segment": 0},
                {"record": "metadata/selected.json", "segment": 0},
            ],
        ),
    ],
)
def test_finalizer_rejects_invented_or_bad_selection(
    selection_config: Path,
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalizer, "_duration", lambda _job: 26.0)
    data = json.loads(selection_config.read_text(encoding="utf-8"))
    data[field] = value
    selection_config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        finalizer.finalize(selection_config)


def test_finalizer_rejects_outside_review(selection_config: Path) -> None:
    data = json.loads(selection_config.read_text(encoding="utf-8"))
    data["review"] = str(selection_config)
    selection_config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="review outside"):
        finalizer.finalize(selection_config)


def test_finalizer_refuses_unrenderable_native_clock(
    selection_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalizer, "_duration", lambda _job: 26.0)
    data = json.loads(selection_config.read_text(encoding="utf-8"))
    recognition = Path(data["source_job"]) / "metadata/selected.json"
    recognition.write_text(
        json.dumps(
            {
                "source_offset": 1,
                "recognition": {
                    "language": "ja",
                    "segments": [{"start": 1, "end": 9, "text": "範囲外"}],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(yt.UnrenderableSegment):
        finalizer.finalize(selection_config)


def test_finalizer_recognition_cannot_escape_source_job(
    selection_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalizer, "_duration", lambda _job: 26.0)
    data = json.loads(selection_config.read_text(encoding="utf-8"))
    data["selection"][0]["record"] = str(selection_config)
    selection_config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="reference outside"):
        finalizer.finalize(selection_config)


def test_finalizer_pcm_extent_is_source_bound(
    selection_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalizer, "_duration", lambda _job: 26.0)
    data = json.loads(selection_config.read_text(encoding="utf-8"))
    recognition = Path(data["source_job"]) / "metadata/selected.json"
    measured = json.loads(recognition.read_text(encoding="utf-8"))
    measured["source_offset"] = 25
    recognition.write_text(json.dumps(measured), encoding="utf-8")
    with pytest.raises(ReplayError, match="audio outside"):
        finalizer.finalize(selection_config)


def test_finalizer_requires_same_input_source(
    selection_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(finalizer, "_duration", lambda _job: 26.0)
    data = json.loads(selection_config.read_text(encoding="utf-8"))
    recognition = Path(data["source_job"]) / "metadata/selected.json"
    measured = json.loads(recognition.read_text(encoding="utf-8"))
    measured["source"] = "another-input"
    recognition.write_text(json.dumps(measured), encoding="utf-8")
    with pytest.raises(ReplayError, match="source mismatch"):
        finalizer.finalize(selection_config)


def test_finalizer_model_change_refused(
    selection_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = json.loads(selection_config.read_text(encoding="utf-8"))
    job = yt.load_job(Path(data["source_config"]))
    (job.model / "weights.npz").write_bytes(b"changed")
    monkeypatch.setattr(finalizer, "_duration", lambda _job: 26.0)
    with pytest.raises(ReplayError, match="provenance changed"):
        finalizer.finalize(selection_config)


def test_empty_shared_renderer_refused(config: Path) -> None:
    with pytest.raises(ReplayError, match="empty subtitle"):
        yt.render_subtitles(yt.load_job(config), output=config.parent, cues=())


def test_finalizer_main(selection_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["finalize", str(selection_config)])
    monkeypatch.setattr(finalizer, "finalize", lambda _path: selection_config.parent)
    finalizer.main()


def test_subprocess_wrapper() -> None:
    assert yt._run(["/usr/bin/printf", "OK"]) == "OK"


def test_unowned_context_tail_is_not_a_reason_to_drop_owned_speech() -> None:
    value = {
        "language": "ja",
        "segments": [
            {"start": 3, "end": 5, "text": "残す発話"},
            {"start": 29.7, "end": 59.68, "text": "時間外の幻覚"},
        ],
    }
    assert yt.owned_segment_cues(value, offset=93, duration=30, core=96) == (
        SpeakerCue(96, 98, "unassigned", "残す発話"),
    )


def test_owned_invalid_clock_still_fails() -> None:
    with pytest.raises(ReplayError, match="Unrenderable"):
        yt.owned_segment_cues(
            {
                "language": "ja",
                "segments": [
                    {"start": 5, "end": 59, "text": "不正"},
                ],
            },
            offset=93,
            duration=30,
            core=96,
        )


def _fake_long_run(command: list[str], *, cwd: Path | None = None) -> str:
    if command[0] == "ffprobe":
        return '{"format":{"duration":"60"}}'
    return _fake_run(command, cwd=cwd)


@pytest.mark.parametrize("output_parent", ["output", "recovery-output"])
def test_new_context_recovers_without_extending_old_clocks(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_parent: str,
) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    data["output_root"] = str(config.parent / output_parent)
    config.write_text(json.dumps(data), encoding="utf-8")
    calls: list[str] = []

    def recognize(path: str, **_kwargs: object) -> object:
        calls.append(path)
        if Path(path).name.startswith("recovery-"):
            return _recognize(path)
        return {
            "language": "ja",
            "segments": [
                {"start": 5, "end": 31, "text": "未確定の切れた発話"},
            ],
        }

    module = SimpleNamespace(transcribe=recognize)
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(yt, "_run", _fake_long_run)
    output = yt.run_job(yt.load_job(config))
    assert any(Path(path).name.startswith("recovery-") for path in calls)
    assert (output / "metadata/chunk-00000.json").is_file()
    assert (output / "metadata/recovery-00000-extended36.json").is_file()
    srt = (output / f"{output.name}.srt").read_text(encoding="utf-8")
    assert "自動字幕" in srt
    assert "未確定の切れた発話" not in srt


def test_second_context_is_new_audio_not_a_relaxed_old_clock(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def recognize(path: str, **_kwargs: object) -> object:
        if "chunk-00000" in path:
            return {"language": "ja", "segments": [{"start": 5, "end": 31, "text": "初回"}]}
        if "recovery-00000" in path:
            return {"language": "ja", "segments": [{"start": 5, "end": 37, "text": "再認識"}]}
        return _recognize(path)

    module = SimpleNamespace(transcribe=recognize)
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(yt, "_run", _fake_long_run)
    output = yt.run_job(yt.load_job(config))
    assert (output / "metadata/recovery-00000-extended60.json").is_file()
    assert "再認識" in (output / f"{output.name}.srt").read_text(encoding="utf-8")


def test_identical_clipped_context_is_not_retried(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def recognize(path: str, **_kwargs: object) -> object:
        calls.append(path)
        return {"language": "ja", "segments": [{"start": 5, "end": 31, "text": "不正"}]}

    module = SimpleNamespace(transcribe=recognize)
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(yt, "_run", _fake_run)
    with pytest.raises(yt.UnrenderableSegment):
        yt.run_job(yt.load_job(config))
    assert len(calls) == 1


def test_missing_segments_cannot_bypass_validation() -> None:
    with pytest.raises(ReplayError, match="Missing"):
        yt.owned_segment_cues({}, offset=0, duration=30, core=0)


def test_resume_uses_completed_chunk_without_repeating_inference(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def interrupted(path: str, **_kwargs: object) -> object:
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("simulated interrupted inference")
        return _recognize(path)

    module = SimpleNamespace(transcribe=interrupted)
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(yt, "_run", _fake_run)
    job = yt.load_job(config)
    with pytest.raises(RuntimeError, match="interrupted"):
        yt.run_job(job)
    output = next(job.output_root.iterdir())
    module.transcribe = _recognize
    assert yt.run_job(job, resume=output) == output
    assert len(calls) == 2
    with pytest.raises(ReplayError, match="already completed"):
        yt.run_job(job, resume=output)
