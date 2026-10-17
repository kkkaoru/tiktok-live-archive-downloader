"""Synthetic fixtures only; no private recordings are read."""

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import caption_ass_completion
from replay import youtube_edit_completion as completion
from replay.core import ReplayError
from replay.youtube_delivery import source_stem

VIDEO_ID = "TEST1234567"
STAMP = 1791811754


@dataclass(frozen=True)
class Recording:
    """Paths of the synthetic recording fixture."""

    root: Path
    work: Path
    movie: Path
    cut_program: Path
    source: Path
    tools: dict[str, Path]


def _rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return [item for item in value if isinstance(item, dict)]


def _missing(report: dict[str, object], index: int) -> dict[str, object]:
    return _row(_rows(report["deliveries"])[index]["missingStages"])


def _row(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def _box(kind: bytes, size: int = 16) -> bytes:
    return struct.pack(">I4s", size, kind) + b"\x00" * (size - 8)


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    bindir = tmp_path / "bin"
    probe_dir = tmp_path / "probes"
    levels_dir = tmp_path / "levels"
    bindir.mkdir()
    probe_dir.mkdir()
    levels_dir.mkdir()
    (bindir / "ffprobe").write_text(
        '#!/bin/sh\nlast=""; for a in "$@"; do last="$a"; done\n'
        'cat "$STUB_PROBE_DIR/$(basename "$last").json"\n',
        encoding="utf-8",
    )
    (bindir / "ffmpeg").write_text(
        '#!/bin/sh\ninput=""; prev=""\nfor a in "$@"; do\n'
        '  if [ "$prev" = "-i" ]; then input="$a"; fi\n  prev="$a"\ndone\n'
        'case "$*" in\n*volumedetect*)\n'
        '  read MEAN MAX < "$STUB_LEVELS_DIR/$(basename "$input").levels"\n'
        '  echo "[Parsed_volumedetect_0] n: 100"\n'
        '  echo "[Parsed_volumedetect_0] mean_volume: $MEAN dB"\n'
        '  echo "[Parsed_volumedetect_0] max_volume: $MAX dB"\n  ;;\n'
        "*format=gray*)\n  head -c 2073600 /dev/zero\n  ;;\n"
        "*rawvideo*)\n  head -c 6220800 /dev/zero\n  ;;\nesac\n"
        "exit ${STUB_FFMPEG_FAIL:-0}\n",
        encoding="utf-8",
    )
    for tool in ("ffprobe", "ffmpeg"):
        (bindir / tool).chmod(0o700)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("STUB_PROBE_DIR", str(probe_dir))
    monkeypatch.setenv("STUB_LEVELS_DIR", str(levels_dir))
    return {"bin": bindir, "probes": probe_dir, "levels": levels_dir}


def _probe(probes: Path, name: str, payload: object) -> None:
    (probes / (name + ".json")).write_text(json.dumps(payload), encoding="utf-8")


def _levels(levels: Path, name: str, mean: str, maximum: str) -> None:
    (levels / (name + ".levels")).write_text(f"{mean} {maximum}\n", encoding="utf-8")


def _media_probe(duration: str) -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "nb_frames": "3000",
                "pix_fmt": "yuv420p",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "profile": "LC",
                "sample_rate": "44100",
                "channels": 2,
                "bit_rate": "320000",
            },
        ],
        "format": {"duration": duration, "size": "100"},
    }


def _alpha_probe(frames: int) -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "prores",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "nb_frames": str(frames),
                "pix_fmt": "yuva444p12le",
            }
        ],
        "format": {"duration": str(frames / 30), "size": "100"},
    }


@pytest.fixture
def recording(tmp_path: Path, tools: dict[str, Path]) -> Recording:
    root = tmp_path / f"youtube-{VIDEO_ID}"
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    publication = media_dir / "publication.json"
    _write(publication, {"id": VIDEO_ID, "release_timestamp": STAMP, "is_live": False})
    stem = source_stem(VIDEO_ID, json.loads(publication.read_text(encoding="utf-8")))
    source = media_dir / f"{stem}.mp4"
    source.write_bytes(b"fake-source-bytes")
    engine = tmp_path / "engine"
    engine.mkdir()
    config = {
        "version": 2,
        "video_id": VIDEO_ID,
        "source": str(source),
        "source_size": source.stat().st_size,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "publication": str(publication),
        "publication_sha256": hashlib.sha256(publication.read_bytes()).hexdigest(),
        "engine": str(engine),
        "executor_scope": str(tmp_path),
        "stages": {
            "bgm_removal": True,
            "silence_cut": True,
            "source_caption_mask": True,
            "transcription": True,
            "dynamic_captions": True,
        },
    }
    root.mkdir()
    _write(root / "edit-config.json", config)
    delivery_dir = root / "delivery" / "2026-01-01_00-00-00_JST_youtube-TEST1234567-edit-x"
    delivery_dir.mkdir(parents=True)
    movie = delivery_dir / "a.mp4"
    movie.write_bytes(_box(b"ftyp") + _box(b"moov") + b"fake-movie-bytes")
    (delivery_dir / "a.srt").write_text(
        "1\n00:00:00,500 --> 00:00:01,000\nあ\n\n2\n00:00:01,500 --> 00:00:02,000\nい\n",
        encoding="utf-8",
    )
    (delivery_dir / "a.srt.summaries").mkdir()
    old_dir = root / "delivery" / "2025-01-01_00-00-00_JST_youtube-TEST1234567-edit-old"
    old_dir.mkdir(parents=True)
    (old_dir / "old.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nold\n", encoding="utf-8")
    work = tmp_path / "work"
    (work / "separation").mkdir(parents=True)
    vocals = work / "vocals.wav"
    vocals.write_bytes(b"fake-vocals")
    _write(work / "separation" / "result.json", {"device": "mps", "vocals": str(vocals)})
    cut = work / "cut-plan" / "p1"
    cut.mkdir(parents=True)
    _write(
        cut / "cuts.json",
        {
            "fps": 30,
            "cuts": [
                {"source_start_frame": 0, "source_end_frame": 3000, "output_start_frame": 0},
            ],
        },
    )
    (cut / "peaks.txt").write_text("peaks", encoding="utf-8")
    labels = [
        {"text": t, "start": float(i), "end": float(i) + 0.5, "provenance": f"w:{i}"}
        for i, t in enumerate(["あ", "い", "う"])
    ]
    _write(
        work / "asr-candidate-filtered.json",
        {
            "version": 1,
            "clock": "input-pcm-not-projected",
            "labels": labels,
            "hallucinationFilter": {"excluded": 1},
        },
    )
    titles = work / "titles"
    titles.mkdir()
    part = work / "part-0000-alpha.mov"
    part.write_bytes(b"fake-part")
    _write(
        work / "composite-full-v1-parts.json",
        {
            "rate": 30,
            "width": 1920,
            "height": 1080,
            "parts": [{"path": str(part), "offsetFrame": 0, "frames": 3000}],
        },
    )
    _write(
        titles / "wrap-evidence-v1.json",
        {
            "failures": 0,
            "widestRewrappedLine": 100.0,
            "safeWidthPx": 1100.0,
            "records": [
                {
                    "part": 0,
                    "startFrame": 10,
                    "lines": [{"line": "あ", "width": 100.0}, {"line": "い", "width": 100.0}],
                }
            ],
        },
    )
    _write(
        titles / "bundle-v1.json",
        {
            "titleParts": [
                {
                    "offsetFrame": 0,
                    "placements": [
                        {"text": "あ", "startFrame": 0, "endFrame": 30},
                        {"text": "い\nろ", "startFrame": 40, "endFrame": 200},
                        {"text": "は\nひ\nふ", "startFrame": 300, "endFrame": 900},
                    ],
                }
            ],
        },
    )
    _write(
        titles / "tail-bundle-v1.json",
        {
            "titleParts": [
                {
                    "offsetFrame": 0,
                    "placements": [
                        {"text": "た\nち", "startFrame": 1000, "endFrame": 2000},
                    ],
                }
            ],
        },
    )
    _write(
        titles / "tail-wrap-evidence-v1.json",
        {"failures": 0, "widestRewrappedLine": 100.0, "safeWidthPx": 1100.0},
    )
    _write(
        titles / "left-edge-v1-settled.json",
        {
            "summary": {"undetected": 0},
            "rows": [{"part": 0, "startFrame": 5, "leftOk": True, "box": [800, 800, 900, 900]}],
        },
    )
    _write(titles / "left-edge-tail-v1.json", {"summary": {"undetected": 0}, "rows": []})
    raw = work / "frames-ocr.json"
    raw.write_text("[]", encoding="utf-8")
    evidence = root / "evidence" / "source-masks-none"
    evidence.mkdir(parents=True)
    _write(
        evidence / "summary.json",
        {
            "reason": "no-repeatable-caption-band-observed",
            "totalFrames": 150,
            "bandBoxes": 0,
            "framesFile": {"path": str(raw), "sha256": hashlib.sha256(b"[]").hexdigest()},
        },
    )
    probes = tools["probes"]
    _probe(probes, source.name, _media_probe("200.0"))
    _probe(probes, vocals.name, {"streams": [], "format": {"duration": "200.0", "size": "50"}})
    _probe(probes, movie.name, _media_probe("100.0"))
    _probe(probes, part.name, _alpha_probe(3000))
    _levels(tools["levels"], movie.name, "-20.0", "-3.0")
    cut_program = work / "cut-program.mov"
    cut_program.write_bytes(b"fake-program")
    _probe(probes, cut_program.name, _media_probe("100.0"))
    _levels(tools["levels"], cut_program.name, "-20.0", "-3.0")
    return Recording(
        root=root, work=work, movie=movie, cut_program=cut_program, source=source, tools=tools
    )


def test_full_check_passes(
    recording: Recording, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(completion, "_diff_band_height", lambda program, movie, time: 300)
    monkeypatch.setattr(completion, "_alpha_box", lambda render, time: (800, 850, 900, 1000))
    monkeypatch.setattr(
        completion, "_diff_band_box", lambda program, movie, time: (800, 850, 900, 1000)
    )
    code = completion.main(
        [
            "check",
            str(recording.root),
            "--workdir",
            str(recording.work),
            "--cut-program",
            str(recording.cut_program),
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["complete"] is True
    assert _missing(report, -1) == {}
    assert report["deliveries"][-1]["complete"] is True
    assert report["deliveries"][0]["superseded"] is True
    assert report["deliveries"][0]["complete"] is False
    assert report["deliveries"][-1]["superseded"] is False
    assert report["requiredStages"] == [
        "bgm_removal",
        "silence_cut",
        "source_caption_mask",
        "transcription",
        "dynamic_captions",
    ]


def test_check_without_cut_program_passes_with_note(
    recording: Recording, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(completion, "_alpha_box", lambda render, time: (800, 850, 900, 1000))
    failures, facts = completion.assess_delivery(
        recording.root,
        completion.find_deliveries(recording.root)[-1],
        workdir=recording.work,
    )
    assert failures == {}
    assert _row(facts["bgmRemoval"])["audioIdentity"] == "unverified-no-reference"


def test_burn_in_paths(recording: Recording, monkeypatch: pytest.MonkeyPatch) -> None:
    movie = recording.movie
    program = recording.cut_program
    real_diff = completion._diff_band_height
    assert real_diff(program, movie, 0.0) == 0
    monkeypatch.setenv("STUB_FFMPEG_FAIL", "1")
    with pytest.raises(ReplayError):
        real_diff(program, movie, 0.0)
    monkeypatch.setenv("STUB_FFMPEG_FAIL", "0")
    failure, _ = completion.check_burn_in(recording.work, "v1", program, movie)
    assert failure is not None and "burn-in" in failure
    monkeypatch.setattr(completion, "_diff_band_height", lambda program, movie, time: 300)
    failure, info = completion.check_burn_in(recording.work, "v1", program, movie)
    assert failure is None and info["bandHeightsPx"] == [300]
    monkeypatch.setattr(completion, "_diff_band_height", lambda program, movie, time: 0)
    failure, _ = completion.check_burn_in(recording.work, "v1", program, movie)
    assert failure is not None and "burn-in" in failure
    failure, info = completion.check_burn_in(recording.work, "v1", None, movie)
    assert failure is None and info["burnIn"] == "unverified-no-reference"


def test_delivery_exception_row(recording: Recording) -> None:
    (recording.work / "separation" / "result.json").unlink()
    (recording.tools["probes"] / (recording.movie.name + ".json")).unlink()
    report = completion.completion_report(recording.root, workdir=recording.work)
    assert report["complete"] is False
    assert "delivery" in _missing(report, 0)
    assert completion.main(["check", str(recording.root), "--workdir", str(recording.work)]) == 2


def test_default_workdir_branch(recording: Recording) -> None:
    report = completion.completion_report(recording.root)
    assert report["complete"] is False
    assert "delivery" in _missing(report, 0)


def test_bgm_failure_paths(recording: Recording) -> None:
    delivery = completion.find_deliveries(recording.root)[-1]
    result = recording.work / "separation" / "result.json"
    data = json.loads(result.read_text(encoding="utf-8"))
    data["device"] = "cpu"
    result.write_text(json.dumps(data), encoding="utf-8")
    failure, _ = completion.check_bgm_removal(recording.source, delivery, recording.work)
    assert failure is not None
    data["device"] = "mps"
    data["vocals"] = str(recording.work / "missing.wav")
    result.write_text(json.dumps(data), encoding="utf-8")
    failure, _ = completion.check_bgm_removal(recording.source, delivery, recording.work)
    assert failure is not None and "vocals" in failure
    (recording.work / "vocals.wav").write_bytes(b"")
    data["vocals"] = str(recording.work / "vocals.wav")
    result.write_text(json.dumps(data), encoding="utf-8")
    failure, _ = completion.check_bgm_removal(recording.source, delivery, recording.work)
    assert failure is not None and "vocals" in failure


def test_bgm_audio_mismatch(recording: Recording, monkeypatch: pytest.MonkeyPatch) -> None:
    delivery = completion.find_deliveries(recording.root)[-1]
    _levels(recording.tools["levels"], recording.cut_program.name, "-30.0", "-3.0")
    failure, _ = completion.check_bgm_removal(
        recording.source, delivery, recording.work, cut_program=recording.cut_program
    )
    assert failure is not None and "differs" in failure
    monkeypatch.setenv("STUB_FFMPEG_FAIL", "1")
    failure, _ = completion.check_bgm_removal(recording.source, delivery, recording.work)
    assert failure is not None and "fully decode" in failure


def test_bgm_more_branches(recording: Recording) -> None:
    delivery = completion.find_deliveries(recording.root)[-1]
    probes = recording.tools["probes"]
    movie_probe = json.loads((probes / (recording.movie.name + ".json")).read_text())
    movie_probe["streams"][1]["codec_name"] = "mp3"
    _probe(probes, recording.movie.name, movie_probe)
    failure, _ = completion.check_bgm_removal(recording.source, delivery, recording.work)
    assert failure == "delivery audio is not AAC"
    movie_probe["streams"][1]["codec_name"] = "aac"
    movie_probe["streams"][1]["profile"] = "HE-AAC"
    _probe(probes, recording.movie.name, movie_probe)
    failure, _ = completion.check_bgm_removal(recording.source, delivery, recording.work)
    assert failure is not None and "LC AAC" in failure
    movie_probe["streams"][1]["profile"] = "LC"
    _probe(probes, recording.movie.name, movie_probe)
    _probe(
        probes,
        (recording.work / "vocals.wav").name,
        {"streams": [], "format": {"duration": "10.0", "size": "1"}},
    )
    failure, _ = completion.check_bgm_removal(recording.source, delivery, recording.work)
    assert failure is not None and "cover" in failure


def test_silence_more_branches(recording: Recording) -> None:
    work = recording.work
    cut = work / "cut-plan" / "p1" / "cuts.json"
    data = json.loads(cut.read_text(encoding="utf-8"))
    data["cuts"] = [{"source_start_frame": "x", "source_end_frame": 1, "output_start_frame": 0}]
    cut.write_text(json.dumps(data), encoding="utf-8")
    failure, _ = completion.check_silence_cut(recording.source, 100.0, work)
    assert failure is not None and "matches" in failure
    (work / "cut-plan" / "p1" / "cuts.json").unlink()
    (work / "cut-plan" / "p1" / "peaks.txt").unlink()
    (work / "cut-plan" / "p1").rmdir()
    (work / "cut-plan").rmdir()
    failure, _ = completion.check_silence_cut(recording.source, 100.0, work)
    assert failure is not None and "no cut plan" in failure


def test_transcription_more_branches(recording: Recording) -> None:
    delivery = completion.find_deliveries(recording.root)[-1]
    work = recording.work
    candidate = work / "asr-candidate-filtered.json"
    data = json.loads(candidate.read_text(encoding="utf-8"))
    data["clock"] = "other"
    candidate.write_text(json.dumps(data), encoding="utf-8")
    failure, _ = completion.check_transcription(delivery, work, 100.0, 200.0)
    assert failure is not None and "missing or empty" in failure
    data["clock"] = "input-pcm-not-projected"
    data["labels"] = data["labels"] * 60
    candidate.write_text(json.dumps(data), encoding="utf-8")
    failure, info = completion.check_transcription(delivery, work, 100.0, 200.0)
    assert failure is not None and "less than half" in failure
    delivery.transcript.write_text("1\n00:00:01,000 --> 00:04:00,000\nlate\n", encoding="utf-8")
    failure, _ = completion.check_transcription(delivery, work, 100.0, 200.0)
    assert failure is not None and "exceeds" in failure
    (work / "cut-plan" / "p1" / "cuts.json").unlink()
    failure, _ = completion.check_transcription(delivery, work, 100.0, 200.0)
    assert failure is not None and "no cut plan" in failure


def test_silence_cut_paths(recording: Recording) -> None:
    failure, info = completion.check_silence_cut(recording.source, 100.0, recording.work)
    assert failure is None and info["cuts"] == 1
    failure, _ = completion.check_silence_cut(recording.source, 50.0, recording.work)
    assert failure is not None and "matches" in failure
    (recording.work / "cut-plan" / "p1" / "peaks.txt").unlink()
    failure, _ = completion.check_silence_cut(recording.source, 100.0, recording.work)
    assert failure is not None and "measurement" in failure


def test_mask_paths(recording: Recording) -> None:
    root = recording.root
    failure, info = completion.check_source_caption_mask(root)
    assert failure is None and info["totalFrames"] == 150
    summary = root / "evidence" / "source-masks-none" / "summary.json"
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["totalFrames"] = 8
    summary.write_text(json.dumps(data), encoding="utf-8")
    failure, _ = completion.check_source_caption_mask(root)
    assert failure is not None and "thinner" in failure
    data["reason"] = "repeated-caption-band-detected"
    summary.write_text(json.dumps(data), encoding="utf-8")
    failure, _ = completion.check_source_caption_mask(root)
    assert failure is not None and "visible" in failure
    summary.unlink()
    failure, _ = completion.check_source_caption_mask(root)
    assert failure is not None and "no source-masks-none" in failure
    summary.write_text(
        json.dumps({"noSourceSubtitlesDetected": True, "observedTextFrames": 0}), encoding="utf-8"
    )
    failure, _ = completion.check_source_caption_mask(root)
    assert failure is None


def test_mask_digest_mismatch(recording: Recording) -> None:
    summary = recording.root / "evidence" / "source-masks-none" / "summary.json"
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["framesFile"] = {"path": str(recording.work / "frames-ocr.json"), "sha256": "0" * 64}
    summary.write_text(json.dumps(data), encoding="utf-8")
    failure, _ = completion.check_source_caption_mask(recording.root)
    assert failure is not None and "digest" in failure


def test_transcription_paths(recording: Recording) -> None:
    delivery = completion.find_deliveries(recording.root)[-1]
    failure, info = completion.check_transcription(delivery, recording.work, 100.0, 200.0)
    assert failure is None and info["srtCues"] == 2
    assert info["mappedLabels"] == 3 and info["unmappedLabels"] == 0
    delivery.transcript.write_text("no cues here\n", encoding="utf-8")
    failure, _ = completion.check_transcription(delivery, recording.work, 100.0, 200.0)
    assert failure is not None and "no transcript" in failure


def test_dynamic_more_branches(recording: Recording, monkeypatch: pytest.MonkeyPatch) -> None:
    delivery = completion.find_deliveries(recording.root)[-1]
    work = recording.work
    probes = recording.tools["probes"]
    manifest = work / "composite-full-v1-parts.json"
    manifest.unlink()
    failure, _ = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "no composite parts manifest" in failure
    _write(manifest, {"rate": 30, "parts": "x"})
    failure, _ = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "matches" in failure
    _write(
        manifest,
        {
            "rate": 30,
            "width": 1920,
            "height": 1080,
            "parts": [{"path": str(work / "missing.mov"), "offsetFrame": 0, "frames": 3000}],
        },
    )
    failure, info = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "missing" in failure and info["missingHead"] != []
    alpha = json.loads((probes / ((work / "part-0000-alpha.mov").name + ".json")).read_text())
    alpha["streams"][0]["pix_fmt"] = "yuv420p"
    _probe(probes, (work / "part-0000-alpha.mov").name, alpha)
    _write(
        manifest,
        {
            "rate": 30,
            "width": 1920,
            "height": 1080,
            "parts": [
                {"path": str(work / "part-0000-alpha.mov"), "offsetFrame": 0, "frames": 3000}
            ],
        },
    )
    failure, _ = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "alpha" in failure
    alpha["streams"][0]["pix_fmt"] = "yuva444p12le"
    alpha["streams"][0]["nb_frames"] = "5"
    _probe(probes, (work / "part-0000-alpha.mov").name, alpha)
    failure, _ = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "frame count" in failure


def test_dynamic_delivery_branches(recording: Recording, monkeypatch: pytest.MonkeyPatch) -> None:
    delivery = completion.find_deliveries(recording.root)[-1]
    work = recording.work
    probes = recording.tools["probes"]
    wrap = work / "titles" / "wrap-evidence-v1.json"
    data = json.loads(wrap.read_text(encoding="utf-8"))
    data["failures"] = 2
    _write(wrap, data)
    failure, _ = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "failures" in failure
    data["failures"] = 0
    data["widestRewrappedLine"] = 9999.0
    _write(wrap, data)
    failure, _ = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "safe width" in failure
    data["widestRewrappedLine"] = 100.0
    _write(wrap, data)
    movie = probes / (recording.movie.name + ".json")
    probe = json.loads(movie.read_text(encoding="utf-8"))
    probe["streams"][0]["codec_name"] = "mpeg4"
    _probe(probes, recording.movie.name, probe)
    failure, _ = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "QuickTime" in failure
    probe["streams"][0]["codec_name"] = "h264"
    probe["streams"][0]["width"] = 1280
    _probe(probes, recording.movie.name, probe)
    failure, _ = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "1920x1080" in failure
    probe["streams"][0]["width"] = 1920
    _probe(probes, recording.movie.name, probe)
    recording.movie.write_bytes(b"no-boxes-here")
    failure, _ = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "moov" in failure
    monkeypatch.setenv("STUB_FFMPEG_FAIL", "1")
    recording.movie.write_bytes(
        b"\x00\x00\x00\x10ftyp\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x10moov\x00\x00\x00\x00\x00\x00\x00\x00"
    )
    failure, _ = completion.check_dynamic_captions(delivery, work, 100.0)
    assert failure is not None and "fully decode" in failure
    monkeypatch.setenv("STUB_FFMPEG_FAIL", "0")


def test_dynamic_caption_paths(recording: Recording, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(completion, "_alpha_box", lambda render, time: (800, 850, 900, 1000))
    delivery = completion.find_deliveries(recording.root)[-1]
    failure, info = completion.check_dynamic_captions(delivery, recording.work, 100.0)
    assert failure is None and info["parts"] == 1
    failure, _ = completion.check_dynamic_captions(delivery, recording.work, 10.0)
    assert failure is not None and "matches" in failure
    manifest = recording.work / "composite-full-v1-parts.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    part_path = str(recording.work / "part-0000-alpha.mov")
    data["parts"] = [
        {"path": part_path, "offsetFrame": 0, "frames": 2000},
        {"path": part_path, "offsetFrame": 1000, "frames": 2000},
    ]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    failure, _ = completion.check_dynamic_captions(delivery, recording.work, 100.0)
    assert failure is not None and "overlap" in failure


def test_mask_and_placement_more_branches(recording: Recording) -> None:
    summary = recording.root / "evidence" / "source-masks-none" / "summary.json"
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["framesFile"] = []
    summary.write_text(json.dumps(data), encoding="utf-8")
    failure, _ = completion.check_source_caption_mask(recording.root)
    assert failure is not None and "bound raw" in failure
    work = recording.work
    rows = [{"part": 9, "startFrame": 0, "leftOk": False, "box": [100, 800, 900, 900]}]
    _write(
        work / "titles" / "left-edge-v1-settled.json", {"summary": {"undetected": 0}, "rows": rows}
    )
    passed, placement = completion.check_placement(work, "v1")
    assert passed is False
    assert _rows(placement["violations"])[0]["part"] == 9
    (work / "titles" / "bundle-v1.json").unlink()
    with pytest.raises(ReplayError):
        completion.check_placement(work, "v1")


def test_assess_volumedetect_raise(recording: Recording) -> None:
    (recording.tools["levels"] / (recording.movie.name + ".levels")).unlink()
    report = completion.completion_report(
        recording.root, workdir=recording.work, cut_program=recording.cut_program
    )
    assert report["complete"] is False
    assert "delivery" in _missing(report, -1)


def test_clip_geometry_paths(recording: Recording, monkeypatch: pytest.MonkeyPatch) -> None:
    work = recording.work
    parts = [{"path": str(work / "part-0000-alpha.mov"), "offsetFrame": 0, "frames": 3000}]
    movie = recording.movie
    program = recording.cut_program
    failure, _ = completion.check_clip_geometry(
        work, "v1", parts, program=None, delivery_movie=movie
    )
    assert failure is not None and "no rendered alpha" in failure
    monkeypatch.setattr(completion, "_alpha_box", lambda render, time: (0, 0, 10, 1079))
    failure, _ = completion.check_clip_geometry(
        work, "v1", parts, program=None, delivery_movie=movie
    )
    assert failure is not None and "exceeds the safe area" in failure
    monkeypatch.setattr(completion, "_alpha_box", lambda render, time: (800, 850, 900, 1000))
    failure, info = completion.check_clip_geometry(
        work, "v1", parts, program=None, delivery_movie=movie
    )
    assert failure is None and info["samples"] == 4
    monkeypatch.setattr(
        completion, "_diff_band_box", lambda program, movie, time: (800, 850, 900, 1079)
    )
    failure, _ = completion.check_clip_geometry(
        work, "v1", parts, program=program, delivery_movie=movie
    )
    assert failure is not None and "canvas edge" in failure
    monkeypatch.setattr(
        completion, "_diff_band_box", lambda program, movie, time: (800, 850, 900, 1000)
    )
    failure, _ = completion.check_clip_geometry(
        work, "v1", parts, program=program, delivery_movie=movie
    )
    assert failure is None
    tail = json.loads((work / "titles" / "tail-bundle-v1.json").read_text(encoding="utf-8"))
    tail["titleParts"][0]["offsetFrame"] = 99999
    (work / "titles" / "tail-bundle-v1.json").write_text(json.dumps(tail), encoding="utf-8")
    failure, _ = completion.check_clip_geometry(
        work, "v1", parts, program=None, delivery_movie=movie
    )
    assert failure is not None and "measurable samples" in failure


def test_placement_violations(recording: Recording) -> None:
    _write(
        recording.work / "titles" / "bundle-v1.json",
        {
            "titleParts": [
                {
                    "offsetFrame": 0,
                    "placements": [
                        {"text": "flash", "startFrame": 0, "endFrame": 1},
                        {"text": "settled caption here", "startFrame": 10, "endFrame": 40},
                    ],
                }
            ],
        },
    )
    rows = [
        {"part": 0, "startFrame": 0, "leftOk": False, "box": [100, 800, 900, 900]},
        {"part": 0, "startFrame": 12, "leftOk": False, "box": [742, 800, 900, 900]},
        {"part": 0, "startFrame": 20, "leftOk": False, "box": [500, 800, 900, 900]},
    ]
    _write(
        recording.work / "titles" / "left-edge-v1-settled.json",
        {"summary": {"undetected": 2}, "rows": rows},
    )
    passed, placement = completion.check_placement(recording.work, "v1")
    assert passed is False
    assert [v["box"] for v in _rows(placement["violations"])] == [[500, 800, 900, 900]]
    assert [t["box"] for t in _rows(placement["transient"])] == [[100, 800, 900, 900]]
    assert [t["box"] for t in _rows(placement["tolerated"])] == [[742, 800, 900, 900]]
    assert placement["undetected"] == 2


def test_survey_analyzer() -> None:
    chat = {"text": " @user hello", "x0": 70.0, "y0": 600.0, "x1": 300.0, "y1": 620.0}
    frames = [{"time": float(i), "boxes": [dict(chat)]} for i in range(5)]
    frames.append({"time": 200.0, "error": "vision-error", "boxes": []})
    result = completion.analyze_mask_survey(frames, width=1280, height=720)
    assert result["reason"] == "no-repeatable-caption-band-observed"
    assert result["bandBoxes"] == 0
    assert result["maxRepeatSpan"] == 0
    assert result["ocrOkFrames"] == 5
    caption = {"text": "同じ字幕", "x0": 500.0, "y0": 600.0, "x1": 900.0, "y1": 660.0}
    frames = [{"time": float(i * 27), "boxes": [dict(caption)]} for i in range(4)]
    result = completion.analyze_mask_survey(frames, width=1280, height=720)
    assert result["reason"] == "repeated-caption-band-detected"
    assert result["maxRepeatSpan"] == 4
    with pytest.raises(ReplayError):
        completion.analyze_mask_survey([], width=0, height=720)
    with pytest.raises(ReplayError):
        completion.analyze_mask_survey([{"time": "x", "boxes": []}], width=1280, height=720)
    with pytest.raises(ReplayError):
        completion.analyze_mask_survey(
            [{"time": 1.0, "boxes": [{"text": "あ", "x0": "x", "y0": 0.0, "x1": 1.0, "y1": 1.0}]}],
            width=1280,
            height=720,
        )


def test_helper_branches(tmp_path: Path) -> None:
    assert completion._evidence_set_complete(tmp_path, "") is False
    assert completion._manifest_version(tmp_path / "x.json") == ""
    with pytest.raises(ReplayError):
        completion._items(5, "nope")
    with pytest.raises(ReplayError):
        completion._integer(True, "nope")
    with pytest.raises(ReplayError):
        completion._integer(1.5, "nope")
    with pytest.raises(ReplayError):
        completion._count(True, "nope")
    with pytest.raises(ReplayError):
        completion._count(1.5, "nope")
    with pytest.raises(ReplayError):
        completion._count("x", "nope")
    assert completion._count("42", "nope") == 42
    with pytest.raises(ReplayError):
        completion._number(True, "nope")
    with pytest.raises(ReplayError):
        completion._run("definitely-not-a-tool-xyz", [], timeout=10)
    with pytest.raises(ReplayError):
        completion.stream_duration_seconds({"format": {}})
    video = completion._video_stream({"streams": [{"width": 10}]})
    assert video == {"width": 10}
    with pytest.raises(ReplayError):
        completion._resolve_media(tmp_path / "missing.mp4")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ReplayError):
        completion._resolve_media(tmp_path / "a.txt")


def test_srt_and_probe_helpers(tmp_path: Path) -> None:
    with pytest.raises(ReplayError):
        completion._srt_cues(tmp_path / "missing.srt")
    bad = tmp_path / "bad.srt"
    bad.write_text("1\nnot a clock --> x\ntext\n", encoding="utf-8")
    with pytest.raises(ReplayError):
        completion._srt_cues(bad)
    with pytest.raises(ReplayError):
        completion.stream_duration_seconds({"format": {"duration": "-1"}})
    with pytest.raises(ReplayError):
        completion._video_stream({"streams": []})
    with pytest.raises(ReplayError):
        completion._audio_stream({"streams": []})
    with pytest.raises(ReplayError):
        completion._resolve_media(tmp_path / "missing.mp4")


def test_volume_levels_unreadable(recording: Recording) -> None:
    _levels(recording.tools["levels"], recording.movie.name, "x", "y")
    with pytest.raises(ReplayError):
        completion.volume_levels(recording.movie, 0.0)


def test_faststart_shapes(tmp_path: Path) -> None:
    good = tmp_path / "good.mp4"
    good.write_bytes(_box(b"ftyp") + _box(b"moov") + _box(b"mdat"))
    assert completion.has_faststart(good) is True
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(_box(b"ftyp") + _box(b"mdat") + _box(b"moov"))
    assert completion.has_faststart(bad) is False
    tiny = tmp_path / "tiny.mp4"
    tiny.write_bytes(b"1234")
    assert completion.has_faststart(tiny) is False
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(_box(b"ftyp", size=4))
    assert completion.has_faststart(broken) is False


def test_main_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert completion.main(["check", str(tmp_path / "missing")]) == 1
    root = tmp_path / "youtube-AAAAAAAAAAA"
    root.mkdir()
    (root / "edit-config.json").write_text("{}", encoding="utf-8")
    assert completion.main(["check", str(root)]) == 1


def test_ass_candidate_route_is_separate_and_preserves_other_gates(
    recording: Recording,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = recording.work / "ass-evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        caption_ass_completion,
        "read_document",
        lambda path: {"artifacts": {"plan": {}, "cues": {}, "cuts": {}}, "frames": 3000},
    )
    monkeypatch.setattr(caption_ass_completion, "bound_path", lambda ref: evidence)
    monkeypatch.setattr(
        caption_ass_completion, "transcription", lambda *args: (None, {"characters": 2})
    )
    monkeypatch.setattr(
        caption_ass_completion,
        "assess",
        lambda *args: (
            "missing measured color",
            {"backend": "libass-pos-phrase"},
            {"violations": ["color"]},
        ),
    )
    result = completion.main(
        [
            "check",
            str(recording.root),
            "--workdir",
            str(recording.work),
            "--cut-program",
            str(recording.cut_program),
            "--candidate",
            str(recording.movie),
            "--transcript",
            str(recording.movie.with_suffix(".srt")),
            "--ass-evidence",
            str(evidence),
        ]
    )
    assert result == 2
    report = json.loads(capsys.readouterr().out)
    assert report["candidateOnly"] is True
    assert report["missingStages"] == {"dynamic_captions": "missing measured color"}
    assert report["facts"]["placement"]["violations"] == ["color"]


def test_candidate_requires_explicit_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        completion.main(["check", str(tmp_path), "--candidate", str(tmp_path / "movie.mp4")]) == 1
    )
    out = capsys.readouterr()
    assert "YOUTUBE_EDIT_COMPLETION_ERROR" in out.out


def test_find_deliveries_empty(tmp_path: Path) -> None:
    assert completion.find_deliveries(tmp_path) == []
    root = tmp_path / "youtube-AAAAAAAAAAA"
    (root / "delivery" / "nodate").mkdir(parents=True)
    assert completion.find_deliveries(root) == []


@pytest.mark.parametrize("registered", [False, True])
def test_ass_published_selection_uses_bound_movie_not_generation_date(
    recording: Recording, monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    later = recording.root / "delivery" / "2027-01-01_00-00-00_JST_youtube-TEST1234567-old"
    later.mkdir()
    (later / "wrong.mp4").write_bytes(b"wrong old generation")
    (later / "wrong.srt").write_text("old", encoding="utf-8")
    evidence = (
        recording.root / "evidence" / "caption-ass-delivery.json"
        if registered
        else recording.work / "proof.json"
    )
    evidence.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        caption_ass_completion, "read_document", lambda path: {"artifacts": {"movie": {}}}
    )
    monkeypatch.setattr(caption_ass_completion, "bound_path", lambda ref: recording.movie)
    monkeypatch.setattr(
        completion,
        "assess_delivery",
        lambda *args, **kwargs: ({}, {"placement": {"violations": []}}),
    )
    report = completion.completion_report(
        recording.root,
        workdir=recording.work,
        ass_evidence=None if registered else evidence,
    )
    assert report["route"] == "youtube-libass-phrase"
    assert report["complete"] is True
    assert json.loads(json.dumps(report))["deliveries"][-1]["superseded"] is True
