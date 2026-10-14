import hashlib
import json
from pathlib import Path

import pytest

from replay import edit_completion
from replay.core import ReplayError
from replay.delivery import require_new_delivery
from replay.edit_completion import (
    assess_delivery,
    completion_report,
    create_config,
    dated_deliveries,
    is_superseded,
    load_config,
    main,
    native_media_path,
    supersede_incomplete,
)
from replay.jobs import register_completed

STEM = "2024-01-01_09-00-00_JST_replay-7"


def test_native_v3_dispatch_reports_recorded_and_strict_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from replay import native_delivery_completion

    root = recording(tmp_path)
    movie = root / "revisions/native/delivery" / f"{STEM}-review.mp4"
    movie.parent.mkdir(parents=True)
    movie.write_bytes(b"candidate")
    write(
        movie.parent.parent / "metadata/native-evidence.json",
        {"version": 3, "backend": "native-fcp-partitioned-whole"},
    )
    monkeypatch.setattr(
        native_delivery_completion,
        "assess",
        lambda *args, **kwargs: {
            "candidateAvailable": True,
            "recordedRequiredStagesComplete": True,
            "strictComplete": False,
            "strictFailures": {"dynamic_captions": ["actual paint missing"]},
        },
    )
    assert assess_delivery(movie, root=root) == {"dynamic_captions": "actual paint missing"}
    report = completion_report(root)
    assert report["complete"] is False
    assert "recordedRequiredStagesComplete" in json.dumps(report)
    assert main(["check", str(root)]) == 2


def test_native_v3_invalid_and_no_truthiness_success(tmp_path: Path) -> None:
    root = recording(tmp_path)
    movie = root / "revisions/native/delivery" / f"{STEM}-review.mp4"
    movie.parent.mkdir(parents=True)
    movie.write_bytes(b"candidate")
    write(movie.parent.parent / "metadata/native-evidence.json", {"version": 3, "backend": "wrong"})
    failures = assess_delivery(movie, root=root)
    assert set(failures) == set(edit_completion.REQUIRED_STAGES)
    assert edit_completion.native_stage_failures(
        {"strictFailures": {}, "strictComplete": False}
    ) == dict.fromkeys(edit_completion.REQUIRED_STAGES, "Native strict verification incomplete")
    assert (
        edit_completion.native_stage_failures({"strictFailures": {}, "strictComplete": True}) == {}
    )


def test_native_v3_live_crypto_error_is_stage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from replay import native_delivery_completion
    from replay.stages import StageError

    root = recording(tmp_path)
    movie = root / "revisions/native/delivery" / f"{STEM}-review.mp4"
    movie.parent.mkdir(parents=True)
    movie.write_bytes(b"candidate")
    write(movie.parent.parent / "metadata/native-evidence.json", {"version": 3})

    def fail(
        index: object, *, root: Path, delivery: Path, verify_content: bool
    ) -> dict[str, object]:
        assert verify_content is True
        raise StageError("changed original expected bytes")

    monkeypatch.setattr(native_delivery_completion, "assess", fail)
    assert "changed original expected bytes" in assess_delivery(movie, root=root)["transcription"]
    assert main(["check", str(root)]) == 2


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def recording(tmp_path: Path) -> Path:
    root = tmp_path / "repo/edits/replay-7"
    source = tmp_path / "downloads" / f"{STEM.removesuffix('-review')}.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    register_completed(root=tmp_path / "jobs", replay_id="7", output=source)
    (tmp_path / "engine").mkdir()
    (tmp_path / "scope").mkdir()
    create_config(
        root, jobs=tmp_path / "jobs", engine=tmp_path / "engine", executor_scope=tmp_path / "scope"
    )
    write(root / "evidence/source-probe.json", {"format": {"filename": str(source)}})
    return root


def render(root: Path, name: str, *, masks: int = 0, output_frames: int = 90) -> Path:
    directory = root / "revisions" / name
    no_text = root / "evidence/source-masks-none/summary.json"
    if not no_text.exists():
        write(no_text, {"noSourceSubtitlesDetected": True, "observedTextFrames": 0})
    write(
        directory / "summary.json",
        {"masks": masks, "captionCues": 2, "outputFrames": output_frames},
    )
    write(directory / "cuts.json", {"cursor": output_frames, "segments": [{"sourceEndFrame": 120}]})
    write(directory / "captions.json", [{"text": "a"}, {"text": "b"}])
    write(
        directory / "render-plan.json",
        [
            {
                "request": {
                    "recipe": {
                        "video": {
                            "captionStyle": {"fontSize": 56},
                            "captions": [{"text": "a"}, {"text": "b"}],
                            "masks": [{"x": 0}] if masks else [],
                        }
                    }
                }
            }
        ],
    )
    write(
        directory / "metadata/complete.json",
        {"fullAVDecodePassed": True, "videoFrames": output_frames},
    )
    write(
        directory / "metadata/mask-frame-audit.json",
        {"missingMaskFrames": 0, "extraMaskFrames": 0},
    )
    (directory / "delivery").mkdir()
    (directory / "delivery/full-review.mp4").write_bytes(b"render")
    return directory


def music_removed(root: Path, render_dir: Path, *, status: str = "listening-approved") -> Path:
    revision = root / "revisions/audio-final"
    write(
        revision / "metadata/verification.json",
        {
            "status": status,
            "full_audio_video_decode": "passed",
            "source": str((render_dir / "delivery/full-review.mp4").relative_to(root.parents[1])),
        },
    )
    (revision / "delivery").mkdir()
    delivery = revision / f"delivery/{STEM}-review.mp4"
    delivery.write_bytes(b"final")
    delivery.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\na\n")
    return delivery


def test_v2_dispatch_keeps_missing_stages_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from replay import delivery_completion_v2

    root = recording(tmp_path)
    movie = root / "revisions/direct/delivery/candidate.mp4"
    write(movie, "synthetic")
    write(movie.parent.parent / "metadata/native-evidence.json", {"version": 2})

    def assess(
        index: Path, *, delivery: Path, root: Path, source: Path, storage: Path | None = None
    ) -> dict[str, str]:
        return {"dynamic_captions": "FCP import has not been observed"}

    monkeypatch.setattr(delivery_completion_v2, "assess", assess)
    assert assess_delivery(movie, root=root) == {
        "dynamic_captions": "FCP import has not been observed"
    }


def test_config_is_created_from_completed_job_and_requires_every_stage(tmp_path: Path) -> None:
    root = recording(tmp_path)
    config = load_config(root)
    assert config.replay_id == "7"
    assert config.source == (tmp_path / "downloads/2024-01-01_09-00-00_JST_replay-7.mp4").resolve()
    assert config.source_size == 5
    data = json.loads((root / "edit-config.json").read_text())
    assert data["stages"] == {
        "bgm_removal": True,
        "silence_cut": True,
        "source_caption_mask": True,
        "transcription": True,
        "dynamic_captions": True,
    }
    assert oct((root / "edit-config.json").stat().st_mode & 0o777) == "0o600"
    with pytest.raises(FileExistsError):
        create_config(
            root,
            jobs=tmp_path / "jobs",
            engine=tmp_path / "engine",
            executor_scope=tmp_path / "scope",
        )


def test_config_rejects_disabled_stage_extra_field_and_changed_source(tmp_path: Path) -> None:
    root = recording(tmp_path)
    path = root / "edit-config.json"
    original = json.loads(path.read_text())
    write(path, {**original, "stages": {**original["stages"], "source_caption_mask": False}})
    with pytest.raises(ReplayError, match="every required stage"):
        load_config(root)
    write(path, {**original, "mask_threshold": 0.3})
    with pytest.raises(ReplayError, match="unknown"):
        load_config(root)
    write(path, {**original, "replay_id": "8"})
    with pytest.raises(ReplayError, match="does not belong"):
        load_config(root)
    write(path, {**original, "source_size": 6})
    with pytest.raises(ReplayError, match="size changed"):
        load_config(root)


def test_config_requires_completed_job_and_directories(tmp_path: Path) -> None:
    (tmp_path / "engine").mkdir()
    with pytest.raises(ReplayError, match="completed download job"):
        create_config(
            tmp_path / "replay-9",
            jobs=tmp_path / "jobs",
            engine=tmp_path / "engine",
            executor_scope=tmp_path / "engine",
        )
    with pytest.raises(ReplayError, match="edits/replay-<id>"):
        create_config(
            tmp_path / "other",
            jobs=tmp_path / "jobs",
            engine=tmp_path / "engine",
            executor_scope=tmp_path / "engine",
        )
    source = tmp_path / "a.mp4"
    source.write_bytes(b"v")
    register_completed(root=tmp_path / "jobs", replay_id="9", output=source)
    with pytest.raises(ReplayError, match="must exist"):
        create_config(
            tmp_path / "replay-9",
            jobs=tmp_path / "jobs",
            engine=tmp_path / "engine",
            executor_scope=tmp_path / "missing",
        )


def test_native_media_path_rejects_extensionless_link_target(tmp_path: Path) -> None:
    target = tmp_path / "proxy-file"
    target.write_bytes(b"video")
    link = tmp_path / "source-video-only.mp4"
    link.symlink_to(target)
    with pytest.raises(ReplayError, match=r"\.mp4/\.mov/\.m4v"):
        native_media_path(link)
    named = tmp_path / "proxy.mp4"
    named.write_bytes(b"video")
    assert native_media_path(named) == named.resolve()


def test_dated_native_media_keeps_required_stage_gate(tmp_path: Path) -> None:
    root = recording(tmp_path)
    revision = root / "revisions/native/delivery"
    revision.mkdir(parents=True)
    delivery = revision / f"{STEM}-review.mov"
    delivery.write_bytes(b"native candidate")
    assert len(dated_deliveries(root)) == 1
    assert completion_report(root) == {
        "replay": "replay-7",
        "complete": False,
        "deliveries": [
            {
                "delivery": "revisions/native/delivery/2024-01-01_09-00-00_JST_replay-7-review.mov",
                "complete": False,
                "superseded": False,
                "missingStages": {
                    "bgm_removal": "no music-removal verification",
                    "silence_cut": "not produced by the shared cut/mask/caption render",
                    "source_caption_mask": "not produced by the shared cut/mask/caption render",
                    "transcription": "not produced by the shared cut/mask/caption render",
                    "dynamic_captions": "not produced by the shared cut/mask/caption render",
                },
            }
        ],
    }


def test_delivery_discovery_accepts_other_native_suffixes(tmp_path: Path) -> None:
    root = recording(tmp_path)
    directory = root / "revisions/native/delivery"
    directory.mkdir(parents=True)
    (directory / f"{STEM}-review.m4v").write_bytes(b"candidate")
    (directory / f"{STEM}-review.MOV").write_bytes(b"candidate")
    found = dated_deliveries(root)
    assert len(found) == 2
    assert found[0].name == "2024-01-01_09-00-00_JST_replay-7-review.MOV"
    assert found[1].name == "2024-01-01_09-00-00_JST_replay-7-review.m4v"


def test_delivery_discovery_excludes_nonmedia_directories_links_and_other_recordings(
    tmp_path: Path,
) -> None:
    root = recording(tmp_path)
    directory = root / "revisions/native/delivery"
    directory.mkdir(parents=True)
    (directory / f"{STEM}-review.mov").mkdir()
    (directory / f"{STEM}-review.mp4").symlink_to(root / "edit-config.json")
    (directory / f"{STEM}-review.srt").write_text("not a movie")
    (directory / "2024-01-01_09-00-00_JST_replay-8-review.mov").write_bytes(b"other")
    assert dated_deliveries(root) == []
    assert completion_report(root) == {"replay": "replay-7", "complete": False, "deliveries": []}


def test_complete_delivery_passes_every_stage(tmp_path: Path) -> None:
    root = recording(tmp_path)
    delivery = music_removed(root, render(root, "render-a"))
    assert assess_delivery(delivery, root=root) == {}
    assert completion_report(root) == {
        "replay": "replay-7",
        "complete": True,
        "deliveries": [
            {
                "delivery": f"revisions/audio-final/delivery/{STEM}-review.mp4",
                "complete": True,
                "superseded": False,
                "missingStages": {},
            }
        ],
    }
    assert main(["check", str(root)]) == 0


def test_masks_none_after_ocr_failure_is_incomplete(tmp_path: Path) -> None:
    root = recording(tmp_path)
    write(
        root / "evidence/source-masks-none/summary.json",
        {"reason": "native-refinement-ocr-failed-source-pixels-retained"},
    )
    delivery = music_removed(root, render(root, "render-a", masks=0))
    assert assess_delivery(delivery, root=root) == {
        "source_caption_mask": (
            "source captions left visible (native-refinement-ocr-failed-source-pixels-retained)"
        )
    }


def test_legacy_fixed_blur_cannot_certify_source_removal(tmp_path: Path) -> None:
    root = recording(tmp_path)
    delivery = music_removed(root, render(root, "render-a", masks=3))
    assert assess_delivery(delivery, root=root) == {
        "source_caption_mask": "fixed source-caption blur is not detected local repair"
    }


def test_masks_none_is_complete_only_without_a_caption_band(tmp_path: Path) -> None:
    root = recording(tmp_path)
    write(
        root / "evidence/source-masks-none/summary.json",
        {"reason": "no-repeatable-caption-band-observed"},
    )
    delivery = music_removed(root, render(root, "render-a", masks=0))
    assert assess_delivery(delivery, root=root) == {}


def test_masks_none_with_empty_text_survey_is_complete(tmp_path: Path) -> None:
    root = recording(tmp_path)
    write(
        root / "evidence/source-masks-none/summary.json",
        {
            "reason": "no-repeatable-subtitle-band",
            "observedTextFrames": 0,
            "noSourceSubtitlesDetected": True,
        },
    )
    delivery = music_removed(root, render(root, "render-a", masks=0))
    assert assess_delivery(delivery, root=root) == {}


def test_masks_none_with_observed_text_stays_incomplete(tmp_path: Path) -> None:
    root = recording(tmp_path)
    write(
        root / "evidence/source-masks-none/summary.json",
        {
            "reason": "no-repeatable-subtitle-band",
            "observedTextFrames": 74,
            "noSourceSubtitlesDetected": True,
        },
    )
    delivery = music_removed(root, render(root, "render-a", masks=0))
    assert assess_delivery(delivery, root=root) == {
        "source_caption_mask": "source captions left visible (no-repeatable-subtitle-band)"
    }


def test_uncut_unverified_music_and_missing_captions_are_reported(tmp_path: Path) -> None:
    root = recording(tmp_path)
    render_dir = render(root, "render-a", output_frames=120)
    write(render_dir / "captions.json", [])
    delivery = music_removed(root, render_dir, status="rejected-music-quality")
    delivery.with_suffix(".srt").write_text("")
    assert assess_delivery(delivery, root=root) == {
        "bgm_removal": "music removal not verified (rejected-music-quality)",
        "silence_cut": "no non-speech interval was removed",
        "dynamic_captions": "captions were not burned by a verified full render",
        "transcription": "delivery has no transcript track file",
    }


def test_malformed_render_evidence_is_reported(tmp_path: Path) -> None:
    root = recording(tmp_path)
    render_dir = render(root, "render-a")
    write(render_dir / "summary.json", {"masks": -1, "captionCues": 0})
    write(render_dir / "cuts.json", {"segments": []})
    write(render_dir / "metadata/mask-frame-audit.json", {"missingMaskFrames": 4})
    delivery = music_removed(root, render_dir)
    assert assess_delivery(delivery, root=root) == {
        "silence_cut": "cut plan is missing",
        "source_caption_mask": "mask count is missing or differs from the render plan",
        "transcription": "no transcript cues",
        "dynamic_captions": "captions were not burned by a verified full render",
    }


def test_unaudited_mask_frames_and_track_only_captions_fail(tmp_path: Path) -> None:
    root = recording(tmp_path)
    render_dir = render(root, "render-a")
    write(render_dir / "metadata/mask-frame-audit.json", {"missingMaskFrames": 4})
    write(render_dir / "metadata/complete.json", {"fullAVDecodePassed": False})
    delivery = music_removed(root, render_dir)
    assert assess_delivery(delivery, root=root) == {
        "source_caption_mask": "rendered mask frames differ from the plan",
        "dynamic_captions": "captions were not burned by a verified full render",
    }


def test_render_plan_must_list_captions_and_masks(tmp_path: Path) -> None:
    root = recording(tmp_path)
    render_dir = render(root, "render-a")
    write(render_dir / "render-plan.json", [])
    delivery = music_removed(root, render_dir)
    with pytest.raises(ReplayError, match="Render plan is empty"):
        assess_delivery(delivery, root=root)
    write(render_dir / "render-plan.json", [{"request": {"recipe": {"video": {}}}}])
    with pytest.raises(ReplayError, match="lacks a mask list"):
        assess_delivery(delivery, root=root)
    write(render_dir / "render-plan.json", [{"request": {"recipe": {"video": {"masks": []}}}}])
    with pytest.raises(ReplayError, match="lacks a caption list"):
        assess_delivery(delivery, root=root)


def test_libass_burn_counts_only_while_its_ass_file_is_unchanged(tmp_path: Path) -> None:
    root = recording(tmp_path)
    render_dir = render(root, "render-a")
    ass = render_dir / "captions/part-000000.ass"
    ass.parent.mkdir()
    ass.write_text("[Script Info]\n", encoding="utf-8")
    burn = {
        "ass": "captions/part-000000.ass",
        "assSha256": hashlib.sha256(b"[Script Info]\n").hexdigest(),
        "captions": [{"text": "a"}, {"text": "b"}],
        "style": {"position": "head-tracked"},
    }
    write(
        render_dir / "render-plan.json",
        [{"captionBurn": burn, "request": {"recipe": {"video": {"masks": []}}}}],
    )
    delivery = music_removed(root, render_dir)
    assert assess_delivery(delivery, root=root) == {}
    ass.write_text("[Script Info]\nchanged\n", encoding="utf-8")
    assert assess_delivery(delivery, root=root) == {
        "dynamic_captions": "captions were not burned by a verified full render"
    }
    write(
        render_dir / "render-plan.json",
        [
            {
                "captionBurn": {**burn, "style": None},
                "request": {"recipe": {"video": {"masks": []}}},
            }
        ],
    )
    assert assess_delivery(delivery, root=root) == {
        "dynamic_captions": "captions were not burned by a verified full render"
    }


def test_render_report_without_music_source_is_not_music_removal(tmp_path: Path) -> None:
    root = recording(tmp_path)
    delivery = music_removed(root, render(root, "render-a"))
    write(delivery.parents[1] / "metadata/verification.json", {"status": "unreviewed"})
    assert assess_delivery(delivery, root=root) == {
        "bgm_removal": "no music-removal verification",
        "silence_cut": "not produced by the shared cut/mask/caption render",
        "source_caption_mask": "not produced by the shared cut/mask/caption render",
        "transcription": "not produced by the shared cut/mask/caption render",
        "dynamic_captions": "not produced by the shared cut/mask/caption render",
    }


def test_source_preserving_copy_misses_every_editing_stage(tmp_path: Path) -> None:
    root = recording(tmp_path)
    delivery = root / f"revisions/source-copy/delivery/{STEM}-review.mp4"
    delivery.parent.mkdir(parents=True)
    delivery.write_bytes(b"copy")
    delivery.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\na\n")
    assert assess_delivery(delivery, root=root) == {
        "bgm_removal": "no music-removal verification",
        "silence_cut": "not produced by the shared cut/mask/caption render",
        "source_caption_mask": "not produced by the shared cut/mask/caption render",
        "transcription": "not produced by the shared cut/mask/caption render",
        "dynamic_captions": "not produced by the shared cut/mask/caption render",
    }
    assert main(["check", str(root)]) == 2


def test_supersede_records_incomplete_delivery_and_allows_new_revision(tmp_path: Path) -> None:
    root = recording(tmp_path)
    delivery = root / f"revisions/source-copy/delivery/{STEM}-review.mp4"
    delivery.parent.mkdir(parents=True)
    delivery.write_bytes(b"copy")
    write(root / f"evidence/{STEM}-review-selection.json", {"status": "selected"})
    with pytest.raises(ReplayError, match="already has a dated delivery"):
        require_new_delivery(root)
    assert supersede_incomplete(root) == [delivery]
    digest = hashlib.sha256(b"copy").hexdigest()
    entry = json.loads((root / f"evidence/superseded/{digest[:16]}.json").read_text())
    assert entry["delivery"] == f"revisions/source-copy/delivery/{STEM}-review.mp4"
    assert entry["status"] == "superseded-incomplete-required-stages"
    assert delivery.read_bytes() == b"copy"
    assert require_new_delivery(root) == f"{STEM}-review"
    assert supersede_incomplete(root) == []
    delivery.write_bytes(b"edit")
    assert is_superseded(root, delivery) is False
    with pytest.raises(ReplayError, match="already has a dated delivery"):
        require_new_delivery(root)


def test_requested_regeneration_supersedes_a_complete_delivery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = recording(tmp_path)
    delivery = music_removed(root, render(root, "render-a"))
    with pytest.raises(ReplayError, match="stated reason"):
        edit_completion.supersede_for_regeneration(root, " ")
    assert main(["regenerate", str(root), "--reason", "new caption style"]) == 0
    assert capsys.readouterr().out == (
        f"SUPERSEDED_FOR_REGENERATION revisions/audio-final/delivery/{STEM}-review.mp4\n"
    )
    assert is_superseded(root, delivery)
    assert completion_report(root)["complete"] is False
    assert require_new_delivery(root) == f"{STEM}-review"
    assert edit_completion.supersede_for_regeneration(root, "again") == []


def test_supersede_never_marks_complete_delivery(tmp_path: Path) -> None:
    root = recording(tmp_path)
    music_removed(root, render(root, "render-a"))
    assert main(["supersede", str(root)]) == 0
    assert list((root / "evidence/superseded").iterdir()) == []
    assert is_superseded(root, root / f"revisions/audio-final/delivery/{STEM}-review.mp4") is False


def test_cli_init_source_scope_and_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = recording(tmp_path)
    assert main(["source", str(root)]) == 0
    assert main(["scope", str(root)]) == 0
    output = capsys.readouterr().out.splitlines()
    assert output == [
        str((tmp_path / "downloads/2024-01-01_09-00-00_JST_replay-7.mp4").resolve()),
        str((tmp_path / "scope").resolve()),
    ]
    other = tmp_path / "repo/edits/replay-8"
    assert (
        main(
            [
                "init",
                str(other),
                "--jobs",
                str(tmp_path / "jobs"),
                "--engine",
                str(tmp_path / "engine"),
                "--executor-scope",
                str(tmp_path / "scope"),
            ]
        )
        == 1
    )
    assert "EDIT_GATE_ERROR" in capsys.readouterr().err
    source = tmp_path / "b.mp4"
    source.write_bytes(b"v")
    register_completed(root=tmp_path / "jobs", replay_id="8", output=source)
    assert (
        main(
            [
                "init",
                str(other),
                "--jobs",
                str(tmp_path / "jobs"),
                "--engine",
                str(tmp_path / "engine"),
                "--executor-scope",
                str(tmp_path / "scope"),
            ]
        )
        == 0
    )
    assert (other / "edit-config.json").is_file()


def test_oversized_evidence_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = recording(tmp_path)
    monkeypatch.setattr(edit_completion, "MAX_JSON_BYTES", 1)
    with pytest.raises(ReplayError, match="size limit"):
        load_config(root)


def storage(tmp_path: Path, root: Path) -> Path:
    directory = tmp_path / "ssd" / root.name
    directory.mkdir(parents=True)
    return directory.resolve()


def selected_candidate(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    root = recording(tmp_path)
    external = storage(tmp_path, root)
    edit_completion.bind_storage(root, external)
    target = external / "revisions/original/delivery" / f"{STEM}-review.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"synthetic final media")
    named = external / "revisions/current/delivery" / target.name
    named.parent.mkdir(parents=True)
    named.symlink_to(Path("../../original/delivery") / target.name)
    transcript = named.with_suffix(".srt")
    transcript.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
    xml = root / "candidate.fcpxml"
    xml.write_text("<fcpxml/>")
    info = target.stat()
    header: dict[str, object] = {
        "version": 1,
        "status": "unverified-candidate",
        "movie": str(named.relative_to(external)),
        "movieTarget": str(target.relative_to(external)),
        "movieSize": info.st_size,
        "movieMtimeNanoseconds": info.st_mtime_ns,
        "transcript": str(transcript.relative_to(external)),
        "transcriptSha256": hashlib.sha256(transcript.read_bytes()).hexdigest(),
        "fcpxml": str(xml),
        "fcpxmlSha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "editConfigSha256": hashlib.sha256((root / "edit-config.json").read_bytes()).hexdigest(),
        "renderConsumptionVerified": False,
        "requiredStagesVerified": False,
    }
    write(external / "evidence/candidate-selection.json", header)
    write(named.parent / "candidate.json", header)
    return root, external, named, header


def test_current_candidate_alias_deduplicates_target_and_ignores_old_alias(tmp_path: Path) -> None:
    root, external, named, _ = selected_candidate(tmp_path)
    old = external / "revisions/old/delivery" / named.name
    old.parent.mkdir(parents=True)
    old.symlink_to(named.resolve())
    assert dated_deliveries(root, external) == [named]


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", True),
        ("status", "complete"),
        ("unknown", True),
        ("movie", "../outside.mp4"),
        ("movie", "/outside.mp4"),
        ("movie", 0),
        ("movie", "revisions/current/delivery/2024-01-02_09-00-00_JST_replay-7-review.mp4"),
        ("movie", "revisions/current/delivery/2024-01-01_09-00-00_JST_replay-8-review.mp4"),
        ("transcriptSha256", "f" * 64),
        ("editConfigSha256", "f" * 64),
        ("fcpxmlSha256", "invalid"),
        ("fcpxml", 4),
        ("fcpxml", "/absent.fcpxml"),
        ("movieSize", 1),
        ("movieMtimeNanoseconds", 1),
        ("requiredStagesVerified", True),
    ],
)
def test_current_candidate_rejects_malformed_header(
    tmp_path: Path, key: str, value: object
) -> None:
    root, external, named, header = selected_candidate(tmp_path)
    header[key] = value
    write(external / "evidence/candidate-selection.json", header)
    write(named.parent / "candidate.json", header)
    with pytest.raises((ReplayError, OSError)):
        dated_deliveries(root, external)


@pytest.mark.parametrize(
    "case",
    [
        "dangling",
        "escape",
        "missing-srt",
        "altered-copy",
        "header-link",
        "duplicate",
        "empty",
        "oversize",
        "other-storage",
    ],
)
def test_current_candidate_rejects_filesystem_substitution(tmp_path: Path, case: str) -> None:
    root, external, named, header = selected_candidate(tmp_path)
    selection = external / "evidence/candidate-selection.json"
    if case in {"dangling", "escape"}:
        named.unlink()
        outside = tmp_path / "outside.mp4"
        if case == "escape":
            outside.write_bytes(b"synthetic final media")
        named.symlink_to(outside)
    elif case == "missing-srt":
        named.with_suffix(".srt").unlink()
    elif case == "altered-copy":
        write(named.parent / "candidate.json", {})
    elif case == "header-link":
        selection.unlink()
        selection.symlink_to(named.parent / "candidate.json")
    elif case == "duplicate":
        selection.write_text('{"version":1,"version":1}')
    elif case == "empty":
        selection.write_bytes(b"")
    elif case == "oversize":
        selection.write_bytes(b" " * 65537)
    else:
        external = storage(tmp_path / "other", root)
        write(external / "evidence/candidate-selection.json", header)
    with pytest.raises((ReplayError, OSError)):
        dated_deliveries(root, external)


def test_current_candidate_mutation_during_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, external, named, _ = selected_candidate(tmp_path)
    digest = edit_completion._digest

    def mutate(path: Path) -> str:
        result = digest(path)
        if path.suffix == ".srt":
            named.resolve().write_bytes(b"changed")
        return result

    monkeypatch.setattr(edit_completion, "_digest", mutate)
    with pytest.raises(ReplayError, match="changed during discovery"):
        dated_deliveries(root, external)


def test_bind_storage_preserves_old_config_and_discovers_external_delivery(
    tmp_path: Path,
) -> None:
    root = recording(tmp_path)
    before = (root / "edit-config.json").read_bytes()
    external = storage(tmp_path, root)
    edit_completion.bind_storage(root, external)
    backups = sorted(root.glob("edit-config.*.json"))
    assert [path.read_bytes() for path in backups] == [before]
    assert load_config(root).storage_root == external
    movie = external / "revisions/native/delivery" / f"{STEM}-review.mp4"
    movie.parent.mkdir(parents=True)
    movie.write_bytes(b"external")
    assert dated_deliveries(root, external) == [movie]
    assert edit_completion.delivery_label(root, movie) == (
        f"storage:revisions/native/delivery/{STEM}-review.mp4"
    )


def test_stored_delivery_link_preserves_identity_and_stage_checks(tmp_path: Path) -> None:
    root = recording(tmp_path)
    movie = music_removed(root, render(root, "native"))
    assert assess_delivery(movie, root=root) == {}
    edit_completion.supersede_for_regeneration(root, "explicit revision")
    external = storage(tmp_path, root)
    edit_completion.bind_storage(root, external)
    target = external / movie.relative_to(root)
    target.parent.mkdir(parents=True)
    movie.rename(target)
    movie.symlink_to(target)

    assert dated_deliveries(root, external) == [movie]
    assert edit_completion.delivery_label(root, movie) == (
        "revisions/audio-final/delivery/2024-01-01_09-00-00_JST_replay-7-review.mp4"
    )
    assert assess_delivery(movie, root=root) == {}
    assert is_superseded(root, movie)


def test_storage_link_without_binding_stays_undiscovered(tmp_path: Path) -> None:
    root = recording(tmp_path)
    movie = music_removed(root, render(root, "native"))
    external = storage(tmp_path, root)
    target = external / movie.relative_to(root)
    target.parent.mkdir(parents=True)
    movie.rename(target)
    movie.symlink_to(target)
    assert dated_deliveries(root) == []


def test_storage_link_must_name_exact_mirror(tmp_path: Path) -> None:
    root = recording(tmp_path)
    movie = music_removed(root, render(root, "native"))
    external = storage(tmp_path, root)
    target = external / "elsewhere.mp4"
    movie.rename(target)
    movie.symlink_to(target)
    assert dated_deliveries(root, external) == []


def test_storage_mirror_must_not_itself_be_linked(tmp_path: Path) -> None:
    root = recording(tmp_path)
    movie = music_removed(root, render(root, "native"))
    external = storage(tmp_path, root)
    target = external / movie.relative_to(root)
    target.parent.mkdir(parents=True)
    unrelated = tmp_path / "unrelated.mp4"
    movie.rename(unrelated)
    target.symlink_to(unrelated)
    movie.symlink_to(target)
    assert dated_deliveries(root, external) == []


def test_storage_mirror_parent_must_not_escape_storage(tmp_path: Path) -> None:
    root = recording(tmp_path)
    movie = music_removed(root, render(root, "native"))
    external = storage(tmp_path, root)
    (external / "revisions").symlink_to(root / "revisions", target_is_directory=True)
    assert dated_deliveries(root, external) == [movie]


def test_disconnected_storage_link_is_not_a_delivery(tmp_path: Path) -> None:
    root = recording(tmp_path)
    movie = music_removed(root, render(root, "native"))
    external = storage(tmp_path, root)
    target = external / movie.relative_to(root)
    target.parent.mkdir(parents=True)
    movie.rename(target)
    movie.symlink_to(target)
    external.rename(tmp_path / "disconnected")
    assert dated_deliveries(root, external) == []


def test_bind_storage_is_final(tmp_path: Path) -> None:
    root = recording(tmp_path)
    external = storage(tmp_path, root)
    edit_completion.bind_storage(root, external)
    with pytest.raises(ReplayError, match="already names a storage root"):
        edit_completion.bind_storage(root, external)


def test_storage_root_must_be_named_after_recording(tmp_path: Path) -> None:
    root = recording(tmp_path)
    other = tmp_path / "ssd/elsewhere"
    other.mkdir(parents=True)
    with pytest.raises(ReplayError, match="named after"):
        edit_completion.bind_storage(root, other.resolve())


def test_storage_root_must_not_be_linked(tmp_path: Path) -> None:
    root = recording(tmp_path)
    real = storage(tmp_path, root)
    link = tmp_path / "links" / root.name
    link.parent.mkdir()
    link.symlink_to(real)
    with pytest.raises(ReplayError, match="unlinked directory"):
        edit_completion.bind_storage(root, link)


def test_storage_root_must_be_absolute(tmp_path: Path) -> None:
    root = recording(tmp_path)
    with pytest.raises(ReplayError, match="absolute path"):
        edit_completion.storage_root("relative", root=root, source=tmp_path / "x.mp4")


def test_storage_root_must_not_contain_original_media(tmp_path: Path) -> None:
    root = recording(tmp_path)
    inside = tmp_path / "downloads" / root.name
    inside.mkdir()
    with pytest.raises(ReplayError, match="outside account and original media"):
        edit_completion.bind_storage(root, inside.resolve())


def test_label_rejects_delivery_outside_both_roots(tmp_path: Path) -> None:
    root = recording(tmp_path)
    with pytest.raises(ReplayError, match="outside the recording"):
        edit_completion.delivery_label(root, tmp_path / "elsewhere.mp4")


def test_cli_bind_storage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = recording(tmp_path)
    external = storage(tmp_path, root)
    assert main(["bind-storage", str(root), str(external)]) == 0
    assert capsys.readouterr().out.strip() == str(root / "edit-config.json")
