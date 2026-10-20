"""A source without caption bands is accepted only from a thick, re-verified profile survey."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from replay import caption_absence as absence
from replay import replay_cleanup_plan as plan
from replay import subtitle_cleanup_workflow as cleanup
from replay.caption_io import FrameObservation
from replay.caption_masks import TextDetection, TextRectangle
from replay.core import ReplayError


def refuse(message: str) -> None:
    raise ValueError(message)


def surveyed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, frames: int = 240) -> Path:
    """A request and a survey whose verifier refuses for want of a caption band."""
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    profile = tmp_path / "survey/profile"
    profile.mkdir(parents=True)
    (profile / "inputs.json").write_text("{}", encoding="utf-8")
    (profile / "complete.json").write_text("[]", encoding="utf-8")
    job = SimpleNamespace(times=tuple(range(frames)), source=SimpleNamespace(sha256="f" * 64))
    monkeypatch.setattr(plan, "load", lambda _path: job)
    monkeypatch.setattr(
        plan,
        "verify_profile",
        lambda _r, _o: refuse("No variable Japanese subtitle bands; refuse a blanket mask"),
    )
    monkeypatch.setattr(absence, "band_free", lambda _r, _o: True)
    return request


def test_only_the_verifier_s_no_band_refusal_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        plan,
        "verify_profile",
        lambda _r, _o: refuse("No variable Japanese subtitle bands; refuse a blanket mask"),
    )
    assert absence.no_band(tmp_path / "r.json", tmp_path) is True
    monkeypatch.setattr(plan, "verify_profile", lambda _r, _o: refuse("clock differs"))
    assert absence.no_band(tmp_path / "r.json", tmp_path) is False
    monkeypatch.setattr(plan, "verify_profile", lambda _r, _o: {})
    assert absence.no_band(tmp_path / "r.json", tmp_path) is False


def test_the_survey_request_samples_at_least_240_frames(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"version": 1, "profileSamples": 64}', encoding="utf-8")
    copy = absence.survey_request(request, tmp_path / "copy.json")
    assert json.loads(copy.read_text(encoding="utf-8")) == {"version": 1, "profileSamples": 240}
    request.write_text('{"profileSamples": 512}', encoding="utf-8")
    copy = absence.survey_request(request, tmp_path / "copy.json")
    assert json.loads(copy.read_text(encoding="utf-8")) == {"profileSamples": 512}


def test_a_thick_band_free_survey_is_summarised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = surveyed(tmp_path, monkeypatch)
    summary = absence.summary(request, tmp_path / "survey")
    assert summary == {
        "version": 1,
        "reason": "no-repeatable-caption-band-observed",
        "sourceSHA256": "f" * 64,
        "totalFrames": 240,
        "request": {"path": str(request), "sha256": hashlib.sha256(b"{}").hexdigest()},
        "survey": str(tmp_path / "survey"),
        "profileInputs": {
            "path": str(tmp_path / "survey/profile/inputs.json"),
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        },
        "profileComplete": {
            "path": str(tmp_path / "survey/profile/complete.json"),
            "sha256": hashlib.sha256(b"[]").hexdigest(),
        },
    }


def test_a_thin_survey_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = surveyed(tmp_path, monkeypatch, frames=119)
    with pytest.raises(ReplayError, match="has 119 frames, fewer than 120"):
        absence.summary(request, tmp_path / "survey")


def test_a_survey_with_a_band_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = surveyed(tmp_path, monkeypatch)
    monkeypatch.setattr(plan, "verify_profile", lambda _r, _o: {})
    with pytest.raises(ReplayError, match="did not verify a source without caption bands"):
        absence.summary(request, tmp_path / "survey")


def test_a_pinned_summary_is_verified_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = surveyed(tmp_path, monkeypatch)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(absence.summary(request, tmp_path / "survey")), encoding="utf-8")
    assert absence.verify(absence.pin(path)) is None


def test_a_changed_summary_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = surveyed(tmp_path, monkeypatch)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(absence.summary(request, tmp_path / "survey")), encoding="utf-8")
    pinned = absence.pin(path)
    path.write_text("{}", encoding="utf-8")
    assert absence.verify(pinned) == "the caption absence summary is missing or changed"


def test_a_summary_whose_survey_changed_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = surveyed(tmp_path, monkeypatch)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(absence.summary(request, tmp_path / "survey")), encoding="utf-8")
    (tmp_path / "survey/profile/complete.json").write_text("[1]", encoding="utf-8")
    assert (
        absence.verify(absence.pin(path)) == "the caption absence summary differs from its survey"
    )


def test_a_summary_whose_profile_now_has_a_band_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = surveyed(tmp_path, monkeypatch)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(absence.summary(request, tmp_path / "survey")), encoding="utf-8")
    monkeypatch.setattr(absence, "band_free", lambda _r, _o: False)
    assert absence.verify(absence.pin(path)) == (
        "caption absence not verified (The survey did not verify a source without caption bands)"
    )


def observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: tuple[FrameObservation, ...]
) -> None:
    job = SimpleNamespace(times=(0.0, 1.0), data={})
    monkeypatch.setattr(plan, "load", lambda _path: job)
    monkeypatch.setattr(cleanup, "collector_frames", lambda _i, _c, _d: (frames, None))


def test_observations_without_text_are_band_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed(tmp_path, monkeypatch, (FrameObservation(0.0, 0.0, 1080, 1920, ()),
                                     FrameObservation(1.0, 1.0, 1080, 1920, ())))  # fmt: skip
    assert absence.band_free(tmp_path / "r.json", tmp_path) is True


def test_observations_off_their_clock_are_not_band_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed(tmp_path, monkeypatch, (FrameObservation(0.0, 0.0, 1080, 1920, ()),))
    assert absence.band_free(tmp_path / "r.json", tmp_path) is False


def test_observations_with_a_caption_band_are_not_band_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    words = ("春の花", "遠い海", "今日の天気", "朝の挨拶", "夕焼けの空", "白い雲", "青い鳥",
             "静かな森", "新しい本", "明日の予定")  # fmt: skip
    frames = tuple(
        FrameObservation(
            float(index),
            float(index),
            1080,
            1920,
            (TextDetection(word, 1, TextRectangle(340, 1296, 400, 32)),),
        )  # fmt: skip
        for index, word in enumerate(words)
    )
    observed(tmp_path, monkeypatch, frames)
    monkeypatch.setattr(
        plan, "load", lambda _path: SimpleNamespace(times=tuple(range(10)), data={})
    )
    assert absence.band_free(tmp_path / "r.json", tmp_path) is False
