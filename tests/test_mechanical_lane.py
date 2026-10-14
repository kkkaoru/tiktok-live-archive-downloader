from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from test_delivery_pipeline import configured as configured

from replay import (
    delivery_render,
    mechanical_lane,
    mechanical_plan_provenance,
    source_caption_binding,
)
from replay.core import ReplayError
from replay.delivery_render import Chunk, Clip, CutRow

MEDIA = Path("/clean.mov")


def chunk(end: str) -> Chunk:
    return Chunk(MEDIA, Fraction(0), Fraction(end))


def test_exact_and_half_frame_overrun_pass_with_exact_frames() -> None:
    clips = mechanical_lane.tolerant_clips([CutRow(0, 30)], [chunk("1")], 30)
    assert [(c.start, c.frames) for c in clips] == [(Fraction(0), 30)]
    # the real Oct6 case: the last cut ends 0.44 frame past the chunk
    end = Fraction(8029052, 1000)
    tail = mechanical_lane.tolerant_clips(
        [CutRow(240868, 240872)], [Chunk(MEDIA, Fraction(0), end)], 30
    )
    assert sum(c.frames for c in tail) == 4


def test_overrun_beyond_half_frame_and_outside_chunk_fail() -> None:
    with pytest.raises(ReplayError, match="exceeds its chunk"):
        mechanical_lane.tolerant_clips(
            [CutRow(0, 1)], [Chunk(MEDIA, Fraction(1, 100), Fraction(1))], 30
        )
    with pytest.raises(ReplayError, match="outside the source chunk index"):
        mechanical_lane.tolerant_clips([CutRow(60, 61)], [chunk("1")], 30)


def test_bind_partition_stages_adds_binding_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Path, str]] = []

    def fake(
        cleanup: source_caption_binding.Cleanup,
        *,
        consumer: Path,
        kind: str,
        output: Path,
    ) -> Path:
        seen.append((consumer, kind))
        output.write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(source_caption_binding, "create_binding", fake)
    request = tmp_path / "request-0000.json"
    request.write_text("{}", encoding="utf-8")
    stage: dict[str, object] = {"name": "partition-0000", "inputs": [str(request)]}
    cleanup = source_caption_binding.Cleanup(
        tmp_path / "cleanup.json", tmp_path / "original.mov", MEDIA, Fraction(1)
    )
    bindings = mechanical_lane.bind_partition_stages(cleanup, [stage])
    assert seen == [(request, "native-request")]
    assert stage["inputs"] == [str(request), str(bindings["partition-0000"])]
    with pytest.raises(ReplayError, match="lacks its request"):
        mechanical_lane.bind_partition_stages(cleanup, [{"name": "x", "inputs": []}])


def test_compose_fixes_patch_for_the_call_and_restore_afterwards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = delivery_render.load_config(configured(tmp_path))
    original = (delivery_render.clips, delivery_render.compose, delivery_render._workflows)
    seen: dict[str, object] = {}

    def fake_compose(config: delivery_render.Config, *, workers: int) -> list[Path]:
        seen["workers"] = workers
        request = tmp_path / "request-0000.json"
        clip = delivery_render._clip_json(Clip(tmp_path / "clean.mov", Fraction(0), 30), 30)
        request.write_text(json.dumps({"recipe": {"clips": [clip]}}))
        stage: dict[str, object] = {
            "name": "partition-0000",
            "command": [
                sys.executable,
                "-m",
                "replay.native_edit_parts",
                str(request),
                str(tmp_path / "parts"),
                "--sha256",
                mechanical_plan_provenance.pin(request).sha256,
            ],
            "inputs": [str(request)],
            "outputs": [str(tmp_path / "parts/manifest.json")],
            "timeout_seconds": 60,
        }
        return delivery_render._workflows(Path("/w"), "partition", [stage], workers)

    def fake_workflows(
        directory: Path,
        name: str,
        stages: list[dict[str, object]],
        workers: int,
        *,
        bindings: dict[str, Path] | None = None,
    ) -> list[Path]:
        seen["bindings"] = bindings
        seen["inputs"] = stages[0]["inputs"]
        return [directory / "partition-0-0000.json"]

    monkeypatch.setattr(delivery_render, "compose", fake_compose)
    monkeypatch.setattr(delivery_render, "_workflows", fake_workflows)
    monkeypatch.setattr(delivery_render, "verified_cleanup", lambda config: object())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["ffprobe"], 0, "1", ""),
    )
    monkeypatch.setattr(
        mechanical_lane,
        "bind_partition_stages",
        lambda cleanup, stages: {"partition-0000": Path("/b.json")} | _extend(stages),
    )
    patched = (delivery_render.clips, delivery_render.compose, delivery_render._workflows)
    with mechanical_lane.compose_fixes():
        assert delivery_render.clips is mechanical_lane.tolerant_clips
        delivery_render.compose(config, workers=4)
    assert seen["workers"] == 4
    assert seen["bindings"] == {"partition-0000": Path("/b.json")}
    inputs = seen["inputs"]
    assert isinstance(inputs, list)
    assert inputs[:2] == [str(tmp_path / "request-0000.json"), "/b.json"]
    assert str(mechanical_plan_provenance.receipt_path(tmp_path / "request-0000.json")) in inputs
    assert (delivery_render.clips, delivery_render.compose, delivery_render._workflows) == patched
    assert original[0] is not mechanical_lane.tolerant_clips


def _extend(stages: list[dict[str, object]]) -> dict[str, Path]:
    inputs = stages[0]["inputs"]
    assert isinstance(inputs, list)
    inputs.append("/b.json")
    return {}


def _limit(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setattr(mechanical_lane, "media_limit", lambda _media: Fraction(value))


def test_native_clip_json_keeps_an_in_range_selection_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _limit(monkeypatch, "3")
    assert mechanical_lane.native_clip_json(Clip(MEDIA, Fraction(0), 90), 30) == {
        "sourcePath": "/clean.mov",
        "selection": {"startSeconds": 0.0, "durationSeconds": 3.0, "rate": 1},
    }


def test_native_clip_json_clamps_a_sub_half_frame_overrun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _limit(monkeypatch, "2985/1000")
    assert mechanical_lane.native_clip_json(Clip(MEDIA, Fraction(0), 90), 30) == {
        "sourcePath": "/clean.mov",
        "selection": {"startSeconds": 0.0, "durationSeconds": 2.985, "rate": 1},
    }


def test_native_clip_json_rejects_an_overrun_beyond_half_a_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _limit(monkeypatch, "2980/1000")
    with pytest.raises(ReplayError, match="more than half a frame"):
        mechanical_lane.native_clip_json(Clip(MEDIA, Fraction(0), 90), 30)


def test_native_clip_json_rejects_a_clip_that_starts_at_the_source_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _limit(monkeypatch, "3")
    with pytest.raises(ReplayError, match="more than half a frame"):
        mechanical_lane.native_clip_json(Clip(MEDIA, Fraction(3), 1), 30)


def test_native_clip_json_rejects_a_zero_length_selection_inside_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _limit(monkeypatch, "3")
    with pytest.raises(ReplayError, match="no duration inside"):
        mechanical_lane.native_clip_json(Clip(MEDIA, Fraction(2), 0), 30)


def test_media_limit_floors_the_real_container_duration_to_a_millisecond(
    tmp_path: Path,
) -> None:
    movie = tmp_path / "one-second.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=30"]
        + ["-frames:v", "30", "-pix_fmt", "yuv420p", str(movie)],
        check=True,
    )
    assert mechanical_lane.media_limit(movie) == Fraction(1)


def test_media_limit_rejects_an_unreadable_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mechanical_lane.media_limit(tmp_path / "missing.mp4")


def test_compose_fixes_install_a_logging_serializer_and_restore_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _limit(monkeypatch, "2985/1000")
    before = delivery_render._clip_json
    with mechanical_lane.compose_fixes() as clamps:
        row = delivery_render._clip_json(Clip(MEDIA, Fraction(0), 90), 30)
    assert row["selection"] == {"startSeconds": 0.0, "durationSeconds": 2.985, "rate": 1}
    assert clamps == [mechanical_lane.Clamp("/clean.mov", "0", 90, "3/200")]
    assert delivery_render._clip_json is before


def test_native_clip_json_logs_every_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    _limit(monkeypatch, "2985/1000")
    log: list[mechanical_lane.Clamp] = []
    mechanical_lane.native_clip_json(Clip(MEDIA, Fraction(0), 90), 30, log)
    assert log == [mechanical_lane.Clamp("/clean.mov", "0", 90, "3/200")]


def test_native_clip_json_logs_nothing_when_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    _limit(monkeypatch, "3")
    log: list[mechanical_lane.Clamp] = []
    mechanical_lane.native_clip_json(Clip(MEDIA, Fraction(0), 90), 30, log)
    assert log == []


def test_write_clamps_records_rows_without_overwriting(tmp_path: Path) -> None:
    path = mechanical_lane.write_clamps(
        tmp_path / "clamps.json", [mechanical_lane.Clamp("/clean.mov", "0", 90, "3/200")]
    )
    assert path == tmp_path / "clamps.json"
    assert json.loads((tmp_path / "clamps.json").read_text(encoding="utf-8")) == {
        "version": 1,
        "clamps": [
            {"media": "/clean.mov", "startSeconds": "0", "frames": 90, "trimmedSeconds": "3/200"}
        ],
    }
    with pytest.raises(FileExistsError):
        mechanical_lane.write_clamps(
            tmp_path / "clamps.json", [mechanical_lane.Clamp("/clean.mov", "0", 90, "3/200")]
        )


def test_write_clamps_writes_nothing_when_no_clamp_occurred(tmp_path: Path) -> None:
    assert mechanical_lane.write_clamps(tmp_path / "clamps.json", []) is None
    assert not (tmp_path / "clamps.json").exists()


@pytest.mark.parametrize("inputs", [None, "request.json", [], [12]])
def test_invalid_partition_request_refuses_binding_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inputs: object
) -> None:
    cleanup = source_caption_binding.Cleanup(
        tmp_path / "cleanup.json", tmp_path / "original.mov", MEDIA, Fraction(1)
    )
    monkeypatch.setattr(
        source_caption_binding,
        "create_binding",
        lambda *_args, **_kwargs: pytest.fail("An invalid request must not create a binding"),
    )
    with pytest.raises(ReplayError, match="lacks its request input"):
        mechanical_lane.bind_partition_stages(cleanup, [{"name": "partition", "inputs": inputs}])


@pytest.mark.parametrize("name", ["partition", "render"])
def test_compose_fixes_preserve_explicit_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    binding = tmp_path / "existing.binding.json"
    stages: list[dict[str, object]] = [{"name": "existing", "inputs": [str(binding)]}]
    captured: list[dict[str, Path] | None] = []

    def workflows(
        directory: Path,
        name: str,
        stages: list[dict[str, object]],
        workers: int,
        *,
        bindings: dict[str, Path] | None = None,
    ) -> list[Path]:
        captured.append(bindings)
        return [directory / "preserved.json"]

    monkeypatch.setattr(delivery_render, "_workflows", workflows)
    monkeypatch.setattr(
        delivery_render,
        "verified_cleanup",
        lambda _config: pytest.fail("Explicit bindings must not be regenerated"),
    )
    with mechanical_lane.compose_fixes():
        result = delivery_render._workflows(
            tmp_path, name, stages, 1, bindings={"existing": binding}
        )
    assert result == [tmp_path / "preserved.json"]
    assert captured == [{"existing": binding}]
    assert stages == [{"name": "existing", "inputs": [str(binding)]}]


def test_nonpartition_workflow_without_bindings_does_not_require_compose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def workflows(
        directory: Path,
        name: str,
        stages: list[dict[str, object]],
        workers: int,
        *,
        bindings: dict[str, Path] | None = None,
    ) -> list[Path]:
        return [directory / f"{name}.json"]

    monkeypatch.setattr(delivery_render, "_workflows", workflows)
    with mechanical_lane.compose_fixes():
        result = delivery_render._workflows(tmp_path, "render", [], 1)
    assert result == [tmp_path / "render.json"]


def test_media_limit_refuses_non_numeric_probe_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "invalid-duration.mp4"
    source.write_bytes(b"synthetic media")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["ffprobe"], 0, "N/A", ""),
    )
    with pytest.raises(ReplayError, match="Could not read the source duration"):
        mechanical_lane.media_limit(source)
