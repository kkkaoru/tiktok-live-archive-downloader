import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_delivery_cut_audit import cleanup_evidence
from test_delivery_render import configured, stages, write

from replay import delivery_render as render
from replay import native_join
from replay.core import ReplayError
from replay.workflow import parse_stages


@pytest.mark.parametrize("with_audio", [False, True])
def test_join_cli_uses_ordered_real_media_without_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_audio: bool
) -> None:
    plan = write(
        tmp_path / "join.json",
        [
            {"sourcePath": str(tmp_path / "second.mp4"), "durationFrames": 2},
            {"sourcePath": str(tmp_path / "first.mp4"), "durationFrames": 4},
        ],
    )
    observed: list[tuple[Path, int]] = []

    def join(
        parts: Sequence[native_join.PartFile], *, rate: int, output: Path, audio: Path | None
    ) -> int:
        assert rate == 30
        assert output == tmp_path / "joined.mp4"
        assert audio == (tmp_path / "whole.wav" if with_audio else None)
        observed.extend((part.path, part.frames) for part in parts)
        return 6

    monkeypatch.setattr(native_join, "join", join)
    arguments = ["join-files", str(plan), "--output", str(tmp_path / "joined.mp4")]
    if with_audio:
        arguments.extend(["--audio", str(tmp_path / "whole.wav")])
    assert render.main(arguments) == 0
    assert observed == [(tmp_path / "second.mp4", 2), (tmp_path / "first.mp4", 4)]
    assert not (tmp_path / "join-inputs").exists()


def test_join_cli_refuses_missing_output_and_invalid_plan(tmp_path: Path) -> None:
    plan = write(tmp_path / "join.json", [{"durationFrames": 4}])
    assert render.main(["join-files", str(plan)]) == 1
    with pytest.raises(ReplayError, match="exact media"):
        render.join_files(plan, tmp_path / "out.mp4", 30)


def test_join_audio_option_is_not_silently_ignored(tmp_path: Path) -> None:
    assert render.main(["join", str(tmp_path / "absent.json"), "--audio", "whole.wav"]) == 1


def test_real_evidence_paths_reach_new_partition_audio_and_mux(tmp_path: Path) -> None:
    path = configured(tmp_path)
    source = Path(json.loads(path.read_text(encoding="utf-8"))["source"])
    source.write_bytes(b"synthetic original")
    cleanup_evidence(path, pts=(0, 150), last_duration=150)
    clock = write(tmp_path / "original-clock.json", {"synthetic": "clock"})
    protected = write(tmp_path / "protected.json", {"synthetic": "not an approval"})
    write(path, {**json.loads(path.read_text()), "cutEvidence": [str(clock), str(protected)]})
    config = render.load_config(path)
    [partition] = render.compose(config, workers=1)
    [audio] = render.audio(config)
    [mux] = render.final_mux(config)
    partition_stage = parse_stages({"version": 1, "stages": stages(partition)})[0]
    audio_stage = parse_stages({"version": 1, "stages": stages(audio)})[0]
    mux_stage = parse_stages({"version": 1, "stages": stages(mux)})[0]
    assert partition_stage.inputs[-2:] == (clock, protected)
    assert audio_stage.inputs[-2:] == (clock, protected)
    assert mux_stage.inputs == (
        config.storage / "joined-video.mp4",
        config.work / "audio-selected/selected.wav",
        clock,
        protected,
    )
    assert protected.read_text() == '{"synthetic": "not an approval"}'


@pytest.mark.parametrize(
    "rows",
    [
        [{"source_start_frame": 10, "source_end_frame": 10}],
        [{"source_start_frame": 10, "source_end_frame": 9}],
        [
            {"source_start_frame": 10, "source_end_frame": 20},
            {"source_start_frame": 19, "source_end_frame": 21},
        ],
    ],
)
def test_invalid_cut_order_is_not_silently_dropped(tmp_path: Path, rows: object) -> None:
    with pytest.raises(ReplayError, match="positive, ordered"):
        render.read_cuts(write(tmp_path / "cuts.json", {"cuts": rows}))


def test_nonpositive_native_timebase_refused(tmp_path: Path) -> None:
    path = write(
        tmp_path / "chunks.json",
        {
            "ranges": [
                {
                    "media": "/source.mp4",
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": 1,
                    "timeBase": "-1/30",
                }
            ]
        },
    )
    with pytest.raises(ReplayError, match="positive"):
        render.read_chunks(path)
