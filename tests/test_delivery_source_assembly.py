"""Synthetic exact VFR source assembly, never private media."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from replay import delivery_source_assembly as assembly
from replay.core import ReplayError
from replay.delivery_render import CutRow, clips, read_chunks
from replay.native_edit_parts import partition
from replay.recordings import record
from replay.subtitle_clock import VideoFrameClock


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": assembly._digest(path)}


def document(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "fault", ["missing-root", "missing-media", "outside", "suffix", "alias", "file-root"]
)
def test_explicit_candidate_media_rejects_invalid_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    media_root = tmp_path / "media"
    media_root.mkdir()
    producer: dict[str, object] = {
        "outputDirectory": str(checkout / "proof"),
        "mediaOutput": str(media_root / "candidate.mp4"),
        "allowedMediaRoot": str(media_root),
    }
    if fault == "missing-root":
        del producer["allowedMediaRoot"]
    elif fault == "missing-media":
        del producer["mediaOutput"]
    elif fault == "outside":
        producer["mediaOutput"] = str(tmp_path / "candidate.mp4")
    elif fault == "suffix":
        producer["mediaOutput"] = str(media_root / "candidate.mov")
    elif fault == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(media_root, target_is_directory=True)
        producer["mediaOutput"] = str(alias / "candidate.mp4")
    else:
        root_file = tmp_path / "file"
        root_file.write_bytes(b"not a directory")
        producer["allowedMediaRoot"] = str(root_file)
    with pytest.raises(ReplayError):
        assembly.candidate_media_path(producer)


def test_explicit_candidate_media_has_separate_canonical_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    media_root = tmp_path / "media"
    media_root.mkdir()
    result = assembly.candidate_media_path(
        {
            "outputDirectory": str(checkout / "proof"),
            "mediaOutput": str(media_root / "candidate.mp4"),
            "allowedMediaRoot": str(media_root),
        }
    )
    assert result == tmp_path / "media/candidate.mp4"


@pytest.fixture
def tools() -> tuple[Path, Path]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    return Path(ffmpeg).resolve(), Path(ffprobe).resolve()


def make_chunk(
    root: Path,
    start: int,
    ffmpeg: Path,
    *,
    codec: str,
    timescale: int = 1000,
    last_duration: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    root.mkdir()
    assert timescale % 1000 == 0
    factor = timescale // 1000
    final = 33 * factor if last_duration is None else last_duration
    codec_options = (
        ["-x265-params", "log-level=error:pools=1:frame-threads=1", "-tag:v", "hvc1"]
        if codec == "libx265"
        else []
    )
    media = root / "candidate.mp4"
    subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=32x32:rate={timescale}",
            "-frames:v",
            "4",
            "-vf",
            f"settb=1/{timescale},setpts='if(eq(N,0),0,"
            f"if(eq(N,1),{33 * factor},if(eq(N,2),{66 * factor},{95 * factor})))'",
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            f"1:{timescale}",
            "-c:v",
            codec,
            *codec_options,
            "-bf",
            "0",
            "-video_track_timescale",
            str(timescale),
            "-bsf:v",
            f"setts=duration='if(eq(N,3),{final},DURATION)'",
            str(media),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=20,
    )
    producer = document(
        root / "producer.json",
        {
            "source": str(root.parent / "original.mp4"),
            "sourceSHA256": "a" * 64,
            "outputDirectory": str(root),
            "clock": {
                "pts": [start, start + 33 * factor, start + 66 * factor, start + 95 * factor],
                "timeBase": f"1/{timescale}",
                "lastDuration": final,
            },
        },
    )
    receipt = document(
        root / "candidate.json",
        {
            "inputs": {
                str(producer): assembly._digest(producer),
                str(root.parent / "original.mp4"): "a" * 64,
            },
            "videoSHA256": assembly._digest(media),
        },
    )
    return (
        {
            "media": str(media),
            "sourceFirstPTS": start,
            "sourceEndPTS": start + 95 * factor + final,
            "frames": 4,
            "timeBase": f"1/{timescale}",
        },
        {"producer": pin(producer), "candidate": pin(receipt)},
    )


@pytest.fixture
def manifest(tmp_path: Path, tools: tuple[Path, Path], request: pytest.FixtureRequest) -> Path:
    codec: object = getattr(request, "param", "libx264")
    assert isinstance(codec, str)
    (tmp_path / "original.mp4").symlink_to(tmp_path / "missing-original.mp4")
    first, first_ref = make_chunk(tmp_path / "first", 1000, tools[0], codec=codec)
    second, second_ref = make_chunk(tmp_path / "second", 1128, tools[0], codec=codec)
    index = document(tmp_path / "index.json", {"ranges": [first, second]})
    return document(
        tmp_path / "manifest.json",
        {"version": 1, "index": pin(index), "chunks": [first_ref, second_ref]},
    )


@pytest.mark.parametrize("inexact_first", [False, True])
def test_nonmicrosecond_terminal_extent(
    tmp_path: Path, tools: tuple[Path, Path], inexact_first: bool
) -> None:
    first, first_ref = make_chunk(
        tmp_path / "first",
        0,
        tools[0],
        codec="libx264",
        timescale=3000,
        last_duration=100 if inexact_first else 99,
    )
    second, second_ref = make_chunk(
        tmp_path / "second",
        385 if inexact_first else 384,
        tools[0],
        codec="libx264",
        timescale=3000,
        last_duration=100,
    )
    index = document(tmp_path / "index.json", {"ranges": [first, second]})
    manifest = document(
        tmp_path / "manifest.json",
        {
            "version": 1,
            "index": pin(index),
            "chunks": [first_ref, second_ref],
        },
    )
    if inexact_first:
        with pytest.raises(ReplayError, match="exact ffconcat serialization"):
            assembly.assemble(
                manifest,
                sha256=assembly._digest(manifest),
                output=tmp_path / "joined.mov",
                evidence=tmp_path / "evidence",
                ffmpeg=tools[0],
                ffprobe=tools[1],
            )
        assert not (tmp_path / "joined.mov").exists()
        return
    receipt = assembly.assemble(
        manifest,
        sha256=assembly._digest(manifest),
        output=tmp_path / "joined.mov",
        evidence=tmp_path / "evidence",
        ffmpeg=tools[0],
        ffprobe=tools[1],
    )
    assert assembly._json(receipt)["compressedPacketsIdentical"] is True
    assert assembly._json(receipt)["frames"] == 8
    result = assembly._json(tmp_path / "evidence/single-source-index.json")
    assert record(assembly._array(result["ranges"], 1)[0])["sourceEndPTS"] == 769
    concat = (tmp_path / "evidence/sources.ffconcat").read_text(encoding="utf-8")
    assert concat.count("duration ") == 1
    assert "duration 0.128000\n" in concat


@pytest.mark.parametrize("manifest", ["libx264", "libx265"], indirect=True)
def test_vfr_seam_lossless_integration(manifest: Path, tools: tuple[Path, Path]) -> None:
    root = manifest.parent
    receipt = assembly.assemble(
        manifest,
        sha256=assembly._digest(manifest),
        output=root / "joined.mov",
        evidence=root / "evidence",
        ffmpeg=tools[0],
        ffprobe=tools[1],
    )
    raw = assembly._json(receipt)
    assert raw["frames"] == 8
    assert raw["compressedPacketsIdentical"] is True
    assert raw["qualityApproved"] is False
    assert raw["originalSourceIndependentlyDecoded"] is False
    index = assembly._json(root / "evidence/single-source-index.json")
    ranges = assembly._array(index["ranges"], 1)
    assert record(ranges[0])["sourceFirstPTS"] == 1000
    assert record(ranges[0])["sourceEndPTS"] == 1256
    cuts = [CutRow(33, 34)]
    with pytest.raises(ReplayError, match="crosses"):
        clips(cuts, read_chunks(root / "index.json"), 30)
    selected = clips(cuts, read_chunks(root / "evidence/single-source-index.json"), 30)
    assert len(selected) == 1
    assert selected[0].frames == 1
    assert selected[0].start == Fraction(1, 10)
    assert cuts == [CutRow(33, 34)]
    layer = {
        "sourcePath": "/synthetic-overlay.mov",
        "selection": {"startSeconds": 0, "durationSeconds": 1 / 30, "rate": 1},
        "offsetSeconds": 0,
        "opacity": 1,
    }
    request = {
        "outputDirectory": str(root / "render-not-executed"),
        "outputName": "candidate.mp4",
        "recipe": {
            "video": {"frameRate": 30, "encoding": {}},
            "clips": [
                {
                    "sourcePath": str(selected[0].media),
                    "selection": {
                        "startSeconds": float(selected[0].start),
                        "durationSeconds": selected[0].frames / 30,
                        "rate": 1,
                    },
                }
            ],
            "additionalVideo": [layer],
        },
    }
    parts = partition(request)
    assert parts[0][:2] == (0, 1)
    assert record(parts[0][2]["recipe"])["additionalVideo"] == [
        {
            "sourcePath": "/synthetic-overlay.mov",
            "selection": {"startSeconds": 0, "durationSeconds": 1 / 30, "rate": 1},
            "offsetSeconds": 0,
            "opacity": 1,
        }
    ]
    # Same exact 30fps selection: the original left chunk cannot support it.
    assert Fraction(1100, 1000) < Fraction(1128, 1000) < Fraction(34, 30)
    assert Fraction(1000, 1000) <= Fraction(33, 30) < Fraction(34, 30) <= Fraction(1256, 1000)
    assert (
        assembly.main(
            [
                str(manifest),
                str(root / "joined.mov"),
                str(root / "other"),
                "--sha256",
                assembly._digest(manifest),
                "--ffmpeg",
                str(tools[0]),
                "--ffprobe",
                str(tools[1]),
            ]
        )
        == 1
    )


@pytest.mark.parametrize("duration", [1, 2])
def test_inexact_duration_rejected(duration: int) -> None:
    with pytest.raises(ReplayError, match="serialization"):
        assembly.duration_text(VideoFrameClock((0,), Fraction(1, 30000), duration))


@pytest.mark.parametrize(
    "value",
    [None, 1, "", "relative", "/a\x00b", "/a\nb", "/a\rb", "/a/../b", "/a/./b", "/" * 4097],
)
def test_invalid_historical_source(value: object) -> None:
    with pytest.raises(ReplayError, match="historical source reference"):
        assembly._historical_source(value)


@pytest.mark.parametrize("linked", [False, True])
def test_historical_source_identity_only(tmp_path: Path, linked: bool) -> None:
    historical = tmp_path / "historical"
    if linked:
        historical.symlink_to(tmp_path / "missing-originals", target_is_directory=True)
    source = f"{historical}//original.mkv"
    media = tmp_path / "candidate.mp4"
    media.write_bytes(b"synthetic candidate bytes; not decoded in this metadata test")
    producer = document(
        tmp_path / "producer.json",
        {
            "source": source,
            "sourceSHA256": "a" * 64,
            "outputDirectory": str(tmp_path),
            "clock": {"pts": [0], "timeBase": "1/1000", "lastDuration": 33},
        },
    )
    receipt = document(
        tmp_path / "candidate.json",
        {
            "inputs": {str(producer): assembly._digest(producer), source: "a" * 64},
            "videoSHA256": assembly._digest(media),
        },
    )
    bindings: dict[str, str] = {}
    chunk = assembly._chunk(
        {
            "media": str(media),
            "sourceFirstPTS": 0,
            "sourceEndPTS": 33,
            "frames": 1,
            "timeBase": "1/1000",
        },
        {"producer": pin(producer), "candidate": pin(receipt)},
        bindings,
    )
    assert chunk.source == (source, "a" * 64)
    assert source not in bindings
    assert len(bindings) == 3


@pytest.mark.parametrize("target", ["candidate.json", "producer.json", "candidate.mp4"])
def test_chunk_io_symlinks_rejected(manifest: Path, target: str) -> None:
    path = manifest.parent / "first" / target
    actual = path.with_suffix(".actual")
    path.rename(actual)
    path.symlink_to(actual)
    with pytest.raises(ReplayError, match="unlinked"):
        assembly.load_chunks(manifest, sha256=assembly._digest(manifest), bindings={})


@pytest.mark.parametrize("value", [None, "relative", "/absolute\npath"])
def test_invalid_path(value: object) -> None:
    with pytest.raises(ReplayError):
        assembly._path(value)


def test_external_evidence_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="checkout-local"):
        assembly._path("/", local=True)
    (tmp_path / "link").symlink_to(tmp_path / "target")
    with pytest.raises(ReplayError, match="unlinked"):
        assembly._path(str(tmp_path / "link"))


@pytest.mark.parametrize("value", [None, "abc", 12])
def test_invalid_sha(value: object) -> None:
    with pytest.raises(ReplayError, match="SHA256"):
        assembly._sha(value)


@pytest.mark.parametrize("value", [False, 1.0, "1"])
def test_invalid_integer(value: object) -> None:
    with pytest.raises(ReplayError, match="integer"):
        assembly._integer(value)


@pytest.mark.parametrize("value", [None, [], [1, 2]])
def test_invalid_array(value: object) -> None:
    with pytest.raises(ReplayError, match="array"):
        assembly._array(value, 1)


def test_changed_pin_and_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = document(tmp_path / "x.json", {"version": 1})
    with pytest.raises(ReplayError, match="Changed"):
        assembly._bind(path, "a" * 64, {})
    with pytest.raises(ReplayError, match="Missing"):
        assembly._digest(tmp_path / "absent")
    monkeypatch.setattr(assembly, "MAX_JSON", 1)
    with pytest.raises(ReplayError, match="size"):
        assembly._json(path)


def test_clock_and_packet_mismatch() -> None:
    chunk = assembly.Chunk(
        Path("/unused"), VideoFrameClock((10, 43), Fraction(1, 1000), 29), ("/original", "a" * 64)
    )
    observed = assembly.Probe(
        {"time_base": "1/1000"}, ((0, 33), (33, 29)), ((0, 33, "a"), (33, 29, "b"))
    )
    assembly.verify_chunk(chunk, observed)
    with pytest.raises(ReplayError, match="time base"):
        assembly.verify_chunk(chunk, replace(observed, stream={"time_base": "1/90000"}))
    with pytest.raises(ReplayError, match="decoded"):
        assembly.verify_chunk(chunk, replace(observed, frames=((0, 33),)))
    with pytest.raises(ReplayError, match="packet"):
        assembly.verify_chunk(chunk, replace(observed, packets=((0, 32, "a"), (33, 29, "b"))))


@pytest.mark.parametrize(
    "change",
    [
        "gap",
        "clock",
        "media",
        "source",
        "source-path",
        "source-spelling",
        "origin-hash",
        "origin-path",
        "receipt",
        "base",
        "refs",
        "version",
        "limit",
        "pin",
    ],
)
def test_manifest_rejects(manifest: Path, change: str, monkeypatch: pytest.MonkeyPatch) -> None:
    root = manifest.parent
    raw = assembly._json(manifest)
    index = assembly._json(root / "index.json")
    rows = assembly._array(index["ranges"], 2)
    second = record(rows[1])
    producer = assembly._json(root / "second/producer.json")
    receipt = assembly._json(root / "second/candidate.json")
    if change == "gap":
        clock = record(producer["clock"])
        clock["pts"] = [1129, 1162, 1195, 1224]
        producer["clock"] = clock
        second.update(sourceFirstPTS=1129, sourceEndPTS=1257)
    elif change == "clock":
        second["frames"] = 3
    elif change == "media":
        second["media"] = str(root / "first/candidate.mp4")
    elif change == "source":
        producer["sourceSHA256"] = "b" * 64
    elif change == "source-path":
        producer["source"] = str(root / "other-original.mp4")
    elif change == "source-spelling":
        producer["source"] = f"{root}//original.mp4"
    elif change in {"origin-hash", "origin-path"}:
        if change == "origin-hash":
            producer["sourceSHA256"] = "b" * 64
        else:
            producer["source"] = str(root / "other-original.mp4")
        inputs = record(receipt["inputs"])
        inputs[str(producer["source"])] = producer["sourceSHA256"]
        receipt["inputs"] = inputs
    elif change == "receipt":
        receipt["inputs"] = {}
    elif change == "base":
        clock = record(producer["clock"])
        clock["timeBase"] = None
        producer["clock"] = clock
    elif change == "limit":
        monkeypatch.setattr(assembly, "MAX_FRAMES", 1)
    elif change == "version":
        raw["version"] = 2
    elif change == "pin":
        receipt["videoSHA256"] = "b" * 64
    producer_path = document(root / "second/producer.json", producer)
    inputs = record(receipt["inputs"])
    if change != "receipt":
        inputs[str(producer_path)] = assembly._digest(producer_path)
        receipt["inputs"] = inputs
    receipt_path = document(root / "second/candidate.json", receipt)
    raw["chunks"] = [
        assembly._array(raw["chunks"], 2)[0],
        {"producer": pin(producer_path), "candidate": pin(receipt_path)},
    ]
    if change == "refs":
        raw["chunks"] = [assembly._array(raw["chunks"], 2)[0]]
    rows[1] = second
    raw["index"] = pin(document(root / "index.json", {"ranges": rows}))
    document(manifest, raw)
    with pytest.raises(ReplayError):
        assembly.load_chunks(manifest, sha256=assembly._digest(manifest), bindings={})


def test_process_failure_and_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ReplayError, match="failed"):
        assembly._run(["/bin/sh", "-c", "exit 2"], tmp_path / "failed.log")
    monkeypatch.setattr(assembly, "PROCESS_SECONDS", -1)
    with pytest.raises(ReplayError, match="budget"):
        assembly._run(["/bin/sleep", "1"], tmp_path / "timeout.log")
    assert (tmp_path / "timeout.command.json").is_file()


@pytest.mark.parametrize("failure", ["codec", "output-clock", "changed-output", "changed-input"])
def test_failed_assembly_keeps_evidence(
    manifest: Path, tools: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    original_probe = assembly.probe

    def altered(
        media: Path, *, directory: Path, name: str, ffprobe: Path, deadline: float | None = None
    ) -> assembly.Probe:
        measured = original_probe(
            media, directory=directory, name=name, ffprobe=ffprobe, deadline=deadline
        )
        if failure == "codec" and name == "input-0001":
            return replace(measured, stream={**measured.stream, "width": 64})
        if name == "output":
            if failure == "output-clock":
                return replace(measured, frames=measured.frames[:-1])
            target = (
                media if failure == "changed-output" else manifest.parent / "first/candidate.mp4"
            )
            with target.open("ab") as handle:
                handle.write(b"changed")
        return measured

    monkeypatch.setattr(assembly, "probe", altered)
    with pytest.raises(ReplayError):
        assembly.assemble(
            manifest,
            sha256=assembly._digest(manifest),
            output=manifest.parent / "joined.mov",
            evidence=manifest.parent / "evidence",
            ffmpeg=tools[0],
            ffprobe=tools[1],
        )
    assert not (manifest.parent / "joined.mov").exists()
    assert not (manifest.parent / "evidence/receipt.json").exists()
    assert (manifest.parent / "evidence/inputs.json").is_file()
    assert (manifest.parent / "joined.mov.assembly").is_dir()


def test_existing_evidence(manifest: Path, tools: tuple[Path, Path]) -> None:
    with pytest.raises(FileExistsError):
        assembly.assemble(
            manifest,
            sha256=assembly._digest(manifest),
            output=manifest.parent / "joined.mov",
            evidence=manifest.parent,
            ffmpeg=tools[0],
            ffprobe=tools[1],
        )


def test_truncated_media(manifest: Path, tools: tuple[Path, Path]) -> None:
    path = manifest.parent / "first/candidate.mp4"
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(ReplayError, match="Changed"):
        assembly.assemble(
            manifest,
            sha256=assembly._digest(manifest),
            output=manifest.parent / "joined.mov",
            evidence=manifest.parent / "evidence",
            ffmpeg=tools[0],
            ffprobe=tools[1],
        )
