"""Chunk probes run concurrently, return in assembly order, and the original probe is restored."""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from replay import delivery_source_assembly as assembly
from replay import delivery_source_assembly_parallel as parallel


def _probe(label: str) -> assembly.Probe:
    return assembly.Probe({"label": label}, (), ())


def _chunks(count: int) -> tuple[SimpleNamespace, ...]:
    return tuple(SimpleNamespace(media=Path(f"chunk-{index}.mp4")) for index in range(count))


def _call_assemble(monkeypatch: pytest.MonkeyPatch, seen: list[str], results: list[str]) -> None:
    def fake_assemble(manifest: Path, **_keywords: object) -> Path:
        for index in range(4):
            probed = assembly.probe(
                Path(f"chunk-{index}.mp4"),
                directory=Path("evidence"),
                name=f"input-{index:04d}",
                ffprobe=Path("ffprobe"),
            )
            results.append(str(probed.stream["label"]))
        joined = assembly.probe(
            Path("joined.mov"), directory=Path("evidence"), name="output", ffprobe=Path("ffprobe")
        )
        results.append(str(joined.stream["label"]))
        return manifest

    monkeypatch.setattr(assembly, "assemble", fake_assemble)
    monkeypatch.setattr(assembly, "load_chunks", lambda *_a, **_k: _chunks(4))
    barrier = threading.Barrier(2, timeout=10)

    def original(media: Path, *, name: str, **_keywords: object) -> assembly.Probe:
        seen.append(name)
        if name in {"input-0000", "input-0001"}:
            barrier.wait()  # only passes when two probes run at the same time
        return _probe(f"{name}:{media.name}")

    monkeypatch.setattr(assembly, "probe", original)


def _run() -> Path:
    return parallel.assemble_parallel(
        Path("manifest.json"),
        sha256="0" * 64,
        output=Path("out.mov"),
        evidence=Path("evidence"),
        ffmpeg=Path("ffmpeg"),
        ffprobe=Path("ffprobe"),
        workers=4,
    )


def test_probes_overlap_and_return_in_assembly_order(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    results: list[str] = []
    _call_assemble(monkeypatch, seen, results)
    assert _run() == Path("manifest.json")
    assert results == [
        "input-0000:chunk-0.mp4",
        "input-0001:chunk-1.mp4",
        "input-0002:chunk-2.mp4",
        "input-0003:chunk-3.mp4",
        "output:joined.mov",
    ]
    assert sorted(seen) == ["input-0000", "input-0001", "input-0002", "input-0003", "output"]


def test_the_original_probe_is_restored_after_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _call_assemble(monkeypatch, [], [])
    original = assembly.probe
    _run()
    assert assembly.probe is original


def test_the_original_probe_is_restored_when_assembly_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing(*_a: object, **_k: object) -> Path:
        raise ValueError("boom")

    monkeypatch.setattr(assembly, "assemble", failing)
    monkeypatch.setattr(assembly, "load_chunks", lambda *_a, **_k: _chunks(1))
    sentinel = assembly.probe
    with pytest.raises(ValueError, match="boom"):
        _run()
    assert assembly.probe is sentinel


def test_invalid_worker_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="worker"):
        parallel.assemble_parallel(
            Path("m"), sha256="0" * 64, output=Path("o"), evidence=Path("e"),
            ffmpeg=Path("f"), ffprobe=Path("p"), workers=0,
        )  # fmt: skip
