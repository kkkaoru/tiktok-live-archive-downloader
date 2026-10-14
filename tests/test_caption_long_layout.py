"""Synthetic producer routing; no apps, fonts or private media are used."""

import json
from pathlib import Path

import pytest
from test_caption_native_layout import fake_native
from test_caption_native_layout import layout as layout

from replay import caption_long_layout
from replay.caption_long_layout import LongLineCaptionLayout
from replay.caption_measure_cache import CachedCaptionLayout
from replay.caption_native_layout import NativeCaptionLayout
from replay.core import ReplayError


@pytest.fixture
def extended(layout: NativeCaptionLayout) -> LongLineCaptionLayout:
    long_executable = layout.directory.parent / "long-native"
    long_executable.write_bytes(b"synthetic extended native producer")
    return LongLineCaptionLayout(
        executable=layout.executable,
        long_executable=long_executable,
        directory=layout.directory.parent / "long-metrics",
        style=layout.style,
        width=layout.width,
        height=layout.height,
        evidence=(layout.executable,),
        execute=fake_native,
    )


def test_routes_intact_long_lines_and_preserves_normal_identity(
    extended: LongLineCaptionLayout,
) -> None:
    calls: list[str] = []

    def execute(argv: tuple[str, ...], output: Path, timeout: int) -> None:
        calls.append(Path(argv[0]).name)
        fake_native(argv, output, timeout)

    extended._execute = execute
    extended.prepare_lines(("ab", "字" * 151, "😀" * 256, "cd"))
    assert calls == ["native", "long-native", "long-native"]
    assert extended.executable.name == "native"
    assert extended.measure("字" * 151) == (1514, 12)
    assert extended.measure("😀" * 256) == (2564, 12)
    assert len(calls) == 3
    request = json.loads((extended.directory / "batch-000001.request.json").read_text())
    assert request["texts"] == ["字" * 151]
    assert extended.verify_inputs()[str(extended.long_executable)]
    extended.verify_long_inputs(extended.verify_inputs())


@pytest.mark.parametrize("text", ["a" * 513, "😀" * 257, "", "a\nb", "a\rb", "a\0b"])
def test_invalid_single_line_is_rejected(extended: LongLineCaptionLayout, text: str) -> None:
    with pytest.raises(ReplayError, match="bounded single-line"):
        extended.prepare_lines((text,))
    assert not tuple(extended.directory.glob("*.request.json"))


def test_failure_restores_normal_executable(extended: LongLineCaptionLayout) -> None:
    def fail(argv: tuple[str, ...], output: Path, timeout: int) -> None:
        raise ReplayError("synthetic measurement failure")

    extended._execute = fail
    with pytest.raises(ReplayError, match="synthetic measurement"):
        extended.prepare_lines(("字" * 151,))
    assert extended.executable.name == "native"


def test_changed_or_unbound_extension_cannot_be_adopted(extended: LongLineCaptionLayout) -> None:
    with pytest.raises(ReplayError, match="producer differs"):
        extended.verify_long_inputs({})
    extended.long_executable.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="inputs changed"):
        extended.prepare_lines(("字" * 151,))


def test_cache_budget_is_preserved(
    extended: LongLineCaptionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(caption_long_layout, "MAX_CACHE", 1)
    with pytest.raises(ReplayError, match="cache budget"):
        extended.prepare_lines(("字" * 151, "short"))


def test_extension_must_be_a_file(layout: NativeCaptionLayout) -> None:
    with pytest.raises(ReplayError, match="executable file"):
        LongLineCaptionLayout(
            executable=layout.executable,
            long_executable=layout.directory,
            directory=layout.directory.parent / "invalid-long",
            style=layout.style,
            width=layout.width,
            height=layout.height,
            evidence=(layout.executable,),
        )


def test_cached_long_line_requires_extension_identity(
    extended: LongLineCaptionLayout, layout: NativeCaptionLayout
) -> None:
    extended.expected_point_size = 10

    def exact(argv: tuple[str, ...], output: Path, timeout: int) -> None:
        fake_native(argv, output, timeout)
        raw = json.loads(output.read_text(encoding="utf-8"))
        raw["font"]["pointSize"] = 10
        output.write_text(json.dumps(raw), encoding="utf-8")

    extended._execute = exact
    extended.measure("字" * 151)
    inputs = extended.verify_inputs()
    cached = CachedCaptionLayout(extended)
    cached.reuse(directory=extended.directory, inputs=inputs)
    assert cached.measure("字" * 151) == (1514, 12)
    del inputs[str(extended.long_executable)]
    with pytest.raises(ReplayError, match="producer differs"):
        CachedCaptionLayout(extended).reuse(directory=extended.directory, inputs=inputs)
    # A normal producer may not adopt long data merely because it has matching style.
    layout.expected_point_size = 10
    with pytest.raises(ReplayError, match="geometry is inconsistent"):
        CachedCaptionLayout(layout).reuse(directory=extended.directory, inputs=inputs)
