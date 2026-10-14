"""Explicit long-line measurements while retaining the ordinary producer identity."""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from replay.caption_native_layout import MAX_CACHE, NativeCaptionLayout, NativeCaptionStyle
from replay.core import ReplayError
from replay.stages import execute_command, file_digest

MAX_SINGLE_LINE_UNITS = 512
ORDINARY_SINGLE_LINE_UNITS = 120


class LongLineCaptionLayout(NativeCaptionLayout):
    """Use the normal producer unchanged; send intact long lines to a pinned extension.

    Like the base class, an instance owns sequential requests and is not shared
    between concurrent callers. No historical source copy is executed.
    """

    def __init__(
        self,
        *,
        executable: Path,
        long_executable: Path,
        directory: Path,
        style: NativeCaptionStyle,
        width: float,
        height: float,
        evidence: tuple[Path, ...],
        expected_point_size: float | None = None,
        execute: Callable[[tuple[str, ...], Path, int], None] = execute_command,
    ) -> None:
        self.long_executable = long_executable.resolve(strict=True)
        if not self.long_executable.is_file():
            raise ReplayError("Long-line measurement requires an explicit executable file")
        self.long_inputs = {
            str(self.long_executable): file_digest(self.long_executable),
            str(Path(__file__).resolve()): file_digest(Path(__file__)),
        }
        super().__init__(
            executable=executable,
            directory=directory,
            style=style,
            width=width,
            height=height,
            evidence=evidence,
            expected_point_size=expected_point_size,
            execute=execute,
        )
        self._inputs.update(self.long_inputs)
        self._save(directory / "long-line-inputs.json", {"inputs": self.long_inputs})

    def prepare_lines(self, texts: Iterable[str]) -> None:
        """Never split or estimate a long line; each uses one bounded request."""
        pending = tuple(text for text in dict.fromkeys(texts) if text not in self._cache)
        if len(self._cache) + len(pending) > MAX_CACHE:
            raise ReplayError("Native caption measurement cache budget exceeded")
        sizes = tuple(len(text.encode("utf-16-le")) // 2 for text in pending)
        if any(
            not 1 <= size <= MAX_SINGLE_LINE_UNITS or any(c in text for c in "\r\n\0")
            for text, size in zip(pending, sizes, strict=True)
        ):
            raise ReplayError("Native caption measurement requires bounded single-line text")
        super().prepare_lines(
            text
            for text, size in zip(pending, sizes, strict=True)
            if size <= ORDINARY_SINGLE_LINE_UNITS
        )
        for text, size in zip(pending, sizes, strict=True):
            if size > ORDINARY_SINGLE_LINE_UNITS:
                self._request_long(text)

    def _request_long(self, text: str) -> None:
        ordinary = self.executable
        self.verify_inputs()
        try:
            self.executable = self.long_executable
            self._request((text,))
        finally:
            self.executable = ordinary

    def verify_long_inputs(self, inputs: Mapping[str, str]) -> None:
        """Long cached bounds require their actual extension producer and adapter pins."""
        self.verify_inputs()
        if any(inputs.get(path) != digest for path, digest in self.long_inputs.items()):
            raise ReplayError("Cached long-line measurement producer differs")
