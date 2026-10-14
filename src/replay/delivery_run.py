"""Deliver one recording end to end from a short per-recording description.

`delivery.json` names only what differs per recording: the recording folder
name, the dated original, its source-clock transcript labels, the frame cut
plan, the caption-removed render configs and (optionally) a separated voice.
`defaults.json` names the machine's tools and roots once for every recording.
Each step is skipped when its output already exists, so a rerun resumes:

proxy/clocks → foreground plan → voice separation → token projection → chunk
index → render config → foreground + caption measurement supervisors (run
together) → render pipeline → FCP timeline → transcript summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from replay import delivery_render, delivery_retention
from replay.core import ReplayError
from replay.recordings import record

CANVAS = (1080, 1920)
RATE = 30
NATIVE_SUFFIXES = frozenset({".mp4", ".mov", ".m4v"})
DELIVERY_KEYS = frozenset(
    {"version", "recording", "source", "labels", "cuts", "sourceRenderConfigs", "vocals"}
)
DEFAULT_KEYS = frozenset(
    {
        "version",
        "workRoot",
        "storageRoot",
        "deliveryRoot",
        "viewerRoot",
        "measureNative",
        "foregroundNative",
        "calibration",
        "motionTemplate",
        "motionFinish",
        "alphaPreset",
        "encodePreset",
        "finalPreset",
        "encoding",
        "separationPython",
        "separationModelHome",
        "summaryExecutable",
        "minimumTop",
        "unknownTop",
        "foregroundWorkers",
        "measureWorkers",
    }
)

Runner = Callable[[Sequence[str]], int]
Launcher = Callable[[Sequence[Sequence[str]]], list[int]]


@dataclass(frozen=True, slots=True)
class Recording:
    name: str
    source: Path
    labels: Path
    cuts: Path
    render_configs: tuple[str, ...]
    vocals: Path | None


def _json(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def _path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ReplayError(f"{field} must be an absolute path")
    return Path(value)


def load_recording(path: Path) -> Recording:
    raw = _json(path)
    if set(raw) - {"vocals"} != DELIVERY_KEYS - {"vocals"} or raw.get("version") != 1:
        raise ReplayError("delivery.json has unknown, missing or unsupported fields")
    name = raw.get("recording")
    if not isinstance(name, str) or not name.startswith(("replay-", "youtube-")):
        raise ReplayError("recording must be replay-<id> or youtube-<id>")
    patterns = raw.get("sourceRenderConfigs")
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        raise ReplayError("sourceRenderConfigs must be a list of glob patterns")
    vocals = raw.get("vocals")
    return Recording(
        name=name,
        source=_path(raw.get("source"), "source"),
        labels=_path(raw.get("labels"), "labels"),
        cuts=_path(raw.get("cuts"), "cuts"),
        render_configs=tuple(str(item) for item in patterns),
        vocals=None if vocals is None else _path(vocals, "vocals"),
    )


def load_defaults(path: Path) -> dict[str, object]:
    raw = _json(path)
    if set(raw) != DEFAULT_KEYS or raw.get("version") != 1:
        raise ReplayError("defaults.json has unknown, missing or unsupported fields")
    return raw


def _run(command: Sequence[str]) -> int:
    return subprocess.run(list(command), stdin=subprocess.DEVNULL, check=False).returncode


def _launch(commands: Sequence[Sequence[str]]) -> list[int]:
    processes = [subprocess.Popen(list(command), stdin=subprocess.DEVNULL) for command in commands]
    return [process.wait() for process in processes]


def _module(name: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", name, *(str(argument) for argument in arguments)]


def _probe_canvas(source: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height = (int(value) for value in result.stdout.strip().split(",")[:2])
    return width, height


@dataclass
class Delivery:
    recording: Recording
    defaults: dict[str, object]
    runner: Runner = _run
    launcher: Launcher = _launch
    probe: Callable[[Path], tuple[int, int]] = _probe_canvas

    def _root(self, key: str) -> Path:
        return _path(self.defaults[key], key) / self.recording.name

    @property
    def work(self) -> Path:
        return self._root("workRoot")

    @property
    def storage(self) -> Path:
        return self._root("storageRoot")

    def _step(self, output: Path, command: Sequence[str], label: str) -> Path:
        if output.exists():
            return output
        print(f"RUN {label}", flush=True)
        if self.runner(command) != 0 or not output.exists():
            raise ReplayError(f"Delivery step failed: {label}")
        return output

    def foreground_source(self) -> Path:
        source = self.recording.source
        if source.suffix.lower() in NATIVE_SUFFIXES and self.probe(source) == CANVAS:
            return source
        proxy = self.storage / "proxy-1080x1920.mov"
        return self._step(proxy, _module("replay.delivery_media", "proxy", source, proxy), "proxy")

    def prepare(self) -> Path:
        """Every input of the measurement and render phases; returns the render config."""
        self.work.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.storage.mkdir(mode=0o700, parents=True, exist_ok=True)
        video = self.foreground_source()
        clock = self._step(
            self.storage / "clock.json",
            _module("replay.delivery_media", "clock", video, self.storage / "clock.json"),
            "video clock",
        )
        foreground = self.work / "foreground"
        if not foreground.exists():
            native = self.defaults["foregroundNative"]
            self._step(
                foreground / "configs" / "0000.json",
                _module(
                    "replay.delivery_media", "foreground-plan", video, clock, native, foreground
                ),
                "foreground plan",
            )
        vocals = self.recording.vocals or self.separate()
        audio_clock = self._step(
            self.work / "audio-clock.json",
            _module(
                "replay.delivery_media",
                "audio-clock",
                self.recording.source,
                self.work / "audio-clock.json",
            ),
            "audio clock",
        )
        self._step(
            self.work / "tokens-v1.json",
            _module(
                "replay.delivery_captions",
                "project",
                self.recording.labels,
                self.recording.cuts,
                self.work / "tokens-v1.json",
            ),
            "token projection",
        )
        configs = sorted(
            {item for pattern in self.recording.render_configs for item in glob.glob(pattern)}
        )
        if not configs:
            raise ReplayError("No caption-removed render configs matched")
        chunks = self._step(
            self.work / "source-chunks.json",
            _module(
                "replay.delivery_media", "chunk-index", self.work / "source-chunks.json", *configs
            ),
            "chunk index",
        )
        return self.render_config(chunks, vocals, audio_clock)

    def separate(self) -> Path:
        directory = self.storage / "separation"
        vocals = directory / "vocals.wav"
        if vocals.exists():
            return vocals
        command = [
            "/usr/bin/env",
            f"PYTHONPATH={Path(__file__).resolve().parents[1]}",
            f"HF_HOME={self.defaults['separationModelHome']}",
            "HF_HUB_OFFLINE=1",
            str(self.defaults["separationPython"]),
            "-m",
            "replay.delivery_audio",
            str(self.recording.source),
            str(directory),
        ]
        return self._step(vocals, command, "voice separation")

    def render_config(self, chunks: Path, vocals: Path, clock: Path) -> Path:
        path = self.work / "render-config.json"
        if path.exists():
            return path
        config = {
            "version": 1,
            "recording": self.recording.name,
            "captions": str(self.work / "captions"),
            "work": str(self.work / "render"),
            "storage": str(self.storage / "render"),
            "rate": RATE,
            "width": CANVAS[0],
            "height": CANVAS[1],
            "cuts": str(self.recording.cuts),
            "sourceChunks": str(chunks),
            "motion": {
                "template": self.defaults["motionTemplate"],
                "expectedText": "Title",
                "finish": self.defaults["motionFinish"],
            },
            "alphaPreset": self.defaults["alphaPreset"],
            "encodePreset": self.defaults["encodePreset"],
            "native": self.defaults["measureNative"],
            "encoding": self.defaults["encoding"],
            "audio": {"source": str(vocals), "clock": str(clock)},
            "source": str(self.recording.source),
            "deliveryRoot": str(self._root("deliveryRoot")),
            "finalPreset": self.defaults["finalPreset"],
            "calibration": self.defaults["calibration"],
            "viewerRoot": str(self._root("viewerRoot")),
        }
        (self.work / "render").mkdir(mode=0o700, exist_ok=True)
        (self.storage / "render").mkdir(mode=0o700, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(config, stream, indent=1)
        return path

    def measure(self) -> None:
        """Observe foreground windows and measure captions concurrently."""
        foreground = self.work / "foreground"
        measure = _module(
            "replay.delivery_measure",
            self.work / "tokens-v1.json",
            self.recording.cuts,
            self.work / "captions",
            "--foreground",
            foreground,
            "--native",
            self.defaults["measureNative"],
            "--calibration",
            self.defaults["calibration"],
            "--minimum-top",
            self.defaults["minimumTop"],
            "--unknown-top",
            self.defaults["unknownTop"],
            "--workers",
            self.defaults["measureWorkers"],
        )
        observe = _module(
            "replay.delivery_foreground",
            foreground,
            "--workers",
            self.defaults["foregroundWorkers"],
        )
        codes = self.launcher([observe, measure])
        if any(code != 0 for code in codes):
            raise ReplayError(f"Measurement phase failed: {codes}")

    def finish(self, config: Path) -> None:
        if delivery_retention.is_retired(config):
            return
        if self.runner(_module("replay.delivery_pipeline", config)) != 0:
            raise ReplayError("Render pipeline failed")
        timeline = self.storage / f"{self.recording.name}.fcpxml"
        if not timeline.exists() and self.runner(
            _module("replay.delivery_fcpxml", config, timeline, "--title", self.recording.name)
        ):
            raise ReplayError("FCP timeline generation failed")
        self.summarize(config)
        self.retire(config, timeline)

    def retire(self, config: Path, timeline: Path) -> None:
        """Remove regenerable render intermediates only after the delivery is accepted."""
        command = _module("replay.delivery_retention", config, "--fcpxml", timeline, "--apply")
        if self.runner(command) != 0:
            raise ReplayError("Delivery finished but intermediate retention failed")

    def summarize(self, config: Path) -> None:
        """Generate or reuse the required summary without rerendering the video."""
        summary = _path(self.defaults.get("summaryExecutable"), "summaryExecutable")
        if not summary.is_file() or not os.access(summary, os.X_OK):
            raise ReplayError("summaryExecutable must name an existing executable file")
        render = delivery_render.load_config(config)
        srt = (
            render.viewer_root
            / "revisions"
            / "native-v1"
            / "delivery"
            / f"{delivery_render.delivery_stem(render)}.srt"
        )
        library = render.viewer_root.parent
        try:
            code = self.runner(
                [str(summary), "generate", "--library", str(library), "--transcript", str(srt)]
            )
        except OSError as error:
            raise ReplayError("Transcript summary could not be started") from error
        if code != 0:
            raise ReplayError(f"Transcript summary generation failed (exit {code})")


def run(recording: Path, defaults: Path) -> None:
    delivery = Delivery(load_recording(recording), load_defaults(defaults))
    config = delivery.prepare()
    delivery.measure()
    delivery.finish(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--defaults", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args.delivery, args.defaults)
    except (ReplayError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"DELIVERY_RUN_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
