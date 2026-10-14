"""Partition a whole recording into bounded measured-caption jobs.

`replay.caption_measured_job` measures one bounded output interval at a time
(at most 120 output seconds, 32 foreground windows and 128 native calls). This
module turns a full output-clock token list, its frame cut plan and the
foreground windows into pinned per-job manifests plus worker workflows, and
writes the combine stage that `replay.fcp_measured` consumes. It is generic for
any recording: every boundary is derived from the cut plan and token gaps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import batched
from math import ceil, floor
from pathlib import Path

from replay import digest_store_run
from replay.caption_projection import CutSpan, project_tokens
from replay.caption_timing import CaptionPolicy, SpokenToken
from replay.core import ReplayError
from replay.recordings import record

TARGET_JOB_SECONDS = 45
MAX_JOB_SECONDS = 110
# Combines and Motion exports accept at most 120 output seconds per segment.
MAX_SEGMENT_SECONDS = 100
SEGMENT_LIMIT_SECONDS = 120
# Token clocks are floats; a shared token boundary on a frame edge is a valid split.
CLOCK_EPSILON = 1e-6
EPSILON_SECONDS = 1e-7
# A caption stays visible for the shared hold after its last token ends.
HOLD_SECONDS = CaptionPolicy().hold_seconds
MAX_JOB_WINDOWS = 32
# About one native measurement call per token; keep well inside the 128-call budget.
MAX_JOB_TOKENS = 48
MEASURE_TIMEOUT_SECONDS = 900
COMBINE_TIMEOUT_SECONDS = 1800
WORKFLOW_STAGES = 512
MAX_INPUT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Cut:
    """Retained source frames [start, end) placed at an output frame."""

    source_start: int
    source_end: int
    output_start: int

    @property
    def frames(self) -> int:
        return self.source_end - self.source_start

    @property
    def output_end(self) -> int:
        return self.output_start + self.frames


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: float
    end: float
    utterance: str

    def json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "utterance": self.utterance,
        }


@dataclass(frozen=True, slots=True)
class Window:
    """A foreground window and its exact source extent; `complete` once observed."""

    config: Path
    provenance: Path
    start: Fraction
    end: Fraction
    complete: bool = True


@dataclass(frozen=True, slots=True)
class Job:
    index: int
    cuts: tuple[Cut, ...]
    tokens: tuple[Token, ...]
    windows: tuple[Window, ...]


@dataclass(frozen=True, slots=True)
class Segment:
    """A bounded output interval with its own local clock starting at `offset`."""

    index: int
    offset: int
    frames: int
    jobs: tuple[Job, ...]


@dataclass(frozen=True, slots=True)
class Settings:
    rate: int
    native: Path
    calibration: Path
    region: tuple[int, int, int, int]
    center_x: int
    clearance: int
    minimum_top: float | None = None
    unknown_top: float | None = None


def _json(path: Path) -> object:
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ReplayError(f"Input exceeds the size limit: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _digest(path)}


def read_cuts(path: Path) -> tuple[Cut, ...]:
    """Read `cuts` rows with source_start_frame/source_end_frame/output_start_frame."""
    rows = record(_json(path)).get("cuts")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Cut plan has no cuts")
    cuts: list[Cut] = []
    cursor = 0
    for raw in rows:
        row = record(raw)
        values = (row.get("source_start_frame"), row.get("source_end_frame"))
        output = row.get("output_start_frame")
        if any(type(value) is not int for value in (*values, output)):
            raise ReplayError("Cut rows need integer frames")
        cut = Cut(int(str(values[0])), int(str(values[1])), int(str(output)))
        if cut.frames <= 0 or cut.output_start != cursor:
            raise ReplayError("Cuts must be nonempty and contiguous in output frames")
        if cuts and cut.source_start < cuts[-1].source_end:
            raise ReplayError("Cuts must be ordered in source frames")
        cuts.append(cut)
        cursor = cut.output_end
    return tuple(cuts)


def read_tokens(path: Path) -> tuple[Token, ...]:
    """Read output-clock `tokens` rows; their order and clocks are kept unchanged."""
    rows = record(_json(path)).get("tokens")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Token file has no tokens")
    tokens: list[Token] = []
    for raw in rows:
        row = record(raw)
        text, start, end, utterance = (
            row.get(key) for key in ("text", "start", "end", "utterance")
        )
        if (
            not isinstance(text, str)
            or not isinstance(utterance, str)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or isinstance(start, bool)
            or isinstance(end, bool)
            or end < start
        ):
            raise ReplayError("Invalid output-clock token")
        tokens.append(Token(text, float(start), float(end), utterance))
    return tuple(tokens)


def merge_same_frame_onsets(
    tokens: Sequence[Token], *, rate: int
) -> tuple[tuple[Token, ...], list[dict[str, object]]]:
    """Merge a token into the next one when both begin within the same output frame.

    A caption step that starts and is replaced inside one frame is never shown,
    which the measured planner rejects. The merged token keeps both texts in order
    and the union of their recognized clocks; no time is invented or moved.
    """
    merged: list[Token] = []
    record_rows: list[dict[str, object]] = []
    for current in tokens:
        previous = merged[-1] if merged else None
        if previous is not None and ceil(current.start * rate - CLOCK_EPSILON) == ceil(
            previous.start * rate - CLOCK_EPSILON
        ):
            joined = Token(
                previous.text + current.text,
                previous.start,
                max(previous.end, current.end),
                current.utterance,
            )
            record_rows.append({"merged": [previous.json(), current.json()], "into": joined.json()})
            merged[-1] = joined
        else:
            merged.append(current)
    return tuple(merged), record_rows


def limit_token_durations(
    tokens: Sequence[Token], *, maximum: float
) -> tuple[tuple[Token, ...], list[dict[str, object]]]:
    """Cap a token's display end at `maximum` seconds after its recognized onset.

    Coarse recognizer tokens can span a silent pause; the focused caption policy
    refuses those. The onset and text are unchanged; only how long the token can
    keep its caption visible is shortened, and each change is recorded.
    """
    limited: list[Token] = []
    rows: list[dict[str, object]] = []
    for token in tokens:
        if token.end - token.start > maximum:
            shortened = Token(token.text, token.start, token.start + maximum, token.utterance)
            rows.append({"clipped": token.json(), "into": shortened.json()})
            limited.append(shortened)
        else:
            limited.append(token)
    return tuple(limited), rows


def prepare_tokens(path: Path, *, rate: int) -> tuple[tuple[Token, ...], dict[str, object]]:
    """Tokens ready for measurement plus the record of every pre-pass change."""
    merged, merges = merge_same_frame_onsets(read_tokens(path), rate=rate)
    limited, clipped = limit_token_durations(
        merged, maximum=CaptionPolicy().maximum_context_seconds - CLOCK_EPSILON
    )
    return limited, {"rate": rate, "merges": merges, "clipped": clipped}


def read_window(config: Path, provenance: Path) -> Window:
    raw = record(_json(config))
    clock = record(raw.get("clock"))
    pts, base, last = clock.get("pts"), clock.get("timeBase"), clock.get("lastDuration")
    if not isinstance(pts, list) or not pts or not isinstance(base, str) or type(last) is not int:
        raise ReplayError("Foreground config lacks its exact clock")
    first, final = pts[0], pts[-1]
    if type(first) is not int or type(final) is not int:
        raise ReplayError("Foreground clock ticks must be integers")
    scale = Fraction(base)
    return Window(
        config, provenance, first * scale, (final + last) * scale, complete=provenance.is_file()
    )


def window_index(directories: Sequence[Path], extras: Sequence[tuple[Path, Path]]) -> list[Window]:
    """Collect windows from foreground plan directories plus explicit pairs, source-ordered."""
    pairs = [(config.absolute(), provenance.absolute()) for config, provenance in extras]
    for directory in (path.absolute() for path in directories):
        for config in sorted((directory / "configs").glob("[0-9][0-9][0-9][0-9].json")):
            pairs.append((config, directory / f"evidence-{config.stem}" / "provenance.json"))
    windows = sorted((read_window(*pair) for pair in pairs), key=_window_start)
    for previous, current in zip(windows, windows[1:], strict=False):
        if current.start < previous.end:
            raise ReplayError("Foreground windows overlap; choose one producer per interval")
    return windows


def _window_start(window: Window) -> Fraction:
    return window.start


def _needed(cuts: Sequence[Cut], windows: Sequence[Window], rate: int) -> tuple[Window, ...]:
    needed: list[Window] = []
    for window in windows:
        if any(
            window.start < Fraction(cut.source_end, rate)
            and Fraction(cut.source_start, rate) < window.end
            for cut in cuts
        ):
            needed.append(window)
    return tuple(needed)


def _straddles(
    tokens: Sequence[Token], starts: Sequence[float], boundary: float, *, held: bool = True
) -> bool:
    """Whether a caption would stay visible across the boundary.

    A token may not cross it, and the first later token must begin at least one
    hold after the latest earlier token ends, so separately planned captions
    never overlap; the earlier job's support covers that hold. Without `held`, a
    back-to-back boundary is allowed and the combine trims the hidden hold.
    """
    split = bisect_left(starts, boundary - EPSILON_SECONDS)
    earlier = tokens[max(split - 8, 0) : split]
    if not earlier:
        return False
    latest = max(token.end for token in earlier)
    if latest > boundary + EPSILON_SECONDS:
        return True
    later = split < len(tokens) and tokens[split].start < latest + HOLD_SECONDS - EPSILON_SECONDS
    return held and later


def _inside(starts: Sequence[float], cut: Cut, rate: int) -> range:
    """Indices of tokens whose onset lies in the cut's output interval."""
    first = bisect_left(starts, cut.output_start / rate)
    return range(first, bisect_left(starts, cut.output_end / rate))


def _free_frame(
    tokens: Sequence[Token],
    starts: Sequence[float],
    index: int,
    rate: int,
    cut: Cut,
    *,
    held: bool,
) -> int | None:
    """Earliest frame between tokens index-1 and index where a job may end."""
    frame = max(ceil(tokens[index - 1].end * rate - CLOCK_EPSILON), cut.output_start + 1)
    if frame > floor(tokens[index].start * rate + CLOCK_EPSILON) or frame >= cut.output_end:
        return None
    return None if _straddles(tokens, starts, frame / rate, held=held) else frame


def _candidates(
    target: int, indices: range, maximum_tokens: int = MAX_JOB_TOKENS
) -> list[tuple[int, bool]]:
    """Split choices: hold-safe gaps first, then back-to-back token boundaries.

    Each pass searches backward from the target, then forward within the cut.
    """
    backward = range(target, indices.start, -1)
    forward = range(target + 1, min(indices.stop, indices.start + 2 * maximum_tokens))
    order = [*backward, *forward]
    return [(index, True) for index in order] + [(index, False) for index in order]


def _split(cut: Cut, frame: int) -> tuple[Cut, Cut]:
    offset = frame - cut.output_start
    head = Cut(cut.source_start, cut.source_start + offset, cut.output_start)
    return head, Cut(cut.source_start + offset, cut.source_end, frame)


def split_cut(
    cut: Cut,
    tokens: Sequence[Token],
    starts: Sequence[float],
    rate: int,
    *,
    maximum_tokens: int = MAX_JOB_TOKENS,
) -> list[Cut]:
    """Split a retained cut at token-free frames so each piece has bounded tokens/time.

    Tokens and clocks are never moved; a cut is divided only between two tokens.
    """
    pieces: list[Cut] = []
    current = cut
    while True:
        indices = _inside(starts, current, rate)
        if len(indices) <= maximum_tokens and current.frames <= MAX_JOB_SECONDS * rate:
            pieces.append(current)
            return pieces
        target = indices.start + min(maximum_tokens, max(len(indices) // 2, 1))
        if maximum_tokens != MAX_JOB_TOKENS:
            time_limit = bisect_left(starts, current.output_start / rate + MAX_JOB_SECONDS)
            target = min(target, max(indices.start + 1, time_limit - 1))
        frame = None
        for index, held in _candidates(target, indices, maximum_tokens):
            frame = _free_frame(tokens, starts, index, rate, current, held=held)
            if (
                maximum_tokens != MAX_JOB_TOKENS
                and frame is not None
                and frame - current.output_start > MAX_JOB_SECONDS * rate
            ):
                frame = None
            if frame is not None:
                break
        if frame is None:
            raise ReplayError("No token-free split point inside a retained cut")
        head, current = _split(current, frame)
        pieces.append(head)


def _group(
    units: Sequence[Cut],
    tokens: Sequence[Token],
    starts: Sequence[float],
    *,
    rate: int,
    over: Callable[[list[Cut], int], bool],
    held: bool,
) -> list[list[Cut]]:
    """Close a group before a unit when `over` holds and no token straddles the boundary."""
    groups: list[list[Cut]] = [[]]
    count = 0
    for unit in units:
        group = groups[-1]
        added = len(_inside(starts, unit, rate))
        if (
            group
            and over([*group, unit], count + added)
            and not _straddles(tokens, starts, unit.output_start / rate, held=held)
        ):
            groups.append([unit])
            count = added
        else:
            group.append(unit)
            count += added
    return groups


def _supported(
    group: Sequence[Cut], following: Sequence[Cut], tokens: Sequence[Token], rate: int
) -> tuple[Cut, ...]:
    """Extend a job's spans into the next retained frames for its final caption hold."""
    if not tokens:
        return tuple(group)
    needed = ceil((max(token.end for token in tokens) + HOLD_SECONDS) * rate - CLOCK_EPSILON)
    spans = list(group)
    for unit in following:
        if spans[-1].output_end >= needed:
            break
        length = min(unit.frames, needed - unit.output_start)
        spans.append(Cut(unit.source_start, unit.source_start + length, unit.output_start))
    return tuple(spans)


def partition(
    cuts: Sequence[Cut],
    tokens: Sequence[Token],
    windows: Sequence[Window],
    *,
    rate: int,
    require_foreground: bool = True,
    maximum_job_tokens: int = MAX_JOB_TOKENS,
) -> list[Segment]:
    """Split the timeline into bounded segments of bounded measurement jobs."""
    if type(maximum_job_tokens) is not int or not 1 <= maximum_job_tokens <= 4096:
        raise ReplayError("Caption job token budget must be bounded")
    starts = [token.start for token in tokens]
    if starts != sorted(starts):
        raise ReplayError("Tokens must be ordered by onset")
    units = [
        piece
        for cut in cuts
        for piece in split_cut(cut, tokens, starts, rate, maximum_tokens=maximum_job_tokens)
    ]
    position = {unit.output_start: index for index, unit in enumerate(units)}

    def segment_over(group: list[Cut], _count: int) -> bool:
        return group[-1].output_end - group[0].output_start > MAX_SEGMENT_SECONDS * rate

    def job_over(group: list[Cut], count: int) -> bool:
        return (
            group[-1].output_end - group[0].output_start > TARGET_JOB_SECONDS * rate
            or count > maximum_job_tokens
            or len(_needed(group, windows, rate)) > MAX_JOB_WINDOWS - 1
        )

    segments: list[Segment] = []
    index = 0
    for number, members in enumerate(
        _group(units, tokens, starts, rate=rate, over=segment_over, held=True)
    ):
        offset = members[0].output_start
        jobs: list[Job] = []
        for group in _group(members, tokens, starts, rate=rate, over=job_over, held=False):
            start, end = group[0].output_start / rate, group[-1].output_end / rate
            selected = tuple(token for token in tokens if start <= token.start < end)
            following = units[position[group[-1].output_start] + 1 :][:4]
            spans = _supported(group, following, selected, rate)
            needed = _needed(spans, windows, rate)
            if (require_foreground and not needed) or len(needed) > MAX_JOB_WINDOWS:
                raise ReplayError(f"Job {index} lacks bounded foreground support")
            jobs.append(Job(index, spans, selected, needed))
            index += 1
        frames = max(job.cuts[-1].output_end for job in jobs) - offset
        if frames > SEGMENT_LIMIT_SECONDS * rate:
            raise ReplayError(f"Segment {number} exceeds the bounded caption interval")
        segments.append(Segment(number, offset, frames, tuple(jobs)))
    if sum(len(job.tokens) for segment in segments for job in segment.jobs) != len(tokens):
        raise ReplayError("Tokens outside the cut plan cannot be assigned to a job")
    return segments


def _rational(frames: int, rate: int) -> dict[str, int]:
    return {"value": frames, "timescale": rate}


def _write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=1)
        stream.write("\n")


def _local_token(token: Token, offset: float) -> dict[str, object]:
    return {
        "text": token.text,
        "start": max(token.start - offset, 0.0),
        "end": max(token.end - offset, 0.0),
        "utterance": token.utterance,
    }


def _manifest(job: Job, offset: int, tokens: Path, settings: Settings) -> dict[str, object]:
    """Pinned job manifest on its segment-local output clock."""
    return {
        "version": 1,
        "tokens": _pin(tokens),
        "spans": [
            {
                "sourceStart": _rational(cut.source_start, settings.rate),
                "outputStart": _rational(cut.output_start - offset, settings.rate),
                "duration": _rational(cut.frames, settings.rate),
            }
            for cut in job.cuts
        ],
        "foreground": [
            {"config": _pin(window.config), "provenance": _pin(window.provenance)}
            for window in job.windows
        ],
        "native": _pin(settings.native),
        "calibration": _pin(settings.calibration),
        "outputFrameRate": settings.rate,
        "region": list(settings.region),
        "centerX": settings.center_x,
        "clearance": settings.clearance,
    }


def _measure_stage(
    directory: Path,
    index: int,
    *,
    minimum_top: float | None = None,
    unknown_top: float | None = None,
) -> dict[str, object]:
    """Measure one job; with `minimum_top`, hold captions in the upper band."""
    manifest = directory / f"manifest-{index:04d}.json"
    tokens = directory / f"tokens-{index:04d}.json"
    output = directory / f"measurement-{index:04d}"
    runner = [
        "replay.digest_store_run",
        str(digest_store_run.default_store()),
        "replay.caption_measured_job",
    ]
    clamp: list[str] = []
    if minimum_top is not None:
        runner, clamp = ["replay.caption_clamp"], ["--minimum-top", str(minimum_top)]
        if unknown_top is not None:
            clamp += ["--unknown-top", str(unknown_top)]
    return {
        "name": f"measure-{index:04d}",
        "command": [
            sys.executable,
            "-m",
            *runner,
            str(manifest),
            str(output),
            "--manifest-sha256",
            _digest(manifest),
            *clamp,
        ],
        "inputs": [str(manifest), str(tokens)],
        "outputs": [str(output / "candidate.json")],
        "timeout_seconds": MEASURE_TIMEOUT_SECONDS,
    }


def write_jobs(
    directory: Path, segments: Sequence[Segment], settings: Settings, *, workers: int
) -> list[Path]:
    """Write segment-local tokens, manifests, a segment index and worker workflows.

    Only jobs whose foreground windows are all observed are written. Rerunning
    into the same directory (same deterministic partition) adds the jobs that
    became ready, with their workflows in a new `round-K` subdirectory.
    """
    if not 1 <= workers <= 8:
        raise ReplayError("Measurement plans need 1-8 workers")
    directory = directory.absolute()
    existing = directory.exists()
    directory.mkdir(mode=0o700, exist_ok=True)
    stages: list[dict[str, object]] = []
    index: list[dict[str, object]] = []
    for segment in segments:
        local = segment.offset / settings.rate
        for job in segment.jobs:
            manifest = directory / f"manifest-{job.index:04d}.json"
            if manifest.exists() or not all(window.complete for window in job.windows):
                continue
            tokens = directory / f"tokens-{job.index:04d}.json"
            rows = [_local_token(token, local) for token in job.tokens]
            _write_new(tokens, {"version": 1, "clock": "output", "tokens": rows})
            _write_new(manifest, _manifest(job, segment.offset, tokens, settings))
            stages.append(
                _measure_stage(
                    directory,
                    job.index,
                    minimum_top=settings.minimum_top,
                    unknown_top=settings.unknown_top,
                )
            )
        index.append(
            {
                "segment": segment.index,
                "offsetFrame": segment.offset,
                "frames": segment.frames,
                "jobs": [job.index for job in segment.jobs],
            }
        )
    plan = {"version": 1, "rate": settings.rate, "segments": index}
    if not existing:
        _write_new(directory / "segments.json", plan)
        return _measure_workflows(directory, stages, workers=workers, prefix="measure")
    if _json(directory / "segments.json") != plan:
        raise ReplayError("The partition changed; plan into a new directory")
    if not stages:
        return []
    rounds = len(list(directory.glob("round-[0-9][0-9][0-9][0-9]")))
    target = directory / f"round-{rounds:04d}"
    target.mkdir(mode=0o700)
    return _measure_workflows(target, stages, workers=workers, prefix="measure")


def _measure_workflows(
    directory: Path, stages: list[dict[str, object]], *, workers: int, prefix: str
) -> list[Path]:
    workflows: list[Path] = []
    for worker in range(workers):
        for part, chunk in enumerate(batched(stages[worker::workers], WORKFLOW_STAGES)):
            path = directory / f"{prefix}-{worker}-{part:04d}.json"
            _write_new(path, {"version": 1, "stages": list(chunk)})
            workflows.append(path)
    return workflows


def retry(
    directory: Path,
    *,
    workers: int,
    exclude: Sequence[Path] = (),
    minimum_top: float | None = None,
    unknown_top: float | None = None,
) -> list[Path]:
    """Write fresh workflows for every job without a published candidate.

    A failed attempt directory is kept under `measurement-NNNN.failed-K`; the job
    then runs again unchanged in a new state directory named by the retry round.
    Jobs named by the `exclude` workflows are still owned by live workers and skipped.
    """
    directory = directory.absolute()
    rounds = len(list(directory.glob("retry-[0-9][0-9][0-9][0-9]")))
    pending: list[int] = []
    owned = {
        str(stage.get("name"))
        for path in exclude
        for stage in (record(item) for item in _rows(record(_json(path)).get("stages")))
    }
    for manifest in sorted(directory.glob("manifest-[0-9][0-9][0-9][0-9].json")):
        index = int(manifest.stem.removeprefix("manifest-"))
        output = directory / f"measurement-{index:04d}"
        if (output / "candidate.json").is_file() or f"measure-{index:04d}" in owned:
            continue
        if output.exists():
            output.rename(output.with_name(f"{output.name}.failed-{rounds}"))
        pending.append(index)
    if not pending:
        raise ReplayError("Every measurement job already has a candidate")
    round_directory = directory / f"retry-{rounds:04d}"
    round_directory.mkdir(mode=0o700)
    stages = [
        _measure_stage(directory, index, minimum_top=minimum_top, unknown_top=unknown_top)
        for index in pending
    ]
    return _measure_workflows(round_directory, stages, workers=workers, prefix="measure")


def rewrite(directory: Path, windows: Sequence[Window], *, rate: int) -> list[int]:
    """Re-derive foreground windows of unfinished jobs from a corrected index.

    Only jobs without a candidate change; the previous manifest is preserved as
    `manifest-NNNN.json.orig-K`. Spans, tokens and other pins are unchanged.
    """
    directory = directory.absolute()
    changed: list[int] = []
    for manifest in sorted(directory.glob("manifest-[0-9][0-9][0-9][0-9].json")):
        index = int(manifest.stem.removeprefix("manifest-"))
        if (directory / f"measurement-{index:04d}" / "candidate.json").is_file():
            continue
        data = record(_json(manifest))
        spans = [_span_cut(record(span), rate) for span in _rows(data.get("spans"))]
        needed = _needed(spans, windows, rate)
        pins = [
            {"config": _pin(window.config), "provenance": _pin(window.provenance)}
            for window in needed
        ]
        if pins == data.get("foreground"):
            continue
        if not needed or len(needed) > MAX_JOB_WINDOWS:
            raise ReplayError(f"Job {index} lacks bounded foreground support")
        rounds = len(list(directory.glob(f"{manifest.name}.orig-*")))
        manifest.rename(manifest.with_name(f"{manifest.name}.orig-{rounds}"))
        _write_new(manifest, {**data, "foreground": pins})
        changed.append(index)
    return changed


def _rows(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ReplayError("Expected a JSON list")
    return value


def _span_cut(span: dict[str, object], rate: int) -> Cut:
    source, output, duration = (
        record(span.get(key)).get("value") for key in ("sourceStart", "outputStart", "duration")
    )
    if type(source) is not int or type(output) is not int or type(duration) is not int:
        raise ReplayError("Manifest spans need integer frame values")
    return Cut(source, source + duration, output)


def _combine_stage(
    directory: Path,
    row: dict[str, object],
    *,
    rate: int,
    width: int,
    height: int,
    calibration: Path,
) -> dict[str, object]:
    number, frames, jobs = row.get("segment"), row.get("frames"), row.get("jobs")
    if type(number) is not int or type(frames) is not int or not isinstance(jobs, list):
        raise ReplayError("Invalid segment index row")
    candidates = [directory / f"measurement-{job:04d}" / "candidate.json" for job in jobs]
    if not candidates or not all(path.is_file() for path in candidates):
        raise ReplayError(f"Segment {number} has unfinished measurement jobs")
    output = directory / f"combined-{number:04d}.json"
    command = [sys.executable, "-m", "replay.fcp_measured", str(output)]
    command += ["--width", str(width), "--height", str(height)]
    command += ["--rate", str(rate), "--frames", str(frames)]
    command += ["--calibration", str(calibration), _digest(calibration)]
    for candidate in candidates:
        command.extend(("--candidate", str(candidate), _digest(candidate)))
    return {
        "name": f"combine-{number:04d}",
        "command": command,
        "inputs": [str(path) for path in candidates],
        "outputs": [str(output)],
        "timeout_seconds": COMBINE_TIMEOUT_SECONDS,
    }


def write_combine(directory: Path, *, width: int, height: int, calibration: Path) -> Path:
    """Write one combine stage per segment once every measurement candidate exists."""
    directory = directory.absolute()
    data = record(_json(directory / "segments.json"))
    rows, rate = data.get("segments"), data.get("rate")
    if not isinstance(rows, list) or not rows or type(rate) is not int:
        raise ReplayError("Invalid segment index")
    stages = [
        _combine_stage(
            directory,
            record(row),
            rate=rate,
            width=width,
            height=height,
            calibration=calibration.absolute(),
        )
        for row in rows
    ]
    path = directory / "combine-workflow.json"
    _write_new(path, {"version": 1, "stages": stages})
    return path


def read_labels(path: Path) -> tuple[tuple[SpokenToken, ...], list[dict[str, object]]]:
    """Read ordered source-clock `labels`; zero-length labels are returned separately.

    A label without duration cannot be displayed, and no duration is invented.
    """
    rows = record(_json(path)).get("labels")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Transcript has no labels")
    labels: list[SpokenToken] = []
    instant: list[dict[str, object]] = []
    for raw in rows:
        row = record(raw)
        if row.get("start") == row.get("end"):
            instant.append(row)
            continue
        text, start, end = row.get("text"), row.get("start"), row.get("end")
        provenance = row.get("provenance")
        if (
            not isinstance(text, str)
            or not isinstance(provenance, str)
            or type(start) not in (int, float)
            or type(end) not in (int, float)
        ):
            raise ReplayError("Invalid source-clock label")
        labels.append(SpokenToken(text, float(str(start)), float(str(end)), provenance))
    return tuple(labels), instant


def project(labels: Path, cuts: Path, output: Path, *, rate: int) -> Path:
    """Project source-clock labels through a frame cut plan with the shared rule.

    Tokens keep their recognized clocks; omitted and boundary-conflicting tokens
    are counted and kept in the output for review rather than silently dropped.
    """
    spans = tuple(
        CutSpan(cut.source_start, cut.source_end, cut.output_start) for cut in read_cuts(cuts)
    )
    tokens, instant = read_labels(labels)
    result = project_tokens(tokens, spans, fps=rate)
    _write_new(
        output,
        {
            "version": 1,
            "clock": "output",
            "labels": {"path": str(labels.absolute()), "sha256": _digest(labels)},
            "cuts": {"path": str(cuts.absolute()), "sha256": _digest(cuts)},
            "tokens": [_spoken(token) for token in result.tokens],
            "omitted": [_spoken(token) for token in result.omitted],
            "boundaryConflicts": [_spoken(token) for token in result.boundary_conflicts],
            "zeroLengthLabels": instant,
        },
    )
    return output


def _spoken(token: SpokenToken) -> dict[str, object]:
    return {
        "text": token.text,
        "start": token.start,
        "end": token.end,
        "utterance": token.utterance,
    }


def _pairs(values: Sequence[str]) -> list[tuple[Path, Path]]:
    if len(values) % 2:
        raise ReplayError("--window needs CONFIG PROVENANCE pairs")
    return [(Path(values[i]), Path(values[i + 1])) for i in range(0, len(values), 2)]


def read_window_list(path: Path | None) -> list[tuple[Path, Path]]:
    """Read `CONFIG PROVENANCE` lines; blank lines are ignored."""
    if path is None:
        return []
    return _pairs(path.read_text(encoding="utf-8").split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Write bounded measurement jobs and workflows")
    plan.add_argument("tokens", type=Path, help="JSON with output-clock `tokens`")
    plan.add_argument("cuts", type=Path, help="JSON with frame `cuts`")
    plan.add_argument("directory", type=Path)
    plan.add_argument("--foreground", type=Path, action="append", default=[])
    plan.add_argument("--window", nargs="+", default=[], metavar="PATH")
    plan.add_argument("--window-list", type=Path)
    plan.add_argument("--native", type=Path, required=True)
    plan.add_argument("--calibration", type=Path, required=True)
    plan.add_argument("--rate", type=int, default=30)
    plan.add_argument("--region", type=int, nargs=4, default=(0, 0, 1080, 1920))
    plan.add_argument("--center-x", type=int, default=540)
    plan.add_argument("--clearance", type=int, default=24)
    plan.add_argument("--workers", type=int, default=4)
    plan.add_argument("--minimum-top", type=float)
    plan.add_argument("--unknown-top", type=float)
    projected = commands.add_parser("project", help="Project source labels through cuts")
    projected.add_argument("labels", type=Path)
    projected.add_argument("cuts", type=Path)
    projected.add_argument("output", type=Path)
    projected.add_argument("--rate", type=int, default=30)
    rewritten = commands.add_parser("rewrite", help="Re-derive windows of unfinished jobs")
    rewritten.add_argument("directory", type=Path)
    rewritten.add_argument("--foreground", type=Path, action="append", default=[])
    rewritten.add_argument("--window-list", type=Path)
    rewritten.add_argument("--rate", type=int, default=30)
    retried = commands.add_parser("retry", help="Rerun jobs without a candidate")
    retried.add_argument("directory", type=Path)
    retried.add_argument("--workers", type=int, default=4)
    retried.add_argument("--exclude", type=Path, nargs="*", default=[])
    retried.add_argument("--minimum-top", type=float)
    retried.add_argument("--unknown-top", type=float)
    combine = commands.add_parser("combine", help="Write the combine workflow")
    combine.add_argument("directory", type=Path)
    combine.add_argument("--calibration", type=Path, required=True)
    combine.add_argument("--width", type=int, default=1080)
    combine.add_argument("--height", type=int, default=1920)
    args = parser.parse_args(argv)
    try:
        if args.command == "project":
            print(project(args.labels, args.cuts, args.output, rate=args.rate))
            return 0
        if args.command == "rewrite":
            index = window_index(args.foreground, read_window_list(args.window_list))
            print(json.dumps(rewrite(args.directory, index, rate=args.rate)))
            return 0
        if args.command == "retry":
            for path in retry(
                args.directory,
                workers=args.workers,
                exclude=args.exclude,
                minimum_top=args.minimum_top,
                unknown_top=args.unknown_top,
            ):
                print(path)
            return 0
        if args.command == "combine":
            print(
                write_combine(
                    args.directory,
                    calibration=args.calibration,
                    width=args.width,
                    height=args.height,
                )
            )
            return 0
        tokens, changes = prepare_tokens(args.tokens, rate=args.rate)
        extras = _pairs(args.window) + read_window_list(args.window_list)
        windows = window_index(args.foreground, extras)
        segments = partition(read_cuts(args.cuts), tokens, windows, rate=args.rate)
        settings = Settings(
            args.rate,
            args.native.resolve(strict=True),
            args.calibration.resolve(strict=True),
            tuple(args.region),
            args.center_x,
            args.clearance,
            args.minimum_top,
            args.unknown_top,
        )
        for path in write_jobs(args.directory, segments, settings, workers=args.workers):
            print(path)
        merge_record = args.directory / "token-merges.json"
        if not merge_record.exists():
            _write_new(merge_record, changes)
    except (ReplayError, OSError, ValueError) as error:
        print(f"DELIVERY_CAPTIONS_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
