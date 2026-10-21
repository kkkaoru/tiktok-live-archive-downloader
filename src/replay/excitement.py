"""Estimate per-video excitement ("盛り上がり") from speech and, for YouTube, live chat.

The result is written beside the transcript as ``<transcript>.excitement.json`` and is
only drawn by the web viewer and TranscriptViewer, so the scoring exists once, here.

Every score is relative to the same video: each raw measure (recognized characters,
speech loudness, chat message count, chat characters) becomes a percentile rank over
the video's bins, the ranks of one source are averaged, lightly smoothed and scaled so
the video's own peak is 1. Raw per-bin values are kept beside the scores as supporting
detail. A measure that is constant over the video carries no relative signal and
scores 0. Chat replay offsets are on the YouTube original clock, so chat is only used
when the transcript is on that clock too, or when the edit's cut filter (the kept
``between(t,start,end)`` source ranges of its ffmpeg ``select``) maps it onto the edited
clock; that filter is only trusted when the edit's own SRT is byte-identical to the
transcript. A comment posted inside a cut part lands on the cut point, so it still counts.
Only per-bin counts leave this module; commenter names and message text never do.

Nothing is recomputed per video version: the decoded loudness is cached by the movie's
content key and the chat replay by YouTube id (``edits/excitement-cache``), so a new
edit, a re-delivered transcript or a moved/offloaded copy reuses them, and the sidecar
is only rewritten when its inputs change. The viewers never compute; they only read it.
"""

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from replay.core import ReplayError
from replay.recordings import is_record, record

VERSION: int = 1
DEFAULT_CACHE: Path = Path("edits/excitement-cache")
KEY_SAMPLE_BYTES: int = 1 << 20
TARGET_BINS: int = 240
MIN_BIN_SECONDS: int = 10
MAX_BIN_SECONDS: int = 60
MERGE_GAP_SECONDS: float = 1.5
PCM_RATE: int = 8000
SILENCE_DB: float = -96.0
# Quieter seconds are pauses: they never pull a mean level (or a BGM under it) down.
AUDIBLE_DB: float = -50.0
PEAK_COUNT: int = 5
PEAK_SEPARATION_BINS: int = 5
SMOOTHING: tuple[float, float, float] = (0.25, 0.5, 0.25)
# Ranks are uniform by construction; squaring keeps the order but lets the top bins stand out.
CONTRAST: float = 2.0
CSV_HEADER: list[str] = ["start_seconds", "end_seconds", "start", "end", "text"]
SRT_TIME: re.Pattern[str] = re.compile(
    r"(\d+):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2}),(\d{3})"
)
SELECT_EXPRESSION: re.Pattern[str] = re.compile(r"select='([^']*)'")
KEEP_SPAN: re.Pattern[str] = re.compile(r"between\(t,([0-9.]+),([0-9.]+)\)")
SPEAKER: re.Pattern[str] = re.compile(r"\[[A-Za-z0-9_-]{1,40}\]\s*")
YOUTUBE_ID: re.Pattern[str] = re.compile(r"[A-Za-z0-9_-]{11}")
# Viewer-authored chat items; membership, gift and system notices are not comments.
COMMENT_RENDERERS: tuple[str, ...] = ("liveChatTextMessageRenderer", "liveChatPaidMessageRenderer")


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Comment:
    offset: float
    characters: int


@dataclass(frozen=True)
class Sources:
    """Inputs of one estimate; ``cut_filter`` maps source-clock chat onto an edit."""

    transcript: Path
    duration: float
    movie: Path | None = None
    chat: Path | None = None
    cut_filter: Path | None = None


@dataclass(frozen=True)
class CatalogOptions:
    cache: Path
    fetch: bool = False
    edits: Path | None = None


def sidecar(transcript: Path, suffix: str) -> Path:
    return transcript.with_name(f"{transcript.name}.{suffix}")


def _srt_utterances(text: str) -> list[Utterance]:
    """Timed cues, accepting zero-length cues as the viewers do."""
    cues: list[Utterance] = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        rows = block.splitlines()
        match = SRT_TIME.match(rows[1]) if len(rows) >= 3 else None
        if match is None:
            raise ReplayError("Malformed SRT cue")
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        cues.append(Utterance(start, max(start, end), "\n".join(rows[2:]).strip()))
    return cues


def _csv_utterances(text: str) -> list[Utterance]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or rows[0] != CSV_HEADER:
        raise ReplayError("Transcript CSV header mismatch")
    return [Utterance(float(row[0]), float(row[1]), row[4]) for row in rows[1:] if row]


def read_utterances(transcript: Path) -> list[Utterance]:
    """Merged utterances, using the viewers' rule that collapses cumulative captions."""
    text = transcript.read_text(encoding="utf-8")
    parse = _csv_utterances if transcript.suffix.lower() == ".csv" else _srt_utterances
    cues = parse(text)
    merged: list[Utterance] = []
    for cue in cues:
        spoken = Utterance(cue.start, cue.end, SPEAKER.sub("", cue.text, count=1))
        last = merged[-1] if merged else None
        if (
            last is not None
            and last.text
            and spoken.start <= last.end + MERGE_GAP_SECONDS
            and spoken.text.startswith(last.text)
        ):
            merged[-1] = Utterance(last.start, max(last.end, spoken.end), spoken.text)
        else:
            merged.append(spoken)
    return merged


def bin_seconds(duration: float) -> int:
    """About TARGET_BINS bins per video, bounded so short and long videos stay readable."""
    return min(MAX_BIN_SECONDS, max(MIN_BIN_SECONDS, round(duration / TARGET_BINS)))


def speech_characters(utterances: Sequence[Utterance], *, width: int, bins: int) -> list[float]:
    """Recognized characters per bin, spread over each utterance's time span."""
    totals = [0.0] * bins
    for utterance in utterances:
        characters = len("".join(utterance.text.split()))
        span = utterance.end - utterance.start
        first = min(int(utterance.start // width), bins - 1)
        if span <= 0:
            totals[first] += characters
            continue
        for index in range(first, min(int(utterance.end // width), bins - 1) + 1):
            overlap = min(utterance.end, (index + 1) * width) - max(utterance.start, index * width)
            totals[index] += characters * max(overlap, 0.0) / span
    return totals


def _pcm_seconds(movie: Path) -> Iterator[float]:
    """Per-second RMS level (dBFS) of the movie's audio, decoded in one-second chunks."""
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(movie), "-vn", "-ac", "1"]
    command += ["-ar", str(PCM_RATE), "-f", "s16le", "-"]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        if process.stdout is None:
            raise ReplayError("ffmpeg produced no audio stream")
        while chunk := process.stdout.read(PCM_RATE * 2):
            samples = np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype="<i2").astype(np.float64)
            rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
            yield 20 * math.log10(rms / 32768) if rms > 0 else SILENCE_DB
    if process.returncode != 0:
        raise ReplayError(f"ffmpeg failed to decode audio ({process.returncode})")


def mean_level(levels: Sequence[float]) -> float | None:
    """Power mean (dBFS) of the audible seconds; None when every second is a pause."""
    audible = [10 ** (level / 10) for level in levels if level > AUDIBLE_DB]
    return 10 * math.log10(sum(audible) / len(audible)) if audible else None


def speech_loudness(
    levels: Sequence[float], utterances: Sequence[Utterance], *, width: int, bins: int
) -> list[float | None]:
    """Mean level of the seconds inside utterances, per bin; None for bins without speech."""
    spoken = [False] * len(levels)
    for utterance in utterances:
        for second in range(int(utterance.start), min(math.ceil(utterance.end), len(levels))):
            spoken[second] = True
    sums, counts = [0.0] * bins, [0] * bins
    for second, level in enumerate(levels):
        index = min(second // width, bins - 1)
        if spoken[second]:
            sums[index] += level
            counts[index] += 1
    return [total / count if count else None for total, count in zip(sums, counts, strict=True)]


def _runs_characters(message: object) -> int:
    runs = record(message).get("runs") if is_record(message) else None
    if not isinstance(runs, list):
        return 0
    # An emoji run counts as one character, like a typed symbol.
    return sum(len(str(run["text"])) if "text" in run else 1 for run in runs if is_record(run))


def read_chat(path: Path, *, duration: float) -> list[Comment]:
    """Viewer comments of a yt-dlp ``live_chat.json`` replay (one JSON object per line)."""
    comments: list[Comment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        action = record(json.loads(line)).get("replayChatItemAction")
        if not is_record(action):
            continue
        offset = float(str(action.get("videoOffsetTimeMsec", "-1"))) / 1000
        actions = action.get("actions")
        if not 0 <= offset <= duration or not isinstance(actions, list):
            continue
        for item in actions:
            added = record(item).get("addChatItemAction") if is_record(item) else None
            body = record(added).get("item") if is_record(added) else None
            for renderer in COMMENT_RENDERERS:
                if is_record(body) and is_record(body.get(renderer)):
                    message = record(body[renderer]).get("message")
                    comments.append(Comment(offset, _runs_characters(message)))
    return comments


def read_keep_spans(cut_filter: Path) -> list[tuple[float, float]]:
    """Kept source ranges of an edit's ffmpeg select filter, in output order."""
    expression = SELECT_EXPRESSION.search(cut_filter.read_text(encoding="utf-8"))
    spans = [
        (float(start), float(end))
        for start, end in KEEP_SPAN.findall(expression.group(1) if expression else "")
    ]
    ordered = all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))
    if not spans or not ordered or any(start >= end for start, end in spans):
        raise ReplayError("Cut filter has no ordered kept ranges")
    return spans


def to_edited_clock(
    comments: Sequence[Comment], spans: Sequence[tuple[float, float]]
) -> list[Comment]:
    """Move source-clock comments onto the edited clock; cut parts collapse to the cut point."""
    starts = [start for start, _ in spans]
    before = [0.0]
    for start, end in spans:
        before.append(before[-1] + end - start)
    moved: list[Comment] = []
    for comment in comments:
        index = bisect.bisect_right(starts, comment.offset) - 1
        if index < 0:
            moved.append(Comment(0.0, comment.characters))
            continue
        start, end = spans[index]
        inside = min(comment.offset, end) - start
        moved.append(Comment(before[index] + inside, comment.characters))
    return moved


def ranks(values: Sequence[float | None]) -> list[float]:
    """Percentile rank in [0, 1] with ties averaged; None ranks lowest; constant -> all 0."""
    keyed = [(-math.inf if value is None else value) for value in values]
    if len(set(keyed)) <= 1:
        return [0.0] * len(keyed)
    order = sorted(range(len(keyed)), key=keyed.__getitem__)
    out = [0.0] * len(keyed)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and keyed[order[stop + 1]] == keyed[order[start]]:
            stop += 1
        for position in order[start : stop + 1]:
            out[position] = (start + stop) / 2 / (len(keyed) - 1)
        start = stop + 1
    return out


def relative_score(*measures: Sequence[float | None]) -> list[float]:
    """Mean of the measures' ranks, contrasted, smoothed and scaled so the peak is 1."""
    ranked = [ranks(measure) for measure in measures]
    mean = [(sum(column) / len(column)) ** CONTRAST for column in zip(*ranked, strict=True)]
    left, middle, right = SMOOTHING
    smooth = [
        left * mean[max(index - 1, 0)]
        + middle * value
        + right * mean[min(index + 1, len(mean) - 1)]
        for index, value in enumerate(mean)
    ]
    peak = max(smooth, default=0.0)
    return [round(value / peak, 4) if peak > 0 else 0.0 for value in smooth]


def peaks(score: Sequence[float], *, width: int) -> list[float]:
    """Start seconds of the strongest bins, at least PEAK_SEPARATION_BINS apart."""
    chosen: list[int] = []
    for index in sorted(range(len(score)), key=lambda i: (-score[i], i)):
        if score[index] <= 0 or len(chosen) == PEAK_COUNT:
            break
        if all(abs(index - other) >= PEAK_SEPARATION_BINS for other in chosen):
            chosen.append(index)
    return [float(index * width) for index in sorted(chosen)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def movie_key(movie: Path) -> str:
    """Content key from size plus the first and last MiB: stable across copies and moves."""
    size = movie.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with movie.open("rb") as stream:
        digest.update(stream.read(KEY_SAMPLE_BYTES))
        stream.seek(max(size - KEY_SAMPLE_BYTES, 0))
        digest.update(stream.read(KEY_SAMPLE_BYTES))
    return digest.hexdigest()[:32]


def cached_levels(movie: Path, *, cache: Path) -> list[float]:
    """Per-second levels of a movie, decoded once per movie content."""
    path = cache / f"levels-{movie_key(movie)}.json"
    if path.is_file():
        levels = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(levels, list) and all(isinstance(v, int | float) for v in levels):
            return [float(v) for v in levels]
    levels = [round(level, 2) for level in _pcm_seconds(movie)]
    cache.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, json.dumps(levels))
    return levels


def fingerprint(sources: Sources) -> dict[str, object]:
    optional = {"chatSha256": sources.chat, "cutsSha256": sources.cut_filter}
    return {
        "version": VERSION,
        "transcriptSha256": _sha256(sources.transcript),
        "movieKey": None if sources.movie is None else movie_key(sources.movie),
        **{name: None if path is None else _sha256(path) for name, path in optional.items()},
    }


def _comments(sources: Sources) -> list[Comment] | None:
    if sources.chat is None:
        return None
    if sources.cut_filter is None:
        return read_chat(sources.chat, duration=sources.duration)
    moved = to_edited_clock(
        read_chat(sources.chat, duration=math.inf), read_keep_spans(sources.cut_filter)
    )
    return [comment for comment in moved if comment.offset <= sources.duration]


def estimate(
    *,
    utterances: Sequence[Utterance],
    duration: float,
    levels: Sequence[float] | None,
    comments: Sequence[Comment] | None,
) -> dict[str, object]:
    width = bin_seconds(duration)
    bins = max(1, math.ceil(duration / width))
    characters = speech_characters(utterances, width=width, bins=bins)
    loudness = (
        None if levels is None else speech_loudness(levels, utterances, width=width, bins=bins)
    )
    speech_score = (
        relative_score(characters) if loudness is None else relative_score(characters, loudness)
    )
    result: dict[str, object] = {
        "version": VERSION,
        "binSeconds": width,
        "duration": duration,
        "speech": {
            "score": speech_score,
            "characters": [round(value, 1) for value in characters],
            "loudnessDb": None
            if loudness is None
            else [None if v is None else round(v, 1) for v in loudness],
            "peaks": peaks(speech_score, width=width),
        },
        "comments": None,
        "combined": None,
    }
    if comments is None:
        return result
    counts, chat_characters = [0.0] * bins, [0.0] * bins
    for comment in comments:
        index = min(int(comment.offset // width), bins - 1)
        counts[index] += 1
        chat_characters[index] += comment.characters
    comment_score = relative_score(counts, chat_characters)
    combined = relative_score(speech_score, comment_score)
    result["comments"] = {
        "score": comment_score,
        "count": [int(value) for value in counts],
        "characters": [int(value) for value in chat_characters],
        "peaks": peaks(comment_score, width=width),
    }
    result["combined"] = {"score": combined, "peaks": peaks(combined, width=width)}
    return result


def _write_atomic(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(sources: Sources, *, cache: Path) -> Path | None:
    """Write the sidecar; returns None when an up-to-date one already exists."""
    out = sidecar(sources.transcript, "excitement.json")
    inputs = fingerprint(sources)
    if out.is_file():
        previous = json.loads(out.read_text(encoding="utf-8"))
        if is_record(previous) and previous.get("inputs") == inputs:
            return None
    result = estimate(
        utterances=read_utterances(sources.transcript),
        duration=sources.duration,
        levels=None if sources.movie is None else cached_levels(sources.movie, cache=cache),
        comments=_comments(sources),
    )
    result["inputs"] = inputs
    _write_atomic(out, json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return out


def chat_path(youtube_id: str, *, cache: Path) -> Path:
    return cache / f"chat-youtube-{youtube_id}.live_chat.json"


def fetch_chat(*, youtube_id: str, cache: Path) -> Path:
    """Download a live chat replay once per YouTube id (kept local, never published)."""
    if not YOUTUBE_ID.fullmatch(youtube_id):
        raise ReplayError("Invalid YouTube video id")
    cache.mkdir(parents=True, exist_ok=True)
    stem = cache / f"chat-youtube-{youtube_id}"
    command = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--js-runtimes", "node"]
    command += ["--no-playlist", "--skip-download", "--write-subs", "--sub-langs", "live_chat"]
    command += ["-o", f"{stem}.%(ext)s", f"https://www.youtube.com/watch?v={youtube_id}"]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    path = chat_path(youtube_id, cache=cache)
    if not path.is_file():
        raise ReplayError("This video has no live chat replay")
    return path


def _existing(value: object) -> Path | None:
    return Path(value) if isinstance(value, str) and Path(value).is_file() else None


def edit_cut_filter(edits: Path | None, *, video_id: str, transcript: Path) -> Path | None:
    """The edit folder's cut filter, trusted only when its SRT is this exact transcript."""
    folder = None if edits is None else edits / video_id
    if folder is None or not (folder / "filter.txt").is_file():
        return None
    srt = folder / f"{video_id}.srt"
    same = srt.is_file() and _sha256(srt) == _sha256(transcript)
    return folder / "filter.txt" if same else None


def _video_sources(
    video: dict[str, object], transcript: Path, *, options: CatalogOptions
) -> Sources:
    duration = video.get("duration")
    if not isinstance(duration, int | float):
        raise ReplayError("Catalog video has no duration")
    original = video.get("original")
    on_original = video.get("movie") is None
    movie = (
        _existing(video.get("localPath"))
        if not on_original
        else (_existing(record(original).get("localPath")) if is_record(original) else None)
    )
    video_id = str(video.get("id"))
    match = re.search(r"_youtube-([A-Za-z0-9_-]{11})", video_id)
    if match is None:
        return Sources(transcript, float(duration), movie)
    cut_filter = (
        None
        if on_original
        else edit_cut_filter(options.edits, video_id=video_id, transcript=transcript)
    )
    chat = chat_path(match.group(1), cache=options.cache)
    # Chat offsets are on the original clock; an edit needs its verified cut filter.
    usable = on_original or cut_filter is not None
    if usable and options.fetch and not chat.is_file():
        fetch_chat(youtube_id=match.group(1), cache=options.cache)
    return Sources(
        transcript, float(duration), movie, chat if usable and chat.is_file() else None, cut_filter
    )


def build_catalog(catalog: Path, *, options: CatalogOptions) -> list[str]:
    """Build sidecars for every catalog video with a transcript; returns one line per video."""
    lines: list[str] = []
    videos = record(json.loads(catalog.read_text(encoding="utf-8"))).get("videos")
    if not isinstance(videos, list):
        raise ReplayError("Catalog has no video list")
    for entry in videos:
        video = record(entry)
        transcript = _existing(video.get("transcriptPath"))
        if transcript is None or not video.get("cueCount"):
            continue
        try:
            sources = _video_sources(video, transcript, options=options)
            written = build(sources, cache=options.cache)
            state = "unchanged" if written is None else "written"
            comments = "no" if sources.chat is None else "edited" if sources.cut_filter else "yes"
            lines.append(
                f"{state} {video.get('id')} volume={'yes' if sources.movie else 'no'} "
                f"comments={comments}"
            )
        except (ReplayError, OSError, ValueError, subprocess.CalledProcessError) as error:
            lines.append(f"failed {video.get('id')} ({error})")
    return lines


def cache_movies(movies: Sequence[Path], *, cache: Path) -> int:
    """Decode each movie's levels once; prints `<key> <seconds> <movie>` or a failure."""
    status = 0
    for movie in movies:
        try:
            seconds = len(cached_levels(movie, cache=cache))
            print(f"{movie_key(movie)} {seconds} {movie}", flush=True)
        except (ReplayError, OSError) as error:
            print(f"failed {movie} ({error})", flush=True)
            status = 1
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m replay.excitement", description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="loudness/chat cache")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("build", help="one transcript")
    one.add_argument("--transcript", type=Path, required=True)
    one.add_argument("--duration", type=float, required=True)
    one.add_argument("--movie", type=Path)
    one.add_argument("--chat", type=Path, help="live chat replay on the source clock")
    one.add_argument("--cut-filter", type=Path, help="edit select filter mapping chat to it")
    many = commands.add_parser("catalog", help="every video of a web viewer catalog")
    many.add_argument("--catalog", type=Path, required=True)
    many.add_argument("--fetch-chat", action="store_true", help="download YouTube live chat")
    many.add_argument("--youtube-edits", type=Path, help="root of <video id>/filter.txt edits")
    levels = commands.add_parser(
        "levels", help="cache per-second levels only (TranscriptViewer's automatic BGM volume)"
    )
    levels.add_argument("movies", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "levels":
        return cache_movies(args.movies, cache=args.cache)
    if args.command == "build":
        sources = Sources(args.transcript, args.duration, args.movie, args.chat, args.cut_filter)
        written = build(sources, cache=args.cache)
        print(written or "unchanged")
        return 0
    options = CatalogOptions(args.cache, args.fetch_chat, args.youtube_edits)
    lines = build_catalog(args.catalog, options=options)
    print("\n".join(lines))
    return int(any(line.startswith("failed") for line in lines))


if __name__ == "__main__":
    raise SystemExit(main())
