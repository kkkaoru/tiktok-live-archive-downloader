"""Bounded speculative measurement candidates; never a caption layout decision.

This standalone planner does not warm production caches or invoke native tools.
Its optional predictions may be incomplete: the exact planner must still measure
all cache misses, and speculative entries must never displace required entries.
"""

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from replay.caption_native_layout import MAX_BATCH_UNITS
from replay.caption_native_paging import read_tokens
from replay.caption_timing import CaptionPolicy, SpokenToken, _attached_tails, _lexical_tokens
from replay.recordings import record
from replay.stages import file_digest

MAX_TOKENS = 4096
MAX_INPUT_CHARACTERS = 327_680
MAX_NATIVE_UNITS = 120
MAX_LOG_BYTES = 64 * 1024 * 1024
MAX_LOG_REQUESTS = 65536


@dataclass(frozen=True)
class PrefetchBudget:
    """Independent discovery and storage limits, not visual fit thresholds."""

    starts_per_end: int = 8
    visits: int = 32768
    strings: int = 4096
    utf16_units: int = 65536

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or not 1 <= value <= 65536
            for value in (self.starts_per_end, self.visits, self.strings, self.utf16_units)
        ):
            raise ValueError("Invalid prefetch resource budget")


@dataclass(frozen=True)
class PrefetchPlan:
    texts: tuple[str, ...]
    visits: int
    utf16_units: int
    exhausted: bool


@dataclass(frozen=True)
class PredictionCoverage:
    requested: int
    predicted: int
    unused: int
    missing: tuple[str, ...]


@dataclass(frozen=True)
class BatchForecast:
    original_requests: int
    warm_requests: int
    remaining_requests: int
    total_requests: int


DEFAULT_BUDGET = PrefetchBudget()


def _phrase_ends(attached: list[bool]) -> tuple[int, ...]:
    ends = [len(attached)] * len(attached)
    for index in range(len(attached) - 2, -1, -1):
        ends[index] = ends[index + 1] if attached[index + 1] else index + 1
    return tuple(ends)


def _joined(words: list[SpokenToken], start: int, end: int, limit: int) -> str:
    parts: list[str] = []
    count = 0
    for index in range(start, end):
        count += len(words[index].text)
        if count > limit:
            return ""
        parts.append(words[index].text)
    return "".join(parts)


def _prefixes(words: list[SpokenToken], *, policy: CaptionPolicy, starts: int) -> Iterator[str]:
    ends = _phrase_ends(_attached_tails(words, policy))
    # The single-token warm-up and reserved complete phrase have highest priority.
    for index, word in enumerate(words):
        yield word.text
        yield _joined(words, index, ends[index], policy.maximum_characters)
    for end in range(1, len(words) + 1):
        yield from _ending_prefixes(
            words, end=end, tail=ends[end - 1], policy=policy, starts=starts
        )


def _ending_prefixes(
    words: list[SpokenToken], *, end: int, tail: int, policy: CaptionPolicy, starts: int
) -> Iterator[str]:
    for start in range(end - 2, max(-1, end - starts - 1), -1):
        if words[start].utterance != words[end - 1].utterance:
            break
        yield _joined(words, start, end, policy.maximum_characters)
        yield _joined(words, start, tail, policy.maximum_characters)


def plan_prefetch(
    tokens: tuple[SpokenToken, ...],
    *,
    policy: CaptionPolicy,
    budget: PrefetchBudget = DEFAULT_BUDGET,
) -> PrefetchPlan:
    """Reuse exact lexical/POS preparation and bound even duplicate discovery."""
    if len(tokens) > MAX_TOKENS or sum(len(token.text) for token in tokens) > MAX_INPUT_CHARACTERS:
        raise ValueError("Prefetch input exceeds resource budget")
    words = _lexical_tokens(tokens)
    selected: dict[str, None] = {}
    units = 0
    visits = 0
    exhausted = False
    for text in _prefixes(words, policy=policy, starts=budget.starts_per_end):
        if visits == budget.visits:
            exhausted = True
            break
        visits += 1
        if not text.strip() or any(mark in text for mark in ("\n", "\r", "\0")) or text in selected:
            continue
        count = len(text.encode("utf-16-le")) // 2
        if count > MAX_NATIVE_UNITS or len(text) > policy.maximum_characters:
            continue
        if len(selected) == budget.strings or units + count > budget.utf16_units:
            exhausted = True
            break
        selected[text] = None
        units += count
    return PrefetchPlan(tuple(selected), visits, units, exhausted)


def prediction_coverage(plan: PrefetchPlan, requested: tuple[str, ...]) -> PredictionCoverage:
    """Compare saved exact request strings without executing any renderer."""
    if len(requested) > 65536:
        raise ValueError("Too many observed measurement strings")
    predicted = set(plan.texts)
    observed = set(requested)
    missing = tuple(dict.fromkeys(text for text in requested if text not in predicted))
    return PredictionCoverage(
        len(requested),
        sum(text in predicted for text in requested),
        len(predicted - observed),
        missing,
    )


def _read_record(path: Path) -> dict[str, object]:
    if not 0 < path.stat().st_size <= MAX_LOG_BYTES:
        raise ValueError("Measurement evaluation document exceeds budget")
    return record(json.loads(path.read_text(encoding="utf-8")))


def forecast_batches(plan: PrefetchPlan, requests: tuple[tuple[str, ...], ...]) -> BatchForecast:
    """Replay saved grouping with unlimited cache; this is not a runtime promise."""
    warm = 0
    count = 0
    units = 0
    for text in plan.texts:
        size = len(text.encode("utf-16-le")) // 2
        if count and (count == 16 or units + size > MAX_BATCH_UNITS):
            warm += 1
            count = 0
            units = 0
        count += 1
        units += size
    warm += int(count > 0)
    cached = set(plan.texts)
    remaining = 0
    for request in requests:
        if any(text not in cached for text in request):
            remaining += 1
            cached.update(request)
    return BatchForecast(len(requests), warm, remaining, warm + remaining)


def evaluate_saved(
    directory: Path,
    *,
    segment: int,
    requests: Path,
    output: Path,
    budget: PrefetchBudget = DEFAULT_BUDGET,
) -> None:
    """Write a hash-bound prediction audit using completed request logs only."""
    if type(segment) is not int or not 0 <= segment < 512:
        raise ValueError("Invalid evaluation segment")
    index_path = directory / "segments.json"
    rows = _read_record(index_path).get("segments")
    if not isinstance(rows, list):
        raise ValueError("Expected segment index")
    matches = [record(row) for row in rows if record(row).get("segment") == segment]
    if len(matches) != 1:
        raise ValueError("Expected exactly one segment")
    combined = directory / f"combined-{segment:04d}.json"
    tokens, pins = read_tokens(directory, matches[0], record(_read_record(combined).get("inputs")))
    paths = sorted(requests.glob("batch-*.request.json"))
    if not 1 <= len(paths) <= MAX_LOG_REQUESTS:
        raise ValueError("Expected bounded saved measurement requests")
    requested: list[str] = []
    batches: list[tuple[str, ...]] = []
    for path in paths:
        values = _read_record(path).get("texts")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("Invalid saved measurement strings")
        requested.extend(value for value in values if isinstance(value, str))
        batches.append(tuple(value for value in values if isinstance(value, str)))
        if len(requested) > MAX_LOG_REQUESTS:
            raise ValueError("Too many saved measurement strings")
        pins[str(path.resolve())] = file_digest(path)
    plan = plan_prefetch(
        tokens,
        policy=CaptionPolicy(linguistic_boundaries=True, maximum_characters=80, frame_rate=30),
        budget=budget,
    )
    pins.update(
        {str(path.resolve()): file_digest(path) for path in (index_path, combined, Path(__file__))}
    )
    result = {
        "version": 1,
        "nativeExecution": False,
        "productionCacheModified": False,
        "budget": asdict(budget),
        "plan": asdict(plan),
        "inputs": pins,
        "coverage": asdict(prediction_coverage(plan, tuple(requested))),
        "batchForecast": asdict(forecast_batches(plan, tuple(batches))),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts-per-end", type=int, default=DEFAULT_BUDGET.starts_per_end)
    args = parser.parse_args()
    evaluate_saved(
        args.directory,
        segment=args.segment,
        requests=args.requests,
        output=args.output,
        budget=PrefetchBudget(starts_per_end=args.starts_per_end),
    )


if __name__ == "__main__":
    main()
