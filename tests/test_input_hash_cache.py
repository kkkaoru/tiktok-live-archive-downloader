import hashlib
import os
from pathlib import Path

import pytest

from replay import delivery_source_assembly, stages
from replay import input_hash_cache as hashes
from replay.core import ReplayError


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_inputs_share_hashes_and_terminal_reads_fresh(tmp_path: Path) -> None:
    source, output = tmp_path / "input", tmp_path / "output"
    source.write_bytes(b"original input")
    output.write_bytes(b"produced output")
    calls: list[Path] = []

    def ordinary(path: Path) -> str:
        calls.append(path)
        return digest(path)

    assert hashes.cached_digest(source, ordinary) == digest(source)
    cache = hashes.InputHashCache([source, source])
    with hashes.input_hash_context(cache):
        cache.prime()
        assert hashes.cached_digest(source, ordinary) == digest(source)
        assert hashes.cached_digest(source, ordinary) == digest(source)
        assert hashes.cached_digest(output, ordinary) == digest(output)
        cache.guard()
        assert cache.verify() == {str(source): digest(source)}
    assert cache.hash_reads == 2
    assert cache.cache_hits == 2
    assert cache.bytes_read == len(b"original input") * 2
    assert calls == [source, output]
    assert hashes.cached_digest(source, ordinary) == digest(source)
    assert calls == [source, output, source]


def test_lazy_digest_requires_fresh_prime_before_verify_for_all_inputs(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    cache = hashes.InputHashCache([first, second])
    assert cache.digest(first, digest) == digest(first)
    with pytest.raises(ReplayError, match="primed"):
        cache.verify()
    cache.prime()
    assert len(cache.verify()) == 2


@pytest.mark.parametrize("change", ["size", "same-size", "restore-mtime", "replace", "symlink"])
@pytest.mark.parametrize("boundary", ["hit", "guard", "terminal"])
def test_mutation_at_every_boundary_is_rejected(tmp_path: Path, change: str, boundary: str) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"original")
    cache = hashes.InputHashCache([source])
    cache.prime()
    before = source.stat()
    if change == "size":
        source.write_bytes(b"changed size")
    elif change in ("same-size", "restore-mtime"):
        source.write_bytes(b"changed!")
        if change == "restore-mtime":
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"original")
        if change == "replace":
            replacement.replace(source)
        else:
            source.unlink()
            source.symlink_to(replacement)
    with pytest.raises(ReplayError):
        if boundary == "hit":
            cache.digest(source, digest)
        elif boundary == "guard":
            cache.guard()
        else:
            cache.verify()


@pytest.mark.parametrize(
    "budget", ["files", "bytes", "bool", "unbounded-files", "unbounded-bytes", "empty"]
)
def test_input_budgets_remain_bounded(tmp_path: Path, budget: str) -> None:
    source, second = tmp_path / "source", tmp_path / "second"
    source.write_bytes(b"bytes")
    second.write_bytes(b"more")
    with pytest.raises(ReplayError):
        if budget == "files":
            hashes.InputHashCache([source, second], max_files=1)
        elif budget == "bytes":
            hashes.InputHashCache([source], max_bytes=1)
        elif budget == "bool":
            hashes.InputHashCache([source], max_files=True)
        elif budget == "unbounded-files":
            hashes.InputHashCache([source], max_files=hashes.MAX_FILES + 1)
        elif budget == "unbounded-bytes":
            hashes.InputHashCache([source], max_bytes=hashes.MAX_BYTES + 1)
        else:
            hashes.InputHashCache([])


def test_empty_directory_and_parent_link_are_not_cached(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"")
    with pytest.raises(ReplayError, match="nonempty regular"):
        hashes.InputHashCache([source])
    with pytest.raises(ReplayError, match="nonempty regular"):
        hashes.InputHashCache([tmp_path])
    source.write_bytes(b"source")
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ReplayError, match="lexical"):
        hashes.InputHashCache([linked / "source"])


def test_repeated_input_iterator_is_bounded(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"input")
    with pytest.raises(ReplayError, match="excessive input references"):
        hashes.InputHashCache(source for _ in range(hashes.MAX_FILES + 1))


def test_context_is_scoped_resets_after_failure_and_never_nests(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"bytes")
    cache = hashes.InputHashCache([source])
    with pytest.raises(RuntimeError, match="operation failed"), hashes.input_hash_context(cache):
        with pytest.raises(ReplayError, match="Nested"), hashes.input_hash_context(cache):
            pytest.fail("nested context entered")
        raise RuntimeError("operation failed")
    assert hashes.cached_digest(source, digest) == digest(source)


def test_open_descriptor_mutation_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"source")
    cache = hashes.InputHashCache([source])
    real_open = os.open

    def changed_open(path: Path, flags: int) -> int:
        descriptor = real_open(path, flags)
        source.write_bytes(b"changed")
        return descriptor

    monkeypatch.setattr(os, "open", changed_open)
    with pytest.raises(ReplayError, match="before opening"):
        cache.prime()


def test_fresh_terminal_digest_must_match_even_if_stat_checks_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"source")
    cache = hashes.InputHashCache([source])
    cache.prime()
    monkeypatch.setattr(cache, "_read", lambda path, entry, *, fresh: "0" * 64)
    with pytest.raises(ReplayError, match="terminal"):
        cache.verify()


def test_descriptor_change_while_hashing_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"source")
    cache = hashes.InputHashCache([source])
    original = hashlib.sha256

    class MutatingHash:
        def __init__(self) -> None:
            self.inner = original()

        def update(self, block: bytes) -> None:
            self.inner.update(block)
            source.write_bytes(b"update")

        def hexdigest(self) -> str:
            return self.inner.hexdigest()

    monkeypatch.setattr(hashlib, "sha256", MutatingHash)
    with pytest.raises(ReplayError, match="descriptor changed"):
        cache.prime()


def test_existing_digest_helpers_share_only_explicit_input_scope(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fixed shared input")
    expected = stages.file_digest(source)
    assert delivery_source_assembly._digest(source) == expected
    cache = hashes.InputHashCache((source,))
    with hashes.input_hash_context(cache):
        cache.prime()
        assert stages.file_digest(source) == expected
        assert delivery_source_assembly._digest(source) == expected
        assert cache.hash_reads == 1 and cache.cache_hits == 2
        assert cache.verify()[str(source)] == expected
    assert stages.file_digest(source) == expected


@pytest.mark.parametrize("budget", ["files", "bytes"])
def test_overflow_is_fresh_crypto_without_losing_closure_guards(
    tmp_path: Path, budget: str
) -> None:
    first, overflow = tmp_path / "first", tmp_path / "overflow"
    first.write_bytes(b"first")
    overflow.write_bytes(b"other")
    cache = hashes.InputHashCache(
        (first, overflow),
        max_files=1 if budget == "files" else 2,
        max_bytes=5 if budget == "bytes" else 10,
        allow_uncached_overflow=True,
    )
    with hashes.input_hash_context(cache):
        cache.prime()
        for _ in range(2):
            assert stages.file_digest(first) == hashlib.sha256(b"first").hexdigest()
            assert stages.file_digest(overflow) == hashlib.sha256(b"other").hexdigest()
        assert cache.hash_reads == 4 and cache.cache_hits == 2
        assert len(cache.verify()) == 2
        overflow.write_bytes(b"mutated")
        with pytest.raises(ReplayError, match="changed"):
            cache.guard()


def test_overflow_closure_has_independent_finite_reference_bound(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fixed")
    with pytest.raises(ReplayError, match="excessive"):
        hashes.InputHashCache(
            (source for _ in range(hashes.MAX_INPUTS + 1)), allow_uncached_overflow=True
        )


def _selection_scope(
    paths: tuple[Path, ...], *, max_files: int, max_bytes: int, first_fit: bool
) -> tuple[int, int, int, dict[str, str]]:
    cache = hashes.InputHashCache(
        paths, max_files=max_files, max_bytes=max_bytes, allow_uncached_overflow=True
    )
    if first_fit:
        selected: set[Path] = set()
        consumed = 0
        for path in paths:
            size = path.stat().st_size
            if path not in selected and len(selected) < max_files and consumed + size <= max_bytes:
                selected.add(path)
                consumed += size
        # Test-only reference selection: execute identical real SHA/FD/guard
        # boundaries, without importing or running an old producer snapshot.
        cache._uncached = set(paths) - selected
    with hashes.input_hash_context(cache):
        cache.prime()
        assert cache.hash_reads == len(paths)
        assert cache.bytes_read == sum(path.stat().st_size for path in paths)
        for _ in range(3):
            for path in paths:
                assert stages.file_digest(path) == digest(path)
            cache.guard()
        before_reads, before_bytes = cache.hash_reads, cache.bytes_read
        verified = cache.verify()
        assert cache.hash_reads - before_reads == len(paths)
        assert cache.bytes_read - before_bytes == sum(path.stat().st_size for path in paths)
    return cache.hash_reads, cache.bytes_read, cache.cache_hits, verified


@pytest.mark.parametrize(
    ("sizes", "max_files", "max_bytes", "before_bytes", "after_bytes", "reads", "hits"),
    [((1, 2, 16), 1, 64, 92, 47, 12, 3), ((3, 8, 16), 2, 20, 102, 78, 9, 6)],
)
def test_late_large_inputs_reduce_actual_sha_bytes_with_same_fresh_boundaries(
    tmp_path: Path,
    sizes: tuple[int, ...],
    max_files: int,
    max_bytes: int,
    before_bytes: int,
    after_bytes: int,
    reads: int,
    hits: int,
) -> None:
    paths = tuple(tmp_path / f"input-{index}" for index in range(len(sizes)))
    for path, size in zip(paths, sizes, strict=True):
        path.write_bytes(b"x" * size)
    before = _selection_scope(paths, max_files=max_files, max_bytes=max_bytes, first_fit=True)
    after = _selection_scope(paths, max_files=max_files, max_bytes=max_bytes, first_fit=False)
    assert before[:3] == (reads, before_bytes, hits)
    assert after[:3] == (reads, after_bytes, hits)
    assert before[3] == after[3] == {str(path): digest(path) for path in paths}
    assert list(after[3]) == [str(path) for path in paths]


def test_large_input_after_global_slot_limit_is_cached_without_losing_manifest_order(
    tmp_path: Path,
) -> None:
    small = tuple(tmp_path / f"small-{index:04d}" for index in range(hashes.MAX_FILES))
    large = tmp_path / "late-large"
    for path in small:
        path.write_bytes(b"s")
    large.write_bytes(b"large input")
    paths = (*small, large)
    cache = hashes.InputHashCache(paths, allow_uncached_overflow=True)
    with hashes.input_hash_context(cache):
        cache.prime()
        before_reads = cache.hash_reads
        assert stages.file_digest(large) == digest(large)
        assert cache.hash_reads == before_reads and cache.cache_hits == 1
        assert len(cache._uncached) == 1 and large not in cache._uncached
        assert list(cache.verify()) == [str(path) for path in paths]
        assert cache.hash_reads == 2 * len(paths)


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_size_ties_are_lexical_and_duplicates_do_not_spend_slots(
    tmp_path: Path, reverse: bool
) -> None:
    first, second = tmp_path / "a", tmp_path / "z"
    first.write_bytes(b"first")
    second.write_bytes(b"other")
    paths = (second, first, second) if reverse else (first, second, first)
    cache = hashes.InputHashCache(paths, max_files=1, allow_uncached_overflow=True)
    with hashes.input_hash_context(cache):
        cache.prime()
        assert cache.hash_reads == 2
        assert stages.file_digest(first) == digest(first)
        assert cache.hash_reads == 2
        assert stages.file_digest(second) == digest(second)
        assert cache.hash_reads == 3
        assert list(cache.verify()) == [str(path) for path in dict.fromkeys(paths)]
    assert cache._uncached == {second}


def test_oversized_input_does_not_prevent_smaller_fitting_inputs_from_being_cached(
    tmp_path: Path,
) -> None:
    oversized, small, medium = tmp_path / "oversized", tmp_path / "small", tmp_path / "medium"
    oversized.write_bytes(b"x" * 21)
    small.write_bytes(b"s" * 4)
    medium.write_bytes(b"m" * 6)
    paths = (oversized, small, medium)
    cache = hashes.InputHashCache(paths, max_files=2, max_bytes=10, allow_uncached_overflow=True)
    with hashes.input_hash_context(cache):
        cache.prime()
        assert cache.hash_reads == 3 and cache.bytes_read == 31
        for path in paths:
            assert stages.file_digest(path) == digest(path)
        assert cache.hash_reads == 4 and cache.bytes_read == 52 and cache.cache_hits == 2
        assert list(cache.verify()) == [str(path) for path in paths]
    assert cache._uncached == {oversized}
    assert cache.hash_reads == 7 and cache.bytes_read == 83


@pytest.mark.parametrize("first_fit", [False, True])
@pytest.mark.parametrize("changed", ["small", "large"])
@pytest.mark.parametrize("boundary", ["hit", "guard", "terminal"])
def test_both_selection_policies_reject_selected_and_overflow_mutation(
    tmp_path: Path, first_fit: bool, changed: str, boundary: str
) -> None:
    small, large = tmp_path / "small", tmp_path / "large"
    small.write_bytes(b"s")
    large.write_bytes(b"large")
    cache = hashes.InputHashCache((small, large), max_files=1, allow_uncached_overflow=True)
    if first_fit:
        # Only baseline admission differs; the same complete closure and
        # current guard/fresh-byte verifier remain under test for both policies.
        cache._uncached = {large}
    cache.prime()
    path = small if changed == "small" else large
    before = path.stat()
    path.write_bytes(b"x" * before.st_size)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ReplayError):
        if boundary == "hit":
            cache.digest(path, digest)
        elif boundary == "guard":
            cache.guard()
        else:
            cache.verify()
