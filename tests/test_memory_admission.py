"""Machine-wide memory reservations: budget, measured peaks, pressure gate and torch caps."""

import json
import os
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from replay import memory_admission as admission

GIB = 2**30
TOTAL = 48 * GIB
NORMAL = (95, 1)


def current(directory: Path) -> admission.Settings:
    return admission.settings(directory, TOTAL)


def test_defaults_budget_physical_memory_minus_headroom(tmp_path: Path) -> None:
    settings = current(tmp_path)
    assert settings.budget == 38 * GIB
    assert settings.peaks["cleanup-chunk"] == 7 * GIB
    assert settings.hard_caps is True
    assert settings.known == frozenset()


def test_settings_override_budget_peaks_and_caps(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text(
        '{"budgetGiB": 30, "peaksGiB": {"asr": 3}, "hardCaps": false}', "utf-8"
    )
    settings = current(tmp_path)
    assert (settings.budget, settings.peaks["asr"], settings.hard_caps) == (
        30 * GIB,
        3 * GIB,
        False,
    )
    assert settings.known == frozenset({"asr"})
    (tmp_path / "settings.json").write_text('{"headroomGiB": 16}', "utf-8")
    assert current(tmp_path).budget == 32 * GIB


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"slots": 1}', "take budgetGiB"),
        ('{"budgetGiB": 0}', "positive GiB"),
        ('{"budgetGiB": true}', "positive GiB"),
        ('{"peaksGiB": [1]}', "map jobs"),
        ('{"hardCaps": 1}', "true or false"),
        (json.dumps({"peaksGiB": {"asr": 1}, "budgetGiB": 1, "x": "1" * 5000}), "too large"),
    ],
)
def test_invalid_settings_are_refused(tmp_path: Path, text: str, message: str) -> None:
    (tmp_path / "settings.json").write_text(text, "utf-8")
    with pytest.raises(ValueError, match=message):
        current(tmp_path)


def test_a_measured_footprint_with_margin_raises_but_never_lowers_the_estimate(
    tmp_path: Path,
) -> None:
    admission.record_peak(tmp_path, "asr", 4 * GIB)
    admission.record_peak(tmp_path, "asr", 3 * GIB)  # smaller: the largest is kept
    admission.record_peak(tmp_path, "asr", 0)
    settings = current(tmp_path)
    assert settings.peaks["asr"] == 6 * GIB  # 4 GiB x 1.25 is below the 6 GiB estimate
    assert settings.caps == {"asr": 8 * GIB}  # twice the measured footprint
    assert settings.known == frozenset({"asr"})
    admission.record_peak(tmp_path, "asr", 8 * GIB)
    assert current(tmp_path).peaks["asr"] == 10 * GIB
    assert json.loads((tmp_path / "footprints.json").read_text("utf-8")) == {
        "asr": {"peak": 8 * GIB, "runs": 3}
    }


def test_old_resident_size_peaks_are_never_read(tmp_path: Path) -> None:
    (tmp_path / "peaks.json").write_text('{"onsets": 1}', "utf-8")
    assert current(tmp_path).known == frozenset()


def test_a_malformed_peaks_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "footprints.json").write_text("[1]", "utf-8")
    with pytest.raises(ValueError, match="map jobs to their peak and runs"):
        current(tmp_path)


def test_reservations_fit_the_budget_and_release(tmp_path: Path) -> None:
    settings = current(tmp_path)
    first = admission.Reservation(tmp_path, "cleanup-chunk", "1", 20 * GIB)
    second = admission.Reservation(tmp_path, "asr", "2", 18 * GIB)
    third = admission.Reservation(tmp_path, "onsets", "3", 1)
    assert first.try_reserve(settings, lambda: NORMAL) is True
    assert second.try_reserve(settings, lambda: NORMAL) is True
    assert admission.reserved(tmp_path) == 38 * GIB
    assert third.try_reserve(settings, lambda: NORMAL) is False
    first.release()
    first.release()
    assert third.try_reserve(settings, lambda: NORMAL) is True
    assert admission.reserved(tmp_path) == 18 * GIB + 1
    second.release()
    third.release()


def test_a_job_alone_beyond_the_budget_runs_once_free_memory_covers_the_budget(
    tmp_path: Path,
) -> None:
    settings = current(tmp_path)  # 48 GiB, budget 38 GiB
    huge = admission.Reservation(tmp_path, "separation", "1", 60 * GIB)
    assert huge.try_reserve(settings, lambda: (85, 4)) is False  # critical: never
    assert huge.try_reserve(settings, lambda: (5, 2)) is False  # 2.4 GiB free: it would thrash
    assert huge.try_reserve(settings, lambda: (85, 2)) is True  # 40.8 GiB free covers 38
    huge.release()


def test_memory_other_processes_use_keeps_a_job_waiting(tmp_path: Path) -> None:
    settings = current(tmp_path)
    clips = admission.Reservation(tmp_path, "clips", "1", 24 * GIB)
    assert clips.try_reserve(settings, lambda: (40, 1)) is False  # 19.2 GiB free: another server
    assert clips.try_reserve(settings, lambda: (55, 1)) is True  # 26.4 GiB free
    clips.release()


def test_low_free_memory_or_pressure_stops_a_second_job(tmp_path: Path) -> None:
    settings = current(tmp_path)
    first = admission.Reservation(tmp_path, "asr", "1", GIB)
    assert first.try_reserve(settings, lambda: NORMAL) is True
    second = admission.Reservation(tmp_path, "asr", "2", GIB)
    assert second.try_reserve(settings, lambda: (19, 1)) is False
    assert second.try_reserve(settings, lambda: (95, 2)) is False
    first.release()


def test_a_killed_holder_releases_its_reservation(tmp_path: Path) -> None:
    held = tmp_path / "held"
    held.mkdir()
    (held / f"asr.9.1.2.{GIB}").write_text("", "utf-8")  # nobody holds its lock
    (held / "not-a-marker").write_text("", "utf-8")
    assert admission.reserved(tmp_path) == 0
    assert sorted(path.name for path in held.iterdir()) == ["not-a-marker"]
    assert admission.reserved(tmp_path / "missing") == 0


def test_unknown_jobs_or_owners_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown reserved job"):
        admission.Reservation(tmp_path, "clocks", "1", 1)
    with pytest.raises(ValueError, match="Unknown reserved job"):
        admission.Reservation(tmp_path, "asr", "../x", 1)


def test_reserve_waits_until_the_job_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission, "physical", lambda: TOTAL)
    blocker = admission.Reservation(tmp_path, "cleanup-chunk", "1", 36 * GIB)
    assert blocker.try_reserve(current(tmp_path), lambda: NORMAL) is True
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        blocker.release()

    with admission.reserve("asr", "2", directory=tmp_path, measure=lambda: NORMAL, sleep=sleep):
        assert admission.reserved(tmp_path) == 6 * GIB
    assert waits == [15.0]
    assert admission.reserved(tmp_path) == 0


def test_reserve_defaults_to_the_checkout_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = Path(admission.__file__).resolve().parents[2]
    assert admission.root() == checkout / "edits/work/memory-reservations"


def test_caps_apply_only_after_two_complete_measured_runs(tmp_path: Path) -> None:
    assert admission.cap_environment("asr", current(tmp_path), TOTAL) == {}
    admission.record_peak(tmp_path, "onsets", 4 * GIB)
    assert admission.cap_environment("onsets", current(tmp_path), TOTAL) == {}  # one run only
    admission.record_peak(tmp_path, "onsets", 3 * GIB)
    assert admission.cap_environment("onsets", current(tmp_path), TOTAL) == {
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.2137",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.2137",
    }
    (tmp_path / "settings.json").write_text('{"hardCaps": false}', "utf-8")
    assert admission.cap_environment("onsets", current(tmp_path), TOTAL) == {}
    (tmp_path / "settings.json").write_text('{"peaksGiB": {"asr": 0.001}}', "utf-8")
    assert admission.cap_environment("asr", current(tmp_path), TOTAL) == {}  # configured only
    admission.record_peak(tmp_path, "asr", 2**20)
    admission.record_peak(tmp_path, "asr", 2**20)
    assert admission.cap_environment("asr", current(tmp_path), TOTAL) == {
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0100",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.0100",
    }


def test_footprint_rows_without_peak_and_runs_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "footprints.json").write_text(
        '{"asr": 5, "onsets": {"peak": 0, "runs": 2}, "compose": {"peak": 9}}', "utf-8"
    )
    assert current(tmp_path).known == frozenset()


def completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout, "")


def test_sysctl_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {"hw.memsize": "51539607552\n", "kern.memorystatus_level": "58\n"}
    monkeypatch.setattr(
        subprocess, "run", lambda command, **_k: completed(answers.get(command[-1], ""))
    )
    assert admission.physical() == 51539607552
    assert admission.pressure() == (58, 4)  # an unreadable pressure counts as critical
    answers.clear()
    with pytest.raises(ValueError, match="hw.memsize"):
        admission.physical()


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   360644.
Pageins:                                       {pageins}.
Pageouts:                                       12.
Swapins:                                        {swapins}.
Swapouts:                                       34.
"""


def vm_stat(pageins: int, swapins: int) -> str:
    return VM_STAT.format(pageins=pageins, swapins=swapins)


def test_paged_in_bytes_sums_swapins_and_pageins() -> None:
    assert admission.paged_in_bytes(vm_stat(3, 1)) == 65536
    assert (
        admission.paged_in_bytes("Mach Virtual Memory Statistics: (page size of 16384 bytes)")
        is None
    )


def test_the_swap_in_rate_is_measured_between_shared_samples(tmp_path: Path) -> None:
    readings = iter([vm_stat(0, 0), vm_stat(0, 0), vm_stat(65536, 0), vm_stat(65536, 0)])
    times = iter([0.0, 5.0, 60.0, 2000.0])

    def rate() -> float:
        return admission.swap_in_rate(
            tmp_path, read=lambda: next(readings), clock=lambda: next(times)
        )

    assert rate() == 0.0  # the first sample
    assert rate() == 0.0  # 5 s later: the last measured rate
    assert rate() == 1024**3  # 1 GiB paged in over one minute
    assert rate() == 0.0  # a sample older than the maximum says nothing about now


def test_an_unreadable_vm_stat_counts_as_paging_without_bound(tmp_path: Path) -> None:
    assert admission.swap_in_rate(tmp_path, read=lambda: "", clock=lambda: 0.0) == float("inf")
    assert list(tmp_path.iterdir()) == []


def test_fast_paging_counts_as_critical_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {"kern.memorystatus_level": "70\n", "kern.memorystatus_vm_pressure_level": "1\n"}
    monkeypatch.setattr(
        subprocess, "run", lambda command, **_k: completed(answers.get(command[-1], ""))
    )
    monkeypatch.setattr(admission, "swap_in_rate", lambda: 8.0 * 1024**3)
    assert admission.pressure() == (70, admission.PRESSURE_CRITICAL)
    monkeypatch.setattr(admission, "swap_in_rate", lambda: 0.5 * 1024**3)
    assert admission.pressure() == (70, admission.PRESSURE_NORMAL)


LISTING = (
    "  10    10 python -m replay.recording_steps onsets /p.json\n"
    "  11    10 python -m replay.speech_onsets tokens.json\n"
    "  12    12 python -m replay.recording_steps onsets /other.json\n"
    "bad\n"
)
# What ps reported resident for these processes; the GPU allocations are not in it.
RESIDENT = {10: GIB // 4, 11: GIB // 2, 12: GIB}
FOOTPRINT = {10: GIB // 4, 11: 3 * GIB, 12: GIB}


def test_group_footprint_sums_the_matching_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: completed(LISTING))
    found = admission.group_footprint(
        lambda arguments: arguments.endswith("onsets /p.json"), read=FOOTPRINT.get
    )
    assert found == 3 * GIB + GIB // 4
    unreadable = admission.group_footprint(lambda arguments: True, read=lambda _pid: None)
    assert unreadable is None


def test_a_small_resident_size_with_a_large_footprint_caps_above_the_footprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: completed(LISTING))
    sampled = threading.Event()

    def sample(match: Callable[[str], bool]) -> int | None:
        sampled.set()
        return admission.group_footprint(match, read=FOOTPRINT.get)

    with admission.measuring(
        tmp_path, "onsets", lambda arguments: arguments.endswith("onsets /p.json"),
        interval=0.01, sample=sample,
    ):  # fmt: skip
        assert sampled.wait(5)
    admission.record_peak(tmp_path, "onsets", 3 * GIB)  # a second run, a little smaller
    settings = current(tmp_path)
    measured = 3 * GIB + GIB // 4
    assert RESIDENT[10] + RESIDENT[11] < GIB  # what the old resident-size sampler saw
    assert settings.caps["onsets"] == 2 * measured
    assert settings.caps["onsets"] > measured
    assert admission.cap_environment("onsets", settings, TOTAL) == {
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.1736",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.1736",
    }


def test_an_incomplete_sample_records_no_peak(tmp_path: Path) -> None:
    sampled = threading.Event()

    def sample(_match: object) -> int | None:
        sampled.set()
        return None

    with admission.measuring(tmp_path, "asr", lambda _a: True, interval=0.01, sample=sample):
        assert sampled.wait(5)
    assert not (tmp_path / "footprints.json").exists()


def test_footprint_reads_this_process_and_an_exited_one() -> None:
    assert (admission.footprint(os.getpid()) or 0) > 0
    assert admission.footprint(2**22 + 12345) == 0


def test_measuring_records_the_peak_when_the_job_ends(tmp_path: Path) -> None:
    sampled = threading.Event()
    values = iter([GIB, 3 * GIB, 2 * GIB])

    def sample(_match: object) -> int:
        value = next(values, 0)
        if value == 0:
            sampled.set()
        return value

    with admission.measuring(tmp_path, "yolo-track", lambda _a: True, interval=0.01, sample=sample):
        assert sampled.wait(5)
    assert json.loads((tmp_path / "footprints.json").read_text("utf-8")) == {
        "yolo-track": {"peak": 3 * GIB, "runs": 1}
    }


def test_a_measured_clips_footprint_replaces_its_placeholder(tmp_path: Path) -> None:
    assert current(tmp_path).peaks["clips"] == 24 * GIB
    admission.record_peak(tmp_path, "clips", 10 * GIB)
    assert current(tmp_path).peaks["clips"] == 12.5 * GIB  # lower than the placeholder
    admission.record_peak(tmp_path, "clips", 26 * GIB)
    assert current(tmp_path).peaks["clips"] == 32.5 * GIB
