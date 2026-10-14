"""Pinned code generations for verifying already-completed operations.

On 2026-10-12 the session cleanup producer gained an interrupted-reservation
recovery capability, changing subtitle_session_cleanup.py,
subtitle_session_cleanup_plan.py and subtitle_session_cleanup_wave.py. The
compatibility rule itself later changed cleanup_lane_successor.py as well.

Operations recorded before those changes can no longer satisfy
`readerCode == _code()`, yet their completed artifacts were verified under
their own code at completion time with retained receipts. This module pins
every production generation below so verification of already-completed work
keeps working. Execution of new work always requires the current code.

Discipline: do not edit the files below (or any other closure member) until
the pending finalizations that consume these operations are terminal. Any
further shared-code change needs a new explicit generation entry here, never
silent reuse.

Only standard-library imports are used here so this module adds a single
closure entry and no transitive replay dependencies.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

COMPAT_BASENAME = "code_generation_compat.py"

MAX_PIN_BYTES = 8 << 20

# Reviewed transitive imports in the completed cleanup producer catalog. Cleanup
# calls caption_io.native_ocr_frames, not the speech-token or title generators.
# These hashes identify the recorded producer; current bytes are still checked
# separately and new execution never uses this compatibility path.
CAPTION_IMPORT_DIGESTS = {
    "subtitle_session_observations.py": (
        "06c635f06115c7142cbd748002715d032ebdb91853c17b1e4b9b5e6802f8d84e"
    ),
    "caption_linguistics.py": "58649a8efa83eb7abd75953a421a3f4df72359f9c0ad0005c13194cbf97d732d",
    "caption_style_defaults.py": "3f6842e81fd9025f07d9ad679a91db69c7947d0377960600d7e8e00b2151a220",
    "caption_timing.py": "a444cc7b353794f69a38af73e0c4dd98630bc6a9922f625c545c3752aa0ae23e",
    "motion_caption.py": "477a3c636122f5b796a619b70930ac739881bb39f2c5fd425222401b355cea06",
    "motion_entrance.py": "54dffd8f4824484b755460dcf0474d0b18f3e52c77984e1a923849efaa657490",
    "motion_stages.py": "cba15ebd82c97a8ba8c309f152e94c2bb78747e1bdecac5a4d7198f53ec6ea02",
    "code_generation_compat.py": "0d650cac0cff24467417088410e0f5b6c22b41083b610af49f87eb74e5317c35",
}

# Files whose bytes changed across the known generations below. Every other
# closure member must be byte-identical between the generations.
RECOVERY_FILES = frozenset(
    {
        "subtitle_session_cleanup.py",
        "subtitle_session_cleanup_plan.py",
        "subtitle_session_cleanup_wave.py",
        "cleanup_lane_successor.py",
        *CAPTION_IMPORT_DIGESTS,
    }
)

# sha256 of the recovery files per known production generation, corroborated
# across every operation recorded in that generation:
# - "pre-recovery": chunks 2..91 era, before the interrupted-reservation
#   capability (e.g. continuation-v1/wave-*/.../wave-operation.json).
# - "recovery-wave": chunks 92..176 era, with the recovery capability but
#   before this compatibility rule (e.g. v4w-*/.../wave-operation.json and
#   wave-00092-00093-recovery-v3/.../wave-operation.json).
PRE_RECOVERY_DIGESTS = {
    **CAPTION_IMPORT_DIGESTS,
    "subtitle_session_cleanup.py": (
        "7bb6ce8bd632c772dc21c67733a29edc63f01f374c5e54fc111586022c8534a6"
    ),
    "subtitle_session_cleanup_plan.py": (
        "88df43e6bda6145ffa22541ac2c0b6e3238f16cdabcf243ae986d7d7efceef93"
    ),
    "subtitle_session_cleanup_wave.py": (
        "050b827b613dbd9f409638d7d2bc9b4303c51285ae03c565d78a08d6bbea6e17"
    ),
    "cleanup_lane_successor.py": (
        "5ed2589ee03fccd7ce2406df0ea0f9d462f46c4603e9919e0a0cf7eaf6678b3d"
    ),
}

RECOVERY_WAVE_DIGESTS = {
    **CAPTION_IMPORT_DIGESTS,
    "subtitle_session_cleanup.py": (
        "82420ae0bd13085bed1b388d9cfed3dd025d520102525b3d4f98f68e9ac0e287"
    ),
    "subtitle_session_cleanup_plan.py": (
        "b8fdfda56514140a4b5a250afb1ce2d865457a10c83369736f9e35cb98b9b617"
    ),
    "subtitle_session_cleanup_wave.py": (
        "d399e1f1a105335f3a170f67cfaef33446904c2f43a6f026b7c2f88b0f964461"
    ),
    "cleanup_lane_successor.py": (
        "5ed2589ee03fccd7ce2406df0ea0f9d462f46c4603e9919e0a0cf7eaf6678b3d"
    ),
}


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _accepted_digests(name: str, current: str | None) -> frozenset[str]:
    known = {
        PRE_RECOVERY_DIGESTS[name],
        RECOVERY_WAVE_DIGESTS[name],
    }
    if current is not None:
        known = known | {current}
    return frozenset(known)


def historical_generation(recorded: object, current: object) -> bool:
    """True when recorded is a known production generation.

    The recorded key set must match current except that it may lack this very
    module (recorded before it existed). Every other file must be
    byte-identical to current, except the recovery files, which must match
    the current bytes or one of the pinned known generations. Never raises.
    """
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    try:
        recorded_keys = set(recorded)
        current_keys = set(current)
    except TypeError:
        return False
    if any(not isinstance(path, str) for path in recorded_keys | current_keys):
        return False
    if recorded_keys - current_keys:
        return False
    if any(_basename(path) != COMPAT_BASENAME for path in current_keys - recorded_keys):
        return False
    for path in recorded_keys:
        digest = recorded[path]
        if not isinstance(digest, str):
            return False
        name = _basename(path)
        if name in RECOVERY_FILES:
            current_digest = current[path] if isinstance(current.get(path), str) else None
            if digest not in _accepted_digests(name, current_digest):
                return False
        elif digest != current[path]:
            return False
    return True


def _bounded_sha(path: str) -> str:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError("Recorded code must be a regular file")
    data = candidate.read_bytes()
    if len(data) > MAX_PIN_BYTES:
        raise ValueError("Recorded code exceeds the bounded pin size")
    return hashlib.sha256(data).hexdigest()


def verify_recorded_pins(recorded: object, current: object) -> None:
    """Re-read code files after a historical generation was accepted.

    Entries pinned at a known historical generation prove the operation was
    built by reviewed code; the running files must then match the current
    generation. Every other entry must still match the bytes on disk.
    """
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        raise ValueError("Recorded code pins must be explicit text digests")
    for path, digest in recorded.items():
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("Recorded code pins must be explicit text digests")
        name = _basename(path)
        current_digest = current.get(path) if isinstance(current.get(path), str) else None
        if name in RECOVERY_FILES and digest in _accepted_digests(name, current_digest):
            if path not in current or _bounded_sha(path) != current[path]:
                raise ValueError("Running recovery code differs from its recorded generation")
        elif _bounded_sha(path) != digest:
            raise ValueError("Recorded code changed")


def accepted_input_digest(path: object, digest: object, current_digest: str) -> bool:
    """True when a recorded receipt input is a reviewed recovery file at a known generation.

    Receipt-recorded code inputs identify which code produced the artifact;
    data inputs (media, models, metadata) must always match byte-for-byte and
    are never accepted here.
    """
    if not isinstance(path, str) or not isinstance(digest, str):
        return False
    name = _basename(path)
    return name in RECOVERY_FILES and digest in _accepted_digests(name, current_digest)


def historical_stage_input_sets(inputs: Mapping[str, str]) -> list[dict[str, str]]:
    """Candidate input-digest maps, one per known historical generation.

    A receipt fingerprint match against any candidate proves the stage
    definition and every data input are unchanged and only reviewed recovery
    files differ. Returns [] when the stage does not depend on the recovery
    files, so the caller keeps the strict current-generation check.
    """
    if not any(_basename(path) in RECOVERY_FILES for path in inputs):
        return []
    candidates = []
    for generation in (PRE_RECOVERY_DIGESTS, RECOVERY_WAVE_DIGESTS):
        candidates.append(
            {path: generation.get(_basename(path), digest) for path, digest in inputs.items()}
        )
    unique: list[dict[str, str]] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique
