"""Read pinned foreground evidence without changing its producer or granting approval."""

import hashlib
import json
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

from replay import foreground_collection as collection
from replay import tracking_observations as storage
from replay.foreground_observations import MAX_SAMPLES, ForegroundFrame, foreground_frames
from replay.recordings import record


@dataclass(frozen=True, slots=True)
class ForegroundWindow:
    job: collection.ForegroundJob
    frames: tuple[ForegroundFrame, ...]
    provenance_sha256: str


def _bound_bytes(path: Path, digest: str, limit: int) -> bytes:
    expected = storage._digest(digest)
    data = storage._read(path, limit)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Pinned foreground evidence changed")
    return data


def read_window(
    config: Path,
    output: Path,
    *,
    config_sha256: str,
    provenance_sha256: str,
) -> ForegroundWindow:
    """Read a completed window using independently pinned config/receipt hashes.

    Reuse the unchanged producer's verification; parse the very response bytes
    whose digests the pinned receipt records. Recheck source, executable, producer
    code and pinned files before returning. No collection, adoption or writes.
    Source-clock provenance and semantic interpretation remain caller obligations;
    a requested-window receipt is not proof of full-video coverage.
    """
    job = collection.load_job(config)
    if job.config_hash != storage._digest(config_sha256):
        raise ValueError("Pinned foreground configuration changed")
    output = storage._checked_output(output)
    provenance = output / "provenance.json"
    receipt = record(json.loads(_bound_bytes(provenance, provenance_sha256, storage.CONFIG_BYTES)))
    collection.verify(config, output)
    batches = receipt.get("batches")
    count = (len(job.clock.pts) + MAX_SAMPLES - 1) // MAX_SAMPLES
    if not isinstance(batches, list) or len(batches) != count:
        raise ValueError("Pinned foreground batch count differs")
    frames: list[ForegroundFrame] = []
    chunks = zip(batched(job.clock.pts, MAX_SAMPLES), batches, strict=True)
    for index, (ticks, batch) in enumerate(chunks):
        digest = storage._digest(record(batch).get("responseSHA256"))
        data = _bound_bytes(
            output / f"batch-{index:04d}" / "response.json", digest, collection.RESPONSE_BYTES
        )
        frames.extend(
            foreground_frames(
                json.loads(data),
                times=tuple(tick * job.clock.time_base for tick in ticks),
                width=job.width,
                height=job.height,
            )
        )
    if collection._bindings(job) != record(receipt.get("inputs")):
        raise ValueError("Foreground inputs changed during evidence reading")
    _bound_bytes(config, config_sha256, storage.CONFIG_BYTES)
    _bound_bytes(provenance, provenance_sha256, storage.CONFIG_BYTES)
    return ForegroundWindow(job, tuple(frames), provenance_sha256)
