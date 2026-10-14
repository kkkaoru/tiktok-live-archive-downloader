"""Bind an immutable audio-only revision without rewriting original execution."""

from pathlib import Path

from replay import native_delivery_evidence as evidence
from replay.stages import file_digest


def validate(
    value: object,
    *,
    descriptor: object,
    movie: object,
    reader: evidence.MetadataReader,
) -> dict[str, object]:
    """Recompute the pinned proof; callers still verify current candidate bytes."""
    # Deferred because the original proof reports native completion gap names.
    from replay import native_audio_revision_evidence as audio

    data = evidence.mapping(value)
    if (
        set(data) != {"version", "kind", "request", "proof", "codePins"}
        or type(data.get("version")) is not int
        or data["version"] != 1
        or data.get("kind") != "payload-identical-video-audio-revision"
    ):
        raise ValueError("Unsupported derived revision contract")
    code = evidence.hashes(data["codePins"])
    modules = (Path(__file__).resolve(), Path(audio.__file__).resolve())
    if code != {str(path): file_digest(path) for path in modules}:
        raise ValueError("Derived revision verifier code changed")
    proof = reader.read(data["proof"])
    recomputed = audio.assess(data["request"], reader)
    if proof != recomputed:
        raise ValueError("Derived revision proof differs from its pinned request")
    evidence.same(proof.get("originalDescriptor"), descriptor)
    evidence.same(proof.get("candidate"), movie)
    if evidence.reference(proof.get("originalMovie")) == evidence.reference(movie):
        raise ValueError("Derived revision must name a distinct movie")
    if proof.get("numericalAudioMeasurementVerified") is not True:
        raise ValueError("Derived revision lacks numerical audio verification")
    return proof
