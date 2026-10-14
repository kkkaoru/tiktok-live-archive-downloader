"""Read hash-bound Motion/Compressor receipts without inventing legacy render evidence.

This consumes local processing evidence, not authenticated attestations. Recorded
execution, full decode, statistical audio checks and source-removal quality are
separate requirements. Unknown quality remains a failure, never an inferred pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from replay.bgm_separation import SeparationPolicy
from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import SpeakerCue, render_srt
from replay.subtitle_retry import RepairEvidence, RepairMethod, decide_repair

MAX_JSON_BYTES = 64 * 1024 * 1024
MINIMUM_CORRELATION = 0.995
# Receipt paths with this prefix are relative to the recording's storage root.
STORAGE_PREFIX = "storage:"
REFERENCES = ("manifest", "encoding", "sourceEncoding", "audio", "cuts", "masks", "captions", "fcp")
INDEX_KEYS = frozenset(
    ("version", "backend", "recording", "configurationSHA256", "sourceSHA256", *REFERENCES)
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ReplayError("Native evidence needs integer counts")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReplayError("Native evidence needs finite measurements")
    return float(value)


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ReplayError("Invalid native evidence path")
    return Path(value)


def _rows(value: object) -> list[object]:
    if not isinstance(value, list) or not 0 < len(value) <= 50000:
        raise ReplayError("Native evidence needs a nonempty bounded list")
    return value


@dataclass
class Evidence:
    """One check's digest cache; never reused after the check returns."""

    root: Path
    source: Path
    storage: Path | None = None
    hashes: dict[Path, str] = field(default_factory=dict)
    snapshots: dict[Path, tuple[int, int, int]] = field(default_factory=dict)

    @staticmethod
    def identity(path: Path) -> tuple[int, int, int]:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    def verify_unchanged(self) -> None:
        for path, identity in self.snapshots.items():
            _require(self.identity(path) == identity, "Native evidence changed during verification")

    def digest(self, path: Path) -> str:
        resolved = path.resolve(strict=True)
        allowed = (
            resolved.is_relative_to(self.root.parents[1])
            or resolved == self.source
            or (self.storage is not None and resolved.is_relative_to(self.storage))
        )
        _require(allowed and resolved.is_file(), "Native evidence leaves the authorized storage")
        _require(
            not resolved.is_relative_to(self.root.parents[1] / "private"),
            "Native delivery evidence must not read private account storage",
        )
        if resolved not in self.hashes:
            identity = self.identity(resolved)
            with resolved.open("rb") as stream:
                self.hashes[resolved] = hashlib.file_digest(stream, "sha256").hexdigest()
            _require(self.identity(resolved) == identity, "Native input changed while hashing")
            self.snapshots[resolved] = identity
        return self.hashes[resolved]

    def read(self, path: Path) -> dict[str, object]:
        _require(
            not path.is_symlink() and path.is_file() and path.stat().st_size <= MAX_JSON_BYTES,
            "Missing, linked or oversized native evidence",
        )
        self.digest(path)
        value: object = json.loads(path.read_text(encoding="utf-8"))
        _require(isinstance(value, dict), "Native evidence must be an object")
        return record(value)

    def reference(self, descriptor: object) -> tuple[Path, dict[str, object]]:
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ReplayError("Invalid native evidence reference")
        ref = record(descriptor)
        text = _path(ref["path"]).as_posix()
        base = self.root
        if text.startswith(STORAGE_PREFIX):
            _require(self.storage is not None, "Native receipt names an unconfigured storage root")
            base, text = self.storage or self.root, text.removeprefix(STORAGE_PREFIX)
        relative = Path(text)
        path = base / relative
        _require(
            not relative.is_absolute() and path.resolve().is_relative_to(base),
            "Native receipt must belong to this recording",
        )
        data = self.read(path)
        _require(self.digest(path) == ref["sha256"], "Native receipt changed")
        return path, data

    def require_binding(self, data: dict[str, object], path: Path) -> None:
        expected = path.resolve()
        _require(
            any(
                _path(name).resolve() == expected and digest == self.digest(path)
                for name, digest in record(data.get("inputs")).items()
            ),
            "Native audit does not bind the selected manifest/source",
        )

    def bindings(self, data: dict[str, object]) -> None:
        inputs = record(data.get("inputs"))
        _require(0 < len(inputs) <= 10000, "Missing native input bindings")
        for name, digest in inputs.items():
            _require(self.digest(_path(name)) == digest, "Native processing input changed")


def _encoding(
    store: Evidence, movie: Path, manifest: dict[str, object], encoding: dict[str, object]
) -> None:
    _require(
        _integer(encoding.get("frames")) == _integer(manifest.get("totalFrames")) > 0
        and _integer(encoding.get("rate")) == _integer(manifest.get("rate")) > 0
        and _integer(encoding.get("audioSamples")) > 0
        and _integer(encoding.get("outputBytes")) == movie.stat().st_size > 0
        and store.digest(movie) == encoding.get("outputSHA256")
        and all(
            encoding.get(key) is True
            for key in ("fullDecodeVerified", "clocksVerified", "audioPCMIdentical")
        ),
        "Native whole encoding/decode evidence does not match the movie",
    )


def _speech(value: object, policy: SeparationPolicy) -> bool:
    data = record(value)
    return (
        _number(data.get("level_delta_db_p50")) >= -policy.maximum_speech_loss_db
        and _number(data.get("level_delta_db_p05")) >= -policy.maximum_speech_loss_db
        and MINIMUM_CORRELATION <= _number(data.get("correlation")) <= 1
    )


def _audio(data: dict[str, object], *, duration: float) -> bool:
    policy = SeparationPolicy()
    if (
        data.get("status") != "candidate-audio-verified"
        or data.get("policy") != asdict(policy)
        or data.get("minimumSpeechCorrelation") != MINIMUM_CORRELATION
        or any(
            data.get(key) is not True
            for key in ("musicGatePassed", "speechGatePassed", "candidateAudioOnly")
        )
        or not _speech(data.get("speech"), policy)
        or not _speech(data.get("candidateSpeechComparedToOriginalFirstPass"), policy)
    ):
        return False
    probes = _rows(data.get("musicProbes"))
    if len(probes) != math.ceil(duration / 60):
        return False
    for index, raw in enumerate(probes):
        probe = record(raw)
        before, after = _number(probe.get("before_dbfs")), _number(probe.get("after_dbfs"))
        drop = _number(probe.get("drop_db"))
        if (
            probe.get("passed") is not True
            or _number(probe.get("start_seconds")) != index * 60
            or not math.isclose(drop, before - after, rel_tol=0, abs_tol=1e-9)
            or (
                before > policy.quiet_instrumental_db and drop < policy.minimum_instrumental_drop_db
            )
        ):
            return False
    return True


def _motion(
    store: Evidence, path: Path, plan: dict[str, object], layout: dict[str, object]
) -> None:
    motion_hash = store.digest(_path(plan.get("motion")))
    _require(motion_hash == plan.get("motionSHA256"), "Motion project changed")
    response_path = path.parent / "alpha-submit.response.json"
    request_path = _path(plan.get("alphaRequest"))
    if "reuseAlphaResponse" in plan:
        response_path = _path(plan["reuseAlphaResponse"])
        _require(response_path.name.endswith(".response.json"), "Invalid reused alpha response")
        request_path = response_path.with_name(
            response_path.name.removesuffix(".response.json") + ".request.json"
        )
    response = store.read(response_path)
    result = record(response.get("data"))
    native = record(result.get("structuredContent"))
    request = store.read(request_path)
    _require(
        response.get("ok") is True
        and result.get("isError", False) is False
        and _integer(native.get("exitStatus")) == 0
        and request.get("bundleID") == "com.apple.CompressorApp"
        and store.digest(_path(request.get("sourcePath"))) == motion_hash
        and store.digest(_path(native.get("outputPath"))) == layout.get("alphaSHA256"),
        "Motion/Compressor alpha identity is unverified",
    )


def _caption_rows(plan: dict[str, object], *, start: int, end: int, rate: int) -> list[SpeakerCue]:
    cues: list[SpeakerCue] = []
    for raw in _rows(plan.get("captions")):
        row = record(raw)
        first, last, lines = row.get("start"), row.get("end"), _rows(row.get("lines"))
        if not isinstance(first, str) or not isinstance(last, str):
            raise ReplayError("Missing rational caption clock")
        a, b = Fraction(first), Fraction(last)
        text: list[str] = []
        for line in lines:
            if not isinstance(line, str) or not line.strip():
                raise ReplayError("Invalid native caption text")
            text.append(line)
        _require(
            start <= a * rate < b * rate <= end
            and (a * rate).denominator == (b * rate).denominator == 1,
            "Native caption lies outside its frame-aligned part",
        )
        cues.append(SpeakerCue(float(a), float(b), "unassigned", "\n".join(text)))
    return cues


def _parts(store: Evidence, manifest: dict[str, object]) -> list[SpeakerCue]:
    cursor = 0
    cues: list[SpeakerCue] = []
    rate = _integer(manifest.get("rate"))
    for raw in _rows(manifest.get("parts")):
        item = record(raw)
        path = _path(item.get("plan"))
        plan = store.read(path)
        _require(store.digest(path) == item.get("planSHA256"), "Native part plan changed")
        duration = _integer(plan.get("durationFrames"))
        _require(
            _integer(plan.get("startFrame")) == cursor
            and duration > 0
            and _integer(plan.get("rate")) == rate,
            "Native parts are not a contiguous output clock",
        )
        encoded = store.read(path.parent / "encoding-verified.json")
        output = _path(encoded.get("output"))
        _require(
            encoded.get("planSHA256") == store.digest(path)
            and _integer(encoded.get("frames")) == duration
            and _integer(encoded.get("rate")) == rate
            and encoded.get("fullDecodeVerified") is True
            and encoded.get("clocksVerified") is True
            and encoded.get("outputSHA256") == store.digest(output)
            and _integer(encoded.get("outputBytes")) == output.stat().st_size,
            "Native encoded part differs from its verified plan",
        )
        layout = store.read(path.parent / "alpha-layout-audit.json")
        _require(
            layout.get("planSHA256") == store.digest(path)
            and layout.get("boundsAndGapsVerified") is True
            and _integer(layout.get("fullCaptionStates")) > 0
            and _integer(layout.get("sampleCount")) > 0,
            "Native Motion caption layout is unverified",
        )
        _motion(store, path, plan, layout)
        cues.extend(_caption_rows(plan, start=cursor, end=cursor + duration, rate=rate))
        cursor += duration
    _require(cursor == manifest.get("totalFrames"), "Native parts omit output frames")
    return cues


def _mask_failure(store: Evidence, masks: dict[str, object], frames: int) -> str | None:
    store.bindings(masks)
    if (
        masks.get("recordedExecutionMatchesPlan") is not True
        or _integer(masks.get("selectedFrames")) != frames
        or _integer(masks.get("selectedMaskFrames")) <= 0
        or any(
            type(masks.get(key)) is not int or masks.get(key) != 0
            for key in ("missingMaskFrames", "extraMaskFrames", "includedEmptyDetections")
        )
    ):
        return "native source-mask execution is incomplete"
    if masks.get("sourceCaptionQualityApproved") is not True:
        return (
            "native source-mask accounting passed; independent removal quality remains unverified"
        )
    quality = store.reference(masks.get("qualityEvidence"))[1]
    store.bindings(quality)
    _require(
        all(
            record(quality.get("inputs")).get(name) == digest
            for name, digest in record(masks.get("inputs")).items()
        ),
        "Source-removal quality does not bind every processed input",
    )
    _require(
        _integer(quality.get("measuredFrames")) == masks.get("selectedMaskFrames"),
        "Source-removal quality omits selected masked frames",
    )
    evidence = RepairEvidence(
        technically_valid=_integer(quality.get("technicallyInvalidFrames")) == 0,
        residual_fraction=_number(quality.get("maximumResidualFraction")),
        unstable_fraction=_number(quality.get("maximumUnstableFraction")),
        readable_text=_integer(quality.get("readableFrames")) != 0,
    )
    decision = decide_repair(RepairMethod.GLYPH, evidence)
    return (
        None
        if decision.accepted
        else "native source-removal quality fails shared repair thresholds"
    )


def assess(
    index: Path, *, delivery: Path, root: Path, source: Path, storage: Path | None = None
) -> dict[str, str]:
    """Assess a private candidate or a published movie using the same native contract.

    The caller validates the recording config first. Malformed/stale common
    evidence raises ReplayError; measured stage failures are returned explicitly.
    This does not publish a file, change approvals or run inference/rendering.
    """
    store = Evidence(root.resolve(), source.resolve(strict=True), storage)
    data = store.read(index)
    _require(
        set(data) == INDEX_KEYS
        and type(data.get("version")) is int
        and data["version"] == 1
        and data.get("backend") == "motion-compressor"
        and data.get("recording") == root.name
        and data.get("configurationSHA256") == store.digest(root / "edit-config.json")
        and data.get("sourceSHA256") == store.digest(source),
        "Native evidence index does not match this recording/configuration/source",
    )
    references = {key: store.reference(data[key]) for key in REFERENCES}
    manifest_path, manifest = references["manifest"]
    encoded = references["encoding"][1]
    original_path, original = references["sourceEncoding"]
    audio_path, audio = references["audio"]
    _encoding(store, delivery, manifest, encoded)
    _encoding(store, _path(original.get("output")), manifest, original)
    _require(
        encoded.get("manifestSHA256")
        == original.get("manifestSHA256")
        == store.digest(manifest_path)
        and encoded.get("sourceVerificationSHA256")
        == audio.get("wholeVerificationSHA256")
        == store.digest(original_path)
        and encoded.get("audioVerificationSHA256") == store.digest(audio_path)
        and encoded.get("videoPayloadIdentical") is True
        and audio.get("sourceVideoSHA256") == original.get("outputSHA256")
        and encoded.get("editedVoiceSHA256")
        == audio.get("candidateVoiceSHA256")
        == store.digest(_path(encoded.get("editedVoice")))
        and _path(encoded.get("editedVoice")).resolve()
        == _path(audio.get("candidateVoice")).resolve(),
        "Native audio/whole-video identity chain is broken",
    )
    failures: dict[str, str] = {}
    frames, rate = _integer(manifest.get("totalFrames")), _integer(manifest.get("rate"))
    if not _audio(audio, duration=frames / rate):
        failures["bgm_removal"] = "native audio measurements fail the unchanged music/speech policy"
    cuts = references["cuts"][1]
    store.require_binding(cuts, manifest_path)
    store.bindings(cuts)
    if (
        cuts.get("selectionPlanMatchesProtectedCuts") is not True
        or _integer(cuts.get("selectedFrames")) != frames
        or _integer(cuts.get("sourceFrames")) - frames != _integer(cuts.get("removedFrames"))
        or _integer(cuts.get("removedFrames")) <= 0
    ):
        failures["silence_cut"] = "native selections do not establish protected non-speech cuts"
    masks = references["masks"][1]
    store.require_binding(masks, manifest_path)
    store.require_binding(masks, source)
    mask = _mask_failure(store, masks, frames)
    if mask is not None:
        failures["source_caption_mask"] = mask
    captions = references["captions"][1]
    cues = _parts(store, manifest)
    srt = delivery.with_suffix(".srt")
    if (
        captions.get("manifestSHA256") != store.digest(manifest_path)
        or captions.get("srtSHA256") != store.digest(srt)
        or _integer(captions.get("cueCount")) != len(cues)
        or _integer(captions.get("outputFrames")) != frames
        or _integer(captions.get("rate")) != rate
        or captions.get("textConserved") is not True
        or captions.get("nativeFrameAligned") is not True
        or srt.read_text(encoding="utf-8") != render_srt(tuple(cues))
    ):
        failures["transcription"] = "native transcript does not match the burned caption plans"
    fcp = references["fcp"][1]
    if (
        fcp.get("importObserved") is not True
        or fcp.get("noModalErrorObserved") is not True
        or fcp.get("xmlSHA256") != store.digest(_path(fcp.get("xml")))
    ):
        failures["dynamic_captions"] = "native editable FCP import was not observed"
    store.verify_unchanged()
    return failures
