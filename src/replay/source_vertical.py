"""A copy of a wide source cropped to a narrower aspect around the tracked character.

Applied only when the recording carries the editing intent ``edits/<recording>/reframe.json``
(``{"version": 1, "aspect": [9, 16], "follow": "character"}``). The trained detector
(``replay.yolo_track``) finds the character's face on sampled frames; samples without a face use
the whole figure (Vision human rectangles, ``video_head_detect``) shifted toward the side its face
points to, so a profile keeps its face and may lose the back of the head; samples with neither hold
the last position (a leading gap takes the first one, a video without any face stays centred). A
face far from the track counts only once it persists. A virtual camera turns those centres into one
crop offset per frame: a median over about half a second drops stray detections, a dead zone keeps
a still character's frame still, and a speed cap with eased acceleration and braking moves it
smoothly; the crop never leaves the frame. The crop keeps the source height (width
``even(round(height * 9 / 16))``) and is never scaled: composition scales to the delivery canvas
last. Every frame keeps its PTS; video is HEVC as in ``source_decodable``, audio is copied or made
ALAC as there.

The derived file keeps the source's name in its own directory, beside the tracking evidence and
a record (source size and SHA-256, evidence SHA-256, crop parameters, code version, output
SHA-256); a rerun reuses it while its inputs and output still match the record, whatever camera
settings made it.
"""

import json
import math
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

from replay import source_decodable as decodable
from replay.core import ReplayError
from replay.recordings import record
from replay.source_lead_in import BITRATE, COLOUR, sha256, video_pts, video_stream
from replay.yolo_track import probe_video

DIRECTORY = "source-vertical"
INTENT = "reframe.json"
RECORD = "vertical.json"
EVIDENCE = "track.json"
BODIES = "bodies.json"
CODE_VERSION = "vertical-reframe-v5"
IDENTITY = ("source", "sourceSize", "sourceSHA256", "evidenceSHA256", "bodiesSHA256", "crop",
            "output")  # fmt: skip
INTENT_KEYS = frozenset({"version", "aspect", "follow"})
# Camera tuning, in crop widths and seconds so it holds for any size and frame rate.
MEDIAN_SECONDS = 1.5
DEAD_ZONE = 0.15
SETTLE = 0.02
MAX_SPEED = 1.0
RAMP_SECONDS = 0.5
# Track gating: a face below RELIABLE confidence is a fallback; one farther than JUMP crop widths
# from the track is a gap until CONFIRM_SECONDS of samples agree on its new place.
RELIABLE = 0.5
JUMP = 0.5
CONFIRM_SECONDS = 0.5
KINDS = ("reliable", "fallback", "extrapolated")
# Faceless samples (a profile or a turned back) look for the whole figure (Vision human
# rectangles) every BODY_SECONDS; the face side of the figure is kept FACE_MARGIN crop widths
# inside the crop once the figure's centre is FACING_MIN crop widths off the last face.
BODY_SECONDS = 0.5
BODY_BATCH = 8
FACING_MIN = 0.05
FACE_MARGIN = 0.05
FACING_STRENGTH = 0.5
FIGURE_DEAD_ZONE = 0.10


def intent(edit_root: Path) -> tuple[int, int] | None:
    """The recording's requested crop aspect (width, height), or None without the intent file."""
    path = edit_root / INTENT
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or set(data) != INTENT_KEYS
        or data["version"] != 1
        or data["aspect"] != [9, 16]
        or data["follow"] != "character"
    ):
        raise ReplayError(f"{INTENT} must be {{version: 1, aspect: [9, 16], follow: character}}")
    return 9, 16


def crop_width(width: int, height: int, aspect: tuple[int, int]) -> int:
    crop = 2 * round(height * aspect[0] / aspect[1] / 2)
    if not 2 <= crop < width:
        raise ReplayError("The source is not wider than the requested aspect")
    return crop


def face_centres(evidence: dict[str, object]) -> list[tuple[int, float | None, float]]:
    """(frame, face centre x, confidence) per sampled frame of ``yolo_track`` evidence."""
    frames = evidence.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ReplayError("The tracking evidence has no frames")
    centres: list[tuple[int, float | None, float]] = []
    for row in frames:
        item = record(row)
        face = item.get("face")
        if face is not None and (not isinstance(face, list) or len(face) != 5):
            raise ReplayError("Malformed face box in the tracking evidence")
        frame = item.get("frame")
        if type(frame) is not int:
            raise ReplayError("Malformed frame in the tracking evidence")
        centre = None if face is None else (float(face[0]) + float(face[2])) / 2
        centres.append((frame, centre, 0.0 if face is None else float(face[4])))
    return centres


def facing(body: tuple[float, float], face: float | None, *, reach: float, least: float) -> float:
    """Where to centre on a faceless figure spanning `body` (left, right) so its face stays in.

    Turning to a profile moves the figure's mass toward the side the face points to while the
    back of the head stays near the last frontal face, so the figure's centre relative to that
    face gives the facing side; the centre is then pulled toward that edge until the edge is
    `reach` from it (the back of the head may leave the crop), and the result lies
    ``FACING_STRENGTH`` of the way from the last face to that place (the user's choice: midway
    keeps both the face and most of the head). Without a clear side it is the figure's centre.
    """
    left, right = body
    middle = (left + right) / 2
    if face is None or abs(middle - face) < least:
        return middle
    edge = min(middle, left + reach) if middle < face else max(middle, right - reach)
    return face + FACING_STRENGTH * (edge - face)


def gated(
    faces: list[tuple[int, float | None, float]],
    *,
    jump: float,
    confirm: int,
    bodies: Mapping[int, tuple[float, float]],
    crop: int,
) -> tuple[list[tuple[int, float | None]], list[str]]:
    """Faces consistent with the track, and each sample's kind; others become gaps.

    A face farther than `jump` from the last accepted one is held back until `confirm` such
    faces agree within `jump` (the character really moved, e.g. during a profile stretch); a
    face near the track meanwhile drops them as strays. A sample without a usable face uses its
    figure (`bodies`, by frame) placed by ``facing`` as a fallback; the remaining gaps are filled
    by ``targets``.
    """
    reach, least = crop / 2 - FACE_MARGIN * crop, FACING_MIN * crop
    track: float | None = None
    pending: list[float] = []
    kept: list[tuple[int, float | None]] = []
    kinds: list[str] = []
    for frame, centre, score in faces:
        if centre is not None and track is not None and abs(centre - track) > jump:
            pending = (
                [*pending, centre] if pending and abs(centre - pending[0]) <= jump else [centre]
            )
            centre = centre if len(pending) >= confirm else None
        if centre is None:
            body = bodies.get(frame)
            figure = None if body is None else facing(body, track, reach=reach, least=least)
            kept.append((frame, figure))
            kinds.append(KINDS[2] if figure is None else KINDS[1])
            continue
        track, pending = centre, []
        kept.append((frame, centre))
        kinds.append(KINDS[0] if score >= RELIABLE else KINDS[1])
    return kept, kinds


def sample_shares(kinds: list[str], *, seconds_per_sample: float) -> dict[str, float]:
    """Share of samples of each kind and the longest run of extrapolated samples, in seconds."""
    longest = run = 0
    for kind in kinds:
        run = run + 1 if kind == KINDS[2] else 0
        longest = max(longest, run)
    shares = {kind: round(kinds.count(kind) / len(kinds), 4) for kind in KINDS}
    return {**shares, "longestGapSeconds": round(longest * seconds_per_sample, 3)}


def targets(
    centres: list[tuple[int, float | None]], frames: int, *, width: int, window: int
) -> list[float]:
    """Per-frame target centre: held between samples, median-filtered over `window` samples.

    A gap holds the last position (no velocity is carried into it); a leading gap takes the first
    found one; only a video without any face stays centred.
    """
    found = [centre for _, centre in centres if centre is not None]
    held: list[float] = []
    last = found[0] if found else width / 2
    for _, centre in centres:
        last = last if centre is None else centre
        held.append(last)
    half = window // 2
    smooth = [
        statistics.median(held[max(0, index - half) : index + half + 1])
        for index in range(len(held))
    ]
    result: list[float] = []
    sample = 0
    for frame in range(frames):
        while sample + 1 < len(centres) and centres[sample + 1][0] <= frame:
            sample += 1
        result.append(smooth[sample])
    return result


def figure_frames(
    points: list[tuple[int, float | None]], kinds: list[str], frames: int
) -> list[bool]:
    """Per frame, whether its place last came from a fallback (a figure) rather than a face."""
    modes: list[bool] = []
    last = False
    for kind in kinds:
        last = last if kind == KINDS[2] else kind == KINDS[1]
        modes.append(last)
    result: list[bool] = []
    sample = 0
    for frame in range(frames):
        while sample + 1 < len(points) and points[sample + 1][0] <= frame:
            sample += 1
        result.append(modes[sample])
    return result


def median_window(rate: float, step: int) -> int:
    """Odd number of samples spanning about ``MEDIAN_SECONDS``."""
    return 2 * round(MEDIAN_SECONDS * rate / step / 2) + 1


def camera(
    target: list[float],
    *,
    width: int,
    crop: int,
    rate: float,
    loose: Sequence[bool] | None = None,
) -> list[int]:
    """Even left crop offsets per frame following `target` without jitter or jumps.

    Frames marked `loose` (placed from a figure) use the narrower ``FIGURE_DEAD_ZONE``: a
    figure's place is already steady, and the facing shift is smaller than the face dead zone.
    """
    settle = SETTLE * crop
    top = MAX_SPEED * crop / rate
    accel = top / (RAMP_SECONDS * rate)
    low, high = crop / 2, width - crop / 2
    position = min(max(target[0], low), high) if target else low
    velocity, moving = 0.0, False
    offsets: list[int] = []
    for index, goal in enumerate(target):
        dead = (FIGURE_DEAD_ZONE if loose is not None and loose[index] else DEAD_ZONE) * crop
        error = min(max(goal, low), high) - position
        moving = abs(error) > (settle if moving else dead)
        # The fastest speed from which braking by `accel` per frame stops at the target.
        brake = accel * (math.sqrt(1 + 8 * abs(error) / accel) - 1) / 2
        wanted = math.copysign(min(top, brake), error) if moving else 0.0
        velocity += min(max(wanted - velocity, -accel), accel)
        position = min(max(position + velocity, low), high)
        if position in (low, high):
            velocity = 0.0
        offsets.append(min(2 * round((position - crop / 2) / 2), 2 * ((width - crop) // 2)))
    return offsets


def commands(offsets: list[int], pts: list[int], time_base: Fraction) -> str:
    """``sendcmd`` lines setting the crop offset half a frame before each frame that moves it.

    The first line sets the first frame's offset, so the list is never empty (``sendcmd`` refuses
    an empty one).
    """
    lines = [f"0.000000 crop x {offsets[0]};"] + [
        f"{float((pts[index - 1] + pts[index]) / 2 * time_base):.6f} crop x {offsets[index]};"
        for index in range(1, len(offsets))
        if offsets[index] != offsets[index - 1]
    ]
    return "\n".join(lines) + "\n"


def render_command(
    source: Path,
    output: Path,
    script: Path,
    stream: dict[str, object],
    audio: dict[str, object],
    *,
    crop: tuple[int, int, int],
    ffmpeg: str,
) -> list[str]:
    if "'" in str(script):
        raise ReplayError("The crop command path must not contain a quote")
    colour = [
        argument
        for key, option in COLOUR
        if isinstance(stream.get(key), str) and stream[key] != "unknown"
        for argument in (option, str(stream[key]))
    ]
    audio_codec = (
        ["-c:a", "copy"]
        if audio.get("codec_name") in decodable.AUDIO_CODECS
        else ["-c:a", "alac", "-sample_fmt", "s32p"]
    )
    width, height, first = crop
    return [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(source), "-map", "0:v", "-map", "0:a",
        "-vf", f"sendcmd=f='{script}',crop=w={width}:h={height}:x={first}:y=0",
        "-fps_mode", "passthrough", "-enc_time_base", str(Fraction(str(stream["time_base"]))),
        "-c:v", "hevc_videotoolbox", "-b:v", BITRATE, "-tag:v", "hvc1", *colour, *audio_codec,
        "-movflags", "+faststart", str(output),
    ]  # fmt: skip


def track(
    source: Path,
    output: Path,
    *,
    frames: int,
    python: str,
    yolo: dict[str, object],
    tools: dict[str, str],
) -> None:
    """Run the pinned detector over the source into `output` (``replay.yolo_track``)."""
    partial = output.with_name(f"partial-{output.name}")
    partial.unlink(missing_ok=True)
    subprocess.run(
        [python, "-B", "-m", "replay.yolo_track", str(source), str(partial),
         "--weights", str(yolo["weights"]), "--weights-sha256", str(yolo["sha256"]),
         "--config-directory", str(yolo["configDirectory"]),
         "--work", str(Path.cwd().resolve() / "edits"), "--end", str(frames),
         "--step", str(yolo["step"]), "--ffmpeg", tools["ffmpeg"], "--ffprobe", tools["ffprobe"]],
        check=True,
    )  # fmt: skip
    partial.rename(output)


def body_frames(faces: list[tuple[int, float | None, float]], *, every: int) -> list[int]:
    """Faceless sampled frames to look for the figure at, at least `every` frames apart."""
    chosen: list[int] = []
    for frame, centre, _ in faces:
        if centre is None and (not chosen or frame - chosen[-1] >= every):
            chosen.append(frame)
    return chosen


def detect_bodies(
    source: Path, frames: list[int], seconds: list[float], *, native: str, directory: Path
) -> list[dict[str, object]]:
    """Vision's whole-figure rectangles at each frame (``video_head_detect``, in batches)."""
    request = directory / "bodies-request.json"
    found: list[dict[str, object]] = []
    for start in range(0, len(frames), BODY_BATCH):
        chunk = frames[start : start + BODY_BATCH]
        times = [{"timeSeconds": seconds[frame]} for frame in chunk]
        request.write_text(
            json.dumps({"path": str(source.resolve()), "samples": times}), encoding="utf-8"
        )
        result = subprocess.run(
            [native, "measure-media", "video_head_detect", str(request.resolve())],
            capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL,
        )  # fmt: skip
        measured = json.loads(result.stdout)
        if not isinstance(measured, list) or len(measured) != len(chunk):
            raise ReplayError("The figure detector returned a different number of frames")
        found += [
            {"frame": frame, "bodies": record(row).get("bodies")}
            for frame, row in zip(chunk, measured, strict=True)
        ]
    request.unlink(missing_ok=True)
    return found


def area(box: dict[str, object]) -> float:
    return float(str(box["width"])) * float(str(box["height"]))


def body_spans(evidence: dict[str, object]) -> dict[int, tuple[float, float]]:
    """(left, right) of each frame's largest figure in figure evidence."""
    rows = evidence.get("frames")
    if not isinstance(rows, list):
        raise ReplayError("Malformed figure evidence")
    spans: dict[int, tuple[float, float]] = {}
    for row in rows:
        item = record(row)
        boxes = item.get("bodies")
        if not isinstance(boxes, list) or type(item.get("frame")) is not int:
            raise ReplayError("Malformed figure evidence")
        sizes = [record(box) for box in boxes]
        if sizes:
            box = max(sizes, key=area)
            left = float(str(box["x"]))
            spans[int(str(item["frame"]))] = (left, left + float(str(box["width"])))
    return spans


def bodies_for(
    source: Path,
    directory: Path,
    digest: str,
    *,
    frames: list[int],
    seconds: list[float],
    native: str,
) -> Path:
    """The figure evidence for this source at `frames`, made once."""
    path = directory / BODIES
    if path.exists():
        found = record(json.loads(path.read_text(encoding="utf-8")))
        if found.get("sourceSHA256") == digest and found.get("requested") == frames:
            return path
        path.unlink()
    rows = detect_bodies(source, frames, seconds, native=native, directory=directory)
    partial = directory / f"partial-{BODIES}"
    partial.write_text(
        json.dumps({"sourceSHA256": digest, "requested": frames, "frames": rows}), encoding="utf-8"
    )
    partial.rename(path)
    return path


def evidence_for(
    source: Path,
    directory: Path,
    digest: str,
    *,
    frames: int,
    python: str,
    yolo: dict[str, object],
    tools: dict[str, str],
) -> Path:
    """The tracking evidence for this source and checkpoint, made once."""
    path = directory / EVIDENCE
    if path.exists():
        found = record(json.loads(path.read_text(encoding="utf-8")))
        if (
            record(found.get("source")).get("sha256") == digest
            and record(found.get("weights")).get("sha256") == yolo["sha256"]
        ):
            return path
        path.unlink()
    track(source, path, frames=frames, python=python, yolo=yolo, tools=tools)
    return path


def prepared(
    source: Path,
    directory: Path,
    *,
    ffmpeg: str,
    ffprobe: str,
    python: str,
    yolo: dict[str, object],
    native: str,
    aspect: tuple[int, int] = (9, 16),
) -> Path:
    """The verified character-following crop of `source` at `aspect`."""
    if directory.resolve() == source.resolve().parent:
        raise ReplayError("The vertical copy needs a directory apart from its source")
    stream = video_stream(source, ffprobe)
    _, audio = decodable.streams(source, ffprobe)
    width, height, rate = probe_video(source, ffprobe)
    crop = crop_width(width, height, aspect)
    output, receipt = directory / source.name, directory / RECORD
    digest = sha256(source)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    tools = {"ffmpeg": ffmpeg, "ffprobe": ffprobe}
    pts = video_pts(source, ffprobe)
    evidence = evidence_for(
        source, directory, digest, frames=len(pts), python=python, yolo=yolo, tools=tools
    )
    loaded = record(json.loads(evidence.read_text(encoding="utf-8")))
    step = int(str(loaded.get("step")))
    faces = face_centres(loaded)
    time_base = Fraction(str(stream["time_base"]))
    figures = bodies_for(
        source,
        directory,
        digest,
        frames=body_frames(faces, every=max(1, round(BODY_SECONDS * rate))),
        seconds=[float(value * time_base) for value in pts],
        native=native,
    )
    expected = {
        "source": str(source),
        "sourceSize": source.stat().st_size,
        "sourceSHA256": digest,
        "evidenceSHA256": sha256(evidence),
        "bodiesSHA256": sha256(figures),
        "crop": {"width": crop, "height": height, "aspect": list(aspect)},
        "camera": {
            "medianSeconds": MEDIAN_SECONDS,
            "deadZone": DEAD_ZONE,
            "settle": SETTLE,
            "maxSpeed": MAX_SPEED,
            "rampSeconds": RAMP_SECONDS,
            "reliable": RELIABLE,
            "jump": JUMP,
            "confirmSeconds": CONFIRM_SECONDS,
            "bodySeconds": BODY_SECONDS,
            "facingMin": FACING_MIN,
            "faceMargin": FACE_MARGIN,
            "facingStrength": FACING_STRENGTH,
            "figureDeadZone": FIGURE_DEAD_ZONE,
        },  # fmt: skip
        "codeVersion": CODE_VERSION,
        "output": str(output),
    }
    if receipt.exists():
        recorded = record(json.loads(receipt.read_text(encoding="utf-8")))
        # A copy is bound to its inputs; the camera and code it records made it, and a change
        # of either applies to new copies only (move a copy aside to remake it).
        kept = {key: recorded.get(key) for key in IDENTITY}
        if kept != {key: expected[key] for key in IDENTITY} or recorded.get(
            "outputSHA256"
        ) != sha256(output):
            raise ReplayError("The vertical copy no longer matches its record; remove it to redo")
        made = recorded.get("camera")
        if (
            not isinstance(made, dict)
            or set(made) != set(record(expected["camera"]))
            or not all(type(value) in (int, float) for value in made.values())
            or not isinstance(recorded.get("codeVersion"), str)
        ):
            raise ReplayError("The vertical copy's record names no camera constants; remove it")
        if made != expected["camera"] or recorded["codeVersion"] != CODE_VERSION:
            print(
                f"Reused vertical copy made by {recorded['codeVersion']} with camera "
                f"{json.dumps(made, sort_keys=True)}; current {CODE_VERSION} defaults "
                f"{json.dumps(expected['camera'], sort_keys=True)} apply to new copies only",
                file=sys.stderr,
            )
        return output
    window = median_window(float(rate), step)
    points, kinds = gated(
        faces,
        jump=JUMP * crop,
        confirm=max(1, round(CONFIRM_SECONDS * rate / step)),
        bodies=body_spans(record(json.loads(figures.read_text(encoding="utf-8")))),
        crop=crop,
    )
    offsets = camera(
        targets(points, len(pts), width=width, window=window),
        width=width,
        crop=crop,
        rate=float(rate),
        loose=figure_frames(points, kinds, len(pts)),
    )
    script, partial = directory / "crop-commands.txt", directory / f"partial-{output.name}"
    partial.unlink(missing_ok=True)
    lines = commands(offsets, pts, time_base)
    script.write_text(lines, encoding="utf-8")
    subprocess.run(
        render_command(
            source, partial, script, stream, audio, crop=(crop, height, offsets[0]), ffmpeg=ffmpeg
        ),
        check=True,
    )
    if video_pts(partial, ffprobe) != pts:
        raise ReplayError("The vertical copy moved a source frame; it is not used")
    partial.rename(output)
    receipt.write_text(
        json.dumps(
            {**expected, "frames": len(offsets), "moves": lines.count(";") - 1,
             "samples": sample_shares(kinds, seconds_per_sample=float(step / rate)),
             "outputSHA256": sha256(output)}, indent=2,
        ) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    print(
        f"Rendered vertical copy with {CODE_VERSION} camera "
        f"{json.dumps(expected['camera'], sort_keys=True)}",
        file=sys.stderr,
    )
    return output
