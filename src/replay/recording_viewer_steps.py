"""Viewer extras of a published recording: summary, excitement and clip points.

- ``upload-original`` (right after clocks): a dry run, then ``sync --originals-only`` sends the
  downloaded originals to ``media-original/`` only, publishing no data or catalog; the delivery's
  upload later finds the original in the bucket instead of sending it in the pipeline's tail;
- ``excitement`` (before the upload): ``replay.excitement build`` writes
  ``<srt>.excitement.json`` from the delivered transcript and movie; the upload carries it;
- ``summarize`` (beside the upload, so the slow gate and on-device model neither hold the movie
  back nor wait for it): the recording's incomplete older revisions are superseded first (the
  Viewer refuses to load a recording with several current deliveries, so the summary CLI found
  "No matching loaded videos" until then); then the completion gate (``edit_completion check``,
  which decodes the movie) runs while ``transcript-summary generate`` (Apple Intelligence) saves
  ``<srt>.summaries/current.json``, reusing a matching one. The step passes only when both do; a
  passed gate is kept (``steps/summarize-gate.json``, bound to this publication and the edit
  configuration) so a retry after a failed generation does not decode the movie again;
- ``summary`` (after the upload and ``summarize``): reuses summarize's gate of this publication
  (or passes the gate itself), generates again (a reuse) and a data-only sync publishes it;
- ``clips``: the Mac computes the clip points from the uploaded data with its local models
  (``bun run clips:mac``) and stores them through the viewer; the step passes once
  ``/api/clips/<id>.json`` reports them ready. Cloudflare AI runs only for a user's reload.
"""

import fcntl
import json
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from replay import delivery_offload, r2_fetch
from replay.core import ReplayError
from replay.edit_completion import completion_report, delivery_label, supersede_incomplete
from replay.recording_steps import Pipeline, Runner, module, pin, sha256
from replay.recordings import record

CHECKOUT = Path(__file__).resolve().parents[2]
SUMMARY_CLI = CHECKOUT / "TranscriptViewer" / ".build" / "release" / "transcript-summary"
WEB_VIEWER = CHECKOUT / "web-viewer"
GATE_KEPT = "summarize-gate.json"
# Cloudflare refuses Python's default User-Agent with 403.
USER_AGENT = "replay-recording-pipeline/1"

Fetch = Callable[[urllib.request.Request], dict[str, object]]


def delivery(pipeline: Pipeline) -> Path:
    """The published movie (``<deliveryRoot>/<recording>/revisions/<rev>/delivery/<name>.mp4``)."""
    published = json.loads((pipeline.steps / "publish.json").read_text(encoding="utf-8"))
    return Path(str(record(published["outputs"])["delivery"]))


def transcript(pipeline: Pipeline) -> Path:
    found = delivery(pipeline).with_suffix(".srt")
    if not found.is_file():
        raise ReplayError(f"The published delivery has no transcript: {found}")
    return found


def sync(run: Runner, log: Path, *options: str) -> None:
    """The web viewer's sync; one runs at a time on this checkout (they share its state)."""
    bun = shutil.which("bun")
    if bun is None:
        raise ReplayError("bun is required to upload to the web viewer")
    with (WEB_VIEWER / "sync.lock").open("a") as lock:
        waited = time.monotonic()
        fcntl.flock(lock, fcntl.LOCK_EX)
        log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"PHASE lock {time.monotonic() - waited:.1f} s\n")
        run([bun, str(WEB_VIEWER / "src" / "bin" / "sync.ts"), *options], log)


def require_uploaded(video: str, movie: Path, *, root: Path | None = None) -> None:
    """The sync left this movie settled in its state: sent now, or found unchanged in the bucket.

    A sync that saw another delivery of the recording sends nothing for it and still succeeds.
    A movie offloaded to R2 (recorded under ``root``) is compared by its recorded size and mtime.
    """
    settled: object = json.loads((WEB_VIEWER / "sync-state.json").read_text(encoding="utf-8"))
    for key in ("sources", video, "movie"):
        settled = settled.get(key) if isinstance(settled, dict) else None
    settled = settled if isinstance(settled, dict) else {}
    entry = None if root is None or movie.exists() else delivery_offload.offloaded(root, movie)
    size, mtime = (
        (entry.bytes, entry.mtime_ns)
        if entry is not None
        else (movie.stat().st_size, movie.stat().st_mtime_ns)
    )
    if settled.get("size") != size or settled.get("mtimeNs") != str(mtime):
        raise ReplayError(f"The web viewer sync did not upload {video}; inspect its upload log")


def upload_original(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    if (pipeline.steps / "upload.json").is_file():  # the delivery's upload already sent it
        return {"originals": "sent with the delivery"}
    log = pipeline.logs / "upload-original.log"
    sync(run, log, "--originals-only", "--dry-run")
    sync(run, log, "--originals-only")
    return {"originals": "sent", "log": str(log)}


def gate(pipeline: Pipeline) -> dict[str, object]:
    """Pass the completion gate, then supersede the incomplete older revisions."""
    root = pipeline.path_of("editRoot")
    r2_fetch.restore_if_offloaded(delivery(pipeline), root=root)  # the gate decodes the movie
    report = completion_report(root)
    saved = pipeline.logs / f"gate-{time.time_ns()}.json"
    saved.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    saved.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report["complete"] is not True:
        raise ReplayError(f"The completion gate did not pass; inspect {saved}")
    rows = report.get("deliveries")
    assessed = {  # the report's assessments: superseding does not decode each movie again
        str(row["delivery"]): {str(k): str(v) for k, v in record(row["missingStages"]).items()}
        for row in (map(record, rows) if isinstance(rows, list) else ())
    }
    superseded = [delivery_label(root, path) for path in supersede_incomplete(root, assessed)]
    return {"complete": True, "report": pin(saved), "superseded": superseded}


def generate(pipeline: Pipeline, run: Runner) -> dict[str, str]:
    """Save the transcript's summary (the CLI reuses a matching saved one) and pin it."""
    srt = transcript(pipeline)
    library = delivery(pipeline).parents[4]
    log = pipeline.logs / "summary.log"
    run([str(SUMMARY_CLI), "generate", "--library", str(library), "--transcript", str(srt)], log)
    return pin(srt.parent / f"{srt.name}.summaries" / "current.json")


def gate_binding(pipeline: Pipeline) -> dict[str, str]:
    """What a passed gate depends on: this publication and the recording's edit configuration."""
    config = pipeline.path_of("editRoot") / "edit-config.json"
    return {"publish": sha256(pipeline.steps / "publish.json"), "editConfig": sha256(config)}


def kept_gate(pipeline: Pipeline) -> dict[str, object] | None:
    """A gate that passed for the same binding before a failed generation, its report unchanged."""
    kept = pipeline.steps / GATE_KEPT
    if not kept.is_file():
        return None
    data = record(json.loads(kept.read_text(encoding="utf-8")))
    passed = record(data.get("gate"))
    report = record(passed.get("report"))
    path = Path(str(report.get("path")))
    if data.get("binding") != gate_binding(pipeline) or passed.get("complete") is not True:
        return None
    return passed if path.is_file() and sha256(path) == report.get("sha256") else None


def release_older(root: Path, published: Path) -> None:
    """Supersede the older incomplete revisions of a new publication before it is gated, so the
    web viewer's catalog and the summary CLI see only it. The publication itself is left to the
    gate; every record still re-checks the movie's identity before it is written."""
    supersede_incomplete(root, {delivery_label(root, published): {}})


def summarize(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    earlier = pipeline.steps / "summary.json"
    if earlier.is_file():  # summarized before this step existed: nothing left to do
        return record(record(json.loads(earlier.read_text(encoding="utf-8"))).get("outputs"))
    passed = kept_gate(pipeline)
    if passed is not None:
        return {"gate": passed, "summary": generate(pipeline, run)}
    release_older(pipeline.path_of("editRoot"), delivery(pipeline))  # a no-op after publish
    # The gate decodes the movie while the on-device model summarizes: neither waits for the other.
    with ThreadPoolExecutor(max_workers=1) as pool:
        checking = pool.submit(gate, pipeline)
        try:
            summary: dict[str, str] | ReplayError = generate(pipeline, run)
        except ReplayError as error:
            summary = error
        passed = checking.result()
    # A retry after a failed generation reuses this gate instead of decoding the movie again.
    kept = {"binding": gate_binding(pipeline), "gate": passed}
    (pipeline.steps / GATE_KEPT).write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
    if isinstance(summary, ReplayError):
        raise summary
    return {"gate": passed, "summary": summary}


def summarized_gate(pipeline: Pipeline) -> dict[str, object] | None:
    """The summarize step's passed gate, when it gated the current publication."""
    receipt = pipeline.steps / "summarize.json"
    if not receipt.is_file():
        return None
    data = record(json.loads(receipt.read_text(encoding="utf-8")))
    published = pipeline.steps / "publish.json"
    if record(data.get("inputs")).get(str(published)) != sha256(published):
        return None
    passed = record(data.get("outputs")).get("gate")  # absent in summaries from before the gate
    return passed if isinstance(passed, dict) and passed.get("complete") is True else None


def summary(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    passed = summarized_gate(pipeline) or gate(pipeline)
    pinned = generate(pipeline, run)
    sync(run, pipeline.logs / "summary-upload.log", "--data-only")
    return {"gate": passed, "summary": pinned}


def duration(movie: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(movie)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return float(result.stdout.strip())


def excitement(pipeline: Pipeline, run: Runner) -> dict[str, object]:
    srt, movie = transcript(pipeline), delivery(pipeline)
    r2_fetch.restore_if_offloaded(movie, root=pipeline.path_of("editRoot"))  # needs the bytes
    seconds = duration(movie, pipeline.text("ffprobe"))
    run(
        module(pipeline, "excitement", "build", "--transcript", str(srt),
               "--duration", f"{seconds:.3f}", "--movie", str(movie)),
        pipeline.logs / "excitement.log",
    )  # fmt: skip
    return {"excitement": pin(srt.parent / f"{srt.name}.excitement.json")}


def _viewer() -> tuple[str, str]:
    """The first viewer domain and the clips API token (never logged)."""
    token = json.loads((WEB_VIEWER / "secrets.json").read_text(encoding="utf-8")).get(
        "CLIPS_API_TOKEN"
    )
    domains = [
        line.split("=", 1)[1].strip().strip("\"'")
        for line in (WEB_VIEWER / "deploy.env").read_text(encoding="utf-8").splitlines()
        if line.startswith("VIEWER_DOMAIN=")
    ]
    if not isinstance(token, str) or not token or not domains or not domains[0]:
        raise ReplayError(
            "web-viewer secrets.json/deploy.env lack CLIPS_API_TOKEN or VIEWER_DOMAIN"
        )
    return domains[0].split(",")[0].strip(), token


def _fetch(request: urllib.request.Request) -> dict[str, object]:
    with urllib.request.urlopen(request, timeout=60) as response:
        return record(json.loads(response.read()))


def clips(pipeline: Pipeline, run: Runner, *, fetch: Fetch = _fetch) -> dict[str, object]:
    """The Mac computes the clip points from the uploaded data, then the viewer must report them.

    ``bun run clips:mac`` runs one at a time machine-wide (its local models take about 22 GB).
    Cloudflare AI is never asked here: it runs only for a user's reload.
    """
    domain, token = _viewer()
    video = delivery(pipeline).stem
    bun = shutil.which("bun")
    if bun is None:
        raise ReplayError("bun is required to compute the web viewer's clip points")
    data = pipeline.work / "clips-data.json"
    data.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (WEB_VIEWER / "sync.lock").open("a") as lock:  # a sync replaces data/ as a whole
        fcntl.flock(lock, fcntl.LOCK_EX)
        shutil.copyfile(WEB_VIEWER / "data" / f"{video}.json", data)
    with (WEB_VIEWER / "clips-mac.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(
            [bun, str(WEB_VIEWER / "src" / "bin" / "clips-mac.ts"), str(data)],
            pipeline.logs / "clips.log",
        )
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    answer = fetch(
        urllib.request.Request(f"https://{domain}/api/clips/{video}.json", headers=headers)
    )
    status = answer.get("status")
    if status != "ready":
        raise ReplayError(
            f"Clip points for {video} are not ready after the Mac stored them: {status}"
        )
    found = record(answer.get("record")).get("clips")
    return {"video": video, "clips": len(found) if isinstance(found, list) else 0}
