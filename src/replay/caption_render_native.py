"""Render caption parts natively from their finished Motion documents, as caption-render does.

The opt-in native backend of the ``caption-render`` step. It consumes the same ``jobs.json``
that ``replay.caption_parts`` writes for Compressor (each job's ``source`` is the finished
``.moti``, its ``output`` the alpha movie) and writes the same outputs and per-job states, so
``replay.caption_compose`` uses them unchanged: ``<state>/<name>.json`` records ``status``
``Successful`` and the ``output`` only after the native renderer
(``caption-alpha-render --moti``, which refuses any part outside its supported template)
finished it. A job whose state is recorded is reused; an output left without a recorded state
is never overwritten and stops the run for inspection.

``gate`` first renders a sample of the parts both ways, Motion through ``replay.compressor_batch``
(one job in flight) and natively, and requires ``replay.caption_parity`` to pass every sampled
part against a pinned floor. Any parse, render or parity failure fails the step; nothing falls
back silently.

    python -m replay.caption_render_native gate PARTS.json JOBS.json --work DIR \
        --binary BIN --binary-sha256 HASH --floor-json FLOOR.json --sample 2
    python -m replay.caption_render_native render JOBS.json --state DIR \
        --binary BIN --binary-sha256 HASH --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

from replay import caption_gradient_audit as gradient_audit
from replay import caption_parity
from replay.core import ReplayError
from replay.recordings import record

SUCCESS: Final[str] = "Successful"
RENDERER: Final[str] = "native-motion-document"
COMPRESSOR_PRESET_KEY: Final[str] = "preset"
GATE_STEP: Final[int] = 4  # every 4th frame: every reveal state lasts at least a frame or two

Run = Callable[[Sequence[str]], str]
# The renderer's refusal of a part outside the template it can verify (exit status 2).
REFUSAL: Final[str] = "Unsupported Motion caption part"


class Refused(ReplayError):
    """The native reader refused a part; Motion + Compressor render it instead."""


def run_checked(command: Sequence[str]) -> str:
    """A command's output; the renderer's refusal of a part raises ``Refused``."""
    done = subprocess.run(list(command), capture_output=True, text=True)
    if done.returncode == 2 and REFUSAL in done.stderr:
        raise Refused(done.stderr.strip())
    if done.returncode:
        raise ReplayError(f"{command[0]} failed ({done.returncode}): {done.stderr.strip()[-2000:]}")
    return done.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_binary(binary: Path, expected: str) -> Path:
    """The renderer, refused unless it is the pinned build."""
    if not binary.is_file() or sha256(binary) != expected:
        raise ReplayError(f"{binary} is not the pinned native caption renderer")
    return binary


def jobs_of(path: Path) -> list[dict[str, str]]:
    rows = record(json.loads(path.read_text(encoding="utf-8"))).get("jobs")
    if not isinstance(rows, list) or not rows:
        raise ReplayError(f"{path} lists no caption jobs")
    jobs = []
    for value in rows:
        item = record(value)
        fields = {key: item.get(key) for key in ("name", "source", "output")}
        if not all(isinstance(field, str) and field for field in fields.values()):
            raise ReplayError("Each caption job needs a name, a source and an output")
        jobs.append({key: str(field) for key, field in fields.items()} | {
            COMPRESSOR_PRESET_KEY: str(item.get(COMPRESSOR_PRESET_KEY, ""))
        })  # fmt: skip
    if len({job["name"] for job in jobs}) != len(jobs):
        raise ReplayError("Caption job names must be unique")
    return jobs


def write_state(path: Path, value: dict[str, object]) -> None:
    """Write a job state atomically, never replacing a recorded one."""
    if path.exists():
        raise ReplayError(f"{path} is already recorded")
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(partial, path)


def recorded(state: Path) -> bool:
    if not state.exists():
        return False
    saved = record(json.loads(state.read_text(encoding="utf-8")))
    return saved.get("status") == SUCCESS and Path(str(saved.get("output"))).is_file()


def render_job(
    job: dict[str, str], states: Path, *, binary: Path, binary_sha256: str, run: Run
) -> str:
    """Render one job unless its success is recorded: ``reused``, ``rendered`` or, when the
    reader refuses the part (nothing is written), the refusal as ``refused: REASON``."""
    state = states / f"{job['name']}.json"
    if recorded(state):
        return "reused"
    output, source = Path(job["output"]), Path(job["source"])
    if state.exists() or output.exists():
        raise ReplayError(f"{job['name']}: an unrecorded or failed output exists; inspect it")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_sha256 = sha256(source)
    try:
        report = record(json.loads(run([str(binary), "--moti", str(source), str(output)])))
    except Refused as refusal:
        return f"refused: {refusal}"
    if sha256(source) != source_sha256:
        raise ReplayError(f"{job['name']}: the Motion document changed while rendering")
    write_state(state, {
        "status": SUCCESS, "renderer": RENDERER, "jobConfig": job, "output": str(output),
        "sourceSha256": source_sha256, "binarySha256": binary_sha256, "report": report,
        "bytes": output.stat().st_size,
    })  # fmt: skip
    return "rendered"


def render(
    jobs_path: Path, states: Path, *, binary: Path, binary_sha256: str, workers: int,
    run: Run = run_checked,
) -> dict[str, object]:  # fmt: skip
    """Render every job, ``workers`` at a time; a failed job fails the run. Parts the reader
    refuses are listed with their reasons under ``refused``; nothing is written for them."""
    verified_binary(binary, binary_sha256)
    if not 1 <= workers <= 8:
        raise ReplayError("Native caption workers must be 1 to 8")
    states.mkdir(mode=0o700, parents=True, exist_ok=True)
    jobs = jobs_of(jobs_path)

    def one(job: dict[str, str]) -> str:
        return render_job(job, states, binary=binary, binary_sha256=binary_sha256, run=run)

    with ThreadPoolExecutor(workers) as pool:
        outcomes = list(pool.map(one, jobs))
    refused = {
        job["name"]: outcome.removeprefix("refused: ")
        for job, outcome in zip(jobs, outcomes, strict=True)
        if outcome.startswith("refused: ")
    }
    counted = [outcome for outcome in outcomes if not outcome.startswith("refused: ")]
    result: dict[str, object] = {name: counted.count(name) for name in sorted(set(counted))}
    return result | ({"refused": refused} if refused else {})


def ranked(parts_path: Path) -> list[str]:
    """Part names, the most stages (then the longest) first, deterministically."""
    rows = [record(value) for value in json.loads(parts_path.read_text(encoding="utf-8"))]
    ordered = sorted(
        rows,
        key=lambda row: (-int(str(row["stages"])), -int(str(row["frames"])), str(row["name"])),
    )
    return [str(row["name"]) for row in ordered]


def sampled(
    parts_path: Path, count: int, by_name: dict[str, dict[str, str]], *, binary: Path, run: Run
) -> list[str]:
    """The busiest ``count`` parts the native reader accepts (refused parts are rendered by
    Motion + Compressor, so they need no parity)."""
    names: list[str] = []
    for name in ranked(parts_path):
        if len(names) == count:
            break
        source = Path(by_name[name]["source"])
        if not source.is_file():
            raise ReplayError(f"The Motion document of {name} is missing: {source}")
        try:
            run([str(binary), "--inspect", str(source)])
        except Refused:
            continue
        names.append(name)
    return names


def gate(
    parts_path: Path, jobs_path: Path, work: Path, *, binary: Path, binary_sha256: str,
    floor_json: Path, sample: int, run: Run = run_checked,
) -> dict[str, object]:  # fmt: skip
    """Render sampled parts both ways and require parity; the receipt is ``work/gate.json``."""
    verified_binary(binary, binary_sha256)
    floor = caption_parity.stored_floor(floor_json)
    if not 1 <= sample <= 8:
        raise ReplayError("Sample 1 to 8 parts for the caption parity gate")
    by_name = {job["name"]: job for job in jobs_of(jobs_path)}
    names = sampled(parts_path, sample, by_name, binary=binary, run=run)
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not names:
        receipt: dict[str, object] = {"pass": False, "parts": [], "reason": "no part was readable"}
        (work / "gate.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
        raise ReplayError(f"Native captions failed the parity gate: {work / 'gate.json'}")
    motion_jobs = [
        {"name": name, "source": by_name[name]["source"],
         COMPRESSOR_PRESET_KEY: by_name[name][COMPRESSOR_PRESET_KEY],
         "output": str(work / "motion" / name / "compressor-batch-v1" / "motion-alpha.mov")}
        for name in names
    ]  # fmt: skip
    native_jobs = [
        {**job, "output": str(work / "native" / f"{job['name']}.mov")} for job in motion_jobs
    ]
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, jobs in (("motion-jobs.json", motion_jobs), ("native-jobs.json", native_jobs)):
        path = work / name
        if not path.exists():
            path.write_text(json.dumps({"jobs": jobs}, indent=1) + "\n", encoding="utf-8")
    run([sys.executable, "-B", "-m", "replay.compressor_batch", str(work / "motion-jobs.json"),
         "--state", str(work / "motion-state"), "--batch-name", "caption parity gate",
         "--in-flight", "1", "--submit-workers", "1", "--retry-controller-failure"])  # fmt: skip
    render(work / "native-jobs.json", work / "native-state", binary=binary,
           binary_sha256=binary_sha256, workers=1, run=run)  # fmt: skip
    results: dict[str, dict[str, object]] = {}
    for motion, native in zip(motion_jobs, native_jobs, strict=True):
        candidate = caption_parity.summary(caption_parity.compare(
            gradient_audit.frames(motion["output"], step=GATE_STEP),
            gradient_audit.frames(native["output"], step=GATE_STEP),
        ))  # fmt: skip
        results[motion["name"]] = {
            "candidate": candidate,
            **caption_parity.verdict(floor, candidate),
        }
    receipt = {
        "pass": all(result["pass"] for result in results.values()),
        "parts": names, "floor": floor, "floorSha256": sha256(floor_json),
        "binarySha256": binary_sha256, "results": results,
    }  # fmt: skip
    (work / "gate.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    if not receipt["pass"]:
        raise ReplayError(f"Native captions failed the parity gate: {work / 'gate.json'}")
    return receipt


def compressor_fallback(
    jobs: Sequence[dict[str, str]], states: Path, work: Path, *, in_flight: int,
    submit_workers: int, run: Run,
) -> None:  # fmt: skip
    """Render ``jobs`` (their own outputs and states) with Motion + Compressor."""
    path = work / "compressor-jobs.json"
    path.write_text(json.dumps({"jobs": list(jobs)}, indent=1) + "\n", encoding="utf-8")
    run([sys.executable, "-B", "-m", "replay.compressor_batch", str(path), "--state",
         str(states), "--in-flight", str(in_flight), "--submit-workers", str(submit_workers),
         "--retry-controller-failure"])  # fmt: skip


def caption_render(
    parts_path: Path, jobs_path: Path, states: Path, work: Path, *, binary: Path,
    binary_sha256: str, floor_json: Path, sample: int, workers: int, in_flight: int,
    submit_workers: int, run: Run = run_checked,
) -> dict[str, object]:  # fmt: skip
    """The caption-render step with native drawing on: gate, then native with fallback.

    A failed gate renders every part with Motion + Compressor; a part the native reader
    refuses is rendered by them alone. Every fallback and its reason is in ``work/receipt.json``.
    """
    jobs = jobs_of(jobs_path)
    try:
        gated: dict[str, object] = gate(
            parts_path, jobs_path, work / "gate", binary=binary, binary_sha256=binary_sha256,
            floor_json=floor_json, sample=sample, run=run,
        )  # fmt: skip
    except ReplayError as failure:
        gated = {"pass": False, "reason": str(failure)}
    fallback: dict[str, str] = {}
    native: dict[str, object] = {}
    if gated["pass"] is True:
        native = render(jobs_path, states, binary=binary, binary_sha256=binary_sha256,
                        workers=workers, run=run)  # fmt: skip
        fallback = {str(name): str(reason)
                    for name, reason in record(native.pop("refused", {})).items()}  # fmt: skip
    else:
        fallback = {job["name"]: f"parity gate failed: {gated.get('reason', 'see gate.json')}"
                    for job in jobs}  # fmt: skip
    if fallback:
        refused = [job for job in jobs if job["name"] in fallback]
        compressor_fallback(refused, states, work, in_flight=in_flight,
                            submit_workers=submit_workers, run=run)  # fmt: skip
    renderer = "native"
    if fallback:
        renderer = "compressor" if len(fallback) == len(jobs) else "mixed"
    receipt: dict[str, object] = {
        "renderer": renderer, "gate": gated, "native": native, "fallback": fallback,
        "binarySha256": binary_sha256,
    }  # fmt: skip
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    (work / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("gate", "render", "run"):
        command = commands.add_parser(name)
        if name in ("gate", "run"):
            command.add_argument("parts", type=Path)
        command.add_argument("jobs", type=Path)
        command.add_argument("--binary", type=Path, required=True)
        command.add_argument("--binary-sha256", required=True)
        if name in ("gate", "run"):
            command.add_argument("--work", type=Path, required=True)
            command.add_argument("--floor-json", type=Path, required=True)
            command.add_argument("--sample", type=int, default=2)
        if name in ("render", "run"):
            command.add_argument("--state", type=Path, required=True)
            command.add_argument("--workers", type=int, default=2)
        if name == "run":
            command.add_argument("--in-flight", type=int, default=2)
            command.add_argument("--submit-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "run":
        receipt = caption_render(
            args.parts, args.jobs, args.state, args.work, binary=args.binary,
            binary_sha256=args.binary_sha256, floor_json=args.floor_json, sample=args.sample,
            workers=args.workers, in_flight=args.in_flight, submit_workers=args.submit_workers,
            run=run_checked,
        )  # fmt: skip
        print(json.dumps({"renderer": receipt["renderer"], "fallback": receipt["fallback"]},
                         ensure_ascii=False))  # fmt: skip
        return 0
    if args.command == "gate":
        result: object = gate(
            args.parts, args.jobs, args.work, binary=args.binary,
            binary_sha256=args.binary_sha256, floor_json=args.floor_json, sample=args.sample,
            run=run_checked,
        )  # fmt: skip
        print(json.dumps({"pass": record(result)["pass"], "parts": record(result)["parts"]}))
        return 0
    print(json.dumps(render(args.jobs, args.state, binary=args.binary,
                            binary_sha256=args.binary_sha256, workers=args.workers,
                            run=run_checked)))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
