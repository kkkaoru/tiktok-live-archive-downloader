"""Metadata-only CPU-to-Metal request preparation and separately sealed workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from replay import isolated_metal_asr as metal
from replay import mixed_isolated_asr as mixed
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256
from replay.native_invocation import durable, identity, pin, require
from replay.recordings import record


def prepare(collection: Path, policy: Path, output: Path, *, timeout: int = 600) -> Path:
    """Reuse original manifest hashes; never hash PCM or model contents here."""
    root = Path.cwd().resolve()
    for path in (collection, policy, output):
        metal._local(str(path), root)
    require(output.is_relative_to(root / "edits/work") and not output.exists(), "Occupied output")
    manifest_path = collection / "inputs.json"
    manifest = metal._read(manifest_path)
    require(
        manifest.get("version") == 1
        and manifest.get("device") == "cpu"
        and manifest.get("sampleRate") == 16000
        and manifest.get("windowSamples") == 960000
        and manifest.get("strideSamples") == 480000
        and manifest.get("clock") == "input-pcm-not-projected",
        "Unsupported original CPU clock/backend",
    )
    started = metal._read(policy)
    require(started.get("manifestSHA256") == pin(manifest_path)["sha256"], "CPU manifest unbound")
    inputs = record(manifest.get("inputs"))
    roles: dict[str, object] = {}
    for name, digest in inputs.items():
        path = metal._local(name, root)
        role = (
            "weights"
            if digest == DEFAULT_WEIGHTS_SHA256
            else (
                "config"
                if digest == DEFAULT_CONFIG_SHA256
                else "pcm"
                if path.suffix == ".wav"
                else None
            )
        )
        if role is not None:
            require(role not in roles and identity(path)["bytes"] > 0, "Ambiguous/missing input")
            roles[role] = {"path": str(path), "sha256": metal._digest(digest)}
    require(set(roles) == {"pcm", "weights", "config"}, "Missing original input roles")
    sources = set(metal.required_code()) | {
        Path(mixed.__file__).resolve(),
        Path(__file__).resolve(),
        Path(__file__).with_name("isolated_cpu_asr.py"),
        Path(__file__).with_name("measured_job_admission.py"),
        Path(__file__).with_name("leaf_retirement.py"),
        Path(__file__).with_name("phase_timer.py"),
    }
    code = {str(path): pin(path)["sha256"] for path in sorted(sources)}
    request = {
        "version": 1,
        "device": "metal",
        "fp16": True,
        "samples": metal._integer(manifest.get("samples"), 1, metal.MAX_SAMPLES),
        "timeoutSeconds": metal._integer(timeout, 1, 3600),
        "inputs": roles,
        "codeInputs": code,
    }
    output.mkdir(mode=0o700)
    metal_path = output / "metal-request.json"
    durable(metal_path, request)
    metal.load_request(metal_path, verify=False)
    draft = output / "mixed-draft.json"
    durable(
        draft,
        {
            "version": 1,
            "kind": "mixed-isolated-asr-draft",
            "candidateOnly": True,
            "canDispatch": False,
            "metalRequest": pin(metal_path),
            "cpuCollection": str(collection),
            "cpuPolicyStarted": pin(policy),
            "cpuManifest": pin(manifest_path),
            "ownerCaptures": None,
            "interruptedStopMaximumSeconds": 30,
            "codeInputs": code,
        },
    )
    return draft


def seal(draft: Path, captures: Path, output: Path) -> Path:
    """Seal actual original capture/barrier refs, without signals or model reads."""
    data = metal._read(draft)
    require(
        data.get("kind") == "mixed-isolated-asr-draft" and data.get("canDispatch") is False,
        "Not an unsubmitted draft",
    )
    mixed._reference(data.get("cpuManifest"))
    for key in ("cpuPolicyStarted", "metalRequest"):
        mixed._reference(data.get(key))
    owners = mixed._rows(metal._read(captures).get("ownerCaptures"))
    require(bool(owners), "Missing actual owners")
    for owner in owners:
        row = record(owner)
        mixed._reference(row)
        mixed._reference(row.get("stopBarrier"))
    request = {
        key: data[key]
        for key in (
            "version",
            "metalRequest",
            "cpuCollection",
            "cpuPolicyStarted",
            "interruptedStopMaximumSeconds",
            "codeInputs",
        )
    }
    request.update(kind="mixed-isolated-asr", ownerCaptures=owners)
    durable(output, request)
    mixed.load(output)
    return output


def trial(request: Path, collection: Path, receipt: Path) -> None:
    result = mixed.run(request, collection, maximum_new_windows=1)
    durable(
        receipt,
        {
            "version": 1,
            "request": pin(request),
            "result": pin(result),
            "collectionComplete": result.name == "complete.json",
        },
    )
    verify_trial(request, collection, receipt)


def verify_trial(request: Path, collection: Path, receipt: Path) -> None:
    plan = mixed.load(request)
    adopted = mixed._adoption(plan, collection)
    result = metal._read(receipt)
    require(result.get("request") == pin(request), "Trial request changed")
    partial = metal._read(mixed._reference(result.get("result")))
    if result.get("collectionComplete") is True:
        mixed.run(request, collection, verify_only=True)
        return
    require(
        partial.get("collectionComplete") is False and partial.get("requestSHA256") == plan.sha256,
        "Invalid partial checkpoint",
    )
    hashes = mixed._rows(partial.get("rawSHA256"))
    refs = mixed._rows(partial.get("windowReceipts"))
    require(0 < len(hashes) == len(refs) < len(mixed._windows(plan)), "Invalid partial coverage")
    for window, digest, ref in zip(mixed._windows(plan), hashes, refs, strict=False):
        raw = collection / f"{window.start_sample:012d}.raw.json"
        clock = collection / f"{window.start_sample:012d}.clock.json"
        require(
            pin(raw)["sha256"] == digest
            and metal._read(clock) == metal._clock(window, str(digest)),
            "Partial raw/clock changed",
        )
        proof = metal._read(mixed._reference(ref))
        if window.start_sample in adopted:
            require(proof == adopted[window.start_sample], "CPU provenance changed")
        else:
            work = collection / f"{window.start_sample:012d}.metal-runtime"
            require(
                proof
                == {
                    "version": 1,
                    "device": "metal",
                    "requestSHA256": plan.metal.sha256,
                    "startSample": window.start_sample,
                    "sampleCount": window.sample_count,
                    "evidence": metal._evidence(work),
                    "deliveryComplete": False,
                },
                "Metal evidence changed",
            )


def workflow(request: Path, collection: Path, output: Path, *, complete: bool = False) -> Path:
    """Generate finite trial or full continuation; original domain verification mandatory."""
    mixed.load(request)
    root = Path.cwd().resolve()
    for path in (collection, output):
        metal._local(str(path), root)
        require(path.is_relative_to(root / "edits/work"), "Work escapes owned scratch")
    python = str(root / ".venv/bin/python")
    prefix = [python, "-B", "-m"]
    inputs = [str(request)]
    stages: list[dict[str, object]] = []
    if not complete:
        adoption = collection / "adoption-complete.json"
        stages.append(
            {
                "name": "adopt-cpu",
                "command": prefix
                + ["replay.mixed_isolated_asr", "adopt", str(request), str(collection)],
                "inputs": inputs,
                "outputs": [str(adoption)],
                "timeout_seconds": 3600,
            }
        )
        receipt = output.parent / "trial.json"
        stages.append(
            {
                "name": "first-metal-window",
                "command": prefix
                + [
                    "replay.metal_asr_request_factory",
                    "trial",
                    str(request),
                    str(collection),
                    str(receipt),
                ],
                "inputs": inputs + [str(adoption)],
                "outputs": [str(receipt)],
                "timeout_seconds": 3600,
                "verify_command": prefix
                + [
                    "replay.metal_asr_request_factory",
                    "verify-trial",
                    str(request),
                    str(collection),
                    str(receipt),
                ],
            }
        )
    else:
        stages.append(
            {
                "name": "remaining-metal",
                "command": prefix
                + ["replay.mixed_isolated_asr", "run", str(request), str(collection)],
                "inputs": inputs + [str(collection / "adoption-complete.json")],
                "outputs": [str(collection / "complete.json")],
                "timeout_seconds": 86400,
                "verify_command": prefix
                + ["replay.mixed_isolated_asr", "verify", str(request), str(collection)],
            }
        )
        stages.append(
            {
                "name": "original-clock-assembly",
                "command": prefix
                + ["replay.asr_assembly", str(collection), str(output.parent / "assembled.json")],
                "inputs": inputs + [str(collection / "complete.json")],
                "outputs": [str(output.parent / "assembled.json")],
                "timeout_seconds": 3600,
            }
        )
    durable(output, {"version": 1, "stages": stages})
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "seal", "workflow", "full-workflow", "trial", "verify-trial")
    )
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.first, args.second, args.output)
    elif args.action == "seal":
        seal(args.first, args.second, args.output)
    elif args.action in {"workflow", "full-workflow"}:
        workflow(args.first, args.second, args.output, complete=args.action == "full-workflow")
    elif args.action == "trial":
        trial(args.first, args.second, args.output)
    else:
        verify_trial(args.first, args.second, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
