"""Read-only domain verification of retained local ASR, including failed envelopes."""

import argparse
from pathlib import Path

from replay.caption_io import mlx_groups
from replay.local_asr import fingerprint, pcm_audio
from replay.native_invocation import canonical, read, require, rows
from replay.stages import _fingerprint
from replay.workflow import _stage


def verify(plan: Path, state: Path, name: str, evidence: Path, pins: Path) -> None:
    """Require observed producer identities and raw-file pins; never infer or rewrite state."""
    historical = read(plan)
    require(historical.get("version") == 1, "Unknown historical workflow")
    # A historical directory-output bug cannot pass the current execution graph.
    # Parse its stage identity only; this verifier never executes that graph.
    stages = [_stage(row) for row in rows(historical.get("stages"))]
    stages = [stage for stage in stages if stage.name == name]
    require(len(stages) == 1, "Missing unique ASR producer")
    stage = stages[0]
    receipt = read(state / f"{name}.json")
    require(
        receipt.get("status") in ("running", "complete")
        and receipt.get("fingerprint") == _fingerprint(stage),
        "Observed producer identity changed",
    )
    require(pins.stat().st_size <= 65536, "Oversized observed pins")
    observed: dict[Path, str] = {}
    for line in pins.read_text(encoding="utf-8").splitlines():
        digest, separator, path = line.partition("  ")
        require(separator == "  " and len(digest) == 64, "Invalid observed pin")
        item = canonical(path)
        require(item not in observed and fingerprint(item) == digest, "Observed artifact changed")
        observed[item] = digest
    inputs = read(evidence / "inputs.json")
    source, model = canonical(inputs.get("source")), canonical(inputs.get("model"))
    required = {
        source,
        evidence / "inputs.json",
        evidence / "raw-transcript.json",
        evidence / "validation.json",
        Path.cwd() / "src/replay/local_asr.py",
        Path.cwd() / "src/replay/caption_io.py",
    }
    require(required <= observed.keys(), "Missing raw/code provenance")
    weights = [
        path for path in (model / "weights.npz", model / "weights.safetensors") if path.exists()
    ]
    require(len(weights) == 1, "Ambiguous retained model")
    require(
        fingerprint(source) == inputs.get("source_sha256")
        and fingerprint(model / "config.json") == inputs.get("config_sha256")
        and fingerprint(weights[0]) == inputs.get("weights_sha256")
        and {source, model / "config.json", weights[0]} <= set(stage.inputs),
        "ASR input/model provenance changed",
    )
    samples = pcm_audio(source)
    require(
        inputs.get("version") == 1
        and inputs.get("sample_rate") == 16000
        and inputs.get("samples") == len(samples)
        and inputs.get("clock") == "continuous-input-pcm"
        and inputs.get("human_reviewed") is False,
        "ASR input clock changed",
    )
    groups = mlx_groups(read(evidence / "raw-transcript.json"), duration=len(samples) / 16000)
    require(bool(groups), "Empty ASR is not absence proof")
    require(
        read(evidence / "validation.json")
        == {
            "word_groups": len(groups),
            "unresolved_points": sum(group.start == group.end for group in groups),
            "human_reviewed": False,
            "source_clock_projected": False,
        },
        "ASR validation differs from retained raw prediction",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("name")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("pins", type=Path)
    args = parser.parse_args(argv)
    verify(args.plan, args.state, args.name, args.evidence.resolve(), args.pins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
