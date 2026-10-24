"""The Mac's local models for the web viewer's clip points (切り抜き), served on 127.0.0.1.

The web viewer computes clip points on the Mac by default (``bun run clips:mac``); Cloudflare AI
runs only when a user asks for a reload. Two models stand in for the Cloudflare ones:

- ``POST /clef`` answers a SystemOne decision request (``state`` plus typed ``questions``) with
  ``mlx-community/clef-OptiQ-4bit``, the 4-bit MLX quant of Cloudflare's clef: the backbone runs
  on MLX and the release's joint schema head (``joint_schema_model.py`` of Cloudflare/clef) is
  ported here to numpy. On 191 judged clips its rubric answers matched Cloudflare clef's token
  counts exactly (r 0.989) and ranked the clips as well (AUC 0.756 with clairic against 0.762);
- ``POST /bonsai`` answers an OpenAI-style chat body with ``Ternary-Bonsai-2-9B`` through
  mlx-vlm (thinking off, greedy), the stand-in for GLM's titles and summaries.

Both models load once from ``edits/models/`` and are never downloaded here; the server refuses
to start while less than ``--min-free`` percent of memory is free, since the two models take
about 22 GB beside the recording pipelines, and stops itself while serving once less than 15% is.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import import_module
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError

type Array = NDArray[np.float32]
type Json = dict[str, object]

CHECKOUT = Path(__file__).resolve().parents[2]
CLEF_DIR = CHECKOUT / "edits" / "models" / "clef-OptiQ-4bit"
BONSAI_DIR = CHECKOUT / "edits" / "models" / "Ternary-Bonsai-2-9B-mlx-2bit"
SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES: dict[str, int] = {"noul": 0, "choice": 1, "score": 2}
NOUL_CRITERIA: dict[str, str] = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}
MAX_LENGTH = 16384
PREFILL_CHUNK = 2048
LAYER_NORM_EPS = 1e-5
LOGIT_SCALE_CAP = math.log(100.0)
CHAT_TOKENS = 1024
MIN_FREE_PERCENT = 40
# Serving stops itself below this so recording pipelines are not pushed into swap.
MIN_RUNNING_FREE_PERCENT = 15
WATCH_SECONDS = 30.0
FREE_PATTERN = re.compile(r"System-wide memory free percentage:\s*(\d+)%")
erf = np.vectorize(math.erf, otypes=[np.float32])


@dataclass(frozen=True)
class Question:
    qid: str
    kind: int
    span: tuple[int, int]
    options: tuple[tuple[str, tuple[int, int]], ...]


@dataclass(frozen=True)
class HeadConfig:
    heads: int
    routing_layers: int
    layers: int


class Backbone(Protocol):
    """The decision model's language backbone: tokens in, final hidden states out."""

    def tokenize(self, text: str) -> list[int]: ...
    def hidden(self, ids: Sequence[int]) -> Array: ...
    def lexicon(self, ids: Sequence[int]) -> Array: ...


class Answerer(Protocol):
    def respond(self, body: Json) -> Json: ...


class Serving(Protocol):
    def serve_forever(self) -> None: ...


def render(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _mapping(value: object, what: str) -> Json:
    if not isinstance(value, dict):
        raise ReplayError(f"{what} must be an object")
    return {str(key): item for key, item in value.items()}


def question_options(question: Mapping[str, object]) -> list[tuple[str, object]]:
    kind = question.get("type")
    if kind == "noul":
        extra = question.get("criteria") or {}
        noul = {**NOUL_CRITERIA, **_mapping(extra, "noul criteria")}
        return [(key, noul[key]) for key in ("true", "false")]
    criteria = question.get("criteria")
    if kind == "choice":
        return sorted((str(key), value) for key, value in _mapping(criteria, "criteria").items())
    if kind == "score" and isinstance(criteria, list) and criteria:
        return [(str(index), value) for index, value in enumerate(criteria)]
    raise ReplayError(f"Unsupported question {kind!r} or empty criteria")


def encode(
    tokenize: Callable[[str], list[int]], record: Mapping[str, object]
) -> tuple[list[int], list[Question]]:
    """Token ids and question spans, built piece by piece exactly as clef's encode_record does."""
    questions = _mapping(record.get("questions"), "questions")
    if not questions or "state" not in record:
        raise ReplayError("A decision request needs a state and at least one question")
    schema = tokenize("\n\nSCHEMA FIELDS:\n")
    parsed: list[Question] = []
    for number, (qid, raw) in enumerate(questions.items()):
        question = _mapping(raw, f"question {qid}")
        kind = question.get("type")
        if kind not in QUESTION_TYPES:
            raise ReplayError(f"{qid}: type must be noul, choice, or score")
        options = question_options(question)
        schema += tokenize(f"\nFIELD {number + 1}\nID: {qid}\nTYPE: {kind}\nINSTRUCTION: ")
        start = len(schema)
        schema += tokenize(render(question.get("instructions") or str(qid)))
        span = (start, len(schema))
        schema += tokenize("\nALLOWED OPTIONS:\n")
        spans: list[tuple[str, tuple[int, int]]] = []
        for index, (oid, description) in enumerate(options):
            schema += tokenize(f"OPTION {index + 1}: ")
            begin = len(schema)
            semantics: Json = {"option_id": oid}
            if description is not None:
                semantics["description"] = description
            schema += tokenize(render(semantics))
            spans.append((oid, (begin, len(schema))))
            schema += tokenize("\n")
        schema += tokenize("END FIELD\n")
        parsed.append(Question(str(qid), QUESTION_TYPES[str(kind)], span, tuple(spans)))
    prefix = tokenize(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix = tokenize(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    )
    room = MAX_LENGTH - len(prefix) - len(schema) - len(suffix)
    if room < 0:
        raise ReplayError(
            f"The schema needs {MAX_LENGTH - room} tokens; the maximum is {MAX_LENGTH}"
        )
    state = tokenize(render(record["state"]))[:room]
    offset = len(prefix) + len(state)
    shifted = [
        Question(
            q.qid,
            q.kind,
            (q.span[0] + offset, q.span[1] + offset),
            tuple((oid, (b + offset, e + offset)) for oid, (b, e) in q.options),
        )
        for q in parsed
    ]
    return prefix + state + schema + suffix, shifted


def layer_norm(x: Array, w: Mapping[str, Array], name: str) -> Array:
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + LAYER_NORM_EPS)
    return np.asarray(normalized * w[f"{name}.weight"] + w[f"{name}.bias"], dtype=np.float32)


def linear(x: Array, w: Mapping[str, Array], name: str, *, bias: bool = True) -> Array:
    y = x @ w[f"{name}.weight"].T
    return y + w[f"{name}.bias"] if bias else y


def gelu(x: Array) -> Array:
    return np.asarray(0.5 * x * (1 + erf(x / math.sqrt(2))), dtype=np.float32)


def softmax(x: Array, axis: int = -1) -> Array:
    shifted = np.exp(x - x.max(axis=axis, keepdims=True))
    return np.asarray(shifted / shifted.sum(axis=axis, keepdims=True), dtype=np.float32)


def unit(x: Array) -> Array:
    return np.asarray(
        x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12), dtype=np.float32
    )


def attention(q: Array, kv: Array, w: Mapping[str, Array], name: str, heads: int) -> Array:
    """torch.nn.MultiheadAttention (batch_first, no mask) on unbatched (length, width) inputs."""
    width = q.shape[-1]
    dh = width // heads
    wq, wk, wv = np.split(w[f"{name}.in_proj_weight"], 3)
    bq, bk, bv = np.split(w[f"{name}.in_proj_bias"], 3)

    def split(x: Array) -> Array:
        return x.reshape(x.shape[0], heads, dh).transpose(1, 0, 2)

    qh, kh, vh = split(q @ wq.T + bq), split(kv @ wk.T + bk), split(kv @ wv.T + bv)
    weights = softmax((qh @ kh.transpose(0, 2, 1)) / math.sqrt(dh))
    out = (weights @ vh).transpose(1, 0, 2).reshape(q.shape[0], width)
    return linear(out, w, f"{name}.out_proj")


def _routed(
    w: Mapping[str, Array], config: HeadConfig, seq: Array, memory: Array, queries: Array
) -> Array:
    for n in range(config.routing_layers):
        p = f"evidence_layers.{n}"
        mem = layer_norm(memory, w, f"{p}.memory_norm")
        normalized = layer_norm(queries, w, f"{p}.query_norm")
        queries = queries + attention(normalized, mem, w, f"{p}.attention", config.heads)
        ff = gelu(linear(layer_norm(queries, w, f"{p}.feedforward_norm"), w, f"{p}.feedforward.0"))
        queries = queries + linear(ff, w, f"{p}.feedforward.3")
    return queries


def _decoded(w: Mapping[str, Array], config: HeadConfig, fields: Array, memory: Array) -> Array:
    for n in range(config.layers):
        p = f"layers.{n}"
        x = layer_norm(fields, w, f"{p}.norm1")
        fields = fields + attention(x, x, w, f"{p}.self_attn", config.heads)
        cross = layer_norm(fields, w, f"{p}.norm2")
        fields = fields + attention(cross, memory, w, f"{p}.multihead_attn", config.heads)
        ff = gelu(linear(layer_norm(fields, w, f"{p}.norm3"), w, f"{p}.linear1"))
        fields = fields + linear(ff, w, f"{p}.linear2")
    return layer_norm(fields, w, "field_norm")


def head(
    w: Mapping[str, Array],
    config: HeadConfig,
    hidden: Array,
    questions: Sequence[Question],
    lexicon: Callable[[Sequence[int]], Array],
    ids: Sequence[int],
) -> list[Array]:
    """JointSchemaHead.forward for one record: the option logits of every question."""
    seq = layer_norm(hidden, w, "hidden_norm")
    memory = linear(seq, w, "memory_projection", bias=False)
    global_vector = seq[-1]
    qvec = np.stack([seq[s:e].mean(0) for s, e in (q.span for q in questions)])
    contexts = [np.stack([seq[b:e].mean(0) for _, (b, e) in q.options]) for q in questions]
    lexical = [np.stack([lexicon(ids[b:e]).mean(0) for _, (b, e) in q.options]) for q in questions]
    queries = np.concatenate(
        [
            linear(c, w, "option_context_projection", bias=False)
            + linear(x, w, "option_lexical_projection", bias=False)
            + linear(qvec[i], w, "option_question_projection", bias=False)[None]
            for i, (c, x) in enumerate(zip(contexts, lexical, strict=True))
        ]
    )
    routed = np.split(
        _routed(w, config, seq, memory, queries),
        np.cumsum([len(q.options) for q in questions])[:-1],
    )
    base = linear(qvec, w, "question_projection", bias=False)
    summaries = [
        (softmax(options @ field / math.sqrt(options.shape[-1]), 0)[:, None] * options).sum(0)
        for field, options in zip(base, routed, strict=True)
    ]
    fields = (
        base
        + layer_norm(np.stack(summaries), w, "option_summary_norm")
        + linear(global_vector, w, "global_projection", bias=False)[None]
        + w["type_embedding.weight"][[q.kind for q in questions]]
    )
    fields = _decoded(w, config, fields, memory)
    prior_scale = math.exp(min(float(w["prior_logit_scale"]), LOGIT_SCALE_CAP))
    joint_scale = math.exp(min(float(w["joint_logit_scale"]), LOGIT_SCALE_CAP))
    gate = 1 / (1 + math.exp(-float(w["residual_gate"])))
    logits: list[Array] = []
    for i, (field, lex, options) in enumerate(zip(fields, lexical, routed, strict=True)):
        prior = prior_scale * (unit(lex) @ unit(qvec[i] + global_vector))
        options = layer_norm(options, w, "option_norm")
        repeated = np.broadcast_to(field[None], options.shape)
        cosine = (unit(repeated) * unit(options)).sum(-1)
        features = np.concatenate(
            [repeated, options, repeated * options, np.abs(repeated - options)], axis=-1
        )
        residual = linear(gelu(linear(features, w, "residual_scorer.0")), w, "residual_scorer.3")
        logits.append(prior + gate * (joint_scale * cosine + residual[:, 0]))
    return logits


def answer(question: Mapping[str, object], probabilities: Mapping[str, float]) -> Json:
    """One SystemOne answer from the per-option probabilities of a question."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(key) for key in _mapping(question["criteria"], "criteria")]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {key: round(probabilities[key], 4) for key in options},
        }
    levels = [str(i) for i in range(len(probabilities))]
    return {
        "type": "score",
        "score": round(sum(i * probabilities[level] for i, level in enumerate(levels)), 4),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "probabilities": {level: round(probabilities[level], 4) for level in levels},
    }


class Decider:
    """clef's SystemOne endpoint over a backbone and the joint schema head's weights."""

    def __init__(self, backbone: Backbone, weights: Mapping[str, Array], config: HeadConfig):
        self.backbone, self.weights, self.config = backbone, weights, config

    def respond(self, body: Json) -> Json:
        request = body
        ids, questions = encode(self.backbone.tokenize, request)
        hidden = self.backbone.hidden(ids)
        logits = head(self.weights, self.config, hidden, questions, self.backbone.lexicon, ids)
        asked = _mapping(request["questions"], "questions")
        answers = {
            q.qid: answer(
                _mapping(asked[q.qid], q.qid),
                dict(zip([oid for oid, _ in q.options], softmax(row).tolist(), strict=True)),
            )
            for q, row in zip(questions, logits, strict=True)
        }
        return {
            "model": request.get("model"),
            "answers": answers,
            "usage": {"input_tokens": len(ids), "output_tokens": 0},
        }


class MlxBackbone:
    """The 4-bit clef backbone on MLX through mlx-lm (``uv sync --extra clips``)."""

    def __init__(self, model_dir: Path):
        self.mx = import_module("mlx.core")
        self.model, self.tokenizer = import_module("mlx_lm").load(str(model_dir))
        lm = self.model.language_model
        self.text = lm.model
        self.output = lm.model.embed_tokens if lm.args.tie_word_embeddings else lm.lm_head

    def tokenize(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def hidden(self, ids: Sequence[int]) -> Array:
        mx = self.mx
        cache = self.model.make_cache()
        parts = []
        for at in range(0, len(ids), PREFILL_CHUNK):
            out = self.text(mx.array(list(ids[at : at + PREFILL_CHUNK]))[None], cache=cache)[0]
            parts.append(np.array(out.astype(mx.float32)))
        mx.clear_cache()
        return np.concatenate(parts).astype(np.float32)

    def lexicon(self, ids: Sequence[int]) -> Array:
        mx, output = self.mx, self.output
        rows = mx.array(list(ids))
        if hasattr(output, "scales"):
            values = mx.dequantize(
                output.weight[rows], output.scales[rows], output.biases[rows],
                output.group_size, output.bits,
            )  # fmt: skip
        else:
            values = output.weight[rows]
        return np.array(values.astype(mx.float32))


def load_decider(model_dir: Path) -> Decider:
    mx = import_module("mlx.core")
    config = json.loads((model_dir / "joint_head_config.json").read_text(encoding="utf-8"))
    raw = mx.load(str(model_dir / "joint_head.safetensors"))
    weights = {key: np.array(value.astype(mx.float32)) for key, value in raw.items()}
    head_config = HeadConfig(config["heads"], config["routing_layers"], config["layers"])
    return Decider(MlxBackbone(model_dir), weights, head_config)


class MlxChat:
    """Ternary-Bonsai-2-9B on mlx-vlm, text only, thinking off and greedy, so equal asks agree."""

    def __init__(self, model_dir: Path):
        vlm = import_module("mlx_vlm")
        self.generate = vlm.generate
        self.model, self.processor = vlm.load(str(model_dir))
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    def respond(self, body: Json) -> Json:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ReplayError("A chat request needs messages")
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        asked = body.get("max_completion_tokens")
        limit = min(asked, CHAT_TOKENS) if isinstance(asked, int) and asked > 0 else CHAT_TOKENS
        result = self.generate(
            self.model, self.processor, prompt, max_tokens=limit, temperature=0.0, verbose=False
        )
        return {"choices": [{"message": {"role": "assistant", "content": str(result.text)}}]}


def free_percent(run: Callable[[list[str]], str]) -> int:
    found = FREE_PATTERN.search(run(["memory_pressure"]))
    if not found:
        raise ReplayError("memory_pressure did not report the free memory percentage")
    return int(found.group(1))


def _run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def watch(
    run: Callable[[list[str]], str],
    stop: Callable[[int], None],
    *,
    sleep: Callable[[float], None] = time.sleep,
    rounds: int | None = None,
) -> None:
    """Stops the process once free memory falls below MIN_RUNNING_FREE_PERCENT."""
    done = 0
    while rounds is None or done < rounds:
        sleep(WATCH_SECONDS)
        done += 1
        free = free_percent(run)
        if free < MIN_RUNNING_FREE_PERCENT:
            print(f"stopping: only {free}% of memory is free", file=sys.stderr, flush=True)
            stop(1)
            return


def handler(routes: Mapping[str, Answerer]) -> type[BaseHTTPRequestHandler]:
    """POST /<name> answers a JSON body with the model registered under that name."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server's hook name)
            route = routes.get(self.path.strip("/"))
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)))
                if route is None:
                    self._send(404, {"error": f"unknown model {self.path}"})
                    return
                self._send(200, route.respond(_mapping(body, "request")))
            except (ReplayError, ValueError, KeyError) as error:
                self._send(400, {"error": str(error)})

        def _send(self, status: int, payload: Json) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            del format, args  # one line per answer would flood a pipeline step's log

    return Handler


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Callable[[list[str]], str] = _run,
    decider: Callable[[Path], Answerer] = load_decider,
    writer: Callable[[Path], Answerer] = MlxChat,
    server: Callable[[tuple[str, int], type[BaseHTTPRequestHandler]], Serving] = HTTPServer,
    guard: Callable[[Callable[[list[str]], str]], None] = lambda run: threading.Thread(
        target=watch, args=(run, os._exit), daemon=True
    ).start(),
) -> int:
    parser = argparse.ArgumentParser(prog="python -m replay.clip_models", description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--clef", type=Path, default=CLEF_DIR)
    parser.add_argument("--bonsai", type=Path, default=BONSAI_DIR)
    parser.add_argument("--min-free", type=int, default=MIN_FREE_PERCENT)
    args = parser.parse_args(argv)
    for folder in (args.clef, args.bonsai):
        if not (folder / "config.json").is_file():
            raise ReplayError(f"Model not found (it is never downloaded here): {folder}")
    free = free_percent(run)
    if free < args.min_free:
        raise ReplayError(f"Only {free}% of memory is free; the local models need {args.min_free}%")
    routes: dict[str, Answerer] = {"clef": decider(args.clef), "bonsai": writer(args.bonsai)}
    http = server(("127.0.0.1", args.port), handler(routes))
    guard(run)
    print(f"ready {args.port}", flush=True)
    http.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
