"""zenz scoring without transformers: GPT-2 BPE ids and a tiny numpy GPT-2 forward pass."""

import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from replay import zenz_lm
from replay.core import ReplayError

SYMBOLS = zenz_lm._byte_symbols()
# Specials first, then every byte symbol except 0xBF (so "ÿ", C3 BF, has an unknown byte).
VOCABULARY = {
    "[UNK]": 0,
    "[PAD]": 1,
    "<s>": 2,
    "</s>": 3,
    **{SYMBOLS[byte]: 4 + byte for byte in range(256) if byte != 0xBF},
    "ab": 300,
    "abc": 301,
}
MERGES = [("a", "b"), ("ab", "c")]
ROWS = 302  # embedding rows: the largest id + 1


def tokenizer() -> zenz_lm.Tokenizer:
    return zenz_lm.Tokenizer(VOCABULARY, MERGES)


def weights(scale: float, seed: int = 0) -> dict[str, np.ndarray]:
    random = np.random.default_rng(seed)
    width, vocabulary = 8, ROWS

    def tensor(*shape: int) -> np.ndarray:
        return random.normal(0.0, scale, shape)

    return {
        "transformer.wte.weight": tensor(vocabulary, width),
        "transformer.wpe.weight": tensor(64, width),
        "transformer.h.0.ln_1.weight": tensor(width) + 1.0,
        "transformer.h.0.ln_1.bias": tensor(width),
        "transformer.h.0.attn.c_attn.weight": tensor(width, 3 * width),
        "transformer.h.0.attn.c_attn.bias": tensor(3 * width),
        "transformer.h.0.attn.c_proj.weight": tensor(width, width),
        "transformer.h.0.attn.c_proj.bias": tensor(width),
        "transformer.h.0.ln_2.weight": tensor(width) + 1.0,
        "transformer.h.0.ln_2.bias": tensor(width),
        "transformer.h.0.mlp.c_fc.weight": tensor(width, 4 * width),
        "transformer.h.0.mlp.c_fc.bias": tensor(4 * width),
        "transformer.h.0.mlp.c_proj.weight": tensor(4 * width, width),
        "transformer.h.0.mlp.c_proj.bias": tensor(width),
        "transformer.ln_f.weight": tensor(width) * 0.0 + (scale > 0),
        "transformer.ln_f.bias": tensor(width),
    }


def test_bpe_applies_merges_by_rank_and_pretokenizes_spaces() -> None:
    assert tokenizer().encode("abc ab") == [301, 4 + 0x20, 300]


def test_bytes_outside_the_vocabulary_map_to_unknown() -> None:
    assert tokenizer().encode("ÿ") == [4 + 0xC3, 0]


def test_cache_returns_the_same_ids() -> None:
    model = tokenizer()
    assert model.encode("abab") == model.encode("abab") == [300, 300]


def test_vocabulary_needs_the_special_tokens() -> None:
    with pytest.raises(ReplayError, match="<s>"):
        zenz_lm.Tokenizer({"[UNK]": 0, "</s>": 1}, [])


def test_tokenizer_loads_vocab_and_versioned_merges(tmp_path: Path) -> None:
    (tmp_path / "vocab.json").write_text(json.dumps(VOCABULARY), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2\na b\nab c\n", encoding="utf-8")
    loaded = zenz_lm.Tokenizer.load(tmp_path)
    assert (loaded.bos, loaded.eos, loaded.unknown, loaded.encode("abc")) == (2, 3, 0, [301])


def test_malformed_merges_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "vocab.json").write_text(json.dumps(VOCABULARY), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("a b c\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="merges"):
        zenz_lm.Tokenizer.load(tmp_path)


def test_malformed_vocabulary_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "vocab.json").write_text('{"a": "b"}', encoding="utf-8")
    (tmp_path / "merges.txt").write_text("", encoding="utf-8")
    with pytest.raises(ReplayError, match="vocab"):
        zenz_lm.Tokenizer.load(tmp_path)


def test_prompt_is_the_v2_family_evaluation_prompt() -> None:
    assert zenz_lm.prompt("カ", "x" * 45) == "カ" + "x" * 40 + ""
    assert zenz_lm.prompt("カ", "") == "カ"


def test_spaces_become_ideographic_and_newlines_vanish() -> None:
    assert zenz_lm.normalized("a b\nc") == "a　bc"


def test_zero_model_assigns_every_token_the_uniform_probability() -> None:
    scorer = zenz_lm.Scorer(tokenizer(), zenz_lm.Model(weights(0.0), np, heads=2))
    expected = -2 * math.log(ROWS)  # "abc" then </s>
    assert scorer.score([("ab", "c", "abc")]) == [pytest.approx(expected)]


def test_right_padding_does_not_change_scores() -> None:
    model = zenz_lm.Model(weights(0.5), np, heads=2)
    items = [("ab", "", "abc"), ("ab", "abc", "a b c ab")]
    alone = zenz_lm.Scorer(tokenizer(), model, batch=1).score(items)
    batched = zenz_lm.Scorer(tokenizer(), model, batch=2).score(items)
    assert batched == pytest.approx(alone)
    assert alone[0] != pytest.approx(alone[1])


def test_input_longer_than_the_model_is_rejected() -> None:
    scorer = zenz_lm.Scorer(tokenizer(), zenz_lm.Model(weights(0.0), np, heads=2))
    with pytest.raises(ReplayError, match="1,024"):
        scorer.score([("a", "", "a" * 1100)])


def test_weights_must_be_gpt2() -> None:
    with pytest.raises(ReplayError, match="GPT-2"):
        zenz_lm.Model({"wte.weight": np.zeros((1, 1))}, np, heads=1)


def model_directory(tmp_path: Path, model_type: str) -> Path:
    (tmp_path / "vocab.json").write_text(json.dumps(VOCABULARY), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("a b\nab c\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type, "n_head": 2}), encoding="utf-8"
    )
    return tmp_path


def test_load_reads_the_checkpoint_through_mlx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = weights(0.0)
    fake = SimpleNamespace(
        float32=np.float32,
        load=lambda path: {name: value.astype(np.float16) for name, value in stored.items()},
        array=np.array,
        **{
            name: getattr(np, name)
            for name in ("mean", "sqrt", "triu", "full", "exp", "max", "sum", "tanh", "log")
        },
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)
    scorer = zenz_lm.load(model_directory(tmp_path, "gpt2"))
    assert scorer.score([("a", "", "a")]) == [pytest.approx(-2 * math.log(ROWS))]


def test_load_rejects_another_architecture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(ReplayError, match="GPT-2"):
        zenz_lm.load(model_directory(tmp_path, "llama"))
