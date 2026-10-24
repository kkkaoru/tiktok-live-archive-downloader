import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import caption_linguistics, clip_terms


@dataclass
class FakeToken:
    text: str
    features: str

    def surface(self) -> str:
        return self.text

    def feature(self) -> str:
        return self.features


class FakeTokenizer:
    """Splits on spaces; each word is `surface/feature,feature,...`."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def tokenize(self, text: str) -> Iterable[FakeToken]:
        self.seen.append(text)
        return [FakeToken(*word.split("/", 1)) for word in text.split(" ") if word]


def test_hints_keep_content_nouns_once_by_base_form() -> None:
    text = (
        "冷やし中華/名詞,一般,*,*,*,*,冷やし中華 は/助詞,係助詞 "
        "アイス/名詞,一般,*,*,*,*,アイス 冷やし中華/名詞,一般,*,*,*,*,冷やし中華 "
        "目/名詞,一般,*,*,*,*,目 こと/名詞,非自立,一般 2024/名詞,一般,*,*,*,*,2024 "
        "カシ/名詞,一般"
    )
    assert clip_terms.hints(text, FakeTokenizer()) == (
        ["冷やし中華", "アイス", "カシ"],
        True,
        True,
    )


def test_hints_mark_a_cue_that_continues_a_sentence() -> None:
    assert clip_terms.hints("を/助詞,格助詞 語った/動詞,自立", FakeTokenizer()) == ([], False, True)
    assert clip_terms.hints("ん/名詞,非自立 だ/助動詞", FakeTokenizer()) == ([], False, True)
    assert clip_terms.hints("られる/動詞,接尾", FakeTokenizer()) == ([], False, True)


def test_hints_mark_a_cue_that_leaves_its_sentence_hanging() -> None:
    assert clip_terms.hints("部屋/名詞,代名詞 に/助詞,格助詞", FakeTokenizer()) == ([], True, False)
    assert clip_terms.hints("行っ/動詞,自立 て/助詞,接続助詞 。/記号,句点", FakeTokenizer()) == (
        [],
        True,
        False,
    )
    assert clip_terms.hints("それ/名詞,代名詞 は/助詞,係助詞", FakeTokenizer()) == ([], True, False)


def test_hints_skip_symbols_and_fillers_before_the_first_word() -> None:
    assert clip_terms.hints("「/記号,括弧開 えー/フィラー,* ね/助詞,終助詞", FakeTokenizer()) == (
        [],
        False,
        True,
    )
    assert clip_terms.hints("、/記号,読点", FakeTokenizer()) == ([], True, True)


def test_hints_ignore_caption_line_breaks() -> None:
    tokenizer = FakeTokenizer()
    clip_terms.hints("部屋/名詞,一般\nに/助詞,格助詞", tokenizer)
    assert tokenizer.seen == ["部屋/名詞,一般に/助詞,格助詞"]


def test_annotate_directory_adds_hints_to_every_data_file(tmp_path: Path) -> None:
    (tmp_path / "catalog.json").write_text('{"videos":[]}', encoding="utf-8")
    (tmp_path / "a.json").write_text(
        json.dumps({"id": "a", "cues": [{"start": 0, "end": 1, "text": "部屋/名詞,一般"}]}),
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text('{"id":"b"}', encoding="utf-8")
    assert clip_terms.annotate_directory(tmp_path, FakeTokenizer()) == 1
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")) == {
        "id": "a",
        "cues": [
            {
                "start": 0,
                "end": 1,
                "text": "部屋/名詞,一般",
                "terms": ["部屋"],
                "opens": True,
                "closes": True,
            }
        ],
    }
    assert (tmp_path / "b.json").read_text(encoding="utf-8") == '{"id":"b"}'
    assert (tmp_path / "catalog.json").read_text(encoding="utf-8") == '{"videos":[]}'


def test_main_annotates_with_the_pinned_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.json").write_text('{"cues":[{"text":"を/助詞"}]}', encoding="utf-8")
    monkeypatch.setattr(caption_linguistics, "_tokenizer", FakeTokenizer)
    assert clip_terms.main([str(tmp_path)]) == 0
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")) == {
        "cues": [{"text": "を/助詞", "terms": [], "opens": False, "closes": True}]
    }
    assert capsys.readouterr().err == "annotated 1 data files\n"
