"""Context-aware interleaved recognition: chunking, streams, guards, rounds and labels."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from replay import asr_context_streams as streams
from replay.asr_context_streams import Chunk
from replay.core import ReplayError

RATE = 16000


def tone(seconds: float, level: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * 220 * t) * level * 32767).astype(np.int16)


def hiss(seconds: float, level: float = 0.0005) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * RATE)) * level * 32767).astype(np.int16)


def test_pcm_must_be_mono_16k_pcm16(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(b"\0\0\0\0" * 10)
    with pytest.raises(ReplayError, match="mono PCM16"):
        streams.read_pcm(path)


def test_speech_is_found_above_the_noise_floor_ignoring_digital_silence() -> None:
    samples = np.concatenate([np.zeros(RATE, np.int16), hiss(2), tone(1), hiss(2)])
    speech = streams.speech_frames(streams.frame_levels(samples))
    seconds = np.nonzero(speech)[0] * streams.FRAME / RATE
    assert (round(float(seconds.min()), 1), round(float(seconds.max()), 1)) == (3.0, 4.0)


def test_only_digital_silence_has_no_speech() -> None:
    levels = streams.frame_levels(np.zeros(RATE, np.int16))
    assert not streams.speech_frames(levels).any()


def test_short_pauses_join_regions_and_long_ones_split_them() -> None:
    speech = np.array([1, 1, 0, 0, 1, 1] + [0] * 20 + [1], dtype=bool)
    assert streams.regions(speech) == [(0, 6), (26, 27)]


def test_a_long_region_is_cut_at_its_quietest_frame() -> None:
    levels = np.full(2000, -20.0)
    levels[[800, 1500]] = -60.0
    assert streams.split_long(0, 2000, levels) == [(0, 800), (800, 1500), (1500, 2000)]


def test_chunks_pack_speech_up_to_the_limit_and_pad_into_pauses() -> None:
    samples = np.concatenate([hiss(1), tone(10), hiss(1), tone(10), hiss(1), tone(15), hiss(1)])
    chunks = streams.speech_chunks(samples)
    spans = [(round(c.start / RATE, 2), round(c.end / RATE, 2)) for c in chunks]
    assert spans == [(0.84, 22.17), (22.83, 38.16)]


def test_streams_are_contiguous_and_balanced() -> None:
    chunks = [Chunk(i, i + 1) for i in range(5)]
    assert [[c.start for c in s] for s in streams.streams_of(chunks, 2)] == [[0, 1, 2], [3, 4]]
    assert streams.streams_of([], 3) == []


def segment(text: str, ratio: float = 1.0, logprob: float = -0.2) -> dict[str, object]:
    return {"text": text, "compression_ratio": ratio, "avg_logprob": logprob}


def test_failing_flags_repetition_improbability_and_prompt_echo() -> None:
    assert streams.failing({"segments": [segment("a", ratio=3.0)]}, "") == "repetition"
    assert streams.failing({"segments": [segment("a", logprob=-2.0)]}, "") == "improbable"
    echo = {"text": "前回の話の続きです今日は", "segments": [segment("x")]}
    assert streams.failing(echo, "そして前回の話の続きです") == "prompt-echo"
    assert (
        streams.failing({"text": "今日は", "segments": [segment("今日は")]}, "前の文脈です長い")
        is None
    )


def test_a_prompted_failure_is_decoded_again_without_the_prompt() -> None:
    calls: list[str] = []

    def decode(chunk: Chunk, prompt: str) -> dict[str, object]:
        calls.append(prompt)
        return {"text": "t", "segments": [segment("t")]}

    first = {"text": "t", "segments": [segment("t", ratio=3.0)]}
    result = streams.guarded(Chunk(0, 1), "文脈", first, decode)
    assert calls == [""]
    assert result["guard"] == {
        "prompted": True, "retry": "repetition", "droppedTail": None, "trusted": True,
    }  # fmt: skip


def test_untrusted_or_disabled_results_do_not_become_context() -> None:
    good = {"text": "続き", "guard": {"trusted": True}}
    bad = {"text": "ああああ", "guard": {"trusted": False}}
    assert streams.carried("前", good, enabled=True) == "前続き"
    assert streams.carried("前", bad, enabled=True) == ""
    assert streams.carried("前", good, enabled=False) == ""


def test_rounds_interleave_streams_carry_context_and_restart_cold_streams() -> None:
    plan = [[Chunk(0, 1), Chunk(1, 2)], [Chunk(2, 3), Chunk(3, 4)]]
    rounds: list[list[tuple[int, int, str]]] = []

    def decode_round(todo: list[tuple[int, Chunk, str]]) -> list[dict[str, object]]:
        rounds.append([(index, chunk.start, prompt) for index, chunk, prompt in todo])
        return [{"text": f"c{chunk.start}", "guard": {"trusted": True}} for _i, chunk, _p in todo]

    results = streams.interleaved(plan, decode_round, context=True)
    assert rounds == [
        [(0, 0, ""), (1, 2, "")],
        [(0, 1, "c0"), (1, 3, "c2")],
        [(1, 2, "c0c1")],
    ]
    assert results["000000000002"] == {"text": "c2", "guard": {"trusted": True}, "coldStart": True}


def test_rounds_without_context_and_with_finished_chunks() -> None:
    plan = [[Chunk(0, 1), Chunk(1, 2)]]
    seen: list[tuple[int, str]] = []
    finished = {"000000000000": {"text": "done", "guard": {"trusted": True}}}

    def decode_round(todo: list[tuple[int, Chunk, str]]) -> list[dict[str, object]]:
        seen.extend((chunk.start, prompt) for _i, chunk, prompt in todo)
        return [{"text": "new", "guard": {"trusted": True}} for _ in todo]

    results = streams.interleaved(
        plan, decode_round, context=False, done=lambda chunk: finished.get(chunk.name)
    )
    assert seen == [(1, "")]
    assert [results[name]["text"] for name in sorted(results)] == ["done", "new"]


def test_labels_are_offset_to_the_input_clock_with_chunk_provenance() -> None:
    result = {
        "language": "ja",
        "text": "こんにちは",
        "chunk": {"start": 32000, "end": 48000},
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "こんにちは",
             "words": [{"word": "こんにちは", "start": 0.1, "end": 0.9, "probability": 0.9}]}
        ],
    }  # fmt: skip
    empty = {"text": " ", "chunk": {"start": 0, "end": 16000}, "segments": []}
    rows = streams.labels({"000000032000": result, "000000000000": empty})
    assert rows == [
        {"text": "こんにちは", "start": 2.1, "end": 2.9, "provenance": "000000032000:0"}
    ]


def fake_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    r = json.loads(line)\n"
        "    t = 'x' if r['prompt'] else 'y'\n"
        "    print(json.dumps({'language': 'ja', 'text': t, 'segments': [{'start': 0.0,"
        " 'end': 0.5, 'text': t, 'compression_ratio': 1.0, 'avg_logprob': -0.1,"
        " 'words': [{'word': t, 'start': 0.0, 'end': 0.5, 'probability': 0.9}]}]}),"
        " flush=True)\n",
        encoding="utf-8",
    )

    class Launcher(streams.Worker):
        def __init__(self, audio: Path, model: Path) -> None:
            import subprocess
            import sys

            self.process = subprocess.Popen(
                [sys.executable, str(script)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, text=True, encoding="utf-8",
            )  # fmt: skip

    monkeypatch.setattr(streams, "Worker", Launcher)


def test_run_writes_resumable_chunks_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_worker(tmp_path, monkeypatch)
    audio = tmp_path / "mono.wav"
    with wave.open(str(audio), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(np.concatenate([hiss(3), tone(2), hiss(3), tone(29), hiss(3)]).tobytes())
    summary = streams.run(audio, tmp_path / "model", tmp_path / "out", streams=2, context=True)
    assert (summary["chunks"], summary["streams"], summary["retries"]) == (2, 2, 0)
    written = json.loads((tmp_path / "out/labels.json").read_text(encoding="utf-8"))
    assert [row["text"] for row in written["labels"]] == ["y", "x"]
    assert len(list((tmp_path / "out/chunks").glob("*.json"))) == 2
    again = streams.run(audio, tmp_path / "model", tmp_path / "out", streams=2, context=True)
    assert again["chunks"] == 2


def test_worker_pipes_must_exist() -> None:
    import subprocess

    worker = object.__new__(streams.Worker)
    worker.process = subprocess.Popen(["true"], text=True)
    worker.process.wait()
    with pytest.raises(ReplayError, match="input pipe"):
        worker.send(Chunk(0, 1), "")
    with pytest.raises(ReplayError, match="output pipe"):
        worker.receive()


def test_the_cli_needs_an_output_for_run(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="output directory"):
        streams.main(["run", str(tmp_path / "a.wav"), str(tmp_path / "m")])


def write_audio(path: Path) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(np.concatenate([hiss(3), tone(2), hiss(3), tone(29), hiss(3)]).tobytes())


def test_publish_binds_labels_to_a_projection_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_worker(tmp_path, monkeypatch)
    audio, model = tmp_path / "mono.wav", tmp_path / "model"
    write_audio(audio)
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    labels, manifest = tmp_path / "labels.json", tmp_path / "asr/inputs.json"
    streams.main(["publish", str(audio), str(model), str(tmp_path / "work"), str(labels),
                  str(manifest)])  # fmt: skip
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert (written["clock"], written["samples"], written["sampleRate"]) == (
        "input-pcm-not-projected", 16000 * 40, 16000,
    )  # fmt: skip
    assert str(audio) in written["inputs"] and str(model / "config.json") in written["inputs"]
    document = json.loads(labels.read_text(encoding="utf-8"))
    assert list(document["parentSHA256"]) == [str(manifest)]
    assert [row["text"] for row in document["labels"]] == ["y", "x"]
    with pytest.raises(ReplayError, match="overwrite"):
        streams.publish(audio, model, tmp_path / "work", labels, manifest)
    with pytest.raises(ReplayError, match="labels and manifest"):
        streams.main(["publish", str(audio), str(model), str(tmp_path / "work")])


def test_the_worker_decodes_each_request_with_the_pinned_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    import sys
    import types

    audio = tmp_path / "mono.wav"
    write_audio(audio)
    seen: list[dict[str, object]] = []

    def transcribe(samples: np.ndarray, **options: object) -> dict[str, object]:
        seen.append({"length": len(samples), **options})
        return {"text": "t", "score": np.float32(0.5)}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    cleared: list[int] = []
    monkeypatch.setitem(
        sys.modules, "mlx.core", types.SimpleNamespace(clear_cache=lambda: cleared.append(1))
    )
    requests = [{"start": 0, "end": 1600, "prompt": "前"}, {"start": 0, "end": 800, "prompt": ""}]
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(r) + "\n" for r in requests)))
    assert streams.main(["worker", str(audio), str(tmp_path / "model")]) == 0
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [
        {"text": "t", "score": 0.5}, {"text": "t", "score": 0.5},
    ]  # fmt: skip
    assert [(item["length"], item["initial_prompt"], item["language"]) for item in seen] == [
        (1600, "前", "ja"), (800, None, "ja"),
    ]  # fmt: skip
    assert len(cleared) == 2


def test_trimming_keeps_long_or_spoken_last_segments() -> None:
    long = {"text": "a", "segments": [{"start": 0.0, "end": 2.0, "no_speech_prob": 0.9}]}
    spoken = {"text": "a", "segments": [{"start": 0.0, "end": 0.3, "no_speech_prob": 0.1}]}
    phantom = {"text": "ab", "segments": [
        {"start": 0.0, "end": 2.0, "text": "a", "no_speech_prob": 0.1},
        {"start": 2.0, "end": 2.3, "text": "b", "no_speech_prob": 0.9},
    ]}  # fmt: skip
    assert streams.trimmed(long) == (long, None)
    assert streams.trimmed(spoken) == (spoken, None)
    assert streams.trimmed({"text": ""}) == ({"text": ""}, None)
    kept, tail = streams.trimmed(phantom)
    assert (kept["text"], tail) == ("a", "b")


def test_mono_is_decoded_from_the_vocals_with_a_receipt(tmp_path: Path) -> None:
    import shutil

    vocals = tmp_path / "vocals.wav"
    with wave.open(str(vocals), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(44100)
        out.writeframes(np.zeros(44100 * 2, dtype=np.int16).tobytes())
    ffmpeg = Path(str(shutil.which("ffmpeg"))).resolve()
    mono, receipt = tmp_path / "mono.wav", tmp_path / "mono-receipt.json"
    assert streams.main(["mono", str(vocals), str(ffmpeg), str(mono), str(receipt)]) == 0
    written = json.loads(receipt.read_text(encoding="utf-8"))
    assert (written["status"], written["samples"], written["sampleRate"]) == (
        "complete",
        16000,
        16000,
    )
    assert list(written["outputs"]) == [str(mono)]
    assert streams.prepare_mono(vocals, ffmpeg, mono, receipt) == written
    with pytest.raises(ReplayError, match="receipt path"):
        streams.main(["mono", str(vocals), str(ffmpeg), str(mono)])


def test_long_word_clocks_shrink_to_their_loudest_voiced_run() -> None:
    speech = np.zeros(400, dtype=bool)
    speech[200:210] = True  # 6.0 s .. 6.3 s
    speech[100:102] = True  # a short blip at 3.0 s
    rows = [
        {"text": "っく", "start": 2.0, "end": 9.0, "provenance": "p:0"},
        {"text": "短い", "start": 6.0, "end": 7.0, "provenance": "p:1"},
        {"text": "無音", "start": 10.0, "end": 12.5, "provenance": "p:2"},
    ]
    result, count = streams.tightened(rows, speech)
    assert count == 1
    assert (round(float(str(result[0]["start"])), 2), round(float(str(result[0]["end"])), 2)) == (
        6.0,
        6.3,
    )
    assert result[0]["voicedClamp"] == {"originalStart": 2.0, "originalEnd": 9.0}
    assert result[1:] == rows[1:]


def test_a_word_starting_before_the_voice_starts_at_its_first_voiced_frame() -> None:
    voice = np.zeros(100, dtype=bool)
    voice[40:60] = True  # 1.2 s .. 1.8 s
    rows = [
        {"text": "早い", "start": 1.0, "end": 1.5, "provenance": "p:0"},
        {"text": "声", "start": 1.3, "end": 1.5, "provenance": "p:1"},
        {"text": "無声", "start": 2.0, "end": 2.4, "provenance": "p:2"},
    ]
    result, count = streams.onset_snapped(rows, voice)
    assert count == 1
    assert (round(float(str(result[0]["start"])), 2), result[0]["onsetSnap"]) == (
        1.2, {"originalStart": 1.0},
    )  # fmt: skip
    assert result[1:] == rows[1:]


def test_a_moved_start_keeps_a_display_frame_before_the_next_word() -> None:
    voice = np.zeros(100, dtype=bool)
    voice[40:60] = True  # first voice at 1.2 s
    rows = [
        {"text": "あ", "start": 1.0, "end": 1.3, "provenance": "p:0"},
        {"text": "い", "start": 1.1, "end": 1.5, "provenance": "p:1"},
    ]
    result, count = streams.onset_snapped(rows, voice)
    starts = [round(float(str(row["start"])), 3) for row in result]
    assert (count, starts) == (2, [1.067, 1.2])


def test_a_fragment_before_a_quiet_pause_inside_a_word_moves_up_to_the_rest_of_it() -> None:
    voice = np.zeros(100, dtype=bool)
    rows = [
        {"text": "ありがとう", "start": 0.0, "end": 0.3, "provenance": "p:0"},
        {"text": "ス", "start": 0.3, "end": 1.9, "provenance": "p:1"},
        {"text": "マ", "start": 2.2, "end": 2.3, "provenance": "p:2"},
        {"text": "ホ", "start": 2.3, "end": 2.4, "provenance": "p:3"},
    ]
    result, count = streams.words_joined(rows, voice)
    assert count == 1
    assert (round(float(str(result[1]["start"])), 3), round(float(str(result[1]["end"])), 3)) == (
        2.05, 2.2,
    )  # fmt: skip
    assert result[1]["wordJoin"] == {"originalStart": 0.3, "originalEnd": 1.9}
    assert [result[0], result[2], result[3]] == [rows[0], rows[2], rows[3]]


def test_a_voiced_or_long_pause_or_a_chunk_change_keeps_the_fragments_apart() -> None:
    voiced = np.zeros(400, dtype=bool)
    voiced[63:73] = True  # 1.9 s .. 2.2 s
    rows = [
        {"text": "ス", "start": 0.3, "end": 1.9, "provenance": "p:1"},
        {"text": "マホ", "start": 2.2, "end": 2.4, "provenance": "p:2"},
    ]
    assert streams.words_joined(rows, voiced) == (rows, 0)
    far = [rows[0], {"text": "マホ", "start": 10.0, "end": 10.2, "provenance": "p:2"}]
    assert streams.words_joined(far, np.zeros(400, dtype=bool)) == (far, 0)
    split = [rows[0], {"text": "マホ", "start": 2.2, "end": 2.4, "provenance": "q:0"}]
    assert streams.words_joined(split, np.zeros(400, dtype=bool)) == (split, 0)


def test_a_short_gap_inside_a_word_closes_by_extending_the_earlier_fragment() -> None:
    rows = [
        {"text": "か", "start": 0.0, "end": 0.1, "provenance": "p:0"},
        {"text": "わいい", "start": 0.15, "end": 0.4, "provenance": "p:1"},
    ]
    result, count = streams.words_joined(rows, np.zeros(100, dtype=bool))
    assert (count, result[0]["end"], result[1]) == (0, 0.15, rows[1])


def test_a_short_gap_after_a_long_fragment_extends_it_instead_of_moving_it() -> None:
    rows = [
        {"text": "ハロ", "start": 0.0, "end": 0.4, "provenance": "p:0"},
        {"text": "ーバ", "start": 0.4, "end": 1.9, "provenance": "p:1"},
        {"text": "ラー", "start": 2.0, "end": 2.2, "provenance": "p:2"},
    ]
    result, count = streams.words_joined(rows, np.zeros(100, dtype=bool))
    assert (count, result[1]["start"], result[1]["end"], "wordJoin" in result[1]) == (
        0, 0.4, 2.0, False,
    )  # fmt: skip
