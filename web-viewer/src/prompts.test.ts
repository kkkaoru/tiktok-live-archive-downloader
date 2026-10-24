// Runs with bun (vitest). The prompt catalog: both languages of an entry agree, English is sent.
import { expect, it } from "vitest";
import { LEVELS, PROMPTS, placeholders, prompt } from "./prompts.ts";

const JAPANESE: RegExp = /[぀-ヿ一-鿿]/;

it("gives every entry both languages with the same placeholders", () => {
  const mismatched = Object.entries(PROMPTS).filter(
    ([, e]) => JSON.stringify(placeholders(e.ja)) !== JSON.stringify(placeholders(e.en)),
  );
  const untranslated = Object.entries(PROMPTS).filter(
    ([, e]) => e.ja === "" || e.en === "" || JAPANESE.test(e.en) || !JAPANESE.test(e.ja),
  );
  expect([mismatched, untranslated]).toStrictEqual([[], []]);
});

it("keeps the level lists the same length in both languages, English without Japanese", () => {
  expect([
    LEVELS.interest.ja.length,
    LEVELS.interest.en.length,
    LEVELS.hook.ja.length,
    LEVELS.hook.en.length,
    [...LEVELS.interest.en, ...LEVELS.hook.en].some((l) => JAPANESE.test(l)),
  ]).toStrictEqual([5, 5, 5, 5, false]);
});

it("sends the Japanese text with its placeholders filled, the values left as given", () => {
  expect([
    prompt("region.settles", { unit: 3 }),
    prompt("clairic.line", { range: 1, text: "{unit} 爆発！" }),
    prompt("topic.strays"),
  ]).toStrictEqual([
    "区間3の最後の発話で、話がオチ・結論・反応の一区切りに達しているか（次の区間が同じ話の続きの途中なら低い）",
    "[区間1] {unit} 爆発！",
    "コメントの読み上げ・あいさつ・お礼・配信の進行・別の話題",
  ]);
});

it("refuses to send a prompt with a placeholder left empty", () => {
  expect(() => prompt("region.boundary", { utterance: 3 })).toThrow(
    "prompt region.boundary needs {previous}",
  );
  expect(() => prompt("region.settles")).toThrow("prompt region.settles needs {unit}");
});
