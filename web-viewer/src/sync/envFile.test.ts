import { expect, it } from "vitest";
import { parseEnvFile } from "./envFile.ts";

it("parses assignments, quotes and skips comments", () => {
  expect(
    parseEnvFile('# note\nVIEWER_DOMAIN=a.example.com\n\nQUOTED="x y"\nbad line\n'),
  ).toStrictEqual({
    VIEWER_DOMAIN: "a.example.com",
    QUOTED: "x y",
  });
});

it("handles single quotes, invalid keys and values with equals signs", () => {
  expect(parseEnvFile("A='x'\n1BAD=2\n=nokey\nB=c=d\n")).toStrictEqual({ A: "x", B: "c=d" });
});

it("returns an empty object for empty text", () => {
  expect(parseEnvFile("")).toStrictEqual({});
});
