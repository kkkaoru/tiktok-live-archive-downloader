import { expect, it } from "vitest";
import { parseOptions } from "./options.ts";

it("defaults to a full sync", () => {
  expect(parseOptions([])).toStrictEqual({ dryRun: false, dataOnly: false, originalsOnly: false });
});

it("reads every flag", () => {
  expect(parseOptions(["--dry-run", "--data-only", "--originals-only"])).toStrictEqual({
    dryRun: true,
    dataOnly: true,
    originalsOnly: true,
  });
});

it("reads the video ID of --only", () => {
  expect(parseOptions(["--only", "a-review", "--dry-run"])).toStrictEqual({
    dryRun: true,
    dataOnly: false,
    originalsOnly: false,
    only: "a-review",
  });
});

it("rejects unknown flags", () => {
  expect(() => parseOptions(["--nope"])).toThrow();
});
