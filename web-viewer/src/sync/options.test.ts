import { expect, it } from "vitest";
import { parseOptions } from "./options.ts";

it("defaults to a full sync", () => {
  expect(parseOptions([])).toStrictEqual({ dryRun: false, dataOnly: false });
});

it("reads both flags", () => {
  expect(parseOptions(["--dry-run", "--data-only"])).toStrictEqual({
    dryRun: true,
    dataOnly: true,
  });
});

it("rejects unknown flags", () => {
  expect(() => parseOptions(["--nope"])).toThrow();
});
