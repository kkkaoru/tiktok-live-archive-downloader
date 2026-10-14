import { expect, it } from "vitest";
import { findWorkersUrl, listRemoteSizes, putJson } from "./cf.ts";
import { createFakeRunner } from "./fakes.fake.ts";

it("merges the media and original listings", async () => {
  const { runner, calls } = createFakeRunner((call) => ({
    stdout: `warning\n${JSON.stringify([{ key: `${call.args[6]}x.mp4`, size: call.args[6] === "media/" ? 1 : 2 }])}\n`,
  }));
  const sizes = await listRemoteSizes({ runner, cwd: "/p", bucket: "b" });
  expect(sizes).toStrictEqual({ "media/x.mp4": 1, "media-original/x.mp4": 2 });
  expect(calls.map((call) => call.args.slice(0, 4))).toStrictEqual([
    ["r2", "objects", "list", "--bucket-name"],
    ["r2", "objects", "list", "--bucket-name"],
  ]);
});

it("fails when a listing command fails", async () => {
  const { runner } = createFakeRunner(() => ({ code: 1, stderr: "boom" }));
  await expect(listRemoteSizes({ runner, cwd: "/p", bucket: "b" })).rejects.toThrow(
    "cf r2 objects list failed: boom",
  );
});

it("fails when a listing has no JSON", async () => {
  const { runner } = createFakeRunner(() => ({ stdout: "nothing" }));
  await expect(listRemoteSizes({ runner, cwd: "/p", bucket: "b" })).rejects.toThrow(
    "cf returned no JSON list",
  );
});

it("puts a JSON object with the right arguments", async () => {
  const { runner, calls } = createFakeRunner(() => ({}));
  await putJson({ runner, cwd: "/p", bucket: "b", key: "data/a.json", path: "/f/a.json" });
  expect(calls).toStrictEqual([
    {
      command: "cf",
      args: [
        "r2",
        "objects",
        "put",
        "data/a.json",
        "--bucket-name",
        "b",
        "--file",
        "/f/a.json",
        "--content-type",
        "application/json",
      ],
      cwd: "/p",
    },
  ]);
});

it("fails when a put fails", async () => {
  const { runner } = createFakeRunner(() => ({ code: 1, stderr: "denied" }));
  await expect(putJson({ runner, cwd: "/p", bucket: "b", key: "k", path: "/f" })).rejects.toThrow(
    "cf r2 objects put failed for k: denied",
  );
});

it("finds a workers.dev URL in deploy output", () => {
  expect(findWorkersUrl("done\n  https://up.example.workers.dev\n")).toBe(
    "https://up.example.workers.dev",
  );
  expect(findWorkersUrl("nothing here")).toBeNull();
});
