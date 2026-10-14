import { EventEmitter } from "node:events";
import { beforeEach, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ spawn: vi.fn() }));
vi.mock("node:child_process", () => ({ spawn: mocks.spawn }));

import { createNodeRunner } from "./nodeProcess.ts";

interface FakeChild extends EventEmitter {
  stdout: EventEmitter;
  stderr: EventEmitter;
}

function child(): FakeChild {
  return Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(),
    stderr: new EventEmitter(),
  });
}

beforeEach(() => {
  vi.resetAllMocks();
});

it("captures output and the exit code without a shell", async () => {
  const fake = child();
  mocks.spawn.mockReturnValue(fake);
  const pending = createNodeRunner({ A: "1" }).run({ command: "cf", args: ["x"], cwd: "/p" });
  fake.stdout.emit("data", Buffer.from("out"));
  fake.stderr.emit("data", Buffer.from("err"));
  fake.emit("close", 3);
  expect(await pending).toStrictEqual({ code: 3, stdout: "out", stderr: "err" });
  expect(mocks.spawn).toHaveBeenCalledWith("cf", ["x"], {
    cwd: "/p",
    env: { A: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
});

it("reports a killed process as a failure", async () => {
  const fake = child();
  mocks.spawn.mockReturnValue(fake);
  const pending = createNodeRunner({}).run({ command: "cf", args: [], cwd: "/p" });
  fake.emit("close", null);
  expect((await pending).code).toBe(1);
});

it("rejects when the command cannot start", async () => {
  const fake = child();
  mocks.spawn.mockReturnValue(fake);
  const pending = createNodeRunner({}).run({ command: "nope", args: [], cwd: "/p" });
  fake.emit("error", new Error("ENOENT"));
  await expect(pending).rejects.toThrow("ENOENT");
});
