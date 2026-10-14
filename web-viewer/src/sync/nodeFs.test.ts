import { EventEmitter } from "node:events";
import { beforeEach, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  copyFile: vi.fn(),
  mkdir: vi.fn(),
  open: vi.fn(),
  readdir: vi.fn(),
  readFile: vi.fn(),
  rename: vi.fn(),
  rm: vi.fn(),
  stat: vi.fn(),
  symlink: vi.fn(),
  writeFile: vi.fn(),
  createReadStream: vi.fn(),
}));

vi.mock("node:fs/promises", () => ({
  copyFile: mocks.copyFile,
  mkdir: mocks.mkdir,
  open: mocks.open,
  readdir: mocks.readdir,
  readFile: mocks.readFile,
  rename: mocks.rename,
  rm: mocks.rm,
  stat: mocks.stat,
  symlink: mocks.symlink,
  writeFile: mocks.writeFile,
}));
vi.mock("node:fs", () => ({ createReadStream: mocks.createReadStream }));

import { createNodeFs } from "./nodeFs.ts";

beforeEach(() => {
  vi.resetAllMocks();
});

it("stats a file with nanosecond mtime", async () => {
  mocks.stat.mockResolvedValue({ size: 5n, mtimeNs: 9n, isFile: () => true });
  expect(await createNodeFs().stat("/f")).toStrictEqual({ size: 5, mtimeNs: 9n, isFile: true });
});

it("returns null for a missing file", async () => {
  mocks.stat.mockRejectedValue(Object.assign(new Error("gone"), { code: "ENOENT" }));
  expect(await createNodeFs().stat("/f")).toBeNull();
});

it("rethrows other stat errors", async () => {
  mocks.stat.mockRejectedValue(new Error("denied"));
  await expect(createNodeFs().stat("/f")).rejects.toThrow("denied");
});

it("delegates directory and text reads", async () => {
  mocks.readdir.mockResolvedValue(["a", "b"]);
  mocks.readFile.mockResolvedValue("text");
  const fs = createNodeFs();
  expect(await fs.readdir("/d")).toStrictEqual(["a", "b"]);
  expect(await fs.readText("/f")).toBe("text");
});

it("writes text with a private parent directory and the requested mode", async () => {
  await createNodeFs().writeText({ path: "/d/f.txt", text: "hi", mode: 0o600 });
  expect(mocks.mkdir).toHaveBeenCalledWith("/d", { recursive: true, mode: 0o700 });
  expect(mocks.writeFile).toHaveBeenCalledWith("/d/f.txt", "hi", { mode: 0o600 });
});

it("creates directories, copies, renames and removes trees", async () => {
  const fs = createNodeFs();
  await fs.mkdir("/d");
  await fs.copy({ from: "/a", to: "/b" });
  await fs.rename({ from: "/b", to: "/c" });
  await fs.removeTree("/c");
  expect(mocks.mkdir).toHaveBeenCalledWith("/d", { recursive: true, mode: 0o700 });
  expect(mocks.copyFile).toHaveBeenCalledWith("/a", "/b");
  expect(mocks.rename).toHaveBeenCalledWith("/b", "/c");
  expect(mocks.rm).toHaveBeenCalledWith("/c", { recursive: true, force: true });
});

it("replaces an existing link", async () => {
  await createNodeFs().link({ target: "/t", path: "/l" });
  expect(mocks.rm).toHaveBeenCalledWith("/l", { recursive: true, force: true });
  expect(mocks.symlink).toHaveBeenCalledWith("/t", "/l");
});

it("hashes a file by streaming it", async () => {
  const stream = new EventEmitter();
  mocks.createReadStream.mockReturnValue(stream);
  const pending = createNodeFs().sha256("/f");
  stream.emit("data", Buffer.from("abc"));
  stream.emit("end");
  expect(await pending).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
});

it("rejects when the stream fails", async () => {
  const stream = new EventEmitter();
  mocks.createReadStream.mockReturnValue(stream);
  const pending = createNodeFs().sha256("/f");
  stream.emit("error", new Error("io"));
  await expect(pending).rejects.toThrow("io");
});

it("reads a chunk and closes the handle", async () => {
  const handle = {
    read: vi.fn().mockResolvedValue({ bytesRead: 2 }),
    close: vi.fn().mockResolvedValue(undefined),
  };
  mocks.open.mockResolvedValue(handle);
  const chunk = await createNodeFs().readChunk({ path: "/f", start: 10, length: 4 });
  expect(chunk.length).toBe(2);
  expect(handle.read).toHaveBeenCalledWith(expect.any(Uint8Array), 0, 4, 10);
  expect(handle.close).toHaveBeenCalledTimes(1);
});

it("closes the handle when a read fails", async () => {
  const handle = {
    read: vi.fn().mockRejectedValue(new Error("read failed")),
    close: vi.fn().mockResolvedValue(undefined),
  };
  mocks.open.mockResolvedValue(handle);
  await expect(createNodeFs().readChunk({ path: "/f", start: 0, length: 1 })).rejects.toThrow(
    "read failed",
  );
  expect(handle.close).toHaveBeenCalledTimes(1);
});
