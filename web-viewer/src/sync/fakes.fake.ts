// Runs with bun. In-memory fakes for the sync ports, used by the tests only.
import type {
  FileStat,
  FileSystem,
  HttpClient,
  HttpResponse,
  ProcessResult,
  ProcessRunner,
} from "./ports.ts";

export interface FakeFile {
  size: number;
  mtimeNs: bigint;
  sha: string;
  text: string;
}

export interface FakeFsHandle {
  fs: FileSystem;
  files: Map<string, FakeFile>;
  calls: string[];
}

export function fakeFile(overrides: Partial<FakeFile>): FakeFile {
  return { size: 10, mtimeNs: 1n, sha: "sha", text: "", ...overrides };
}

function entriesUnder(files: Map<string, FakeFile>, directory: string): string[] {
  const prefix = `${directory}/`;
  const names = [...files.keys()]
    .filter((path) => path.startsWith(prefix))
    .map((path) => path.slice(prefix.length).split("/")[0] ?? "");
  return [...new Set(names)];
}

export function createFakeFs(initial: Record<string, Partial<FakeFile>>): FakeFsHandle {
  const files = new Map<string, FakeFile>(
    Object.entries(initial).map(([path, file]) => [path, fakeFile(file)]),
  );
  const calls: string[] = [];
  const stat = async (path: string): Promise<FileStat | null> => {
    const file = files.get(path);
    if (file !== undefined) {
      return { size: file.size, mtimeNs: file.mtimeNs, isFile: true };
    }
    return entriesUnder(files, path).length > 0 ? { size: 0, mtimeNs: 0n, isFile: false } : null;
  };
  const fs: FileSystem = {
    stat,
    readdir: async (path) => entriesUnder(files, path),
    sha256: async (path) => files.get(path)?.sha ?? "missing",
    readText: async (path) => files.get(path)?.text ?? "",
    mkdir: async (path) => {
      calls.push(`mkdir ${path}`);
    },
    writeText: async ({ path, text, mode }) => {
      calls.push(`write ${path} ${mode.toString(8)}`);
      files.set(path, fakeFile({ text, size: text.length }));
    },
    copy: async ({ from, to }) => {
      calls.push(`copy ${from} ${to}`);
      files.set(to, fakeFile({}));
    },
    link: async ({ target, path }) => {
      calls.push(`link ${target} ${path}`);
    },
    removeTree: async (path) => {
      calls.push(`remove ${path}`);
      for (const key of [...files.keys()]) {
        if (key === path || key.startsWith(`${path}/`)) {
          files.delete(key);
        }
      }
    },
    rename: async ({ from, to }) => {
      calls.push(`rename ${from} ${to}`);
      const moved = [...files.entries()].filter(
        ([key]) => key === from || key.startsWith(`${from}/`),
      );
      for (const [key, file] of moved) {
        files.delete(key);
        files.set(`${to}${key.slice(from.length)}`, file);
      }
    },
    readChunk: async ({ length }) => new Uint8Array(Math.min(length, 4)),
  };
  return { fs, files, calls };
}

export interface RunnerCall {
  command: string;
  args: string[];
  cwd: string;
}

export interface FakeRunnerHandle {
  runner: ProcessRunner;
  calls: RunnerCall[];
}

export function createFakeRunner(
  respond: (call: RunnerCall) => Partial<ProcessResult>,
): FakeRunnerHandle {
  const calls: RunnerCall[] = [];
  const runner: ProcessRunner = {
    run: async (call) => {
      calls.push(call);
      return { code: 0, stdout: "", stderr: "", ...respond(call) };
    },
  };
  return { runner, calls };
}

export interface HttpCall {
  method: string;
  url: string;
  headers: Record<string, string>;
  bodyBytes: number | null;
}

export interface FakeHttpHandle {
  http: HttpClient;
  calls: HttpCall[];
}

export type HttpReply = Omit<HttpResponse, "setCookie"> & { setCookie?: string | null };

export function createFakeHttp(respond: (call: HttpCall) => HttpReply | Error): FakeHttpHandle {
  const calls: HttpCall[] = [];
  const http: HttpClient = {
    request: async ({ method, url, headers, body }) => {
      const bodyBytes = body === null ? null : body.length;
      const call: HttpCall = { method, url, headers, bodyBytes };
      calls.push(call);
      const result = respond(call);
      if (result instanceof Error) {
        throw result;
      }
      return { setCookie: null, ...result };
    },
  };
  return { http, calls };
}
