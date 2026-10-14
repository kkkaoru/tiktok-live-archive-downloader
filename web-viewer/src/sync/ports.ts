// Runs with bun. Ports for everything that touches the outside world, so the sync logic can
// be tested with fakes and the real adapters stay tiny.
export interface FileStat {
  size: number;
  mtimeNs: bigint;
  isFile: boolean;
}

export interface FileSystem {
  stat(path: string): Promise<FileStat | null>;
  readdir(path: string): Promise<string[]>;
  sha256(path: string): Promise<string>;
  readText(path: string): Promise<string>;
  mkdir(path: string): Promise<void>;
  writeText(args: { path: string; text: string; mode: number }): Promise<void>;
  copy(args: { from: string; to: string }): Promise<void>;
  link(args: { target: string; path: string }): Promise<void>;
  removeTree(path: string): Promise<void>;
  rename(args: { from: string; to: string }): Promise<void>;
  readChunk(args: { path: string; start: number; length: number }): Promise<Uint8Array>;
}

export interface ProcessResult {
  code: number;
  stdout: string;
  stderr: string;
}

export interface ProcessRunner {
  run(args: { command: string; args: string[]; cwd: string }): Promise<ProcessResult>;
}

export interface Logger {
  info(message: string): void;
}

export interface HttpResponse {
  status: number;
  body: unknown;
  setCookie: string | null;
}

export interface HttpClient {
  request(args: {
    method: string;
    url: string;
    headers: Record<string, string>;
    body: Uint8Array | string | null;
  }): Promise<HttpResponse>;
}
