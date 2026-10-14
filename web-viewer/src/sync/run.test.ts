import { expect, it } from "vitest";
import type { HttpCall } from "./fakes.fake.ts";
import {
  createFakeFs,
  createFakeHttp,
  createFakeRunner,
  type FakeFsHandle,
  fakeFile,
  type RunnerCall,
} from "./fakes.fake.ts";
import { runSync, type SyncServices } from "./run.ts";

const CONFIG = JSON.stringify({
  streamer: "s",
  roots: ["roots/a"],
  youtubeOriginals: "yt",
  bucket: "bucket",
  uploaderWorker: "up",
});

interface Scenario {
  fsHandle: FakeFsHandle;
  services: SyncServices;
  runs: RunnerCall[];
  http: HttpCall[];
  logs: string[];
}

interface ScenarioOptions {
  catalogOriginal: { movie: string; localPath: string } | null;
  remoteMediaSize: number | null;
  buildExitCode: number;
  partStatus: number;
  existing: FakeFsHandle | null;
}

function scenario(options: ScenarioOptions): Scenario {
  const fsHandle =
    options.existing ??
    createFakeFs({
      "/p/sync.config.json": { text: CONFIG },
      "/lib/a-review.mp4": { size: 100, mtimeNs: 5n },
      "/lib/a-review.srt": { size: 1, mtimeNs: 2n, sha: "t" },
      "/lib/a-review.srt.summaries/current.json": { size: 2, mtimeNs: 3n, sha: "s" },
      "/o/orig.mkv": { size: 500, mtimeNs: 6n },
    });
  const catalog = JSON.stringify({
    videos: [
      {
        id: "a-review",
        movie: "a-review.mp4",
        localPath: "/lib/a-review.mp4",
        transcriptPath: "/lib/a-review.srt",
        original: options.catalogOriginal,
      },
    ],
  });
  const logs: string[] = [];
  const { runner, calls: runs } = createFakeRunner((call) => {
    if (call.command === "python3") {
      fsHandle.files.set("/p/data.next/catalog.json", fakeFile({ text: catalog, sha: "c1" }));
      fsHandle.files.set("/p/data.next/a-review.json", fakeFile({ sha: "d1" }));
      return { code: options.buildExitCode, stderr: "build error" };
    }
    if (
      call.command === "cf" &&
      call.args[0] === "r2" &&
      call.args[1] === "objects" &&
      call.args[2] === "list"
    ) {
      const listed =
        call.args[6] === "media/" && options.remoteMediaSize !== null
          ? [{ key: "media/a-review.mp4", size: options.remoteMediaSize }]
          : [];
      return { stdout: JSON.stringify(listed) };
    }
    if (call.command === "ffmpeg") {
      fsHandle.files.set(call.args.at(-1) ?? "", fakeFile({ size: 500 }));
      return {};
    }
    if (call.command === "cf" && call.args[0] === "deploy") {
      return { stdout: "https://up.example.workers.dev" };
    }
    return {};
  });
  const { http, calls: httpCalls } = createFakeHttp((call) => {
    if (call.url.endsWith("/ready")) {
      return { status: 200, body: { ready: true } };
    }
    if (call.url.endsWith("/create")) {
      return { status: 200, body: { uploadId: "u" } };
    }
    if (call.url.includes("/part")) {
      return { status: options.partStatus, body: { partNumber: 1, etag: "e" } };
    }
    return { status: 200, body: {} };
  });
  const services: SyncServices = {
    fs: fsHandle.fs,
    runner,
    http,
    logger: { info: (message) => logs.push(message) },
    sleep: async () => {},
    newToken: () => "token",
    projectDirectory: "/p",
  };
  return { fsHandle, services, runs, http: httpCalls, logs };
}

const base: ScenarioOptions = {
  catalogOriginal: null,
  remoteMediaSize: null,
  buildExitCode: 0,
  partStatus: 200,
  existing: null,
};

const cfSteps = (runs: RunnerCall[]) =>
  runs.map((run) => `${run.command} ${run.args.slice(0, 3).join(" ")}`.trim());

it("plans without changing anything on a dry run", async () => {
  const { services, runs, logs, fsHandle } = scenario(base);
  await runSync({ options: { dryRun: true, dataOnly: false }, services });
  expect(logs).toStrictEqual([
    "changed [media] media/a-review.mp4 - not in the bucket",
    "changed [data] data/a-review.json - not uploaded yet",
    "changed [catalog] catalog.json - videos changed",
    "changed: 3",
  ]);
  expect(cfSteps(runs).some((step) => step.startsWith("cf deploy"))).toBe(false);
  expect(fsHandle.calls.includes("remove /p/data.next")).toBe(true);
});

it("uploads media, then data, then the catalog and removes the uploader", async () => {
  const { services, runs, fsHandle, logs } = scenario(base);
  await runSync({ options: { dryRun: false, dataOnly: false }, services });
  expect(cfSteps(runs)).toStrictEqual([
    "python3 /p/tools/build_catalog.py --streamer s",
    "cf r2 objects list",
    "cf r2 objects list",
    "cf deploy --secrets-file secrets.json",
    "cf workers delete up",
    "cf r2 objects put",
    "cf r2 objects put",
  ]);
  expect(logs.at(-1)).toBe("SYNC DONE");
  expect(fsHandle.calls.at(-1)).toBe("rename /p/data.next /p/data");
  expect(fsHandle.files.has("/p/sync-state.json")).toBe(true);
});

it("has nothing to do on the second run", async () => {
  const first = scenario(base);
  await runSync({ options: { dryRun: false, dataOnly: false }, services: first.services });
  const second = scenario({ ...base, existing: first.fsHandle });
  await runSync({ options: { dryRun: false, dataOnly: false }, services: second.services });
  expect(second.logs).toStrictEqual(["changed: 0", "SYNC DONE"]);
  expect(cfSteps(second.runs).some((step) => step.startsWith("cf deploy"))).toBe(false);
});

it("detects a re-edited summary on the next run", async () => {
  const first = scenario(base);
  await runSync({ options: { dryRun: false, dataOnly: false }, services: first.services });
  first.fsHandle.files.set(
    "/lib/a-review.srt.summaries/current.json",
    fakeFile({ size: 9, mtimeNs: 99n, sha: "s2" }),
  );
  const second = scenario({ ...base, existing: first.fsHandle });
  await runSync({ options: { dryRun: true, dataOnly: false }, services: second.services });
  expect(second.logs).toStrictEqual([
    "changed [data] data/a-review.json - source files changed: summary",
    "changed [catalog] catalog.json - videos changed",
    "changed: 2",
  ]);
});

it("skips media and the catalog with --data-only", async () => {
  const { services, runs } = scenario(base);
  await runSync({ options: { dryRun: false, dataOnly: true }, services });
  expect(cfSteps(runs)).toStrictEqual([
    "python3 /p/tools/build_catalog.py --streamer s",
    "cf r2 objects list",
    "cf r2 objects list",
    "cf r2 objects put",
  ]);
});

it("deletes the uploader even when an upload fails", async () => {
  const { services, runs } = scenario({ ...base, partStatus: 500 });
  await expect(runSync({ options: { dryRun: false, dataOnly: false }, services })).rejects.toThrow(
    "Part 1 of media/a-review.mp4 failed",
  );
  expect(cfSteps(runs).at(-1)).toBe("cf workers delete up");
});

it("rebuilds the proxy for an MKV original before uploading it", async () => {
  const { services, runs } = scenario({
    ...base,
    remoteMediaSize: 100,
    catalogOriginal: { movie: "orig.mp4", localPath: "/o/orig.mkv" },
  });
  await runSync({ options: { dryRun: false, dataOnly: false }, services });
  const steps = cfSteps(runs);
  expect(steps.includes("ffprobe -v error -select_streams")).toBe(true);
  expect(steps.includes("ffmpeg -y -v error")).toBe(true);
});

it("stops when the catalog build fails", async () => {
  const { services } = scenario({ ...base, buildExitCode: 1 });
  await expect(runSync({ options: { dryRun: false, dataOnly: false }, services })).rejects.toThrow(
    "build_catalog failed: build error",
  );
});

it("omits the YouTube originals option when none is configured", async () => {
  const handle = createFakeFs({
    "/p/sync.config.json": {
      text: JSON.stringify({
        streamer: "s",
        roots: ["r"],
        youtubeOriginals: null,
        bucket: "b",
        uploaderWorker: "up",
      }),
    },
    "/lib/a-review.mp4": {},
    "/lib/a-review.srt": {},
  });
  const { services, runs } = scenario({ ...base, existing: handle });
  await runSync({ options: { dryRun: true, dataOnly: false }, services });
  expect(runs[0]?.args.includes("--youtube-originals")).toBe(false);
});

it("adopts a proxied original when the existing proxy matches the bucket", async () => {
  const handle = createFakeFs({
    "/p/sync.config.json": { text: CONFIG },
    "/lib/a-review.mp4": { size: 100, mtimeNs: 5n },
    "/lib/a-review.srt": { size: 1, mtimeNs: 2n, sha: "t" },
    "/o/orig.mkv": { size: 900, mtimeNs: 6n },
    "/p/proxies/orig.mp4": { size: 300 },
  });
  const { services, logs, runs } = scenario({
    ...base,
    existing: handle,
    remoteMediaSize: 100,
    catalogOriginal: { movie: "orig.mp4", localPath: "/o/orig.mkv" },
  });
  await runSync({ options: { dryRun: true, dataOnly: false }, services });
  expect(logs.some((line) => line.includes("media-original"))).toBe(true);
  expect(runs.length).toBeGreaterThan(0);
});
