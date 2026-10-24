import { expect, it } from "vitest";
import type { HttpCall } from "./fakes.fake.ts";
import {
  createFakeFs,
  createFakeHttp,
  createFakeRunner,
  type FakeFsHandle,
  fakeFile,
  type RunnerCall,
  zerosEtag,
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
  termsExitCode: number;
  partStatus: number;
  existing: FakeFsHandle | null;
  kept: boolean;
  second: boolean;
  localPath: string | null;
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
  const second = {
    id: "b-review",
    movie: "b-review.mp4",
    localPath: "/lib/b-review.mp4",
    transcriptPath: "/lib/b-review.srt",
    original: null,
    kept: false,
  };
  const catalog = JSON.stringify({
    videos: [
      {
        id: "a-review",
        movie: "a-review.mp4",
        localPath: options.localPath,
        transcriptPath: "/lib/a-review.srt",
        original: options.catalogOriginal,
        kept: options.kept,
      },
      ...(options.second ? [second] : []),
    ],
  });
  const logs: string[] = [];
  const { runner, calls: runs } = createFakeRunner((call) => {
    if (call.command === "python3") {
      fsHandle.files.set("/p/data.next/catalog.json", fakeFile({ text: catalog, sha: "c1" }));
      fsHandle.files.set("/p/data.next/a-review.json", fakeFile({ sha: "d1" }));
      fsHandle.files.set("/p/data.next/b-review.json", fakeFile({ sha: "d2" }));
      return { code: options.buildExitCode, stderr: "build error" };
    }
    if (call.command === "uv") {
      return { code: options.termsExitCode, stderr: "terms error" };
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
      return {
        status: options.partStatus,
        body: { partNumber: 1, etag: zerosEtag(call.bodyBytes) },
      };
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
  termsExitCode: 0,
  partStatus: 200,
  existing: null,
  kept: false,
  second: false,
  localPath: "/lib/a-review.mp4",
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
    "uv run --no-sync python",
    "cf r2 objects list",
    "cf r2 objects list",
    "cf deploy --secrets-file secrets.json",
    "cf workers delete up",
    "cf r2 objects put",
    "cf r2 objects put",
  ]);
  expect(logs.at(-1)).toBe("SYNC DONE");
  expect(fsHandle.calls.at(-1)).toBe("rename /p/data.next /p/data");
  expect(JSON.parse(fsHandle.files.get("/p/sync-state.json")?.text ?? "{}").uploads).toStrictEqual({
    "media/a-review.mp4": {
      size: 100,
      sha256: "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
      etag: "bb4512f7b430c4ff12d53c6796475fbe-1",
    },
  });
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
    "uv run --no-sync python",
    "cf r2 objects list",
    "cf r2 objects list",
    "cf r2 objects put",
  ]);
});

it("plans only the originals on a dry run with --originals-only", async () => {
  const first = scenario(base);
  first.fsHandle.files.set("/o/orig.mp4", fakeFile({ size: 500, mtimeNs: 6n }));
  const { services, logs, runs } = scenario({
    ...base,
    existing: first.fsHandle,
    catalogOriginal: { movie: "orig.mp4", localPath: "/o/orig.mp4" },
  });
  await runSync({ options: { dryRun: true, dataOnly: false, originalsOnly: true }, services });
  expect(logs).toStrictEqual([
    "changed [media] media-original/orig.mp4 - not in the bucket",
    "changed: 1",
  ]);
  expect(cfSteps(runs).some((step) => step.startsWith("cf deploy"))).toBe(false);
});

it("uploads only the original and publishes nothing else with --originals-only", async () => {
  const first = scenario(base);
  first.fsHandle.files.set("/o/orig.mp4", fakeFile({ size: 500, mtimeNs: 6n }));
  const { services, runs, logs, http, fsHandle } = scenario({
    ...base,
    existing: first.fsHandle,
    catalogOriginal: { movie: "orig.mp4", localPath: "/o/orig.mp4" },
  });
  await runSync({ options: { dryRun: false, dataOnly: false, originalsOnly: true }, services });
  expect(logs).toStrictEqual([
    "changed [media] media-original/orig.mp4 - not in the bucket",
    "changed: 1",
    "uploading media-original/orig.mp4 (500 B)",
    "uploaded media-original/orig.mp4 in 1 part(s)",
    "SYNC DONE",
  ]);
  expect(http.filter((call) => call.url.endsWith("/create")).length).toBe(1);
  expect(cfSteps(runs).includes("cf r2 objects put")).toBe(false);
  expect(fsHandle.files.has("/p/data/catalog.json")).toBe(false);
  const state = JSON.parse(await fsHandle.fs.readText("/p/sync-state.json"));
  expect(Object.keys(state.sources["a-review"])).toStrictEqual(["original"]);
  expect(state.data).toStrictEqual({});

  const full = scenario({
    ...base,
    existing: fsHandle,
    remoteMediaSize: 100,
    catalogOriginal: { movie: "orig.mp4", localPath: "/o/orig.mp4" },
  });
  await runSync({ options: { dryRun: true, dataOnly: false }, services: full.services });
  expect(full.logs.some((line) => line.includes("media-original/orig.mp4"))).toBe(false);
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

it("stops when the part-of-speech hints fail", async () => {
  const { services, runs } = scenario({ ...base, termsExitCode: 1 });
  await expect(runSync({ options: { dryRun: true, dataOnly: false }, services })).rejects.toThrow(
    "clip_terms failed: terms error",
  );
  expect(runs[1]).toStrictEqual({
    command: "uv",
    args: ["run", "--no-sync", "python", "-m", "replay.clip_terms", "/p/data.next"],
    cwd: "/",
  });
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
  expect(runs[0]?.args.includes("--raw")).toBe(false);
});

it("passes the raw downloads directory to the catalog build", async () => {
  const handle = createFakeFs({
    "/p/sync.config.json": {
      text: JSON.stringify({
        streamer: "s",
        roots: ["r"],
        youtubeOriginals: null,
        raw: "../downloads",
        bucket: "b",
        uploaderWorker: "up",
      }),
    },
    "/lib/a-review.mp4": {},
    "/lib/a-review.srt": {},
  });
  const { services, runs } = scenario({ ...base, existing: handle });
  await runSync({ options: { dryRun: true, dataOnly: false }, services });
  expect(runs[0]?.args.slice(-4)).toStrictEqual(["--raw", "/downloads", "--keep", "/p/data"]);
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

it("carries a kept published video without reading its retired files", async () => {
  const handle = createFakeFs({
    "/p/sync.config.json": { text: CONFIG },
    "/p/sync-state.json": {
      text: JSON.stringify({
        sources: { "a-review": { movie: { size: 100, mtimeNs: "5", sha256: null } } },
        data: { "data/a-review.json": "d1" },
        catalogSha256: "c1",
      }),
    },
  });
  const { services, logs } = scenario({ ...base, existing: handle, kept: true });
  await runSync({ options: { dryRun: true, dataOnly: false }, services });
  expect(logs).toStrictEqual(["changed: 0"]);
});

it("publishes an offloaded video's new data without sending its movie again", async () => {
  const handle = createFakeFs({
    "/p/sync.config.json": { text: CONFIG },
    "/lib/a-review.srt": { size: 1, mtimeNs: 2n, sha: "t" },
    "/lib/a-review.srt.summaries/current.json": { size: 3, mtimeNs: 9n, sha: "new summary" },
    "/p/sync-state.json": {
      text: JSON.stringify({
        sources: {
          "a-review": {
            movie: { size: 100, mtimeNs: "5", sha256: null },
            transcript: { size: 1, mtimeNs: "2", sha256: "t" },
            summary: { size: 2, mtimeNs: "3", sha256: "s" },
          },
        },
        data: { "data/a-review.json": "d0" },
        catalogSha256: "c1",
      }),
    },
  });
  const { services, runs, fsHandle } = scenario({ ...base, existing: handle, localPath: null });
  await runSync({ options: { dryRun: false, dataOnly: false }, services });
  expect(cfSteps(runs).some((step) => step.startsWith("cf deploy"))).toBe(false);
  const state = JSON.parse(fsHandle.files.get("/p/sync-state.json")?.text ?? "{}");
  expect(state.sources["a-review"]).toStrictEqual({
    movie: { size: 100, mtimeNs: "5", sha256: null },
    transcript: { size: 1, mtimeNs: "2", sha256: "t" },
    summary: { size: 3, mtimeNs: "9", sha256: "new summary" },
  });
  expect(state.data).toStrictEqual({ "data/a-review.json": "d1" });
});

it("uploads only the named video's movie with --only and defers the catalog", async () => {
  const handle = createFakeFs({
    "/p/sync.config.json": { text: CONFIG },
    "/lib/a-review.mp4": { size: 100, mtimeNs: 5n },
    "/lib/a-review.srt": { size: 1, mtimeNs: 2n, sha: "t" },
    "/lib/b-review.mp4": { size: 200, mtimeNs: 7n },
    "/lib/b-review.srt": { size: 1, mtimeNs: 8n, sha: "u" },
  });
  const first = scenario({ ...base, existing: handle, second: true });
  await runSync({
    options: { dryRun: false, dataOnly: false, only: "a-review" },
    services: first.services,
  });
  expect(first.logs).toStrictEqual([
    "changed [media] media/a-review.mp4 - not in the bucket",
    "changed [data] data/a-review.json - not uploaded yet",
    "changed [data] data/b-review.json - not uploaded yet",
    "changed [catalog] catalog.json - videos changed",
    "changed: 4",
    "uploading media/a-review.mp4 (100 B)",
    "uploaded media/a-review.mp4 in 1 part(s)",
    "catalog deferred: 1 movie(s) not uploaded yet",
    "uploaded data/a-review.json",
    "uploaded data/b-review.json",
    "SYNC DONE",
  ]);

  const full = scenario({ ...base, existing: first.fsHandle, second: true });
  await runSync({ options: { dryRun: true, dataOnly: false }, services: full.services });
  // B's movie is still pending; data is recorded as published only with the catalog, as for
  // --data-only, so the small data files are sent again with it.
  expect(full.logs).toStrictEqual([
    "changed [media] media/b-review.mp4 - not in the bucket",
    "changed [data] data/a-review.json - not uploaded yet",
    "changed [data] data/b-review.json - not uploaded yet",
    "changed [catalog] catalog.json - videos changed",
    "changed: 4",
  ]);
});

it("refuses --only for a video the catalog does not list", async () => {
  const { services, runs } = scenario(base);
  await expect(
    runSync({ options: { dryRun: true, dataOnly: false, only: "z-review" }, services }),
  ).rejects.toThrow("--only z-review is not in the catalog");
  expect(cfSteps(runs)).toStrictEqual([
    "python3 /p/tools/build_catalog.py --streamer s",
    "uv run --no-sync python",
  ]);
});

it("logs one timing line per phase when given a clock", async () => {
  const { services, logs } = scenario(base);
  let now = 0;
  const clock = () => {
    now += 500;
    return now;
  };
  await runSync({ options: { dryRun: false, dataOnly: false }, services: { ...services, clock } });
  expect(logs.filter((line) => line.startsWith("PHASE"))).toStrictEqual([
    "PHASE build 0.5 s",
    "PHASE measure 0.5 s",
    "PHASE list 0.5 s",
    "PHASE deploy 0.5 s",
    "PHASE transfer 0.5 s 100 B 0.0 MB/s",
    "PHASE delete 0.5 s",
    "PHASE put 0.5 s 2 object(s)",
  ]);
  expect(logs.some((line) => line.includes("token"))).toBe(false);
});
