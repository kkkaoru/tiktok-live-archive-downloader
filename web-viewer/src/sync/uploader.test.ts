import { expect, it } from "vitest";
import { createFakeFs, createFakeHttp, createFakeRunner, zerosEtag } from "./fakes.fake.ts";
import type { HttpClient } from "./ports.ts";
import {
  CHUNK_BYTES,
  deleteUploader,
  deployUploader,
  ensureProxy,
  PARALLEL_PARTS,
  PART_ATTEMPTS,
  READY_ATTEMPTS,
  uploaderConfig,
  uploadFile,
  waitUntilReady,
} from "./uploader.ts";

// The fake file system reads 4 zero bytes per part.
const ZEROS_MD5 = "f1d3ff8443297732862df21dc4e57262";

const project = {
  directory: "/p/uploader",
  projectDirectory: "/p",
  name: "uploader-worker",
  bucket: "bucket",
};

function sleeper(): { sleep: (ms: number) => Promise<void>; slept: number[] } {
  const slept: number[] = [];
  return {
    sleep: async (ms) => {
      slept.push(ms);
    },
    slept,
  };
}

it("renders the uploader Worker config", () => {
  const config = uploaderConfig({ name: "w", bucket: "b" });
  expect(config.includes('name: "w"')).toBe(true);
  expect(config.includes('bindings.r2({ name: "b" })')).toBe(true);
});

it("deploys the uploader with a private secrets file and returns its URL", async () => {
  const { fs, calls } = createFakeFs({});
  const { runner, calls: runs } = createFakeRunner(() => ({
    stdout: "Deployed\n  https://uploader-worker.example.workers.dev\n",
  }));
  const url = await deployUploader({ services: { fs, runner }, project, token: "tok" });
  expect(url).toBe("https://uploader-worker.example.workers.dev");
  expect(calls).toStrictEqual([
    "mkdir /p/uploader",
    "copy /p/src/upload-worker.js /p/uploader/worker.js",
    "write /p/uploader/cloudflare.config.ts 644",
    "write /p/uploader/package.json 644",
    "write /p/uploader/secrets.json 600",
    "link /p/node_modules /p/uploader/node_modules",
  ]);
  expect(runs).toStrictEqual([
    { command: "cf", args: ["deploy", "--secrets-file", "secrets.json"], cwd: "/p/uploader" },
  ]);
});

it("fails when the deploy fails", async () => {
  const { fs } = createFakeFs({});
  const { runner } = createFakeRunner(() => ({ code: 1, stderr: "no wrangler" }));
  await expect(deployUploader({ services: { fs, runner }, project, token: "t" })).rejects.toThrow(
    "Uploader deploy failed: no wrangler",
  );
});

it("fails when the deploy prints no URL", async () => {
  const { fs } = createFakeFs({});
  const { runner } = createFakeRunner(() => ({ stdout: "ok" }));
  await expect(deployUploader({ services: { fs, runner }, project, token: "t" })).rejects.toThrow(
    "Uploader deploy failed",
  );
});

it("waits until the Worker itself answers /ready", async () => {
  const responses = [
    new Error("network"),
    { status: 404, body: null },
    { status: 200, body: { ready: false } },
    { status: 200, body: { ready: true } },
  ];
  const { http, calls } = createFakeHttp(
    () => responses.shift() ?? { status: 200, body: { ready: true } },
  );
  const { sleep, slept } = sleeper();
  await waitUntilReady({ http, sleep, url: "https://u", token: "tok" });
  expect(calls.length).toBe(4);
  expect(slept.length).toBe(3);
  expect(calls[0]?.url).toBe("https://u/ready");
  expect(calls[0]?.headers["X-Upload-Token"]).toBe("tok");
});

it("gives up when the Worker never becomes ready", async () => {
  const { http, calls } = createFakeHttp(() => ({ status: 404, body: null }));
  const { sleep } = sleeper();
  await expect(waitUntilReady({ http, sleep, url: "https://u", token: "t" })).rejects.toThrow(
    "Uploader Worker did not become ready",
  );
  expect(calls.length).toBe(READY_ATTEMPTS);
});

it("uploads a small file as one part", async () => {
  const { fs } = createFakeFs({});
  const { http, calls } = createFakeHttp((call) =>
    call.url.endsWith("/create")
      ? { status: 200, body: { uploadId: "u1" } }
      : call.url.includes("/part")
        ? { status: 200, body: { partNumber: 1, etag: ZEROS_MD5 } }
        : { status: 200, body: { ok: true } },
  );
  const { sleep } = sleeper();
  const parts = await uploadFile({
    services: { fs, http },
    sleep,
    url: "https://u",
    token: "t",
    key: "media/a b.mp4",
    path: "/f",
    size: 10,
  });
  expect(parts).toStrictEqual({
    parts: 1,
    sha256: "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119",
    etag: "bb4512f7b430c4ff12d53c6796475fbe-1",
  });
  expect(calls.map((call) => `${call.method} ${call.url.replace("https://u", "")}`)).toStrictEqual([
    "POST /create",
    "PUT /part?key=media%2Fa%20b.mp4&uploadId=u1&n=1",
    "POST /complete",
  ]);
});

it("splits a large file into several parts", async () => {
  const { fs } = createFakeFs({});
  const { http, calls } = createFakeHttp((call) =>
    call.url.endsWith("/create")
      ? { status: 200, body: { uploadId: "u" } }
      : call.url.includes("/part")
        ? {
            status: 200,
            body: { partNumber: Number(call.url.split("n=")[1]), etag: zerosEtag(call.bodyBytes) },
          }
        : { status: 200, body: {} },
  );
  const { sleep } = sleeper();
  const parts = await uploadFile({
    services: { fs, http },
    sleep,
    url: "https://u",
    token: "t",
    key: "k",
    path: "/f",
    size: CHUNK_BYTES + 1,
  });
  expect(parts).toStrictEqual({
    parts: 2,
    sha256: "8855508aade16ec573d21e6a485dfd0a7624085c1a14b5ecdd6485de0c6839a4",
    etag: "c97bba22a3c3989fbe2132f63bcfe1a5-2",
  });
  expect(calls.length).toBe(4);
});

it("retries a failing part", async () => {
  const { fs } = createFakeFs({});
  const statuses = [500, 200];
  const { http } = createFakeHttp((call) =>
    call.url.endsWith("/create")
      ? { status: 200, body: { uploadId: "u" } }
      : call.url.includes("/part")
        ? { status: statuses.shift() ?? 200, body: { partNumber: 1, etag: ZEROS_MD5 } }
        : { status: 200, body: {} },
  );
  const { sleep, slept } = sleeper();
  await uploadFile({
    services: { fs, http },
    sleep,
    url: "https://u",
    token: "t",
    key: "k",
    path: "/f",
    size: 5,
  });
  expect(slept.length).toBe(1);
});

it("retries a part whose ETag is not its MD5, then gives up", async () => {
  const { fs } = createFakeFs({});
  const etags = ['"0123"', `"${ZEROS_MD5}"`];
  const { http } = createFakeHttp((call) =>
    call.url.endsWith("/create")
      ? { status: 200, body: { uploadId: "u" } }
      : call.url.includes("/part")
        ? { status: 200, body: { partNumber: 1, etag: etags.shift() ?? "bad" } }
        : { status: 200, body: {} },
  );
  const { sleep, slept } = sleeper();
  const request = { services: { fs, http }, sleep, url: "https://u", token: "t", key: "k" };
  const sent = await uploadFile({ ...request, path: "/f", size: 5 });
  expect(sent.etag).toBe("bb4512f7b430c4ff12d53c6796475fbe-1");
  expect(slept).toStrictEqual([3000]);
  await expect(uploadFile({ ...request, path: "/f", size: 5 })).rejects.toThrow(
    "Part 1 of k failed or did not match its MD5",
  );
});

it("aborts without a hash when a part cannot be read", async () => {
  const { fs } = createFakeFs({});
  const { http, calls } = createFakeHttp((call) =>
    call.url.endsWith("/create")
      ? { status: 200, body: { uploadId: "u" } }
      : { status: 200, body: {} },
  );
  const failing = { ...fs, readChunk: async () => Promise.reject(new Error("disk")) };
  const { sleep } = sleeper();
  await expect(
    uploadFile({
      services: { fs: failing, http },
      sleep,
      url: "https://u",
      token: "t",
      key: "k",
      path: "/f",
      size: CHUNK_BYTES + 1,
    }),
  ).rejects.toThrow("disk");
  expect(calls.map((call) => call.url)).toStrictEqual(["https://u/create", "https://u/abort"]);
});

it("doubles the delay between attempts of a part", async () => {
  const { fs } = createFakeFs({});
  const { http } = createFakeHttp((call) =>
    call.url.endsWith("/create")
      ? { status: 200, body: { uploadId: "u" } }
      : call.url.includes("/part")
        ? new Error("network")
        : { status: 200, body: {} },
  );
  const { sleep, slept } = sleeper();
  await expect(
    uploadFile({
      services: { fs, http },
      sleep,
      url: "https://u",
      token: "t",
      key: "k",
      path: "/f",
      size: 5,
    }),
  ).rejects.toThrow("Part 1 of k failed");
  expect(slept).toStrictEqual([3000, 6000, 12000, 24000, 48000, 96000]);
  expect(slept.length).toBe(PART_ATTEMPTS - 1);
});

it("starts the next part as soon as any part finishes, never more than the pool at once", async () => {
  const { fs } = createFakeFs({});
  const count = PARALLEL_PARTS + 2;
  const started: number[] = [];
  const inFlight = { now: 0, most: 0 };
  const release: { first: () => void } = { first: () => undefined };
  const firstDone = new Promise<void>((resolve) => {
    release.first = resolve;
  });
  const http: HttpClient = {
    request: async ({ url, body }) => {
      if (!url.includes("/part")) {
        return { status: 200, body: { uploadId: "u", ok: true }, setCookie: null };
      }
      const etag = zerosEtag(body?.length ?? 0);
      const partNumber = Number(url.split("n=")[1]);
      started.push(partNumber);
      inFlight.now += 1;
      inFlight.most = Math.max(inFlight.most, inFlight.now);
      if (partNumber === count) {
        release.first(); // the last part started while part 1 is still in flight
      }
      await (partNumber === 1 ? firstDone : new Promise((resolve) => setTimeout(resolve, 0)));
      inFlight.now -= 1;
      return { status: 200, body: { partNumber, etag }, setCookie: null };
    },
  };
  const { sleep } = sleeper();
  const parts = await uploadFile({
    services: { fs, http },
    sleep,
    url: "https://u",
    token: "t",
    key: "k",
    path: "/f",
    size: CHUNK_BYTES * (count - 1) + 1,
  });
  expect(parts.parts).toBe(count);
  expect(started).toStrictEqual(Array.from({ length: count }, (_, index) => index + 1));
  expect(inFlight.most).toBe(PARALLEL_PARTS);
});

it("starts no further part once the upload is aborted", async () => {
  const { fs } = createFakeFs({});
  const urls: string[] = [];
  const release: { aborted: () => void } = { aborted: () => undefined };
  const aborted = new Promise<void>((resolve) => {
    release.aborted = resolve;
  });
  const http: HttpClient = {
    request: async ({ url }) => {
      urls.push(url);
      if (url.endsWith("/abort")) {
        release.aborted();
      }
      if (!url.includes("/part")) {
        return { status: 200, body: { uploadId: "u", ok: true }, setCookie: null };
      }
      if (url.endsWith("n=1")) {
        throw new Error("network");
      }
      await aborted; // the other parts are still in flight when the upload is aborted
      return { status: 200, body: { partNumber: 2, etag: ZEROS_MD5 }, setCookie: null };
    },
  };
  const { sleep } = sleeper();
  await expect(
    uploadFile({
      services: { fs, http },
      sleep,
      url: "https://u",
      token: "t",
      key: "k",
      path: "/f",
      size: CHUNK_BYTES * 40,
    }),
  ).rejects.toThrow("Part 1 of k failed");
  await new Promise((resolve) => setTimeout(resolve, 0));
  const parts = urls.filter((url) => url.includes("/part"));
  expect(parts.length).toBe(PART_ATTEMPTS + PARALLEL_PARTS - 1);
});

it("aborts the multipart upload when a part keeps failing", async () => {
  const { fs } = createFakeFs({});
  const { http, calls } = createFakeHttp((call) =>
    call.url.endsWith("/create")
      ? { status: 200, body: { uploadId: "u" } }
      : call.url.includes("/part")
        ? new Error("network")
        : { status: 200, body: {} },
  );
  const { sleep } = sleeper();
  await expect(
    uploadFile({
      services: { fs, http },
      sleep,
      url: "https://u",
      token: "t",
      key: "k",
      path: "/f",
      size: 5,
    }),
  ).rejects.toThrow("Part 1 of k failed");
  expect(calls.at(-1)?.url).toBe("https://u/abort");
});

it("fails when the upload cannot be created", async () => {
  const { fs } = createFakeFs({});
  const { http } = createFakeHttp(() => ({ status: 500, body: null }));
  const { sleep } = sleeper();
  await expect(
    uploadFile({
      services: { fs, http },
      sleep,
      url: "https://u",
      token: "t",
      key: "k",
      path: "/f",
      size: 5,
    }),
  ).rejects.toThrow("Uploader request failed with HTTP 500");
});

it("keeps an existing proxy", async () => {
  const { fs } = createFakeFs({ "/p/proxies/a.mp4": { size: 5 } });
  const { runner, calls } = createFakeRunner(() => ({}));
  await ensureProxy({
    services: { fs, runner },
    source: "/o/a.mkv",
    destination: "/p/proxies/a.mp4",
    cwd: "/p",
  });
  expect(calls).toStrictEqual([]);
});

it("copies the video stream of an H.264 original", async () => {
  const { fs, calls: fsCalls } = createFakeFs({});
  const { runner, calls } = createFakeRunner((call) =>
    call.command === "ffprobe" ? { stdout: "h264\n" } : {},
  );
  await ensureProxy({
    services: { fs, runner },
    source: "/o/a.mkv",
    destination: "/p/proxies/a.mp4",
    cwd: "/p",
  });
  expect(calls[1]?.args.slice(5, 7)).toStrictEqual(["-c:v", "copy"]);
  expect(fsCalls.at(-1)).toBe("rename /p/proxies/a.mp4.part.mp4 /p/proxies/a.mp4");
});

it("re-encodes other codecs with VideoToolbox", async () => {
  const { fs } = createFakeFs({});
  const { runner, calls } = createFakeRunner((call) =>
    call.command === "ffprobe" ? { stdout: "vp9" } : {},
  );
  await ensureProxy({
    services: { fs, runner },
    source: "/o/a.mkv",
    destination: "/p/proxies/a.mp4",
    cwd: "/p",
  });
  expect(calls[1]?.args.slice(5, 9)).toStrictEqual(["-c:v", "h264_videotoolbox", "-b:v", "8M"]);
});

it("fails when ffmpeg fails", async () => {
  const { fs } = createFakeFs({});
  const { runner } = createFakeRunner((call) =>
    call.command === "ffmpeg" ? { code: 1, stderr: "bad" } : {},
  );
  await expect(
    ensureProxy({
      services: { fs, runner },
      source: "/o/a.mkv",
      destination: "/p/proxies/a.mp4",
      cwd: "/p",
    }),
  ).rejects.toThrow("ffmpeg proxy failed: bad");
});

it("deletes the uploader Worker and its directory", async () => {
  const { fs, calls: fsCalls } = createFakeFs({});
  const { runner, calls } = createFakeRunner(() => ({}));
  await deleteUploader({ services: { fs, runner }, project });
  expect(calls).toStrictEqual([
    { command: "cf", args: ["workers", "delete", "uploader-worker", "--force"], cwd: "/p" },
  ]);
  expect(fsCalls).toStrictEqual(["remove /p/uploader"]);
});
