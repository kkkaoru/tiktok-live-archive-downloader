// Runs with bun (vitest). The Mac's clip points: routing to the local server, the viewer's bearer
// API and which videos are computed.
import { expect, it } from "vitest";
import { type ClipRecord, inputHash } from "./clips.ts";
import { localAi, matches, runMac, viewerOf } from "./clips-mac.ts";

interface Sent {
  url: string;
  method: string;
  body: string | undefined;
}

function fetcher(replies: { ok: boolean; status: number; body: unknown }[]) {
  const sent: Sent[] = [];
  const queue = [...replies];
  return {
    sent,
    fetch: async (
      url: string,
      init: { method: string; headers: Record<string, string>; body?: string },
    ) => {
      sent.push({ url, method: init.method, body: init.body });
      const reply = queue.shift() ?? { ok: false, status: 599, body: null };
      return { ok: reply.ok, status: reply.status, json: async () => reply.body };
    },
  };
}

const EMPTY = { id: "v1", cues: [] };
const RECORD: ClipRecord = {
  version: 9,
  videoId: "v1",
  movieKey: "k",
  inputHash: "old",
  models: { score: "s", summary: "m" },
  createdAt: "t",
  clips: [],
};

it("sends clef and the summary model to the local server and nothing else", async () => {
  const local = fetcher([
    { ok: true, status: 200, body: { answers: {} } },
    { ok: true, status: 200, body: { choices: [] } },
    { ok: false, status: 400, body: null },
  ]);
  const ai = localAi(local.fetch, "http://127.0.0.1:8791");
  const clef = await ai.run("@cf/cloudflare/clef", { state: 1 });
  const glm = await ai.run("@cf/zai-org/glm-5.3-flash", { messages: [] });
  const failed = await ai.run("@cf/cloudflare/clef", {}).catch((error: unknown) => String(error));
  const clairic = await ai
    .run("@cf/cloudflare/clairic", {})
    .catch((error: unknown) => String(error));
  expect([clef, glm, failed, clairic, local.sent]).toStrictEqual([
    { answers: {} },
    { choices: [] },
    "Error: local clef answered 400",
    "Error: @cf/cloudflare/clairic is not served on the Mac",
    [
      { url: "http://127.0.0.1:8791/clef", method: "POST", body: '{"state":1}' },
      { url: "http://127.0.0.1:8791/bonsai", method: "POST", body: '{"messages":[]}' },
      { url: "http://127.0.0.1:8791/clef", method: "POST", body: "{}" },
    ],
  ]);
});

it("reads ready records and stores new ones with the bearer", async () => {
  const remote = fetcher([
    { ok: true, status: 200, body: { status: "ready", record: RECORD } },
    { ok: true, status: 200, body: { status: "missing" } },
    { ok: false, status: 404, body: { status: "missing" } },
    { ok: true, status: 200, body: { outcome: "written" } },
    { ok: false, status: 400, body: { outcome: "invalid" } },
  ]);
  const viewer = viewerOf(remote.fetch, "https://viewer.test", "t0ken");
  const ready = await viewer.current("v1");
  const missing = await viewer.current("v1");
  const absent = await viewer.current("v2");
  await viewer.store("v1", RECORD);
  const refused = await viewer.store("v1", RECORD).catch((error: unknown) => String(error));
  expect([
    ready?.inputHash,
    missing,
    absent,
    refused,
    remote.sent.map((s) => `${s.method} ${s.url}`),
  ]).toStrictEqual([
    "old",
    null,
    null,
    "Error: storing clips for v1 answered 400",
    [
      "GET https://viewer.test/api/clips/v1.json",
      "GET https://viewer.test/api/clips/v1.json",
      "GET https://viewer.test/api/clips/v2.json",
      "PUT https://viewer.test/api/clips/v1.json",
      "PUT https://viewer.test/api/clips/v1.json",
    ],
  ]);
});

it("keeps records made from the same data on either place", async () => {
  const mac = { ...RECORD, inputHash: await inputHash(EMPTY, "mac") };
  const cloud = { ...RECORD, inputHash: await inputHash(EMPTY, "cloudflare") };
  expect([
    await matches(mac, EMPTY),
    await matches(cloud, EMPTY),
    await matches(RECORD, EMPTY),
  ]).toStrictEqual([true, true, false]);
});

it("computes only videos without a matching record and stores them", async () => {
  const kept = {
    ...RECORD,
    videoId: "v2",
    inputHash: await inputHash({ id: "v2", cues: [] }, "cloudflare"),
  };
  const stored: ClipRecord[] = [];
  const results = await runMac({
    videos: [EMPTY, { id: "v2", cues: [] }],
    ai: { run: async () => Promise.reject(new Error("no model is needed without cues")) },
    viewer: {
      current: async (videoId) => (videoId === "v2" ? kept : null),
      store: async (_videoId, record) => {
        stored.push(record);
      },
    },
    now: new Date(0),
  });
  expect([results, stored.map((r) => [r.videoId, r.movieKey, r.models.summary])]).toStrictEqual([
    [
      { videoId: "v1", outcome: "written", clips: 0 },
      { videoId: "v2", outcome: "current", clips: 0 },
    ],
    [["v1", "mac", "prism-ml/Ternary-Bonsai-2-9B-mlx-2bit"]],
  ]);
});
