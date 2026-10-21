// Runs with bun (vitest). Clip generation with a fake Workers AI and an in-memory R2 bucket.
import { createHash } from "node:crypto";
import { expect, it } from "vitest";
import {
  type Ai,
  type Bucket,
  bandOf,
  type ClipEnv,
  type ClipRecord,
  candidates,
  clefRequests,
  compose,
  enqueue,
  generate,
  inputHash,
  lookup,
  movieKey,
  openings,
  parseSummary,
  patient,
  peaksOf,
  refresh,
  requestOf,
  type Segment,
  scored,
  summaryRequest,
  utterances,
  type VideoData,
} from "./clips.ts";

interface FakeBucket extends Bucket {
  objects: Map<string, string | Uint8Array>;
}

function bucket(entries: Record<string, string | Uint8Array>): FakeBucket {
  const objects = new Map(Object.entries(entries));
  const body = (value: string | Uint8Array) => ({
    json: async () => JSON.parse(String(value)),
    arrayBuffer: async () =>
      typeof value === "string" ? new TextEncoder().encode(value).buffer : value.slice().buffer,
  });
  return {
    objects,
    head: async (key) => {
      const value = objects.get(key);
      return value === undefined ? null : { size: value.length };
    },
    get: async (key, options) => {
      const value = objects.get(key);
      if (value === undefined) return null;
      const range = options?.range;
      return range && typeof value !== "string"
        ? body(value.slice(range.offset, range.offset + range.length))
        : body(value);
    },
    put: async (key, value) => objects.set(key, value),
    delete: async (key) => objects.delete(key),
  };
}

const cue = (start: number, end: number, text: string) => ({ start, end, text });

// 10 s bins: a peak at 100-110 s and its lead-up from 70 s, one utterance per bin.
const DATA: VideoData = {
  id: "v1",
  cues: [
    cue(72, 80, "なんでこうなったと思う？"),
    cue(82, 90, "それでね"),
    cue(92, 99, "まさかの展開"),
    cue(101, 110, "うわーすごい！"),
  ],
  excitement: {
    binSeconds: 10,
    speech: { score: [0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5, 0.9, 0] },
    combined: null,
  },
};

const HOOKED = {
  answers: {
    "clip.0": { score: 3 },
    "clip.1": { score: 3 },
    "clip.2": { score: 3 },
    "clip.3": { score: 3 },
    "question.0": { noul: 0.9 },
    "hook.0.0": { score: 4 },
    "hook.1.0": { score: 1 },
    "hook.2.0": { score: 1 },
    "hook.3.0": { score: 4 },
    "link.1": { noul: 0.9 },
    "link.2": { noul: 0.9 },
    "link.3": { noul: 0.9 },
  },
};

function fakeAi(): Ai & { calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    run: async (model) => {
      calls.push(model);
      return model === "@cf/cloudflare/clef"
        ? HOOKED
        : { choices: [{ message: { content: 'ok {"title":"題","summary":"要約"}' } }] };
    },
  };
}

it("merges cumulative captions into one utterance", () => {
  expect(utterances([cue(0, 1, "あ"), cue(1, 2, "あい"), cue(5, 6, "う")])).toStrictEqual([
    { start: 0, end: 2, text: "あい" },
    { start: 5, end: 6, text: "う" },
  ]);
});

it("puts every clip length into one of the two bands", () => {
  expect([bandOf(9.9), bandOf(10), bandOf(30), bandOf(30.5), bandOf(90), bandOf(91)]).toStrictEqual(
    [null, "short", "short", "long", "long", null],
  );
});

it("takes the strongest bins whose lead-ups do not overlap", () => {
  expect(peaksOf([0.9, 0.8, 0, 0, 0, 0.7, 0])).toStrictEqual([0, 5]);
});

it("takes a unit from the peak bin and each lead-up bin before it", () => {
  expect(candidates(DATA)).toStrictEqual([
    {
      start: 72,
      end: 80,
      text: "なんでこうなったと思う？",
      excitement: 0.2,
      peak: 10,
      parts: [{ start: 72, text: "なんでこうなったと思う？" }],
    },
    {
      start: 82,
      end: 90,
      text: "それでね",
      excitement: 0.3,
      peak: 10,
      parts: [{ start: 82, text: "それでね" }],
    },
    {
      start: 92,
      end: 99,
      text: "まさかの展開",
      excitement: 0.5,
      peak: 10,
      parts: [{ start: 92, text: "まさかの展開" }],
    },
    {
      start: 101,
      end: 110,
      text: "うわーすごい！",
      excitement: 0.9,
      peak: 10,
      parts: [{ start: 101, text: "うわーすごい！" }],
    },
  ]);
});

it("has no candidates without excitement", () => {
  expect(candidates({ id: "x", cues: DATA.cues })).toStrictEqual([]);
});

it("prefers the combined score when comments exist", () => {
  const data = {
    ...DATA,
    excitement: {
      binSeconds: 10,
      speech: { score: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
      combined: { score: [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0] },
    },
  };
  expect(candidates(data).map((s) => [s.start, s.peak])).toStrictEqual([
    [72, 8],
    [82, 8],
  ]);
});

const UNIT: Segment = {
  start: 0,
  end: 10,
  text: "",
  excitement: 0,
  peak: 0,
  parts: [
    { start: 0, text: "a" },
    { start: 1, text: "b" },
    { start: 2.5, text: "c" },
    { start: 9, text: "d" },
  ],
};

it("offers the first utterances as openings with what their first 3 s say", () => {
  expect(openings(UNIT)).toStrictEqual([
    { start: 0, text: "a b c" },
    { start: 1, text: "b c" },
    { start: 2.5, text: "c" },
  ]);
  expect(openings({ ...UNIT, end: 3.5 })).toStrictEqual([{ start: 0, text: "a b c" }]);
});

it("asks every question in requests of at most 64", () => {
  const one = clefRequests(candidates(DATA));
  const many = clefRequests(
    Array.from({ length: 22 }, (_, i) => ({
      ...UNIT,
      start: i * 100,
      end: i * 100 + 4,
      peak: i,
      parts: [{ start: i * 100, text: "x" }],
    })),
  );
  expect([
    one.length,
    Object.keys(Object(one[0]).questions),
    many.map((r) => Object.keys(Object(r).questions).length),
  ]).toStrictEqual([
    1,
    [
      "clip.0",
      "question.0",
      "hook.0.0",
      "clip.1",
      "question.1",
      "hook.1.0",
      "link.1",
      "clip.2",
      "question.2",
      "hook.2.0",
      "link.2",
      "clip.3",
      "question.3",
      "hook.3.0",
      "link.3",
    ],
    [64, 2],
  ]);
});

it("turns clef answers into quality, question, links and the strongest opening", () => {
  const segments = candidates(DATA);
  expect(
    scored(segments, [HOOKED]).map((s) => [s.quality, s.question, s.linked, s.hook, s.hookStart]),
  ).toStrictEqual([
    [0.585, 0.9, false, 1, 72],
    [0.615, 0, true, 0.25, 82],
    [0.675, 0, true, 0.25, 92],
    [0.795, 0, true, 1, 101],
  ]);
});

it("tells each linked chain as climax-first and hook-to-climax, short and long", () => {
  const clips = compose(scored(candidates(DATA), [HOOKED]));
  expect(
    clips.map((c) => [c.pattern, c.kind, c.duration, c.ranges.map((r) => r.start)]),
  ).toStrictEqual([
    ["hook-to-climax", "long", 32, [72, 82, 92, 101]],
    ["hook-to-climax", "short", 24, [72, 92, 101]],
    ["climax-first", "long", 32, [101, 72, 82, 92]],
    ["climax-first", "short", 24, [101, 82, 92]],
  ]);
});

it("keeps no clip whose first 3 s do not hook", () => {
  const weak = {
    answers: { ...HOOKED.answers, "hook.0.0": { score: 1 }, "hook.3.0": { score: 1 } },
  };
  const unlinked = {
    answers: {
      ...HOOKED.answers,
      "link.1": { noul: 0.1 },
      "link.2": { noul: 0.1 },
      "link.3": { noul: 0.1 },
    },
  };
  expect([
    compose(scored(candidates(DATA), [weak])),
    compose(scored(candidates(DATA), [unlinked])),
  ]).toStrictEqual([[], []]);
});

it("starts the first range at the strongest opening", () => {
  const unit = (start: number, excitement: number) => ({
    ...UNIT,
    start,
    end: start + 10,
    excitement,
    parts: [
      { start, text: "x" },
      { start: start + 4, text: "y" },
    ],
  });
  const segments = [unit(0, 0.2), unit(20, 0.9)];
  const answers = {
    answers: {
      "hook.0.0": { score: 4 },
      "hook.1.0": { score: 0 },
      "hook.1.1": { score: 4 },
      "link.1": { noul: 1 },
    },
  };
  expect(compose(scored(segments, [answers])).map((c) => [c.pattern, c.ranges])).toStrictEqual([
    [
      "hook-to-climax",
      [
        { start: 0, end: 10 },
        { start: 20, end: 30 },
      ],
    ],
    [
      "climax-first",
      [
        { start: 24, end: 30 },
        { start: 0, end: 10 },
      ],
    ],
  ]);
});

it("keeps the densest run of each bin", () => {
  const data = {
    id: "d",
    cues: [cue(0, 3, "a"), cue(3, 6, "bbbb"), cue(6, 9, "cc")],
    excitement: { binSeconds: 10, speech: { score: [1] }, combined: null },
  };
  expect(candidates(data).map((s) => [s.start, s.end, s.text])).toStrictEqual([
    [0, 9, "a bbbb cc"],
  ]);
});

it("keeps at most two clips per peak in a band", () => {
  const split = {
    answers: { ...HOOKED.answers, "link.2": { noul: 0.1 }, "hook.2.0": { score: 4 } },
  };
  expect(
    compose(scored(candidates(DATA), [split])).map((c) => [
      c.pattern,
      c.ranges.map((r) => r.start),
    ]),
  ).toStrictEqual([
    ["hook-to-climax", [92, 101]],
    ["climax-first", [101, 92]],
  ]);
});

it("drops a story shorter than 10 s", () => {
  const unit = (start: number, excitement: number) => ({
    ...UNIT,
    start,
    end: start + 5,
    excitement,
    parts: [
      { start, text: "x" },
      { start: start + 2, text: "y" },
    ],
  });
  const answers = { answers: { "hook.1.1": { score: 4 }, "link.1": { noul: 1 } } };
  expect(compose(scored([unit(0, 0.2), unit(20, 0.9)], [answers]))).toStrictEqual([]);
});

it("summarizes a range outside every unit as silence", () => {
  const request = summaryRequest(
    {
      kind: "short",
      pattern: "hook-to-climax",
      duration: 12,
      score: 1,
      ranges: [{ start: 500, end: 512 }],
    },
    [],
  );
  expect(JSON.parse(Object(request).messages[1].content).ranges).toStrictEqual([
    { time: "0:08:20〜0:08:32", excitement: 0, text: "" },
  ]);
});

it("writes a summary request with the story and each range's own words", () => {
  const segments = candidates(DATA);
  const request = summaryRequest(
    {
      kind: "short",
      pattern: "climax-first",
      duration: 17,
      score: 1,
      ranges: [
        { start: 101, end: 110 },
        { start: 92, end: 99 },
      ],
    },
    segments,
  );
  expect(JSON.parse(Object(request).messages[1].content)).toStrictEqual({
    story:
      "最初の区間がこの話の山場（一番の盛り上がり）で、続く区間はそこに至るまでの経緯を時系列で見せる構成",
    seconds: 17,
    ranges: [
      { time: "0:01:41〜0:01:50", excitement: 0.9, text: "うわーすごい！" },
      { time: "0:01:32〜0:01:39", excitement: 0.5, text: "まさかの展開" },
    ],
  });
});

it("reads the JSON object out of the model's reply", () => {
  expect(
    parseSummary({
      choices: [{ message: { content: '```json\n{"title":"t","summary":"s"}\n```' } }],
    }),
  ).toStrictEqual({
    title: "t",
    summary: "s",
  });
});

it("rejects a reply without a title", () => {
  expect(() => parseSummary({ choices: [{ message: { content: "{}" } }] })).toThrow();
});

it("generates titled clips and reuses summaries of unchanged stories", async () => {
  const ai = fakeAi();
  const now = new Date("2026-10-21T00:00:00Z");
  const first = await generate({ ai, data: DATA, movieKey: "k", previous: null, now });
  const second = await generate({ ai, data: DATA, movieKey: "k", previous: first, now });
  expect([
    first.clips.map((c) => [c.id, c.pattern, c.title]),
    ai.calls.length,
    second.clips,
  ]).toStrictEqual([
    [
      ["c1", "hook-to-climax", "題"],
      ["c2", "hook-to-climax", "題"],
      ["c3", "climax-first", "題"],
      ["c4", "climax-first", "題"],
    ],
    6,
    first.clips,
  ]);
});

it("generates nothing and calls no model without candidates", async () => {
  const ai = fakeAi();
  const record = await generate({
    ai,
    data: { id: "x", cues: [] },
    movieKey: "k",
    previous: null,
    now: new Date(0),
  });
  expect([record.clips, record.excitement, ai.calls]).toStrictEqual([[], false, []]);
});

it("hashes inputs so a changed cue changes the hash", async () => {
  const changed = { ...DATA, cues: [cue(0, 1, "x")] };
  expect((await inputHash(DATA)) === (await inputHash(changed))).toBe(false);
});

it("computes the same movie key as replay.excitement.movie_key", async () => {
  const movie = Uint8Array.from({ length: 1572864 }, (_, i) => i % 251);
  const store = bucket({ "media/a.mp4": movie, "media/abc.mp4": "abc", "media/empty.mp4": "" });
  expect([
    await movieKey(store, "media/a.mp4"),
    await movieKey(store, "media/abc.mp4"),
    await movieKey(store, "media/empty.mp4"),
    await movieKey(store, "media/none.mp4"),
  ]).toStrictEqual([
    "3d890c891d3c9bf5344b66383deee2a9",
    "2e8e504c5e6b886af70f8c03cbfc31ac",
    "5feceb66ffc86f38d952786c6d696c79",
    null,
  ]);
});

function clipEnv(entries: Record<string, string | Uint8Array>) {
  const sent: unknown[] = [];
  const env: ClipEnv & { MEDIA: FakeBucket } = {
    AI: fakeAi(),
    MEDIA: bucket(entries),
    CLIPS_QUEUE: { send: async (body) => sent.push(body) },
  };
  return { env, sent };
}

const ABC_KEY = "2e8e504c5e6b886af70f8c03cbfc31ac";

it("refreshes once per input and keeps a pointer from the video to its movie key", async () => {
  const { env } = clipEnv({
    "data/v1.json": JSON.stringify(DATA),
    "media/v1.mp4": "abc",
    "clips/pending/v1.json": "{}",
  });
  const now = new Date(0);
  const outcomes = [
    await refresh(env, { videoId: "v1", force: false }, now),
    await refresh(env, { videoId: "v1", force: false }, now),
    await refresh(env, { videoId: "v1", force: true }, now),
  ];
  expect([
    outcomes,
    env.MEDIA.objects.get("clips/videos/v1.json"),
    env.MEDIA.objects.has("clips/pending/v1.json"),
  ]).toStrictEqual([
    ["written", "unchanged", "written"],
    '{"movieKey":"2e8e504c5e6b886af70f8c03cbfc31ac"}',
    false,
  ]);
});

it("uses the original movie of a video that is not edited yet", async () => {
  const { env } = clipEnv({
    "data/v1.json": JSON.stringify(DATA),
    "catalog.json": JSON.stringify({ videos: [{ id: "v1", original: { movie: "raw.mp4" } }] }),
    "media-original/raw.mp4": "abc",
  });
  await refresh(env, { videoId: "v1", force: false }, new Date(0));
  expect(env.MEDIA.objects.has(`clips/${ABC_KEY}.json`)).toBe(true);
});

it("skips videos without data or movie", async () => {
  const { env } = clipEnv({ "data/v1.json": JSON.stringify(DATA), "catalog.json": "{}" });
  expect([
    await refresh(env, { videoId: "../x", force: false }, new Date(0)),
    await refresh(env, { videoId: "v2", force: false }, new Date(0)),
    await refresh(env, { videoId: "v1", force: false }, new Date(0)),
  ]).toStrictEqual(["no-data", "no-data", "no-movie"]);
});

it("reads own requests and R2 upload events from the queue", () => {
  expect([
    requestOf({ videoId: "a", force: true }),
    requestOf({ videoId: "a" }),
    requestOf({ action: "PutObject", object: { key: "data/b.json" } }),
    requestOf({ object: { key: "media/b.mp4" } }),
    requestOf("x"),
  ]).toStrictEqual([
    { videoId: "a", force: true },
    { videoId: "a", force: false },
    { videoId: "b", force: false },
    null,
    null,
  ]);
});

it("queues once per ten minutes unless a reload asks for more", async () => {
  const { env, sent } = clipEnv({});
  await enqueue(env, { videoId: "v1", force: false }, new Date(0));
  await enqueue(env, { videoId: "v1", force: false }, new Date(60_000));
  await enqueue(env, { videoId: "v1", force: true }, new Date(60_000));
  await enqueue(env, { videoId: "v1", force: true }, new Date(120_000));
  await enqueue(env, { videoId: "v1", force: false }, new Date(700_000));
  await enqueue(env, { videoId: "../x", force: false }, new Date(0));
  expect(sent).toStrictEqual([
    { videoId: "v1", force: false },
    { videoId: "v1", force: true },
    { videoId: "v1", force: false },
  ]);
});

const RECORD: ClipRecord = {
  version: 1,
  videoId: "v1",
  movieKey: ABC_KEY,
  inputHash: "h",
  models: { score: "s", summary: "m" },
  createdAt: "t",
  clips: [
    {
      id: "c1",
      kind: "short",
      duration: 12,
      score: 1,
      ranges: [{ start: 0, end: 12 }],
      title: "t",
      summary: "s",
    },
  ],
};

it("looks clips up by the video's pointer or a viewer's own key", async () => {
  const { env } = clipEnv({
    "clips/videos/v1.json": JSON.stringify({ movieKey: ABC_KEY }),
    [`clips/${ABC_KEY}.json`]: JSON.stringify(RECORD),
    "clips/pending/v2.json": JSON.stringify({ at: 0, force: false }),
    "clips/pending/v3.json": JSON.stringify({ at: 0, force: true }),
  });
  const now = new Date(60_000);
  expect([
    await lookup(env, { videoId: "v1", key: null, now }),
    await lookup(env, { videoId: "v9", key: ABC_KEY, now }),
    await lookup(env, { videoId: "v1", key: "f".repeat(32), now }),
    await lookup(env, { videoId: "v2", key: null, now }),
    await lookup(env, { videoId: "v3", key: null, now }),
    await lookup(env, { videoId: "v3", key: null, now: new Date(700_000) }),
    await lookup(env, { videoId: "v4", key: null, now }),
    await lookup(env, { videoId: "../x", key: null, now }),
  ]).toStrictEqual([
    { status: "ready", record: RECORD },
    { status: "ready", record: RECORD },
    { status: "mismatch" },
    { status: "pending", force: false },
    { status: "pending", force: true },
    { status: "missing" },
    { status: "missing" },
    { status: "missing" },
  ]);
});

it("waits and retries only rate-limited model calls", async () => {
  const waits: number[] = [];
  const replies = [new Error("AiError: 3021: rate limiting"), "ok"];
  const flaky: Ai = {
    run: async () => {
      const next = replies.shift();
      if (next instanceof Error) throw next;
      return next;
    },
  };
  const broken: Ai = {
    run: async () => {
      throw new Error("AiError: 5000: bad input");
    },
  };
  const limited: Ai = {
    run: async () => {
      throw new Error("AiError: 3021: rate limiting");
    },
  };
  const wait = async (ms: number) => waits.push(ms);
  const result = await patient(flaky, wait).run("m", {});
  await expect(patient(broken, wait).run("m", {})).rejects.toThrow("5000");
  await expect(patient(limited, wait).run("m", {})).rejects.toThrow("3021");
  expect([result, waits]).toStrictEqual(["ok", [20000, 20000, 20000, 20000, 20000]]);
});

it("matches the Python key of a 1.5 MiB pattern through node's hash too", () => {
  const movie = Uint8Array.from({ length: 1572864 }, (_, i) => i % 251);
  const digest = createHash("sha256")
    .update("1572864")
    .update(movie.subarray(0, 1 << 20))
    .update(movie.subarray(524288));
  expect(digest.digest("hex").slice(0, 32)).toBe("3d890c891d3c9bf5344b66383deee2a9");
});
