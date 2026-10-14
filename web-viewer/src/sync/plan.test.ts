import { expect, it } from "vitest";
import { changedLabels, type PlanInput, planChanges, settledKey } from "./plan.ts";
import { EMPTY_STATE, type Fingerprint, type Sources } from "./schemas.ts";

const fp = (size: number, mtimeNs: string, sha256: string | null): Fingerprint => ({
  size,
  mtimeNs,
  sha256,
});

const sources: Sources = {
  movie: fp(100, "1", null),
  transcript: fp(1, "1", "t"),
  summary: fp(2, "1", "s"),
};

function input(overrides: Partial<PlanInput>): PlanInput {
  return {
    videos: [
      {
        id: "a-review",
        movie: "a-review.mp4",
        localPath: "/lib/a-review.mp4",
        transcriptPath: "/lib/a-review.srt",
        original: null,
      },
    ],
    current: { "a-review": sources },
    dataSha: { "a-review": "d1" },
    catalogSha: "c1",
    dataDirectory: "/next",
    state: {
      sources: { "a-review": sources },
      data: { "data/a-review.json": "d1" },
      catalogSha256: "c1",
    },
    remoteSizes: {},
    proxySizes: {},
    ...overrides,
  };
}

it("plans nothing when everything matches the last upload", () => {
  expect(planChanges(input({}))).toStrictEqual({
    changes: [],
    settledMedia: ["a-review:movie"],
  });
});

it("adopts media already in the bucket with the same size", () => {
  const plan = planChanges(
    input({ state: EMPTY_STATE, remoteSizes: { "media/a-review.mp4": 100 } }),
  );
  expect(plan.changes.map((change) => change.kind)).toStrictEqual(["data", "catalog"]);
  expect(plan.settledMedia).toStrictEqual(["a-review:movie"]);
});

it("plans a media upload when the bucket has a different size", () => {
  const plan = planChanges(
    input({ state: EMPTY_STATE, remoteSizes: { "media/a-review.mp4": 90 } }),
  );
  expect(plan.changes[0]).toStrictEqual({
    kind: "media",
    key: "media/a-review.mp4",
    path: "/lib/a-review.mp4",
    reason: "size differs (bucket 90 B, local 100 B)",
    videoId: "a-review",
    label: "movie",
    proxy: false,
  });
  expect(plan.settledMedia).toStrictEqual([]);
});

it("plans a media upload when the movie is not in the bucket", () => {
  const plan = planChanges(input({ state: EMPTY_STATE }));
  expect(plan.changes[0]?.reason).toBe("not in the bucket");
});

it("plans a media upload when the file changed since the last upload", () => {
  const edited: Sources = { ...sources, movie: fp(100, "2", null) };
  const plan = planChanges(input({ current: { "a-review": edited } }));
  expect(plan.changes.map((change) => `${change.kind}:${change.reason}`)).toStrictEqual([
    "media:file changed since the last upload",
    "catalog:videos changed",
  ]);
});

it("plans data when a summary or neighbouring file changed", () => {
  const edited: Sources = {
    ...sources,
    summary: fp(3, "2", "s2"),
    "sibling:a.csv": fp(1, "1", "c"),
  };
  const plan = planChanges(input({ current: { "a-review": edited } }));
  expect(plan.changes[0]?.reason).toBe("source files changed: summary, sibling:a.csv");
  expect(plan.changes[0]?.path).toBe("/next/a-review.json");
});

it("plans data when the generated JSON changed", () => {
  const plan = planChanges(input({ dataSha: { "a-review": "d2" } }));
  expect(plan.changes[0]?.reason).toBe("generated data changed");
});

it("plans only the catalog when only its hash changed", () => {
  const plan = planChanges(input({ catalogSha: "c2" }));
  expect(plan.changes).toStrictEqual([
    {
      kind: "catalog",
      key: "catalog.json",
      path: "/next/catalog.json",
      reason: "catalog changed",
    },
  ]);
});

it("marks an MKV original for proxy conversion", () => {
  const withOriginal = {
    id: "a-review",
    movie: "a-review.mp4",
    localPath: "/lib/a-review.mp4",
    transcriptPath: "/lib/a-review.srt",
    original: { movie: "orig.mp4", localPath: "/orig/orig.MKV" },
  };
  const current: Sources = { ...sources, original: fp(500, "5", null) };
  const plan = planChanges(
    input({
      videos: [withOriginal],
      current: { "a-review": current },
      remoteSizes: { "media-original/orig.mp4": 1 },
    }),
  );
  const original = plan.changes.find((change) => change.key === "media-original/orig.mp4");
  expect(original).toStrictEqual({
    kind: "media",
    key: "media-original/orig.mp4",
    path: "/orig/orig.MKV",
    reason: "size differs (bucket 1 B, local 500 B)",
    videoId: "a-review",
    label: "original",
    proxy: true,
  });
});

it("lists changed non-media labels including removed ones", () => {
  expect(
    changedLabels({
      previous: { movie: fp(1, "1", null), "sibling:x": fp(1, "1", "x"), summary: fp(1, "1", "s") },
      current: { movie: fp(9, "9", null), summary: fp(1, "1", "s") },
    }),
  ).toStrictEqual(["sibling:x"]);
  expect(
    changedLabels({ previous: undefined, current: { summary: fp(1, "1", "s") } }),
  ).toStrictEqual(["summary"]);
});

it("builds settled keys", () => {
  expect(settledKey("a", "movie")).toBe("a:movie");
});

it("treats a catalog original without a fingerprint as settled", () => {
  const withOriginal = {
    id: "a-review",
    movie: "a-review.mp4",
    localPath: "/lib/a-review.mp4",
    transcriptPath: "/lib/a-review.srt",
    original: { movie: "orig.mp4", localPath: "/orig/orig.mp4" },
  };
  const plan = planChanges(input({ videos: [withOriginal] }));
  expect(plan.settledMedia).toStrictEqual(["a-review:movie", "a-review:original"]);
  expect(plan.changes).toStrictEqual([]);
});

it("adopts a proxied original when the bucket matches the local proxy size", () => {
  const withOriginal = {
    id: "a-review",
    movie: "a-review.mp4",
    localPath: "/lib/a-review.mp4",
    transcriptPath: "/lib/a-review.srt",
    original: { movie: "orig.mp4", localPath: "/orig/orig.mkv" },
  };
  const current: Sources = { ...sources, original: fp(900, "5", null) };
  const plan = planChanges(
    input({
      videos: [withOriginal],
      current: { "a-review": current },
      remoteSizes: { "media-original/orig.mp4": 300 },
      proxySizes: { "media-original/orig.mp4": 300 },
    }),
  );
  expect(plan.settledMedia).toStrictEqual(["a-review:movie", "a-review:original"]);
  expect(plan.changes).toStrictEqual([]);
});

it("plans a proxied original when no local proxy exists to compare", () => {
  const withOriginal = {
    id: "a-review",
    movie: "a-review.mp4",
    localPath: "/lib/a-review.mp4",
    transcriptPath: "/lib/a-review.srt",
    original: { movie: "orig.mp4", localPath: "/orig/orig.mkv" },
  };
  const current: Sources = { ...sources, original: fp(900, "5", null) };
  const plan = planChanges(
    input({
      videos: [withOriginal],
      current: { "a-review": current },
      remoteSizes: { "media-original/orig.mp4": 300 },
    }),
  );
  expect(plan.changes.map((change) => change.key)).toStrictEqual([
    "media-original/orig.mp4",
    "catalog.json",
  ]);
});
