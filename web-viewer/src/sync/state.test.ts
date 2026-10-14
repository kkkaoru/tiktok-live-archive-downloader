import { expect, it } from "vitest";
import { createFakeFs } from "./fakes.fake.ts";
import { EMPTY_STATE, type Fingerprint, type Sources } from "./schemas.ts";
import { loadState, nextState, saveState } from "./state.ts";

const fp = (size: number, sha256: string | null): Fingerprint => ({ size, mtimeNs: "1", sha256 });

const current: Record<string, Sources> = {
  a: { movie: fp(100, null), transcript: fp(1, "t2"), summary: fp(2, "s2") },
};

const previous = {
  sources: { a: { movie: fp(90, null), transcript: fp(1, "t1"), "sibling:gone": fp(1, "g") } },
  data: { "data/a.json": "old" },
  catalogSha256: "old-catalog",
};

it("loads an empty state when no file exists", async () => {
  const { fs } = createFakeFs({});
  expect(await loadState({ fs, path: "/s.json" })).toStrictEqual(EMPTY_STATE);
});

it("loads and validates a saved state", async () => {
  const text = JSON.stringify({ sources: {}, data: { "data/a.json": "x" }, catalogSha256: null });
  const { fs } = createFakeFs({ "/s.json": { text } });
  expect(await loadState({ fs, path: "/s.json" })).toStrictEqual({
    sources: {},
    data: { "data/a.json": "x" },
    catalogSha256: null,
  });
});

it("rejects a malformed state file", async () => {
  const { fs } = createFakeFs({ "/s.json": { text: '{"sources": 1}' } });
  await expect(loadState({ fs, path: "/s.json" })).rejects.toThrow();
});

it("saves through a private temporary file and a rename", async () => {
  const { fs, calls } = createFakeFs({});
  await saveState({ fs, path: "/s.json", state: EMPTY_STATE });
  expect(calls).toStrictEqual(["write /s.json.tmp 600", "rename /s.json.tmp /s.json"]);
});

it("records published data, catalog and settled media", () => {
  const state = nextState({
    previous,
    current,
    dataSha: { a: "new" },
    catalogSha: "new-catalog",
    settledMedia: new Set(["a:movie"]),
    dataPublished: true,
    catalogPublished: true,
  });
  expect(Object.keys(state.sources.a ?? {}).sort()).toStrictEqual([
    "movie",
    "summary",
    "transcript",
  ]);
  expect(state.sources.a?.movie).toStrictEqual(fp(100, null));
  expect(state.data).toStrictEqual({ "data/a.json": "new" });
  expect(state.catalogSha256).toBe("new-catalog");
});

it("keeps the previous media record until the upload succeeds", () => {
  const state = nextState({
    previous,
    current,
    dataSha: { a: "new" },
    catalogSha: "new-catalog",
    settledMedia: new Set(),
    dataPublished: true,
    catalogPublished: false,
  });
  expect(state.sources.a?.movie).toStrictEqual(fp(90, null));
  expect(state.catalogSha256).toBe("old-catalog");
});

it("keeps previous non-media records and data until data is published", () => {
  const state = nextState({
    previous,
    current,
    dataSha: { a: "new" },
    catalogSha: "new-catalog",
    settledMedia: new Set(["a:movie"]),
    dataPublished: false,
    catalogPublished: false,
  });
  expect(state.sources.a?.transcript).toStrictEqual(fp(1, "t1"));
  expect(state.sources.a?.["sibling:gone"]).toStrictEqual(fp(1, "g"));
  expect(state.sources.a?.movie).toStrictEqual(fp(100, null));
  expect(state.data).toStrictEqual({ "data/a.json": "old" });
});
