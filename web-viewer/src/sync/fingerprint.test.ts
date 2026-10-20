import { expect, it } from "vitest";
import { createFakeFs } from "./fakes.fake.ts";
import {
  collectSources,
  fingerprintPath,
  HASH_LIMIT_BYTES,
  isMediaLabel,
  sameFingerprint,
} from "./fingerprint.ts";
import type { CatalogVideo } from "./schemas.ts";

const video: CatalogVideo = {
  id: "a-review",
  movie: "a-review.mp4",
  localPath: "/lib/rev/a-review.mp4",
  transcriptPath: "/lib/rev/delivery/a-review.srt",
  kept: false,
  original: null,
};

it("fingerprints a small file with its hash", async () => {
  const { fs } = createFakeFs({ "/x": { size: 5, mtimeNs: 7n, sha: "abc" } });
  expect(await fingerprintPath({ fs, path: "/x" })).toStrictEqual({
    size: 5,
    mtimeNs: "7",
    sha256: "abc",
  });
});

it("skips the hash for files above the limit", async () => {
  const { fs } = createFakeFs({ "/big": { size: HASH_LIMIT_BYTES + 1, mtimeNs: 9n } });
  expect(await fingerprintPath({ fs, path: "/big" })).toStrictEqual({
    size: HASH_LIMIT_BYTES + 1,
    mtimeNs: "9",
    sha256: null,
  });
});

it("throws for a missing file", async () => {
  const { fs } = createFakeFs({});
  await expect(fingerprintPath({ fs, path: "/nope" })).rejects.toThrow("Missing file: /nope");
});

it("collects movie, transcript, summary and neighbouring files", async () => {
  const { fs } = createFakeFs({
    "/lib/rev/a-review.mp4": { size: 100, mtimeNs: 5n, sha: "m" },
    "/lib/rev/delivery/a-review.srt": { size: 1, mtimeNs: 2n, sha: "t" },
    "/lib/rev/delivery/a-review.csv": { size: 2, mtimeNs: 3n, sha: "c" },
    "/lib/rev/delivery/a-review.srt.summaries/current.json": { size: 3, mtimeNs: 4n, sha: "s" },
    "/lib/rev/delivery/other.srt": { size: 9, mtimeNs: 9n, sha: "o" },
  });
  const sources = await collectSources({ fs, video });
  expect(Object.keys(sources).sort()).toStrictEqual([
    "movie",
    "sibling:a-review.csv",
    "summary",
    "transcript",
  ]);
  expect(sources.summary).toStrictEqual({ size: 3, mtimeNs: "4", sha256: "s" });
});

it("includes the original when the catalog has one", async () => {
  const { fs } = createFakeFs({
    "/lib/rev/a-review.mp4": {},
    "/lib/rev/delivery/a-review.srt": {},
    "/orig/a.mkv": { size: 77, mtimeNs: 8n, sha: "o" },
  });
  const sources = await collectSources({
    fs,
    video: { ...video, original: { movie: "a.mp4", localPath: "/orig/a.mkv" } },
  });
  expect(Object.keys(sources).sort()).toStrictEqual(["movie", "original", "transcript"]);
});

it("throws when a required source is missing", async () => {
  const { fs } = createFakeFs({ "/lib/rev/delivery/a-review.srt": {} });
  await expect(collectSources({ fs, video })).rejects.toThrow(
    "Missing source file for a-review: movie",
  );
});

it("compares fingerprints on size, mtime and hash", () => {
  const base = { size: 1, mtimeNs: "2", sha256: "x" };
  expect(sameFingerprint(base, { ...base })).toBe(true);
  expect(sameFingerprint(base, { ...base, size: 2 })).toBe(false);
  expect(sameFingerprint(base, { ...base, mtimeNs: "3" })).toBe(false);
  expect(sameFingerprint(base, { ...base, sha256: "y" })).toBe(false);
});

it("recognises media labels", () => {
  expect(isMediaLabel("movie")).toBe(true);
  expect(isMediaLabel("original")).toBe(true);
  expect(isMediaLabel("summary")).toBe(false);
});

it("collects no movie for a recording not edited yet", async () => {
  const { fs } = createFakeFs({
    "/next/raw.json": { size: 1, mtimeNs: 2n, sha: "d" },
    "/dl/raw.mp4": { size: 300, mtimeNs: 4n },
  });
  const sources = await collectSources({
    fs,
    video: {
      ...video,
      movie: null,
      localPath: null,
      transcriptPath: "/next/raw.json",
      original: { movie: "raw.mp4", localPath: "/dl/raw.mp4" },
    },
  });
  expect(Object.keys(sources).sort()).toStrictEqual(["original", "transcript"]);
});
