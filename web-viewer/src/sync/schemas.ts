// Runs with bun. Validated shapes for the sync configuration, the catalog built from the
// delivery libraries, and the state of what was last uploaded.
import { z } from "zod";

export const configSchema = z.object({
  streamer: z.string().min(1),
  roots: z.array(z.string().min(1)).min(1),
  youtubeOriginals: z.string().min(1).nullable(),
  // Downloads without a delivery yet, published raw and without captions.
  raw: z.string().min(1).nullable().default(null),
  bucket: z.string().min(1),
  uploaderWorker: z.string().min(1),
});

const originalSchema = z.object({ movie: z.string().min(1), localPath: z.string().min(1) });

export const catalogSchema = z.object({
  videos: z.array(
    z.object({
      id: z.string().regex(/^[A-Za-z0-9._-]+$/),
      // Both null for a recording not edited yet: only its original is published.
      movie: z.string().min(1).nullable(),
      localPath: z.string().min(1).nullable(),
      transcriptPath: z.string().min(1),
      original: originalSchema.nullable(),
      // Published before and no longer found locally: carried over without re-reading files.
      kept: z.boolean().default(false),
    }),
  ),
});

export const fingerprintSchema = z.object({
  size: z.number().int().nonnegative(),
  mtimeNs: z.string().regex(/^[0-9]+$/),
  sha256: z.string().nullable(),
});

export const stateSchema = z.object({
  sources: z.record(z.string(), z.record(z.string(), fingerprintSchema)),
  data: z.record(z.string(), z.string()),
  catalogSha256: z.string().nullable(),
  // Per media key, what an upload sent: SHA-256 streamed while reading it, and the multipart ETag.
  uploads: z
    .record(
      z.string(),
      z.object({
        size: z.number().int().nonnegative(),
        sha256: z.string().regex(/^[0-9a-f]{64}$/),
        etag: z.string().min(1),
      }),
    )
    .default({}),
});

export type SyncConfig = z.infer<typeof configSchema>;
export type Catalog = z.infer<typeof catalogSchema>;
export type CatalogVideo = Catalog["videos"][number];
export type Fingerprint = z.infer<typeof fingerprintSchema>;
export type SyncState = z.infer<typeof stateSchema>;
export type Sources = Record<string, Fingerprint>;

export const EMPTY_STATE: SyncState = { sources: {}, data: {}, catalogSha256: null, uploads: {} };
