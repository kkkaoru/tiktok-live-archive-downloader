import { defineConfig, bindings, triggers } from "cf/config";

const CLIPS_QUEUE = "transcript-viewer-clips";

export default defineConfig({
  worker: {
    name: "transcript-viewer-web",
    entrypoint: "src/worker.js",
    compatibilityDate: "2025-06-01",
    workersDev: true,
    domains: process.env.VIEWER_DOMAIN ? process.env.VIEWER_DOMAIN.split(",") : [],
    observability: { enabled: true },
    assets: { runWorkerFirst: true },
    // Clip generation: R2 event notifications for data/*.json and the viewers' requests arrive
    // here one at a time, so a video is never computed twice in parallel.
    triggers: [triggers.queue({ name: CLIPS_QUEUE, maxBatchSize: 1, maxConcurrency: 1, maxRetries: 5, retryDelay: 120 })],
    env: {
      ASSETS: bindings.assets(),
      ALLOW_KV: bindings.kv(),
      MEDIA: bindings.r2({ name: "transcript-viewer-media" }),
      AI: bindings.ai(),
      CLIPS_QUEUE: bindings.queue({ name: CLIPS_QUEUE }),
    },
  },
});
