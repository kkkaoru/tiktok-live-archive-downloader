import { defineConfig, bindings } from "cf/config";

export default defineConfig({
  worker: {
    name: "transcript-viewer-web",
    entrypoint: "src/worker.js",
    compatibilityDate: "2025-06-01",
    workersDev: true,
    domains: process.env.VIEWER_DOMAIN ? process.env.VIEWER_DOMAIN.split(",") : [],
    observability: { enabled: true },
    assets: { runWorkerFirst: true },
    env: {
      ASSETS: bindings.assets(),
      ALLOW_KV: bindings.kv(),
      MEDIA: bindings.r2({ name: "transcript-viewer-media" }),
    },
  },
});
