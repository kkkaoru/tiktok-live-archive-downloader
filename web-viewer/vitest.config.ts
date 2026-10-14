// Runs with bun (vitest under Node-compatible APIs). Entry points in src/bin stay thin and are
// excluded from coverage; every other source file must reach 90% on its own.
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.ts"],
      exclude: ["src/**/*.test.ts", "src/**/*.fake.ts", "src/bin/**"],
      thresholds: { perFile: true, statements: 90, branches: 90, functions: 90, lines: 90 },
    },
  },
});
