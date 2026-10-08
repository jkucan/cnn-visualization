import { defineConfig } from "vite";

const target = process.env.VITE_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  build: { target: "es2022" }, // top-level await in main.ts
  server: {
    port: 5173,
    proxy: {
      "/api": target,
      "/ws": { target, ws: true },
    },
  },
});
