import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxies to the backend so the UI and API share an origin in development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    proxy: {
      "/api": { target: "http://127.0.0.1:8090", changeOrigin: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
