import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE ?? "/",
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.NYELETI_API ?? "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: { chunkSizeWarningLimit: 6000 },
});
