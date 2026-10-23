import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "es2022",
    cssCodeSplit: false,
    assetsInlineLimit: 0,
    reportCompressedSize: false,
    chunkSizeWarningLimit: 300,
  },
});
