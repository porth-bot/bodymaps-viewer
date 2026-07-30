import { defineConfig } from 'vite';

// `base` is overridden by CI so the same source deploys to a project page
// (/<repo>/) and to a custom domain or local preview (/) without edits.
export default defineConfig({
  base: process.env.VITE_BASE ?? '/',
  build: {
    target: 'es2022',
    sourcemap: true,
    // The .nii.gz sample case lives in public/ and is copied verbatim; nothing
    // here should be inlined as base64.
    assetsInlineLimit: 0,
  },
  worker: {
    format: 'es',
  },
  server: {
    port: 5173,
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
