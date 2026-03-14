import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

const backendPort = process.env.BACKEND_PORT || '8000';
const apiTarget = process.env.VITE_API_TARGET || `http://localhost:${backendPort}`;

export default defineConfig({
  plugins: [react()],

  // Build output to backend/static
  build: {
    outDir: '../backend/static',
    emptyOutDir: true,
  },

  // Development proxy to backend
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        ws: true,
      },
    },
  },

  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
