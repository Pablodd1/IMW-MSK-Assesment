import build from '@hono/vite-build/cloudflare-pages'
import devServer from '@hono/vite-dev-server'
import adapter from '@hono/vite-dev-server/cloudflare'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [
    build(),
    devServer({
      adapter,
      entry: 'src/index.tsx'
    })
  ],
  build: {
    target: 'esnext',
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          'three': ['three'],
          'tensorflow': ['@tensorflow/tfjs'],
          'mediapipe': ['@mediapipe/pose']
        }
      }
    }
  },
  optimizeDeps: {
    include: ['three', 'zod', 'bcryptjs']
  },
  server: {
    port: 3000,
    host: true
  }
})