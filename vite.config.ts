import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [tailwindcss()],
  build: {
    target: 'esnext',
    outDir: 'dist',
    emptyOutDir: true,
    copyPublicDir: true
  },
  server: {
    port: 3000,
    host: true
  }
})