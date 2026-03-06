import { defineConfig } from 'vite'
import devServer from '@hono/vite-dev-server'

export default defineConfig({
  plugins: [
    devServer({
      entry: 'src/index.tsx',
    })
  ],
  build: {
    outDir: 'dist',
    target: 'node18',
    rollupOptions: {
      input: 'api/index.ts',
      external: ['os', 'path', 'crypto', 'fs'],
      output: {
        format: 'es',
        entryFileNames: 'api/index.js'
      }
    }
  },
  ssr: {
    noExternal: true,
    external: ['os', 'path', 'crypto', 'fs']
  }
})
