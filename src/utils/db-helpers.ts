import type { Bindings } from '../types.js'

// Global mockD1 for Cloudflare context
export const mockD1 = {
  prepare: (query: string) => {
    // For dev without Cloudflare context
    return {
      bind: (...args: any[]) => ({
        first: async () => null,
        all: async () => ({ results: [], success: true, meta: { last_row_id: null } }),
        run: async () => ({ success: true, meta: { last_row_id: null } })
      }),
      first: async () => null,
      all: async () => ({ results: [], success: true, meta: { last_row_id: null } }),
      run: async () => ({ success: true, meta: { last_row_id: null } })
    }
  }
}

// Export database helpers for routes
export function getDB(c: any) {
  return c.env?.DB || mockD1
}