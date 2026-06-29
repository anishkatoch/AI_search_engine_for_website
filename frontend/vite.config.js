import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During `npm run dev`, the React dev server proxies /api to the FastAPI
// backend so the frontend and API behave as one origin (just like in prod,
// where FastAPI serves the built files directly).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8005',
    },
  },
  build: {
    outDir: 'dist',
  },
})
