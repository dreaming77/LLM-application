import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8055',
        changeOrigin: true
      }
    }
  },
  optimizeDeps: {
    include: ['vue', 'element-plus', '@element-plus/icons-.eslintrc.cjsvue', 'axios']
  }
})
