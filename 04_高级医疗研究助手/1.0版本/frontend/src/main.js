import { createApp } from 'vue'
import App from './App.vue'
import axios from 'axios'

// 设置axios默认配置
axios.defaults.timeout = 30000

const app = createApp(App)
app.mount('#app')
