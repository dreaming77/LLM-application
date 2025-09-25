import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1'

// 创建 axios 实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30秒超时
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log(`发起 API 请求: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API 响应成功: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('API 请求失败:', error)

    if (error.response) {
      // 服务器返回错误状态码
      console.error('错误详情:', error.response.data)
      error.message = error.response.data.detail || `服务器错误: ${error.response.status}`
    } else if (error.request) {
      // 请求已发出但没有收到响应
      error.message = '网络错误，请检查服务器是否运行'
    } else {
      // 其他错误
      error.message = error.message || '未知错误'
    }

    return Promise.reject(error)
  }
)

export const researchService = {
  async conductResearch(query) {
    try {
      const response = await apiClient.post('/research', {
        query,
        max_documents: 10
      })
      return response.data
    } catch (error) {
      console.error('研究请求失败:', error)
      throw error // 直接抛出错误，由调用者处理
    }
  },

  async checkHealth() {
    try {
      const response = await apiClient.get('/health')
      return response.data
    } catch (error) {
      console.error('健康检查失败:', error)
      throw error
    }
  },

  async startResearch(query) {
    try {
      const response = await apiClient.post('/research/start', {
        query,
        max_documents: 10
      })
      return response.data
    } catch (error) {
      console.error('启动研究请求失败:', error)
      throw new Error(error.response?.data?.detail || '启动研究请求失败')
    }
  },

  async getResearchProgress(researchId) {
    try {
      const response = await apiClient.get(`/research/progress/${researchId}`)
      return response.data
    } catch (error) {
      console.error('获取研究进度失败:', error)
      throw new Error(error.response?.data?.detail || '获取研究进度失败')
    }
  }

}
