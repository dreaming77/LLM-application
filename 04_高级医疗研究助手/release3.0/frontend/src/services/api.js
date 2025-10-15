// frontend/src/services/api.js

class ApiService {
  constructor() {
    // 不要在构造函数中立即使用store，延迟初始化
    this.baseURL = 'http://localhost:8888'
    this.apiKey = 'medical_research_demo_key'
  }

  // 获取store的辅助方法
  _getStore() {
    try {
      const { useAppStore } = require('../stores/index.js')
      return useAppStore()
    } catch (error) {
      console.warn('Store not available yet, using fallback values')
      return {
        baseURL: this.baseURL,
        apiKey: this.apiKey,
        setLoading: () => {},
        isLoading: false
      }
    }
  }

  async request(endpoint, options = {}) {
    const store = this._getStore()
    const url = `${store.baseURL || this.baseURL}${endpoint}`
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': store.apiKey || this.apiKey,
        ...options.headers
      },
      ...options
    }
  
    try {
      if (store.setLoading) {
        store.setLoading(true)
      }
      
      const response = await fetch(url, config)
      
      if (!response.ok) {
        // 如果后端返回错误状态，尝试获取错误信息
        let errorMessage = `HTTP error! status: ${response.status}`
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorData.error || errorMessage
        } catch (e) {
          // 如果无法解析JSON，使用默认错误信息
        }
        throw new Error(errorMessage)
      }
      
      return await response.json()
    } catch (error) {
      console.error('API request failed:', error)
      
      // 返回一个结构化的错误响应，而不是抛出异常
      return {
        error: true,
        message: error.message,
        endpoint: endpoint
      }
    } finally {
      if (store.setLoading) {
        store.setLoading(false)
      }
    }
  }

  // 健康检查
  async healthCheck() {
    return this.request('/health')
  }

  async detailedHealthCheck() {
    return this.request('/api/health/detail')
  }

  async modelHealthCheck() {
    return this.request('/api/health/models/detailed')
  }

  // 处理医疗查询
  async processQuery(queryData) {
    return this.request('/api/query', {
      method: 'POST',
      body: JSON.stringify(queryData)
    })
  }

  // 继续对话
  async continueConversation(sessionId, queryData) {
    return this.request(`/api/conversation/${sessionId}/continue`, {
      method: 'POST',
      body: JSON.stringify(queryData)
    })
  }

  // 获取会话状态
  async getSessionStatus(sessionId) {
    return this.request(`/api/session/${sessionId}`)
  }

  // 中断会话
  async interruptSession(sessionId) {
    return this.request(`/api/session/${sessionId}/interrupt`, {
      method: 'POST'
    })
  }

  // 获取系统信息
  async getSystemInfo() {
    return this.request('/api/system/info')
  }

  // 获取默认配置
  async getDefaultConfig() {
    return this.request('/api/config/default')
  }

  // 调试模型状态
  async debugModels() {
    return this.request('/api/debug/models')
  }
}

// 创建全局实例
const apiService = new ApiService()

export default apiService
