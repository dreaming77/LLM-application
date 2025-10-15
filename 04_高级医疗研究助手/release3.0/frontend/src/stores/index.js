
import { defineStore } from 'pinia'

export const useAppStore = defineStore('app', {
  state: () => ({
    apiKey: 'medical_research_demo_key',
    baseURL: 'http://localhost:8888',
    currentSession: null,
    conversationHistory: [],
    systemStatus: {},
    isLoading: false
  }),
  actions: {
    setSession(sessionId) {
      this.currentSession = sessionId
    },
    addToHistory(message) {
      this.conversationHistory.push(message)
    },
    clearHistory() {
      this.conversationHistory = []
      this.currentSession = null
    },
    setLoading(loading) {
      this.isLoading = loading
    },
    updateSystemStatus(status) {
      this.systemStatus = status
    }
  },
  getters: {
    hasActiveSession: (state) => !!state.currentSession
  }
})
