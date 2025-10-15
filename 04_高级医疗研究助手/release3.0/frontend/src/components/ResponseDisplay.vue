<!-- frontend/src/components/ResponseDisplay.vue -->
<template>
  <div class="response-display">
    <div v-if="responses.length === 0" class="empty-state">
      <div class="empty-icon">💬</div>
      <h3>暂无对话记录</h3>
      <p>请输入您的医疗问题开始咨询</p>
    </div>

    <div v-else class="conversation">
      <div 
        v-for="(message, index) in responses"
        :key="index"
        :class="['message', message.type]"
      >
        <div class="message-header">
          <span class="sender">{{ message.type === 'user' ? '您' : '医疗助手' }}</span>
          <span class="timestamp">{{ formatTime(message.timestamp) }}</span>
        </div>
        
        <div class="message-content">
          {{ message.content }}
        </div>

        <div v-if="message.type === 'assistant' && message.metadata" class="message-metadata">
          <div class="metadata-item">
            <span class="label">置信度:</span>
            <span 
              class="confidence" 
              :class="getConfidenceClass(message.metadata.confidence_score)"
            >
              {{ (message.metadata.confidence_score * 100).toFixed(1) }}%
            </span>
          </div>
          
          <div v-if="message.metadata.citations_count > 0" class="metadata-item">
            <span class="label">参考来源:</span>
            <span class="citations">{{ message.metadata.citations_count }} 个</span>
          </div>
        </div>

        <div v-if="message.type === 'assistant' && message.sessionId" class="session-info">
          会话ID: {{ message.sessionId }}
        </div>
      </div>
    </div>

    <div v-if="isLoading" class="loading-indicator">
      <div class="loading-spinner"></div>
      <span>AI助手正在思考中...</span>
    </div>
  </div>
</template>

<script setup>
defineProps({
  responses: {
    type: Array,
    default: () => []
  },
  isLoading: {
    type: Boolean,
    default: false
  },
  currentSession: {
    type: String,
    default: null
  }
})

const formatTime = (timestamp) => {
  return new Date(timestamp).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

const getConfidenceClass = (score) => {
  if (score >= 0.8) return 'high'
  if (score >= 0.6) return 'medium'
  return 'low'
}
</script>

<style scoped>
.response-display {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  min-height: 400px;
  max-height: 600px;
  overflow-y: auto;
}

.empty-state {
  text-align: center;
  padding: 3rem;
  color: #7f8c8d;
}

.empty-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.conversation {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.message {
  padding: 1.5rem;
  border-radius: 12px;
  max-width: 80%;
}

.message.user {
  align-self: flex-end;
  background: #3498db;
  color: white;
}

.message.assistant {
  align-self: flex-start;
  background: #f8f9fa;
  border: 1px solid #e9ecef;
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
}

.message.user .message-header {
  color: rgba(255, 255, 255, 0.9);
}

.message.assistant .message-header {
  color: #7f8c8d;
}

.sender {
  font-weight: 600;
}

.message-content {
  line-height: 1.6;
  white-space: pre-wrap;
}

.message-metadata {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  display: flex;
  gap: 1rem;
  font-size: 0.9rem;
}

.message.user .message-metadata {
  border-top-color: rgba(255, 255, 255, 0.3);
}

.metadata-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.label {
  color: #7f8c8d;
}

.confidence.high { color: #27ae60; font-weight: 600; }
.confidence.medium { color: #f39c12; font-weight: 600; }
.confidence.low { color: #e74c3c; font-weight: 600; }

.citations {
  color: #3498db;
  font-weight: 600;
}

.session-info {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #95a5a6;
  font-family: monospace;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 8px;
  color: #856404;
}

.loading-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #f3f3f3;
  border-top: 2px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
