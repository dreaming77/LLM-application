<template>
  <div class="history-page">
    <div class="container">
      <div class="header">
        <p>查看之前的医疗咨询记录</p>
      </div>

      <div class="history-list">
        <div 
          v-for="(item, index) in history" 
          :key="index"
          class="history-item"
        >
          <div class="history-header">
            <span class="session-id">会话: {{ item.session_id }}</span>
            <span class="timestamp">{{ formatDate(item.timestamp) }}</span>
          </div>
          <div class="query-preview">
            {{ item.user_query }}
          </div>
          <div class="response-preview">
            {{ truncateText(item.response, 150) }}
          </div>
          <div class="history-metrics">
            <span class="confidence" :class="getConfidenceClass(item.response_metadata.confidence_score)">
              置信度: {{ (item.response_metadata.confidence_score * 100).toFixed(1) }}%
            </span>
            <span class="duration">耗时: {{ item.duration_seconds.toFixed(2) }}s</span>
          </div>
        </div>

        <div v-if="history.length === 0" class="empty-state">
          <p>暂无咨询历史</p>
          <router-link to="/query" class="start-query-link">
            开始第一次咨询
          </router-link>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useAppStore } from '../stores'

const store = useAppStore()
const history = computed(() => store.conversationHistory)

const formatDate = (timestamp) => {
  return new Date(timestamp * 1000).toLocaleString('zh-CN')
}

const truncateText = (text, length) => {
  return text.length > length ? text.substring(0, length) + '...' : text
}

const getConfidenceClass = (score) => {
  if (score >= 0.8) return 'high'
  if (score >= 0.6) return 'medium'
  return 'low'
}
</script>

<style scoped>
.history-page {
  padding: 2rem 0;
  flex: 1;
}

.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 0 1rem;
}

.header {
  text-align: center;
  margin-bottom: 2rem;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.history-item {
  background: white;
  border-radius: 8px;
  padding: 1.5rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  border-left: 4px solid #3498db;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.session-id {
  font-weight: 600;
  color: #2c3e50;
}

.timestamp {
  color: #7f8c8d;
  font-size: 0.9rem;
}

.query-preview {
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: #2c3e50;
}

.response-preview {
  color: #7f8c8d;
  line-height: 1.5;
  margin-bottom: 1rem;
}

.history-metrics {
  display: flex;
  gap: 1rem;
  font-size: 0.9rem;
}

.confidence.high { color: #27ae60; }
.confidence.medium { color: #f39c12; }
.confidence.low { color: #e74c3c; }

.duration {
  color: #7f8c8d;
}

.empty-state {
  text-align: center;
  padding: 3rem;
  color: #7f8c8d;
}

.start-query-link {
  color: #3498db;
  text-decoration: none;
  margin-top: 1rem;
  display: inline-block;
}

.start-query-link:hover {
  text-decoration: underline;
}
</style>
