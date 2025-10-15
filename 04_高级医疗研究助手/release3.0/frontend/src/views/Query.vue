<!-- frontend/src/views/Query.vue -->
<template>
  <div class="chat-page">
    <!-- 顶部导航栏 -->
    <div class="chat-header">
      <div class="header-content">
        <div class="header-left">
          <h1>医疗研究助手</h1>
          <span class="subtitle">专业医疗问答与咨询</span>
        </div>
        <div class="header-right">
          <div class="session-info" v-if="hasActiveSession">
            <span class="session-badge">会话进行中</span>
            <span class="session-id">ID: {{ currentSessionId }}</span>
          </div>
          <button @click="handleClear" class="new-chat-btn" :disabled="isLoading">
            新对话
          </button>
        </div>
      </div>
    </div>

    <!-- 主要对话区域 -->
    <div class="chat-container" ref="chatContainer">
      <!-- 欢迎消息 -->
      <div v-if="responses.length === 0 && !isLoading" class="welcome-section">
        <div class="welcome-content">
          <div class="welcome-icon">🔥</div>
          <h2>欢迎使用医疗研究助手</h2>
          <p>我是您的专业医疗AI助手，基于权威医疗知识库为您提供专业的医疗咨询和建议</p>
          <div class="suggestions">
            <div class="suggestion-grid">
              <button 
                v-for="suggestion in quickSuggestions" 
                :key="suggestion.id"
                @click="useSuggestion(suggestion.text)"
                class="suggestion-btn"
                :disabled="isLoading"
              >
                {{ suggestion.text }}
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- 对话消息 -->
      <div v-else class="messages-container">
        <div 
          v-for="(response, index) in responses"
          :key="index"
          class="message-group"
        >
          <!-- 用户消息 -->
          <div class="message user-message">
            <div class="message-avatar">
              <div class="avatar user-avatar">👤</div>
            </div>
            <div class="message-content">
              <div class="message-text">
                {{ response.question }}
              </div>
              <div class="message-time">
                {{ formatTime(response.timestamp) }}
              </div>
            </div>
          </div>

          <!-- AI助手消息 -->
          <div class="message assistant-message">
            <div class="message-avatar">
              <div class="avatar assistant-avatar">🏥</div>
            </div>
            <div class="message-content">
              <div class="message-text" v-html="formatResponse(response.answer)"></div>
              <div class="message-metadata">
                <span class="confidence" :class="getConfidenceClass(response.metadata?.confidence_score)">
                  置信度: {{ ((response.metadata?.confidence_score || 0.5) * 100).toFixed(1) }}%
                </span>
                <span class="sources" v-if="response.metadata?.citations_count > 0">
                  参考来源: {{ response.metadata.citations_count }} 个
                </span>
                <span class="duration">
                  响应时间: {{ (response.metadata?.duration_seconds || 0).toFixed(2) }}秒
                </span>
              </div>
              <div class="message-time">
                {{ formatTime(response.timestamp) }}
              </div>
            </div>
          </div>
        </div>

        <!-- 加载指示器 -->
        <div v-if="isLoading" class="message assistant-message loading-message">
          <div class="message-avatar">
            <div class="avatar assistant-avatar">🏥</div>
          </div>
          <div class="message-content">
            <div class="loading-indicator">
              <div class="typing-animation">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <span class="loading-text">AI助手正在思考中...</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 错误消息 -->
      <div v-if="error" class="error-banner">
        <div class="error-content">
          <span class="error-icon">⚠️</span>
          <span class="error-text">{{ error }}</span>
          <button @click="clearError" class="error-close">×</button>
        </div>
      </div>
    </div>

    <!-- 悬浮底部输入区域 -->
    <div class="floating-input-container" :class="{ 'with-context': showContextPanel }">
      <div class="floating-input-wrapper">
        <!-- 上下文信息快速设置 -->
        <div v-if="showContextPanel" class="context-panel">
          <div class="context-header">
            <span>上下文信息</span>
            <button @click="toggleContextPanel" class="close-context">×</button>
          </div>
          <div class="context-fields">
            <div class="context-field">
              <label>年龄</label>
              <input
                v-model="userContext.age"
                type="number"
                placeholder="请输入年龄"
                class="context-input"
              />
            </div>
            <div class="context-field">
              <label>性别</label>
              <select v-model="userContext.gender" class="context-input">
                <option value="">请选择</option>
                <option value="male">男性</option>
                <option value="female">女性</option>
                <option value="other">其他</option>
              </select>
            </div>
            <div class="context-field">
              <label>已知疾病</label>
              <input
                v-model="userContext.condition"
                type="text"
                placeholder="如高血压、糖尿病等"
                class="context-input"
              />
            </div>
          </div>
        </div>

        <div class="input-main">
          <!-- 上下文信息按钮 -->
          <button 
            @click="toggleContextPanel" 
            class="context-btn"
            :class="{ active: showContextPanel }"
            title="设置上下文信息"
          >
            🏷️
          </button>

          <!-- 文本输入框 -->
          <div class="text-input-wrapper">
            <textarea
              v-model="currentQuery"
              ref="textInput"
              placeholder="请输入您的医疗问题..."
              rows="1"
              @keydown="handleKeydown"
              @input="autoResize"
              :disabled="isLoading"
            ></textarea>
            
            <!-- 快捷操作按钮 -->
            <div v-if="!currentQuery" class="quick-actions">
              <button 
                v-for="action in quickActions" 
                :key="action.id"
                @click="performQuickAction(action)"
                class="quick-action-btn"
                :title="action.title"
              >
                {{ action.icon }}
              </button>
            </div>
          </div>

          <!-- 发送按钮 -->
          <button 
            @click="submitQuery"
            class="send-btn"
            :disabled="!currentQuery.trim() || isLoading"
            :class="{ loading: isLoading }"
          >
            <span v-if="isLoading" class="send-loading">
              <div class="spinner"></div>
            </span>
            <span v-else class="send-icon">➤</span>
          </button>
        </div>

        <!-- 输入提示 -->
        <div class="input-footer">
          <span class="tip-text">
            💡 按 Enter 发送，Shift + Enter 换行
          </span>
          <span class="model-info">
            基于 Qwen2.5-7B 医疗模型
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { useAppStore } from '../stores'
import apiService from '../services/api'

const store = useAppStore()
const chatContainer = ref(null)
const textInput = ref(null)

const currentQuery = ref('')
const userContext = ref({
  age: '',
  gender: '',
  condition: ''
})
const responses = ref([])
const error = ref('')
const isLoading = ref(false)
const showContextPanel = ref(false)

const hasActiveSession = computed(() => store.hasActiveSession)
const currentSessionId = computed(() => store.currentSession)

// 快速建议问题
const quickSuggestions = ref([
  { id: 1, text: '感冒了应该吃什么药？' },
  { id: 2, text: '高血压患者饮食要注意什么？' },
  { id: 3, text: '糖尿病早期有什么症状？' },
  { id: 4, text: '如何预防心脏病？' },
  { id: 5, text: '癌症的早期筛查方法有哪些？' },
  { id: 6, text: '心理健康问题如何识别？' }
])

// 快捷操作
const quickActions = ref([
  { id: 1, icon: '📋', title: '整理症状', action: () => '请帮我整理一下我的症状描述' },
  { id: 2, icon: '💊', title: '药物咨询', action: () => '关于这种药物，请给我一些信息' },
  { id: 3, icon: '🩺', title: '检查解读', action: () => '请帮我解读这份检查报告' },
  { id: 4, icon: '🥗', title: '饮食建议', action: () => '针对我的情况，饮食上有什么建议？' }
])

// 自动调整文本框高度
const autoResize = () => {
  nextTick(() => {
    if (textInput.value) {
      textInput.value.style.height = 'auto'
      textInput.value.style.height = Math.min(textInput.value.scrollHeight, 120) + 'px'
    }
  })
}

// 键盘事件处理
const handleKeydown = (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submitQuery()
  }
}

// 使用快速建议
const useSuggestion = (text) => {
  currentQuery.value = text
  submitQuery()
}

// 执行快捷操作
const performQuickAction = (action) => {
  currentQuery.value = action.action()
  textInput.value?.focus()
}

// 切换上下文面板
const toggleContextPanel = () => {
  showContextPanel.value = !showContextPanel.value
}

// 提交查询
const submitQuery = async () => {
  const queryText = currentQuery.value.trim()
  if (!queryText || isLoading.value) return

  isLoading.value = true
  error.value = ''

  try {
    // 准备查询数据
    const queryData = {
      query: queryText,
      user_context: Object.fromEntries(
        Object.entries(userContext.value).filter(([_, value]) => value)
      ),
      session_id: currentSessionId.value
    }

    // 调用后端API
    const result = await apiService.processQuery(queryData)
    
    if (result.error) {
      throw new Error(result.message || '处理查询时发生错误')
    }

    if (!result.success) {
      throw new Error(result.error || 'API返回失败状态')
    }

    // 保存响应
    responses.value.push({
      question: queryText,
      answer: result.response,
      metadata: result.response_metadata,
      sessionId: result.session_id,
      timestamp: new Date()
    })

    // 更新会话状态
    store.setSession(result.session_id)
    store.addToHistory(result)
    
    // 清空当前查询
    currentQuery.value = ''
    
    // 重置文本框高度
    if (textInput.value) {
      textInput.value.style.height = 'auto'
    }
    
    // 滚动到底部
    scrollToBottom()
    
  } catch (err) {
    console.error('查询失败:', err)
    error.value = err.message || '处理您的查询时出现了问题，请稍后重试。'
  } finally {
    isLoading.value = false
  }
}

// 格式化响应文本
const formatResponse = (text) => {
  if (!text) return ''
  
  // 简单的格式化：将数字列表转换为带样式的列表
  return text
    .replace(/\n/g, '<br>')
    .replace(/(\d+[\.\)])/g, '<strong>$1</strong>')
}

// 格式化时间
const formatTime = (timestamp) => {
  return new Date(timestamp).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 获取置信度样式类
const getConfidenceClass = (score) => {
  if (!score) return 'medium'
  if (score >= 0.8) return 'high'
  if (score >= 0.6) return 'medium'
  return 'low'
}

// 滚动到底部
const scrollToBottom = () => {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}

// 清除错误
const clearError = () => {
  error.value = ''
}

// 开始新对话
const handleClear = () => {
  if (isLoading.value) return
  
  store.clearHistory()
  responses.value = []
  currentQuery.value = ''
  userContext.value = { age: '', gender: '', condition: '' }
  showContextPanel.value = false
  
  // 重置文本框高度
  if (textInput.value) {
    textInput.value.style.height = 'auto'
  }
}

// 监听响应变化，自动滚动
watch(responses, () => {
  scrollToBottom()
}, { deep: true })

// 组件挂载时聚焦输入框
onMounted(() => {
  textInput.value?.focus()
})
</script>

<style scoped>
.chat-page {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #f8f9fa;
  position: relative;
}

/* 头部样式 */
.chat-header {
  background: white;
  border-bottom: 1px solid #e9ecef;
  padding: 1rem 0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  flex-shrink: 0;
  z-index: 100;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-left h1 {
  font-size: 1.5rem;
  color: #2c3e50;
  margin: 0;
}

.subtitle {
  color: #7f8c8d;
  font-size: 0.9rem;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.session-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8rem;
}

.session-badge {
  background: #e3f2fd;
  color: #1976d2;
  padding: 0.25rem 0.5rem;
  border-radius: 12px;
  font-weight: 500;
}

.session-id {
  color: #7f8c8d;
  font-family: monospace;
}

.new-chat-btn {
  background: #3498db;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: background 0.3s ease;
}

.new-chat-btn:hover:not(:disabled) {
  background: #2980b9;
}

.new-chat-btn:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
}

/* 对话容器 - 现在占据整个屏幕减去头部高度 */
.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background: #f8f9fa;
  /* 为悬浮输入框留出空间 */
  padding-bottom: 120px;
}

/* 欢迎区域 */
.welcome-section {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  max-width: 800px;
  margin: 0 auto;
}

.welcome-content {
  text-align: center;
  padding: 2rem;
  margin-top: -15vh;
}

.welcome-icon {
  font-size: 6rem;
  margin-bottom: 1rem;
}

.welcome-content h2 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.welcome-content p {
  color: #7f8c8d;
  font-size: 1.1rem;
  margin-bottom: 2rem;
}

.suggestion-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  max-width: 600px;
  margin: 0 auto;
}

.suggestion-btn {
  background: white;
  border: 1px solid #e9ecef;
  border-radius: 12px;
  padding: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  text-align: left;
  font-size: 0.9rem;
}

.suggestion-btn:hover {
  border-color: #3498db;
  box-shadow: 0 2px 8px rgba(52, 152, 219, 0.2);
  transform: translateY(-1px);
}

/* 消息容器 */
.messages-container {
  max-width: 800px;
  margin: 0 auto;
  padding-bottom: 2rem;
}

.message-group {
  margin-bottom: 2rem;
}

.message {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.user-message {
  flex-direction: row-reverse;
}

.message-avatar {
  flex-shrink: 0;
}

.avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
}

.user-avatar {
  background: #3498db;
}

.assistant-avatar {
  background: #2ecc71;
}

.message-content {
  flex: 1;
  max-width: calc(100% - 60px);
}

.user-message .message-content {
  text-align: right;
}

.message-text {
  background: white;
  padding: 1rem 1.5rem;
  border-radius: 18px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  line-height: 1.5;
  word-wrap: break-word;
}

.user-message .message-text {
  background: #3498db;
  color: white;
  border-bottom-right-radius: 4px;
}

.assistant-message .message-text {
  background: white;
  color: #2c3e50;
  border-bottom-left-radius: 4px;
}

.message-metadata {
  display: flex;
  gap: 1rem;
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #7f8c8d;
  flex-wrap: wrap;
}

.confidence.high { color: #27ae60; }
.confidence.medium { color: #f39c12; }
.confidence.low { color: #e74c3c; }

.user-message .message-metadata {
  justify-content: flex-end;
}

.message-time {
  font-size: 0.75rem;
  color: #95a5a6;
  margin-top: 0.25rem;
}

/* 加载消息 */
.loading-message .message-text {
  background: transparent;
  box-shadow: none;
  padding: 0;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.5rem;
  background: white;
  border-radius: 18px;
  border-bottom-left-radius: 4px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.typing-animation {
  display: flex;
  gap: 4px;
}

.typing-animation span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #7f8c8d;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-animation span:nth-child(1) { animation-delay: -0.32s; }
.typing-animation span:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
  0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
  40% { transform: scale(1); opacity: 1; }
}

.loading-text {
  color: #7f8c8d;
  font-size: 0.9rem;
}

/* 错误横幅 */
.error-banner {
  position: fixed;
  top: 80px;
  left: 50%;
  transform: translateX(-50%);
  background: #f8d7da;
  border: 1px solid #f5c6cb;
  border-radius: 8px;
  padding: 0.75rem 1rem;
  max-width: 600px;
  width: 90%;
  z-index: 1000;
  animation: slideDown 0.3s ease;
}

@keyframes slideDown {
  from { transform: translateX(-50%) translateY(-20px); opacity: 0; }
  to { transform: translateX(-50%) translateY(0); opacity: 1; }
}

.error-content {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.error-text {
  flex: 1;
  color: #721c24;
  font-size: 0.9rem;
}

.error-close {
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  color: #721c24;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 悬浮输入容器 */
.floating-input-container {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-top: 1px solid rgba(233, 236, 239, 0.8);
  padding: 1rem;
  z-index: 100;
  transition: all 0.3s ease;
}

.floating-input-container.with-context {
  background: white;
  backdrop-filter: none;
}

.floating-input-wrapper {
  max-width: 800px;
  margin: 0 auto;
  position: relative;
}

/* 上下文面板 */
.context-panel {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 1rem;
  animation: slideUp 0.3s ease;
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.context-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  font-weight: 600;
  color: #2c3e50;
}

.close-context {
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  color: #7f8c8d;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.context-fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
}

.context-field {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.context-field label {
  font-size: 0.8rem;
  color: #7f8c8d;
  font-weight: 500;
}

.context-input {
  padding: 0.5rem;
  border: 1px solid #e9ecef;
  border-radius: 6px;
  font-size: 0.9rem;
}

.context-input:focus {
  outline: none;
  border-color: #3498db;
}

/* 主输入区域 */
.input-main {
  display: flex;
  align-items: flex-end;
  gap: 0.5rem;
  background: white;
  border: 2px solid #e9ecef;
  border-radius: 24px;
  padding: 0.75rem;
  transition: border-color 0.3s ease;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.input-main:focus-within {
  border-color: #3498db;
  box-shadow: 0 2px 15px rgba(52, 152, 219, 0.2);
}

.context-btn {
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 50%;
  transition: background 0.3s ease;
  flex-shrink: 0;
}

.context-btn:hover {
  background: #f8f9fa;
}

.context-btn.active {
  background: #e3f2fd;
  color: #1976d2;
}

.text-input-wrapper {
  flex: 1;
  position: relative;
  min-height: 40px;
  display: flex;
  align-items: center;
}

.text-input-wrapper textarea {
  width: 100%;
  border: none;
  outline: none;
  resize: none;
  font-size: 1rem;
  line-height: 1.5;
  max-height: 120px;
  background: transparent;
  font-family: inherit;
}

.quick-actions {
  position: absolute;
  right: 0.5rem;
  display: flex;
  gap: 0.25rem;
}

.quick-action-btn {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 6px;
  padding: 0.25rem;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.3s ease;
}

.quick-action-btn:hover {
  background: #e9ecef;
  transform: scale(1.1);
}

.send-btn {
  background: #3498db;
  color: white;
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: all 0.3s ease;
  box-shadow: 0 2px 5px rgba(52, 152, 219, 0.3);
}

.send-btn:hover:not(:disabled) {
  background: #2980b9;
  transform: scale(1.05);
  box-shadow: 0 3px 8px rgba(52, 152, 219, 0.4);
}

.send-btn:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.send-btn.loading {
  background: #95a5a6;
}

.send-loading {
  display: flex;
  align-items: center;
  justify-content: center;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid transparent;
  border-top: 2px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.send-icon {
  font-size: 1.2rem;
}

/* 输入底部 */
.input-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 0.5rem;
  font-size: 0.75rem;
  color: #7f8c8d;
}

.tip-text, .model-info {
  opacity: 0.7;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .chat-container {
    padding: 0.5rem;
    padding-bottom: 100px;
  }
  
  .header-content {
    flex-direction: column;
    gap: 0.5rem;
    text-align: center;
  }
  
  .suggestion-grid {
    grid-template-columns: 1fr;
  }
  
  .context-fields {
    grid-template-columns: 1fr;
  }
  
  .input-main {
    padding: 0.5rem;
  }
  
  .quick-actions {
    position: static;
    margin-top: 0.5rem;
    justify-content: center;
  }
  
  .input-footer {
    flex-direction: column;
    gap: 0.25rem;
    text-align: center;
  }
  
  .error-banner {
    top: 120px;
    width: 95%;
  }
  
  .floating-input-container {
    padding: 0.75rem;
  }
}

@media (max-width: 480px) {
  .message {
    gap: 0.5rem;
  }
  
  .avatar {
    width: 32px;
    height: 32px;
    font-size: 1rem;
  }
  
  .message-content {
    max-width: calc(100% - 48px);
  }
  
  .message-text {
    padding: 0.75rem 1rem;
    font-size: 0.9rem;
  }
  
  .message-metadata {
    font-size: 0.7rem;
  }
  
  .floating-input-container {
    padding: 0.5rem;
  }
}
</style>
