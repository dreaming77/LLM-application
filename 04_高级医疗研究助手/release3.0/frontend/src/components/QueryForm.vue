<!-- frontend/src/components/QueryForm.vue -->
<template>
  <div class="query-form">
    <form @submit.prevent="handleSubmit" class="form">
      <div class="form-group">
        <label for="medical-query">医疗问题描述</label>
        <textarea
          id="medical-query"
          v-model="query"
          placeholder="请详细描述您的医疗问题，例如症状、病史、检查结果等..."
          rows="4"
          required
          :disabled="isLoading"
        ></textarea>
      </div>

      <div class="form-group">
        <label>上下文信息（可选）</label>
        <div class="context-fields">
          <input
            v-model="userContext.age"
            type="number"
            placeholder="年龄"
            class="context-input"
          />
          <select v-model="userContext.gender" class="context-input">
            <option value="">选择性别</option>
            <option value="male">男性</option>
            <option value="female">女性</option>
          </select>
          <input
            v-model="userContext.condition"
            type="text"
            placeholder="已知疾病"
            class="context-input"
          />
        </div>
      </div>

      <div class="form-actions">
        <button 
          type="submit" 
          class="submit-btn"
          :disabled="!query.trim() || isLoading"
        >
          <span v-if="isLoading">处理中...</span>
          <span v-else>提交咨询</span>
        </button>
        
        <button 
          v-if="hasActiveSession"
          type="button"
          class="clear-btn"
          @click="handleClear"
        >
          新会话
        </button>
      </div>
    </form>

    <!-- 继续对话输入 -->
    <div v-if="hasActiveSession" class="continue-section">
      <div class="continue-form">
        <input
          v-model="continueQuery"
          placeholder="继续提问..."
          class="continue-input"
          @keypress.enter="handleContinue"
        />
        <button 
          @click="handleContinue"
          class="continue-btn"
          :disabled="!continueQuery.trim()"
        >
          发送
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useAppStore } from '../stores'

const emit = defineEmits(['submit-query', 'continue-conversation'])

const store = useAppStore()
const query = ref('')
const continueQuery = ref('')
const userContext = ref({
  age: '',
  gender: '',
  condition: ''
})

const isLoading = computed(() => store.isLoading)
const hasActiveSession = computed(() => store.hasActiveSession)

const handleSubmit = () => {
  const queryData = {
    query: query.value,
    user_context: Object.fromEntries(
      Object.entries(userContext.value).filter(([_, value]) => value)
    )
  }

  emit('submit-query', queryData)
  
  // 清空表单
  query.value = ''
  userContext.value = { age: '', gender: '', condition: '' }
}

const handleContinue = () => {
  if (continueQuery.value.trim()) {
    emit('continue-conversation', continueQuery.value)
    continueQuery.value = ''
  }
}

const handleClear = () => {
  store.clearHistory()
}
</script>

<style scoped>
.query-form {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.form-group {
  margin-bottom: 1.5rem;
}

label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: #2c3e50;
}

textarea {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e9ecef;
  border-radius: 8px;
  font-size: 1rem;
  resize: vertical;
  transition: border-color 0.3s ease;
}

textarea:focus {
  outline: none;
  border-color: #3498db;
}

.context-fields {
  display: grid;
  grid-template-columns: 1fr 1fr 2fr;
  gap: 1rem;
}

.context-input {
  padding: 0.75rem;
  border: 2px solid #e9ecef;
  border-radius: 6px;
  font-size: 0.9rem;
}

.context-input:focus {
  outline: none;
  border-color: #3498db;
}

.form-actions {
  display: flex;
  gap: 1rem;
}

.submit-btn,
.clear-btn {
  padding: 1rem 2rem;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.3s ease;
}

.submit-btn {
  background: #3498db;
  color: white;
  flex: 1;
}

.submit-btn:hover:not(:disabled) {
  background: #2980b9;
}

.submit-btn:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
}

.clear-btn {
  background: #95a5a6;
  color: white;
}

.clear-btn:hover {
  background: #7f8c8d;
}

.continue-section {
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 1px solid #e9ecef;
}

.continue-form {
  display: flex;
  gap: 1rem;
}

.continue-input {
  flex: 1;
  padding: 1rem;
  border: 2px solid #e9ecef;
  border-radius: 8px;
  font-size: 1rem;
}

.continue-input:focus {
  outline: none;
  border-color: #3498db;
}

.continue-btn {
  padding: 1rem 2rem;
  background: #2ecc71;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
}

.continue-btn:hover:not(:disabled) {
  background: #27ae60;
}

.continue-btn:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
}
</style>
