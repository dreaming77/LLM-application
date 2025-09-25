<template>
  <div id="app">
    <header class="header">
      <h1>高级医疗研究助手&报告生成器</h1>
      <p>基于LangGraph + Qwen-7B + BGE + Milvus + Huatuo-26M数据集</p>
    </header>

    <main class="main-content">
      <div class="input-section">
        <h2>输入医疗问题</h2>
        <textarea
          v-model="userQuery"
          placeholder="请输入您想研究的医疗问题，例如：糖尿病的病因、症状和治疗方法..."
          rows="4"
        ></textarea>
        <button
          @click="startResearch"
          :disabled="isProcessing"
          class="submit-btn"
        >
          {{ isProcessing ? '处理中...' : '开始研究' }}
        </button>
      </div>

      <div v-if="currentTaskId" class="progress-section">
        <h3>研究进度</h3>
        <div class="progress-bar">
          <div
            class="progress-fill"
            :style="{ width: progress + '%' }"
          ></div>
        </div>
        <p>{{ progress }}% - {{ statusText }}</p>
      </div>

      <div v-if="error" class="error-section">
        <h3>错误信息</h3>
        <p class="error-text">{{ error }}</p>

        <div v-if="currentTaskId" class="task-info">
          <p><strong>任务ID:</strong> {{ currentTaskId }}</p>
          <p><strong>状态:</strong> {{ status }}</p>
          <p><strong>进度:</strong> {{ progress }}%</p>
        </div>

        <button @click="showDebugInfo = !showDebugInfo" class="debug-btn">
          {{ showDebugInfo ? '隐藏' : '显示' }}调试信息
        </button>

        <div v-if="showDebugInfo" class="debug-info">
          <h4>调试信息:</h4>
          <pre>{{ debugInfo }}</pre>
        </div>

        <button @click="retryResearch" class="retry-btn" v-if="status === 'error'">
          重试
        </button>
      </div>

      <div v-if="finalReport" class="result-section">
        <h2>研究报告</h2>
        <div class="report-content">
          <pre>{{ finalReport }}</pre>
        </div>

        <div v-if="searchQueries.length > 0" class="search-queries">
          <h3>搜索子问题</h3>
          <ul>
            <li v-for="(query, index) in searchQueries" :key="index">
              {{ query }}
            </li>
          </ul>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import axios from 'axios';

// 使用相对路径，让代理处理
const API_BASE_URL = '/api';

export default {
  name: 'App',
  data() {
    return {
      userQuery: '',
      currentTaskId: null,
      progress: 0,
      status: '',
      finalReport: '',
      searchQueries: [],
      error: null,
      pollingInterval: null,
      showDebugInfo: false,
      debugInfo: ''
    };
  },
  computed: {
    isProcessing() {
      return this.status === 'pending' || this.status === 'processing';
    },
    statusText() {
      const statusMap = {
        'pending': '等待中',
        'processing': '处理中',
        'completed': '完成',
        'error': '错误'
      };
      return statusMap[this.status] || this.status;
    }
  },
  methods: {
    async startResearch() {
      if (!this.userQuery.trim()) {
        alert('请输入医疗问题');
        return;
      }

      try {
        this.resetState();

        console.log('发送研究请求:', this.userQuery);
        const response = await axios.post(`${API_BASE_URL}/research`, {
          query: this.userQuery
        });

        console.log('收到响应:', response.data);
        this.currentTaskId = response.data.task_id;
        this.status = response.data.status;

        this.startPolling();
      } catch (error) {
        console.error('开始研究错误详情:', error);
        this.debugInfo = JSON.stringify({
          error: error.message,
          response: error.response ? error.response.data : '无响应数据',
          config: error.config
        }, null, 2);

        this.error = `无法开始研究任务: ${error.message}`;
      }
    },

  async startPolling() {
    // 清除之前的轮询
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
    }

    // 设置超时（10分钟）
    const timeout = setTimeout(() => {
      if (this.pollingInterval) {
        clearInterval(this.pollingInterval);
        this.pollingInterval = null;
      }
      this.error = "研究任务执行超时，请重试";
    }, 600000);

    // 每2秒轮询一次任务状态
    this.pollingInterval = setInterval(async () => {
      try {
        console.log('轮询任务状态:', this.currentTaskId);
        const response = await axios.get(`${API_BASE_URL}/research/${this.currentTaskId}`);
        const taskData = response.data;

        console.log('任务状态响应:', taskData);
        this.status = taskData.status;
        this.progress = taskData.progress;

        if (taskData.status === 'completed') {
          clearTimeout(timeout);
          this.handleCompletedTask(taskData.result);
          clearInterval(this.pollingInterval);
        } else if (taskData.status === 'error') {
          clearTimeout(timeout);
          this.error = taskData.error || '研究过程中发生错误';
          clearInterval(this.pollingInterval);
        }
      } catch (error) {
        console.error('轮询任务状态错误:', error);
        this.debugInfo = JSON.stringify({
          error: error.message,
          response: error.response ? error.response.data : '无响应数据',
          config: error.config
        }, null, 2);

        this.error = '获取任务状态失败';
        clearTimeout(timeout);
        clearInterval(this.pollingInterval);
      }
    }, 2000);
  },

    handleCompletedTask(result) {
      if (result) {
        this.finalReport = result.final_report || '无报告内容';
        this.searchQueries = result.search_queries || [];
      }
    },

    resetState() {
      this.currentTaskId = null;
      this.progress = 0;
      this.status = '';
      this.finalReport = '';
      this.searchQueries = [];
      this.error = null;
      this.debugInfo = '';

      if (this.pollingInterval) {
        clearInterval(this.pollingInterval);
        this.pollingInterval = null;
      }
    },

    retryResearch() {
      this.resetState();
      this.startResearch();
      }
    },

  beforeUnmount() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
    }
  }
};
</script>

<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Arial', sans-serif;
  line-height: 1.6;
  color: #333;
  background-color: #f5f7fa;
}

#app {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  text-align: center;
  margin-bottom: 30px;
  padding: 20px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.header h1 {
  color: #2c3e50;
  margin-bottom: 10px;
}

.header p {
  color: #7f8c8d;
}

.main-content {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.input-section {
  margin-bottom: 30px;
}

.input-section h2 {
  margin-bottom: 15px;
  color: #2c3e50;
}

textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 16px;
  margin-bottom: 15px;
  resize: vertical;
}

.submit-btn {
  background-color: #3498db;
  color: white;
  border: none;
  padding: 12px 24px;
  font-size: 16px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.submit-btn:hover:not(:disabled) {
  background-color: #2980b9;
}

.submit-btn:disabled {
  background-color: #bdc3c7;
  cursor: not-allowed;
}

.progress-section {
  margin-bottom: 20px;
}

.progress-bar {
  height: 20px;
  background-color: #ecf0f1;
  border-radius: 10px;
  overflow: hidden;
  margin: 10px 0;
}

.progress-fill {
  height: 100%;
  background-color: #2ecc71;
  transition: width 0.3s;
}

.error-section {
  margin-bottom: 20px;
  padding: 15px;
  background-color: #ffecec;
  border-left: 4px solid #e74c3c;
  border-radius: 4px;
}

.error-text {
  color: #c0392b;
}

.result-section {
  margin-top: 30px;
}

.result-section h2 {
  margin-bottom: 15px;
  color: #2c3e50;
}

.report-content {
  background-color: #f9f9f9;
  padding: 20px;
  border-radius: 4px;
  border: 1px solid #eee;
  white-space: pre-wrap;
}

.search-queries {
  margin-top: 20px;
}

.search-queries h3 {
  margin-bottom: 10px;
  color: #2c3e50;
}

.search-queries ul {
  list-style-type: none;
}

.search-queries li {
  padding: 8px 12px;
  background-color: #f1f8ff;
  margin-bottom: 8px;
  border-radius: 4px;
  border-left: 3px solid #3498db;
}

.task-info {
  margin: 10px 0;
  padding: 10px;
  background-color: #f8f9fa;
  border-radius: 4px;
  border-left: 3px solid #6c757d;
}

.task-info p {
  margin: 5px 0;
}

.retry-btn {
  background-color: #28a745;
  color: white;
  border: none;
  padding: 8px 16px;
  margin-top: 10px;
  border-radius: 4px;
  cursor: pointer;
}

.retry-btn:hover {
  background-color: #218838;
}
</style>
