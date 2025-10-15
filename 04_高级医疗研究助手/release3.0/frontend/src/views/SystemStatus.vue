<!-- frontend/src/views/SystemStatus.vue -->
<template>
  <div class="status-page">
    <div class="container">
      <div class="header">
        <p>实时监控医疗研究助手系统的运行状态</p>
      </div>

      <div class="status-grid">
        <div class="status-card overall-status">
          <h3>总体状态</h3>
          <div class="status-indicator" :class="overallStatus">
            {{ overallStatusText }}
          </div>
          <p>最后检查: {{ lastCheckTime }}</p>
          <p>活跃会话: {{ systemInfo.workflow_info?.active_sessions || 0 }}</p>
        </div>

        <div class="status-card component-status">
          <h3>组件状态</h3>
          <div class="component-list">
            <div class="component-item">
              <span class="component-name">Milvus数据库</span>
              <span class="component-status" :class="milvusStatus">
                {{ getStatusText(milvusStatus) }}
              </span>
            </div>
            <div class="component-item">
              <span class="component-name">工作流引擎</span>
              <span class="component-status" :class="workflowStatus">
                {{ getStatusText(workflowStatus) }}
              </span>
            </div>
            <div class="component-item">
              <span class="component-name">AI模型服务</span>
              <span class="component-status" :class="aiModelStatus">
                {{ getStatusText(aiModelStatus) }}
              </span>
            </div>
          </div>
        </div>

        <div class="status-card model-status">
          <h3>模型状态</h3>
          <div class="model-list">
            <div class="model-item">
              <span>嵌入模型</span>
              <span :class="modelStatus.embedding">
                {{ getModelStatusText(modelStatus.embedding) }}
              </span>
            </div>
            <div class="model-item">
              <span>生成模型</span>
              <span :class="modelStatus.generation">
                {{ getModelStatusText(modelStatus.generation) }}
              </span>
            </div>
            <div class="model-item">
              <span>GPU状态</span>
              <span :class="modelStatus.gpu">
                {{ getModelStatusText(modelStatus.gpu) }}
              </span>
            </div>
          </div>
        </div>

        <div class="status-card system-info">
          <h3>系统信息</h3>
          <div class="info-list">
            <div class="info-item">
              <span>版本</span>
              <span>{{ systemInfo.version || '1.0.0' }}</span>
            </div>
            <div class="info-item">
              <span>总执行次数</span>
              <span>{{ systemInfo.workflow_info?.statistics?.total_executions || 0 }}</span>
            </div>
            <div class="info-item">
              <span>Milvus文档数</span>
              <span>{{ milvusEntityCount || 0 }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 详细模型信息 -->
      <div v-if="modelDetails" class="model-details">
        <h3>详细模型信息</h3>
        <div class="details-grid">
          <div class="detail-item">
            <label>嵌入模型</label>
            <span>{{ getModelName(modelDetails.embedding_model?.model_name) || '未知' }}</span>
          </div>
          <div class="detail-item">
            <label>生成模型</label>
            <span>{{ getModelName(modelDetails.generation_model?.model_name) || '未知' }}</span>
          </div>
          <div class="detail-item">
            <label>嵌入模型设备</label>
            <span>{{ modelDetails.embedding_model?.device || '未知' }}</span>
          </div>
          <div class="detail-item">
            <label>生成模型设备</label>
            <span>{{ modelDetails.generation_model?.device || '未知' }}</span>
          </div>
          <div class="detail-item">
            <label>GPU/CPU</label>
            <span>{{ modelDetails.gpu_available !== undefined ? (modelDetails.gpu_available ? 'GPU' : 'CPU') : '未知' }}</span>
          </div>
          <div class="detail-item">
            <label>GPU数量</label>
            <span>{{ modelDetails.gpu_count !== undefined ? modelDetails.gpu_count : '未知' }}</span>
          </div>
        </div>
      </div>

       <div class="actions">
        <button @click="refreshStatus" class="refresh-btn" :disabled="isLoading">
          {{ isLoading ? '检查中...' : '刷新状态' }}
        </button>
        <button @click="testModels" class="test-btn" :disabled="isLoading">
          测试模型
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import apiService from '../services/api'

const isLoading = ref(false)
const detailedStatus = ref({})
const systemInfo = ref({})
const modelDetails = ref(null)
const modelStatus = ref({
  embedding: 'unknown',
  generation: 'unknown',
  gpu: 'unknown'
})

// 计算各个组件的状态
const milvusStatus = computed(() => {
  return detailedStatus.value.components?.milvus?.status || 'unknown'
})

const workflowStatus = computed(() => {
  const workflow = detailedStatus.value.components?.workflow
  if (!workflow) return 'unknown'
  return workflow.initialized ? 'healthy' : 'unhealthy'
})

const aiModelStatus = computed(() => {
  // 使用模型详细状态来判断AI模型服务状态
  return modelStatus.value.embedding === 'healthy' && modelStatus.value.generation === 'healthy' ? 'healthy' : 'unhealthy'
})

const milvusEntityCount = computed(() => {
  return detailedStatus.value.components?.milvus?.entity_count || 0
})

const overallStatus = computed(() => {
  // 综合判断整体状态
  if (milvusStatus.value === 'healthy' && 
      workflowStatus.value === 'healthy' && 
      aiModelStatus.value === 'healthy') {
    return 'healthy'
  } else if (milvusStatus.value === 'error' || 
             workflowStatus.value === 'error' || 
             aiModelStatus.value === 'error') {
    return 'error'
  } else {
    return 'degraded'
  }
})

const overallStatusText = computed(() => {
  const statusMap = {
    healthy: '健康',
    degraded: '降级',
    unhealthy: '不健康',
    error: '错误',
    unknown: '未知'
  }
  return statusMap[overallStatus.value] || '未知'
})

const lastCheckTime = computed(() => {
  return detailedStatus.value.timestamp 
    ? new Date(detailedStatus.value.timestamp * 1000).toLocaleString('zh-CN')
    : '从未检查'
})

const getStatusText = (status) => {
  const statusMap = {
    healthy: '正常',
    degraded: '降级',
    unhealthy: '异常',
    error: '错误',
    unknown: '未知'
  }
  return statusMap[status] || '未知'
}

const getModelStatusText = (status) => {
  const statusMap = {
    healthy: '正常',
    degraded: '降级',
    unhealthy: '异常',
    error: '错误',
    unknown: '未知'
  }
  return statusMap[status] || '未知'
}

const calculateMemoryUsage = (gpu) => {
  const totalMemory = gpu.memory_allocated + gpu.memory_free
  if (totalMemory === 0) return 0
  return (gpu.memory_allocated / totalMemory) * 100
}

const refreshStatus = async () => {
  isLoading.value = true
  try {
    // 并行调用所有状态API
    const [status, info, modelHealth, debugInfo] = await Promise.all([
      apiService.detailedHealthCheck(),
      apiService.getSystemInfo(),
      apiService.modelHealthCheck(),
      apiService.debugModels()  // 同时获取调试信息来获得模型名称
    ])
    
    detailedStatus.value = status
    systemInfo.value = info
    
    // 更新模型状态
    updateModelStatusFromHealth(modelHealth)
    
    // 如果调试信息可用，使用它来获取模型名称
    if (debugInfo && !debugInfo.error && debugInfo.model_info) {
      modelDetails.value = debugInfo.model_info

            modelDetails.value.gpu_available = modelHealth.gpu_available
      modelDetails.value.gpu_count = modelHealth.gpu_count
      
      // 如果 modelHealth 中没有 GPU 信息，但 debugInfo 中有，则使用 debugInfo 的
      if (modelDetails.value.gpu_available === undefined && debugInfo.model_info.gpu_info) {
        modelDetails.value.gpu_available = Object.keys(debugInfo.model_info.gpu_info).length > 0
        modelDetails.value.gpu_count = Object.keys(debugInfo.model_info.gpu_info).length
      }
    } else {
      // 如果调试信息不可用，至少设置基本模型信息
      modelDetails.value = {
        embedding_model: {
          model_name: 'BAAI/bge-large-zh-v1.5',
          device: modelHealth.embedding_model?.device || '未知'
        },
        generation_model: {
          model_name: 'Qwen2.5-7B-Instruct', 
          device: modelHealth.generation_model?.device || '未知',
          config: {
            max_new_tokens: 1000,
            temperature: 0.3,
            do_sample: true
          }
        },
        gpu_available: modelHealth.gpu_available,
        gpu_count: modelHealth.gpu_count
      }
    }
    
  } catch (error) {
    console.error('Failed to refresh status:', error)
    // 设置所有状态为未知
    modelStatus.value = {
      embedding: 'unknown',
      generation: 'unknown',
      gpu: 'unknown'
    }
  } finally {
    isLoading.value = false
  }
}

const updateModelStatusFromHealth = (modelHealth) => {
  if (!modelHealth || modelHealth.error) {
    modelStatus.value = {
      embedding: 'error',
      generation: 'error',
      gpu: 'error'
    }
    return
  }
  
  // 使用正确的字段路径
  modelStatus.value = {
    embedding: modelHealth.embedding_model?.working ? 'healthy' : 'unhealthy',
    generation: modelHealth.generation_model?.working ? 'healthy' : 'unhealthy',
    gpu: modelHealth.gpu_available ? 'healthy' : 'unhealthy'
  }
  
  // 确保 GPU 状态正确
  if (modelHealth.gpu_available) {
    modelStatus.value.gpu = 'healthy'
  }

  // 保存基本模型信息（即使没有完整名称）
  if (!modelDetails.value) {
    modelDetails.value = {
      embedding_model: {
        device: modelHealth.embedding_model?.device || '未知'
      },
      generation_model: {
        device: modelHealth.generation_model?.device || '未知'
      },
      gpu_available: modelHealth.gpu_available,
      gpu_count: modelHealth.gpu_count
    }
  }
}

const testModels = async () => {
  isLoading.value = true
  try {
    // 调用模型调试API
    const debugResult = await apiService.debugModels()
    
    if (debugResult.error) {
      console.error('Model debug failed:', debugResult.error)
      return
    }
    
    // 更新模型状态
    modelStatus.value = {
      embedding: debugResult.embedding_test === '成功' ? 'healthy' : 'unhealthy',
      generation: debugResult.generation_test === '成功' ? 'healthy' : 'unhealthy',
      gpu: debugResult.model_info?.gpu_info ? 'healthy' : 'unhealthy'
    }
    
    // 保存详细模型信息
    modelDetails.value = debugResult.model_info
    
        // 确保 GPU 信息正确设置
    if (debugResult.model_info?.gpu_info) {
      modelDetails.value.gpu_available = Object.keys(debugResult.model_info.gpu_info).length > 0
      modelDetails.value.gpu_count = Object.keys(debugResult.model_info.gpu_info).length
    }
    
    console.log('模型测试完成:', debugResult)
    
  } catch (error) {
    console.error('Model test failed:', error)
  } finally {
    isLoading.value = false
  }
}

const getModelName = (modelPath) => {
  if (!modelPath) return '未知'
  // 从路径中提取最后的目录名作为模型名称
  const parts = modelPath.split('/')
  return parts[parts.length - 1] || '未知'
}

onMounted(() => {
  refreshStatus()
})
</script>

<style scoped>
.status-page {
  padding: 2rem 0;
  flex: 1;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}

.header {
  text-align: center;
  margin-bottom: 2rem;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin-bottom: 2rem;
}

.status-card {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.status-card h3 {
  margin-bottom: 1rem;
  color: #2c3e50;
}

.status-indicator {
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 1rem;
  padding: 0.5rem;
  border-radius: 6px;
  text-align: center;
}

.status-indicator.healthy { background: #d5f4e6; color: #27ae60; }
.status-indicator.degraded { background: #fff3cd; color: #f39c12; }
.status-indicator.unhealthy { background: #f8d7da; color: #e74c3c; }
.status-indicator.error { background: #f8d7da; color: #e74c3c; }
.status-indicator.unknown { background: #e9ecef; color: #6c757d; }

.component-list,
.model-list,
.info-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.component-item,
.model-item,
.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
  border-bottom: 1px solid #e9ecef;
}

.component-status.healthy { color: #27ae60; font-weight: 600; }
.component-status.degraded { color: #f39c12; font-weight: 600; }
.component-status.unhealthy { color: #e74c3c; font-weight: 600; }
.component-status.error { color: #e74c3c; font-weight: 600; }
.component-status.unknown { color: #6c757d; font-weight: 600; }

.model-status .healthy { color: #27ae60; font-weight: 600; }
.model-status .unhealthy { color: #e74c3c; font-weight: 600; }
.model-status .unknown { color: #6c757d; font-weight: 600; }
.model-status .error { color: #e74c3c; font-weight: 600; }

.model-details {
  background: white;
  border-radius: 12px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.model-details h3 {
  margin-bottom: 1rem;
  color: #2c3e50;
}

.details-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.detail-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.detail-item label {
  font-weight: 600;
  color: #7f8c8d;
  font-size: 0.9rem;
}

.detail-item span {
  color: #2c3e50;
  font-family: monospace;
  background: #f8f9fa;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
}

.gpu-info h4 {
  margin-bottom: 1rem;
  color: #2c3e50;
}

.gpu-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.gpu-item {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 8px;
}

.gpu-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.gpu-name {
  font-weight: 600;
  color: #2c3e50;
}

.gpu-memory {
  font-size: 0.9rem;
  color: #7f8c8d;
}

.memory-bar {
  width: 100%;
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
}

.memory-used {
  height: 100%;
  background: linear-gradient(90deg, #2ecc71, #f39c12, #e74c3c);
  transition: width 0.3s ease;
}

.actions {
  text-align: center;
}

.refresh-btn,
.test-btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  margin: 0 0.5rem;
  font-size: 1rem;
  transition: all 0.3s ease;
}

.refresh-btn {
  background: #3498db;
  color: white;
}

.refresh-btn:hover:not(:disabled) {
  background: #2980b9;
  transform: translateY(-1px);
}

.test-btn {
  background: #2ecc71;
  color: white;
}

.test-btn:hover:not(:disabled) {
  background: #27ae60;
  transform: translateY(-1px);
}

.refresh-btn:disabled,
.test-btn:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
  transform: none;
}

@media (max-width: 768px) {
  .status-grid {
    grid-template-columns: 1fr;
  }
  
  .details-grid {
    grid-template-columns: 1fr;
  }
  
  .gpu-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
  }
  
  .actions {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  
  .refresh-btn,
  .test-btn {
    margin: 0;
  }
}
</style>
