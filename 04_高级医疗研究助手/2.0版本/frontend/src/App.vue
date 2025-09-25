<template>
  <div id="app">
    <el-container>
      <el-header class="app-header">
        <h1>
          <el-icon><FirstAidKit /></el-icon>
          高级医疗研究助手 & 报告生成器
        </h1>
        <el-tag v-if="serviceStatus" type="success">服务正常</el-tag>
        <el-tag v-else type="danger">服务异常</el-tag>
      </el-header>

      <el-main class="app-main">
        <el-progress
            v-if="loading"
            :percentage="progress"
            :text-inside="true"
            :stroke-width="20"
            status="success"
            style="margin-bottom: 20px;"
        />
        <el-row :gutter="20">
          <el-col :span="10">
            <el-card>
              <template #header>
                <div class="card-header">
                  <span>研究问题输入</span>
                </div>
              </template>

              <ResearchForm
                :loading="loading"
                @research="handleResearch"
              />
            </el-card>
          </el-col>

          <el-col :span="14">
            <el-card>
              <template #header>
                <div class="card-header">
                  <span>研究报告</span>
                  <el-button
                    v-if="report"
                    :loading="loading"
                    :icon="Refresh"
                    @click="handleResearch(lastQuery)"
                  >
                    重新生成
                  </el-button>
                </div>
              </template>

              <div v-loading="loading" element-loading-text="研究中..." element-loading-background="rgba(255, 255, 255, 0.8)">
                <ReportViewer
                  :report="report"
                  :query="lastQuery"
                />
              </div>
            </el-card>
          </el-col>
        </el-row>

        <!-- 全局进度条 -->
        <el-progress
          v-if="loading"
          :percentage="progress"
          :indeterminate="progress === 0"
          :color="progressColors"
          class="global-progress"
        />
      </el-main>
    </el-container>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElNotification } from 'element-plus'
import { FirstAidKit, Refresh } from '@element-plus/icons-vue'
import ResearchForm from './components/ResearchForm.vue'
import ReportViewer from './components/ReportViewer.vue'
import { researchService } from './services/api'

const loading = ref(false)
const report = ref('')
const lastQuery = ref('')
const serviceStatus = ref(false)
const researchId = ref(null) // 新增：研究任务ID
const progress = ref(0) // 新增：进度
const progressTimer = ref(null) // 新增：进度定时器

onMounted(async () => {
  try {
    await researchService.checkHealth()
    serviceStatus.value = true
    ElMessage.success('服务连接正常')
  } catch (error) {
    ElMessage.error('服务连接失败: ' + error.message)
  }
})

// 新增：检查研究进度
const checkResearchProgress = async (id) => {
  try {
    const response = await researchService.getResearchProgress(id)

    if (response.status === 'completed') {
      clearInterval(progressTimer.value)
      progress.value = 100
      report.value = response.report
      loading.value = false
      ElMessage.success('研究完成')
    } else if (response.status === 'processing') {
      progress.value = response.progress || 0
      // 继续轮询
    } else if (response.status === 'failed') {
      clearInterval(progressTimer.value)
      loading.value = false
      ElMessage.error('研究失败: ' + response.message)
    }
  } catch (error) {
    console.error('检查进度失败:', error)
    // 不停止轮询，可能是临时网络问题
  }
}

const handleResearch = async (query) => {
  if (!query.trim()) {
    ElMessage.warning('请输入有效的研究问题')
    return
  }

  loading.value = true
  lastQuery.value = query
  progress.value = 0

  try {
    // 开始研究，获取任务ID
    const response = await researchService.startResearch(query)
    researchId.value = response.research_id

    // 显示进度通知
    ElNotification({
      title: '研究已开始',
      message: '研究任务正在后台处理，完成后将自动显示结果',
      type: 'info',
      duration: 3000
    })

    // 启动进度轮询
    progressTimer.value = setInterval(() => {
      if (researchId.value) {
        checkResearchProgress(researchId.value)
      }
    }, 3000) // 每3秒检查一次进度

  } catch (error) {
    loading.value = false
    ElMessage.error('启动研究失败: ' + error.message)
    console.error('研究错误:', error)
  }
}
</script>


<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  min-height: 100vh;
  background: #f5f7fa;
  position: relative;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #fff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: 0 20px;
  position: sticky;
  top: 0;
  z-index: 1000;
}

.app-header h1 {
  display: flex;
  align-items: center;
  gap: 10px;
}

.app-main {
  padding: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.global-progress {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 2000;
  border-radius: 0;
}
</style>
