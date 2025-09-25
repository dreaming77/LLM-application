<template>
  <div class="report-viewer">
    <div v-if="report" class="report-content">
      <div class="report-header">
        <h2>研究报告: {{ query }}</h2>
        <el-button
          type="primary"
          :icon="Download"
          @click="downloadReport"
        >
          下载报告
        </el-button>
      </div>

      <div class="report-body">
        <div v-html="formattedReport"></div>
      </div>
    </div>

    <div v-else class="empty-state">
      <el-empty description="暂无研究报告，请先提交研究问题" />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { Download } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const props = defineProps({
  report: String,
  query: String
})

const formattedReport = computed(() => {
  if (!props.report) return ''

  // 简单的格式转换：将数字标题转换为h3标签
  return props.report
    .replace(/(\d+\.\s+)([^\n]+)/g, '<h3>$1$2</h3>')
    .replace(/\n/g, '<br>')
})

const downloadReport = () => {
  if (!props.report) {
    ElMessage.warning('没有可下载的报告')
    return
  }

  const blob = new Blob([props.report], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `医疗研究报告-${props.query.substring(0, 20)}.txt`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
</script>

<style scoped>
.report-viewer {
  margin-top: 20px;
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 1px solid #eee;
}

.report-header h2 {
  margin: 0;
  color: #303133;
}

.report-body {
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
  line-height: 1.8;
}

.report-body h3 {
  color: #409EFF;
  margin-top: 20px;
  margin-bottom: 10px;
}

.empty-state {
  padding: 40px 0;
}
</style>
