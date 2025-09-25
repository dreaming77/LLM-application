<template>
  <div class="research-form">
    <el-form :model="form" :rules="rules" ref="formRef" @submit.prevent="handleSubmit">
      <el-form-item label="医疗研究问题" prop="query">
        <el-input
          v-model="form.query"
          type="textarea"
          :rows="4"
          placeholder="请输入您要研究的医疗问题，例如：糖尿病的症状、治疗方法和预防措施"
          resize="none"
        ></el-input>
      </el-form-item>

      <el-form-item>
        <el-button
          type="primary"
          :loading="loading"
          @click="handleSubmit"
          :icon="Search"
        >
          开始研究
        </el-button>
        <el-button @click="resetForm">重置</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { Search } from '@element-plus/icons-vue'

const props = defineProps({
  loading: Boolean
})

const emit = defineEmits(['research'])

const formRef = ref()
const form = reactive({
  query: ''
})

const rules = reactive({
  query: [
    { required: true, message: '请输入研究问题', trigger: 'blur' },
    { min: 10, message: '问题描述至少需要10个字符', trigger: 'blur' }
  ]
})

const handleSubmit = async () => {
  if (!formRef.value) return

  try {
    await formRef.value.validate()
    emit('research', form.query)
  } catch (error) {
    ElMessage.warning('请正确填写研究问题')
  }
}

const resetForm = () => {
  if (!formRef.value) return
  formRef.value.resetFields()
}
</script>

<style scoped>
.research-form {
  padding: 20px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}
</style>
