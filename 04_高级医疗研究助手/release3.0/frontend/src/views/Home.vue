<template>
  <div class="home">
    <div class="hero-section">
      <p class="subtitle">基于AI的智能医疗问答与报告生成系统</p>
      
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-icon">🔬</div>
          <h3>专业医疗知识</h3>
          <p>基于CHuatuo-26M医疗数据集的权威知识库</p>
        </div>
        
        <div class="feature-card">
          <div class="feature-icon">🧠</div>
          <h3>智能问答</h3>
          <p>使用Qwen大模型提供专业的医疗咨询</p>
        </div>
        
        <div class="feature-card">
          <div class="feature-icon">⚡</div>
          <h3>快速响应</h3>
          <p>RAG Fusion技术确保准确快速的回答</p>
        </div>
      </div>

      <div class="cta-section">
        <button @click="startQuery" class="cta-button">
          开始医疗咨询
        </button>
        <router-link to="/status" class="status-link">
          查看系统状态
        </router-link>
      </div>
    </div>

  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const startQuery = () => {
  router.push('/query')
}
</script>

<style scoped>
.home {
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.hero-section {
  text-align: center;
  margin-bottom: 3rem;
}

.hero-section h1 {
  font-size: 3rem;
  color: #2c3e50;
  margin-bottom: 1rem;
}

.subtitle {
  font-size: 1.2rem;
  color:rgb(99, 106, 107);
  margin-bottom: 3rem;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
}

.feature-card {
  padding: 2rem;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
  transition: transform 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-5px);
}

.feature-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.feature-card h3 {
  color: #2c3e50;
  margin-bottom: 1rem;
}

.feature-card p {
  color: #7f8c8d;
  line-height: 1.6;
}

.cta-section {
  margin: 3rem 0;
}

.cta-button {
  background: #3498db;
  color: white;
  border: none;
  padding: 1rem 2rem;
  font-size: 1.1rem;
  border-radius: 8px;
  cursor: pointer;
  margin-right: 1rem;
  transition: background 0.3s ease;
}

.cta-button:hover {
  background: #2980b9;
}

.status-link {
  color: #3498db;
  text-decoration: none;
  font-weight: 500;
}

.status-link:hover {
  text-decoration: underline;
}
</style>
