<!-- frontend/src/components/AppFooter.vue -->
<template>
  <footer class="app-footer">
    <div class="footer-content">
      <div class="footer-section">
        <h4>医疗研究助手</h4>
        <p>基于AI的智能医疗问答系统</p>
      </div>
      
      <div class="footer-section">
        <h4>技术栈</h4>
        <ul>
          <li>LangGraph & LangChain</li>
          <li>Milvus 向量数据库</li>
          <li>Qwen 大语言模型</li>
          <li>FastAPI & Vue3</li>
        </ul>
      </div>
      
      <div class="footer-section">
        <h4>重要声明</h4>
        <p class="disclaimer">
          本系统提供的信息仅供参考，不能替代专业医疗建。
        </p>
      </div>
    </div>
    
    <div class="footer-bottom">
      <p>&copy; 2025 高级医疗研究助手系统. </p>
    </div>
  </footer>
</template>

<style scoped>
.app-footer {
  background: #2c3e50;
  color: #ecf0f1;
  margin-top: auto;
}

.footer-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
}

.footer-section h4 {
  margin-bottom: 1rem;
  color: #3498db;
}

.footer-section ul {
  list-style: none;
  padding: 0;
}

.footer-section li {
  margin-bottom: 0.5rem;
  color: #bdc3c7;
}

.disclaimer {
  font-size: 0.9rem;
  color: #bdc3c7;
  line-height: 1.5;
}

.footer-bottom {
  border-top: 1px solid #34495e;
  padding: 1rem;
  text-align: center;
  color: #95a5a6;
}
</style>
