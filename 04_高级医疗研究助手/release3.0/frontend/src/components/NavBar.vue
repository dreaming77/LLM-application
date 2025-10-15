<!-- frontend/src/components/NavBar.vue -->
<template>
  <nav class="navbar">
    <div class="nav-container">
      <router-link to="/" class="nav-brand">
        🕸️ 医疗研究助手
      </router-link>
      
      <div class="nav-links">
        <router-link to="/" class="nav-link">首页</router-link>
        <router-link to="/query" class="nav-link">医疗咨询</router-link>
        <router-link to="/history" class="nav-link">咨询历史</router-link>
        <router-link to="/status" class="nav-link">系统状态</router-link>
      </div>
      
      <div class="nav-status">
        <span class="status-indicator">●</span>
        <span class="status-text">系统正常</span>
      </div>
    </div>
  </nav>
</template>

<script setup>
// 暂时移除复杂的store逻辑
</script>

<style scoped>
.navbar {
  background: #2c3e50;
  color: white;
  padding: 1rem 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.nav-brand {
  font-size: 1.5rem;
  font-weight: bold;
  color: white;
  text-decoration: none;
}

.nav-links {
  display: flex;
  gap: 2rem;
}

.nav-link {
  color: #ecf0f1;
  text-decoration: none;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: background 0.3s ease;
}

.nav-link:hover,
.nav-link.router-link-active {
  background: #34495e;
}

.nav-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.status-indicator {
  font-size: 0.8rem;
  color: #2ecc71;
}

.status-text {
  font-size: 0.9rem;
  color: #ecf0f1;
}

@media (max-width: 768px) {
  .nav-container {
    flex-direction: column;
    gap: 1rem;
  }
  
  .nav-links {
    gap: 1rem;
  }
}
</style>
