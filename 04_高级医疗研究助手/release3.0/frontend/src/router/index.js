import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Query from '../views/Query.vue'
import History from '../views/History.vue'
import SystemStatus from '../views/SystemStatus.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/query',
    name: 'Query',
    component: Query
  },
  {
    path: '/history',
    name: 'History',
    component: History
  },
  {
    path: '/status',
    name: 'SystemStatus',
    component: SystemStatus
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
