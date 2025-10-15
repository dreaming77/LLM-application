import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException
import time
from core.workflow.workflow_manager import workflow_manager
from core.state.state_manager import StateManager
from core.model_manager import model_manager
from milvus.milvus_client import health_check as milvus_health_check
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class MedicalResearchController:
    """
    医疗研究助手API控制器
    """
    
    def __init__(self):
        self.workflow_manager = workflow_manager
        self.state_manager = StateManager()

    async def initialize(self) -> None:
        """初始化控制器"""
        try:
            # 先初始化模型
            if not model_manager.initialize_models():
                raise RuntimeError("模型初始化失败，请检查模型路径和GPU配置")

            # 再初始化工作流
            await self.workflow_manager.initialize()
            logger.info("医疗研究助手控制器初始化完成")

        except Exception as e:
            logger.error(f"控制器初始化失败: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        try:
            # Milvus健康检查
            
            milvus_health = milvus_health_check()
            
            # 工作流健康检查
            workflow_health = {
                "initialized": self.workflow_manager.workflow is not None,
                "active_sessions": len(self.workflow_manager.active_sessions),
                "total_executions": self.workflow_manager.workflow_stats["total_executions"]
            }
            
            # 模型健康检查（简化）
            model_health = {
                "embedding_model_loaded": hasattr(self, 'embedding_model'),
                "generation_model_loaded": hasattr(self, 'generation_model')
            }
            
            overall_status = "healthy"
            if milvus_health.get("status") != "healthy":
                overall_status = "degraded"
            if not workflow_health["initialized"]:
                overall_status = "unhealthy"
            
            return {
                "status": overall_status,
                "timestamp": time.time(),
                "components": {
                    "milvus": milvus_health,
                    "workflow": workflow_health,
                    "models": model_health
                }
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "status": "error",
                "timestamp": time.time(),
                "error": str(e)
            }

    async def process_medical_query(self, query: str, user_context: Optional[Dict[str, Any]] = None,
                                  session_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理医疗查询
        
        Args:
            query: 用户查询
            user_context: 用户上下文
            session_id: 会话ID
            config: 配置参数
            
        Returns:
            处理结果
        """
        try:
            # 验证查询
            if not query or len(query.strip()) < 5:
                raise HTTPException(status_code=400, detail="查询过短，请提供更详细的问题描述")
            
            if len(query) > 1000:
                raise HTTPException(status_code=400, detail="查询过长，请简化问题表述")
            
            # 处理查询
            result = await self.workflow_manager.process_query(
                query, user_context, session_id, config
            )
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"处理医疗查询失败: {e}")
            raise HTTPException(status_code=500, detail=f"系统处理错误: {str(e)}")
    
    async def continue_conversation(self, session_id: str, new_query: Optional[str] = None,
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        继续对话 - 支持对话记忆
        """
        try:
            if not session_id:
                raise HTTPException(status_code=400, detail="需要提供会话ID")
            
            # 获取现有状态
            state = self.state_manager.get_state(session_id)
            if not state:
                raise ValueError(f"会话 {session_id} 不存在")
            
            # 更新查询
            if new_query:
                # 保存当前对话到历史
                if "conversation_history" not in state:
                    state["conversation_history"] = []
                
                # 添加上一轮对话到历史
                if state.get("user_query") and state.get("refined_response"):
                    state["conversation_history"].append(
                        HumanMessage(content=state["user_query"])  # 用户消息
                    )
                    state["conversation_history"].append(
                        AIMessage(content=state["refined_response"])  # 助手消息
                    )
                
                # 更新当前查询
                state = self.state_manager.update_state(session_id, {
                    "user_query": new_query,
                    "original_query": new_query,
                    "current_query": new_query,
                    "conversation_history": state.get("conversation_history", [])
                })
            
            # 应用配置
            if config:
                state = self._apply_config_to_state(state, config)
            
            # 重置工作流状态（从意图识别重新开始）
            state = self.state_manager.update_state(session_id, {
                "current_step": "medical_intent_detection",
                "next_step": "medical_intent_detection",
                "retry_count": 0,
                "error_message": None
            })
            
            # 执行工作流
            return await self.workflow_manager.process_query(state["user_query"], state.get("user_context"), session_id, config)
            
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"继续对话失败: {e}")
            raise HTTPException(status_code=500, detail=f"系统错误: {str(e)}")
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """获取会话状态"""
        try:
            status = self.workflow_manager.get_session_status(session_id)
            return status
        except Exception as e:
            logger.error(f"获取会话状态失败: {e}")
            raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            workflow_info = self.workflow_manager.get_workflow_info()
            
            return {
                "system": "高级医疗研究助手",
                "version": "1.0.0",
                "status": "运行中",
                "workflow_info": workflow_info,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"获取系统信息失败: {e}")
            raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")
    
    async def interrupt_processing(self, session_id: str) -> Dict[str, Any]:
        """中断处理"""
        try:
            success = await self.workflow_manager.interrupt_session(session_id)
            
            return {
                "success": success,
                "message": "会话中断请求已提交" if success else "会话不存在或无法中断",
                "session_id": session_id
            }
        except Exception as e:
            logger.error(f"中断处理失败: {e}")
            raise HTTPException(status_code=500, detail=f"中断失败: {str(e)}")
    
    async def close(self) -> None:
        """关闭控制器"""
        await self.workflow_manager.close()
        logger.info("医疗研究助手控制器已关闭")


    async def model_health_check(self) -> Dict[str, Any]:
        """模型健康检查"""
        try:
            from core.model_manager import model_manager
            
            model_info = model_manager.get_model_info()
            
            # 执行简单测试
            test_results = {}
            
            # 测试嵌入模型
            embedding_test = model_manager.embed_text("健康检查测试")
            test_results["embedding"] = {
                "working": embedding_test is not None and len(embedding_test) == 1024,
                "dimension": len(embedding_test) if embedding_test else 0
            }
            
            # 测试生成模型
            generation_test = model_manager.generate_response("回复'OK'进行健康检查。", max_tokens=10)
            test_results["generation"] = {
                "working": generation_test is not None and len(generation_test) > 0,
                "response_length": len(generation_test) if generation_test else 0
            }
            
            overall_healthy = test_results["embedding"]["working"] and test_results["generation"]["working"]
            
            return {
                "status": "healthy" if overall_healthy else "degraded",
                "timestamp": time.time(),
                "model_info": model_info,
                "test_results": test_results
            }
            
        except Exception as e:
            logger.error(f"模型健康检查失败: {e}")
            return {
                "status": "error",
                "timestamp": time.time(),
                "error": str(e)
            }

# 创建全局控制器实例
medical_research_controller = MedicalResearchController()
