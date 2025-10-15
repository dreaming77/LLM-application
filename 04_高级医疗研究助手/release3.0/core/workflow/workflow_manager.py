import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph
from core.state.state import GraphState
from core.state.state_manager import StateManager
from core.workflow.graph_definition import create_medical_research_graph, get_workflow_visualization
from config.settings import MAX_SESSION_AGE_HOURS

logger = logging.getLogger(__name__)

class WorkflowManager:
    """
    工作流管理器 - 负责工作流的执行、监控和管理
    """

    def __init__(self):
        self.workflow: Optional[StateGraph] = None
        self.state_manager = StateManager()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.workflow_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0,
            "session_history": []
        }

    async def initialize(self) -> None:
        """初始化工作流"""
        logger.info("初始化医疗研究助手工作流")

        try:
            # 创建图但不编译（避免检查点问题）
            self.workflow = create_medical_research_graph()
            logger.info("工作流初始化成功")

            # 启动会话清理任务
            asyncio.create_task(self._session_cleanup_task())

        except Exception as e:
            logger.error(f"工作流初始化失败: {e}")
            raise

    async def process_query(self, user_query: str, user_context: Optional[Dict[str, Any]] = None,
                            session_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[
        str, Any]:
        """
        处理用户查询

        Args:
            user_query: 用户查询文本
            user_context: 用户上下文信息
            session_id: 会话ID（如未提供则创建新会话）
            config: 工作流配置参数

        Returns:
            处理结果
        """
        if not self.workflow:
            raise RuntimeError("工作流未初始化")

        start_time = time.time()
        execution_id = f"exec_{int(start_time * 1000)}"

        try:
            # 初始化状态
            state = self.state_manager.initialize_state(user_query, user_context, session_id)

            # 应用配置参数
            if config:
                state = self._apply_config_to_state(state, config)

            session_id = state["session_id"]

            # 记录活跃会话
            self.active_sessions[session_id] = {
                "start_time": start_time,
                "execution_id": execution_id,
                "user_query": user_query,
                "status": "running"
            }

            logger.info(f"开始处理会话 {session_id}, 执行ID: {execution_id}")

            # 手动执行工作流节点，而不是使用ainvoke
            final_state = await self._execute_workflow_manually(state)

            # 记录完成时间
            end_time = time.time()
            duration = end_time - start_time

            # 更新会话状态
            self.active_sessions[session_id].update({
                "end_time": end_time,
                "duration": duration,
                "status": "completed",
                "final_state": final_state.get("final_response", {})
            })

            # 更新统计信息
            self._update_statistics(duration, True)

            logger.info(f"会话 {session_id} 处理完成, 耗时: {duration:.2f}秒")

            return self._prepare_final_result(final_state, duration, execution_id)

        except Exception as e:
            # 处理执行错误
            error_duration = time.time() - start_time
            self._update_statistics(error_duration, False)

            logger.error(f"工作流执行失败: {e}")

            # 清理活跃会话
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "failed"
                self.active_sessions[session_id]["error"] = str(e)

            return self._prepare_error_result(user_query, str(e), execution_id)

    async def _execute_workflow_manually(self, initial_state: GraphState) -> GraphState:
        """
        手动执行工作流节点

        Args:
            initial_state: 初始状态

        Returns:
            最终状态
        """
        from core.node_manager import node_manager

        current_state = initial_state.copy()
        iteration = 0

        logger.info("开始执行工作流")

        while iteration:
            iteration += 1
            current_step = current_state.get("current_step", "initialize")
            next_step = current_state.get("next_step", "medical_intent_detection")

            logger.info(f"工作流步骤 {iteration}: {current_step} -> {next_step}")

            # 检查是否到达结束
            if next_step == "__end__" or next_step == "finalize":
                logger.info("工作流执行完成")
                # 确保设置完成标志
                current_state["workflow_completed"] = True
                break

            # 执行当前节点
            try:
                node_result = await node_manager.execute_node(current_state)

                # 更新状态
                current_state.update(node_result)

                # 记录步骤完成
                logger.info(f"节点 {next_step} 执行完成")

            except Exception as e:
                logger.error(f"节点 {next_step} 执行失败: {e}")
                current_state.update({
                    "current_step": next_step,
                    "next_step": "error",
                    "error_message": f"节点执行失败: {str(e)}"
                })
        else:
            current_state["workflow_completed"] = True

        return current_state
    
    async def continue_session(self, session_id: str, new_query: Optional[str] = None,
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        继续现有会话
        
        Args:
            session_id: 会话ID
            new_query: 新的用户查询（如提供则更新）
            config: 工作流配置参数
            
        Returns:
            处理结果
        """
        if not self.workflow:
            raise RuntimeError("工作流未初始化")
        
        # 获取现有状态
        state = self.state_manager.get_state(session_id)
        if not state:
            raise ValueError(f"会话 {session_id} 不存在")
        
        # 更新查询（如果提供）
        if new_query:
            state = self.state_manager.update_state(session_id, {
                "user_query": new_query,
                "original_query": new_query,
                "current_query": new_query
            })
        
        # 重置工作流状态（从意图识别重新开始）
        state = self.state_manager.update_state(session_id, {
            "current_step": "medical_intent_detection",
            "next_step": "medical_intent_detection",
            "retry_count": 0,
            "error_message": None
        })
        
        # 执行工作流
        return await self.process_query(state["user_query"], state.get("user_context"), session_id, config)
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """获取工作流信息"""
        if not self.workflow:
            return {"status": "未初始化"}
        
        visualization = get_workflow_visualization(self.workflow)
        
        return {
            "status": "已初始化",
            "节点数量": len(self.workflow.nodes) if hasattr(self.workflow, 'nodes') else "未知",
            "可视化": visualization,
            "统计信息": self.workflow_stats,
            "活跃会话数": len(self.active_sessions)
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """获取会话状态"""
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id].copy()
            # 移除可能过大的状态数据
            session_info.pop("final_state", None)
            return session_info
        else:
            state = self.state_manager.get_state(session_id)
            if state:
                return {
                    "session_id": session_id,
                    "status": "completed" if state.get("workflow_completed") else "unknown",
                    "user_query": state.get("user_query"),
                    "current_step": state.get("current_step")
                }
            else:
                return {"error": f"会话 {session_id} 不存在"}
    
    async def interrupt_session(self, session_id: str) -> bool:
        """中断会话执行"""
        # 注意：LangGraph目前没有直接的中断API
        # 这里我们标记会话为中断状态，在下次检查时处理
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "interrupted"
            logger.info(f"会话 {session_id} 已被标记为中断")
            return True
        return False
    
    def _apply_config_to_state(self, state: GraphState, config: Dict[str, Any]) -> GraphState:
        """应用配置参数到状态"""
        config_mapping = {
            "search_k": "search_k",
            "fusion_top_k": "fusion_top_k", 
            "generation_temperature": "generation_temperature",
            "max_retries": "max_retries"
        }
        
        updates = {}
        for config_key, state_key in config_mapping.items():
            if config_key in config:
                updates[state_key] = config[config_key]
        
        return self.state_manager.update_state(state["session_id"], updates)
    
    def _update_statistics(self, duration: float, success: bool) -> None:
        """更新工作流统计信息"""
        self.workflow_stats["total_executions"] += 1
        
        if success:
            self.workflow_stats["successful_executions"] += 1
        else:
            self.workflow_stats["failed_executions"] += 1
        
        # 更新平均耗时
        total_duration = self.workflow_stats["average_duration"] * (self.workflow_stats["total_executions"] - 1)
        self.workflow_stats["average_duration"] = (total_duration + duration) / self.workflow_stats["total_executions"]
        
        # 记录执行历史（保留最近100条）
        execution_record = {
            "timestamp": time.time(),
            "duration": duration,
            "success": success
        }
        
        self.workflow_stats["session_history"].append(execution_record)
        if len(self.workflow_stats["session_history"]) > 100:
            self.workflow_stats["session_history"] = self.workflow_stats["session_history"][-100:]

    def _prepare_final_result(self, final_state: GraphState, duration: float, execution_id: str) -> Dict[str, Any]:
        """准备最终结果 - 修复版"""
        final_response = final_state.get("final_response", {})
        session_metrics = final_state.get("session_metrics", {})

        # 如果最终响应为空，尝试从其他字段获取
        if not final_response:
            # 从精炼响应或初始响应构建最终响应
            response_content = final_state.get("refined_response") or final_state.get("initial_response", "")
            if not response_content:
                response_content = "抱歉，无法生成完整的医疗回答。建议您咨询专业医生。"

            final_response = {
                "content": response_content,
                "type": "medical_response",
                "timestamp": time.time(),
                "has_error": final_state.get("error_message") is not None,
                "confidence_score": final_state.get("confidence_score", 0.3),
                "sources_count": len(final_state.get("citations", []))
            }

        # 如果会话指标为空，创建基本指标
        if not session_metrics:
            session_metrics = {
                "session_id": final_state.get("session_id", "unknown"),
                "duration_seconds": round(duration, 2),
                "steps_completed": self._count_completed_steps(final_state),
                "has_errors": final_state.get("error_message") is not None,
                "completion_status": "success" if not final_state.get("error_message") else "partial"
            }

        medical_metadata = final_response.get("medical_metadata", {})

        return {
            "success": True,
            "execution_id": execution_id,
            "session_id": final_state.get("session_id", "unknown"),
            "duration_seconds": round(duration, 2),
            "response": final_response.get("content", ""),
            "response_metadata": {
                "confidence_score": final_response.get("confidence_score", 0.5),
                "has_error": final_response.get("has_error", False),
                "citations_count": final_response.get("sources_count", 0)
            },
            "session_metrics": session_metrics,
            "medical_metadata": medical_metadata,
            "timestamp": time.time()
        }
    
    def _prepare_error_result(self, user_query: str, error_message: str, execution_id: str) -> Dict[str, Any]:
        """准备错误结果"""
        return {
            "success": False,
            "execution_id": execution_id,
            "error": error_message,
            "user_query": user_query,
            "fallback_response": "抱歉，系统处理您的请求时出现错误。请稍后重试或联系技术支持。",
            "timestamp": time.time(),
            "suggested_actions": [
                "稍后重试",
                "简化问题表述",
                "联系人工客服"
            ]
        }

    def _count_completed_steps(self, state: GraphState) -> int:
        """计算完成的步骤数量"""
        completed_steps = [
            "medical_intent_detection", "query_rewriting", "document_retrieval",
            "rag_fusion", "response_generation", "response_refinement"
        ]

        count = 0
        for step in completed_steps:
            # 检查步骤是否有有效输出
            if step == "medical_intent_detection" and state.get("medical_intent"):
                count += 1
            elif step == "query_rewriting" and state.get("rewritten_queries"):
                count += 1
            elif step == "document_retrieval" and state.get("retrieved_documents"):
                count += 1
            elif step == "rag_fusion" and state.get("fusion_ranked_docs"):
                count += 1
            elif step == "response_generation" and state.get("initial_response"):
                count += 1
            elif step == "response_refinement" and state.get("refined_response"):
                count += 1

        return count

    async def _session_cleanup_task(self) -> None:
        """会话清理任务（后台运行）"""
        import asyncio
        
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时检查一次
                
                current_time = time.time()
                cleanup_threshold = current_time - (MAX_SESSION_AGE_HOURS * 3600)
                
                sessions_to_cleanup = []
                for session_id, session_info in self.active_sessions.items():
                    if session_info.get("start_time", 0) < cleanup_threshold:
                        sessions_to_cleanup.append(session_id)
                
                for session_id in sessions_to_cleanup:
                    self.state_manager.cleanup_session(session_id)
                    self.active_sessions.pop(session_id, None)
                
                if sessions_to_cleanup:
                    logger.info(f"清理了 {len(sessions_to_cleanup)} 个过期会话")
                    
            except Exception as e:
                logger.error(f"会话清理任务错误: {e}")
    
    async def close(self) -> None:
        """关闭工作流管理器"""
        logger.info("关闭工作流管理器")
        
        # 清理所有活跃会话
        for session_id in list(self.active_sessions.keys()):
            self.state_manager.cleanup_session(session_id)
        
        self.active_sessions.clear()
        self.workflow = None
        
        logger.info("工作流管理器已关闭")

# 创建全局工作流管理器实例
workflow_manager = WorkflowManager()
