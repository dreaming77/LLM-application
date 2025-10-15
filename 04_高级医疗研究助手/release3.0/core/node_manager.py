import logging
from typing import Dict, Any, Callable
from core.state.state import GraphState
from core.nodes.medical_intent_detection import medical_intent_detection_node
from core.nodes.query_rewriting import query_rewriting_node
from core.nodes.document_retrieval import document_retrieval_node
from core.nodes.rag_fusion import rag_fusion_node
from core.nodes.response_generation import response_generation_node
from core.nodes.response_refinement import response_refinement_node
from core.nodes.error_handling import error_handling_node
from core.nodes.finalize import finalize_node

logger = logging.getLogger(__name__)


class NodeManager:
    """
    节点管理器 - 统一管理和调度所有工作流节点
    """

    def __init__(self):
        self.nodes = self._initialize_nodes()

    def _initialize_nodes(self) -> Dict[str, Callable]:
        """初始化所有节点"""
        return {
            "medical_intent_detection": medical_intent_detection_node,
            "query_rewriting": query_rewriting_node,
            "document_retrieval": document_retrieval_node,
            "rag_fusion": rag_fusion_node,
            "response_generation": response_generation_node,
            "response_refinement": response_refinement_node,
            "error_handling": error_handling_node,
            "finalize": finalize_node,
            "retry": error_handling_node,  # 重试使用错误处理节点
            "regenerate": response_generation_node,  # 重新生成使用响应生成节点
        }

    async def execute_node(self, state: GraphState) -> Dict[str, Any]:
        """执行当前步骤对应的节点"""
        current_step = state.get("current_step", "initialize")
        next_step = state.get("next_step", "medical_intent_detection")

        logger.info(f"执行节点: {current_step} -> {next_step}")

        # 获取对应的节点函数
        node_func = self.nodes.get(next_step)

        if not node_func:
            logger.error(f"未找到节点: {next_step}")
            return await self._handle_missing_node(state, next_step)

        try:
            # 执行节点
            result = await node_func(state)
            logger.info(f"节点 {next_step} 执行完成")
            return result

        except Exception as e:
            logger.error(f"节点 {next_step} 执行失败: {e}")
            return await self._handle_node_error(state, next_step, e)

    async def _handle_missing_node(self, state: GraphState, missing_node: str) -> Dict[str, Any]:
        """处理缺失节点错误"""
        error_message = f"工作流节点缺失: {missing_node}"

        return {
            "current_step": state.get("current_step", "unknown"),
            "next_step": "error",
            "error_message": error_message,
            "workflow_completed": False
        }

    async def _handle_node_error(self, state: GraphState, failed_node: str, error: Exception) -> Dict[str, Any]:
        """处理节点执行错误"""
        error_message = f"节点 {failed_node} 执行错误: {str(error)}"

        return {
            "current_step": failed_node,
            "next_step": "error",
            "error_message": error_message,
            "retry_count": state.get("retry_count", 0) + 1,
            "workflow_completed": False
        }

    def validate_workflow_path(self, current_step: str, next_step: str) -> bool:
        """验证工作流路径是否有效（更新为移除安全检查后的路径）"""
        valid_transitions = {
            "initialize": ["medical_intent_detection", "error"],
            "medical_intent_detection": ["query_rewriting", "error"],
            "query_rewriting": ["document_retrieval", "error"],
            "document_retrieval": ["rag_fusion", "error"],
            "rag_fusion": ["response_generation", "error"],
            "response_generation": ["response_refinement", "error"],
            "response_refinement": ["finalize", "error"],
            "error": ["retry", "finalize"],
            "retry": ["medical_intent_detection", "error"],
            "finalize": ["__end__"]
        }

        return next_step in valid_transitions.get(current_step, [])

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """获取工作流统计信息"""
        return {
            "total_nodes": len(self.nodes),
            "node_names": list(self.nodes.keys()),
            "primary_flow": [
                "medical_intent_detection",
                "query_rewriting",
                "document_retrieval",
                "rag_fusion",
                "response_generation",
                "response_refinement",
                "finalize"
            ],
            "error_handling_flow": [
                "error",
                "retry",
                "regenerate"
            ]
        }


# 创建全局节点管理器实例
node_manager = NodeManager()
