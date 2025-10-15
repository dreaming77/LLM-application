import logging
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from core.state.state import GraphState
from core.node_manager import node_manager

logger = logging.getLogger(__name__)


def create_medical_research_graph() -> StateGraph:
    """
    创建医疗研究助手的工作流图

    Returns:
        StateGraph: 配置完成的工作流图
    """
    logger.info("创建医疗研究助手工作流图")

    # 创建图实例
    workflow = StateGraph(GraphState)

    # 添加所有节点到图中
    _add_nodes_to_graph(workflow)

    # 定义节点之间的边（流程）
    _define_workflow_edges(workflow)

    # 设置条件边（分支逻辑）
    _define_conditional_edges(workflow)

    # 编译图（不启用检查点，避免版本兼容性问题）
    workflow = workflow.compile()
    logger.info("工作流图编译完成（无检查点模式）")

    return workflow

def _add_nodes_to_graph(workflow: StateGraph) -> None:
    """添加所有节点到图中"""
    
    nodes = {
        "medical_intent_detection": node_manager.nodes["medical_intent_detection"],
        "query_rewriting": node_manager.nodes["query_rewriting"],
        "document_retrieval": node_manager.nodes["document_retrieval"],
        "rag_fusion": node_manager.nodes["rag_fusion"],
        "response_generation": node_manager.nodes["response_generation"],
        "response_refinement": node_manager.nodes["response_refinement"],
        "error_handling": node_manager.nodes["error_handling"],
        "finalize": node_manager.nodes["finalize"],
    }
    
    for node_name, node_func in nodes.items():
        workflow.add_node(node_name, node_func)
        logger.debug(f"已添加节点: {node_name}")

def _define_workflow_edges(workflow: StateGraph) -> None:
    """定义工作流的主要边（流程顺序）"""
    
    # 主流程边
    workflow.add_edge("medical_intent_detection", "query_rewriting")
    workflow.add_edge("query_rewriting", "document_retrieval")
    workflow.add_edge("document_retrieval", "rag_fusion")
    workflow.add_edge("rag_fusion", "response_generation")
    workflow.add_edge("response_generation", "response_refinement")
    
    logger.info("主流程边定义完成")


def _define_conditional_edges(workflow: StateGraph) -> None:
    """定义条件边（分支逻辑）"""

    # 设置入口点
    workflow.set_entry_point("medical_intent_detection")

    # 响应精炼后的流程
    workflow.add_edge("response_refinement", "finalize")

    # 错误处理后的条件分支
    workflow.add_conditional_edges(
        "error_handling",
        _route_after_error,
        {
            "retry": "medical_intent_detection",  # 重试从意图识别开始
            "finalize": "finalize",  # 终止流程
            "error": "error_handling"  # 继续错误处理
        }
    )
    
    # 最终节点连接到结束
    workflow.add_edge("finalize", END)
    
    logger.info("条件边定义完成")


def _route_after_error(state: GraphState) -> Literal["retry", "finalize", "error"]:
    """
    错误处理后的路由逻辑
    """
    try:
        error_metrics = state.get("error_metrics", {})
        handling_strategy = error_metrics.get("handling_strategy", "terminate")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        if handling_strategy == "retry" and retry_count < max_retries:
            return "retry"
        else:
            return "finalize"  # 终止流程

    except Exception as e:
        logger.error(f"错误处理路由失败: {e}")
        return "finalize"


def get_workflow_visualization(workflow: StateGraph) -> str:
    """获取工作流的可视化表示"""
    description = "医疗研究助手工作流（优化版）:\n\n"
    description += "节点序列:\n"
    description += "1. medical_intent_detection → 医疗意图识别\n"
    description += "2. query_rewriting → 查询重写\n"
    description += "3. document_retrieval → 文档检索\n"
    description += "4. rag_fusion → RAG Fusion\n"
    description += "5. response_generation → 响应生成\n"
    description += "6. response_refinement → 响应精炼\n"
    description += "7. finalize → 最终化\n\n"

    return description
        