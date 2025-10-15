import logging
import time
from typing import Dict, Any
from core.state.state import GraphState

logger = logging.getLogger(__name__)

async def finalize_node(state: GraphState) -> Dict[str, Any]:
    """
    最终化节点
    完成工作流，准备最终响应和清理资源
    """
    logger.info("执行最终化节点")

    try:
        # 记录结束时间
        end_time = time.time()
        start_time = state.get("start_time", end_time)

        # 保存当前对话到历史（如果工作流成功完成）
        if state.get("workflow_completed") and not state.get("error_message"):
            current_conversation = {
                "user_query": state["user_query"],
                "assistant_response": state.get("refined_response") or state.get("initial_response", ""),
                "timestamp": end_time
            }
            # 这里可以将对话历史保存到持久化存储
        
        # 准备最终响应
        final_response = _prepare_final_response(state)

        # 收集会话指标
        session_metrics = _collect_session_metrics(state, start_time, end_time)

        # 准备清理数据
        cleanup_data = _prepare_cleanup_data(state)

        logger.info(f"工作流完成，响应长度: {len(final_response.get('content', ''))}")

        return {
            "final_response": final_response,
            "session_metrics": session_metrics,
            "cleanup_data": cleanup_data,
            "end_time": end_time,
            "current_step": "finalize",
            "next_step": "__end__",
            "workflow_completed": True
        }

    except Exception as e:
        logger.error(f"最终化过程失败: {e}")
        # 即使最终化失败，也返回基本响应
        return {
            "final_response": _create_error_fallback_response(state),
            "session_metrics": {"error": "finalization_failed", "duration_seconds": 0},
            "cleanup_data": {},
            "end_time": time.time(),
            "current_step": "finalize",
            "next_step": "__end__",
            "error_message": f"最终化错误: {str(e)}",
            "workflow_completed": True
        }


def _prepare_final_response(state: GraphState) -> Dict[str, Any]:
    """准备最终响应"""

    # 使用精炼后的响应或初始响应
    response_text = state.get("refined_response") or state.get("initial_response", "")

    # 最终清理响应文本
    if response_text:
        response_text = _clean_final_response(response_text)

    # 检查是否有错误
    has_error = state.get("error_message") is not None

    # 构建响应对象
    response = {
        "content": response_text,
        "type": "medical_response",
        "timestamp": time.time(),
        "has_error": has_error,
        "confidence_score": state.get("confidence_score", 0.5),
        "sources_count": len(state.get("citations", []))
    }

    # 添加引用信息
    citations = state.get("citations", [])
    if citations:
        response["citations"] = citations
        response["sources_count"] = len(citations)

    # 添加医疗元数据
    medical_metadata = {
        "intent": state.get("medical_intent", "unknown"),
        "entities": state.get("medical_entities", []),
        "query_type": _classify_query_type(state["user_query"])
    }
    response["medical_metadata"] = medical_metadata

    # 如果有错误，添加错误信息
    if has_error:
        response["error_info"] = {
            "message": state.get("error_message", "未知错误"),
            "recovered": state.get("workflow_completed", False)
        }

    return response


def _clean_final_response(text: str) -> str:
    """最终清理响应文本"""
    if not text:
        return ""
    import re
    # 移除所有不自然的空格
    cleaned = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)

    # 清理标点符号周围的空格
    cleaned = re.sub(r'\s*([，。！？；：,.!?;:])\s*', r'\1', cleaned)

    # 移除行首尾空格
    lines = cleaned.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    cleaned = '\n'.join(cleaned_lines)

    return cleaned

def _classify_query_type(query: str) -> str:
    """简单分类查询类型"""
    query_lower = query.lower()

    if any(word in query_lower for word in ["怎么", "如何", "怎么办", "治疗方法", "治疗"]):
        return "treatment_inquiry"
    elif any(word in query_lower for word in ["症状", "表现", "特征"]):
        return "symptom_inquiry"
    elif any(word in query_lower for word in ["原因", "为什么", "病因"]):
        return "cause_inquiry"
    elif any(word in query_lower for word in ["预防", "避免", "防范"]):
        return "prevention_inquiry"
    else:
        return "general_inquiry"


def _collect_session_metrics(state: GraphState, start_time: float, end_time: float) -> Dict[str, Any]:
    """收集会话指标"""

    duration = end_time - start_time

    metrics = {
        "session_id": state.get("session_id", "unknown"),
        "duration_seconds": round(duration, 2),
        "retry_count": state.get("retry_count", 0),
        "steps_completed": _count_completed_steps(state),
        "documents_retrieved": len(state.get("retrieved_documents", [])),
        "documents_used": len(state.get("fusion_ranked_docs", [])),
        "final_confidence": state.get("confidence_score", 0.0),
        "has_errors": state.get("error_message") is not None,
        "error_count": 1 if state.get("error_message") else 0,
        "completion_status": "success" if not state.get("error_message") else "partial",
        "response_length": len(state.get("refined_response", "") or state.get("initial_response", ""))
    }

    # 添加性能指标
    if duration > 0:
        metrics["steps_per_second"] = round(metrics["steps_completed"] / duration, 2)

    return metrics


def _count_completed_steps(state: GraphState) -> int:
    """计算完成的步骤数量"""
    completed_steps = [
        "medical_intent_detection", "query_rewriting", "document_retrieval",
        "rag_fusion", "response_generation", "response_refinement"
    ]

    # 检查哪些步骤有有效数据
    count = 0
    for step in completed_steps:
        # 简单的完成检查逻辑
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


def _prepare_cleanup_data(state: GraphState) -> Dict[str, Any]:
    """准备清理数据"""

    cleanup = {
        "session_id": state.get("session_id"),
        "completion_time": time.time(),
        "resources_to_clean": [
            "temporary_embeddings",
            "intermediate_results"
        ],
        "persist_data": [
            "session_metrics",
            "final_response"
        ]
    }

    return cleanup


def _create_error_fallback_response(state: GraphState) -> Dict[str, Any]:
    """创建错误回退响应"""

    fallback_text = """抱歉，我在处理您的医疗咨询时遇到了技术困难。

虽然无法提供完整的专业回答，但我强烈建议您：

1. 咨询专业医生获取准确信息
2. 提供详细的症状描述
3. 如有紧急医疗问题，请立即就医

您的健康安全是最重要的！"""

    return {
        "content": fallback_text,
        "type": "error_fallback",
        "timestamp": time.time(),
        "has_error": True,
        "confidence_score": 0.1,
        "sources_count": 0,
        "medical_metadata": {
            "intent": "unknown",
            "entities": [],
            "query_type": "general_inquiry"
        },
        "error_info": {
            "message": "系统处理过程中出现意外错误",
            "recovered": False
        }
    }
    