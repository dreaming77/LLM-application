import logging
from typing import Dict, Any
from core.state.state import GraphState

logger = logging.getLogger(__name__)

async def error_handling_node(state: GraphState) -> Dict[str, Any]:
    """
    错误处理节点
    处理工作流中出现的错误，决定重试或终止
    """
    logger.info("执行错误处理节点")
    
    error_message = state.get("error_message", "未知错误")
    current_step = state["current_step"]
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    logger.error(f"步骤 {current_step} 发生错误: {error_message}")
    logger.info(f"重试计数: {retry_count}/{max_retries}")
    
    # 分析错误类型
    error_type = _classify_error(error_message, current_step)
    
    # 决定处理策略
    handling_strategy = _determine_handling_strategy(error_type, retry_count, max_retries)
    
    # 记录错误指标
    error_metrics = {
        "error_type": error_type,
        "error_step": current_step,
        "retry_count": retry_count,
        "handling_strategy": handling_strategy
    }
    
    result = {
        "error_metrics": error_metrics,
        "current_step": "error",
        "retry_count": retry_count + 1 if handling_strategy == "retry" else retry_count
    }
    
    # 根据策略决定下一步
    if handling_strategy == "retry" and retry_count < max_retries:
        result["next_step"] = "retry"
        result["error_message"] = f"将重试: {error_message}"
        logger.info(f"决定重试，第 {retry_count + 1} 次")
    else:
        result["next_step"] = "finalize"
        result["error_message"] = f"终止处理: {error_message}"
        logger.warning("达到最大重试次数或决定终止")
    
    return result

def _classify_error(error_message: str, current_step: str) -> str:
    """分类错误类型"""
    error_message_lower = error_message.lower()
    
    # 连接相关错误
    connection_errors = ["连接", "网络", "timeout", "connection", "network"]
    if any(keyword in error_message_lower for keyword in connection_errors):
        return "connection_error"
    
    # 模型相关错误
    model_errors = ["模型", "model", "gpu", "内存", "memory", "token"]
    if any(keyword in error_message_lower for keyword in model_errors):
        return "model_error"
    
    # 数据相关错误
    data_errors = ["数据", "data", "格式", "format", "json", "解析"]
    if any(keyword in error_message_lower for keyword in data_errors):
        return "data_error"
    
    # 医疗特定错误
    medical_errors = ["医疗", "安全", "safety", "风险", "risk"]
    if any(keyword in error_message_lower for keyword in medical_errors):
        return "medical_safety_error"
    
    # 根据步骤分类
    step_based_errors = {
        "medical_intent_detection": "intent_detection_error",
        "query_rewriting": "query_processing_error", 
        "document_retrieval": "retrieval_error",
        "rag_fusion": "fusion_error",
        "response_generation": "generation_error",
        "safety_check": "safety_check_error",
        "response_refinement": "refinement_error"
    }
    
    return step_based_errors.get(current_step, "unknown_error")

def _determine_handling_strategy(error_type: str, retry_count: int, max_retries: int) -> str:
    """确定错误处理策略"""
    
    # 可重试的错误类型
    retryable_errors = {
        "connection_error": 3,  # 连接错误可重试3次
        "model_error": 2,       # 模型错误可重试2次
        "data_error": 1,        # 数据错误可重试1次
        "retrieval_error": 2,   # 检索错误可重试2次
        "fusion_error": 2       # 融合错误可重试2次
    }
    
    # 不可重试的错误类型
    non_retryable_errors = [
        "medical_safety_error",  # 医疗安全错误不可重试
        "safety_check_error",    # 安全检查错误不可重试
        "unknown_error"          # 未知错误谨慎处理
    ]
    
    if error_type in non_retryable_errors:
        return "terminate"
    
    max_retries_for_error = retryable_errors.get(error_type, 1)
    
    if retry_count < min(max_retries_for_error, max_retries):
        return "retry"
    else:
        return "terminate"


async def retry_node(state: GraphState) -> Dict[str, Any]:
    """
    重试节点
    根据错误类型决定重试策略
    """
    logger.info("执行重试节点")

    error_metrics = state.get("error_metrics", {})
    error_type = error_metrics.get("error_type", "unknown_error")
    original_step = error_metrics.get("error_step", "medical_intent_detection")

    # 根据错误类型决定重试的起始步骤
    restart_step = _determine_restart_step(error_type, original_step)

    logger.info(f"错误类型: {error_type}, 从步骤 {restart_step} 重新开始")

    # 准备重试状态
    retry_state = {
        "current_step": "retry",
        "next_step": restart_step,
        "retry_count": state.get("retry_count", 0),
        "last_error": state.get("error_message"),
        "error_retry_context": f"重试原因: {error_type}"
    }

    # 根据错误类型进行特定的重试准备
    if error_type == "connection_error":
        retry_state.update(_prepare_connection_retry())
    elif error_type == "model_error":
        retry_state.update(_prepare_model_retry())

    return retry_state

def _determine_restart_step(error_type: str, original_step: str) -> str:
    """确定重试的起始步骤"""
    
    # 某些错误可以从更早的步骤重新开始
    restart_strategies = {
        "connection_error": "medical_intent_detection",  # 连接错误从头开始
        "model_error": "medical_intent_detection",       # 模型错误从头开始
        "data_error": original_step,                     # 数据错误从原步骤开始
        "retrieval_error": "document_retrieval",         # 检索错误从检索步骤开始
        "fusion_error": "rag_fusion",                    # 融合错误从融合步骤开始
        "generation_error": "response_generation",       # 生成错误从生成步骤开始
    }
    
    return restart_strategies.get(error_type, "medical_intent_detection")

def _prepare_connection_retry() -> Dict[str, Any]:
    """准备连接错误的重试"""
    return {
        "retry_delay": 5,  # 5秒延迟
        "connection_timeout_increased": True
    }

def _prepare_model_retry() -> Dict[str, Any]:
    """准备模型错误的重试"""
    return {
        "generation_temperature": 0.1,  # 降低温度提高稳定性
        "max_retries_reduced": True
    }
    