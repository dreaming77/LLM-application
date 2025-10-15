from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator

class GraphState(TypedDict):
    """
    图状态定义 - 跟踪整个工作流的状态
    """
    # 用户输入
    user_query: str  # 原始用户查询
    user_context: Optional[Dict[str, Any]]  # 用户上下文信息（如病史、偏好等）
    
    # 查询处理
    original_query: str  # 原始查询
    rewritten_queries: List[str]  # RAG Fusion重写后的多个查询
    current_query: str  # 当前使用的查询
    
    # 检索相关
    retrieved_documents: List[Dict[str, Any]]  # 检索到的文档列表
    document_scores: Dict[str, float]  # 文档得分映射
    fusion_ranked_docs: List[Dict[str, Any]]  # RAG Fusion重排序后的文档
    
    # 生成相关
    generation_context: str  # 用于生成的上下文
    initial_response: str  # 初始生成的回答
    refined_response: str  # 精炼后的最终回答
    citations: List[Dict[str, Any]]  # 引用的文档信息
    
    # 医疗特定字段
    medical_intent: Optional[str]  # 医疗意图分类（诊断、治疗、预防等）
    medical_entities: List[Dict[str, Any]]  # 医疗实体识别结果
    confidence_score: float  # 回答置信度
    
    # 工作流控制
    current_step: str  # 当前执行步骤
    next_step: str  # 下一步执行步骤
    error_message: Optional[str]  # 错误信息
    max_retries: int  # 最大重试次数
    retry_count: int  # 当前重试次数
    workflow_completed: bool  # 控制流是否完成
    
    # 对话历史
    conversation_history: Annotated[List[BaseMessage], operator.add]  # 对话消息历史
    
    # 系统配置
    search_k: int  # 检索文档数量
    fusion_top_k: int  # RAG Fusion保留文档数量
    generation_temperature: float  # 生成温度
    
    # 时间戳和元数据
    start_time: Optional[float]  # 开始时间
    end_time: Optional[float]  # 结束时间
    session_id: str  # 会话ID

    # 最终结果字段
    final_response: Optional[Dict[str, Any]]  # 最终响应
    session_metrics: Optional[Dict[str, Any]]  # 会话指标

    # 其他临时字段
    is_fallback: Optional[bool]
    error_metrics: Optional[Dict[str, Any]]
    last_error: Optional[str]
    error_retry_context: Optional[str]
