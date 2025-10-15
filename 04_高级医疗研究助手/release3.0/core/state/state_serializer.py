import json
from typing import Dict, Any
from datetime import datetime
from core.state.state import GraphState

class StateSerializer:
    """
    状态序列化器 - 负责状态的序列化和反序列化
    """
    
    @staticmethod
    def serialize(state: GraphState) -> Dict[str, Any]:
        """
        序列化状态为可存储格式
        
        Args:
            state: 图状态
            
        Returns:
            序列化后的字典
        """
        serialized = {}
        
        # 序列化基本字段
        for key, value in state.items():
            if key == "conversation_history":
                # 特殊处理消息历史
                serialized[key] = [str(msg) for msg in value]
            elif key in ["start_time", "end_time"] and value is not None:
                # 转换时间戳为可读格式
                serialized[key] = datetime.fromtimestamp(value).isoformat()
            else:
                serialized[key] = value
        
        return serialized
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> GraphState:
        """
        从序列化数据恢复状态
        
        Args:
            data: 序列化数据
            
        Returns:
            恢复的图状态
        """
        state: GraphState = {
            "user_query": data.get("user_query", ""),
            "user_context": data.get("user_context", {}),
            "original_query": data.get("original_query", ""),
            "rewritten_queries": data.get("rewritten_queries", []),
            "current_query": data.get("current_query", ""),
            "retrieved_documents": data.get("retrieved_documents", []),
            "document_scores": data.get("document_scores", {}),
            "fusion_ranked_docs": data.get("fusion_ranked_docs", []),
            "generation_context": data.get("generation_context", ""),
            "initial_response": data.get("initial_response", ""),
            "refined_response": data.get("refined_response", ""),
            "citations": data.get("citations", []),
            "medical_intent": data.get("medical_intent"),
            "medical_entities": data.get("medical_entities", []),
            "confidence_score": data.get("confidence_score", 0.0),
            "safety_check_passed": data.get("safety_check_passed", True),
            "current_step": data.get("current_step", "initialize"),
            "next_step": data.get("next_step", "medical_intent_detection"),
            "error_message": data.get("error_message"),
            "max_retries": data.get("max_retries", 3),
            "retry_count": data.get("retry_count", 0),
            "conversation_history": [],  # 消息历史需要特殊处理
            "search_k": data.get("search_k", 10),
            "fusion_top_k": data.get("fusion_top_k", 5),
            "generation_temperature": data.get("generation_temperature", 0.3),
            "start_time": StateSerializer._parse_timestamp(data.get("start_time")),
            "end_time": StateSerializer._parse_timestamp(data.get("end_time")),
            "session_id": data.get("session_id", "")
        }
        
        return state
    
    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> float:
        """解析时间戳字符串"""
        if timestamp_str is None:
            return None
        try:
            dt = datetime.fromisoformat(timestamp_str)
            return dt.timestamp()
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def to_json(state: GraphState) -> str:
        """转换为JSON字符串"""
        serialized = StateSerializer.serialize(state)
        return json.dumps(serialized, ensure_ascii=False, indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> GraphState:
        """从JSON字符串恢复"""
        data = json.loads(json_str)
        return StateSerializer.deserialize(data)
