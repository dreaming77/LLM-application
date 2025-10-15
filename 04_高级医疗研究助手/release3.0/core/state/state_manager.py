import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from core.state.state import GraphState

class StateManager:
    """
    状态管理器 - 负责状态的初始化、验证和持久化
    """
    
    def __init__(self):
        self.sessions: Dict[str, GraphState] = {}
    
    def initialize_state(self, user_query: str, user_context: Optional[Dict[str, Any]] = None, 
                    session_id: Optional[str] = None) -> GraphState:
        """
        初始化新的图状态 - 支持对话历史
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # 基础状态配置
        state: GraphState = {
            "user_query": user_query,
            "user_context": user_context or {},
            "original_query": user_query,
            "rewritten_queries": [],
            "current_query": user_query,
            "retrieved_documents": [],
            "document_scores": {},
            "fusion_ranked_docs": [],
            "generation_context": "",
            "initial_response": "",
            "refined_response": "",
            "citations": [],
            "medical_intent": None,
            "medical_entities": [],
            "confidence_score": 0.0,
            "current_step": "initialize",
            "next_step": "medical_intent_detection",
            "error_message": None,
            "max_retries": 3,
            "retry_count": 0,
            "workflow_completed": False,
            "conversation_history": [],  
            "search_k": 10,
            "fusion_top_k": 5,
            "generation_temperature": 0.3,
            "start_time": time.time(),
            "end_time": None,
            "session_id": session_id,
            "final_response": None,
            "session_metrics": None,
            "is_fallback": None,
            "error_metrics": None,
            "last_error": None,
            "error_retry_context": None
        }
        
        self.sessions[session_id] = state
        return state
    
    def update_state(self, session_id: str, updates: Dict[str, Any]) -> GraphState:
        """
        更新状态
        
        Args:
            session_id: 会话ID
            updates: 要更新的字段
            
        Returns:
            更新后的状态
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # 更新状态
        for key, value in updates.items():
            if key in self.sessions[session_id]:
                self.sessions[session_id][key] = value
        
        return self.sessions[session_id]
    
    def get_state(self, session_id: str) -> GraphState:
        """
        获取当前状态
        
        Args:
            session_id: 会话ID
            
        Returns:
            当前状态
        """
        return self.sessions.get(session_id)
    
    def validate_state(self, state: GraphState) -> bool:
        """
        验证状态完整性
        
        Args:
            state: 要验证的状态
            
        Returns:
            是否有效
        """
        required_fields = [
            "user_query", "original_query", "current_query", 
            "session_id", "current_step", "next_step"
        ]
        
        for field in required_fields:
            if field not in state or state[field] is None:
                return False
        
        return True
    
    def add_conversation_message(self, session_id: str, message: Any) -> None:
        """
        添加对话消息到历史
        
        Args:
            session_id: 会话ID
            message: 要添加的消息
        """
        if session_id in self.sessions:
            if "conversation_history" not in self.sessions[session_id]:
                self.sessions[session_id]["conversation_history"] = []
            self.sessions[session_id]["conversation_history"].append(message)
    
    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话指标
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话指标字典
        """
        state = self.get_state(session_id)
        if not state:
            return {}
        
        metrics = {
            "session_id": session_id,
            "duration": state.get("end_time", time.time()) - state.get("start_time", time.time()),
            "documents_retrieved": len(state.get("retrieved_documents", [])),
            "confidence_score": state.get("confidence_score", 0.0),
            "steps_completed": self._count_completed_steps(state),
            "error_occurred": state.get("error_message") is not None
        }
        
        return metrics
    
    def _count_completed_steps(self, state: GraphState) -> int:
        """计算完成的步骤数量"""
        completed_steps = [
            "medical_intent_detection", "query_rewriting", "document_retrieval",
            "rag_fusion", "response_generation", "safety_check", "response_refinement"
        ]
        
        return sum(1 for step in completed_steps if state.get("current_step") == step or 
                  state.get("next_step") == step)
    
    def cleanup_session(self, session_id: str) -> None:
        """
        清理会话
        
        Args:
            session_id: 要清理的会话ID
        """
        if session_id in self.sessions:
            # 记录结束时间
            self.sessions[session_id]["end_time"] = time.time()
            
            # 可以在这里添加持久化逻辑
            # 例如保存到数据库或文件
            
            # 从内存中移除（或保留用于分析）
            # del self.sessions[session_id]
