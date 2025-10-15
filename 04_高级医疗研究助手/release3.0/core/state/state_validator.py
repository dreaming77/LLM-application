from typing import List, Dict, Any
from core.state.state import GraphState

class StateValidator:
    """
    状态验证器 - 负责验证状态的一致性和完整性
    """
    
    @staticmethod
    def validate_step_transition(current_step: str, next_step: str) -> bool:
        """
        验证步骤转换是否有效
        
        Args:
            current_step: 当前步骤
            next_step: 下一步骤
            
        Returns:
            转换是否有效
        """
        valid_transitions = {
            "initialize": ["medical_intent_detection", "error"],
            "medical_intent_detection": ["query_rewriting", "error"],
            "query_rewriting": ["document_retrieval", "error"],
            "document_retrieval": ["rag_fusion", "error"],
            "rag_fusion": ["response_generation", "error"],
            "response_generation": ["safety_check", "error"],
            "safety_check": ["response_refinement", "error", "regenerate"],
            "response_refinement": ["finalize", "error"],
            "regenerate": ["response_generation", "error"],
            "error": ["retry", "finalize"],
            "retry": ["medical_intent_detection", "error"],
            "finalize": []
        }
        
        return next_step in valid_transitions.get(current_step, [])
    
    @staticmethod
    def validate_state_completeness(state: GraphState, current_step: str) -> List[str]:
        """
        验证状态完整性
        
        Args:
            state: 图状态
            current_step: 当前步骤
            
        Returns:
            缺失的必需字段列表
        """
        step_requirements = {
            "medical_intent_detection": ["user_query"],
            "query_rewriting": ["medical_intent"],
            "document_retrieval": ["rewritten_queries"],
            "rag_fusion": ["retrieved_documents"],
            "response_generation": ["fusion_ranked_docs"],
            "safety_check": ["initial_response"],
            "response_refinement": ["initial_response", "safety_check_passed"]
        }
        
        required_fields = step_requirements.get(current_step, [])
        missing_fields = []
        
        for field in required_fields:
            if not state.get(field):
                missing_fields.append(field)
        
        return missing_fields
    
    @staticmethod
    def validate_medical_safety(state: GraphState) -> Dict[str, Any]:
        """
        验证医疗安全性
        
        Args:
            state: 图状态
            
        Returns:
            安全性检查结果
        """
        safety_issues = []
        
        # 检查置信度
        if state.get("confidence_score", 0) < 0.5:
            safety_issues.append("低置信度回答")
        
        # 检查是否有明确的医疗声明
        response = state.get("refined_response", "") or state.get("initial_response", "")
        if not response:
            safety_issues.append("空回答")
        
        # 检查是否有适当的免责声明
        disclaimer_keywords = ["建议", "咨询", "医生", "专业", "仅供参考"]
        has_disclaimer = any(keyword in response for keyword in disclaimer_keywords)
        if not has_disclaimer:
            safety_issues.append("缺少医疗免责声明")
        
        return {
            "is_safe": len(safety_issues) == 0,
            "issues": safety_issues,
            "requires_human_review": len(safety_issues) > 1
        }
        