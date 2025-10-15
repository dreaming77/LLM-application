import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, validator
from config.settings import (
    DEFAULT_SEARCH_K, DEFAULT_FUSION_TOP_K, DEFAULT_GENERATION_TEMPERATURE,
    MAX_RETRY_ATTEMPTS, MIN_CONFIDENCE_SCORE
)

logger = logging.getLogger(__name__)

class WorkflowConfig(BaseModel):
    """工作流配置模型"""
    
    search_k: int = DEFAULT_SEARCH_K
    fusion_top_k: int = DEFAULT_FUSION_TOP_K
    generation_temperature: float = DEFAULT_GENERATION_TEMPERATURE
    max_retries: int = MAX_RETRY_ATTEMPTS
    session_timeout_minutes: int = 60
    
    @validator('search_k')
    def validate_search_k(cls, v):
        if not 1 <= v <= 50:
            raise ValueError('search_k必须在1-50之间')
        return v
    
    @validator('fusion_top_k')
    def validate_fusion_top_k(cls, v):
        if not 1 <= v <= 20:
            raise ValueError('fusion_top_k必须在1-20之间')
        return v
    
    @validator('generation_temperature')
    def validate_temperature(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError('generation_temperature必须在0.1-1.0之间')
        return v
    
    @validator('max_retries')
    def validate_max_retries(cls, v):
        if not 0 <= v <= 5:
            raise ValueError('max_retries必须在0-5之间')
        return v

class UserQuery(BaseModel):
    """用户查询模型"""
    
    query: str
    user_context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    config: Optional[WorkflowConfig] = None
    
    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) < 5:
            raise ValueError('查询过短，请提供更详细的问题')
        if len(v) > 2000:
            raise ValueError('查询过长，请简化表述')
        return v.strip()

class MedicalResponse(BaseModel):
    """医疗响应模型"""
    
    success: bool
    execution_id: str
    session_id: str
    duration_seconds: float
    response: str
    response_metadata: Dict[str, Any]
    session_metrics: Dict[str, Any]
    medical_metadata: Dict[str, Any]
    timestamp: float

class ErrorResponse(BaseModel):
    """错误响应模型"""
    
    success: bool = False
    error: str
    execution_id: str
    fallback_response: str
    timestamp: float
    suggested_actions: List[str]

class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_workflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """验证工作流配置"""
        try:
            validated_config = WorkflowConfig(**config)
            return validated_config.dict()
        except Exception as e:
            logger.warning(f"配置验证失败，使用默认值: {e}")
            return WorkflowConfig().dict()
    
    @staticmethod
    def validate_user_query(query_data: Dict[str, Any]) -> UserQuery:
        """验证用户查询"""
        try:
            return UserQuery(**query_data)
        except Exception as e:
            logger.error(f"用户查询验证失败: {e}")
            raise ValueError(f"查询验证失败: {str(e)}")
    
    @staticmethod
    def validate_medical_intent(intent: str) -> bool:
        """验证医疗意图"""
        valid_intents = {
            "diagnosis_inquiry", "treatment_inquiry", "symptom_analysis",
            "prevention_advice", "medication_guidance", "prognosis_question",
            "medical_test_interpretation", "lifestyle_advice", "second_opinion",
            "emergency_advice", "general_inquiry"
        }
        
        return intent in valid_intents
    
    @staticmethod
    def validate_confidence_score(score: float) -> bool:
        """验证置信度分数"""
        return 0.0 <= score <= 1.0

# 创建全局验证器实例
config_validator = ConfigValidator()
