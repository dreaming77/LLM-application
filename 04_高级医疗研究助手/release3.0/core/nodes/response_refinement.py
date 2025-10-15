import logging
from typing import Dict, Any, List
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from core.model_manager import model_manager
from core.state.state import GraphState

logger = logging.getLogger(__name__)

# 响应精炼提示模板
RESPONSE_REFINEMENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="""你是一个医疗内容精炼专家。基于安全检查结果和改进建议，优化AI医疗回答。

精炼原则：
1. 保持专业准确性的同时提高安全性
2. 增强风险提示和免责声明
3. 改善语言表达，使其更加清晰易懂
4. 确保回答结构合理、重点突出
5. 保持对原始问题的高度相关性

精炼重点：
- 加强医疗免责声明
- 明确建议咨询专业医生
- 避免绝对化表述
- 突出紧急情况的处理建议
- 提高可读性和实用性"""),
    HumanMessage(content="""原始用户问题：{question}
初始AI回答：{initial_response}
医疗意图：{medical_intent}
安全检查问题：{safety_issues}
改进建议：{improvement_suggestions}
风险等级：{risk_level}

请基于以上信息精炼回答，使其更加安全、专业、有用：""")
])


async def response_refinement_node(state: GraphState) -> Dict[str, Any]:
    """
    响应精炼节点
    """
    logger.info("执行响应精炼节点")

    try:
        initial_response = state.get("initial_response", "")
        question = state["user_query"]
        medical_intent = state.get("medical_intent", "general_inquiry")
        confidence = state.get("confidence_score", 0.5)

        # 如果初始响应有问题，重新生成
        if not initial_response or len(initial_response.strip()) < 20:
            logger.warning("初始响应无效，创建默认响应")
            refined_response = _create_default_response(question, medical_intent)
            confidence = 0.3
        else:
            # 基础精炼：格式优化和质量提升
            refined_response = _basic_refinement(initial_response, question, medical_intent)
            # 精炼后略微提升置信度
            confidence = min(confidence + 0.1, 0.9)

        logger.info(f"响应精炼完成，长度: {len(refined_response)}")

        return {
            "refined_response": refined_response,
            "confidence_score": confidence,
            "current_step": "response_refinement",
            "next_step": "finalize"
        }

    except Exception as e:
        logger.error(f"响应精炼失败: {e}")
        # 创建错误回退响应
        return {
            "refined_response": _create_error_response(state["user_query"]),
            "confidence_score": 0.2,
            "current_step": "response_refinement",
            "next_step": "finalize",
            "error_message": f"响应精炼错误: {str(e)}"
        }

async def _refine_response(question: str, initial_response: str, medical_intent: str,
                          safety_issues: List[str], improvement_suggestions: List[str],
                          risk_level: str) -> str:
    """精炼响应内容"""
    
    # 如果是高风险情况，使用更严格的精炼
    if risk_level == "high":
        return await _high_risk_refinement(question, initial_response, medical_intent)
    
    # 普通精炼流程
    try:
        prompt = RESPONSE_REFINEMENT_PROMPT.format(
            question=question,
            initial_response=initial_response,
            medical_intent=medical_intent,
            safety_issues=", ".join(safety_issues) if safety_issues else "无",
            improvement_suggestions=", ".join(improvement_suggestions) if improvement_suggestions else "无",
            risk_level=risk_level
        )
        
        response = model_manager.generate_response(
            prompt,
            max_new_tokens=1000,
            temperature=0.2
        )
        
        refined = response[0] if isinstance(response, list) else str(response)
        return refined
        
    except Exception as e:
        logger.error(f"精炼过程失败: {e}")
        # 精炼失败时应用基本的安全改进
        return _apply_basic_safety_improvements(initial_response, risk_level)

async def _high_risk_refinement(question: str, initial_response: str, medical_intent: str) -> str:
    """高风险情况的精炼处理"""
    high_risk_prompt = f"""高风险医疗问题精炼要求：

用户问题：{question}
医疗意图：{medical_intent}
原始回答：{initial_response}

这是一个高风险医疗咨询，请严格按照以下要求精炼回答：

1. 开头必须明确声明："这是一个需要紧急医疗关注的情况，请立即寻求专业医疗帮助。"
2. 强调AI建议的局限性，不能替代专业医疗诊断
3. 提供具体的紧急行动建议（如拨打急救电话、前往急诊等）
4. 避免提供任何具体的医疗诊断或治疗建议
5. 保持冷静、专业的语气

请生成精炼后的回答："""

    try:
        response = model_manager.generation_model.chat(
            model_manager.generation_tokenizer,
            high_risk_prompt,
            max_new_tokens=800,
            temperature=0.1
        )
        
        return response[0] if isinstance(response, list) else str(response)
    except Exception as e:
        logger.error(f"高风险精炼失败: {e}")
        return _apply_emergency_safety_template(initial_response)

def _apply_basic_safety_improvements(response: str, risk_level: str) -> str:
    """应用基本的安全改进"""
    improved_response = response
    
    # 确保包含免责声明
    disclaimer = "\n\n重要提示：以上信息仅供参考，不能替代专业医疗建议。如有医疗问题，请咨询专业医生。"
    if disclaimer not in improved_response:
        improved_response += disclaimer
    
    # 根据风险等级调整语气
    if risk_level == "medium":
        # 在开头添加风险提示
        risk_notice = "请注意：以下信息需要谨慎对待，建议咨询医疗专业人士确认。\n\n"
        if risk_notice not in improved_response:
            improved_response = risk_notice + improved_response
    
    return improved_response


def _basic_refinement(response: str, question: str, medical_intent: str) -> str:
    """基础响应精炼 - 修复空格问题"""
    # 首先清理文本
    refined = response

    # 清理多余的空白字符
    refined = re.sub(r'\n\s*\n', '\n\n', refined)

    # 确保以合适的标点结束
    if not refined.endswith(('.', '!', '?', '。', '！', '？')):
        refined += '。'

    # 根据医疗意图调整语气
    if medical_intent == "emergency_advice":
        if "紧急" not in refined and "立即" not in refined:
            refined = "重要提醒：这可能是紧急医疗情况。\n\n" + refined

    return refined

def _create_default_response(question: str, medical_intent: str) -> str:
    """创建默认响应"""
    if medical_intent == "emergency_advice":
        return f"""关于"{question}"，这可能是紧急医疗情况。

紧急建议：
• 立即联系专业医生或前往急诊科
• 不要延误寻求医疗帮助
• 如有生命危险症状，请拨打急救电话

注意：在线咨询不能处理紧急医疗情况。"""
    else:
        return f"""关于"{question}"，我目前提供以下一般性信息：

由于您的具体情况未知，建议：
1. 咨询专业医疗人员获取个性化建议
2. 提供详细的症状和病史信息
3. 遵循医生的专业指导

重要提示：请勿依赖AI回答进行医疗决策，务必咨询合格医疗专业人员。"""


def _create_error_response(question: str) -> str:
    """创建错误响应"""
    return f"""抱歉，在处理您关于"{question}"的咨询时遇到了技术问题。

虽然无法提供完整的专业回答，但建议您：
• 咨询专业医生获取准确信息
• 提供详细的症状描述
• 如有紧急情况请立即就医

您的健康安全是最重要的！"""

def _apply_emergency_safety_template(response: str) -> str:
    """应用紧急情况安全模板"""
    emergency_template = """重要安全提示：您描述的情况可能需要紧急医疗关注。

鉴于AI的局限性，我强烈建议您：

1. 如有生命危险症状（如呼吸困难、胸痛、大出血等），请立即拨打急救电话或前往最近医院的急诊科
2. 不要依赖AI建议进行医疗决策
3. 尽快咨询专业医生获取准确诊断

您的健康安全是最重要的，请优先寻求专业医疗帮助。"""
    
    return emergency_template

def _calculate_refined_confidence(original_confidence: float, safety_issue_count: int) -> float:
    """计算精炼后的置信度"""
    # 精炼通常会提高置信度，但安全问题越多，置信度提升越小
    improvement_factor = max(0.1, 1.0 - (safety_issue_count * 0.05))
    refined_confidence = min(1.0, original_confidence + (0.1 * improvement_factor))
    
    return refined_confidence
    