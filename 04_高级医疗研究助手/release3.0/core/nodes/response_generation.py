import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from core.model_manager import model_manager
from core.state.state import GraphState
import re

logger = logging.getLogger(__name__)

# 响应生成提示模板
RESPONSE_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
你是一个专业的医疗AI助手，基于提供的医疗知识参考和对话历史为用户提供准确、专业、负责任的回答。

回答要求：
1. 专业准确：基于参考信息，使用准确的医学术语
2. 考虑上下文：结合对话历史理解用户的完整意图
3. 清晰易懂：用通俗语言解释专业概念
4. 全面覆盖：回答用户问题的各个方面
5. 负责任：强调专业医疗建议的重要性
6. 结构化：使用清晰的段落结构，重要信息突出

重要格式要求：
1. 提供自然流畅的回答内容
2. 确保文字连贯，不要在中文词语中间添加空格
3. 如果对话历史中有相关信息，请合理引用

回答结构：
- 一、简要确认理解用户问题（可结合历史上下文）
- 二、基于参考信息和对话历史提供详细解答  
- 三、提供具体的、详细的下一步行动建议
- 四、强调需要专业医疗咨询

请基于参考信息和对话历史生成回答，不要编造信息。如果参考信息不足，请明确说明。"""),
    HumanMessage(content="""对话历史：
{conversation_history}

参考信息：
{context}

当前用户问题：
{question}

医疗意图：
{medical_intent}

请生成专业、负责任的医疗回答：""")
])


async def response_generation_node(state: GraphState) -> Dict[str, Any]:
    """
    响应生成节点 - 支持对话历史
    """
    logger.info("执行响应生成节点")

    try:
        context = state.get("generation_context", "")
        question = state["user_query"]
        medical_intent = state.get("medical_intent", "general_inquiry")
        temperature = state.get("generation_temperature", 0.3)
        
        # 获取对话历史
        conversation_history = state.get("conversation_history", [])
        conversation_history_text = _format_conversation_history(conversation_history)

        # 检查上下文是否充足
        if not context and not conversation_history_text:
            logger.warning("上下文信息和对话历史都为空，使用备用方案")
            return await _generate_fallback_response(question, medical_intent)

        # 生成专业回答
        prompt = RESPONSE_GENERATION_PROMPT.format()
        prompt = prompt.format(
            context=context,
            question=question,
            medical_intent=medical_intent,
            conversation_history=conversation_history_text
        )

        # 确保模型管理器可用
        if not hasattr(model_manager, 'generate_response') or model_manager.generation_pipeline is None:
            logger.error("生成模型不可用，使用备用方案")
            return await _generate_fallback_response(question, medical_intent)

        response = model_manager.generate_response(
            prompt,
            max_tokens=800,  # 增加token限制以容纳更长的对话
            temperature=temperature
        )

        # 处理响应格式
        if isinstance(response, list) and len(response) > 0:
            initial_response = response[0].get('generated_text', '') if isinstance(response[0], dict) else str(
                response[0])
        else:
            initial_response = str(response) if response else ""

        # 如果响应为空，使用备用方案
        if not initial_response or len(initial_response.strip()) < 10:
            logger.warning("生成的响应过短，使用备用方案")
            return await _generate_fallback_response(question, medical_intent)

        # 预处理响应文本
        initial_response = _preprocess_response(initial_response)

        # 提取引用信息
        citations = _extract_citations(initial_response, state.get("fusion_ranked_docs", []))

        logger.info(f"响应生成完成，长度: {len(initial_response)}")
        logger.info(f"使用了 {len(conversation_history)} 条历史对话")

        return {
            "initial_response": initial_response,
            "citations": citations,
            "current_step": "response_generation",
            "next_step": "response_refinement",
            "confidence_score": _calculate_response_confidence(initial_response, context)
        }

    except Exception as e:
        logger.error(f"响应生成失败: {e}")
        return await _generate_fallback_response(state["user_query"], state.get("medical_intent", "general_inquiry"))


def _preprocess_response(text: str) -> str:
    """预处理响应文本"""
    if not text:
        return ""

    # 移除Assistant:等前缀
    text = re.sub(r'^(Assistant:|助手:|AI:|AI助手:)\s*', '', text.strip())

    # 清理多余空格
    text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

async def _generate_fallback_response(question: str, medical_intent: str) -> Dict[str, Any]:
    """生成备用回答（当检索信息不足或生成失败时）"""
    logger.info(f"生成备用回答，问题: {question}, 意图: {medical_intent}")

    # 基于医疗意图生成不同的备用回答
    if medical_intent == "emergency_advice":
        fallback_text = """您描述的情况可能需要紧急医疗关注。

    紧急建议：
    1. 如有生命危险症状，请立即拨打急救电话或前往急诊科
    2. 不要依赖在线咨询进行紧急医疗决策
    3. 尽快寻求专业医疗帮助

    您的健康安全是最重要的！"""
    else:
        fallback_text = f"""关于"{question}"，基于现有信息，我为您提供以下参考：

    由于信息有限，建议您：
    • 咨询专业医生获取准确诊断
    • 提供更详细的症状描述
    • 结合个人具体情况寻求医疗帮助

    重要提示：以上信息仅供参考，不能替代专业医疗建议。请咨询医疗专业人士获取个性化指导。"""

    return {
        "initial_response": fallback_text,
        "citations": [],
        "current_step": "response_generation",
        "next_step": "response_refinement",
        "confidence_score": 0.4,  # 备用回答的置信度较低
        "is_fallback": True  # 标记为备用回答
    }


def _extract_citations(response: str, ranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从响应中提取引用信息"""
    citations = []

    if not ranked_docs:
        return citations

    for i, doc in enumerate(ranked_docs[:3], 1):
        # 检查文档是否有有效内容
        if not doc.get('question') or not doc.get('answer'):
            continue

        # 简单的关键词匹配
        key_phrases = [
            doc["question"][:30],
            doc.get("label", ""),
            doc.get("related_diseases", "")[:20]
        ]

        for phrase in key_phrases:
            if phrase and phrase.strip() and phrase in response:
                citations.append({
                    "document_id": doc.get("id", f"doc_{i}"),
                    "question": doc["question"][:100],
                    "score": doc.get("score", 0),
                    "label": doc.get("label", ""),
                    "matched_phrase": phrase
                })
                break

    return citations

def _format_conversation_history(conversation_history: List[Dict]) -> str:
    """格式化对话历史为文本"""
    if not conversation_history:
        return "无对话历史"
    
    formatted_history = []
    for i, message in enumerate(conversation_history[-6:], 1):  # 只保留最近6条消息
        role = "用户" if message.get("role") == "user" else "助手"
        content = message.get("content", "")
        formatted_history.append(f"{i}. {role}: {content}")
    
    return "\n".join(formatted_history)


def _calculate_response_confidence(response: str, context: str) -> float:
    """计算回答置信度"""
    
    if not response or response.strip() == "":
        return 0.0
    
    confidence_score = 0.0
    response = response.strip()
    
    # 1. 回答质量评估 (权重: 40%)
    quality_score = 0.0
    
    # 长度合理性 (不是越长越好，也不是越短越好)
    response_length = len(response)
    if 50 <= response_length <= 500:  # 适中的长度范围
        length_score = 0.3
    elif response_length < 20:  # 太短的回复
        length_score = 0.1
    elif response_length > 1000:  # 可能过于冗长
        length_score = 0.2
    else:
        length_score = 0.25
    
    # 结构完整性
    sentence_count = response.count('。') + response.count('!') + response.count('?')
    if sentence_count >= 2:  # 包含多个句子，结构更完整
        structure_score = 0.4
    else:
        structure_score = 0.2
    
    # 专业性词汇密度
    medical_terms = ["建议", "治疗", "诊断", "症状", "药物", "检查", "就医", "专业", "医生", "医院"]
    term_count = sum(1 for term in medical_terms if term in response)
    term_density = term_count / max(1, len(response.split()))
    term_score = min(term_density * 2, 0.3)  # 专业词汇密度得分
    
    quality_score = (length_score + structure_score + term_score) * 0.4
    
    # 2. 上下文相关性评估 (权重: 30%)
    relevance_score = 0.0
    
    if context and len(context) > 0:
        # 简单的关键词匹配
        context_words = set(context.split()[:20])  # 取前20个词作为上下文关键词
        response_words = set(response.split())
        common_words = context_words.intersection(response_words)
        
        if len(context_words) > 0:
            word_overlap = len(common_words) / len(context_words)
            relevance_score = min(word_overlap * 0.5, 0.3)
    
    # 3. 可信度标记评估 (权重: 20%)
    credibility_score = 0.0
    
    # 积极的专业标记
    positive_indicators = [
        "研究表明", "临床经验", "根据指南", "专业建议", "循证医学",
        "常见方案", "标准治疗", "医学共识"
    ]
    
    # 谨慎的免责声明 (适度的免责声明增加可信度，过度则降低)
    disclaimer_terms = ["咨询医生", "专业建议", "仅供参考", "不能替代", "建议就医"]
    disclaimer_count = sum(1 for term in disclaimer_terms if term in response)
    
    positive_count = sum(1 for indicator in positive_indicators if indicator in response)
    
    if positive_count > 0:
        credibility_score += 0.15
    if disclaimer_count == 1:  # 适度的免责声明
        credibility_score += 0.05
    elif disclaimer_count > 1:  # 过多的免责声明可能降低可信度
        credibility_score += 0.02
    
    credibility_score = min(credibility_score, 0.2)
    
    # 4. 回答确定性评估 (权重: 10%)
    certainty_score = 0.0
    
    # 不确定的表达
    uncertain_phrases = ["可能", "也许", "大概", "不一定", "不确定", "疑似"]
    uncertain_count = sum(1 for phrase in uncertain_phrases if phrase in response)
    
    # 确定的表达
    certain_phrases = ["明确", "确定", "肯定", "必须", "一定", "毫无疑问"]
    certain_count = sum(1 for phrase in certain_phrases if phrase in response)
    
    if certain_count > uncertain_count:
        certainty_score = 0.1
    elif certain_count == uncertain_count:
        certainty_score = 0.05
    else:
        certainty_score = 0.02
    
    # 综合计算最终置信度
    confidence_score = quality_score + relevance_score + credibility_score + certainty_score
    
    # 最终调整和限制
    confidence_score = min(max(confidence_score, 0.0), 1.0)
    
    # 对极端情况做特殊处理
    if "不知道" in response or "不清楚" in response or "无法回答" in response:
        confidence_score *= 0.5
    
    return round(confidence_score, 2)

    