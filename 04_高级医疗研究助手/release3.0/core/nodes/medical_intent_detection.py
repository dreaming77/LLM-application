import logging
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from core.model_manager import model_manager
from core.state.state import GraphState

logger = logging.getLogger(__name__)

# 医疗意图分类提示模板 - 使用字符串格式
INTENT_CLASSIFICATION_TEMPLATE = """你是一个专业的医疗意图分类器。你的任务是根据用户的问题准确识别医疗意图。

可用的意图类别：
1. diagnosis_inquiry - 诊断咨询：询问症状可能对应的疾病或诊断方法
2. treatment_inquiry - 治疗咨询：询问特定疾病的治疗方法、药物、手术等
3. symptom_analysis - 症状分析：描述症状请求分析可能原因
4. prevention_advice - 预防建议：询问疾病预防措施或健康维护
5. medication_guidance - 用药指导：询问药物用法、副作用、相互作用等
6. prognosis_question - 预后咨询：询问疾病发展前景或康复时间
7. medical_test_interpretation - 检查结果解读：询问化验单、影像学检查结果含义
8. lifestyle_advice - 生活方式建议：询问饮食、运动等生活习惯对健康的影响
9. second_opinion - 第二意见：对已有诊断或治疗方案寻求确认
10. emergency_advice - 紧急建议：需要立即医疗干预的紧急情况
11. general_inquiry - 一般咨询：其他医疗相关问题

请严格按照以上类别进行分类，只返回类别名称，不要额外解释。

用户问题：{question}

医疗意图类别："""


async def medical_intent_detection_node(state: GraphState) -> Dict[str, Any]:
    """
    医疗意图识别节点
    分析用户问题的医疗意图，为后续处理提供方向
    """
    logger.info("执行医疗意图识别节点")

    try:
        user_query = state["user_query"]

        # 构建提示
        prompt = INTENT_CLASSIFICATION_TEMPLATE.format(question=user_query)

        # 使用模型进行意图分类
        response = model_manager.generate_response(
            prompt,
            max_tokens=50,
            temperature=0.1
        )

        # 提取意图类别
        intent_text = response.strip() if response else ""

        # 映射到标准意图类别
        medical_intent = _map_intent_category(intent_text)

        # 识别医疗实体
        medical_entities = await _extract_medical_entities(user_query)

        logger.info(f"识别到医疗意图: {medical_intent}")
        logger.info(f"识别到医疗实体: {[entity['entity'] for entity in medical_entities]}")

        return {
            "medical_intent": medical_intent,
            "medical_entities": medical_entities,
            "current_step": "medical_intent_detection",
            "next_step": "query_rewriting",
            "confidence_score": _calculate_intent_confidence(intent_text, medical_intent)
        }

    except Exception as e:
        logger.error(f"医疗意图识别失败: {e}")
        return {
            "medical_intent": "general_inquiry",  # 默认类别
            "medical_entities": [],
            "current_step": "medical_intent_detection",
            "next_step": "error",
            "error_message": f"意图识别错误: {str(e)}",
            "confidence_score": 0.1
        }


def _map_intent_category(intent_text: str) -> str:
    """将模型输出映射到标准意图类别"""
    intent_mapping = {
        "diagnosis_inquiry": "diagnosis_inquiry",
        "treatment_inquiry": "treatment_inquiry",
        "symptom_analysis": "symptom_analysis",
        "prevention_advice": "prevention_advice",
        "medication_guidance": "medication_guidance",
        "prognosis_question": "prognosis_question",
        "medical_test_interpretation": "medical_test_interpretation",
        "lifestyle_advice": "lifestyle_advice",
        "second_opinion": "second_opinion",
        "emergency_advice": "emergency_advice",
        "general_inquiry": "general_inquiry"
    }

    # 直接匹配
    if intent_text in intent_mapping:
        return intent_text

    # 关键词匹配
    intent_text_lower = intent_text.lower()
    for key, value in intent_mapping.items():
        if key in intent_text_lower:
            return value

    return "general_inquiry"  # 默认类别


async def _extract_medical_entities(question: str) -> List[Dict[str, Any]]:
    """提取医疗实体（疾病、症状、药物等）"""
    # 这里可以使用专业的医疗NER模型，暂时使用规则+模型的方式
    entities = []

    # 简单的关键词匹配（实际应该使用专业医疗词典）
    medical_keywords = {
        "疾病": ["糖尿病", "高血压", "冠心病", "哮喘", "肺炎", "胃炎", "关节炎", "感冒", "发烧", "头痛"],
        "症状": ["头痛", "发热", "咳嗽", "胸痛", "腹痛", "乏力", "头晕", "恶心", "呕吐", "腹泻"],
        "检查": ["血常规", "CT", "MRI", "超声", "心电图", "胃镜", "血压", "血糖"],
        "药物": ["阿司匹林", "胰岛素", "降压药", "抗生素", "止痛药", "感冒药"]
    }

    for category, keywords in medical_keywords.items():
        for keyword in keywords:
            if keyword in question:
                entities.append({
                    "entity": keyword,
                    "category": category,
                    "start_pos": question.find(keyword),
                    "end_pos": question.find(keyword) + len(keyword)
                })

    return entities


def _calculate_intent_confidence(intent_text: str, mapped_intent: str) -> float:
    """计算意图识别置信度"""
    if mapped_intent != "general_inquiry" and intent_text.strip():
        return 0.9
    elif mapped_intent != "general_inquiry":
        return 0.7
    else:
        return 0.3
