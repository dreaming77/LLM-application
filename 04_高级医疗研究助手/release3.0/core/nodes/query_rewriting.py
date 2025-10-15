import logging
from typing import Dict, Any, List
import json
import re
from core.model_manager import model_manager
from core.state.state import GraphState

logger = logging.getLogger(__name__)

# 查询重写提示模板
QUERY_REWRITING_TEMPLATE = """你是一个专业的医疗查询重写专家。基于用户的原始问题和医疗意图，生成3-5个相关的查询变体用于信息检索。

重写原则：
1. 保持医疗专业性：使用准确的医学术语
2. 多角度覆盖：从不同角度重新表达相同问题
3. 细化扩展：将宽泛问题分解为具体子问题
4. 同义词替换：使用医学术语同义词
5. 上下文补充：基于医疗意图补充相关上下文

原始问题：{question}
医疗意图：{medical_intent}
医疗实体：{medical_entities}

请生成3-5个重写查询，以JSON格式返回：
{{
    "rewritten_queries": ["查询1", "查询2", "查询3"]
}}

重写查询："""


async def query_rewriting_node(state: GraphState) -> Dict[str, Any]:
    """
    查询重写节点
    基于医疗意图生成多个相关查询，用于RAG Fusion
    """
    logger.info("执行查询重写节点")

    try:
        question = state["user_query"]
        medical_intent = state["medical_intent"]
        medical_entities = state["medical_entities"]

        # 准备实体信息
        entities_text = ", ".join([entity["entity"] for entity in medical_entities])

        # 构建提示
        prompt = QUERY_REWRITING_TEMPLATE.format(
            question=question,
            medical_intent=medical_intent,
            medical_entities=entities_text
        )

        # 生成重写查询
        response = model_manager.generate_response(
            prompt,
            max_tokens=300,
            temperature=0.3
        )

        # 解析响应
        rewritten_queries = _parse_rewritten_queries(response)

        # 确保包含原始查询
        if question not in rewritten_queries:
            rewritten_queries.insert(0, question)

        logger.info(f"生成 {len(rewritten_queries)} 个重写查询")
        for i, query in enumerate(rewritten_queries, 1):
            logger.debug(f"查询 {i}: {query}")

        return {
            "rewritten_queries": rewritten_queries,
            "current_query": question,  # 默认使用原始查询
            "current_step": "query_rewriting",
            "next_step": "document_retrieval"
        }

    except Exception as e:
        logger.error(f"查询重写失败: {e}")
        # 失败时使用原始查询
        return {
            "rewritten_queries": [state["user_query"]],
            "current_query": state["user_query"],
            "current_step": "query_rewriting",
            "next_step": "document_retrieval",
            "error_message": f"查询重写错误: {str(e)}"
        }


def _parse_rewritten_queries(response: str) -> List[str]:
    """解析模型返回的重写查询"""
    try:
        if not response:
            return []

        # 尝试解析JSON
        json_match = re.search(r'\{.*?"rewritten_queries".*?\}', response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group()
                data = json.loads(json_str)
                queries = data.get("rewritten_queries", [])
                if queries:
                    return queries
            except json.JSONDecodeError:
                logger.warning("JSON解析失败，尝试其他解析方法")

        # 如果JSON解析失败，尝试按行提取
        queries = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # 匹配带引号的查询或带编号的查询
            if line and len(line) > 10 and not line.startswith(('{', '}', '```')):
                # 清理编号和引号
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
                clean_line = re.sub(r'^["\']|["\']$', '', clean_line)
                clean_line = clean_line.strip()
                if clean_line and clean_line not in queries and len(clean_line) > 5:
                    queries.append(clean_line)

        return queries[:5]  # 最多返回5个查询

    except Exception as e:
        logger.warning(f"重写查询解析失败，使用备用方案: {e}")
        return []
        