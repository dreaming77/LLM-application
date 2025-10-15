import logging
from typing import Dict, Any, List
from core.state.state import GraphState

logger = logging.getLogger(__name__)

async def rag_fusion_node(state: GraphState) -> Dict[str, Any]:
    """
    RAG Fusion节点
    使用加权RRF算法对多个查询的检索结果进行融合重排序
    """
    logger.info("执行RAG Fusion节点")
    
    try:
        retrieved_documents = state["retrieved_documents"]
        fusion_top_k = state.get("fusion_top_k", 5)
        
        if not retrieved_documents:
            logger.warning("没有检索到文档，跳过RAG Fusion")
            return {
                "fusion_ranked_docs": [],
                "current_step": "rag_fusion",
                "next_step": "response_generation"
            }
        
        # 按查询分组文档
        query_groups = {}
        for doc in retrieved_documents:
            query_idx = doc["query_index"]
            if query_idx not in query_groups:
                query_groups[query_idx] = []
            query_groups[query_idx].append(doc)
        # 计算每个查询的权重（基于查询与原始问题的相似度）
        query_weights = _calculate_query_weights(query_groups, state["user_query"])
        
        # 应用RRF算法进行融合排序
        fused_documents = _reciprocal_rank_fusion(query_groups, query_weights, k=60)
        # 取前N个文档
        ranked_docs = fused_documents[:fusion_top_k]
        
        logger.info(f"RAG Fusion完成，排名前 {len(ranked_docs)} 个文档")
        
        # 构建生成上下文
        generation_context = _build_generation_context(ranked_docs, state["user_query"])
        
        return {
            "fusion_ranked_docs": ranked_docs,
            "generation_context": generation_context,
            "current_step": "rag_fusion",
            "next_step": "response_generation"
        }
        
    except Exception as e:
        logger.error(f"RAG Fusion失败: {e}")
        # 失败时使用原始排序
        original_docs = retrieved_documents[:state.get("fusion_top_k", 5)]
        return {
            "fusion_ranked_docs": original_docs,
            "generation_context": _build_generation_context(original_docs, state["user_query"]),
            "current_step": "rag_fusion",
            "next_step": "response_generation",
            "error_message": f"RAG Fusion错误: {str(e)}"
        }

def _calculate_query_weights(query_groups: Dict[int, List], original_query: str) -> Dict[int, float]:
    """计算每个查询的权重"""
    weights = {}
    total_queries = len(query_groups)
    
    # 简单的权重分配：第一个查询（原始查询）权重最高
    for i, query_idx in enumerate(query_groups.keys()):
        if i == 0:  # 原始查询
            weights[query_idx] = 1.0
        else:  # 重写查询
            weights[query_idx] = 0.7 - (i * 0.1)  # 递减权重
    
    # 归一化权重
    total_weight = sum(weights.values())
    if total_weight > 0:
        for query_idx in weights:
            weights[query_idx] /= total_weight
    
    return weights

def _reciprocal_rank_fusion(query_groups: Dict[int, List], weights: Dict[int, float], k: int = 60) -> List[Dict[str, Any]]:
    """应用加权 Reciprocal Rank Fusion 算法"""
    
    # 收集所有唯一文档
    all_documents = {}
    for query_idx, documents in query_groups.items():
        for rank, doc in enumerate(documents):
            doc_key = doc["question"]  # 使用问题作为唯一标识
            if doc_key not in all_documents:
                all_documents[doc_key] = doc.copy()
                all_documents[doc_key]["fusion_score"] = 0.0
                all_documents[doc_key]["appearances"] = 0
    
    # 计算每个查询中文档的RRF分数
    for query_idx, documents in query_groups.items():
        query_weight = weights.get(query_idx, 0.1)
        
        for rank, doc in enumerate(documents):
            doc_key = doc["question"]
            if doc_key in all_documents:
                # RRF公式: score += weight / (k + rank)
                rrf_score = query_weight / (k + rank + 1)
                all_documents[doc_key]["fusion_score"] += rrf_score
                all_documents[doc_key]["appearances"] += 1
    
    # 转换为列表并按融合分数排序
    fused_list = list(all_documents.values())
    fused_list.sort(key=lambda x: x["fusion_score"], reverse=True)
    
    return fused_list

def _build_generation_context(ranked_docs: List[Dict[str, Any]], user_query: str) -> str:
    """构建用于生成的上下文"""
    context_parts = []
    
    context_parts.append(f"用户问题: {user_query}\n\n")
    context_parts.append("相关医疗知识参考:\n")
    
    for i, doc in enumerate(ranked_docs, 1):
        context_parts.append(f"{i}. 问题: {doc['question']}\n")
        context_parts.append(f"   答案: {doc['answer']}\n")
        if doc.get('related_diseases'):
            context_parts.append(f"   相关疾病: {doc['related_diseases']}\n")
        context_parts.append(f"   专业领域: {doc['label']}\n")
        context_parts.append(f"   质量评分: {doc['score']}/5\n\n")
    
    # 添加医疗免责声明
    context_parts.append("重要提示: 以上信息仅供参考，不能替代专业医疗建议。如有医疗问题，请咨询专业医生。")
    
    return "".join(context_parts)
    