import logging
from typing import Dict, Any, List
from milvus.milvus_client import get_milvus_client
from core.state.state import GraphState
from core.model_manager import model_manager

logger = logging.getLogger(__name__)

async def document_retrieval_node(state: GraphState) -> Dict[str, Any]:
    """
    文档检索节点
    使用重写后的查询从Milvus中检索相关文档
    """
    logger.info("执行文档检索节点")
    
    try:
        rewritten_queries = state["rewritten_queries"]
        search_k = state.get("search_k", 10)
        
        # 获取Milvus客户端
        milvus_client = get_milvus_client()
        
        # 确保连接
        if not milvus_client.ensure_connected():
            logger.error("无法连接到Milvus数据库")
            return {
                "retrieved_documents": [],
                "document_scores": {},
                "current_step": "document_retrieval", 
                "next_step": "error",
                "error_message": "无法连接到向量数据库"
            }
        
        retrieved_documents = []
        document_scores = {}
        
        # 生成所有查询的向量
        query_vectors = []
        valid_queries = []
        
        for i, query in enumerate(rewritten_queries):
            logger.info(f"生成查询 {i+1} 的向量: {query}")
            
            query_vector = await _get_query_embedding(query)
            if query_vector:
                query_vectors.append(query_vector)
                valid_queries.append((i, query))
            else:
                logger.warning(f"查询 {i+1} 的向量生成失败: {query}")
        
        if not query_vectors:
            logger.error("所有查询的向量生成都失败")
            return {
                "retrieved_documents": [],
                "document_scores": {},
                "current_step": "document_retrieval", 
                "next_step": "error",
                "error_message": "查询向量化失败"
            }
        
        # 执行批量搜索
        search_results = milvus_client.search_similar_questions(
            query_vectors=query_vectors,
            top_k=search_k,
            output_fields=["id", "question", "answer", "label", "score", "related_diseases"]
        )

        # 处理搜索结果
        for (query_idx, original_query), results in zip(valid_queries, search_results):
            logger.info(f"处理查询 {query_idx+1} 的 {len(results)} 个结果")
            
            for j, hit in enumerate(results):
                document_id = f"doc_{query_idx}_{j}"
                document = {
                    "id": document_id,
                    "milvus_id": hit.get("id", ""),
                    "query_index": query_idx,
                    "query": original_query,
                    "question": hit.get("question", ""),
                    "answer": hit.get("answer", ""),
                    "label": hit.get("label", ""),
                    "score": hit.get("score", 0),
                    "related_diseases": hit.get("related_diseases", ""),
                    "similarity_score": hit.get("similarity_score", 0),
                    "distance": hit.get("distance", 1.0),
                    "rank": j + 1
                }
                
                retrieved_documents.append(document)
                document_scores[document_id] = hit.get("distance", 1.0)
        
        # 去重并排序
        unique_documents = _remove_duplicate_documents(retrieved_documents)
        
        logger.info(f"检索到 {len(unique_documents)} 个唯一文档")
        
        return {
            "retrieved_documents": unique_documents,
            "document_scores": document_scores,
            "current_step": "document_retrieval",
            "next_step": "rag_fusion"
        }
        
    except Exception as e:
        logger.error(f"文档检索失败: {e}")
        return {
            "retrieved_documents": [],
            "document_scores": {},
            "current_step": "document_retrieval", 
            "next_step": "error",
            "error_message": f"文档检索错误: {str(e)}"
        }

async def _get_query_embedding(query: str) -> List[float]:
    """获取查询的向量表示"""
    try:
        # 使用嵌入模型生成向量
        embedding = model_manager.embed_text(query)
        
        # 验证向量维度
        if len(embedding) != 1024:
            logger.warning(f"向量维度不正确: {len(embedding)}，期望 1024")
            # 尝试调整维度（简单的填充或截断）
            if len(embedding) > 1024:
                embedding = embedding[:1024]
            else:
                embedding = embedding + [0.0] * (1024 - len(embedding))
        
        return embedding
        
    except Exception as e:
        logger.error(f"查询向量化失败: {e}")
        return None

def _remove_duplicate_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去除重复文档（基于问题内容的相似性）"""
    seen_questions = set()
    unique_documents = []
    
    for doc in documents:
        question = doc["question"]
        
        if not question:  # 跳过空问题
            continue
            
        # 简单的基于内容的去重
        is_duplicate = False
        for seen_question in seen_questions:
            if _calculate_similarity(question, seen_question) > 0.9:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_documents.append(doc)
            seen_questions.add(question)
    
    # 按相似度排序
    unique_documents.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return unique_documents

def _calculate_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（综合策略）"""
    if not text1 or not text2:
        return 0.0
        
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    if text1 == text2:
        return 1.0
    
    # 策略1：字符集合相似度
    def char_set_similarity(s1, s2):
        set1, set2 = set(s1), set(s2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0.0
    
    # 策略2：最长公共子序列相似度
    def lcs_similarity(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_len = dp[m][n]
        return lcs_len / max(m, n) if max(m, n) > 0 else 0.0
    
    # 策略3：编辑距离相似度
    def edit_distance_similarity(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        edit_dist = dp[m][n]
        max_len = max(m, n)
        return 1 - (edit_dist / max_len) if max_len > 0 else 0.0
    
    # 计算各种相似度
    char_sim = char_set_similarity(text1, text2)
    lcs_sim = lcs_similarity(text1, text2)
    edit_sim = edit_distance_similarity(text1, text2)
    
    # 加权平均，可以根据需要调整权重
    similarity = 0.3 * char_sim + 0.4 * lcs_sim + 0.3 * edit_sim
    
    return similarity
    