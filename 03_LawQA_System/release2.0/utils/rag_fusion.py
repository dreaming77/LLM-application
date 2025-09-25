import re
from typing import List, Any
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings
from utils.model_manager import model_manager
from config.settings import VECTOR_STORE_PATH
import torch
from pydantic import Field, PrivateAttr
import warnings

# 忽略 LangChain 的警告
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")


class RAGFusionRetriever(BaseRetriever):
    """RAG Fusion检索器，继承自BaseRetriever"""

    # 声明 Pydantic 字段
    num_variants: int = Field(default=3)
    top_k: int = Field(default=10)

    # 使用 PrivateAttr 声明私有字段（不会被 Pydantic 验证）
    _embeddings: Embeddings = PrivateAttr()
    _vectorstore: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _generation_config: dict = PrivateAttr()

    def __init__(self, **kwargs):
        """
        初始化RAG Fusion检索器
        """
        # 先调用父类的初始化
        super().__init__(**kwargs)

        # 初始化私有字段
        self._embeddings = model_manager.embedding_model

        # 确保向量存储只加载一次
        if not hasattr(self, '_vectorstore_loaded'):
            self._vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH,
                self._embeddings,
                allow_dangerous_deserialization=True
            )
            self._vectorstore_loaded = True

        self._tokenizer = model_manager.generation_tokenizer
        self._model = model_manager.generation_model

        # 设置生成参数
        self._generation_config = {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1
        }

    def generate_query_variants(self, original_query: str) -> List[str]:
        """生成查询变体"""
        prompt = f"""你是一个法律检索助手。请为以下查询生成{self.num_variants}个不同的变体查询，这些变体应该从不同角度表达相同或相似的法律检索需求。

                原始查询: {original_query}
                
                请生成{self.num_variants}个变体查询，每个变体一行，只输出查询内容，不要包含任何其他文本:
                1."""

        # 编码输入
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        # 生成文本
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **self._generation_config
            )

        # 解码生成结果
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取变体查询
        lines = generated_text.split('\n')
        variants = []

        for line in lines:
            # 匹配编号开头的行
            if re.match(r'^\d+\.', line):
                # 移除编号和可能的空格
                variant = re.sub(r'^\d+\.\s*', '', line).strip()
                # 过滤掉空行和过长的行
                if variant and len(variant) < 100 and not variant.startswith('请生成'):
                    variants.append(variant)

        # 确保有足够的变体，如果不够则使用原始查询补充
        while len(variants) < self.num_variants:
            variants.append(original_query)

        # 添加原始查询作为第一个变体
        variants = [original_query] + variants[:self.num_variants]

        return variants

    def reciprocal_rank_fusion(self, search_results: List[List[Document]], k: int = 60) -> List[Document]:
        """
        使用倒数排名融合(RRF)算法融合多个查询的结果

        Args:
            search_results: 多个查询的检索结果列表
            k: RRF参数，通常设置为60

        Returns:
            融合排序后的文档列表
        """
        # 初始化文档得分字典
        doc_scores = {}

        # 遍历每个查询的结果
        for i, results in enumerate(search_results):
            # 遍历当前查询的每个结果
            for rank, doc in enumerate(results):
                # 获取文档的唯一标识
                doc_key = f"{doc.metadata.get('law_name', '')}_{doc.metadata.get('article_id', '')}"

                # 更新文档得分
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {
                        'score': 0.0,
                        'doc': doc
                    }

                # 应用RRF公式
                doc_scores[doc_key]['score'] += 1.0 / (rank + k)

        # 按得分排序
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        # 返回文档对象
        return [item['doc'] for item in sorted_docs]

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        """实现BaseRetriever的核心方法"""
        # 生成查询变体
        query_variants = self.generate_query_variants(query)

        # 对每个查询变体进行检索
        all_results = []
        for variant in query_variants:
            try:
                results = self._vectorstore.similarity_search(variant, k=self.top_k * 2)
                all_results.append(results)
            except Exception as e:
                print(f"查询 '{variant}' 检索失败: {e}")
                # 如果某个查询失败，使用原始查询的结果
                results = self._vectorstore.similarity_search(query, k=self.top_k * 2)
                all_results.append(results)

        # 使用RRF融合结果
        fused_results = self.reciprocal_rank_fusion(all_results)

        # 返回top_k个结果
        return fused_results[:self.top_k]

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        """异步版本（可选实现）"""
        # 这里我们简单地调用同步版本
        return self._get_relevant_documents(query, run_manager=run_manager)

if __name__ == "__main__":
    """单元测试"""
    # 初始化RAG Fusion检索器
    faiss_index_path = "faiss_vector_db/law_faiss_index"
    embedding_model_path = "models/text2vec-large-chinese"  # 替换为你的嵌入模型路径
    generation_model_path = "models/Qwen2.5-3B-Instruct"  # 替换为你的生成模型路径

    retriever = RAGFusionRetriever(faiss_index_path, embedding_model_path, generation_model_path)

    # 示例查询
    query = "帮信罪该如何判罪？"

    # 进行检索
    results = retriever._get_relevant_documents(query, num_variants=3, top_k=5)

    # 打印结果
    print("\n最终检索结果:")
    for i, doc in enumerate(results):
        print(f"\n结果 {i + 1}:")
        print(f"法律: {doc.metadata.get('law_name', '未知')}")
        print(f"章节: {doc.metadata.get('chapter_title', '未知')}")
        if doc.metadata.get('section_title'):
            print(f"节: {doc.metadata.get('section_title')}")
        print(f"条文: {doc.metadata.get('article_id', '未知')}")
        print(f"内容: {doc.page_content}")
