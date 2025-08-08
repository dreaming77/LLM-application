from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain.chains import RetrievalQA
from prompt import prompt
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json
import torch
from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


model_path = "./Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)


def load_vector_db():
    model_path = "./text2vec-large-chinese"
    db_path = "./faiss_vector_db"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'}
    )
    print(f"✅ 本地嵌入模型加载成功: {model_path}")

    vector_db = FAISS.load_local(
        folder_path=db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ 本地向量数据库加载成功: {db_path}")
    return vector_db


vector_db = load_vector_db()


# ============= RAG Fusion 核心实现 =============
class RAGFusionRetriever(BaseRetriever):
    """实现RAG Fusion的检索器，包含多查询生成和RRF融合"""
    base_retriever: BaseRetriever
    llm_chain: LLMChain
    num_queries: int = 4  # 生成的查询数量（包含原始查询）
    fusion_k: int = 5  # 融合后返回的文档数量
    rrf_k: int = 60  # RRF算法参数（经验值）

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """执行RAG Fusion检索流程"""
        # 1. 生成多样化查询
        generated_queries = self._generate_queries(query)
        all_queries = [query] + generated_queries
        print(f"🔍 生成的查询列表: {all_queries}")

        # 2. 并行执行所有查询检索
        all_results = []
        for q in all_queries:
            docs = self.base_retriever.get_relevant_documents(q)
            all_results.append(docs)
            print(f"📌 查询 '{q}' 检索到 {len(docs)} 个文档")

        # 3. 应用RRF融合算法
        fused_docs = self._reciprocal_rank_fusion(all_results)
        return fused_docs[:self.fusion_k]

    def _generate_queries(self, original_query: str) -> List[str]:
        """使用LLM生成多样化查询"""
        response = self.llm_chain.invoke({
            "original_query": original_query,
            "num_queries": self.num_queries - 1  # 额外生成n-1个查询
        })
        try:
            queries = json.loads(response["text"])
            return queries[:self.num_queries - 1]
        except Exception as e:
            print(f"⚠️ 查询生成解析失败: {e}")
            return []

    def _reciprocal_rank_fusion(
            self, results: List[List[Document]]
    ) -> List[Document]:
        """实现倒数排序融合算法"""
        doc_scores = {}

        # 给每个文档集合分配权重
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_hash = hash((doc.page_content, tuple(doc.metadata.items())))
                doc_scores.setdefault(doc_hash, {"doc": doc, "score": 0.0})
                doc_scores[doc_hash]["score"] += 1.0 / (self.rrf_k + rank + 1)

        # 按融合得分排序
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        return [item["doc"] for item in sorted_docs]


# ============= 查询生成提示模板 =============
QUERY_GEN_PROMPT = PromptTemplate(
    input_variables=["original_query", "num_queries"],
    template="""作为搜索优化专家，请为以下查询生成{num_queries}个语义相关的变体查询。
保持查询的简洁性和多样性，确保覆盖不同表达方式和相关概念。
原始查询：{original_query}

要求：
1. 输出必须是纯JSON数组格式
2. 不要包含任何解释性文本
3. 所有查询必须使用中文

示例输出格式：
["查询变体1", "查询变体2", ...]"""
)

# ============= 初始化LLM组件 =============
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
    temperature=0.1
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ============= 配置RAG Fusion检索器 =============
base_retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}  # 每个查询检索更多结果
)

# 创建查询生成链
query_gen_chain = LLMChain(
    llm=llm,
    prompt=QUERY_GEN_PROMPT,
    output_key="text"
)

# 创建RAG Fusion检索器
fusion_retriever = RAGFusionRetriever(
    base_retriever=base_retriever,
    llm_chain=query_gen_chain,
    num_queries=5,  # 原始查询+4个变体
    fusion_k=5  # 返回前5个融合文档
)

# ============= 创建优化后的QA链 =============
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=fusion_retriever,  # 使用RAG Fusion检索器
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

