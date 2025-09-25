from langchain.chains import RetrievalQA
from utils.rag_fusion import RAGFusionRetriever
from utils.prompt_templates import QA_WITH_SOURCES_PROMPT
from utils.model_manager import model_manager
import re


class RetrievalQAWithSources:
    def __init__(self):
        # 初始化RAG Fusion检索器
        self.retriever = RAGFusionRetriever(num_variants=3, top_k=10)

        # 使用模型管理器中的生成模型
        self.llm = model_manager.get_huggingface_pipeline()

        # 创建QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": QA_WITH_SOURCES_PROMPT},
            return_source_documents=True
        )

    def answer_question(self, question: str) -> dict:
        """回答问题并返回来源"""
        result = self.qa_chain({"query": question})

        # 使用与ConversationalQA相同的清理逻辑
        cleaned_answer = self._clean_and_format_answer(result["result"])

        # 提取来源信息
        sources = []
        for doc in result["source_documents"][:3]:  # 只显示前3个来源
            source_info = {
                "law": doc.metadata.get("law_name", "未知"),
                "article": doc.metadata.get("article_id", "未知"),
                "chapter": doc.metadata.get("chapter_title", "未知"),
                "content": self._summarize_content(doc.page_content)
            }
            sources.append(source_info)

        return {
            "answer": cleaned_answer,
            "sources": sources
        }

    def _clean_and_format_answer(self, answer: str) -> str:
        """清理和格式化回答（与ConversationalQA相同）"""
        # 这里使用与上面ConversationalQA中相同的_clean_and_format_answer方法
        # 为了简洁，您可以将其提取到工具函数中，或者直接复制实现

        # 简化的清理逻辑
        filter_patterns = [
            "你是一个专业的法律助手",
            "相关法律条文：",
            "问题：",
            "请按照以下要求回答问题：",
            "请开始回答："
        ]

        for pattern in filter_patterns:
            answer = answer.replace(pattern, "")

        # 按句子分割并去重
        sentences = re.split(r'[。！？]', answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        unique_sentences = []
        seen_content = set()

        for sentence in sentences:
            if len(sentence) < 10:
                continue

            simplified = re.sub(r'[《》第条]', '', sentence)
            simplified = re.sub(r'\d+', '#', simplified)
            simplified = re.sub(r'\s+', '', simplified)

            if simplified not in seen_content:
                unique_sentences.append(sentence)
                seen_content.add(simplified)

        if not unique_sentences:
            return "抱歉，我无法提供准确的法律建议，请咨询专业律师。"

        # 限制句子数量
        if len(unique_sentences) > 5:
            unique_sentences = unique_sentences[:5]

        formatted_answer = '。'.join(unique_sentences) + '。'
        formatted_answer = re.sub(r'\.{2,}', '。', formatted_answer)
        formatted_answer = re.sub(r'\s+', ' ', formatted_answer)

        return formatted_answer

    def _summarize_content(self, content: str, max_length: int = 120) -> str:
        """摘要化法律条文内容"""
        if len(content) <= max_length:
            return content

        sentences = re.split(r'[。！？]', content)
        if sentences and len(sentences[0]) > 10:
            return sentences[0] + '...'

        return content[:max_length] + '...'

    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear()

    def _fallback_answer_question(self, question: str) -> dict:
        """备用方法"""
        return {
            "answer": "抱歉，我暂时无法回答这个问题。建议您咨询专业律师获取准确的法律意见。",
            "sources": []
        }


if __name__ == "__main__":
    """单元测试"""
    pass
