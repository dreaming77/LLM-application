from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from utils.rag_fusion import RAGFusionRetriever
from langchain.chains import RetrievalQA
from utils.prompt_templates import CONVERSATIONAL_PROMPT
from utils.model_manager import model_manager
import re


class CustomConversationBufferWindowMemory(ConversationBufferWindowMemory):
    """自定义对话内存，明确指定输出键"""

    def __init__(self, *args, output_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_key = output_key

    def save_context(self, inputs, outputs):
        """重写保存上下文方法，明确指定输出键"""
        if self.output_key and self.output_key in outputs:
            # 只保存指定的输出键
            filtered_outputs = {self.output_key: outputs[self.output_key]}
            super().save_context(inputs, filtered_outputs)
        else:
            # 如果没有指定输出键，使用默认行为
            super().save_context(inputs, outputs)


class ConversationalQA:
    """改进的对话QA系统，提供更连贯的回答"""

    def __init__(self):
        # 初始化RAG Fusion检索器·
        self.retriever = RAGFusionRetriever(num_variants=2, top_k=5)

        # 使用模型管理器中的生成模型
        self.llm = model_manager.get_huggingface_pipeline()

        # 初始化对话记忆
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=3
        )

    def answer_question(self, question: str) -> dict:
        """回答问题并返回来源"""
        try:
            # 获取并简化对话历史
            memory_vars = self.memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            chat_history_summary = self._summarize_chat_history(chat_history)

            # 使用基础的检索QA链，但手动构建包含历史摘要的查询
            # 首先检索相关文档
            retrieved_docs = self.retriever._get_relevant_documents(question)

            # 构建上下文
            context = "\n".join([doc.page_content for doc in retrieved_docs[:3]])  # 只使用前3个文档

            # 构建完整的提示词
            full_prompt = CONVERSATIONAL_PROMPT.format(
                chat_history=chat_history_summary,
                context=context,
                question=question
            )

            # 使用LLM生成回答
            response = self.llm(full_prompt)

            # 过滤回答
            filtered_answer = self._filter_answer(response)

            # 保存到记忆（只保存问题和回答，不保存上下文）
            self.memory.save_context(
                {"question": question},
                {"answer": filtered_answer}
            )

            # 提取来源信息
            sources = []
            for doc in retrieved_docs[:5]:  # 最多显示5个来源
                source_info = {
                    "law": doc.metadata.get("law_name", "未知"),
                    "article": doc.metadata.get("article_id", "未知"),
                    "chapter": doc.metadata.get("chapter_title", "未知"),
                    "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                }
                sources.append(source_info)

            return {
                "answer": filtered_answer,
                "sources": sources
            }

        except Exception as e:
            print(f"对话QA出错: {e}")
            # 备用方案
            return self._fallback_answer_question(question)


    def _summarize_chat_history(self, chat_history):
        """将对话历史简化为摘要，而不是完整文本"""
        if not chat_history:
            return "这是第一次对话。"

        # 提取最近几轮对话的关键信息
        summary_parts = []
        for i, message in enumerate(chat_history[-3:]):  # 只考虑最近3条消息
            content = message.content if hasattr(message, 'content') else str(message)
            # 简化内容，只保留关键信息
            simplified = re.sub(r'[{}]', '', content)  # 移除大括号内容
            simplified = re.sub(r'第[一二三四五六七八九十百千]+章.*?条', '相关法律条文', simplified)  # 简化法律引用
            simplified = re.sub(r'\s+', ' ', simplified)  # 合并空格
            simplified = simplified[:100] + "..." if len(simplified) > 100 else simplified  # 截断

            role = "用户" if i % 2 == 0 else "助手"
            summary_parts.append(f"{role}: {simplified}")

        return " | ".join(summary_parts)

    def _filter_answer(self, answer: str) -> str:
        """过滤回答，移除不需要的内容"""
        # 移除提示词模板相关内容
        filter_patterns = [
            "你是一个专业的法律助手",
            "对话历史摘要：",
            "当前问题：",
            "相关法律条文：",
            "请直接回答问题",
            "不要重复对话历史或问题本身",
            "回答要简洁明了",
            "专注于当前问题"
        ]

        lines = answer.split('\n')
        filtered_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 跳过包含过滤模式的行
            if any(pattern in line for pattern in filter_patterns):
                continue

            filtered_lines.append(line)

        # 重新组合
        filtered_answer = '\n'.join(filtered_lines)

        # 如果为空，返回原始回答
        return filtered_answer if filtered_answer.strip() else answer

    def _fallback_answer_question(self, question: str) -> dict:
        """备用方法"""
        return {
            "answer": "抱歉，我暂时无法回答这个问题。建议您咨询专业律师获取准确的法律意见。",
            "sources": []
        }

    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear()


if __name__ == "__main__":
    """单元测试"""
    pass
