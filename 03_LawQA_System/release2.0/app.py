import streamlit as st
import torch
import gc
from chains.retrieval_qa import RetrievalQAWithSources
from utils.document_processor import process_documents
from utils.model_manager import model_manager
from config.settings import VECTOR_STORE_PATH
import os


# 设置内存管理
def manage_memory():
    """管理内存使用"""
    gc.collect()  # 强制垃圾回收
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空CUDA缓存


# 清理GPU缓存
model_manager.clear_cache()
manage_memory()

# 初始化应用
st.set_page_config(page_title="法律问答系统", page_icon="⚖️")
st.title("⚖️ 法律问答系统")

# 检查向量存储是否存在，如果不存在则处理文档
if not os.path.exists(VECTOR_STORE_PATH):
    st.info("正在初始化法律知识库，这可能需要一些时间...")
    process_documents()
    st.success("法律知识库初始化完成！")
    manage_memory()  # 处理文档后清理内存


# 初始化系统
@st.cache_resource(show_spinner=False)
def load_systems():
    # 清理GPU缓存
    model_manager.clear_cache()
    manage_memory()

    retrieval_qa = RetrievalQAWithSources()
    return retrieval_qa


retrieval_qa = load_systems()

# 会话状态初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "use_agent" not in st.session_state:
    st.session_state.use_agent = False
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# 侧边栏设置
with st.sidebar:
    st.header("设置")
    st.session_state.use_agent = st.checkbox("启用外部知识查询", value=False)

    if st.button("清空对话历史"):
        retrieval_qa.clear_memory()
        st.session_state.messages = []
        st.session_state.message_count = 0
        manage_memory()  # 清空历史后清理内存
        st.rerun()

    # 显示内存使用情况
    st.subheader("系统状态")
    st.write(f"对话轮数: {st.session_state.message_count}")
    if torch.cuda.is_available():
        st.write(
            f"GPU内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB / {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 显示来源信息（如果是助理的消息）
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("查看来源"):
                for source in message["sources"]:
                    st.write(f"**{source.get('law', '未知')}** - {source.get('article', '未知')}")
                    st.caption(source.get('content', ''))

# 用户输入
if prompt := st.chat_input("请输入您的法律问题..."):
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.message_count += 1

    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    # 在显示回答的部分，使用更简单的格式化
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = conversational_qa.answer_question(prompt)

            # 直接显示清理后的回答，不使用复杂格式化
            message_placeholder = st.empty()
            full_response = ""

            # 简单的流式输出
            for chunk in result["answer"].split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # 显示来源信息
            if "sources" in result and result["sources"]:
                with st.expander("查看法律依据"):
                    for i, source in enumerate(result["sources"]):
                        st.write(f"**{source.get('law', '未知')}** - {source.get('article', '未知')}")
                        if source.get('chapter'):
                            st.caption(f"章节: {source.get('chapter')}")
                        st.write(source.get('content', ''))
                        if i < len(result["sources"]) - 1:
                            st.divider()

                # 添加助理消息到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": result.get("sources", [])
                })

                # 每5轮对话后强制清理一次内存
                if st.session_state.message_count % 5 == 0:
                    model_manager.clear_cache()
                    manage_memory()
