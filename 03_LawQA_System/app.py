import streamlit as st
from langchain_core.exceptions import OutputParserException

from rag_chain import qa_chain

# 初始化组件
st.title("⚖️ 法律智能问答系统")
question = st.text_input("请输入法律问题：", placeholder="例如：盗窃罪如何量刑？")

if st.button("获取答案"):
    with st.spinner("🔍 检索法律条款中..."):
        try:
            # 使用 invoke() 获取结构化响应
            response = qa_chain.invoke({"query": question})

            answer = response["result"]

            # 提取源文档用于参考
            source_docs = response["source_documents"]

            st.success("⚖️ **法律解答：**")
            st.markdown(f"> {response}")

        except OutputParserException as e:
            st.error(f"解析答案时出错: {str(e)}")
        except Exception as e:
            st.error(f"处理问题时发生错误: {str(e)}")


                