from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class ResearchState(TypedDict):
    # 用户输入
    user_query: str
    # 生成的搜索查询
    search_queries: List[str]
    # 检索到的文档
    retrieved_documents: List[Dict[str, Any]]
    # 过滤后的文档
    filtered_documents: List[Dict[str, Any]]
    # 综合后的内容
    synthesized_content: str
    # 最终报告
    final_report: str
