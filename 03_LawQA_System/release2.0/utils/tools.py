import requests
from langchain.tools import Tool
from config.settings import SERPAPI_API_KEY


def search_external_knowledge(query: str) -> str:
    """使用SerpAPI搜索外部知识"""
    if not SERPAPI_API_KEY:
        return "未配置外部搜索API密钥"

    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "num": 5  # 返回5个结果
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        results = response.json()

        # 提取有机搜索结果
        organic_results = results.get("organic_results", [])
        snippets = [f"{result.get('title', '')}: {result.get('snippet', '')}" for result in organic_results]

        return "\n".join(snippets)
    except Exception as e:
        return f"搜索外部知识时出错: {str(e)}"


# 创建外部搜索工具
external_search_tool = Tool(
    name="ExternalSearch",
    func=search_external_knowledge,
    description="当问题超出法律知识库范围时，使用此工具搜索外部知识"
)

if __name__ == '__main__':
    """单元测试"""
