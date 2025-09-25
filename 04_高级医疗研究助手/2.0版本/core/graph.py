from langgraph.graph import StateGraph, END
from state import ResearchState
from nodes import (
    plan_research,
    retrieve_information,
    filter_documents,
    synthesize_information,
    generate_report
)

def create_research_workflow():
    # 创建图
    workflow = StateGraph(ResearchState)

    # 添加节点
    workflow.add_node("plan_research", plan_research)
    workflow.add_node("retrieve_information", retrieve_information)
    workflow.add_node("filter_documents", filter_documents)
    workflow.add_node("synthesize_information", synthesize_information)
    workflow.add_node("generate_report", generate_report)

    # 设置入口点
    workflow.set_entry_point("plan_research")

    # 添加条件边，确保状态正确
    def should_generate_report(state):
        return state.get("synthesized_content") is not None and len(state["synthesized_content"]) > 0

    # 添加边
    workflow.add_edge("plan_research", "retrieve_information")
    workflow.add_edge("retrieve_information", "filter_documents")
    workflow.add_edge("filter_documents", "synthesize_information")
    workflow.add_conditional_edges(
        "synthesize_information",
        should_generate_report,
        {
            True: "generate_report",
            False: END  # 如果综合内容为空，直接结束
        }
    )
    workflow.add_edge("generate_report", END)

    # 编译图
    return workflow.compile()

# 全局图实例
research_graph = create_research_workflow()
