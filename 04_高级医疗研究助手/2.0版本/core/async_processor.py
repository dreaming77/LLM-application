import uuid
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from nodes import (
    plan_research,
    retrieve_information,
    filter_documents,
    synthesize_information,
    generate_report
)

# 存储研究任务状态
research_tasks = {}
# 全局线程池
thread_pool = ThreadPoolExecutor(max_workers=2)  # 限制并发数，避免GPU内存溢出

def execute_research_task(research_id: str, query: str):
    """在后台线程中执行研究任务"""
    try:
        # 初始化状态
        state = {
            "user_query": query,
            "search_queries": [],
            "retrieved_documents": [],
            "filtered_documents": [],
            "synthesized_content": "",
            "final_report": ""
        }

        # 更新任务状态
        research_tasks[research_id] = {
            "status": "processing",
            "progress": 10,
            "state": state
        }

        # 步骤1: 计划研究
        state.update(plan_research(state))
        research_tasks[research_id]["progress"] = 25

        # 步骤2: 检索信息
        state.update(retrieve_information(state))
        research_tasks[research_id]["progress"] = 40

        # 步骤3: 过滤文档
        state.update(filter_documents(state))
        research_tasks[research_id]["progress"] = 55

        # 步骤4: 综合信息
        state.update(synthesize_information(state))
        research_tasks[research_id]["progress"] = 70

        # 步骤5: 生成报告
        state.update(generate_report(state))
        research_tasks[research_id]["progress"] = 90

        # 完成
        research_tasks[research_id] = {
            "status": "completed",
            "progress": 100,
            "report": state["final_report"],
            "query": query
        }

    except Exception as e:
        research_tasks[research_id] = {
            "status": "failed",
            "progress": 100,
            "message": str(e),
            "query": query
        }


def start_research_task(query: str) -> str:
    """启动研究任务并返回任务ID"""
    research_id = str(uuid.uuid4())

    # 初始化任务状态
    research_tasks[research_id] = {
        "status": "pending",
        "progress": 0,
        "query": query
    }

    # 在线程池中执行任务
    thread_pool.submit(execute_research_task, research_id, query)

    return research_id


def get_research_progress(research_id: str) -> Dict:
    """获取研究任务进度"""
    return research_tasks.get(research_id, {"status": "not_found"})
