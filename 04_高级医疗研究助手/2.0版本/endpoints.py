import asyncio
import sys
import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict

# 添加core目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+ 'core')

from core.graph import research_graph
from core.state import ResearchState
from core.async_processor import start_research_task, get_research_progress

research_router = APIRouter()

class ResearchRequest(BaseModel):
    query: str
    max_documents: Optional[int] = 10


class ResearchResponse(BaseModel):
    report: str
    status: str

class ResearchStartResponse(BaseModel):
    research_id: str
    status: str

class ResearchProgressResponse(BaseModel):
    status: str
    progress: Optional[int] = 0
    report: Optional[str] = None
    message: Optional[str] = None
    query: Optional[str] = None

@research_router.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    try:
        # 初始化状态
        initial_state = ResearchState(
            user_query=request.query,
            search_queries=[],
            retrieved_documents=[],
            filtered_documents=[],
            synthesized_content="",
            final_report=""
        )

        # 运行研究图
        # 使用线程池执行同步代码，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: research_graph.invoke(initial_state)
        )

        return ResearchResponse(
            report=result.get("final_report", "未能生成报告"),
            status="success",
            query=request.query
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"研究过程中出错: {str(e)}")


@research_router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "医疗研究助手API"}


@research_router.post("/research/start", response_model=ResearchStartResponse)
async def start_research(request: ResearchRequest):
    try:
        research_id = start_research_task(request.query)

        return ResearchStartResponse(
            research_id=research_id,
            status="started"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动研究任务失败: {str(e)}")


@research_router.get("/research/progress/{research_id}", response_model=ResearchProgressResponse)
async def get_research_progress_endpoint(research_id: str):
    progress_data = get_research_progress(research_id)

    if progress_data["status"] == "not_found":
        raise HTTPException(status_code=404, detail="研究任务不存在")

    return ResearchProgressResponse(**progress_data)

