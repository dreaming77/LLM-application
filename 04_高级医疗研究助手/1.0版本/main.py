from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import threading
import uuid
import time
from threading import Thread
import os
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from core.graph import research_graph
from core.nodes import init_models
from core.state import ResearchState

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 初始化FastAPI应用
app = FastAPI(title="高级医疗研究助手&报告生成器")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储任务状态
tasks = {}
task_lock = threading.Lock()
logger = logging.getLogger(__name__)

# 请求模型
class ResearchRequest(BaseModel):
    query: str


# 响应模型
class ResearchResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: str = None


# 状态模型
class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: int
    result: Optional[Dict[str, Any]] = None  # 使用Optional允许None值
    error: Optional[str] = None  # 添加错误信息字段

# 添加全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"内部服务器错误: {str(exc)}", "detail": str(exc)}
    )


# 添加请求验证错误处理
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "请求数据验证失败", "detail": exc.errors()}
    )


# 根路由
@app.get("/")
async def root():
    return {
        "message": "高级医疗研究助手API服务",
        "version": "1.0.0",
        "docs": "/docs",
        "health_check": "/health"
    }


def cleanup_old_tasks():
    """清理超过30分钟的任务"""
    while True:
        time.sleep(300)  # 每5分钟检查一次
        current_time = time.time()
        with task_lock:
            # 找出所有超过30分钟的任务
            tasks_to_remove = []
            for task_id, task_info in list(tasks.items()):
                if 'start_time' in task_info and current_time - task_info['start_time'] > 1800:  # 30分钟
                    tasks_to_remove.append(task_id)

            # 移除超时任务
            for task_id in tasks_to_remove:
                print(f"移除超时任务: {task_id}")
                del tasks[task_id]


# 在应用启动时启动清理线程
@app.on_event("startup")
async def startup_event():
    # 初始化模型（在单独的线程中运行，避免阻塞事件循环）
    def init_models_thread():
        embedding_model_path = os.path.join(PROJECT_ROOT, "models", "BAAI", "bge-large-zh-v1.5")
        llm_model_path = os.path.join(PROJECT_ROOT, "models", "Qwen-7B-Chat")

        # 初始化模型，指定GPU设备
        init_models(
            embedding_model_path=embedding_model_path,
            llm_model_path=llm_model_path,
            embedding_device_id=5,  # 根据实际情况调整
            llm_device_id=6  # 根据实际情况调整
        )
        print("模型初始化完成")

    # 启动初始化线程
    thread = threading.Thread(target=init_models_thread)
    thread.daemon = True
    thread.start()

    # 启动任务清理线程
    cleanup_thread = threading.Thread(target=cleanup_old_tasks, daemon=True)
    cleanup_thread.start()

# 研究任务处理函数
def process_research_task(task_id, query):
    try:
        # 更新任务状态
        with task_lock:
            tasks[task_id]["status"] = "processing"
            tasks[task_id]["progress"] = 10
            tasks[task_id]["result"] = {}

        # 初始化研究状态
        initial_state = ResearchState(
            user_query=query,
            search_queries=[],
            retrieved_documents=[],
            filtered_documents=[],
            synthesized_content="",
            final_report=""
        )

        # 设置超时（15分钟）
        timeout_seconds = 900

        # 直接调用研究图，但设置超时
        start_time = time.time()
        result = research_graph.invoke(initial_state)
        elapsed_time = time.time() - start_time

        if elapsed_time > timeout_seconds:
            raise Exception(f"研究任务执行超时，耗时 {elapsed_time:.2f} 秒")

        with task_lock:
            tasks[task_id]["progress"] = 100
            tasks[task_id]["result"] = dict(result)
            tasks[task_id]["status"] = "completed"

        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"任务处理错误: {task_id}, 错误: {str(e)}")
        print(f"错误详情: {error_details}")

        with task_lock:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = f"{str(e)}\n\n错误详情:\n{error_details}"
            tasks[task_id]["result"] = {}


# 创建研究任务端点
@app.post("/research", response_model=ResearchResponse)
async def create_research_task(request: ResearchRequest):
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())

    # 初始化任务状态
    with task_lock:
        tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "result": {},
            "error": None,
            "start_time": time.time()  # 记录开始时间
        }

    # 在后台线程中处理任务
    thread = threading.Thread(
        target=process_research_task,
        args=(task_id, request.query)
    )
    thread.daemon = True
    thread.start()

    return ResearchResponse(
        task_id=task_id,
        status="pending"
    )


# 获取任务状态端点
@app.get("/research/{task_id}", response_model=TaskStatus)
async def get_research_status(task_id: str):
    with task_lock:
        task = tasks.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # 检查任务是否超时（30分钟）
    if task["status"] == "processing" and "start_time" in task:
        elapsed_time = time.time() - task["start_time"]
        if elapsed_time > 1800:  # 30分钟
            task["status"] = "error"
            task["error"] = f"任务执行超时（已运行 {elapsed_time:.0f} 秒）"

    # 确保result字段始终是字典，即使为None也转换为空字典
    result_data = task["result"] if task["result"] is not None else {}

    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        result=result_data,
        error=task.get("error")
    )

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

# 在应用启动时启动清理线程
@app.on_event("startup")
async def startup_event():
    # 初始化模型...

    # 启动任务清理线程
    cleanup_thread = Thread(target=cleanup_old_tasks, daemon=True)
    cleanup_thread.start()


if __name__ == "__main__":
    import uvicorn

    # 禁用热重载，设置工作目录为当前目录
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8055,
        reload=False,  # 禁用热重载
        workers=1  # 使用单个工作进程
    )
