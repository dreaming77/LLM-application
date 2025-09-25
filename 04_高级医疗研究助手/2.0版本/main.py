from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import research_router
import uvicorn
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
core_dir = os.path.join(project_root, "core")

if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

app = FastAPI(
    title="高级医疗研究助手API",
    description="基于LangGraph和RAG的医疗研究助手后端API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vue开发服务器
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(research_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "API服务运行中"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8055)
