#!/usr/bin/env python3
"""
高级医疗研究助手 - FastAPI主应用
"""

import os
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from contextlib import asynccontextmanager
import torch
from core.model_manager import model_manager
import uvicorn

from core.api.controller import medical_research_controller
from config.validator import UserQuery
from config.settings import API_HOST, API_PORT, API_WORKERS, API_DEBUG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("初始化医疗研究助手系统...")
    try:
        await medical_research_controller.initialize()
        logger.info("系统初始化完成")
        yield
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise
    finally:
        # 关闭时清理
        logger.info("关闭医疗研究助手系统...")
        await medical_research_controller.close()
        logger.info("系统已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="高级医疗研究助手 API",
    description="基于LangGraph + Milvus + RAG Fusion的智能医疗问答系统",
    version="1.0.0",
    docs_url="/docs",  # 启用默认文档
    redoc_url="/redoc",  # 启用Redoc文档
    lifespan=lifespan
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# 挂载静态文件（如果需要自定义Swagger UI资源）
# app.mount("/static", StaticFiles(directory=static_dir), name="static")

# API路由
@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "高级医疗研究助手 API",
        "version": "1.0.0",
        "status": "运行中",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        system_info = medical_research_controller.get_system_info()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system_info": system_info
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服务异常: {str(e)}")

@app.get("/api/health/detail")
async def detailed_health_check():
    """详细健康检查端点"""
    try:
        health_info = await medical_research_controller.health_check()
        return health_info
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"健康检查失败: {str(e)}")


@app.get("/api/health/models/detailed")
async def detailed_model_health():
    """详细的模型健康检查"""
    try:
        model_info = model_manager.get_model_info()

        # 测试嵌入模型
        embedding_test = model_manager.embed_text("健康检查")
        embedding_ok = embedding_test is not None and len(embedding_test) == 1024

        # 测试生成模型
        generation_test = model_manager.generate_response("回复OK")
        generation_ok = generation_test is not None and len(generation_test) > 0

        return {
            "status": "healthy" if (embedding_ok and generation_ok) else "degraded",
            "embedding_model": {
                "loaded": model_info["embedding_model"]["loaded"],
                "working": embedding_ok,
                "device": model_info["embedding_model"]["device"]
            },
            "generation_model": {
                "loaded": model_info["generation_model"]["loaded"],
                "working": generation_ok,
                "device": model_info["generation_model"]["device"]
            },
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/api/query", response_model=dict)
async def process_medical_query(query_data: UserQuery):
    """
    处理医疗查询

    Args:
        query_data: 用户查询数据

    Returns:
        处理结果
    """
    try:
        # 验证查询
        if not query_data.query or len(query_data.query.strip()) < 5:
            raise HTTPException(status_code=400, detail="查询过短，请提供更详细的问题描述")

        if len(query_data.query) > 1000:
            raise HTTPException(status_code=400, detail="查询过长，请简化问题表述")

        # 处理查询
        result = await medical_research_controller.process_medical_query(
            query=query_data.query,
            user_context=query_data.user_context,
            session_id=query_data.session_id,
            config=query_data.config.dict() if query_data.config else None
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理医疗查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统处理错误: {str(e)}")

@app.post("/api/conversation/{session_id}/continue")
async def continue_conversation(session_id: str, query_data: dict):
    """
    继续对话
    
    Args:
        session_id: 会话ID
        query_data: 新查询数据
        
    Returns:
        处理结果
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="需要提供会话ID")
        
        new_query = query_data.get("query")
        config = query_data.get("config")
        
        result = await medical_research_controller.continue_conversation(
            session_id=session_id,
            new_query=new_query,
            config=config
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"继续对话失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统错误: {str(e)}")

@app.get("/api/session/{session_id}")
async def get_session_status(session_id: str):
    """
    获取会话状态
    
    Args:
        session_id: 会话ID
        
    Returns:
        会话状态信息
    """
    try:
        status = medical_research_controller.get_session_status(session_id)
        return status
    except Exception as e:
        logger.error(f"获取会话状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@app.post("/api/session/{session_id}/interrupt")
async def interrupt_session(session_id: str):
    """
    中断会话处理
    
    Args:
        session_id: 会话ID
        
    Returns:
        中断结果
    """
    try:
        result = await medical_research_controller.interrupt_processing(session_id)
        return result
    except Exception as e:
        logger.error(f"中断会话失败: {e}")
        raise HTTPException(status_code=500, detail=f"中断失败: {str(e)}")

@app.get("/api/system/info")
async def get_system_info():
    """
    获取系统信息
    
    Returns:
        系统状态和信息
    """
    try:
        info = medical_research_controller.get_system_info()
        return info
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

@app.get("/api/config/default")
async def get_default_config():
    """
    获取默认配置
    
    Returns:
        默认工作流配置
    """
    from config.settings import WORKFLOW_CONFIG
    return WORKFLOW_CONFIG

# 自定义文档界面
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "内部服务器错误",
            "timestamp": time.time()
        }
    )

@app.get("/api/health/models")
async def model_health_check():
    """模型健康检查端点"""
    try:
        health_info = await medical_research_controller.model_health_check()
        return health_info
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"模型健康检查失败: {str(e)}")


@app.get("/api/debug/models")
async def debug_models():
    """调试模型状态"""
    try:
        model_info = model_manager.get_model_info()

        # 测试嵌入模型
        embedding_test = None
        if model_manager.embedding_model:
            try:
                embedding_test = model_manager.embed_text("测试")
            except Exception as e:
                embedding_test = f"错误: {e}"

        # 测试生成模型
        generation_test = None
        if model_manager.generation_pipeline:
            try:
                generation_test = model_manager.generate_response("回复OK")
            except Exception as e:
                generation_test = f"错误: {e}"

        return {
            "model_info": model_info,
            "embedding_test": "成功" if embedding_test and len(embedding_test) == 1024 else f"失败: {embedding_test}",
            "generation_test": "成功" if generation_test and len(generation_test) > 0 else f"失败: {generation_test}",
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        reload=API_DEBUG,
        log_level="info"
    )
