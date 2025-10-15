"""
应用配置
"""

import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """应用设置"""
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_debug: bool = False
    
    # 安全配置
    api_keys: list = ["medical_research_demo_key"]
    
    # CORS配置
    cors_origins: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # 静态文件配置
    static_files_dir: str = "./static"
    
    class Config:
        env_file = ".env"

settings = Settings()

