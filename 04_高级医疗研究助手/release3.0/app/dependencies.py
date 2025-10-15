"""
API依赖项
"""

from fastapi import Header, HTTPException
from typing import Optional

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """验证API密钥（基础实现）"""
    # 生产环境应该使用更安全的认证方式
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="需要API密钥")
    
    # 这里可以添加更复杂的密钥验证逻辑
    valid_keys = ["medical_research_demo_key"]  # 示例密钥
    
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="无效的API密钥")
    
    return x_api_key
    