#!/usr/bin/env python3
import os
import sys
import uvicorn

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
core_dir = os.path.join(project_root, "core")

if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

# 导入主应用
from main import app

if __name__ == "__main__":
    uvicorn.run(
        "start_server:app",
        host="0.0.0.0",
        port=8055,
        reload=True,  # 开发模式下启用热重载
        reload_dirs=[current_dir, core_dir]  # 监视这些目录的变化
    )
