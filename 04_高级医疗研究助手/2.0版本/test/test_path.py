# 测试python包路径问题

import sys
import os
sys.path.append(os.path.dirname(os.getcwd()) + '/core')

# path = os.path.dirname(os.getcwd() + '/core')

from ..core.graph import research_graph
from ..core.state import ResearchState

target_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(target_path + '/core')
path = os.path.join(target_path + '/core')
print(path)

# import sys
# import os
#
# # 获取当前文件的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取项目根目录（假设core在项目根目录下）
# project_root = os.path.dirname(current_dir)
# core_path = os.path.join(project_root, 'core')
#
# # 检查路径是否存在
# if os.path.exists(core_path):
#     if core_path not in sys.path:
#         sys.path.insert(0, core_path)  # 插入到开头优先查找
#     try:
#         from core.graph import research_graph
#         from core.state import ResearchState
#         print("成功导入core模块!")
#     except ImportError as e:
#         print(f"导入错误: {e}")
# else:
#     print(f"core路径不存在: {core_path}")
