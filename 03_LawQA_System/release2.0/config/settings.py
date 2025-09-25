import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent

# 模型路径
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "text2vec-large-chinese")
GENERATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-3B-Instruct")

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# 向量存储路径
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "faiss_vector_db", "law_faiss_index")

# 外部API配置
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")  # 用于外部搜索

# GPU设备
GENERATION_MODEL_DEVICE = "cuda:5"
EMBEDDING_MODEL_DEVICE = "cpu"

