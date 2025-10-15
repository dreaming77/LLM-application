import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent

# 模型路径
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "BAAI", "bge-large-zh-v1.5")
GENERATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-7B-Instruct")

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DATA_DIR = os.path.join(DATA_DIR, "huatuo_dataset")
CHUATUO_DATA_PATH = os.path.join(RAW_DATA_DIR, "chuatuo_26m.jsonl")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# 向量存储路径
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "volumes")

# Milvus配置
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION_NAME = "CHuaTuo_26M_medical_qa"

# GPU设备，请根据实际情况修改
GENERATION_MODEL_DEVICE = "cuda:5"
EMBEDDING_MODEL_DEVICE = "cuda:3"

#模型配置
MODEL_CONFIG = {
    "embedding": {
        "model_name": EMBEDDING_MODEL_PATH,
        "device": EMBEDDING_MODEL_DEVICE,
        "normalize_embeddings": True
    },
    "generation": {
        "model_name": GENERATION_MODEL_PATH,
        "device": GENERATION_MODEL_DEVICE,
        "max_new_tokens": 1000,
        "temperature": 0.3,
        "do_sample": True
    }
}

# 数据处理配置
BATCH_SIZE = 2048
MAX_RECORDS = None  # 设为None处理全部数据


# 状态管理配置
MAX_SESSION_AGE_HOURS = 24  # 会话最大保存时间（小时）
STATE_PERSISTENCE_ENABLED = True  # 是否启用状态持久化
STATE_BACKUP_PATH = os.path.join(BASE_DIR, "dataset", "state_backups")

# 工作流配置
MAX_RETRY_ATTEMPTS = 3
DEFAULT_SEARCH_K = 10
DEFAULT_FUSION_TOP_K = 5
DEFAULT_GENERATION_TEMPERATURE = 0.3

# 医疗安全配置
MIN_CONFIDENCE_SCORE = 0.5
REQUIRE_MEDICAL_DISCLAIMER = True

# 工作流配置
WORKFLOW_CONFIG = {
    "search_k": DEFAULT_SEARCH_K,
    "fusion_top_k": DEFAULT_FUSION_TOP_K, 
    "generation_temperature": DEFAULT_GENERATION_TEMPERATURE,
    "max_retries": MAX_RETRY_ATTEMPTS,
    "session_timeout_minutes": 30
}

# API配置
API_HOST = "0.0.0.0"
API_PORT = 8888
API_WORKERS = 4
API_DEBUG = False

# 性能配置
MAX_CONCURRENT_SESSIONS = 10
SESSION_CLEANUP_INTERVAL_HOURS = 1
