import multiprocessing as mp
from functools import partial
import json
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import torch
import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 初始化嵌入模型 - 支持多GPU
def load_embedding_model(embedding_model_path, device_id):

    device = f"cuda:{device_id}"
    print(f"嵌入模型将使用设备: {device}")

    model = SentenceTransformer(embedding_model_path, device=device)

    return model

# 初始化Milvus向量数据库
def init_milvus_connection(host='localhost', port='19530'):
    try:
        connections.connect(host=host, port=port)
        print("成功连接到Milvus")
    except Exception as e:
        print(f"连接Milvus失败: {e}")

def create_milvus_collection(collection_name="medical_knowledge", dim=1024):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    # 创建schema
    schema = CollectionSchema(fields, "医疗知识库集合")

    # 创建集合
    collection = Collection(name=collection_name, schema=schema)

    # 创建索引
    index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }

    collection.create_index("vector", index_params)
    print(f"集合 {collection_name} 创建成功")
    return collection

# 设置多进程启动方法为 'spawn'
mp.set_start_method('spawn', force=True)

def process_huatuo_dataset_single(input_path, output_path, collection, device_id, batch_size=128, max_samples=None):
    """
    使用单进程处理JSONL格式的Huatuo-26M数据集，并指定GPU设备

    参数:
    - device_id: 要使用的GPU设备ID
    """

    # 初始化嵌入模型到指定设备
    embedding_model = SentenceTransformer(
        os.path.join(PROJECT_ROOT, "models", "BAAI", "bge-large-zh-v1.5"),
        device=f"cuda:{device_id}"
    )
    print(f"嵌入模型已加载到设备: cuda:{device_id}")

    # 读取所有数据
    all_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            try:
                data = json.loads(line.strip())

                # 根据数据结构提取文本
                question = data.get("question", "")
                answer = data.get("answer", "")
                text = f"问题: {question}\n答案: {answer}"

                # 获取类别信息
                category = data.get("related_diseases", "未知")

                all_data.append({
                    "text": text,
                    "source": "huatuo-26m",
                    "category": category
                })

            except json.JSONDecodeError as e:
                print(f"解析JSON错误 (行 {i+1}): {e}")
                continue

    print(f"总共读取 {len(all_data)} 条数据，开始处理...")

    # 分批处理数据
    for i in range(0, len(all_data), batch_size):
        batch_data = all_data[i:i+batch_size]

        texts = [item["text"] for item in batch_data]
        sources = [item["source"] for item in batch_data]
        categories = [item["category"] for item in batch_data]

        # 生成嵌入向量
        embeddings = embedding_model.encode(texts).tolist()

        # 准备插入数据
        entities = [
            texts,  # text字段
            sources,  # source字段
            categories,  # category字段
            embeddings  # vector字段
        ]

        # 插入到Milvus
        collection.insert(entities)

        # 清理GPU内存
        torch.cuda.empty_cache()

        if (i // batch_size) % 10 == 0:  # 每10批打印一次进度
            print(f"已处理 {min(i+batch_size, len(all_data))}/{len(all_data)} 条数据")

    # 将集合加载到内存
    collection.load()
    print(f"数据集处理完成，总共处理了 {len(all_data)} 条数据")

    # 保存处理后的数据
    if output_path:
        processed_data = {
            "total_samples": len(all_data),
            "collection_name": collection.name,
            "batch_size": batch_size,
            "device_used": f"cuda:{device_id}"
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # 清理资源
    del embedding_model
    torch.cuda.empty_cache()

