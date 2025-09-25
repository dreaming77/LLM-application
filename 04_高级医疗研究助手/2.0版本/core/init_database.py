# init_database.py
from utils import init_milvus_connection, create_milvus_collection, process_huatuo_dataset_single
import os
import torch


def main(input_path, output_path):
    # 检查可用GPU

    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        memory_free = memory_total - memory_allocated

        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  可用内存: {memory_free:.2f}GB")

    # 初始化Milvus连接
    init_milvus_connection()

    # 创建集合
    collection = create_milvus_collection()

    # 确保输入文件存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件 {input_path} 不存在")
        return

    # 选择最空闲的GPU (1-7)
    best_device = 3  # 默认使用GPU 3
    best_memory_free = 0

    # 处理数据集（使用单进程处理，指定GPU设备）
    process_huatuo_dataset_single(
        input_path=input_path,
        output_path=output_path,
        collection=collection,
        batch_size=128,  # 减小批次大小以减少内存使用
        device_id=best_device  # 使用选择的GPU设备
    )

    print("数据库初始化完成")

if __name__ == "__main__":
    INPUT_PATH = '../data/huatuo_dataset/format_data.jsonl'
    OUTPUT_PATH = '../data/processed/processed_data.json'

    # 设置多进程启动方法
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main(input_path=INPUT_PATH, output_path=OUTPUT_PATH)
