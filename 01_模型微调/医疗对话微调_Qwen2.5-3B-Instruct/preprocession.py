import pandas as pd
import json
import os
import chardet

# 定义系统提示内容 - 可以根据需要调整
SYSTEM_PROMPT = "你是一个专业、友善的医疗健康助手，请根据你的医学知识用中文回答用户的问题。如果不知道答案或者问题超出你的专业范围，请诚实告知。"

# 设置最小长度阈值
MIN_QUESTION_LENGTH = 5  # 用户问题最小长度
MIN_ANSWER_LENGTH = 5  # 助理回答最小长度


def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(100000)
        result = chardet.detect(raw_data)
        return result['encoding'], result['confidence']


def process_csv_with_encoding(input_csv_path, output_jsonl_path, encoding, batch_size=1000):
    """使用指定编码处理CSV文件"""
    try:
        print(f"尝试使用编码: {encoding}")
        df = pd.read_csv(input_csv_path, encoding=encoding)
        print(f"成功使用 {encoding} 编码读取文件")
        return df, encoding
    except (UnicodeDecodeError, LookupError) as e:
        print(f"编码 {encoding} 失败: {e}")
        return None, None


def process_csv_in_batches(input_csv_path, output_jsonl_path, batch_size=1000):
    # 检测文件编码
    detected_encoding, confidence = detect_encoding(input_csv_path)
    print(f"检测到编码: {detected_encoding}, 置信度: {confidence}")

    # 尝试使用检测到的编码
    df, used_encoding = process_csv_with_encoding(input_csv_path, output_jsonl_path, detected_encoding)

    # 如果失败，尝试其他常见中文编码
    if df is None:
        for enc in ['gbk', 'gb18030', 'utf-8', 'latin1']:
            df, used_encoding = process_csv_with_encoding(input_csv_path, output_jsonl_path, enc)
            if df is not None:
                break

    # 如果所有编码都失败，尝试使用错误处理
    if df is None:
        print("所有编码尝试失败，尝试使用错误处理模式")
        try:
            df = pd.read_csv(input_csv_path, encoding='gbk', errors='replace')
            used_encoding = 'gbk_with_replace'
            print("使用错误替换模式成功读取文件")
        except Exception as e:
            raise ValueError(f"无法读取文件 {input_csv_path}，尝试了多种编码和错误处理") from e

    # 分批处理数据
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size

    processed_count = 0
    skipped_count = 0  # 记录跳过的条目数

    with open(output_jsonl_path, 'a', encoding='utf-8') as f_out:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_rows)

            batch_df = df.iloc[start_idx:end_idx]

            for _, row in batch_df.iterrows():
                # 确保所有字段都是字符串类型
                ask = str(row['ask']) if pd.notna(row.get('ask', '')) else ""
                answer = str(row['answer']) if pd.notna(row.get('answer', '')) else ""

                # 检查是否有乱码字符
                if used_encoding == 'latin1' and any(ord(c) > 127 for c in ask + answer):
                    print(f"警告: 可能检测到乱码字符，使用编码: {used_encoding}")

                # 过滤掉简短的问题或回答
                if len(ask.strip()) < MIN_QUESTION_LENGTH or len(answer.strip()) < MIN_ANSWER_LENGTH:
                    skipped_count += 1
                    continue

                # 修改为Qwen2.5所需的messages格式，并添加系统提示
                messages = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": ask},
                        {"role": "assistant", "content": answer}
                    ]
                }
                f_out.write(json.dumps(messages, ensure_ascii=False) + '\n')
                processed_count += 1

            print(f"已处理 {end_idx}/{total_rows} 条记录，有效记录: {processed_count}，跳过记录: {skipped_count}")

    print(
        f"完成处理文件: {input_csv_path}, 使用编码: {used_encoding}, 总处理记录: {processed_count}, 跳过记录: {skipped_count}")
    return used_encoding, processed_count, skipped_count


# 处理所有科室的CSV文件
data_dir = "./dataset/Chinese-medical-dialogue-data/Data"
output_dir = "./dataset/processed_data"
os.makedirs(output_dir, exist_ok=True)

# 最终合并的文件
all_data_path = os.path.join(output_dir, "all_data.jsonl")

# 如果已存在合并文件，先删除
if os.path.exists(all_data_path):
    os.remove(all_data_path)

# 科室列表和对应的文件名
departments = {
    "Andriatria": "Andriatria.csv",
    "IM": "IM.csv",
    "OAGD": "OAGD.csv",
    "Oncology": "Oncology.csv",
    "Pediatric": "Pediatric.csv",
    "Surgical": "Surgical.csv"
}

# 记录每个文件使用的编码和处理统计
processing_stats = {}

# 处理每个科室的数据
for dept_name, csv_file in departments.items():
    csv_path = os.path.join(data_dir, dept_name, csv_file)
    if os.path.exists(csv_path):
        print(f"\n开始处理: {csv_path}")
        encoding, processed_count, skipped_count = process_csv_in_batches(csv_path, all_data_path, batch_size=4096)
        processing_stats[csv_path] = {
            "encoding": encoding,
            "processed": processed_count,
            "skipped": skipped_count
        }
    else:
        print(f"文件不存在: {csv_path}")

# 打印处理统计
total_processed = 0
total_skipped = 0
print("\n处理统计:")
for path, stats in processing_stats.items():
    print(
        f"{os.path.basename(path)}: 编码={stats['encoding']}, 有效记录={stats['processed']}, 跳过记录={stats['skipped']}")
    total_processed += stats['processed']
    total_skipped += stats['skipped']

print(f"\n总计: 有效记录={total_processed}, 跳过记录={total_skipped}")


def split_train_test_val(input_file, output_dir, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    """划分训练集、测试集和验证集"""
    # 确保比例之和为1
    assert train_ratio + test_ratio + val_ratio == 1.0, "比例之和必须为1"

    # 读取所有数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    import numpy as np
    # 随机打乱数据
    np.random.seed(42)  # 设置随机种子以确保可重复性
    np.random.shuffle(lines)

    # 计算划分点
    total_size = len(lines)
    train_end = int(total_size * train_ratio)
    test_end = train_end + int(total_size * test_ratio)

    # 划分数据
    train_data = lines[:train_end]
    test_data = lines[train_end:test_end]
    val_data = lines[test_end:]

    # 保存到不同文件
    train_path = os.path.join(output_dir, "train_data.jsonl")
    test_path = os.path.join(output_dir, "test_data.jsonl")
    val_path = os.path.join(output_dir, "val_data.jsonl")

    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_data)

    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_data)

    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_data)

    print(f"数据划分完成:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    print(f"  验证集: {len(val_data)} 条")

    return train_path, test_path, val_path


split_train_test_val("./dataset/processed_data/first_3000.jsonl", "./dataset/processed_data/")

print("\n所有数据预处理完成!")
