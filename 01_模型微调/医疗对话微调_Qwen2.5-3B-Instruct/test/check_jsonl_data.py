import json


def preview_jsonl(file_path, num_lines=10):
    """预览JSONL文件的前几行"""
    print(f"预览文件: {file_path}")
    print("-" * 50)

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            try:
                data = json.loads(line.strip())
                print(f"第 {i + 1} 行:")
                print(json.dumps(data, ensure_ascii=False, indent=2))
                print("-" * 30)
            except json.JSONDecodeError as e:
                print(f"第 {i + 1} 行解析错误: {e}")
                print(f"原始内容: {line}")
                print("-" * 30)


# 指定文件路径
file_path = "../dataset/processed_data/all_data.jsonl"

# 预览前10行
preview_jsonl(file_path, 300)

import json


def extract_first_n_jsonl(input_path, output_path, n=5000):
    count = 0
    # 读取输入文件并提取前n个条目
    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if count >= n:
                break  # 达到目标数量，停止读取
            try:
                # 验证JSON格式（可选，视文件完整性而定）
                json.loads(line)
                # 写入到输出文件
                outfile.write(line)
                count += 1
            except json.JSONDecodeError:
                print(f"跳过无效JSON行（行号：{count + 1}）")
                continue

    print(f"已提取前{count}个条目到 {output_path}")


# 使用示例
if __name__ == "__main__":
    input_file = "../dataset/processed_data/all_data.jsonl"  # 替换为你的输入JSONL文件路径
    output_file = "../dataset/processed_data/first_30000.jsonl"  # 输出文件路径
    # extract_first_n_jsonl(input_file, output_file, n=30000)
