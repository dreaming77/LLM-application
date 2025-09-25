import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json
import pandas as pd
from collections import defaultdict


# 加载嵌入模型（使用与创建索引时相同的模型）
model_path = "../models/text2vec-large-chinese"  # 替换为你的实际路径
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={'device': 'cuda:5'},
    encode_kwargs={"normalize_embeddings": True}
)

# 加载FAISS索引
print("加载FAISS索引...")
vectorstore = FAISS.load_local(
                "../faiss_vector_db/law_faiss_index",
                            embeddings,
                            allow_dangerous_deserialization=True
    )

# 获取所有存储的文档
print("提取存储的文档...")
all_documents = vectorstore.docstore._dict

print(f"总共存储了 {len(all_documents)} 个文档")

# 创建一个列表来存储所有文档的详细信息
all_docs_info = []

# 遍历所有文档
for doc_id, document in all_documents.items():
    doc_info = {
        "id": doc_id,
        "content": document.page_content,
        "metadata": document.metadata
    }
    all_docs_info.append(doc_info)


# 按法律、章节和条文编号排序
def article_key(article_id):
    """将中文数字转换为阿拉伯数字以便排序"""
    chinese_numbers = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                       '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                       '百': 100, '千': 1000, '零': 0}

    if not article_id or not article_id.startswith('第'):
        return 0

    # 提取数字部分
    num_str = article_id[1:].replace('条', '')

    # 简单转换（实际应用中可能需要更复杂的转换）
    try:
        # 尝试直接转换阿拉伯数字
        return int(num_str)
    except:
        # 如果是中文数字，进行简单映射
        total = 0
        for char in num_str:
            total += chinese_numbers.get(char, 0)
        return total


all_docs_info.sort(key=lambda x: (
    x["metadata"].get("law_name", ""),
    x["metadata"].get("chapter_title", ""),
    article_key(x["metadata"].get("article_id", ""))
))

# 打印前几个文档作为示例
print("\n=== 前10个文档示例 ===")
for i, doc_info in enumerate(all_docs_info[:100]):
    print(f"\n文档 {i + 1}:")
    print(f"ID: {doc_info['id']}")
    print(f"法律: {doc_info['metadata'].get('law_name', 'N/A')}")
    print(f"章节: {doc_info['metadata'].get('chapter_title', 'N/A')}")
    print(f"节: {doc_info['metadata'].get('section_title', 'N/A')}")
    print(f"条文: {doc_info['metadata'].get('article_id', 'N/A')}")
    print(f"内容预览: {doc_info['content']}")

# 将全部内容保存到JSON文件以便查看
output_file = "faiss_contents.json"
print(f"\n将所有文档内容保存到 {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_docs_info, f, ensure_ascii=False, indent=2)

# 创建Excel文件，按法律和章节组织
print("创建Excel文件...")
excel_data = []

for doc_info in all_docs_info:
    excel_data.append({
        "法律": doc_info["metadata"].get("law_name", ""),
        "章节": doc_info["metadata"].get("chapter_title", ""),
        "节": doc_info["metadata"].get("section_title", ""),
        "条文": doc_info["metadata"].get("article_id", ""),
        "内容": doc_info["content"]
    })

df = pd.DataFrame(excel_data)
excel_file = "faiss_contents.xlsx"
df.to_excel(excel_file, index=False, engine='openpyxl')

# 统计信息
law_counts = defaultdict(int)
chapter_counts = defaultdict(int)
section_counts = defaultdict(int)

for doc_info in all_docs_info:
    law_name = doc_info["metadata"].get("law_name", "未知")
    chapter_title = doc_info["metadata"].get("chapter_title", "未知")
    section_title = doc_info["metadata"].get("section_title", "未知")

    law_counts[law_name] += 1
    chapter_counts[f"{law_name} - {chapter_title}"] += 1
    if section_title:  # 只统计有节标题的
        section_counts[f"{law_name} - {chapter_title} - {section_title}"] += 1

print("\n=== 统计信息 ===")
print("各法律文档数量:")
for law, count in sorted(law_counts.items()):
    print(f"  {law}: {count} 条")

print("\n各章节文档数量:")
for chapter, count in sorted(chapter_counts.items()):
    print(f"  {chapter}: {count} 条")

if section_counts:
    print("\n各节文档数量:")
    for section, count in sorted(section_counts.items()):
        print(f"  {section}: {count} 条")

# 检查是否有重复的条文
article_ids = defaultdict(list)
for doc_info in all_docs_info:
    law_name = doc_info["metadata"].get("law_name", "未知")
    article_id = doc_info["metadata"].get("article_id", "未知")
    key = f"{law_name}-{article_id}"
    article_ids[key].append(doc_info["id"])

duplicates = {k: v for k, v in article_ids.items() if len(v) > 1}
if duplicates:
    print(f"\n警告: 发现 {len(duplicates)} 个重复条文:")
    for article, ids in list(duplicates.items())[:5]:  # 只显示前5个重复
        print(f"  {article}: {len(ids)} 个实例")
else:
    print("\n没有发现重复条文")

print(f"\n完成! 已生成:")
print(f"- JSON文件: {output_file}")
print(f"- Excel文件: {excel_file}")

