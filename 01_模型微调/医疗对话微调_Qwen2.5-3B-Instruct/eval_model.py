import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import json
import re
import nltk
import jieba  # 添加jieba用于中文分词


# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 模型路径
base_model_path = "./models/Qwen2.5-3B-Instruct"
fine_tuned_model_path = "./models/Qwen2.5-3B-instruct-finetuned"
lora_adapter_path = "./models/qwen2.5-3b-instruct-lora-medical-adapter"
test_dataset_path = "./dataset/processed_data/test_data.jsonl"  # 需要准备的测试集
sbert_model_path = "./models/paraphrase-multilingual-MiniLM-L12-v2"

# 加载语义相似度模型
print("加载语义相似度模型...")
sbert_model = SentenceTransformer(sbert_model_path)
sbert_model = sbert_model.to(device)

# 系统提示
SYSTEM_PROMPT = ("你是一个专业、友善的医疗健康助手，请根据你的医学知识用中文回答用户的问题。"
                 "首先分析用户提问的知识点做一个概念介绍，随后针对用户的问题提出一个方案")

# 加载测试数据集
test_dataset = load_dataset("json", data_files=test_dataset_path, split="train")
print(f"测试集大小: {len(test_dataset)}")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

results = None

# 加载基础模型
print("加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

# 加载微调模型
print("加载微调模型...")
fine_tuned_model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)


# 定义生成函数
def generate_response(model, tokenizer, messages, max_new_tokens=256):
    """使用模型生成响应"""
    # 格式化输入
    formatted_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]  # 记录输入token的长度

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    # 直接截取生成部分的token
    response_tokens = outputs[0][input_length:]

    # 解码生成部分
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)

    return response
"""
    1. do_sample=True（启用采样生成）
作用：控制模型生成时的选择策略，决定是 “采样生成” 还是 “贪婪解码”。
两种策略的区别：
    do_sample=False（贪婪解码）：每次选择概率最高的下一个 token，生成结果确定但可能单调（如重复句式）；
    do_sample=True（采样生成）：从 token 的概率分布中随机选择（概率高的 token 被选中的可能性大），
输出更多样、更自然（适合对话、创作等需要灵活性的场景）

    2. temperature=0.7（温度参数，控制采样随机性）
作用：调整 token 概率分布的 “陡峭程度”，间接控制生成的随机性和确定性。
原理：对 token 的 logits（未归一化的概率）进行缩放（logits / temperature），再通过 softmax 计算概率：
temperature→0：高概率 token 的概率被放大，几乎只选最高概率 token（接近贪婪解码，确定性高，多样性低）；
temperature→1：保持原始概率分布，随机性适中；
temperature>1：概率分布被拉平，低概率 token 也可能被选中（随机性高，可能更有创造力，但易出现不合理内容）。

    3. repetition_penalty=1.1（重复惩罚参数）
作用：减少生成文本中的重复内容（如重复短语、句式循环）。
原理：生成过程中，对已出现过的 token，降低其在后续生成中的概率（惩罚系数越大，概率降低越多）。
1.1 是温和的惩罚：既能抑制明显的重复（如 “你好，你好，你好”），又不会过度影响正常的语义连贯（惩罚过高可能导致语句断裂）。
适用场景：对话生成、长文本创作（这类场景容易出现重复），纯事实性输出（如问答）可适当降低（如 1.05）。
"""

# 中文分词函数
def chinese_tokenize(text):
    """使用jieba进行中文分词"""
    if not text:
        return []
    # 使用jieba进行分词
    return list(jieba.cut(text))


# 计算语义相似度
def calculate_semantic_similarity(reference, candidate):
    """计算两个文本的语义相似度"""
    try:
        # 生成嵌入向量
        ref_embedding = sbert_model.encode(reference, convert_to_tensor=True)
        cand_embedding = sbert_model.encode(candidate, convert_to_tensor=True)

        # 计算余弦相似度
        cosine_scores = util.pytorch_cos_sim(ref_embedding, cand_embedding)
        return cosine_scores.item()
    except Exception as e:
        print(f"语义相似度计算错误: {e}")
        return 0


# 修复ROUGE计算函数
def calculate_rouge_scores(reference, candidate):
    """计算ROUGE分数，处理中文文本"""
    try:
        # 使用空格分隔的中文文本（ROUGE库需要这样处理）
        ref_text = ' '.join(list(reference))
        cand_text = ' '.join(list(candidate))

        # 初始化ROUGE计算器
        rouge = Rouge()

        # 计算ROUGE分数
        scores = rouge.get_scores(cand_text, ref_text)
        return scores[0]
    except Exception as e:
        print(f"ROUGE计算错误: {e}")
        # 返回默认值
        return {
            "rouge-1": {"f": 0, "p": 0, "r": 0},
            "rouge-2": {"f": 0, "p": 0, "r": 0},
            "rouge-l": {"f": 0, "p": 0, "r": 0}
        }


# 评估函数
def evaluate_models(test_data, num_samples=100):
    """评估两个模型在测试集上的表现"""
    # 随机选择样本
    if num_samples < len(test_data):
        indices = np.random.choice(len(test_data), num_samples, replace=False)
        test_data = test_data.select(indices)

    base_scores = {
        "bleu": [],
        "rouge": [],
        "semantic_similarity": []
    }
    fine_tuned_scores = {
        "bleu": [],
        "rouge": [],
        "semantic_similarity": []
    }

    # 创建详细结果文件
    detailed_results = []

    smoothing = SmoothingFunction().method1

    for i, example in enumerate(test_data):
        print(f"处理样本 {i + 1}/{len(test_data)}")

        # 提取用户问题和真实回答
        user_query = None
        true_answer = None
        for msg in example["messages"]:
            if msg["role"] == "user":
                user_query = msg["content"]
            elif msg["role"] == "assistant":
                true_answer = msg["content"]

        if not user_query or not true_answer:
            continue

        # 构建消息格式
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        # 基础模型生成回答
        try:
            base_answer = generate_response(base_model, tokenizer, messages)
            print(f"基础模型回答: {base_answer[:100]}..." if len(base_answer) > 100 else f"基础模型回答: {base_answer}")
        except Exception as e:
            print(f"基础模型生成错误: {e}")
            continue

        # 微调模型生成回答
        try:
            fine_tuned_answer = generate_response(fine_tuned_model, tokenizer, messages)
            print(f"微调模型回答: {fine_tuned_answer[:100]}..." if len(
                fine_tuned_answer) > 100 else f"微调模型回答: {fine_tuned_answer}")
        except Exception as e:
            print(f"微调模型生成错误: {e}")
            continue

        # 计算BLEU分数
        base_tokens = chinese_tokenize(base_answer)
        fine_tuned_tokens = chinese_tokenize(fine_tuned_answer)
        true_tokens = chinese_tokenize(true_answer)

        base_bleu = sentence_bleu([true_tokens], base_tokens, smoothing_function=smoothing)
        fine_tuned_bleu = sentence_bleu([true_tokens], fine_tuned_tokens, smoothing_function=smoothing)

        base_scores["bleu"].append(base_bleu)
        fine_tuned_scores["bleu"].append(fine_tuned_bleu)

        # 计算ROUGE分数
        base_rouge = calculate_rouge_scores(true_answer, base_answer)
        fine_tuned_rouge = calculate_rouge_scores(true_answer, fine_tuned_answer)

        base_scores["rouge"].append(base_rouge)
        fine_tuned_scores["rouge"].append(fine_tuned_rouge)

        # 计算语义相似度
        base_semantic = calculate_semantic_similarity(true_answer, base_answer)
        fine_tuned_semantic = calculate_semantic_similarity(true_answer, fine_tuned_answer)

        base_scores["semantic_similarity"].append(base_semantic)
        fine_tuned_scores["semantic_similarity"].append(fine_tuned_semantic)

        # 保存详细结果
        detailed_results.append({
            "user_query": user_query,
            "true_answer": true_answer,
            "base_answer": base_answer,
            "fine_tuned_answer": fine_tuned_answer,
            "base_bleu": float(base_bleu),
            "fine_tuned_bleu": float(fine_tuned_bleu),
            "base_rouge": base_rouge,
            "fine_tuned_rouge": fine_tuned_rouge,
            "base_semantic_similarity": float(base_semantic),
            "fine_tuned_semantic_similarity": float(fine_tuned_semantic)
        })

    # 计算平均分数
    results = {
        "base_model": {
            "bleu": np.mean(base_scores["bleu"]) if base_scores["bleu"] else 0,
            "rouge_1": np.mean([score["rouge-1"]["f"] for score in base_scores["rouge"]]) if base_scores[
                "rouge"] else 0,
            "rouge_2": np.mean([score["rouge-2"]["f"] for score in base_scores["rouge"]]) if base_scores[
                "rouge"] else 0,
            "rouge_l": np.mean([score["rouge-l"]["f"] for score in base_scores["rouge"]]) if base_scores[
                "rouge"] else 0,
            "semantic_similarity": np.mean(base_scores["semantic_similarity"]) if base_scores[
                "semantic_similarity"] else 0
        },
        "fine_tuned_model": {
            "bleu": np.mean(fine_tuned_scores["bleu"]) if fine_tuned_scores["bleu"] else 0,
            "rouge_1": np.mean([score["rouge-1"]["f"] for score in fine_tuned_scores["rouge"]]) if fine_tuned_scores[
                "rouge"] else 0,
            "rouge_2": np.mean([score["rouge-2"]["f"] for score in fine_tuned_scores["rouge"]]) if fine_tuned_scores[
                "rouge"] else 0,
            "rouge_l": np.mean([score["rouge-l"]["f"] for score in fine_tuned_scores["rouge"]]) if fine_tuned_scores[
                "rouge"] else 0,
            "semantic_similarity": np.mean(fine_tuned_scores["semantic_similarity"]) if fine_tuned_scores[
                "semantic_similarity"] else 0
        },
        "detailed_results": detailed_results
    }

    return results


# 运行评估
print("开始评估模型...")
results = evaluate_models(test_dataset, num_samples=50)  # 评估50个样本

# 打印结果
print("\n评估结果:")
print("=" * 60)
print("基础模型:")
print(f"  BLEU: {results['base_model']['bleu']:.4f}")
print(f"  ROUGE-1: {results['base_model']['rouge_1']:.4f}")
print(f"  ROUGE-2: {results['base_model']['rouge_2']:.4f}")
print(f"  ROUGE-L: {results['base_model']['rouge_l']:.4f}")
print(f"  语义相似度: {results['base_model']['semantic_similarity']:.4f}")

print("\n微调模型:")
print(f"  BLEU: {results['fine_tuned_model']['bleu']:.4f}")
print(f"  ROUGE-1: {results['fine_tuned_model']['rouge_1']:.4f}")
print(f"  ROUGE-2: {results['fine_tuned_model']['rouge_2']:.4f}")
print(f"  ROUGE-L: {results['fine_tuned_model']['rouge_l']:.4f}")
print(f"  语义相似度: {results['fine_tuned_model']['semantic_similarity']:.4f}")

# 计算改进百分比
metrics = ["bleu", "rouge_1", "rouge_2", "rouge_l", "semantic_similarity"]
improvement = {}

for metric in metrics:
    base_val = results["base_model"][metric]
    fine_tuned_val = results["fine_tuned_model"][metric]
    if base_val > 0:
        improvement[metric] = (fine_tuned_val - base_val) / base_val * 100
    else:
        improvement[metric] = float('inf') if fine_tuned_val > 0 else 0

print("\n改进百分比:")
print("=" * 60)
for metric, percent in improvement.items():
    if percent == float('inf'):
        print(f"{metric.upper()}: +∞%")
    else:
        print(f"{metric.upper()}: {percent:+.2f}%")

# 保存结果
with open("model_comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n结果已保存到 model_comparison_results.json")

# 保存详细结果到单独文件
with open("detailed_comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(results["detailed_results"], f, ensure_ascii=False, indent=2)

print("详细结果已保存到 detailed_comparison_results.json")

# 打印一些示例对比
print("\n示例对比:")
print("=" * 60)
# 随机选择几个示例展示
sample_indices = np.random.choice(len(results["detailed_results"]), min(3, len(results["detailed_results"])),
                                  replace=False)

for i, idx in enumerate(sample_indices):
    sample = results["detailed_results"][idx]
    print(f"\n示例 {i + 1}:")
    print(f"用户问题: {sample['user_query']}")
    print(f"真实回答: {sample['true_answer']}")
    print(f"基础模型回答: {sample['base_answer']}")
    print(f"微调模型回答: {sample['fine_tuned_answer']}")
    print(f"基础模型ROUGE-L: {sample['base_rouge']['rouge-l']['f']:.4f}")
    print(f"微调模型ROUGE-L: {sample['fine_tuned_rouge']['rouge-l']['f']:.4f}")
    print(f"基础模型语义相似度: {sample['base_semantic_similarity']:.4f}")
    print(f"微调模型语义相似度: {sample['fine_tuned_semantic_similarity']:.4f}")
    print("-" * 60)
