from typing import List, Dict, Any
from state import ResearchState
from utils import init_milvus_connection, load_embedding_model
from pymilvus import Collection
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import json


# 全局变量，避免重复加载模型
_milvus_collection = None
_embedding_model = None
_llm_model = None
_llm_tokenizer = None
_device_choice = 4          # 使用GPU:4，请根据实际情况修改
_device = f"cuda:{_device_choice}" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {_device}")

def get_milvus_collection():
    """获取Milvus集合实例"""
    global _milvus_collection
    if _milvus_collection is None:
        init_milvus_connection()
        _milvus_collection = Collection("medical_knowledge")
        _milvus_collection.load()
    return _milvus_collection


def get_embedding_model(device_id=5):
    """获取嵌入模型实例"""
    global _embedding_model
    if _embedding_model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "models", "BAAI", "bge-large-zh-v1.5")
        _embedding_model = load_embedding_model(model_path, device_id)      # 尽量使嵌入模型与大模型存入的gpu不一样，减少使用空间
    return _embedding_model


def get_llm_model():
    """获取LLM模型和分词器实例"""
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "models", "Qwen2.5-7B-Instruct")

        # 配置bitsandbytes量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # 加载模型和分词器
        _llm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=_device ,
            trust_remote_code=True
        )

        _llm_model = _llm_model.to(_device)
    return _llm_model, _llm_tokenizer


def plan_research(state: ResearchState) -> dict:
    """根据用户查询生成研究计划（搜索查询）"""
    user_query = state["user_query"]

    # 获取LLM模型和分词器
    model, tokenizer = get_llm_model()

    # 构建提示词
    prompt = f"""你是一个专业的医疗研究助手。请根据以下用户查询，生成3-5个相关的搜索查询，用于从医疗知识库中检索相关信息。
        
        用户查询: {user_query}
        
        请以JSON格式返回结果，包含一个"queries"字段，值为字符串数组。
        示例: {{"queries": ["查询1", "查询2", "查询3"]}}
        
        请直接返回JSON格式，不要有其他内容:"""

    # 生成响应
    inputs = tokenizer(prompt, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取JSON部分
    try:
        json_start = response.find('{')
        json_end = response.find('}') + 1
        json_str = response[json_start:json_end]
        result = json.loads(json_str)
        queries = result.get("queries", [])
    except (json.JSONDecodeError, AttributeError):
        # 如果解析失败，使用备用查询
        queries = [user_query, f"{user_query} 医疗", f"{user_query} 治疗方法"]

    print(f"生成的搜索查询: {queries}")
    return {"search_queries": queries}


def retrieve_information(state: ResearchState) -> dict:
    """从Milvus向量数据库中检索相关信息"""
    search_queries = state["search_queries"]
    collection = get_milvus_collection()
    embedding_model = get_embedding_model()

    retrieved_documents = []

    for query in search_queries:
        # 生成查询向量
        query_embedding = embedding_model.encode([query])

        # 搜索参数
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 10}
        }

        # 执行搜索
        results = collection.search(
            data=query_embedding,
            anns_field="vector",
            param=search_params,
            limit=3,  # 每个查询返回3个最相关文档
            output_fields=["text", "source", "category"]
        )

        # 处理结果
        for hits in results:
            for hit in hits:
                document = {
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "category": hit.entity.get("category"),
                    "score": hit.score
                }
                retrieved_documents.append(document)

    print(f"检索到 {len(retrieved_documents)} 个文档")
    return {"retrieved_documents": retrieved_documents}


def filter_documents(state: ResearchState) -> dict:
    """过滤检索到的文档，去除重复和低质量内容"""
    retrieved_documents = state["retrieved_documents"]

    # 去重：基于文本内容的简单去重
    seen_texts = set()
    filtered_documents = []

    for doc in retrieved_documents:
        text = doc["text"]
        # 简单去重和过滤低分文档
        if text not in seen_texts and doc.get("score", 0) > 0.45:
            seen_texts.add(text)
            filtered_documents.append(doc)

    print(f"过滤后剩余 {len(filtered_documents)} 个文档")
    return {"filtered_documents": filtered_documents}


def synthesize_information(state: ResearchState) -> dict:
    """综合过滤后的文档信息"""
    user_query = state["user_query"]
    filtered_documents = state["filtered_documents"]

    if not filtered_documents:
        return {"synthesized_content": "未找到相关信息"}

    # 获取LLM模型和分词器
    model, tokenizer = get_llm_model()

    # 构建文档上下文
    context = "。\n\n".join([f"文档 {i + 1}:\n{doc['text']}" for i, doc in enumerate(filtered_documents[:3])])  # 限制前3个文档

    # 构建提示词
    prompt = f"""你是一个专业的医疗研究员。请根据以下文档内容，综合回答用户的问题。
        
        用户问题: {user_query}
        
        相关文档:
        {context}
        
        请基于以上文档内容，提供一个综合、准确且有条理的回答。确保回答专业且易于理解。
        注意: 只使用提供文档中的信息，不要添加外部知识。
        
        综合回答:"""

    # 生成响应
    inputs = tokenizer(prompt, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取回答部分（去除提示词）
    synthesized_content = response[len(prompt):].strip()

    print(f"生成长度为 {len(synthesized_content)} 的综合内容")
    return {"synthesized_content": synthesized_content}


def generate_report(state: ResearchState) -> dict:
    """生成最终研究报告"""
    user_query = state["user_query"]
    synthesized_content = state["synthesized_content"]

    # 获取LLM模型和分词器
    model, tokenizer = get_llm_model()

    # 构建提示词
    prompt = f"""你是一个专业的医疗报告撰写人。请基于以下综合信息，为用户查询生成一份结构完整、专业的医疗研究报告。
        用户查询: {user_query}
        综合信息:
        {synthesized_content}
        
        请生成一份包含以下部分的完整报告:
        1. 概述
        2. 病因与风险因素
        3. 症状与诊断
        4. 治疗方法
        5. 预防与建议
        6. 总结
        
        确保报告专业、准确、易于理解，并基于提供的综合信息。不要有太多的换行符。
        
        医疗研究报告:"""

    # 生成响应
    inputs = tokenizer(prompt, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=600,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取报告部分（去除提示词）
    final_report = response[len(prompt):].strip()

    print(f"最终报告已生成")
    return {"final_report": final_report}

if __name__ == "__main__":
    """单元测试"""
    # 初始化状态
    initial_state = ResearchState(user_query="没有做任何安全措施，也不是安全期，请问最早什么时候能知道怀孕了啊？")

    from graph import research_graph
    # 运行工作流
    result = research_graph.invoke(initial_state)
    # 输出结果
    print(result["final_report"])
