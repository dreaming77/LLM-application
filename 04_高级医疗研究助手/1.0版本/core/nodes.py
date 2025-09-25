from .utils import load_embedding_model
from pymilvus import Collection
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
import functools
import logging

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 初始化模型（全局变量，避免重复加载）
embedding_model = None
milvus_collection = None
llm_model = None
llm_tokenizer = None

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行出错: {str(e)}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise
    return wrapper


def init_models(embedding_model_path, llm_model_path, embedding_device_id=5, llm_device_id=6):
    """
    初始化所有模型，并指定使用的GPU设备，对Qwen-7B-Chat进行bitsandbytes量化
    """
    global embedding_model, milvus_collection, llm_model, llm_tokenizer

    # 初始化嵌入模型
    if embedding_model is None:
        print("正在初始化嵌入模型...")
        embedding_model = load_embedding_model(embedding_model_path, device_id=embedding_device_id)
        print("嵌入模型初始化完成")

    # 初始化Milvus连接
    if milvus_collection is None:
        print("正在初始化Milvus连接...")
        from pymilvus import connections
        connections.connect(host='localhost', port='19530')
        milvus_collection = Collection("medical_knowledge")
        milvus_collection.load()
        print("Milvus连接初始化完成")

    # 初始化Qwen-7B-Chat模型，使用bitsandbytes量化
    if llm_model is None or llm_tokenizer is None:
        print("正在初始化LLM模型和分词器...")
        try:
            # 配置bitsandbytes量化
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            print(f"正在加载分词器从: {llm_model_path}")
            llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_model_path,
                trust_remote_code=True,
                padding_side='left'  # 确保填充在左侧
            )

            # 手动设置特殊令牌 - 这是关键修复
            # Qwen模型使用特定的特殊令牌
            if hasattr(llm_tokenizer, 'special_tokens_map'):
                special_tokens = llm_tokenizer.special_tokens_map
                print(f"特殊令牌映射: {special_tokens}")

                # 设置eos_token
                if 'eos_token' in special_tokens and special_tokens['eos_token'] is not None:
                    llm_tokenizer.eos_token = special_tokens['eos_token']
                else:
                    # 使用Qwen默认的结束令牌
                    llm_tokenizer.eos_token = '<|endoftext|>'

                # 设置pad_token
                if 'pad_token' in special_tokens and special_tokens['pad_token'] is not None:
                    llm_tokenizer.pad_token = special_tokens['pad_token']
                else:
                    # 使用eos_token作为pad_token
                    llm_tokenizer.pad_token = llm_tokenizer.eos_token

                # 设置bos_token
                if 'bos_token' in special_tokens and special_tokens['bos_token'] is not None:
                    llm_tokenizer.bos_token = special_tokens['bos_token']
                else:
                    # 使用Qwen默认的开始令牌
                    llm_tokenizer.bos_token = '<|endoftext|>'

                # 设置unk_token
                if 'unk_token' in special_tokens and special_tokens['unk_token'] is not None:
                    llm_tokenizer.unk_token = special_tokens['unk_token']
                else:
                    # 使用Qwen默认的未知令牌
                    llm_tokenizer.unk_token = '<|endoftext|>'
            else:
                # 如果没有特殊令牌映射，使用Qwen默认值
                llm_tokenizer.eos_token = '<|endoftext|>'
                llm_tokenizer.pad_token = '<|endoftext|>'
                llm_tokenizer.bos_token = '<|endoftext|>'
                llm_tokenizer.unk_token = '<|endoftext|>'

            # 确保令牌ID正确设置
            if llm_tokenizer.eos_token_id is None:
                # 尝试通过编码获取ID
                eos_encoded = llm_tokenizer.encode(llm_tokenizer.eos_token, add_special_tokens=False)
                if eos_encoded:
                    llm_tokenizer.eos_token_id = eos_encoded[0]
                else:
                    # 使用默认ID
                    llm_tokenizer.eos_token_id = 151643

            if llm_tokenizer.pad_token_id is None:
                llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

            if llm_tokenizer.bos_token_id is None:
                llm_tokenizer.bos_token_id = llm_tokenizer.eos_token_id

            if llm_tokenizer.unk_token_id is None:
                llm_tokenizer.unk_token_id = llm_tokenizer.eos_token_id

            print(f"设置特殊令牌:")
            print(f"  eos_token: {llm_tokenizer.eos_token} (ID: {llm_tokenizer.eos_token_id})")
            print(f"  pad_token: {llm_tokenizer.pad_token} (ID: {llm_tokenizer.pad_token_id})")
            print(f"  bos_token: {llm_tokenizer.bos_token} (ID: {llm_tokenizer.bos_token_id})")
            print(f"  unk_token: {llm_tokenizer.unk_token} (ID: {llm_tokenizer.unk_token_id})")

            print("分词器加载成功")

            print(f"正在加载模型从: {llm_model_path}")
            # 使用量化配置加载模型
            llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                device_map={"": f"cuda:{llm_device_id}"},
                quantization_config=quantization_config,
                trust_remote_code=True
            )

            # 确保模型知道特殊令牌ID
            if hasattr(llm_model.config, 'pad_token_id') and llm_model.config.pad_token_id is None:
                llm_model.config.pad_token_id = llm_tokenizer.pad_token_id

            if hasattr(llm_model.config, 'eos_token_id') and llm_model.config.eos_token_id is None:
                llm_model.config.eos_token_id = llm_tokenizer.eos_token_id

            if hasattr(llm_model.config, 'bos_token_id') and llm_model.config.bos_token_id is None:
                llm_model.config.bos_token_id = llm_tokenizer.bos_token_id

            print("使用bitsandbytes量化加载Qwen-7B-Chat模型成功")

            # 检查模型和设备
            print(f"模型设备: {llm_model.device}")
            print(f"模型参数数量: {sum(p.numel() for p in llm_model.parameters()):,}")

        except Exception as e:
            print(f"量化加载失败: {str(e)}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            # 如果量化加载失败，尝试非量化加载
            try:
                llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_path,
                    device_map={"": f"cuda:{llm_device_id}"},
                    trust_remote_code=True
                )
                print("使用非量化方式加载模型成功")
            except Exception as e2:
                print(f"非量化加载也失败: {str(e2)}")
                # 最后尝试CPU加载
                try:
                    llm_model = AutoModelForCausalLM.from_pretrained(
                        llm_model_path,
                        device_map={"": "cpu"},
                        trust_remote_code=True
                    )
                    print("使用CPU加载模型成功")
                except Exception as e_cpu:
                    print(f"CPU加载也失败: {str(e_cpu)}")
                    raise e_cpu

                
def log_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"函数 {func.__name__} 执行出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    return wrapper

@log_exceptions
def plan_research(state):
    """规划研究步骤"""
    print(f"开始规划研究: {state['user_query']}")
    global llm_model, llm_tokenizer
    # 检查模型和分词器是否已初始化
    if llm_model is None or llm_tokenizer is None:
        raise ValueError("模型或分词器未初始化")

    # 在开始前清理GPU内存
    torch.cuda.empty_cache()

    # 手动构建 Qwen 模型的对话格式
    system_prompt = "你是一位专业的医学研究助手。请将用户的医学问题分解为3-5个具体的、可搜索的子问题。确保这些问题覆盖了问题的不同方面，并且适合用于检索相关的医学知识。"
    user_query = f"用户问题: {state['user_query']}\n\n请列出搜索子问题:"

    # Qwen 模型的特殊格式
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

    # 准备模型输入
    inputs = prepare_model_inputs(llm_tokenizer, text, llm_model.device)

    if inputs is None:
        print("警告: 无法准备模型输入")
        # 使用默认问题
        sub_questions = [
            f"{state['user_query']}的定义和病因",
            f"{state['user_query']}的临床表现和诊断标准",
            f"{state['user_query']}的治疗方法和药物",
            f"{state['user_query']}的预防和预后"
        ]
        return {"search_queries": sub_questions}

    # 生成响应
    outputs = safe_model_generate(
        llm_model,
        inputs,
        max_new_tokens=512,  # 调整为合适的值
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=llm_tokenizer.pad_token_id,
        eos_token_id=llm_tokenizer.eos_token_id
    )

    if outputs is None:
        print("警告: 模型生成失败")
        # 使用默认问题
        sub_questions = [
            f"{state['user_query']}的定义和病因",
            f"{state['user_query']}的临床表现和诊断标准",
            f"{state['user_query']}的治疗方法和药物",
            f"{state['user_query']}的预防和预后"
        ]
        return {"search_queries": sub_questions}

    # 解码响应
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取生成的子问题
    lines = response.split('\n')
    sub_questions = []
    for line in lines:
        if line.strip() and any(char.isdigit() for char in line):
            # 移除数字和点
            question = line.split('.', 1)[-1].strip()
            if question:
                sub_questions.append(question)

    # 如果解析失败，使用默认问题
    if not sub_questions:
        sub_questions = [
            f"{state['user_query']}的定义和病因",
            f"{state['user_query']}的临床表现和诊断标准",
            f"{state['user_query']}的治疗方法和药物",
            f"{state['user_query']}的预防和预后"
        ]
    # 在返回前再次清理内存
    torch.cuda.empty_cache()
    print(f"生成的搜索查询: {sub_questions}")
    return {"search_queries": sub_questions}

@log_exceptions
def retrieve_information(state):
    """从知识库检索信息"""
    print(f"开始检索信息: {state['search_queries']}")
    torch.cuda.empty_cache()

    search_queries = state["search_queries"]
    all_documents = []

    for query in search_queries:
        # 将查询转换为向量
        query_vector = embedding_model.encode([query])[0].tolist()

        # 在Milvus中搜索
        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        results = milvus_collection.search(
            [query_vector],
            "vector",
            search_params,
            limit=5,
            output_fields=["text", "source", "category"]
        )

        # 处理搜索结果
        for hits in results:
            for hit in hits:
                document = {
                    "text": hit.entity.get("text"),
                    "source": hit.entity.get("source"),
                    "category": hit.entity.get("category"),
                    "score": hit.score,
                    "query": query
                }
                all_documents.append(document)

    torch.cuda.empty_cache()
    print(f"检索到的文档数量: {len(all_documents)}")
    return {"retrieved_documents": all_documents}

@log_exceptions
def filter_documents(state):
    """过滤和评估检索到的文档"""
    # 简单的基于分数的过滤
    print(f"开始过滤文档: {len(state['retrieved_documents'])} 个文档")
    filtered_docs = [
        doc for doc in state["retrieved_documents"]
        if doc["score"] > 0.3  # 调整阈值
    ]

    # 去重
    seen_texts = set()
    unique_docs = []
    for doc in filtered_docs:
        if doc["text"] not in seen_texts:
            seen_texts.add(doc["text"])
            unique_docs.append(doc)

    print(f"过滤后的文档数量: {len(unique_docs)}")
    return {"filtered_documents": unique_docs}


@log_exceptions
def synthesize_information(state):
    """综合信息并生成初步内容"""
    try:
        logger.info("开始综合信息")
        documents = state["filtered_documents"]

        # 限制文档数量，避免输入过长
        max_documents = 10
        if len(documents) > max_documents:
            documents = documents[:max_documents]
            logger.warning(f"文档数量过多，仅使用前 {max_documents} 个文档")

        context = "\n\n".join([f"来源: {doc['source']}\n内容: {doc['text']}" for doc in documents])
        logger.info(f"综合信息上下文长度: {len(context)}")

        # 手动构建 Qwen 模型的对话格式
        system_prompt = ("你是一位医学专家。请基于以下检索到的信息，综合回答用户的医学问题。请确保回答专业、准确，并注明信息的来源。"
                         "如果信息不足或存在矛盾，请明确指出。")
        user_query = f"用户问题: {state['user_query']}\n\n检索到的信息:\n{context}\n\n请基于以上信息提供综合回答:"

        # Qwen 模型的特殊格式
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

        logger.info(f"准备模型输入，文本长度: {len(text)}")

        # 准备模型输入
        inputs = prepare_model_inputs(llm_tokenizer, text, llm_model.device, max_length=2048)

        if inputs is None:
            logger.error("无法准备模型输入")
            return {"synthesized_content": "无法处理输入，请重试。"}

        logger.info(f"模型输入准备完成，输入ID形状: {inputs['input_ids'].shape}")

        # 生成响应 - 使用更小的 max_new_tokens
        logger.info("开始模型生成...")
        outputs = safe_model_generate(
            llm_model,
            inputs,
            max_new_tokens=512,  # 调整为合适的值
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id
        )

        if outputs is None:
            logger.error("模型生成返回None")
            return {"synthesized_content": "模型生成失败，请重试。"}

        logger.info("模型生成完成")

        # 解码响应
        synthesized = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 移除输入文本部分，只保留生成的响应
        if text in synthesized:
            synthesized = synthesized.replace(text, "").strip()

        logger.info(f"综合信息完成，响应长度: {len(synthesized)}")
        return {"synthesized_content": synthesized}

    except Exception as e:
        logger.error(f"综合信息过程中发生错误: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return {"synthesized_content": f"综合信息过程中发生错误: {str(e)}"}


@log_exceptions
def generate_report(state):
    """生成最终报告"""
    try:
        logger.info("开始生成报告")

        # 检查必要的输入是否存在
        if not state.get("synthesized_content") or not state.get("user_query"):
            logger.error("缺少必要的输入数据")
            return {"final_report": "无法生成报告：缺少必要的输入数据"}

        # 手动构建 Qwen 模型的对话格式
        system_prompt = "你是一位医学写作专家。请基于以下综合内容，撰写一份结构清晰、专业准确的医学报告。报告应包括以下部分:\n1. 概述\n2. 病因与发病机制\n3. 临床表现\n4. 诊断与鉴别诊断\n5. 治疗与管理\n6. 预防与预后\n7. 参考文献\n\n请使用专业的医学语言，并确保内容准确、完整。"
        user_query = f"用户问题: {state['user_query']}\n\n综合内容:\n{state['synthesized_content']}\n\n请撰写医学报告:"

        # Qwen 模型的特殊格式
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

        logger.info(f"准备模型输入，文本长度: {len(text)}")

        # 准备模型输入
        inputs = prepare_model_inputs(llm_tokenizer, text, llm_model.device, max_length=2048)

        if inputs is None:
            logger.error("无法准备模型输入")
            return {"final_report": "无法处理输入，请重试。"}

        logger.info(f"模型输入准备完成，输入ID形状: {inputs['input_ids'].shape}")

        # 生成响应
        logger.info("开始模型生成...")
        outputs = safe_model_generate(
            llm_model,
            inputs,
            max_new_tokens=512,  # 调整为合适的值
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id
        )

        if outputs is None:
            logger.error("模型生成返回None")
            return {"final_report": "模型生成失败，请重试。"}

        logger.info("模型生成完成")

        # 解码响应
        report = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 移除输入文本部分，只保留生成的响应
        if text in report:
            report = report.replace(text, "").strip()

        logger.info(f"报告生成完成，响应长度: {len(report)}")
        return {"final_report": report}

    except Exception as e:
        logger.error(f"报告生成过程中发生错误: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return {"final_report": f"报告生成过程中发生错误: {str(e)}"}


def safe_model_generate(model, inputs, **kwargs):
    """安全的模型生成函数，处理各种潜在问题"""
    try:
        # 在生成前清理GPU内存
        torch.cuda.empty_cache()

        # 确保输入格式正确
        if not isinstance(inputs, dict):
            logger.error("错误: 输入不是字典格式")
            return None

        # 确保必要的键存在
        if 'input_ids' not in inputs:
            logger.error("错误: 输入缺少 'input_ids' 键")
            return None

        # 确保输入不为空
        if inputs['input_ids'] is None or inputs['input_ids'].numel() == 0:
            logger.error("错误: 输入为空")
            return None

        # 确保输入在正确的设备上
        if inputs['input_ids'].device != model.device:
            inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}

        # 设置合理的默认生成参数
        default_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
            'do_sample': kwargs.get('do_sample', True),
            'top_p': kwargs.get('top_p', 0.9),
            'pad_token_id': model.config.pad_token_id,  # 使用模型的pad_token_id
            'eos_token_id': model.config.eos_token_id,  # 使用模型的eos_token_id
            'use_cache': True,  # 启用缓存以提高性能
        }

        # 更新用户提供的参数
        default_kwargs.update(kwargs)

        # 添加额外的安全检查
        if 'attention_mask' not in inputs:
            # 如果没有 attention_mask，创建一个
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            logger.warning("创建默认的 attention_mask")

        logger.info(f"模型生成参数: input_ids形状={inputs['input_ids'].shape}, kwargs={default_kwargs}")

        # 生成响应
        with torch.no_grad():  # 禁用梯度计算以节省内存
            result = model.generate(
                **inputs,
                **default_kwargs
            )

        # 生成后清理GPU内存
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        logger.error(f"模型生成错误: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 生成后清理GPU内存
        torch.cuda.empty_cache()
        return None


def validate_and_clean_input(text, max_length=4096):  # 增加最大长度
    """验证和清理输入文本"""
    if not text or not isinstance(text, str):
        return ""

    # 移除多余的空格和换行
    text = ' '.join(text.split())

    # 只在实际长度超过最大长度时截断
    if len(text) > max_length:
        # 尝试在句子边界处截断
        if '.' in text:
            truncated = text[:max_length]
            last_dot = truncated.rfind('.')
            if last_dot > 0:
                text = truncated[:last_dot+1] + "..."
            else:
                text = truncated + "..."
        else:
            text = text[:max_length] + "..."
        logger.warning(f"输入文本过长，已截断为 {len(text)} 字符")

    return text


def prepare_model_inputs(tokenizer, text, device, max_length=4096):
    """准备模型输入，确保格式正确"""
    try:
        # 验证和清理输入
        text = validate_and_clean_input(text, max_length)

        if not text:
            logger.error("输入文本为空")
            return None

        logger.info(f"处理文本长度: {len(text)}")

        # 确保分词器有填充令牌
        if tokenizer.pad_token is None:
            logger.error("分词器没有设置填充令牌")
            return None

        # 分词 - 使用左侧填充
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,  # 启用填充
            pad_to_multiple_of=8  # 优化GPU性能
        )

        # 检查输入是否有效
        if inputs is None or 'input_ids' not in inputs:
            logger.error("分词器返回了无效的输入")
            return None

        # 记录输入形状
        input_shape = inputs['input_ids'].shape
        logger.info(f"分词后输入形状: {input_shape}")

        # 确保所有必要的键都存在
        required_keys = ['input_ids', 'attention_mask']
        for key in required_keys:
            if key not in inputs:
                logger.warning(f"输入缺少键 '{key}'，使用默认值")
                if key == 'input_ids':
                    inputs[key] = torch.tensor([[tokenizer.pad_token_id]], device=device)
                elif key == 'attention_mask':
                    inputs[key] = torch.ones_like(inputs['input_ids'])

        # 移动到正确的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logger.info(f"输入准备完成: input_ids形状={inputs['input_ids'].shape}")
        return inputs

    except Exception as e:
        logger.error(f"准备模型输入时发生错误: {str(e)}")
        return None
