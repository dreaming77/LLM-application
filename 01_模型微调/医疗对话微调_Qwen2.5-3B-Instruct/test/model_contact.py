# 测试模型交互能力

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 加载模型
llm_model_path = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-3B-Instruct")

# 配置bitsandbytes量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
llm_device_id = 2

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    llm_model_path,
    device_map={"": f"cuda:{llm_device_id}"},
    dtype="auto",
    quantization_config=quantization_config,
    trust_remote_code=True
).eval()

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
