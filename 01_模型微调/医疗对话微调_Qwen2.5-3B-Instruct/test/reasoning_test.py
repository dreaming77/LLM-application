import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# 加载基础模型和tokenizer
model_path = "../models/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 确保有填充token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(
    base_model,
    "./models/qwen2.5-3b-instruct-lora-medical-adapter"
)

# 合并LoRA权重到基础模型（可选，可以提高推理速度）
model = model.merge_and_unload()

# 设置模型为评估模式
model.eval()

# 系统提示（与训练时相同）
SYSTEM_PROMPT = "你是一个专业、友善的医疗健康助手，请根据你的医学知识用中文回答用户的问题。如果不知道答案或者问题超出你的专业范围，请诚实告知。"

# 测试问题
test_questions = [
    "最近给孩子洗澡的时候发现孩子身上有一块红点儿.于是带孩子去医生检查.大夫说是疝气.需要手术治疗.请问手术过程要多长时间啊？",
    "我是一个在校大学生，平时除了上课就喜欢在寝室玩电脑，坐的时间比较多，平时也缺乏运动，最近一段时间感觉自己上厕所时尿不干净，"
    "还有白色分泌物流出，自己查了好像是前列腺炎，我还那么年轻，请问：得前列腺炎需要怎么样治疗好？",
    "我现在主要是勃起不坚，有时候阴茎勃起正常，但是坚挺时间不长，去医院仔细检查，"
    "医生说是前列腺发炎和供血不足导致的阳痿。请问勃起不坚是怎么样的病症？",
    "男44岁最近比较疲惫，身体有感觉到了不健康，会是得这个病了吗，请问：做爱三分钟是不是属于早泄？",
    "男，目前51岁，近半年，发现，房事大不如前，此外，不足2,3分钟就射了，请问：男性早泄是由哪些方面引发的。"
]


def generate_response(question, max_length=512, temperature=0.7):
    """生成回答"""
    # 构建对话格式
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # 减少重复
        )

    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取助手的回答
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

    return response.strip()


# 进行测试
print("开始测试微调后的模型...\n")

for i, question in enumerate(test_questions, 1):
    print(f"问题 {i}: {question}")
    response = generate_response(question)
    print(f"回答: {response}\n")
    print("-" * 80)


# 与原始模型对比（可选）
def compare_with_original(question):
    """与原始模型对比"""
    print(f"问题: {question}")

    # 微调模型的回答
    fine_tuned_response = generate_response(question)
    print(f"微调模型回答: {fine_tuned_response}\n")

    # 原始模型的回答（需要单独加载原始模型）
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    original_model.eval()

    # 构建对话格式
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    inputs = tokenizer(text, return_tensors="pt").to(original_model.device)

    # 生成回答
    with torch.no_grad():
        outputs = original_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    # 解码输出
    original_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取助手的回答
    if "<|im_start|>assistant" in original_response:
        original_response = original_response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in original_response:
            original_response = original_response.split("<|im_end|>")[0]

    print(f"原始模型回答: {original_response.strip()}\n")
    print("=" * 80)


# 对比测试（可选）
print("\n\n对比测试：微调模型 vs 原始模型")
compare_questions = [    "医生说是前列腺发炎和供血不足导致的阳痿。请问勃起不坚是怎么样的病症？",
                        "男44岁最近比较疲惫，身体有感觉到了不健康，会是得这个病了吗，请问：做爱三分钟是不是属于早泄？"]
for q in compare_questions:
    compare_with_original(q)


# 交互式测试
def interactive_test():
    """交互式测试"""
    print("\n进入交互式测试模式（输入'退出'结束测试）")

    while True:
        user_input = input("\n请输入您的问题: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            break

        response = generate_response(user_input)
        print(f"\n助手回答: {response}")


# 启动交互式测试
interactive_test()
