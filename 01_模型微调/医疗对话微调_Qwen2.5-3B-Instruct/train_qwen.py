import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

# 模型配置
model_path = "./models/Qwen2.5-3B-Instruct"
output_dir = "./models/Qwen2.5-3B-instruct-finetuned"
logging_dir = "./logs"
dataset_path = "./dataset/processed_data/first_3000.jsonl"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
"""
为什么用 eos_token 作为 pad_token？
    很多预训练模型（如 Qwen、GPT 系列、Llama）默认没有定义 pad_token（训练时主要关注 “生成文本到结束符为止”，而非填充场景），需要手动指定。
eos_token（结束符，如 Qwen 中的 <|end_of_solution|> 或 <|end_of_text|>）是模型训练中天然存在的特殊符号，
"""

tokenizer.padding_side = "right"  # 确保填充在右侧
"""
填充方向对模型的影响：
    因果语言模型（如 Qwen2.5、GPT）的核心是 “从左到右生成文本”（下一个词只依赖左侧已生成的词），因此填充位置必须与模型的 “生成逻辑” 匹配：
若 padding_side = "right"（右侧填充）：短句子的填充符加在末尾（如 “我爱中国”→“我爱中国<eos><eos>”），模型处理时会先关注左侧的有效文本，再忽略右侧填充符，符合生成逻辑。
若 padding_side = "left"（左侧填充）：填充符加在句首（如 “我爱中国”→“<eos><eos>我爱中国”），模型会先看到填充符，可能干扰对有效文本开头的理解（尤其对长文本生成影响明显）。
"""


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

# 配置LoRA
lora_config = LoraConfig(
    r=8,  # 秩，定义低秩矩阵的维度（秩），决定了 LoRA 适配器的参数规模和表达能力。
    lora_alpha=16,  # alpha参数，控制低秩矩阵输出的缩放比例，影响 LoRA 更新对原始模型的 “干预强度”。
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # 目标模块，指定模型中哪些模块需要添加 LoRA 适配器
    lora_dropout=0.05,  # dropout率
    bias="none",  # 偏置
    task_type="CAUSAL_LM"  # 任务类型
)
"""指定在模型微调时哪些模块需要添加低秩矩阵适配器（Adapter）、适配器的结构参数及任务类型。
    q_proj/k_proj/v_proj：注意力机制中的 “查询（Query）”“键（Key）”“值（Value）” 投影层，
负责将输入特征转换为注意力计算所需的向量，是模型捕捉上下文关联的关键。
    o_proj：注意力输出的投影层，将多头注意力的结果整合为单向量，影响注意力信息的最终传递。
    gate_proj/up_proj/down_proj：前馈网络（FFN）中的核心层，gate_proj 是门控机制（控制特征流过的强度），
up_proj 是升维层（扩大特征维度），down_proj 是降维层（将特征映射回原维度），负责对注意力输出进行非线性变换，增强模型的特征表达能力。
"""

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数数量

# 加载预处理后的数据集
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 划分训练集和验证集
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(eval_dataset)}")


# 数据预处理函数
def preprocess_function(examples):
    # 将messages格式化为模型输入
    texts = []
    for message in examples["messages"]:
        # 将对话历史格式化为模型接受的格式
        formatted_text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(formatted_text)

    # 对文本进行tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=2048,  # 根据GPU内存调整
        padding=False,
        return_tensors=None
    )

    # 对于因果语言建模，标签与输入相同
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


# 预处理数据集
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training data"
)

tokenized_eval = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing evaluation data"
)

# 数据收集器 - 使用Seq2Seq的数据收集器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,  # 使用-100忽略损失计算中的填充部分
    pad_to_multiple_of=8  # 填充到8的倍数，提高GPU效率
)

# 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # 根据GPU内存调整
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # 梯度累积
    eval_steps=13000,
    save_steps=13000,
    logging_steps=150,
    learning_rate=1e-6,
    weight_decay=0.01,
    warmup_steps=1000,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),  # 使用FP16如果可用
    report_to="tensorboard",
    max_grad_norm=1.0,  # 添加梯度裁剪
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    group_by_length=True,  # 按长度分组，提高训练效率
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# 开始训练
print("开始训练...")
trainer.train()

# 保存最终模型
trainer.save_model()
print("训练完成，模型已保存")

# 保存LoRA适配器
model.save_pretrained("./models/qwen2.5-3b-instruct-lora-medical-adapter")
print("LoRA适配器已保存")
