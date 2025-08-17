#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/8/17
Description:If using codes, please indicate the source. 
"""

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model
from data_preprocessing import train_ds, val_ds
from peft import LoraConfig, TaskType
from sklearn.metrics import f1_score


# 加载模型和tokenizer
model_name = "../distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    return_dict=True  # 确保返回字典格式
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_lin", "v_lin"],
    modules_to_save=["classifier"]  # 关键修复: 确保分类层可训练
)

# 应用PEFT
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 显示可训练参数占比

# 数据预处理
def tokenize_fn(examples):
    return tokenizer(
        examples["Tweet text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

# 使用正确的数据收集器
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="max_length",  # 确保固定长度填充
    max_length=256,
    return_tensors="pt"
)

# 训练参数
training_args = TrainingArguments(
    output_dir="../results",
    eval_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=8,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    gradient_accumulation_steps=4,
    metric_for_best_model="f1",
    report_to="none",
    logging_steps=50
)

# 自定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": (predictions == labels).mean(),
        "f1": f1_score(labels, predictions)
    }

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=data_collator,  # 使用修复后的数据收集器
    tokenizer=tokenizer
)

# 开始训练
trainer.train()

# 保存最佳模型
trainer.save_model("best_model")
tokenizer.save_pretrained("best_model")

