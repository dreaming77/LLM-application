#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/8/17
Description:If using codes, please indicate the source. 
"""
from transformers import pipeline
from data_preprocessing import test_ds
import panda as pd
from transformers import AutoTokenizer

# 加载模型和tokenizer
model_name = "../distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建预测管道
classifier = pipeline(
    "text-classification",
    model="best_model",
    tokenizer=tokenizer,
    device=0  # 使用GPU
)

# 生成预测
predictions = []
for example in test_ds:
    pred = classifier(example["Tweet text"])[0]
    predictions.append({
        "ID": example["ID"],
        "Label": 1 if pred["label"] == "LABEL_1" else 0,
        "Confidence": pred["score"]
    })

# 保存结果
pd.DataFrame(predictions).to_csv("submission.csv", index=False)
