#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: DreamingWay
Date: 2025/8/17
Description:If using codes, please indicate the source. 
"""

import pandas as pd
from datasets import Dataset, ClassLabel
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Label'] = df['Label'].map({'complaint': 1, 'no complaint': 0})
    return df

TRAIN_PATH = '../dataset/twitter_complaints/data/train.csv'
TEST_PATH = '../dataset/twitter_complaints/data/test_unlabeled.csv'

# 处理训练数据
train_df = load_data(TRAIN_PATH)
train_data, val_data = train_test_split(train_df, test_size=0.15, random_state=42)

# 处理测试数据
test_df = pd.read_csv(TEST_PATH)
test_df['Label'] = -1  # 添加伪标签

# 转换为Dataset对象并添加特征类型
features = {
    'Tweet text': 'string',
    'Label': ClassLabel(num_classes=2, names=['no complaint', 'complaint']),
    'ID': 'int32'
}

# 转换为Dataset对象
train_ds = Dataset.from_pandas(train_data)
val_ds = Dataset.from_pandas(val_data)
test_ds = Dataset.from_pandas(test_df)


