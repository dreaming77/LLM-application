# **BERT模型微调与量化_Twitter_complaints**

技术栈：PEFT、LoRA、模型量化

基座模型：distilbert-base-uncased模型（将模型下载到本地）

数据集：Twitter_complaints [地址](https://huggingface.co/datasets/ought/raft) 将data文件夹里的csv文件下载到数据集中。

![image-20250806205729297](img src="ai.jpeg" style="zoom:33%;" )

$$
平平无奇
$$

## 微调目的

	微调BERT模型，对用户评论做情感分析分类。

## 内容概述

步骤概述：

1. 数据准备：加载并预处理数据并将标签列进行数值映射。
2. 加载模型和对应的tokenizer，使用tokenizer对数据集逐行分词。
3. 配置PEFT/LoRA：使用PEFT库配置LoRA，将其应用到模型上。
4. 训练设置：设置训练参数，使用GPU进行训练。
5. 训练模型：使用训练数据微调模型。
6. 评估模型：在测试集上进行预测并生成提交文件（由于测试集没有标签，我们只需要生成预测结果）。
7. 保存和加载模型：保存微调后的模型，以便后续使用。

