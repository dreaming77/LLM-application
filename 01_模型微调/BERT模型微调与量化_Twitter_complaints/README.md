# **BERT模型微调与量化_Twitter_complaints**

技术栈：PEFT、LoRA、模型量化

基座模型：distilbert-base-uncased模型（将模型下载到本地）

数据集：Twitter_complaints [地址](https://huggingface.co/datasets/ought/raft) 将data文件夹里的csv文件下载到数据集中。

>
>`distilbert-base-uncased` 基于 **DistilBERT** 架构，是 BERT 模型的 “蒸馏版”。
>
>“uncased” 表示模型在预训练时对文本进行了 “不区分大小写” 处理（所有输入文本会被转为小写）。
>
>**模型规模**：
>
>- 层数：6 层（仅为 BERT-base 的一半，BERT-base 为 12 层）。
>
>- 隐藏层维度：768（与 BERT-base 一致）。
>
>- 注意力头数：12（与 BERT-base 一致）。
>
>- 参数量：约 6600 万（BERT-base 为 1.1 亿，参数量减少约 40%）。
>

![image-20250806205729297](ai.jpeg)

$$
平平无奇
$$

## 微调目的

	微调BERT模型，对用户评论做情感分析分类。

## 内容概述

步骤概述：

1. 数据准备：加载并预处理数据并将标签列进行数值映射。数据集划分并转换成dataset格式。
2. 加载模型和对应的tokenizer，使用tokenizer对数据集逐行分词。
3. 配置PEFT/LoRA：使用PEFT库配置LoRA，将其应用到模型上。
4. 训练设置：设置训练参数，使用GPU进行训练。
5. 训练模型：使用训练数据微调模型。
6. 评估模型：在测试集上进行预测并生成提交文件（由于测试集没有标签，我们只需要生成预测结果）。
7. 保存和加载模型：保存微调后的模型，以便后续使用。
