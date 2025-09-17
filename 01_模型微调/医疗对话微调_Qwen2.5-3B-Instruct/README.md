# **针对医学领域的模型微调**

技术栈：PEFT、LoRA

基座模型：Qwen-3B-Instruct（将模型下载到本地）

句子转换器模型：paraphrase-multilingual-MiniLM-L12-v2 

数据集：Chinese-medical-dialogue-data [地址](https://github.com/Toyhom/Chinese-medical-dialogue-data) 将数据集下载到dataset数据集中，解压。

![image-20250806205729297](dataset/ai.png)

$$
我不吃牛肉
$$

## 模型介绍

1）Qwen2.5-3B-Instruct 模型
	Transformer 解码器架构：采用标准 Transformer 解码器，集成 RoPE 旋转位置编码（支持长序列建模）、SwiGLU 激活函数（提升非线性表达能力）和 RMSNorm 归一化（增强训练稳定性）。
	分组查询注意力（GQA）：配置 16 个查询头（Q）和 2 个键值头（KV），在保持计算效率的同时提升上下文建模能力，尤其适合多轮对话场景。
	共享词嵌入与多语言支持：使用统一的字节对编码（BBPE）分词器，覆盖 29 种语言，支持跨语言指令理解和生成。
	
2）paraphrase-multilingual-MiniLM-L12-v2 句子转换器模型————它将句子和段落映射到 384 维密集向量空间，可用于聚类或语义搜索等任务。

​	属于 sentence-transformers 库（SBERT 开源生态），该生态的核心是通过 对比学习 微调 Transformer 模型，让输出的句子向量可直接用于余弦相似度计算。

| 名称片段       | 含义解释                                                     |
| -------------- | :----------------------------------------------------------- |
| `paraphrase`   | 核心预训练任务围绕「 paraphrase （复述句）」展开，侧重捕捉句子语义等价性 |
| `multilingual` | 支持多语言，覆盖 **100+ 种语言**（含中英、日韩、欧洲小语种、部分低资源语言） |
| `MiniLM`       | 基础模型架构基于微软的 **MiniLM**（BERT 的轻量级变体，通过知识蒸馏压缩参数） |
| `L12`          | 模型包含 **12 层 Transformer 编码器**（平衡语义捕捉能力与计算效率） |
| `v2`           | 第二个版本，相比 v1 优化了低资源语言的嵌入质量和跨语言一致性 |

## 内容概述

1. #### 数据集预处理`preprocession.py`

​	1）定义系统提示模版（如“你是{department}的医疗专家，请根据你的医学知识用中文回答用户的问题。”department占位符填写对应的科室）。

​	2）数据过滤，数据集中存在打来空白数据、不相干提问、“无或没”、提问乱码等问题。

解决方案：

- 基于嵌入的语义相似度检测，使用SentenceTransformer模型 “paraphrase-multilingual-MiniLM-L12-v2”。
- 基于聚类的异常检测。

​	3）将 CSV 文件中的问答数据（包含title、ask和answer字段）处理并转换为 Qwen2.5 模型微调所需的 JSONL 格式。

​	4）划分数据集，按照8：1：1划分训练集、测试集和验证集。

2. #### 模型微调`train_qwen.py`

​	1）加载分词器tokenizer和model，给tokenizer指定填充符。

​	2）配置LoRA。如低秩矩阵的维度、缩放因子以及微调的目标模块。

​	3）定义数据预处理函数，将数据messages格式化为模型输入。

​	4）配置数据收集器和训练参数。

3. #### **模型评估`eval_model.py`**

​	1）定义响应生成函数，包括设置模型温度参数`temperature=0.7`、重复惩罚悉数`repetition_penalty=1.1`。

​	2）加载基础模型与微调模型，根据BLEU分数、ROUGE分数和语义相似度指标比较。

​	3）中文分词函数（`chinese_tokenize`）
​	基于`jieba`库实现中文分词，将输入文本拆分为词语列表（如 “我爱中国”→`["我", "爱", "中国"]`），为空文本返回空列表。
*作用*：为后续 BLEU 等基于 n-gram 的指标提供分词后的 token，确保中文处理的准确性（中文无天然空格分隔）。

​	4）语义相似度计算（`calculate_semantic_similarity`）
使用预训练的 SBERT 模型（句子转换器）生成参考文本（真实回答）和候选文本（模型生成回答）的语义嵌入向量，通过余弦相似度计算两者的语义关联度（值越接近 1，语义越相似）。
*作用*：弥补传统 n-gram 指标（如 BLEU）仅关注字面重叠的缺陷，从语义层面评估生成质量。

​	5）ROUGE 分数计算（`calculate_rouge_scores`）
​	针对中文文本特点，将参考文本和候选文本转换为 “字符 + 空格” 格式（如 “我爱中国”→`"我 爱 中 国"`），适配 ROUGE 库的输入要求，计算 ROUGE-1（1-gram 重叠）、ROUGE-2（2-gram 重叠）、ROUGE-L（最长公共子序列）的精确率（p）、召回率（r）和 F1 分数（f）。
*作用*：评估生成文本与真实回答在局部短语和整体结构上的重叠度，常用于文本摘要、对话生成任务

| 分数    | Qwen2.5-3B-Instuct-finetuned |
| ------- | ---------------------------- |
| BLEU-4  | 3.21                         |
| Rouge-1 | 17.19                        |
| Rouge-2 | 3.07                         |
| Rouge-L | 15.47                        |
