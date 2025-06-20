##### 1. 数据集准备（例:民法典.txt）

[《中华人民共和国民法典 》全文免费下载 ](https://zhuanlan.zhihu.com/p/146891846)

2.虚拟环境准备（根据Llama-Factory官方文档自行配置）

3. 数据集处理

- 文本分割：text_segmentation.py文件

- 数据集生成：dataset.py

> [如何获取API-KEY_大模型服务平台百炼(Model Studio)-阿里云帮助中心 (aliyun.com)](https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key?spm=a2c4g.11186623.0.0.433444ceustKOk)
>
> 获取api 选择所需模型即可
>
> model=“qwen-max-latest”, # 修改自己选择的 模型

- 数据格式转换：text_transform.py

- 训练测试集划分：partitioning.py

3. 开始微调

一、配置文件修改
	LLaMA-Factory/data/dataset_info.json

​	LLaMA-Factory/data/dataset_info.json

同时alpaca_train_dataset.json和alpaca_test_dataset.json放入LLaMA-Factory/data/路径

```
  "train": {
    "file_name": "alpaca_train_dataset.json"
  },
  "test": {
    "file_name": "alpaca_test_dataset.json"
  }
```

二、进入llamafactory界面微调：llamafactory-cli webui

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9089ff8c53f7413a9df3bcc253421cf3.png#pic_center)

教学地址：[llama-factory实战: 基于qwen2.5-7b 手把手实战 自定义数据集清洗 微调_llamafactory qwen2.5](https://blog.csdn.net/yierbubu1212/article/details/142635578)

