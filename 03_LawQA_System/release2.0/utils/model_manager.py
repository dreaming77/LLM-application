import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL_PATH, GENERATION_MODEL_PATH, EMBEDDING_MODEL_DEVICE, GENERATION_MODEL_DEVICE
import gc


class ModelManager:
    _instance = None
    _embedding_model = None
    _generation_model = None
    _generation_tokenizer = None
    _generation_pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 清理GPU缓存
        self.clear_cache()

        # 加载嵌入模型
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={'device': EMBEDDING_MODEL_DEVICE}
        )

        # 加载生成模型和分词器
        self._generation_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_PATH)

        # 使用4位量化减少内存使用
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self._generation_model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=GENERATION_MODEL_DEVICE,
            quantization_config=quantization_config if GENERATION_MODEL_DEVICE.startswith("cuda") else None
        )

        # 创建生成管道
        self._generation_pipeline = pipeline(
            "text-generation",
            model=self._generation_model,
            tokenizer=self._generation_tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            device_map=GENERATION_MODEL_DEVICE
        )

        self._initialized = True

    @property
    def embedding_model(self):
        return self._embedding_model

    @property
    def generation_model(self):
        return self._generation_model

    @property
    def generation_tokenizer(self):
        return self._generation_tokenizer

    @property
    def generation_pipeline(self):
        return self._generation_pipeline

    def get_huggingface_pipeline(self):
        """获取HuggingFacePipeline实例"""
        return HuggingFacePipeline(pipeline=self._generation_pipeline)

    def clear_cache(self):
        """清理GPU缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 创建全局模型管理器实例
model_manager = ModelManager()
