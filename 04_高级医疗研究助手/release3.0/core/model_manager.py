#!/usr/bin/env python3
"""
统一模型管理器 - 负责加载和管理所有AI模型
"""

import logging
import gc
import os
import torch
from typing import Optional, Dict, Any, List
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from config.settings import (
    EMBEDDING_MODEL_PATH,
    GENERATION_MODEL_PATH, 
    EMBEDDING_MODEL_DEVICE,
    GENERATION_MODEL_DEVICE
)

logger = logging.getLogger(__name__)

class ModelManager:
    """
    统一模型管理器
    负责加载、管理和提供生成模型与嵌入模型
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.embedding_model = None
        self.generation_model = None
        self.generation_tokenizer = None
        self.generation_pipeline = None
        self.embedding_device = EMBEDDING_MODEL_DEVICE
        self.generation_device = GENERATION_MODEL_DEVICE
        
        # 模型配置
        self.model_config = {
            "embedding": {
                "model_name": EMBEDDING_MODEL_PATH,
                "device": self.embedding_device,
                "model_kwargs": {'device': self.embedding_device}
            },
            "generation": {
                "model_name": GENERATION_MODEL_PATH,
                "device": self.generation_device,
                "torch_dtype": torch.float16,
                "max_new_tokens": 1000,
                "temperature": 0.3,
                "do_sample": True
            }
        }
        
        self._initialized = True
        logger.info("模型管理器初始化完成")
    
    def load_embedding_model(self) -> bool:
        """
        加载嵌入模型
        
        Returns:
            加载是否成功
        """
        try:
            logger.info(f"正在加载嵌入模型: {self.model_config['embedding']['model_name']}")
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_config['embedding']['model_name'],
                model_kwargs=self.model_config['embedding']['model_kwargs'],
                encode_kwargs={'normalize_embeddings': True}  # 归一化嵌入向量
            )
            
            # 测试模型
            test_embedding = self.embedding_model.embed_query("测试文本")
            if len(test_embedding) != 1024:
                logger.warning(f"嵌入向量维度异常: {len(test_embedding)}，期望 1024")
            
            logger.info("嵌入模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            self.embedding_model = None
            return False

    def load_generation_model(self) -> bool:
        """
        加载生成模型
        """
        try:
            logger.info(f"正在加载生成模型: {self.model_config['generation']['model_name']}")

            # 检查模型路径是否存在
            model_path = self.model_config['generation']['model_name']
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False

            # 根据设备类型选择不同的加载策略
            if self.generation_device.startswith("cuda") and torch.cuda.is_available():
                # GPU加载 - 使用量化减少显存占用
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

                self.generation_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch.float16,
                    device_map=GENERATION_MODEL_DEVICE,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                # CPU加载
                logger.warning("使用CPU加载生成模型，性能会受影响")
                self.generation_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

            # 加载tokenizer
            self.generation_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            if self.generation_tokenizer.pad_token is None:
                self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token

            # 创建pipeline
            self.generation_pipeline = pipeline(
                "text-generation",
                model=self.generation_model,
                tokenizer=self.generation_tokenizer,
                max_new_tokens=self.model_config['generation']['max_new_tokens'],
                temperature=self.model_config['generation']['temperature'],
                do_sample=self.model_config['generation']['do_sample'],
                pad_token_id=self.generation_tokenizer.eos_token_id,
                return_full_text=False
            )

            # 测试模型
            test_response = self.generate_response("你好，请回复'OK'确认模型工作正常。")
            if test_response and 'OK' in test_response:
                logger.info("生成模型加载成功")
                return True
            else:
                logger.error("生成模型测试失败")
                return False

        except Exception as e:
            logger.error(f"生成模型加载失败: {e}", exc_info=True)
            self.generation_model = None
            self.generation_tokenizer = None
            self.generation_pipeline = None
            return False

    def initialize_models(self) -> bool:
        """
        初始化所有模型

        Returns:
            初始化是否成功
        """
        logger.info("开始初始化所有模型...")

        success_count = 0

        # 加载嵌入模型
        if self.load_embedding_model():
            success_count += 1
        else:
            logger.error("嵌入模型初始化失败")

        # 加载生成模型
        if self.load_generation_model():
            success_count += 1
        else:
            logger.error("生成模型初始化失败")

        # 检查初始化结果
        if success_count == 2:
            logger.info("所有模型初始化成功")
            return True
        else:
            logger.error(f"模型初始化部分失败: {success_count}/2")
            return False

    def get_embedding_model(self) -> Optional[HuggingFaceEmbeddings]:
        """
        获取嵌入模型

        Returns:
            嵌入模型实例，如果未加载则返回None
        """
        if self.embedding_model is None:
            logger.warning("嵌入模型未加载，尝试重新加载...")
            if not self.load_embedding_model():
                return None
        return self.embedding_model

    def get_generation_model(self) -> Optional[Any]:
        """
        获取生成模型

        Returns:
            生成模型实例，如果未加载则返回None
        """
        if self.generation_model is None:
            logger.warning("生成模型未加载，尝试重新加载...")
            if not self.load_generation_model():
                return None
        return self.generation_model

    def get_generation_tokenizer(self) -> Optional[Any]:
        """
        获取生成模型的分词器

        Returns:
            分词器实例，如果未加载则返回None
        """
        if self.generation_tokenizer is None:
            logger.warning("生成模型分词器未加载")
            return None
        return self.generation_tokenizer

    def get_generation_pipeline(self) -> Optional[Any]:
        """
        获取生成pipeline

        Returns:
            生成pipeline实例，如果未加载则返回None
        """
        if self.generation_pipeline is None:
            logger.warning("生成pipeline未加载")
            return None
        return self.generation_pipeline

    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        生成文本的嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量，如果失败则返回None
        """
        try:
            embedding_model = self.get_embedding_model()
            if embedding_model is None:
                return None

            return embedding_model.embed_query(text)

        except Exception as e:
            logger.error(f"文本嵌入失败: {e}")
            return None

    def embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        批量生成文本的嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表，如果失败则返回None
        """
        try:
            embedding_model = self.get_embedding_model()
            if embedding_model is None:
                return None

            return embedding_model.embed_documents(texts)

        except Exception as e:
            logger.error(f"批量文本嵌入失败: {e}")
            return None

    def generate_response(self,
                          prompt: str,
                          max_tokens: int = None,
                          temperature: float = None,
                          **kwargs) -> Optional[str]:
        """
        生成文本响应

        Args:
            prompt: 输入提示
            max_tokens: 最大token数
            temperature: 生成温度
            **kwargs: 其他生成参数

        Returns:
            生成的文本，如果失败则返回None
        """
        try:
            pipeline = self.get_generation_pipeline()
            if pipeline is None:
                return None

            # 设置生成参数
            generation_kwargs = {
                "max_new_tokens": max_tokens or self.model_config['generation']['max_new_tokens'],
                "temperature": temperature or self.model_config['generation']['temperature'],
                "do_sample": self.model_config['generation']['do_sample'],
            }
            generation_kwargs.update(kwargs)

            # 生成响应
            response = pipeline(
                prompt,
                **generation_kwargs
            )

            # 提取生成的文本
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0]['generated_text']
                return generated_text.strip()
            else:
                logger.error("生成响应格式异常")
                return None

        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return None

    def chat(self,
             messages: List[Dict[str, str]],
             max_tokens: int = None,
             temperature: float = None,
             **kwargs) -> Optional[str]:
        """
        聊天式生成（兼容旧接口）

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            max_tokens: 最大token数
            temperature: 生成温度
            **kwargs: 其他生成参数

        Returns:
            生成的回复，如果失败则返回None
        """
        try:
            # 构建对话提示
            prompt = self._build_chat_prompt(messages)
            if not prompt:
                return None

            return self.generate_response(prompt, max_tokens, temperature, **kwargs)

        except Exception as e:
            logger.error(f"聊天生成失败: {e}")
            return None

    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        构建聊天提示

        Args:
            messages: 消息列表

        Returns:
            格式化后的提示文本
        """
        try:
            # 根据Qwen模型的要求构建提示
            prompt_parts = []

            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")

                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
                else:
                    prompt_parts.append(content)

            # 添加助手前缀以引导生成
            prompt_parts.append("Assistant: ")

            return "\n".join(prompt_parts)

        except Exception as e:
            logger.error(f"构建聊天提示失败: {e}")
            return None
    
    def update_generation_config(self, **kwargs):
        """
        更新生成配置
        
        Args:
            **kwargs: 配置参数
        """
        valid_keys = ["max_new_tokens", "temperature", "do_sample"]
        for key, value in kwargs.items():
            if key in valid_keys and key in self.model_config['generation']:
                self.model_config['generation'][key] = value
                logger.info(f"更新生成配置: {key} = {value}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            "embedding_model": {
                "loaded": self.embedding_model is not None,
                "model_name": self.model_config['embedding']['model_name'],
                "device": self.embedding_device
            },
            "generation_model": {
                "loaded": self.generation_model is not None,
                "model_name": self.model_config['generation']['model_name'],
                "device": self.generation_device,
                "config": {
                    "max_new_tokens": self.model_config['generation']['max_new_tokens'],
                    "temperature": self.model_config['generation']['temperature'],
                    "do_sample": self.model_config['generation']['do_sample']
                }
            }
        }
        
        # 添加GPU内存信息
        if torch.cuda.is_available():
            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                gpu_info[f"cuda:{i}"] = {
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                    "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3,    # GB
                    "memory_free": torch.cuda.get_device_properties(i).total_memory / 1024**3 - 
                                 (torch.cuda.memory_allocated(i) / 1024**3)        # GB
                }
            info["gpu_info"] = gpu_info
        
        return info
    
    def clear_cache(self):
        """清理GPU缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")
    
    def unload_models(self):
        """卸载所有模型"""
        logger.info("卸载所有模型...")
        
        # 卸载生成模型
        if self.generation_model is not None:
            del self.generation_model
            self.generation_model = None
        
        if self.generation_tokenizer is not None:
            del self.generation_tokenizer
            self.generation_tokenizer = None
        
        if self.generation_pipeline is not None:
            del self.generation_pipeline
            self.generation_pipeline = None
        
        # 卸载嵌入模型
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
        
        # 清理缓存
        self.clear_cache()
        
        logger.info("所有模型已卸载")


# 创建全局模型管理器实例
model_manager = ModelManager()

