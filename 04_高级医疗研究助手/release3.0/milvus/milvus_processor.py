import json
import logging
from typing import List, Dict, Any
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from tqdm import tqdm
import os
import sys
from config.settings import MILVUS_COLLECTION_NAME

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_manager import model_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MilvusDataProcessor:
    def __init__(self, milvus_host: str = "localhost", milvus_port: str = "19530", batch_size: int = 100):
        """
        初始化Milvus数据处理器
        
        Args:
            milvus_host: Milvus服务地址
            milvus_port: Milvus服务端口
            batch_size: 批量处理大小
        """
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.batch_size = batch_size
        self.collection_name = MILVUS_COLLECTION_NAME
        
        # 连接Milvus
        self._connect_milvus()
        
        # 获取embedding模型
        self.embedding_model = model_manager.embedding_model
        logger.info("Embedding模型已加载")
    
    def _connect_milvus(self):
        """连接Milvus数据库"""
        try:
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            logger.info(f"成功连接到Milvus: {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本的embedding向量
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding向量列表
        """
        try:
            # 使用模型管理器中的embedding模型
            embeddings = self.embedding_model.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"生成embedding失败: {e}")
            # 如果批量处理失败，尝试单个处理
            logger.info("尝试单个文本处理...")
            single_embeddings = []
            for text in texts:
                try:
                    embedding = self.embedding_model.embed_query(text)
                    single_embeddings.append(embedding)
                except Exception as ex:
                    logger.error(f"单个文本embedding生成失败: {ex}, 文本: {text[:100]}...")
                    # 返回零向量作为兜底
                    single_embeddings.append([0.0] * 1024)
            return single_embeddings
    
    def create_collection(self, drop_existing: bool = True):
        """创建Milvus集合"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="related_diseases", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="score", dtype=DataType.INT8),
            FieldSchema(name="question_vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]
        
        # 创建schema
        schema = CollectionSchema(fields, description="CHuatuo-26M医疗问答数据集")
        
        # 检查并删除已存在的集合
        if drop_existing and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"已存在的集合 {self.collection_name} 已被删除")
        
        # 创建集合
        self.collection = Collection(self.collection_name, schema)
        logger.info(f"集合 {self.collection_name} 创建成功")
        
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024}
        }
        
        self.collection.create_index("question_vector", index_params)
        logger.info("向量索引创建成功")
    
    def process_jsonl_file(self, file_path: str, max_records: int = None):
        """
        处理jsonl文件并导入Milvus
        
        Args:
            file_path: jsonl文件路径
            max_records: 最大处理记录数（用于测试）
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return
        
        # 读取并处理数据
        data_batch = []
        total_count = 0
        processed_count = 0
        
        # 首先统计总行数用于进度条
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        if max_records:
            total_lines = min(total_lines, max_records)
        
        logger.info(f"开始处理文件: {file_path}, 总行数: {total_lines}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            with tqdm(total=total_lines, desc="处理进度") as pbar:
                for line in file:
                    if max_records and processed_count >= max_records:
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        
                        # 数据清洗和验证
                        processed_data = self._process_single_record(data)
                        if processed_data:
                            data_batch.append(processed_data)
                            processed_count += 1
                        
                        # 批量处理
                        if len(data_batch) >= self.batch_size:
                            success_count = self._insert_batch(data_batch)
                            total_count += success_count
                            data_batch = []
                        
                        pbar.update(1)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON解析错误: {e}, 行内容: {line[:200]}...")
                    except Exception as e:
                        logger.error(f"处理数据时发生错误: {e}")
                        logger.error(f"错误行内容: {line[:200]}...")
                
                # 处理最后一批数据
                if data_batch:
                    success_count = self._insert_batch(data_batch)
                    total_count += success_count
        
        logger.info(f"数据处理完成，成功导入 {total_count} 条记录")
        
        # 加载集合到内存
        try:
            self.collection.load()
            logger.info("集合已加载到内存")
        except Exception as e:
            logger.error(f"集合加载失败: {e}")
    
    def _process_single_record(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单条记录
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的数据
        """
        # 验证必要字段
        required_fields = ['id', 'question', 'answer', 'score', 'label']
        for field in required_fields:
            if field not in data:
                logger.warning(f"记录缺少必要字段 {field}: {data.get('id', 'unknown')}")
                return None
        
        try:
            # 处理缺失值
            processed_data = {
                'id': int(data['id']),
                'question': str(data['question'])[:1990],  # 限制长度
                'answer': str(data['answer'])[:9990],      # 限制长度
                'label': str(data['label'])[:190],         # 限制长度
                'related_diseases': str(data.get('related_diseases', ''))[:490],  # 限制长度
                'score': min(max(int(data['score']), 0), 5)  # 确保score在0-5范围内
            }
            
            return processed_data
        except (ValueError, TypeError) as e:
            logger.warning(f"数据格式错误: {e}, 记录ID: {data.get('id', 'unknown')}")
            return None
    
    def _insert_batch(self, data_batch: List[Dict[str, Any]]) -> int:
        """
        批量插入数据到Milvus
        
        Args:
            data_batch: 批量数据
            
        Returns:
            成功插入的记录数
        """
        if not data_batch:
            return 0
            
        try:
            # 提取问题文本用于生成embedding
            questions = [item['question'] for item in data_batch]
            
            # 生成embedding
            question_vectors = self._get_embedding(questions)
            
            # 准备插入数据
            insert_data = [
                [item['id'] for item in data_batch],  # id
                [item['question'] for item in data_batch],  # question
                [item['answer'] for item in data_batch],  # answer
                [item['label'] for item in data_batch],  # label
                [item['related_diseases'] for item in data_batch],  # related_diseases
                [item['score'] for item in data_batch],  # score
                question_vectors  # question_vector
            ]
            
            # 插入数据
            insert_result = self.collection.insert(insert_data)
            logger.info(f"成功插入 {len(data_batch)} 条记录")
            
            return len(data_batch)
            
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            # 尝试单条插入
            success_count = 0
            for i, item in enumerate(data_batch):
                try:
                    question_vector = self._get_embedding([item['question']])[0]
                    insert_data = [
                        [item['id']],
                        [item['question']],
                        [item['answer']],
                        [item['label']],
                        [item['related_diseases']],
                        [item['score']],
                        [question_vector]
                    ]
                    self.collection.insert(insert_data)
                    success_count += 1
                except Exception as single_error:
                    logger.error(f"单条插入失败 (ID: {item['id']}): {single_error}")
            
            logger.info(f"单条插入完成，成功 {success_count}/{len(data_batch)} 条记录")
            return success_count
    
    def search_similar_questions(self, query: str, top_k: int = 5):
        """
        搜索相似问题
        
        Args:
            query: 查询文本
            top_k: 返回最相似的前k个结果
        """
        try:
            # 生成查询向量
            query_vector = self._get_embedding([query])[0]
            
            # 搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            results = self.collection.search(
                [query_vector],
                "question_vector",
                search_params,
                limit=top_k,
                output_fields=["question", "answer", "label", "score", "related_diseases"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return None
    
    def get_collection_info(self):
        """获取集合信息"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return self.collection.num_entities
        return 0
    
    def close(self):
        """关闭连接"""
        connections.disconnect("default")
        logger.info("Milvus连接已关闭")


def main():
    """主函数"""
    from config.settings import MILVUS_HOST, MILVUS_PORT, BATCH_SIZE, MAX_RECORDS, CHUATUO_DATA_PATH
    
    processor = None
    try:
        # 初始化处理器
        processor = MilvusDataProcessor(
            milvus_host=MILVUS_HOST,
            milvus_port=MILVUS_PORT,
            batch_size=BATCH_SIZE
        )
        
        # 创建集合
        processor.create_collection(drop_existing=True)
        
        # 处理数据文件
        processor.process_jsonl_file(CHUATUO_DATA_PATH, max_records=MAX_RECORDS)
        
        # 显示集合信息
        entity_count = processor.get_collection_info()
        logger.info(f"集合中的实体数量: {entity_count}")
        
        # 测试搜索功能
        if entity_count > 0:
            test_query = "脸上长痘痘怎么治疗？"
            results = processor.search_similar_questions(test_query)
            
            if results:
                logger.info(f"查询: '{test_query}' 的搜索结果:")
                for i, hit in enumerate(results[0]):
                    logger.info(f"{i+1}. 问题: {hit.entity.get('question')}")
                    logger.info(f"   答案: {hit.entity.get('answer')[:100]}...")
                    logger.info(f"   标签: {hit.entity.get('label')}, 评分: {hit.entity.get('score')}")
                    logger.info(f"   相关疾病: {hit.entity.get('related_diseases')}")
                    logger.info(f"   距离: {hit.distance:.4f}")
                    logger.info("   " + "-"*50)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
    finally:
        if processor:
            processor.close()

if __name__ == "__main__":
    main()


""""
执行完成后，如果你是安装milvus standalone服务，
那么可以访问http://xxx:8000/ 查看milvus的web界面，查看数据是否插入成功。
"""
