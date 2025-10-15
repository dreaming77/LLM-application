import logging
import time
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections, 
    utility, 
    Collection, 
    DataType,
    FieldSchema, 
    CollectionSchema,
    MilvusException
)
from config.settings import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION_NAME

logger = logging.getLogger(__name__)

class MilvusClient:
    """
    Milvus客户端 - 负责与Milvus向量数据库的交互
    """
    
    _instance = None
    _collection = None
    _connected = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MilvusClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.host = MILVUS_HOST
        self.port = MILVUS_PORT
        self.collection_name = MILVUS_COLLECTION_NAME
        self.collection = None
        self._initialized = True
        
        # 延迟连接，在第一次使用时建立
        logger.info("Milvus客户端初始化完成（延迟连接）")
    
    def connect(self, retry_attempts: int = 3, retry_delay: int = 5) -> bool:
        """
        连接到Milvus数据库
        
        Args:
            retry_attempts: 重试次数
            retry_delay: 重试延迟（秒）
            
        Returns:
            连接是否成功
        """
        if self._connected:
            return True
            
        for attempt in range(retry_attempts):
            try:
                logger.info(f"尝试连接到Milvus: {self.host}:{self.port} (尝试 {attempt + 1}/{retry_attempts})")
                
                # 连接Milvus
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port
                )
                
                # 验证连接
                if utility.has_collection(self.collection_name):
                    self.collection = Collection(self.collection_name)
                    self.collection.load()
                    logger.info(f"成功连接到Milvus并加载集合: {self.collection_name}")
                else:
                    logger.warning(f"集合 {self.collection_name} 不存在，需要先创建")
                
                self._connected = True
                return True
                
            except MilvusException as e:
                logger.error(f"Milvus连接失败 (尝试 {attempt + 1}): {e}")
                if attempt < retry_attempts - 1:
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"经过 {retry_attempts} 次尝试后仍无法连接到Milvus")
                    return False
            except Exception as e:
                logger.error(f"连接过程中发生未知错误: {e}")
                return False
        
        return False
    
    def ensure_connected(self) -> bool:
        """确保已连接，如果未连接则尝试连接"""
        if not self._connected:
            return self.connect()
        return True
    
    def get_collection(self) -> Optional[Collection]:
        """
        获取集合实例
        
        Returns:
            Milvus集合实例，如果失败则返回None
        """
        if not self.ensure_connected():
            return None
            
        if self.collection is None and utility.has_collection(self.collection_name):
            try:
                self.collection = Collection(self.collection_name)
                self.collection.load()
                logger.info(f"集合 {self.collection_name} 已加载")
            except Exception as e:
                logger.error(f"加载集合失败: {e}")
                return None
        
        return self.collection
    
    def search_similar_questions(self, query_vectors: List[List[float]], 
                               top_k: int = 10, 
                               output_fields: Optional[List[str]] = None) -> List[List[Dict[str, Any]]]:
        """
        搜索相似问题
        
        Args:
            query_vectors: 查询向量列表
            top_k: 返回最相似的前k个结果
            output_fields: 需要返回的字段
            
        Returns:
            搜索结果列表
        """
        if not self.ensure_connected():
            return []
        
        collection = self.get_collection()
        if collection is None:
            logger.error("无法获取集合，搜索失败")
            return []
        
        # 默认返回字段
        if output_fields is None:
            output_fields = ["id", "question", "answer", "label", "score", "related_diseases"]
        
        try:
            # 搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nlist": 1024, "nprobe": 16}
            }
            
            # 执行搜索
            results = collection.search(
                query_vectors,
                "question_vector",
                search_params,
                limit=top_k,
                output_fields=output_fields
            )
            
            # 格式化结果
            formatted_results = []
            for i, result in enumerate(results):
                hits = []
                for hit in result:
                    hit_data = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "similarity_score": 1 - hit.distance,  # 转换为相似度分数
                        "entity_data": hit.entity._row_data if hasattr(hit.entity, '_row_data') else {}
                    }
                    
                    # 添加实体字段
                    for field in output_fields:
                        if hasattr(hit.entity, field):
                            hit_data[field] = getattr(hit.entity, field)
                    
                    hits.append(hit_data)
                
                formatted_results.append(hits)
            
            logger.info(f"成功搜索到 {sum(len(r) for r in formatted_results)} 个结果")
            return formatted_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息
        
        Returns:
            集合信息字典
        """
        if not self.ensure_connected():
            return {"error": "未连接到Milvus"}
        
        try:
            info = {}
            
            # 基本集合信息
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                info["entity_count"] = collection.num_entities
                info["collection_name"] = self.collection_name
                load_state = utility.load_state(self.collection_name)
                info["loaded"] = load_state == "Loaded"
                
                # 索引信息
                indexes = collection.indexes
                info["indexes"] = []
                for idx in indexes:
                    # 优先获取index_type，若不存在则获取type（兼容旧版本）
                    index_type = getattr(idx, "index_type", None)
                    if index_type is None:
                        index_type = getattr(idx, "type", "unknown")  # 旧版本属性名

                    info["indexes"].append({
                        "field_name": idx.field_name,
                        "index_type": index_type,  # 统一用index_type作为键
                        "params": idx.params  # params属性在各版本中通常稳定存在
                    })
                
                # 字段信息
                schema = collection.schema
                info["fields"] = [{
                    "name": field.name,
                    "dtype": str(field.dtype),
                    "is_primary": field.is_primary,
                    "auto_id": field.auto_id,
                    "max_length": getattr(field, 'max_length', None)
                } for field in schema.fields]
                
            else:
                info["error"] = f"集合 {self.collection_name} 不存在"
            
            return info
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        health_info = {
            "timestamp": time.time(),
            "milvus_connected": self._connected,
            "collection_available": False,
            "entity_count": 0,
            "status": "unknown"
        }
        
        try:
            if self.ensure_connected():
                collection_info = self.get_collection_info()
                
                if "error" not in collection_info:
                    health_info.update({
                        "collection_available": True,
                        "entity_count": collection_info.get("entity_count", 0),
                        "status": "healthy" if collection_info.get("entity_count", 0) > 0 else "empty"
                    })
                else:
                    health_info["status"] = "collection_missing"
            else:
                health_info["status"] = "connection_failed"
                
        except Exception as e:
            health_info["status"] = "error"
            health_info["error"] = str(e)
        
        return health_info
    
    def create_collection_if_not_exists(self, fields: List[FieldSchema], 
                                      index_params: Dict[str, Any] = None) -> bool:
        """
        如果集合不存在则创建
        
        Args:
            fields: 字段定义
            index_params: 索引参数
            
        Returns:
            创建是否成功
        """
        if not self.ensure_connected():
            return False
        
        if utility.has_collection(self.collection_name):
            logger.info(f"集合 {self.collection_name} 已存在")
            return True
        
        try:
            # 创建schema
            schema = CollectionSchema(
                fields, 
                description="CHuatuo-26M医疗问答数据集"
            )
            
            # 创建集合
            collection = Collection(self.collection_name, schema)
            
            # 创建索引
            if index_params is None:
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 1024}
                }
            
            collection.create_index("question_vector", index_params)
            
            logger.info(f"成功创建集合: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def insert_data(self, data: List[List[Any]]) -> bool:
        """
        插入数据到集合
        
        Args:
            data: 要插入的数据，格式与字段顺序对应
            
        Returns:
            插入是否成功
        """
        if not self.ensure_connected():
            return False
        
        collection = self.get_collection()
        if collection is None:
            logger.error("无法获取集合，插入失败")
            return False
        
        try:
            # 插入数据
            result = collection.insert(data)
            
            # 刷新使数据可搜索
            collection.flush()
            
            logger.info(f"成功插入 {len(result.primary_keys)} 条数据")
            return True
            
        except Exception as e:
            logger.error(f"数据插入失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        try:
            if self.collection:
                self.collection.release()
            
            connections.disconnect("default")
            self._connected = False
            self.collection = None
            
            logger.info("Milvus连接已关闭")
        except Exception as e:
            logger.error(f"关闭连接时发生错误: {e}")

# 创建全局Milvus客户端实例
milvus_client = MilvusClient()

# 便捷函数
def get_milvus_collection() -> Optional[Collection]:
    """获取Milvus集合实例（用于节点函数）"""
    return milvus_client.get_collection()

def get_milvus_client() -> MilvusClient:
    """获取Milvus客户端实例"""
    return milvus_client

def health_check() -> Dict[str, Any]:
    """健康检查便捷函数"""
    return milvus_client.health_check()
    