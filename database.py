# database.py
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from config import VECTOR_DB_CONFIG, INDEX_DIR

class VectorDatabase:
    """向量数据库管理类"""
    
    def __init__(self, dim=None, nlist=None, index_type=None):
        """
        初始化向量数据库
        
        参数:
        - dim: 向量维度
        - nlist: IVF聚类中心数
        - index_type: 索引类型
        """
        self.dim = dim or VECTOR_DB_CONFIG["dimension"]
        self.nlist = nlist or VECTOR_DB_CONFIG["nlist"]
        self.index_type = index_type or VECTOR_DB_CONFIG["index_type"]
        
        # 初始化索引
        self.index = None
        self.quantizer = None
        
        # 存储数据
        self.documents = []
        self.metadata = []
        self.document_ids = []
        self.next_id = 0
        
        # 嵌入向量存储（用于重新训练）
        self.embeddings_cache = []
        
        print(f"向量数据库初始化: dim={self.dim}, nlist={self.nlist}, type={self.index_type}")
    
    def create_index(self):
        """创建索引"""
        if self.index_type == "flat":
            # 平面索引（精确搜索）
            self.index = faiss.IndexFlatL2(self.dim)
            
        elif self.index_type == "ivfflat":
            # IVF平面索引
            self.quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(self.quantizer, self.dim, self.nlist)
            
        elif self.index_type == "ivfpq":
            # IVF产品量化索引
            self.quantizer = faiss.IndexFlatL2(self.dim)
            m = 8  # 子空间数
            bits = 8  # 比特数
            self.index = faiss.IndexIVFPQ(self.quantizer, self.dim, self.nlist, m, bits)
            
        elif self.index_type == "hnsw":
            # HNSW图索引
            M = 16  # 每个节点的连接数
            self.index = faiss.IndexHNSWFlat(self.dim, M)
            
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        print(f"创建 {self.index_type} 索引成功")
    
    def train(self, embeddings):
        """训练索引（如果需要）"""
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print(f"训练索引，使用 {len(embeddings)} 个样本...")
            
            # 检查训练数据是否足够
            if len(embeddings) < self.nlist * 39:
                print(f"警告: 训练数据不足，建议至少 {self.nlist * 39} 个样本")
            
            self.index.train(embeddings)
            print("索引训练完成")
    
    def add_documents(self, documents: List[str], metadata_list: List[Dict] = None):
        """
        添加文档到数据库
        
        注意: 这个方法需要接收预处理的嵌入向量
        实际使用中应该先调用 preprocess_data 生成嵌入向量
        """
        if not documents:
            return
        
        # 为每个文档分配ID
        start_id = self.next_id
        new_ids = list(range(start_id, start_id + len(documents)))
        
        # 存储文档
        self.documents.extend(documents)
        self.document_ids.extend(new_ids)
        
        # 存储元数据
        if metadata_list:
            for i, meta in enumerate(metadata_list):
                if i < len(documents):
                    meta["id"] = new_ids[i]
                    self.metadata.append(meta)
        else:
            for doc_id in new_ids:
                self.metadata.append({"id": doc_id})
        
        self.next_id += len(documents)
        
        print(f"添加 {len(documents)} 个文档，当前总数: {len(self.documents)}")
    
    def add_embeddings(self, embeddings: np.ndarray, documents: List[str], metadata_list: List[Dict] = None):
        """
        添加嵌入向量和对应的文档
        """
        # 确保索引存在
        if self.index is None:
            self.create_index()
        
        # 训练索引（如果需要）
        if hasattr(self.index, 'is_trained'):
            # 检查是否已经训练过
            if not self.index.is_trained:
                self.train(embeddings)
        
        # 添加嵌入向量到索引
        self.index.add(embeddings)
        self.embeddings_cache.extend(embeddings)
        
        # 添加文档信息
        self.add_documents(documents, metadata_list)
        
        print(f"向量数据库当前包含 {self.index.ntotal} 个向量")
    
    def search(self, query_vector: np.ndarray, k: int = 3, threshold: float = None):
        """
        搜索最相似的向量
        
        参数:
        - query_vector: 查询向量
        - k: 返回的最近邻数量
        - threshold: 相似度阈值
        
        返回:
        - distances: 距离数组
        - indices: 索引数组
        - documents: 文档列表
        """
        if self.index is None or self.index.ntotal == 0:
            return [], [], []
        
        # 设置nprobe（如果是IVF索引）
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = VECTOR_DB_CONFIG["nprobe"]
        
        # 搜索
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        # 过滤无效结果
        valid_indices = []
        valid_distances = []
        valid_documents = []
        
        for i in range(k):
            idx = indices[0][i]
            if idx == -1:  # 无效索引
                continue
            
            # 应用阈值过滤
            if threshold is not None:
                # 计算相似度（距离越小越相似）
                similarity = 1.0 / (1.0 + distances[0][i])
                if similarity < threshold:
                    continue
            
            valid_indices.append(idx)
            valid_distances.append(distances[0][i])
            valid_documents.append(self.documents[idx])
        
        return valid_distances, valid_indices, valid_documents
    
    def get_context(self, query_vector: np.ndarray, k: int = 5, max_length: int = 2000):
        """
        检索相关上下文
        
        参数:
        - query_vector: 查询向量
        - k: 检索数量
        - max_length: 最大上下文长度
        
        返回:
        - context: 合并的上下文
        """
        distances, indices, documents = self.search(query_vector, k=k)
        
        if not documents:
            return ""
        
        # 构建上下文
        context_parts = []
        total_length = 0
        
        for i, (doc, dist) in enumerate(zip(documents, distances)):
            # 计算相似度
            similarity = 1.0 / (1.0 + dist)
            
            # 截断过长的文档
            if len(doc) > 500:
                doc = doc[:500] + "..."
            
            context_part = f"[相似度: {similarity:.3f}] {doc}"
            
            # 检查长度限制
            if total_length + len(context_part) > max_length:
                break
            
            context_parts.append(context_part)
            total_length += len(context_part)
        
        return "\n\n".join(context_parts)
    
    def save(self, path: str):
        """保存向量数据库"""
        # 保存索引
        if self.index is not None:
            faiss.write_index(self.index, f"{path}.faiss")
        
        # 保存数据
        data = {
            "documents": self.documents,
            "metadata": self.metadata,
            "document_ids": self.document_ids,
            "next_id": self.next_id,
            "dim": self.dim,
            "nlist": self.nlist,
            "index_type": self.index_type,
            "embeddings_cache": self.embeddings_cache if hasattr(self, 'embeddings_cache') else []
        }
        
        with open(f"{path}_data.pkl", "wb") as f:
            pickle.dump(data, f)
        
        print(f"向量数据库已保存到: {path}")
    
    def load(self, path: str):
        """加载向量数据库"""
        # 加载索引
        if os.path.exists(f"{path}.faiss"):
            self.index = faiss.read_index(f"{path}.faiss")
        else:
            print(f"警告: 索引文件不存在: {path}.faiss")
            self.index = None
        
        # 加载数据
        if os.path.exists(f"{path}_data.pkl"):
            with open(f"{path}_data.pkl", "rb") as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.document_ids = data["document_ids"]
            self.next_id = data["next_id"]
            self.dim = data["dim"]
            self.nlist = data["nlist"]
            self.index_type = data["index_type"]
            self.embeddings_cache = data.get("embeddings_cache", [])
            
            print(f"向量数据库已加载，包含 {len(self.documents)} 个文档")
        else:
            print(f"警告: 数据文件不存在: {path}_data.pkl")
    
    def get_stats(self):
        """获取统计信息"""
        return {
            "total_documents": len(self.documents),
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dim,
            "index_type": self.index_type,
            "nlist": self.nlist
        }