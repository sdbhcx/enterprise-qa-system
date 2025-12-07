# utils.py
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME, DEVICE

# 全局实例
_tokenizer = None
_embedding_model = None

def get_tokenizer():
    """获取分词器"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """获取嵌入模型"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

def preprocess_data(texts, mode="embedding", max_length=512):
    """
    预处理文本数据
    
    参数:
    - texts: 文本列表
    - mode: 模式，'embedding' 或 'tokenization'
    - max_length: 最大长度
    
    返回:
    - 嵌入向量或tokenized输入
    """
    if mode == "embedding":
        # 使用sentence transformer生成嵌入向量
        model = get_embedding_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.astype('float32')
    
    elif mode == "tokenization":
        # 使用GPT-2的分词器
        tokenizer = get_tokenizer()
        
        embeddings = []
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length" if len(texts) > 1 else False
            ).to(DEVICE)
            
            # 使用input_ids作为简单表示
            embedding = inputs["input_ids"].cpu().detach().numpy()
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    else:
        raise ValueError(f"不支持的预处理模式: {mode}")

def batch_preprocess(texts, batch_size=32, **kwargs):
    """批量预处理"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_result = preprocess_data(batch, **kwargs)
        results.append(batch_result)
    
    if len(results) == 0:
        return np.array([])
    
    return np.vstack(results) if kwargs.get('mode') == 'embedding' else np.concatenate(results, axis=0)

def calculate_similarity(query_vector, target_vectors):
    """计算相似度"""
    # 归一化向量
    query_norm = query_vector / np.linalg.norm(query_vector)
    target_norms = target_vectors / np.linalg.norm(target_vectors, axis=1, keepdims=True)
    
    # 计算余弦相似度
    similarities = np.dot(target_norms, query_norm.T)
    return similarities.flatten()

def validate_inputs(context, question, min_length=5):
    """验证输入"""
    errors = []
    
    if not context or not isinstance(context, str):
        errors.append("context必须是非空字符串")
    elif len(context.strip()) < min_length:
        errors.append(f"context长度必须至少{min_length}个字符")
    
    if not question or not isinstance(question, str):
        errors.append("question必须是非空字符串")
    elif len(question.strip()) < min_length:
        errors.append(f"question长度必须至少{min_length}个字符")
    
    return errors