# config.py
import torch
import os

# Flask配置
FLASK_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "threaded": True,
}

# 模型配置
MODEL_NAME = "gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 生成参数
GENERATION_CONFIG = {
    "max_length": 200,
    "temperature": 0.6,
    "top_k": 40,
    "top_p": 0.85,
    "num_return_sequences": 1,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

# 向量数据库配置
VECTOR_DB_CONFIG = {
    "dimension": 768,
    "nlist": 100,
    "nprobe": 10,
    "index_type": "ivfflat",  # flat, ivfflat, ivfpq, hnsw
}

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")

# 创建必要的目录
for directory in [DATA_DIR, MODEL_DIR, INDEX_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"配置加载完成，使用设备: {DEVICE}")