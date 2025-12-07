# app.py
from flask import Flask, request, jsonify
import numpy as np
import traceback
import logging
from datetime import datetime

from model import generate_answer
from database import VectorDatabase
from utils import preprocess_data, validate_inputs
from config import FLASK_CONFIG, INDEX_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__)

# 初始化向量数据库
vector_db = VectorDatabase()
vector_db_path = f"{INDEX_DIR}/vector_db"

# 尝试加载现有数据库
try:
    vector_db.load(vector_db_path)
    logger.info(f"已加载向量数据库，包含 {len(vector_db.documents)} 个文档")
except Exception as e:
    logger.warning(f"加载向量数据库失败: {e}")
    logger.info("将创建新的向量数据库")

# 存储查询历史
query_history = []

@app.route('/')
def home():
    """首页"""
    return jsonify({
        "service": "企业问答系统API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - 查询答案",
            "/add_document": "POST - 添加文档",
            "/stats": "GET - 系统统计",
            "/history": "GET - 查询历史",
            "/health": "GET - 健康检查"
        },
        "status": "running"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vector_db": vector_db.get_stats(),
        "model": "gpt2-medium"
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """获取系统统计"""
    stats = vector_db.get_stats()
    
    # 添加查询历史统计
    history_stats = {
        "total_queries": len(query_history),
        "recent_queries": query_history[-10:] if query_history else [],
        "average_response_time": np.mean([q.get('response_time', 0) for q in query_history]) if query_history else 0
    }
    
    stats.update(history_stats)
    return jsonify(stats)

@app.route('/history', methods=['GET'])
def get_history():
    """获取查询历史"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        "total": len(query_history),
        "history": query_history[-limit:]
    })

@app.route('/add_document', methods=['POST'])
def add_document():
    """添加文档到向量数据库"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "请求体必须为JSON格式"}), 400
        
        documents = data.get("documents")
        if not documents:
            return jsonify({"error": "documents字段不能为空"}), 400
        
        # 预处理文档
        embeddings = preprocess_data(documents, mode="embedding")
        
        # 获取元数据（可选）
        metadata_list = data.get("metadata", [])
        
        # 添加到向量数据库
        vector_db.add_embeddings(embeddings, documents, metadata_list)
        
        # 保存数据库
        vector_db.save(vector_db_path)
        
        return jsonify({
            "message": f"成功添加 {len(documents)} 个文档",
            "total_documents": len(vector_db.documents),
            "total_vectors": vector_db.index.ntotal if vector_db.index else 0
        })
    
    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        return jsonify({"error": f"添加文档失败: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def query():
    """
    API端点，接收查询请求并返回回答
    
    请求体:
    {
        "context": "背景信息文本",
        "question": "问题文本",
        "k": 3,  # 可选，检索的文档数量
        "threshold": 0.5,  # 可选，相似度阈值
        "generation_config": {}  # 可选，生成参数
    }
    """
    start_time = datetime.now()
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "请求体必须为JSON格式"}), 400
        
        # 获取参数
        context = data.get("context")
        question = data.get("question")
        k = data.get("k", 3)
        threshold = data.get("threshold", 0.5)
        generation_config = data.get("generation_config", {})
        
        # 验证输入
        validation_errors = validate_inputs(context, question)
        if validation_errors:
            return jsonify({"error": "输入验证失败", "details": validation_errors}), 400
        
        # 1. 检索最相似的向量
        logger.info(f"检索相似向量: question='{question[:50]}...'")
        
        # 生成查询嵌入向量
        query_embedding = preprocess_data([question], mode="embedding")
        
        # 搜索向量数据库
        distances, indices, retrieved_docs = vector_db.search(query_embedding, k=k, threshold=threshold)
        
        logger.info(f"检索结果: 找到 {len(retrieved_docs)} 个相关文档")
        
        # 2. 构建上下文（如果提供了额外上下文，与检索结果合并）
        if context:
            # 使用提供的上下文作为主要信息
            combined_context = context
            if retrieved_docs:
                combined_context += "\n\n相关补充信息:\n" + "\n".join(retrieved_docs)
        else:
            # 仅使用检索结果
            combined_context = "\n".join(retrieved_docs) if retrieved_docs else "暂无相关信息"
        
        # 3. 调用生成模型
        logger.info("调用生成模型...")
        answer = generate_answer(combined_context, question, generation_config)
        
        # 4. 记录查询历史
        response_time = (datetime.now() - start_time).total_seconds()
        query_record = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "context_preview": context[:100] + "..." if context and len(context) > 100 else context,
            "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
            "retrieved_docs_count": len(retrieved_docs),
            "response_time": response_time,
            "retrieval_distances": distances if distances else []
        }
        
        # 保持历史记录不超过100条
        query_history.append(query_record)
        if len(query_history) > 100:
            query_history.pop(0)
        
        # 5. 返回结果
        return jsonify({
            "answer": answer,
            "retrieval_info": {
                "retrieved_count": len(retrieved_docs),
                "distances": distances,
                "indices": indices,
                "documents_preview": [doc[:100] + "..." if len(doc) > 100 else doc for doc in retrieved_docs]
            },
            "context_used": combined_context[:500] + "..." if len(combined_context) > 500 else combined_context,
            "response_time": response_time,
            "model": "gpt2-medium"
        })
    
    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            "error": "处理查询时发生错误",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/batch_query', methods=['POST'])
def batch_query():
    """批量查询"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "请求体必须为JSON格式"}), 400
        
        queries = data.get("queries")
        if not queries or not isinstance(queries, list):
            return jsonify({"error": "queries必须是列表"}), 400
        
        results = []
        for query_item in queries:
            context = query_item.get("context", "")
            question = query_item.get("question", "")
            
            if not question:
                results.append({"error": "问题不能为空"})
                continue
            
            # 处理单个查询
            try:
                response = query()
                # 这里需要模拟query()函数的处理逻辑
                # 实际实现中可能需要重构
                results.append({
                    "question": question,
                    "answer": "模拟回答（批量查询需要重构）"
                })
            except Exception as e:
                results.append({
                    "question": question,
                    "error": str(e)
                })
        
        return jsonify({
            "total": len(queries),
            "results": results
        })
    
    except Exception as e:
        logger.error(f"批量查询失败: {e}")
        return jsonify({"error": f"批量查询失败: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "端点不存在"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "请求方法不允许"}), 405

if __name__ == "__main__":
    logger.info(f"启动Flask应用，监听 {FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    
    # 初始化示例数据（如果数据库为空）
    if not vector_db.documents:
        logger.info("数据库为空，添加示例文档...")
        example_docs = [
            "公司政策规定，所有员工每年可以享受10天的带薪年假。",
            "员工在公司入职满一年后可以获得额外的年终奖金。",
            "公司支持员工每周三在家办公，支持远程工作。",
            "我们的医疗保险包括门诊和住院费用的报销。",
            "公司设有内部学习与培训计划，员工可以自由报名参加。"
        ]
        
        # 预处理并添加
        embeddings = preprocess_data(example_docs, mode="embedding")
        vector_db.add_embeddings(embeddings, example_docs)
        vector_db.save(vector_db_path)
    
    # 启动Flask应用
    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug'],
        threaded=FLASK_CONFIG['threaded']
    )