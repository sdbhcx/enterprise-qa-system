# model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_NAME, DEVICE, GENERATION_CONFIG

# 全局模型和分词器实例
_model = None
_tokenizer = None

def init_model():
    """初始化模型和分词器"""
    global _model, _tokenizer
    
    if _model is None:
        print(f"加载模型: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        
        # 设置pad_token
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        _model.eval()
        print("模型加载完成")
    
    return _model, _tokenizer

def generate_answer(context, question, generation_config=None):
    """
    生成回答
    
    参数:
    - context: 上下文信息
    - question: 问题
    - generation_config: 生成参数，默认为全局配置
    
    返回:
    - answer: 生成的回答
    """
    # 初始化模型
    model, tokenizer = init_model()
    
    # 合并配置
    config = GENERATION_CONFIG.copy()
    if generation_config:
        config.update(generation_config)
    
    # 构建提示词
    prompt = f"""基于以下信息，回答问题：

信息：
{context}

问题：
{question}

回答："""
    
    # 编码输入
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024
    ).to(DEVICE)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=config["max_length"],
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
            num_return_sequences=config["num_return_sequences"],
            repetition_penalty=config.get("repetition_penalty", 1.0),
            do_sample=config["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码并提取回答
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取回答部分（去除提示词）
    if full_text.startswith(prompt):
        answer = full_text[len(prompt):].strip()
    else:
        answer = full_text.strip()
    
    return answer

def batch_generate(contexts_questions):
    """批量生成回答"""
    results = []
    for context, question in contexts_questions:
        answer = generate_answer(context, question)
        results.append({
            "context": context,
            "question": question,
            "answer": answer
        })
    return results