# test_app.py
import requests
import json
import time
from typing import List, Dict

# API配置
API_URL = "http://127.0.0.1:5000"  # 本地测试地址
# API_URL = "http://your-server-ip:5000"  # 生产环境地址

def test_health():
    """测试健康检查端点"""
    print("测试健康检查...")
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        print(f"✓ 健康检查通过: {response.json()}")
        return True
    else:
        print(f"✗ 健康检查失败: {response.status_code}")
        return False

def test_stats():
    """测试统计端点"""
    print("\n测试统计信息...")
    response = requests.get(f"{API_URL}/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print(f"✓ 统计信息获取成功:")
        print(f"  文档总数: {stats.get('total_documents', 0)}")
        print(f"  向量总数: {stats.get('total_vectors', 0)}")
        print(f"  查询总数: {stats.get('total_queries', 0)}")
        return True
    else:
        print(f"✗ 统计信息获取失败: {response.status_code}")
        return False

def test_add_document():
    """测试添加文档"""
    print("\n测试添加文档...")
    
    documents = [
        "公司提供免费的午餐和晚餐，食堂位于办公楼3层。",
        "每年6月公司会组织团建活动，所有员工都可以参加。",
        "公司设有健身房和瑜伽室，员工可以免费使用。"
    ]
    
    payload = {
        "documents": documents,
        "metadata": [
            {"type": "福利政策", "category": "餐饮"},
            {"type": "团队活动", "category": "团建"},
            {"type": "福利政策", "category": "健康"}
        ]
    }
    
    response = requests.post(f"{API_URL}/add_document", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 添加文档成功: {result}")
        return True
    else:
        print(f"✗ 添加文档失败: {response.status_code} - {response.text}")
        return False

def test_query_api():
    """测试问答API"""
    print("\n测试问答API...")
    
    # 测试数据
    test_data = [
        {
            "context": "公司政策规定，所有员工每年可以享受10天的带薪年假。",
            "question": "公司的带薪年假政策是什么？",
            "expected_keywords": ["带薪年假", "10天"]
        },
        {
            "context": "员工在公司入职满一年后可以获得额外的年终奖金。",
            "question": "公司对年终奖金是如何规定的？",
            "expected_keywords": ["年终奖金", "满一年"]
        },
        {
            "context": "公司支持员工每周三在家办公。",
            "question": "公司是否允许远程工作？",
            "expected_keywords": ["在家办公", "远程工作"]
        },
        {
            "context": "我们的医疗保险包括门诊和住院费用的报销。",
            "question": "公司的医疗保险覆盖哪些方面？",
            "expected_keywords": ["医疗保险", "门诊", "住院", "报销"]
        },
        {
            "context": "公司设有内部学习与培训计划，员工可以自由报名。",
            "question": "公司提供哪些员工培训计划？",
            "expected_keywords": ["培训", "学习", "报名"]
        }
    ]
    
    all_passed = True
    
    for idx, item in enumerate(test_data):
        print(f"\n测试用例 {idx + 1}/{len(test_data)}")
        print(f"问题: {item['question']}")
        
        payload = {
            "context": item["context"],
            "question": item["question"],
            "k": 3,
            "threshold": 0.3
        }
        
        try:
            # 发送POST请求
            start_time = time.time()
            response = requests.post(f"{API_URL}/query", json=payload)
            response_time = time.time() - start_time
            
            # 检查响应状态码
            if response.status_code != 200:
                print(f"✗ API调用失败: 状态码 {response.status_code}")
                print(f"  响应: {response.text}")
                all_passed = False
                continue
            
            # 解析响应
            result = response.json()
            
            if "error" in result:
                print(f"✗ API返回错误: {result['error']}")
                all_passed = False
                continue
            
            # 获取生成的回答
            answer = result.get("answer", "")
            print(f"生成的回答: {answer}")
            print(f"响应时间: {response_time:.2f}秒")
            
            # 检查回答是否包含预期关键字
            missing_keywords = []
            for keyword in item["expected_keywords"]:
                if keyword not in answer:
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                print(f"✗ 缺少关键字: {missing_keywords}")
                all_passed = False
            else:
                print(f"✓ 测试通过")
            
            # 显示检索信息
            retrieval_info = result.get("retrieval_info", {})
            print(f"检索到 {retrieval_info.get('retrieved_count', 0)} 个相关文档")
            
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            all_passed = False
    
    return all_passed

def test_error_cases():
    """测试错误情况"""
    print("\n测试错误情况...")
    
    # 测试1: 空请求体
    print("测试空请求体...")
    response = requests.post(f"{API_URL}/query", json={})
    if response.status_code == 400:
        print(f"✓ 正确处理空请求体")
    else:
        print(f"✗ 空请求体测试失败: {response.status_code}")
    
    # 测试2: 缺少必要字段
    print("\n测试缺少字段...")
    response = requests.post(f"{API_URL}/query", json={"question": "test"})
    if response.status_code == 400:
        print(f"✓ 正确处理缺少字段")
    else:
        print(f"✗ 缺少字段测试失败: {response.status_code}")
    
    # 测试3: 无效的JSON
    print("\n测试无效JSON...")
    response = requests.post(f"{API_URL}/query", data="invalid json")
    if response.status_code == 400:
        print(f"✓ 正确处理无效JSON")
    else:
        print(f"✗ 无效JSON测试失败: {response.status_code}")
    
    return True

def test_batch_query():
    """测试批量查询"""
    print("\n测试批量查询...")
    
    batch_payload = {
        "queries": [
            {"context": "公司提供免费咖啡和茶", "question": "公司提供哪些饮料？"},
            {"context": "上班时间是9点到18点", "question": "工作时间是什么？"},
            {"question": "这个问题没有上下文"}
        ]
    }
    
    response = requests.post(f"{API_URL}/batch_query", json=batch_payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 批量查询成功，处理了 {result['total']} 个查询")
        for i, item in enumerate(result["results"]):
            print(f"  结果{i+1}: {item}")
        return True
    else:
        print(f"✗ 批量查询失败: {response.status_code} - {response.text}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始运行RAG系统API测试")
    print("=" * 60)
    
    tests = [
        ("健康检查", test_health),
        ("统计信息", test_stats),
        ("添加文档", test_add_document),
        ("问答API", test_query_api),
        ("错误情况", test_error_cases),
        ("批量查询", test_batch_query),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"测试: {test_name}")
        print('='*40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            time.sleep(1)  # 避免请求过快
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败，请检查问题")
    
    return passed == total

if __name__ == "__main__":
    # 等待API启动
    print("等待API启动...")
    time.sleep(3)
    
    # 运行测试
    success = run_all_tests()
    
    if success:
        exit(0)
    else:
        exit(1)