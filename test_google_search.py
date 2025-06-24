#!/usr/bin/env python3
"""
测试脚本：验证新的 Google 搜索功能
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目路径到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from open_deep_research.utils import google_search_async, gemini_google_search_async
from open_deep_research.configuration import SearchAPI, MultiAgentConfiguration


async def test_google_search():
    """测试 Google Custom Search API 或网页抓取功能"""
    print("🔍 测试 Google Search 功能...")
    
    test_queries = ["Rust programming", "machine learning"]
    
    try:
        results = await google_search_async(
            search_queries=test_queries,
            max_results=3,
            include_raw_content=False
        )
        
        print(f"✅ Google Search 测试成功！")
        print(f"📊 返回结果数量: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"\n--- 查询 {i+1}: {result['query']} ---")
            print(f"结果数量: {len(result['results'])}")
            for j, item in enumerate(result['results'][:2]):  # 只显示前2个结果
                print(f"  {j+1}. {item['title']}")
                print(f"     URL: {item['url']}")
                print(f"     摘要: {item['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Google Search 测试失败: {str(e)}")
        return False


async def test_gemini_google_search():
    """测试 Gemini Google Search 功能"""
    print("\n🔍 测试 Gemini Google Search 功能...")
    
    # 检查是否有 Gemini API Key
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠️  未设置 GEMINI_API_KEY 环境变量，跳过 Gemini Google Search 测试")
        return True
    
    test_queries = ["AI news 2025"]
    
    try:
        results = await gemini_google_search_async(
            search_queries=test_queries,
            max_results=2,
            include_raw_content=False
        )
        
        print(f"✅ Gemini Google Search 测试成功！")
        print(f"📊 返回结果数量: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"\n--- 查询 {i+1}: {result['query']} ---")
            print(f"结果数量: {len(result['results'])}")
            if result.get('answer'):
                print(f"AI 回答: {result['answer'][:200]}...")
            for j, item in enumerate(result['results'][:2]):
                print(f"  {j+1}. {item['title']}")
                print(f"     URL: {item['url']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini Google Search 测试失败: {str(e)}")
        return False


def test_configuration():
    """测试配置更新"""
    print("\n⚙️  测试配置更新...")
    
    try:
        # 测试新的搜索 API 枚举
        assert hasattr(SearchAPI, 'GOOGLESEARCH')
        assert hasattr(SearchAPI, 'GEMINIGOOGLESEARCH')
        assert hasattr(SearchAPI, 'AZUREAISEARCH')
        
        print("✅ 配置枚举测试成功！")
        
        # 测试配置类
        config = MultiAgentConfiguration(
            search_api=SearchAPI.GOOGLESEARCH,
            search_api_config={"max_results": 5}
        )
        
        assert config.search_api == SearchAPI.GOOGLESEARCH
        print("✅ 配置类测试成功！")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {str(e)}")
        return False


async def main():
    """主测试函数"""
    print("🚀 开始测试 Google Search 集成...")
    
    results = []
    
    # 测试配置
    results.append(test_configuration())
    
    # 测试 Google Search
    results.append(await test_google_search())
    
    # 测试 Gemini Google Search
    results.append(await test_gemini_google_search())
    
    # 总结测试结果
    print("\n" + "="*50)
    print("📋 测试总结:")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 所有测试通过！({passed}/{total})")
        print("\n✨ Google Search 集成配置完成！")
        print("\n📝 使用说明:")
        print("1. Google Custom Search API: 设置 GOOGLE_API_KEY 和 GOOGLE_CX 环境变量")
        print("2. Gemini Google Search: 设置 GEMINI_API_KEY 环境变量")
        print("3. 在配置中设置 search_api='googlesearch' 或 'geminigooglesearch'")
    else:
        print(f"⚠️  部分测试失败 ({passed}/{total})")
        print("请检查错误信息并修复相关问题")
    
    return passed == total


if __name__ == "__main__":
    # 运行测试
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 