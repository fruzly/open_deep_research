#!/usr/bin/env python3
"""
Google Gemini 模型使用示例
演示如何在 open_deep_research 项目中配置和使用 Google Gemini 模型
"""

import os
import uuid
import asyncio
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 检查是否安装了必要的包
try:
    from open_deep_research.graph import builder
    from langgraph.checkpoint.memory import MemorySaver
    print("✅ open_deep_research 导入成功")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装 open_deep_research 包")
    exit(1)


async def test_gemini_basic():
    """测试 Gemini 模型的基本功能"""
    
    # 检查 API 密钥
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ 请设置 GOOGLE_API_KEY 环境变量")
        return
    
    print("🧪 测试 Gemini 模型集成...")
    
    # 编译图
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    print("✅ 图编译成功")
    
    # 配置 Gemini 模型
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": "tavily",  # 确保也设置了 TAVILY_API_KEY
            
            # 使用 Gemini 作为规划器和写作器
            "planner_provider": "google_genai",
            "planner_model": "gemini-2.5-flash-lite-preview-06-17",  # 使用较经济的模型
            
            "writer_provider": "google_genai",
            "writer_model": "gemini-2.5-flash-lite-preview-06-17",
            
            "max_search_depth": 1,  # 减少搜索深度以节省成本
            "number_of_queries": 1,  # 减少查询数量
        }
    }
    
    # 简单的研究主题
    topic = "Google Gemini 模型的特点和应用"
    
    print(f"🔍 开始研究主题: {topic}")
    
    try:
        # 运行工作流
        final_report = None
        async for event in graph.astream({"topic": topic}, config, stream_mode="updates"):
            if '__interrupt__' in event:
                # 自动批准计划
                print("📋 收到计划审批请求，自动批准...")
                async for event2 in graph.astream(
                    {"type": "interrupt", "value": True}, 
                    config, 
                    stream_mode="updates"
                ):
                    if 'compile_final_report' in event2:
                        print("✅ 报告生成完成")
                        break
            elif 'compile_final_report' in event:
                print("✅ 报告编译完成")
        
        # 获取最终结果
        final_state = graph.get_state(config)
        if final_state.values.get('final_report'):
            final_report = final_state.values['final_report']
            print("\n" + "="*80)
            print("📄 最终研究报告")
            print("="*80)
            print(final_report)
            print("="*80)
        else:
            print("❌ 未能生成最终报告")
            
    except Exception as e:
        print(f"❌ 执行错误: {e}")
        print("请检查 API 密钥和网络连接")


async def test_gemini_multi_agent():
    """测试多智能体模式的 Gemini 配置"""
    
    try:
        from open_deep_research.multi_agent import graph
        print("✅ 多智能体模块导入成功")
    except ImportError as e:
        print(f"❌ 多智能体模块导入失败: {e}")
        return
    
    # 配置多智能体系统
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": "tavily",
            
            # 两个角色都使用 Gemini
            "supervisor_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "researcher_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            
            "ask_for_clarification": False,  # 关闭澄清以简化测试
        }
    }
    
    print("🤖 测试多智能体 Gemini 配置...")
    
    # 执行简单的研究任务
    messages = [{
        "role": "user", 
        "content": "请简要介绍人工智能的发展历程"
    }]
    
    try:
        result = await graph.ainvoke({"messages": messages}, config=config)
        
        if result.get("final_report"):
            print("\n" + "="*80)
            print("🤖 多智能体研究报告")
            print("="*80)
            print(result["final_report"])
            print("="*80)
        else:
            print("❌ 多智能体模式未能生成报告")
            
    except Exception as e:
        print(f"❌ 多智能体执行错误: {e}")


def check_environment():
    """检查环境配置"""
    print("🔧 检查环境配置...")
    
    required_keys = ["GOOGLE_API_KEY"]
    optional_keys = ["TAVILY_API_KEY", "OPENAI_API_KEY"]
    
    all_good = True
    
    for key in required_keys:
        if os.getenv(key):
            print(f"✅ {key}: 已设置")
        else:
            print(f"❌ {key}: 未设置（必需）")
            all_good = False
    
    for key in optional_keys:
        if os.getenv(key):
            print(f"✅ {key}: 已设置")
        else:
            print(f"⚠️  {key}: 未设置（可选，某些功能可能需要）")
    
    return all_good


async def main():
    """主函数"""
    print("🚀 Google Gemini 模型测试开始")
    print("="*60)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境配置不完整，请设置必需的 API 密钥")
        print("\n📝 设置方法：")
        print("1. 创建 .env 文件")
        print("2. 添加: GOOGLE_API_KEY=your_api_key_here")
        print("3. 可选: TAVILY_API_KEY=your_tavily_key")
        return
    
    print("\n" + "="*60)
    
    # 测试基础功能
    print("🧪 测试 1: 基础 Graph 模式")
    await test_gemini_basic()
    
    print("\n" + "="*60)
    
    # 测试多智能体功能
    print("🧪 测试 2: 多智能体模式")
    # await test_gemini_multi_agent()
    
    print("\n" + "="*60)
    print("🎉 测试完成！")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main()) 