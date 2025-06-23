#!/usr/bin/env python3
"""
调试supervisor函数接收到的消息
"""

import sys
import os
sys.path.append('src')

# 修改multi_agent.py来添加调试信息
def patch_supervisor_function():
    """给supervisor函数添加调试信息"""
    
    import src.open_deep_research.multi_agent as multi_agent_module
    
    # 保存原始函数
    original_supervisor = multi_agent_module.supervisor
    
    async def debug_supervisor(state, config):
        """带调试信息的supervisor函数"""
        print(f"\n🔍 [DEBUG] supervisor函数被调用")
        print(f"📊 state类型: {type(state)}")
        print(f"🔑 state键: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
        
        if 'messages' in state:
            messages = state['messages']
            print(f"📝 messages类型: {type(messages)}")
            print(f"📏 messages长度: {len(messages) if messages else 0}")
            
            if messages:
                for i, msg in enumerate(messages):
                    print(f"  消息 {i}: 类型={type(msg)}")
                    if hasattr(msg, '__dict__'):
                        print(f"    属性: {msg.__dict__}")
                    elif isinstance(msg, dict):
                        print(f"    字典键: {list(msg.keys())}")
                        print(f"    内容: {msg}")
                    else:
                        print(f"    值: {msg}")
        else:
            print("❌ state中没有'messages'键")
        
        # 调用原始函数
        return await original_supervisor(state, config)
    
    # 替换函数
    multi_agent_module.supervisor = debug_supervisor
    print("✅ supervisor函数已被调试版本替换")

if __name__ == "__main__":
    # 先打补丁
    patch_supervisor_function()
    
    # 然后运行测试
    import asyncio
    import os
    from datetime import datetime
    from langgraph.checkpoint.memory import MemorySaver
    from src.open_deep_research.multi_agent import supervisor_builder
    
    async def run_debug_test():
        """运行调试测试"""
        print("🚀 开始调试测试")
        
        # 设置API密钥
        if not os.environ.get("GOOGLE_API_KEY"):
            print("❌ 缺少 GOOGLE_API_KEY 环境变量")
            return
        
        try:
            checkpointer = MemorySaver()
            agent = supervisor_builder.compile(name="debug_test", checkpointer=checkpointer)
            
            config = {
                "thread_id": "debug_test_123",
                "search_api": "none",  # 无搜索模式
                "supervisor_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
                "researcher_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
                "number_of_queries": 0,
                "ask_for_clarification": False,
                "include_source_str": False,
            }
            
            thread_config = {
                "configurable": config,
                "recursion_limit": 10
            }
            
            # 简单的测试消息
            test_msg = [{"role": "user", "content": "Write a short report about Python."}]
            
            print(f"\n📝 发送的消息: {test_msg}")
            
            # 执行工作流
            response = await agent.ainvoke({"messages": test_msg}, config=thread_config)
            
            print(f"✅ 测试完成")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行测试
    asyncio.run(run_debug_test()) 