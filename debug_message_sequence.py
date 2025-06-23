"""
消息序列深度调试脚本
==================

专门分析和调试Gemini消息序列问题
"""

import asyncio
import uuid
import os
from datetime import datetime
from src.open_deep_research.multi_agent import supervisor_builder, fix_gemini_message_sequence, ensure_user_message_ending
from langgraph.checkpoint.memory import MemorySaver

def analyze_message_sequence(messages, title=""):
    """详细分析消息序列"""
    print(f"\n🔍 {title} 消息序列分析")
    print("="*50)
    
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = str(msg.get('content', ''))[:100]
            tool_calls = msg.get('tool_calls', [])
            print(f"  [{i}] {role}: {content}{'...' if len(str(msg.get('content', ''))) > 100 else ''}")
            if tool_calls:
                print(f"      工具调用: {len(tool_calls)} 个")
        else:
            # LangChain 消息对象
            msg_type = type(msg).__name__
            role = getattr(msg, 'role', 'unknown')
            content = str(getattr(msg, 'content', ''))[:100]
            tool_calls = getattr(msg, 'tool_calls', [])
            print(f"  [{i}] {msg_type}({role}): {content}{'...' if len(str(getattr(msg, 'content', ''))) > 100 else ''}")
            if tool_calls:
                print(f"      工具调用: {len(tool_calls)} 个")
    
    # 检查序列问题
    problems = []
    for i in range(len(messages) - 1):
        current_msg = messages[i]
        next_msg = messages[i + 1]
        
        # 获取角色
        current_role = current_msg.get('role') if isinstance(current_msg, dict) else getattr(current_msg, 'role', type(current_msg).__name__.lower())
        next_role = next_msg.get('role') if isinstance(next_msg, dict) else getattr(next_msg, 'role', type(next_msg).__name__.lower())
        
        # 检查工具调用
        current_has_tools = bool(current_msg.get('tool_calls') if isinstance(current_msg, dict) else getattr(current_msg, 'tool_calls', []))
        
        if current_role == 'assistant' and current_has_tools and next_role == 'assistant':
            problems.append(f"位置 {i}-{i+1}: 连续的assistant消息，且第一个包含tool_calls")
    
    if problems:
        print(f"\n⚠️ 发现问题:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print(f"\n✅ 消息序列看起来正常")
    
    return problems

async def test_message_sequence_fix():
    """测试消息序列修复功能"""
    print("🧪 测试消息序列修复功能")
    print("="*50)
    
    try:
        # 创建 agent
        checkpointer = MemorySaver()
        agent = supervisor_builder.compile(name="sequence_debug", checkpointer=checkpointer)
        
        # 配置
        config = {
            "thread_id": str(uuid.uuid4()),
            "search_api": "none",
            "supervisor_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "researcher_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "number_of_queries": 1,
            "ask_for_clarification": False,
            "include_source_str": False,
        }
        
        thread_config = {
            "configurable": config,
            "recursion_limit": 15  # 降低限制以减少复杂度
        }
        
        # 简单查询
        query = "Write a short report about Python with just 2 sections: Introduction and Conclusion."
        test_msg = [{"role": "user", "content": query}]
        
        print(f"📝 测试查询: {query}")
        
        # 分析初始消息
        analyze_message_sequence(test_msg, "初始")
        
        # 执行工作流，但在每一步都分析消息序列
        print(f"\n🚀 开始执行工作流...")
        
        # 使用 astream 来逐步观察
        step_count = 0
        async for chunk in agent.astream({"messages": test_msg}, config=thread_config):
            step_count += 1
            print(f"\n📊 步骤 {step_count}: {chunk}")
            
            # 分析当前状态的消息
            if 'messages' in chunk:
                current_messages = chunk['messages']
                if current_messages:
                    analyze_message_sequence(current_messages, f"步骤 {step_count}")
            
            # 限制步骤数以避免无限循环
            if step_count > 10:
                print("⚠️ 达到最大步骤数，停止执行")
                break
        
        print(f"\n✅ 工作流执行完成，共 {step_count} 步")
        
        # 获取最终状态
        final_state = agent.get_state(thread_config)
        if hasattr(final_state, 'values') and 'messages' in final_state.values:
            analyze_message_sequence(final_state.values['messages'], "最终状态")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        
        # 分析错误类型
        error_str = str(e)
        if "Please ensure that function call turn" in error_str:
            print("\n🔧 Gemini消息序列错误详细分析:")
            print("  - 错误类型: 400 function call turn 错误")
            print("  - 可能原因: 连续的assistant消息包含tool_calls")
            print("  - 修复函数可能没有正确处理复杂的消息序列")
        
        import traceback
        traceback.print_exc()
        return False

async def test_fix_functions():
    """测试修复函数的独立功能"""
    print("\n🔧 测试修复函数")
    print("="*50)
    
    # 模拟问题消息序列
    problematic_messages = [
        {"role": "user", "content": "请生成报告"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "Section", "args": {}, "id": "1"}]},
        {"role": "tool", "content": "Section 1 result", "name": "Section", "tool_call_id": "1"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "Section", "args": {}, "id": "2"}]},
        {"role": "tool", "content": "Section 2 result", "name": "Section", "tool_call_id": "2"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "FinishResearch", "args": {}, "id": "3"}]},
    ]
    
    print("🔍 原始问题消息序列:")
    analyze_message_sequence(problematic_messages, "原始")
    
    # 应用修复函数
    print("\n🔧 应用修复函数...")
    fixed_messages = fix_gemini_message_sequence(problematic_messages)
    analyze_message_sequence(fixed_messages, "修复后")
    
    # 确保用户消息结尾
    final_messages = ensure_user_message_ending(fixed_messages, "请继续")
    analyze_message_sequence(final_messages, "最终")
    
    return final_messages

async def main():
    """主函数"""
    print("🔍 Gemini 消息序列深度调试")
    print("="*60)
    print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # API密钥检查
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ 缺少 GOOGLE_API_KEY 环境变量")
        return
    
    print("✅ GOOGLE_API_KEY: 已设置")
    
    # 测试1: 修复函数独立测试
    print(f"\n{'='*60}")
    await test_fix_functions()
    
    # 测试2: 实际工作流测试
    print(f"\n{'='*60}")
    success = await test_message_sequence_fix()
    
    # 总结
    print(f"\n{'='*60}")
    print("📊 调试总结")
    print("="*60)
    
    if success:
        print("✅ 调试成功完成")
    else:
        print("❌ 发现问题需要进一步修复")
        print("\n💡 建议:")
        print("  1. 检查修复函数是否正确处理所有边界情况")
        print("  2. 考虑在工具调用后强制插入tool响应消息")
        print("  3. 确保最后一条消息总是user消息")

if __name__ == "__main__":
    print("""
🔍 Gemini 消息序列深度调试工具
==============================

本脚本将详细分析和调试Gemini消息序列问题：
1. 分析每一步的消息序列
2. 识别具体的问题模式
3. 测试修复函数的效果
4. 提供详细的调试信息

运行前请确保设置 GOOGLE_API_KEY 环境变量。
""")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断调试")
    except Exception as e:
        print(f"\n❌ 调试失败: {e}")
        import traceback
        traceback.print_exc() 