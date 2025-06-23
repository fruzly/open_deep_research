"""
测试完整报告生成
================

简化版本的完整报告生成测试脚本
"""

import asyncio
import uuid
import os
from datetime import datetime
from src.open_deep_research.multi_agent import supervisor_builder
from langgraph.checkpoint.memory import MemorySaver

async def test_full_report_generation():
    """测试完整报告生成功能"""
    print("🚀 测试完整报告生成功能")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查必需的 API 密钥
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ 缺少 GOOGLE_API_KEY 环境变量")
        print("请设置: os.environ['GOOGLE_API_KEY'] = 'your_api_key'")
        return False
    
    try:
        # 创建 agent
        checkpointer = MemorySaver()
        agent = supervisor_builder.compile(name="full_report_test", checkpointer=checkpointer)
        
        # 🔧 完整报告配置 - 启用搜索功能
        config = {
            "thread_id": str(uuid.uuid4()),
            # 使用 DuckDuckGo 搜索（免费，无需额外 API 密钥）
            "search_api": "tavily",  # 🔑 关键：启用搜索API
            "supervisor_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "researcher_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "number_of_queries": 20,  # 每个研究节的查询数量
            "ask_for_clarification": False,  # 不要求澄清，直接生成
            "include_source_str": True,  # 包含源信息用于评估
        }
        
        # 🔧 线程配置 - 增加递归限制以支持完整工作流
        thread_config = {
            "configurable": config,
            "recursion_limit": 100  # 🔑 关键：增加递归限制支持完整工作流
        }
        
        # 🔧 测试查询 - 使用具体的研究主题
        test_query = """
        Write a comprehensive report about Python programming language in 2024. 
        Cover the following aspects:
        1. Current state and recent developments
        2. Popular frameworks and libraries
        3. Use cases and applications
        4. Community and ecosystem
        5. Future outlook
        
        Please provide a detailed, well-structured report with specific examples.
        """
        
        test_msg = [{"role": "user", "content": test_query.strip()}]
        
        print(f"📝 测试查询: {test_query.strip()}")
        print(f"⚙️ 配置: {config}")
        print("🔍 预期: 生成包含引言、主体章节和结论的完整报告")
        print("⏳ 预计时间: 2-5 分钟（包含网络搜索）")
        
        # 执行工作流
        print("\n🚀 开始执行完整报告生成工作流...")
        start_time = datetime.now()
        
        response = await agent.ainvoke({"messages": test_msg}, config=thread_config)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 工作流执行完成！耗时: {duration:.1f} 秒")
        
        # 检查结果
        state = agent.get_state(thread_config)
        
        # 显示消息统计
        if hasattr(state, 'values') and 'messages' in state.values:
            message_count = len(state.values['messages'])
            print(f"📊 总消息数: {message_count}")
            
            # 分析消息类型
            roles = []
            for msg in state.values['messages']:
                if hasattr(msg, 'type'):
                    msg_type = msg.type
                elif hasattr(msg, 'role'):
                    msg_type = msg.role
                else:
                    msg_type = type(msg).__name__.lower()
                    if 'human' in msg_type:
                        msg_type = 'user'
                    elif 'ai' in msg_type:
                        msg_type = 'assistant'
                roles.append(msg_type)
            
            print(f"📋 消息序列: {' -> '.join(roles[:10])}{'...' if len(roles) > 10 else ''}")
        
        # 检查最终报告
        if hasattr(state, 'values') and 'final_report' in state.values:
            final_report = state.values['final_report']
            if final_report and final_report.strip():
                print("\n🎉 成功生成完整报告！")
                print("="*60)
                print("📄 生成的完整报告:")
                print("="*60)
                print(final_report)
                print("="*60)
                
                # 保存报告到文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_full_report_{timestamp}.md"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# 测试生成的完整研究报告\n\n")
                    f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**耗时:** {duration:.1f} 秒\n")
                    f.write(f"**消息数:** {message_count}\n")
                    f.write(f"**配置:** {config}\n\n")
                    f.write("---\n\n")
                    f.write(final_report)
                
                print(f"💾 报告已保存到: {filename}")
                
                # 报告质量分析
                lines = final_report.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                print(f"\n📈 报告统计:")
                print(f"  - 总行数: {len(lines)}")
                print(f"  - 非空行数: {len(non_empty_lines)}")
                print(f"  - 字符数: {len(final_report)}")
                print(f"  - 是否包含标题: {'✅' if any('#' in line for line in lines) else '❌'}")
                print(f"  - 是否包含引言: {'✅' if any('引言' in line or 'Introduction' in line or '介绍' in line for line in lines) else '❌'}")
                print(f"  - 是否包含结论: {'✅' if any('结论' in line or 'Conclusion' in line or '总结' in line for line in lines) else '❌'}")
                
                return True
            else:
                print("❌ 报告生成失败：final_report 为空")
                print("🔍 可能原因:")
                print("  - 工作流没有完成完整的报告生成过程")
                print("  - 配置不正确")
                print("  - 递归限制不足")
                return False
        else:
            print("❌ 报告生成失败：未找到 final_report")
            print(f"🔍 状态键: {list(state.values.keys()) if hasattr(state, 'values') else 'N/A'}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("\n🔴 完整错误追踪:")
        import traceback
        traceback.print_exc()
        
        print("\n💡 故障排除建议:")
        print("1. 检查 GOOGLE_API_KEY 是否正确设置")
        print("2. 检查网络连接是否正常")
        print("3. 尝试减少 number_of_queries 或 recursion_limit")
        print("4. 检查是否有其他 API 限制")
        
        return False

async def main():
    """主函数"""
    print("🌟 Open Deep Research - 完整报告生成测试")
    print("="*50)
    
    # API 密钥检查
    if os.environ.get("GOOGLE_API_KEY"):
        print("✅ GOOGLE_API_KEY: 已设置")
    else:
        print("❌ GOOGLE_API_KEY: 未设置")
        print("\n请先设置 API 密钥:")
        print("os.environ['GOOGLE_API_KEY'] = 'your_api_key_here'")
        return
    
    # 执行测试
    success = await test_full_report_generation()
    
    if success:
        print("\n🎉 测试成功！完整报告生成功能正常工作。")
        print("💡 提示: 您可以修改 test_query 来生成不同主题的报告。")
    else:
        print("\n💥 测试失败！请检查配置和错误信息。")

if __name__ == "__main__":
    print("""
🚀 快速开始指南:

1. 设置 API 密钥:
   os.environ['GOOGLE_API_KEY'] = 'your_google_api_key'

2. 运行测试:
   python test_full_report.py

3. 检查生成的报告文件

注意: 完整报告生成需要 2-5 分钟，请耐心等待。
""")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc() 