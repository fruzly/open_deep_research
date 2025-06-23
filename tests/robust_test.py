"""
稳定版完整报告生成测试
======================

解决DuckDuckGo限制和消息序列问题的增强版测试脚本
"""

import asyncio
import uuid
import os
from datetime import datetime
from src.open_deep_research.multi_agent import supervisor_builder
from langgraph.checkpoint.memory import MemorySaver

async def test_no_search_mode():
    """测试无搜索模式 - 最稳定的配置"""
    print("🧪 测试无搜索模式（最稳定）")
    print("="*50)
    
    try:
        # 创建 agent
        checkpointer = MemorySaver()
        agent = supervisor_builder.compile(name="no_search_test", checkpointer=checkpointer)
        
        # 🔧 无搜索配置 - 避免所有网络问题
        config = {
            "thread_id": str(uuid.uuid4()),
            "search_api": "none",  # 🔑 关键：禁用搜索
            "supervisor_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "researcher_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "number_of_queries": 1,
            "ask_for_clarification": False,
            "include_source_str": False,  # 无搜索时不需要源信息
        }
        
        thread_config = {
            "configurable": config,
            "recursion_limit": 25  # 适中的递归限制
        }
        
        # 简化的测试查询
        query = """
        Write a comprehensive report about Python programming language. 
        Include the following sections:
        1. Introduction - What is Python and why is it popular
        2. Key Features and Advantages
        3. Main Applications and Use Cases
        4. Popular Libraries and Frameworks
        5. Community and Ecosystem
        6. Future Outlook and Conclusion
        
        Please provide a well-structured, detailed report.
        """
        
        test_msg = [{"role": "user", "content": query.strip()}]
        
        print(f"📝 查询: {query.strip()[:100]}...")
        print(f"⚙️ 配置: 无搜索模式，递归限制: {thread_config['recursion_limit']}")
        print(f"⏱️ 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 执行工作流
        start_time = datetime.now()
        
        response = await agent.ainvoke({"messages": test_msg}, config=thread_config)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 工作流执行完成！耗时: {duration:.1f} 秒")
        
        # 检查结果
        state = agent.get_state(thread_config)
        
        # 消息统计
        message_count = 0
        if hasattr(state, 'values') and 'messages' in state.values:
            message_count = len(state.values['messages'])
            print(f"📊 总消息数: {message_count}")
        
        # 检查最终报告
        if hasattr(state, 'values') and 'final_report' in state.values:
            final_report = state.values['final_report']
            if final_report and final_report.strip():
                print(f"\n🎉 成功生成完整报告！")
                print(f"📏 报告长度: {len(final_report)} 字符")
                
                # 保存报告
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stable_report_{timestamp}.md"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# 稳定模式生成的Python报告\n\n")
                    f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**配置模式:** 无搜索（最稳定）\n")
                    f.write(f"**耗时:** {duration:.1f} 秒\n")
                    f.write(f"**消息数:** {message_count}\n\n")
                    f.write("---\n\n")
                    f.write(final_report)
                
                print(f"💾 报告已保存到: {filename}")
                
                # 显示报告摘要
                lines = final_report.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                has_headers = any('#' in line for line in lines)
                
                print(f"\n📊 报告分析:")
                print(f"  - 总行数: {len(lines)}")
                print(f"  - 有效行数: {len(non_empty_lines)}")
                print(f"  - 包含标题: {'✅' if has_headers else '❌'}")
                
                # 显示报告开头预览
                preview_lines = final_report.split('\n')[:10]
                print(f"\n📖 报告预览（前10行）:")
                for i, line in enumerate(preview_lines):
                    if line.strip():
                        print(f"  {i+1}: {line[:80]}{'...' if len(line) > 80 else ''}")
                
                return True
            else:
                print("❌ 报告生成失败：final_report 为空")
                return False
        else:
            print("❌ 报告生成失败：未找到 final_report")
            print(f"🔍 可用状态键: {list(state.values.keys()) if hasattr(state, 'values') else 'N/A'}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print(f"\n🔍 错误分析:")
        
        error_str = str(e)
        if "Please ensure that function call turn" in error_str:
            print("  - 类型: Gemini消息序列错误")
            print("  - 原因: 即使在无搜索模式下仍有消息序列问题")
            print("  - 建议: 检查multi_agent.py中的修复函数是否正确应用")
        else:
            print(f"  - 类型: 其他错误")
            print(f"  - 详情: {error_str}")
        
        import traceback
        traceback.print_exc()
        return False

async def test_conservative_search():
    """测试保守搜索模式"""
    print("\n🧪 测试保守搜索模式")
    print("="*50)
    
    try:
        checkpointer = MemorySaver()
        agent = supervisor_builder.compile(name="conservative_test", checkpointer=checkpointer)
        
        # 🔧 保守搜索配置
        config = {
            "thread_id": str(uuid.uuid4()),
            "search_api": "duckduckgo",
            "supervisor_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "researcher_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "number_of_queries": 1,  # 🔑 只做1次查询
            "ask_for_clarification": False,
            "include_source_str": True,
        }
        
        thread_config = {
            "configurable": config,
            "recursion_limit": 20  # 较低的递归限制
        }
        
        # 更简单的查询
        query = "Write a brief report about Python programming language, covering its main features and applications."
        
        test_msg = [{"role": "user", "content": query}]
        
        print(f"📝 查询: {query}")
        print(f"⚙️ 配置: 保守搜索（1次查询），递归限制: 20")
        print("⚠️ 注意: 可能遇到DuckDuckGo速率限制")
        
        # 添加延迟避免速率限制
        print("🕐 等待3秒避免速率限制...")
        await asyncio.sleep(3)
        
        start_time = datetime.now()
        response = await agent.ainvoke({"messages": test_msg}, config=thread_config)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 保守搜索测试完成！耗时: {duration:.1f} 秒")
        
        # 检查结果
        state = agent.get_state(thread_config)
        if hasattr(state, 'values') and 'final_report' in state.values:
            final_report = state.values['final_report']
            if final_report and final_report.strip():
                print(f"🎉 保守搜索模式成功生成报告！({len(final_report)} 字符)")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conservative_report_{timestamp}.md"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# 保守搜索模式生成的报告\n\n")
                    f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**配置模式:** 保守搜索（1次查询）\n")
                    f.write(f"**耗时:** {duration:.1f} 秒\n\n")
                    f.write("---\n\n")
                    f.write(final_report)
                
                print(f"💾 报告已保存到: {filename}")
                return True
            else:
                print("❌ 保守搜索模式未生成有效报告")
                return False
        else:
            print("❌ 保守搜索模式失败：未找到final_report")
            return False
            
    except Exception as e:
        print(f"❌ 保守搜索测试失败: {e}")
        
        if "Ratelimit" in str(e) or "202" in str(e):
            print("🚫 确认遇到DuckDuckGo速率限制")
            print("💡 建议等待几分钟后重试，或使用无搜索模式")
        
        return False

async def main():
    """主测试函数"""
    print("🛡️ 稳定版完整报告生成测试")
    print("="*60)
    print(f"🕐 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # API密钥检查
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ 缺少 GOOGLE_API_KEY 环境变量")
        print("请设置: os.environ['GOOGLE_API_KEY'] = 'your_api_key'")
        return
    
    print("✅ GOOGLE_API_KEY: 已设置")
    
    # 分层测试策略
    print("\n🎯 测试策略: 从最稳定到较复杂")
    print("1. 无搜索模式（基于模型知识）")
    print("2. 保守搜索模式（如果无搜索成功）")
    
    results = {}
    
    # 测试1: 无搜索模式
    print(f"\n{'='*60}")
    success1 = await test_no_search_mode()
    results["no_search"] = success1
    
    if success1:
        print("\n✅ 无搜索模式测试成功！这是最稳定的配置。")
        
        # 询问是否继续测试搜索模式
        print("\n🤔 无搜索模式已成功，是否继续测试搜索模式？")
        print("  - 搜索模式可能遇到速率限制")
        print("  - 但可以生成更丰富的内容")
        
        # 自动继续测试（可以根据需要修改）
        print("⏭️ 继续测试搜索模式...")
        await asyncio.sleep(5)  # 等待避免速率限制
        
        # 测试2: 保守搜索模式
        print(f"\n{'='*60}")
        success2 = await test_conservative_search()
        results["conservative_search"] = success2
    else:
        print("\n❌ 无搜索模式测试失败")
        print("💡 这表明存在基础配置问题，跳过搜索模式测试")
    
    # 测试总结
    print(f"\n{'='*60}")
    print("📊 测试结果总结")
    print("="*60)
    
    successful_modes = [mode for mode, success in results.items() if success]
    failed_modes = [mode for mode, success in results.items() if not success]
    
    if successful_modes:
        print(f"✅ 成功的模式: {', '.join(successful_modes)}")
        print(f"💡 推荐使用: {successful_modes[0]}")
        
        print(f"\n📁 生成的文件:")
        for mode in successful_modes:
            if mode == "no_search":
                print(f"  - stable_report_*.md (无搜索模式)")
            elif mode == "conservative_search":
                print(f"  - conservative_report_*.md (保守搜索模式)")
    
    if failed_modes:
        print(f"❌ 失败的模式: {', '.join(failed_modes)}")
    
    # 最终建议
    print(f"\n🎯 使用建议:")
    if "no_search" in successful_modes:
        print("  ✅ 日常使用推荐: 无搜索模式")
        print("     - 稳定可靠，不依赖外部API")
        print("     - 基于模型训练数据生成报告")
        print("     - 适合大多数通用主题")
    
    if "conservative_search" in successful_modes:
        print("  ✅ 高质量需求: 保守搜索模式")
        print("     - 包含最新信息")
        print("     - 需要稳定的网络连接")
        print("     - 注意API速率限制")
    
    if not successful_modes:
        print("  ❌ 所有模式都失败了")
        print("     - 检查API密钥和网络连接")
        print("     - 查看详细错误信息")
        print("     - 考虑联系技术支持")
    
    print(f"\n🏁 测试完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    print("""
🛡️ 稳定版完整报告生成测试
============================

本脚本专门解决以下问题：
✅ DuckDuckGo API 速率限制 (202 Ratelimit)
✅ Gemini 消息序列错误 (400 function call turn)
✅ 网络连接不稳定
✅ 配置参数不当

测试包含两种模式：
1. 无搜索模式 - 最稳定，基于模型知识
2. 保守搜索模式 - 包含搜索，但减少API调用

运行前请确保设置 GOOGLE_API_KEY 环境变量。
""")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试运行失败: {e}")
        import traceback
        traceback.print_exc() 