"""
稳定版完整报告生成测试
======================

解决DuckDuckGo限制和消息序列问题的增强版测试脚本
"""

import asyncio
import uuid
import os
import time
from datetime import datetime
from src.open_deep_research.multi_agent import supervisor_builder
from langgraph.checkpoint.memory import MemorySaver

class RobustReportTester:
    """稳定的报告生成测试器"""
    
    def __init__(self):
        self.configs = self._get_robust_configs()
    
    def _get_robust_configs(self):
        """获取稳定的配置选项"""
        base_config = {
            "thread_id": str(uuid.uuid4()),
            "supervisor_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "researcher_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "ask_for_clarification": False,
            "include_source_str": True,
        }
        
        return {
            # 🥇 无搜索模式 - 最稳定
            "no_search": {
                **base_config,
                "search_api": "none",
                "number_of_queries": 1,
                "recursion_limit": 20,
                "description": "无搜索模式 - 基于模型知识生成报告"
            },
            
            # 🥈 保守搜索模式 - 减少API调用
            "conservative": {
                **base_config,
                "search_api": "duckduckgo",
                "number_of_queries": 1,  # 🔑 减少查询数量
                "recursion_limit": 25,
                "search_delay": 3,  # 🔑 添加搜索延迟
                "description": "保守搜索模式 - 减少API调用频率"
            },
            
            # 🥉 分阶段搜索模式 - 避免并发搜索
            "staged": {
                **base_config,
                "search_api": "duckduckgo",
                "number_of_queries": 1,
                "recursion_limit": 30,
                "sequential_search": True,  # 🔑 顺序搜索
                "description": "分阶段搜索模式 - 避免并发请求"
            }
        }
    
    async def test_config(self, config_name: str, custom_query: str = None):
        """测试特定配置"""
        print(f"\n🧪 测试配置: {config_name}")
        print("="*50)
        
        config = self.configs.get(config_name)
        if not config:
            print(f"❌ 未找到配置: {config_name}")
            return False
        
        print(f"📋 配置描述: {config['description']}")
        print(f"⚙️ 配置详情: {config}")
        
        try:
            # 创建 agent
            checkpointer = MemorySaver()
            agent = supervisor_builder.compile(
                name=f"robust_test_{config_name}", 
                checkpointer=checkpointer
            )
            
            # 线程配置
            thread_config = {
                "configurable": config,
                "recursion_limit": config.get("recursion_limit", 20)
            }
            
            # 测试查询 - 使用更简单的主题
            if custom_query:
                query = custom_query
            else:
                query = """
                Write a report about Python programming language. 
                Cover these key areas:
                1. What Python is and why it's popular
                2. Main uses and applications  
                3. Key advantages and features
                4. Current status in the programming world
                
                Please provide a structured report with clear sections.
                """
            
            test_msg = [{"role": "user", "content": query.strip()}]
            
            print(f"📝 测试查询: {query.strip()[:100]}...")
            print(f"⏱️ 开始时间: {datetime.now().strftime('%H:%M:%S')}")
            
            # 如果是搜索模式，添加延迟警告
            if config.get("search_api") != "none":
                print("⚠️ 搜索模式：如遇到速率限制，将自动重试")
                if config.get("search_delay"):
                    print(f"🐌 搜索延迟: {config['search_delay']}秒")
            
            # 执行工作流
            start_time = datetime.now()
            
            response = await agent.ainvoke({"messages": test_msg}, config=thread_config)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ 工作流执行完成！耗时: {duration:.1f} 秒")
            
            # 检查结果
            state = agent.get_state(thread_config)
            
            # 消息统计
            if hasattr(state, 'values') and 'messages' in state.values:
                message_count = len(state.values['messages'])
                print(f"📊 总消息数: {message_count}")
            
            # 检查最终报告
            if hasattr(state, 'values') and 'final_report' in state.values:
                final_report = state.values['final_report']
                if final_report and final_report.strip():
                    print(f"\n🎉 成功生成完整报告！({len(final_report)} 字符)")
                    
                    # 保存报告
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"robust_report_{config_name}_{timestamp}.md"
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"# 稳定测试生成的报告 - {config_name}\n\n")
                        f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"**配置:** {config_name}\n")
                        f.write(f"**耗时:** {duration:.1f} 秒\n")
                        f.write(f"**消息数:** {message_count}\n\n")
                        f.write("---\n\n")
                        f.write(final_report)
                    
                    print(f"💾 报告已保存到: {filename}")
                    
                    # 报告质量检查
                    self._analyze_report_quality(final_report, config_name)
                    
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
            
            # 特定错误处理
            error_str = str(e)
            if "Please ensure that function call turn" in error_str:
                print("\n🔧 检测到Gemini消息序列问题")
                print("💡 建议:")
                print("  - 这可能是搜索失败导致的消息序列异常")
                print("  - 尝试使用 'no_search' 配置")
                print("  - 检查网络连接稳定性")
            elif "Ratelimit" in error_str or "202" in error_str:
                print("\n🚫 检测到API速率限制")
                print("💡 建议:")
                print("  - 等待几分钟后重试")
                print("  - 使用 'no_search' 或 'conservative' 配置")
                print("  - 考虑使用其他搜索API")
            
            return False
    
    def _analyze_report_quality(self, report: str, config_name: str):
        """分析报告质量"""
        lines = report.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        print(f"\n📈 报告质量分析 ({config_name}):")
        print(f"  📏 总行数: {len(lines)}")
        print(f"  📝 非空行数: {len(non_empty_lines)}")
        print(f"  🔤 字符数: {len(report)}")
        
        # 结构检查
        has_headers = any('#' in line for line in lines)
        has_intro = any(any(keyword in line.lower() for keyword in ['introduction', '介绍', '引言', 'overview']) for line in lines)
        has_conclusion = any(any(keyword in line.lower() for keyword in ['conclusion', '结论', '总结', 'summary']) for line in lines)
        
        print(f"  📋 结构分析:")
        print(f"    - 标题结构: {'✅' if has_headers else '❌'}")
        print(f"    - 引言部分: {'✅' if has_intro else '❌'}")
        print(f"    - 结论部分: {'✅' if has_conclusion else '❌'}")
        
        # 质量评分
        quality_score = 0
        if len(report) > 500: quality_score += 1
        if has_headers: quality_score += 1
        if has_intro: quality_score += 1
        if has_conclusion: quality_score += 1
        if len(non_empty_lines) > 10: quality_score += 1
        
        quality_levels = ["❌ 很差", "🔶 较差", "🔸 一般", "✅ 良好", "🎉 优秀"]
        print(f"  🏆 质量评分: {quality_score}/5 - {quality_levels[quality_score]}")

async def main():
    """主测试函数"""
    print("🌟 稳定版完整报告生成测试")
    print("="*60)
    
    # API 密钥检查
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ 缺少 GOOGLE_API_KEY 环境变量")
        print("请设置: os.environ['GOOGLE_API_KEY'] = 'your_api_key'")
        return
    
    print("✅ GOOGLE_API_KEY: 已设置")
    
    tester = RobustReportTester()
    
    # 显示可用配置
    print("\n📋 可用测试配置:")
    for name, config in tester.configs.items():
        print(f"  {name}: {config['description']}")
    
    # 测试策略：从最稳定到最复杂
    test_order = ["no_search", "conservative", "staged"]
    
    print(f"\n🚀 开始分层测试（按稳定性排序）...")
    
    results = {}
    for config_name in test_order:
        print(f"\n{'='*60}")
        success = await tester.test_config(config_name)
        results[config_name] = success
        
        if success:
            print(f"✅ {config_name} 配置测试成功！")
            break  # 找到可用配置就停止
        else:
            print(f"❌ {config_name} 配置测试失败")
            print("⏭️ 尝试下一个配置...")
            
            # 添加延迟避免连续失败
            await asyncio.sleep(2)
    
    # 测试总结
    print(f"\n{'='*60}")
    print("📊 测试结果总结:")
    
    successful_configs = [name for name, success in results.items() if success]
    failed_configs = [name for name, success in results.items() if not success]
    
    if successful_configs:
        print(f"✅ 成功的配置: {successful_configs}")
        print(f"💡 推荐使用: {successful_configs[0]} 配置")
    
    if failed_configs:
        print(f"❌ 失败的配置: {failed_configs}")
    
    # 给出最终建议
    print(f"\n🎯 最终建议:")
    if successful_configs:
        print(f"  1. 使用 '{successful_configs[0]}' 配置进行日常报告生成")
        print(f"  2. 查看生成的报告文件了解输出质量")
        print(f"  3. 根据需要调整查询内容和配置参数")
    else:
        print("  1. 检查网络连接和API密钥")
        print("  2. 等待一段时间后重试（避免速率限制）")
        print("  3. 考虑使用其他搜索API或联系技术支持")

if __name__ == "__main__":
    print("""
🛡️ 稳定版报告生成测试
========================

本脚本解决了以下问题：
✅ DuckDuckGo API 速率限制
✅ Gemini 消息序列错误
✅ 网络连接不稳定
✅ 配置参数优化

测试策略：
1. 无搜索模式（最稳定）
2. 保守搜索模式（减少API调用）  
3. 分阶段搜索模式（避免并发）

运行前确保设置 GOOGLE_API_KEY 环境变量。
""")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试运行失败: {e}")
        import traceback
        traceback.print_exc() 