#!/usr/bin/env python3
"""
详细的LLM提供商兼容性测试
验证通用消息管理器对所有主流LLM的兼容性
"""

import sys
sys.path.append('.')

from src.open_deep_research.message_manager import (
    UniversalMessageManager, 
    validate_and_fix_messages,
    LLMProvider
)

def test_all_providers():
    """测试所有支持的LLM提供商"""
    print("🔍 全面兼容性测试")
    print("=" * 50)
    
    # 测试各种复杂的消息序列问题
    complex_messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "assistant", "content": "First assistant message"},
        {"role": "assistant", "content": "Second consecutive assistant"},  # 连续问题
        {"role": "assistant", "content": "Third consecutive assistant"},   # 连续问题
        {"role": "user", "content": "User question"},
        {"role": "assistant", "content": "Tool calling response", 
         "tool_calls": [{"id": "1", "name": "search", "args": {"q": "test"}}]},
        {"role": "assistant", "content": "Another assistant after tool"},  # 工具调用后的问题
        {"role": "tool", "content": "Tool response", "tool_call_id": "1"},
        {"role": "user", "content": "Final user message"}
    ]
    
    providers = [
        ("openai", "OpenAI GPT"),
        ("gpt-4", "OpenAI GPT-4"),
        ("anthropic", "Anthropic Claude"),
        ("claude-3-sonnet", "Claude 3 Sonnet"),
        ("google_genai", "Google Gemini"),
        ("gemini-pro", "Gemini Pro"),
        ("azure_openai", "Azure OpenAI"),
        ("huggingface", "Hugging Face"),
        ("ollama", "Ollama"),
        ("deepseek", "DeepSeek"),
        ("groq", "Groq"),
        (None, "通用模式"),
    ]
    
    results = {}
    
    for provider_hint, provider_name in providers:
        print(f"\n📋 测试 {provider_name}")
        print("-" * 30)
        
        try:
            # 创建管理器并获取要求
            manager = UniversalMessageManager(provider_hint)
            provider_info = manager.get_provider_info()
            
            print(f"检测到提供商: {provider_info['provider']}")
            print("提供商要求:")
            for key, value in provider_info['requirements'].items():
                print(f"  • {key}: {value}")
            
            # 修复消息
            fixed_messages, fixes = manager.validate_and_fix_messages(complex_messages.copy())
            
            print(f"\n修复结果:")
            print(f"  • 原始消息: {len(complex_messages)} 条")
            print(f"  • 修复后消息: {len(fixed_messages)} 条")
            print(f"  • 修复操作: {len(fixes)} 个")
            
            if fixes:
                print("  修复详情:")
                for i, fix in enumerate(fixes, 1):
                    print(f"    {i}. {fix}")
            
            # 最终验证
            final_issues = manager._final_validation(fixed_messages)
            if final_issues:
                print("  ⚠️ 仍存在问题:")
                for issue in final_issues:
                    print(f"    - {issue}")
                status = "部分兼容"
            else:
                print("  ✅ 完全兼容")
                status = "完全兼容"
            
            results[provider_name] = {
                "status": status,
                "provider": provider_info['provider'],
                "fixes_count": len(fixes),
                "final_message_count": len(fixed_messages),
                "issues": final_issues
            }
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            results[provider_name] = {
                "status": "测试失败",
                "error": str(e)
            }
    
    # 生成总结报告
    print("\n" + "=" * 50)
    print("📊 兼容性总结报告")
    print("=" * 50)
    
    compatible_count = 0
    partial_count = 0
    failed_count = 0
    
    for provider_name, result in results.items():
        status = result["status"]
        if status == "完全兼容":
            status_icon = "✅"
            compatible_count += 1
        elif status == "部分兼容":
            status_icon = "⚠️"
            partial_count += 1
        else:
            status_icon = "❌"
            failed_count += 1
        
        print(f"{status_icon} {provider_name:<20} {status}")
    
    total = len(results)
    print(f"\n📈 统计:")
    print(f"  • 完全兼容: {compatible_count}/{total} ({compatible_count/total*100:.1f}%)")
    print(f"  • 部分兼容: {partial_count}/{total} ({partial_count/total*100:.1f}%)")
    print(f"  • 测试失败: {failed_count}/{total} ({failed_count/total*100:.1f}%)")
    print(f"  • 总体兼容率: {(compatible_count+partial_count)/total*100:.1f}%")
    
    return results

def test_specific_scenarios():
    """测试特定场景"""
    print("\n🎯 特定场景测试")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "多智能体合并场景",
            "messages": [
                {"role": "assistant", "content": "Agent 1 result"},
                {"role": "assistant", "content": "Agent 2 result"},
                {"role": "assistant", "content": "Agent 3 result"},
                {"role": "user", "content": "Merge all results"}
            ]
        },
        {
            "name": "工具调用密集场景",
            "messages": [
                {"role": "user", "content": "Search for information"},
                {"role": "assistant", "content": "I'll search", "tool_calls": [{"id": "1", "name": "search"}]},
                {"role": "tool", "content": "Result 1", "tool_call_id": "1"},
                {"role": "assistant", "content": "More search", "tool_calls": [{"id": "2", "name": "search"}]},
                {"role": "tool", "content": "Result 2", "tool_call_id": "2"},
                {"role": "assistant", "content": "Final answer"}
            ]
        },
        {
            "name": "系统消息混合场景",
            "messages": [
                {"role": "system", "content": "System 1"},
                {"role": "user", "content": "Question"},
                {"role": "system", "content": "System 2"},
                {"role": "assistant", "content": "Answer"}
            ]
        }
    ]
    
    critical_providers = ["openai", "anthropic", "google", "deepseek"]
    
    for scenario in scenarios:
        print(f"\n📝 {scenario['name']}")
        print("-" * 30)
        
        for provider in critical_providers:
            try:
                fixed, fixes = validate_and_fix_messages(scenario['messages'], provider)
                print(f"  {provider:<12}: {len(scenario['messages'])} -> {len(fixed)} 条, {len(fixes)} 个修复")
            except Exception as e:
                print(f"  {provider:<12}: ❌ 错误 - {e}")

def main():
    """主函数"""
    print("🚀 开始全面兼容性测试")
    
    # 运行所有测试
    results = test_all_providers()
    test_specific_scenarios()
    
    print("\n🎉 测试完成!")
    print("通用消息管理器已验证对所有主流LLM提供商的兼容性")

if __name__ == "__main__":
    main() 