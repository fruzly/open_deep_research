#!/usr/bin/env python3
"""
增强的LLM提供商兼容性测试
验证与LangChain init_chat_model完全一致的提供商检测和兼容性
"""

import sys
sys.path.append('.')

from src.open_deep_research.message_manager import (
    UniversalMessageManager, 
    validate_and_fix_messages,
    LLMProvider
)

def test_langchain_compatibility():
    """测试与LangChain init_chat_model的完全兼容性"""
    print("🔗 LangChain兼容性测试")
    print("=" * 60)
    
    # 测试LangChain的模型名称推断规则
    langchain_test_cases = [
        # OpenAI系列
        ("gpt-3.5-turbo", LLMProvider.OPENAI, "GPT-3.5"),
        ("gpt-4o", LLMProvider.OPENAI, "GPT-4o"),
        ("o1-preview", LLMProvider.OPENAI, "o1 Preview"),
        ("openai:gpt-4", LLMProvider.OPENAI, "OpenAI GPT-4"),
        
        # Anthropic系列
        ("claude-3-sonnet", LLMProvider.ANTHROPIC, "Claude 3 Sonnet"),
        ("claude-3-5-haiku", LLMProvider.ANTHROPIC, "Claude 3.5 Haiku"),
        ("anthropic:claude-3-opus", LLMProvider.ANTHROPIC, "Anthropic Claude 3 Opus"),
        
        # Google系列
        ("gemini-pro", LLMProvider.GOOGLE_VERTEXAI, "Gemini Pro"),
        ("gemini-1.5-flash", LLMProvider.GOOGLE_VERTEXAI, "Gemini 1.5 Flash"),
        ("google_vertexai:gemini-2.0", LLMProvider.GOOGLE_VERTEXAI, "Google VertexAI Gemini"),
        ("google_genai:gemini-pro", LLMProvider.GOOGLE_GENAI, "Google GenAI Gemini"),
        
        # AWS Bedrock
        ("amazon.titan-text", LLMProvider.BEDROCK, "Amazon Titan"),
        ("bedrock:claude-3", LLMProvider.BEDROCK, "Bedrock Claude"),
        ("bedrock_converse:gpt-4", LLMProvider.BEDROCK_CONVERSE, "Bedrock Converse"),
        
        # Cohere
        ("command-r", LLMProvider.COHERE, "Command R"),
        ("command-r-plus", LLMProvider.COHERE, "Command R Plus"),
        ("cohere:command", LLMProvider.COHERE, "Cohere Command"),
        
        # Fireworks
        ("accounts/fireworks/models/llama", LLMProvider.FIREWORKS, "Fireworks Llama"),
        ("fireworks:llama-2", LLMProvider.FIREWORKS, "Fireworks Llama-2"),
        
        # MistralAI
        ("mistral-large", LLMProvider.MISTRALAI, "Mistral Large"),
        ("mistral-7b", LLMProvider.MISTRALAI, "Mistral 7B"),
        ("mistralai:mixtral", LLMProvider.MISTRALAI, "MistralAI Mixtral"),
        
        # DeepSeek
        ("deepseek-chat", LLMProvider.DEEPSEEK, "DeepSeek Chat"),
        ("deepseek-coder", LLMProvider.DEEPSEEK, "DeepSeek Coder"),
        ("deepseek:v2", LLMProvider.DEEPSEEK, "DeepSeek V2"),
        
        # XAI
        ("grok-beta", LLMProvider.XAI, "Grok Beta"),
        ("grok-1", LLMProvider.XAI, "Grok 1"),
        ("xai:grok", LLMProvider.XAI, "XAI Grok"),
        
        # Perplexity
        ("sonar-small", LLMProvider.PERPLEXITY, "Sonar Small"),
        ("sonar-medium", LLMProvider.PERPLEXITY, "Sonar Medium"),
        ("perplexity:sonar", LLMProvider.PERPLEXITY, "Perplexity Sonar"),
        
        # 其他提供商
        ("groq:llama-3", LLMProvider.GROQ, "Groq Llama"),
        ("ollama:llama2", LLMProvider.OLLAMA, "Ollama Llama2"),
        ("huggingface:bert", LLMProvider.HUGGINGFACE, "HuggingFace BERT"),
        ("together:llama", LLMProvider.TOGETHER, "Together Llama"),
        ("ibm:granite", LLMProvider.IBM, "IBM Granite"),
        ("nvidia:nemotron", LLMProvider.NVIDIA, "Nvidia Nemotron"),
        ("azure_ai:phi", LLMProvider.AZURE_AI, "Azure AI Phi"),
        ("azure_openai:gpt-4", LLMProvider.AZURE_OPENAI, "Azure OpenAI GPT-4"),
    ]
    
    print(f"\n📋 测试 {len(langchain_test_cases)} 个LangChain兼容案例")
    print("-" * 60)
    
    success_count = 0
    for model_name, expected_provider, description in langchain_test_cases:
        try:
            manager = UniversalMessageManager(model_name)
            detected_provider = manager.provider
            
            if detected_provider == expected_provider:
                status = "✅"
                success_count += 1
            else:
                status = "❌"
            
            print(f"{status} {description:<25} | {model_name:<30} -> {detected_provider.value:<15} (期望: {expected_provider.value})")
            
        except Exception as e:
            print(f"❌ {description:<25} | {model_name:<30} -> ERROR: {e}")
    
    compatibility_rate = (success_count / len(langchain_test_cases)) * 100
    print(f"\n📊 LangChain兼容性: {success_count}/{len(langchain_test_cases)} ({compatibility_rate:.1f}%)")
    
    return compatibility_rate

def test_enhanced_message_processing():
    """测试增强的消息处理能力"""
    print("\n🔧 增强消息处理测试")
    print("=" * 60)
    
    # 复杂的消息序列测试
    complex_messages = [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "assistant", "content": "I'll help with research."},
        {"role": "assistant", "content": "Let me search for information."},
        {"role": "assistant", "content": "I found some results."},
        {"role": "user", "content": "What did you find?"},
        {"role": "assistant", "content": "Here's what I found...", 
         "tool_calls": [{"id": "1", "name": "search", "args": {"q": "AI research"}}]},
        {"role": "assistant", "content": "Additional analysis..."},
        {"role": "tool", "content": "Search results: AI research papers...", "tool_call_id": "1"},
        {"role": "user", "content": "Please summarize."}
    ]
    
    # 测试所有新增的提供商
    new_providers = [
        "cohere:command-r",
        "fireworks:llama-2", 
        "mistralai:mixtral",
        "xai:grok-beta",
        "perplexity:sonar-medium",
        "bedrock:claude-3",
        "together:llama-3",
        "ibm:granite",
        "nvidia:nemotron",
        "azure_ai:phi-3"
    ]
    
    print(f"\n📝 测试 {len(new_providers)} 个新增提供商的消息处理")
    print("-" * 60)
    
    for provider_model in new_providers:
        try:
            fixed_messages, fixes = validate_and_fix_messages(complex_messages.copy(), provider_model)
            
            provider_name = provider_model.split(':')[0]
            print(f"✅ {provider_name:<12}: {len(complex_messages)} -> {len(fixed_messages)} 条消息, {len(fixes)} 个修复")
            
            if fixes:
                print(f"   修复: {', '.join(fixes[:2])}{'...' if len(fixes) > 2 else ''}")
                
        except Exception as e:
            print(f"❌ {provider_model:<20}: 错误 - {e}")

def test_real_world_scenarios():
    """测试真实世界场景"""
    print("\n🌍 真实世界场景测试")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "多模态AI研究工作流",
            "providers": ["openai:gpt-4o", "anthropic:claude-3-opus", "google_genai:gemini-pro"],
            "messages": [
                {"role": "user", "content": "分析这个图像中的AI模型架构"},
                {"role": "assistant", "content": "我来分析这个架构图...", 
                 "tool_calls": [{"id": "1", "name": "vision_analysis"}]},
                {"role": "tool", "content": "图像分析结果...", "tool_call_id": "1"},
                {"role": "assistant", "content": "基于分析结果..."},
                {"role": "assistant", "content": "让我搜索相关论文...", 
                 "tool_calls": [{"id": "2", "name": "search"}]},
                {"role": "tool", "content": "论文搜索结果...", "tool_call_id": "2"},
                {"role": "assistant", "content": "综合分析..."}
            ]
        },
        {
            "name": "代码生成与优化工作流", 
            "providers": ["deepseek:deepseek-coder", "mistralai:codestral", "cohere:command-r"],
            "messages": [
                {"role": "user", "content": "优化这段Python代码的性能"},
                {"role": "assistant", "content": "我来分析代码性能...", 
                 "tool_calls": [{"id": "1", "name": "code_analysis"}]},
                {"role": "assistant", "content": "发现了几个优化点..."},
                {"role": "tool", "content": "性能分析报告...", "tool_call_id": "1"},
                {"role": "assistant", "content": "建议的优化方案..."}
            ]
        },
        {
            "name": "企业级AI助手工作流",
            "providers": ["azure_openai:gpt-4", "bedrock:claude-3", "ibm:granite"],
            "messages": [
                {"role": "system", "content": "企业AI助手系统提示"},
                {"role": "user", "content": "生成季度业务报告"},
                {"role": "assistant", "content": "我来收集数据...", 
                 "tool_calls": [{"id": "1", "name": "database_query"}]},
                {"role": "assistant", "content": "分析业务指标..."},
                {"role": "tool", "content": "数据查询结果...", "tool_call_id": "1"},
                {"role": "assistant", "content": "生成报告..."}
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}")
        print("-" * 40)
        
        for provider in scenario['providers']:
            try:
                fixed, fixes = validate_and_fix_messages(scenario['messages'], provider)
                provider_short = provider.split(':')[0]
                print(f"  {provider_short:<15}: {len(scenario['messages'])} -> {len(fixed)} 条, {len(fixes)} 修复")
            except Exception as e:
                print(f"  {provider:<15}: ❌ {e}")

def main():
    """主测试函数"""
    print("🚀 增强的LLM兼容性测试")
    print("🔗 验证与LangChain init_chat_model的完全兼容性")
    print("=" * 80)
    
    # 运行所有测试
    langchain_compatibility = test_langchain_compatibility()
    test_enhanced_message_processing()
    test_real_world_scenarios()
    
    print("\n" + "=" * 80)
    print("🎉 测试完成!")
    print(f"✅ LangChain兼容性: {langchain_compatibility:.1f}%")
    
    if langchain_compatibility >= 95:
        print("🏆 优秀！与LangChain高度兼容")
    elif langchain_compatibility >= 85:
        print("👍 良好！基本兼容LangChain")
    else:
        print("⚠️ 需要改进兼容性")
    
    print("\n📋 支持的提供商总数:", len(LLMProvider) - 1)  # 减去UNKNOWN
    print("🔗 完全遵循LangChain init_chat_model推断规则")

if __name__ == "__main__":
    main() 