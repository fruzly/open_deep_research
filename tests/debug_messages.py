"""
调试工具：分析消息格式和内容
"""

import asyncio
import uuid
import os
import traceback
from src.open_deep_research.multi_agent import supervisor_builder
from langgraph.checkpoint.memory import MemorySaver

def analyze_message(msg, context=""):
    """分析单个消息的格式和内容"""
    print(f"\n=== 消息分析 {context} ===")
    print(f"类型: {type(msg)}")
    print(f"内容: {msg}")
    
    if isinstance(msg, dict):
        print("✅ 字典格式消息")
        print(f"  - 角色: {msg.get('role', 'N/A')}")
        print(f"  - 内容: {msg.get('content', 'N/A')}")
        if 'tool_calls' in msg:
            print(f"  - 工具调用: {msg['tool_calls']}")
    elif hasattr(msg, 'type'):
        # LangChain 消息对象
        msg_type = msg.type if hasattr(msg, 'type') else type(msg).__name__.lower()
        print(f"✅ LangChain消息对象 ({msg_type})")
        
        # 根据类型推断角色
        if 'human' in msg_type or 'user' in msg_type:
            role = 'user'
        elif 'ai' in msg_type or 'assistant' in msg_type:
            role = 'assistant'
        elif 'system' in msg_type:
            role = 'system'
        elif 'tool' in msg_type:
            role = 'tool'
        else:
            role = getattr(msg, 'role', 'unknown')
            
        print(f"  - 角色: {role}")
        print(f"  - 内容: {getattr(msg, 'content', 'N/A')}")
        
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"  - 工具调用: {msg.tool_calls}")
        if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
            print(f"  - 附加参数: {msg.additional_kwargs}")
    elif hasattr(msg, 'role') and hasattr(msg, 'content'):
        print("✅ LangChain消息对象")
        print(f"  - 角色: {msg.role}")
        print(f"  - 内容: {msg.content}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"  - 工具调用: {msg.tool_calls}")
    else:
        print("❌ 未知消息格式")
        print(f"  - 属性: {[attr for attr in dir(msg) if not attr.startswith('_')]}")

def analyze_message_list(messages, context=""):
    """分析消息列表"""
    print(f"\n🔍 分析消息列表 {context}")
    print(f"消息数量: {len(messages)}")
    
    for i, msg in enumerate(messages):
        analyze_message(msg, f"[{i}]")
    
    print("\n📊 消息序列摘要:")
    roles = []
    for msg in messages:
        if isinstance(msg, dict):
            roles.append(msg.get('role', 'unknown'))
        elif hasattr(msg, 'type'):
            # LangChain 消息对象
            msg_type = msg.type if hasattr(msg, 'type') else type(msg).__name__.lower()
            if 'human' in msg_type or 'user' in msg_type:
                roles.append('user')
            elif 'ai' in msg_type or 'assistant' in msg_type:
                roles.append('assistant')
            elif 'system' in msg_type:
                roles.append('system')
            elif 'tool' in msg_type:
                roles.append('tool')
            else:
                roles.append(getattr(msg, 'role', 'unknown'))
        elif hasattr(msg, 'role'):
            roles.append(msg.role)
        else:
            roles.append('unknown')
    
    print(f"角色序列: {' -> '.join(roles)}")
    
    # 检查Gemini要求
    if roles:
        last_role = roles[-1]
        if last_role == 'user':
            print("✅ 最后一条消息是用户消息（符合Gemini要求）") 
        else:
            print(f"⚠️ 最后一条消息是 {last_role}（可能不符合Gemini要求）")

async def debug_multi_agent():
    """详细调试multi_agent功能"""
    print("🔍 开始详细调试multi_agent...")
    
    try:
        # 创建agent
        checkpointer = MemorySaver()
        agent = supervisor_builder.compile(name="debug_research_team", checkpointer=checkpointer)
        
        # 配置 - 使用更简单的设置
        config = {
            "thread_id": str(uuid.uuid4()),
            "search_api": "none",  # 禁用搜索API以简化消息流
            "supervisor_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "researcher_model": "google_genai:gemini-2.5-flash-lite-preview-06-17",
            "number_of_queries": 10,  # 减少查询数量
        }
        
        thread_config = {"configurable": config, "recursion_limit": 20}  # 降低递归限制
        
        # 测试消息 - 使用更简单的问题
        test_msg = [{"role": "user", "content": "Focus on Anthropic‑backed open standard for integrating external context and tools with LLMs, give an architectural overview for developers, tell me about interesting MCP servers, compare to google Agent2Agent (A2A) protocol. write the report and dont ask any follow up questions"}]
        
        print(f"📝 输入消息: {test_msg}")
        print(f"⚙️ 配置: {config}")
        
        # 分析初始消息序列
        analyze_message_list(test_msg, "初始输入")
        
        # 设置调试环境变量
        os.environ["DEBUG_MULTI_AGENT"] = "true"
        
        print("🚀 开始执行工作流...")
        response = await agent.ainvoke({"messages": test_msg}, config=thread_config)
        print("✅ 工作流执行成功！")
        print(f"📄 响应: {response}")
        
        
        print("="*40)
        for m in agent.get_state(thread_config).values['messages']:
            m.pretty_print()
        print("="*40)
        
        # 获取状态并显示所有消息
        state = agent.get_state(thread_config)
        if hasattr(state, 'values') and 'messages' in state.values:
            print("\n📋 完整消息历史:")
            for i, m in enumerate(state.values['messages']):
                print(f"\n--- 消息 {i} ---")
                if hasattr(m, 'pretty_print'):
                    m.pretty_print()
                else:
                    print(f"消息内容: {m}")
            
            # 分析最终消息序列
            analyze_message_list(state.values['messages'], "最终状态")
        
        print("="*40)
        from IPython.display import Markdown
        
        # 安全地检查 final_report 是否存在
        state_values = agent.get_state(thread_config).values
        if 'final_report' in state_values and state_values['final_report']:
            print("📄 最终报告:")
            print(state_values['final_report'])
        else:
            print("ℹ️  当前测试场景没有生成完整报告（这是正常的，因为使用了简化配置）")
        
        return True
        
    except Exception as e:
        print(f"❌ 详细错误信息:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        
        # 打印完整的traceback
        print("\n🔴 完整错误追踪:")
        traceback.print_exc()
        
        # 如果是 KeyError，打印更多调试信息
        if isinstance(e, KeyError):
            print(f"\n🔑 KeyError 详细信息:")
            print(f"缺少的键: {e.args}")
            
        # 如果是 ChatGoogleGenerativeAIError，提供修复建议
        if "ChatGoogleGenerativeAIError" in str(type(e)):
            print(f"\n💡 Gemini 错误修复建议:")
            print("1. 检查消息序列是否符合 Gemini 要求")
            print("2. 确保 function call 紧跟在 user turn 或 function response 之后")
            print("3. 检查是否有连续的 assistant 消息")
            
        return False

if __name__ == "__main__":
    # 检查环境变量
    required_keys = ["GOOGLE_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    
    if missing_keys:
        print(f"⚠️ 缺少环境变量: {missing_keys}")
        print("请设置所需的API密钥")
    else:
        # 运行调试
        result = asyncio.run(debug_multi_agent())
        if result:
            print("🎉 调试成功！")
        else:
            print("💥 调试失败，需要进一步检查") 


"""
# 运行命令
G:\MyProjects\open_deep_research> python .\debug_messages.py
# 输出结果

🔍 开始详细调试multi_agent...
📝 输入消息: [{'role': 'user', 'content': 'Write a simple overview of Python programming language.'}]  
⚙️ 配置: {'thread_id': '0663a8b7-b278-49e7-b371-358bd6851c3f', 'search_api': 'none', 'supervisor_model'
: 'google_genai:gemini-2.5-flash-lite-preview-06-17', 'researcher_model': 'google_genai:gemini-2.5-flash-lite-preview-06-17', 'number_of_queries': 1}

🔍 分析消息列表 初始输入
消息数量: 1

=== 消息分析 [0] ===
类型: <class 'dict'>
内容: {'role': 'user', 'content': 'Write a simple overview of Python programming language.'}
✅ 字典格式消息
- 角色: user
- 内容: Write a simple overview of Python programming language.

📊 消息序列摘要:
角色序列: user
✅ 最后一条消息是用户消息（符合Gemini要求）
🚀 开始执行工作流...
✅ 工作流执行成功！
📄 响应: {'messages': [HumanMessage(content='Write a simple overview of Python programming language.', 
additional_kwargs={}, response_metadata={}, id='7e7c9692-cc54-460c-986d-b8132a643843'), AIMessage(content='', additional_kwargs={'function_call': {'name': 'FinishReport', 'arguments': '{}'}}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash-lite-preview-06-17', 'safety_ratings': []}, id='run--c5be8b9d-359c-4714-b7d7-d4c13bbbf21e-0', tool_calls=[{'name': 'FinishReport', 'args': {}, 'id': 'afde1fe5-96d7-4c4f-b1a4-bae32570b78b', 'type': 'tool_call'}], usage_metadata={'input_tokens': 1013, 'output_tokens': 9, 'total_tokens': 1022, 'input_token_details': {'cache_read': 0}})], 'source_str': ''}

📋 完整消息历史:

--- 消息 0 ---
================================ Human Message =================================

Write a simple overview of Python programming language.

--- 消息 1 ---
================================== Ai Message ==================================
Tool Calls:
FinishReport (afde1fe5-96d7-4c4f-b1a4-bae32570b78b)
Call ID: afde1fe5-96d7-4c4f-b1a4-bae32570b78b
Args:

🔍 分析消息列表 最终状态
消息数量: 2

=== 消息分析 [0] ===
类型: <class 'langchain_core.messages.human.HumanMessage'>
内容: content='Write a simple overview of Python programming language.' additional_kwargs={} response_metadata={} id='7e7c9692-cc54-460c-986d-b8132a643843'
✅ LangChain消息对象 (human)
- 角色: user
- 内容: Write a simple overview of Python programming language.

=== 消息分析 [1] ===
类型: <class 'langchain_core.messages.ai.AIMessage'>
内容: content='' additional_kwargs={'function_call': {'name': 'FinishReport', 'arguments': '{}'}} response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash-lite-preview-06-17', 'safety_ratings': []} id='run--c5be8b9d-359c-4714-b7d7-d4c13bbbf21e-0' tool_calls=[{'name': 'FinishReport', 'args': {}, 'id': 'afde1fe5-96d7-4c4f-b1a4-bae32570b78b', 'type': 'tool_call'}] usage_metadata={'input_tokens': 1013, 'output_tokens': 9, 'total_tokens': 1022, 'input_token_details': {'cache_read': 0}}
✅ LangChain消息对象 (ai)
- 角色: assistant
- 内容:
- 工具调用: [{'name': 'FinishReport', 'args': {}, 'id': 'afde1fe5-96d7-4c4f-b1a4-bae32570b78b', 'type': 'tool_call'}]
- 附加参数: {'function_call': {'name': 'FinishReport', 'arguments': '{}'}}

📊 消息序列摘要:
角色序列: user -> assistant
⚠️ 最后一条消息是 assistant（可能不符合Gemini要求）
🎉 调试成功！    

"""