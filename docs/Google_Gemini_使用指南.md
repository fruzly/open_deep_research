# 在 open_deep_research 项目中使用 Google Gemini 模型

## 📋 前置条件

1. 确保已安装项目依赖（项目已包含 Google AI 支持）
2. 获取 Google AI API 密钥

## 🔑 API 密钥设置

### 获取 API 密钥

1. 访问 [Google AI Studio](https://makersuite.google.com/)
2. 登录 Google 账号
3. 创建新的 API 密钥
4. 复制生成的 API 密钥

### 环境变量配置

在项目根目录创建 `.env` 文件（如果不存在），并添加：

```bash
# Google AI API 密钥
GOOGLE_API_KEY=your_google_api_key_here

# 可选：如果使用 Vertex AI
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### PowerShell 中设置环境变量

```powershell
# 设置 Google AI API 密钥
$env:GOOGLE_API_KEY="your_google_api_key_here"

# 验证设置
echo $env:GOOGLE_API_KEY
```

## 🚀 使用方法

### 1. Graph-based 工作流模式

在 Jupyter Notebook 或 Python 脚本中：

```python
import uuid
from open_deep_research.graph import builder
from langgraph.checkpoint.memory import MemorySaver

# 编译图
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 配置 Gemini 模型
thread = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
        "search_api": "tavily",  # 搜索API
        
        # 使用 Gemini 作为规划器
        "planner_provider": "google_genai",
        "planner_model": "gemini-2.0-flash",
        
        # 使用 Gemini 作为写作器
        "writer_provider": "google_genai", 
        "writer_model": "gemini-2.0-flash",
        
        # 可选：摘要模型也使用 Gemini
        "summarization_model_provider": "google_genai",
        "summarization_model": "gemini-2.0-flash",
        
        "max_search_depth": 2,
    }
}

# 执行研究任务
topic = "AI模型在自然语言处理中的最新发展"

async for event in graph.astream({"topic": topic}, thread, stream_mode="updates"):
    if '__interrupt__' in event:
        interrupt_value = event['__interrupt__'][0].value
        print(interrupt_value)
```

### 2. Multi-Agent 模式

```python
import uuid
from open_deep_research.multi_agent import graph

# 配置多智能体系统
config = {
    "thread_id": str(uuid.uuid4()),
    "search_api": "tavily",
    
    # Supervisor 使用 Gemini
    "supervisor_model": "google_genai:gemini-2.0-flash",
    
    # Researcher 使用 Gemini  
    "researcher_model": "google_genai:gemini-2.0-flash",
    
    # 可选配置
    "ask_for_clarification": True,
}

thread_config = {"configurable": config}

# 执行研究
msg = [{"role": "user", "content": "请研究量子计算的最新进展"}]
response = await graph.ainvoke({"messages": msg}, config=thread_config)
```

## 🎯 可用的 Gemini 模型

### Google AI (google_genai provider)

```python
# 最新的 Gemini 2.0 模型
"planner_provider": "google_genai"
"planner_model": "gemini-2.0-flash"

# Gemini 1.5 系列
"planner_model": "gemini-1.5-pro"
"planner_model": "gemini-1.5-flash" 
"planner_model": "gemini-1.5-flash-8b"

# Gemini 1.0 系列（较老版本）
"planner_model": "gemini-pro"
"planner_model": "gemini-pro-vision"
```

### Vertex AI (google-vertexai provider)

```python
# 使用 Vertex AI
"planner_provider": "google-vertexai"
"planner_model": "gemini-2.0-flash"

# 需要额外配置
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

## ⚙️ 高级配置

### 模型参数调优

```python
thread = {
    "configurable": {
        # ... 其他配置 ...
        
        # 自定义模型参数
        "planner_model_kwargs": {
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.9,
        },
        
        "writer_model_kwargs": {
            "temperature": 0.8,
            "max_tokens": 8192,
        }
    }
}
```

### 混合模型策略

```python
# 规划用 Gemini，写作用其他模型
thread = {
    "configurable": {
        "planner_provider": "google_genai",
        "planner_model": "gemini-2.0-flash",
        
        "writer_provider": "anthropic", 
        "writer_model": "claude-3-5-sonnet-latest",
        
        "summarization_model_provider": "google_genai",
        "summarization_model": "gemini-1.5-flash",
    }
}
```

## 🔍 实用示例

### 示例 1：基础研究报告

```python
import os
import asyncio
from open_deep_research.graph import builder
from langgraph.checkpoint.memory import MemorySaver

# 设置 API 密钥
os.environ["GOOGLE_API_KEY"] = "your_api_key"
os.environ["TAVILY_API_KEY"] = "your_tavily_key"

async def generate_report():
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    config = {
        "configurable": {
            "thread_id": "gemini-test-1",
            "search_api": "tavily",
            "planner_provider": "google_genai",
            "planner_model": "gemini-2.0-flash",
            "writer_provider": "google_genai",
            "writer_model": "gemini-1.5-pro",
            "max_search_depth": 2,
        }
    }
    
    topic = "人工智能在医疗诊断中的应用现状与发展趋势"
    
    async for event in graph.astream({"topic": topic}, config):
        print(f"Event: {event}")
    
    # 获取最终报告
    final_state = graph.get_state(config)
    if final_state.values.get('final_report'):
        print("=== 最终报告 ===")
        print(final_state.values['final_report'])

# 运行
asyncio.run(generate_report())
```

### 示例 2：多智能体协作

```python
from open_deep_research.multi_agent import graph

async def multi_agent_research():
    config = {
        "configurable": {
            "thread_id": "gemini-multi-1",
            "search_api": "googlesearch",
            "supervisor_model": "google_genai:gemini-2.0-flash",
            "researcher_model": "google_genai:gemini-1.5-pro",
        }
    }
    
    messages = [{
        "role": "user", 
        "content": "请研究并分析区块链技术在供应链管理中的应用案例"
    }]
    
    result = await graph.ainvoke({"messages": messages}, config=config)
    print(result["final_report"])

asyncio.run(multi_agent_research())
```

## 📊 性能优化建议

### 1. 模型选择建议

- **规划任务**: `gemini-2.0-flash` - 速度快，推理能力强
- **长文本写作**: `gemini-1.5-pro` - 更好的文本生成质量
- **摘要任务**: `gemini-1.5-flash` - 高效且经济

### 2. 成本优化

```python
# 成本敏感的配置
config = {
    "planner_provider": "google_genai",
    "planner_model": "gemini-1.5-flash",  # 更经济的选择
    
    "writer_provider": "google_genai", 
    "writer_model": "gemini-1.5-flash",   # 平衡性能和成本
    
    "max_search_depth": 1,  # 减少搜索迭代
}
```

### 3. 错误处理

```python
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

try:
    result = await graph.ainvoke({"messages": messages}, config=config)
except Exception as e:
    logging.error(f"Gemini API 错误: {e}")
    # 回退到其他模型
    config["configurable"]["planner_provider"] = "anthropic"
    config["configurable"]["planner_model"] = "claude-3-haiku-20240307"
```

## 🚨 注意事项

1. **API 限制**: Gemini API 有请求速率限制，高频使用时注意控制并发
2. **Token 限制**: 不同模型有不同的上下文长度限制
3. **地区限制**: 某些地区可能无法访问 Gemini API
4. **成本控制**: 监控 API 使用量，避免意外高额费用

## 🔧 故障排除

### 常见问题

1. **模型提供商推断错误**
   
   **错误信息**: `Unable to infer model provider for model='google_genai:gemini-1.5-flash'`
   
   **原因**: 在多智能体模式中使用了错误的 provider 名称格式
   
   **解决方案**:
   ```python
   # ❌ 错误写法
   config = {
       "configurable": {
           "supervisor_model": "google_genai:gemini-1.5-flash"
       }
   }
   
   # ✅ 正确写法
   config = {
       "configurable": {
           "supervisor_model": "google_genai:gemini-1.5-flash"  # 注意下划线
       }
   }
   ```

2. **认证错误**
   ```python
   # 确保 API 密钥正确设置
   import os
   print(os.environ.get("GOOGLE_API_KEY"))
   ```

3. **模型不可用**
   ```python
   # 检查模型名称是否正确
   from langchain.chat_models import init_chat_model
   
   try:
       model = init_chat_model(
           model="gemini-2.0-flash",
           model_provider="google_genai"
       )
       print("模型初始化成功")
   except Exception as e:
       print(f"错误: {e}")
   ```

4. **配置格式差异**
   
   不同模式下的配置格式略有不同：
   ```python
   # Graph 模式 - 使用分离的 provider 和 model 字段
   graph_config = {
       "configurable": {
           "planner_provider": "google_genai",
           "planner_model": "gemini-1.5-flash",
       }
   }
   
   # Multi-Agent 模式 - 使用组合的模型字符串
   multi_agent_config = {
       "configurable": {
           "supervisor_model": "google_genai:gemini-1.5-flash",  # 注意 provider 名称
           "researcher_model": "google_genai:gemini-1.5-flash",
       }
   }
   ```

5. **网络连接问题**
   - 确保网络可以访问 Google AI API
   - 如在中国大陆，可能需要代理

### 获取更多帮助

- [Google AI 官方文档](https://ai.google.dev/)
- [LangChain Google 集成文档](https://python.langchain.com/docs/integrations/chat/google_generative_ai)
- [项目 GitHub Issues](https://github.com/langchain-ai/open_deep_research/issues) 