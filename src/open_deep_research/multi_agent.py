from typing import List, Annotated, TypedDict, Literal, cast
from pydantic import BaseModel, Field
import operator
import warnings
import asyncio
import json
import logging
import threading
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import MessagesState

from langgraph.types import Command, Send
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from open_deep_research.configuration import MultiAgentConfiguration
from open_deep_research.utils import (
    get_config_value,
    tavily_search,
    duckduckgo_search,
    gemini_google_search,
    azureaisearch_search,
    deduplicate_and_format_sources,
    get_today_str,
    intelligent_search_web_unified,
)
from open_deep_research.message_manager import validate_and_fix_messages
from open_deep_research.message_converter import convert_langchain_messages_to_dict

from open_deep_research.prompts import SUPERVISOR_INSTRUCTIONS, RESEARCH_INSTRUCTIONS

import structlog
logger = structlog.get_logger(__name__)

# Monkey patch for debugging Gemini API calls
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Store the original _generate method
    if hasattr(ChatGoogleGenerativeAI, '_generate'):
        _original_generate = ChatGoogleGenerativeAI._generate
        
        def _patched_generate(self, messages, stop=None, run_manager=None, **kwargs):
            """Patched _generate method to log message contents before sending to Gemini"""
            logger.info("[GeminiPatch] Intercepting _generate call")
            
            # Log each message
            for i, msg in enumerate(messages):
                logger.info(f"[GeminiPatch] Message {i}: type={type(msg).__name__}")
                if hasattr(msg, 'content'):
                    content_length = len(str(msg.content)) if msg.content is not None else 0
                    logger.info(f"[GeminiPatch] Message {i} content length: {content_length}")
                    if content_length == 0:
                        logger.error(f"[GeminiPatch] WARNING: Message {i} has empty content!")
                
            # Call the original method
            try:
                return _original_generate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                logger.error(f"[GeminiPatch] Error in _generate: {str(e)}")
                # Log the exact message that caused the error
                if "contents[" in str(e):
                    import re
                    match = re.search(r'contents\[(\d+)\]', str(e))
                    if match:
                        idx = int(match.group(1))
                        if 0 <= idx < len(messages):
                            logger.error(f"[GeminiPatch] Problem message at index {idx}: {messages[idx]}")
                raise
        
        # Apply the patch
        ChatGoogleGenerativeAI._generate = _patched_generate
        logger.info("[GeminiPatch] Successfully patched ChatGoogleGenerativeAI._generate")
        
    # Also patch the async version if it exists
    if hasattr(ChatGoogleGenerativeAI, '_agenerate'):
        _original_agenerate = ChatGoogleGenerativeAI._agenerate
        
        async def _patched_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            """Patched _agenerate method to log message contents before sending to Gemini"""
            logger.info("[GeminiPatch] Intercepting _agenerate call")
            
            # Log each message
            for i, msg in enumerate(messages):
                logger.info(f"[GeminiPatch] Message {i}: type={type(msg).__name__}")
                if hasattr(msg, 'content'):
                    content_length = len(str(msg.content)) if msg.content is not None else 0
                    logger.info(f"[GeminiPatch] Message {i} content length: {content_length}")
                    if content_length == 0:
                        logger.error(f"[GeminiPatch] WARNING: Message {i} has empty content!")
                
            # Call the original method
            try:
                return await _original_agenerate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                logger.error(f"[GeminiPatch] Error in _agenerate: {str(e)}")
                # Log the exact message that caused the error
                if "contents[" in str(e):
                    import re
                    match = re.search(r'contents\[(\d+)\]', str(e))
                    if match:
                        idx = int(match.group(1))
                        if 0 <= idx < len(messages):
                            logger.error(f"[GeminiPatch] Problem message at index {idx}: {messages[idx]}")
                raise
        
        # Apply the patch
        ChatGoogleGenerativeAI._agenerate = _patched_agenerate
        logger.info("[GeminiPatch] Successfully patched ChatGoogleGenerativeAI._agenerate")
        
except ImportError:
    pass  # langchain_google_genai not installed
except Exception as e:
    logger.warning(f"[GeminiPatch] Failed to apply patch: {e}")

# Global tool manager for caching initialized tools
class GlobalToolManager:
    """全局工具管理器，用于缓存已初始化的工具，避免重复初始化"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._supervisor_tools_cache = {}
            self._research_tools_cache = {}
            self._cache_lock = threading.Lock()
            self._initialized = True
    
    def _get_cache_key(self, config: RunnableConfig) -> str:
        """生成配置的缓存键"""
        configurable = MultiAgentConfiguration.from_runnable_config(config)
        
        # 使用关键配置参数生成缓存键
        key_parts = [
            f"supervisor_model:{get_config_value(configurable.supervisor_model)}",
            f"researcher_model:{get_config_value(configurable.researcher_model)}",
            f"search_api:{get_config_value(configurable.search_api)}",
        ]
        return "|".join(key_parts)
    
    async def get_supervisor_tools(self, config: RunnableConfig) -> list[BaseTool]:
        """获取supervisor工具，使用缓存机制"""
        cache_key = self._get_cache_key(config)
        
        with self._cache_lock:
            if cache_key in self._supervisor_tools_cache:
                logger.info(f"[ToolManager] Using cached supervisor tools for key: {cache_key[:50]}...")
                return self._supervisor_tools_cache[cache_key]
        
        # 不在缓存中，需要初始化
        logger.info(f"[ToolManager] Initializing supervisor tools for key: {cache_key[:50]}...")
        tools = await get_supervisor_tools_impl(config)
        
        with self._cache_lock:
            self._supervisor_tools_cache[cache_key] = tools
        
        logger.info(f"[ToolManager] Cached supervisor tools ({len(tools)} tools)")
        return tools
    
    async def get_research_tools(self, config: RunnableConfig) -> list[BaseTool]:
        """获取research工具，使用缓存机制"""
        cache_key = self._get_cache_key(config)
        
        with self._cache_lock:
            if cache_key in self._research_tools_cache:
                logger.info(f"[ToolManager] Using cached research tools for key: {cache_key[:50]}...")
                return self._research_tools_cache[cache_key]
        
        # 不在缓存中，需要初始化
        logger.info(f"[ToolManager] Initializing research tools for key: {cache_key[:50]}...")
        tools = await get_research_tools_impl(config)
        
        with self._cache_lock:
            self._research_tools_cache[cache_key] = tools
        
        logger.info(f"[ToolManager] Cached research tools ({len(tools)} tools)")
        return tools
    
    def clear_cache(self):
        """清空工具缓存"""
        with self._cache_lock:
            self._supervisor_tools_cache.clear()
            self._research_tools_cache.clear()
        logger.info("[ToolManager] Tool cache cleared")

# 全局工具管理器实例
tool_manager = GlobalToolManager()

def _infer_message_role(msg) -> str:
    """Infer message role, supporting both dictionary format and LangChain message objects"""
    if isinstance(msg, dict):
        return msg.get("role", "unknown")
    
    # Try to get role attribute directly
    if hasattr(msg, "role"):
        return msg.role
    
    # Infer role based on type name
    msg_type = type(msg).__name__.lower()
    role_mapping = {
        "human": "user",
        "user": "user", 
        "ai": "assistant",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool"
    }
    
    for key, role in role_mapping.items():
        if key in msg_type:
            return role
    
    return "unknown"


def _fix_gemini_message_sequence(messages) -> list:
    """
    Fix message sequence to comply with Gemini API requirements
    
    Gemini requirements:
    - function call must follow immediately after user turn or function response turn
    - cannot have consecutive assistant messages
    - the last message should preferably be a user message
    
    Args:
        messages: Message list, supporting both dictionary format and LangChain message objects
        
    Returns:
        Fixed message list
    """
    if not messages:
        return messages
        
    fixed_messages = []
    last_role = None
    
    for msg in messages:
        current_role = _infer_message_role(msg)
        
        # Check if there are consecutive assistant messages
        if current_role == "assistant" and last_role == "assistant":
            # Insert separator user message
            fixed_messages.append({
                "role": "user", 
                "content": "Continue with the next step."
            })
        
        fixed_messages.append(msg)
        last_role = current_role
        
    return fixed_messages


def _ensure_user_message_ending(messages, default_content: str = "Please continue.") -> list:
    """
    Ensure message sequence ends with user message (Gemini recommendation)
    
    Args:
        messages: Message list
        default_content: Default user message content
        
    Returns:
        Message list ensuring user message ending
    """
    if not messages:
        return messages
    
    last_msg = messages[-1]
    last_role = _infer_message_role(last_msg)
    
    if last_role != "user":
        messages = messages.copy()  # Avoid modifying original list
        messages.append({
            "role": "user", 
            "content": default_content
        })
    
    return messages

## Tools factory - will be initialized based on configuration
def get_search_tool(config: RunnableConfig):
    """Get the appropriate search tool based on configuration"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    
    logger.info(f"[SearchTool] Configuring search API: {search_api}")

    # Return None if no search tool is requested
    if search_api.lower() == "none":
        logger.info("[SearchTool] Search function disabled")
        return None

    # A dictionary mapping search API names to their corresponding search functions
    search_functions = {
        "tavily": lambda queries, config: intelligent_search_web_unified(queries, config, "tavily"),
        "duckduckgo": lambda queries, config: intelligent_search_web_unified(queries, config, "duckduckgo"),
        "googlesearch": lambda queries, config: intelligent_search_web_unified(queries, config, "googlesearch"),
        "geminigooglesearch": lambda queries, config: intelligent_search_web_unified(queries, config, "geminigooglesearch"),
        "azureaisearch": azureaisearch_search,
    }

    # Get the search function from the dictionary
    search_function = search_functions.get(search_api.lower())

    if not search_function:
        logger.error(f"[SearchTool] Unsupported search API: {search_api}")
        raise NotImplementedError(
            f"The search API '{search_api}' is not yet supported in the multi-agent implementation. "
            f"Currently, only Tavily/DuckDuckGo/GoogleSearch/GeminiGoogleSearch/AzureAISearch/None is supported. Please use the graph-based implementation in "
            f"src/open_deep_research/graph.py for other search APIs, or set search_api to one of the supported options."
        )

    # Create a generic search tool
    @tool
    async def search_tool(queries: List[str]) -> str:
        """Perform intelligent search using the configured search engine"""
        return await search_function(queries, config)

    tool_metadata = {**(search_tool.metadata or {}), "type": "search"}
    search_tool.metadata = tool_metadata
    logger.info(f"[SearchTool] Successfully created search tool: {search_tool.name}")
    return search_tool

class Section(BaseModel):
    """Section of the report."""
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Research scope for this section of the report.",
    )
    content: str = Field(
        description="The content of the section."
    )

class Sections(BaseModel):
    """List of section titles of the report."""
    sections: List[str] = Field(
        description="Sections of the report.",
    )

class Introduction(BaseModel):
    """Introduction to the report."""
    name: str = Field(
        description="Name for the report.",
    )
    content: str = Field(
        description="The content of the introduction, giving an overview of the report."
    )

class Conclusion(BaseModel):
    """Conclusion to the report."""
    name: str = Field(
        description="Name for the conclusion of the report.",
    )
    content: str = Field(
        description="The content of the conclusion, summarizing the report."
    )

class Question(BaseModel):
    """Ask a follow-up question to clarify the report scope."""
    question: str = Field(
        description="A specific question to ask the user to clarify the scope, focus, or requirements of the report."
    )

# No-op tool to indicate that the research is complete
class FinishResearch(BaseModel):
    """Finish the research."""

# No-op tool to indicate that the report writing is complete
class FinishReport(BaseModel):
    """Finish the report."""

## State
class AppState(TypedDict):
    messages: Annotated[list, operator.add]
    sections: list[str]
    completed_sections: Annotated[list[Section], operator.add]
    final_report: str
    source_str: Annotated[str, operator.add]


async def _load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    if not configurable.mcp_server_config:
        return []

    mcp_server_config = configurable.mcp_server_config
    client = MultiServerMCPClient(mcp_server_config)
    mcp_tools = await client.get_tools()
    filtered_mcp_tools: list[BaseTool] = []
    for tool in mcp_tools:
        # TODO: this will likely be hard to manage
        # on a remote server that's not controlled by the developer
        # best solution here is allowing tool name prefixes in MultiServerMCPClient
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue

        if configurable.mcp_tools_to_include and tool.name not in configurable.mcp_tools_to_include:
            continue

        filtered_mcp_tools.append(tool)

    return filtered_mcp_tools


# Tool lists will be built dynamically based on configuration
async def get_supervisor_tools_impl(config: RunnableConfig) -> list[BaseTool]:
    """Get supervisor tools based on configuration (implementation)"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    search_tool = get_search_tool(config)
    tools = [tool(Sections), tool(Introduction), tool(Conclusion), tool(FinishReport)]
    if configurable.ask_for_clarification:
        tools.append(tool(Question))
    if search_tool is not None:
        tools.append(search_tool)  # Add search tool, if available
    existing_tool_names = {cast(BaseTool, tool).name for tool in tools}
    mcp_tools = await _load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools


async def get_research_tools_impl(config: RunnableConfig) -> list[BaseTool]:
    """Get research tools based on configuration (implementation)"""
    logger.info("[ResearcherTools] Starting to get research tools")
    search_tool = get_search_tool(config)
    tools = [tool(Section), tool(FinishResearch)]
    if search_tool is not None:
        tools.append(search_tool)  # Add search tool, if available
    existing_tool_names = {cast(BaseTool, tool).name for tool in tools}
    mcp_tools = await _load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    logger.info(f"[ResearcherTools] Research tools: {tools}")
    return tools


# Helper: safely get tool_calls from either LangChain message objects or plain dictionaries
def _get_tool_calls(msg):
    """Safely retrieve tool calls from a message (handles both dict and LangChain objects)."""
    if isinstance(msg, dict):
        return msg.get('tool_calls', [])
    elif hasattr(msg, 'tool_calls'):
        return msg.tool_calls or []
    return []

def _validate_messages_for_llm(messages: list, model_name: str, context: str = "Unknown") -> list:
    """
    Final validation to ensure all messages have non-empty content before sending to LLM.
    This is especially critical for Gemini models.
    
    Args:
        messages: List of messages to validate
        model_name: Name of the model being used
        context: Context string for logging (e.g., "Supervisor", "Researcher")
    
    Returns:
        List of validated messages with guaranteed non-empty content
    """
    is_gemini = model_name.lower().startswith('gemini') or 'google' in model_name.lower()
    validated_messages = []
    
    logger.info(f"[{context}] Pre-flight validation for {len(messages)} messages (Model: {model_name})")
    
    # Special handling for Gemini - filter out problematic message patterns
    if is_gemini:
        # Remove consecutive tool messages that might cause issues
        filtered_messages = []
        prev_was_tool = False
        for msg in messages:
            if isinstance(msg, dict):
                is_tool = msg.get('role') == 'tool'
                # Skip consecutive tool messages for Gemini
                if is_tool and prev_was_tool:
                    logger.warning(f"[{context}] Skipping consecutive tool message for Gemini compatibility")
                    continue
                filtered_messages.append(msg)
                prev_was_tool = is_tool
            else:
                filtered_messages.append(msg)
                prev_was_tool = False
        messages = filtered_messages
    
    for i, msg in enumerate(messages):
        msg_copy = msg.copy() if isinstance(msg, dict) else msg
        
        # Ensure we have required fields
        if isinstance(msg_copy, dict):
            role = msg_copy.get('role', 'unknown')
            original_content = msg_copy.get('content', '')
            content = original_content
            
            # Convert None to empty string
            if content is None:
                content = ''
            
            # Ensure content is string
            content = str(content)
            
            logger.debug(f"[{context}] _validate_messages_for_llm: Before fix - Message {i} (Role: {role}, Original Content: '{str(original_content)[:100] if original_content else ''}', Current Content: '{str(content)[:100] if content else ''}', Full Message: {str(msg_copy)[:100] if msg_copy else ''})")

            # For Gemini, ensure ALL messages have substantial content
            if is_gemini:
                # Remove any zero-width or special characters that might cause issues
                content = content.strip()
                
                # If content is empty or too short, provide more substantial default
                # Ensure content is never truly empty after stripping
                if not content:
                    content = "."  # Use a period instead of space to ensure it's not stripped
                    
            # Check content length for all models
            if len(content) < 2 and is_gemini:  # Less than 2 chars is problematic for Gemini
                has_tool_calls = bool(msg_copy.get('tool_calls'))

                # Provide more substantial default content for Gemini
                if role == 'assistant' and has_tool_calls:
                    content = 'I am calling the necessary tools to help with your request.'
                elif role == 'tool':
                    tool_name = msg_copy.get('name', 'unknown')
                    content = f'The {tool_name} tool has completed execution successfully with no specific output.'
                elif role == 'user':
                    content = 'Please continue with the task.'
                elif role == 'system':
                    content = 'System instructions provided.'
                else:
                    content = 'Processing your request.'

                logger.warning(f"[{context}] Gemini fix: Enhanced message {i} ({role}) content to: '{content[:100] if content else ''}'")
            else:
                # For non-Gemini models, use simpler defaults
                if not content.strip():
                    has_tool_calls = bool(msg_copy.get('tool_calls'))

                    if role == 'assistant' and has_tool_calls:
                        content = 'Calling tools.'
                    elif role == 'tool':
                        tool_name = msg_copy.get('name', 'unknown')
                        content = f'Tool {tool_name} completed with no specific output.'
                    elif role == 'user':
                        content = 'Please continue.'
                    elif role == 'system':
                        content = 'System message.'
                    else:
                        content = 'Continue.'

                    logger.warning(f"[{context}] Pre-flight fix: Message {i} ({role}) had empty content, "
                                 f"set to: '{content[:100] if content else ''}'")

            msg_copy['content'] = content

            # Final safety check for Gemini
            if is_gemini and len(msg_copy['content']) < 2:
                msg_copy['content'] = 'Processing request.'
                logger.error(f"[{context}] Gemini safety: Message {i} content too short, fixed!")

            validated_messages.append(msg_copy)
        else:
            # For non-dict messages, just append as-is
            validated_messages.append(msg_copy)
        logger.debug(f"[{context}] _validate_messages_for_llm: After fix - Message {i} (Role: {msg_copy.get('role')}, Current Content: '{str(msg_copy.get('content'))[:100] if msg_copy.get('content') else ''}')")
        
    # Log summary
    empty_count = sum(1 for msg in messages if isinstance(msg, dict) and not str(msg.get('content', '')).strip())
    if empty_count > 0:
        logger.warning(f"[{context}] Pre-flight summary: Fixed {empty_count} empty messages out of {len(messages)}")

    # Additional Gemini-specific validation
    if is_gemini:
        logger.info(f"[{context}] Gemini validation complete - {len(validated_messages)} messages processed")

    return validated_messages

def _format_messages_for_gemini(messages: list, context: str = "Unknown") -> list:
    """
    Special formatter for Gemini models to work around known API issues.
    Gemini has strict requirements about message formatting and content.

    Args:
        messages: List of messages to format
        context: Context string for logging

    Returns:
        List of messages formatted for Gemini API
    """
    logger.info(f"[{context}] Applying Gemini-specific message formatting")

    formatted_messages = []
    message_count = len(messages)

    # Track message roles to ensure proper alternation
    last_role = None

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            logger.warning(f"[{context}] Skipping non-dict message at position {i}")
            continue

        msg_copy = msg.copy()
        role = msg_copy.get('role', 'unknown')
        original_content = msg_copy.get('content', '')
        content = str(original_content)

        logger.debug(f"[{context}] _format_messages_for_gemini: Before fix - Message {i} (Role: {role}, Original Content: '{str(original_content)[:100] if original_content else ''}', Current Content: '{str(content)[:100] if content else ''}')")

        # Ensure content is never empty or too short after stripping
        content = content.strip()
        if not content:
            content = "."  # Use a period instead of space to ensure it's not stripped
            
        if len(content) < 5:
            if role == 'system':
                content = "You are a helpful AI assistant. Follow the instructions carefully."
            elif role == 'user':
                content = "Please proceed with the current task as instructed."
            elif role == 'assistant':
                if msg_copy.get('tool_calls'):
                    content = "I will use the appropriate tools to help you with this request."
                else:
                    content = "I understand. Let me help you with that."
            elif role == 'tool':
                tool_name = msg_copy.get('name', 'tool')
                content = f"The {tool_name} has been executed and returned its results successfully with no specific output."
            else:
                content = "Continuing with the current operation."

            logger.info(f"[{context}] Enhanced content for {role} message at position {i}")

        msg_copy['content'] = content

        # Ensure proper message alternation for Gemini
        if last_role == role and role in ['user', 'assistant']:
            # Insert a bridging message to maintain alternation
            if role == 'user':
                bridge_msg = {
                    'role': 'assistant',
                    'content': 'I understand your request. Let me process that for you.'
                }
                formatted_messages.append(bridge_msg)
                logger.info(f"[{context}] Added bridge assistant message for alternation")
            elif role == 'assistant':
                bridge_msg = {
                    'role': 'user',
                    'content': 'Please continue with the analysis.'
                }
                formatted_messages.append(bridge_msg)
                logger.info(f"[{context}] Added bridge user message for alternation")

        formatted_messages.append(msg_copy)
        last_role = role
        logger.debug(f"[{context}] _format_messages_for_gemini: After fix - Message {i} (Role: {msg_copy.get('role')}, Content: '{str(msg_copy.get('content', ''))[:100]}')")

    # Ensure the message sequence ends properly for Gemini
    if formatted_messages and formatted_messages[-1].get('role') == 'tool':
        # Add a user message after tool response
        formatted_messages.append({
            'role': 'user',
            'content': 'Please continue based on the tool results.'
        })
        logger.info(f"[{context}] Added closing user message after tool response")

    logger.info(f"[{context}] Gemini formatting complete: {message_count} -> {len(formatted_messages)} messages")
    return formatted_messages

async def supervisor(state: AppState, config: RunnableConfig):
    """LLM decides whether to call a tool or not
    tool list: Sections、Introduction、Conclusion、FinishReport、Question、search_tool
    """
    
    logger.info("[Supervisor] Starting supervisor task execution")
    
    # Messages
    messages = state["messages"]
    logger.info(f"[Supervisor] Received {len(messages)} messages")

    # Get configuration
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    supervisor_model = get_config_value(configurable.supervisor_model)
    logger.info(f"[Supervisor] Using model: {supervisor_model}")

    # Initialize the model
    llm = init_chat_model(model=supervisor_model)
    
    # If sections have been completed, but we don't yet have the final report, then we need to initiate writing the introduction and conclusion
    if state.get("completed_sections") and not state.get("final_report"):
        completed_count = len(state["completed_sections"])
        logger.info(f"[Supervisor] Research completed, starting report writing. Finished {completed_count} sections")
        research_complete_message = {"role": "user", "content": "Research is complete. Now write the introduction and conclusion for the report. Here are the completed main body sections: \n\n" + "\n\n".join([s.content for s in state["completed_sections"]])}
        messages = messages + [research_complete_message]

    # Get tools from global cache
    supervisor_tool_list = await tool_manager.get_supervisor_tools(config)
    logger.info(f"[Supervisor] Loaded {len(supervisor_tool_list)} tools from cache: {[tool.name for tool in supervisor_tool_list]}")

    # Convert LangChain message objects to dict format for message manager
    dict_messages = convert_langchain_messages_to_dict(messages)
    logger.debug(f"[Supervisor] Converted {len(dict_messages)} messages to dictionary format")

    # Fix Gemini message sequence problem:
    # Invalid argument provided to Gemini: 400 Please ensure that function call turn comes immediately after a user turn or after a function response turn response turn is not allowed to be followed by a function call turn
    # Use global message manager to handle message sequence
    supervisor_model = get_config_value(configurable.supervisor_model)
    provider_hint = supervisor_model.split(":")[0] if ":" in supervisor_model else supervisor_model
    messages, fixes = validate_and_fix_messages(dict_messages, provider_hint)
    if fixes:
        logger.info(f"[Supervisor] Message sequence fixes: {', '.join(fixes)}")
    
    # Determine tool_choice based on state
    # If we haven't generated sections yet, we should encourage tool use
    sections_generated = bool(state.get("sections")) or bool(state.get("completed_sections"))
    report_completed = bool(state.get("final_report"))
    
    if report_completed:
        # Report is done, don't force tool calls
        tool_choice = "auto"
        logger.info("[Supervisor] Report completed, using tool_choice='auto'")
    elif not sections_generated:
        # We haven't even started, encourage tool use
        tool_choice = "any"
        logger.info("[Supervisor] No sections generated yet, using tool_choice='any' to start workflow")
    else:
        # In progress, let the model decide
        tool_choice = "auto" 
        logger.info("[Supervisor] Work in progress, using tool_choice='auto'")
    
    llm_with_tools = (
        llm
        .bind_tools(
            supervisor_tool_list,
            # parallel_tool_calls=False,
            tool_choice=tool_choice
        )
    )

    # Get system prompt
    system_prompt = SUPERVISOR_INSTRUCTIONS.format(today=get_today_str())
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # Prepare the final messages for LLM (system + user messages)
    llm_messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ] + messages

    # Ultra-strict safety check: ensure no empty content for Gemini API
    safe_messages = []
    for i, msg in enumerate(llm_messages):
        msg_copy = msg.copy()
        
        # Ensure content field exists
        if 'content' not in msg_copy:
            msg_copy['content'] = ''
        
        # Ensure content is not None
        if msg_copy['content'] is None:
            msg_copy['content'] = ''
            
        # Convert to string if not already
        msg_copy['content'] = str(msg_copy['content'])
        
        # Check if content is effectively empty
        if not msg_copy['content'].strip():
            role = msg_copy.get('role', 'unknown')
            has_tool_calls = bool(msg_copy.get('tool_calls'))
            
            # Get more detailed info for debugging
            tool_info = f" (has {len(msg_copy.get('tool_calls', []))} tool calls)" if has_tool_calls else ""
            logger.warning(f"[Supervisor] Empty content detected in message {i} - role: {role}{tool_info}")
            
            # Log the full message for debugging
            logger.debug(f"[Supervisor] Full empty message {i}: {msg_copy}")
            
            if role == 'assistant' and has_tool_calls:
                msg_copy['content'] = 'Calling tools.'
            elif role == 'user':
                msg_copy['content'] = 'Please continue.'
            elif role == 'system':
                msg_copy['content'] = 'System prompt.'
            elif role == 'tool':
                # For tool messages, try to get more specific content
                tool_name = msg_copy.get('name', 'unknown_tool')
                msg_copy['content'] = f'Tool {tool_name} execution completed.'
            else:
                msg_copy['content'] = 'Continue.'
            
            logger.warning(f"[Supervisor] Fixed empty content in message {i} ({role}) - new content: '{msg_copy['content']}'")
        
        # Final verification: ensure content is never empty
        if not msg_copy['content'].strip():
            msg_copy['content'] = 'Continue.'
            logger.error(f"[Supervisor] Applied emergency fallback for message {i} - this should not happen!")
            
        # Additional check for Gemini: ensure no empty strings after all processing
        if supervisor_model.lower().startswith('gemini') or 'google' in supervisor_model.lower():
            # Extra validation for Gemini
            if 'content' in msg_copy and isinstance(msg_copy['content'], str) and len(msg_copy['content']) == 0:
                msg_copy['content'] = 'Continue.'
                logger.error(f"[Supervisor] Gemini safety: Found zero-length string in message {i}, fixed")
            
        safe_messages.append(msg_copy)

    # Final pre-flight validation
    validated_messages = _validate_messages_for_llm(safe_messages, supervisor_model, "Supervisor")
    
    # Apply Gemini-specific formatting if needed
    if supervisor_model.lower().startswith('gemini') or 'google' in supervisor_model.lower():
        validated_messages = _format_messages_for_gemini(validated_messages, "Supervisor")
    
    # Final safety net for Gemini: ensure no message has empty content before invoking
    # This is critical because langchain_google_genai may not handle all empty cases properly
    final_messages = []
    for i, msg in enumerate(validated_messages):
        msg_copy = msg.copy() if isinstance(msg, dict) else msg
        
        if isinstance(msg_copy, dict):
            # Ensure content exists and is substantial
            content = str(msg_copy.get('content', ''))
            
            # If content is empty or just whitespace, provide a meaningful default
            if not content or not content.strip() or content.strip() in ['.', ' ']:
                role = msg_copy.get('role', 'unknown')
                has_tool_calls = bool(msg_copy.get('tool_calls'))
                
                # Provide role-specific substantial content
                if role == 'assistant' and has_tool_calls:
                    msg_copy['content'] = 'I am calling the necessary tools to complete the research.'
                elif role == 'assistant':
                    msg_copy['content'] = 'I am processing the research task.'
                elif role == 'user':
                    msg_copy['content'] = 'Please continue with the research.'
                elif role == 'system':
                    msg_copy['content'] = 'System instructions for research have been provided.'
                elif role == 'tool':
                    tool_name = msg_copy.get('name', 'tool')
                    msg_copy['content'] = f'The {tool_name} tool has returned results.'
                else:
                    msg_copy['content'] = 'Continuing with the research process.'
                
                logger.warning(f"[Supervisor] Final safety net: Enhanced empty content for message {i} ({role})")
        
        final_messages.append(msg_copy)
    
    logger.info(f"[Supervisor] Invoking LLM with {len(final_messages)} messages.")
    logger.debug(f"[Supervisor] LLM Messages (final before invoke): {final_messages}")
    
    # 添加详细的消息内容调试日志
    if supervisor_model.lower().startswith('gemini') or 'google' in supervisor_model.lower():
        logger.info(f"[Supervisor] Gemini API Debug - Checking all messages before invoke:")
        for i, msg in enumerate(final_messages):
            msg_role = msg.get('role', 'unknown')
            msg_content = msg.get('content', '')
            content_length = len(str(msg_content)) if msg_content is not None else 0
            has_tool_calls = bool(msg.get('tool_calls'))
            
            logger.info(f"[Supervisor] Message {i}: role={msg_role}, content_length={content_length}, "
                       f"has_tool_calls={has_tool_calls}, content_preview='{str(msg_content)[:50] if msg_content else 'NONE'}'")
            
            # 检查是否有空内容
            if not msg_content or (isinstance(msg_content, str) and not msg_content.strip()):
                logger.error(f"[Supervisor] WARNING: Message {i} has empty/None content! Full message: {msg}")
            
            # 如果是工具调用消息，记录工具信息
            if has_tool_calls:
                tool_calls = msg.get('tool_calls', [])
                logger.info(f"[Supervisor] Message {i} tool calls: {[tc.get('name', 'unknown') for tc in tool_calls]}")
    
    """When the research agent is given a task but the search fails (due to fetch errors), it still MUST call a tool because of tool_choice="any". But if it can't get any useful search results, it might keep trying to call the search tool repeatedly, or it might not know what to do.
The issue is:
1. tool_choice="any" forces the LLM to always call at least one tool
2. If the search keeps failing, the research agent can't call the Section tool (because it has no content)
3. It can't call FinishResearch either (because it hasn't completed the research)
4. So it gets stuck in a loop, always being forced to call a tool but unable to make progress
    """
    
    try:
        response = await llm_with_tools.ainvoke(final_messages)
        logger.info(f"[Supervisor] LLM response: {response}")
        logger.info(f"[Supervisor] LLM response completed with {len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0} tool calls")
    except Exception as e:
        logger.error(f"[Supervisor] LLM invocation failed: {str(e)}")
        # 如果是 Gemini 相关错误，记录更多信息
        if "GenerateContentRequest.contents" in str(e) and "parts" in str(e):
            logger.error(f"[Supervisor] Gemini API error detected - checking message structure")
            # 尝试找出哪个消息可能有问题
            import re
            match = re.search(r'contents\[(\d+)\]', str(e))
            if match:
                msg_index = int(match.group(1))
                logger.error(f"[Supervisor] Error indicates problem with message at index {msg_index}")
                if 0 <= msg_index < len(final_messages):
                    problem_msg = final_messages[msg_index]
                    logger.error(f"[Supervisor] Problematic message: {problem_msg}")
        
        # 改为优雅降级而不是抛出异常
        logger.warning(f"[Supervisor] Falling back to error response due to LLM failure")
        
        # 创建一个降级响应，模拟 AI 决定结束研究
        from langchain_core.messages import AIMessage
        
        # 根据当前状态决定使用哪个工具
        sections_generated = bool(state.get("sections")) or bool(state.get("completed_sections"))
        report_completed = bool(state.get("final_report"))
        
        if report_completed:
            # 报告已完成，使用 FinishReport
            fallback_response = AIMessage(
                content="Due to technical difficulties, I'm unable to continue. The report generation is complete.",
                tool_calls=[{
                    "id": "fallback_finish",
                    "name": "FinishReport",
                    "args": {}
                }]
            )
            logger.info(f"[Supervisor] Created fallback response with FinishReport tool call")
        elif not sections_generated:
            # 还没有生成章节，创建一个简单的章节列表
            fallback_response = AIMessage(
                content="Due to technical difficulties, I'll create a basic report structure.",
                tool_calls=[{
                    "id": "fallback_sections",
                    "name": "Sections",
                    "args": {"sections": ["Overview", "Analysis", "Conclusion"]}
                }]
            )
            logger.info(f"[Supervisor] Created fallback response with Sections tool call")
        else:
            # 正在进行中，询问用户
            fallback_response = AIMessage(
                content="Due to technical difficulties, I'm unable to continue the report generation.",
                tool_calls=[{
                    "id": "fallback_question",
                    "name": "Question",
                    "args": {"question": "I encountered technical difficulties. Would you like me to continue with a simplified version of the report?"}
                }]
            )
            logger.info(f"[Supervisor] Created fallback response with Question tool call")
        
        response = fallback_response

    return {
        "messages": [response]
    }

async def _execute_tool(tool_call, tools_by_name, config):
    """A helper function to execute a single tool call."""
    tool_name = tool_call["name"]
    args = tool_call.get('args', {})
    logger.info(f"[ToolExecutor] Executing tool: {tool_name} with args: {args}")
    
    tool = tools_by_name[tool_name]
    try:
        observation = await tool.ainvoke(args, config)
        logger.info(f"[ToolExecutor] Tool {tool_name} executed successfully")
    except NotImplementedError:
        observation = tool.invoke(args, config)
        logger.info(f"[ToolExecutor] Tool {tool_name} executed successfully (sync)")
    except Exception as e:
        logger.error(f"[ToolExecutor] Tool {tool_name} with args {args} execution failed: {e}")
        raise

    # Ensure observation is never None or empty (critical for Gemini)
    if observation is None:
        observation = f"Tool {tool_name} completed with no output."
        logger.warning(f"[ToolExecutor] Tool {tool_name} returned None, using default message.")
    elif hasattr(observation, '__str__'):
        observation_str = str(observation)
        if not observation_str.strip():
            observation = f"Tool {tool_name} completed successfully with no specific output."
            logger.warning(f"[ToolExecutor] Tool {tool_name} returned empty string, using default message.")
    elif isinstance(observation, str) and not observation.strip():
        observation = f"Tool {tool_name} completed successfully with no specific output."
        logger.warning(f"[ToolExecutor] Tool {tool_name} returned empty string, using default message.")
    
    # Build the tool response message with guaranteed non-empty content
    tool_response = {
        "role": "tool", 
        "content": str(observation),  # Ensure it's a string
        "name": tool_call["name"], 
        "tool_call_id": tool_call["id"]
    }
    
    # Final safety check for Gemini: ensure content is never empty after all processing
    if not tool_response["content"].strip():
        tool_response["content"] = f"Tool {tool_name} execution completed with no specific output."
        logger.error(f"[ToolExecutor] Emergency fix: Tool response was empty after all checks!")
    
    # Ensure content is never just an empty string, especially for Gemini
    if not tool_response["content"]:
        tool_response["content"] = "Tool executed successfully with no specific output."
        logger.warning(f"[ToolExecutor] Final content check: Tool response content was empty, set to default.")
    
    return tool_response, observation


async def supervisor_tools(state: AppState, config: RunnableConfig) -> Command[Literal["supervisor", "research_team", "__end__"]]:
    """Performs the tool call and sends to the research agent"""
    logger.info("[SupervisorTools] Starting tool execution")

    configurable = MultiAgentConfiguration.from_runnable_config(config)
    tool_calls = state["messages"][-1].tool_calls if state["messages"] and hasattr(state["messages"][-1], 'tool_calls') else []
    logger.info(f"[SupervisorTools] Need to execute {len(tool_calls)} tool calls: {[tc.get('name', 'unknown') for tc in tool_calls]}")

    supervisor_tool_list = await tool_manager.get_supervisor_tools(config)
    supervisor_tools_by_name = {tool.name: tool for tool in supervisor_tool_list}
    search_tool_names = {tool.name for tool in supervisor_tool_list if tool.metadata and tool.metadata.get("type") == "search"}

    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None
    source_str = ""

    for tool_call in tool_calls:
        tool_name = tool_call["name"]

        tool_prompt_result, observation = await _execute_tool(tool_call, supervisor_tools_by_name, config)
        result.append(tool_prompt_result)

        if tool_name == "Question":
            question_obj = cast(Question, observation)
            logger.info(f"[SupervisorTools] Generated question: {question_obj.question}")
            result.append({"role": "assistant", "content": question_obj.question})
            return Command(goto=END, update={"messages": result})
        elif tool_name == "Sections":
            sections_list = cast(Sections, observation).sections
            logger.info(f"[SupervisorTools] Generated {len(sections_list)} sections: {sections_list}")
        elif tool_name == "Introduction":
            observation = cast(Introduction, observation)
            intro_content = f"""
# {observation.name}

{observation.content}
"""
            logger.info(f"[SupervisorTools] Generated introduction: {observation.name}")
        elif tool_name == "Conclusion":
            observation = cast(Conclusion, observation)
            conclusion_content = f"""
## {observation.name}

{observation.content}
"""
            logger.info(f"[SupervisorTools] Generated conclusion: {observation.name}")
        elif tool_name in search_tool_names:
            if configurable.include_source_str:
                source_str += cast(str, observation)

    if sections_list:
        logger.info(f"[SupervisorTools] Assigning {len(sections_list)} sections to research team")
        # Pass section information through messages instead of state to avoid concurrent update conflicts
        section_messages = []
        for section in sections_list:
            section_msg = {"messages": [{"role": "user", "content": f"Research and write the following section: {section}"}]}
            section_messages.append(Send("research_team", section_msg))
        return Command(goto=section_messages, update={"messages": result})

    state_update = {"messages": result}
    if intro_content:
        logger.info("[SupervisorTools] Introduction completed, waiting for conclusion")
        result.append({"role": "user", "content": "Introduction written. Now write a conclusion section."})
        state_update["final_report"] = intro_content
    elif conclusion_content:
        intro = state.get("final_report", "")
        body_sections = """
        
        """.join([s.content for s in state["completed_sections"]])
        complete_report = f"""
{intro}

{body_sections}
{conclusion_content}
"""
        logger.info(f"[SupervisorTools] Report completed! Total length: {len(complete_report)} characters")
        result.append({"role": "user", "content": "Report is now complete with introduction, body sections, and conclusion."})
        state_update["final_report"] = complete_report

    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str

    logger.info(f"[SupervisorTools] Returning state update with keys: {state_update.keys()}")
    return Command(goto="supervisor", update=state_update)

async def supervisor_should_continue(state: AppState) -> str:
    """Decide if we should continue the loop or stop based on whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # Safely retrieve tool call lists (works for dict or LangChain message objects)
    last_tool_calls = _get_tool_calls(last_message)
    prev_tool_calls = _get_tool_calls(messages[-2]) if len(messages) > 2 else []

    # Check if the last two messages are tool calls to the same tool
    if len(last_tool_calls) > 0 and len(prev_tool_calls) > 0:
        if last_tool_calls[0]["name"] == prev_tool_calls[0]["name"]:
            logger.info("[SupervisorControl] Detected a potential loop, forcing end.")
            return END

    # End because the supervisor asked a question or is finished
    if not last_tool_calls or (len(last_tool_calls) == 1 and last_tool_calls[0]["name"] == "FinishReport"):
        logger.info("[SupervisorControl] Supervisor decided to end process")
        return END

    # If the LLM makes a tool call, then perform an action
    tool_names = [tc.get("name", "unknown") for tc in last_tool_calls]
    logger.info(f"[SupervisorControl] Supervisor continuing with tools: {tool_names}")
    return "supervisor_tools_node"

async def research_agent(state: AppState, config: RunnableConfig):
    """LLM decides whether to call a tool or not
    tool list: Section、FinishResearch、search_tool
    """
    
    # Extract section description from the latest user message
    section_description = "General research topic"
    messages = state.get("messages", [])
    
    # Find the latest user message that contains section assignment
    for message in reversed(messages):
        if (isinstance(message, dict) and message.get("role") == "user" and 
            "Research and write the following section:" in message.get("content", "")):
            # Extract section from message content
            content = message["content"]
            section_start = content.find("Research and write the following section:") + len("Research and write the following section:")
            section_description = content[section_start:].strip()
            break
        elif hasattr(message, 'content') and "Research and write the following section:" in str(message.content):
            # Handle LangChain message objects
            content = str(message.content)
            section_start = content.find("Research and write the following section:") + len("Research and write the following section:")
            section_description = content[section_start:].strip()
            break
    
    logger.info(f"[Researcher] Starting research for section: {section_description}")
    
    # Get configuration
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    researcher_model = get_config_value(configurable.researcher_model)
    logger.info(f"[Researcher] Using model: {researcher_model}")
    
    # Initialize the model
    llm = init_chat_model(model=researcher_model)

    # Get tools from global cache
    research_tool_list = await tool_manager.get_research_tools(config)
    logger.info(f"[Researcher] Loaded {len(research_tool_list)} tools from cache: {[tool.name for tool in research_tool_list]}")
    
    system_prompt = RESEARCH_INSTRUCTIONS.format(
        section_description=section_description,
        number_of_queries=configurable.number_of_queries,
        today=get_today_str(),
    )
    logger.debug(f"[Researcher] System prompt length: {len(system_prompt)} characters")
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # Ensure we have at least one user message (required by Anthropic)
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": f"Please research and write the section: {section_description}"}]
        logger.info("[Researcher] Created initial research message")
    
    logger.info(f"[Researcher] Processing {len(messages)} messages")

    # Convert LangChain message objects to dict format for message manager
    dict_messages = convert_langchain_messages_to_dict(messages)

    # Fix Gemini message sequence problem:
    # Invalid argument provided to Gemini: 400 Please ensure that function call turn comes immediately after a user turn or after a function response turn response turn is not allowed to be followed by a function call turn
    # Use global message manager to handle message sequence
    provider_hint = researcher_model.split(":")[0] if ":" in researcher_model else researcher_model
    messages, fixes = validate_and_fix_messages(dict_messages, provider_hint)
    if fixes:
        logger.info(f"[Researcher] Message sequence fixes: {', '.join(fixes)}")

    # Prepare the final messages for LLM (system + user messages)
    llm_messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ] + messages

    # Ultra-strict safety check: ensure no empty content for Gemini API
    safe_messages = []
    
    # Ultra-strict safety check: ensure no empty content for Gemini API
    safe_messages = []
    problematic_indices = [11, 14, 23, 26, 29, 32, 35] # Add all problematic indices here

    for i, msg in enumerate(llm_messages):
        msg_copy = msg.copy()
        
        # Ensure content field exists
        if 'content' not in msg_copy:
            msg_copy['content'] = ''
        
        # Ensure content is not None
        if msg_copy['content'] is None:
            msg_copy['content'] = ''
            
        # Convert to string if not already
        msg_copy['content'] = str(msg_copy['content'])
        
        # Check if content is effectively empty
        if not msg_copy['content'].strip():
            role = msg_copy.get('role', 'unknown')
            has_tool_calls = bool(msg_copy.get('tool_calls'))
            
            # Get more detailed info for debugging
            tool_info = f" (has {len(msg_copy.get('tool_calls', []))} tool calls)" if has_tool_calls else ""
            logger.warning(f"[Researcher] Empty content detected in message {i} - role: {role}{tool_info}")
            
            # Log the full message for debugging if it's at a problematic position
            if i in problematic_indices:
                logger.error(f"[Researcher] Problematic position {i} has empty content! Full message: {msg_copy}")
            
            if role == 'assistant' and has_tool_calls:
                msg_copy['content'] = 'Calling tools.'
            elif role == 'user':
                msg_copy['content'] = 'Please continue.'
            elif role == 'system':
                msg_copy['content'] = 'System prompt.'
            elif role == 'tool':
                # For tool messages, try to get more specific content
                tool_name = msg_copy.get('name', 'unknown_tool')
                msg_copy['content'] = f'Tool {tool_name} execution completed.'
            else:
                msg_copy['content'] = 'Continue.'
            
            logger.warning(f"[Researcher] Fixed empty content in message {i} ({role}) - new content: '{msg_copy['content']}'")
        
        # Final verification: ensure content is never empty
        if not msg_copy['content'].strip():
            msg_copy['content'] = 'Continue.'
            logger.error(f"[Researcher] Applied emergency fallback for message {i} - this should not happen!")
        
        # Additional check for Gemini: ensure no empty strings after all processing
        if researcher_model.lower().startswith('gemini') or 'google' in researcher_model.lower():
            # Extra validation for Gemini
            if 'content' in msg_copy and isinstance(msg_copy['content'], str) and len(msg_copy['content']) == 0:
                msg_copy['content'] = 'Continue.'
                logger.error(f"[Researcher] Gemini safety: Found zero-length string in message {i}, fixed")
            
        safe_messages.append(msg_copy)

    # Final pre-flight validation
    validated_messages = _validate_messages_for_llm(safe_messages, researcher_model, "Researcher")
    
    # Apply Gemini-specific formatting if needed
    if researcher_model.lower().startswith('gemini') or 'google' in researcher_model.lower():
        validated_messages = _format_messages_for_gemini(validated_messages, "Researcher")
    
    # Final safety net for Gemini: ensure no message has empty content before invoking
    # This is critical because langchain_google_genai may not handle all empty cases properly
    final_messages = []
    for i, msg in enumerate(validated_messages):
        msg_copy = msg.copy() if isinstance(msg, dict) else msg
        
        if isinstance(msg_copy, dict):
            # Ensure content exists and is substantial
            content = str(msg_copy.get('content', ''))
            
            # If content is empty or just whitespace, provide a meaningful default
            if not content or not content.strip() or content.strip() in ['.', ' ']:
                role = msg_copy.get('role', 'unknown')
                has_tool_calls = bool(msg_copy.get('tool_calls'))
                
                # Provide role-specific substantial content
                if role == 'assistant' and has_tool_calls:
                    msg_copy['content'] = 'I am calling the necessary tools to complete the research.'
                elif role == 'assistant':
                    msg_copy['content'] = 'I am processing the research task.'
                elif role == 'user':
                    msg_copy['content'] = 'Please continue with the research.'
                elif role == 'system':
                    msg_copy['content'] = 'System instructions for research have been provided.'
                elif role == 'tool':
                    tool_name = msg_copy.get('name', 'tool')
                    msg_copy['content'] = f'The {tool_name} tool has returned results.'
                else:
                    msg_copy['content'] = 'Continuing with the research process.'
                
                logger.warning(f"[Researcher] Final safety net: Enhanced empty content for message {i} ({role})")
        
        final_messages.append(msg_copy)
    
    logger.info(f"[Researcher] Invoking LLM with {len(final_messages)} messages.")
    logger.debug(f"[Researcher] LLM Messages (final before invoke): {final_messages}")
    
    # 添加详细的消息内容调试日志
    if researcher_model.lower().startswith('gemini') or 'google' in researcher_model.lower():
        logger.info(f"[Researcher] Gemini API Debug - Checking all messages before invoke:")
        for i, msg in enumerate(final_messages):
            msg_role = msg.get('role', 'unknown')
            msg_content = msg.get('content', '')
            content_length = len(str(msg_content)) if msg_content is not None else 0
            has_tool_calls = bool(msg.get('tool_calls'))
            
            logger.info(f"[Researcher] Message {i}: role={msg_role}, content_length={content_length}, "
                       f"has_tool_calls={has_tool_calls}, content_preview='{str(msg_content)[:50] if msg_content else 'NONE'}'")
            
            # 检查是否有空内容
            if not msg_content or (isinstance(msg_content, str) and not msg_content.strip()):
                logger.error(f"[Researcher] WARNING: Message {i} has empty/None content! Full message: {msg}")
            
            # 如果是工具调用消息，记录工具信息
            if has_tool_calls:
                tool_calls = msg.get('tool_calls', [])
                logger.info(f"[Researcher] Message {i} tool calls: {[tc.get('name', 'unknown') for tc in tool_calls]}")
    
    try:
        response = await llm.bind_tools(research_tool_list,             
                          #  parallel_tool_calls=False,
                           # Remove forced tool call to allow graceful exit on search failures
                           # tool_choice="any"
                           ).ainvoke(final_messages)
        logger.info(f"[Researcher] LLM response type: {type(response)}")
        logger.info(f"[Researcher] Tool calls in response: {len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0}")
    except Exception as e:
        logger.error(f"[Researcher] LLM invocation failed: {str(e)}")
        # 如果是 Gemini 相关错误，记录更多信息
        if "GenerateContentRequest.contents" in str(e) and "parts" in str(e):
            logger.error(f"[Researcher] Gemini API error detected - checking message structure")
            # 尝试找出哪个消息可能有问题
            import re
            match = re.search(r'contents\[(\d+)\]', str(e))
            if match:
                msg_index = int(match.group(1))
                logger.error(f"[Researcher] Error indicates problem with message at index {msg_index}")
                if 0 <= msg_index < len(final_messages):
                    problem_msg = final_messages[msg_index]
                    logger.error(f"[Researcher] Problematic message: {problem_msg}")
        
        # 改为优雅降级而不是抛出异常
        logger.warning(f"[Researcher] Falling back to error response due to LLM failure")
        
        # 创建一个降级响应，模拟 AI 决定结束研究
        from langchain_core.messages import AIMessage
        
        # 创建一个包含 FinishResearch 工具调用的响应
        fallback_response = AIMessage(
            content="Due to technical difficulties, I'm unable to complete the research at this moment. Finishing the research task.",
            tool_calls=[{
                "id": f"fallback_{section_description[:10]}",
                "name": "FinishResearch",
                "args": {}
            }]
        )
        
        logger.info(f"[Researcher] Created fallback response with FinishResearch tool call")
        response = fallback_response

    result = {
        "messages": [response]
    }
    
    logger.debug(f"[Researcher] Returning result with {len(result['messages'])} messages")
    
    return result

async def research_agent_tools(state: AppState, config: RunnableConfig):
    """Performs the tool call and route to supervisor or continue the research loop"""
    logger.info("[ResearcherTools] Starting research tool execution")
    
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result = []
    completed_section = None
    source_str = ""
    
    # Count tool calls
    tool_calls = state["messages"][-1].tool_calls if state["messages"] and hasattr(state["messages"][-1], 'tool_calls') else []
    logger.info(f"[ResearcherTools] Need to execute {len(tool_calls)} tool calls: {[tc.get('name', 'unknown') for tc in tool_calls]}")
    
    # Get tools from global cache
    research_tool_list = await tool_manager.get_research_tools(config)
    research_tools_by_name = {tool.name: tool for tool in research_tool_list}
    logger.info(f"[ResearcherTools] Research tool names: {research_tools_by_name.keys()}")
    search_tool_names = {
        tool.name
        for tool in research_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }
    logger.info(f"[ResearcherTools] Search tools: {search_tool_names}")
    
    # Process all tool calls first (required for OpenAI)
    for tool_call in tool_calls:
        tool_prompt_result, observation = await _execute_tool(tool_call, research_tools_by_name, config)
        result.append(tool_prompt_result)
        
        # Store the section observation if a Section tool was called
        if tool_call["name"] == "Section":
            completed_section = cast(Section, observation)
            logger.info(f"[ResearcherTools] Completed section: {completed_section.name}, content length: {len(completed_section.content)} characters")

        # Intelligent search integration: Check if intelligent research mode is enabled
        if tool_call["name"] in search_tool_names:
            research_mode = get_config_value(configurable.research_mode, "simple")
            logger.info(f"research_mode: {research_mode}")
            if research_mode and research_mode != "simple":
                try:
                    logger.info(f"Researcher enabled intelligent research mode: {research_mode}")
                    # observation should already be the result of intelligent search
                    pass
                except Exception as e:
                    logger.error(f"Researcher intelligent search failed: {e}")
            
            # Store the source string if a search tool was called
            if configurable.include_source_str:
                source_str += cast(str, observation)
    
    # After processing all tools, decide what to do next
    state_update = {"messages": result}
    if completed_section:
        # Write the completed section to state and return to the supervisor
        logger.info(f"[ResearcherTools] Section research completed, returning to supervisor: {completed_section.name}")
        state_update["completed_sections"] = [completed_section]
    else:
        # If no section was completed but we're here, it might be due to search failures
        # Check if all searches failed and create a fallback section
        all_searches_failed = all(
            "Fetch error" in str(msg.get("content", "")) 
            for msg in result 
            if msg.get("role") == "tool" and "search" in msg.get("name", "").lower()
        )
        
        if all_searches_failed and len(result) > 0:
            # Extract section description from state
            section_description = "General topic"
            for message in reversed(state.get("messages", [])):
                if (isinstance(message, dict) and 
                    "Research and write the following section:" in message.get("content", "")):
                    content = message["content"]
                    section_start = content.find("Research and write the following section:") + len("Research and write the following section:")
                    section_description = content[section_start:].strip()
                    break
            
            # Create a fallback section with limited content
            fallback_section = Section(
                name=section_description,
                description=f"Section about {section_description}",
                content=f"""# {section_description}

*Note: This section was generated with limited information due to technical difficulties during research.*

## Overview

This section provides an analysis of {section_description} based on general knowledge and understanding.

## Key Points

Due to technical limitations during the research phase, this section presents a foundational understanding of the topic. For more detailed and up-to-date information, additional research may be required.

## Summary

{section_description} represents an important area of study that warrants further investigation. While comprehensive real-time data was not available during this analysis, the fundamental concepts and considerations outlined above provide a starting point for understanding this topic.

*Please note: This content was generated as a fallback due to research limitations and may not reflect the most current information available.*"""
            )
            logger.warning(f"[ResearcherTools] All searches failed, creating fallback section: {section_description}")
            state_update["completed_sections"] = [fallback_section]

    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str
        logger.debug(f"[ResearcherTools] Including source string, length: {len(source_str)} characters")

    logger.info(f"[ResearcherTools] Returning state update with keys: {state_update.keys()}")
    return state_update

async def research_agent_should_continue(state: AppState) -> str:
    """Decide if we should continue the loop or stop based on whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    last_tool_calls = _get_tool_calls(last_message)

    # If the model did not make a tool call, or called FinishResearch/FinishReport, then end the current research branch
    if not last_tool_calls:
        logger.info("[ResearcherControl] No tool calls made, checking if section needs fallback handling")
        
        # Check if we should create a fallback section due to search failures
        completed_sections = state.get("completed_sections", [])
        if not completed_sections:
            # Extract section info and create a minimal section
            section_description = "General topic"
            for message in reversed(messages):
                if (isinstance(message, dict) and 
                    "Research and write the following section:" in message.get("content", "")):
                    content = message["content"]
                    section_start = content.find("Research and write the following section:") + len("Research and write the following section:")
                    section_description = content[section_start:].strip()
                    break
            
            logger.warning(f"[ResearcherControl] Creating minimal section for: {section_description}")
            # The actual section creation will be handled by research_agent_tools
        
        logger.info("[ResearcherControl] Research task completed (no tools), returning to supervisor")
        return END
    elif last_tool_calls[0]["name"] in {"FinishResearch", "FinishReport"}:
        logger.info("[ResearcherControl] Research task explicitly finished, returning to supervisor")
        return END
    else:
        tool_names = [tc.get("name", "unknown") for tc in last_tool_calls]
        logger.info(f"[ResearcherControl] Research task continuing with tools: {tool_names}")
        return "research_agent_tools_node"
    
async def initialize_tools(state: AppState, config: RunnableConfig) -> AppState:
    """Initialize the tools for the supervisor and research agents in global cache."""
    logger.info("[InitializeTools] Starting tool initialization")
    # Pre-load tools into global cache to ensure they're available
    await tool_manager.get_supervisor_tools(config)
    await tool_manager.get_research_tools(config)
    logger.info("[InitializeTools] Tool initialization completed")
    return state


"""Build the multi-agent workflow"""

logger.info("Starting to build multi-agent workflow")

# Research agent workflow
logger.info("Building research agent workflow # Start")
research_builder = StateGraph(AppState, output=AppState, config_schema=MultiAgentConfiguration)
research_builder.add_node("research_agent", research_agent)
research_builder.add_node("research_agent_tools_node", research_agent_tools)
research_builder.add_edge(START, "research_agent") 
research_builder.add_conditional_edges(
    "research_agent",
    research_agent_should_continue,
    ["research_agent_tools_node", END]
)
research_builder.add_edge("research_agent_tools_node", "research_agent")
logger.info("Building research agent workflow # Completed")
# Supervisor workflow
logger.info("Building supervisor workflow # Start")
supervisor_builder = StateGraph(AppState, input=AppState, output=AppState, config_schema=MultiAgentConfiguration)
supervisor_builder.add_node("initialize_tools", initialize_tools)
supervisor_builder.add_node("supervisor", supervisor)
# LangGraph requires that node names and state keys be unique. You cannot have a node and a state key with the same name.
supervisor_builder.add_node("supervisor_tools_node", supervisor_tools)
supervisor_builder.add_node("research_team", research_builder.compile())

# Flow of the supervisor agent
supervisor_builder.add_edge(START, "initialize_tools")
supervisor_builder.add_edge("initialize_tools", "supervisor")
supervisor_builder.add_conditional_edges(
    "supervisor",
    supervisor_should_continue,
    ["supervisor_tools_node", END]
)
supervisor_builder.add_edge("research_team", "supervisor")
logger.info("Building supervisor workflow # Completed")
graph = supervisor_builder.compile()
logger.info("Multi-agent workflow construction completed")

# 工具管理器便捷函数
def clear_tool_cache():
    """清空全局工具缓存，强制重新初始化"""
    tool_manager.clear_cache()

def get_tool_cache_stats():
    """获取工具缓存统计信息"""
    with tool_manager._cache_lock:
        supervisor_count = len(tool_manager._supervisor_tools_cache)
        research_count = len(tool_manager._research_tools_cache)
    
    return {
        "supervisor_tools_cached": supervisor_count,
        "research_tools_cached": research_count,
        "total_cached_configs": supervisor_count + research_count
    }
