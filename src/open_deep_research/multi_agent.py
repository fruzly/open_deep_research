from typing import List, Annotated, TypedDict, Literal, cast
from pydantic import BaseModel, Field
import operator
import warnings

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import MessagesState

from langgraph.types import Command, Send
from langgraph.graph import START, END, StateGraph

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


def fix_gemini_message_sequence(messages) -> list:
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


def ensure_user_message_ending(messages, default_content: str = "Please continue.") -> list:
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
    
    logger.info(f"🔍 [SearchTool] Configuring search API: {search_api}")

    # Return None if no search tool is requested
    if search_api.lower() == "none":
        logger.info("🚫 [SearchTool] Search function disabled")
        return None

    # TODO: Configure other search functions as tools
    if search_api.lower() == "tavily":
        # Create intelligent search wrapper
        @tool
        async def intelligent_tavily_search(queries: List[str]) -> str:
            """Perform intelligent search using Tavily search engine"""
            return await intelligent_search_web_unified(queries, config, "tavily")
        search_tool = intelligent_tavily_search
        
    elif search_api.lower() == "duckduckgo":
        # Create intelligent search wrapper
        @tool
        async def intelligent_duckduckgo_search(queries: List[str]) -> str:
            """Perform intelligent search using DuckDuckGo search engine"""
            return await intelligent_search_web_unified(queries, config, "duckduckgo")
        search_tool = intelligent_duckduckgo_search
    elif search_api.lower() == "googlesearch":
        # Create intelligent Google search wrapper
        @tool
        async def intelligent_google_search_tool(queries: List[str]) -> str:
            """Perform intelligent search using Google search engine"""
            return await intelligent_search_web_unified(queries, config, "googlesearch")
        search_tool = intelligent_google_search_tool
    elif search_api.lower() == "geminigooglesearch":
        @tool
        async def gemini_google_search_tool(queries: List[str]) -> str:
            """Perform intelligent search using Gemini Google search engine"""
            logger.info(f"🔍 [SearchTool] [geminigooglesearch] Performing Gemini Google search with queries: {queries}")
            # return await gemini_google_search(queries, config) # search + analysis
            return await intelligent_search_web_unified(queries, config, "geminigooglesearch")
        search_tool = gemini_google_search_tool
    elif search_api.lower() == "azureaisearch":
        @tool
        async def azureaisearch_search_tool(queries: List[str]) -> str:
            """Perform intelligent search using Azure AI search engine"""
            return await azureaisearch_search(queries, config)
        search_tool = azureaisearch_search_tool
    else:
        logger.error(f"❌ [SearchTool] Unsupported search API: {search_api}")
        raise NotImplementedError(
            f"The search API '{search_api}' is not yet supported in the multi-agent implementation. "
            f"Currently, only Tavily/DuckDuckGo/GoogleSearch/GeminiGoogleSearch/AzureAISearch/None is supported. Please use the graph-based implementation in "
            f"src/open_deep_research/graph.py for other search APIs, or set search_api to one of the supported options."
        )

    tool_metadata = {**(search_tool.metadata or {}), "type": "search"}
    search_tool.metadata = tool_metadata
    logger.info(f"✅ [SearchTool] Successfully created search tool: {search_tool.name}")
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
class ReportStateOutput(MessagesState):
    final_report: str # Final report
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: str # String of formatted source content from web search

class ReportState(MessagesState):
    sections: list[str] # List of report sections 
    completed_sections: Annotated[list[Section], operator.add] # Send() API key
    final_report: str # Final report
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: Annotated[str, operator.add] # String of formatted source content from web search

class SectionState(MessagesState):
    section: str # Report section  
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: str # String of formatted source content from web search

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: str # String of formatted source content from web search


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
async def get_supervisor_tools(config: RunnableConfig) -> list[BaseTool]:
    """Get supervisor tools based on configuration"""
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


async def get_research_tools(config: RunnableConfig) -> list[BaseTool]:
    """Get research tools based on configuration"""
    logger.info("🔧 [ResearcherTools] Starting to get research tools")
    search_tool = get_search_tool(config)
    tools = [tool(Section), tool(FinishResearch)]
    if search_tool is not None:
        tools.append(search_tool)  # Add search tool, if available
    existing_tool_names = {cast(BaseTool, tool).name for tool in tools}
    mcp_tools = await _load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    logger.info(f"🔧 [ResearcherTools] Research tools: {tools}")
    return tools


async def supervisor(state: ReportState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    
    logger.info("🎯 [Supervisor] Starting supervisor task execution")
    
    # Messages
    messages = state["messages"]
    logger.info(f"📨 [Supervisor] Received {len(messages)} messages")

    # Get configuration
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    supervisor_model = get_config_value(configurable.supervisor_model)
    logger.info(f"🤖 [Supervisor] Using model: {supervisor_model}")

    # Initialize the model
    llm = init_chat_model(model=supervisor_model)
    
    # If sections have been completed, but we don't yet have the final report, then we need to initiate writing the introduction and conclusion
    if state.get("completed_sections") and not state.get("final_report"):
        completed_count = len(state["completed_sections"])
        logger.info(f"📝 [Supervisor] Research completed, starting report writing. Finished {completed_count} sections")
        research_complete_message = {"role": "user", "content": "Research is complete. Now write the introduction and conclusion for the report. Here are the completed main body sections: \n\n" + "\n\n".join([s.content for s in state["completed_sections"]])}
        messages = messages + [research_complete_message]

    # Get tools based on configuration
    supervisor_tool_list = await get_supervisor_tools(config)
    logger.info(f"🔧 [Supervisor] Loaded {len(supervisor_tool_list)} tools: {[tool.name for tool in supervisor_tool_list]}")

    # 🔧 Convert LangChain message objects to dict format for message manager
    dict_messages = convert_langchain_messages_to_dict(messages)
    logger.debug(f"🔄 [Supervisor] Converted {len(dict_messages)} messages to dictionary format")

    # 🔧 Fix Gemini message sequence problem:
    # Invalid argument provided to Gemini: 400 Please ensure that function call turn comes immediately after a user turn or after a function response turn response turn is not allowed to be followed by a function call turn
    # Use global message manager to handle message sequence
    supervisor_model = get_config_value(configurable.supervisor_model)
    provider_hint = supervisor_model.split(":")[0] if ":" in supervisor_model else supervisor_model
    messages, fixes = validate_and_fix_messages(dict_messages, provider_hint)
    if fixes:
        logger.info(f"[Supervisor] Message sequence fixes: {', '.join(fixes)}")
    
    llm_with_tools = (
        llm
        .bind_tools(
            supervisor_tool_list,
            # parallel_tool_calls=False,
            # force at least one tool call
            tool_choice="any"
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

    # Invoke
    logger.info(f"🚀 [Supervisor] Invoking LLM with {len(llm_messages)} messages")
    logger.info()
    logger.info(llm_messages)
    logger.info()
    
    response = await llm_with_tools.ainvoke(llm_messages)
    logger.info(f"✅ [Supervisor] LLM response completed with {len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0} tool calls")
    
    return {
        "messages": [response]
    }

async def supervisor_tools(state: ReportState, config: RunnableConfig)  -> Command[Literal["supervisor", "research_team", "__end__"]]:
    """Performs the tool call and sends to the research agent"""
    logger.info("🔧 [SupervisorTools] Starting tool execution")
    
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None
    source_str = ""
    
    # Count tool calls
    tool_calls = state["messages"][-1].tool_calls if state["messages"] and hasattr(state["messages"][-1], 'tool_calls') else []
    logger.info(f"📋 [SupervisorTools] Need to execute {len(tool_calls)} tool calls: {[tc.get('name', 'unknown') for tc in tool_calls]}")

    # Get tools based on configuration
    supervisor_tool_list = await get_supervisor_tools(config)
    supervisor_tools_by_name = {tool.name: tool for tool in supervisor_tool_list}
    search_tool_names = {
        tool.name
        for tool in supervisor_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }

    # First process all tool calls to ensure we respond to each one (required for OpenAI)
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        logger.info(f"🛠️ [SupervisorTools] Executing tool: {tool_name}, args: {tool_call.get('args', {})}")
        
        # Get the tool
        tool = supervisor_tools_by_name[tool_name]
        # Perform the tool call - use ainvoke for async tools
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
            logger.info(f"✅ [SupervisorTools] Tool {tool_name} executed successfully")
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)
            logger.info(f"✅ [SupervisorTools] Tool {tool_name} executed successfully (sync)")
        except Exception as e:
            logger.error(f"❌ [SupervisorTools] Tool {tool_name} execution failed: {e}")
            raise

        # Append to messages 
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_call["name"], 
                       "tool_call_id": tool_call["id"]})
        
        # Store special tool results for processing after all tools have been called
        if tool_call["name"] == "Question":
            # Question tool was called - return to supervisor to ask the question
            question_obj = cast(Question, observation)
            logger.info(f"❓ [SupervisorTools] Generated question: {question_obj.question}")
            result.append({"role": "assistant", "content": question_obj.question})
            return Command(goto=END, update={"messages": result})
        elif tool_call["name"] == "Sections":
            sections_list = cast(Sections, observation).sections
            logger.info(f"📋 [SupervisorTools] Generated {len(sections_list)} sections: {sections_list}")
        elif tool_call["name"] == "Introduction":
            # Format introduction with proper H1 heading if not already formatted
            observation = cast(Introduction, observation)
            if not observation.content.startswith("# "):
                intro_content = f"# {observation.name}\n\n{observation.content}"
            else:
                intro_content = observation.content
            logger.info(f"📝 [SupervisorTools] Generated introduction: {observation.name}")
        elif tool_call["name"] == "Conclusion":
            # Format conclusion with proper H2 heading if not already formatted
            observation = cast(Conclusion, observation)
            if not observation.content.startswith("## "):
                conclusion_content = f"## {observation.name}\n\n{observation.content}"
            else:
                conclusion_content = observation.content
            logger.info(f"🏁 [SupervisorTools] Generated conclusion: {observation.name}")
        elif tool_call["name"] in search_tool_names:
            # 🧠 Intelligent search integration: Check if intelligent research mode is enabled
            research_mode = get_config_value(configurable.research_mode, "simple")
            
            if research_mode and research_mode != "simple":
                # Use intelligent search interface
                try:
                    logger.info(f"🧠 Supervisor enabled intelligent research mode: {research_mode}")
                    # observation should already be the result of intelligent search
                    pass
                except Exception as e:
                    logger.error(f"⚠️ Supervisor intelligent search failed: {e}")
            
            if configurable.include_source_str:
                source_str += cast(str, observation)

    # After processing all tool calls, decide what to do next
    if sections_list:
        # Send the sections to the research agents
        logger.info(f"🎯 [SupervisorTools] Assigning {len(sections_list)} sections to research team")
        return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], update={"messages": result})
    elif intro_content:
        # Store introduction while waiting for conclusion
        # Append to messages to guide the LLM to write conclusion next
        logger.info("📝 [SupervisorTools] Introduction completed, waiting for conclusion")
        result.append({"role": "user", "content": "Introduction written. Now write a conclusion section."})
        state_update = {
            "final_report": intro_content,
            "messages": result,
        }
    elif conclusion_content:
        # Get all sections and combine in proper order: Introduction, Body Sections, Conclusion
        intro = state.get("final_report", "")
        body_sections = "\n\n".join([s.content for s in state["completed_sections"]])
        
        # Assemble final report in correct order
        complete_report = f"{intro}\n\n{body_sections}\n\n{conclusion_content}"
        
        logger.info(f"🎉 [SupervisorTools] Report completed! Total length: {len(complete_report)} characters")
        
        # Append to messages to indicate completion
        result.append({"role": "user", "content": "Report is now complete with introduction, body sections, and conclusion."})

        state_update = {
            "final_report": complete_report,
            "messages": result,
        }
    else:
        # Default case (for search tools, etc.)
        state_update = {"messages": result}

    # Include source string for evaluation
    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str

    return Command(goto="supervisor", update=state_update)

async def supervisor_should_continue(state: ReportState) -> str:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    
    # End because the supervisor asked a question or is finished
    if not last_message.tool_calls or (len(last_message.tool_calls) == 1 and last_message.tool_calls[0]["name"] == "FinishReport"):
        # Exit the graph
        logger.info("🏁 [SupervisorControl] Supervisor decided to end process")
        return END

    # If the LLM makes a tool call, then perform an action
    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
    logger.info(f"➡️ [SupervisorControl] Supervisor continuing with tools: {tool_names}")
    return "supervisor_tools"

async def research_agent(state: SectionState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    
    section_name = state.get("section", "Unknown section")
    logger.info(f"🔬 [Researcher] Starting research for section: {section_name}")
    
    # Get configuration
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    researcher_model = get_config_value(configurable.researcher_model)
    logger.info(f"🤖 [Researcher] Using model: {researcher_model}")
    
    # Initialize the model
    llm = init_chat_model(model=researcher_model)

    # Get tools based on configuration
    research_tool_list = await get_research_tools(config)
    logger.info(f"🔧 [Researcher] Loaded {len(research_tool_list)} tools: {[tool.name for tool in research_tool_list]}")
    
    system_prompt = RESEARCH_INSTRUCTIONS.format(
        section_description=state["section"],
        number_of_queries=configurable.number_of_queries,
        today=get_today_str(),
    )
    logger.debug(f"📋 [Researcher] System prompt length: {len(system_prompt)} characters")
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # Ensure we have at least one user message (required by Anthropic)
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": f"Please research and write the section: {state['section']}"}]
        logger.info("📝 [Researcher] Created initial research message")
    
    logger.info(f"📨 [Researcher] Processing {len(messages)} messages")

    # 🔧 Convert LangChain message objects to dict format for message manager
    dict_messages = convert_langchain_messages_to_dict(messages)

    # 🔧 Fix Gemini message sequence problem:
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

    logger.info(f"🚀 [Researcher] Invoking LLM with {len(llm_messages)} messages")
    
    response = await llm.bind_tools(research_tool_list,             
                          #  parallel_tool_calls=False,
                           # force at least one tool call
                           tool_choice="any").ainvoke(llm_messages)
    
    tool_calls_count = len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0
    logger.info(f"✅ [Researcher] LLM response completed with {tool_calls_count} tool calls")
    
    result = {
        "messages": [response]
    }
    
    logger.debug(f"📤 [Researcher] Returning result with {len(result['messages'])} messages")
    
    return result

async def research_agent_tools(state: SectionState, config: RunnableConfig):
    """Performs the tool call and route to supervisor or continue the research loop"""
    logger.info("🔧 [ResearcherTools] Starting research tool execution")
    
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result = []
    completed_section = None
    source_str = ""
    
    # Count tool calls
    tool_calls = state["messages"][-1].tool_calls if state["messages"] and hasattr(state["messages"][-1], 'tool_calls') else []
    logger.info(f"📋 [ResearcherTools] Need to execute {len(tool_calls)} tool calls: {[tc.get('name', 'unknown') for tc in tool_calls]}")
    
    # Get tools based on configuration
    research_tool_list = await get_research_tools(config)
    research_tools_by_name = {tool.name: tool for tool in research_tool_list}
    logger.info(f"🔧 [ResearcherTools] Research tool names: {research_tools_by_name.keys()}")
    search_tool_names = {
        tool.name
        for tool in research_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }
    logger.info(f"🔧 [ResearcherTools] Search tools: {search_tool_names}")
    
    # Process all tool calls first (required for OpenAI)
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        logger.info(f"🛠️ [ResearcherTools] Executing tool: {tool_name}, args: {tool_call.get('args', {})}")
        
        # Get the tool
        tool = research_tools_by_name[tool_name]
        # Perform the tool call - use ainvoke for async tools
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
            logger.info(f"✅ [ResearcherTools] Tool {tool_name} executed successfully")
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)
            logger.info(f"✅ [ResearcherTools] Tool {tool_name} executed successfully (sync)")
        except Exception as e:
            logger.error(f"❌ [ResearcherTools] Tool {tool_name} execution failed: {e}")
            raise

        # Append to messages 
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_call["name"], 
                       "tool_call_id": tool_call["id"]})
        
        # Store the section observation if a Section tool was called
        if tool_call["name"] == "Section":
            completed_section = cast(Section, observation)
            logger.info(f"📝 [ResearcherTools] Completed section: {completed_section.name}, content length: {len(completed_section.content)} characters")

        # 🧠 Intelligent search integration: Check if intelligent research mode is enabled
        if tool_call["name"] in search_tool_names:
            research_mode = get_config_value(configurable.research_mode, "simple")
            logger.info(f"research_mode: {research_mode}")
            if research_mode and research_mode != "simple":
                try:
                    logger.info(f"🧠 Researcher enabled intelligent research mode: {research_mode}")
                    # observation should already be the result of intelligent search
                    pass
                except Exception as e:
                    logger.error(f"⚠️ Researcher intelligent search failed: {e}")
            
            # Store the source string if a search tool was called
            if configurable.include_source_str:
                source_str += cast(str, observation)
    
    # After processing all tools, decide what to do next
    state_update = {"messages": result}
    if completed_section:
        # Write the completed section to state and return to the supervisor
        logger.info(f"🎯 [ResearcherTools] Section research completed, returning to supervisor: {completed_section.name}")
        state_update["completed_sections"] = [completed_section]
    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str
        logger.debug(f"📊 [ResearcherTools] Including source string, length: {len(source_str)} characters")

    logger.info(f"📤 [ResearcherTools] Returning state update with {len(state_update)} fields")
    return state_update

async def research_agent_should_continue(state: SectionState) -> str:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    section_name = state.get("section", "Unknown section")
    
    # If the model did not make a tool call, or called FinishResearch, then end the current research branch
    if not last_message.tool_calls or last_message.tool_calls[0]["name"] == "FinishResearch":
        # Research is done - return to supervisor
        logger.info(f"🏁 [ResearcherControl] Section '{section_name}' research completed, returning to supervisor")
        return END
    else:
        tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
        logger.info(f"➡️ [ResearcherControl] Section '{section_name}' continuing with tools: {tool_names}")
        return "research_agent_tools"
    
"""Build the multi-agent workflow"""

logger.info("🔧 [ResearchBuilder] Starting to build multi-agent workflow")

# Research agent workflow
logger.info("📊 [ResearchBuilder] Building research agent workflow # Start")
research_builder = StateGraph(SectionState, output=SectionOutputState, config_schema=MultiAgentConfiguration)
research_builder.add_node("research_agent", research_agent)
research_builder.add_node("research_agent_tools", research_agent_tools)
research_builder.add_edge(START, "research_agent") 
research_builder.add_conditional_edges(
    "research_agent",
    research_agent_should_continue,
    ["research_agent_tools", END]
)
research_builder.add_edge("research_agent_tools", "research_agent")
logger.info("📊 [ResearchBuilder] Building research agent workflow # Completed")
# Supervisor workflow
logger.info("👑 [SupervisorBuilder] Building supervisor workflow # Start")
supervisor_builder = StateGraph(ReportState, input=MessagesState, output=ReportStateOutput, config_schema=MultiAgentConfiguration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("research_team", research_builder.compile())

# Flow of the supervisor agent
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_conditional_edges(
    "supervisor",
    supervisor_should_continue,
    ["supervisor_tools", END]
)
supervisor_builder.add_edge("research_team", "supervisor")
logger.info("👑 [SupervisorBuilder] Building supervisor workflow # Completed")
graph = supervisor_builder.compile()
logger.info("✅ [MultiAgentBuilder] Multi-agent workflow construction completed")

def debug_next_ainvoke(agent, new_msg, config):
    """Wrap an ainvoke call and print the final message list entering the LLM"""
    state_before = agent.get_state(config).values.get('messages', [])
    logger.info(f"📋 Existing message count: {len(state_before)}")

    # ➜ Convert LangChain Message to dict
    dict_msgs = convert_langchain_messages_to_dict(state_before)
    # ➜ Run validate_and_fix_messages again to see what else can be fixed
    fixed, fixes = validate_and_fix_messages(dict_msgs, "google_genai")
    logger.info("🛠 validate_and_fix_messages fixes:", fixes)

    # ➜ Find entries that are still empty
    for idx, m in enumerate(fixed):
        if (not m.get('content')) or (not str(m['content']).strip()):
            logger.info(f"❗ Empty content at index={idx}, role={m['role']}, msg={m}")

    # ➜ Print final list for manual inspection
    for i, m in enumerate(fixed):
        logger.info(f"{i:02d} {m['role']}: {repr(m['content'])[:80]}")

    # ➜ Actually invoke (if you want to continue execution)
    return agent.ainvoke({"messages": new_msg}, config=config)