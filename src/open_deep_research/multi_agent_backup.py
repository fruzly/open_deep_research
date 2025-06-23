from typing import List, Annotated, TypedDict, Literal, cast
from pydantic import BaseModel, Field
import operator
import warnings
import os

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
    get_today_str,
)

from open_deep_research.prompts import SUPERVISOR_INSTRUCTIONS, RESEARCH_INSTRUCTIONS

## Tools factory - will be initialized based on configuration
def get_search_tool(config: RunnableConfig):
    """Get the appropriate search tool based on configuration"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)

    # Return None if no search tool is requested
    if search_api.lower() == "none":
        return None

    # TODO: Configure other search functions as tools
    if search_api.lower() == "tavily":
        search_tool = tavily_search
    elif search_api.lower() == "duckduckgo":
        search_tool = duckduckgo_search
    else:
        raise NotImplementedError(
            f"The search API '{search_api}' is not yet supported in the multi-agent implementation. "
            f"Currently, only Tavily/DuckDuckGo/None is supported. Please use the graph-based implementation in "
            f"src/open_deep_research/graph.py for other search APIs, or set search_api to 'tavily', 'duckduckgo', or 'none'."
        )

    tool_metadata = {**(search_tool.metadata or {}), "type": "search"}
    search_tool.metadata = tool_metadata
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
    search_tool = get_search_tool(config)
    tools = [tool(Section), tool(FinishResearch)]
    if search_tool is not None:
        tools.append(search_tool)  # Add search tool, if available
    existing_tool_names = {cast(BaseTool, tool).name for tool in tools}
    mcp_tools = await _load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools


async def supervisor(state: ReportState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""

    # Messages
    messages = state["messages"]

    # Get configuration
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    supervisor_model = get_config_value(configurable.supervisor_model)

    # Initialize the model
    llm = init_chat_model(model=supervisor_model)
    
    # 🔧 检查研究是否完成，如果完成则触发报告写作
    sections = state.get("sections", [])
    completed_sections = state.get("completed_sections", [])
    
    # If sections have been completed, but we don't yet have the final report, then we need to initiate writing the introduction and conclusion
    if completed_sections and not state.get("final_report"):
        research_complete_message = {"role": "user", "content": "Research is complete. Now write the introduction and conclusion for the report. Here are the completed main body sections: \n\n" + "\n\n".join([s.content for s in completed_sections])}
        messages = messages + [research_complete_message]

    # Get tools based on configuration
    supervisor_tool_list = await get_supervisor_tools(config)
    
    
    # Get system prompt
    system_prompt = SUPERVISOR_INSTRUCTIONS.format(today=get_today_str())
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # Check if this is a Google Gemini model for special handling
    is_gemini = "gemini" in supervisor_model.lower() or "google" in supervisor_model.lower()
    
    if is_gemini:
        DEBUG_MESSAGES = os.environ.get("DEBUG_MULTI_AGENT", "").lower() == "true"
        
        if DEBUG_MESSAGES:
            print(f"🔧 Improved Gemini supervisor message handling")
            print(f"🔧 Current state: sections={len(sections)}, completed={len(completed_sections)}, final_report={bool(state.get('final_report'))}")
        
        # Improved Gemini processing logic: build appropriate messages based on workflow state
        current_state_info = ""
        
        # 🔧 改进的状态检查逻辑
        if not sections:
            # Stage 1: Need to create research plan
            current_state_info = "Research planning stage: Please analyze the topic and define the sections to be researched."
        elif sections and len(completed_sections) < len(sections):
            # Stage 1.5: Research in progress
            current_state_info = f"Research in progress: {len(completed_sections)}/{len(sections)} sections completed, waiting for remaining research."
        elif len(completed_sections) >= len(sections) and not state.get("final_report"):
            # Stage 2: Research completed, need to write introduction and conclusion
            current_state_info = f"Report writing stage: {len(completed_sections)} research sections completed, now need to write introduction and conclusion."
        elif state.get("final_report"):
            # Stage 3: Work completed
            current_state_info = "Work completion stage: Report has been generated, please call FinishReport tool."
        else:
            # Other cases
            current_state_info = "Workflow in progress, please select appropriate operation based on current state."
        
        # Extract original question
        original_question = ""
        if messages:
            first_msg = messages[0]
            if isinstance(first_msg, dict):
                original_question = first_msg.get("content", "")
            elif hasattr(first_msg, 'content'):
                original_question = first_msg.content
            else:
                original_question = str(first_msg)
        
        # Build improved message sequence
        simplified_messages = [{
            "role": "user", 
            "content": f"""System prompt: {system_prompt}

Original question: {original_question}

Current workflow state: {current_state_info}

Please select appropriate tools to operate based on current state. If work is completed, please call FinishReport tool."""
        }]
        
        # Adjust tool selection strategy based on workflow state
        if state.get("final_report"):
            # If report is completed, do not force tool calls
            llm_with_tools = llm.bind_tools(supervisor_tool_list, tool_choice="auto")
        else:
            # When work is not completed, encourage tool calls
            llm_with_tools = llm.bind_tools(supervisor_tool_list, tool_choice="any")
        
        final_messages = simplified_messages
    else:
        llm_with_tools = llm.bind_tools(
            supervisor_tool_list,
            parallel_tool_calls=False,
            # force at least one tool call
            tool_choice="any"
        )
        
        final_messages = [{"role": "system", "content": system_prompt}] + messages

    # Invoke
    return {
        "messages": [
            await llm_with_tools.ainvoke(final_messages)
        ]
    }

async def supervisor_tools(state: ReportState, config: RunnableConfig)  -> Command[Literal["supervisor", "research_team", "__end__"]]:
    """Performs the tool call and sends to the research agent"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None
    source_str = ""

    # Get tools based on configuration
    supervisor_tool_list = await get_supervisor_tools(config)
    supervisor_tools_by_name = {tool.name: tool for tool in supervisor_tool_list}
    search_tool_names = {
        tool.name
        for tool in supervisor_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }

    # 🔧 添加调试信息
    DEBUG_MESSAGES = os.environ.get("DEBUG_MULTI_AGENT", "").lower() == "true"
    if DEBUG_MESSAGES:
        print(f"🔧 Supervisor tools available: {list(supervisor_tools_by_name.keys())}")
        if state["messages"]:
            last_msg = state["messages"][-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"🔧 Supervisor tool calls to process: {[tc['name'] for tc in last_msg.tool_calls]}")

    # First process all tool calls to ensure we respond to each one (required for OpenAI)
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        
        # 🔧 添加工具存在性验证
        if tool_name not in supervisor_tools_by_name:
            error_message = f"❌ Supervisor tool '{tool_name}' not found in available tools: {list(supervisor_tools_by_name.keys())}"
            print(error_message)
            
            # 返回错误消息而不是抛出异常
            result.append({
                "role": "tool", 
                "content": f"Error: Tool '{tool_name}' is not available. Available tools are: {', '.join(supervisor_tools_by_name.keys())}", 
                "name": tool_name, 
                "tool_call_id": tool_call["id"]
            })
            continue
        
        # Get the tool
        tool = supervisor_tools_by_name[tool_name]
        # Perform the tool call - use ainvoke for async tools
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)

        # Append to messages 
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_name, 
                       "tool_call_id": tool_call["id"]})
        
        # Store special tool results for processing after all tools have been called
        if tool_name == "Question":
            # Question tool was called - return to supervisor to ask the question
            question_obj = cast(Question, observation)
            result.append({"role": "assistant", "content": question_obj.question})
            return Command(goto=END, update={"messages": result})
        elif tool_name == "Sections":
            sections_list = cast(Sections, observation).sections
        elif tool_name == "Introduction":
            # Format introduction with proper H1 heading if not already formatted
            observation = cast(Introduction, observation)
            if not observation.content.startswith("# "):
                intro_content = f"# {observation.name}\n\n{observation.content}"
            else:
                intro_content = observation.content
        elif tool_name == "Conclusion":
            # Format conclusion with proper H2 heading if not already formatted
            observation = cast(Conclusion, observation)
            if not observation.content.startswith("## "):
                conclusion_content = f"## {observation.name}\n\n{observation.content}"
            else:
                conclusion_content = observation.content
        elif tool_name in search_tool_names and configurable.include_source_str:
            source_str += cast(str, observation)

    # After processing all tool calls, decide what to do next
    if sections_list:
        # 🔧 修复：确保sections被存储到状态中
        if DEBUG_MESSAGES:
            print(f"🔧 Sending sections to research team: {sections_list}")
        # Send the sections to the research agents
        return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], update={"messages": result, "sections": sections_list})
    elif intro_content:
        # Store introduction while waiting for conclusion
        # Append to messages to guide the LLM to write conclusion next
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
    """Improved termination condition logic"""
    
    messages = state["messages"]
    if not messages:
        return END
        
    last_message = messages[-1]
    
    # 🔧 增强调试信息
    DEBUG_MESSAGES = os.environ.get("DEBUG_MULTI_AGENT", "").lower() == "true"
    if DEBUG_MESSAGES:
        sections_count = len(state.get("sections", []))
        completed_count = len(state.get("completed_sections", []))
        has_final_report = bool(state.get("final_report"))
        print(f"🔧 Supervisor decision: sections={sections_count}, completed={completed_count}, final_report={has_final_report}")
        print(f"🔧 Last message type: {type(last_message).__name__}")
        if hasattr(last_message, 'tool_calls'):
            print(f"🔧 Has tool calls: {bool(last_message.tool_calls)}")
            if last_message.tool_calls:
                print(f"🔧 Tool calls: {[tc['name'] for tc in last_message.tool_calls]}")
    
    # Check if there are tool calls
    has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
    
    # Case 1: Explicitly call FinishReport tool
    if has_tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "FinishReport":
                if DEBUG_MESSAGES:
                    print("🔧 FinishReport tool called, ending workflow")
                return END
    
    # Case 2: 检查是否应该结束工作流的其他条件
    if not has_tool_calls:
        if DEBUG_MESSAGES:
            print("🔧 No tool calls in last message")
        
        # 如果有最终报告，应该结束
        if state.get("final_report"):
            if DEBUG_MESSAGES:
                print("🔧 Final report exists, ending workflow")
            return END
        
        # 🔧 如果没有sections且没有tool calls，可能陷入循环，强制结束
        if not state.get("sections") and len(messages) > 5:
            if DEBUG_MESSAGES:
                print("🔧 No sections defined after multiple messages, ending to prevent recursion")
            return END
        
        # 如果没有工具调用且没有最终报告，可能需要继续，但要防止无限循环
        return END
    
    # Case 3: Has tool calls but not FinishReport, continue processing
    if DEBUG_MESSAGES:
        print("🔧 Continuing to supervisor_tools")
    return "supervisor_tools"

async def research_agent(state: SectionState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    
    # Get configuration
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    researcher_model = get_config_value(configurable.researcher_model)
    
    # Initialize the model
    llm = init_chat_model(model=researcher_model)

    # Get tools based on configuration
    research_tool_list = await get_research_tools(config)
    
    # 🔧 检查是否有搜索工具可用
    has_search_tool = any(
        tool.metadata is not None and tool.metadata.get("type") == "search" 
        for tool in research_tool_list
    )
    
    # 🔧 根据实际可用工具动态调整系统提示词
    if has_search_tool:
        system_prompt = RESEARCH_INSTRUCTIONS.format(
            section_description=state["section"],
            number_of_queries=configurable.number_of_queries,
            today=get_today_str(),
        )
    else:
        # 当没有搜索工具时，使用修改版的提示词
        system_prompt = f"""
You are a researcher responsible for completing a specific section of a report.

<Section Description>
{state["section"]}
</Section Description>

### Your Task:
Since no search tools are available, you need to write the section based on your existing knowledge.

**REQUIRED: Two-Step Completion Process**
You MUST complete your work in exactly two steps:

**Step 1: Write Your Section**
- Call the Section tool to write your section based on your existing knowledge
- The Section tool parameters are:
  - `name`: The title of the section
  - `description`: The scope of research you completed (brief, 1-2 sentences)
  - `content`: The completed body of text for the section, which MUST:
    - Begin with the section title formatted as "## [Section Title]" (H2 level with ##)
    - Be formatted in Markdown style
    - Be MAXIMUM 200 words (strictly enforce this limit)
    - Use clear, concise language with bullet points where appropriate
    - Include relevant facts and general knowledge about the topic

**Step 2: Signal Completion**
- Immediately after calling the Section tool, call the FinishResearch tool
- This signals that your research work is complete and the section is ready

### Notes:
- **CRITICAL**: You MUST call the Section tool to complete your work - this is not optional
- Write based on your general knowledge since no search tools are available
- Keep a professional, factual tone
- Always follow markdown formatting
- Stay within the 200 word limit for the main content

Today is {get_today_str()}
"""
    
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # Ensure we have at least one user message (required by Anthropic)
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": f"Please research and write the section: {state['section']}"}]

    # Check if this is a Google Gemini model for special handling
    is_gemini = "gemini" in researcher_model.lower() or "google" in researcher_model.lower()
    
    if is_gemini:
        DEBUG_MESSAGES = os.environ.get("DEBUG_MULTI_AGENT", "").lower() == "true"
        
        if DEBUG_MESSAGES:
            print(f"🔧 Improved Gemini research agent message handling")
            print(f"🔧 Research section: {state['section']}")
            print(f"🔧 Has search tool: {has_search_tool}")
        
        # 🔧 改进的研究智能体消息处理
        # 检查是否有之前的研究工作
        has_previous_research = len(messages) > 1
        
        # 🔧 根据工具可用性构建不同的任务描述
        if has_search_tool:
            if has_previous_research:
                research_progress = f"Continue researching section: {state['section']}. Based on previous conversation, please decide whether to continue searching for more information or start writing section content."
            else:
                research_progress = f"Start researching section: {state['section']}. Please search for relevant information first, then write complete section content."
            
            tool_instructions = """Please select appropriate tools:
1. If more information is needed, use available search tools
2. If information is sufficient, use Section tool to write content
3. If unable to continue, use FinishResearch tool"""
        else:
            # 没有搜索工具时的指令
            research_progress = f"Write section: {state['section']}. Since no search tools are available, write the section based on your existing knowledge."
            
            tool_instructions = """Please select appropriate tools:
1. Use Section tool to write content based on your knowledge
2. Then use FinishResearch tool to complete the task"""
        
        simplified_messages = [{
            "role": "user", 
            "content": f"""System prompt: {system_prompt}

Task: {research_progress}

{tool_instructions}"""
        }]
        
        # 智能工具选择策略
        llm_with_tools = llm.bind_tools(research_tool_list, tool_choice="auto")
        final_messages = simplified_messages
    else:
        llm_with_tools = llm.bind_tools(
            research_tool_list,
            parallel_tool_calls=False,
            # force at least one tool call
            tool_choice="any"
        )
        
        final_messages = [{"role": "system", "content": system_prompt}] + messages

    return {
        "messages": [
            # Enforce tool calling to either perform more search or call the Section tool to write the section
            await llm_with_tools.ainvoke(final_messages)
        ]
    }

async def research_agent_tools(state: SectionState, config: RunnableConfig):
    """Performs the tool call and route to supervisor or continue the research loop"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result = []
    completed_section = None
    source_str = ""
    
    # Get tools based on configuration
    research_tool_list = await get_research_tools(config)
    research_tools_by_name = {tool.name: tool for tool in research_tool_list}
    search_tool_names = {
        tool.name
        for tool in research_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }
    
    # 🔧 添加调试信息
    DEBUG_MESSAGES = os.environ.get("DEBUG_MULTI_AGENT", "").lower() == "true"
    if DEBUG_MESSAGES:
        print(f"🔧 Available tools: {list(research_tools_by_name.keys())}")
        if state["messages"]:
            last_msg = state["messages"][-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"🔧 Tool calls to process: {[tc['name'] for tc in last_msg.tool_calls]}")
    
    # Process all tool calls first (required for OpenAI)
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        
        # 🔧 添加工具存在性验证
        if tool_name not in research_tools_by_name:
            error_message = f"❌ Tool '{tool_name}' not found in available tools: {list(research_tools_by_name.keys())}"
            print(error_message)
            
            # 返回错误消息而不是抛出异常
            result.append({
                "role": "tool", 
                "content": f"Error: Tool '{tool_name}' is not available. Available tools are: {', '.join(research_tools_by_name.keys())}", 
                "name": tool_name, 
                "tool_call_id": tool_call["id"]
            })
            continue
        
        # Get the tool
        tool = research_tools_by_name[tool_name]
        # Perform the tool call - use ainvoke for async tools
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)

        # Append to messages 
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_name, 
                       "tool_call_id": tool_call["id"]})
        
        # Store the section observation if a Section tool was called
        if tool_name == "Section":
            if DEBUG_MESSAGES:
                print(f"🔧 Section tool called, processing observation...")
                print(f"🔧 Observation type: {type(observation)}")
                print(f"🔧 Observation content: {observation}")
            completed_section = cast(Section, observation)
            if DEBUG_MESSAGES:
                print(f"🔧 Cast result type: {type(completed_section)}")
                print(f"🔧 Cast result: {completed_section}")

        # Store the source string if a search tool was called
        if tool_name in search_tool_names and configurable.include_source_str:
            source_str += cast(str, observation)
    
    # After processing all tools, decide what to do next
    state_update = {"messages": result}
    if completed_section:
        # Write the completed section to state and return to the supervisor
        if DEBUG_MESSAGES:
            print(f"🔧 Research completed for section: {completed_section.name}")
            print(f"🔧 Adding to completed_sections: {completed_section}")
        state_update["completed_sections"] = [completed_section]
    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str
    
    if DEBUG_MESSAGES:
        print(f"🔧 Research agent returning state_update: {list(state_update.keys())}")

    return state_update

async def research_agent_should_continue(state: SectionState) -> str:
    """Improved research agent termination condition logic"""

    messages = state["messages"]
    if not messages:
        return END
        
    last_message = messages[-1]
    
    # 🔧 增强调试信息
    DEBUG_MESSAGES = os.environ.get("DEBUG_MULTI_AGENT", "").lower() == "true"
    if DEBUG_MESSAGES:
        print(f"🔧 Research agent decision for section: {state.get('section', 'unknown')}")
        print(f"🔧 Messages count: {len(messages)}")
        print(f"🔧 Last message type: {type(last_message).__name__}")

    # Check if there are tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if DEBUG_MESSAGES:
                print(f"🔧 Processing tool call: {tool_call['name']}")
            
            if tool_call["name"] == "FinishResearch":
                if DEBUG_MESSAGES:
                    print("🔧 FinishResearch called, ending research")
                return END
            elif tool_call["name"] == "Section":
                if DEBUG_MESSAGES:
                    print("🔧 Section tool called, ending research")
                return END
        
        # Has tool calls but not terminal ones, continue processing
        if DEBUG_MESSAGES:
            print("🔧 Non-terminal tool calls, continuing to research_agent_tools")
        return "research_agent_tools"
    else:
        # No tool calls, 可能出现问题
        if DEBUG_MESSAGES:
            print("🔧 No tool calls, ending to prevent infinite loop")
        
        # 🔧 防止无限循环：如果消息很多但没有工具调用，强制结束
        if len(messages) > 3:
            if DEBUG_MESSAGES:
                print("🔧 Too many messages without tool calls, forcing END")
            return END
        
        return END
    
"""Build the multi-agent workflow"""

# Research agent workflow
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

# Supervisor workflow
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

graph = supervisor_builder.compile()