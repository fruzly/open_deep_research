from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)

from open_deep_research.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from open_deep_research.configuration import WorkflowConfiguration
from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search,
    get_today_str,
    intelligent_search_web_unified
)

import structlog
logger = structlog.get_logger(__name__)

## Nodes -- 

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Generate the initial report plan with sections.
    
    This node:
    1. Gets configuration for the report structure and search parameters
    2. Generates search queries to gather context for planning
    3. Performs web searches using those queries
    4. Uses an LLM to generate a structured plan with sections
    
    Args:
        state: Current graph state containing the report topic
        config: Configuration for models, search APIs, etc.
        
    Returns:
        Dict containing the generated sections
    """
    # Inputs
    topic = state["topic"]
    logger.info(f"Starting to generate report plan - Topic: {topic}")

    # Get list of feedback on the report plan
    feedback_list = state.get("feedback_on_report_plan", [])

    # Concatenate feedback on the report plan into a single string
    feedback = " /// ".join(feedback_list) if feedback_list else ""
    logger.debug(f"Feedback info - Feedback count: {len(feedback_list)}, Combined length: {len(feedback)}")

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters
    
    logger.debug(f"Configuration parameters - Number of queries: {number_of_queries}, Search API: {search_api}")

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Set writer model (model used for query writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    logger.debug(f"Writer model configuration - Provider: {writer_provider}, Model: {writer_model_name}")
    
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries,
        today=get_today_str()
    )

    # Generate queries  
    logger.debug("Starting to generate search queries")
    results = await structured_llm.ainvoke([SystemMessage(content=system_instructions_query),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

    # Web search
    query_list = [query.search_query for query in results.queries]
    logger.info(f"Query generation complete - Number of queries: {len(query_list)}")

    # 🧠 Intelligent search integration: Check if intelligent research mode is enabled
    research_mode = get_config_value(configurable.research_mode, "simple")
    
    if research_mode and research_mode != "simple":
        logger.info(f"🧠 Enabling intelligent research mode for graph report planning: {research_mode}")
        source_str = await intelligent_search_web_unified(query_list, config, search_api, params_to_pass)
    else:
        # Use traditional search
        logger.debug("Using traditional search mode")
        source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    logger.info(f"Search complete - Search result length: {len(source_str) if source_str else 0}")
    logger.info(f"Search results - {source_str}")

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str, feedback=feedback)

    # Set the planner
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})
    logger.debug(f"Planner model configuration - Provider: {planner_provider}, Model: {planner_model}")

    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, research, and content fields."""

    # Run the planner
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        logger.debug("Using Claude 3.7 Sonnet model, enabling thinking mode")
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider, 
                                      max_tokens=20_000, 
                                      thinking={"type": "enabled", "budget_tokens": 16_000})

    else:
        # With other models, thinking tokens are not specifically allocated
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider,
                                      model_kwargs=planner_model_kwargs)
    
    # Generate the report sections
    logger.debug("Starting to generate report sections")
    structured_llm = planner_llm.with_structured_output(Sections)
    report_sections = await structured_llm.ainvoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])

    # Get sections
    sections = report_sections.sections
    logger.info(f"Report plan generation complete - Number of sections: {len(sections)}")
    logger.info(f"Full report plan - {report_sections}")

    return {"sections": sections}

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """Get human feedback on the report plan and route to next steps.
    
    This node:
    1. Formats the current report plan for human review
    2. Gets feedback via an interrupt
    3. Routes to either:
       - Section writing if plan is approved
       - Plan regeneration if feedback is provided
    
    Args:
        state: Current graph state with sections to review
        config: Configuration for the workflow
        
    Returns:
        Command to either regenerate plan or start section writing
    """

    # Get sections
    topic = state["topic"]
    sections = state['sections']
    logger.info(f"Awaiting human feedback - Topic: {topic}, Number of sections: {len(sections)}")
    
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan from interrupt
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    
    logger.debug("Requesting feedback from the user")
    feedback = interrupt(interrupt_message)
    logger.info(f"Received user feedback - Type: {type(feedback)}")

    # If the user approves the report plan, kick off section writing
    if isinstance(feedback, bool) and feedback is True:
        # Treat this as approve and kick off section writing
        research_sections = [s for s in sections if s.research]
        logger.info(f"User approved the plan - Starting section writing, number of research sections: {len(research_sections)}")
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in research_sections
        ])
    
    # If the user provides feedback, regenerate the report plan 
    elif isinstance(feedback, str):
        # Treat this as feedback and append it to the existing list
        logger.info(f"User provided feedback - Regenerating plan, feedback length: {len(feedback)}")
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": [feedback]})
    else:
        logger.error(f"Unsupported feedback type: {type(feedback)}")
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")
    
async def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for researching a specific section.
    
    This node uses an LLM to generate targeted search queries based on the 
    section topic and description.
    
    Args:
        state: Current state containing section details
        config: Configuration including number of queries to generate
        
    Returns:
        Dict containing the generated search queries
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    logger.info(f"Starting to generate queries for section - Topic: {topic}, Section: {section.name}")

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries
    logger.debug(f"Query generation configuration - Number of queries: {number_of_queries}")

    # Generate queries 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    logger.debug(f"Query generation model - Provider: {writer_provider}, Model: {writer_model_name}")
    
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(topic=topic, 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries,
                                                           today=get_today_str())

    # Generate queries  
    logger.debug("Calling LLM to generate queries")
    queries = await structured_llm.ainvoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])

    logger.info(f"Section query generation complete - Number of queries generated: {len(queries.queries)}")
    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """Execute web searches for the section queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]
    current_iterations = state["search_iterations"]
    logger.info(f"Starting section search - Number of queries: {len(search_queries)}, Current iteration: {current_iterations}")

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters
    logger.debug(f"Search configuration - API: {search_api}, Parameters: {list(params_to_pass.keys()) if params_to_pass else 'None'}")

    # Web search
    query_list = [query.search_query for query in search_queries]

    # 🧠 Intelligent search integration: Check if intelligent research mode is enabled
    research_mode = get_config_value(configurable.research_mode, "simple")
    
    if research_mode and research_mode != "simple":
        logger.info(f"🧠 Enabling intelligent research mode for graph section search: {research_mode}")
        source_str = await intelligent_search_web_unified(query_list, config, search_api, params_to_pass)
    else:
        # Use traditional search
        logger.debug("Using traditional search mode")
        source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    new_iterations = current_iterations + 1
    logger.info(f"Section search complete - Search result length: {len(source_str) if source_str else 0}, New iteration count: {new_iterations}")
    return {"source_str": source_str, "search_iterations": new_iterations}

async def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report and evaluate if more research is needed.
    
    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete section or do more research
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]
    current_iterations = state["search_iterations"]
    
    logger.info(f"Starting to write section - Topic: {topic}, Section: {section.name}, Iteration: {current_iterations}")

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)
    max_search_depth = configurable.max_search_depth
    logger.debug(f"Writing configuration - Max search depth: {max_search_depth}")

    # Format system instructions
    section_writer_inputs_formatted = section_writer_inputs.format(topic=topic, 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    logger.debug(f"Section writing model - Provider: {writer_provider}, Model: {writer_model_name}")
    
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 

    logger.debug("Calling LLM to generate section content")
    section_content = await writer_model.ainvoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # Write content to the section object  
    section.content = section_content.content
    logger.info(f"Section content generation complete - Content length: {len(section.content) if section.content else 0}")

    # Grade prompt 
    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})
    logger.debug(f"Reflection model - Provider: {planner_provider}, Model: {planner_model}")

    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        logger.debug("Using Claude 3.7 Sonnet for reflection, enabling thinking mode")
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, 
                                           max_tokens=20_000, 
                                           thinking={"type": "enabled", "budget_tokens": 16_000}).with_structured_output(Feedback)
    else:
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, model_kwargs=planner_model_kwargs).with_structured_output(Feedback)
    
    # Generate feedback
    logger.debug("Starting to evaluate section quality")
    feedback = await reflection_model.ainvoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    logger.info(f"Section quality evaluation complete - Grade: {feedback.grade}")

    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or current_iterations >= max_search_depth:
        # Publish the section to completed sections 
        update = {"completed_sections": [section]}
        if configurable.include_source_str:
            update["source_str"] = source_str
        
        if feedback.grade == "pass":
            logger.info(f"Section complete - Quality passed, Section name: {section.name}")
        else:
            logger.info(f"Section complete - Max search depth {max_search_depth} reached, Section name: {section.name}")
        
        return Command(update=update, goto=END)

    # Update the existing section with new content and update search queries
    else:
        follow_up_count = len(feedback.follow_up_queries) if feedback.follow_up_queries else 0
        logger.info(f"Section needs more research - Number of follow-up queries: {follow_up_count}")
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="search_web"
        )
    
async def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.
    
    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.
    
    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model
        
    Returns:
        Dict containing the newly written section
    """

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Get state 
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    logger.info(f"Starting to write final section - Topic: {topic}, Section: {section.name}")
    logger.debug(f"Reference content length: {len(completed_report_sections) if completed_report_sections else 0}")
    
    # Format system instructions
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    logger.debug(f"Final section writing model - Provider: {writer_provider}, Model: {writer_model_name}")
    
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    
    logger.debug("Calling LLM to generate final section content")
    section_content = await writer_model.ainvoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # Write content to section 
    section.content = section_content.content
    logger.info(f"Final section generation complete - Section: {section.name}, Content length: {len(section.content) if section.content else 0}")

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """Format completed sections as context for writing final sections.
    
    This node takes all completed research sections and formats them into
    a single context string for writing summary sections.
    
    Args:
        state: Current state with completed sections
        
    Returns:
        Dict with formatted sections as context
    """

    # List of completed sections
    completed_sections = state["completed_sections"]
    logger.info(f"Gathering completed sections - Number of completed sections: {len(completed_sections)}")

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)
    logger.debug(f"Sections formatted - Formatted content length: {len(completed_report_sections) if completed_report_sections else 0}")

    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState, config: RunnableConfig):
    """Compile all sections into the final report.
    
    This node:
    1. Gets all completed sections
    2. Orders them according to original plan
    3. Combines them into the final report
    
    Args:
        state: Current state with all completed sections
        
    Returns:
        Dict containing the complete report
    """

    # Get configuration
    configurable = WorkflowConfiguration.from_runnable_config(config)

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}
    
    logger.info(f"Starting to compile final report - Total sections: {len(sections)}, Completed sections: {len(completed_sections)}")

    # Update sections with completed content while maintaining original order
    for section in sections:
        if section.name in completed_sections:
            section.content = completed_sections[section.name]
            logger.debug(f"Updating section content - Section: {section.name}")
        else:
            logger.warning(f"Missing section content - Section: {section.name}")

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections if s.content])
    logger.info(f"Final report compilation complete - Total report length: {len(all_sections)}")

    if configurable.include_source_str:
        source_str = state.get("source_str", "")
        logger.debug(f"Including source string - Source string length: {len(source_str) if source_str else 0}")
        return {"final_report": all_sections, "source_str": source_str}
    else:
        return {"final_report": all_sections}

def initiate_final_section_writing(state: ReportState):
    """Create parallel tasks for writing non-research sections.
    
    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one.
    
    Args:
        state: Current state with all sections and research context
        
    Returns:
        List of Send commands for parallel section writing
    """

    # Kick off section writing in parallel via Send() API for any sections that do not require research
    sections = state["sections"]
    final_sections = [s for s in sections if not s.research]
    logger.info(f"Initiating final section writing - Total sections: {len(sections)}, Final sections: {len(final_sections)}")
    
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in final_sections
    ]

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=WorkflowConfiguration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()
