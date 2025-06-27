"""
Reflection and Evaluation Engine

Implements the core reflection mechanism from the reference project:
- Assesses the quality and completeness of search results
- Identifies knowledge gaps
- Determines whether further research is needed
- Prepares for the extended thinking mode of Supervisor 3.0
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
from dataclasses import dataclass
from enum import Enum

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Add logging configuration
import structlog
logger = structlog.get_logger(__name__)

from open_deep_research.intelligent_research.core import ResearchState
from open_deep_research.intelligent_research.prompts import (
    INTELLIGENT_REFLECTION_PROMPT,
    KNOWLEDGE_GAP_IDENTIFICATION_PROMPT,
    get_research_prompt
)
from open_deep_research.utils import get_config_value, get_today_str, extract_json_from_markdown

class ReflectionQuality(Enum):
    """Reflection quality levels"""
    INSUFFICIENT = "insufficient"  # Insufficient information
    PARTIAL = "partial"           # Partially satisfied
    ADEQUATE = "adequate"         # Basically satisfied
    COMPREHENSIVE = "comprehensive"  # Comprehensively satisfied


class ReflectionResult(BaseModel):
    """Reflection evaluation result"""
    quality: ReflectionQuality = Field(description="Information quality assessment")
    completeness_score: float = Field(description="Completeness score (0-1)")
    knowledge_gaps: List[str] = Field(description="Identified knowledge gaps")
    continue_research: bool = Field(description="Whether to continue research")
    reasoning: str = Field(description="Reasoning process for the assessment")
    suggested_focus: Optional[str] = Field(description="Suggested research focus", default=None)


@dataclass
class ReflectionContext:
    """Reflection evaluation context"""
    original_query: str
    current_iteration: int
    search_results: List[Dict[str, Any]]
    previous_reflections: List[ReflectionResult]
    search_queries: List[str] = None
    research_objective: Optional[str] = None


class ReflectionEngine:
    """
    Reflection and Evaluation Engine
    
    Based on the reflection mechanism from gemini-fullstack-langgraph-quickstart:
    1. Analyzes the quality of search results
    2. Identifies information gaps
    3. Assesses research completeness
    4. Decides whether to continue research
    
    Reserved for Stage Two upgrade:
    - Extended thinking mode (Supervisor 3.0)
    - Interleaved thinking loop (Researcher 3.0)
    """
    
    def __init__(self):
        self.quality_thresholds = {
            ReflectionQuality.INSUFFICIENT: 0.3,
            ReflectionQuality.PARTIAL: 0.6,
            ReflectionQuality.ADEQUATE: 0.8,
            ReflectionQuality.COMPREHENSIVE: 0.95
        }
    
    async def reflect_on_research(
        self,
        context: ReflectionContext,
        config: RunnableConfig,
        quality_threshold: float = 0.8
    ) -> ReflectionResult:
        """
        Reflect on and evaluate the research results
        
        Args:
            context: The context for reflection evaluation
            config: The runnable config
            quality_threshold: The quality threshold
            
        Returns:
            The result of the reflection evaluation
        """
        
        logger.info(f"Starting basic reflection assessment - iteration: {context.current_iteration}, results_count: {len(context.search_results)}, quality_threshold: {quality_threshold}")
        
        # Phase 1: Basic reflection logic
        if not context.search_results:
            logger.warning(f"Search results are empty, cannot perform reflection - query: {context.original_query}")
            return self._create_insufficient_result("No search results to evaluate")
        
        # Calculate basic metrics
        completeness_score = await self._calculate_completeness_score(context)
        quality = self._determine_quality_level(completeness_score)
        knowledge_gaps = await self._identify_knowledge_gaps(context)
        
        # Decide whether to continue research
        continue_research = (
            completeness_score < quality_threshold and 
            context.current_iteration < 3  # Max iterations
        )
        
        reasoning = self._generate_basic_reasoning(
            completeness_score, quality, len(knowledge_gaps)
        )
        
        logger.info(f"Basic reflection assessment complete - quality: {quality.value}, completeness_score: {completeness_score:.3f}, continue_research: {continue_research}, gaps_count: {len(knowledge_gaps)}")
        
        return ReflectionResult(
            quality=quality,
            completeness_score=completeness_score,
            knowledge_gaps=knowledge_gaps,
            continue_research=continue_research,
            reasoning=reasoning,
            suggested_focus=knowledge_gaps[0] if knowledge_gaps else None
        )
    
    async def intelligent_reflection(
        self,
        context: ReflectionContext,
        config: RunnableConfig,
        quality_threshold: float = 0.8
    ) -> ReflectionResult:
        """
        Perform intelligent reflection using an LLM (integrating the design pattern from the reference project)
        
        This implements the core reflection logic from the reference project:
        - In-depth analysis of search results
        - Intelligent identification of knowledge gaps
        - Dynamic adjustment of research strategy
        """
        
        logger.info(f"Starting intelligent reflection assessment - iteration: {context.current_iteration}, results_count: {len(context.search_results)}, previous_reflections: {len(context.previous_reflections)}")
        
        # Build prompt context
        prompt_context = {
            "original_query": context.original_query,
            "iteration": context.current_iteration,
            "research_objective": context.research_objective or "Comprehensive research on the topic",
            "search_queries": ", ".join(context.search_queries) if context.search_queries else "Unknown queries",
            "result_count": len(context.search_results),
            "search_results_content": self._format_search_results(context.search_results),
            "previous_reflections_summary": self._format_previous_reflections(context.previous_reflections)
        }
        
        # Get formatted prompt
        prompt = get_research_prompt("reflection", prompt_context)
        
        # Initialize LLM
        configurable = config.get("configurable", {})
        model_name = configurable.get("researcher_model", "anthropic:claude-3-5-sonnet-latest")
        logger.debug(f"Using model for intelligent reflection - model: {model_name}")
        llm = init_chat_model(model_name)
        
        # Invoke LLM for reflection
        messages = [HumanMessage(content=prompt)]
        logger.debug(f"Sending reflection prompt - prompt_length: {len(prompt)}")
        response = await llm.ainvoke(messages)
        logger.info(f"Intelligent reflection assessment complete - response: {response}")
        
        # Parse response
        try:
            json_content = extract_json_from_markdown(response.content)
            result_data = json.loads(json_content)
            
            # Parse quality level
            quality_str = result_data.get("quality", "partial")
            quality = ReflectionQuality(quality_str) if quality_str in [q.value for q in ReflectionQuality] else ReflectionQuality.PARTIAL
            
            result = ReflectionResult(
                quality=quality,
                completeness_score=result_data.get("completeness_score", 0.5),
                knowledge_gaps=result_data.get("knowledge_gaps", []),
                continue_research=result_data.get("continue_research", True),
                reasoning=result_data.get("reasoning", ""),
                suggested_focus=result_data.get("suggested_focus")
            )
            
            logger.info(f"Intelligent reflection assessment complete - quality: {quality.value}, completeness_score: {result.completeness_score:.3f}, continue_research: {result.continue_research}, gaps_count: {len(result.knowledge_gaps)}")
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse intelligent reflection response, falling back to basic reflection - error: {str(e)}, response_content: {response.content[:200]}...")
            # Fallback to basic reflection
            return await self.reflect_on_research(context, config, quality_threshold)
    
    async def identify_knowledge_gaps(
        self,
        context: ReflectionContext,
        config: RunnableConfig
    ) -> Dict[str, List[str]]:
        """
        Use an LLM for specialized knowledge gap identification.
        """
        
        logger.info(f"Starting knowledge gap identification - query: {context.original_query}, results_count: {len(context.search_results)}")
        
        # Build prompt context
        prompt_context = {
            "original_query": context.original_query,
            "research_objective": context.research_objective or "Comprehensive research on the topic",
            "collected_information": self._format_search_results(context.search_results)
        }
        
        # Get formatted prompt
        prompt = get_research_prompt("gap_identification", prompt_context)
        
        # Initialize LLM
        configurable = config.get("configurable", {})
        model_name = configurable.get("researcher_model", "anthropic:claude-3-5-sonnet-latest")
        llm = init_chat_model(model_name)
        
        # Invoke LLM to identify gaps
        messages = [HumanMessage(content=prompt)]
        logger.debug(f"Sending knowledge gap identification prompt - prompt_length: {len(prompt)}")
        response = await llm.ainvoke(messages)
        logger.info(f"Knowledge gap identification complete - response: {response}")
        
        # Parse response
        try:
            json_content = extract_json_from_markdown(response.content)
            result_data = json.loads(json_content)
            
            gaps_result = {
                "critical_gaps": result_data.get("critical_gaps", []),
                "minor_gaps": result_data.get("minor_gaps", []),
                "verification_needed": result_data.get("verification_needed", []),
                "new_directions": result_data.get("new_directions", []),
                "priority_ranking": result_data.get("priority_ranking", [])
            }
            
            total_gaps = len(gaps_result["critical_gaps"]) + len(gaps_result["minor_gaps"])
            logger.info(f"Knowledge gap identification complete - critical_gaps: {len(gaps_result['critical_gaps'])}, minor_gaps: {len(gaps_result['minor_gaps'])}, total_gaps: {total_gaps}")
            
            return gaps_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse knowledge gap identification response, falling back to basic identification - error: {str(e)}")
            # Fallback to basic identification
            basic_gaps = await self._identify_knowledge_gaps(context)
            return {
                "critical_gaps": basic_gaps[:2],
                "minor_gaps": basic_gaps[2:],
                "verification_needed": [],
                "new_directions": [],
                "priority_ranking": basic_gaps
            }
    
    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results"""
        if not search_results:
            return "No search results available"
        
        formatted = []
        for i, result in enumerate(search_results[:5]):  # Limit to the first 5 results
            title = result.get('title', 'Unknown Title')
            content = result.get('content', '')
            url = result.get('url', 'Unknown Source')
            
            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "..."
            
            formatted.append(f"{i+1}. Title: {title}\n   Content: {content}\n   Source: {url}")
        
        return "\n\n".join(formatted)
    
    def _format_previous_reflections(self, reflections: List[ReflectionResult]) -> str:
        """Format previous reflection results"""
        if not reflections:
            return "No previous reflection records available"
        
        formatted = []
        for i, reflection in enumerate(reflections[-2:]):  # Only show the last 2 reflections
            formatted.append(
                f"Reflection #{i+1}:\n"
                f"  Quality: {reflection.quality.value}\n"
                f"  Completeness: {reflection.completeness_score:.2f}\n"
                f"  Gaps: {', '.join(reflection.knowledge_gaps[:3])}"
            )
        
        return "\n\n".join(formatted)
    
    async def _calculate_completeness_score(self, context: ReflectionContext) -> float:
        """Calculate completeness score"""
        
        # Basic evaluation metrics
        result_count = len(context.search_results)
        
        # Score based on number of results (0.4 weight)
        count_score = min(result_count / 5.0, 1.0) * 0.4
        
        # Score based on content quality (0.6 weight)
        content_score = await self._evaluate_content_quality(context.search_results) * 0.6
        
        final_score = count_score + content_score
        logger.debug(f"Completeness score calculation - result_count: {result_count}, count_score: {count_score:.3f}, content_score: {content_score:.3f}, final_score: {final_score:.3f}")
        
        return final_score
    
    async def _evaluate_content_quality(self, search_results: List[Dict[str, Any]]) -> float:
        """Evaluate content quality"""
        
        if not search_results:
            return 0.0
        
        quality_scores = []
        
        for result in search_results:
            content = result.get('content', '')
            title = result.get('title', '')
            
            # Basic quality metrics
            content_length_score = min(len(content) / 500.0, 1.0) * 0.5
            title_relevance_score = 0.3 if title else 0.0
            has_url_score = 0.2 if result.get('url') else 0.0
            
            quality_score = content_length_score + title_relevance_score + has_url_score
            quality_scores.append(quality_score)
        
        return sum(quality_scores) / len(quality_scores)
    
    def _determine_quality_level(self, completeness_score: float) -> ReflectionQuality:
        """Determine quality level"""
        
        if completeness_score >= self.quality_thresholds[ReflectionQuality.COMPREHENSIVE]:
            return ReflectionQuality.COMPREHENSIVE
        elif completeness_score >= self.quality_thresholds[ReflectionQuality.ADEQUATE]:
            return ReflectionQuality.ADEQUATE
        elif completeness_score >= self.quality_thresholds[ReflectionQuality.PARTIAL]:
            return ReflectionQuality.PARTIAL
        else:
            return ReflectionQuality.INSUFFICIENT
    
    async def _identify_knowledge_gaps(self, context: ReflectionContext) -> List[str]:
        """Identify knowledge gaps"""
        
        logger.debug(f"Starting basic knowledge gap identification - results_count: {len(context.search_results)}")
        
        # Phase 1: Basic gap identification
        gaps = []
        
        # Check number of results
        if len(context.search_results) < 3:
            gaps.append("Insufficient number of search results, more information sources needed")
            logger.debug("Identified gap: Insufficient number of search results")
        
        # Check content depth
        avg_content_length = sum(len(result.get('content', '')) for result in context.search_results) / len(context.search_results) if context.search_results else 0
        if avg_content_length < 200:
            gaps.append("Insufficient content depth, more detailed analysis needed")
            logger.debug(f"Identified gap: Insufficient content depth - avg_length: {avg_content_length:.1f}")
        
        # Check source diversity
        unique_domains = set()
        for result in context.search_results:
            url = result.get('url', '')
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    unique_domains.add(domain)
                except:
                    pass
        
        if len(unique_domains) < 2:
            gaps.append("Lack of source diversity, diverse information sources needed")
            logger.debug(f"Identified gap: Lack of source diversity - unique_domains: {len(unique_domains)}")
        
        # Check for timeliness
        from datetime import datetime
        current_year = str(datetime.now().year)
        has_recent_info = any(current_year in result.get('content', '') for result in context.search_results)
        if not has_recent_info:
            gaps.append("Lacks up-to-date information and recent developments")
            logger.debug("Identified gap: Lacks up-to-date information")
        
        final_gaps = gaps[:5]  # Limit to a maximum of 5 gaps
        logger.debug(f"Basic knowledge gap identification complete - gaps_count: {len(final_gaps)}, gaps: {final_gaps}")
        
        return final_gaps
    
    def _generate_basic_reasoning(
        self, 
        completeness_score: float, 
        quality: ReflectionQuality, 
        gap_count: int
    ) -> str:
        """Generate basic reasoning process"""
        
        reasoning_parts = [
            f"Completeness score: {completeness_score:.2f}",
            f"Quality level: {quality.value}",
            f"Number of identified gaps: {gap_count}"
        ]
        
        if completeness_score >= 0.8:
            reasoning_parts.append("Information quality is high, basically meeting research needs")
        elif completeness_score >= 0.6:
            reasoning_parts.append("Information has some value, but there are significant gaps")
        else:
            reasoning_parts.append("Information is severely insufficient, needs substantial supplementation")
        
        return "; ".join(reasoning_parts)
    
    def _create_insufficient_result(self, reason: str) -> ReflectionResult:
        """Create an insufficient information result"""
        return ReflectionResult(
            quality=ReflectionQuality.INSUFFICIENT,
            completeness_score=0.0,
            knowledge_gaps=[reason],
            continue_research=True,
            reasoning=f"Assessment failed: {reason}",
            suggested_focus="Restart search"
        )
    
    # ================================
    # Methods reserved for Stage Two upgrade
    # ================================
    
    async def extended_thinking_reflection(
        self,
        context: ReflectionContext,
        config: RunnableConfig
    ) -> ReflectionResult:
        """
        Reflection in extended thinking mode (Supervisor 3.0)
        
        Conducts deep reflection through an inner monologue:
        - State assessment
        - Strategy adjustment
        - Dynamic planning
        """
        
        # Build prompt context for extended thinking
        prompt_context = {
            "research_state": {
                "original_query": context.original_query,
                "current_iteration": context.current_iteration,
                "search_results_count": len(context.search_results),
                "previous_reflections_count": len(context.previous_reflections)
            }
        }
        
        # Get prompt for extended thinking
        prompt = get_research_prompt("supervisor_thinking", prompt_context)
        
        # Initialize LLM
        configurable = config.get("configurable", {})
        model_name = configurable.get("researcher_model", "anthropic:claude-3-5-sonnet-latest")
        llm = init_chat_model(model_name)
        
        # Invoke LLM for extended thinking
        messages = [HumanMessage(content=prompt)]
        logger.debug(f"Sending reflection prompt - prompt_length: {len(prompt)}")
        response = await llm.ainvoke(messages)
        logger.info(f"Extended thinking mode reflection complete - response: {response}")
        
        try:
            json_content = extract_json_from_markdown(response.content)
            result_data = json.loads(json_content)
            
            # Reflect based on the results of extended thinking
            inner_monologue = result_data.get("inner_monologue", "")
            strategic_adjustments = result_data.get("strategic_adjustments", [])
            
            # Combine basic reflection with extended thinking
            basic_reflection = await self.reflect_on_research(context, config)
            
            # Enhance the reasoning process
            enhanced_reasoning = f"""
Extended Thinking Mode Reflection:

Inner Monologue: {inner_monologue}

Strategic Adjustments: {', '.join(strategic_adjustments)}

Basic Assessment: {basic_reflection.reasoning}
"""
            
            return ReflectionResult(
                quality=basic_reflection.quality,
                completeness_score=min(basic_reflection.completeness_score + 0.1, 1.0),  # Extended thinking improves quality
                knowledge_gaps=basic_reflection.knowledge_gaps,
                continue_research=basic_reflection.continue_research,
                reasoning=enhanced_reasoning,
                suggested_focus=strategic_adjustments[0] if strategic_adjustments else basic_reflection.suggested_focus
            )
            
        except (json.JSONDecodeError, ValueError):
            # Fallback to basic reflection
            basic_reflection = await self.reflect_on_research(context, config)
            basic_reflection.reasoning += " (Extended thinking mode failed, using basic reflection)"
            return basic_reflection
    
    async def interleaved_thinking_reflection(
        self,
        context: ReflectionContext,
        config: RunnableConfig
    ) -> ReflectionResult:
        """
        Reflection in the interleaved thinking loop (Researcher 3.0)
        
        Reflects using a "broad-to-narrow" heuristic:
        - Breadth exploration
        - Depth analysis
        - Interleaved validation
        - Synthesis and improvement
        """
        
        # Build prompt context for interleaved thinking
        prompt_context = {
            "research_task": {
                "original_query": context.original_query,
                "current_results": self._format_search_results(context.search_results[:3]),  # Limit number of results
                "iteration": context.current_iteration
            }
        }
        
        # Get prompt for interleaved thinking
        prompt = get_research_prompt("researcher_thinking", prompt_context)
        
        # Initialize LLM
        configurable = config.get("configurable", {})
        model_name = configurable.get("researcher_model", "anthropic:claude-3-5-sonnet-latest")
        llm = init_chat_model(model_name)
        
        # Invoke LLM for interleaved thinking
        messages = [HumanMessage(content=prompt)]
        logger.debug(f"Sending reflection prompt - prompt_length: {len(prompt)}")
        response = await llm.ainvoke(messages)
        logger.info(f"Interleaved thinking mode reflection complete - response: {response}")
        
        try:
            json_content = extract_json_from_markdown(response.content)
            result_data = json.loads(json_content)
            
            # Extract the different stages of interleaved thinking
            breadth_exploration = result_data.get("breadth_exploration", "")
            depth_analysis = result_data.get("depth_analysis", "")
            interleaved_insights = result_data.get("interleaved_insights", "")
            synthesis = result_data.get("synthesis", "")
            next_cycle_focus = result_data.get("next_cycle_focus", "")
            
            # Assess quality based on the results of interleaved thinking
            quality_indicators = [
                len(breadth_exploration) > 50,  # Is breadth exploration sufficient?
                len(depth_analysis) > 50,       # Is depth analysis sufficient?
                len(interleaved_insights) > 30, # Are interleaved insights valuable?
                len(synthesis) > 50             # Is synthesis complete?
            ]
            
            quality_score = sum(quality_indicators) / len(quality_indicators)
            
            # Determine quality level
            if quality_score >= 0.9:
                quality = ReflectionQuality.COMPREHENSIVE
                completeness_score = 0.95
            elif quality_score >= 0.7:
                quality = ReflectionQuality.ADEQUATE
                completeness_score = 0.85
            elif quality_score >= 0.5:
                quality = ReflectionQuality.PARTIAL
                completeness_score = 0.65
            else:
                quality = ReflectionQuality.INSUFFICIENT
                completeness_score = 0.4
            
            # Extract knowledge gaps from interleaved thinking
            knowledge_gaps = []
            if next_cycle_focus:
                knowledge_gaps.append(f"Needs further exploration: {next_cycle_focus}")
            if "lacks" in synthesis or "insufficient" in synthesis:
                knowledge_gaps.append("Insufficient depth or breadth of information")
            
            # Build a detailed reasoning process
            detailed_reasoning = f"""
Interleaved Thinking Loop Reflection:

1. Breadth Exploration Phase:
{breadth_exploration}

2. Depth Analysis Phase:
{depth_analysis}

3. Interleaved Insights:
{interleaved_insights}

4. Synthesis and Improvement:
{synthesis}

5. Next Cycle Focus: {next_cycle_focus}

Quality Assessment: {quality.value} (Score: {completeness_score:.2f})
"""
            
            return ReflectionResult(
                quality=quality,
                completeness_score=completeness_score,
                knowledge_gaps=knowledge_gaps,
                continue_research=completeness_score < 0.8 and context.current_iteration < 5,
                reasoning=detailed_reasoning,
                suggested_focus=next_cycle_focus
            )
            
        except (json.JSONDecodeError, ValueError):
            # Fallback to basic reflection
            basic_reflection = await self.reflect_on_research(context, config)
            basic_reflection.reasoning += " (Interleaved thinking mode failed, using basic reflection)"
            return basic_reflection 