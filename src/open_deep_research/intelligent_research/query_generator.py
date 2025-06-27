"""
Dynamic Query Generator

Implements the core query generation logic of the reference project:
- Dynamically generates queries based on the current research state
- Identifies knowledge gaps and generates targeted queries
- Supports multi-angle, multi-level query strategies
"""

from typing import List, Dict, Any, Optional
import asyncio
import json
from dataclasses import dataclass
import re
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Add logging configuration
import structlog
logger = structlog.get_logger(__name__)

from open_deep_research.intelligent_research.core import ResearchState
from open_deep_research.intelligent_research.prompts import (
    INTELLIGENT_QUERY_GENERATION_PROMPT,
    get_research_prompt
)
from open_deep_research.utils import get_config_value, get_today_str, extract_json_from_markdown


class QueryGenerationRequest(BaseModel):
    """Query Generation Request"""
    queries: List[str] = Field(description="List of generated search queries")
    reasoning: str = Field(description="Reasoning process for query generation")
    focus_areas: List[str] = Field(description="Key research areas of focus")


@dataclass 
class QueryGenerationContext:
    """Query Generation Context"""
    original_query: str
    iteration: int
    previous_results: List[Dict[str, Any]]
    knowledge_gaps: List[str]
    research_focus: Optional[str] = None


class DynamicQueryGenerator:
    """
    Dynamic Query Generator
    
    References the query generation strategy from gemini-fullstack-langgraph-quickstart:
    1. Analyze the current research state
    2. Identify knowledge gaps
    3. Generate multi-angle queries
    4. Optimize query diversity
    """
    
    def __init__(self):
        self.query_templates = self._load_query_templates()
        self.logger = logging.getLogger(__name__)
    
    def _load_query_templates(self) -> Dict[str, List[str]]:
        """Load query templates"""
        return {
            "broad_exploration": [
                "What is {topic}?",
                "Overview of {topic}",
                "{topic} comprehensive guide",
                "Latest developments in {topic}",
                "{topic} current trends 2024"
            ],
            "deep_analysis": [
                "{topic} detailed analysis",
                "Technical aspects of {topic}",
                "{topic} implementation strategies",
                "Best practices for {topic}",
                "{topic} case studies and examples"
            ],
            "comparative": [
                "{topic} vs alternatives",
                "Comparison of {topic} approaches",
                "{topic} advantages and disadvantages",
                "Different types of {topic}",
                "{topic} market analysis"
            ],
            "practical": [
                "How to implement {topic}",
                "{topic} step-by-step guide",
                "{topic} practical applications",
                "Real-world {topic} examples",
                "{topic} tools and resources"
            ],
            "gap_filling": [
                "{gap} in context of {topic}",
                "How {gap} relates to {topic}",
                "{gap} detailed explanation",
                "Missing information about {gap}",
                "Latest research on {gap}"
            ]
        }
    
    async def generate_queries(
        self,
        context: QueryGenerationContext,
        config: RunnableConfig,
        max_queries: int = 3
    ) -> List[str]:
        """
        Generate dynamic queries.
        
        Args:
            context: The query generation context.
            config: The runtime configuration.
            max_queries: The maximum number of queries.
            
        Returns:
            A list of generated queries.
        """
        
        logger.info(f"Starting dynamic query generation - iteration: {context.iteration}, max_queries: {max_queries}, gaps_count: {len(context.knowledge_gaps)}")
        
        # Phase 1: Generate using templates (simple implementation)
        if context.iteration == 0:
            queries = await self._generate_initial_queries(context, max_queries)
            logger.info(f"Initial query generation complete - queries_count: {len(queries)}, queries: {queries}")
            return queries
        else:
            queries = await self._generate_iterative_queries(context, max_queries)
            logger.info(f"Iterative query generation complete - queries_count: {len(queries)}, queries: {queries}")
            return queries
    
    async def _generate_initial_queries(
        self,
        context: QueryGenerationContext,
        max_queries: int
    ) -> List[str]:
        """Generate initial queries (first round)"""
        
        logger.debug(f"Generating initial queries - topic: {context.original_query}, max_queries: {max_queries}")
        
        topic = context.original_query
        queries = []
        
        # Generate queries using different strategies
        strategies = ["broad_exploration", "deep_analysis", "practical"]
        
        for i, strategy in enumerate(strategies[:max_queries]):
            template = self.query_templates[strategy][0]  # Use the first template of each strategy
            query = template.format(topic=topic)
            queries.append(query)
            logger.debug(f"Initial query strategy {strategy} - query: {query}")
        
        return queries
    
    async def _generate_iterative_queries(
        self,
        context: QueryGenerationContext,
        max_queries: int
    ) -> List[str]:
        """Generate iterative queries (based on previous round's results)
        
        Args:
            context: The query generation context.  
            max_queries: The maximum number of queries.
            
        Returns:
            A list of generated queries.
        """
        
        # Phase 2 will implement intelligent query generation based on LLM
        # Current stage: Generate queries based on knowledge gaps
        
        queries = []
        topic = context.original_query
        
        # Generate queries based on knowledge gaps
        available_gaps = context.knowledge_gaps[:max_queries]
        for gap in available_gaps:
            template = self.query_templates["gap_filling"][0]
            query = template.format(gap=gap, topic=topic)
            queries.append(query)
        
        # If there are not enough gaps, supplement with deep analysis strategies
        remaining = max_queries - len(queries)
        if remaining > 0:
            deep_analysis_templates = self.query_templates["deep_analysis"]
            
            for i in range(remaining):
                template_idx = i % len(deep_analysis_templates)
                template = deep_analysis_templates[template_idx]
                query = template.format(topic=topic)
                queries.append(query)
        
        return queries
    
    def _extract_json_from_markdown(self, content: str) -> str:
        """Extracts a JSON string from a Markdown formatted response."""
        # Match content between ```json and ```
        json_pattern = r'```json\s*\n(.*?)\n```'
        match = re.search(json_pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If ```json format is not found, try to match any ``` code block
        code_block_pattern = r'```.*?\n(.*?)\n```'
        match = re.search(code_block_pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If neither is found, return the original content
        return content.strip()

    async def generate_intelligent_queries(
        self,
        context: QueryGenerationContext,
        config: RunnableConfig,
        max_queries: int = 3
    ) -> QueryGenerationRequest:
        """Use LLM to generate intelligent queries, core query generation logic:
        1. Analyze the current research state
        2. Identify knowledge gaps
        3. Generate targeted queries
        """
        
        logger.info(f"Starting intelligent query generation - iteration: {context.iteration}, query: {context.original_query}, knowledge_gaps: {context.knowledge_gaps}")
        
        # Build prompt
        previous_results_text = self._format_previous_results(context.previous_results)
        knowledge_gaps_text = ", ".join(context.knowledge_gaps) if context.knowledge_gaps else "No specific knowledge gaps"
        
        prompt = f"""As a professional research assistant, I need you to generate high-quality search queries based on the following information.

Research Background:
- Original query: {context.original_query}
- Current iteration: Round {context.iteration}
- Identified knowledge gaps: {knowledge_gaps_text}
- Research focus: {context.research_focus or "Comprehensive research"}

Summary of previous search results:
{previous_results_text}

Please generate {max_queries} search queries from different angles, ensuring:
1. Each query targets a different research dimension
2. Fully utilize the identified knowledge gaps
3. Avoid repeating previous search content
4. Queries should be specific, searchable, and in-depth

Please return in the following JSON format:
```json
{{
    "queries": [
        "Query 1",
        "Query 2", 
        "Query 3"
    ],
    "reasoning": "Detail the thinking process and expected goal for each query",
    "focus_areas": [
        "Focus Area 1",
        "Focus Area 2",
        "Focus Area 3"
    ]
}}
```"""
        logger.info(f"Intelligent query generation prompt - prompt: {prompt}")
        
        # Initialize LLM
        configurable = config.get("configurable", {})
        model_name = configurable.get("researcher_model", "anthropic:claude-3-5-sonnet-latest")
        logger.debug(f"Using model to generate intelligent queries - model: {model_name}")
        llm = init_chat_model(model_name)
        
        # Call LLM to generate queries
        messages = [HumanMessage(content=prompt)]
        logger.debug(f"Sending query generation prompt - prompt_length: {len(prompt)} \n prompt: {prompt}")
        response = await llm.ainvoke(messages)
        logger.info(f"Intelligent query generation complete - response: {response}")
        
        # Parse response
        try:
            # Extract JSON string from Markdown format
            json_content = extract_json_from_markdown(response.content)
            logger.debug(f"Extracted JSON content from Markdown: {json_content[:200]}...")
            
            result_data = json.loads(json_content)
            result = QueryGenerationRequest(
                queries=result_data.get("queries", []),
                reasoning=result_data.get("reasoning", ""),
                focus_areas=result_data.get("focus_areas", [])
            )
            
            logger.info(f"Intelligent query generation complete - queries_count: {len(result.queries)}, focus_areas: {result.focus_areas}, queries: {result.queries}")
            
            # 🆕 Add diversity evaluation
            if len(result.queries) > 1:
                diversity_score = self.evaluate_query_diversity(result.queries)
                logger.info(f"Intelligent generated query diversity evaluation - diversity_score: {diversity_score:.3f}, queries_count: {len(result.queries)}")
                
                # If diversity is insufficient, try to optimize
                if diversity_score < 0.3:
                    logger.warning(f"Intelligent generated query diversity is insufficient - diversity_score: {diversity_score:.3f}, attempting to enhance diversity")
                    
                    # Use templates to supplement diversity
                    enhanced_queries = list(result.queries)  # 复制原查询
                    topic = context.original_query
                    
                    # Try adding queries from different strategies to enhance diversity
                    enhancement_strategies = ["comparative", "gap_filling", "practical"]
                    
                    for strategy in enhancement_strategies:
                        if len(enhanced_queries) >= max_queries:
                            break
                        
                        if strategy in self.query_templates:
                            if strategy == "gap_filling" and context.knowledge_gaps:
                                # Generate query using knowledge gap
                                gap = context.knowledge_gaps[0] if context.knowledge_gaps else "Detailed information"
                                template = self.query_templates[strategy][0]
                                enhanced_query = template.format(gap=gap, topic=topic)
                            else:
                                template = self.query_templates[strategy][0]
                                enhanced_query = template.format(topic=topic)
                            
                            # Avoid duplicate queries
                            if enhanced_query not in enhanced_queries:
                                enhanced_queries.append(enhanced_query)
                                logger.debug(f"Adding diversity enhancement query - strategy: {strategy}, query: {enhanced_query}")
                    
                    # Re-evaluate diversity
                    if len(enhanced_queries) > len(result.queries):
                        new_diversity_score = self.evaluate_query_diversity(enhanced_queries)
                        logger.info(f"Evaluation after diversity enhancement - original_score: {diversity_score:.3f}, new_score: {new_diversity_score:.3f}, queries_added: {len(enhanced_queries) - len(result.queries)}")
                        
                        if new_diversity_score > diversity_score:
                            # Update result
                            result.queries = enhanced_queries[:max_queries]
                            result.reasoning += f"\nDiversity enhancement: Original diversity {diversity_score:.3f} -> Enhanced {new_diversity_score:.3f}"
                            logger.info(f"Intelligent query diversity enhancement successful - final_diversity: {new_diversity_score:.3f}, final_count: {len(result.queries)}")
                        else:
                            logger.warning(f"Diversity enhancement has limited effect - keeping original queries")
                else:
                    logger.info(f"Intelligent generated query diversity is good - diversity_score: {diversity_score:.3f}")
            else:
                logger.debug(f"Insufficient number of queries, skipping diversity evaluation - queries_count: {len(result.queries)}")
            
            logger.info(f"Intelligent query generation complete - result: {result}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Intelligent query response parsing failed, falling back to template generation - error: {str(e)}, response_content: {response.content[:200]}..., extracted_json: {json_content[:200] if 'json_content' in locals() else 'N/A'}...")
            # Fallback to template generation
            queries = await self.generate_queries(context, config, max_queries)
            return QueryGenerationRequest(
                queries=queries,
                reasoning="LLM response parsing failed, using template generation",
                focus_areas=["Basic Research"]
            )
    
    def _format_previous_results(self, results: List[Dict[str, Any]]) -> str:
        """Format previous search results"""
        if not results:
            return "No previous search results yet"
        
        formatted = []
        for i, result in enumerate(results[:3]):  # Only show the top 3 results
            title = result.get('title', 'Unknown Title')
            content = result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', '')
            formatted.append(f"{i+1}. {title}\n   {content}")
        
        return "\n".join(formatted)
    
    def evaluate_query_diversity(self, queries: List[str]) -> float:
        """Evaluate query diversity"""
        logger.debug(f"Starting query diversity evaluation - queries_count: {len(queries)}")
        
        if len(queries) <= 1:
            logger.debug("Insufficient number of queries, diversity score is 0")
            return 0.0
        
        # Simple diversity evaluation: based on word overlap
        all_words = set()
        query_words = []
        
        for query in queries:
            words = set(query.lower().split())
            query_words.append(words)
            all_words.update(words)
        
        # Calculate average unique word ratio
        uniqueness_scores = []
        for words in query_words:
            unique_ratio = len(words) / len(all_words) if all_words else 0
            uniqueness_scores.append(unique_ratio)
        
        diversity_score = sum(uniqueness_scores) / len(uniqueness_scores)
        logger.debug(f"Query diversity evaluation complete - diversity_score: {diversity_score:.3f}, total_words: {len(all_words)}")
        
        return diversity_score
    
    async def optimize_query_quality(
        self,
        queries: List[str],
        context: QueryGenerationContext
    ) -> List[str]:
        """Optimize query quality (Phase 2 extension)"""
        
        logger.debug(f"Starting query quality optimization - original_count: {len(queries)}")
        
        # Current implementation: basic optimization
        optimized_queries = []
        
        for query in queries:
            # Ensure query is not empty and is meaningful
            if query.strip() and len(query.strip()) > 3:
                optimized_queries.append(query.strip())
            else:
                logger.debug(f"Skipping invalid query - query: '{query}'")
        
        # Deduplicate
        unique_queries = []
        for query in optimized_queries:
            if query not in unique_queries:
                unique_queries.append(query)
            else:
                logger.debug(f"Skipping duplicate query - query: '{query}'")
        
        final_queries = unique_queries[:3]  # Limit to a maximum of 3 queries
        
        # Add diversity evaluation and optimization
        if len(final_queries) > 1:
            diversity_score = self.evaluate_query_diversity(final_queries)
            logger.info(f"Optimized query diversity evaluation - diversity_score: {diversity_score:.3f}, query_count: {len(final_queries)}")
            
            # If diversity is insufficient, try to enhance it
            if diversity_score < 0.3 and len(final_queries) < 3:
                logger.warning(f"Query diversity insufficient, attempting to enhance - current_diversity: {diversity_score:.3f}")
                
                # Try to generate supplementary queries from different strategy templates
                topic = context.original_query
                enhancement_strategies = ["comparative", "practical"]
                
                for strategy in enhancement_strategies:
                    if len(final_queries) >= 3:
                        break
                    
                    if strategy in self.query_templates:
                        template = self.query_templates[strategy][0]
                        enhanced_query = template.format(topic=topic)
                        
                        # Avoid adding duplicate queries
                        if enhanced_query not in final_queries:
                            final_queries.append(enhanced_query)
                            logger.debug(f"Adding diversity enhancement query - strategy: {strategy}, query: {enhanced_query}")
                
                # Re-evaluate diversity
                if len(final_queries) > 1:
                    new_diversity_score = self.evaluate_query_diversity(final_queries)
                    logger.info(f"Re-evaluating after diversity enhancement - old_score: {diversity_score:.3f}, new_score: {new_diversity_score:.3f}, improvement: {new_diversity_score - diversity_score:.3f}")
                    
                    if new_diversity_score > diversity_score:
                        logger.info(f"Diversity enhancement successful - final_query_count: {len(final_queries)}")
                    else:
                        logger.warning(f"Diversity enhancement has limited effect - final_diversity: {new_diversity_score:.3f}")
            else:
                logger.info(f"Query diversity meets requirements - diversity_score: {diversity_score:.3f}")
        else:
            logger.debug(f"Insufficient number of queries, skipping diversity evaluation - query_count: {len(final_queries)}")
        
        logger.debug(f"Query quality optimization complete - final_count: {len(final_queries)}, queries: {final_queries}")
        
        return final_queries 