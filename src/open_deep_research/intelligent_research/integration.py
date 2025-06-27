"""
Intelligent Research System Integration Module

Provides seamless integration with existing workflows:
- Intelligent search interface
- Backward compatibility guarantee
- Error handling and fallback mechanisms
- Reserved interfaces for Stage Two upgrade
"""

from typing import List, Dict, Any, Optional, Union
import traceback
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig

from open_deep_research.intelligent_research.core import (
    IntelligentResearchManager,
    ResearchMode,
    ResearchState
)
from open_deep_research.intelligent_research.config import IntelligentResearchConfig
from open_deep_research.intelligent_research.query_generator import (
    DynamicQueryGenerator,
    QueryGenerationContext
)
from open_deep_research.intelligent_research.reflection import (
    ReflectionEngine,
    ReflectionContext
)
from open_deep_research.intelligent_research.iterative_controller import (
    IterativeController,
    ResearchSession
)
from open_deep_research.utils import (
    select_and_execute_search,
    deduplicate_and_format_sources,
    get_today_str,
    get_config_value,
    get_config_value_from_runnable
)

# Import the correct configuration
from open_deep_research.configuration import MultiAgentConfiguration

# Configure logging
import structlog
logger = structlog.get_logger(__name__)

@dataclass
class IntelligentSearchResult:
    """Intelligent search result"""
    content: str
    sources: List[Dict[str, Any]]
    research_quality: str
    iterations_used: int
    reasoning: str
    knowledge_gaps: List[str]
    success: bool = True
    error_message: Optional[str] = None




class IntelligentResearchIntegration:
    """
    Intelligent Research System Integrator
    
    Provides seamless integration with existing systems, supporting:
    - Switching between intelligent and simple modes
    - Error handling and automatic fallback
    - Performance monitoring and optimization
    - Reserved interfaces for Stage Two upgrade
    """
    
    def __init__(self):
        logger.info("Initializing Intelligent Research System Integrator")
        self.research_manager = IntelligentResearchManager()
        self.query_generator = DynamicQueryGenerator()
        self.reflection_engine = ReflectionEngine()
        self.controller = IterativeController()
        
        # Performance statistics
        self.stats = {
            "intelligent_searches": 0,
            "fallback_searches": 0,
            "average_iterations": 0.0,
            "success_rate": 0.0
        }
        logger.info("Intelligent Research System Integrator initialized")
    
    async def intelligent_search_web(
        self,
        queries: List[str],
        config: RunnableConfig,
        research_mode: str = "reflective",
        max_iterations: int = 3,
        quality_threshold: float = 0.8
    ) -> IntelligentSearchResult:
        """
        Intelligent Web Search Interface
        
        This is the main integration interface, which can directly replace the existing search_web function.
        
        Args:
            queries: List of search queries
            config: Runnable config
            research_mode: Research mode ("simple", "reflective", "collaborative")
            max_iterations: Maximum number of iterations
            quality_threshold: Quality threshold
            
        Returns:
            Intelligent search result
        """
        
        logger.info(f"Starting intelligent web search - queries_count: {len(queries)}, research_mode: {research_mode}, max_iterations: {max_iterations}, quality_threshold: {quality_threshold}")
        logger.debug(f"Search query details: {queries[:3]}")  # Only show the first 3 queries to avoid long logs
        
        try:
            self.stats["intelligent_searches"] += 1
            
            # Validate input
            if not queries or not queries[0].strip():
                logger.error("Search query is empty")
                return self._create_error_result("Query cannot be empty")
            
            # Parse research mode
            try:
                mode = ResearchMode(research_mode)
                logger.info(f"Using research mode: {mode.value}")
            except ValueError:
                logger.warning(f"Invalid research mode '{research_mode}', using default mode: reflective")
                mode = ResearchMode.REFLECTIVE  # Default mode
            
            # Create research session
            session = ResearchSession(
                original_query=queries[0],
                research_mode=mode,
                max_iterations=max_iterations,
                quality_threshold=quality_threshold
            )
            logger.info(f"Created research session - original_query: {session.original_query[:100]}...")
            
            # Conduct intelligent research
            if mode == ResearchMode.SIMPLE:
                logger.info("Executing simple search mode")
                return await self._simple_search(queries, config)
            elif mode == ResearchMode.REFLECTIVE:
                logger.info("Executing reflective search mode")
                return await self._reflective_search(session, config)
            elif mode == ResearchMode.COLLABORATIVE:
                logger.info("Executing collaborative search mode")
                return await self._collaborative_search(session, config)
            else:
                logger.warning(f"Unrecognized research mode {mode}, falling back to simple search")
                return await self._simple_search(queries, config)
                
        except Exception as e:
            logger.error(f"An exception occurred during intelligent search: {str(e)}")
            logger.error(f"Exception details: {traceback.format_exc()}")
            self.stats["fallback_searches"] += 1
            return await self._fallback_search(queries, config, str(e))
    
    async def _simple_search(
        self,
        queries: List[str],
        config: RunnableConfig
    ) -> IntelligentSearchResult:
        """Simple search mode (compatible with original search)"""
        
        logger.info(f"Starting simple search - queries_count: {len(queries)}")
        logger.debug(f"Query content: {queries}")
        
        try:
            # Execute traditional search - requires correct parameters
            # Get search API settings from config, default to tavily
            search_api = get_config_value_from_runnable(config, 'search_api', 'tavily')
            params_to_pass = get_config_value_from_runnable(config, 'search_params', {})
            
            logger.info(f"Using search API: {search_api}, parameters: {params_to_pass}")
            
            # Call search function
            formatted_content = await select_and_execute_search(
                search_api=search_api,
                query_list=queries,
                params_to_pass=params_to_pass
            )
            
            logger.info(f"Simple search complete - content length: {len(formatted_content)} characters")
            
            # Parse search results to get sources
            # formatted_content is a formatted string, we need to create virtual sources
            sources = [{"title": "Search Result", "content": formatted_content, "url": "multiple"}]
            
            result = IntelligentSearchResult(
                content=formatted_content,
                sources=sources,
                research_quality="simple",
                iterations_used=1,
                reasoning="Using simple search mode, directly returning search results",
                knowledge_gaps=[],
                success=True
            )
            
            logger.info(f"Simple search result created successfully - success: {result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Simple search failed: {str(e)}")
            logger.error(f"Exception details: {traceback.format_exc()}")
            return self._create_error_result(f"Simple search failed: {str(e)}")
    
    async def _reflective_search(
        self,
        session: ResearchSession,
        config: RunnableConfig
    ) -> IntelligentSearchResult:
        """Reflective search mode (core intelligent feature)"""
        
        logger.info(f"Starting reflective search - original_query: {session.original_query[:100]}..., max_iterations: {session.max_iterations}")
        
        all_results = []
        all_reasoning = []
        current_iteration = 0
        
        # Generate initial query
        logger.info("Generating initial query context")
        query_context = QueryGenerationContext(
            original_query=session.original_query,
            iteration=0,
            previous_results=[],
            knowledge_gaps=[]
        )
        
        try:
            # Try to use intelligent query generation
            logger.info("Trying to use intelligent query generation")
            query_request = await self.query_generator.generate_intelligent_queries(
                query_context, config, max_queries=3
            )
            current_queries = query_request.queries
            logger.info(f"Intelligent query generation successful - generated_queries_count: {len(current_queries)}")
            logger.debug(f"Generated queries: {current_queries}")
            all_reasoning.append(f"Initial query generation: {query_request.reasoning}")
            
            # 🆕 Add initial query diversity evaluation
            if len(current_queries) > 1:
                diversity_score = self.query_generator.evaluate_query_diversity(current_queries)
                logger.info(f"Initial query diversity evaluation - diversity_score: {diversity_score:.3f}")
                all_reasoning.append(f"Initial query diversity score: {diversity_score:.3f}")
                
                if diversity_score < 0.3:
                    logger.warning(f"Low initial query diversity - diversity_score: {diversity_score:.3f}")
                    all_reasoning.append(f"Note: Low initial query diversity ({diversity_score:.3f})")
        except Exception as e:
            # Fallback to template-based query generation
            logger.warning(f"Intelligent query generation failed, using template generation: {str(e)}")
            current_queries = await self.query_generator.generate_queries(
                query_context, config, max_queries=3
            )
            logger.info(f"Template query generation complete - queries_count: {len(current_queries)}")
            all_reasoning.append(f"Using template-based query generation (intelligent generation failed: {str(e)})")
            
            # 🆕 Add template query diversity evaluation
            if len(current_queries) > 1:
                diversity_score = self.query_generator.evaluate_query_diversity(current_queries)
                logger.info(f"Template query diversity evaluation - diversity_score: {diversity_score:.3f}")
                all_reasoning.append(f"Template query diversity score: {diversity_score:.3f}")
                
                if diversity_score < 0.3:
                    logger.warning(f"Low template query diversity - diversity_score: {diversity_score:.3f}")
                    all_reasoning.append(f"Note: Low template query diversity ({diversity_score:.3f})")
        
        while current_iteration < session.max_iterations:
            current_iteration += 1
            logger.info(f"Starting search iteration {current_iteration}")
            
            # Execute search - provide correct parameters
            search_api = get_config_value_from_runnable(config, 'search_api', 'tavily')
            params_to_pass = get_config_value_from_runnable(config, 'search_params', {})
            
            logger.info(f"Iteration {current_iteration} - Using search API: {search_api}, queries_count: {len(current_queries)}")
            logger.debug(f"Iteration {current_iteration} - Query content: {current_queries}")
            
            search_content = await select_and_execute_search(
                search_api=search_api,
                query_list=current_queries,
                params_to_pass=params_to_pass
            )
            
            logger.info(f"Iteration {current_iteration} search complete - content length: {len(search_content)} characters")
            
            # Convert search content to result format
            search_results = [{"title": f"Query results - Iteration {current_iteration}", "content": search_content, "url": "multiple"}]
            all_results.extend(search_results)
            
            # Create reflection context
            logger.info(f"Iteration {current_iteration} - Creating reflection context")
            reflection_context = ReflectionContext(
                original_query=session.original_query,
                current_iteration=current_iteration,
                search_results=search_results,
                previous_reflections=[],
                search_queries=current_queries
            )
            
            # Perform reflection and evaluation
            try:
                # Try to use intelligent reflection
                logger.info(f"Iteration {current_iteration} - Attempting intelligent reflection")
                reflection_result = await self.reflection_engine.intelligent_reflection(
                    reflection_context, config, session.quality_threshold
                )
                logger.info(f"Iteration {current_iteration} - Intelligent reflection complete - continue_research: {reflection_result.continue_research}")
            except Exception as e:
                # Fallback to basic reflection
                logger.warning(f"Iteration {current_iteration} - Intelligent reflection failed, using basic reflection: {str(e)}")
                reflection_result = await self.reflection_engine.reflect_on_research(
                    reflection_context, config, session.quality_threshold
                )
                logger.info(f"Iteration {current_iteration} - Basic reflection complete - continue_research: {reflection_result.continue_research}")
                all_reasoning.append(f"Using basic reflection (intelligent reflection failed: {str(e)})")
            
            all_reasoning.append(f"Iteration {current_iteration} reflection: {reflection_result.reasoning}")
            
            # Check if we need to continue
            if not reflection_result.continue_research:
                logger.info(f"Iteration {current_iteration} - Reflection suggests stopping research, ending iteration")
                break
            
            # Generate next round of queries
            if current_iteration < session.max_iterations:
                logger.info(f"Preparing for iteration {current_iteration + 1} query generation")
                query_context = QueryGenerationContext(
                    original_query=session.original_query,
                    iteration=current_iteration,
                    previous_results=search_results,
                    knowledge_gaps=reflection_result.knowledge_gaps
                )
                
                try:
                    logger.info(f"Iteration {current_iteration + 1} - Attempting intelligent query generation")
                    query_request = await self.query_generator.generate_intelligent_queries(
                        query_context, config, max_queries=2
                    )
                    current_queries = query_request.queries
                    logger.info(f"Iteration {current_iteration + 1} - Intelligent query generation successful - queries_count: {len(current_queries)}")
                    all_reasoning.append(f"Iteration {current_iteration+1} query generation: {query_request.reasoning}")
                    
                    # 🆕 Add iterative query diversity evaluation
                    if len(current_queries) > 1:
                        diversity_score = self.query_generator.evaluate_query_diversity(current_queries)
                        logger.info(f"Iteration {current_iteration + 1} query diversity evaluation - diversity_score: {diversity_score:.3f}")
                        all_reasoning.append(f"Iteration {current_iteration+1} query diversity score: {diversity_score:.3f}")
                        
                        if diversity_score < 0.3:
                            logger.warning(f"Iteration {current_iteration + 1} low query diversity - diversity_score: {diversity_score:.3f}")
                            all_reasoning.append(f"Note: Iteration {current_iteration+1} low query diversity ({diversity_score:.3f})")
                except Exception as e:
                    logger.warning(f"Iteration {current_iteration + 1} - Intelligent query generation failed, using template generation: {str(e)}")
                    current_queries = await self.query_generator.generate_queries(
                        query_context, config, max_queries=2
                    )
                    logger.info(f"Iteration {current_iteration + 1} - Template query generation complete - queries_count: {len(current_queries)}")
                    all_reasoning.append(f"Using template-based query generation (intelligent generation failed: {str(e)})")
                    
                    # 🆕 Add template query diversity evaluation
                    if len(current_queries) > 1:
                        diversity_score = self.query_generator.evaluate_query_diversity(current_queries)
                        logger.info(f"Iteration {current_iteration + 1} template query diversity evaluation - diversity_score: {diversity_score:.3f}")
                        all_reasoning.append(f"Iteration {current_iteration+1} template query diversity score: {diversity_score:.3f}")
                        
                        if diversity_score < 0.3:
                            logger.warning(f"Iteration {current_iteration + 1} low template query diversity - diversity_score: {diversity_score:.3f}")
                            all_reasoning.append(f"Note: Iteration {current_iteration+1} low template query diversity ({diversity_score:.3f})")
        
        logger.info(f"Reflective search complete - total_iterations: {current_iteration}, total_results_count: {len(all_results)}")
        
        # Format final result
        logger.info("Starting to format search results")
        formatted_content = deduplicate_and_format_sources(
            all_results,
            max_tokens_per_source=10000,
            deduplication_strategy="keep_first"
        )
        
        logger.info(f"Result formatting complete - final content length: {len(formatted_content)} characters")
        
        # Update statistics
        self._update_stats(current_iteration, True)
        logger.info(f"Statistics updated - current success rate: {self.stats['success_rate']:.2f}")
        
        result = IntelligentSearchResult(
            content=formatted_content,
            sources=all_results,
            research_quality=reflection_result.quality.value if 'reflection_result' in locals() else "adequate",
            iterations_used=current_iteration,
            reasoning="\n".join(all_reasoning),
            knowledge_gaps=reflection_result.knowledge_gaps if 'reflection_result' in locals() else [],
            success=True
        )
        
        logger.info(f"Reflective search result created successfully - research_quality: {result.research_quality}, iterations_used: {result.iterations_used}")
        return result
    
    async def _collaborative_search(
        self,
        session: ResearchSession,
        config: RunnableConfig
    ) -> IntelligentSearchResult:
        """Collaborative search mode (to be implemented in Stage Two)"""
        
        logger.info("Collaborative search mode: falling back to reflective search in the current stage")
        # Current stage: Fallback to reflective search
        return await self._reflective_search(session, config)
    
    async def _fallback_search(
        self,
        queries: List[str],
        config: RunnableConfig,
        error_message: str
    ) -> IntelligentSearchResult:
        """Fallback search (used in case of errors)"""
        
        logger.warning(f"Initiating fallback search - reason: {error_message}")
        logger.info(f"Fallback search - queries_count: {len(queries)}")
        
        try:
            # Use the most basic search method - provide correct parameters
            search_api = get_config_value_from_runnable(config, 'search_api_fallback', get_config_value_from_runnable(config, 'search_api', 'tavily'))
            params_to_pass = get_config_value_from_runnable(config, 'search_params', {})
            
            logger.info(f"Fallback search using API: {search_api}, using only the first query")
            
            formatted_content = await select_and_execute_search(
                search_api=search_api,
                query_list=queries[:1],  # Only use the first query
                params_to_pass=params_to_pass
            )
            
            logger.info(f"Fallback search complete - content length: {len(formatted_content)} characters")
            logger.info(f"Fallback search complete - content: {formatted_content}")
            
            # Create virtual search results
            search_results = [{"title": "Fallback Search Result", "content": formatted_content, "url": "fallback"}]
            
            result = IntelligentSearchResult(
                content=formatted_content,
                sources=search_results,
                research_quality="fallback",
                iterations_used=1,
                reasoning=f"Intelligent search failed, using fallback mode: {error_message}",
                knowledge_gaps=["Intelligent features unavailable"],
                success=True,
                error_message=error_message
            )
            
            logger.info(f"Fallback search result created successfully - success: {result.success}")
            return result
            
        except Exception as fallback_error:
            logger.error(f"Fallback search also failed: {str(fallback_error)}")
            logger.error(f"Fallback search exception details: {traceback.format_exc()}")
            return self._create_error_result(
                f"Both intelligent search and fallback search failed: {error_message} | {str(fallback_error)}"
            )
    
    def _create_error_result(self, error_message: str) -> IntelligentSearchResult:
        """Create an error result"""
        logger.error(f"Creating error result: {error_message}")
        return IntelligentSearchResult(
            content="Search failed, unable to retrieve information.",
            sources=[],
            research_quality="error",
            iterations_used=0,
            reasoning=f"An error occurred during the search process: {error_message}",
            knowledge_gaps=["Search functionality unavailable"],
            success=False,
            error_message=error_message
        )
    
    def _update_stats(self, iterations: int, success: bool):
        """Update performance statistics"""
        logger.debug(f"Updating statistics - iterations: {iterations}, success: {success}")
        total_searches = self.stats["intelligent_searches"] + self.stats["fallback_searches"]
        if total_searches > 0:
            # Update average iterations
            current_avg = self.stats["average_iterations"]
            self.stats["average_iterations"] = (current_avg * (total_searches - 1) + iterations) / total_searches
            
            # Update success rate
            if success:
                successful_searches = total_searches - self.stats["fallback_searches"]
                self.stats["success_rate"] = successful_searches / total_searches
        
        logger.debug(f"Statistics update complete - average_iterations: {self.stats['average_iterations']:.2f}, success_rate: {self.stats['success_rate']:.2f}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            **self.stats,
            "total_searches": self.stats["intelligent_searches"] + self.stats["fallback_searches"],
            "intelligent_ratio": self.stats["intelligent_searches"] / max(1, self.stats["intelligent_searches"] + self.stats["fallback_searches"])
        }
        logger.info(f"Getting performance statistics - total_searches: {stats['total_searches']}, intelligent_ratio: {stats['intelligent_ratio']:.2f}")
        return stats


# ================================
# Global integration instance and convenience functions
# ================================

# Global integration instance
_integration_instance = None

def get_integration_instance() -> IntelligentResearchIntegration:
    """Get the global integration instance"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = IntelligentResearchIntegration()
    return _integration_instance


async def intelligent_search_web(
    queries: List[str],
    config: RunnableConfig,
    research_mode: str = "reflective",
    max_iterations: int = 3,
    quality_threshold: float = 0.8
) -> str:
    """
    Convenience interface for intelligent web search
    
    This function can directly replace the existing search_web function, providing backward compatibility.
    
    Args:
        queries: List of search queries
        config: Runnable config
        research_mode: Research mode ("simple", "reflective", "collaborative")
        max_iterations: Maximum number of iterations
        quality_threshold: Quality threshold
        
    Returns:
        Formatted content of the search results
    """
    
    logger.info(f"Convenience interface called - queries_count: {len(queries)}, research_mode: {research_mode}")
    
    integration = get_integration_instance()
    result = await integration.intelligent_search_web(
        queries, config, research_mode, max_iterations, quality_threshold
    )
    
    logger.info(f"Convenience interface returned - success: {result.success}, content_length: {len(result.content)}")
    return result.content


async def get_intelligent_search_details(
    queries: List[str],
    config: RunnableConfig,
    research_mode: str = "reflective",
    max_iterations: int = 3,
    quality_threshold: float = 0.8
) -> IntelligentSearchResult:
    """
    Get detailed intelligent search results
    
    Args:
        queries: List of search queries
        config: Runnable config
        research_mode: Research mode
        max_iterations: Maximum number of iterations
        quality_threshold: Quality threshold
        
    Returns:
        Complete intelligent search result
    """
    
    logger.info(f"Detailed search interface called - queries_count: {len(queries)}, research_mode: {research_mode}")
    
    integration = get_integration_instance()
    result = await integration.intelligent_search_web(
        queries, config, research_mode, max_iterations, quality_threshold
    )
    
    logger.info(f"Detailed search interface returned - success: {result.success}, iterations_used: {result.iterations_used}")
    return result


def get_research_performance_stats() -> Dict[str, Any]:
    """Get research performance statistics"""
    logger.info("Getting research performance statistics")
    integration = get_integration_instance()
    stats = integration.get_performance_stats()
    logger.info(f"Performance statistics retrieval complete - total_searches: {stats.get('total_searches', 0)}")
    return stats


# ================================
# Reserved interfaces for Stage Two upgrade
# ================================

async def supervisor_3_0_search(
    queries: List[str],
    config: RunnableConfig,
    thinking_mode: str = "extended"
) -> IntelligentSearchResult:
    """
    Supervisor 3.0 extended thinking mode search (now implemented)
    
    Args:
        queries: Search queries
        config: Runnable config
        thinking_mode: Thinking mode ("extended", "dynamic_planning")
        
    Returns:
        Intelligent search result
    """
    
    logger.info(f"Supervisor 3.0 search started - queries_count: {len(queries)}, thinking_mode: {thinking_mode}")
    
    integration = get_integration_instance()
    
    # Create extended thinking session
    session = ResearchSession(
        original_query=queries[0] if queries else "Supervisor 3.0 extended thinking search",
        research_mode=ResearchMode.EXTENDED,
        max_iterations=5,
        quality_threshold=0.85
    )
    
    logger.info(f"Supervisor 3.0 - Created extended thinking session: {session.original_query[:100]}...")
    
    try:
        # Search using extended thinking mode
        all_results = []
        all_reasoning = []
        current_iteration = 0
        
        # Generate initial query
        logger.info("Supervisor 3.0 - Generating initial query")
        query_context = QueryGenerationContext(
            original_query=session.original_query,
            iteration=0,
            previous_results=[],
            knowledge_gaps=[],
            research_focus="Extended thinking mode in-depth research"
        )
        
        # Use intelligent query generation
        try:
            logger.info("Supervisor 3.0 - Attempting intelligent query generation")
            query_request = await integration.query_generator.generate_intelligent_queries(
                query_context, config, max_queries=4
            )
            current_queries = query_request.queries
            logger.info(f"Supervisor 3.0 - Intelligent query generation successful: {len(current_queries)} queries")
            all_reasoning.append(f"Supervisor 3.0 initial query generation: {query_request.reasoning}")
        except Exception as e:
            logger.warning(f"Supervisor 3.0 - Intelligent query generation failed, using original query: {str(e)}")
            current_queries = queries[:3]  # Use original queries
            all_reasoning.append(f"Using original query (intelligent generation failed: {str(e)})")
        
        while current_iteration < session.max_iterations:
            current_iteration += 1
            logger.info(f"Supervisor 3.0 - Starting iteration {current_iteration} of extended thinking")
            
            # Execute search - provide correct parameters (Supervisor 3.0)
            search_api = get_config_value_from_runnable(config, 'search_api', 'tavily')
            params_to_pass = get_config_value_from_runnable(config, 'search_params', {})
            
            logger.info(f"Supervisor 3.0 Iteration {current_iteration} - Using API: {search_api}, queries_count: {len(current_queries)}")
            
            search_content = await select_and_execute_search(
                search_api=search_api,
                query_list=current_queries,
                params_to_pass=params_to_pass
            )
            
            logger.info(f"Supervisor 3.0 Iteration {current_iteration} search complete - content length: {len(search_content)}")
            
            # Convert search content to result format
            search_results = [{"title": f"Supervisor 3.0 search - Iteration {current_iteration}", "content": search_content, "url": "supervisor_3_0"}]
            all_results.extend(search_results)
            
            # Create reflection context
            logger.info(f"Supervisor 3.0 Iteration {current_iteration} - Creating extended thinking reflection context")
            reflection_context = ReflectionContext(
                original_query=session.original_query,
                current_iteration=current_iteration,
                search_results=search_results,
                previous_reflections=[],
                search_queries=current_queries,
                research_objective="Supervisor 3.0 extended thinking mode research"
            )
            
            # Use extended thinking reflection
            try:
                logger.info(f"Supervisor 3.0 Iteration {current_iteration} - Attempting extended thinking reflection")
                reflection_result = await integration.reflection_engine.extended_thinking_reflection(
                    reflection_context, config
                )
                logger.info(f"Supervisor 3.0 Iteration {current_iteration} - Extended thinking reflection complete - continue_research: {reflection_result.continue_research}")
                all_reasoning.append(f"Iteration {current_iteration} extended thinking reflection: {reflection_result.reasoning}")
            except Exception as e:
                # Fallback to basic reflection
                logger.warning(f"Supervisor 3.0 Iteration {current_iteration} - Extended thinking failed, using basic reflection: {str(e)}")
                reflection_result = await integration.reflection_engine.reflect_on_research(
                    reflection_context, config, session.quality_threshold
                )
                logger.info(f"Supervisor 3.0 Iteration {current_iteration} - Basic reflection complete")
                all_reasoning.append(f"Using basic reflection (extended thinking failed: {str(e)})")
            
            # Check if we need to continue
            if not reflection_result.continue_research or reflection_result.completeness_score >= session.quality_threshold:
                logger.info(f"Supervisor 3.0 Iteration {current_iteration} - Stopping conditions met, ending search")
                break
            
            # Generate next round of queries (based on strategic adjustments from extended thinking)
            if current_iteration < session.max_iterations:
                logger.info(f"Supervisor 3.0 - Preparing for iteration {current_iteration + 1} query")
                query_context = QueryGenerationContext(
                    original_query=session.original_query,
                    iteration=current_iteration,
                    previous_results=search_results,
                    knowledge_gaps=reflection_result.knowledge_gaps,
                    research_focus=reflection_result.suggested_focus or "In-depth extended analysis"
                )
                
                try:
                    query_request = await integration.query_generator.generate_intelligent_queries(
                        query_context, config, max_queries=3
                    )
                    current_queries = query_request.queries
                    logger.info(f"Supervisor 3.0 Iteration {current_iteration + 1} - Intelligent query generation successful: {len(current_queries)} queries")
                    all_reasoning.append(f"Iteration {current_iteration+1} extended query generation: {query_request.reasoning}")
                except Exception as e:
                    logger.warning(f"Supervisor 3.0 Iteration {current_iteration + 1} - Intelligent query generation failed: {str(e)}")
                    current_queries = [f"{session.original_query} in-depth analysis"]
                    all_reasoning.append(f"Using basic query (intelligent generation failed: {str(e)})")
        
        logger.info(f"Supervisor 3.0 search complete - total_iterations: {current_iteration}")
        
        # Format final result
        formatted_content = deduplicate_and_format_sources(
            all_results,
            max_tokens_per_source=5000,  # Supervisor 3.0 supports longer content
            deduplication_strategy="keep_first"
        )
        
        result = IntelligentSearchResult(
            content=formatted_content,
            sources=all_results,
            research_quality=f"supervisor_3.0_{reflection_result.quality.value}" if 'reflection_result' in locals() else "supervisor_3.0_adequate",
            iterations_used=current_iteration,
            reasoning="\n".join(all_reasoning),
            knowledge_gaps=reflection_result.knowledge_gaps if 'reflection_result' in locals() else [],
            success=True
        )
        
        logger.info(f"Supervisor 3.0 search result created successfully - quality: {result.research_quality}")
        return result
        
    except Exception as e:
        logger.error(f"Supervisor 3.0 search failed: {str(e)}")
        logger.error(f"Supervisor 3.0 exception details: {traceback.format_exc()}")
        # Fallback to reflective search
        return await integration._reflective_search(session, config)


async def researcher_3_0_search(
    queries: List[str],
    config: RunnableConfig,
    thinking_mode: str = "interleaved"
) -> IntelligentSearchResult:
    """
    Researcher 3.0 interleaved thinking mode search (now implemented)
    
    Args:
        queries: Search queries
        config: Runnable config
        thinking_mode: Thinking mode ("interleaved", "broad_to_narrow")
        
    Returns:
        Intelligent search result
    """
    
    logger.info(f"Researcher 3.0 search started - queries_count: {len(queries)}, thinking_mode: {thinking_mode}")
    
    integration = get_integration_instance()
    
    # Create interleaved thinking session
    session = ResearchSession(
        original_query=queries[0] if queries else "Researcher 3.0 interleaved thinking search",
        research_mode=ResearchMode.ITERATIVE,
        max_iterations=7,
        quality_threshold=0.9
    )
    
    logger.info(f"Researcher 3.0 - Created interleaved thinking session: {session.original_query[:100]}...")
    
    try:
        # Search using interleaved thinking mode
        all_results = []
        all_reasoning = []
        current_iteration = 0
        
        # Stage 1: Breadth exploration
        logger.info("Researcher 3.0 - Stage 1: Breadth exploration")
        broad_queries = []
        for query in queries[:2]:  # Limit number of queries
            broad_queries.extend([
                f"{query} overview",
                f"{query} multi-angle analysis",
                f"{query} related fields"
            ])
        
        logger.info(f"Researcher 3.0 - Breadth exploration query generation complete: {len(broad_queries)} queries")
        print(f"🔍 Researcher 3.0 - Breadth exploration phase")
        search_api = get_config_value_from_runnable(config, 'search_api', 'tavily')
        params_to_pass = get_config_value_from_runnable(config, 'search_params', {})
        
        broad_content = await select_and_execute_search(
            search_api=search_api,
            query_list=broad_queries[:4],
            params_to_pass=params_to_pass
        )
        broad_results = [{"title": "Researcher 3.0 Breadth Exploration", "content": broad_content, "url": "researcher_3_0_broad"}]
        all_results.extend(broad_results)
        logger.info(f"Researcher 3.0 - Breadth exploration complete - content length: {len(broad_content)}")
        all_reasoning.append(f"Breadth exploration: Generated {len(broad_queries[:4])} broad queries, obtained {len(broad_results)} results")
        
        # Stage 2: Depth analysis
        logger.info("Researcher 3.0 - Stage 2: Depth analysis")
        print(f"🎯 Researcher 3.0 - Depth analysis phase")
        depth_queries = []
        for query in queries[:2]:
            depth_queries.extend([
                f"{query} detailed technical analysis",
                f"{query} implementation principles",
                f"{query} best practice case studies"
            ])
        
        logger.info(f"Researcher 3.0 - Depth query generation complete: {len(depth_queries)} queries")
        
        depth_content = await select_and_execute_search(
            search_api=search_api,
            query_list=depth_queries[:4],
            params_to_pass=params_to_pass
        )
        depth_results = [{"title": "Researcher 3.0 Depth Analysis", "content": depth_content, "url": "researcher_3_0_depth"}]
        all_results.extend(depth_results)
        logger.info(f"Researcher 3.0 - Depth analysis complete - content length: {len(depth_content)}")
        all_reasoning.append(f"Depth analysis: Generated {len(depth_queries[:4])} depth queries, obtained {len(depth_results)} results")
        
        # Stage 3: Interleaved validation and reflection
        logger.info("Researcher 3.0 - Stage 3: Interleaved thinking reflection")
        print(f"💡 Researcher 3.0 - Interleaved thinking reflection")
        
        reflection_context = ReflectionContext(
            original_query=session.original_query,
            current_iteration=2,  # After breadth and depth stages
            search_results=all_results,
            previous_reflections=[],
            search_queries=broad_queries + depth_queries,
            research_objective="Researcher 3.0 interleaved thinking mode research"
        )
        
        # Use interleaved thinking reflection
        try:
            logger.info("Researcher 3.0 - Attempting interleaved thinking reflection")
            reflection_result = await integration.reflection_engine.interleaved_thinking_reflection(
                reflection_context, config
            )
            logger.info(f"Researcher 3.0 - Interleaved thinking reflection complete - continue_research: {reflection_result.continue_research}")
            all_reasoning.append(f"Interleaved thinking reflection: {reflection_result.reasoning}")
        except Exception as e:
            # Fallback to basic reflection
            logger.warning(f"Researcher 3.0 - Interleaved thinking reflection failed, using basic reflection: {str(e)}")
            reflection_result = await integration.reflection_engine.reflect_on_research(
                reflection_context, config, session.quality_threshold
            )
            logger.info("Researcher 3.0 - Basic reflection complete")
            all_reasoning.append(f"Using basic reflection (interleaved thinking failed: {str(e)})")
        
        # Stage 4: Synthesis and improvement (if needed)
        if reflection_result.continue_research and reflection_result.suggested_focus:
            logger.info("Researcher 3.0 - Stage 4: Synthesis and improvement")
            print(f"📊 Researcher 3.0 - Synthesis and improvement phase")
            
            synthesis_queries = [
                f"{session.original_query} {reflection_result.suggested_focus}",
                f"{reflection_result.suggested_focus} comprehensive analysis",
                f"how to integrate various aspects of {session.original_query}"
            ]
            
            logger.info(f"Researcher 3.0 - Synthesis query generation: {len(synthesis_queries)} queries")
            
            synthesis_content = await select_and_execute_search(
                search_api=search_api,
                query_list=synthesis_queries,
                params_to_pass=params_to_pass
            )
            synthesis_results = [{"title": "Researcher 3.0 Synthesis and Improvement", "content": synthesis_content, "url": "researcher_3_0_synthesis"}]
            all_results.extend(synthesis_results)
            logger.info(f"Researcher 3.0 - Synthesis and improvement complete - content length: {len(synthesis_content)}")
            all_reasoning.append(f"Synthesis and improvement: Generated {len(synthesis_queries)} synthesis queries based on reflection, obtained {len(synthesis_results)} results")
        
        logger.info(f"Researcher 3.0 search complete - total_results_count: {len(all_results)}")
        
        # Format final result
        formatted_content = deduplicate_and_format_sources(
            all_results,
            max_tokens_per_source=4500,  # Researcher 3.0 supports longer content
            deduplication_strategy="keep_best"  # Keep the best results
        )
        
        result = IntelligentSearchResult(
            content=formatted_content,
            sources=all_results,
            research_quality=f"researcher_3.0_{reflection_result.quality.value}",
            iterations_used=4,  # Breadth, depth, reflection, synthesis stages
            reasoning="\n".join(all_reasoning),
            knowledge_gaps=reflection_result.knowledge_gaps,
            success=True
        )
        
        logger.info(f"Researcher 3.0 search result created successfully - quality: {result.research_quality}")
        return result
        
    except Exception as e:
        logger.error(f"Researcher 3.0 search failed: {str(e)}")
        logger.error(f"Researcher 3.0 exception details: {traceback.format_exc()}")
        # Fallback to reflective search
        return await integration._reflective_search(session, config) 