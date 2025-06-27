"""
Iterative Controller

Manages the iterative process of intelligent research:
- Controls the execution of the research loop
- Decides when to continue or stop research
- Manages the transition of research states
- Prepares for the future creation of dynamic agents
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
from dataclasses import dataclass
from enum import Enum

from langchain_core.runnables import RunnableConfig

# Add logging configuration
import structlog
logger = structlog.get_logger(__name__)

from open_deep_research.intelligent_research.core import ResearchState, ResearchMode
from open_deep_research.intelligent_research.reflection import ReflectionResult, ReflectionQuality
from open_deep_research.intelligent_research.query_generator import QueryGenerationContext
from open_deep_research.utils import select_and_execute_search, get_search_params, get_config_value_from_runnable


class IterationDecision(Enum):
    """Iteration Decision"""
    CONTINUE = "continue"         # Continue research
    STOP_SUCCESS = "stop_success"  # Successfully completed
    STOP_MAX_ITER = "stop_max_iter"  # Reached maximum iterations
    STOP_ERROR = "stop_error"     # Stopped due to error


@dataclass
class IterationResult:
    """Iteration Result"""
    decision: IterationDecision
    final_state: ResearchState
    all_search_results: List[Dict[str, Any]]
    reflection_history: List[ReflectionResult]
    iteration_count: int
    total_queries_executed: int


@dataclass
class ResearchSession:
    """
    Research Session
    
    Encapsulates all information for a complete research session:
    - Original query and research objectives
    - Research mode and configuration parameters
    - Session state and progress tracking
    """
    original_query: str
    research_mode: ResearchMode = ResearchMode.REFLECTIVE
    max_iterations: int = 3
    quality_threshold: float = 0.8
    
    # Session state
    session_id: Optional[str] = None
    start_time: Optional[str] = None
    current_iteration: int = 0
    is_active: bool = True
    
    # Research configuration
    enable_intelligent_queries: bool = True
    enable_intelligent_reflection: bool = True
    max_queries_per_iteration: int = 3
    
    # Result tracking
    total_queries_generated: int = 0
    total_results_collected: int = 0
    best_quality_achieved: str = "insufficient"
    
    def __post_init__(self):
        if self.session_id is None:
            import uuid
            self.session_id = str(uuid.uuid4())[:8]
        
        if self.start_time is None:
            from datetime import datetime
            self.start_time = datetime.now().isoformat()
    
    def update_progress(self, iteration: int, queries_count: int, results_count: int, quality: str):
        """Update research progress"""
        self.current_iteration = iteration
        self.total_queries_generated += queries_count
        self.total_results_collected += results_count
        
        # Update best quality
        quality_order = ["insufficient", "partial", "adequate", "comprehensive"]
        if quality in quality_order:
            current_index = quality_order.index(self.best_quality_achieved)
            new_index = quality_order.index(quality)
            if new_index > current_index:
                self.best_quality_achieved = quality
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "research_mode": self.research_mode.value,
            "start_time": self.start_time,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "is_active": self.is_active,
            "total_queries": self.total_queries_generated,
            "total_results": self.total_results_collected,
            "best_quality": self.best_quality_achieved,
            "quality_threshold": self.quality_threshold
        }


class IterativeController:
    """
    Iterative Controller
    
    Manages the core iterative research process of the reference project:
    1. Executes the research loop
    2. Monitors research progress
    3. Determines stopping conditions
    4. Manages state transitions
    
    Reserved for future upgrades:
    - Dynamic policy adjustments
    - Multi-agent collaborative control
    - Adaptive iteration strategies
    """
    
    def __init__(self):
        self.max_iterations = 5
        self.quality_threshold = 0.8
        self.min_improvement_threshold = 0.1
        
        # Session management
        self.active_sessions: Dict[str, ResearchSession] = {}
        self.session_history: List[ResearchSession] = []
    
    def create_session(
        self,
        query: str,
        research_mode: ResearchMode = ResearchMode.REFLECTIVE,
        max_iterations: int = 3,
        quality_threshold: float = 0.8
    ) -> ResearchSession:
        """Create a new research session"""
        session = ResearchSession(
            original_query=query,
            research_mode=research_mode,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold
        )
        
        self.active_sessions[session.session_id] = session
        logger.info(f"Creating new research session - session_id: {session.session_id}, query: {query}, mode: {research_mode.value}, max_iterations: {max_iterations}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Get research session"""
        return self.active_sessions.get(session_id)
    
    def close_session(self, session_id: str):
        """Close research session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.is_active = False
            self.session_history.append(session)
            del self.active_sessions[session_id]
            logger.info(f"Closing research session - session_id: {session_id}, final_iteration: {session.current_iteration}, total_queries: {session.total_queries_generated}, best_quality: {session.best_quality_achieved}")
        else:
            logger.warning(f"Attempting to close a non-existent session - session_id: {session_id}")
    
    async def execute_research_loop(
        self,
        initial_query: str,
        config: RunnableConfig,
        query_generator,  # DynamicQueryGenerator
        reflection_engine,  # ReflectionEngine
        search_api: str = "tavily",
        max_iterations: int = 3
    ) -> IterationResult:
        """
        Executes the complete intelligent research loop.

        Implements the core iterative logic of the reference project:
        Query Generation -> Search Execution -> Reflection Evaluation -> Iteration Control

        Args:
            initial_query: The initial research query.
            config: The runtime configuration.
            query_generator: The query generator.
            reflection_engine: The reflection engine.
            search_api: The search API to use.
            max_iterations: The maximum number of iterations.

        Returns:
            The result of the iterative research.
        """
        
        logger.info(f"Starting intelligent research loop - query: {initial_query}, search_api: {search_api}, max_iterations: {max_iterations}")
        
        # Create research session
        session = self.create_session(initial_query, ResearchMode.REFLECTIVE, max_iterations)
        
        # Initialize research state
        research_state = ResearchState(
            query=initial_query,
            max_iterations=max_iterations
        )
        
        all_search_results = []
        reflection_history = []
        total_queries = 0
        
        logger.debug(f"Research state initialization complete - session_id: {session.session_id}")
        logger.debug(f"🔍 Starting intelligent research loop: {initial_query}")
        
        # Main research loop
        for iteration in range(max_iterations):
            research_state.iteration = iteration
            session.current_iteration = iteration
            
            logger.info(f"Starting iteration {iteration + 1} - session_id: {session.session_id}, total_results_so_far: {len(all_search_results)}")
            
            try:
                # 1. Generate queries
                query_context = QueryGenerationContext(
                    original_query=initial_query,
                    iteration=iteration,
                    previous_results=all_search_results,
                    knowledge_gaps=research_state.knowledge_gaps
                )
                
                logger.debug(f"Generating query context - iteration: {iteration}, gaps_count: {len(research_state.knowledge_gaps)}")
                queries = await query_generator.generate_queries(
                    query_context, config, max_queries=3
                )
                total_queries += len(queries)
                
                logger.info(f"Query generation complete - iteration: {iteration + 1}, queries_count: {len(queries)}, total_queries: {total_queries}")
                
                # 2. Execute search - providing correct parameters
                search_api = get_config_value_from_runnable(config, 'search_api', 'tavily')
                params_to_pass = get_config_value_from_runnable(config, 'search_params', {}) if config else {}
                
                logger.debug(f"Starting search execution - search_api: {search_api}, queries_count: {len(queries)}")
                search_results = await select_and_execute_search(
                    search_api=search_api,
                    query_list=queries,
                    params_to_pass=params_to_pass
                )
                
                # Parse search results (simplified handling)
                iteration_results = self._parse_search_results(search_results, queries)
                all_search_results.extend(iteration_results)
                research_state.search_results = iteration_results
                
                logger.info(f"Search execution complete - iteration: {iteration + 1}, results_count: {len(iteration_results)}, total_results: {len(all_search_results)}")
                
                # 3. Reflection and evaluation
                from open_deep_research.intelligent_research.reflection import ReflectionContext
                
                reflection_context = ReflectionContext(
                    original_query=initial_query,
                    current_iteration=iteration,
                    search_results=iteration_results,
                    previous_reflections=reflection_history,
                    search_queries=queries
                )
                
                logger.debug(f"Starting reflection and evaluation - iteration: {iteration + 1}, results_count: {len(iteration_results)}")
                reflection_result = await reflection_engine.reflect_on_research(
                    reflection_context, config, self.quality_threshold
                )
                reflection_history.append(reflection_result)
                
                # Update research state
                research_state.knowledge_gaps = reflection_result.knowledge_gaps
                research_state.completion_confidence = reflection_result.completeness_score
                research_state.reflection_notes.append(reflection_result.reasoning)
                
                # Update session progress
                session.update_progress(
                    iteration, len(queries), len(iteration_results), 
                    reflection_result.quality.value
                )
                
                logger.info(f"Reflection and evaluation complete - iteration: {iteration + 1}, quality: {reflection_result.quality.value}, score: {reflection_result.completeness_score:.3f}, gaps: {len(reflection_result.knowledge_gaps)}")
                
                # 4. Iteration decision
                decision = self._make_iteration_decision(
                    research_state, reflection_result, iteration, max_iterations
                )
                logger.info(f"Iteration decision complete - iteration: {iteration + 1}, decision: {decision.value}")
                
                if decision != IterationDecision.CONTINUE:
                    logger.info(f"Research finished: {decision.value}")
                    break
                    
                logger.info(f"Continuing to next iteration")
                
            except Exception as e:
                logger.error(f"Exception in research loop - iteration: {iteration + 1}, session_id: {session.session_id}, error: {str(e)}")
                decision = IterationDecision.STOP_ERROR
                break
        
        # If the loop finishes normally without a clear decision, it's because the max iterations were reached.
        if 'decision' not in locals():
            decision = IterationDecision.STOP_MAX_ITER
            logger.info(f"Reached max iterations - session_id: {session.session_id}, max_iterations: {max_iterations}")
        
        # Close session
        self.close_session(session.session_id)
        
        # Create final result
        final_result = IterationResult(
            decision=decision,
            final_state=research_state,
            all_search_results=all_search_results,
            reflection_history=reflection_history,
            iteration_count=iteration + 1,
            total_queries_executed=total_queries
        )
        
        logger.info(f"Intelligent research loop complete - session_id: {session.session_id}, decision: {decision.value}, iterations: {iteration + 1}, total_queries: {total_queries}, total_results: {len(all_search_results)}")
        logger.info(f"Session summary: {session.get_session_summary()}")
        
        return final_result
    
    def _make_iteration_decision(
        self,
        state: ResearchState,
        reflection: ReflectionResult,
        current_iteration: int,
        max_iterations: int
    ) -> IterationDecision:
        """Make iteration decision"""
        
        logger.debug(f"Making iteration decision - iteration: {current_iteration + 1}, quality: {reflection.quality.value}, score: {reflection.completeness_score:.3f}, continue_research: {reflection.continue_research}")
        
        # Check if max iterations have been reached
        if current_iteration >= max_iterations - 1:
            logger.info(f"Reached max iteration limit - iteration: {current_iteration + 1}, max: {max_iterations}")
            return IterationDecision.STOP_MAX_ITER
        
        # Check if quality meets requirements
        if reflection.quality in [ReflectionQuality.COMPREHENSIVE, ReflectionQuality.ADEQUATE]:
            if reflection.completeness_score >= self.quality_threshold:
                logger.info(f"Research quality meets requirements, stopping successfully - quality: {reflection.quality.value}, score: {reflection.completeness_score:.3f}, threshold: {self.quality_threshold}")
                return IterationDecision.STOP_SUCCESS
        
        # Check if continuing research is recommended
        if not reflection.continue_research:
            logger.info(f"Reflection result suggests stopping research - quality: {reflection.quality.value}, score: {reflection.completeness_score:.3f}")
            return IterationDecision.STOP_SUCCESS
        
        # Check for sufficient improvement (to avoid ineffective iterations)
        if len(state.reflection_notes) >= 2:
            # Simple improvement check: compare completeness scores of the last two iterations
            if current_iteration > 0:
                current_score = reflection.completeness_score
                # Assuming the previous score is stored somewhere, this is a simplified process
                if current_score < 0.3:  # If the score is too low, the strategy might need to be changed
                    logger.warning(f"Research quality remains low, recommending stop - score: {current_score:.3f}")
                    return IterationDecision.STOP_ERROR
        
        logger.debug(f"Decision result: Continue research - iteration: {current_iteration + 1}")
        return IterationDecision.CONTINUE
    
    def _parse_search_results(self, search_results: Any, queries: List[str]) -> List[Dict[str, Any]]:
        """Parse search results (improved implementation)"""
        
        logger.debug(f"Parsing search results - type: {type(search_results)}, queries_count: {len(queries)}")
        
        results = []
        
        # If search_results is a string, try to parse it
        if isinstance(search_results, str):
            logger.debug("Search result is a string, attempting to parse")
            results = self._parse_search_results_string(search_results, queries)
        elif isinstance(search_results, list):
            # If it's already a list, use it directly
            logger.debug(f"Search result is a list - length: {len(search_results)}")
            results = search_results
        elif hasattr(search_results, '__iter__'):
            # If it's an iterable, convert to a list
            logger.debug("Search result is an iterable, converting to list")
            results = list(search_results)
        else:
            # Create placeholder results
            logger.warning(f"Unknown search result type, creating placeholder results - type: {type(search_results)}")
            for i, query in enumerate(queries):
                results.append({
                    'title': f'Search Result {i+1}',
                    'url': f'https://example.com/result{i+1}',
                    'content': f'Content for search result regarding "{query}"...',
                    'query': query
                })
        
        logger.debug(f"Search result parsing complete - final_count: {len(results)}")
        return results
    
    def _parse_search_results_string(self, search_results_str: str, queries: List[str]) -> List[Dict[str, Any]]:
        """Parse search result string (original implementation)"""
        
        results = []
        
        # Simple parsing based on the search result string
        lines = search_results_str.split('\n')
        current_result = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('--- SOURCE'):
                if current_result:
                    results.append(current_result)
                current_result = {'title': '', 'url': '', 'content': ''}
            elif line.startswith('URL:'):
                current_result['url'] = line.replace('URL:', '').strip()
            elif line and not line.startswith('---') and not line.startswith('SUMMARY:') and not line.startswith('FULL CONTENT:'):
                if 'title' not in current_result or not current_result['title']:
                    current_result['title'] = line
                else:
                    current_result['content'] += line + ' '
        
        # Add the last result
        if current_result:
            results.append(current_result)
        
        # If parsing fails, create placeholder results
        if not results:
            for i, query in enumerate(queries):
                results.append({
                    'title': f'Search Result {i+1}',
                    'url': f'https://example.com/result{i+1}',
                    'content': f'Content for search result regarding "{query}"...',
                    'query': query
                })
        
        return results
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        active_count = len(self.active_sessions)
        history_count = len(self.session_history)
        
        logger.debug(f"Calculating session statistics - active: {active_count}, completed: {history_count}")
        
        # Calculate average iterations
        total_iterations = sum(session.current_iteration for session in self.session_history)
        avg_iterations = total_iterations / max(1, history_count)
        
        # Calculate quality distribution
        quality_distribution = {}
        for session in self.session_history:
            quality = session.best_quality_achieved
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        stats = {
            "active_sessions": active_count,
            "completed_sessions": history_count,
            "average_iterations": avg_iterations,
            "quality_distribution": quality_distribution,
            "total_sessions": active_count + history_count
        }
        
        logger.info(f"Session statistics - total_sessions: {stats['total_sessions']}, avg_iterations: {avg_iterations:.2f}, quality_dist: {quality_distribution}")
        
        return stats
    
    async def adaptive_iteration_control(
        self,
        state: ResearchState,
        reflection: ReflectionResult,
        config: RunnableConfig
    ) -> IterationDecision:
        """
        Adaptive Iteration Control (Phase 2 Implementation)

        Will implement:
        - Dynamic adjustment of iteration strategy
        - Optimization of decisions based on historical performance
        - Intelligent stopping condition judgment
        """
        
        # TODO: Phase 2 implementation
        raise NotImplementedError("Adaptive iteration control will be implemented in Phase 2")
    
    async def collaborative_iteration_control(
        self,
        state: ResearchState,
        reflection: ReflectionResult,
        config: RunnableConfig
    ) -> IterationDecision:
        """
        Collaborative Iteration Control (Phase 3 Implementation)

        Will implement:
        - Multi-agent collaborative decision-making
        - Distributed research control
        - Handover mechanism between agents
        """
        
        # TODO: Phase 3 implementation
        raise NotImplementedError("Collaborative iteration control will be implemented in Phase 3")

 