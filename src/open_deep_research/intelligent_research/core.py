"""
Core implementation of the intelligent research node

Following the architectural pattern of gemini-fullstack-langgraph-quickstart,
this module implements extensible intelligent research capabilities.
"""

from typing import Dict, List, Any, Optional, Literal
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import traceback

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage

# Import logging system
from open_deep_research.util.logging import get_logger
from open_deep_research.utils import get_config_value_from_runnable

# Create logger
logger = get_logger(__name__)

class ResearchMode(Enum):
    """Research mode enum - reserved for stage two upgrade"""
    SIMPLE = "simple"           # Current: simple search mode
    REFLECTIVE = "reflective"   # Phase 2: basic reflection mode
    EXTENDED = "extended"       # Stage two: extended thinking mode (Supervisor 3.0)
    ITERATIVE = "iterative"     # Stage two: interleaved thinking loop (Researcher 3.0)
    COLLABORATIVE = "collaborative"  # Stage three: multi-agent collaborative mode


@dataclass
class ResearchState:
    """Research state - tracks research progress and knowledge gaps"""
    query: str
    iteration: int = 0
    max_iterations: int = 3
    knowledge_gaps: List[str] = None
    search_results: List[Dict[str, Any]] = None
    reflection_notes: List[str] = None
    completion_confidence: float = 0.0
    
    def __post_init__(self):
        if self.knowledge_gaps is None:
            self.knowledge_gaps = []
        if self.search_results is None:
            self.search_results = []
        if self.reflection_notes is None:
            self.reflection_notes = []


class IntelligentResearchInterface(ABC):
    """Intelligent research interface - reserved for future GenericAgent"""
    
    @abstractmethod
    async def generate_queries(self, state: ResearchState, config: RunnableConfig) -> List[str]:
        """Dynamically generate search queries"""
        pass
    
    @abstractmethod
    async def reflect_on_results(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Reflect on the quality of search results"""
        pass
    
    @abstractmethod
    async def should_continue_research(self, state: ResearchState, config: RunnableConfig) -> bool:
        """Determine whether to continue research"""
        pass


class IntelligentResearchNode(IntelligentResearchInterface):
    """
    Main implementation of the intelligent research node
    
    Integrates the core intelligent mechanisms of the reference project:
    1. Dynamic query generation
    2. Intelligent reflection loop
    3. Iterative optimization control
    """
    
    def __init__(self, mode: ResearchMode = ResearchMode.SIMPLE):
        logger.info(f"Initializing intelligent research node - mode: {mode.value}")
        self.mode = mode
        self.query_generator = None  # To be set via dependency injection
        self.reflection_engine = None  # To be set via dependency injection
        self.iterative_controller = None  # To be set via dependency injection
        logger.debug(f"Intelligent research node initialization complete - mode: {mode.value}")
    
    def set_components(self, query_generator, reflection_engine, iterative_controller):
        """Set component dependencies"""
        logger.info("Setting intelligent research node component dependencies")
        self.query_generator = query_generator
        self.reflection_engine = reflection_engine
        self.iterative_controller = iterative_controller
        
        # Log component status
        components_status = {
            "query_generator": query_generator is not None,
            "reflection_engine": reflection_engine is not None,
            "iterative_controller": iterative_controller is not None
        }
        logger.info(f"Component dependencies set - query_generator: {components_status['query_generator']}, reflection_engine: {components_status['reflection_engine']}, iterative_controller: {components_status['iterative_controller']}")
    
    async def research_with_intelligence(
        self, 
        initial_query: str, 
        config: RunnableConfig,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Main intelligent research flow
        
        Implements the core research loop from the reference project:
        Initial query -> Dynamic query generation -> Parallel search -> Reflection & evaluation -> Iterative control
        """
        logger.info(f"Starting intelligent research - query: {initial_query}, mode: {self.mode.value}, max_iterations: {max_iterations}")
        
        try:
            state = ResearchState(
                query=initial_query,
                max_iterations=max_iterations
            )
            
            logger.debug(f"Created research state - query: {initial_query}, max_iterations: {max_iterations}")
            
            # Current stage: simple mode, directly return the original query
            if self.mode == ResearchMode.SIMPLE:
                logger.info("Using simple research mode")
                result = await self._simple_research(state, config)
                logger.info(f"Simple research complete - result_queries_count: {len(result.get('queries', []))}")
                return result
            
            # Implement the full intelligent loop
            logger.info("Starting intelligent research loop")
            result = await self._intelligent_research_loop(state, config)
            logger.info(f"Intelligent research complete - mode: {self.mode.value}, total_iterations: {result.get('iterations', 0)}, confidence: {result.get('completion_confidence', 0.0)}")
            return result
            
        except Exception as e:
            logger.error(f"Intelligent research execution failed - query: {initial_query}, mode: {self.mode.value}, error: {str(e)}", exc_info=True)
            # Return an error result instead of raising an exception
            return {
                "queries": [initial_query],
                "mode": "error",
                "iterations": 0,
                "completion_confidence": 0.0,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _simple_research(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Simple research mode - current implementation"""
        logger.debug(f"Executing simple research mode - query: {state.query}")
        
        try:
            # Maintain compatibility with the existing system
            result = {
                "queries": [state.query],
                "mode": "simple",
                "iterations": 1,
                "completion_confidence": 1.0
            }
            
            logger.debug(f"Simple research mode complete - result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Simple research mode execution failed - error: {str(e)}", exc_info=True)
            raise
    
    async def _intelligent_research_loop(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Intelligent research loop - now implemented"""
        logger.info(f"Starting intelligent research loop - query: {state.query}, max_iterations: {state.max_iterations}")
        
        if not self.query_generator or not self.reflection_engine:
            logger.warning(f"Components not set, falling back to simple mode - has_query_generator: {self.query_generator is not None}, has_reflection_engine: {self.reflection_engine is not None}")
            # Fallback to simple mode if components are not set
            return await self._simple_research(state, config)
        
        try:
            from open_deep_research.intelligent_research.query_generator import QueryGenerationContext
            from open_deep_research.intelligent_research.reflection import ReflectionContext
            from open_deep_research.utils import select_and_execute_search
            
            logger.debug("Successfully imported intelligent research related modules")
            
            all_queries = []
            all_results = []
            iteration_details = []
            
            while state.iteration < state.max_iterations:
                logger.info(f"Starting research iteration - iteration: {state.iteration + 1}, max_iterations: {state.max_iterations}, current_confidence: {state.completion_confidence}")
                
                try:
                    # 1. Dynamic query generation
                    logger.debug(f"Starting dynamic query generation - iteration: {state.iteration}")
                    query_context = QueryGenerationContext(
                        original_query=state.query,
                        iteration=state.iteration,
                        previous_results=state.search_results,
                        knowledge_gaps=state.knowledge_gaps
                    )
                    
                    queries = await self.query_generator.generate_queries(
                        query_context, config, max_queries=3
                    )
                    all_queries.extend(queries)
                    
                    logger.info(f"Query generation complete - iteration: {state.iteration}, generated_queries: {len(queries)}, queries: {queries}")
                    
                    # 🆕 Add query diversity evaluation
                    if len(queries) > 1:
                        diversity_score = self.query_generator.evaluate_query_diversity(queries)
                        logger.info(f"Query diversity evaluation - iteration: {state.iteration}, diversity_score: {diversity_score:.3f}")
                        
                        # Log a warning if diversity is low
                        if diversity_score < 0.3:
                            logger.warning(f"Low query diversity - iteration: {state.iteration}, diversity_score: {diversity_score:.3f}, suggest increasing query variation")
                            state.reflection_notes.append(f"Iteration {state.iteration} query diversity is low: {diversity_score:.3f}")
                        else:
                            logger.info(f"Good query diversity - iteration: {state.iteration}, diversity_score: {diversity_score:.3f}")
                    else:
                        logger.debug(f"Not enough queries, skipping diversity evaluation - iteration: {state.iteration}, query_count: {len(queries)}")
                    
                    # 2. Parallel search execution - provide correct parameters
                    logger.debug("Starting parallel search execution")
                    search_api = get_config_value_from_runnable(config, 'search_api', 'tavily')
                    params_to_pass = get_config_value_from_runnable(config, 'search_params', {})
                    
                    logger.debug(f"Search parameters - search_api: {search_api}, params: {params_to_pass}, query_count: {len(queries)}")
                    
                    search_content = await select_and_execute_search(
                        search_api=search_api,
                        query_list=queries,
                        params_to_pass=params_to_pass
                    )
                    
                    # Convert search content to result format
                    search_results = [{"title": f"Core search results - Iteration {state.iteration}", "content": search_content, "url": "core_search"}]
                    state.search_results = search_results
                    all_results.extend(search_results)
                    
                    logger.info(f"Search execution complete - iteration: {state.iteration}, search_content_length: {len(search_content)}, results_count: {len(search_results)}")
                    
                    # 3. Reflection & evaluation
                    logger.debug("Starting reflection and evaluation")
                    reflection_context = ReflectionContext(
                        original_query=state.query,
                        current_iteration=state.iteration,
                        search_results=search_results,
                        previous_reflections=[],
                        search_queries=queries
                    )
                    
                    reflection_result = await self.reflection_engine.reflect_on_research(
                        reflection_context, config
                    )
                    
                    # Update state
                    state.knowledge_gaps = reflection_result.knowledge_gaps
                    state.completion_confidence = reflection_result.completeness_score
                    state.reflection_notes.append(reflection_result.reasoning)
                    
                    logger.info(f"Reflection and evaluation complete - iteration: {state.iteration}, quality: {reflection_result.quality.value}, confidence: {reflection_result.completeness_score}, knowledge_gaps: {len(reflection_result.knowledge_gaps)}, continue_research: {reflection_result.continue_research}")
                    
                    # Record iteration details
                    iteration_detail = {
                        "iteration": state.iteration,
                        "queries": queries,
                        "results_count": len(search_results),
                        "quality": reflection_result.quality.value,
                        "confidence": reflection_result.completeness_score,
                        "continue": reflection_result.continue_research
                    }
                    iteration_details.append(iteration_detail)
                    
                    logger.debug(f"Iteration detail recorded - iteration_detail: {iteration_detail}")
                    
                    # 4. Iteration control
                    if not reflection_result.continue_research:
                        logger.info(f"Reflection decided to stop research - iteration: {state.iteration}, reason: continue_research=False")
                        break
                        
                    state.iteration += 1
                    logger.debug(f"Continuing to next iteration - next_iteration: {state.iteration}")
                    
                except Exception as iteration_error:
                    logger.error(f"Research iteration failed - iteration: {state.iteration}, error: {str(iteration_error)}", exc_info=True)
                    break
            
            final_result = {
                "queries": all_queries,
                "mode": self.mode.value,
                "iterations": state.iteration + 1,
                "completion_confidence": state.completion_confidence,
                "all_results": all_results,
                "iteration_details": iteration_details,
                "knowledge_gaps": state.knowledge_gaps,
                "reflection_notes": state.reflection_notes
            }
            
            logger.info(f"Intelligent research loop complete - total_queries: {len(all_queries)}, total_iterations: {state.iteration + 1}, final_confidence: {state.completion_confidence}, mode: {self.mode.value}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Intelligent research loop failed - query: {state.query}, iteration: {state.iteration}, error: {str(e)}", exc_info=True)
            print(f"Intelligent research loop error: {str(e)}")
            # Return partial results instead of raising an exception
            return {
                "queries": all_queries if 'all_queries' in locals() else [state.query],
                "mode": self.mode.value,
                "iterations": state.iteration,
                "completion_confidence": state.completion_confidence,
                "error": str(e),
                "partial_results": True
            }
    
    # IntelligentResearchInterface implementation
    async def generate_queries(self, state: ResearchState, config: RunnableConfig) -> List[str]:
        """Dynamically generate search queries"""
        logger.debug(f"Generating queries - query: {state.query}, iteration: {state.iteration}")
        
        try:
            if self.query_generator:
                from open_deep_research.intelligent_research.query_generator import QueryGenerationContext
                context = QueryGenerationContext(
                    original_query=state.query,
                    iteration=state.iteration,
                    previous_results=state.search_results,
                    knowledge_gaps=state.knowledge_gaps
                )
                queries = await self.query_generator.generate_queries(context, config)
                logger.info(f"Query generation successful - query_count: {len(queries)}, queries: {queries}")
                return queries
            else:
                # Fallback to original query
                logger.warning("Query generator not set, falling back to original query")
                return [state.query]
                
        except Exception as e:
            logger.error(f"Query generation failed - error: {str(e)}", exc_info=True)
            return [state.query]  # Return original query as fallback
    
    async def reflect_on_results(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        """Reflect on the quality of search results"""
        logger.debug(f"Reflecting on search results - query: {state.query}, iteration: {state.iteration}, results_count: {len(state.search_results)}")
        
        try:
            if self.reflection_engine:
                from open_deep_research.intelligent_research.reflection import ReflectionContext
                context = ReflectionContext(
                    original_query=state.query,
                    current_iteration=state.iteration,
                    search_results=state.search_results,
                    previous_reflections=[]
                )
                reflection_result = await self.reflection_engine.reflect_on_research(context, config)
                
                result = {
                    "quality_score": reflection_result.completeness_score,
                    "knowledge_gaps": reflection_result.knowledge_gaps,
                    "continue_research": reflection_result.continue_research,
                    "reasoning": reflection_result.reasoning
                }
                
                logger.info(f"Reflection complete - quality_score: {result['quality_score']}, gaps_count: {len(result['knowledge_gaps'])}, continue_research: {result['continue_research']}")
                return result
            else:
                logger.warning("Reflection engine not set, returning default result")
                return {
                    "quality_score": 1.0,
                    "knowledge_gaps": [],
                    "continue_research": False
                }
                
        except Exception as e:
            logger.error(f"Reflection execution failed - error: {str(e)}", exc_info=True)
            return {
                "quality_score": 0.5,
                "knowledge_gaps": [],
                "continue_research": False,
                "error": str(e)
            }
    
    async def should_continue_research(self, state: ResearchState, config: RunnableConfig) -> bool:
        """Determine whether to continue research"""
        should_continue = state.iteration < state.max_iterations and state.completion_confidence < 0.8
        
        logger.debug(f"Determining whether to continue research - iteration: {state.iteration}, max_iterations: {state.max_iterations}, confidence: {state.completion_confidence}, should_continue: {should_continue}")
        
        return should_continue


class IntelligentResearchManager:
    """
    Intelligent Research Manager
    
    Manages the coordination and configuration of the entire intelligent research system:
    - Component initialization and dependency injection
    - Research mode switching
    - Performance monitoring and optimization
    - Reserves interfaces for stage two upgrade
    """
    
    def __init__(self):
        logger.info("Initializing Intelligent Research Manager")
        self.research_nodes: Dict[ResearchMode, IntelligentResearchNode] = {}
        self.components_initialized = False
        
        # Performance statistics
        self.stats = {
            "total_research_sessions": 0,
            "successful_sessions": 0,
            "average_iterations": 0.0,
            "mode_usage": {mode.value: 0 for mode in ResearchMode}
        }
        
        logger.debug(f"Intelligent Research Manager initialized - stats: {self.stats}")
    
    async def initialize_components(self):
        """Initialize all components"""
        if self.components_initialized:
            logger.debug("Components already initialized, skipping")
            return
        
        logger.info("Initializing intelligent research components")
        
        try:
            # Lazy import to avoid circular dependencies
            from open_deep_research.intelligent_research.query_generator import DynamicQueryGenerator
            from open_deep_research.intelligent_research.reflection import ReflectionEngine
            from open_deep_research.intelligent_research.iterative_controller import IterativeController
            
            logger.debug("Successfully imported component classes")
            
            # Create component instances
            query_generator = DynamicQueryGenerator()
            reflection_engine = ReflectionEngine()
            iterative_controller = IterativeController()
            
            logger.debug("Successfully created component instances")
            
            # Create research nodes for each mode
            for mode in ResearchMode:
                logger.debug(f"Creating research node - mode: {mode.value}")
                node = IntelligentResearchNode(mode)
                node.set_components(query_generator, reflection_engine, iterative_controller)
                self.research_nodes[mode] = node
            
            self.components_initialized = True
            logger.info(f"✅ Intelligent research components initialized - modes_count: {len(self.research_nodes)}")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed - error: {str(e)}", exc_info=True)
            self.components_initialized = False
            print(f"❌ Component initialization failed: {str(e)}")
    
    async def conduct_research(
        self,
        query: str,
        config: RunnableConfig,
        mode: ResearchMode = ResearchMode.REFLECTIVE,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Conduct intelligent research
        
        Args:
            query: Research query
            config: Runnable config
            mode: Research mode
            max_iterations: Maximum number of iterations
            
        Returns:
            Research results
        """
        logger.info(f"Starting to conduct intelligent research - query: {query}, mode: {mode.value}, max_iterations: {max_iterations}")
        
        # Ensure components are initialized
        await self.initialize_components()
        
        if not self.components_initialized:
            logger.error("Component initialization failed, returning fallback result")
            # Fallback to the simplest implementation
            return {
                "queries": [query],
                "mode": "fallback",
                "iterations": 1,
                "completion_confidence": 0.5,
                "error": "Component initialization failed"
            }
        
        # Update statistics
        self.stats["total_research_sessions"] += 1
        self.stats["mode_usage"][mode.value] += 1
        
        logger.debug(f"Updating research statistics - total_sessions: {self.stats['total_research_sessions']}, mode_usage: {self.stats['mode_usage'][mode.value]}")
        
        try:
            # Get the research node for the corresponding mode
            research_node = self.research_nodes.get(mode)
            if not research_node:
                logger.warning(f"Research node for specified mode not found, using simple mode - requested_mode: {mode.value}")
                research_node = self.research_nodes[ResearchMode.SIMPLE]
            
            # Conduct research
            logger.debug("Starting to execute research node")
            result = await research_node.research_with_intelligence(
                query, config, max_iterations
            )
            
            # Update success statistics
            self.stats["successful_sessions"] += 1
            
            # Update average iterations
            total_sessions = self.stats["total_research_sessions"]
            current_avg = self.stats["average_iterations"]
            iterations = result.get("iterations", 1)
            self.stats["average_iterations"] = (current_avg * (total_sessions - 1) + iterations) / total_sessions
            
            logger.info(f"Intelligent research conducted successfully - query: {query}, mode: {mode.value}, iterations: {iterations}, confidence: {result.get('completion_confidence', 0.0)}, success_rate: {self.stats['successful_sessions'] / self.stats['total_research_sessions']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Research execution failed - query: {query}, mode: {mode.value}, error: {str(e)}", exc_info=True)
            print(f"❌ Research execution failed: {str(e)}")
            return {
                "queries": [query],
                "mode": "error",
                "iterations": 0,
                "completion_confidence": 0.0,
                "error": str(e)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_sessions = self.stats["total_research_sessions"]
        success_rate = self.stats["successful_sessions"] / max(1, total_sessions)
        
        stats_result = {
            **self.stats,
            "success_rate": success_rate,
            "components_initialized": self.components_initialized
        }
        
        logger.debug(f"Getting performance statistics - stats: {stats_result}")
        return stats_result
    
    def get_available_modes(self) -> List[str]:
        """Get available research modes"""
        modes = [mode.value for mode in ResearchMode]
        logger.debug(f"Getting available modes - modes: {modes}")
        return modes
    
    # ================================
    # Methods reserved for Stage Two upgrade
    # ================================
    
    async def supervisor_3_0_research(
        self,
        query: str,
        config: RunnableConfig,
        thinking_mode: str = "extended"
    ) -> Dict[str, Any]:
        """
        Supervisor 3.0 extended thinking mode research (to be implemented in Stage Two)
        """
        logger.info(f"Attempting to call Supervisor 3.0 research - query: {query}, thinking_mode: {thinking_mode}")
        # TODO: Implement in Stage Two
        logger.warning("Supervisor 3.0 research will be implemented in Stage Two")
        raise NotImplementedError("Supervisor 3.0 research will be implemented in Stage Two")
    
    async def researcher_3_0_research(
        self,
        query: str,
        config: RunnableConfig,
        thinking_mode: str = "interleaved"
    ) -> Dict[str, Any]:
        """
        Researcher 3.0 interleaved thinking mode research (to be implemented in Stage Two)
        """
        logger.info(f"Attempting to call Researcher 3.0 research - query: {query}, thinking_mode: {thinking_mode}")
        # TODO: Implement in Stage Two
        logger.warning("Researcher 3.0 research will be implemented in Stage Two")
        raise NotImplementedError("Researcher 3.0 research will be implemented in Stage Two")

 