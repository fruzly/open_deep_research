"""
Intelligent Research Module - Provides core intelligence for Supervisor 3.0 and Researcher 3.0

This module implements the core intelligent mechanisms from the reference project gemini-fullstack-langgraph-quickstart:
- Dynamic Query Generation
- Intelligent Reflection Loop
- Iterative Optimization Control

Design Principles:
1. Progressive Complexity: From simple reflection to extended thinking patterns
2. Extensible Architecture: Reserve interfaces for future GenericAgent and dynamic tool encapsulation
3. Backward Compatibility: Does not break existing workflows
"""

from .core import (
    IntelligentResearchNode, 
    IntelligentResearchManager,
    ResearchMode,
    ResearchState,
    IntelligentResearchInterface
)
from .query_generator import (
    DynamicQueryGenerator,
    QueryGenerationContext,
    QueryGenerationRequest
)
from .reflection import (
    ReflectionEngine,
    ReflectionContext,
    ReflectionResult,
    ReflectionQuality
)
from .iterative_controller import (
    IterativeController,
    ResearchSession,
    IterationResult,
    IterationDecision
)
from .config import (
    IntelligentResearchConfig,
    IntelligentResearchConfigManager,
    config_manager
)
from .integration import (
    IntelligentResearchIntegration,
    IntelligentSearchResult,
    intelligent_search_web,
    get_intelligent_search_details,
    get_research_performance_stats,
    supervisor_3_0_search,
    researcher_3_0_search
)

__all__ = [
    # Core Components
    "IntelligentResearchNode",
    "IntelligentResearchManager", 
    "IntelligentResearchInterface",
    
    # Enums and States
    "ResearchMode",
    "ResearchState",
    "ReflectionQuality",
    "IterationDecision",
    
    # Query Generation
    "DynamicQueryGenerator",
    "QueryGenerationContext",
    "QueryGenerationRequest",
    
    # Reflection Evaluation
    "ReflectionEngine",
    "ReflectionContext", 
    "ReflectionResult",
    
    # Iterative Control
    "IterativeController",
    "ResearchSession",
    "IterationResult",
    
    # Configuration Management
    "IntelligentResearchConfig",
    "IntelligentResearchConfigManager",
    "config_manager",
    
    # Integration Interface
    "IntelligentResearchIntegration",
    "IntelligentSearchResult",
    "intelligent_search_web",
    "get_intelligent_search_details", 
    "get_research_performance_stats",
    
    # Stage Two Upgrade Interface
    "supervisor_3_0_search",
    "researcher_3_0_search"
]

__version__ = "0.2.0"  # Upgraded version number reflects new features 