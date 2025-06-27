"""
Intelligent Research Module Configuration Management

Provides flexible configuration options for intelligent research features,
supporting a progressive upgrade from simple to advanced intelligent modes.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from open_deep_research.intelligent_research.core import ResearchMode


@dataclass
class IntelligentResearchConfig:
    """Intelligent research configuration"""
    
    # Basic configuration
    mode: ResearchMode = ResearchMode.SIMPLE
    max_iterations: int = 3
    enable_reflection: bool = False  # Enabled in Phase 2
    enable_dynamic_queries: bool = False  # Enabled in Phase 2
    
    # Query generation configuration (Phase 2)
    max_queries_per_iteration: int = 3
    query_diversity_threshold: float = 0.7
    
    # Reflection configuration (Phase 2) 
    reflection_model: Optional[str] = None
    reflection_temperature: float = 0.1
    quality_threshold: float = 0.8
    
    # Iterative control configuration (Phase 2)
    early_stopping_enabled: bool = True
    confidence_threshold: float = 0.85
    
    # Configuration reserved for Stage Two
    extended_thinking_enabled: bool = False  # Supervisor 3.0
    interleaved_thinking_enabled: bool = False  # Researcher 3.0
    
    # Configuration reserved for Stage Three
    collaborative_mode_enabled: bool = False
    reflection_tool_enabled: bool = False
    debate_tool_enabled: bool = False


class IntelligentResearchConfigManager:
    """Intelligent research configuration manager"""
    
    def __init__(self):
        self._configs: Dict[str, IntelligentResearchConfig] = {}
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Set up default configurations"""
        
        # Phase 1: Simple mode (current)
        self._configs["simple"] = IntelligentResearchConfig(
            mode=ResearchMode.SIMPLE,
            max_iterations=1,
            enable_reflection=False,
            enable_dynamic_queries=False
        )
        
        # Phase 2: Basic reflection mode
        self._configs["reflective"] = IntelligentResearchConfig(
            mode=ResearchMode.REFLECTIVE,
            max_iterations=3,
            enable_reflection=True,
            enable_dynamic_queries=True,
            max_queries_per_iteration=3,
            quality_threshold=0.8
        )
        
        # Stage Two: Extended thinking mode (Supervisor 3.0)
        self._configs["extended"] = IntelligentResearchConfig(
            mode=ResearchMode.EXTENDED,
            max_iterations=5,
            enable_reflection=True,
            enable_dynamic_queries=True,
            extended_thinking_enabled=True,
            max_queries_per_iteration=5,
            quality_threshold=0.85
        )
        
        # Stage Two: Interleaved thinking loop (Researcher 3.0)
        self._configs["iterative"] = IntelligentResearchConfig(
            mode=ResearchMode.ITERATIVE,
            max_iterations=7,
            enable_reflection=True,
            enable_dynamic_queries=True,
            interleaved_thinking_enabled=True,
            max_queries_per_iteration=4,
            quality_threshold=0.9
        )
        
        # Stage Three: Multi-agent collaborative mode
        self._configs["collaborative"] = IntelligentResearchConfig(
            mode=ResearchMode.COLLABORATIVE,
            max_iterations=10,
            enable_reflection=True,
            enable_dynamic_queries=True,
            collaborative_mode_enabled=True,
            reflection_tool_enabled=True,
            debate_tool_enabled=True,
            max_queries_per_iteration=6,
            quality_threshold=0.95
        )
    
    def get_config(self, name: str) -> IntelligentResearchConfig:
        """Get configuration"""
        if name not in self._configs:
            raise ValueError(f"Unknown config: {name}. Available: {list(self._configs.keys())}")
        return self._configs[name]
    
    def register_config(self, name: str, config: IntelligentResearchConfig):
        """Register a custom configuration"""
        self._configs[name] = config
    
    def list_configs(self) -> List[str]:
        """List all available configurations"""
        return list(self._configs.keys())
    
    def get_current_phase_configs(self) -> List[str]:
        """Get configurations available in the current phase"""
        # Phase 1: Only simple mode
        return ["simple"]
    
    def get_phase2_configs(self) -> List[str]:
        """Get configurations available in Phase 2"""
        return ["simple", "reflective"]
    
    def get_stage2_configs(self) -> List[str]:
        """Get configurations available in Stage Two"""
        return ["simple", "reflective", "extended", "iterative"]
    
    def get_stage3_configs(self) -> List[str]:
        """Get configurations available in Stage Three"""
        return list(self._configs.keys())


# Global configuration manager instance
config_manager = IntelligentResearchConfigManager() 