#!/usr/bin/env python3
"""
Haystack Auto-Instrumentation Registration System

Provides zero-code setup for Haystack AI pipeline governance by automatically
instrumenting pipelines, components, and workflows with GenOps telemetry.

Usage:
    from genops.providers.haystack import auto_instrument
    auto_instrument()
    
    # Your existing Haystack code works unchanged
    from haystack import Pipeline
    from haystack.components.generators import OpenAIGenerator
    
    pipeline = Pipeline()
    pipeline.add_component("generator", OpenAIGenerator())
    # ... add more components ...
    
    result = pipeline.run({"query": "What is RAG?"})
    # ✅ Automatic cost tracking and governance added!

Features:
    - Zero-code instrumentation for existing Haystack applications
    - Automatic pipeline and component monitoring
    - Multi-provider cost tracking and governance
    - RAG and agent workflow specialization
    - Configurable instrumentation policies
    - Production-ready auto-instrumentation with minimal overhead
"""

import functools
import logging
import sys
import threading
import weakref
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union, TYPE_CHECKING
import inspect

# GenOps imports - using TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from genops.providers.haystack_adapter import GenOpsHaystackAdapter
    from genops.providers.haystack_monitor import HaystackMonitor
    from genops.providers.haystack_cost_aggregator import HaystackCostAggregator

logger = logging.getLogger(__name__)

# Check for Haystack availability
try:
    import haystack
    from haystack import Pipeline, component
    from haystack.core.component import Component
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
    HAS_HAYSTACK = True
    logger.debug(f"Haystack {haystack.__version__} detected for auto-instrumentation")
except ImportError:
    HAS_HAYSTACK = False
    Pipeline = None
    Component = None
    component = None
    logger.warning("Haystack not installed - auto-instrumentation disabled")


class InstrumentationRegistry:
    """Registry for managing auto-instrumentation state and configuration."""
    
    def __init__(self):
        self.is_instrumented = False
        self.instrumented_classes: Set[Type] = set()
        self.original_methods: Dict[str, Callable] = {}
        self.adapter: Optional['GenOpsHaystackAdapter'] = None
        self.monitor: Optional['HaystackMonitor'] = None
        self.cost_aggregator: Optional['HaystackCostAggregator'] = None
        self._lock = threading.RLock()
        
        # Configuration
        self.config = {
            "team": "auto-instrumented",
            "project": "haystack-app",
            "environment": "development",
            "enable_component_tracking": True,
            "enable_cost_tracking": True,
            "enable_rag_specialization": True,
            "enable_agent_tracking": True,
            "daily_budget_limit": 100.0,
            "governance_policy": "advisory"
        }
        
        # Component patterns to instrument
        self.component_patterns = {
            "generators": [
                "OpenAIGenerator", "AnthropicGenerator", "CohereGenerator",
                "HuggingFaceGenerator", "MistralGenerator"
            ],
            "retrievers": [
                "InMemoryEmbeddingRetriever", "ChromaEmbeddingRetriever",
                "ElasticsearchRetriever", "PineconeRetriever"
            ],
            "embedders": [
                "OpenAIDocumentEmbedder", "OpenAITextEmbedder",
                "HuggingFaceDocumentEmbedder", "CohereDocumentEmbedder"
            ],
            "rankers": [
                "TransformersRanker", "CohereRanker", "SentenceTransformersRanker"
            ],
            "converters": [
                "HTMLToDocument", "PDFToDocument", "TextFileToDocument"
            ]
        }
    
    def update_config(self, **kwargs):
        """Update instrumentation configuration."""
        with self._lock:
            self.config.update(kwargs)
            
            # Reinitialize components if already instrumented
            if self.is_instrumented:
                self._initialize_components()
    
    def _initialize_components(self):
        """Initialize GenOps components with current configuration."""
        # Import at runtime to avoid circular imports
        from genops.providers.haystack_adapter import GenOpsHaystackAdapter
        from genops.providers.haystack_monitor import HaystackMonitor
        from genops.providers.haystack_cost_aggregator import HaystackCostAggregator
        
        self.adapter = GenOpsHaystackAdapter(
            team=self.config["team"],
            project=self.config["project"], 
            environment=self.config["environment"],
            daily_budget_limit=self.config["daily_budget_limit"],
            governance_policy=self.config["governance_policy"]
        )
        
        self.monitor = HaystackMonitor(
            team=self.config["team"],
            project=self.config["project"],
            environment=self.config["environment"],
            enable_performance_monitoring=True,
            enable_cost_tracking=self.config["enable_cost_tracking"],
            enable_rag_specialization=self.config["enable_rag_specialization"],
            enable_agent_tracking=self.config["enable_agent_tracking"]
        )
        
        self.cost_aggregator = HaystackCostAggregator(
            budget_limit=self.config["daily_budget_limit"]
        )


# Global registry instance
_registry = InstrumentationRegistry()


def configure_auto_instrumentation(**kwargs):
    """
    Configure auto-instrumentation settings.
    
    Args:
        team: Team name for governance
        project: Project name for governance
        environment: Environment name
        enable_component_tracking: Enable component-level tracking
        enable_cost_tracking: Enable cost tracking
        enable_rag_specialization: Enable RAG workflow specialization
        enable_agent_tracking: Enable agent workflow tracking
        daily_budget_limit: Daily budget limit
        governance_policy: Governance policy ("advisory", "enforced")
        
    Example:
        configure_auto_instrumentation(
            team="ml-team",
            project="rag-chatbot",
            daily_budget_limit=50.0,
            governance_policy="enforced"
        )
    """
    _registry.update_config(**kwargs)
    logger.info(f"Auto-instrumentation configured: {kwargs}")


def is_instrumented() -> bool:
    """Check if auto-instrumentation is currently active."""
    return _registry.is_instrumented


def get_instrumentation_stats() -> Dict[str, Any]:
    """Get current instrumentation statistics."""
    return {
        "is_instrumented": _registry.is_instrumented,
        "instrumented_classes": [cls.__name__ for cls in _registry.instrumented_classes],
        "config": _registry.config.copy(),
        "has_adapter": _registry.adapter is not None,
        "has_monitor": _registry.monitor is not None,
        "pipeline_executions": len(_registry.monitor._execution_results) if _registry.monitor else 0
    }


def _create_instrumented_pipeline_run():
    """Create instrumented version of Pipeline.run method."""
    if not HAS_HAYSTACK:
        return None
    
    # Store original method
    original_run = Pipeline.run
    _registry.original_methods["Pipeline.run"] = original_run
    
    @functools.wraps(original_run)
    def instrumented_run(self, inputs: Dict[str, Any], **kwargs):
        """Instrumented version of Pipeline.run with governance tracking."""
        pipeline_name = getattr(self, 'name', 'unnamed-pipeline') or f"pipeline-{id(self)}"
        
        # Use adapter for tracking
        with _registry.adapter.track_pipeline(pipeline_name) as context:
            try:
                # Execute original pipeline
                result = original_run(self, inputs, **kwargs)
                
                # Try to extract component information
                if hasattr(self, 'graph') and hasattr(self.graph, 'nodes'):
                    for node_name in self.graph.nodes():
                        try:
                            # Get component from pipeline
                            node = self.graph.nodes[node_name].get('instance')
                            if node:
                                component_type = node.__class__.__name__
                                
                                # Import at runtime to avoid circular imports
                                from genops.providers.haystack_adapter import HaystackComponentResult
                                from decimal import Decimal
                                
                                # Estimate cost based on component type
                                estimated_cost = _estimate_component_cost(component_type, inputs)
                                
                                component_result = HaystackComponentResult(
                                    component_name=node_name,
                                    component_type=component_type,
                                    execution_time_seconds=0.1,  # Placeholder - actual timing would need hooks
                                    cost=estimated_cost,
                                    provider=_get_provider_for_component(component_type)
                                )
                                
                                context.add_component_result(component_result)
                                
                        except Exception as e:
                            logger.debug(f"Could not track component {node_name}: {e}")
                            continue
                
                return result
                
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                raise
    
    return instrumented_run


def _estimate_component_cost(component_type: str, inputs: Dict[str, Any]) -> 'Decimal':
    """Estimate cost for a component based on its type and inputs."""
    from decimal import Decimal
    
    # Cost estimates by component type
    cost_estimates = {
        'OpenAIGenerator': Decimal("0.002"),
        'AnthropicGenerator': Decimal("0.001"),
        'CohereGenerator': Decimal("0.0005"),
        'HuggingFaceGenerator': Decimal("0.0001"),
        'OpenAIDocumentEmbedder': Decimal("0.0001"),
        'OpenAITextEmbedder': Decimal("0.0001"),
        'InMemoryEmbeddingRetriever': Decimal("0.00001"),
        'ChromaEmbeddingRetriever': Decimal("0.0001"),
    }
    
    base_cost = cost_estimates.get(component_type, Decimal("0.001"))
    
    # Scale based on input size
    if inputs:
        input_text = str(inputs)
        length_multiplier = max(1, len(input_text) / 1000)  # Scale by text length
        return base_cost * Decimal(str(length_multiplier))
    
    return base_cost


def _get_provider_for_component(component_type: str) -> str:
    """Get provider name for a component type."""
    if 'OpenAI' in component_type:
        return 'openai'
    elif 'Anthropic' in component_type:
        return 'anthropic'
    elif 'Cohere' in component_type:
        return 'cohere'
    elif 'HuggingFace' in component_type:
        return 'huggingface'
    elif 'Mistral' in component_type:
        return 'mistral'
    else:
        return 'haystack'


def _create_instrumented_component_run():
    """Create instrumented version of Component.run method."""
    if not HAS_HAYSTACK or not Component:
        return None
    
    # Store original method
    original_component_run = Component.run
    _registry.original_methods["Component.run"] = original_component_run
    
    @functools.wraps(original_component_run)
    def instrumented_component_run(self, **kwargs):
        """Instrumented version of Component.run with tracking."""
        component_name = getattr(self, 'name', self.__class__.__name__)
        component_type = self.__class__.__name__
        
        # Only track if component tracking is enabled
        if not _registry.config["enable_component_tracking"]:
            return original_component_run(self, **kwargs)
        
        # Track component execution with monitor
        if _registry.monitor:
            try:
                return _registry.monitor.component_monitor.track_component_execution(
                    component_name,
                    component_type,
                    lambda: original_component_run(self, **kwargs)
                )
            except Exception as e:
                logger.debug(f"Component tracking failed for {component_name}: {e}")
                return original_component_run(self, **kwargs)
        else:
            return original_component_run(self, **kwargs)
    
    return instrumented_component_run


def _instrument_pipeline_class():
    """Instrument the Haystack Pipeline class."""
    if not HAS_HAYSTACK or Pipeline in _registry.instrumented_classes:
        return
    
    # Create instrumented run method
    instrumented_run = _create_instrumented_pipeline_run()
    if instrumented_run:
        # Monkey patch the Pipeline.run method
        Pipeline.run = instrumented_run
        _registry.instrumented_classes.add(Pipeline)
        logger.debug("Pipeline class instrumented")


def _instrument_component_classes():
    """Instrument Haystack component classes."""
    if not HAS_HAYSTACK:
        return
    
    # Instrument base Component class
    if Component and Component not in _registry.instrumented_classes:
        instrumented_component_run = _create_instrumented_component_run()
        if instrumented_component_run:
            Component.run = instrumented_component_run
            _registry.instrumented_classes.add(Component)
            logger.debug("Component base class instrumented")


def _instrument_specific_components():
    """Instrument specific component types for enhanced tracking."""
    if not HAS_HAYSTACK:
        return
    
    # Try to instrument known component types
    component_classes = []
    
    # Import and collect component classes
    try:
        from haystack.components.generators import OpenAIGenerator
        component_classes.append(OpenAIGenerator)
    except ImportError:
        pass
    
    try:
        from haystack.components.retrievers import InMemoryEmbeddingRetriever
        component_classes.append(InMemoryEmbeddingRetriever)
    except ImportError:
        pass
    
    try:
        from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
        component_classes.extend([OpenAIDocumentEmbedder, OpenAITextEmbedder])
    except ImportError:
        pass
    
    # Instrument each component class
    for component_class in component_classes:
        if component_class not in _registry.instrumented_classes:
            _instrument_single_component_class(component_class)


def _instrument_single_component_class(component_class: Type):
    """Instrument a single component class."""
    if component_class in _registry.instrumented_classes:
        return
    
    class_name = component_class.__name__
    
    # Store original run method
    original_method = component_class.run
    method_key = f"{class_name}.run"
    _registry.original_methods[method_key] = original_method
    
    @functools.wraps(original_method)
    def instrumented_run(self, **kwargs):
        """Enhanced instrumented run for specific component types."""
        component_name = getattr(self, 'name', class_name)
        
        # Use specialized monitoring if available
        if _registry.monitor and _registry.config["enable_component_tracking"]:
            try:
                result = _registry.monitor.component_monitor.monitor_component(
                    self, component_name, kwargs
                )
                return result
            except Exception as e:
                logger.debug(f"Enhanced component monitoring failed for {component_name}: {e}")
        
        # Fallback to original method
        return original_method(self, **kwargs)
    
    # Monkey patch the component class
    component_class.run = instrumented_run
    _registry.instrumented_classes.add(component_class)
    logger.debug(f"Component class {class_name} enhanced instrumentation applied")


def auto_instrument(**config):
    """
    Enable automatic instrumentation for all Haystack components and pipelines.
    
    This function monkey-patches Haystack classes to automatically add GenOps
    governance tracking to all pipeline executions and component operations.
    
    Args:
        **config: Configuration options for instrumentation
        
    Usage:
        from genops.providers.haystack import auto_instrument
        
        # Basic setup
        auto_instrument()
        
        # Custom configuration
        auto_instrument(
            team="ml-team",
            project="rag-chatbot",
            daily_budget_limit=50.0,
            governance_policy="enforced"
        )
        
        # Your existing Haystack code works unchanged
        pipeline = Pipeline()
        # ... add components ...
        result = pipeline.run({"query": "What is RAG?"})
        # ✅ Automatic cost tracking and governance added!
    """
    if not HAS_HAYSTACK:
        logger.error("Cannot enable auto-instrumentation: Haystack not installed")
        logger.error("Install with: pip install haystack-ai")
        return False
    
    with _registry._lock:
        if _registry.is_instrumented:
            logger.info("Auto-instrumentation already enabled")
            if config:
                _registry.update_config(**config)
            return True
        
        try:
            # Update configuration
            if config:
                _registry.update_config(**config)
            
            # Initialize GenOps components
            _registry._initialize_components()
            
            # Instrument Haystack classes
            _instrument_pipeline_class()
            _instrument_component_classes()
            _instrument_specific_components()
            
            # Mark as instrumented
            _registry.is_instrumented = True
            
            logger.info("Haystack auto-instrumentation enabled successfully")
            logger.info(f"Configuration: {_registry.config}")
            logger.info(f"Instrumented classes: {len(_registry.instrumented_classes)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable auto-instrumentation: {e}")
            # Attempt to rollback
            disable_auto_instrumentation()
            return False


def disable_auto_instrumentation():
    """
    Disable automatic instrumentation and restore original Haystack behavior.
    
    This function removes all monkey patches and restores the original
    Haystack class methods.
    """
    with _registry._lock:
        if not _registry.is_instrumented:
            logger.info("Auto-instrumentation not currently enabled")
            return
        
        try:
            # Restore original methods
            if HAS_HAYSTACK:
                # Restore Pipeline.run
                if "Pipeline.run" in _registry.original_methods:
                    Pipeline.run = _registry.original_methods["Pipeline.run"]
                
                # Restore Component.run
                if Component and "Component.run" in _registry.original_methods:
                    Component.run = _registry.original_methods["Component.run"]
                
                # Restore specific component methods
                for method_key, original_method in _registry.original_methods.items():
                    if "." in method_key and method_key not in ["Pipeline.run", "Component.run"]:
                        class_name, method_name = method_key.split(".", 1)
                        
                        # Find the class and restore method
                        for cls in _registry.instrumented_classes:
                            if cls.__name__ == class_name:
                                setattr(cls, method_name, original_method)
                                break
            
            # Clear registry
            _registry.is_instrumented = False
            _registry.instrumented_classes.clear()
            _registry.original_methods.clear()
            _registry.adapter = None
            _registry.monitor = None
            _registry.cost_aggregator = None
            
            logger.info("Auto-instrumentation disabled - original Haystack behavior restored")
            
        except Exception as e:
            logger.error(f"Error disabling auto-instrumentation: {e}")


def get_current_adapter() -> Optional['GenOpsHaystackAdapter']:
    """Get the current auto-instrumentation adapter."""
    return _registry.adapter


def get_current_monitor() -> Optional['HaystackMonitor']:
    """Get the current auto-instrumentation monitor."""
    return _registry.monitor


def get_cost_summary() -> Dict[str, Any]:
    """Get cost summary from auto-instrumentation."""
    if _registry.adapter:
        return _registry.adapter.get_cost_summary()
    else:
        return {"error": "Auto-instrumentation not enabled"}


def get_execution_metrics() -> Dict[str, Any]:
    """Get execution metrics from auto-instrumentation."""
    if _registry.monitor:
        return _registry.monitor.get_performance_summary()
    else:
        return {"error": "Auto-instrumentation not enabled"}


# Context manager for temporary instrumentation
class TemporaryInstrumentation:
    """Context manager for temporary auto-instrumentation."""
    
    def __init__(self, **config):
        self.config = config
        self.was_instrumented = False
    
    def __enter__(self):
        self.was_instrumented = is_instrumented()
        if not self.was_instrumented:
            auto_instrument(**self.config)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.was_instrumented:
            disable_auto_instrumentation()


# Export main functions
__all__ = [
    'auto_instrument',
    'disable_auto_instrumentation',
    'configure_auto_instrumentation',
    'is_instrumented',
    'get_instrumentation_stats',
    'get_current_adapter',
    'get_current_monitor',
    'get_cost_summary',
    'get_execution_metrics',
    'TemporaryInstrumentation'
]