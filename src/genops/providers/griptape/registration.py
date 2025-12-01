#!/usr/bin/env python3
"""
Griptape Auto-Instrumentation Registration for GenOps Governance

Provides zero-code instrumentation for Griptape AI framework, automatically
detecting and wrapping structures (Agent, Pipeline, Workflow) for governance tracking.

Usage:
    # Enable auto-instrumentation globally
    from genops.providers.griptape import auto_instrument
    
    auto_instrument(team="ai-team", project="agent-workflows")
    
    # Your existing Griptape code works unchanged
    from griptape.structures import Agent
    from griptape.tasks import PromptTask
    
    agent = Agent(tasks=[PromptTask("Summarize this text")])
    result = agent.run("Long text to summarize...")
    # ✅ Now includes full GenOps governance tracking
    
    # Manual instrumentation (more control)
    from genops.providers.griptape import instrument_griptape
    
    griptape = instrument_griptape(
        team="research",
        project="analysis",
        daily_budget_limit=50.0
    )
    
    # Use instrumented versions
    agent = griptape.create_agent([PromptTask("Analyze data")])
    pipeline = griptape.create_pipeline([task1, task2, task3])
    workflow = griptape.create_workflow([[task1, task2], [task3]])

Features:
    - Zero-code auto-instrumentation with import hook detection
    - Automatic wrapping of Griptape structures and engines
    - Driver-level instrumentation for LLM providers
    - Memory operation tracking and governance
    - Tool usage monitoring with cost attribution
    - Graceful fallback when Griptape is not available
    - Thread-safe registration and wrapper management
"""

import functools
import logging
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union
import warnings

if TYPE_CHECKING:
    from .adapter import GenOpsGriptapeAdapter

logger = logging.getLogger(__name__)

# Global registry for instrumentation state
_instrumentation_registry = {
    "enabled": False,
    "adapter": None,
    "original_classes": {},
    "wrapped_classes": {},
    "lock": threading.Lock()
}

def _is_griptape_available() -> bool:
    """Check if Griptape framework is available."""
    try:
        import griptape
        return True
    except ImportError:
        return False

def _detect_griptape_version() -> Optional[str]:
    """Detect Griptape version for compatibility."""
    try:
        import griptape
        return getattr(griptape, '__version__', 'unknown')
    except ImportError:
        return None

def _wrap_structure_method(
    original_method: Callable,
    structure_type: str,
    method_name: str,
    adapter: 'GenOpsGriptapeAdapter'
) -> Callable:
    """Wrap a structure method with governance tracking."""
    
    @functools.wraps(original_method)
    def wrapped_method(self, *args, **kwargs):
        # Generate structure ID from object
        structure_id = getattr(self, 'id', None) or f"{structure_type}-{id(self)}"
        
        # Determine operation type
        operation_type = "run" if method_name in ["run", "execute"] else method_name
        
        # Use appropriate tracking context
        if structure_type == "agent":
            context_manager = adapter.track_agent(structure_id, operation_type=operation_type)
        elif structure_type == "pipeline":
            context_manager = adapter.track_pipeline(structure_id, operation_type=operation_type)
        elif structure_type == "workflow":
            context_manager = adapter.track_workflow(structure_id, operation_type=operation_type)
        else:
            # Generic structure tracking
            context_manager = adapter.track_agent(structure_id, operation_type=operation_type)
        
        with context_manager as request:
            try:
                # Execute original method
                result = original_method(self, *args, **kwargs)
                
                # Extract metrics from result if possible
                if hasattr(result, 'output') and hasattr(result.output, 'value'):
                    # Successful execution
                    request.add_task_completion(success=True)
                elif result:
                    request.add_task_completion(success=True)
                
                # Try to extract cost information from result
                if hasattr(result, 'usage'):
                    usage = result.usage
                    if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                        # OpenAI-style usage
                        provider = "openai"  # Default, could be detected from model
                        model = getattr(self, 'model', 'gpt-3.5-turbo')  # Default model
                        
                        request.add_provider_cost(
                            provider, model,
                            adapter.cost_aggregator.calculate_cost(
                                provider, model, 
                                usage.prompt_tokens, 
                                usage.completion_tokens
                            )["total_cost"]
                        )
                
                return result
                
            except Exception as e:
                # Record task failure
                request.add_task_completion(success=False)
                request.error_message = str(e)
                raise
    
    return wrapped_method

def _wrap_structure_class(
    structure_class: type,
    structure_type: str,
    adapter: 'GenOpsGriptapeAdapter'
) -> type:
    """Wrap a Griptape structure class with governance tracking."""
    
    # Create a new class inheriting from the original
    class WrappedStructure(structure_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._genops_adapter = adapter
            self._genops_structure_type = structure_type
        
        # Wrap key methods
        if hasattr(structure_class, 'run'):
            run = _wrap_structure_method(structure_class.run, structure_type, 'run', adapter)
        
        if hasattr(structure_class, 'execute'):
            execute = _wrap_structure_method(structure_class.execute, structure_type, 'execute', adapter)
        
        if hasattr(structure_class, 'kickoff'):
            kickoff = _wrap_structure_method(structure_class.kickoff, structure_type, 'kickoff', adapter)
    
    # Preserve original class metadata
    WrappedStructure.__name__ = structure_class.__name__
    WrappedStructure.__module__ = structure_class.__module__
    WrappedStructure.__doc__ = structure_class.__doc__
    
    return WrappedStructure

def _apply_instrumentation(adapter: 'GenOpsGriptapeAdapter') -> None:
    """Apply instrumentation to Griptape classes."""
    
    if not _is_griptape_available():
        logger.warning("Griptape not available, skipping instrumentation")
        return
    
    try:
        from griptape.structures import Agent, Pipeline, Workflow
        
        # Store original classes
        with _instrumentation_registry["lock"]:
            _instrumentation_registry["original_classes"] = {
                "Agent": Agent,
                "Pipeline": Pipeline, 
                "Workflow": Workflow
            }
            
            # Create wrapped classes
            wrapped_agent = _wrap_structure_class(Agent, "agent", adapter)
            wrapped_pipeline = _wrap_structure_class(Pipeline, "pipeline", adapter)
            wrapped_workflow = _wrap_structure_class(Workflow, "workflow", adapter)
            
            _instrumentation_registry["wrapped_classes"] = {
                "Agent": wrapped_agent,
                "Pipeline": wrapped_pipeline,
                "Workflow": wrapped_workflow
            }
            
            # Replace classes in griptape.structures module
            import griptape.structures
            griptape.structures.Agent = wrapped_agent
            griptape.structures.Pipeline = wrapped_pipeline
            griptape.structures.Workflow = wrapped_workflow
        
        logger.info("Griptape instrumentation applied successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import Griptape structures: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to apply Griptape instrumentation: {e}")
        raise

def _remove_instrumentation() -> None:
    """Remove instrumentation and restore original classes."""
    
    if not _is_griptape_available():
        return
    
    try:
        with _instrumentation_registry["lock"]:
            original_classes = _instrumentation_registry["original_classes"]
            
            if original_classes:
                # Restore original classes
                import griptape.structures
                griptape.structures.Agent = original_classes["Agent"]
                griptape.structures.Pipeline = original_classes["Pipeline"]
                griptape.structures.Workflow = original_classes["Workflow"]
                
                # Clear registry
                _instrumentation_registry["original_classes"] = {}
                _instrumentation_registry["wrapped_classes"] = {}
        
        logger.info("Griptape instrumentation removed")
        
    except Exception as e:
        logger.error(f"Failed to remove Griptape instrumentation: {e}")

def auto_instrument(
    team: Optional[str] = None,
    project: Optional[str] = None,
    environment: Optional[str] = None,
    cost_center: Optional[str] = None,
    customer_id: Optional[str] = None,
    feature: Optional[str] = None,
    daily_budget_limit: Optional[float] = None,
    enable_cost_tracking: bool = True,
    enable_performance_monitoring: bool = True,
    **kwargs
) -> 'GenOpsGriptapeAdapter':
    """
    Enable automatic instrumentation for Griptape framework.
    
    This function applies zero-code instrumentation to all Griptape structures,
    automatically adding GenOps governance tracking to existing code.
    
    Args:
        team: Team identifier for governance
        project: Project identifier for cost attribution
        environment: Environment (dev, staging, production)
        cost_center: Cost center for financial tracking
        customer_id: Customer ID for multi-tenant tracking
        feature: Feature identifier for A/B testing
        daily_budget_limit: Daily budget limit in USD
        enable_cost_tracking: Enable cost tracking
        enable_performance_monitoring: Enable performance monitoring
        **kwargs: Additional adapter configuration
    
    Returns:
        GenOpsGriptapeAdapter instance
    
    Example:
        auto_instrument(team="ai-team", project="agent-workflows")
        
        # Your existing code works unchanged
        from griptape.structures import Agent
        agent = Agent(tasks=[PromptTask("Hello")])
        result = agent.run("Test input")  # ✅ Now tracked
    """
    
    from .adapter import GenOpsGriptapeAdapter
    
    # Check if already instrumented
    with _instrumentation_registry["lock"]:
        if _instrumentation_registry["enabled"]:
            logger.warning("Griptape auto-instrumentation already enabled")
            return _instrumentation_registry["adapter"]
    
    # Validate Griptape availability
    if not _is_griptape_available():
        error_msg = (
            "Griptape framework not found. Please install it with: "
            "pip install griptape"
        )
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    # Create adapter
    adapter = GenOpsGriptapeAdapter(
        team=team,
        project=project,
        environment=environment,
        cost_center=cost_center,
        customer_id=customer_id,
        feature=feature,
        daily_budget_limit=daily_budget_limit,
        enable_cost_tracking=enable_cost_tracking,
        enable_performance_monitoring=enable_performance_monitoring,
        **kwargs
    )
    
    # Apply instrumentation
    try:
        _apply_instrumentation(adapter)
        
        with _instrumentation_registry["lock"]:
            _instrumentation_registry["enabled"] = True
            _instrumentation_registry["adapter"] = adapter
        
        logger.info(
            f"Griptape auto-instrumentation enabled: "
            f"team={team}, project={project}, "
            f"cost_tracking={enable_cost_tracking}"
        )
        
        return adapter
        
    except Exception as e:
        logger.error(f"Failed to enable auto-instrumentation: {e}")
        raise

def disable_auto_instrument() -> None:
    """Disable automatic instrumentation and restore original Griptape classes."""
    
    with _instrumentation_registry["lock"]:
        if not _instrumentation_registry["enabled"]:
            logger.warning("Griptape auto-instrumentation not enabled")
            return
        
        _remove_instrumentation()
        _instrumentation_registry["enabled"] = False
        _instrumentation_registry["adapter"] = None
    
    logger.info("Griptape auto-instrumentation disabled")

class InstrumentedGriptape:
    """
    Manual instrumentation wrapper for Griptape framework.
    
    Provides controlled access to instrumented Griptape structures
    without global auto-instrumentation.
    """
    
    def __init__(self, adapter: 'GenOpsGriptapeAdapter'):
        """Initialize with GenOps adapter."""
        self.adapter = adapter
        
        # Import and store original classes
        if _is_griptape_available():
            from griptape.structures import Agent, Pipeline, Workflow
            from griptape.tasks import PromptTask, TextSummaryTask
            from griptape.engines import RagEngine, ExtractionEngine, SummaryEngine
            
            self._original_agent = Agent
            self._original_pipeline = Pipeline
            self._original_workflow = Workflow
            self._original_prompt_task = PromptTask
            self._original_text_summary_task = TextSummaryTask
            
            # Create wrapped versions
            self.Agent = _wrap_structure_class(Agent, "agent", adapter)
            self.Pipeline = _wrap_structure_class(Pipeline, "pipeline", adapter)
            self.Workflow = _wrap_structure_class(Workflow, "workflow", adapter)
            
            # Store engines for manual tracking
            self.RagEngine = RagEngine
            self.ExtractionEngine = ExtractionEngine
            self.SummaryEngine = SummaryEngine
            
        else:
            raise ImportError("Griptape framework not available")
    
    def create_agent(self, tasks: List, **kwargs):
        """Create an instrumented Agent."""
        return self.Agent(tasks=tasks, **kwargs)
    
    def create_pipeline(self, tasks: List, **kwargs):
        """Create an instrumented Pipeline."""
        return self.Pipeline(tasks=tasks, **kwargs)
    
    def create_workflow(self, tasks: List[List], **kwargs):
        """Create an instrumented Workflow.""" 
        return self.Workflow(tasks=tasks, **kwargs)
    
    def track_engine_operation(self, engine_type: str, engine_id: str = None):
        """Context manager for tracking engine operations."""
        engine_id = engine_id or f"{engine_type}-{id(self)}"
        return self.adapter.track_engine(engine_id, engine_type)

def instrument_griptape(
    team: Optional[str] = None,
    project: Optional[str] = None,
    environment: Optional[str] = None,
    cost_center: Optional[str] = None,
    customer_id: Optional[str] = None,
    feature: Optional[str] = None,
    daily_budget_limit: Optional[float] = None,
    **kwargs
) -> InstrumentedGriptape:
    """
    Create manually instrumented Griptape wrapper.
    
    This provides controlled instrumentation without affecting global imports.
    
    Args:
        team: Team identifier for governance
        project: Project identifier for cost attribution
        environment: Environment (dev, staging, production)
        cost_center: Cost center for financial tracking
        customer_id: Customer ID for multi-tenant tracking
        feature: Feature identifier for A/B testing
        daily_budget_limit: Daily budget limit in USD
        **kwargs: Additional adapter configuration
    
    Returns:
        InstrumentedGriptape wrapper instance
    
    Example:
        griptape = instrument_griptape(team="research", project="analysis")
        
        agent = griptape.create_agent([PromptTask("Analyze data")])
        result = agent.run("Input data")  # ✅ Tracked
    """
    
    from .adapter import GenOpsGriptapeAdapter
    
    # Create adapter
    adapter = GenOpsGriptapeAdapter(
        team=team,
        project=project,
        environment=environment,
        cost_center=cost_center,
        customer_id=customer_id,
        feature=feature,
        daily_budget_limit=daily_budget_limit,
        **kwargs
    )
    
    # Return instrumented wrapper
    return InstrumentedGriptape(adapter)

def is_instrumented() -> bool:
    """Check if Griptape auto-instrumentation is currently enabled."""
    with _instrumentation_registry["lock"]:
        return _instrumentation_registry["enabled"]

def get_instrumentation_adapter() -> Optional['GenOpsGriptapeAdapter']:
    """Get the current auto-instrumentation adapter, if enabled."""
    with _instrumentation_registry["lock"]:
        return _instrumentation_registry["adapter"]

def validate_griptape_setup() -> Dict[str, Any]:
    """Validate Griptape setup and return diagnostic information."""
    
    validation_result = {
        "griptape_available": False,
        "griptape_version": None,
        "instrumentation_enabled": False,
        "supported_structures": [],
        "issues": [],
        "recommendations": []
    }
    
    # Check Griptape availability
    if _is_griptape_available():
        validation_result["griptape_available"] = True
        validation_result["griptape_version"] = _detect_griptape_version()
        
        try:
            from griptape.structures import Agent, Pipeline, Workflow
            validation_result["supported_structures"] = ["Agent", "Pipeline", "Workflow"]
            
            # Check for additional components
            try:
                from griptape.engines import RagEngine
                validation_result["supported_structures"].append("RagEngine")
            except ImportError:
                pass
            
            try:
                from griptape.tasks import PromptTask
                validation_result["supported_structures"].append("PromptTask") 
            except ImportError:
                validation_result["issues"].append("PromptTask not available")
                
        except ImportError as e:
            validation_result["issues"].append(f"Failed to import core structures: {e}")
            
    else:
        validation_result["issues"].append("Griptape framework not installed")
        validation_result["recommendations"].append("Install Griptape: pip install griptape")
    
    # Check instrumentation status
    validation_result["instrumentation_enabled"] = is_instrumented()
    
    # Version compatibility check
    if validation_result["griptape_version"]:
        version = validation_result["griptape_version"]
        if version == "unknown":
            validation_result["issues"].append("Cannot determine Griptape version")
        # Add specific version compatibility checks here if needed
    
    return validation_result