"""Ollama provider for GenOps AI governance."""

from .adapter import (
    GenOpsOllamaAdapter,
    OllamaOperation,
    LocalModelMetrics,
    instrument_ollama,
)
from .resource_monitor import (
    OllamaResourceMonitor,
    ResourceMetrics,
    ModelPerformanceTracker,
    HardwareMetrics,
    get_resource_monitor,
    set_resource_monitor,
    create_resource_monitor,
)
from .model_manager import (
    OllamaModelManager,
    ModelInfo,
    ModelOptimizer,
    ModelComparison,
    get_model_manager,
    set_model_manager,
)
from .validation import (
    validate_ollama_setup,
    print_validation_result,
    quick_validate,
    ValidationResult,
    ValidationIssue,
    OllamaValidator,
)
from .registration import (
    auto_instrument,
)

# Auto-register with instrumentation system if available
try:
    from .registration import auto_register
    auto_register()
except ImportError:
    # Registration system not available
    pass

__all__ = [
    # Main adapter classes
    "GenOpsOllamaAdapter",
    "OllamaOperation", 
    "LocalModelMetrics",
    
    # Resource monitoring
    "OllamaResourceMonitor",
    "ResourceMetrics",
    "ModelPerformanceTracker",
    "HardwareMetrics",
    "get_resource_monitor",
    "set_resource_monitor", 
    "create_resource_monitor",
    
    # Model management
    "OllamaModelManager",
    "ModelInfo",
    "ModelOptimizer",
    "ModelComparison",
    "get_model_manager",
    "set_model_manager",
    
    # Validation
    "validate_ollama_setup",
    "print_validation_result",
    "quick_validate",
    "ValidationResult",
    "ValidationIssue", 
    "OllamaValidator",
    
    # Main factory functions
    "instrument_ollama",
    "auto_instrument",
]