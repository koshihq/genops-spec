"""Ollama provider for GenOps AI governance."""

import logging

logger = logging.getLogger(__name__)

try:
    from .adapter import (
        GenOpsOllamaAdapter,
        LocalModelMetrics,
        OllamaOperation,
        instrument_ollama,
    )
    from .model_manager import (
        ModelComparison,
        ModelInfo,
        ModelOptimizer,
        OllamaModelManager,
        get_model_manager,
        set_model_manager,
    )
    from .registration import (
        auto_instrument,
    )
    from .resource_monitor import (
        HardwareMetrics,
        ModelPerformanceTracker,
        OllamaResourceMonitor,
        ResourceMetrics,
        create_resource_monitor,
        get_resource_monitor,
        set_resource_monitor,
    )
    from .validation import (
        OllamaValidator,
        ValidationIssue,
        ValidationResult,
        print_validation_result,
        quick_validate,
        validate_ollama_setup,
    )

    # Auto-register with instrumentation system if available
    try:
        from .registration import auto_register

        auto_register()
    except ImportError:
        pass

    OLLAMA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Ollama provider not fully available: {e}")
    OLLAMA_AVAILABLE = False

if OLLAMA_AVAILABLE:
    __all__ = [
        "OLLAMA_AVAILABLE",
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
else:
    __all__ = [
        "OLLAMA_AVAILABLE",
    ]
