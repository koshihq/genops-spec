"""
GenOps Kubetorch Integration - Compute Governance for ML Training.

This package extends GenOps governance to the compute execution layer, providing:
- GPU resource allocation tracking
- Multi-resource cost aggregation (GPU, CPU, storage, network)
- Distributed training governance
- OpenTelemetry-based telemetry emission
- Auto-instrumentation for zero-code setup

Example (Zero-Code Auto-Instrumentation):
    >>> from genops.providers.kubetorch import auto_instrument_kubetorch
    >>> auto_instrument_kubetorch(team="ml-research", project="llm-training")
    >>> # All Kubetorch operations now automatically tracked!

Example (Manual Instrumentation):
    >>> from genops.providers.kubetorch import instrument_kubetorch
    >>> adapter = instrument_kubetorch(team="ml-engineering")
    >>> # Use adapter to track specific operations

Example (Cost Calculation):
    >>> from genops.providers.kubetorch import calculate_gpu_cost
    >>> cost = calculate_gpu_cost("a100", num_devices=8, duration_seconds=3600)
    >>> print(f"Cost: ${cost:.2f}")  # Cost: $262.16
"""

import logging

logger = logging.getLogger(__name__)

# Version
__version__ = "0.1.0"

# Pricing module is always available (no dependencies)
from .pricing import (
    GPUInstancePricing,
    KubetorchPricing,
    GPU_PRICING,
    calculate_gpu_cost,
    get_pricing_info,
    STORAGE_COST_PER_GB_MONTH,
    NETWORK_COST_PER_GB,
)

# Adapter module
try:
    from .adapter import (
        GenOpsKubetorchAdapter,
        instrument_kubetorch,
        create_compute_context,
        KubetorchOperation,
    )
    _ADAPTER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Kubetorch adapter import failed: {e}")
    _ADAPTER_AVAILABLE = False

    # Provide stub functions
    def instrument_kubetorch(**kwargs):
        """Stub: Adapter module not available."""
        raise ImportError(
            "Kubetorch adapter import failed. Check dependencies."
        )

    def create_compute_context(**kwargs):
        """Stub: Adapter module not available."""
        raise ImportError(
            "Kubetorch adapter import failed. Check dependencies."
        )

    GenOpsKubetorchAdapter = None  # type: ignore
    KubetorchOperation = None  # type: ignore


try:
    from .cost_aggregator import (
        ComputeResourceCost,
        ComputeCostSummary,
        KubetorchCostAggregator,
        create_compute_cost_context,
        get_cost_aggregator,
        reset_cost_aggregator,
    )
    _COST_AGGREGATOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Cost aggregator import failed: {e}")
    _COST_AGGREGATOR_AVAILABLE = False

    # Provide stubs
    ComputeResourceCost = None  # type: ignore
    ComputeCostSummary = None  # type: ignore
    KubetorchCostAggregator = None  # type: ignore

    def create_compute_cost_context(**kwargs):
        """Stub: Cost aggregator module not available."""
        raise ImportError("Cost aggregator import failed. Check dependencies.")

    def get_cost_aggregator():
        """Stub: Cost aggregator module not available."""
        raise ImportError("Cost aggregator import failed. Check dependencies.")

    def reset_cost_aggregator():
        """Stub: Cost aggregator module not available."""
        raise ImportError("Cost aggregator import failed. Check dependencies.")


# Compute Monitor module
try:
    from .compute_monitor import (
        KubetorchComputeMonitor,
        create_compute_monitor,
    )
    _COMPUTE_MONITOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Compute monitor import failed: {e}")
    _COMPUTE_MONITOR_AVAILABLE = False

    # Provide stubs
    KubetorchComputeMonitor = None  # type: ignore

    def create_compute_monitor(**kwargs):
        """Stub: Compute monitor module not available."""
        raise ImportError("Compute monitor import failed. Check dependencies.")


try:
    from .validation import (
        validate_kubetorch_setup,
        print_validation_result,
        ValidationResult,
        ValidationIssue,
    )
    _VALIDATION_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Validation module not yet available: {e}")
    _VALIDATION_AVAILABLE = False

    # Provide stubs
    ValidationResult = None  # type: ignore
    ValidationIssue = None  # type: ignore

    def validate_kubetorch_setup(**kwargs):
        """Stub: Validation module not yet implemented."""
        raise NotImplementedError("Validation module not yet implemented.")

    def print_validation_result(*args, **kwargs):
        """Stub: Validation module not yet implemented."""
        raise NotImplementedError("Validation module not yet implemented.")


try:
    from .registration import (
        auto_instrument_kubetorch,
        uninstrument_kubetorch,
        is_kubetorch_instrumented,
    )
    _REGISTRATION_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Registration module not yet available: {e}")
    _REGISTRATION_AVAILABLE = False

    # Provide stubs
    def auto_instrument_kubetorch(**kwargs):
        """Stub: Registration module not yet implemented."""
        raise NotImplementedError("Auto-instrumentation not yet implemented.")

    def uninstrument_kubetorch():
        """Stub: Registration module not yet implemented."""
        raise NotImplementedError("Uninstrumentation not yet implemented.")

    def is_kubetorch_instrumented():
        """Stub: Registration module not yet implemented."""
        return False


# Public API exports
__all__ = [
    # Version
    "__version__",

    # Pricing (always available)
    "GPUInstancePricing",
    "KubetorchPricing",
    "GPU_PRICING",
    "calculate_gpu_cost",
    "get_pricing_info",
    "STORAGE_COST_PER_GB_MONTH",
    "NETWORK_COST_PER_GB",

    # Adapter (available when implemented)
    "GenOpsKubetorchAdapter",
    "KubetorchOperation",
    "instrument_kubetorch",
    "create_compute_context",

    # Cost Aggregator (available when implemented)
    "ComputeResourceCost",
    "ComputeCostSummary",
    "KubetorchCostAggregator",
    "create_compute_cost_context",
    "get_cost_aggregator",
    "reset_cost_aggregator",

    # Compute Monitor (available when implemented)
    "KubetorchComputeMonitor",
    "create_compute_monitor",

    # Validation (available when implemented)
    "validate_kubetorch_setup",
    "print_validation_result",
    "ValidationResult",
    "ValidationIssue",

    # Registration (available when implemented)
    "auto_instrument_kubetorch",
    "uninstrument_kubetorch",
    "is_kubetorch_instrumented",
]


# Module availability status (for debugging)
def get_module_status() -> dict:
    """
    Get status of all Kubetorch modules.

    Returns:
        Dict with module availability status
    """
    return {
        "pricing": True,  # Always available
        "adapter": _ADAPTER_AVAILABLE,
        "cost_aggregator": _COST_AGGREGATOR_AVAILABLE,
        "compute_monitor": _COMPUTE_MONITOR_AVAILABLE,
        "validation": _VALIDATION_AVAILABLE,
        "registration": _REGISTRATION_AVAILABLE,
    }
