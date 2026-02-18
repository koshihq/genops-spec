"""
GenOps Anyscale Provider - Governance for Anyscale Endpoints

Provides comprehensive governance tracking for Anyscale managed LLM endpoints:
- Cost attribution and tracking across all models
- OpenTelemetry traces with governance semantics
- Zero-code auto-instrumentation
- Multi-model support with unified governance

Quick Start:
    from genops.providers.anyscale import instrument_anyscale

    adapter = instrument_anyscale(
        team="ml-research",
        project="chatbot"
    )

    response = adapter.completion_create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=[{"role": "user", "content": "Hello!"}],
        customer_id="acme-corp"
    )

For detailed documentation, see: docs/anyscale-quickstart.md
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Core adapter
from .adapter import (  # noqa: E402
    AnyscaleCostSummary,
    AnyscaleOperation,
    GenOpsAnyscaleAdapter,
    instrument_anyscale,
)

# Budget management
from .budget import (  # noqa: E402
    BudgetExceededError,
    BudgetManager,
    create_budget_manager,
)

# Pricing utilities
from .pricing import (  # noqa: E402
    ANYSCALE_PRICING,
    AnyscalePricing,
    ModelPricing,
    calculate_completion_cost,
    calculate_embedding_cost,
    get_model_pricing,
)

# Auto-instrumentation
from .registration import (  # noqa: E402
    auto_instrument,
    disable_auto_instrument,
)

# Validation utilities
from .validation import (  # noqa: E402
    ValidationIssue,
    ValidationResult,
    print_validation_result,
    validate_setup,
)

# Version info
__version__ = "0.1.0"

# Export public API
__all__ = [
    # Adapter
    "GenOpsAnyscaleAdapter",
    "AnyscaleOperation",
    "AnyscaleCostSummary",
    "instrument_anyscale",
    # Pricing
    "ModelPricing",
    "AnyscalePricing",
    "ANYSCALE_PRICING",
    "calculate_completion_cost",
    "calculate_embedding_cost",
    "get_model_pricing",
    # Auto-instrumentation
    "auto_instrument",
    "disable_auto_instrument",
    # Validation
    "validate_setup",
    "print_validation_result",
    "ValidationResult",
    "ValidationIssue",
]


# Auto-registration with GenOps instrumentation system
def auto_register():
    """Automatically register Anyscale provider with GenOps instrumentation."""
    try:
        from genops.auto_instrumentation import _instrumentor

        from .registration import register_anyscale_provider

        register_anyscale_provider(_instrumentor)
        logger.debug("Anyscale provider registered with auto-instrumentation system")
    except ImportError as e:
        logger.debug(f"Auto-instrumentation not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to auto-register Anyscale provider: {e}")


# Attempt auto-registration on import
try:
    auto_register()
except Exception as e:
    logger.debug(f"Auto-registration skipped: {e}")
