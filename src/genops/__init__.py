"""GenOps AI - OpenTelemetry-native governance for AI."""

# Auto-instrumentation system
from genops.auto_instrumentation import (
    get_available_frameworks,
    get_framework_status,
    init,
    register_framework_provider,
    status,
    uninstrument,
)

# Context management (avoiding duplicate import of get_default_attributes)
from genops.core.context import (
    clear_context,
    clear_default_attributes,
    get_context,
    get_default_attributes,
    get_effective_attributes,
    set_context,
    set_customer_context,
    set_default_attributes,
    set_team_defaults,
    set_user_context,
    update_default_attributes,
)

# Provider integrations - Key functions for easy access
try:
    from genops.providers import instrument_helicone

    _helicone_available = True
except ImportError:

    def instrument_helicone(*args, **kwargs):
        raise ImportError(
            "Helicone provider not available. Install with: pip install 'genops[helicone]'"
        )

    _helicone_available = False

try:
    from genops.providers import instrument_langfuse

    _langfuse_available = True
except ImportError:

    def instrument_langfuse(*args, **kwargs):
        raise ImportError(
            "Langfuse provider not available. Install with: pip install 'genops[langfuse]'"
        )

    _langfuse_available = False

try:
    from genops.providers import instrument_dust

    _dust_available = True
except ImportError:

    def instrument_dust(*args, **kwargs):
        raise ImportError(
            "Dust provider not available. Install with: pip install requests"
        )

    _dust_available = False
from genops.core.context_manager import track, track_enhanced

# Multi-provider cost aggregation
from genops.core.multi_provider_costs import (
    MultiProviderCostAggregator,
    MultiProviderCostSummary,
    ProviderCostEntry,
    compare_provider_costs,
    estimate_migration_costs,
    multi_provider_cost_tracking,
)
from genops.core.policy import enforce_policy
from genops.core.telemetry import GenOpsTelemetry
from genops.core.tracker import track_usage

# Tag validation and enforcement
from genops.core.validation import (
    TagValidationError,
    TagValidator,
    ValidationRule,
    ValidationSeverity,
    add_validation_rule,
    create_enum_rule,
    create_pattern_rule,
    create_required_rule,
    enforce_tags,
    get_validator,
    remove_validation_rule,
    validate_tags,
)

# Export configuration and validation
from genops.exporters.otlp import configure_otlp_exporter
from genops.exporters.validation import (
    ValidationResult,
    print_validation_result,
    validate_export_setup,
)

__version__ = "0.1.0"


# Auto-instrumentation convenience function
def auto_instrument(**kwargs):
    """Convenience function for auto-instrumentation. Alias for init()."""
    return init(**kwargs)


__all__ = [
    # Core functions
    "track_usage",
    "track",
    "track_enhanced",
    "enforce_policy",
    "GenOpsTelemetry",
    # Auto-instrumentation
    "init",
    "auto_instrument",
    "uninstrument",
    "status",
    "register_framework_provider",
    "get_available_frameworks",
    "get_framework_status",
    # Provider integrations
    "instrument_helicone",
    "instrument_langfuse",
    "instrument_dust",
    # Attribution context management
    "set_default_attributes",
    "get_default_attributes",
    "clear_default_attributes",
    "update_default_attributes",
    "set_context",
    "get_context",
    "clear_context",
    "get_effective_attributes",
    "set_team_defaults",
    "set_customer_context",
    "set_user_context",
    # Multi-provider cost aggregation
    "MultiProviderCostAggregator",
    "MultiProviderCostSummary",
    "ProviderCostEntry",
    "multi_provider_cost_tracking",
    "compare_provider_costs",
    "estimate_migration_costs",
    # Tag validation and enforcement
    "ValidationSeverity",
    "ValidationRule",
    "TagValidator",
    "TagValidationError",
    "validate_tags",
    "enforce_tags",
    "add_validation_rule",
    "remove_validation_rule",
    "get_validator",
    "create_required_rule",
    "create_enum_rule",
    "create_pattern_rule",
    # Export configuration and validation
    "configure_otlp_exporter",
    "validate_export_setup",
    "print_validation_result",
    "ValidationResult",
]
