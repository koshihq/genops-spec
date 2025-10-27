"""GenOps AI - OpenTelemetry-native governance for AI."""

__version__ = "0.1.0"

# Core instrumentation functions
# Auto-instrumentation system
from genops.auto_instrumentation import (
    get_default_attributes,
    init,
    status,
    uninstrument,
)
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
from genops.core.context_manager import track, track_enhanced
from genops.core.policy import enforce_policy
from genops.core.telemetry import GenOpsTelemetry
from genops.core.tracker import track_usage

# Tag validation and enforcement
from genops.core.validation import (
    ValidationSeverity,
    ValidationRule,
    TagValidator,
    TagValidationError,
    validate_tags,
    enforce_tags,
    add_validation_rule,
    remove_validation_rule,
    get_validator,
    create_required_rule,
    create_enum_rule,
    create_pattern_rule,
)

__all__ = [
    # Core functions
    "track_usage",
    "track",
    "track_enhanced", 
    "enforce_policy",
    "GenOpsTelemetry",
    # Auto-instrumentation
    "init",
    "uninstrument",
    "status",
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
]
