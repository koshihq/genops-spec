"""
GenOps Collibra Integration.

Bidirectional integration between GenOps AI and Collibra Data Governance Center:
- Export: GenOps governance telemetry → Collibra Assets
- Import: Collibra governance policies → GenOps PolicyEngine

Example usage:

    # Auto-instrumentation (zero-code)
    from genops.providers.collibra import auto_instrument
    adapter = auto_instrument()

    # Manual instrumentation
    from genops.providers.collibra import GenOpsCollibraAdapter
    adapter = GenOpsCollibraAdapter(
        collibra_url="https://company.collibra.com",
        team="ml-platform",
        project="ai-governance"
    )

    with adapter.track_ai_operation("model-inference") as span:
        # Your AI operations
        pass
"""

from genops.providers.collibra.adapter import GenOpsCollibraAdapter
from genops.providers.collibra.client import (
    CollibraAPIClient,
    CollibraAPIError,
    CollibraAsset,
    CollibraAuthenticationError,
    CollibraPolicy,
    CollibraRateLimitError,
)
from genops.providers.collibra.policy_importer import PolicyImporter, PolicySyncStats
from genops.providers.collibra.validation import (
    CollibraValidationResult,
    print_validation_result,
    validate_setup,
)

__version__ = "0.1.0"

__all__ = [
    # Main adapter
    "GenOpsCollibraAdapter",
    "auto_instrument",
    "instrument_collibra",
    # Client
    "CollibraAPIClient",
    "CollibraAsset",
    "CollibraPolicy",
    # Policy importer
    "PolicyImporter",
    "PolicySyncStats",
    # Errors
    "CollibraAPIError",
    "CollibraAuthenticationError",
    "CollibraRateLimitError",
    # Validation
    "validate_setup",
    "print_validation_result",
    "CollibraValidationResult",
]


def auto_instrument(
    collibra_url=None, team=None, project=None, environment="development", **kwargs
):
    """
    Auto-instrument GenOps with Collibra integration.

    Args:
        collibra_url: Collibra instance URL (or from COLLIBRA_URL env var)
        team: Team name for governance attribution
        project: Project name for governance attribution
        environment: Environment (development, staging, production)
        **kwargs: Additional configuration options (see GenOpsCollibraAdapter)

    Returns:
        GenOpsCollibraAdapter: Configured adapter

    Example:
        >>> from genops.providers.collibra import auto_instrument
        >>> adapter = auto_instrument(team="data-science", project="llm-experiment")
        >>> # Your AI code now automatically exports to Collibra
        >>> with adapter.track_ai_operation("completion") as span:
        ...     result = openai.chat.completions.create(...)
    """
    return GenOpsCollibraAdapter(
        collibra_url=collibra_url,
        team=team,
        project=project,
        environment=environment,
        **kwargs,
    )


def instrument_collibra(
    team: str = "default-team",
    project: str = "collibra-integration",
    environment: str = "development",
    export_mode: str = "batch",
    enable_policy_sync: bool = False,
    **kwargs,
) -> GenOpsCollibraAdapter:
    """
    Convenience function to instrument Collibra with common settings.

    This function provides a standardized way to create a Collibra adapter
    with sensible defaults for typical use cases.

    Args:
        team: Team name for cost attribution (default: "default-team")
        project: Project name (default: "collibra-integration")
        environment: Environment (development, staging, production)
        export_mode: Export mode - "batch", "realtime", or "hybrid" (default: "batch")
        enable_policy_sync: Enable policy import from Collibra (default: False)
        **kwargs: Additional configuration options passed to GenOpsCollibraAdapter

    Returns:
        GenOpsCollibraAdapter: Configured adapter instance

    Example:
        >>> from genops.providers.collibra import instrument_collibra
        >>> adapter = instrument_collibra(
        ...     team="ml-platform",
        ...     project="model-inference",
        ...     export_mode="realtime",
        ...     enable_policy_sync=True
        ... )
        >>> with adapter.track_ai_operation("inference") as span:
        ...     # Your AI operations
        ...     pass
    """
    return GenOpsCollibraAdapter(
        team=team,
        project=project,
        environment=environment,
        export_mode=export_mode,
        enable_policy_sync=enable_policy_sync,
        **kwargs,
    )


def get_version() -> str:
    """
    Get Collibra integration version.

    Returns:
        Version string
    """
    return __version__
