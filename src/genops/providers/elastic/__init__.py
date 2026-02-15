"""
GenOps Elasticsearch Integration - Export AI governance telemetry to Elasticsearch.

Provides zero-code auto-instrumentation and manual instrumentation patterns
for tracking AI operations, cost, policy, and budget telemetry in Elasticsearch.

Quick Start (Auto-Instrumentation):
    from genops.providers.elastic import auto_instrument

    adapter = auto_instrument(team="ml-platform", project="recommendations")

    # AI operations are now tracked automatically
    with adapter.track_ai_operation("gpt4-completion") as span:
        # Your AI code here
        adapter.record_cost(span, cost=0.05, provider="openai", model="gpt-4")

Quick Start (Manual Instrumentation):
    from genops.providers.elastic import instrument_elastic

    adapter = instrument_elastic(
        elastic_url="http://localhost:9200",
        team="ml-platform",
        project="recommendations"
    )

    with adapter.track_ai_operation("gpt4-completion") as span:
        adapter.record_cost(span, cost=0.05, provider="openai", model="gpt-4")

Validation:
    from genops.providers.elastic import validate_setup, print_validation_result

    result = validate_setup()
    print_validation_result(result)
"""

import logging
import os
from typing import Optional

from .adapter import GenOpsElasticAdapter
from .client import (
    ElasticAPIClient,
    ElasticAPIError,
    ElasticAuthenticationError,
    ElasticConnectionError,
    ElasticDocument,
    ElasticIndexError,
)
from .event_exporter import EventExporter, ExportMode, ExportStats
from .validation import (
    ElasticValidationResult,
    print_validation_result,
    validate_setup,
)

logger = logging.getLogger(__name__)

# Public API
__all__ = [
    # Main functions
    "auto_instrument",
    "instrument_elastic",
    "validate_setup",
    "print_validation_result",
    # Core classes
    "GenOpsElasticAdapter",
    "ElasticAPIClient",
    "EventExporter",
    # Data classes
    "ElasticDocument",
    "ElasticValidationResult",
    "ExportStats",
    # Enums
    "ExportMode",
    # Exceptions
    "ElasticAPIError",
    "ElasticAuthenticationError",
    "ElasticConnectionError",
    "ElasticIndexError",
]


def auto_instrument(
    team: Optional[str] = None,
    project: Optional[str] = None,
    environment: Optional[str] = None,
    customer_id: Optional[str] = None,
    cost_center: Optional[str] = None,
    export_mode: str = "batch",
    batch_size: int = 100,
    batch_interval_seconds: int = 60,
    ilm_enabled: bool = True,
    ilm_retention_days: int = 90,
    auto_validate: bool = True,
    **kwargs,
) -> GenOpsElasticAdapter:
    """
    Zero-code auto-instrumentation for Elasticsearch telemetry export.

    Automatically configures Elasticsearch connection using environment variables:
    - ELASTIC_URL or ELASTIC_CLOUD_ID (required)
    - ELASTIC_API_KEY (recommended) or ELASTIC_USERNAME + ELASTIC_PASSWORD

    Usage:
        # Set environment variables first
        export ELASTIC_URL=http://localhost:9200
        export ELASTIC_API_KEY=your-api-key

        # Auto-instrument with zero additional configuration
        adapter = auto_instrument(team="ml-platform", project="recommendations")

        # Track AI operations
        with adapter.track_ai_operation("gpt4-completion") as span:
            adapter.record_cost(span, cost=0.05, provider="openai", model="gpt-4")

    Args:
        team: Team for governance attribution (recommended)
        project: Project for cost tracking (recommended)
        environment: Environment (development/staging/production)
        customer_id: Customer ID for multi-tenant tracking
        cost_center: Cost center for financial reporting
        export_mode: Export mode - "batch", "realtime", or "hybrid"
        batch_size: Maximum batch size before flush (batch mode)
        batch_interval_seconds: Flush interval in seconds (batch mode)
        ilm_enabled: Enable Index Lifecycle Management
        ilm_retention_days: Data retention period in days
        auto_validate: Automatically validate setup on initialization
        **kwargs: Additional arguments passed to GenOpsElasticAdapter

    Returns:
        Configured GenOpsElasticAdapter instance

    Raises:
        ImportError: If elasticsearch package not installed
        ElasticConnectionError: If connection configuration is invalid
        ElasticAuthenticationError: If authentication fails

    Environment Variables:
        ELASTIC_URL: Elasticsearch cluster URL (http://localhost:9200)
        ELASTIC_CLOUD_ID: Elastic Cloud deployment ID (alternative to ELASTIC_URL)
        ELASTIC_USERNAME: Basic auth username
        ELASTIC_PASSWORD: Basic auth password
        ELASTIC_API_KEY: API key (recommended over basic auth)
        ELASTIC_API_ID: API key ID (optional)
    """
    # Get connection config from environment
    elastic_url = os.getenv("ELASTIC_URL")
    cloud_id = os.getenv("ELASTIC_CLOUD_ID")

    if not elastic_url and not cloud_id:
        raise ValueError(
            "No Elasticsearch connection configured. "
            "Set ELASTIC_URL or ELASTIC_CLOUD_ID environment variable.\n\n"
            "Examples:\n"
            "  export ELASTIC_URL=http://localhost:9200\n"
            "  export ELASTIC_CLOUD_ID=<your-cloud-id>\n\n"
            "For more help, run: python -m genops.providers.elastic.validation"
        )

    # Create adapter with auto-detected configuration
    adapter = GenOpsElasticAdapter(
        elastic_url=elastic_url,
        cloud_id=cloud_id,
        team=team,
        project=project,
        environment=environment or "development",
        customer_id=customer_id,
        cost_center=cost_center,
        export_mode=export_mode,
        batch_size=batch_size,
        batch_interval_seconds=batch_interval_seconds,
        ilm_enabled=ilm_enabled,
        ilm_retention_days=ilm_retention_days,
        auto_validate=auto_validate,
        **kwargs,
    )

    logger.info(
        f"Elasticsearch auto-instrumentation enabled "
        f"(team: {team}, project: {project}, mode: {export_mode})"
    )

    return adapter


def instrument_elastic(
    elastic_url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
    api_id: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    environment: str = "development",
    customer_id: Optional[str] = None,
    cost_center: Optional[str] = None,
    export_mode: str = "batch",
    batch_size: int = 100,
    batch_interval_seconds: int = 60,
    ilm_enabled: bool = True,
    ilm_retention_days: int = 90,
    verify_certs: bool = True,
    ca_certs: Optional[str] = None,
    auto_validate: bool = True,
    **kwargs,
) -> GenOpsElasticAdapter:
    """
    Manual instrumentation for Elasticsearch telemetry export.

    Provides full control over configuration. Falls back to environment variables
    for connection parameters if not explicitly provided.

    Usage:
        adapter = instrument_elastic(
            elastic_url="http://localhost:9200",
            api_key="your-api-key",
            team="ml-platform",
            project="recommendations",
            export_mode="batch"
        )

        with adapter.track_ai_operation("gpt4-completion") as span:
            adapter.record_cost(span, cost=0.05, provider="openai", model="gpt-4")

    Args:
        elastic_url: Elasticsearch cluster URL (env: ELASTIC_URL)
        cloud_id: Elastic Cloud deployment ID (env: ELASTIC_CLOUD_ID)
        username: Basic auth username (env: ELASTIC_USERNAME)
        password: Basic auth password (env: ELASTIC_PASSWORD)
        api_key: API key for authentication (env: ELASTIC_API_KEY)
        api_id: API key ID (env: ELASTIC_API_ID)
        team: Team for governance attribution
        project: Project for cost tracking
        environment: Environment (development/staging/production)
        customer_id: Customer ID for multi-tenant tracking
        cost_center: Cost center for financial reporting
        export_mode: Export mode - "batch", "realtime", or "hybrid"
        batch_size: Maximum batch size before flush (batch mode)
        batch_interval_seconds: Flush interval in seconds (batch mode)
        ilm_enabled: Enable Index Lifecycle Management
        ilm_retention_days: Data retention period in days
        verify_certs: Verify SSL certificates
        ca_certs: Path to CA certificate bundle
        auto_validate: Automatically validate setup on initialization
        **kwargs: Additional arguments passed to GenOpsElasticAdapter

    Returns:
        Configured GenOpsElasticAdapter instance

    Raises:
        ImportError: If elasticsearch package not installed
        ElasticConnectionError: If connection fails
        ElasticAuthenticationError: If authentication fails
    """
    adapter = GenOpsElasticAdapter(
        elastic_url=elastic_url,
        cloud_id=cloud_id,
        username=username,
        password=password,
        api_key=api_key,
        api_id=api_id,
        team=team,
        project=project,
        environment=environment,
        customer_id=customer_id,
        cost_center=cost_center,
        export_mode=export_mode,
        batch_size=batch_size,
        batch_interval_seconds=batch_interval_seconds,
        ilm_enabled=ilm_enabled,
        ilm_retention_days=ilm_retention_days,
        verify_certs=verify_certs,
        ca_certs=ca_certs,
        auto_validate=auto_validate,
        **kwargs,
    )

    logger.info(
        f"Elasticsearch instrumentation enabled "
        f"(team: {team}, project: {project}, mode: {export_mode})"
    )

    return adapter
