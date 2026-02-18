"""
Shared configuration for Flowise examples.

This module provides common configuration and utilities used across
all Flowise integration examples.
"""

import os
from typing import Any

# Default Flowise configuration
FLOWISE_CONFIG = {
    "base_url": os.getenv("FLOWISE_BASE_URL", "http://localhost:3000"),
    "api_key": os.getenv("FLOWISE_API_KEY"),
    "team": os.getenv("GENOPS_TEAM", "flowise-examples"),
    "project": os.getenv("GENOPS_PROJECT", "examples"),
    "environment": os.getenv("GENOPS_ENVIRONMENT", "development"),
}

# OpenTelemetry configuration
OTEL_CONFIG = {
    "endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
    "headers": os.getenv("OTEL_EXPORTER_OTLP_HEADERS"),
}

# Example-specific settings
EXAMPLE_SETTINGS = {
    "enable_console_output": os.getenv("GENOPS_CONSOLE_OUTPUT", "true").lower()
    == "true",
    "log_level": os.getenv("GENOPS_LOG_LEVEL", "INFO"),
    "timeout_seconds": int(os.getenv("GENOPS_TIMEOUT_SECONDS", "30")),
    "max_retries": int(os.getenv("GENOPS_MAX_RETRIES", "3")),
}


def get_flowise_config(**overrides) -> dict[str, Any]:
    """Get Flowise configuration with optional overrides."""
    config = FLOWISE_CONFIG.copy()
    config.update(overrides)
    return config


def validate_required_config() -> tuple[bool, list[str]]:
    """Validate that required configuration is present."""
    errors = []

    if not FLOWISE_CONFIG["base_url"]:
        errors.append("FLOWISE_BASE_URL is required")

    # API key is optional for local development
    if (
        FLOWISE_CONFIG["base_url"] != "http://localhost:3000"
        and not FLOWISE_CONFIG["api_key"]
    ):
        errors.append("FLOWISE_API_KEY is required for non-local Flowise instances")

    return len(errors) == 0, errors


# Export for use in examples
__all__ = [
    "FLOWISE_CONFIG",
    "OTEL_CONFIG",
    "EXAMPLE_SETTINGS",
    "get_flowise_config",
    "validate_required_config",
]
