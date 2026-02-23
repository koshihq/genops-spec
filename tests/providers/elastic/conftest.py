"""
Pytest fixtures and configuration for Elastic integration tests.
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def sample_elastic_config() -> dict[str, Any]:
    """Sample Elastic configuration for testing."""
    return {
        "url": "https://localhost:9200",
        "api_key": "test-api-key",
        "index_prefix": "genops-test",
        "team": "test-team",
        "project": "test-project",
        "environment": "test",
        "customer_id": "test-customer",
        "cost_center": "engineering",
        "export_mode": "batch",
        "batch_size": 100,
        "batch_interval_seconds": 60,
    }


@pytest.fixture
def minimal_elastic_config() -> dict[str, Any]:
    """Minimal Elastic configuration for testing."""
    return {"url": "https://localhost:9200", "api_key": "test-api-key"}


@pytest.fixture
def mock_elasticsearch_client():
    """Mock Elasticsearch client."""
    mock_client = MagicMock()

    # Mock successful responses
    mock_client.info.return_value = {
        "version": {"number": "8.12.0"},
        "cluster_name": "test-cluster",
    }

    mock_client.indices.exists.return_value = False
    mock_client.indices.create.return_value = {"acknowledged": True}

    mock_client.bulk.return_value = {"took": 10, "errors": False, "items": []}

    mock_client.index.return_value = {"result": "created", "_id": "test-doc-id"}

    return mock_client


@pytest.fixture
def mock_elastic_adapter(sample_elastic_config, mock_elasticsearch_client):
    """Mock GenOps Elastic adapter with mocked client."""
    with patch(
        "genops.providers.elastic.client.Elasticsearch",
        return_value=mock_elasticsearch_client,
    ):
        from genops.providers.elastic import instrument_elastic

        adapter = instrument_elastic(**sample_elastic_config)
        adapter._client = mock_elasticsearch_client
        return adapter


@pytest.fixture
def sample_telemetry_event() -> dict[str, Any]:
    """Sample telemetry event for testing."""
    return {
        "timestamp": "2024-01-18T12:00:00Z",
        "operation_id": "test-op-123",
        "operation_type": "llm.completion",
        "provider": "openai",
        "model": "gpt-4",
        "cost": 0.05,
        "tokens_in": 100,
        "tokens_out": 200,
        "latency_ms": 1500,
        "team": "test-team",
        "project": "test-project",
        "environment": "test",
        "customer_id": "test-customer",
        "cost_center": "engineering",
    }


@pytest.fixture
def sample_batch_events(sample_telemetry_event) -> list:
    """Sample batch of telemetry events for testing."""
    events = []
    for i in range(10):
        event = sample_telemetry_event.copy()
        event["operation_id"] = f"test-op-{i}"
        event["cost"] = 0.01 * (i + 1)
        events.append(event)
    return events


@pytest.fixture
def mock_env_vars(sample_elastic_config):
    """Mock environment variables for auto-instrumentation."""
    env_vars = {
        "GENOPS_ELASTIC_URL": sample_elastic_config["url"],
        "GENOPS_ELASTIC_API_KEY": sample_elastic_config["api_key"],
        "GENOPS_ELASTIC_INDEX_PREFIX": sample_elastic_config["index_prefix"],
        "GENOPS_TEAM": sample_elastic_config["team"],
        "GENOPS_PROJECT": sample_elastic_config["project"],
        "GENOPS_ENVIRONMENT": sample_elastic_config["environment"],
    }

    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def validation_result_success():
    """Sample successful validation result."""
    return {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "config": {
            "url": "https://localhost:9200",
            "index_prefix": "genops-test",
            "export_mode": "batch",
        },
    }


@pytest.fixture
def validation_result_with_errors():
    """Sample validation result with errors."""
    return {
        "is_valid": False,
        "errors": [
            "GENOPS_ELASTIC_URL environment variable is not set",
            "GENOPS_ELASTIC_API_KEY environment variable is not set",
        ],
        "warnings": ["GENOPS_TEAM is not set - cost attribution will be limited"],
        "config": {},
    }
