"""
Pytest configuration and fixtures for Fireworks AI provider tests.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest


# Mock Fireworks AI client
@pytest.fixture
def mock_fireworks_client():
    """Mock Fireworks AI client for testing."""
    mock_client = Mock()

    # Mock chat completions
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[
        0
    ].message.content = "Test response from Fireworks AI with 4x speed optimization"
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 25
    mock_response.usage.total_tokens = 75
    mock_response.model = "accounts/fireworks/models/llama-v3p1-8b-instruct"

    mock_client.chat.completions.create.return_value = mock_response

    # Mock embeddings
    mock_embedding_response = Mock()
    mock_embedding_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3] * 256),  # 768-dim embedding
        Mock(embedding=[0.4, 0.5, 0.6] * 256),
    ]
    mock_embedding_response.usage.total_tokens = 100
    mock_embedding_response.model = "accounts/fireworks/models/nomic-embed-text-v1p5"

    mock_client.embeddings.create.return_value = mock_embedding_response

    return mock_client


@pytest.fixture
def mock_fireworks_models():
    """Mock Fireworks model enums and pricing data."""
    from genops.providers.fireworks import FireworksModel

    # Ensure all test models are available
    test_models = {
        FireworksModel.LLAMA_3_2_1B_INSTRUCT: {
            "input_price": Decimal("0.0001"),
            "output_price": Decimal("0.0001"),
            "context_length": 131072,
            "tier": "tiny",
        },
        FireworksModel.LLAMA_3_1_8B_INSTRUCT: {
            "input_price": Decimal("0.0002"),
            "output_price": Decimal("0.0002"),
            "context_length": 131072,
            "tier": "small",
        },
        FireworksModel.LLAMA_3_1_70B_INSTRUCT: {
            "input_price": Decimal("0.0009"),
            "output_price": Decimal("0.0009"),
            "context_length": 131072,
            "tier": "large",
        },
        FireworksModel.MIXTRAL_8X7B: {
            "input_price": Decimal("0.0005"),
            "output_price": Decimal("0.0005"),
            "context_length": 32768,
            "tier": "medium",
        },
        FireworksModel.NOMIC_EMBED_TEXT: {
            "input_price": Decimal("0.00008"),
            "output_price": Decimal("0.0"),
            "context_length": 8192,
            "tier": "embedding",
        },
    }

    return test_models


@pytest.fixture
def sample_fireworks_config():
    """Sample configuration for Fireworks AI adapter testing."""
    return {
        "team": "test-team",
        "project": "test-project",
        "environment": "test",
        "daily_budget_limit": 100.0,
        "monthly_budget_limit": 2000.0,
        "governance_policy": "advisory",
        "enable_cost_alerts": True,
        "enable_governance": True,
        "api_key": "fw-test-key-12345",
    }


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant optimized for 4x faster inference.",
        },
        {
            "role": "user",
            "content": "Explain the benefits of Fireworks AI's speed optimization.",
        },
    ]


@pytest.fixture
def sample_embedding_texts():
    """Sample texts for embedding testing."""
    return [
        "Fireworks AI provides 4x faster inference with Fireattention optimization",
        "Cost optimization is crucial for production AI deployments",
        "Multimodal AI enables vision and language understanding together",
    ]


@pytest.fixture
def mock_validation_result():
    """Mock validation result for testing."""
    from genops.providers.fireworks_validation import ValidationResult

    return ValidationResult(
        is_valid=True,
        api_key_valid=True,
        connectivity_ok=True,
        model_access=["accounts/fireworks/models/llama-v3p1-8b-instruct"],
        performance_metrics={
            "avg_response_time": 0.85,  # 4x faster than baseline
            "tokens_per_second": 120,
        },
        diagnostics={
            "fireattention_enabled": True,
            "batch_processing_available": True,
            "supported_modalities": ["text", "vision", "audio", "embeddings"],
        },
    )


@pytest.fixture
def mock_cost_summary():
    """Mock cost summary for testing."""
    return {
        "daily_costs": Decimal("5.25"),
        "monthly_costs": Decimal("147.50"),
        "daily_budget_utilization": 5.25,
        "monthly_budget_utilization": 7.375,
        "operations_count": 150,
        "avg_cost_per_operation": Decimal("0.035"),
        "cost_by_model": {
            "llama-v3p1-8b-instruct": Decimal("3.20"),
            "llama-v3p1-70b-instruct": Decimal("2.05"),
        },
        "fireattention_savings": Decimal("1.75"),  # Speed-based efficiency savings
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key-12345")
    monkeypatch.setenv("GENOPS_TEAM", "test-team")
    monkeypatch.setenv("GENOPS_PROJECT", "fireworks-testing")
    monkeypatch.setenv("GENOPS_ENVIRONMENT", "test")


@pytest.fixture
def mock_session_context():
    """Mock session context for testing."""
    mock_session = Mock()
    mock_session.session_id = "test-session-123"
    mock_session.session_name = "test-session"
    mock_session.total_operations = 5
    mock_session.total_cost = Decimal("0.25")
    mock_session.start_time = 1234567890
    mock_session.governance_attrs = {
        "customer_id": "test-customer",
        "use_case": "testing",
    }
    return mock_session


@pytest.fixture
def mock_batch_operation():
    """Mock batch operation context for testing."""
    return {
        "batch_id": "test-batch-123",
        "operation_index": 1,
        "is_batch": True,
        "batch_discount": 0.5,  # 50% batch savings
        "estimated_batch_size": 100,
    }


# Test utilities
def create_mock_fireworks_result(
    response: str = "Test response",
    cost: float = 0.001,
    tokens: int = 75,
    model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
    execution_time: float = 0.85,
):
    """Create a mock Fireworks result object."""
    mock_result = Mock()
    mock_result.response = response
    mock_result.cost = Decimal(str(cost))
    mock_result.tokens_used = tokens
    mock_result.model_used = model
    mock_result.execution_time_seconds = execution_time
    mock_result.governance_attrs = {
        "team": "test-team",
        "project": "test-project",
        "feature": "testing",
    }
    mock_result.fireattention_optimized = True
    return mock_result


def create_mock_pricing_recommendation(
    model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
    cost: float = 0.001,
    reasoning: str = "Optimal balance of speed and cost",
):
    """Create a mock pricing recommendation."""
    mock_rec = Mock()
    mock_rec.recommended_model = model
    mock_rec.estimated_cost = Decimal(str(cost))
    mock_rec.reasoning = reasoning
    mock_rec.alternatives = [
        {
            "model": "accounts/fireworks/models/llama-v3p2-1b-instruct",
            "cost": Decimal("0.0005"),
            "tier": "tiny",
        },
        {
            "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "cost": Decimal("0.002"),
            "tier": "large",
        },
    ]
    return mock_rec
