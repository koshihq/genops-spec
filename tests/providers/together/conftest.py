#!/usr/bin/env python3
"""
pytest configuration and fixtures for Together AI tests.

Provides shared fixtures, test configuration, and utilities
for comprehensive Together AI provider testing.
"""

import os
import sys
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from src.genops.providers.together import GenOpsTogetherAdapter, TogetherModel
    from src.genops.providers.together_pricing import TogetherPricingCalculator
    from src.genops.providers.together_validation import (
        ValidationError,
        ValidationResult,
    )
except ImportError:
    # Skip all tests if Together AI provider is not available
    pytest.skip("Together AI provider not available", allow_module_level=True)


@pytest.fixture(scope="session")
def test_config():
    """Session-wide test configuration."""
    return {
        "test_team": "together-test-suite",
        "test_project": "comprehensive-testing",
        "test_environment": "test",
        "default_budget": 5.0,
        "default_governance": "advisory"
    }


@pytest.fixture
def mock_together_response():
    """Fixture providing standard mock Together API response."""
    return MagicMock(
        choices=[{"message": {"content": "Test response from Together AI"}}],
        usage={"prompt_tokens": 15, "completion_tokens": 25},
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        id="test-response-id",
        created=1234567890,
        object="chat.completion"
    )


@pytest.fixture
def mock_together_client(mock_together_response):
    """Fixture providing fully mocked Together client."""
    with patch('src.genops.providers.together.Together') as mock_together:
        client = MagicMock()

        # Mock chat completions
        client.chat.completions.create.return_value = mock_together_response

        # Mock models list
        client.models.list.return_value = MagicMock(data=[
            {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "object": "model"},
            {"id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "object": "model"},
            {"id": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "object": "model"},
            {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "object": "model"},
            {"id": "deepseek-ai/DeepSeek-Coder-V2-Instruct", "object": "model"},
            {"id": "Qwen/Qwen2.5-VL-72B-Instruct", "object": "model"}
        ])

        mock_together.return_value = client
        yield client


@pytest.fixture
def standard_test_adapter(test_config):
    """Fixture providing standard test adapter."""
    return GenOpsTogetherAdapter(
        team=test_config["test_team"],
        project=test_config["test_project"],
        environment=test_config["test_environment"],
        daily_budget_limit=test_config["default_budget"],
        governance_policy=test_config["default_governance"]
    )


@pytest.fixture
def enterprise_test_adapter(test_config):
    """Fixture providing enterprise-configured test adapter."""
    return GenOpsTogetherAdapter(
        team=test_config["test_team"],
        project="enterprise-testing",
        environment="production",
        customer_id="enterprise-customer-123",
        cost_center="ai-research",
        daily_budget_limit=25.0,
        monthly_budget_limit=500.0,
        governance_policy="strict",
        enable_cost_alerts=True,
        tags={
            "tier": "enterprise",
            "department": "engineering",
            "priority": "high"
        }
    )


@pytest.fixture
def pricing_calculator():
    """Fixture providing pricing calculator instance."""
    return TogetherPricingCalculator()


@pytest.fixture
def sample_messages():
    """Fixture providing sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful AI assistant specialized in testing."},
        {"role": "user", "content": "This is a test message for the Together AI integration."}
    ]


@pytest.fixture
def validation_success_result():
    """Fixture providing successful validation result."""
    return ValidationResult(
        is_valid=True,
        errors=[],
        model_access=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        ],
        api_key_valid=True,
        dependencies_available=True,
        connectivity_working=True
    )


@pytest.fixture
def validation_failure_result():
    """Fixture providing failed validation result."""
    return ValidationResult(
        is_valid=False,
        errors=[
            ValidationError(
                code="API_KEY_MISSING",
                message="Together AI API key not found",
                remediation="Set TOGETHER_API_KEY environment variable with your API key"
            ),
            ValidationError(
                code="DEPENDENCY_MISSING",
                message="Together AI client library not installed",
                remediation="Install with: pip install together"
            )
        ],
        api_key_valid=False,
        dependencies_available=False,
        connectivity_working=False
    )


@pytest.fixture(autouse=True)
def clean_environment():
    """Auto-use fixture to ensure clean test environment."""
    # Store original environment
    original_env = os.environ.copy()

    # Set up test environment variables if not present
    test_env_vars = {
        "GENOPS_TEAM": "test-team",
        "GENOPS_PROJECT": "test-project",
        "GENOPS_ENVIRONMENT": "test"
    }

    for key, value in test_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_budget_exceeded_adapter():
    """Fixture providing adapter that will exceed budget for testing."""
    return GenOpsTogetherAdapter(
        team="budget-test",
        project="budget-exceeded",
        daily_budget_limit=0.001,  # Very low budget
        governance_policy="strict",  # Strict enforcement
        enable_cost_alerts=True
    )


@pytest.fixture
def performance_test_data():
    """Fixture providing data for performance testing."""
    return {
        "small_message": [{"role": "user", "content": "Hi"}],
        "medium_message": [{"role": "user", "content": "Please explain machine learning in simple terms for a beginner audience."}],
        "large_message": [{"role": "user", "content": "Write a comprehensive analysis of artificial intelligence trends, including deep learning, natural language processing, computer vision, and their applications across various industries like healthcare, finance, automotive, and entertainment. Include both current developments and future predictions."}],
        "batch_messages": [
            [{"role": "user", "content": f"Batch message {i}"}]
            for i in range(50)
        ]
    }


@pytest.fixture
def models_for_testing():
    """Fixture providing list of models for testing."""
    return [
        TogetherModel.LLAMA_3_1_8B_INSTRUCT,
        TogetherModel.LLAMA_3_1_70B_INSTRUCT,
        TogetherModel.DEEPSEEK_R1,
        TogetherModel.DEEPSEEK_CODER_V2,
        TogetherModel.QWEN_VL_72B
    ]


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for end-to-end workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load testing"
    )
    config.addinivalue_line(
        "markers", "cross_provider: Cross-provider compatibility tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: Tests that need real API key"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file names."""
    for item in items:
        # Add markers based on test file names
        if "test_adapter.py" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_pricing.py" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_validation.py" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "test_integration.py" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_cross_provider.py" in str(item.fspath):
            item.add_marker(pytest.mark.cross_provider)
        elif "test_performance.py" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)


# Helper functions for tests
def assert_valid_governance_result(result):
    """Helper function to assert result has valid governance attributes."""
    assert hasattr(result, 'response')
    assert hasattr(result, 'tokens_used')
    assert hasattr(result, 'cost')
    assert hasattr(result, 'model_used')
    assert hasattr(result, 'governance_attributes')

    assert result.response is not None
    assert result.tokens_used > 0
    assert isinstance(result.cost, Decimal)
    assert result.cost > 0
    assert result.model_used is not None
    assert isinstance(result.governance_attributes, dict)

    # Check essential governance attributes
    required_attrs = ["team", "project", "environment"]
    for attr in required_attrs:
        assert attr in result.governance_attributes
        assert result.governance_attributes[attr] is not None


def assert_valid_cost_summary(summary):
    """Helper function to assert cost summary is valid."""
    required_keys = [
        "daily_costs", "daily_budget_limit", "daily_budget_utilization",
        "governance_policy", "operations_count"
    ]

    for key in required_keys:
        assert key in summary

    assert isinstance(summary["daily_costs"], (int, float, Decimal))
    assert summary["daily_costs"] >= 0
    assert isinstance(summary["daily_budget_limit"], (int, float))
    assert summary["daily_budget_limit"] > 0
    assert isinstance(summary["daily_budget_utilization"], (int, float))
    assert 0 <= summary["daily_budget_utilization"] <= 100
    assert summary["governance_policy"] in ["advisory", "enforced", "strict"]
    assert isinstance(summary["operations_count"], int)
    assert summary["operations_count"] >= 0


def assert_valid_pricing_calculation(cost, expected_min=0, expected_max=float('inf')):
    """Helper function to assert pricing calculation is valid."""
    assert isinstance(cost, Decimal)
    assert cost > 0
    assert expected_min <= float(cost) <= expected_max

    # Cost should have reasonable precision (at least 6 decimal places)
    cost_str = str(cost)
    if '.' in cost_str:
        decimal_places = len(cost_str.split('.')[1])
        assert decimal_places >= 4  # At least 4 decimal places for precision
