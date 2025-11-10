"""
Pytest configuration and fixtures for PromptLayer tests.

Provides common fixtures, markers, and test configuration
for the PromptLayer test suite.
"""

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_promptlayer():
    """Mock PromptLayer SDK for testing."""
    with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
        mock_client = Mock()
        mock_pl.return_value = mock_client
        mock_client.run.return_value = {
            'response': 'Mock response',
            'usage': {
                'input_tokens': 10,
                'output_tokens': 20,
                'total_tokens': 30
            }
        }
        yield mock_client


@pytest.fixture
def sample_governance_config():
    """Sample governance configuration for tests."""
    return {
        'team': 'test-team',
        'project': 'test-project',
        'environment': 'test',
        'customer_id': 'test-customer',
        'cost_center': 'test-cost-center',
        'daily_budget_limit': 10.0,
        'max_operation_cost': 1.0
    }


@pytest.fixture
def promptlayer_adapter(mock_promptlayer, sample_governance_config):
    """PromptLayer adapter with mocked client."""
    from genops.providers.promptlayer import GenOpsPromptLayerAdapter

    return GenOpsPromptLayerAdapter(
        promptlayer_api_key='pl-test-key',
        **sample_governance_config
    )


@pytest.fixture
def sample_prompt_operations():
    """Sample prompt operation data for tests."""
    return [
        {
            'prompt_name': 'test_prompt_1',
            'input_variables': {'query': 'Test query 1'},
            'expected_cost': 0.015
        },
        {
            'prompt_name': 'test_prompt_2',
            'input_variables': {'query': 'Test query 2'},
            'expected_cost': 0.025
        },
        {
            'prompt_name': 'test_prompt_3',
            'input_variables': {'query': 'Test query 3'},
            'expected_cost': 0.008
        }
    ]


# Test markers
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle slow tests."""
    if config.getoption("--runslow"):
        # Don't skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "performance" in item.keywords:
            item.add_marker(skip_slow)
