"""
Pytest configuration and fixtures for SkyRouter tests.

Provides shared test fixtures, mock configurations, and test utilities
for the SkyRouter provider test suite.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime

# Import the modules under test (with graceful fallback)
try:
    from genops.providers.skyrouter import GenOpsSkyRouterAdapter
    from genops.providers.skyrouter_pricing import SkyRouterPricingConfig, SkyRouterPricingCalculator
    from genops.providers.skyrouter_validation import SkyRouterValidator
    from genops.providers.skyrouter_cost_aggregator import SkyRouterCostAggregator
    SKYROUTER_AVAILABLE = True
except ImportError:
    SKYROUTER_AVAILABLE = False


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "sk-test-api-key-12345678901234567890"


@pytest.fixture
def test_governance_attrs():
    """Provide test governance attributes."""
    return {
        'team': 'test-team',
        'project': 'test-project',
        'customer_id': 'test-customer-123',
        'environment': 'test',
        'cost_center': 'TEST-001'
    }


@pytest.fixture
def mock_skyrouter_response():
    """Provide a mock SkyRouter API response."""
    response = Mock()
    response.model = 'gpt-4'
    response.usage = {'total_tokens': 150, 'prompt_tokens': 100, 'completion_tokens': 50}
    response.choices = [Mock(message=Mock(content='Mock response content'))]
    response.route = 'balanced'
    response.route_efficiency_score = 0.85
    return response


@pytest.fixture
def mock_skyrouter_multi_model_response():
    """Provide a mock multi-model routing response."""
    response = Mock()
    response.model = 'claude-3-sonnet'
    response.usage = {'total_tokens': 200, 'prompt_tokens': 120, 'completion_tokens': 80}
    response.route = 'cost_optimized'
    response.route_efficiency_score = 0.92
    response.optimization_savings = 0.025
    response.routing_strategy = 'cost_optimized'
    return response


@pytest.fixture
def sample_adapter(mock_api_key, test_governance_attrs):
    """Provide a sample SkyRouter adapter for testing."""
    if not SKYROUTER_AVAILABLE:
        pytest.skip("SkyRouter provider not available")
    
    return GenOpsSkyRouterAdapter(
        skyrouter_api_key=mock_api_key,
        team=test_governance_attrs['team'],
        project=test_governance_attrs['project'],
        environment=test_governance_attrs['environment'],
        daily_budget_limit=50.0
    )


@pytest.fixture
def sample_pricing_calculator():
    """Provide a sample pricing calculator for testing."""
    if not SKYROUTER_AVAILABLE:
        pytest.skip("SkyRouter provider not available")
    
    config = SkyRouterPricingConfig()
    return SkyRouterPricingCalculator(config=config)


@pytest.fixture
def sample_cost_aggregator(test_governance_attrs):
    """Provide a sample cost aggregator for testing."""
    if not SKYROUTER_AVAILABLE:
        pytest.skip("SkyRouter provider not available")
    
    return SkyRouterCostAggregator(
        team=test_governance_attrs['team'],
        project=test_governance_attrs['project'],
        daily_budget_limit=100.0
    )


@pytest.fixture
def sample_validator():
    """Provide a sample validator for testing."""
    if not SKYROUTER_AVAILABLE:
        pytest.skip("SkyRouter provider not available")
    
    return SkyRouterValidator()


@pytest.fixture
def mock_network_success():
    """Mock successful network requests."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'active',
            'models': ['gpt-4', 'claude-3-sonnet', 'gpt-3.5-turbo'],
            'permissions': ['read', 'write', 'route']
        }
        mock_response.elapsed.total_seconds.return_value = 0.5
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_network_failure():
    """Mock network request failures."""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = ConnectionError("Network unreachable")
        yield mock_get


@pytest.fixture
def mock_api_error():
    """Mock API error responses."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {'error': 'Invalid API key'}
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def sample_operations_data():
    """Provide sample operations data for testing."""
    return [
        {
            'model': 'gpt-4',
            'cost': 0.06,
            'input_tokens': 1000,
            'output_tokens': 500,
            'routing_strategy': 'reliability_first',
            'complexity': 'enterprise',
            'timestamp': datetime.now()
        },
        {
            'model': 'claude-3-sonnet',
            'cost': 0.015,
            'input_tokens': 800,
            'output_tokens': 300,
            'routing_strategy': 'balanced',
            'complexity': 'moderate',
            'timestamp': datetime.now()
        },
        {
            'model': 'gpt-3.5-turbo',
            'cost': 0.002,
            'input_tokens': 500,
            'output_tokens': 200,
            'routing_strategy': 'cost_optimized',
            'complexity': 'simple',
            'timestamp': datetime.now()
        }
    ]


@pytest.fixture
def sample_workflow_steps():
    """Provide sample workflow steps for testing."""
    return [
        {
            'model': 'gpt-3.5-turbo',
            'input': {'task': 'intent_classification'},
            'complexity': 'simple',
            'optimization': 'cost_optimized'
        },
        {
            'model': 'claude-3-sonnet',
            'input': {'task': 'solution_generation'},
            'complexity': 'moderate',
            'optimization': 'balanced'
        },
        {
            'model': 'gpt-4',
            'input': {'task': 'quality_review'},
            'complexity': 'complex',
            'optimization': 'reliability_first'
        }
    ]


@pytest.fixture
def enterprise_configuration():
    """Provide enterprise configuration for testing."""
    return {
        'environments': [
            {
                'name': 'development',
                'budget': 10.0,
                'policy': 'advisory',
                'compliance': []
            },
            {
                'name': 'staging',
                'budget': 50.0,
                'policy': 'enforced',
                'compliance': ['soc2']
            },
            {
                'name': 'production',
                'budget': 500.0,
                'policy': 'strict',
                'compliance': ['soc2', 'gdpr', 'hipaa']
            }
        ],
        'departments': [
            {'name': 'engineering', 'budget': 500.0, 'cost_center': 'TECH-001'},
            {'name': 'product', 'budget': 200.0, 'cost_center': 'PROD-002'},
            {'name': 'customer_success', 'budget': 150.0, 'cost_center': 'CS-003'},
            {'name': 'sales', 'budget': 100.0, 'cost_center': 'SALES-004'}
        ]
    }


@pytest.fixture
def mock_skyrouter_module():
    """Mock the entire skyrouter module for testing."""
    with patch('genops.providers.skyrouter.skyrouter') as mock_module:
        # Configure the mock module
        mock_module.route.return_value = Mock(
            model='gpt-4',
            usage={'total_tokens': 150},
            choices=[Mock(message=Mock(content='Mock content'))]
        )
        
        mock_module.route_to_best_model.return_value = Mock(
            model='claude-3-sonnet',
            usage={'total_tokens': 200},
            route='balanced',
            route_efficiency_score=0.85
        )
        
        yield mock_module


@pytest.fixture
def environment_variables(mock_api_key):
    """Set up environment variables for testing."""
    env_vars = {
        'SKYROUTER_API_KEY': mock_api_key,
        'GENOPS_TEAM': 'test-team',
        'GENOPS_PROJECT': 'test-project',
        'GENOPS_ENVIRONMENT': 'test'
    }
    
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def performance_test_data():
    """Provide data for performance testing."""
    return {
        'high_volume_operations': 1000,
        'concurrent_sessions': 10,
        'batch_size': 100,
        'expected_max_latency': 5.0,  # seconds
        'expected_throughput': 50  # operations per second
    }


@pytest.fixture(autouse=True)
def cleanup_auto_instrumentation():
    """Automatically cleanup auto-instrumentation after each test."""
    yield
    
    # Clean up any auto-instrumentation that might have been set up
    try:
        from genops.providers.skyrouter import restore_skyrouter
        restore_skyrouter()
    except ImportError:
        pass  # Module not available, nothing to clean up


@pytest.fixture
def validation_test_cases():
    """Provide test cases for validation testing."""
    return [
        {
            'name': 'missing_api_key',
            'env_vars': {},
            'expected_issues': ['MISSING_API_KEY']
        },
        {
            'name': 'invalid_api_key',
            'env_vars': {'SKYROUTER_API_KEY': 'invalid-key'},
            'mock_response_status': 401,
            'expected_issues': ['INVALID_API_KEY']
        },
        {
            'name': 'valid_setup',
            'env_vars': {'SKYROUTER_API_KEY': 'sk-valid-key-123'},
            'mock_response_status': 200,
            'expected_issues': []
        }
    ]


@pytest.fixture
def cost_optimization_scenarios():
    """Provide cost optimization test scenarios."""
    return [
        {
            'name': 'high_cost_operations',
            'operations': [
                {'model': 'gpt-4', 'cost': 0.08, 'count': 50},
                {'model': 'claude-3-opus', 'cost': 0.075, 'count': 30}
            ],
            'expected_recommendations': ['model_optimization', 'route_optimization']
        },
        {
            'name': 'unoptimized_routing',
            'operations': [
                {'model': 'gpt-4', 'cost': 0.06, 'routing_strategy': 'reliability_first', 'count': 100}
            ],
            'expected_recommendations': ['route_optimization']
        },
        {
            'name': 'well_optimized',
            'operations': [
                {'model': 'gpt-3.5-turbo', 'cost': 0.002, 'routing_strategy': 'cost_optimized', 'count': 1000}
            ],
            'expected_recommendations': []
        }
    ]


def pytest_configure(config):
    """Configure pytest for SkyRouter tests."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "enterprise: mark test as enterprise feature test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.name or "test_integration.py" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in item.name or "high_volume" in item.name:
            item.add_marker(pytest.mark.performance)
        
        # Add enterprise marker to enterprise tests
        if "enterprise" in item.name or "compliance" in item.name or "governance" in item.name:
            item.add_marker(pytest.mark.enterprise)


@pytest.fixture
def skip_if_no_skyrouter():
    """Skip test if SkyRouter provider is not available."""
    if not SKYROUTER_AVAILABLE:
        pytest.skip("SkyRouter provider not available")


# Test utilities
class SkyRouterTestHelper:
    """Helper class for SkyRouter testing utilities."""
    
    @staticmethod
    def create_mock_operation_result(
        model: str = 'gpt-4',
        cost: float = 0.05,
        tokens: int = 150,
        routing_strategy: str = 'balanced'
    ):
        """Create a mock operation result for testing."""
        from genops.providers.skyrouter import SkyRouterOperationResult
        
        return SkyRouterOperationResult(
            model=model,
            total_cost=cost,
            input_tokens=tokens // 2,
            output_tokens=tokens // 2,
            routing_strategy=routing_strategy,
            session_id='test-session-123',
            governance_attrs=Mock(team='test-team', project='test-project')
        )
    
    @staticmethod
    def assert_cost_within_range(actual_cost: float, expected_min: float, expected_max: float):
        """Assert that cost is within expected range."""
        assert expected_min <= actual_cost <= expected_max, \
            f"Cost {actual_cost} not within range [{expected_min}, {expected_max}]"
    
    @staticmethod
    def assert_governance_attributes_present(result):
        """Assert that governance attributes are present and valid."""
        assert hasattr(result, 'governance_attrs')
        assert hasattr(result.governance_attrs, 'team')
        assert hasattr(result.governance_attrs, 'project')
        assert result.governance_attrs.team is not None
        assert result.governance_attrs.project is not None


@pytest.fixture
def test_helper():
    """Provide test helper utilities."""
    return SkyRouterTestHelper