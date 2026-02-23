#!/usr/bin/env python3
"""
Pytest configuration and fixtures for CrewAI tests

Provides common test fixtures and configuration for CrewAI + GenOps testing.
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest

# Import CrewAI components for fixture creation
try:
    from genops.providers.crewai import (
        CrewAIAgentMonitor,
        CrewAICostAggregator,
        GenOpsCrewAIAdapter,
        ValidationIssue,
        ValidationResult,
        disable_auto_instrumentation,
    )

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False


@pytest.fixture(scope="function")
def clean_environment():
    """Fixture to ensure clean test environment."""
    if CREWAI_AVAILABLE:
        try:
            disable_auto_instrumentation()
        except Exception:
            pass

    yield

    if CREWAI_AVAILABLE:
        try:
            disable_auto_instrumentation()
        except Exception:
            pass


@pytest.fixture
def mock_crewai_adapter():
    """Fixture providing a mock CrewAI adapter."""
    if not CREWAI_AVAILABLE:
        pytest.skip("CrewAI not available")

    adapter = GenOpsCrewAIAdapter(
        team="test-team",
        project="test-project",
        environment="testing",
        daily_budget_limit=10.0,
        governance_policy="advisory",
    )
    return adapter


@pytest.fixture
def mock_agent_monitor():
    """Fixture providing a mock agent monitor."""
    if not CREWAI_AVAILABLE:
        pytest.skip("CrewAI not available")

    return CrewAIAgentMonitor()


@pytest.fixture
def mock_cost_aggregator():
    """Fixture providing a mock cost aggregator."""
    if not CREWAI_AVAILABLE:
        pytest.skip("CrewAI not available")

    return CrewAICostAggregator(daily_budget_limit=50.0)


@pytest.fixture
def sample_validation_result():
    """Fixture providing a sample validation result."""
    if not CREWAI_AVAILABLE:
        pytest.skip("CrewAI not available")

    issues = [
        ValidationIssue(
            category="api_key",
            severity="warning",
            message="OpenAI API key not set",
            fix_suggestion="Set OPENAI_API_KEY environment variable",
        )
    ]

    return ValidationResult(
        is_valid=False,
        issues=issues,
        summary="1 warning found",
        timestamp=datetime.now().isoformat(),
    )


@pytest.fixture
def mock_crewai_crew():
    """Fixture providing a mock CrewAI crew."""

    class MockAgent:
        def __init__(self, role, goal, backstory):
            self.role = role
            self.goal = goal
            self.backstory = backstory

    class MockTask:
        def __init__(self, description, agent):
            self.description = description
            self.agent = agent

    class MockCrew:
        def __init__(self, agents, tasks):
            self.agents = agents
            self.tasks = tasks

        def kickoff(self, inputs=None):
            return f"Mock crew executed with {len(self.agents)} agents and {len(self.tasks)} tasks"

    # Create mock agents
    agent1 = MockAgent(
        role="Researcher", goal="Conduct research", backstory="Expert researcher"
    )
    agent2 = MockAgent(role="Writer", goal="Write content", backstory="Skilled writer")

    # Create mock tasks
    task1 = MockTask("Research task", agent1)
    task2 = MockTask("Writing task", agent2)

    # Create mock crew
    crew = MockCrew(agents=[agent1, agent2], tasks=[task1, task2])

    return crew


@pytest.fixture
def temp_api_key():
    """Fixture providing temporary API key for testing."""
    test_key = "sk-test-key-for-testing-1234567890abcdef"
    original_key = os.environ.get("OPENAI_API_KEY")

    os.environ["OPENAI_API_KEY"] = test_key

    yield test_key

    if original_key is not None:
        os.environ["OPENAI_API_KEY"] = original_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture
def temp_config_dir():
    """Fixture providing temporary configuration directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="session")
def crewai_test_config():
    """Session-scoped fixture providing test configuration."""
    return {
        "team": "pytest-team",
        "project": "crewai-tests",
        "environment": "testing",
        "budget_limit": 5.0,
        "governance_policy": "advisory",
        "enable_cost_tracking": True,
        "enable_agent_tracking": True,
    }


@pytest.fixture
def mock_instrumentation_state():
    """Fixture to mock instrumentation state."""
    with patch(
        "genops.providers.crewai.registration._instrumentation_state"
    ) as mock_state:
        mock_state.return_value = {
            "instrumented": False,
            "adapter": None,
            "monitor": None,
            "config": {},
        }
        yield mock_state


@pytest.fixture
def performance_test_config():
    """Fixture providing configuration for performance tests."""
    return {
        "max_execution_time": 5.0,  # 5 seconds
        "max_memory_mb": 100,  # 100 MB
        "max_crews": 50,  # 50 concurrent crews
        "timeout_seconds": 30,  # 30 second timeout
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for CrewAI tests."""
    config.addinivalue_line(
        "markers", "crewai: mark test as requiring CrewAI integration"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark all CrewAI tests
        if "crewai" in str(item.fspath):
            item.add_marker(pytest.mark.crewai)

        # Mark integration tests
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if "performance" in item.name or "load" in item.name:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """Setup before running each test."""
    # Skip CrewAI tests if not available
    if item.get_closest_marker("crewai") and not CREWAI_AVAILABLE:
        pytest.skip("CrewAI integration not available")


@pytest.fixture(autouse=True)
def test_isolation():
    """Fixture to ensure test isolation."""
    # Clean up any global state before test
    if CREWAI_AVAILABLE:
        try:
            disable_auto_instrumentation()
        except Exception:
            pass

    yield

    # Clean up any global state after test
    if CREWAI_AVAILABLE:
        try:
            disable_auto_instrumentation()
        except Exception:
            pass


# Helper functions for tests
def create_mock_cost_entry(provider="openai", model="gpt-4", cost=0.045):
    """Helper function to create mock cost entries."""
    if not CREWAI_AVAILABLE:
        return None

    from genops.providers.crewai import AgentCostEntry

    return AgentCostEntry(
        provider=provider,
        model=model,
        agent_id="test_agent",
        tokens_in=100,
        tokens_out=50,
        cost=cost,
        timestamp=datetime.now(),
    )


def create_mock_agent_metrics(agent_id="test_agent", execution_time=0.5):
    """Helper function to create mock agent metrics."""
    if not CREWAI_AVAILABLE:
        return None

    from genops.providers.crewai import AgentExecutionMetrics

    return AgentExecutionMetrics(
        agent_id=agent_id,
        agent_role="Test Agent",
        execution_time=execution_time,
        start_time=datetime.now(),
        end_time=datetime.now(),
        success=True,
        custom_metrics={},
    )
