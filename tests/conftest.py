"""Pytest configuration and shared fixtures for GenOps AI tests."""

from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

try:
    from opentelemetry.test.spantestutil import SpanRecorder
except ImportError:
    # Fallback implementation for SpanRecorder
    from typing import List

    from opentelemetry.sdk.trace import Span

    class SpanRecorder:
        """Simple span recorder for testing."""

        def __init__(self):
            self._spans: List[Span] = []

        def export(self, spans):
            self._spans.extend(spans)
            return None

        def shutdown(self):
            pass

        def on_start(self, span, parent_context):
            pass

        def on_end(self, span):
            self._spans.append(span)

        def get_finished_spans(self):
            return list(self._spans)
        
        def get_spans(self):
            """Alias for get_finished_spans for compatibility."""
            return self.get_finished_spans()

        def clear(self):
            self._spans.clear()


from opentelemetry.sdk.resources import Resource

from genops.core.policy import PolicyConfig, PolicyResult
from genops.core.telemetry import GenOpsTelemetry


@pytest.fixture
def mock_otel_setup() -> Generator[SpanRecorder, None, None]:
    """Set up in-memory OpenTelemetry for isolated testing."""
    # Get existing tracer provider or create new one
    current_tracer_provider = trace.get_tracer_provider()
    
    if not hasattr(current_tracer_provider, "add_span_processor"):
        # Create a tracer provider with test resource only if none exists
        resource = Resource.create({"service.name": "genops-test"})
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
    else:
        tracer_provider = current_tracer_provider

    # Set up span recorder for verification
    span_recorder = SpanRecorder()
    span_processor = SimpleSpanProcessor(span_recorder)
    tracer_provider.add_span_processor(span_processor)

    yield span_recorder

    # Cleanup
    span_recorder.clear()


@pytest.fixture
def telemetry(mock_otel_setup) -> GenOpsTelemetry:
    """Provide a GenOpsTelemetry instance with mock OpenTelemetry."""
    return GenOpsTelemetry("genops-test")


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing without API calls."""
    mock_client = MagicMock()

    # Mock chat completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test AI response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = "gpt-3.5-turbo"

    mock_client.chat.completions.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing without API calls."""
    mock_client = MagicMock()

    # Mock message response
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Test Claude response"
    mock_response.usage.input_tokens = 12
    mock_response.usage.output_tokens = 8
    mock_response.model = "claude-3-sonnet-20240229"

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def sample_messages() -> List[Dict[str, str]]:
    """Provide sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ]


@pytest.fixture
def sample_policy_config() -> PolicyConfig:
    """Provide a sample policy configuration."""
    return PolicyConfig(
        name="test_cost_limit",
        description="Test cost limit policy",
        enforcement_level=PolicyResult.BLOCKED,
        conditions={"max_cost": 1.0},
    )


@pytest.fixture
def sample_policies() -> List[PolicyConfig]:
    """Provide sample policy configurations for testing."""
    return [
        PolicyConfig(
            name="cost_limit",
            description="Limit AI operation costs",
            enforcement_level=PolicyResult.BLOCKED,
            conditions={"max_cost": 5.0},
        ),
        PolicyConfig(
            name="rate_limit",
            description="Rate limit AI operations",
            enforcement_level=PolicyResult.RATE_LIMITED,
            conditions={"max_requests": 100, "time_window": 3600},
        ),
        PolicyConfig(
            name="content_filter",
            description="Filter inappropriate content",
            enforcement_level=PolicyResult.WARNING,
            conditions={"blocked_patterns": ["violence", "explicit"]},
        ),
    ]


@pytest.fixture
def governance_attributes() -> Dict[str, Any]:
    """Provide sample governance attributes."""
    return {
        "team": "ai-platform",
        "project": "chatbot-service",
        "environment": "testing",
        "feature": "conversation",
        "customer_id": "test-customer-123",
        "cost_center": "engineering",
        "model": "gpt-3.5-turbo",
        "provider": "openai",
    }


@pytest.fixture
def cost_data() -> Dict[str, Any]:
    """Provide sample cost calculation data."""
    return {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "model": "gpt-3.5-turbo",
        "cost_per_input_token": 0.0005,
        "cost_per_output_token": 0.0015,
        "total_cost": 0.125,
    }


@pytest.fixture
def mock_span_recorder(mock_otel_setup) -> SpanRecorder:
    """Provide direct access to span recorder for assertions."""
    return mock_otel_setup


class SpanAssertions:
    """Helper class for making assertions about OpenTelemetry spans."""

    @staticmethod
    def assert_span_exists(spans: List, name: str) -> Any:
        """Assert that a span with the given name exists."""
        matching_spans = [s for s in spans if s.name == name]
        assert len(matching_spans) > 0, f"No span found with name '{name}'"
        return matching_spans[0]

    @staticmethod
    def assert_span_attribute(span: Any, key: str, expected_value: Any = None):
        """Assert that a span has a specific attribute."""
        attributes = getattr(span, "attributes", {})
        assert key in attributes, f"Attribute '{key}' not found in span"

        if expected_value is not None:
            actual_value = attributes[key]
            assert actual_value == expected_value, (
                f"Attribute '{key}': expected '{expected_value}', got '{actual_value}'"
            )

    @staticmethod
    def assert_governance_attributes(span: Any, expected_attrs: Dict[str, Any]):
        """Assert that a span contains expected governance attributes."""
        for key, expected_value in expected_attrs.items():
            genops_key = f"genops.{key}" if not key.startswith("genops.") else key
            SpanAssertions.assert_span_attribute(span, genops_key, expected_value)


@pytest.fixture
def span_assertions() -> SpanAssertions:
    """Provide span assertion helper."""
    return SpanAssertions()


# Mock provider patches for isolated testing
@pytest.fixture
def mock_openai_import():
    """Mock OpenAI import for testing without dependency."""
    with patch("genops.providers.openai.HAS_OPENAI", True):
        with patch("genops.providers.openai.OpenAI") as mock_openai_class:
            yield mock_openai_class


@pytest.fixture
def mock_anthropic_import():
    """Mock Anthropic import for testing without dependency."""
    with patch("genops.providers.anthropic.HAS_ANTHROPIC", True):
        with patch("genops.providers.anthropic.Anthropic") as mock_anthropic_class:
            yield mock_anthropic_class


# Test data generators
class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def generate_chat_messages(count: int = 3) -> List[Dict[str, str]]:
        """Generate sample chat messages."""
        messages = []
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Test message {i + 1} from {role}"
            messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def generate_policy_violations() -> List[Dict[str, Any]]:
        """Generate sample policy violation scenarios."""
        return [
            {
                "policy": "cost_limit",
                "violation_type": "cost_exceeded",
                "cost": 10.0,
                "limit": 5.0,
                "metadata": {"model": "gpt-4", "tokens": 2000},
            },
            {
                "policy": "content_filter",
                "violation_type": "blocked_content",
                "content": "This contains violence",
                "patterns": ["violence"],
                "metadata": {"severity": "high"},
            },
            {
                "policy": "rate_limit",
                "violation_type": "rate_exceeded",
                "requests": 150,
                "limit": 100,
                "time_window": 3600,
                "metadata": {"user_id": "test-user"},
            },
        ]


@pytest.fixture
def test_data_generator() -> TestDataGenerator:
    """Provide test data generator."""
    return TestDataGenerator()


# Cleanup fixture to ensure test isolation
@pytest.fixture
def cleanup_test_state():
    """Ensure clean state between tests."""
    yield

    # Clean up instrumentation without breaking telemetry
    from genops.auto_instrumentation import GenOpsInstrumentor

    if hasattr(GenOpsInstrumentor, "_instance") and GenOpsInstrumentor._instance:
        instrumentor = GenOpsInstrumentor._instance
        if instrumentor and instrumentor._initialized:
            try:
                instrumentor.uninstrument()
            except Exception:
                pass  # Ignore cleanup errors
        # Only reset initialization flag, not the instance itself
        GenOpsInstrumentor._initialized = False
