"""Tests for GenOps AI telemetry engine."""

import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.trace import Status, StatusCode

from genops.core.telemetry import GenOpsTelemetry


class TestGenOpsTelemetry:
    """Test the GenOpsTelemetry class."""
    
    def test_initialization(self, telemetry):
        """Test GenOpsTelemetry initialization."""
        assert telemetry is not None
        assert hasattr(telemetry, 'tracer')
    
    def test_create_span_basic(self, telemetry, mock_span_recorder):
        """Test basic span creation."""
        span_name = "test.operation"
        attributes = {"genops.test": "value"}
        
        span = telemetry.create_span(span_name, attributes)
        
        assert span is not None
        spans = mock_span_recorder.get_finished_spans()
        # Span won't be finished until we end it
        assert len(spans) == 0
        
        # End the span to verify attributes
        span.end()
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == span_name
        assert spans[0].attributes["genops.test"] == "value"
    
    def test_create_span_with_none_values(self, telemetry, mock_span_recorder):
        """Test span creation filters out None values."""
        attributes = {
            "genops.valid": "value",
            "genops.none": None,
            "genops.empty": "",
            "genops.zero": 0
        }
        
        span = telemetry.create_span("test.span", attributes)
        span.end()
        
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        span_attrs = spans[0].attributes
        assert "genops.valid" in span_attrs
        assert "genops.none" not in span_attrs  # None values filtered out
        assert "genops.empty" in span_attrs  # Empty string is kept
        assert "genops.zero" in span_attrs  # Zero is kept
    
    def test_trace_operation_context_manager(self, telemetry, mock_span_recorder):
        """Test the trace_operation context manager."""
        operation_name = "ai.inference"
        
        with telemetry.trace_operation(
            operation_name=operation_name,
            operation_type="ai.inference",
            model="gpt-3.5-turbo",
            team="test-team"
        ) as span:
            # Verify span is active during context
            assert span is not None
            span.set_attribute("genops.custom", "test-value")
        
        # Verify span was recorded
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        finished_span = spans[0]
        assert finished_span.name == operation_name
        
        # Check core attributes are set
        attrs = finished_span.attributes
        assert attrs["genops.operation.type"] == "ai.inference"
        assert attrs["genops.operation.name"] == operation_name
        assert "genops.timestamp" in attrs
        assert attrs["genops.model"] == "gpt-3.5-turbo"
        assert attrs["genops.team"] == "test-team"
        assert attrs["genops.custom"] == "test-value"
    
    def test_trace_operation_success(self, telemetry, mock_span_recorder):
        """Test trace_operation with successful execution."""
        with telemetry.trace_operation("test.success") as span:
            result = "operation completed"
        
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.UNSET  # Default for success
    
    def test_trace_operation_with_exception(self, telemetry, mock_span_recorder):
        """Test trace_operation handles exceptions properly."""
        test_error = ValueError("Test error")
        
        with pytest.raises(ValueError):
            with telemetry.trace_operation("test.error") as span:
                raise test_error
        
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        finished_span = spans[0]
        assert finished_span.status.status_code == StatusCode.ERROR
        
        # Check that exception details are recorded
        events = getattr(finished_span, 'events', [])
        exception_event = next((e for e in events if e.name == "exception"), None)
        assert exception_event is not None
    
    def test_record_cost(self, telemetry, mock_span_recorder):
        """Test cost recording functionality."""
        with telemetry.trace_operation("test.cost") as span:
            telemetry.record_cost(
                span=span,
                cost=1.50,
                currency="USD",
                cost_type="inference",
                input_tokens=100,
                output_tokens=50
            )
        
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        attrs = spans[0].attributes
        assert attrs["genops.cost.total"] == 1.50
        assert attrs["genops.cost.currency"] == "USD"
        assert attrs["genops.cost.type"] == "inference"
        assert attrs["genops.tokens.input"] == 100
        assert attrs["genops.tokens.output"] == 50
        assert attrs["genops.tokens.total"] == 150
    
    def test_record_policy(self, telemetry, mock_span_recorder):
        """Test policy recording functionality."""
        with telemetry.trace_operation("test.policy") as span:
            telemetry.record_policy(
                span=span,
                policy_name="cost_limit",
                result="allowed",
                reason="Under cost threshold",
                metadata={"threshold": 5.0, "actual": 1.5}
            )
        
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        attrs = spans[0].attributes
        assert attrs["genops.policy.name"] == "cost_limit"
        assert attrs["genops.policy.result"] == "allowed"
        assert attrs["genops.policy.reason"] == "Under cost threshold"
        assert attrs["genops.policy.metadata.threshold"] == 5.0
        assert attrs["genops.policy.metadata.actual"] == 1.5
    
    def test_record_evaluation(self, telemetry, mock_span_recorder):
        """Test evaluation recording functionality."""
        with telemetry.trace_operation("test.evaluation") as span:
            telemetry.record_evaluation(
                span=span,
                metric_name="response_quality",
                score=0.85,
                evaluator="human_review",
                metadata={"reviewer_id": "reviewer_123"}
            )
        
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        attrs = spans[0].attributes
        assert attrs["genops.eval.metric"] == "response_quality"
        assert attrs["genops.eval.score"] == 0.85
        assert attrs["genops.eval.evaluator"] == "human_review"
        assert attrs["genops.eval.metadata.reviewer_id"] == "reviewer_123"
    
    def test_record_budget(self, telemetry, mock_span_recorder):
        """Test budget recording functionality."""
        with telemetry.trace_operation("test.budget") as span:
            telemetry.record_budget(
                span=span,
                budget_name="monthly_ai_spend",
                allocated=1000.0,
                consumed=150.0,
                remaining=850.0,
                period="2024-01"
            )
        
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        attrs = spans[0].attributes
        assert attrs["genops.budget.name"] == "monthly_ai_spend"
        assert attrs["genops.budget.allocated"] == 1000.0
        assert attrs["genops.budget.consumed"] == 150.0
        assert attrs["genops.budget.remaining"] == 850.0
        assert attrs["genops.budget.period"] == "2024-01"
    
    def test_multiple_governance_signals(self, telemetry, mock_span_recorder):
        """Test recording multiple governance signals in one operation."""
        with telemetry.trace_operation("test.multi_signals") as span:
            # Record cost
            telemetry.record_cost(span, cost=2.0, currency="USD")
            
            # Record policy
            telemetry.record_policy(span, "rate_limit", "allowed", "Within limits")
            
            # Record evaluation
            telemetry.record_evaluation(span, "accuracy", 0.92, "auto_eval")
            
            # Record budget
            telemetry.record_budget(span, "team_budget", 500.0, 50.0, 450.0)
        
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        attrs = spans[0].attributes
        
        # Verify all governance signals are present
        assert attrs["genops.cost.total"] == 2.0
        assert attrs["genops.policy.name"] == "rate_limit"
        assert attrs["genops.eval.metric"] == "accuracy"
        assert attrs["genops.budget.name"] == "team_budget"
    
    def test_nested_spans(self, telemetry, mock_span_recorder):
        """Test nested span operations."""
        with telemetry.trace_operation("parent.operation") as parent_span:
            telemetry.record_cost(parent_span, cost=1.0, currency="USD")
            
            with telemetry.trace_operation("child.operation") as child_span:
                telemetry.record_cost(child_span, cost=0.5, currency="USD")
        
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 2
        
        # Verify both spans have their respective cost data
        child_span, parent_span = spans  # Finished in reverse order
        
        assert parent_span.name == "parent.operation"
        assert parent_span.attributes["genops.cost.total"] == 1.0
        
        assert child_span.name == "child.operation"
        assert child_span.attributes["genops.cost.total"] == 0.5
    
    @patch('time.time', return_value=1234567890)
    def test_timestamp_recording(self, mock_time, telemetry, mock_span_recorder):
        """Test that timestamps are recorded correctly."""
        with telemetry.trace_operation("test.timestamp") as span:
            pass
        
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["genops.timestamp"] == 1234567890
    
    def test_custom_tracer_name(self, mock_otel_setup):
        """Test GenOpsTelemetry with custom tracer name."""
        custom_telemetry = GenOpsTelemetry("custom-tracer")
        
        with custom_telemetry.trace_operation("test.custom") as span:
            pass
        
        spans = mock_otel_setup.get_finished_spans()
        assert len(spans) == 1
        # Note: The tracer name affects the span's instrumentation_scope,
        # but that's not easily accessible in the test span recorder