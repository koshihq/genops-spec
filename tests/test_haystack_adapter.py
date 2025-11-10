#!/usr/bin/env python3
"""
Comprehensive test suite for Haystack adapter functionality.

Tests cover core adapter functionality, context management, governance patterns,
and error handling scenarios as required by CLAUDE.md standards.
"""

import time
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

# Core test imports
from genops.providers.haystack_adapter import (
    GenOpsComponentMixin,
    GenOpsHaystackAdapter,
    HaystackComponentResult,
    HaystackPipelineResult,
)


class TestGenOpsHaystackAdapter:
    """Core adapter functionality tests."""

    def test_adapter_initialization_with_defaults(self):
        """Test adapter creates with default values."""
        adapter = GenOpsHaystackAdapter(team="test-team", project="test-project")

        assert adapter.team == "test-team"
        assert adapter.project == "test-project"
        assert adapter.environment == "development"
        assert adapter.daily_budget_limit == Decimal("100.0")
        assert adapter.governance_policy == "advisory"

    def test_adapter_initialization_with_custom_values(self):
        """Test adapter creates with custom configuration."""
        adapter = GenOpsHaystackAdapter(
            team="custom-team",
            project="custom-project",
            environment="production",
            daily_budget_limit=250.0,
            governance_policy="enforcing",
            monthly_budget_limit=5000.0
        )

        assert adapter.team == "custom-team"
        assert adapter.project == "custom-project"
        assert adapter.environment == "production"
        assert adapter.daily_budget_limit == Decimal("250.0")
        assert adapter.monthly_budget_limit == Decimal("5000.0")
        assert adapter.governance_policy == "enforcing"

    def test_adapter_invalid_governance_policy(self):
        """Test adapter rejects invalid governance policy."""
        with pytest.raises(ValueError, match="Invalid governance policy"):
            GenOpsHaystackAdapter(
                team="test-team",
                project="test-project",
                governance_policy="invalid-policy"
            )

    def test_adapter_negative_budget_limit(self):
        """Test adapter rejects negative budget limits."""
        with pytest.raises(ValueError, match="Budget limits must be positive"):
            GenOpsHaystackAdapter(
                team="test-team",
                project="test-project",
                daily_budget_limit=-50.0
            )

    def test_adapter_enables_component_tracking_by_default(self):
        """Test adapter enables component tracking by default."""
        adapter = GenOpsHaystackAdapter(team="test-team", project="test-project")
        assert adapter.enable_component_tracking is True

    def test_adapter_cost_aggregator_initialization(self):
        """Test adapter initializes cost aggregator properly."""
        adapter = GenOpsHaystackAdapter(team="test-team", project="test-project")
        assert adapter.cost_aggregator is not None
        assert hasattr(adapter.cost_aggregator, 'add_component_cost')

    def test_adapter_monitor_initialization(self):
        """Test adapter initializes monitor properly."""
        adapter = GenOpsHaystackAdapter(team="test-team", project="test-project")
        assert adapter.monitor is not None
        assert hasattr(adapter.monitor, 'start_pipeline_execution')


class TestHaystackPipelineContext:
    """Pipeline context manager tests."""

    @pytest.fixture
    def adapter(self):
        """Create test adapter."""
        return GenOpsHaystackAdapter(team="test-team", project="test-project")

    def test_pipeline_context_creation(self, adapter):
        """Test pipeline context manager creation."""
        context = adapter.track_pipeline("test-pipeline")

        assert context.pipeline_name == "test-pipeline"
        assert context.customer_id is None
        assert context.use_case is None
        assert context.adapter == adapter

    def test_pipeline_context_with_governance_attributes(self, adapter):
        """Test pipeline context with governance attributes."""
        context = adapter.track_pipeline(
            "test-pipeline",
            customer_id="customer-123",
            use_case="document-qa",
            feature="rag-system"
        )

        assert context.pipeline_name == "test-pipeline"
        assert context.customer_id == "customer-123"
        assert context.use_case == "document-qa"
        assert context.feature == "rag-system"

    def test_pipeline_context_manager_lifecycle(self, adapter):
        """Test pipeline context manager __enter__ and __exit__."""
        with adapter.track_pipeline("test-pipeline") as context:
            assert context.pipeline_id is not None
            assert isinstance(context.pipeline_id, str)
            assert len(context.pipeline_id) > 0

            # Verify context is tracking
            assert hasattr(context, 'start_time')
            assert context.start_time is not None

    def test_pipeline_context_component_tracking(self, adapter):
        """Test pipeline context tracks components."""
        with adapter.track_pipeline("test-pipeline") as context:
            # Add mock component results
            component_result = HaystackComponentResult(
                component_name="test-component",
                component_type="Generator",
                execution_time_seconds=1.5,
                cost=Decimal("0.005"),
                provider_name="OpenAI",
                model_name="gpt-3.5-turbo"
            )

            context.add_component_result(component_result)

            assert len(context.component_results) == 1
            assert context.component_results[0].component_name == "test-component"

    def test_pipeline_context_get_metrics(self, adapter):
        """Test pipeline context metrics calculation."""
        with adapter.track_pipeline("test-pipeline") as context:
            # Add mock component results
            context.add_component_result(HaystackComponentResult(
                component_name="component1",
                component_type="Generator",
                execution_time_seconds=1.0,
                cost=Decimal("0.003"),
                provider_name="OpenAI"
            ))

            context.add_component_result(HaystackComponentResult(
                component_name="component2",
                component_type="Retriever",
                execution_time_seconds=0.5,
                cost=Decimal("0.002"),
                provider_name="OpenAI"
            ))

        metrics = context.get_metrics()
        assert metrics.total_cost == Decimal("0.005")
        assert metrics.total_components == 2
        assert metrics.total_execution_time_seconds >= 1.5
        assert "OpenAI" in metrics.cost_by_provider

    def test_pipeline_context_exception_handling(self, adapter):
        """Test pipeline context handles exceptions properly."""
        try:
            with adapter.track_pipeline("test-pipeline") as context:
                # Add some component results before exception
                context.add_component_result(HaystackComponentResult(
                    component_name="component1",
                    component_type="Generator",
                    execution_time_seconds=1.0,
                    cost=Decimal("0.003"),
                    provider_name="OpenAI"
                ))
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception

        # Context should still have metrics available
        metrics = context.get_metrics()
        assert metrics.total_cost == Decimal("0.003")


class TestHaystackSessionContext:
    """Session context manager tests."""

    @pytest.fixture
    def adapter(self):
        """Create test adapter."""
        return GenOpsHaystackAdapter(team="test-team", project="test-project")

    def test_session_context_creation(self, adapter):
        """Test session context manager creation."""
        session = adapter.track_session("test-session")

        assert session.session_name == "test-session"
        assert session.customer_id is None
        assert session.use_case is None
        assert session.adapter == adapter

    def test_session_context_with_governance_attributes(self, adapter):
        """Test session context with governance attributes."""
        session = adapter.track_session(
            "test-session",
            customer_id="customer-456",
            use_case="multi-pipeline-analysis"
        )

        assert session.session_name == "test-session"
        assert session.customer_id == "customer-456"
        assert session.use_case == "multi-pipeline-analysis"

    def test_session_context_manager_lifecycle(self, adapter):
        """Test session context manager __enter__ and __exit__."""
        with adapter.track_session("test-session") as session:
            assert session.session_id is not None
            assert isinstance(session.session_id, str)
            assert len(session.session_id) > 0

    def test_session_context_pipeline_tracking(self, adapter):
        """Test session context tracks multiple pipelines."""
        with adapter.track_session("test-session") as session:
            # Track first pipeline
            with adapter.track_pipeline("pipeline1") as p1:
                p1.add_component_result(HaystackComponentResult(
                    component_name="comp1",
                    component_type="Generator",
                    execution_time_seconds=1.0,
                    cost=Decimal("0.005"),
                    provider_name="OpenAI"
                ))

            session.add_pipeline_result(p1.get_metrics())

            # Track second pipeline
            with adapter.track_pipeline("pipeline2") as p2:
                p2.add_component_result(HaystackComponentResult(
                    component_name="comp2",
                    component_type="Retriever",
                    execution_time_seconds=0.5,
                    cost=Decimal("0.002"),
                    provider_name="Anthropic"
                ))

            session.add_pipeline_result(p2.get_metrics())

            assert session.total_pipelines == 2
            assert session.total_cost == Decimal("0.007")


class TestHaystackComponentResult:
    """Component result data structure tests."""

    def test_component_result_creation(self):
        """Test component result creation with required fields."""
        result = HaystackComponentResult(
            component_name="test-generator",
            component_type="Generator",
            execution_time_seconds=2.5,
            cost=Decimal("0.01"),
            provider_name="OpenAI"
        )

        assert result.component_name == "test-generator"
        assert result.component_type == "Generator"
        assert result.execution_time_seconds == 2.5
        assert result.cost == Decimal("0.01")
        assert result.provider_name == "OpenAI"

    def test_component_result_with_optional_fields(self):
        """Test component result with optional fields."""
        result = HaystackComponentResult(
            component_name="test-generator",
            component_type="Generator",
            execution_time_seconds=2.5,
            cost=Decimal("0.01"),
            provider_name="OpenAI",
            model_name="gpt-4",
            tokens_used=150,
            success=True,
            error_message=None
        )

        assert result.model_name == "gpt-4"
        assert result.tokens_used == 150
        assert result.success is True
        assert result.error_message is None

    def test_component_result_with_error(self):
        """Test component result with error information."""
        result = HaystackComponentResult(
            component_name="failing-component",
            component_type="Generator",
            execution_time_seconds=0.1,
            cost=Decimal("0.0"),
            provider_name="OpenAI",
            success=False,
            error_message="Rate limit exceeded"
        )

        assert result.success is False
        assert result.error_message == "Rate limit exceeded"


class TestHaystackPipelineResult:
    """Pipeline result data structure tests."""

    def test_pipeline_result_creation(self):
        """Test pipeline result creation."""
        result = HaystackPipelineResult(
            pipeline_name="test-pipeline",
            total_cost=Decimal("0.015"),
            total_components=3,
            total_execution_time_seconds=4.2,
            cost_by_provider={"OpenAI": Decimal("0.01"), "Anthropic": Decimal("0.005")},
            cost_by_component={"gen1": Decimal("0.01"), "ret1": Decimal("0.005")}
        )

        assert result.pipeline_name == "test-pipeline"
        assert result.total_cost == Decimal("0.015")
        assert result.total_components == 3
        assert result.total_execution_time_seconds == 4.2
        assert len(result.cost_by_provider) == 2
        assert len(result.cost_by_component) == 2

    def test_pipeline_result_most_expensive_component(self):
        """Test pipeline result identifies most expensive component."""
        result = HaystackPipelineResult(
            pipeline_name="test-pipeline",
            total_cost=Decimal("0.015"),
            total_components=2,
            total_execution_time_seconds=2.0,
            cost_by_provider={"OpenAI": Decimal("0.015")},
            cost_by_component={"generator": Decimal("0.012"), "retriever": Decimal("0.003")}
        )

        assert result.most_expensive_component == "generator"

    def test_pipeline_result_empty_components(self):
        """Test pipeline result with no components."""
        result = HaystackPipelineResult(
            pipeline_name="empty-pipeline",
            total_cost=Decimal("0.0"),
            total_components=0,
            total_execution_time_seconds=0.0,
            cost_by_provider={},
            cost_by_component={}
        )

        assert result.most_expensive_component is None


class TestGenOpsComponentMixin:
    """Component mixin functionality tests."""

    def test_component_mixin_integration(self):
        """Test component mixin adds GenOps functionality."""
        # Create mock component with mixin
        class MockHaystackComponent(GenOpsComponentMixin):
            def __init__(self):
                super().__init__()
                self.component_config = {}

        component = MockHaystackComponent()

        assert hasattr(component, '_genops_adapter')
        assert hasattr(component, 'set_genops_adapter')
        assert hasattr(component, 'track_execution')

    def test_component_mixin_adapter_setting(self):
        """Test component mixin adapter setting."""
        class MockHaystackComponent(GenOpsComponentMixin):
            def __init__(self):
                super().__init__()
                self.component_config = {}

        component = MockHaystackComponent()
        adapter = GenOpsHaystackAdapter(team="test", project="test")

        component.set_genops_adapter(adapter)
        assert component._genops_adapter == adapter

    def test_component_mixin_execution_tracking(self):
        """Test component mixin execution tracking."""
        class MockHaystackComponent(GenOpsComponentMixin):
            def __init__(self):
                super().__init__()
                self.component_config = {}

            def run(self, **kwargs):
                with self.track_execution("MockComponent") as context:
                    # Simulate component execution
                    time.sleep(0.1)
                    return {"result": "test"}

        component = MockHaystackComponent()
        adapter = GenOpsHaystackAdapter(team="test", project="test")
        component.set_genops_adapter(adapter)

        result = component.run(test_input="value")
        assert result["result"] == "test"


class TestAdapterBudgetEnforcement:
    """Budget enforcement and governance tests."""

    def test_adapter_budget_warning_advisory_mode(self):
        """Test adapter warns about budget in advisory mode."""
        adapter = GenOpsHaystackAdapter(
            team="test-team",
            project="test-project",
            daily_budget_limit=0.01,  # Very low limit
            governance_policy="advisory"
        )

        # Mock cost aggregator to report high costs
        adapter.cost_aggregator.get_daily_costs = Mock(return_value=Decimal("0.015"))

        # Should not raise exception in advisory mode
        with adapter.track_pipeline("test-pipeline") as context:
            context.add_component_result(HaystackComponentResult(
                component_name="expensive-component",
                component_type="Generator",
                execution_time_seconds=1.0,
                cost=Decimal("0.005"),
                provider_name="OpenAI"
            ))

    def test_adapter_budget_enforcement_enforcing_mode(self):
        """Test adapter enforces budget in enforcing mode."""
        adapter = GenOpsHaystackAdapter(
            team="test-team",
            project="test-project",
            daily_budget_limit=0.01,  # Very low limit
            governance_policy="enforcing"
        )

        # Mock cost aggregator to report high costs
        adapter.cost_aggregator.get_daily_costs = Mock(return_value=Decimal("0.015"))

        # Should raise exception in enforcing mode
        with pytest.raises(RuntimeError, match="Daily budget limit exceeded"):
            with adapter.track_pipeline("test-pipeline") as context:
                context.add_component_result(HaystackComponentResult(
                    component_name="expensive-component",
                    component_type="Generator",
                    execution_time_seconds=1.0,
                    cost=Decimal("0.005"),
                    provider_name="OpenAI"
                ))


class TestAdapterTelemetryIntegration:
    """OpenTelemetry integration tests."""

    @patch('genops.providers.haystack_adapter.trace')
    def test_adapter_creates_telemetry_spans(self, mock_trace):
        """Test adapter creates proper telemetry spans."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_trace.get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

        adapter = GenOpsHaystackAdapter(team="test-team", project="test-project")

        with adapter.track_pipeline("test-pipeline"):
            pass

        # Verify telemetry spans were created
        mock_trace.get_tracer.assert_called()
        mock_tracer.start_as_current_span.assert_called()

    @patch('genops.providers.haystack_adapter.trace')
    def test_adapter_sets_span_attributes(self, mock_trace):
        """Test adapter sets proper span attributes."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_trace.get_tracer.return_value = mock_tracer
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

        adapter = GenOpsHaystackAdapter(
            team="test-team",
            project="test-project",
            environment="production"
        )

        with adapter.track_pipeline("test-pipeline", customer_id="cust-123"):
            pass

        # Verify governance attributes were set
        mock_span.set_attribute.assert_any_call("genops.team", "test-team")
        mock_span.set_attribute.assert_any_call("genops.project", "test-project")
        mock_span.set_attribute.assert_any_call("genops.environment", "production")
        mock_span.set_attribute.assert_any_call("genops.customer_id", "cust-123")


class TestAdapterErrorHandling:
    """Error handling and resilience tests."""

    def test_adapter_handles_missing_dependencies_gracefully(self):
        """Test adapter handles missing Haystack gracefully."""
        with patch('genops.providers.haystack_adapter.HAS_HAYSTACK', False):
            adapter = GenOpsHaystackAdapter(team="test", project="test")
            assert adapter is not None

    def test_adapter_handles_cost_aggregator_failures(self):
        """Test adapter handles cost aggregator failures."""
        adapter = GenOpsHaystackAdapter(team="test", project="test")

        # Mock cost aggregator to raise exception
        adapter.cost_aggregator.add_component_cost = Mock(side_effect=Exception("Cost calc failed"))

        # Should not crash pipeline execution
        with adapter.track_pipeline("test-pipeline") as context:
            context.add_component_result(HaystackComponentResult(
                component_name="test-component",
                component_type="Generator",
                execution_time_seconds=1.0,
                cost=Decimal("0.005"),
                provider_name="OpenAI"
            ))

    def test_adapter_handles_monitor_failures(self):
        """Test adapter handles monitor failures."""
        adapter = GenOpsHaystackAdapter(team="test", project="test")

        # Mock monitor to raise exception
        adapter.monitor.start_pipeline_execution = Mock(side_effect=Exception("Monitor failed"))

        # Should not crash pipeline execution
        with adapter.track_pipeline("test-pipeline"):
            pass

    def test_adapter_context_manager_cleanup_on_exception(self):
        """Test adapter cleans up properly on exception."""
        adapter = GenOpsHaystackAdapter(team="test", project="test")

        try:
            with adapter.track_pipeline("test-pipeline") as context:
                # Add some results
                context.add_component_result(HaystackComponentResult(
                    component_name="test-component",
                    component_type="Generator",
                    execution_time_seconds=1.0,
                    cost=Decimal("0.005"),
                    provider_name="OpenAI"
                ))
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Context should still be accessible and have results
        metrics = context.get_metrics()
        assert metrics.total_cost == Decimal("0.005")
