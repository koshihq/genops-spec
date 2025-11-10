"""
Comprehensive tests for GenOps PromptLayer Adapter.

Tests the core adapter functionality including:
- Prompt management with governance enhancement
- Cost attribution and tracking
- Policy enforcement and budget management
- Auto-instrumentation patterns
- Context manager lifecycle
- Error handling and resilience
- Performance monitoring
"""

import time
from unittest.mock import Mock, patch

import pytest

# Import the modules under test
try:
    from genops.providers.promptlayer import (
        EnhancedPromptLayerSpan,
        GenOpsPromptLayerAdapter,
        GovernancePolicy,
        PromptLayerResponse,
        PromptLayerUsage,
        auto_instrument,
        instrument_promptlayer,
    )
    PROMPTLAYER_AVAILABLE = True
except ImportError:
    PROMPTLAYER_AVAILABLE = False


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestGenOpsPromptLayerAdapter:
    """Test suite for the main PromptLayer adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            self.adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-test-key',
                team='test-team',
                project='test-project',
                environment='test',
                daily_budget_limit=10.0,
                max_operation_cost=1.0,
                enable_governance=True
            )

        self.sample_governance_attrs = {
            'team': 'test-team',
            'project': 'test-project',
            'customer_id': 'test-customer',
            'environment': 'test'
        }

    def test_adapter_initialization(self):
        """Test adapter initialization with various configurations."""
        # Test basic initialization
        assert self.adapter.team == 'test-team'
        assert self.adapter.project == 'test-project'
        assert self.adapter.environment == 'test'
        assert self.adapter.daily_budget_limit == 10.0
        assert self.adapter.max_operation_cost == 1.0
        assert self.adapter.enable_governance is True

        # Test default initialization
        with patch('genops.providers.promptlayer.PromptLayer'):
            default_adapter = GenOpsPromptLayerAdapter()
            assert default_adapter.enable_governance is True
            assert default_adapter.governance_policy == GovernancePolicy.ADVISORY

    def test_adapter_initialization_with_environment_variables(self):
        """Test adapter initialization using environment variables."""
        with patch.dict('os.environ', {
            'PROMPTLAYER_API_KEY': 'pl-env-key',
            'GENOPS_TEAM': 'env-team',
            'GENOPS_PROJECT': 'env-project'
        }):
            with patch('genops.providers.promptlayer.PromptLayer'):
                adapter = GenOpsPromptLayerAdapter()
                assert adapter.promptlayer_api_key == 'pl-env-key'
                assert adapter.team == 'env-team'
                assert adapter.project == 'env-project'

    def test_adapter_initialization_without_promptlayer(self):
        """Test adapter initialization when PromptLayer is not available."""
        with patch('genops.providers.promptlayer.HAS_PROMPTLAYER', False):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-test-key',
                team='test-team'
            )
            # Should use MockPromptLayer
            assert adapter.client is not None

    def test_governance_policy_enforcement(self):
        """Test governance policy enforcement levels."""
        # Test advisory policy
        with patch('genops.providers.promptlayer.PromptLayer'):
            advisory_adapter = GenOpsPromptLayerAdapter(
                governance_policy=GovernancePolicy.ADVISORY,
                daily_budget_limit=0.01  # Very low limit
            )
            assert advisory_adapter.governance_policy == GovernancePolicy.ADVISORY

        # Test enforced policy
        with patch('genops.providers.promptlayer.PromptLayer'):
            enforced_adapter = GenOpsPromptLayerAdapter(
                governance_policy=GovernancePolicy.ENFORCED,
                daily_budget_limit=0.01
            )
            assert enforced_adapter.governance_policy == GovernancePolicy.ENFORCED

    def test_track_prompt_operation_context_manager(self):
        """Test the track_prompt_operation context manager."""
        with self.adapter.track_prompt_operation(
            prompt_name='test_prompt',
            operation_type='test_operation'
        ) as span:
            assert isinstance(span, EnhancedPromptLayerSpan)
            assert span.prompt_name == 'test_prompt'
            assert span.operation_type == 'test_operation'
            assert span.team == 'test-team'
            assert span.project == 'test-project'
            assert span.operation_id in self.adapter.active_spans

        # After context exits, span should be removed from active spans
        assert span.operation_id not in self.adapter.active_spans
        assert span.end_time is not None

    def test_track_prompt_operation_with_custom_attributes(self):
        """Test context manager with custom attributes and limits."""
        with self.adapter.track_prompt_operation(
            prompt_name='custom_prompt',
            prompt_version='v2.1',
            customer_id='customer_123',
            cost_center='marketing',
            tags={'campaign': 'q4_promo', 'priority': 'high'},
            max_cost=0.50
        ) as span:
            assert span.prompt_name == 'custom_prompt'
            assert span.prompt_version == 'v2.1'
            assert span.customer_id == 'customer_123'
            assert span.cost_center == 'marketing'
            assert span.tags['campaign'] == 'q4_promo'
            assert span.max_cost == 0.50

    def test_budget_enforcement_advisory(self):
        """Test budget enforcement in advisory mode."""
        # Set up adapter with low budget limit
        self.adapter.governance_policy = GovernancePolicy.ADVISORY
        self.adapter.daily_budget_limit = 0.01
        self.adapter.daily_usage = 0.005  # Half budget used

        with self.adapter.track_prompt_operation('budget_test') as span:
            # Simulate operation that would exceed budget
            span.update_cost(0.02)  # Would exceed daily limit
            assert len(span.policy_violations) > 0
            assert 'budget limit' in span.policy_violations[0].lower()

    def test_budget_enforcement_enforced(self):
        """Test budget enforcement in enforced mode."""
        # Set up adapter with enforced policy and exceeded budget
        self.adapter.governance_policy = GovernancePolicy.ENFORCED
        self.adapter.daily_budget_limit = 0.01
        self.adapter.daily_usage = 0.02  # Budget already exceeded

        # Should raise exception due to budget violation
        with pytest.raises(ValueError, match="Daily budget limit"):
            with self.adapter.track_prompt_operation('budget_violation_test') as span:
                pass

    def test_operation_cost_limit_enforcement(self):
        """Test max operation cost enforcement."""
        with self.adapter.track_prompt_operation(
            'cost_limit_test',
            max_cost=0.05
        ) as span:
            # Cost within limit should be fine
            span.update_cost(0.03)
            assert len([v for v in span.policy_violations if 'operation cost' in v.lower()]) == 0

            # Cost exceeding limit should trigger violation
            span.update_cost(0.08)
            assert len([v for v in span.policy_violations if 'operation cost' in v.lower()]) > 0

    def test_usage_tracking_updates(self):
        """Test that usage tracking is properly updated."""
        initial_usage = self.adapter.daily_usage
        initial_count = self.adapter.operation_count

        with self.adapter.track_prompt_operation('usage_test') as span:
            span.update_cost(0.05)

        assert self.adapter.daily_usage == initial_usage + 0.05
        assert self.adapter.operation_count == initial_count + 1

    @patch('genops.providers.promptlayer.PromptLayer')
    def test_run_prompt_with_governance(self, mock_promptlayer_class):
        """Test running prompts with governance tracking."""
        # Setup mock
        mock_client = Mock()
        mock_promptlayer_class.return_value = mock_client
        mock_client.run.return_value = {
            'response': 'Test response',
            'usage': {'input_tokens': 10, 'output_tokens': 20}
        }

        # Create adapter with mock
        adapter = GenOpsPromptLayerAdapter(
            promptlayer_api_key='pl-test-key',
            team='test-team'
        )

        # Test prompt execution
        result = adapter.run_prompt_with_governance(
            prompt_name='test_prompt',
            input_variables={'query': 'test query'},
            prompt_version='v1.0',
            tags=['test_tag']
        )

        # Verify mock was called correctly
        mock_client.run.assert_called_once()
        call_args = mock_client.run.call_args

        assert call_args[1]['prompt_name'] == 'test_prompt'
        assert call_args[1]['input_variables'] == {'query': 'test query'}
        assert call_args[1]['version'] == 'v1.0'
        assert 'team:test-team' in call_args[1]['tags']

        # Verify governance context in response
        assert 'governance' in result
        assert result['governance']['team'] == 'test-team'

    def test_run_prompt_with_governance_mock_client(self):
        """Test prompt execution with mock client when PromptLayer unavailable."""
        # Force use of mock client
        self.adapter.client = Mock()
        self.adapter.client.run = Mock(return_value={'mock': True, 'message': 'PromptLayer not available'})

        result = self.adapter.run_prompt_with_governance(
            prompt_name='mock_test',
            input_variables={'test': 'value'}
        )

        assert result['response']['mock'] is True
        assert 'governance' in result

    def test_get_metrics(self):
        """Test metrics collection and retrieval."""
        # Update some usage
        self.adapter.daily_usage = 0.15
        self.adapter.operation_count = 5

        metrics = self.adapter.get_metrics()

        assert metrics['team'] == 'test-team'
        assert metrics['project'] == 'test-project'
        assert metrics['environment'] == 'test'
        assert metrics['daily_usage'] == 0.15
        assert metrics['operation_count'] == 5
        assert metrics['budget_remaining'] == 9.85  # 10.0 - 0.15
        assert metrics['governance_enabled'] is True
        assert metrics['policy_level'] == GovernancePolicy.ADVISORY.value

    def test_error_handling_in_context_manager(self):
        """Test error handling within the context manager."""
        with pytest.raises(ValueError, match="Test error"):
            with self.adapter.track_prompt_operation('error_test') as span:
                # Simulate error during operation
                raise ValueError("Test error")

        # Span should still be finalized and removed from active spans
        assert span.operation_id not in self.adapter.active_spans
        assert span.end_time is not None
        assert span.metadata.get('error') == "Test error"

    def test_concurrent_operations_tracking(self):
        """Test tracking multiple concurrent operations."""
        spans = []

        # Start multiple operations
        span1 = self.adapter.track_prompt_operation('concurrent_1').__enter__()
        span2 = self.adapter.track_prompt_operation('concurrent_2').__enter__()
        span3 = self.adapter.track_prompt_operation('concurrent_3').__enter__()

        spans.extend([span1, span2, span3])

        # All should be tracked as active
        assert len(self.adapter.active_spans) == 3
        assert all(span.operation_id in self.adapter.active_spans for span in spans)

        # Close operations
        for span in spans:
            self.adapter.track_prompt_operation('dummy').__exit__(None, None, None)

        # Clean up manually since we didn't use proper context managers
        for span in spans:
            if span.operation_id in self.adapter.active_spans:
                del self.adapter.active_spans[span.operation_id]

    def test_governance_policy_checking(self):
        """Test the governance policy checking mechanism."""
        span = EnhancedPromptLayerSpan(
            operation_type='test',
            operation_name='policy_test',
            team='test-team',
            max_cost=0.10
        )

        # Test within limits
        span.update_cost(0.05)
        self.adapter._check_governance_policies(span)
        assert len(span.policy_violations) == 0

        # Test exceeding operation cost limit
        span.update_cost(0.15)
        self.adapter._check_governance_policies(span)
        assert len(span.policy_violations) > 0
        assert any('operation cost' in v.lower() for v in span.policy_violations)

    def test_custom_tags_propagation(self):
        """Test that custom tags are properly propagated."""
        custom_tags = {'environment': 'staging', 'feature': 'recommendation'}

        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            adapter = GenOpsPromptLayerAdapter(
                team='test-team',
                tags=custom_tags
            )

            mock_client = Mock()
            mock_pl.return_value = mock_client
            mock_client.run.return_value = {'response': 'test'}
            adapter.client = mock_client

            with adapter.track_prompt_operation('tag_test') as span:
                assert span.tags['environment'] == 'staging'
                assert span.tags['feature'] == 'recommendation'

    def test_span_metrics_calculation(self):
        """Test span metrics calculation and aggregation."""
        with self.adapter.track_prompt_operation('metrics_test') as span:
            span.update_cost(0.025)
            span.update_token_usage(50, 100, 'gpt-3.5-turbo')
            span.add_attributes({'custom_metric': 'test_value'})

            metrics = span.get_metrics()

            assert metrics['estimated_cost'] == 0.025
            assert metrics['input_tokens'] == 50
            assert metrics['output_tokens'] == 100
            assert metrics['total_tokens'] == 150
            assert metrics['model'] == 'gpt-3.5-turbo'
            assert metrics['team'] == 'test-team'
            assert metrics['metadata']['custom_metric'] == 'test_value'


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestEnhancedPromptLayerSpan:
    """Test suite for the EnhancedPromptLayerSpan class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.span = EnhancedPromptLayerSpan(
            operation_type='test_operation',
            operation_name='test_span',
            prompt_name='test_prompt',
            team='test-team',
            project='test-project'
        )

    def test_span_initialization(self):
        """Test span initialization with various parameters."""
        assert self.span.operation_type == 'test_operation'
        assert self.span.operation_name == 'test_span'
        assert self.span.prompt_name == 'test_prompt'
        assert self.span.team == 'test-team'
        assert self.span.project == 'test-project'
        assert self.span.start_time is not None
        assert len(self.span.operation_id) > 0

    def test_cost_update_and_limits(self):
        """Test cost updating and limit checking."""
        # Test normal cost update
        self.span.update_cost(0.05)
        assert self.span.estimated_cost == 0.05

        # Test cost limit violation
        span_with_limit = EnhancedPromptLayerSpan(
            operation_type='test',
            operation_name='limit_test',
            max_cost=0.10
        )

        span_with_limit.update_cost(0.15)  # Exceeds limit
        assert len(span_with_limit.policy_violations) > 0
        assert any('exceeds maximum' in v for v in span_with_limit.policy_violations)

    def test_token_usage_update(self):
        """Test token usage tracking and cost estimation."""
        # Test GPT-4 pricing
        self.span.update_token_usage(1000, 500, 'gpt-4')
        assert self.span.input_tokens == 1000
        assert self.span.output_tokens == 500
        assert self.span.total_tokens == 1500
        assert self.span.model == 'gpt-4'
        # Should estimate cost based on GPT-4 pricing
        assert self.span.estimated_cost > 0.05  # GPT-4 is expensive

        # Test GPT-3.5 pricing
        span_35 = EnhancedPromptLayerSpan('test', 'test')
        span_35.update_token_usage(1000, 500, 'gpt-3.5-turbo')
        assert span_35.estimated_cost < 0.01  # GPT-3.5 is cheaper

    def test_attributes_management(self):
        """Test custom attribute management."""
        test_attrs = {
            'custom_field': 'test_value',
            'priority': 'high',
            'team': 'override-team'  # Should override existing team
        }

        self.span.add_attributes(test_attrs)

        assert self.span.metadata['custom_field'] == 'test_value'
        assert self.span.metadata['priority'] == 'high'
        assert self.span.team == 'override-team'  # Should be overridden

    def test_metrics_generation(self):
        """Test comprehensive metrics generation."""
        # Setup span with various data
        self.span.update_cost(0.032)
        self.span.update_token_usage(75, 125, 'gpt-3.5-turbo')
        self.span.add_attributes({'quality_score': 0.85})

        # Simulate some duration
        time.sleep(0.01)  # Small delay
        self.span.finalize()

        metrics = self.span.get_metrics()

        # Verify all expected fields
        assert metrics['operation_id'] == self.span.operation_id
        assert metrics['operation_type'] == 'test_operation'
        assert metrics['prompt_name'] == 'test_prompt'
        assert metrics['estimated_cost'] == 0.032
        assert metrics['total_tokens'] == 200
        assert metrics['team'] == 'test-team'
        assert metrics['duration_seconds'] > 0
        assert 'quality_score' in metrics['metadata']

    def test_span_finalization(self):
        """Test span finalization process."""
        assert self.span.end_time is None

        self.span.finalize()

        assert self.span.end_time is not None
        assert self.span.end_time >= self.span.start_time


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestPromptLayerInstrumentationFunctions:
    """Test suite for instrumentation helper functions."""

    @patch('genops.providers.promptlayer.GenOpsPromptLayerAdapter')
    def test_instrument_promptlayer_function(self, mock_adapter_class):
        """Test the instrument_promptlayer convenience function."""
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter

        result = instrument_promptlayer(
            promptlayer_api_key='pl-test-key',
            team='function-test-team',
            project='function-test-project'
        )

        # Verify adapter was created with correct parameters
        mock_adapter_class.assert_called_once_with(
            promptlayer_api_key='pl-test-key',
            team='function-test-team',
            project='function-test-project'
        )
        assert result == mock_adapter

    @patch('genops.providers.promptlayer.GenOpsPromptLayerAdapter')
    def test_auto_instrument_function(self, mock_adapter_class):
        """Test the auto_instrument function for zero-code integration."""
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter

        auto_instrument(
            promptlayer_api_key='pl-auto-key',
            team='auto-team',
            project='auto-project',
            environment='production'
        )

        # Verify global adapter was created
        mock_adapter_class.assert_called_once_with(
            promptlayer_api_key='pl-auto-key',
            team='auto-team',
            project='auto-project',
            environment='production'
        )

        # Test get_current_adapter
        from genops.providers.promptlayer import get_current_adapter
        current_adapter = get_current_adapter()
        assert current_adapter == mock_adapter

    def test_governance_policy_enum(self):
        """Test GovernancePolicy enum values."""
        assert GovernancePolicy.ADVISORY.value == 'advisory'
        assert GovernancePolicy.ENFORCED.value == 'enforced'
        assert GovernancePolicy.AUDIT_ONLY.value == 'audit_only'


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestPromptLayerDataClasses:
    """Test suite for PromptLayer data classes."""

    def test_promptlayer_usage_dataclass(self):
        """Test PromptLayerUsage data class."""
        usage = PromptLayerUsage(
            operation_id='test-op-123',
            operation_type='prompt_run',
            prompt_name='test_prompt',
            prompt_version='v1.0',
            model='gpt-3.5-turbo',
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
            cost=0.025,
            latency_ms=1250.5,
            team='data-team',
            project='data-project'
        )

        assert usage.operation_id == 'test-op-123'
        assert usage.total_tokens == 150
        assert usage.cost == 0.025
        assert usage.environment == 'production'  # Default value
        assert len(usage.policy_violations) == 0  # Default empty list

    def test_promptlayer_response_dataclass(self):
        """Test PromptLayerResponse data class."""
        response = PromptLayerResponse(
            content='Test response content',
            usage=PromptLayerUsage(
                operation_id='resp-test',
                operation_type='test',
                prompt_name='test',
                prompt_version=None,
                model='gpt-3.5-turbo',
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cost=0.01,
                latency_ms=500
            ),
            prompt_id='prompt-123',
            request_id='req-456'
        )

        assert response.content == 'Test response content'
        assert response.usage.operation_id == 'resp-test'
        assert response.prompt_id == 'prompt-123'
        assert response.governance_status == 'compliant'  # Default
        assert len(response.cost_optimization_suggestions) == 0  # Default empty


@pytest.mark.integration
@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestPromptLayerRealIntegration:
    """Integration tests that require real PromptLayer API keys."""

    def test_real_promptlayer_connection(self):
        """Test connection to real PromptLayer API."""
        import os

        api_key = os.getenv('PROMPTLAYER_API_KEY')
        if not api_key:
            pytest.skip("PROMPTLAYER_API_KEY not set for integration tests")

        adapter = GenOpsPromptLayerAdapter(
            promptlayer_api_key=api_key,
            team='integration-test',
            project='test-suite'
        )

        # Test basic initialization
        assert adapter.promptlayer_api_key == api_key
        assert adapter.team == 'integration-test'

        # Test metrics collection
        metrics = adapter.get_metrics()
        assert 'team' in metrics
        assert 'daily_usage' in metrics

    def test_real_governance_tracking(self):
        """Test governance tracking with real operations."""
        import os

        api_key = os.getenv('PROMPTLAYER_API_KEY')
        if not api_key:
            pytest.skip("PROMPTLAYER_API_KEY not set for integration tests")

        adapter = GenOpsPromptLayerAdapter(
            promptlayer_api_key=api_key,
            team='integration-test',
            daily_budget_limit=0.10  # Small budget for testing
        )

        with adapter.track_prompt_operation('integration_test') as span:
            span.update_cost(0.01)
            span.add_attributes({'integration_test': True})

        # Verify tracking worked
        assert adapter.daily_usage == 0.01
        assert adapter.operation_count == 1

        # Test metrics
        metrics = adapter.get_metrics()
        assert metrics['daily_usage'] == 0.01
        assert metrics['budget_remaining'] == 0.09
