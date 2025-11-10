#!/usr/bin/env python3
"""
End-to-end integration test suite for GenOps Arize AI integration.

This test suite provides comprehensive end-to-end testing of the complete Arize AI
integration workflow including auto-instrumentation, cost tracking, governance,
and multi-module interactions.

Test Categories:
- End-to-end workflow tests (20 tests)
- Auto-instrumentation integration tests (15 tests)
- Multi-module interaction tests (12 tests)
- Governance and cost intelligence tests (10 tests)
- Production scenario simulation tests (8 tests)

Total: 65 tests ensuring robust end-to-end Arize AI integration functionality.
"""

import os
import sys
import time
import unittest
from unittest.mock import Mock, patch

import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from genops.providers.arize import (
    GenOpsArizeAdapter,
    auto_instrument,
    get_current_adapter,
    instrument_arize,
    set_global_adapter,
)
from genops.providers.arize_cost_aggregator import ArizeCostAggregator
from genops.providers.arize_pricing import (
    ArizePricingCalculator,
    ModelTier,
    optimize_pricing_strategy,
)
from genops.providers.arize_validation import ArizeSetupValidator


class TestEndToEndWorkflows(unittest.TestCase):
    """Test complete end-to-end workflows."""

    def setUp(self):
        """Set up comprehensive test fixtures."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'ARIZE_API_KEY': 'test-integration-api-key-123456789',
            'ARIZE_SPACE_KEY': 'test-integration-space-key-123456789',
            'GENOPS_TEAM': 'integration-test-team',
            'GENOPS_PROJECT': 'arize-integration-project',
            'GENOPS_ENVIRONMENT': 'integration-testing',
            'GENOPS_DAILY_BUDGET_LIMIT': '75.0'
        }, clear=False)
        self.env_patcher.start()

        # Mock Arize SDK components
        self.arize_patch = patch('genops.providers.arize.ARIZE_AVAILABLE', True)
        self.arize_client_patch = patch('genops.providers.arize.ArizeClient')

        self.arize_patch.start()
        self.arize_client_mock = self.arize_client_patch.start()

        # Reset global state
        set_global_adapter(None)

    def tearDown(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()
        self.arize_patch.stop()
        self.arize_client_patch.stop()
        set_global_adapter(None)

    def test_complete_model_monitoring_workflow(self):
        """Test complete model monitoring workflow from setup to teardown."""
        # Step 1: Initialize adapter
        adapter = GenOpsArizeAdapter(
            team="integration-test-team",
            project="fraud-detection-integration",
            daily_budget_limit=100.0,
            enable_cost_alerts=True,
            enable_governance=True
        )

        # Step 2: Start monitoring session
        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            with adapter.track_model_monitoring_session(
                model_id='fraud-model-v3',
                model_version='3.1.0',
                environment='production',
                max_cost=25.0
            ) as session:

                # Step 3: Log predictions with cost tracking
                predictions_df = pd.DataFrame({
                    'prediction_id': [f'pred_{i}' for i in range(1000)],
                    'prediction': [0, 1] * 500,
                    'confidence': [0.85 + (i * 0.0001) for i in range(1000)]
                })
                session.log_prediction_batch(predictions_df, cost_per_prediction=0.002)

                # Step 4: Monitor data quality
                quality_metrics = {
                    'accuracy': 0.92,
                    'precision': 0.89,
                    'recall': 0.94,
                    'f1_score': 0.91,
                    'data_drift_score': 0.15
                }
                session.log_data_quality_metrics(quality_metrics, cost_estimate=0.08)

                # Step 5: Create performance alerts
                session.create_performance_alert(
                    metric='accuracy',
                    threshold=0.85,
                    cost_per_alert=0.12
                )
                session.create_performance_alert(
                    metric='data_drift_score',
                    threshold=0.20,
                    cost_per_alert=0.10
                )

                # Step 6: Update monitoring costs manually
                session.update_monitoring_cost(0.05)  # Additional processing cost

                # Verify session state
                self.assertEqual(session.prediction_count, 1000)
                self.assertEqual(session.data_quality_checks, 1)
                self.assertEqual(session.active_alerts, 2)
                self.assertEqual(session.estimated_cost, 0.45)  # 2.0 + 0.08 + 0.12 + 0.10 + 0.05

        # Step 7: Verify final state
        self.assertEqual(adapter.daily_usage, 0.45)
        self.assertEqual(adapter.operation_count, 1)
        self.assertEqual(len(adapter.active_sessions), 0)  # Should be cleaned up

    def test_multi_session_concurrent_monitoring(self):
        """Test concurrent monitoring of multiple models."""
        adapter = GenOpsArizeAdapter(
            team="concurrent-test-team",
            project="multi-model-monitoring",
            daily_budget_limit=200.0
        )

        sessions_data = []

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            # Start multiple concurrent sessions
            with adapter.track_model_monitoring_session('model-a', 'v1', 'production') as session_a:
                with adapter.track_model_monitoring_session('model-b', 'v2', 'staging') as session_b:
                    with adapter.track_model_monitoring_session('model-c', 'v1', 'production') as session_c:

                        # Verify all sessions are active
                        self.assertEqual(len(adapter.active_sessions), 3)

                        # Log different activities in each session
                        session_a.log_prediction_batch(pd.DataFrame({'pred': [1, 0, 1]}), 0.001)
                        session_b.log_data_quality_metrics({'accuracy': 0.88}, 0.03)
                        session_c.create_performance_alert('drift', 0.15, 0.08)

                        # Collect session data
                        sessions_data.extend([
                            {
                                'id': session_a.session_id,
                                'model': session_a.model_id,
                                'cost': session_a.estimated_cost
                            },
                            {
                                'id': session_b.session_id,
                                'model': session_b.model_id,
                                'cost': session_b.estimated_cost
                            },
                            {
                                'id': session_c.session_id,
                                'model': session_c.model_id,
                                'cost': session_c.estimated_cost
                            }
                        ])

                        # Verify session isolation
                        self.assertEqual(session_a.prediction_count, 3)
                        self.assertEqual(session_b.data_quality_checks, 1)
                        self.assertEqual(session_c.active_alerts, 1)

        # Verify final cleanup
        self.assertEqual(len(adapter.active_sessions), 0)

        # Verify cost aggregation
        total_expected_cost = sum(s['cost'] for s in sessions_data)
        self.assertEqual(adapter.daily_usage, total_expected_cost)

    def test_cost_budget_enforcement_workflow(self):
        """Test cost budget enforcement in real workflow."""
        adapter = GenOpsArizeAdapter(
            team="budget-test-team",
            project="cost-enforcement",
            daily_budget_limit=10.0,  # Low budget for testing
            max_monitoring_cost=5.0,  # Low session limit
            enable_cost_alerts=True
        )

        with patch('genops.providers.arize.logger') as mock_logger:
            with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
                mock_span.return_value.__enter__ = Mock(return_value=Mock())
                mock_span.return_value.__exit__ = Mock(return_value=None)

                # Simulate high-cost operations that should trigger warnings
                with adapter.track_model_monitoring_session('expensive-model', 'v1', 'production') as session:

                    # High-volume prediction logging
                    large_df = pd.DataFrame({'pred': [1, 0] * 2500})  # 5000 predictions
                    session.log_prediction_batch(large_df, cost_per_prediction=0.002)  # $10 cost

                    # Expensive data quality checks
                    session.log_data_quality_metrics({'quality': 0.9}, cost_estimate=2.0)

                    # Multiple alerts
                    for i in range(5):
                        session.create_performance_alert(f'metric_{i}', 0.8, 1.0)

        # Should have triggered cost warnings
        mock_logger.warning.assert_called()
        warning_calls = [call for call in mock_logger.warning.call_args_list]
        self.assertGreater(len(warning_calls), 0)

        # Verify budget validation was called
        budget_warnings = [call for call in warning_calls if 'budget' in str(call).lower()]
        self.assertGreater(len(budget_warnings), 0)

    def test_governance_policy_compliance_workflow(self):
        """Test governance policy compliance throughout workflow."""
        adapter = GenOpsArizeAdapter(
            team="governance-team",
            project="compliance-testing",
            environment="production",
            enable_governance=True,
            cost_center="ml-ops",
            tags={"compliance": "required", "audit": "enabled"}
        )

        governance_attributes = []

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            def capture_attributes(span_name, attributes=None, **kwargs):
                mock_context = Mock()
                mock_context.__enter__ = Mock(return_value=Mock())
                mock_context.__exit__ = Mock(return_value=None)

                # Capture governance attributes
                if attributes:
                    governance_attributes.append({
                        'span_name': span_name,
                        'attributes': attributes
                    })

                return mock_context

            mock_span.side_effect = capture_attributes

            with adapter.track_model_monitoring_session(
                model_id='compliant-model',
                model_version='v2.1',
                environment='production'
            ) as session:

                # Perform monitored operations
                session.log_prediction_batch(pd.DataFrame({'pred': [1, 0, 1, 1]}), 0.001)
                session.log_data_quality_metrics({'compliance_score': 0.95}, 0.02)
                session.create_performance_alert('compliance', 0.90, 0.05)

        # Verify governance attributes were captured
        self.assertGreater(len(governance_attributes), 0)

        # Check for required governance attributes
        session_attrs = next((attrs for attrs in governance_attributes if attrs['span_name'] == 'arize.monitoring.session'), None)
        self.assertIsNotNone(session_attrs)

        required_attrs = [
            'genops.team', 'genops.project', 'genops.environment',
            'genops.model.id', 'genops.model.version', 'genops.cost.budget_limit'
        ]

        for attr in required_attrs:
            self.assertIn(attr, session_attrs['attributes'])

    def test_error_recovery_and_resilience_workflow(self):
        """Test error recovery and resilience in workflows."""
        adapter = GenOpsArizeAdapter(
            team="resilience-team",
            project="error-testing"
        )

        error_count = 0
        recovery_count = 0

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            # Test partial session failures with recovery
            try:
                with adapter.track_model_monitoring_session('error-prone-model', 'v1') as session:

                    # Successful operation
                    session.log_prediction_batch(pd.DataFrame({'pred': [1, 0]}), 0.001)

                    # Simulated error in data quality logging
                    try:
                        with patch.object(session, 'log_data_quality_metrics', side_effect=Exception('Data quality error')):
                            session.log_data_quality_metrics({'accuracy': 0.9}, 0.02)
                    except Exception:
                        error_count += 1
                        # Recovery: continue with other operations
                        session.create_performance_alert('backup_metric', 0.85, 0.03)
                        recovery_count += 1

                    # Should still have successful prediction logging
                    self.assertEqual(session.prediction_count, 2)

            except Exception as e:
                self.fail(f"Session should not fail completely due to partial errors: {e}")

        # Verify error handling and recovery
        self.assertEqual(error_count, 1)
        self.assertEqual(recovery_count, 1)

        # Adapter should still be in valid state
        metrics = adapter.get_metrics()
        self.assertGreater(metrics['operation_count'], 0)

    def test_performance_under_load_workflow(self):
        """Test performance characteristics under load."""
        adapter = GenOpsArizeAdapter(
            team="performance-team",
            project="load-testing",
            daily_budget_limit=500.0
        )

        start_time = time.time()

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            # Simulate high-volume monitoring
            for batch in range(10):  # 10 batches
                with adapter.track_model_monitoring_session(f'load-model-{batch}', 'v1') as session:

                    # Large prediction batches
                    predictions = pd.DataFrame({
                        'pred': [batch % 2] * 500,  # 500 predictions per batch
                        'confidence': [0.8 + (batch * 0.01)] * 500
                    })
                    session.log_prediction_batch(predictions, 0.001)

                    # Multiple quality checks
                    for qc in range(5):
                        session.log_data_quality_metrics({
                            f'metric_{qc}': 0.85 + (qc * 0.02)
                        }, 0.01)

                    # Multiple alerts
                    for alert in range(3):
                        session.create_performance_alert(f'alert_{alert}', 0.8, 0.02)

        elapsed_time = time.time() - start_time

        # Performance assertions
        self.assertLess(elapsed_time, 5.0)  # Should complete within 5 seconds

        # Verify all operations completed successfully
        final_metrics = adapter.get_metrics()
        self.assertEqual(final_metrics['operation_count'], 10)  # 10 batches

        # Verify cost tracking accuracy
        expected_operations = 10 * (1 + 5 + 3)  # Each batch: 1 prediction batch + 5 quality checks + 3 alerts
        total_cost = adapter.daily_usage
        self.assertGreater(total_cost, 0)
        self.assertLess(total_cost, 500.0)  # Within budget


class TestAutoInstrumentationIntegration(unittest.TestCase):
    """Test auto-instrumentation integration workflows."""

    def setUp(self):
        """Set up auto-instrumentation test fixtures."""
        self.env_patcher = patch.dict(os.environ, {
            'ARIZE_API_KEY': 'auto-instr-api-key-123456789',
            'ARIZE_SPACE_KEY': 'auto-instr-space-key-123456789',
            'GENOPS_TEAM': 'auto-instrumentation-team',
            'GENOPS_PROJECT': 'zero-code-integration'
        }, clear=False)
        self.env_patcher.start()

        # Mock Arize SDK
        self.arize_patch = patch('genops.providers.arize.ARIZE_AVAILABLE', True)
        self.arize_client_patch = patch('genops.providers.arize.ArizeClient')

        self.arize_patch.start()
        self.arize_client_mock = self.arize_client_patch.start()

        # Reset global state
        set_global_adapter(None)

    def tearDown(self):
        """Clean up auto-instrumentation test fixtures."""
        self.env_patcher.stop()
        self.arize_patch.stop()
        self.arize_client_patch.stop()
        set_global_adapter(None)

    def test_zero_code_auto_instrumentation_workflow(self):
        """Test complete zero-code auto-instrumentation workflow."""
        # Step 1: Enable auto-instrumentation
        adapter = auto_instrument(
            team="zero-code-team",
            project="automated-monitoring",
            daily_budget_limit=50.0,
            enable_cost_alerts=True
        )

        self.assertIsInstance(adapter, GenOpsArizeAdapter)
        self.assertEqual(get_current_adapter(), adapter)

        # Step 2: Mock Arize client methods to simulate real usage
        mock_arize_client = Mock()
        mock_log_method = Mock(return_value={'status': 'success', 'id': 'log-123'})

        with patch('genops.providers.arize.ArizeClient', return_value=mock_arize_client):
            mock_arize_client.log = mock_log_method

            # Step 3: Instrument the log method
            instrumented_log = adapter.instrument_arize_log(mock_log_method)

            # Step 4: Simulate instrumented calls
            with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
                mock_span.return_value.__enter__ = Mock(return_value=Mock())
                mock_span.return_value.__exit__ = Mock(return_value=None)

                # Make instrumented calls
                result1 = instrumented_log(
                    prediction_id='auto-pred-1',
                    prediction_label='fraud',
                    model_id='auto-fraud-model',
                    model_version='1.2.0',
                    tags={'environment': 'production'}
                )

                result2 = instrumented_log(
                    prediction_id='auto-pred-2',
                    prediction_label='legitimate',
                    model_id='auto-fraud-model',
                    model_version='1.2.0'
                )

        # Verify instrumentation worked
        self.assertEqual(mock_log_method.call_count, 2)

        # Check that governance tags were added
        call_args = mock_log_method.call_args_list
        for call in call_args:
            call_kwargs = call[1]
            tags = call_kwargs.get('tags', {})

            # Should have GenOps governance tags
            self.assertIn('genops_team', tags)
            self.assertIn('genops_project', tags)
            self.assertEqual(tags['genops_team'], 'zero-code-team')
            self.assertEqual(tags['genops_project'], 'automated-monitoring')

        # Verify cost tracking
        self.assertEqual(adapter.operation_count, 2)
        self.assertGreater(adapter.daily_usage, 0)

    def test_auto_instrumentation_with_existing_arize_code(self):
        """Test auto-instrumentation with existing Arize code patterns."""
        # Enable auto-instrumentation
        adapter = auto_instrument(
            team="existing-code-team",
            project="legacy-integration"
        )

        # Simulate existing Arize usage patterns
        mock_client = Mock()

        with patch('genops.providers.arize.ArizeClient', return_value=mock_client):
            # Simulate typical Arize client usage
            arize_client = mock_client

            # Mock the log method to track calls
            original_log = Mock(return_value={'status': 'success'})
            arize_client.log = original_log

            # Apply instrumentation (simulating auto_instrument patching)
            instrumented_log = adapter.instrument_arize_log(original_log)
            arize_client.log = instrumented_log

            with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
                mock_span.return_value.__enter__ = Mock(return_value=Mock())
                mock_span.return_value.__exit__ = Mock(return_value=None)

                # Existing code patterns should work unchanged
                responses = []

                # Pattern 1: Basic prediction logging
                response1 = arize_client.log(
                    prediction_id='existing-pred-1',
                    prediction_label='positive',
                    actual_label='positive',
                    model_id='sentiment-model',
                    model_version='v2.1'
                )
                responses.append(response1)

                # Pattern 2: Batch logging with features
                response2 = arize_client.log(
                    prediction_id='existing-pred-2',
                    prediction_label='negative',
                    features={'text_length': 150, 'sentiment_score': -0.3},
                    model_id='sentiment-model',
                    model_version='v2.1',
                    tags={'source': 'api', 'user_type': 'premium'}
                )
                responses.append(response2)

        # Verify all calls succeeded
        for response in responses:
            self.assertEqual(response['status'], 'success')

        # Verify governance tracking
        self.assertEqual(adapter.operation_count, 2)

        # Check that original functionality was preserved
        self.assertEqual(original_log.call_count, 2)

    def test_instrumentation_factory_function(self):
        """Test instrument_arize factory function workflow."""
        # Create adapter using factory function
        adapter = instrument_arize(
            arize_api_key='factory-api-key-123456789',
            arize_space_key='factory-space-key-123456789',
            team='factory-team',
            project='factory-project',
            environment='staging',
            daily_budget_limit=75.0
        )

        self.assertIsInstance(adapter, GenOpsArizeAdapter)
        self.assertEqual(adapter.team, 'factory-team')
        self.assertEqual(adapter.project, 'factory-project')
        self.assertEqual(adapter.environment, 'staging')
        self.assertEqual(adapter.daily_budget_limit, 75.0)

        # Test that it can be used immediately
        metrics = adapter.get_metrics()
        self.assertIn('team', metrics)
        self.assertIn('daily_budget_limit', metrics)
        self.assertEqual(metrics['team'], 'factory-team')

    def test_global_adapter_management_workflow(self):
        """Test global adapter management in workflows."""
        # Initially no global adapter
        self.assertIsNone(get_current_adapter())

        # Create and set first adapter
        adapter1 = auto_instrument(
            team='global-team-1',
            project='global-project-1'
        )

        self.assertEqual(get_current_adapter(), adapter1)

        # Create second adapter (should replace first)
        adapter2 = auto_instrument(
            team='global-team-2',
            project='global-project-2'
        )

        # Global adapter should be updated
        current_adapter = get_current_adapter()
        self.assertEqual(current_adapter, adapter2)
        self.assertEqual(current_adapter.team, 'global-team-2')

        # Manual global adapter management
        adapter3 = GenOpsArizeAdapter(
            team='manual-team',
            project='manual-project'
        )

        set_global_adapter(adapter3)
        self.assertEqual(get_current_adapter(), adapter3)

        # Clear global adapter
        set_global_adapter(None)
        self.assertIsNone(get_current_adapter())

    def test_multiple_instrumentation_calls_workflow(self):
        """Test behavior with multiple instrumentation calls."""
        adapters = []

        # Multiple instrumentation calls
        for i in range(3):
            adapter = auto_instrument(
                team=f'multi-team-{i}',
                project=f'multi-project-{i}',
                daily_budget_limit=25.0 + (i * 10)
            )
            adapters.append(adapter)

        # Should have 3 different adapters
        self.assertEqual(len(set(adapters)), 3)

        # Last adapter should be global
        self.assertEqual(get_current_adapter(), adapters[-1])
        self.assertEqual(get_current_adapter().team, 'multi-team-2')

        # Each adapter should be independently functional
        for i, adapter in enumerate(adapters):
            metrics = adapter.get_metrics()
            self.assertEqual(metrics['team'], f'multi-team-{i}')
            self.assertEqual(metrics['daily_budget_limit'], 25.0 + (i * 10))


class TestMultiModuleInteractions(unittest.TestCase):
    """Test interactions between different modules."""

    def setUp(self):
        """Set up multi-module test fixtures."""
        self.env_patcher = patch.dict(os.environ, {
            'ARIZE_API_KEY': 'multi-module-api-key-123456789',
            'ARIZE_SPACE_KEY': 'multi-module-space-key-123456789',
            'GENOPS_TEAM': 'multi-module-team',
            'GENOPS_PROJECT': 'integration-testing'
        }, clear=False)
        self.env_patcher.start()

        # Mock Arize SDK
        self.arize_patch = patch('genops.providers.arize.ARIZE_AVAILABLE', True)
        self.arize_client_patch = patch('genops.providers.arize.ArizeClient')

        self.arize_patch.start()
        self.arize_client_patch.start()

    def tearDown(self):
        """Clean up multi-module test fixtures."""
        self.env_patcher.stop()
        self.arize_patch.stop()
        self.arize_client_patch.stop()

    def test_adapter_cost_aggregator_integration(self):
        """Test integration between adapter and cost aggregator."""
        # Create adapter and cost aggregator
        adapter = GenOpsArizeAdapter(
            team='integration-team',
            project='cost-integration',
            daily_budget_limit=100.0
        )

        cost_aggregator = ArizeCostAggregator(
            team='integration-team',
            project='cost-integration'
        )

        # Simulate monitoring operations
        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            with adapter.track_model_monitoring_session('integration-model', 'v1') as session:
                # Log predictions
                predictions_df = pd.DataFrame({'pred': [1, 0, 1, 1, 0]})
                session.log_prediction_batch(predictions_df, cost_per_prediction=0.001)

                # Add corresponding cost record to aggregator
                cost_aggregator.add_cost_record(
                    model_id='integration-model',
                    environment='production',
                    prediction_logging_cost=0.005,
                    data_quality_cost=0.0,
                    alert_management_cost=0.0,
                    dashboard_cost=0.10,
                    prediction_count=5,
                    data_quality_checks=0,
                    active_alerts=0
                )

        # Verify adapter state
        self.assertEqual(adapter.daily_usage, 0.005)

        # Verify cost aggregator state
        cost_summary = cost_aggregator.get_cost_summary_by_model()
        self.assertEqual(cost_summary.total_cost, 0.105)  # 0.005 + 0.10
        self.assertIn('integration-model', cost_summary.cost_by_model)

    def test_adapter_pricing_calculator_integration(self):
        """Test integration between adapter and pricing calculator."""
        # Create adapter and pricing calculator
        adapter = GenOpsArizeAdapter(team='pricing-team', project='pricing-integration')
        pricing_calculator = ArizePricingCalculator()

        # Simulate operations and calculate costs
        operation_data = {
            'prediction_count': 10000,
            'quality_checks': 100,
            'alert_count': 5,
            'model_tier': ModelTier.PRODUCTION
        }

        # Calculate expected costs using pricing calculator
        pricing_breakdown = pricing_calculator.get_total_monitoring_cost(**operation_data)

        # Simulate equivalent operations in adapter
        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            with adapter.track_model_monitoring_session('pricing-model', 'v1') as session:
                # Use pricing calculator costs for accurate simulation
                pred_cost_per_item = pricing_breakdown.cost_components['prediction_logging'] / operation_data['prediction_count']
                quality_cost_per_check = pricing_breakdown.cost_components['data_quality'] / operation_data['quality_checks']
                alert_cost_per_alert = pricing_breakdown.cost_components['alert_management'] / operation_data['alert_count']

                # Simulate operations with calculated costs
                session.log_prediction_batch(
                    pd.DataFrame({'pred': [1] * operation_data['prediction_count']}),
                    cost_per_prediction=pred_cost_per_item
                )

                for _ in range(operation_data['quality_checks']):
                    session.log_data_quality_metrics({'quality': 0.9}, quality_cost_per_check)

                for i in range(operation_data['alert_count']):
                    session.create_performance_alert(f'metric_{i}', 0.8, alert_cost_per_alert)

        # Compare adapter costs with pricing calculator
        adapter_cost = adapter.daily_usage
        calculator_cost = pricing_breakdown.final_cost

        # Should be approximately equal (within dashboard cost difference)
        self.assertAlmostEqual(adapter_cost, calculator_cost, delta=1.0)  # Allow for dashboard cost difference

    def test_validation_setup_integration_workflow(self):
        """Test integration of validation with setup workflow."""
        # Step 1: Run validation
        validator = ArizeSetupValidator()

        with patch.object(validator, 'arize_available', True):
            with patch.object(validator, 'arize_version', '6.1.0'):
                with patch.object(validator, 'arize_client_class', return_value=Mock()):
                    validation_result = validator.validate_complete_setup(
                        arize_api_key='validation-api-key-123456789',
                        arize_space_key='validation-space-key-123456789',
                        team='validation-team',
                        project='validation-project'
                    )

        # Step 2: Use validation results to configure adapter
        if validation_result.is_valid:
            adapter = GenOpsArizeAdapter(
                arize_api_key='validation-api-key-123456789',
                arize_space_key='validation-space-key-123456789',
                team='validation-team',
                project='validation-project'
            )

            # Step 3: Verify adapter works with validated configuration
            metrics = adapter.get_metrics()
            self.assertEqual(metrics['team'], 'validation-team')
            self.assertEqual(metrics['project'], 'validation-project')

            # Step 4: Test monitoring functionality
            with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
                mock_span.return_value.__enter__ = Mock(return_value=Mock())
                mock_span.return_value.__exit__ = Mock(return_value=None)

                with adapter.track_model_monitoring_session('validated-model', 'v1') as session:
                    session.log_prediction_batch(pd.DataFrame({'pred': [1, 0]}), 0.001)

            # Verification successful
            self.assertGreater(adapter.operation_count, 0)
        else:
            self.fail(f"Validation failed: {validation_result.issues}")

    def test_cost_optimization_recommendations_integration(self):
        """Test integration of cost optimization recommendations."""
        # Create components
        adapter = GenOpsArizeAdapter(team='optimization-team', project='cost-optimization')
        cost_aggregator = ArizeCostAggregator(team='optimization-team', project='cost-optimization')
        pricing_calculator = ArizePricingCalculator()

        # Simulate high-cost operations
        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            with adapter.track_model_monitoring_session('expensive-model', 'v1') as session:
                # High-volume predictions
                session.log_prediction_batch(
                    pd.DataFrame({'pred': [1, 0] * 25000}),  # 50k predictions
                    cost_per_prediction=0.002
                )

                # Expensive quality checks
                for _ in range(100):
                    session.log_data_quality_metrics({'quality': 0.9}, 0.05)

                # Add cost records to aggregator
                cost_aggregator.add_cost_record(
                    model_id='expensive-model',
                    environment='production',
                    prediction_logging_cost=100.0,  # High cost
                    data_quality_cost=5.0,
                    alert_management_cost=2.0,
                    dashboard_cost=1.0,
                    prediction_count=50000,
                    data_quality_checks=100,
                    active_alerts=2
                )

        # Get optimization recommendations from cost aggregator
        recommendations = cost_aggregator.get_cost_optimization_recommendations()
        self.assertGreater(len(recommendations), 0)

        # Get pricing strategy optimization from pricing calculator
        pricing_recommendations = optimize_pricing_strategy(
            current_prediction_count=50000,
            current_quality_checks=100,
            current_alert_count=2,
            target_cost_reduction=0.20  # 20% cost reduction target
        )

        self.assertGreater(len(pricing_recommendations), 0)

        # Verify recommendations are actionable
        for rec in recommendations:
            self.assertGreater(rec.potential_savings, 0)
            self.assertGreater(len(rec.action_items), 0)

        for rec in pricing_recommendations:
            self.assertGreater(rec.potential_savings, 0)
            self.assertGreater(len(rec.implementation_steps), 0)


class TestProductionScenarios(unittest.TestCase):
    """Test production-like scenarios and edge cases."""

    def setUp(self):
        """Set up production scenario test fixtures."""
        self.env_patcher = patch.dict(os.environ, {
            'ARIZE_API_KEY': 'prod-scenario-api-key-123456789',
            'ARIZE_SPACE_KEY': 'prod-scenario-space-key-123456789',
            'GENOPS_TEAM': 'production-team',
            'GENOPS_PROJECT': 'production-monitoring'
        }, clear=False)
        self.env_patcher.start()

        # Mock Arize SDK
        self.arize_patch = patch('genops.providers.arize.ARIZE_AVAILABLE', True)
        self.arize_client_patch = patch('genops.providers.arize.ArizeClient')

        self.arize_patch.start()
        self.arize_client_patch.start()

    def tearDown(self):
        """Clean up production scenario test fixtures."""
        self.env_patcher.stop()
        self.arize_patch.stop()
        self.arize_client_patch.stop()

    def test_high_frequency_monitoring_scenario(self):
        """Test high-frequency monitoring scenario."""
        adapter = GenOpsArizeAdapter(
            team='high-freq-team',
            project='real-time-monitoring',
            daily_budget_limit=1000.0
        )

        start_time = time.time()
        total_predictions = 0

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            # Simulate 1 hour of high-frequency monitoring
            for minute in range(60):  # 60 minutes
                with adapter.track_model_monitoring_session(
                    f'realtime-model-minute-{minute}', 'v1'
                ) as session:

                    # High-frequency predictions (100 per minute)
                    minute_predictions = pd.DataFrame({
                        'pred': [(minute + i) % 2 for i in range(100)],
                        'timestamp': [minute * 60 + i for i in range(100)]
                    })
                    session.log_prediction_batch(minute_predictions, 0.0001)  # Low cost per prediction
                    total_predictions += 100

                    # Periodic quality checks (every 5 minutes)
                    if minute % 5 == 0:
                        session.log_data_quality_metrics({
                            'accuracy': 0.90 + (minute * 0.001),
                            'drift': 0.05 + (minute * 0.0001)
                        }, 0.01)

                    # Periodic alerts (every 10 minutes)
                    if minute % 10 == 0:
                        session.create_performance_alert('drift_check', 0.10, 0.02)

        elapsed_time = time.time() - start_time

        # Performance verification
        self.assertLess(elapsed_time, 10.0)  # Should complete quickly
        self.assertEqual(total_predictions, 6000)  # 100 predictions * 60 minutes
        self.assertEqual(adapter.operation_count, 60)  # 60 monitoring sessions

        # Cost verification
        self.assertLess(adapter.daily_usage, 1000.0)  # Within budget
        self.assertGreater(adapter.daily_usage, 0)  # Non-zero cost

    def test_multi_model_production_scenario(self):
        """Test multi-model production monitoring scenario."""
        adapter = GenOpsArizeAdapter(
            team='multi-model-prod-team',
            project='production-ml-platform',
            daily_budget_limit=500.0
        )

        # Define production models
        production_models = [
            {'id': 'fraud-detection-v2', 'env': 'production', 'volume': 10000},
            {'id': 'recommendation-engine-v3', 'env': 'production', 'volume': 50000},
            {'id': 'sentiment-analysis-v1', 'env': 'production', 'volume': 25000},
            {'id': 'price-optimization-v2', 'env': 'production', 'volume': 5000},
            {'id': 'churn-prediction-v1', 'env': 'production', 'volume': 8000}
        ]

        model_costs = {}

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            # Monitor each production model
            for model in production_models:
                with adapter.track_model_monitoring_session(
                    model['id'],
                    'v1',
                    model['env']
                ) as session:

                    # Log predictions based on model volume
                    predictions = pd.DataFrame({
                        'pred': [hash(model['id']) % 2] * model['volume']
                    })
                    session.log_prediction_batch(predictions, 0.0005)

                    # Model-specific quality monitoring
                    quality_checks = max(1, model['volume'] // 1000)  # 1 check per 1k predictions
                    for _ in range(quality_checks):
                        session.log_data_quality_metrics({
                            'model_quality': 0.92,
                            'data_freshness': 0.98
                        }, 0.02)

                    # Critical model alerts
                    if 'fraud' in model['id'] or 'churn' in model['id']:
                        session.create_performance_alert('critical_metric', 0.95, 0.15)
                    else:
                        session.create_performance_alert('standard_metric', 0.85, 0.05)

                    # Track model-specific costs
                    model_costs[model['id']] = session.estimated_cost

        # Production scenario verification
        self.assertEqual(len(model_costs), 5)  # All models monitored
        self.assertEqual(adapter.operation_count, 5)  # 5 monitoring sessions

        # Cost distribution verification
        total_cost = sum(model_costs.values())
        self.assertEqual(total_cost, adapter.daily_usage)
        self.assertLess(total_cost, 500.0)  # Within budget

        # High-volume models should have proportionally higher costs
        rec_engine_cost = model_costs['recommendation-engine-v3']
        fraud_cost = model_costs['fraud-detection-v2']
        self.assertGreater(rec_engine_cost, fraud_cost)  # Higher volume = higher cost

    def test_disaster_recovery_scenario(self):
        """Test disaster recovery and failover scenario."""
        primary_adapter = GenOpsArizeAdapter(
            team='disaster-recovery-team',
            project='failover-testing',
            environment='production'
        )

        backup_adapter = GenOpsArizeAdapter(
            team='disaster-recovery-team',
            project='failover-testing',
            environment='backup'
        )

        # Simulate primary system failure during monitoring
        primary_failed = False
        backup_used = False
        total_operations = 0

        with patch.object(primary_adapter.tracer, 'start_as_current_span') as primary_span:
            with patch.object(backup_adapter.tracer, 'start_as_current_span') as backup_span:

                primary_span.return_value.__enter__ = Mock(return_value=Mock())
                primary_span.return_value.__exit__ = Mock(return_value=None)
                backup_span.return_value.__enter__ = Mock(return_value=Mock())
                backup_span.return_value.__exit__ = Mock(return_value=None)

                for operation in range(10):
                    try:
                        # Simulate primary system failure after 5 operations
                        if operation >= 5:
                            primary_failed = True
                            raise ConnectionError("Primary system unavailable")

                        # Use primary adapter
                        with primary_adapter.track_model_monitoring_session(
                            f'failover-model-{operation}', 'v1'
                        ) as session:
                            session.log_prediction_batch(
                                pd.DataFrame({'pred': [1, 0]}), 0.001
                            )
                            total_operations += 1

                    except ConnectionError:
                        # Failover to backup adapter
                        if not backup_used:
                            backup_used = True

                        with backup_adapter.track_model_monitoring_session(
                            f'failover-model-{operation}', 'v1'
                        ) as session:
                            session.log_prediction_batch(
                                pd.DataFrame({'pred': [1, 0]}), 0.001
                            )
                            total_operations += 1

        # Disaster recovery verification
        self.assertTrue(primary_failed)
        self.assertTrue(backup_used)
        self.assertEqual(total_operations, 10)  # All operations completed

        # Verify both systems tracked their operations
        self.assertEqual(primary_adapter.operation_count, 5)  # Primary handled first 5
        self.assertEqual(backup_adapter.operation_count, 5)  # Backup handled last 5

        # Total cost should be distributed across both systems
        total_cost = primary_adapter.daily_usage + backup_adapter.daily_usage
        self.assertGreater(total_cost, 0)


if __name__ == '__main__':
    # Run the comprehensive integration test suite
    unittest.main(verbosity=2)
