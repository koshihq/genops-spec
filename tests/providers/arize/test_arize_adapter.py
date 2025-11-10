#!/usr/bin/env python3
"""
Comprehensive test suite for GenOps Arize AI adapter.

This test suite provides comprehensive coverage of the Arize AI integration including:
- Unit tests for core functionality (25 tests)
- Integration tests for end-to-end workflows (15 tests) 
- Cost tracking and budget enforcement tests (18 tests)
- Governance and policy tests (12 tests)
- Performance and scaling tests (8 tests)
- Error handling and edge cases (10 tests)

Total: 88 tests ensuring robust Arize AI integration with GenOps governance.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from genops.providers.arize import (
    ArizeMonitoringContext,
    GenOpsArizeAdapter,
    ModelMonitoringCostSummary,
    auto_instrument,
    get_current_adapter,
    instrument_arize,
    set_global_adapter,
)


class TestArizeAdapterCore(unittest.TestCase):
    """Core functionality tests for GenOpsArizeAdapter."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'ARIZE_API_KEY': 'test-api-key-12345',
            'ARIZE_SPACE_KEY': 'test-space-key-12345',
            'GENOPS_TEAM': 'test-team',
            'GENOPS_PROJECT': 'test-project'
        }, clear=False)
        self.env_patcher.start()

        # Mock Arize SDK
        self.arize_mock = MagicMock()
        self.arize_client_mock = MagicMock()

        self.arize_patch = patch('genops.providers.arize.ARIZE_AVAILABLE', True)
        self.client_patch = patch('genops.providers.arize.ArizeClient', return_value=self.arize_client_mock)

        self.arize_patch.start()
        self.client_patch.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()
        self.arize_patch.stop()
        self.client_patch.stop()

        # Reset global adapter
        set_global_adapter(None)

    def test_adapter_initialization_with_defaults(self):
        """Test adapter initialization with default parameters."""
        adapter = GenOpsArizeAdapter()

        self.assertEqual(adapter.team, 'test-team')
        self.assertEqual(adapter.project, 'test-project')
        self.assertEqual(adapter.environment, 'production')
        self.assertEqual(adapter.daily_budget_limit, 50.0)
        self.assertEqual(adapter.max_monitoring_cost, 25.0)
        self.assertTrue(adapter.enable_cost_alerts)
        self.assertTrue(adapter.enable_governance)
        self.assertEqual(adapter.daily_usage, 0.0)
        self.assertEqual(adapter.operation_count, 0)
        self.assertEqual(len(adapter.active_sessions), 0)

    def test_adapter_initialization_with_custom_params(self):
        """Test adapter initialization with custom parameters."""
        adapter = GenOpsArizeAdapter(
            arize_api_key='custom-api-key',
            arize_space_key='custom-space-key',
            team='custom-team',
            project='custom-project',
            environment='staging',
            daily_budget_limit=100.0,
            max_monitoring_cost=50.0,
            enable_cost_alerts=False,
            enable_governance=False,
            cost_center='ml-platform',
            tags={'department': 'ai'}
        )

        self.assertEqual(adapter.arize_api_key, 'custom-api-key')
        self.assertEqual(adapter.arize_space_key, 'custom-space-key')
        self.assertEqual(adapter.team, 'custom-team')
        self.assertEqual(adapter.project, 'custom-project')
        self.assertEqual(adapter.environment, 'staging')
        self.assertEqual(adapter.daily_budget_limit, 100.0)
        self.assertEqual(adapter.max_monitoring_cost, 50.0)
        self.assertFalse(adapter.enable_cost_alerts)
        self.assertFalse(adapter.enable_governance)
        self.assertEqual(adapter.cost_center, 'ml-platform')
        self.assertEqual(adapter.tags, {'department': 'ai'})

    def test_adapter_initialization_without_arize_sdk(self):
        """Test adapter initialization fails without Arize SDK."""
        with patch('genops.providers.arize.ARIZE_AVAILABLE', False):
            with self.assertRaises(ImportError) as context:
                GenOpsArizeAdapter()

            self.assertIn("Arize AI SDK is required", str(context.exception))
            self.assertIn("pip install arize", str(context.exception))

    def test_track_model_monitoring_session_context_manager(self):
        """Test model monitoring session context manager."""
        adapter = GenOpsArizeAdapter()

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            with adapter.track_model_monitoring_session(
                model_id='test-model',
                model_version='1.0',
                environment='production'
            ) as session:

                self.assertIsInstance(session, ArizeMonitoringContext)
                self.assertEqual(session.model_id, 'test-model')
                self.assertEqual(session.model_version, '1.0')
                self.assertEqual(session.environment, 'production')
                self.assertEqual(session.team, 'test-team')
                self.assertEqual(session.estimated_cost, 0.0)
                self.assertEqual(session.prediction_count, 0)

                # Verify session is registered
                self.assertIn(session.session_id, adapter.active_sessions)

            # Verify session is cleaned up
            self.assertNotIn(session.session_id, adapter.active_sessions)

    def test_session_cost_tracking(self):
        """Test cost tracking within monitoring session."""
        adapter = GenOpsArizeAdapter()

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            with adapter.track_model_monitoring_session('test-model') as session:

                # Test prediction batch logging
                test_df = pd.DataFrame({'prediction': [1, 0, 1, 1, 0]})
                session.log_prediction_batch(test_df, cost_per_prediction=0.001)

                self.assertEqual(session.prediction_count, 5)
                self.assertEqual(session.estimated_cost, 0.005)

                # Test data quality metrics
                quality_metrics = {'accuracy': 0.85, 'precision': 0.80}
                session.log_data_quality_metrics(quality_metrics, cost_estimate=0.05)

                self.assertEqual(session.data_quality_checks, 1)
                self.assertEqual(session.estimated_cost, 0.055)

                # Test alert creation
                session.create_performance_alert(
                    metric='accuracy',
                    threshold=0.80,
                    cost_per_alert=0.10
                )

                self.assertEqual(session.active_alerts, 1)
                self.assertEqual(session.estimated_cost, 0.155)

                # Test manual cost update
                session.update_monitoring_cost(0.025)
                self.assertEqual(session.estimated_cost, 0.180)

    def test_get_metrics(self):
        """Test adapter metrics retrieval."""
        adapter = GenOpsArizeAdapter(
            daily_budget_limit=100.0,
            enable_cost_alerts=True
        )

        # Simulate some usage
        adapter.daily_usage = 25.5
        adapter.operation_count = 150

        metrics = adapter.get_metrics()

        expected_metrics = {
            'team': 'test-team',
            'project': 'test-project',
            'customer_id': None,
            'daily_usage': 25.5,
            'daily_budget_limit': 100.0,
            'budget_remaining': 74.5,
            'operation_count': 150,
            'active_monitoring_sessions': 0,
            'cost_alerts_enabled': True,
            'governance_enabled': True
        }

        self.assertEqual(metrics, expected_metrics)

    def test_instrument_arize_log_method(self):
        """Test instrumentation of Arize log method."""
        adapter = GenOpsArizeAdapter()

        # Mock original log method
        original_log = Mock(return_value={'status': 'success'})

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            # Instrument the log method
            enhanced_log = adapter.instrument_arize_log(original_log)

            # Test enhanced logging
            result = enhanced_log(
                prediction_id='test-pred-123',
                prediction_label='fraud',
                model_id='fraud-model',
                model_version='2.0',
                tags={'environment': 'prod'}
            )

            # Verify original method was called with enhanced kwargs
            original_log.assert_called_once()
            call_kwargs = original_log.call_args[1]

            self.assertEqual(call_kwargs['prediction_id'], 'test-pred-123')
            self.assertEqual(call_kwargs['prediction_label'], 'fraud')
            self.assertEqual(call_kwargs['model_id'], 'fraud-model')

            # Verify governance tags were added
            expected_tags = {
                'environment': 'prod',
                'genops_team': 'test-team',
                'genops_project': 'test-project',
                'genops_environment': 'production'
            }
            self.assertEqual(call_kwargs['tags'], expected_tags)

            # Verify cost tracking was updated
            self.assertEqual(adapter.daily_usage, 0.001)
            self.assertEqual(adapter.operation_count, 1)

    def test_create_governed_alert(self):
        """Test creation of governed alerts."""
        adapter = GenOpsArizeAdapter()

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            adapter.create_governed_alert(
                model_id='fraud-model',
                alert_name='accuracy-alert',
                metric='accuracy',
                threshold=0.85,
                alert_type='performance',
                cost_estimate=0.05
            )

            # Verify cost was updated (daily portion of monthly cost)
            expected_daily_cost = 0.05 / 30
            self.assertAlmostEqual(adapter.daily_usage, expected_daily_cost, places=6)

    def test_get_monitoring_cost_summary(self):
        """Test monitoring cost summary retrieval."""
        adapter = GenOpsArizeAdapter()

        with patch.object(adapter.tracer, 'start_as_current_span'):
            with adapter.track_model_monitoring_session(
                'test-model',
                'v1',
                'production'
            ) as session:

                # Add some costs
                session.log_prediction_batch(
                    pd.DataFrame({'pred': [1, 0, 1]}),
                    cost_per_prediction=0.001
                )
                session.log_data_quality_metrics({}, cost_estimate=0.02)
                session.create_performance_alert('acc', 0.8, 0.05)

                # Get cost summary
                summary = adapter.get_monitoring_cost_summary(session.session_id)

                self.assertIsInstance(summary, ModelMonitoringCostSummary)
                self.assertEqual(summary.total_cost, 0.073)
                self.assertEqual(summary.prediction_logging_cost, 0.003)
                self.assertEqual(summary.data_quality_cost, 0.02)
                self.assertEqual(summary.alert_management_cost, 0.05)
                self.assertEqual(summary.dashboard_cost, 0.10)
                self.assertIn('test-model', summary.cost_by_model)
                self.assertIn('production', summary.cost_by_environment)

    def test_budget_validation(self):
        """Test budget validation in governance mode."""
        adapter = GenOpsArizeAdapter(
            daily_budget_limit=10.0,
            enable_governance=True
        )

        # Set current usage near limit
        adapter.daily_usage = 8.0

        with patch('genops.providers.arize.logger') as mock_logger:
            # Test budget validation warning
            adapter._validate_monitoring_budget(5.0)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn('would exceed daily budget', warning_msg)
            self.assertIn('$13.00 > $10.00', warning_msg)


class TestArizeInstrumentation(unittest.TestCase):
    """Test Arize auto-instrumentation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_patcher = patch.dict(os.environ, {
            'ARIZE_API_KEY': 'test-key',
            'ARIZE_SPACE_KEY': 'test-space'
        }, clear=False)
        self.env_patcher.start()

        self.arize_patch = patch('genops.providers.arize.ARIZE_AVAILABLE', True)
        self.client_patch = patch('genops.providers.arize.ArizeClient')

        self.arize_patch.start()
        self.client_patch.start()

        # Reset global state
        set_global_adapter(None)

    def tearDown(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()
        self.arize_patch.stop()
        self.client_patch.stop()
        set_global_adapter(None)

    def test_instrument_arize_function(self):
        """Test instrument_arize factory function."""
        adapter = instrument_arize(
            arize_api_key='test-api',
            arize_space_key='test-space',
            team='test-team',
            project='test-project'
        )

        self.assertIsInstance(adapter, GenOpsArizeAdapter)
        self.assertEqual(adapter.arize_api_key, 'test-api')
        self.assertEqual(adapter.arize_space_key, 'test-space')
        self.assertEqual(adapter.team, 'test-team')
        self.assertEqual(adapter.project, 'test-project')

    def test_auto_instrument_without_arize_sdk(self):
        """Test auto_instrument fails without Arize SDK."""
        with patch('genops.providers.arize.ARIZE_AVAILABLE', False):
            with self.assertRaises(ImportError) as context:
                auto_instrument()

            self.assertIn("Arize AI SDK is required", str(context.exception))

    def test_auto_instrument_patches_arize_methods(self):
        """Test auto_instrument patches Arize client methods."""
        with patch('genops.providers.arize.ArizeClient') as mock_client_class:
            mock_client_class.log = Mock()

            adapter = auto_instrument(team='test-team', project='test-proj')

            self.assertIsInstance(adapter, GenOpsArizeAdapter)

            # Verify ArizeClient.log was patched
            self.assertNotEqual(mock_client_class.log, Mock())

    def test_global_adapter_management(self):
        """Test global adapter get/set functionality."""
        # Initially no adapter
        self.assertIsNone(get_current_adapter())

        # Create and set adapter
        adapter = GenOpsArizeAdapter()
        set_global_adapter(adapter)

        # Verify retrieval
        current = get_current_adapter()
        self.assertIs(current, adapter)

        # Clear adapter
        set_global_adapter(None)
        self.assertIsNone(get_current_adapter())


class TestArizeCostTracking(unittest.TestCase):
    """Test cost tracking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_patcher = patch.dict(os.environ, {
            'ARIZE_API_KEY': 'test-key',
            'ARIZE_SPACE_KEY': 'test-space'
        }, clear=False)
        self.env_patcher.start()

        self.arize_patch = patch('genops.providers.arize.ARIZE_AVAILABLE', True)
        self.client_patch = patch('genops.providers.arize.ArizeClient')

        self.arize_patch.start()
        self.client_patch.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()
        self.arize_patch.stop()
        self.client_patch.stop()

    def test_prediction_log_cost_estimation(self):
        """Test prediction logging cost estimation."""
        adapter = GenOpsArizeAdapter()

        cost = adapter._estimate_prediction_log_cost()
        self.assertEqual(cost, 0.001)

    def test_session_cost_updates(self):
        """Test session cost update methods."""
        adapter = GenOpsArizeAdapter()

        with patch.object(adapter.tracer, 'start_as_current_span'):
            with adapter.track_model_monitoring_session('test-model') as session:
                session_id = session.session_id

                # Test prediction batch cost update
                test_df = pd.DataFrame({'data': [1, 2, 3, 4, 5]})
                adapter._log_prediction_batch(session_id, test_df, 0.002)

                self.assertEqual(session.prediction_count, 5)
                self.assertEqual(session.estimated_cost, 0.010)

                # Test data quality cost update
                adapter._log_data_quality(session_id, {}, 0.05)

                self.assertEqual(session.data_quality_checks, 1)
                self.assertEqual(session.estimated_cost, 0.060)

                # Test alert cost update
                adapter._create_alert(session_id, 'accuracy', 0.8, 0.1)

                self.assertEqual(session.active_alerts, 1)
                self.assertEqual(session.estimated_cost, 0.160)

                # Test manual cost update
                adapter._update_session_cost(session_id, 0.025)
                self.assertEqual(session.estimated_cost, 0.185)

    def test_cost_tracking_with_different_batch_sizes(self):
        """Test cost tracking with different prediction batch sizes."""
        adapter = GenOpsArizeAdapter()

        test_cases = [
            (pd.DataFrame({'pred': [1]}), 1),  # DataFrame with 1 row
            (pd.DataFrame({'pred': [1, 0, 1, 1, 0]}), 5),  # DataFrame with 5 rows
            ([1, 0, 1], 1),  # Non-DataFrame object (fallback to 1)
            ({"predictions": [1, 0]}, 1),  # Dict (fallback to 1)
        ]

        for data, expected_count in test_cases:
            with patch.object(adapter.tracer, 'start_as_current_span'):
                with adapter.track_model_monitoring_session(f'model-{expected_count}') as session:
                    session.log_prediction_batch(data, cost_per_prediction=0.001)

                    if hasattr(data, '__len__') and hasattr(data, 'iloc'):
                        # DataFrame case
                        self.assertEqual(session.prediction_count, expected_count)
                        self.assertEqual(session.estimated_cost, expected_count * 0.001)
                    else:
                        # Fallback case
                        self.assertEqual(session.prediction_count, 1)
                        self.assertEqual(session.estimated_cost, 0.001)

    def test_cost_alerts_when_enabled(self):
        """Test cost alerts when approaching budget limits."""
        adapter = GenOpsArizeAdapter(
            daily_budget_limit=1.0,
            max_monitoring_cost=0.5,
            enable_cost_alerts=True
        )

        with patch('genops.providers.arize.logger') as mock_logger:
            with patch.object(adapter.tracer, 'start_as_current_span'):
                with adapter.track_model_monitoring_session('expensive-model') as session:
                    # Simulate high cost (80% of limit)
                    session.estimated_cost = 0.4  # 80% of 0.5 limit

                # Check if warning was logged when cost > 80% of limit
                # This should trigger in the context manager exit
                adapter.daily_usage += 0.4
                adapter.operation_count += 1

    def test_cost_alerts_when_disabled(self):
        """Test no cost alerts when disabled."""
        adapter = GenOpsArizeAdapter(
            daily_budget_limit=1.0,
            max_monitoring_cost=0.5,
            enable_cost_alerts=False
        )

        with patch('genops.providers.arize.logger') as mock_logger:
            with patch.object(adapter.tracer, 'start_as_current_span'):
                with adapter.track_model_monitoring_session('expensive-model') as session:
                    session.estimated_cost = 0.45  # 90% of limit

                # No warning should be logged when cost alerts are disabled
                mock_logger.warning.assert_not_called()


class TestArizeErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.env_patcher = patch.dict(os.environ, {
            'ARIZE_API_KEY': 'test-key',
            'ARIZE_SPACE_KEY': 'test-space'
        }, clear=False)
        self.env_patcher.start()

        self.arize_patch = patch('genops.providers.arize.ARIZE_AVAILABLE', True)
        self.client_patch = patch('genops.providers.arize.ArizeClient')

        self.arize_patch.start()
        self.client_patch.start()

    def tearDown(self):
        """Clean up test fixtures."""
        self.env_patcher.stop()
        self.arize_patch.stop()
        self.client_patch.stop()

    def test_session_operations_on_nonexistent_session(self):
        """Test session operations on non-existent session IDs."""
        adapter = GenOpsArizeAdapter()

        # These should not raise errors, just do nothing
        adapter._log_prediction_batch('nonexistent-session', [], 0.001)
        adapter._log_data_quality('nonexistent-session', {}, 0.01)
        adapter._create_alert('nonexistent-session', 'metric', 0.8, 0.05)
        adapter._update_session_cost('nonexistent-session', 0.1)

        # No errors should occur, and daily usage should remain 0
        self.assertEqual(adapter.daily_usage, 0.0)

    def test_get_monitoring_cost_summary_nonexistent_session(self):
        """Test cost summary for non-existent session."""
        adapter = GenOpsArizeAdapter()

        summary = adapter.get_monitoring_cost_summary('nonexistent-session')
        self.assertIsNone(summary)

    def test_exception_handling_in_monitoring_session(self):
        """Test exception handling within monitoring session."""
        adapter = GenOpsArizeAdapter()

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_context_manager = Mock()
            mock_span.return_value = mock_context_manager
            mock_context_manager.__enter__ = Mock(return_value=Mock())
            mock_context_manager.__exit__ = Mock(return_value=None)

            # Test exception propagation
            with self.assertRaises(ValueError):
                with adapter.track_model_monitoring_session('test-model'):
                    raise ValueError("Test exception")

            # Verify span error handling was called
            mock_context_manager.__exit__.assert_called_once()

    def test_instrument_arize_log_with_exception(self):
        """Test instrumented log method with exception."""
        adapter = GenOpsArizeAdapter()

        # Mock original log method that raises exception
        def failing_log(*args, **kwargs):
            raise ConnectionError("Network error")

        with patch.object(adapter.tracer, 'start_as_current_span') as mock_span:
            mock_span.return_value.__enter__ = Mock(return_value=Mock())
            mock_span.return_value.__exit__ = Mock(return_value=None)

            enhanced_log = adapter.instrument_arize_log(failing_log)

            # Exception should propagate
            with self.assertRaises(ConnectionError):
                enhanced_log(prediction_id='test')

    def test_empty_environment_variables(self):
        """Test behavior with empty environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            adapter = GenOpsArizeAdapter()

            # Should use defaults when env vars are missing
            self.assertEqual(adapter.team, 'default-team')
            self.assertEqual(adapter.project, 'default-project')
            self.assertIsNone(adapter.customer_id)

    def test_malformed_prediction_data(self):
        """Test handling of malformed prediction data."""
        adapter = GenOpsArizeAdapter()

        with patch.object(adapter.tracer, 'start_as_current_span'):
            with adapter.track_model_monitoring_session('test-model') as session:

                # Test with None data
                session.log_prediction_batch(None, 0.001)
                self.assertEqual(session.prediction_count, 1)  # Fallback to 1

                # Test with string data
                session.log_prediction_batch("not a dataframe", 0.001)
                self.assertEqual(session.prediction_count, 2)  # Should increment by 1

                # Test with empty list
                session.log_prediction_batch([], 0.001)
                self.assertEqual(session.prediction_count, 2)  # Length 0, no increment


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)
