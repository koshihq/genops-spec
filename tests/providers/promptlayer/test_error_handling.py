"""
Tests for PromptLayer error handling and resilience.

Tests error scenarios, recovery patterns, graceful degradation,
and robustness under various failure conditions.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

try:
    from genops.providers.promptlayer import (
        GenOpsPromptLayerAdapter,
        EnhancedPromptLayerSpan,
        GovernancePolicy,
        MockPromptLayer
    )
    PROMPTLAYER_AVAILABLE = True
except ImportError:
    PROMPTLAYER_AVAILABLE = False


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestAPIErrorHandling:
    """Test handling of PromptLayer API errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            self.mock_client = Mock()
            mock_pl.return_value = self.mock_client
            
            self.adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-error-test',
                team='error-test-team'
            )
    
    def test_api_connection_error(self):
        """Test handling of API connection errors."""
        self.mock_client.run.side_effect = ConnectionError("Failed to connect to PromptLayer API")
        
        with pytest.raises(ConnectionError):
            with self.adapter.track_prompt_operation('connection_error_test') as span:
                self.adapter.run_prompt_with_governance(
                    prompt_name='connection_error_test',
                    input_variables={'test': 'connection_error'}
                )
        
        # Span should still be properly finalized
        assert span.end_time is not None
        assert 'error' in span.metadata
        assert span.metadata['error_type'] == 'ConnectionError'
    
    def test_api_authentication_error(self):
        """Test handling of authentication errors."""
        self.mock_client.run.side_effect = Exception("Invalid API key")
        
        with pytest.raises(Exception, match="Invalid API key"):
            with self.adapter.track_prompt_operation('auth_error_test') as span:
                self.adapter.run_prompt_with_governance(
                    prompt_name='auth_error_test',
                    input_variables={'test': 'auth_error'}
                )
        
        # Error should be captured in span metadata
        assert 'Invalid API key' in span.metadata.get('error', '')
    
    def test_api_rate_limit_error(self):
        """Test handling of rate limit errors."""
        self.mock_client.run.side_effect = Exception("Rate limit exceeded")
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            with self.adapter.track_prompt_operation('rate_limit_test') as span:
                self.adapter.run_prompt_with_governance(
                    prompt_name='rate_limit_test',
                    input_variables={'test': 'rate_limit'}
                )
    
    def test_api_timeout_error(self):
        """Test handling of API timeout errors."""
        self.mock_client.run.side_effect = TimeoutError("Request timed out")
        
        with pytest.raises(TimeoutError):
            with self.adapter.track_prompt_operation('timeout_test') as span:
                self.adapter.run_prompt_with_governance(
                    prompt_name='timeout_test',
                    input_variables={'test': 'timeout'}
                )
    
    def test_malformed_api_response(self):
        """Test handling of malformed API responses."""
        # Return invalid response format
        self.mock_client.run.return_value = "invalid response format"
        
        with self.adapter.track_prompt_operation('malformed_test') as span:
            result = self.adapter.run_prompt_with_governance(
                prompt_name='malformed_test',
                input_variables={'test': 'malformed'}
            )
        
        # Should handle gracefully
        assert 'governance' in result
        assert result['governance']['team'] == 'error-test-team'


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestGracefulDegradation:
    """Test graceful degradation when dependencies are unavailable."""
    
    def test_promptlayer_sdk_unavailable(self):
        """Test graceful degradation when PromptLayer SDK is not available."""
        with patch('genops.providers.promptlayer.HAS_PROMPTLAYER', False):
            # Should use MockPromptLayer
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-unavailable-test',
                team='degradation-test-team'
            )
            
            assert isinstance(adapter.client, MockPromptLayer)
            
            # Operations should still work with mock
            with adapter.track_prompt_operation('degradation_test') as span:
                result = adapter.run_prompt_with_governance(
                    prompt_name='degradation_test',
                    input_variables={'test': 'degradation'}
                )
                span.update_cost(0.01)
            
            # Should return mock response with governance
            assert 'governance' in result
            assert result['response']['mock'] is True
            assert adapter.daily_usage == 0.01
    
    def test_missing_api_key_graceful_handling(self):
        """Test graceful handling when API key is missing."""
        with patch('genops.providers.promptlayer.PromptLayer', side_effect=Exception("Missing API key")):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key=None,  # No API key
                team='no-key-test-team'
            )
            
            # Should fall back to MockPromptLayer
            assert isinstance(adapter.client, MockPromptLayer)
    
    def test_network_unavailable_handling(self):
        """Test handling when network is unavailable."""
        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            mock_client = Mock()
            mock_pl.return_value = mock_client
            mock_client.run.side_effect = OSError("Network is unreachable")
            
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-network-test',
                team='network-test-team'
            )
            
            with pytest.raises(OSError):
                with adapter.track_prompt_operation('network_test') as span:
                    adapter.run_prompt_with_governance(
                        prompt_name='network_test',
                        input_variables={'test': 'network'}
                    )
            
            # Governance tracking should still work
            assert span.team == 'network-test-team'
            assert span.end_time is not None


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestContextManagerErrorHandling:
    """Test error handling in context managers."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            self.adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-context-error-test',
                team='context-error-team'
            )
    
    def test_exception_in_context_manager(self):
        """Test exception handling within context manager."""
        with pytest.raises(ValueError, match="Test exception"):
            with self.adapter.track_prompt_operation('context_exception_test') as span:
                span.update_cost(0.05)
                raise ValueError("Test exception")
        
        # Span should be finalized even with exception
        assert span.end_time is not None
        assert 'Test exception' in span.metadata.get('error', '')
        assert span.metadata.get('error_type') == 'ValueError'
        
        # Usage should still be tracked
        assert self.adapter.daily_usage == 0.05
        assert self.adapter.operation_count == 1
    
    def test_keyboard_interrupt_handling(self):
        """Test handling of KeyboardInterrupt."""
        with pytest.raises(KeyboardInterrupt):
            with self.adapter.track_prompt_operation('keyboard_interrupt_test') as span:
                span.update_cost(0.02)
                raise KeyboardInterrupt()
        
        # Should still track usage before interruption
        assert self.adapter.daily_usage == 0.02
    
    def test_system_exit_handling(self):
        """Test handling of SystemExit."""
        with pytest.raises(SystemExit):
            with self.adapter.track_prompt_operation('system_exit_test') as span:
                span.update_cost(0.03)
                raise SystemExit(1)
        
        # Should still track usage
        assert self.adapter.daily_usage == 0.03
    
    def test_nested_context_manager_errors(self):
        """Test error handling in nested context managers."""
        outer_completed = False
        inner_completed = False
        
        try:
            with self.adapter.track_prompt_operation('outer_context') as outer_span:
                outer_span.update_cost(0.01)
                
                try:
                    with self.adapter.track_prompt_operation('inner_context') as inner_span:
                        inner_span.update_cost(0.02)
                        raise RuntimeError("Inner context error")
                        inner_completed = True
                except RuntimeError:
                    pass
                
                outer_completed = True
                
        except Exception:
            pass
        
        # Both spans should be properly finalized
        assert outer_span.end_time is not None
        assert inner_span.end_time is not None
        assert not inner_completed
        assert outer_completed
        
        # Both costs should be tracked
        assert self.adapter.daily_usage == 0.03
        assert self.adapter.operation_count == 2


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestGovernanceErrorHandling:
    """Test error handling in governance scenarios."""
    
    def test_budget_violation_with_enforced_policy(self):
        """Test budget violation handling with enforced policy."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-governance-error',
                governance_policy=GovernancePolicy.ENFORCED,
                daily_budget_limit=0.05
            )
            
            # Exceed budget in enforced mode
            adapter.daily_usage = 0.06  # Already over budget
            
            with pytest.raises(ValueError, match="Daily budget limit"):
                with adapter.track_prompt_operation('budget_violation_test') as span:
                    pass
    
    def test_invalid_governance_configuration(self):
        """Test handling of invalid governance configuration."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            # Test with negative budget limit
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-invalid-config',
                daily_budget_limit=-5.0  # Invalid negative budget
            )
            
            # Should handle gracefully - governance features might be disabled
            with adapter.track_prompt_operation('invalid_config_test') as span:
                span.update_cost(0.01)
            
            assert adapter.operation_count == 1
    
    def test_cost_calculation_overflow(self):
        """Test handling of cost calculation edge cases."""
        span = EnhancedPromptLayerSpan(
            operation_type='overflow_test',
            operation_name='cost_overflow'
        )
        
        # Test with very large token counts
        span.update_token_usage(1000000, 500000, 'gpt-4')
        
        # Should handle large calculations without overflow
        assert span.estimated_cost > 0
        assert span.total_tokens == 1500000
    
    def test_concurrent_access_errors(self):
        """Test error handling with concurrent access to adapter state."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-concurrent-error',
                team='concurrent-team'
            )
            
            # Simulate concurrent modification
            spans = []
            for i in range(3):
                span = adapter.track_prompt_operation(f'concurrent_{i}').__enter__()
                spans.append(span)
            
            # Modify adapter state while operations are active
            adapter.daily_usage = 100.0  # Large value
            
            # Close all spans
            for span in spans:
                span.update_cost(0.01)
                adapter.track_prompt_operation('dummy').__exit__(None, None, None)
            
            # Should handle state modifications gracefully
            assert len(adapter.active_spans) <= 3  # May have cleanup issues but shouldn't crash


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestRecoveryPatterns:
    """Test error recovery and retry patterns."""
    
    def test_retry_after_transient_error(self):
        """Test retry behavior after transient errors."""
        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            mock_client = Mock()
            mock_pl.return_value = mock_client
            
            # First call fails, second succeeds
            mock_client.run.side_effect = [
                ConnectionError("Temporary connection error"),
                {'response': 'Success after retry'}
            ]
            
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-retry-test',
                team='retry-team'
            )
            
            # First attempt should fail
            with pytest.raises(ConnectionError):
                with adapter.track_prompt_operation('retry_test_1') as span:
                    adapter.run_prompt_with_governance(
                        prompt_name='retry_test_1',
                        input_variables={'attempt': 1}
                    )
            
            # Second attempt should succeed
            with adapter.track_prompt_operation('retry_test_2') as span:
                result = adapter.run_prompt_with_governance(
                    prompt_name='retry_test_2',
                    input_variables={'attempt': 2}
                )
            
            assert 'Success after retry' in result['response']['response']
            assert adapter.operation_count == 2
    
    def test_fallback_to_mock_after_persistent_failure(self):
        """Test fallback to mock client after persistent failures."""
        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            # PromptLayer initialization fails
            mock_pl.side_effect = Exception("PromptLayer service unavailable")
            
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-fallback-test',
                team='fallback-team'
            )
            
            # Should fall back to MockPromptLayer
            assert isinstance(adapter.client, MockPromptLayer)
            
            # Operations should work with degraded functionality
            with adapter.track_prompt_operation('fallback_test') as span:
                result = adapter.run_prompt_with_governance(
                    prompt_name='fallback_test',
                    input_variables={'test': 'fallback'}
                )
                span.update_cost(0.01)
            
            assert result['response']['mock'] is True
            assert 'governance' in result
    
    def test_partial_operation_recovery(self):
        """Test recovery from partial operation failures."""
        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            mock_client = Mock()
            mock_pl.return_value = mock_client
            mock_client.run.return_value = {'response': 'Partial recovery test'}
            
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-partial-recovery',
                team='recovery-team'
            )
            
            # Start operation and simulate partial failure
            with adapter.track_prompt_operation('partial_recovery_test') as span:
                # Simulate successful prompt execution
                result = adapter.run_prompt_with_governance(
                    prompt_name='partial_recovery_test',
                    input_variables={'test': 'partial'}
                )
                
                # Simulate error in cost calculation
                try:
                    span.update_cost(float('inf'))  # Invalid cost
                except (ValueError, OverflowError):
                    span.update_cost(0.01)  # Fallback to reasonable cost
            
            # Should complete successfully with fallback cost
            assert adapter.daily_usage == 0.01
            assert 'governance' in result


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestRobustnessUnderLoad:
    """Test robustness under high load and stress conditions."""
    
    def test_high_concurrency_error_handling(self):
        """Test error handling under high concurrency."""
        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            mock_client = Mock()
            mock_pl.return_value = mock_client
            mock_client.run.return_value = {'response': 'Concurrency test'}
            
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-concurrency-stress',
                team='stress-team',
                daily_budget_limit=50.0
            )
            
            # Simulate many concurrent operations
            active_spans = []
            num_operations = 20
            
            # Start many operations
            for i in range(num_operations):
                try:
                    ctx = adapter.track_prompt_operation(f'stress_test_{i}')
                    span = ctx.__enter__()
                    active_spans.append((ctx, span))
                except Exception:
                    pass  # Some may fail under stress
            
            # Update costs and close operations
            successful_operations = 0
            for ctx, span in active_spans:
                try:
                    span.update_cost(0.05)
                    ctx.__exit__(None, None, None)
                    successful_operations += 1
                except Exception:
                    pass  # Some operations may fail
            
            # Should handle at least some operations successfully
            assert successful_operations > 0
            assert adapter.operation_count >= successful_operations
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-memory-pressure',
                team='memory-team'
            )
            
            # Create many spans with large metadata
            large_metadata = {'large_data': 'x' * 10000}  # 10KB of data per span
            
            for i in range(100):  # 1MB total
                try:
                    with adapter.track_prompt_operation(f'memory_test_{i}') as span:
                        span.add_attributes(large_metadata)
                        span.update_cost(0.001)
                except MemoryError:
                    # Should handle memory pressure gracefully
                    break
                except Exception:
                    # Other errors are acceptable under pressure
                    pass
            
            # Should have processed some operations
            assert adapter.operation_count > 0
    
    def test_long_running_operation_stability(self):
        """Test stability during long-running operations."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-long-running',
                team='stability-team'
            )
            
            with adapter.track_prompt_operation('long_running_test') as span:
                # Simulate long-running operation
                start_time = time.time()
                
                # Update span multiple times during execution
                for i in range(10):
                    span.add_attributes({f'checkpoint_{i}': time.time()})
                    time.sleep(0.001)  # Small delays
                
                span.update_cost(0.25)
                end_time = time.time()
            
            # Verify span tracked the entire duration
            assert span.end_time is not None
            assert span.end_time >= span.start_time
            assert len(span.metadata) >= 10  # All checkpoints recorded
            assert adapter.daily_usage == 0.25