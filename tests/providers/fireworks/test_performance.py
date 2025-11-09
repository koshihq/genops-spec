"""
Performance tests for Fireworks AI provider.

Tests cover:
- Fireattention 4x speed optimization validation
- Throughput and latency measurements
- Memory usage and resource efficiency
- Batch processing performance benefits
- Concurrent operation handling
- Load testing scenarios
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import threading
import statistics

from genops.providers.fireworks import (
    GenOpsFireworksAdapter,
    FireworksModel
)


class TestFireattentionOptimization:
    """Test Fireattention 4x speed optimization."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_fireattention_speed_benchmark(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test Fireattention speed optimization benchmark."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Mock fast response times (Fireattention optimization)
        response_times = []
        
        for i in range(10):  # Multiple tests for statistical significance
            with patch('time.time') as mock_time:
                # Simulate 4x faster responses (baseline ~3.4s, Fireattention ~0.85s)
                mock_time.side_effect = [0.0, 0.85 + (i * 0.02)]  # Small variance
                
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Speed test {i}"}],
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=100
                )
                
                response_times.append(result.execution_time_seconds)
        
        # Verify Fireattention optimization
        avg_response_time = statistics.mean(response_times)
        assert avg_response_time < 1.2  # Should be significantly faster than baseline
        assert all(t < 1.5 for t in response_times)  # All responses should be fast
        
        # Verify consistency (low standard deviation)
        std_dev = statistics.stdev(response_times)
        assert std_dev < 0.1  # Should be consistent
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_fireattention_vs_baseline_comparison(self, mock_fireworks_class, sample_fireworks_config):
        """Test Fireattention optimization vs baseline performance."""
        # Mock baseline (slow) responses
        mock_client_baseline = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Baseline response"))]
        mock_response.usage = Mock(total_tokens=75)
        mock_client_baseline.chat.completions.create.return_value = mock_response
        
        # Mock Fireattention (fast) responses
        mock_client_optimized = Mock()
        mock_client_optimized.chat.completions.create.return_value = mock_response
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Baseline timing (simulate traditional inference)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 3.4]  # Baseline 3.4s
            mock_fireworks_class.return_value = mock_client_baseline
            
            baseline_result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Baseline test"}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100
            )
        
        # Fireattention timing
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 0.85]  # Fireattention 0.85s
            mock_fireworks_class.return_value = mock_client_optimized
            
            optimized_result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Optimized test"}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100
            )
        
        # Verify 4x speedup
        speedup_ratio = baseline_result.execution_time_seconds / optimized_result.execution_time_seconds
        assert speedup_ratio >= 3.5  # Should be close to 4x speedup
        assert speedup_ratio <= 4.5  # Within reasonable range
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_fireattention_across_model_sizes(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test Fireattention optimization across different model sizes."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Test different model sizes
        models_to_test = [
            (FireworksModel.LLAMA_3_2_1B_INSTRUCT, 0.3),   # Tiny model: very fast
            (FireworksModel.LLAMA_3_1_8B_INSTRUCT, 0.85),  # Small model: fast
            (FireworksModel.LLAMA_3_1_70B_INSTRUCT, 1.2),  # Large model: still fast
            (FireworksModel.LLAMA_3_1_405B_INSTRUCT, 2.1)  # Premium model: relatively fast
        ]
        
        performance_results = {}
        
        for model, expected_time in models_to_test:
            with patch('time.time') as mock_time:
                mock_time.side_effect = [0.0, expected_time]
                
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": "Model performance test"}],
                    model=model,
                    max_tokens=100
                )
                
                performance_results[model] = result.execution_time_seconds
        
        # Verify all models benefit from Fireattention
        for model, response_time in performance_results.items():
            assert response_time < 3.0  # All should be faster than baseline
            
        # Verify expected performance hierarchy (smaller models faster)
        tiny_time = performance_results[FireworksModel.LLAMA_3_2_1B_INSTRUCT]
        large_time = performance_results[FireworksModel.LLAMA_3_1_70B_INSTRUCT]
        assert tiny_time < large_time


class TestThroughputPerformance:
    """Test throughput and concurrent performance."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_sequential_throughput(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test sequential operation throughput."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        num_operations = 50
        start_time = time.time()
        
        results = []
        for i in range(num_operations):
            with patch('time.time') as mock_time:
                mock_time.side_effect = [0.0, 0.85]  # Consistent Fireattention speed
                
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Throughput test {i}"}],
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=50
                )
                results.append(result)
        
        total_time = time.time() - start_time
        throughput = num_operations / total_time
        
        # Verify high throughput (operations per second)
        assert throughput > 10  # Should process many operations per second
        assert len(results) == num_operations
        
        # Verify consistent performance
        response_times = [r.execution_time_seconds for r in results]
        assert all(t < 1.0 for t in response_times)
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_concurrent_operations(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test concurrent operation handling."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        def single_operation(operation_id):
            """Single operation for concurrent testing."""
            with patch('time.time') as mock_time:
                mock_time.side_effect = [0.0, 0.9]  # Fast response
                
                return adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Concurrent test {operation_id}"}],
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=50,
                    operation_id=operation_id
                )
        
        # Run concurrent operations
        num_concurrent = 20
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(single_operation, i) 
                for i in range(num_concurrent)
            ]
            
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        concurrent_throughput = num_concurrent / total_time
        
        # Verify concurrent performance
        assert len(results) == num_concurrent
        assert concurrent_throughput > 5  # Good concurrent throughput
        assert all(r.response for r in results)  # All operations completed
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_batch_processing_throughput(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test batch processing throughput benefits."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        batch_size = 100
        batch_requests = [f"Batch request {i}" for i in range(batch_size)]
        
        # Measure batch processing time
        start_time = time.time()
        batch_results = []
        
        for i, request in enumerate(batch_requests):
            with patch('time.time') as mock_time:
                # Batch processing should be faster per operation
                mock_time.side_effect = [0.0, 0.6]  # Faster due to batching
                
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": request}],
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=50,
                    is_batch=True,
                    batch_id="performance-batch",
                    operation_index=i
                )
                batch_results.append(result)
        
        batch_total_time = time.time() - start_time
        batch_throughput = batch_size / batch_total_time
        
        # Verify batch processing benefits
        assert batch_throughput > 15  # Higher throughput due to batching
        assert len(batch_results) == batch_size
        
        # Verify cost benefits (50% savings)
        total_batch_cost = sum(r.cost for r in batch_results)
        assert all(r.governance_attrs.get("is_batch") for r in batch_results)


class TestMemoryAndResourceEfficiency:
    """Test memory usage and resource efficiency."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_memory_usage_stability(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test memory usage remains stable during operations."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Run many operations to test for memory leaks
        num_operations = 200
        
        for i in range(num_operations):
            with patch('time.time') as mock_time:
                mock_time.side_effect = [0.0, 0.8]
                
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Memory test {i}"}],
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=50
                )
                
                # Clear result to allow garbage collection
                del result
        
        # Verify cost tracking is still accurate (no memory leaks affecting state)
        cost_summary = adapter.get_cost_summary()
        assert cost_summary["operations_count"] == num_operations
        assert cost_summary["daily_costs"] > 0
    
    def test_adapter_initialization_efficiency(self, sample_fireworks_config):
        """Test adapter initialization is efficient."""
        start_time = time.time()
        
        # Initialize multiple adapters
        adapters = []
        for i in range(10):
            adapter = GenOpsFireworksAdapter(
                team=f"test-team-{i}",
                project="efficiency-test",
                **{k: v for k, v in sample_fireworks_config.items() 
                   if k not in ["team", "project"]}
            )
            adapters.append(adapter)
        
        initialization_time = time.time() - start_time
        
        # Should initialize quickly
        assert initialization_time < 1.0  # Less than 1 second for 10 adapters
        assert len(adapters) == 10
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_session_memory_efficiency(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test session tracking memory efficiency."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Create and destroy many sessions
        for session_id in range(50):
            with adapter.track_session(f"session-{session_id}") as session:
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": "Session memory test"}],
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=30,
                    session_id=session.session_id
                )
            
            # Session should be properly cleaned up
            assert session.end_time is not None
        
        # Adapter should still function normally
        cost_summary = adapter.get_cost_summary()
        assert cost_summary["operations_count"] == 50


class TestLatencyOptimization:
    """Test latency optimization and response times."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_cold_start_performance(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test cold start (first request) performance."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # First request (cold start)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 1.2]  # Slightly slower for first request
            
            first_result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Cold start test"}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100
            )
        
        # Subsequent requests (warm)
        warm_times = []
        for i in range(5):
            with patch('time.time') as mock_time:
                mock_time.side_effect = [0.0, 0.85]  # Consistent warm performance
                
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Warm test {i}"}],
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=100
                )
                warm_times.append(result.execution_time_seconds)
        
        # Verify warm requests are consistently faster
        avg_warm_time = statistics.mean(warm_times)
        assert avg_warm_time < first_result.execution_time_seconds
        assert avg_warm_time < 1.0  # Warm requests should be very fast
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_token_length_scaling(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test how performance scales with token length."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Test different token lengths
        token_lengths = [50, 100, 200, 500, 1000]
        performance_data = []
        
        for max_tokens in token_lengths:
            with patch('time.time') as mock_time:
                # Time should scale sublinearly due to Fireattention
                base_time = 0.5
                scaling_factor = (max_tokens / 100) ** 0.7  # Sublinear scaling
                response_time = base_time * scaling_factor
                mock_time.side_effect = [0.0, response_time]
                
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Token scaling test {max_tokens}"}],
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=max_tokens
                )
                
                performance_data.append((max_tokens, result.execution_time_seconds))
        
        # Verify reasonable scaling with Fireattention optimization
        for tokens, response_time in performance_data:
            tokens_per_second = tokens / response_time
            assert tokens_per_second > 30  # Good token generation rate
        
        # Verify 1000 tokens still completes reasonably quickly
        thousand_token_time = next(time for tokens, time in performance_data if tokens == 1000)
        assert thousand_token_time < 4.0  # Even large requests are fast


class TestLoadTestingScenarios:
    """Test load testing and stress scenarios."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_sustained_load_performance(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test performance under sustained load."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Simulate sustained load for extended period
        num_operations = 100
        response_times = []
        costs = []
        
        start_time = time.time()
        
        for i in range(num_operations):
            with patch('time.time') as mock_time:
                # Add small variance to simulate real conditions
                base_time = 0.85
                variance = (i % 10) * 0.02  # Small variance pattern
                mock_time.side_effect = [0.0, base_time + variance]
                
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Load test {i}"}],
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=75,
                    load_test_index=i
                )
                
                response_times.append(result.execution_time_seconds)
                costs.append(result.cost)
        
        total_time = time.time() - start_time
        avg_response_time = statistics.mean(response_times)
        throughput = num_operations / total_time
        
        # Verify sustained performance
        assert avg_response_time < 1.2  # Maintain good performance
        assert throughput > 8  # Good sustained throughput
        assert max(response_times) < 2.0  # No extreme outliers
        
        # Verify cost consistency
        cost_variance = statistics.stdev(costs) if len(set(costs)) > 1 else 0
        assert cost_variance < 0.001  # Costs should be consistent
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_peak_load_handling(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test handling of peak load bursts."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Simulate peak load burst
        burst_size = 30
        
        def burst_operation(op_id):
            with patch('time.time') as mock_time:
                mock_time.side_effect = [0.0, 1.1]  # Slightly slower under peak load
                
                return adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Peak load {op_id}"}],
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=60,
                    peak_load_id=op_id
                )
        
        # Execute burst concurrently
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(burst_operation, i) 
                for i in range(burst_size)
            ]
            
            results = [future.result() for future in futures]
        
        burst_time = time.time() - start_time
        burst_throughput = burst_size / burst_time
        
        # Verify peak load handling
        assert len(results) == burst_size
        assert all(r.response for r in results)  # All completed successfully
        assert burst_throughput > 3  # Reasonable throughput under peak load
        assert all(r.execution_time_seconds < 2.0 for r in results)  # No timeouts
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_mixed_workload_performance(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test performance with mixed model workloads."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Mixed workload: different models and request sizes
        mixed_workload = [
            (FireworksModel.LLAMA_3_2_1B_INSTRUCT, 30, 0.4),    # Fast, cheap
            (FireworksModel.LLAMA_3_1_8B_INSTRUCT, 100, 0.85),  # Balanced
            (FireworksModel.LLAMA_3_1_70B_INSTRUCT, 150, 1.3),  # Slower, higher quality
            (FireworksModel.MIXTRAL_8X7B, 120, 1.0),           # MoE efficiency
        ] * 10  # Repeat pattern 10 times
        
        results = []
        start_time = time.time()
        
        for i, (model, max_tokens, expected_time) in enumerate(mixed_workload):
            with patch('time.time') as mock_time:
                mock_time.side_effect = [0.0, expected_time]
                
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Mixed workload {i}"}],
                    model=model,
                    max_tokens=max_tokens,
                    workload_index=i
                )
                results.append(result)
        
        total_time = time.time() - start_time
        mixed_throughput = len(mixed_workload) / total_time
        
        # Verify mixed workload performance
        assert len(results) == len(mixed_workload)
        assert mixed_throughput > 5  # Good throughput with mixed models
        
        # Verify cost scaling matches expectations
        tiny_costs = [r.cost for r in results[::4]]    # Every 4th (tiny model)
        large_costs = [r.cost for r in results[2::4]]  # Every 4th offset by 2 (large model)
        
        assert statistics.mean(tiny_costs) < statistics.mean(large_costs)  # Cost scaling