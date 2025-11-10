"""
Performance and benchmarking tests for PromptLayer integration.

Tests performance characteristics, scalability, memory usage,
and optimization features under various load conditions.
"""

import gc
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import psutil
import pytest

try:
    from genops.providers.promptlayer import (
        EnhancedPromptLayerSpan,
        GenOpsPromptLayerAdapter,
        auto_instrument,
    )
    PROMPTLAYER_AVAILABLE = True
except ImportError:
    PROMPTLAYER_AVAILABLE = False


@pytest.mark.performance
@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestPerformanceBenchmarks:
    """Benchmark performance characteristics of the PromptLayer integration."""

    def setup_method(self):
        """Set up performance test fixtures."""
        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            mock_client = Mock()
            mock_pl.return_value = mock_client
            mock_client.run.return_value = {'response': 'Benchmark response'}

            self.adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-benchmark-test',
                team='benchmark-team',
                daily_budget_limit=100.0
            )

    def test_single_operation_latency(self):
        """Benchmark latency of single operation."""
        latencies = []
        num_operations = 100

        for i in range(num_operations):
            start_time = time.perf_counter()

            with self.adapter.track_prompt_operation(f'latency_test_{i}') as span:
                self.adapter.run_prompt_with_governance(
                    prompt_name=f'latency_test_{i}',
                    input_variables={'iteration': i}
                )
                span.update_cost(0.001)

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate performance metrics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

        # Performance assertions
        assert avg_latency < 10.0  # Less than 10ms average
        assert min_latency < 5.0   # Less than 5ms minimum
        assert p95_latency < 20.0  # Less than 20ms for 95th percentile

        print(f"Single operation latency - Avg: {avg_latency:.2f}ms, "
              f"Min: {min_latency:.2f}ms, Max: {max_latency:.2f}ms, "
              f"P95: {p95_latency:.2f}ms")

    def test_span_creation_overhead(self):
        """Benchmark span creation and finalization overhead."""
        num_spans = 1000

        start_time = time.perf_counter()

        for i in range(num_spans):
            span = EnhancedPromptLayerSpan(
                operation_type='benchmark',
                operation_name=f'overhead_test_{i}',
                team='benchmark-team'
            )
            span.update_cost(0.001)
            span.add_attributes({'test_id': i})
            span.finalize()

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000  # ms
        avg_time_per_span = total_time / num_spans

        # Performance assertions
        assert avg_time_per_span < 1.0  # Less than 1ms per span
        assert total_time < 500.0  # Less than 500ms total

        print(f"Span creation overhead - Total: {total_time:.2f}ms, "
              f"Per span: {avg_time_per_span:.3f}ms")

    def test_concurrent_operations_performance(self):
        """Benchmark performance under concurrent operations."""
        num_threads = 10
        operations_per_thread = 20

        def run_operations(thread_id):
            thread_latencies = []

            for i in range(operations_per_thread):
                start_time = time.perf_counter()

                with self.adapter.track_prompt_operation(f'concurrent_{thread_id}_{i}') as span:
                    self.adapter.run_prompt_with_governance(
                        prompt_name=f'concurrent_{thread_id}_{i}',
                        input_variables={'thread': thread_id, 'operation': i}
                    )
                    span.update_cost(0.002)

                end_time = time.perf_counter()
                thread_latencies.append((end_time - start_time) * 1000)

            return thread_latencies

        # Run concurrent operations
        overall_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_operations, i) for i in range(num_threads)]
            all_latencies = []

            for future in as_completed(futures):
                thread_latencies = future.result()
                all_latencies.extend(thread_latencies)

        overall_end = time.perf_counter()
        overall_time = (overall_end - overall_start) * 1000

        # Calculate metrics
        total_operations = num_threads * operations_per_thread
        avg_latency = sum(all_latencies) / len(all_latencies)
        throughput = total_operations / (overall_time / 1000)  # ops/second

        # Performance assertions
        assert avg_latency < 50.0  # Less than 50ms average under concurrency
        assert throughput > 100    # More than 100 operations per second
        assert self.adapter.operation_count == total_operations

        print(f"Concurrent performance - Throughput: {throughput:.1f} ops/sec, "
              f"Avg latency: {avg_latency:.2f}ms, Total time: {overall_time:.1f}ms")

    def test_high_volume_operations_performance(self):
        """Benchmark performance with high volume of operations."""
        num_operations = 5000
        batch_size = 100

        start_time = time.perf_counter()

        for batch in range(0, num_operations, batch_size):
            batch_start = time.perf_counter()

            for i in range(batch, min(batch + batch_size, num_operations)):
                with self.adapter.track_prompt_operation(f'high_volume_{i}') as span:
                    span.update_cost(0.0005)  # Small cost per operation

            batch_end = time.perf_counter()
            batch_time = (batch_end - batch_start) * 1000

            # Ensure batch processing is efficient
            assert batch_time < 1000  # Less than 1 second per batch

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        avg_time_per_op = total_time / num_operations

        # Performance assertions
        assert avg_time_per_op < 0.5  # Less than 0.5ms per operation
        assert total_time < 10000     # Less than 10 seconds total
        assert self.adapter.operation_count == num_operations

        print(f"High volume performance - {num_operations} ops in {total_time:.1f}ms, "
              f"Avg: {avg_time_per_op:.3f}ms per op")


@pytest.mark.performance
@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestMemoryUsage:
    """Test memory usage characteristics and optimization."""

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_memory_usage_single_operations(self):
        """Test memory usage for single operations."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-memory-test',
                team='memory-team'
            )

            initial_memory = self.get_memory_usage()

            # Perform operations and measure memory
            num_operations = 1000
            for i in range(num_operations):
                with adapter.track_prompt_operation(f'memory_test_{i}') as span:
                    span.update_cost(0.001)
                    span.add_attributes({'test_data': f'operation_{i}'})

            # Force garbage collection
            gc.collect()
            final_memory = self.get_memory_usage()

            memory_increase = final_memory - initial_memory
            memory_per_operation = memory_increase / num_operations * 1024  # KB

            # Memory usage assertions
            assert memory_increase < 50.0  # Less than 50MB total increase
            assert memory_per_operation < 50  # Less than 50KB per operation

            print(f"Memory usage - Total increase: {memory_increase:.2f}MB, "
                  f"Per operation: {memory_per_operation:.2f}KB")

    def test_memory_cleanup_after_operations(self):
        """Test memory cleanup after operations complete."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-cleanup-test',
                team='cleanup-team'
            )

            initial_memory = self.get_memory_usage()

            # Create large operations with substantial metadata
            large_data = 'x' * 1000  # 1KB per operation
            num_operations = 500

            for i in range(num_operations):
                with adapter.track_prompt_operation(f'cleanup_test_{i}') as span:
                    span.update_cost(0.002)
                    span.add_attributes({
                        'large_metadata': large_data,
                        'operation_id': i,
                        'timestamp': time.time()
                    })

            # Verify active spans are cleaned up
            assert len(adapter.active_spans) == 0

            # Force garbage collection
            gc.collect()
            time.sleep(0.1)  # Allow cleanup
            gc.collect()

            final_memory = self.get_memory_usage()
            memory_increase = final_memory - initial_memory

            # Memory should be mostly cleaned up
            assert memory_increase < 10.0  # Less than 10MB retained

            print(f"Memory cleanup - Retained: {memory_increase:.2f}MB after {num_operations} operations")

    def test_long_running_memory_stability(self):
        """Test memory stability over long-running operations."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-stability-test',
                team='stability-team'
            )

            memory_samples = []
            num_cycles = 10
            operations_per_cycle = 100

            for cycle in range(num_cycles):
                cycle_start_memory = self.get_memory_usage()

                # Perform operations
                for i in range(operations_per_cycle):
                    with adapter.track_prompt_operation(f'stability_{cycle}_{i}') as span:
                        span.update_cost(0.001)
                        span.add_attributes({'cycle': cycle, 'operation': i})

                # Force cleanup
                gc.collect()
                cycle_end_memory = self.get_memory_usage()
                memory_samples.append(cycle_end_memory)

            # Analyze memory stability
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
            memory_variation = max_memory - min_memory

            # Memory should remain stable
            assert memory_variation < 20.0  # Less than 20MB variation

            # Should not show consistent upward trend (memory leak)
            first_half_avg = sum(memory_samples[:5]) / 5
            second_half_avg = sum(memory_samples[5:]) / 5
            memory_trend = second_half_avg - first_half_avg

            assert memory_trend < 10.0  # Less than 10MB trend increase

            print(f"Memory stability - Variation: {memory_variation:.2f}MB, "
                  f"Trend: {memory_trend:.2f}MB")


@pytest.mark.performance
@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestScalabilityPatterns:
    """Test scalability patterns and optimization strategies."""

    def test_batch_operation_efficiency(self):
        """Test efficiency of batch operations vs individual operations."""
        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            mock_client = Mock()
            mock_pl.return_value = mock_client
            mock_client.run.return_value = {'response': 'Batch test'}

            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-batch-test',
                team='batch-team'
            )

            # Test individual operations
            individual_start = time.perf_counter()
            for i in range(50):
                with adapter.track_prompt_operation(f'individual_{i}') as span:
                    adapter.run_prompt_with_governance(
                        prompt_name=f'individual_{i}',
                        input_variables={'batch': False, 'id': i}
                    )
                    span.update_cost(0.002)
            individual_end = time.perf_counter()
            individual_time = (individual_end - individual_start) * 1000

            # Reset adapter state
            adapter.daily_usage = 0.0
            adapter.operation_count = 0

            # Test batch-style operation
            batch_start = time.perf_counter()
            with adapter.track_prompt_operation('batch_operation') as batch_span:
                # Simulate processing multiple items in one operation
                batch_items = []
                for i in range(50):
                    batch_items.append({'id': i, 'batch': True})

                adapter.run_prompt_with_governance(
                    prompt_name='batch_operation',
                    input_variables={'items': batch_items, 'count': 50}
                )
                batch_span.update_cost(0.05)  # Lower total cost due to efficiency
            batch_end = time.perf_counter()
            batch_time = (batch_end - batch_start) * 1000

            # Batch should be more efficient
            efficiency_ratio = batch_time / individual_time
            assert efficiency_ratio < 0.5  # Batch should be at least 2x faster

            print(f"Batch efficiency - Individual: {individual_time:.1f}ms, "
                  f"Batch: {batch_time:.1f}ms, Ratio: {efficiency_ratio:.2f}")

    def test_connection_pooling_simulation(self):
        """Simulate connection pooling benefits."""
        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            # Simulate connection setup overhead
            connection_setup_delay = 0.01  # 10ms setup time

            def mock_client_with_delay():
                time.sleep(connection_setup_delay)
                mock_client = Mock()
                mock_client.run.return_value = {'response': 'Pooled response'}
                return mock_client

            mock_pl.side_effect = mock_client_with_delay

            # Test without connection reuse (new client each time)
            no_pooling_start = time.perf_counter()
            for i in range(10):
                adapter = GenOpsPromptLayerAdapter(
                    promptlayer_api_key='pl-no-pooling',
                    team='no-pooling-team'
                )
                with adapter.track_prompt_operation(f'no_pool_{i}') as span:
                    span.update_cost(0.001)
            no_pooling_end = time.perf_counter()
            no_pooling_time = (no_pooling_end - no_pooling_start) * 1000

            # Test with connection reuse (single client)
            mock_pl.side_effect = None
            mock_pl.return_value = Mock()
            mock_pl.return_value.run.return_value = {'response': 'Pooled response'}

            pooling_start = time.perf_counter()
            shared_adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-pooling',
                team='pooling-team'
            )
            for i in range(10):
                with shared_adapter.track_prompt_operation(f'pooled_{i}') as span:
                    span.update_cost(0.001)
            pooling_end = time.perf_counter()
            pooling_time = (pooling_end - pooling_start) * 1000

            # Connection reuse should be much faster
            efficiency_gain = no_pooling_time / pooling_time
            assert efficiency_gain > 2.0  # At least 2x improvement

            print(f"Connection pooling - No pooling: {no_pooling_time:.1f}ms, "
                  f"With pooling: {pooling_time:.1f}ms, Gain: {efficiency_gain:.1f}x")

    def test_auto_instrumentation_overhead(self):
        """Test overhead of auto-instrumentation vs manual instrumentation."""
        # Test manual instrumentation
        with patch('genops.providers.promptlayer.PromptLayer') as mock_pl:
            mock_client = Mock()
            mock_pl.return_value = mock_client
            mock_client.run.return_value = {'response': 'Manual test'}

            manual_adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-manual',
                team='manual-team'
            )

            manual_start = time.perf_counter()
            for i in range(100):
                with manual_adapter.track_prompt_operation(f'manual_{i}') as span:
                    span.update_cost(0.001)
            manual_end = time.perf_counter()
            manual_time = (manual_end - manual_start) * 1000

            # Test auto-instrumentation
            with patch('genops.providers.promptlayer.GenOpsPromptLayerAdapter') as mock_adapter_class:
                mock_adapter = Mock()
                mock_adapter.track_prompt_operation.return_value.__enter__ = Mock()
                mock_adapter.track_prompt_operation.return_value.__exit__ = Mock()
                mock_adapter_class.return_value = mock_adapter

                auto_instrument(team='auto-team', project='auto-test')

                auto_start = time.perf_counter()
                # Simulate auto-instrumented operations
                for i in range(100):
                    # In real scenario, this would be intercepted automatically
                    mock_adapter.track_prompt_operation(f'auto_{i}').__enter__()
                    mock_adapter.track_prompt_operation(f'auto_{i}').__exit__(None, None, None)
                auto_end = time.perf_counter()
                auto_time = (auto_end - auto_start) * 1000

                # Auto-instrumentation overhead should be minimal
                overhead_ratio = auto_time / manual_time
                assert overhead_ratio < 1.5  # Less than 50% overhead

                print(f"Auto-instrumentation overhead - Manual: {manual_time:.1f}ms, "
                      f"Auto: {auto_time:.1f}ms, Overhead: {overhead_ratio:.2f}x")


@pytest.mark.performance
@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestCacheOptimization:
    """Test caching and optimization features."""

    def test_span_metadata_caching(self):
        """Test optimization of span metadata handling."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-cache-test',
                team='cache-team'
            )

            # Test with repeated metadata patterns
            common_attrs = {
                'common_field_1': 'value_1',
                'common_field_2': 'value_2',
                'common_field_3': 'value_3'
            }

            cache_start = time.perf_counter()
            for i in range(200):
                with adapter.track_prompt_operation(f'cache_test_{i}') as span:
                    span.add_attributes(common_attrs)
                    span.add_attributes({'unique_field': i})
                    span.update_cost(0.001)
            cache_end = time.perf_counter()
            cache_time = (cache_end - cache_start) * 1000

            # Should handle repeated metadata efficiently
            avg_time_per_op = cache_time / 200
            assert avg_time_per_op < 2.0  # Less than 2ms per operation

            print(f"Metadata caching - {cache_time:.1f}ms total, "
                  f"{avg_time_per_op:.2f}ms per operation")

    def test_cost_calculation_optimization(self):
        """Test optimization of cost calculations."""
        # Test cost calculation performance
        num_calculations = 1000

        calc_start = time.perf_counter()
        for i in range(num_calculations):
            span = EnhancedPromptLayerSpan(
                operation_type='cost_calc_test',
                operation_name=f'calc_{i}'
            )

            # Mix of different cost calculation patterns
            if i % 3 == 0:
                span.update_token_usage(100, 50, 'gpt-3.5-turbo')
            elif i % 3 == 1:
                span.update_token_usage(200, 100, 'gpt-4')
            else:
                span.update_cost(0.005)
        calc_end = time.perf_counter()
        calc_time = (calc_end - calc_start) * 1000

        avg_calc_time = calc_time / num_calculations
        assert avg_calc_time < 0.1  # Less than 0.1ms per calculation

        print(f"Cost calculation performance - {calc_time:.1f}ms total, "
              f"{avg_calc_time:.3f}ms per calculation")


@pytest.mark.performance
@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestResourceUsageOptimization:
    """Test resource usage optimization and efficiency."""

    def test_thread_safety_performance(self):
        """Test performance impact of thread safety measures."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-thread-safety',
                team='thread-team'
            )

            def worker_function(worker_id, num_ops):
                for i in range(num_ops):
                    with adapter.track_prompt_operation(f'worker_{worker_id}_{i}') as span:
                        span.update_cost(0.001)
                        span.add_attributes({'worker_id': worker_id})

            # Test with multiple threads
            num_workers = 5
            ops_per_worker = 50

            thread_start = time.perf_counter()
            threads = []
            for worker_id in range(num_workers):
                thread = threading.Thread(
                    target=worker_function,
                    args=(worker_id, ops_per_worker)
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            thread_end = time.perf_counter()

            thread_time = (thread_end - thread_start) * 1000
            total_ops = num_workers * ops_per_worker

            # Should handle concurrent access efficiently
            assert thread_time < 5000  # Less than 5 seconds
            assert adapter.operation_count == total_ops

            throughput = total_ops / (thread_time / 1000)
            print(f"Thread safety performance - {throughput:.1f} ops/sec, "
                  f"{thread_time:.1f}ms total")

    def test_garbage_collection_impact(self):
        """Test performance impact of garbage collection."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-gc-test',
                team='gc-team'
            )

            # Disable automatic garbage collection
            gc.disable()

            no_gc_start = time.perf_counter()
            for i in range(500):
                with adapter.track_prompt_operation(f'no_gc_{i}') as span:
                    span.update_cost(0.001)
                    span.add_attributes({'large_data': 'x' * 100})
            no_gc_end = time.perf_counter()
            no_gc_time = (no_gc_end - no_gc_start) * 1000

            # Reset and enable garbage collection
            adapter.daily_usage = 0.0
            adapter.operation_count = 0
            gc.enable()
            gc.collect()

            gc_start = time.perf_counter()
            for i in range(500):
                with adapter.track_prompt_operation(f'with_gc_{i}') as span:
                    span.update_cost(0.001)
                    span.add_attributes({'large_data': 'x' * 100})
            gc_end = time.perf_counter()
            gc_time = (gc_end - gc_start) * 1000

            # GC impact should be reasonable
            gc_overhead = (gc_time - no_gc_time) / no_gc_time
            assert gc_overhead < 0.5  # Less than 50% overhead from GC

            print(f"GC impact - No GC: {no_gc_time:.1f}ms, "
                  f"With GC: {gc_time:.1f}ms, Overhead: {gc_overhead:.1%}")

    def test_operation_cleanup_efficiency(self):
        """Test efficiency of operation cleanup processes."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-cleanup-efficiency',
                team='cleanup-team'
            )

            cleanup_times = []
            num_batches = 10
            ops_per_batch = 100

            for batch in range(num_batches):
                # Create operations
                for i in range(ops_per_batch):
                    with adapter.track_prompt_operation(f'cleanup_{batch}_{i}') as span:
                        span.update_cost(0.001)
                        span.add_attributes({
                            'batch': batch,
                            'operation': i,
                            'data': f'cleanup_data_{i}'
                        })

                # Measure cleanup time
                cleanup_start = time.perf_counter()
                gc.collect()  # Force cleanup
                cleanup_end = time.perf_counter()

                cleanup_time = (cleanup_end - cleanup_start) * 1000
                cleanup_times.append(cleanup_time)

                # Verify cleanup occurred
                assert len(adapter.active_spans) == 0

            avg_cleanup_time = sum(cleanup_times) / len(cleanup_times)
            max_cleanup_time = max(cleanup_times)

            # Cleanup should be efficient
            assert avg_cleanup_time < 50  # Less than 50ms average
            assert max_cleanup_time < 100  # Less than 100ms maximum

            print(f"Cleanup efficiency - Avg: {avg_cleanup_time:.1f}ms, "
                  f"Max: {max_cleanup_time:.1f}ms per batch of {ops_per_batch} ops")
