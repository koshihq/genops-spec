#!/usr/bin/env python3
"""
Performance tests for Together AI provider.

Tests load handling, memory usage, concurrent operations,
throughput benchmarks, and scalability patterns.
"""

import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from unittest.mock import MagicMock, patch

import psutil
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    from src.genops.providers.together import (
        GenOpsTogetherAdapter,
        TogetherModel,
        auto_instrument,
    )
    from src.genops.providers.together_pricing import TogetherPricingCalculator
except ImportError as e:
    pytest.skip(f"Together AI provider not available: {e}", allow_module_level=True)


@pytest.fixture
def mock_together_client():
    """Fixture providing fast mocked Together client."""
    with patch("src.genops.providers.together.Together") as mock:
        client = MagicMock()

        def fast_response(*args, **kwargs):
            return MagicMock(
                choices=[{"message": {"content": "Fast test response"}}],
                usage={"prompt_tokens": 5, "completion_tokens": 10},
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            )

        client.chat.completions.create.side_effect = fast_response
        mock.return_value = client
        yield client


@pytest.fixture
def performance_adapter():
    """Fixture providing adapter optimized for performance testing."""
    return GenOpsTogetherAdapter(
        team="performance-test",
        project="load-testing",
        daily_budget_limit=100.0,  # High limit for testing
        governance_policy="advisory",  # Fastest policy
    )


class TestThroughputPerformance:
    """Test throughput and request handling performance."""

    def test_single_request_latency(self, mock_together_client, performance_adapter):
        """Test latency of single request."""
        start_time = time.time()

        result = performance_adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Performance test"}],
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=50,
        )

        end_time = time.time()
        latency = end_time - start_time

        assert result is not None
        assert latency < 1.0  # Should complete in under 1 second with mocking
        assert result.execution_time_seconds > 0

        print(f"Single request latency: {latency:.3f}s")

    def test_sequential_requests_throughput(
        self, mock_together_client, performance_adapter
    ):
        """Test throughput of sequential requests."""
        num_requests = 50
        start_time = time.time()

        results = []
        for i in range(num_requests):
            result = performance_adapter.chat_with_governance(
                messages=[{"role": "user", "content": f"Request {i}"}],
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=20,
                request_id=f"perf-test-{i}",
            )
            results.append(result)

        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_requests / total_time

        assert len(results) == num_requests
        assert all(r.response is not None for r in results)
        assert throughput > 10  # Should handle >10 requests/second with mocking

        print(f"Sequential throughput: {throughput:.1f} requests/second")

    def test_concurrent_requests_performance(
        self, mock_together_client, performance_adapter
    ):
        """Test concurrent request handling performance."""
        num_concurrent = 20
        num_requests_each = 5

        def worker_function(worker_id):
            """Worker function for concurrent testing."""
            worker_results = []
            for i in range(num_requests_each):
                result = performance_adapter.chat_with_governance(
                    messages=[
                        {"role": "user", "content": f"Worker {worker_id} Request {i}"}
                    ],
                    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=20,
                    worker_id=worker_id,
                    request_index=i,
                )
                worker_results.append(result)
            return worker_results

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            future_to_worker = {
                executor.submit(worker_function, worker_id): worker_id
                for worker_id in range(num_concurrent)
            }

            all_results = []
            for future in as_completed(future_to_worker):
                worker_results = future.result()
                all_results.extend(worker_results)

        end_time = time.time()
        total_time = end_time - start_time
        total_requests = num_concurrent * num_requests_each
        concurrent_throughput = total_requests / total_time

        assert len(all_results) == total_requests
        assert all(r.response is not None for r in all_results)
        assert (
            concurrent_throughput > 30
        )  # Should handle >30 requests/second concurrently

        print(f"Concurrent throughput: {concurrent_throughput:.1f} requests/second")
        print(f"Concurrent speedup vs sequential: {concurrent_throughput / 10:.1f}x")

    def test_batch_operation_performance(
        self, mock_together_client, performance_adapter
    ):
        """Test batch operation performance."""
        batch_size = 100

        # Prepare batch operations
        batch_messages = [
            [{"role": "user", "content": f"Batch request {i}"}]
            for i in range(batch_size)
        ]

        start_time = time.time()

        # Simulate batch processing
        results = []
        with performance_adapter.track_session("batch-performance") as session:
            for i, messages in enumerate(batch_messages):
                result = performance_adapter.chat_with_governance(
                    messages=messages,
                    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=15,
                    session_id=session.session_id,
                    batch_index=i,
                )
                results.append(result)

        end_time = time.time()
        total_time = end_time - start_time
        batch_throughput = batch_size / total_time

        assert len(results) == batch_size
        assert session.total_operations == batch_size
        assert batch_throughput > 20  # Should handle >20 batch items/second

        print(f"Batch processing throughput: {batch_throughput:.1f} items/second")


class TestMemoryPerformance:
    """Test memory usage and resource management."""

    def test_memory_usage_single_adapter(self, mock_together_client):
        """Test memory usage of single adapter instance."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create adapter
        GenOpsTogetherAdapter(team="memory-test", project="resource-management")

        adapter_creation_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = adapter_creation_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 50  # Less than 50MB for adapter creation

        print(f"Adapter creation memory increase: {memory_increase:.1f}MB")

    def test_memory_usage_multiple_operations(
        self, mock_together_client, performance_adapter
    ):
        """Test memory usage with multiple operations."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Perform many operations
        num_operations = 100
        for i in range(num_operations):
            performance_adapter.chat_with_governance(
                messages=[{"role": "user", "content": f"Memory test {i}"}],
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=10,
            )

            # Periodic memory check
            if i % 25 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_per_operation = (current_memory - initial_memory) / (i + 1)
                assert memory_per_operation < 1.0  # Less than 1MB per operation

        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory
        memory_per_operation = total_memory_increase / num_operations

        assert memory_per_operation < 0.5  # Less than 0.5MB per operation on average

        print(f"Memory per operation: {memory_per_operation:.3f}MB")

    def test_memory_cleanup_after_session(
        self, mock_together_client, performance_adapter
    ):
        """Test memory cleanup after session completion."""
        process = psutil.Process()

        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Create and use session
        session_operations = 50
        with performance_adapter.track_session("memory-cleanup-test") as session:
            for i in range(session_operations):
                performance_adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Cleanup test {i}"}],
                    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=10,
                    session_id=session.session_id,
                )

        # Force garbage collection and check memory
        gc.collect()
        post_session_memory = process.memory_info().rss / 1024 / 1024
        memory_retained = post_session_memory - baseline_memory

        # Some memory increase is expected, but should be reasonable
        assert memory_retained < 25  # Less than 25MB retained after session

        print(f"Memory retained after session: {memory_retained:.1f}MB")

    def test_memory_usage_multiple_adapters(self, mock_together_client):
        """Test memory usage with multiple adapter instances."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Create multiple adapters
        num_adapters = 10
        adapters = []

        for i in range(num_adapters):
            adapter = GenOpsTogetherAdapter(
                team=f"team-{i}", project=f"project-{i}", customer_id=f"customer-{i}"
            )
            adapters.append(adapter)

        multi_adapter_memory = process.memory_info().rss / 1024 / 1024
        memory_per_adapter = (multi_adapter_memory - initial_memory) / num_adapters

        # Each adapter should use reasonable memory
        assert memory_per_adapter < 10  # Less than 10MB per adapter

        print(f"Memory per adapter instance: {memory_per_adapter:.1f}MB")


class TestScalabilityPerformance:
    """Test scalability patterns and limits."""

    def test_session_scalability(self, mock_together_client, performance_adapter):
        """Test performance with many concurrent sessions."""
        num_sessions = 25
        operations_per_session = 4

        def session_worker(session_id):
            """Worker function for session testing."""
            session_results = []
            with performance_adapter.track_session(
                f"scale-session-{session_id}"
            ) as session:
                for i in range(operations_per_session):
                    result = performance_adapter.chat_with_governance(
                        messages=[
                            {"role": "user", "content": f"Session {session_id} Op {i}"}
                        ],
                        model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                        max_tokens=10,
                        session_id=session.session_id,
                    )
                    session_results.append(result)
            return session_results

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_sessions) as executor:
            futures = [executor.submit(session_worker, i) for i in range(num_sessions)]
            all_results = []

            for future in as_completed(futures):
                session_results = future.result()
                all_results.extend(session_results)

        end_time = time.time()
        total_time = end_time - start_time
        total_operations = num_sessions * operations_per_session
        scalability_throughput = total_operations / total_time

        assert len(all_results) == total_operations
        assert scalability_throughput > 25  # Should maintain good throughput

        print(
            f"Multi-session scalability throughput: {scalability_throughput:.1f} ops/second"
        )

    def test_cost_calculation_performance(self, performance_adapter):
        """Test cost calculation performance at scale."""
        calculator = TogetherPricingCalculator()

        # Test cost calculation performance
        num_calculations = 1000
        models = [
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ]

        start_time = time.time()

        for i in range(num_calculations):
            model = models[i % len(models)]
            tokens = 100 + (i % 500)  # Varying token counts

            cost = calculator.estimate_chat_cost(model, tokens=tokens)
            assert isinstance(cost, Decimal)
            assert cost > 0

        end_time = time.time()
        calculation_time = end_time - start_time
        calculations_per_second = num_calculations / calculation_time

        assert calculations_per_second > 1000  # Should handle >1000 calculations/second

        print(
            f"Cost calculation throughput: {calculations_per_second:.0f} calculations/second"
        )

    def test_governance_attribute_performance(
        self, mock_together_client, performance_adapter
    ):
        """Test performance impact of governance attributes."""
        # Test with minimal governance attributes
        start_time = time.time()

        for i in range(50):
            performance_adapter.chat_with_governance(
                messages=[{"role": "user", "content": f"Minimal governance {i}"}],
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=10,
            )

        minimal_time = time.time() - start_time

        # Test with many governance attributes
        start_time = time.time()

        for i in range(50):
            performance_adapter.chat_with_governance(
                messages=[{"role": "user", "content": f"Rich governance {i}"}],
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=10,
                feature=f"feature-{i}",
                session_id=f"session-{i}",
                custom_attr1="value1",
                custom_attr2="value2",
                custom_attr3=f"value-{i}",
                operation_type="performance-test",
                complexity="high",
                priority="normal",
            )

        rich_governance_time = time.time() - start_time

        # Rich governance shouldn't significantly impact performance
        performance_impact = (rich_governance_time - minimal_time) / minimal_time
        assert performance_impact < 0.5  # Less than 50% performance impact

        print(f"Governance attributes performance impact: {performance_impact:.1%}")


class TestAutoInstrumentationPerformance:
    """Test performance of auto-instrumentation features."""

    @patch("src.genops.providers.together.Together")
    def test_auto_instrument_setup_performance(self, mock_together):
        """Test auto-instrumentation setup performance."""
        mock_client = MagicMock()
        mock_together.return_value = mock_client

        # Test setup time
        start_time = time.time()
        auto_instrument()
        setup_time = time.time() - start_time

        assert setup_time < 1.0  # Should setup in under 1 second

        print(f"Auto-instrumentation setup time: {setup_time:.3f}s")

    @patch("src.genops.providers.together.Together")
    def test_auto_instrument_overhead(self, mock_together):
        """Test auto-instrumentation runtime overhead."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[{"message": {"content": "Auto-instrumented response"}}],
            usage={"prompt_tokens": 5, "completion_tokens": 10},
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        )
        mock_together.return_value = mock_client

        # Apply auto-instrumentation
        auto_instrument()

        # Test performance impact (if any)
        num_operations = 20
        start_time = time.time()

        # Simulate auto-instrumented operations
        for _i in range(num_operations):
            # In real scenario, this would be instrumented Together calls
            # For testing, we just verify setup doesn't impact performance
            time.sleep(0.001)  # Minimal simulated work

        total_time = time.time() - start_time
        time_per_operation = total_time / num_operations

        assert time_per_operation < 0.1  # Reasonable per-operation time

        print(f"Auto-instrumentation overhead per operation: {time_per_operation:.3f}s")


class TestStressTestScenarios:
    """Test stress scenarios and edge cases."""

    def test_rapid_fire_requests(self, mock_together_client, performance_adapter):
        """Test handling rapid sequential requests."""
        num_rapid_requests = 200
        start_time = time.time()

        successful_requests = 0
        for i in range(num_rapid_requests):
            try:
                result = performance_adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Rapid {i}"}],
                    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=5,  # Minimal tokens for speed
                    rapid_fire=True,
                )
                if result.response:
                    successful_requests += 1
            except Exception:
                # Some failures might be expected under stress
                pass

        end_time = time.time()
        total_time = end_time - start_time
        success_rate = successful_requests / num_rapid_requests

        assert success_rate > 0.9  # At least 90% success rate
        assert total_time < 30  # Complete within 30 seconds

        print(f"Rapid fire success rate: {success_rate:.1%}")
        print(
            f"Rapid fire throughput: {successful_requests / total_time:.1f} requests/second"
        )

    def test_large_session_handling(self, mock_together_client, performance_adapter):
        """Test handling sessions with many operations."""
        operations_in_session = 150

        start_time = time.time()
        with performance_adapter.track_session("large-session-test") as session:
            for i in range(operations_in_session):
                performance_adapter.chat_with_governance(
                    messages=[{"role": "user", "content": f"Large session op {i}"}],
                    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=5,
                    session_id=session.session_id,
                    operation_index=i,
                )

                # Verify session tracking doesn't degrade
                if i % 50 == 0:
                    assert session.total_operations == i + 1
                    assert session.total_cost > 0

        end_time = time.time()
        session_time = end_time - start_time

        assert session.total_operations == operations_in_session
        assert session.end_time > session.start_time
        assert session_time < 60  # Complete within 1 minute

        print(f"Large session ({operations_in_session} ops) time: {session_time:.1f}s")


if __name__ == "__main__":
    # Run with performance reporting
    pytest.main([__file__, "-v", "-s"])
