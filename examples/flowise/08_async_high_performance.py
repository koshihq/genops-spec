#!/usr/bin/env python3
"""
Example: Async High-Performance Processing

Complexity: ⭐⭐⭐ Advanced

This example demonstrates high-throughput, async processing patterns for
Flowise with connection pooling, concurrent execution, batch processing,
and performance optimization techniques.

Prerequisites:
- Flowise instance running
- GenOps package installed
- aiohttp and asyncio for async processing

Usage:
    python 08_async_high_performance.py
    python 08_async_high_performance.py --benchmark  # Run performance benchmarks

Environment Variables:
    FLOWISE_BASE_URL: Flowise instance URL
    FLOWISE_API_KEY: API key
    MAX_CONCURRENT: Maximum concurrent requests (default: 10)
"""

import argparse
import asyncio
import logging
import os
import statistics
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Optional

# Async HTTP dependencies
try:
    import aiofiles  # noqa: F401
    import aiohttp

    HAS_ASYNC_DEPS = True
except ImportError:
    print("⚠️  Install async dependencies: pip install aiohttp aiofiles")
    HAS_ASYNC_DEPS = False

from genops.providers.flowise import instrument_flowise
from genops.providers.flowise_validation import validate_flowise_setup

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AsyncRequestResult:
    """Result of an async Flowise request."""

    request_id: str
    success: bool
    response_data: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: int = 0
    tokens_estimated: int = 0
    cost_estimated: float = 0.0
    retry_count: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for async processing."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_ms: int = 0
    min_duration_ms: int = float("inf")
    max_duration_ms: int = 0
    durations: list[int] = field(default_factory=list)
    throughput_rps: float = 0.0
    success_rate: float = 0.0

    def add_result(self, result: AsyncRequestResult):
        """Add a result to the metrics."""
        self.total_requests += 1
        self.total_duration_ms += result.duration_ms
        self.durations.append(result.duration_ms)

        if result.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.min_duration_ms = min(self.min_duration_ms, result.duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, result.duration_ms)

    def calculate_final_metrics(self, total_time_seconds: float):
        """Calculate final metrics after all requests complete."""
        if self.total_requests > 0:
            self.success_rate = (self.successful_requests / self.total_requests) * 100
            self.throughput_rps = self.total_requests / total_time_seconds

    def get_percentiles(self) -> dict[str, float]:
        """Get response time percentiles."""
        if not self.durations:
            return {}

        sorted_durations = sorted(self.durations)
        return {
            "p50": statistics.median(sorted_durations),
            "p95": statistics.quantiles(sorted_durations, n=20)[18]
            if len(sorted_durations) >= 20
            else max(sorted_durations),
            "p99": statistics.quantiles(sorted_durations, n=100)[98]
            if len(sorted_durations) >= 100
            else max(sorted_durations),
        }


class AsyncFlowiseClient:
    """High-performance async client for Flowise API."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        max_connections: int = 100,
        connection_timeout: int = 10,
        request_timeout: int = 30,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        self._session = None
        self._connector = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Configure connection pooling
        self._connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections,
            ttl_dns_cache=300,
            ttl_socket_reuse=30,
            enable_cleanup_closed=True,
        )

        # Configure timeouts
        timeout = aiohttp.ClientTimeout(
            total=self.request_timeout, connect=self.connection_timeout
        )

        # Set up headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._session = aiohttp.ClientSession(
            connector=self._connector, timeout=timeout, headers=headers
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
        if self._connector:
            await self._connector.close()

    async def predict_flow(
        self,
        chatflow_id: str,
        question: str,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncRequestResult:
        """Execute Flowise flow asynchronously with retry logic."""

        if not request_id:
            request_id = f"async-{int(time.time() * 1000)}"

        url = f"{self.base_url}/api/v1/prediction/{chatflow_id}"
        data = {"question": question}

        if session_id:
            data["sessionId"] = session_id

        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in ["request_id"]:
                data[key] = value

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                async with self._session.post(url, json=data) as response:
                    duration_ms = int((time.time() - start_time) * 1000)

                    if response.status == 200:
                        response_data = await response.json()

                        # Estimate tokens and cost
                        response_text = self._extract_response_text(response_data)
                        tokens_estimated = len(question.split()) + len(
                            response_text.split()
                        )
                        cost_estimated = tokens_estimated * 0.000002  # Rough estimate

                        return AsyncRequestResult(
                            request_id=request_id,
                            success=True,
                            response_data=response_data,
                            duration_ms=duration_ms,
                            tokens_estimated=tokens_estimated,
                            cost_estimated=cost_estimated,
                            retry_count=attempt,
                        )
                    else:
                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text[:200]}"

                        # Don't retry on client errors (4xx)
                        if 400 <= response.status < 500 and response.status != 429:
                            break

            except asyncio.TimeoutError:
                last_error = "Request timeout"
            except aiohttp.ClientError as e:
                last_error = f"Client error: {str(e)}"
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"

            # Exponential backoff for retries
            if attempt < self.max_retries:
                delay = min(2**attempt + 0.1, 10)  # Cap at 10 seconds
                await asyncio.sleep(delay)

        duration_ms = int((time.time() - start_time) * 1000)

        return AsyncRequestResult(
            request_id=request_id,
            success=False,
            error=last_error,
            duration_ms=duration_ms,
            retry_count=self.max_retries,
        )

    def _extract_response_text(self, response_data: dict) -> str:
        """Extract text content from response data."""
        if isinstance(response_data, dict):
            return (
                response_data.get("text", "")
                or response_data.get("answer", "")
                or response_data.get("content", "")
                or str(response_data)
            )
        return str(response_data)


class BatchProcessor:
    """Process requests in batches with concurrency control."""

    def __init__(
        self,
        client: AsyncFlowiseClient,
        max_concurrent: int = 10,
        batch_size: int = 100,
        progress_callback: Optional[callable] = None,
    ):
        self.client = client
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_requests(
        self, requests: list[dict[str, Any]]
    ) -> list[AsyncRequestResult]:
        """Process a list of requests with concurrency control."""

        results = []
        total_requests = len(requests)

        # Process requests in batches
        for batch_start in range(0, total_requests, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_requests)
            batch_requests = requests[batch_start:batch_end]

            # Process batch concurrently
            batch_tasks = [
                self._process_single_request(
                    req, f"batch-{batch_start // self.batch_size}-{i}"
                )
                for i, req in enumerate(batch_requests)
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append(
                        AsyncRequestResult(
                            request_id=f"batch-{batch_start // self.batch_size}-{i}",
                            success=False,
                            error=str(result),
                            duration_ms=0,
                        )
                    )
                else:
                    results.append(result)

            # Progress callback
            if self.progress_callback:
                progress = (batch_end / total_requests) * 100
                self.progress_callback(batch_end, total_requests, progress)

        return results

    async def _process_single_request(
        self, request_data: dict, request_id: str
    ) -> AsyncRequestResult:
        """Process a single request with semaphore control."""
        async with self.semaphore:
            return await self.client.predict_flow(request_id=request_id, **request_data)


async def demonstrate_async_performance():
    """Demonstrate high-performance async processing."""

    print("⚡ Async High-Performance Processing")
    print("=" * 50)

    if not HAS_ASYNC_DEPS:
        print("❌ Missing async dependencies. Install with:")
        print("   pip install aiohttp aiofiles")
        return False

    # Configuration
    base_url = os.getenv("FLOWISE_BASE_URL", "http://localhost:3000")
    api_key = os.getenv("FLOWISE_API_KEY")
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "10"))

    print(f"Flowise URL: {base_url}")
    print(f"Max Concurrent: {max_concurrent}")

    # Step 1: Validate setup
    print("\n📋 Step 1: Validating async setup...")

    try:
        result = validate_flowise_setup(base_url, api_key)
        if not result.is_valid:
            print("❌ Setup validation failed.")
            return False

        # Get available chatflows
        sync_flowise = instrument_flowise(base_url=base_url, api_key=api_key)
        chatflows = sync_flowise.get_chatflows()
        if not chatflows:
            print("❌ No chatflows available.")
            return False

        chatflow_id = chatflows[0].get("id")
        chatflow_name = chatflows[0].get("name", "Unnamed")
        print(f"✅ Using chatflow: {chatflow_name}")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

    # Step 2: Create test workload
    print("\n📋 Step 2: Creating test workload...")

    # Generate variety of test requests
    test_requests = []

    # Quick requests (simple questions)
    quick_questions = [
        "What is AI?",
        "How does machine learning work?",
        "Explain neural networks.",
        "What are the benefits of automation?",
        "Define data science.",
    ]

    for i, question in enumerate(quick_questions * 4):  # 20 quick requests
        test_requests.append(
            {
                "chatflow_id": chatflow_id,
                "question": question,
                "session_id": f"quick-session-{i % 5}",
            }
        )

    # Medium requests (more detailed)
    medium_questions = [
        "Explain the differences between supervised and unsupervised learning with examples.",
        "How can businesses implement AI solutions effectively?",
        "What are the key considerations for AI ethics and responsible AI development?",
        "Describe the process of training a machine learning model from data collection to deployment.",
    ]

    for i, question in enumerate(medium_questions * 3):  # 12 medium requests
        test_requests.append(
            {
                "chatflow_id": chatflow_id,
                "question": question,
                "session_id": f"medium-session-{i % 3}",
            }
        )

    # Complex requests (detailed analysis)
    complex_questions = [
        "Conduct a comprehensive analysis of how artificial intelligence is transforming healthcare, including current applications, benefits, challenges, and future prospects.",
        "Develop a strategic framework for implementing AI in enterprise environments, covering technology selection, change management, risk mitigation, and ROI measurement.",
    ]

    for i, question in enumerate(complex_questions * 4):  # 8 complex requests
        test_requests.append(
            {
                "chatflow_id": chatflow_id,
                "question": question,
                "session_id": f"complex-session-{i % 2}",
            }
        )

    print(f"✅ Created {len(test_requests)} test requests")
    print("   Quick requests: 20")
    print("   Medium requests: 12")
    print("   Complex requests: 8")

    # Step 3: Execute async processing
    print("\n⚡ Step 3: Executing async processing...")

    metrics = PerformanceMetrics()

    def progress_callback(completed: int, total: int, percent: float):
        print(f"   Progress: {completed}/{total} ({percent:.1f}%)")

    async with AsyncFlowiseClient(
        base_url=base_url,
        api_key=api_key,
        max_connections=max_concurrent * 2,
        request_timeout=60,  # Longer timeout for complex requests
    ) as client:
        processor = BatchProcessor(
            client=client,
            max_concurrent=max_concurrent,
            batch_size=20,
            progress_callback=progress_callback,
        )

        start_time = time.time()
        results = await processor.process_requests(test_requests)
        total_time = time.time() - start_time

        # Calculate metrics
        for result in results:
            metrics.add_result(result)

        metrics.calculate_final_metrics(total_time)

    # Step 4: Analyze performance results
    print("\n📊 Step 4: Performance Analysis")
    print("=" * 40)

    print("Execution Summary:")
    print(f"   Total Requests: {metrics.total_requests}")
    print(f"   Successful: {metrics.successful_requests}")
    print(f"   Failed: {metrics.failed_requests}")
    print(f"   Success Rate: {metrics.success_rate:.2f}%")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Throughput: {metrics.throughput_rps:.2f} requests/second")

    if metrics.durations:
        percentiles = metrics.get_percentiles()
        avg_duration = statistics.mean(metrics.durations)

        print("\nResponse Time Analysis:")
        print(f"   Average: {avg_duration:.0f}ms")
        print(f"   Min: {metrics.min_duration_ms}ms")
        print(f"   Max: {metrics.max_duration_ms}ms")
        print(f"   P50 (median): {percentiles.get('p50', 0):.0f}ms")
        print(f"   P95: {percentiles.get('p95', 0):.0f}ms")
        print(f"   P99: {percentiles.get('p99', 0):.0f}ms")

    # Show error analysis if there were failures
    if metrics.failed_requests > 0:
        print("\nError Analysis:")
        error_counts = {}
        for result in results:
            if not result.success and result.error:
                error_type = (
                    result.error.split(":")[0] if ":" in result.error else result.error
                )
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        for error_type, count in error_counts.items():
            print(f"   {error_type}: {count} occurrences")

    # Cost estimation
    total_estimated_cost = sum(
        result.cost_estimated for result in results if result.success
    )
    total_estimated_tokens = sum(
        result.tokens_estimated for result in results if result.success
    )

    print("\nCost Estimation:")
    print(f"   Total Tokens: {total_estimated_tokens:,}")
    print(f"   Estimated Cost: ${total_estimated_cost:.4f}")
    print(
        f"   Cost per Request: ${total_estimated_cost / max(metrics.successful_requests, 1):.6f}"
    )

    return metrics.success_rate > 80  # Consider successful if >80% success rate


async def run_performance_benchmark():
    """Run comprehensive performance benchmarks."""

    print("🏁 Performance Benchmark Suite")
    print("=" * 50)

    base_url = os.getenv("FLOWISE_BASE_URL", "http://localhost:3000")
    api_key = os.getenv("FLOWISE_API_KEY")

    # Get chatflow for testing
    sync_flowise = instrument_flowise(base_url=base_url, api_key=api_key)
    chatflows = sync_flowise.get_chatflows()
    if not chatflows:
        print("❌ No chatflows available for benchmarking.")
        return False

    chatflow_id = chatflows[0].get("id")

    # Benchmark scenarios
    scenarios = [
        {"name": "Low Concurrency", "concurrent": 5, "requests": 25},
        {"name": "Medium Concurrency", "concurrent": 15, "requests": 50},
        {"name": "High Concurrency", "concurrent": 30, "requests": 100},
    ]

    benchmark_results = []

    for scenario in scenarios:
        print(f"\n🧪 Running {scenario['name']} Benchmark:")
        print(f"   Concurrent Requests: {scenario['concurrent']}")
        print(f"   Total Requests: {scenario['requests']}")

        # Create test requests
        test_requests = []
        for i in range(scenario["requests"]):
            test_requests.append(
                {
                    "chatflow_id": chatflow_id,
                    "question": f"Test question {i}: What are the applications of AI in business?",
                    "session_id": f"benchmark-session-{i % 10}",
                }
            )

        metrics = PerformanceMetrics()

        async with AsyncFlowiseClient(
            base_url=base_url,
            api_key=api_key,
            max_connections=scenario["concurrent"] * 2,
        ) as client:
            processor = BatchProcessor(
                client=client,
                max_concurrent=scenario["concurrent"],
                batch_size=scenario["concurrent"],
            )

            start_time = time.time()
            results = await processor.process_requests(test_requests)
            total_time = time.time() - start_time

            for result in results:
                metrics.add_result(result)

            metrics.calculate_final_metrics(total_time)

        # Store results
        percentiles = metrics.get_percentiles()
        benchmark_results.append(
            {
                "scenario": scenario["name"],
                "concurrent": scenario["concurrent"],
                "total_requests": metrics.total_requests,
                "success_rate": metrics.success_rate,
                "throughput_rps": metrics.throughput_rps,
                "avg_response_time": statistics.mean(metrics.durations)
                if metrics.durations
                else 0,
                "p95_response_time": percentiles.get("p95", 0),
                "total_time": total_time,
            }
        )

        print("   ✅ Results:")
        print(f"      Success Rate: {metrics.success_rate:.1f}%")
        print(f"      Throughput: {metrics.throughput_rps:.2f} req/sec")
        print(
            f"      Avg Response Time: {statistics.mean(metrics.durations) if metrics.durations else 0:.0f}ms"
        )
        print(f"      P95 Response Time: {percentiles.get('p95', 0):.0f}ms")

    # Summary comparison
    print("\n📊 Benchmark Comparison Summary")
    print("=" * 60)

    print(
        f"{'Scenario':<20} {'Concurrent':<10} {'Success%':<10} {'RPS':<8} {'Avg(ms)':<10} {'P95(ms)':<10}"
    )
    print("-" * 60)

    for result in benchmark_results:
        print(
            f"{result['scenario']:<20} {result['concurrent']:<10} {result['success_rate']:<10.1f} "
            f"{result['throughput_rps']:<8.1f} {result['avg_response_time']:<10.0f} {result['p95_response_time']:<10.0f}"
        )

    return True


async def demonstrate_streaming_processing():
    """Demonstrate streaming and real-time processing patterns."""

    print("\n🌊 Streaming Processing Patterns")
    print("=" * 50)

    base_url = os.getenv("FLOWISE_BASE_URL", "http://localhost:3000")
    api_key = os.getenv("FLOWISE_API_KEY")

    sync_flowise = instrument_flowise(base_url=base_url, api_key=api_key)
    chatflows = sync_flowise.get_chatflows()
    if not chatflows:
        return

    chatflow_id = chatflows[0].get("id")

    async def request_generator() -> AsyncGenerator[dict[str, Any], None]:
        """Generate requests continuously (simulating real-time data)."""
        request_templates = [
            "Analyze current market trends in {topic}",
            "What are the latest developments in {topic}?",
            "How is {topic} impacting business today?",
            "Provide insights on {topic} for decision makers",
        ]

        topics = [
            "AI",
            "cloud computing",
            "cybersecurity",
            "blockchain",
            "IoT",
            "automation",
        ]

        for i in range(20):  # Generate 20 streaming requests
            template = request_templates[i % len(request_templates)]
            topic = topics[i % len(topics)]

            yield {
                "chatflow_id": chatflow_id,
                "question": template.format(topic=topic),
                "session_id": f"stream-session-{i}",
                "priority": "high" if i % 5 == 0 else "normal",
            }

            await asyncio.sleep(0.1)  # Simulate real-time arrival

    print("🔄 Processing streaming requests...")

    processed_count = 0
    start_time = time.time()

    async with AsyncFlowiseClient(
        base_url=base_url, api_key=api_key, max_connections=20
    ) as client:
        # Process requests as they arrive
        async for request_data in request_generator():
            # Process high-priority requests immediately
            if request_data.get("priority") == "high":
                print(
                    f"   🔥 High-priority request: {request_data['question'][:50]}..."
                )
                result = await client.predict_flow(**request_data)
                print(
                    f"      {'✅' if result.success else '❌'} Completed in {result.duration_ms}ms"
                )
            else:
                # Queue normal requests (simplified - in production use proper queue)
                print(f"   📋 Queued: {request_data['question'][:50]}...")
                # Simulate background processing
                asyncio.create_task(client.predict_flow(**request_data))

            processed_count += 1

    total_time = time.time() - start_time
    print(f"✅ Processed {processed_count} streaming requests in {total_time:.2f}s")
    print(f"   Stream throughput: {processed_count / total_time:.2f} requests/second")


def main():
    """Main example function."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Async High-Performance Flowise Example"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )
    args = parser.parse_args()

    async def run_examples():
        try:
            print("🚀 Async High-Performance Processing Example")
            print("=" * 60)

            if args.benchmark:
                # Run benchmarks only
                success = await run_performance_benchmark()
            else:
                # Run full demonstration
                success = await demonstrate_async_performance()

                if success:
                    # Show streaming patterns
                    await demonstrate_streaming_processing()

            if success:
                print("\n🎉 Async High-Performance Example Complete!")
                print("=" * 50)
                print("✅ You've learned how to:")
                print("   • Build high-throughput async Flowise clients")
                print("   • Implement connection pooling and concurrency control")
                print("   • Process requests in batches with error handling")
                print("   • Measure and optimize performance metrics")
                print("   • Handle streaming and real-time processing patterns")

                print("\n⚡ Performance Features Demonstrated:")
                print("   • Async/await patterns for maximum concurrency")
                print("   • Connection pooling for efficient resource usage")
                print("   • Batch processing with progress tracking")
                print("   • Comprehensive performance metrics and analysis")
                print("   • Error handling and retry logic with backoff")

                print("\n📚 Next Steps:")
                print("   • Integrate async patterns into production applications")
                print("   • Implement proper queue systems for request management")
                print("   • Set up load balancing across multiple Flowise instances")
                print("   • Monitor performance metrics in production environments")

            return success

        except Exception as e:
            logger.error(f"Example failed: {e}")
            return False

    try:
        success = asyncio.run(run_examples())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
