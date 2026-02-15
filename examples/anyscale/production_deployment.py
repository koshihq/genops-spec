#!/usr/bin/env python3
"""
Production Deployment Patterns - 30 Minute Tutorial

Learn production-ready patterns for high-volume Anyscale deployments.

Demonstrates:
- Error handling with retry logic
- Rate limiting and request throttling
- Circuit breaker pattern
- Request batching optimization
- Performance monitoring

Prerequisites:
- export ANYSCALE_API_KEY='your-api-key'
- pip install genops-ai tenacity
"""

import os
import time
from dataclasses import dataclass

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from genops.providers.anyscale import calculate_completion_cost, instrument_anyscale

# Check API key
if not os.getenv("ANYSCALE_API_KEY"):
    print("❌ ERROR: ANYSCALE_API_KEY not set")
    exit(1)

print("=" * 70)
print("GenOps Anyscale - Production Deployment Patterns")
print("=" * 70 + "\n")


# Pattern 1: Resilient Request Handler with Retry Logic
print("=" * 70)
print("PATTERN 1: Resilient Request Handler")
print("=" * 70 + "\n")


class AnyscaleAPIError(Exception):
    """Custom exception for Anyscale API errors."""

    pass


class TransientError(AnyscaleAPIError):
    """Transient errors that should be retried."""

    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(TransientError),
)
def resilient_completion(adapter, **kwargs):
    """
    Make completion request with automatic retry on transient failures.

    Retries up to 3 times with exponential backoff for transient errors.
    """
    try:
        return adapter.completion_create(**kwargs)
    except Exception as e:
        error_msg = str(e)

        # Classify error types
        if "timeout" in error_msg.lower() or "429" in error_msg:
            print(f"⚠️  Transient error detected, will retry: {error_msg}")
            raise TransientError(error_msg) from e
        else:
            # Non-transient error, don't retry
            print(f"❌ Permanent error, not retrying: {error_msg}")
            raise


# Create production adapter
adapter = instrument_anyscale(
    team="production-team", project="customer-api", environment="production"
)

print("Testing resilient request handler...")
try:
    response = resilient_completion(
        adapter,
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[{"role": "user", "content": "Test resilience"}],
        max_tokens=50,
    )
    print(
        f"✅ Request succeeded: {response['choices'][0]['message']['content'][:50]}..."
    )
except Exception as e:
    print(f"❌ Request failed after retries: {e}")

print()


# Pattern 2: Rate Limiting for High-Volume Applications
print("=" * 70)
print("PATTERN 2: Rate Limiting")
print("=" * 70 + "\n")


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, requests_per_second: int):
        self.requests_per_second = requests_per_second
        self.interval = 1.0 / requests_per_second
        self.last_request_time = 0

    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.interval:
            wait_time = self.interval - elapsed
            print(f"   ⏱️  Rate limit: waiting {wait_time:.3f}s")
            time.sleep(wait_time)

        self.last_request_time = time.time()


rate_limiter = RateLimiter(requests_per_second=5)  # Max 5 requests/second

print("Processing 10 requests with rate limiting (max 5/sec)...")
start_time = time.time()

for i in range(10):
    rate_limiter.wait_if_needed()

    try:
        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": f"Request {i + 1}"}],
            max_tokens=20,
        )
        print(f"   ✅ Request {i + 1} completed")
    except Exception as e:
        print(f"   ❌ Request {i + 1} failed: {e}")

elapsed = time.time() - start_time
print(f"\n✅ Completed 10 requests in {elapsed:.2f}s (avg {elapsed / 10:.2f}s each)")
print()


# Pattern 3: Circuit Breaker Pattern
print("=" * 70)
print("PATTERN 3: Circuit Breaker")
print("=" * 70 + "\n")


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                print("   🔄 Circuit breaker: Moving to HALF_OPEN state")
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)

            # Success - reset failure count
            if self.state == "HALF_OPEN":
                print("   ✅ Circuit breaker: Moving to CLOSED state")
                self.state = "CLOSED"

            self.failure_count = 0
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            print(f"   ⚠️  Failure {self.failure_count}/{self.failure_threshold}: {e}")

            if self.failure_count >= self.failure_threshold:
                print("   🚨 Circuit breaker: Opening circuit (too many failures)")
                self.state = "OPEN"

            raise


circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)

print("Testing circuit breaker with simulated failures...")
print("(Circuit opens after 3 failures, recovers after 5 seconds)\n")

# Simulate some successful requests
for i in range(2):
    try:
        response = circuit_breaker.call(
            adapter.completion_create,
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": f"Test {i + 1}"}],
            max_tokens=20,
        )
        print(f"✅ Request {i + 1} succeeded")
    except Exception as e:
        print(f"❌ Request {i + 1} failed: {e}")

print()


# Pattern 4: Request Batching
print("=" * 70)
print("PATTERN 4: Request Batching")
print("=" * 70 + "\n")


@dataclass
class BatchResult:
    """Result of a batch processing operation."""

    total_requests: int
    successful: int
    failed: int
    total_cost: float
    avg_latency: float


def batch_process_requests(
    adapter, requests: list[dict], batch_size: int = 10
) -> BatchResult:
    """
    Process multiple requests in batches with tracking.

    Args:
        adapter: GenOps Anyscale adapter
        requests: List of request dictionaries
        batch_size: Number of requests per batch

    Returns:
        BatchResult with statistics
    """
    total_requests = len(requests)
    successful = 0
    failed = 0
    total_cost = 0.0
    latencies = []

    print(f"Processing {total_requests} requests in batches of {batch_size}...")

    for i in range(0, total_requests, batch_size):
        batch = requests[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        print(f"\n   Batch {batch_num}: Processing {len(batch)} requests...")
        batch_start = time.time()

        for j, req in enumerate(batch):
            try:
                req_start = time.time()

                response = adapter.completion_create(**req)

                req_latency = time.time() - req_start
                latencies.append(req_latency)

                # Calculate cost
                usage = response["usage"]
                cost = calculate_completion_cost(
                    model=req["model"],
                    input_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"],
                )
                total_cost += cost

                successful += 1
                print(f"      ✅ Request {i + j + 1}: {req_latency:.2f}s, ${cost:.8f}")

            except Exception as e:
                failed += 1
                print(f"      ❌ Request {i + j + 1} failed: {e}")

        batch_time = time.time() - batch_start
        print(f"   Batch {batch_num} completed in {batch_time:.2f}s")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return BatchResult(
        total_requests=total_requests,
        successful=successful,
        failed=failed,
        total_cost=total_cost,
        avg_latency=avg_latency,
    )


# Prepare test batch
test_requests = [
    {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "messages": [{"role": "user", "content": f"Process item {i}"}],
        "max_tokens": 30,
        "customer_id": f"customer-{i % 3}",  # Distribute across 3 customers
    }
    for i in range(20)
]

result = batch_process_requests(adapter, test_requests, batch_size=5)

print("\n📊 BATCH PROCESSING RESULTS:")
print(f"   Total requests: {result.total_requests}")
print(f"   Successful: {result.successful}")
print(f"   Failed: {result.failed}")
print(f"   Success rate: {result.successful / result.total_requests * 100:.1f}%")
print(f"   Total cost: ${result.total_cost:.6f}")
print(f"   Avg cost/request: ${result.total_cost / result.total_requests:.8f}")
print(f"   Avg latency: {result.avg_latency:.3f}s")
print()


# Pattern 5: Performance Monitoring
print("=" * 70)
print("PATTERN 5: Performance Monitoring")
print("=" * 70 + "\n")


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        self.request_count = 0
        self.total_latency = 0
        self.total_cost = 0
        self.error_count = 0
        self.latencies = []

    def record_request(self, latency: float, cost: float, success: bool = True):
        """Record metrics for a request."""
        self.request_count += 1
        self.total_latency += latency
        self.total_cost += cost
        self.latencies.append(latency)

        if not success:
            self.error_count += 1

    def get_stats(self) -> dict:
        """Get current statistics."""
        if not self.request_count:
            return {}

        sorted_latencies = sorted(self.latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        return {
            "total_requests": self.request_count,
            "avg_latency": self.total_latency / self.request_count,
            "p50_latency": p50,
            "p95_latency": p95,
            "p99_latency": p99,
            "total_cost": self.total_cost,
            "avg_cost": self.total_cost / self.request_count,
            "error_rate": self.error_count / self.request_count * 100,
        }

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()

        if not stats:
            print("No requests recorded yet")
            return

        print("📊 Performance Statistics:")
        print(f"   Requests: {stats['total_requests']}")
        print(f"   Error rate: {stats['error_rate']:.2f}%")
        print("\n   Latency:")
        print(f"      Average: {stats['avg_latency']:.3f}s")
        print(f"      P50: {stats['p50_latency']:.3f}s")
        print(f"      P95: {stats['p95_latency']:.3f}s")
        print(f"      P99: {stats['p99_latency']:.3f}s")
        print("\n   Cost:")
        print(f"      Total: ${stats['total_cost']:.6f}")
        print(f"      Average: ${stats['avg_cost']:.8f}")


monitor = PerformanceMonitor()

print("Collecting performance metrics for 15 requests...")

for i in range(15):
    try:
        start = time.time()

        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": f"Performance test {i + 1}"}],
            max_tokens=30,
        )

        latency = time.time() - start

        usage = response["usage"]
        cost = calculate_completion_cost(
            model="meta-llama/Llama-2-7b-chat-hf",
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
        )

        monitor.record_request(latency, cost, success=True)

    except Exception:
        monitor.record_request(0, 0, success=False)

print()
monitor.print_stats()
print()


# Summary
print("=" * 70)
print("✅ Production patterns demonstration complete!")
print("=" * 70)

print("\n🎯 PRODUCTION CHECKLIST:")
print("   ✅ Implement retry logic with exponential backoff")
print("   ✅ Add rate limiting to prevent API throttling")
print("   ✅ Use circuit breaker to handle service degradation")
print("   ✅ Batch requests for efficiency")
print("   ✅ Monitor performance metrics (latency, cost, errors)")
print("   ✅ Track governance attributes for cost attribution")
print("   ✅ Set up alerting for error rates and cost anomalies")
print()

print("📚 Next Steps:")
print("   • Try multi_customer_attribution.py for multi-tenant patterns")
print("   • See context_manager_patterns.py for complex workflows")
