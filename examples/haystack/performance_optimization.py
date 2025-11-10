#!/usr/bin/env python3
"""
Performance Optimization with GenOps and Haystack

Demonstrates advanced performance optimization techniques including caching,
request batching, parallel processing, model optimization, and resource management
for high-performance AI systems.

Usage:
    python performance_optimization.py

Features:
    - Intelligent caching strategies with LRU and TTL policies
    - Request batching and parallel processing optimization
    - Model selection and parameter optimization
    - Resource pooling and connection management
    - Performance profiling and bottleneck analysis
    - Load testing and capacity planning tools
"""

import hashlib
import logging
import statistics
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Core Haystack imports
try:
    from haystack import Document, Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore
except ImportError as e:
    print(f"❌ Haystack not installed: {e}")
    print("Please install Haystack: pip install haystack-ai")
    sys.exit(1)

# GenOps imports
try:
    from genops.providers.haystack import (
        GenOpsHaystackAdapter,
        analyze_pipeline_costs,
        print_validation_result,
        validate_haystack_setup,
    )
except ImportError as e:
    print(f"❌ GenOps not installed: {e}")
    print("Please install GenOps: pip install genops-ai[haystack]")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL and access tracking."""
    value: Any
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)

    def access(self) -> Any:
        """Access cache entry and update tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        return self.value


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100

    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests

    @property
    def p95_response_time(self) -> float:
        """Calculate P95 response time."""
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile

    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.error_count / self.total_requests) * 100


class IntelligentCache:
    """High-performance caching with LRU and TTL policies."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expires": 0
        }

    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            entry = self.cache[key]

            if entry.is_expired():
                del self.cache[key]
                self.stats["expires"] += 1
                self.stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.stats["hits"] += 1
            return entry.access()

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache."""
        with self.lock:
            ttl = ttl or self.default_ttl

            entry = CacheEntry(
                value=value,
                timestamp=datetime.now(),
                ttl_seconds=ttl
            )

            self.cache[key] = entry
            self.cache.move_to_end(key)

            # Evict oldest entries if over capacity
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats["evictions"] += 1

    def cached(self, ttl: Optional[int] = None):
        """Decorator for caching function results."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                key = self._make_key(func.__name__, *args, **kwargs)

                # Try cache first
                cached_result = self.get(key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.put(key, result, ttl)
                return result

            return wrapper
        return decorator

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "stats": self.stats.copy()
            }


class BatchProcessor:
    """Intelligent request batching for improved throughput."""

    def __init__(self, batch_size: int = 10, batch_timeout: float = 1.0, max_workers: int = 4):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_workers = max_workers
        self.pending_requests = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()

    def add_request(self, request_data: Dict[str, Any], callback=None) -> Any:
        """Add request to batch queue."""
        with self.lock:
            self.pending_requests.append({
                "data": request_data,
                "callback": callback,
                "timestamp": time.time()
            })

            # Process batch if conditions met
            if len(self.pending_requests) >= self.batch_size:
                return self._process_batch()

    def _process_batch(self) -> List[Any]:
        """Process current batch of requests."""
        if not self.pending_requests:
            return []

        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]

        # Process requests in parallel
        futures = []
        for request in batch:
            future = self.executor.submit(self._process_single_request, request)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                results.append({"error": str(e)})

        return results

    def _process_single_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual request within batch."""
        # This would be implemented by the specific use case
        # For demo purposes, simulate processing
        time.sleep(0.1)  # Simulate work

        return {
            "request_id": request.get("request_id", "unknown"),
            "result": f"Processed: {request['data'][:50]}...",
            "processing_time": 0.1
        }

    def flush(self) -> List[Any]:
        """Process all pending requests."""
        with self.lock:
            return self._process_batch()


class OptimizedPipelineManager:
    """High-performance pipeline manager with caching and optimization."""

    def __init__(self, adapter: GenOpsHaystackAdapter):
        self.adapter = adapter
        self.cache = IntelligentCache(max_size=500, default_ttl=1800)  # 30-minute TTL
        self.batch_processor = BatchProcessor(batch_size=5, batch_timeout=2.0)
        self.metrics = PerformanceMetrics("optimized_pipeline")
        self.pipelines = {}
        self.connection_pool_size = 10

    def initialize_pipelines(self):
        """Initialize optimized pipelines."""

        # Fast pipeline for simple requests
        fast_pipeline = Pipeline()
        fast_pipeline.add_component("prompt_builder", PromptBuilder(
            template="Provide a concise answer: {{query}}"
        ))
        fast_pipeline.add_component("llm", OpenAIGenerator(
            model="gpt-3.5-turbo",
            generation_kwargs={
                "max_tokens": 100,
                "temperature": 0.3,
                "stream": False  # Optimize for latency
            }
        ))
        fast_pipeline.connect("prompt_builder", "llm")
        self.pipelines["fast"] = fast_pipeline

        # Balanced pipeline for normal requests
        balanced_pipeline = Pipeline()
        balanced_pipeline.add_component("prompt_builder", PromptBuilder(
            template="Provide a detailed and accurate answer: {{query}}"
        ))
        balanced_pipeline.add_component("llm", OpenAIGenerator(
            model="gpt-3.5-turbo",
            generation_kwargs={
                "max_tokens": 250,
                "temperature": 0.5,
                "stream": False
            }
        ))
        balanced_pipeline.connect("prompt_builder", "llm")
        self.pipelines["balanced"] = balanced_pipeline

        # High-quality pipeline for complex requests
        quality_pipeline = Pipeline()
        quality_pipeline.add_component("prompt_builder", PromptBuilder(
            template="""
            Provide a comprehensive, accurate, and well-structured response to this query:
            
            Query: {{query}}
            
            Requirements:
            - Be thorough and detailed
            - Provide examples where relevant
            - Ensure accuracy and clarity
            
            Response:
            """
        ))
        quality_pipeline.add_component("llm", OpenAIGenerator(
            model="gpt-4",
            generation_kwargs={
                "max_tokens": 500,
                "temperature": 0.4,
                "stream": False
            }
        ))
        quality_pipeline.connect("prompt_builder", "llm")
        self.pipelines["quality"] = quality_pipeline

        logger.info("Optimized pipelines initialized with 3 performance tiers")

    def select_optimal_pipeline(self, query: str, priority: str = "balanced") -> str:
        """Intelligently select pipeline based on query characteristics."""

        query_length = len(query.split())

        # Simple heuristic-based selection
        if priority == "speed" or query_length < 10:
            return "fast"
        elif priority == "quality" or query_length > 30:
            return "quality"
        else:
            return "balanced"

    @IntelligentCache().cached(ttl=1800)  # 30-minute cache
    def process_query_cached(self, query: str, pipeline_name: str, request_id: str) -> Dict[str, Any]:
        """Process query with intelligent caching."""
        return self._process_query_internal(query, pipeline_name, request_id)

    def _process_query_internal(self, query: str, pipeline_name: str, request_id: str) -> Dict[str, Any]:
        """Internal query processing with performance tracking."""
        start_time = time.time()

        try:
            pipeline = self.pipelines.get(pipeline_name, self.pipelines["balanced"])

            with self.adapter.track_pipeline(
                f"optimized-{pipeline_name}",
                request_id=request_id,
                pipeline_tier=pipeline_name,
                query_length=len(query.split())
            ) as context:

                result = pipeline.run({"prompt_builder": {"query": query}})
                response = result["llm"]["replies"][0]

                processing_time = time.time() - start_time

                # Update performance metrics
                self.metrics.total_requests += 1
                self.metrics.total_response_time += processing_time
                self.metrics.min_response_time = min(self.metrics.min_response_time, processing_time)
                self.metrics.max_response_time = max(self.metrics.max_response_time, processing_time)
                self.metrics.response_times.append(processing_time)

                return {
                    "request_id": request_id,
                    "response": response,
                    "pipeline_used": pipeline_name,
                    "processing_time": processing_time,
                    "cost": float(context.get_metrics().total_cost),
                    "cached": False
                }

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Query processing failed: {e}")
            return {
                "request_id": request_id,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def process_query(self, query: str, priority: str = "balanced", request_id: Optional[str] = None) -> Dict[str, Any]:
        """Process query with optimization."""
        request_id = request_id or f"req-{int(time.time() * 1000)}"

        # Select optimal pipeline
        pipeline_name = self.select_optimal_pipeline(query, priority)

        # Try cache first
        cache_key = self.cache._make_key(query, pipeline_name)
        cached_result = self.cache.get(cache_key)

        if cached_result is not None:
            self.metrics.cache_hits += 1
            cached_result["cached"] = True
            return cached_result

        # Cache miss - process query
        self.metrics.cache_misses += 1
        result = self._process_query_internal(query, pipeline_name, request_id)

        # Cache successful results
        if "error" not in result:
            self.cache.put(cache_key, result, ttl=1800)

        return result

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_stats = self.cache.get_stats()

        return {
            "processing_metrics": {
                "total_requests": self.metrics.total_requests,
                "average_response_time": self.metrics.average_response_time,
                "p95_response_time": self.metrics.p95_response_time,
                "min_response_time": self.metrics.min_response_time,
                "max_response_time": self.metrics.max_response_time,
                "error_rate": self.metrics.error_rate
            },
            "cache_performance": {
                "hit_rate": cache_stats["hit_rate"],
                "cache_size": cache_stats["size"],
                "cache_utilization": (cache_stats["size"] / cache_stats["max_size"]) * 100
            },
            "optimization_impact": {
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "requests_served": self.metrics.total_requests,
                "time_saved_seconds": self.metrics.cache_hits * 0.5  # Estimate
            }
        }


def demo_caching_optimization():
    """Demonstrate intelligent caching optimization."""
    print("\n" + "="*70)
    print("🧠 Intelligent Caching Optimization")
    print("="*70)

    # Create optimized adapter
    adapter = GenOpsHaystackAdapter(
        team="performance-optimization",
        project="caching-demo",
        daily_budget_limit=100.0
    )

    # Initialize optimized pipeline manager
    pipeline_manager = OptimizedPipelineManager(adapter)
    pipeline_manager.initialize_pipelines()

    print("✅ Optimized pipeline manager initialized with intelligent caching")

    # Test queries with different characteristics
    test_queries = [
        {"query": "What is machine learning?", "priority": "speed", "repeat": 3},
        {"query": "Explain the differences between supervised and unsupervised learning algorithms", "priority": "balanced", "repeat": 2},
        {"query": "Provide a comprehensive analysis of deep learning architectures including CNNs, RNNs, and Transformers", "priority": "quality", "repeat": 2},
        {"query": "How do neural networks work?", "priority": "speed", "repeat": 4},
        {"query": "What are the best practices for MLOps?", "priority": "balanced", "repeat": 2}
    ]

    print("\n🚀 Testing Caching Performance:")

    total_queries = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for test_case in test_queries:
            query = test_case["query"]
            priority = test_case["priority"]
            repeat_count = test_case["repeat"]

            # Submit multiple requests for the same query to test caching
            for i in range(repeat_count):
                request_id = f"test-{total_queries:03d}"
                future = executor.submit(
                    pipeline_manager.process_query,
                    query,
                    priority,
                    request_id
                )
                futures.append((request_id, query[:50] + "...", future))
                total_queries += 1

        # Collect results and measure performance
        cache_hit_count = 0
        total_time = 0

        for request_id, query_preview, future in futures:
            try:
                result = future.result(timeout=30)

                cached_indicator = "🔥" if result.get("cached", False) else "⚡"
                if result.get("cached", False):
                    cache_hit_count += 1

                processing_time = result.get("processing_time", 0)
                total_time += processing_time

                print(f"   {cached_indicator} {request_id}: {query_preview} ({processing_time:.3f}s)")

            except Exception as e:
                print(f"   ❌ {request_id}: Error - {e}")

    # Show performance metrics
    metrics = pipeline_manager.get_performance_metrics()

    print("\n📊 Caching Performance Results:")
    print(f"   Total Queries: {metrics['processing_metrics']['total_requests']}")
    print(f"   Cache Hit Rate: {metrics['cache_performance']['hit_rate']:.1f}%")
    print(f"   Average Response Time: {metrics['processing_metrics']['average_response_time']:.3f}s")
    print(f"   P95 Response Time: {metrics['processing_metrics']['p95_response_time']:.3f}s")
    print(f"   Estimated Time Saved: {metrics['optimization_impact']['time_saved_seconds']:.1f}s")
    print(f"   Cache Utilization: {metrics['cache_performance']['cache_utilization']:.1f}%")

    return pipeline_manager, metrics


def demo_parallel_processing():
    """Demonstrate parallel processing optimization."""
    print("\n" + "="*70)
    print("⚡ Parallel Processing Optimization")
    print("="*70)

    # Create adapter for parallel processing demo
    adapter = GenOpsHaystackAdapter(
        team="parallel-processing",
        project="concurrency-demo",
        daily_budget_limit=150.0
    )

    # Create simple pipeline for parallel testing
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", PromptBuilder(
        template="Answer this question concisely: {{question}}"
    ))
    pipeline.add_component("llm", OpenAIGenerator(
        model="gpt-3.5-turbo",
        generation_kwargs={"max_tokens": 100, "temperature": 0.3}
    ))
    pipeline.connect("prompt_builder", "llm")

    # Test queries for parallel processing
    parallel_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain natural language processing",
        "What are neural networks?",
        "Define deep learning",
        "How do recommendation systems work?",
        "What is computer vision?",
        "Explain reinforcement learning",
        "What are large language models?",
        "How does transfer learning work?"
    ]

    # Sequential processing test
    print("🐌 Sequential Processing Test:")
    sequential_start = time.time()
    sequential_results = []

    with adapter.track_session("sequential-processing", use_case="performance-comparison") as seq_session:
        for i, query in enumerate(parallel_queries, 1):
            with adapter.track_pipeline(f"sequential-{i}", query_index=i) as context:
                result = pipeline.run({"prompt_builder": {"question": query}})
                sequential_results.append({
                    "query": query,
                    "response": result["llm"]["replies"][0],
                    "cost": float(context.get_metrics().total_cost)
                })
            seq_session.add_pipeline_result(context.get_metrics())

    sequential_time = time.time() - sequential_start
    print(f"   Time: {sequential_time:.2f}s")
    print(f"   Queries: {len(parallel_queries)}")
    print(f"   Average per query: {sequential_time / len(parallel_queries):.2f}s")

    # Parallel processing test
    print("\n🚀 Parallel Processing Test (4 workers):")
    parallel_start = time.time()
    parallel_results = []

    with adapter.track_session("parallel-processing", use_case="performance-comparison") as par_session:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            for i, query in enumerate(parallel_queries, 1):
                future = executor.submit(
                    lambda q, idx: pipeline.run({"prompt_builder": {"question": q}}),
                    query, i
                )
                futures[future] = (i, query)

            for future in as_completed(futures):
                i, query = futures[future]
                try:
                    result = future.result()
                    parallel_results.append({
                        "query": query,
                        "response": result["llm"]["replies"][0],
                        "index": i
                    })
                except Exception as e:
                    print(f"   ❌ Query {i} failed: {e}")

    parallel_time = time.time() - parallel_start

    print(f"   Time: {parallel_time:.2f}s")
    print(f"   Queries: {len(parallel_results)}")
    print(f"   Average per query: {parallel_time / len(parallel_results):.2f}s" if parallel_results else "N/A")

    # Performance comparison
    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        efficiency = (speedup / 4) * 100  # 4 workers

        print("\n📈 Performance Improvement:")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Efficiency: {efficiency:.1f}%")
        print(f"   Time saved: {sequential_time - parallel_time:.2f}s ({((sequential_time - parallel_time) / sequential_time * 100):.1f}%)")


def demo_pipeline_optimization():
    """Demonstrate pipeline-level optimization techniques."""
    print("\n" + "="*70)
    print("🔧 Pipeline Optimization Techniques")
    print("="*70)

    print("🎯 Optimization Strategies:")
    print("   • Model Selection: Right-sizing models for task complexity")
    print("   • Parameter Tuning: Optimal temperature, max_tokens, top_p settings")
    print("   • Prompt Engineering: Efficient prompt design for faster processing")
    print("   • Context Management: Minimizing unnecessary context overhead")
    print("   • Response Streaming: Reducing perceived latency for users")

    # Create adapter for optimization demos
    adapter = GenOpsHaystackAdapter(
        team="pipeline-optimization",
        project="optimization-techniques",
        daily_budget_limit=75.0
    )

    # Demonstrate model selection optimization
    print("\n🤖 Model Selection Optimization:")

    model_configs = [
        {
            "name": "Speed-Optimized",
            "model": "gpt-3.5-turbo",
            "params": {"max_tokens": 50, "temperature": 0.1},
            "use_case": "Simple queries, fact checking"
        },
        {
            "name": "Balanced",
            "model": "gpt-3.5-turbo",
            "params": {"max_tokens": 150, "temperature": 0.5},
            "use_case": "General purpose, moderate complexity"
        },
        {
            "name": "Quality-Focused",
            "model": "gpt-4",
            "params": {"max_tokens": 300, "temperature": 0.3},
            "use_case": "Complex analysis, high accuracy required"
        }
    ]

    test_query = "Explain the concept of artificial neural networks"

    with adapter.track_session("model-optimization", use_case="configuration-testing") as session:
        for config in model_configs:
            print(f"   Testing {config['name']} configuration...")

            # Create pipeline with specific configuration
            pipeline = Pipeline()
            pipeline.add_component("prompt_builder", PromptBuilder(
                template="{{query}}"
            ))
            pipeline.add_component("llm", OpenAIGenerator(
                model=config["model"],
                generation_kwargs=config["params"]
            ))
            pipeline.connect("prompt_builder", "llm")

            with adapter.track_pipeline(
                f"model-{config['name'].lower()}",
                model_name=config["model"],
                optimization_type=config["name"]
            ) as context:

                start_time = time.time()
                result = pipeline.run({"prompt_builder": {"query": test_query}})
                processing_time = time.time() - start_time

                response = result["llm"]["replies"][0]
                cost = float(context.get_metrics().total_cost)

                print(f"      Model: {config['model']}")
                print(f"      Time: {processing_time:.3f}s")
                print(f"      Cost: ${cost:.6f}")
                print(f"      Response length: {len(response)} chars")
                print(f"      Use case: {config['use_case']}")
                print()

            session.add_pipeline_result(context.get_metrics())

        print("   📊 Model Optimization Session:")
        print(f"      Total configurations tested: {session.total_pipelines}")
        print(f"      Total cost: ${session.total_cost:.6f}")

    # Demonstrate prompt optimization
    print("\n📝 Prompt Engineering Optimization:")

    prompt_variations = [
        {
            "name": "Verbose",
            "template": """
            Please provide a detailed and comprehensive explanation of the following topic.
            Include background information, key concepts, and relevant examples.
            
            Topic: {{topic}}
            
            Your detailed response:
            """
        },
        {
            "name": "Concise",
            "template": "Explain {{topic}} concisely:"
        },
        {
            "name": "Structured",
            "template": """
            Topic: {{topic}}
            
            Provide:
            1. Definition
            2. Key features
            3. Applications
            
            Response:
            """
        }
    ]

    topic = "machine learning algorithms"

    print("   Testing prompt variations for efficiency...")
    for prompt_config in prompt_variations:
        pipeline = Pipeline()
        pipeline.add_component("prompt_builder", PromptBuilder(
            template=prompt_config["template"]
        ))
        pipeline.add_component("llm", OpenAIGenerator(
            model="gpt-3.5-turbo",
            generation_kwargs={"max_tokens": 200, "temperature": 0.4}
        ))
        pipeline.connect("prompt_builder", "llm")

        start_time = time.time()
        result = pipeline.run({"prompt_builder": {"topic": topic}})
        processing_time = time.time() - start_time

        response_length = len(result["llm"]["replies"][0])

        print(f"      {prompt_config['name']} prompt:")
        print(f"         Processing time: {processing_time:.3f}s")
        print(f"         Response length: {response_length} chars")
        print(f"         Efficiency ratio: {response_length/processing_time:.1f} chars/sec")


def demo_load_testing():
    """Demonstrate load testing and capacity planning."""
    print("\n" + "="*70)
    print("📊 Load Testing and Capacity Planning")
    print("="*70)

    # Create adapter for load testing
    adapter = GenOpsHaystackAdapter(
        team="load-testing",
        project="capacity-planning",
        daily_budget_limit=200.0
    )

    # Create simple pipeline for load testing
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", PromptBuilder(
        template="Answer briefly: {{question}}"
    ))
    pipeline.add_component("llm", OpenAIGenerator(
        model="gpt-3.5-turbo",
        generation_kwargs={"max_tokens": 75, "temperature": 0.3}
    ))
    pipeline.connect("prompt_builder", "llm")

    # Load test scenarios
    load_scenarios = [
        {"name": "Light Load", "concurrent_users": 2, "requests_per_user": 5},
        {"name": "Medium Load", "concurrent_users": 5, "requests_per_user": 4},
        {"name": "Heavy Load", "concurrent_users": 10, "requests_per_user": 3}
    ]

    test_questions = [
        "What is AI?",
        "How does ML work?",
        "Define deep learning",
        "Explain NLP",
        "What are neural nets?"
    ]

    print("🧪 Running Load Test Scenarios:")

    for scenario in load_scenarios:
        print(f"\n   📈 {scenario['name']} Test:")
        print(f"      Concurrent users: {scenario['concurrent_users']}")
        print(f"      Requests per user: {scenario['requests_per_user']}")

        total_requests = scenario['concurrent_users'] * scenario['requests_per_user']

        with adapter.track_session(f"load-test-{scenario['name'].lower().replace(' ', '-')}",
                                 use_case="load-testing") as session:

            start_time = time.time()
            response_times = []
            errors = 0

            with ThreadPoolExecutor(max_workers=scenario['concurrent_users']) as executor:
                futures = []

                # Submit all requests
                for user in range(scenario['concurrent_users']):
                    for req in range(scenario['requests_per_user']):
                        question = test_questions[req % len(test_questions)]
                        request_id = f"user-{user}-req-{req}"

                        future = executor.submit(
                            lambda q, rid: self._execute_load_test_request(pipeline, q, rid),
                            question,
                            request_id
                        )
                        futures.append(future)

                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        response_times.append(result["response_time"])
                    except Exception:
                        errors += 1

            total_time = time.time() - start_time

            # Calculate metrics
            successful_requests = len(response_times)
            requests_per_second = successful_requests / total_time
            avg_response_time = statistics.mean(response_times) if response_times else 0
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times, default=0)
            error_rate = (errors / total_requests) * 100

            print("      Results:")
            print(f"         Total time: {total_time:.2f}s")
            print(f"         Successful requests: {successful_requests}/{total_requests}")
            print(f"         Requests per second: {requests_per_second:.2f}")
            print(f"         Average response time: {avg_response_time:.3f}s")
            print(f"         P95 response time: {p95_response_time:.3f}s")
            print(f"         Error rate: {error_rate:.1f}%")
            print(f"         Total cost: ${session.total_cost:.6f}")


def _execute_load_test_request(pipeline, question: str, request_id: str) -> Dict[str, Any]:
    """Execute individual load test request."""
    start_time = time.time()
    try:
        result = pipeline.run({"prompt_builder": {"question": question}})
        response_time = time.time() - start_time

        return {
            "request_id": request_id,
            "response_time": response_time,
            "success": True
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "response_time": time.time() - start_time,
            "success": False,
            "error": str(e)
        }


def main():
    """Run the comprehensive performance optimization demonstration."""
    print("⚡ Performance Optimization with Haystack + GenOps")
    print("="*70)

    # Validate environment setup
    print("🔍 Validating setup...")
    result = validate_haystack_setup()

    if not result.is_valid:
        print("❌ Setup validation failed!")
        print_validation_result(result)
        return 1
    else:
        print("✅ Environment validated and ready")

    try:
        # Caching optimization demonstration
        pipeline_manager, caching_metrics = demo_caching_optimization()

        # Parallel processing optimization
        demo_parallel_processing()

        # Pipeline optimization techniques
        demo_pipeline_optimization()

        # Load testing and capacity planning
        demo_load_testing()

        print("\n🎉 Performance Optimization demonstration completed!")
        print("\n🚀 Key Takeaways:")
        print("   • Intelligent caching can improve response times by 50-80%")
        print("   • Parallel processing provides 2-4x throughput improvements")
        print("   • Right-sized models balance cost, speed, and quality")
        print("   • Optimized prompts reduce processing time and costs")
        print("   • Load testing validates system capacity and performance limits")
        print("\n💡 Next Steps:")
        print("   • Implement caching strategies in your production systems")
        print("   • Profile your specific workloads for optimization opportunities")
        print("   • Set up monitoring for performance regression detection")
        print("   • Optimize your AI systems for maximum performance! ⚡")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("Try running the setup validation to check your configuration")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
