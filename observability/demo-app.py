#!/usr/bin/env python3
"""
GenOps AI Demo Application with Full Observability

This demo application showcases GenOps AI with a complete observability stack:
- FastAPI web service with AI endpoints
- OpenTelemetry tracing, metrics, and logging
- Redis for caching and session management
- Prometheus metrics endpoint
- Full integration with Grafana/Tempo/Loki/Mimir stack
"""

import json
import logging
import os
import random
import time
from typing import Any

import redis
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException

# OpenTelemetry setup
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from genops.core.policy import PolicyResult, _policy_engine, register_policy

# GenOps AI imports
from genops.core.telemetry import GenOpsTelemetry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry
otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
service_name = os.getenv("OTEL_SERVICE_NAME", "genops-demo")

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer_provider = trace.get_tracer_provider()
otlp_trace_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))

# Setup metrics
otlp_metric_exporter = OTLPMetricExporter(endpoint=f"{otlp_endpoint}/v1/metrics")
metric_reader = PeriodicExportingMetricReader(otlp_metric_exporter, export_interval_millis=5000)
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# Get tracer and meter
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Prometheus metrics
request_count = Counter('genops_demo_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('genops_demo_request_duration_seconds', 'Request duration')
ai_operations = Counter('genops_demo_ai_operations_total', 'AI operations', ['provider', 'model', 'team'])
active_sessions = Gauge('genops_demo_active_sessions', 'Active user sessions')

# FastAPI app
app = FastAPI(
    title="GenOps AI Demo",
    description="Demo application showcasing GenOps AI with full observability",
    version="1.0.0"
)

# Initialize Redis
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# Initialize GenOps telemetry
genops_telemetry = GenOpsTelemetry()

# Register governance policies
register_policy(
    name="cost_limit_demo",
    enforcement_level=PolicyResult.WARNING,
    conditions={"max_cost": 1.0}
)

register_policy(
    name="content_safety_demo",
    enforcement_level=PolicyResult.BLOCKED,
    conditions={"blocked_patterns": ["violence", "hate", "explicit"]}
)

# Instrument FastAPI and Redis
FastAPIInstrumentor.instrument_app(app)
RedisInstrumentor().instrument()


class MockAIProvider:
    """Mock AI provider that simulates real AI API calls with realistic costs and latencies"""

    MODELS = {
        "gpt-3.5-turbo": {"cost_per_token": 0.0000015, "avg_latency": 0.8},
        "gpt-4": {"cost_per_token": 0.00003, "avg_latency": 2.1},
        "claude-3-sonnet": {"cost_per_token": 0.000003, "avg_latency": 1.2},
        "claude-3-opus": {"cost_per_token": 0.000075, "avg_latency": 3.2}
    }

    @classmethod
    def simulate_ai_call(cls, model: str, prompt: str, max_tokens: int = 150) -> dict[str, Any]:
        """Simulate an AI API call with realistic behavior"""
        model_config = cls.MODELS.get(model, cls.MODELS["gpt-3.5-turbo"])

        # Simulate latency
        latency = random.uniform(model_config["avg_latency"] * 0.7, model_config["avg_latency"] * 1.3)
        time.sleep(latency)

        # Calculate tokens and cost
        prompt_tokens = len(prompt.split()) * 1.3  # Rough token estimate
        completion_tokens = min(max_tokens, random.randint(20, max_tokens))
        total_tokens = prompt_tokens + completion_tokens

        cost = total_tokens * model_config["cost_per_token"]

        return {
            "response": f"Mock AI response for: {prompt[:50]}...",
            "model": model,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
            "cost": round(cost, 6),
            "latency": round(latency, 2)
        }


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add metrics and tracing to all requests"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    request_duration.observe(process_time)
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "GenOps AI Demo is running!", "service": service_name}


@app.get("/health")
async def health():
    """Detailed health check with dependencies"""
    try:
        # Test Redis connection
        redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy",
        "service": service_name,
        "dependencies": {
            "redis": redis_status,
            "otel_endpoint": otlp_endpoint
        }
    }


@app.post("/ai/chat")
async def ai_chat(request: dict[str, Any], background_tasks: BackgroundTasks):
    """AI chat endpoint with full governance tracking"""

    # Extract request parameters
    prompt = request.get("message", "")
    model = request.get("model", "gpt-3.5-turbo")
    team = request.get("team", "demo-team")
    customer_id = request.get("customer_id", "demo-customer")
    max_tokens = request.get("max_tokens", 150)

    if not prompt:
        raise HTTPException(status_code=400, detail="Message is required")

    # Start GenOps telemetry tracking
    with genops_telemetry.trace_operation(
        operation_name="ai_chat",
        team=team,
        project="demo-app",
        customer_id=customer_id,
        feature="chat"
    ) as span:

        try:
            # Policy evaluation before operation
            estimated_tokens = len(prompt.split()) * 2
            estimated_cost = estimated_tokens * MockAIProvider.MODELS.get(model, MockAIProvider.MODELS["gpt-3.5-turbo"])["cost_per_token"]

            # Check cost policy
            cost_policy_result = _policy_engine.evaluate_policy(
                "cost_limit_demo",
                {"cost": estimated_cost}
            )

            # Check content safety policy
            content_policy_result = _policy_engine.evaluate_policy(
                "content_safety_demo",
                {"content": prompt}
            )

            # Record policy evaluations
            genops_telemetry.record_policy(span, "cost_limit_demo", cost_policy_result.result.value, cost_policy_result.reason)
            genops_telemetry.record_policy(span, "content_safety_demo", content_policy_result.result.value, content_policy_result.reason)

            # Block if content policy failed
            if content_policy_result.result == PolicyResult.BLOCKED:
                raise HTTPException(
                    status_code=400,
                    detail=f"Content policy violation: {content_policy_result.reason}"
                )

            # Simulate AI call
            ai_result = MockAIProvider.simulate_ai_call(model, prompt, max_tokens)

            # Record comprehensive telemetry
            genops_telemetry.record_cost(
                span=span,
                cost=ai_result["cost"],
                currency="USD",
                provider="demo-provider",
                model=model
            )

            genops_telemetry.record_tokens(
                span=span,
                prompt_tokens=ai_result["prompt_tokens"],
                completion_tokens=ai_result["completion_tokens"],
                total_tokens=ai_result["total_tokens"]
            )

            # Simulate quality evaluation
            quality_score = random.uniform(0.7, 0.95)
            genops_telemetry.record_evaluation(
                span=span,
                metric_name="response_quality",
                score=quality_score,
                threshold=0.8,
                passed=quality_score > 0.8
            )

            # Update Prometheus metrics
            ai_operations.labels(
                provider="demo-provider",
                model=model,
                team=team
            ).inc()

            # Cache the result
            cache_key = f"chat:{customer_id}:{hash(prompt)}"
            background_tasks.add_task(
                redis_client.setex,
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(ai_result)
            )

            logger.info(f"AI chat completed - Team: {team}, Customer: {customer_id}, Cost: ${ai_result['cost']:.6f}")

            return {
                "response": ai_result["response"],
                "metadata": {
                    "model": model,
                    "tokens_used": ai_result["total_tokens"],
                    "cost": ai_result["cost"],
                    "latency": ai_result["latency"],
                    "quality_score": quality_score,
                    "policies": {
                        "cost_check": cost_policy_result.result.value,
                        "content_safety": content_policy_result.result.value
                    }
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"AI chat failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")


@app.post("/ai/analyze")
async def ai_analyze(request: dict[str, Any]):
    """AI analysis endpoint for document/data processing"""

    content = request.get("content", "")
    analysis_type = request.get("type", "general")
    team = request.get("team", "data-team")
    customer_id = request.get("customer_id", "demo-customer")

    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    with genops_telemetry.trace_operation(
        operation_name="ai_analysis",
        team=team,
        project="demo-app",
        customer_id=customer_id,
        feature="analysis"
    ) as span:

        # Use a more expensive model for analysis
        model = "gpt-4" if analysis_type == "complex" else "gpt-3.5-turbo"
        ai_result = MockAIProvider.simulate_ai_call(model, f"Analyze: {content}", max_tokens=300)

        # Record telemetry
        genops_telemetry.record_cost(span, ai_result["cost"], "USD", "openai", model)
        genops_telemetry.record_tokens(span, ai_result["prompt_tokens"], ai_result["completion_tokens"], ai_result["total_tokens"])

        # Simulate confidence score
        confidence = random.uniform(0.6, 0.9)
        genops_telemetry.record_evaluation(span, "analysis_confidence", confidence, 0.7, confidence > 0.7)

        ai_operations.labels(provider="openai", model=model, team=team).inc()

        return {
            "analysis": ai_result["response"],
            "confidence": confidence,
            "metadata": {
                "model": model,
                "cost": ai_result["cost"],
                "tokens": ai_result["total_tokens"]
            }
        }


@app.get("/metrics/dashboard")
async def metrics_dashboard():
    """Get aggregated metrics for dashboard display"""

    # Get some Redis stats
    try:
        redis_info = redis_client.info()
        connected_clients = redis_info.get('connected_clients', 0)
        active_sessions.set(connected_clients)
    except:
        connected_clients = 0

    # Simulate getting metrics from the telemetry system
    return {
        "active_sessions": connected_clients,
        "ai_operations_today": random.randint(100, 500),
        "total_cost_today": round(random.uniform(10, 50), 2),
        "policy_violations_today": random.randint(0, 5),
        "top_models": [
            {"model": "gpt-3.5-turbo", "usage": 65},
            {"model": "gpt-4", "usage": 25},
            {"model": "claude-3-sonnet", "usage": 10}
        ]
    }


@app.post("/simulate/load")
async def simulate_load(request: dict[str, Any]):
    """Simulate load for testing observability stack"""

    operations = request.get("operations", 10)
    teams = ["engineering", "product", "support", "data-science"]
    customers = ["enterprise-1", "startup-2", "mid-market-3"]
    models = list(MockAIProvider.MODELS.keys())

    results = []

    for i in range(operations):
        team = random.choice(teams)
        customer = random.choice(customers)
        model = random.choice(models)

        with genops_telemetry.trace_operation(
            operation_name=f"load_test_op_{i}",
            team=team,
            customer_id=customer,
            feature="load_testing"
        ) as span:

            prompt = f"Load test operation {i} for {team} team"
            ai_result = MockAIProvider.simulate_ai_call(model, prompt)

            genops_telemetry.record_cost(span, ai_result["cost"], "USD", "demo", model)
            genops_telemetry.record_tokens(span, ai_result["prompt_tokens"], ai_result["completion_tokens"], ai_result["total_tokens"])

            ai_operations.labels(provider="demo", model=model, team=team).inc()

            results.append({
                "operation": i,
                "team": team,
                "customer": customer,
                "model": model,
                "cost": ai_result["cost"]
            })

    total_cost = sum(r["cost"] for r in results)

    return {
        "message": f"Simulated {operations} AI operations",
        "total_cost": round(total_cost, 4),
        "operations": results
    }


if __name__ == "__main__":
    # Start Prometheus metrics server on a separate port
    start_http_server(8001)
    logger.info("Started Prometheus metrics server on port 8001")

    # Start the main FastAPI server
    logger.info("Starting GenOps AI Demo on port 8000")
    logger.info(f"OTLP endpoint: {otlp_endpoint}")
    logger.info(f"Redis URL: {os.getenv('REDIS_URL', 'redis://localhost:6379')}")

    uvicorn.run(
        "demo-app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
