#!/usr/bin/env python3
"""
GenOps AI + OpenTelemetry Collector - 5-Minute Quickstart Example

This zero-configuration example demonstrates:
- Auto-instrumentation with GenOps AI
- OTLP export to local OTel Collector (http://localhost:4318)
- Complete governance telemetry (cost, policy, evaluation)
- Immediate visibility in Grafana dashboards

Prerequisites:
    1. Docker Compose observability stack running:
       docker-compose -f docker-compose.observability.yml up -d

    2. GenOps AI installed:
       pip install genops-ai

Run this script:
    python otel_collector_quickstart.py

Then open Grafana:
    http://localhost:3000 (admin/genops)
    Navigate to: Dashboards → GenOps AI - Governance Overview

You'll see your governance data in real-time!
"""

import os
import random
import time
from typing import Any

from genops.core.policy import PolicyResult, register_policy
from genops.core.telemetry import GenOpsTelemetry

# OpenTelemetry setup
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

print("=" * 70)
print("GenOps AI + OpenTelemetry Collector Quickstart")
print("=" * 70)
print()

# Configure OpenTelemetry to export to local OTel Collector
# Auto-detects http://localhost:4318 if OTEL_EXPORTER_OTLP_ENDPOINT not set
otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
service_name = os.getenv("OTEL_SERVICE_NAME", "genops-quickstart")

print(f"📡 Configuring OTLP export to: {otlp_endpoint}")
print(f"🏷️  Service name: {service_name}")
print()

# Setup tracing with OTLP exporter
trace.set_tracer_provider(TracerProvider())
tracer_provider = trace.get_tracer_provider()
otlp_trace_exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))

print("✅ GenOps configured to send telemetry to OTel Collector")
print()

# Initialize GenOps telemetry engine
genops_telemetry = GenOpsTelemetry()

# Register sample governance policies
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

print("🛡️  Registered governance policies:")
print("   • cost_limit_demo (WARNING level)")
print("   • content_safety_demo (BLOCKED level)")
print()


class MockAIProvider:
    """Mock AI provider that simulates realistic costs and latencies"""

    MODELS = {
        "gpt-3.5-turbo": {"cost_per_token": 0.0000015, "avg_latency": 0.8},
        "gpt-4": {"cost_per_token": 0.00003, "avg_latency": 2.1},
        "claude-3-sonnet": {"cost_per_token": 0.000003, "avg_latency": 1.2},
        "claude-3-opus": {"cost_per_token": 0.000075, "avg_latency": 3.2}
    }

    @classmethod
    def simulate_ai_call(cls, model: str, prompt: str, max_tokens: int = 150) -> dict[str, Any]:
        """Simulate an AI API call"""
        model_config = cls.MODELS.get(model, cls.MODELS["gpt-3.5-turbo"])

        # Simulate latency
        latency = random.uniform(model_config["avg_latency"] * 0.5, model_config["avg_latency"] * 1.2)
        time.sleep(latency)

        # Calculate tokens and cost
        prompt_tokens = int(len(prompt.split()) * 1.3)
        completion_tokens = min(max_tokens, random.randint(20, max_tokens))
        total_tokens = prompt_tokens + completion_tokens

        cost = total_tokens * model_config["cost_per_token"]

        return {
            "response": f"Mock AI response for: {prompt[:50]}...",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(cost, 6),
            "latency": round(latency, 2)
        }


def run_quickstart_demo():
    """Run comprehensive quickstart demo with governance tracking"""

    print("📊 Simulating AI operations with governance tracking...")
    print()

    operations = [
        {
            "name": "AI Chat",
            "team": "engineering",
            "customer": "demo-customer-1",
            "model": "gpt-4",
            "prompt": "Explain the benefits of OpenTelemetry for AI governance",
            "feature": "chat"
        },
        {
            "name": "AI Analysis",
            "team": "data-science",
            "customer": "demo-customer-2",
            "model": "claude-3-sonnet",
            "prompt": "Analyze this dataset for cost optimization opportunities",
            "feature": "analysis"
        },
        {
            "name": "Content Generation",
            "team": "product",
            "customer": "demo-customer-3",
            "model": "gpt-3.5-turbo",
            "prompt": "Generate marketing copy for a new AI product launch",
            "feature": "content-gen"
        }
    ]

    total_cost = 0.0

    for i, op in enumerate(operations, 1):
        print(f"Operation {i}: {op['name']} (team={op['team']}, customer={op['customer']})")

        # Create governance-tracked span
        with genops_telemetry.trace_operation(
            operation_name=op["name"],
            team=op["team"],
            project="quickstart-demo",
            customer_id=op["customer"],
            feature=op["feature"]
        ) as span:

            # Simulate AI call
            ai_result = MockAIProvider.simulate_ai_call(op["model"], op["prompt"])

            # Determine provider from model
            if op["model"].startswith("gpt"):
                provider = "openai"
            elif op["model"].startswith("claude"):
                provider = "anthropic"
            else:
                provider = "unknown"

            # Record comprehensive telemetry
            genops_telemetry.record_cost(
                span=span,
                cost=ai_result["cost"],
                currency="USD",
                provider=provider,
                model=op["model"]
            )

            genops_telemetry.record_tokens(
                span=span,
                prompt_tokens=ai_result["prompt_tokens"],
                completion_tokens=ai_result["completion_tokens"],
                total_tokens=ai_result["total_tokens"]
            )

            # Simulate quality evaluation
            quality_score = random.uniform(0.75, 0.95)
            genops_telemetry.record_evaluation(
                span=span,
                metric_name="response_quality",
                score=quality_score,
                threshold=0.8,
                passed=quality_score > 0.8
            )

            # Record policy evaluation
            policy_result = "PASSED" if ai_result["cost"] < 1.0 else "WARNING"
            genops_telemetry.record_policy(
                span=span,
                policy_name="cost_limit_demo",
                result=policy_result,
                reason=f"Cost ${ai_result['cost']:.4f} {'within' if ai_result['cost'] < 1.0 else 'exceeds'} limit"
            )

            total_cost += ai_result["cost"]

            print(f"   Provider: {provider}, Model: {op['model']}")
            print(f"   Cost: ${ai_result['cost']:.4f}, Tokens: {ai_result['total_tokens']}")
            print(f"   Quality Score: {quality_score:.2f}, Policy: {policy_result}")
            print()

    print("✅ Sent 3 operations to OTel Collector!")
    print(f"   Total cost: ${total_cost:.4f}")
    print()

    # Give time for telemetry to be exported
    print("⏳ Waiting for telemetry export (5 seconds)...")
    time.sleep(5)
    print()

    print("=" * 70)
    print("📊 View your data in Grafana:")
    print("=" * 70)
    print()
    print("1. Open: http://localhost:3000")
    print("   Login: admin / genops")
    print()
    print("2. Navigate to: Dashboards → GenOps AI - Governance Overview")
    print()
    print("3. What you'll see:")
    print("   • Cost tracking by team/customer/model")
    print("   • Token usage distribution")
    print("   • Policy evaluation results")
    print("   • Recent operations table")
    print()
    print("4. Explore traces:")
    print("   • Click 'Explore' in left sidebar")
    print("   • Select 'Tempo' data source")
    print("   • Search for: {.genops.team=\"engineering\"}")
    print("   • Click any trace to see governance attributes")
    print()
    print("=" * 70)
    print("🎉 Quickstart complete! You're now tracking AI governance with OTel!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_quickstart_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        print()
        print("Troubleshooting:")
        print("1. Ensure Docker containers are running:")
        print("   docker-compose -f docker-compose.observability.yml ps")
        print()
        print("2. Check OTel Collector is accessible:")
        print("   curl http://localhost:4318/v1/traces")
        print()
        print("3. View collector logs:")
        print("   docker-compose -f docker-compose.observability.yml logs otel-collector")
        print()
        print("4. Run validation script:")
        print("   python examples/observability/validate_otel_collector.py")
