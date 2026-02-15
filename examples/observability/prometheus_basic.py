"""
GenOps AI - Prometheus Basic Example

Demonstrates zero-code Prometheus metrics export with governance tracking.

This example shows:
- Auto-instrumentation setup (zero code changes)
- Governance context configuration
- Multi-provider cost tracking
- Metrics validation

Metrics available at: http://localhost:8000/metrics

Usage:
    python examples/observability/prometheus_basic.py
"""

import os
import time

from openai import OpenAI

# Set API key (or use OPENAI_API_KEY environment variable)
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  OPENAI_API_KEY environment variable not set")
    print("   Set it with: export OPENAI_API_KEY='your-api-key'")
    exit(1)


def main():
    print("=" * 80)
    print("GenOps AI - Prometheus Basic Example")
    print("=" * 80)
    print()

    # Step 1: Validate setup before starting
    print("Step 1: Validating Prometheus setup...")
    print("-" * 80)

    from genops.exporters.prometheus import print_validation_result, validate_setup

    result = validate_setup()
    print_validation_result(result)

    if not result.success:
        print("\n⚠️  Validation failed. Please fix issues before continuing.")
        print("   Install dependencies: pip install genops-ai[prometheus]")
        return

    # Step 2: Start Prometheus metrics exporter (zero-code auto-instrumentation)
    print("\nStep 2: Starting Prometheus metrics exporter...")
    print("-" * 80)

    from genops.exporters.prometheus import auto_instrument

    exporter = auto_instrument()
    print(
        f"✅ Prometheus metrics server started at http://localhost:{exporter.config.port}/metrics"
    )
    print()

    # Step 3: Set governance context (cost attribution)
    print("Step 3: Configuring governance context...")
    print("-" * 80)

    from genops.core.context import set_governance_context

    set_governance_context(
        {
            "team": "ml-research",
            "project": "prometheus-demo",
            "environment": "development",
            "customer_id": "demo-customer",
        }
    )
    print("✅ Governance context configured:")
    print("   - team: ml-research")
    print("   - project: prometheus-demo")
    print("   - environment: development")
    print("   - customer_id: demo-customer")
    print()

    # Step 4: Use OpenAI (metrics automatically tracked)
    print("Step 4: Making AI requests (metrics tracked automatically)...")
    print("-" * 80)

    client = OpenAI()

    # Make a few requests
    for i in range(3):
        print(f"\nRequest {i + 1}/3:")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"What is {i + 1} + {i + 1}?"}],
                max_tokens=50,
            )

            print(f"   ✅ Response: {response.choices[0].message.content.strip()}")
            print(f"   📊 Tokens: {response.usage.total_tokens}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        time.sleep(1)  # Brief pause between requests

    print()
    print("✅ All requests completed")
    print()

    # Step 5: View metrics
    print("Step 5: Viewing exported metrics...")
    print("-" * 80)
    print(
        f"\nMetrics are now available at: http://localhost:{exporter.config.port}/metrics"
    )
    print()
    print("Key metrics to check:")
    print("  • genops_cost_total_usd - Total cost across all operations")
    print("  • genops_tokens_total - Total tokens consumed")
    print("  • genops_tokens_input_total - Input tokens")
    print("  • genops_tokens_output_total - Output tokens")
    print("  • genops_operations_total - Total operations count")
    print()
    print("View in browser:")
    print(f"  curl http://localhost:{exporter.config.port}/metrics | grep genops")
    print()

    # Step 6: PromQL query examples
    print("Step 6: Example PromQL queries for Prometheus...")
    print("-" * 80)
    print()
    print(
        "After configuring Prometheus to scrape http://localhost:8000, try these queries:"
    )
    print()
    print("# Total cost")
    print("sum(genops_cost_total_usd)")
    print()
    print("# Cost by model")
    print("sum(genops_cost_total_usd) by (model)")
    print()
    print("# Cost by team")
    print("sum(genops_cost_total_usd) by (team)")
    print()
    print("# Hourly cost rate")
    print("sum(rate(genops_cost_total_usd[1h])) * 3600")
    print()
    print("# Token efficiency (tokens per dollar)")
    print("sum(genops_tokens_total) / sum(genops_cost_total_usd)")
    print()
    print("# Operations per second")
    print("sum(rate(genops_operations_total[1m]))")
    print()

    # Step 7: Prometheus configuration
    print("Step 7: Prometheus scrape configuration...")
    print("-" * 80)
    print()
    print("Add this to your prometheus.yml:")
    print()
    print("scrape_configs:")
    print("  - job_name: 'genops-ai'")
    print("    static_configs:")
    print("      - targets: ['localhost:8000']")
    print("    scrape_interval: 15s")
    print()
    print("Then restart Prometheus:")
    print("  docker restart prometheus")
    print("  # or")
    print("  systemctl restart prometheus")
    print()

    # Keep server running
    print("=" * 80)
    print("Metrics server is running. Press Ctrl+C to stop.")
    print("=" * 80)
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        exporter.stop()
        print("✅ Metrics server stopped")


if __name__ == "__main__":
    main()
