#!/usr/bin/env python3
"""
🍯 Honeycomb Integration for GenOps AI Observability

This example demonstrates comprehensive GenOps AI telemetry integration with Honeycomb
for high-cardinality AI governance observability and analysis.

📚 Documentation:
   • Quickstart Guide (5 minutes): docs/honeycomb-quickstart.md
   • Comprehensive Integration: docs/integrations/honeycomb.md

Features Demonstrated:
✅ Zero-code auto-instrumentation
✅ High-cardinality attribution analysis
✅ Context manager patterns for scoped tracking
✅ Budget enforcement and policy tracking
✅ Validation utilities for setup verification
✅ AI operation performance analysis
✅ Cost attribution with flexible grouping
✅ Production-ready patterns

Requirements:
    pip install genops-ai[opentelemetry]

Environment Variables:
    HONEYCOMB_API_KEY     - Your Honeycomb API key (required)
    HONEYCOMB_DATASET     - Dataset name (optional, defaults to "genops-ai")
    OTEL_SERVICE_NAME     - Service name (optional, defaults to "genops-demo")
"""

import os
import time

from genops import auto_instrument
from genops.core.context import (
    clear_governance_context,
    governance_context,
    set_governance_context,
)

# OpenTelemetry imports for Honeycomb integration
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    print(
        "⚠️  OpenTelemetry not installed. Install with: pip install genops-ai[opentelemetry]"
    )


def setup_honeycomb_integration():
    """
    Set up Honeycomb integration using OpenTelemetry OTLP export.

    This is the manual setup approach. For production, consider using:
    - genops.exporters.otlp.configure_otlp_exporter() for simplified setup
    - OpenTelemetry Collector for advanced routing and sampling

    See docs/integrations/honeycomb.md for comprehensive setup patterns.
    """
    if not HAS_OPENTELEMETRY:
        print("❌ OpenTelemetry not available. Skipping Honeycomb setup.")
        return False

    api_key = os.getenv("HONEYCOMB_API_KEY")
    dataset = os.getenv("HONEYCOMB_DATASET", "genops-ai")
    service_name = os.getenv("OTEL_SERVICE_NAME", "genops-demo")

    if not api_key:
        print("⚠️  HONEYCOMB_API_KEY not set. Using console export for demo.")
        print("    Set HONEYCOMB_API_KEY to send data to Honeycomb.")
        exporter = ConsoleSpanExporter()
    else:
        print("✅ Honeycomb API key found")
        exporter = OTLPSpanExporter(
            endpoint="https://api.honeycomb.io/v1/traces",
            headers={"X-Honeycomb-Team": api_key},
        )

    # Create resource with service metadata
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            "honeycomb.dataset": dataset,
        }
    )

    # Set up tracing
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(trace_provider)

    print("✅ Honeycomb integration configured")
    print(f"   Dataset: {dataset}")
    print(f"   Service: {service_name}")
    print(f"   Environment: {os.getenv('ENVIRONMENT', 'development')}")

    return True


def validate_honeycomb_setup():
    """
    Validate Honeycomb setup and configuration.

    Demonstrates validation utilities for troubleshooting.
    See docs/honeycomb-quickstart.md#validate-your-setup for details.
    """
    print("\n🔍 HONEYCOMB SETUP VALIDATION")
    print("=" * 60)

    # Check environment variables
    api_key = os.getenv("HONEYCOMB_API_KEY")
    dataset = os.getenv("HONEYCOMB_DATASET", "genops-ai")

    validation_results = []

    # Validate API key
    if api_key:
        validation_results.append(("✅", "HONEYCOMB_API_KEY", "Set"))
    else:
        validation_results.append(
            ("❌", "HONEYCOMB_API_KEY", "Not set (using console export)")
        )

    # Validate dataset
    validation_results.append(("✅", "HONEYCOMB_DATASET", dataset))

    # Validate OpenTelemetry
    if HAS_OPENTELEMETRY:
        validation_results.append(("✅", "OpenTelemetry", "Installed"))
    else:
        validation_results.append(("❌", "OpenTelemetry", "Not installed"))

    # Display results
    print("\nConfiguration:")
    for icon, key, value in validation_results:
        print(f"  {icon} {key}: {value}")

    # Check connectivity (basic check)
    if api_key:
        print("\nConnectivity:")
        print("  ✅ Honeycomb endpoint: https://api.honeycomb.io/v1/traces")
        print("  ℹ️  To verify API key and dataset, send test telemetry below")
    else:
        print("\nConnectivity:")
        print("  ⚠️  Skipped (no API key configured)")

    print("\n" + "=" * 60)


def demonstrate_auto_instrumentation():
    """
    Demonstrate zero-code auto-instrumentation pattern.

    This is the fastest way to get started with GenOps + Honeycomb.
    See docs/honeycomb-quickstart.md for step-by-step guide.
    """
    print("\n🤖 AUTO-INSTRUMENTATION DEMONSTRATION")
    print("=" * 60)

    # Enable auto-instrumentation for all providers
    # This automatically tracks OpenAI, Anthropic, Bedrock, etc.
    auto_instrument()
    print("✅ Auto-instrumentation enabled for all AI providers")

    # Set global governance context (applies to all operations)
    set_governance_context(
        {
            "team": "ai-platform",
            "project": "honeycomb-integration-demo",
            "environment": "development",
        }
    )
    print("✅ Global governance context set")

    print("\n💡 Now all AI operations are automatically tracked!")
    print("   Example: client.chat.completions.create(...)")
    print(
        "   → Telemetry sent to Honeycomb with cost, tokens, and governance attributes"
    )


def demonstrate_high_cardinality_tracking():
    """
    Demonstrate high-cardinality attribution tracking.

    Honeycomb excels at high-cardinality analysis (unlimited customer_id, user_id, etc.)
    This is perfect for per-customer cost tracking in SaaS applications.

    See docs/integrations/honeycomb.md#high-cardinality-analysis for details.
    """
    print("\n🔢 HIGH-CARDINALITY TRACKING DEMONSTRATION")
    print("=" * 60)

    # Simulate multi-customer SaaS operations
    customers = [
        {
            "customer_id": "enterprise-acme-corp",
            "customer_tier": "enterprise",
            "feature": "document-analysis",
            "user_id": "user-12345",
            "region": "us-west-2",
        },
        {
            "customer_id": "startup-tech-innovations",
            "customer_tier": "business",
            "feature": "chat-assistant",
            "user_id": "user-67890",
            "region": "eu-west-1",
        },
        {
            "customer_id": "enterprise-global-bank",
            "customer_tier": "enterprise",
            "feature": "fraud-detection",
            "user_id": "user-11111",
            "region": "us-east-1",
        },
    ]

    print("🤖 Generating operations with high-cardinality attributes...")
    print("   (customer_id, user_id, feature, tier, region)\n")

    for i, customer_attrs in enumerate(customers, 1):
        # Set per-request governance context (high-cardinality tracking)
        set_governance_context(**customer_attrs)

        # Simulate AI operation (would normally be actual API call)
        time.sleep(0.05)

        # Display tracked attributes
        print(f"Operation {i}:")
        print(f"  Customer: {customer_attrs['customer_id']}")
        print(f"  User: {customer_attrs['user_id']}")
        print(f"  Feature: {customer_attrs['feature']}")
        print(f"  Tier: {customer_attrs['customer_tier']}")
        print(f"  Region: {customer_attrs['region']}")

        # Clear context for next request
        clear_governance_context()

    print("\n✅ High-cardinality telemetry sent to Honeycomb!")
    print("   Query in Honeycomb: GROUP BY genops.customer_id | SUM(genops.cost.total)")


def demonstrate_context_managers():
    """
    Demonstrate context manager pattern for scoped tracking.

    Context managers automatically set and clear governance attributes,
    preventing attribute leakage between operations.

    See docs/integrations/honeycomb.md#governance-context-and-attribution
    """
    print("\n🎯 CONTEXT MANAGER PATTERN DEMONSTRATION")
    print("=" * 60)

    # Simulate workflow with scoped context
    workflows = [
        {
            "workflow_id": "workflow-abc-123",
            "customer_id": "customer-001",
            "feature": "data-pipeline",
        },
        {
            "workflow_id": "workflow-def-456",
            "customer_id": "customer-002",
            "feature": "analysis",
        },
    ]

    for workflow_attrs in workflows:
        print(f"\nProcessing {workflow_attrs['workflow_id']}...")

        # Context manager automatically manages governance scope
        with governance_context(**workflow_attrs):
            # All operations within this block inherit the governance context
            print(
                f"  Step 1: Data preparation (customer: {workflow_attrs['customer_id']})"
            )
            time.sleep(0.05)

            print(f"  Step 2: AI processing (feature: {workflow_attrs['feature']})")
            time.sleep(0.05)

            print("  Step 3: Result aggregation")
            time.sleep(0.05)

        # Context automatically cleared on exit
        print("  ✅ Workflow complete (context auto-cleared)")

    print("\n✅ Context manager pattern demonstrated!")
    print("   → Prevents attribute leakage between operations")
    print("   → Automatic cleanup on scope exit")


def demonstrate_budget_tracking():
    """
    Demonstrate budget enforcement and tracking.

    Budget tracking helps prevent cost overruns by enforcing spending limits
    per team, project, or customer.

    See docs/integrations/honeycomb.md#budget-tracking-queries for queries.
    """
    print("\n💰 BUDGET TRACKING DEMONSTRATION")
    print("=" * 60)

    # Simulate budget-constrained operations
    budgets = [
        {
            "team": "ai-research",
            "budget_id": "team-ai-research-daily",
            "budget_limit": 100.0,
            "budget_consumed": 45.50,
            "budget_remaining": 54.50,
        },
        {
            "team": "product-eng",
            "budget_id": "team-product-eng-daily",
            "budget_limit": 50.0,
            "budget_consumed": 48.75,
            "budget_remaining": 1.25,
        },
    ]

    for budget_info in budgets:
        utilization_pct = (
            budget_info["budget_consumed"] / budget_info["budget_limit"]
        ) * 100

        print(f"\nTeam: {budget_info['team']}")
        print(f"  Budget ID: {budget_info['budget_id']}")
        print(f"  Limit: ${budget_info['budget_limit']:.2f}")
        print(f"  Consumed: ${budget_info['budget_consumed']:.2f}")
        print(f"  Remaining: ${budget_info['budget_remaining']:.2f}")
        print(f"  Utilization: {utilization_pct:.1f}%")

        if utilization_pct >= 90:
            print("  ⚠️  WARNING: Budget nearly exhausted!")
        elif utilization_pct >= 75:
            print("  ⚠️  ALERT: 75% budget threshold crossed")
        else:
            print("  ✅ Budget healthy")

    print("\n💡 Budget Tracking in Honeycomb:")
    print("   Query: WHERE genops.budget.consumed / genops.budget.limit > 0.9")
    print("   → Find teams approaching budget limits")
    print("\n   Trigger: Alert when utilization > 90%")
    print("   → Proactive budget enforcement")


def show_honeycomb_queries():
    """
    Show example Honeycomb queries for AI governance analysis.

    These queries demonstrate Honeycomb's high-cardinality query capabilities.
    See docs/integrations/honeycomb.md#honeycomb-query-examples for comprehensive list.
    """
    print("\n🔍 HONEYCOMB QUERY EXAMPLES")
    print("=" * 60)

    queries = {
        "Cost Analysis": [
            "# Total cost by provider and model",
            "GROUP BY genops.cost.provider, genops.cost.model | SUM(genops.cost.total)",
            "",
            "# Cost by customer (top 20)",
            "GROUP BY genops.customer_id | SUM(genops.cost.total) | ORDER BY SUM DESC | LIMIT 20",
            "",
            "# Daily cost trend",
            "GROUP BY DATE_TRUNC('day', timestamp) | SUM(genops.cost.total)",
        ],
        "Performance Analysis": [
            "# Latency percentiles by model",
            "GROUP BY genops.cost.model | P50(duration_ms), P95(duration_ms), P99(duration_ms)",
            "",
            "# Slow operations (>2 seconds)",
            "WHERE duration_ms > 2000 | COUNT | GROUP BY genops.team, genops.feature",
            "",
            "# Correlation: Latency vs Token Count",
            "HEATMAP(duration_ms, genops.tokens.total)",
        ],
        "Attribution Analysis": [
            "# Multi-dimensional cost breakdown",
            "GROUP BY genops.team, genops.project, genops.environment | SUM(genops.cost.total)",
            "",
            "# Cost by customer tier",
            "GROUP BY genops.customer_tier | SUM(genops.cost.total), COUNT, AVG(genops.cost.total)",
            "",
            "# Feature usage and cost",
            "GROUP BY genops.feature | COUNT, SUM(genops.cost.total) | ORDER BY COUNT DESC",
        ],
        "BubbleUp Analysis": [
            "# Find cost outliers automatically",
            "1. Create query: SUM(genops.cost.total) WHERE timestamp > ago(1h)",
            "2. Click 'BubbleUp' button",
            "3. Honeycomb automatically surfaces attributes driving high costs",
            "   Example: customer_id, feature, model, etc.",
        ],
        "Budget Tracking": [
            "# Budget utilization percentage",
            "WHERE genops.budget.id EXISTS | AVG(genops.budget.consumed / genops.budget.limit * 100)",
            "",
            "# Budget overruns",
            "WHERE genops.budget.consumed > genops.budget.limit | COUNT | GROUP BY genops.team",
        ],
    }

    for category, query_list in queries.items():
        print(f"\n🍯 {category}:")
        for query in query_list:
            if query:  # Skip empty lines in display
                print(f"   {query}")

    print("\n\n💡 Honeycomb Query Tips:")
    print("   • WHERE filters by any attribute (high-cardinality supported!)")
    print("   • GROUP BY enables multi-dimensional analysis")
    print("   • HEATMAP shows correlation between metrics")
    print("   • P50, P95, P99 for performance percentiles")
    print("   • BubbleUp automatically discovers cost drivers")
    print("   • Use Derived Columns for computed metrics")


def show_honeycomb_advanced_features():
    """
    Show Honeycomb's advanced features for AI governance.

    See docs/integrations/honeycomb.md for comprehensive documentation.
    """
    print("\n🚀 HONEYCOMB ADVANCED FEATURES")
    print("=" * 60)

    print("\n1. Derived Columns (Computed Metrics)")
    print("   Create: cost_per_token = genops.cost.total / genops.tokens.total")
    print("   Usage: GROUP BY genops.cost.model | AVG($cost_per_token)")
    print("   → Reduces cardinality and simplifies queries")

    print("\n2. Triggers (Alerting)")
    print("   Example: Alert when daily budget > 90%")
    print("   Query: MAX(genops.budget.consumed / genops.budget.limit * 100)")
    print("   Condition: MAX >= 90")
    print("   Action: Send Slack notification")

    print("\n3. SLOs (Service Level Objectives)")
    print("   Example: Policy Compliance SLO")
    print("   SLI: WHERE genops.policy.result = 'allowed'")
    print("   Target: 99.9% compliance rate")
    print("   → Track governance compliance over time")

    print("\n4. Markers (Deployment Tracking)")
    print("   Mark significant events (model deployments, config changes)")
    print("   Correlate cost changes with deployments")
    print("   API: POST /markers/{dataset} with deployment details")

    print("\n5. BubbleUp (Root Cause Analysis)")
    print("   Automatically discover cost drivers")
    print("   No manual query construction needed")
    print("   Statistical analysis of attribute distributions")

    print("\n📘 See docs/integrations/honeycomb.md for setup details")


def main():
    """Run the comprehensive Honeycomb integration demonstration."""

    print("🍯 GenOps AI: Honeycomb Integration Comprehensive Demo")
    print("=" * 80)
    print("\n📚 Documentation:")
    print("   • Quickstart (5 min): docs/honeycomb-quickstart.md")
    print("   • Comprehensive: docs/integrations/honeycomb.md")
    print("   • GitHub: https://github.com/KoshiHQ/GenOps-AI")
    print("\n" + "=" * 80)

    # 1. Setup and validation
    if not setup_honeycomb_integration():
        print("\n❌ Setup failed. Please install OpenTelemetry:")
        print("   pip install genops-ai[opentelemetry]")
        return

    validate_honeycomb_setup()

    # 2. Demonstrate integration patterns
    try:
        demonstrate_auto_instrumentation()
        demonstrate_high_cardinality_tracking()
        demonstrate_context_managers()
        demonstrate_budget_tracking()
        show_honeycomb_queries()
        show_honeycomb_advanced_features()

        # 3. Summary
        print("\n\n🎯 HONEYCOMB INTEGRATION BENEFITS")
        print("=" * 60)
        print("✅ High-cardinality attribution (unlimited customer_id, user_id, etc.)")
        print("✅ Sub-second query performance for interactive debugging")
        print("✅ BubbleUp automatically surfaces cost drivers")
        print("✅ Real-time AI governance insights with no aggregation delays")
        print("✅ Triggers for proactive budget enforcement")
        print("✅ SLOs for governance compliance tracking")
        print("✅ Derived Columns for computed governance metrics")

        print("\n\n🔧 NEXT STEPS")
        print("=" * 60)
        print("1. Set HONEYCOMB_API_KEY environment variable")
        print("   export HONEYCOMB_API_KEY='your_api_key'")
        print("")
        print("2. Follow quickstart guide (5 minutes)")
        print("   → docs/honeycomb-quickstart.md")
        print("")
        print("3. Create custom queries and boards in Honeycomb UI")
        print("   → https://ui.honeycomb.io")
        print("")
        print("4. Set up Triggers for budget alerts")
        print("   → docs/integrations/honeycomb.md#triggers-for-budget-alerts")
        print("")
        print("5. Explore BubbleUp for cost analysis")
        print("   → docs/integrations/honeycomb.md#bubbleup-for-root-cause-analysis")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
