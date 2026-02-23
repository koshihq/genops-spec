#!/usr/bin/env python3
"""
GenOps AI + Cribl Stream Integration Example

This example demonstrates how to configure GenOps AI to send governance telemetry
to Cribl Stream for intelligent routing, enrichment, and distribution to multiple
observability platforms.

Architecture:
    GenOps AI → OTLP → Cribl Stream → [Datadog, Splunk, S3, etc.]

Features demonstrated:
- OTLP configuration for Cribl endpoint
- Cost tracking with multi-destination routing
- Policy enforcement with SIEM integration
- Budget alerting via webhooks
- Compliance audit trail preservation

Prerequisites:
- Cribl Stream v4.0+ running at http://cribl-stream:4318
- CRIBL_AUTH_TOKEN environment variable set
- Downstream platforms configured (Datadog, Splunk, etc.)

Usage:
    export CRIBL_OTLP_ENDPOINT="http://localhost:4318"
    export CRIBL_AUTH_TOKEN="your-cribl-token"
    export OPENAI_API_KEY="your-openai-key"  # Optional for real demos
    python cribl_integration.py
"""

import os
import sys
import time


def validate_cribl_connection():
    """Validate Cribl setup before running demos."""
    print("=" * 70)
    print("STEP 0: Validating Cribl Stream Setup")
    print("=" * 70 + "\n")

    from genops.providers.cribl.validation import (
        print_validation_result,
        validate_setup,
    )

    result = validate_setup()
    print_validation_result(result)

    if not result.is_valid:
        print(
            "\n❌ Setup validation failed - please fix errors above before continuing"
        )
        sys.exit(1)

    print("\n✅ Validation passed - proceeding with demos...\n")


def setup_genops_cribl_integration():
    """Configure GenOps to send telemetry to Cribl Stream."""
    from genops import init

    # Get Cribl endpoint and credentials
    cribl_endpoint = os.getenv("CRIBL_OTLP_ENDPOINT", "http://localhost:4318")
    cribl_token = os.getenv("CRIBL_AUTH_TOKEN")

    if not cribl_token:
        print("⚠️  CRIBL_AUTH_TOKEN not set - using anonymous mode")
        print("   For production, set: export CRIBL_AUTH_TOKEN='your-token'\n")

    # Initialize GenOps with Cribl as OTLP destination
    init(
        service_name="genops-cribl-demo",
        exporter_type="otlp",
        otlp_endpoint=cribl_endpoint,
        otlp_headers={
            "Authorization": f"Bearer {cribl_token}" if cribl_token else "",
            "X-Scope-OrgID": "my-organization",
            "X-GenOps-Version": "0.1.0",
        },
        default_team="ai-platform",
        default_project="cribl-integration-demo",
        default_environment="production",
    )

    print(f"✅ GenOps configured to send telemetry to Cribl at {cribl_endpoint}\n")


def demonstrate_cost_tracking():
    """Demonstrate cost tracking routed through Cribl to cost platforms."""
    print("=" * 70)
    print("DEMO 1: Cost Tracking with Cribl Routing")
    print("=" * 70 + "\n")

    from genops.core import track_enhanced
    from genops.core.telemetry import GenOpsTelemetry

    print("📊 Generating AI operations with cost tracking...\n")

    # Simulate AI operations with varying costs
    operations = [
        {
            "customer": "enterprise-123",
            "feature": "chat",
            "model": "gpt-4",
            "tokens_in": 150,
            "tokens_out": 350,
            "cost": 0.0075,
        },
        {
            "customer": "enterprise-456",
            "feature": "summarization",
            "model": "gpt-3.5-turbo",
            "tokens_in": 800,
            "tokens_out": 200,
            "cost": 0.0012,
        },
        {
            "customer": "free-tier-789",
            "feature": "simple-qa",
            "model": "gpt-3.5-turbo",
            "tokens_in": 50,
            "tokens_out": 30,
            "cost": 0.0001,
        },
    ]

    telemetry = GenOpsTelemetry()

    for op in operations:
        with track_enhanced(
            operation_name=f"ai_operation_{op['feature']}",
            customer_id=op["customer"],
            feature=op["feature"],
        ) as span:
            print(
                f"  🤖 Processing: {op['customer']} - {op['feature']} ({op['model']})"
            )

            # Record cost telemetry
            telemetry.record_cost(
                span,
                provider="openai",
                model=op["model"],
                input_tokens=op["tokens_in"],
                output_tokens=op["tokens_out"],
                total_cost=op["cost"],
                metadata={
                    "cost_per_token": op["cost"] / (op["tokens_in"] + op["tokens_out"]),
                    "operation_type": op["feature"],
                },
            )

            # Cost telemetry sent to Cribl → routed to cost dashboards
            print(f"     💰 Cost: ${op['cost']:.6f} - Tracked and routed via Cribl")

            # Small delay for realistic timing
            time.sleep(0.1)

    print("\n✅ Cost data sent to Cribl → routed to Datadog/Grafana\n")


def demonstrate_policy_enforcement():
    """Demonstrate policy violations routed to SIEM via Cribl."""
    print("=" * 70)
    print("DEMO 2: Policy Enforcement with SIEM Routing")
    print("=" * 70 + "\n")

    from genops.core import track_enhanced
    from genops.core.telemetry import GenOpsTelemetry

    print("🔐 Simulating policy evaluation and violations...\n")

    policies = [
        {
            "name": "cost_limit",
            "result": "allowed",
            "reason": "Within budget ($5.23 of $100 daily limit)",
            "severity": "info",
        },
        {
            "name": "pii_detection",
            "result": "warning",
            "reason": "Potential PII detected in prompt (email address)",
            "severity": "medium",
        },
        {
            "name": "content_safety",
            "result": "blocked",
            "reason": "Harmful content detected (violence threshold exceeded)",
            "severity": "high",
        },
        {
            "name": "data_residency",
            "result": "allowed",
            "reason": "Request originated from allowed region (US-EAST)",
            "severity": "info",
        },
    ]

    telemetry = GenOpsTelemetry()

    for policy in policies:
        with track_enhanced(
            operation_name="policy_evaluation",
            customer_id="regulated-customer-001",
            environment="production",
        ) as span:
            # Determine status emoji
            status_emoji = {"allowed": "✅", "warning": "⚠️", "blocked": "🚫"}.get(
                policy["result"], "❓"
            )

            print(
                f"  {status_emoji} Policy: {policy['name']} → {policy['result'].upper()}"
            )
            print(f"     Reason: {policy['reason']}")
            print(f"     Severity: {policy['severity']}")

            # Record policy result
            telemetry.record_policy(
                span,
                policy_name=policy["name"],
                policy_result=policy["result"],
                policy_reason=policy["reason"],
                metadata={
                    "severity": policy["severity"],
                    "compliance_framework": "SOC2",
                    "evaluated_at": time.time(),
                    "requires_audit": policy["result"] in ["blocked", "warning"],
                },
            )

            # Cribl routes violations to SIEM, warnings to monitoring
            if policy["result"] == "blocked":
                destination = "SIEM (Splunk)"
            elif policy["result"] == "warning":
                destination = "Monitoring (Datadog) + SIEM (Splunk)"
            else:
                destination = "Monitoring (Datadog)"

            print(f"     📤 Routed to: {destination}\n")

            time.sleep(0.1)

    print("✅ Policy events sent to Cribl → routed by severity to SIEM/Monitoring\n")


def demonstrate_budget_alerting():
    """Demonstrate budget tracking with webhook alerts via Cribl."""
    print("=" * 70)
    print("DEMO 3: Budget Alerting via Cribl Webhooks")
    print("=" * 70 + "\n")

    from genops.core import track_enhanced
    from genops.core.telemetry import GenOpsTelemetry

    print("💸 Tracking budget utilization with threshold alerts...\n")

    budgets = [
        {
            "name": "team-nlp-daily",
            "limit": 100.0,
            "used": 75.0,
            "threshold": 75,
            "period": "daily",
        },
        {
            "name": "team-vision-daily",
            "limit": 200.0,
            "used": 180.0,
            "threshold": 90,
            "period": "daily",
        },
        {
            "name": "customer-enterprise-monthly",
            "limit": 10000.0,
            "used": 10500.0,
            "threshold": 100,
            "period": "monthly",
        },
        {
            "name": "project-research-weekly",
            "limit": 500.0,
            "used": 125.0,
            "threshold": 25,
            "period": "weekly",
        },
    ]

    telemetry = GenOpsTelemetry()

    for budget in budgets:
        utilization = (budget["used"] / budget["limit"]) * 100
        remaining = max(0, budget["limit"] - budget["used"])

        with track_enhanced(
            operation_name="budget_tracking",
            team=budget["name"].split("-")[1]
            if "team" in budget["name"]
            else "platform",
            budget_name=budget["name"],
        ) as span:
            print(f"  📊 Budget: {budget['name']} ({budget['period']})")
            print(
                f"     💰 ${budget['used']:.2f} / ${budget['limit']:.2f} USD ({utilization:.1f}%)"
            )
            print(f"     📉 Remaining: ${remaining:.2f}")

            # Record budget telemetry
            telemetry.record_budget(
                span,
                budget_name=budget["name"],
                budget_limit=budget["limit"],
                budget_used=budget["used"],
                budget_remaining=remaining,
                metadata={
                    "utilization_percent": utilization,
                    "threshold": budget["threshold"],
                    "alert_triggered": utilization >= budget["threshold"],
                    "period": budget["period"],
                    "exceeded": utilization > 100,
                },
            )

            # Cribl triggers webhook alert if threshold exceeded
            if utilization >= 100:
                print(
                    f"     🚨 CRITICAL ALERT: Budget exceeded by ${budget['used'] - budget['limit']:.2f}"
                )
                print("        → PagerDuty incident created (P1)")
                print("        → Slack alert: #budget-alerts-critical")
            elif utilization >= budget["threshold"]:
                severity = "HIGH" if utilization >= 90 else "MEDIUM"
                print(f"     ⚠️  {severity} ALERT: {utilization:.1f}% utilization")
                print("        → Slack alert: #budget-alerts")
            else:
                print(f"     ✅ Normal: Below {budget['threshold']}% threshold")

            print()
            time.sleep(0.1)

    print("✅ Budget data sent to Cribl → alerts triggered via webhooks\n")


def demonstrate_compliance_audit_trail():
    """Demonstrate compliance audit trail preserved in data lake via Cribl."""
    print("=" * 70)
    print("DEMO 4: Compliance Audit Trail to Data Lake")
    print("=" * 70 + "\n")

    from genops.core import track_enhanced
    from genops.core.telemetry import GenOpsTelemetry

    print("📋 Creating compliance audit trail for regulated operations...\n")

    regulated_operations = [
        {
            "operation": "phi_processing",
            "compliance": "HIPAA",
            "customer": "healthcare-provider-001",
            "data_classification": "PHI",
            "retention_years": 7,
        },
        {
            "operation": "pii_analysis",
            "compliance": "GDPR",
            "customer": "eu-customer-002",
            "data_classification": "Personal Data",
            "retention_years": 7,
        },
        {
            "operation": "financial_modeling",
            "compliance": "SOC2",
            "customer": "fintech-company-003",
            "data_classification": "Financial Data",
            "retention_years": 7,
        },
        {
            "operation": "pci_transaction",
            "compliance": "PCI-DSS",
            "customer": "payment-processor-004",
            "data_classification": "Payment Card Data",
            "retention_years": 5,
        },
    ]

    GenOpsTelemetry()

    for op in regulated_operations:
        with track_enhanced(
            operation_name=op["operation"],
            customer_id=op["customer"],
            environment="production",
            metadata={
                "compliance_framework": op["compliance"],
                "data_classification": op["data_classification"],
                "audit_required": True,
                "retention_period_years": op["retention_years"],
            },
        ) as span:
            print(f"  🔐 Operation: {op['operation']}")
            print(f"     📋 Compliance: {op['compliance']}")
            print(f"     🏷️  Classification: {op['data_classification']}")
            print(f"     📅 Retention: {op['retention_years']} years")

            # Record compliance metadata
            span.set_attribute("genops.compliance.framework", op["compliance"])
            span.set_attribute(
                "genops.compliance.data_classification", op["data_classification"]
            )
            span.set_attribute("genops.compliance.audit_trail_required", True)
            span.set_attribute(
                "genops.compliance.retention_years", op["retention_years"]
            )

            # Cribl routes to long-term storage (S3/Snowflake) for compliance
            print("     📤 Routed to:")
            print(
                f"        → S3 (compliance bucket with {op['retention_years']}-year retention)"
            )
            print("        → Snowflake (audit database)")
            print("        → Cribl Lake (searchable archive)")
            print()

            time.sleep(0.1)

    print("✅ Audit trail sent to Cribl → routed to compliant long-term storage\n")


def print_cribl_pipeline_summary():
    """Print summary of Cribl pipeline configurations needed."""
    print("=" * 70)
    print("CRIBL PIPELINE CONFIGURATION SUMMARY")
    print("=" * 70 + "\n")

    print("📋 Required Cribl Pipelines:\n")

    pipelines = [
        {
            "name": "GenOps Cost Governance",
            "description": "Route cost telemetry to dashboards",
            "filters": "genops.cost.* attributes present",
            "destinations": ["Datadog", "Grafana", "InfluxDB"],
            "sampling": "100% if cost > $10/hr, 10% otherwise",
        },
        {
            "name": "GenOps Policy & Compliance",
            "description": "Route policy events to SIEM",
            "filters": "genops.policy.result == 'blocked' OR 'warning'",
            "destinations": ["Splunk", "Elastic", "Sentinel"],
            "sampling": "100% for violations, 1% for allowed",
        },
        {
            "name": "GenOps Budget Alerting",
            "description": "Trigger alerts on budget thresholds",
            "filters": "genops.budget.utilization_percent >= 80",
            "destinations": ["Webhook (Slack)", "Webhook (PagerDuty)"],
            "sampling": "100% for alerts",
        },
        {
            "name": "GenOps Compliance Audit",
            "description": "Preserve audit trail for compliance",
            "filters": "genops.compliance.audit_trail_required == true",
            "destinations": ["S3 (compliance bucket)", "Snowflake", "Cribl Lake"],
            "sampling": "100% for regulated data",
        },
    ]

    for i, pipeline in enumerate(pipelines, 1):
        print(f"{i}. {pipeline['name']}")
        print(f"   Description: {pipeline['description']}")
        print(f"   Filters: {pipeline['filters']}")
        print(f"   Destinations: {', '.join(pipeline['destinations'])}")
        print(f"   Sampling: {pipeline['sampling']}")
        print()

    print("✅ See docs/integrations/cribl.md for detailed pipeline configurations\n")


def print_integration_architecture():
    """Print the integration architecture diagram."""
    print("=" * 70)
    print("INTEGRATION ARCHITECTURE")
    print("=" * 70 + "\n")

    architecture = """
    ┌─────────────────────────────────────────────────────────┐
    │                   AI Application                         │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │         GenOps AI Instrumentation                  │  │
    │  │  - Cost tracking    - Policy enforcement           │  │
    │  │  - Token counting   - Evaluation metrics           │  │
    │  └────────────────┬──────────────────────────────────┘  │
    └────────────────────┼──────────────────────────────────────┘
                         │ OTLP (HTTP/gRPC)
                         ↓
    ┌─────────────────────────────────────────────────────────┐
    │                   Cribl Stream                           │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  HTTP Receiver Source (OTLP)                       │  │
    │  │  - Endpoint: http://cribl:4318/v1/traces          │  │
    │  │  - Authentication: Bearer token                    │  │
    │  └────────────────┬──────────────────────────────────┘  │
    │                   ↓                                      │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │  Cribl Pipelines                                   │  │
    │  │  - Cost Governance (route to dashboards)          │  │
    │  │  - Policy & Compliance (route to SIEM)            │  │
    │  │  - Budget Alerting (trigger webhooks)             │  │
    │  │  - Audit Trail (preserve for compliance)          │  │
    │  └────────────────┬──────────────────────────────────┘  │
    └────────────────────┼──────────────────────────────────────┘
                         │ Routed & Enriched Telemetry
                         ↓
    ┌─────────────────────────────────────────────────────────┐
    │        Destination Routing (Policy-Based)                │
    │  ├─→ Datadog (cost dashboards & alerting)               │
    │  ├─→ Splunk (compliance audit logs)                     │
    │  ├─→ Elastic (security analytics)                       │
    │  ├─→ S3/Snowflake (long-term cost analysis)             │
    │  ├─→ Webhooks (Slack/PagerDuty for alerts)              │
    │  └─→ Cribl Lake (internal telemetry store)              │
    └─────────────────────────────────────────────────────────┘
    """

    print(architecture)
    print()


def print_key_benefits():
    """Print key benefits of GenOps + Cribl integration."""
    print("=" * 70)
    print("KEY BENEFITS: GenOps AI + Cribl Stream")
    print("=" * 70 + "\n")

    benefits = [
        {
            "category": "Multi-Platform Distribution",
            "items": [
                "Route governance telemetry to 100+ destinations simultaneously",
                "Unified telemetry pipeline for all AI governance data",
                "No vendor lock-in - works with any observability platform",
            ],
        },
        {
            "category": "Intelligent Cost Optimization",
            "items": [
                "Cost-aware sampling reduces telemetry costs by 90%+",
                "Route high-value events to premium platforms, low-value to storage",
                "Policy-based routing optimizes downstream platform costs",
            ],
        },
        {
            "category": "Compliance & Audit",
            "items": [
                "Automatic routing to compliant long-term storage (S3, Snowflake)",
                "Immutable audit trail for regulated industries (HIPAA, SOC2, GDPR)",
                "Configurable retention policies per compliance framework",
            ],
        },
        {
            "category": "Real-Time Alerting",
            "items": [
                "Budget threshold alerts via webhooks (Slack, PagerDuty)",
                "Policy violation routing to SIEM for immediate response",
                "Configurable alert severity and escalation paths",
            ],
        },
        {
            "category": "Operational Excellence",
            "items": [
                "Enrichment with organizational metadata via lookup tables",
                "Centralized telemetry governance across all AI systems",
                "Performance optimization through intelligent sampling",
            ],
        },
    ]

    for benefit in benefits:
        print(f"🎯 {benefit['category']}")
        for item in benefit["items"]:
            print(f"   ✅ {item}")
        print()


def main():
    """Run all Cribl integration demonstrations."""
    print("\n" + "=" * 70)
    print("GenOps AI + Cribl Stream Integration Demo")
    print("=" * 70 + "\n")

    print("This demo shows how GenOps governance telemetry flows through Cribl")
    print(
        "for intelligent routing to multiple observability and compliance platforms.\n"
    )

    # Validate Cribl setup before proceeding
    validate_cribl_connection()

    # Print architecture
    print_integration_architecture()

    # Setup
    setup_genops_cribl_integration()

    # Run demonstrations
    demonstrate_cost_tracking()
    demonstrate_policy_enforcement()
    demonstrate_budget_alerting()
    demonstrate_compliance_audit_trail()

    # Summary
    print_cribl_pipeline_summary()
    print_key_benefits()

    print("=" * 70)
    print("🎉 Demo Complete!")
    print("=" * 70 + "\n")

    print("📚 Next Steps:")
    print("1. Review docs/integrations/cribl.md for detailed setup")
    print("2. Configure Cribl Stream pipelines (see summary above)")
    print("3. Set up downstream destinations (Datadog, Splunk, S3, etc.)")
    print("4. Test with production workloads")
    print("5. Monitor Cribl metrics and optimize routing rules")
    print()

    print("💡 Configuration Files:")
    print("   - GenOps config: See setup_genops_cribl_integration() above")
    print("   - Cribl pipelines: examples/cribl/pipelines/*.yml (coming soon)")
    print("   - Quickstart guide: docs/cribl-quickstart.md (coming soon)")
    print()

    print("🔗 Documentation:")
    print("   - Full integration guide: docs/integrations/cribl.md")
    print("   - Cribl Stream docs: https://docs.cribl.io")
    print("   - GenOps AI docs: https://github.com/KoshiHQ/GenOps-AI")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running demo: {e}")
        print("   Check that GenOps AI is installed: pip install genops-ai")
        print("   Set CRIBL_OTLP_ENDPOINT and CRIBL_AUTH_TOKEN if needed")
