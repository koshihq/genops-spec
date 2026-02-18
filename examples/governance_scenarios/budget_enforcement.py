#!/usr/bin/env python3
"""
🚨 Prevent AI Budget Overruns - Complete Governance Scenario

This example demonstrates how GenOps AI prevents runaway AI costs through
automatic budget enforcement and real-time monitoring.

BUSINESS PROBLEM:
Your team's OpenAI bill went from $500 to $5,000 last month because someone
accidentally ran an expensive batch job. Finance is asking tough questions.

GENOPS SOLUTION:
- Set monthly/daily/per-operation budget limits
- Automatically block operations that would exceed budgets
- Get real-time alerts when approaching limits
- Full audit trail of all AI spending

Run this example to see budget enforcement in action!
"""

import logging
import os

# GenOps imports
from genops.core.policy import PolicyResult, PolicyViolationError, register_policy
from genops.core.telemetry import GenOpsTelemetry
from genops.providers.openai import instrument_openai

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_budget_policies():
    """
    Set up realistic budget policies for different scenarios.

    In production, these would be configured via config files or environment variables.
    """
    print("\n🏛️ SETTING UP BUDGET GOVERNANCE POLICIES")
    print("=" * 60)

    # Policy 1: Monthly team budget limit
    register_policy(
        name="monthly_team_budget",
        description="Prevent team from exceeding monthly AI budget",
        enforcement_level=PolicyResult.BLOCKED,
        conditions={
            "max_cost": 100.0,  # $100/month team budget
            "time_period": "monthly",
        },
    )
    print("✅ Monthly team budget: $100.00")

    # Policy 2: Per-operation cost limit
    register_policy(
        name="operation_cost_limit",
        description="Block individual operations over cost threshold",
        enforcement_level=PolicyResult.BLOCKED,
        conditions={
            "max_cost": 5.0  # $5 per operation max
        },
    )
    print("✅ Per-operation limit: $5.00")

    # Policy 3: Daily customer budget (with warning)
    register_policy(
        name="customer_daily_budget",
        description="Warn when customer approaches daily budget",
        enforcement_level=PolicyResult.WARNING,
        conditions={
            "max_cost": 25.0,  # $25/day per customer
            "time_period": "daily",
        },
    )
    print("✅ Customer daily warning: $25.00")


def demonstrate_budget_enforcement():
    """
    Show budget policies in action with realistic AI operations.
    """
    print("\n🤖 DEMONSTRATING AI OPERATIONS WITH BUDGET ENFORCEMENT")
    print("=" * 60)

    # Initialize telemetry
    telemetry = GenOpsTelemetry()

    # Scenario 1: Normal operation within budget
    print("\n📊 Scenario 1: Normal AI operation (within budget)")
    try:
        with telemetry.trace_operation(
            operation_name="customer_support_classification",
            operation_type="ai.inference",
            team="support-team",
            project="ticket-classifier",
            customer_id="enterprise-123",
        ) as span:
            # Simulate a normal AI operation cost
            estimated_cost = 0.15  # $0.15 - well within limits

            # Check budget policies before operation
            from genops.core.policy import _policy_engine

            context = {
                "cost": estimated_cost,
                "team": "support-team",
                "customer": "enterprise-123",
                "operation": "customer_support_classification",
            }

            # Check operation cost policy
            result = _policy_engine.evaluate_policy("operation_cost_limit", context)
            print(f"   💰 Operation cost: ${estimated_cost:.2f}")
            print(f"   🛡️ Policy check: {result.result.value}")

            if result.result == PolicyResult.BLOCKED:
                raise PolicyViolationError(
                    "operation_cost_limit", result.reason, result.metadata
                )

            # Record the operation telemetry
            telemetry.record_cost(
                span=span,
                cost=estimated_cost,
                currency="USD",
                provider="openai",
                model="gpt-3.5-turbo",
                input_tokens=120,
                output_tokens=45,
            )

            print("   ✅ Operation completed successfully!")

    except PolicyViolationError as e:
        print(f"   🚫 BLOCKED: {e}")

    # Scenario 2: Operation exceeding per-operation limit
    print("\n🚨 Scenario 2: Expensive operation (exceeds per-operation limit)")
    try:
        with telemetry.trace_operation(
            operation_name="document_analysis_batch",
            operation_type="ai.inference",
            team="content-team",
            project="document-processor",
        ) as span:
            # Simulate expensive batch operation
            estimated_cost = 7.50  # $7.50 - exceeds $5 limit!

            context = {
                "cost": estimated_cost,
                "team": "content-team",
                "operation": "document_analysis_batch",
            }

            result = _policy_engine.evaluate_policy("operation_cost_limit", context)
            print(f"   💰 Operation cost: ${estimated_cost:.2f}")
            print(f"   🛡️ Policy check: {result.result.value}")

            if result.result == PolicyResult.BLOCKED:
                # Record the policy violation in telemetry
                telemetry.record_policy(
                    span=span,
                    policy_name="operation_cost_limit",
                    result="blocked",
                    reason=result.reason,
                    metadata=result.metadata,
                )
                raise PolicyViolationError(
                    "operation_cost_limit", result.reason, result.metadata
                )

    except PolicyViolationError as e:
        print(f"   🚫 BLOCKED: {e}")
        print(
            "   💡 Suggestion: Break this into smaller operations or request budget increase"
        )

    # Scenario 3: Customer approaching daily budget (warning)
    print("\n⚠️  Scenario 3: Customer approaching daily budget (warning level)")
    try:
        with telemetry.trace_operation(
            operation_name="product_recommendations",
            operation_type="ai.inference",
            team="ml-team",
            project="recommendation-engine",
            customer_id="premium-456",
        ) as span:
            # Simulate customer close to daily budget
            estimated_cost = 22.0  # $22 - approaching $25 limit

            context = {
                "cost": estimated_cost,
                "customer": "premium-456",
                "operation": "product_recommendations",
            }

            result = _policy_engine.evaluate_policy("customer_daily_budget", context)
            print(f"   💰 Operation cost: ${estimated_cost:.2f}")
            print(f"   🛡️ Policy check: {result.result.value}")

            if result.result == PolicyResult.WARNING:
                print(f"   ⚠️  WARNING: {result.reason}")
                print(
                    "   📧 Alert would be sent to: finance@company.com, ml-team@company.com"
                )

                # Record warning in telemetry
                telemetry.record_policy(
                    span=span,
                    policy_name="customer_daily_budget",
                    result="warning",
                    reason=result.reason,
                    metadata=result.metadata,
                )

            # Operation proceeds with warning
            telemetry.record_cost(
                span=span,
                cost=estimated_cost,
                currency="USD",
                provider="openai",
                model="gpt-4",
                input_tokens=850,
                output_tokens=320,
            )

            print("   ✅ Operation completed with warning logged")

    except PolicyViolationError as e:
        print(f"   🚫 BLOCKED: {e}")


def demonstrate_real_openai_integration():
    """
    Show how budget enforcement works with real OpenAI API calls.

    This requires OPENAI_API_KEY environment variable to be set.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Skipping OpenAI integration demo (no API key)")
        print("   Set OPENAI_API_KEY environment variable to see real API integration")
        return

    print("\n🔗 REAL OPENAI API INTEGRATION WITH BUDGET ENFORCEMENT")
    print("=" * 60)

    try:
        # Instrument OpenAI client with governance
        client = instrument_openai(api_key=os.getenv("OPENAI_API_KEY"))

        print("✅ OpenAI client instrumented with GenOps governance")

        # Make a real API call with governance attributes
        print("\n📞 Making real OpenAI API call with budget tracking...")

        response = client.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Explain AI governance in one sentence."}
            ],
            max_tokens=50,
            # Governance attributes
            team="demo-team",
            project="governance-demo",
            customer_id="demo-customer",
        )

        print("📝 Response:", response.choices[0].message.content.strip())
        print("✅ Cost and governance telemetry automatically recorded!")

    except Exception as e:
        print(f"❌ Error with OpenAI integration: {e}")


def show_telemetry_data():
    """
    Show what telemetry data is captured for budget monitoring.
    """
    print("\n📊 TELEMETRY DATA CAPTURED")
    print("=" * 60)

    sample_telemetry = {
        "genops.operation.name": "customer_support_classification",
        "genops.operation.type": "ai.inference",
        "genops.team": "support-team",
        "genops.project": "ticket-classifier",
        "genops.customer": "enterprise-123",
        "genops.cost.total": 0.15,
        "genops.cost.currency": "USD",
        "genops.cost.provider": "openai",
        "genops.cost.model": "gpt-3.5-turbo",
        "genops.tokens.input": 120,
        "genops.tokens.output": 45,
        "genops.tokens.total": 165,
        "genops.policy.name": "operation_cost_limit",
        "genops.policy.result": "allowed",
        "genops.policy.reason": "Under cost threshold",
    }

    print("📈 Sample telemetry attributes sent to your observability platform:")
    for key, value in sample_telemetry.items():
        print(f"   {key}: {value}")

    print("\n💡 This data enables:")
    print("   • Cost dashboards by team, project, customer")
    print("   • Budget alerts and notifications")
    print("   • Audit trails for compliance")
    print("   • Predictive budget forecasting")
    print("   • Chargeback and cost attribution")


def main():
    """
    Run the complete budget enforcement demonstration.
    """
    print("🚨 GenOps AI: Prevent AI Budget Overruns Demo")
    print("=" * 80)
    print("\nThis demo shows how GenOps AI prevents runaway AI costs through")
    print(
        "automatic budget enforcement, real-time monitoring, and governance policies."
    )

    # Setup
    setup_budget_policies()

    # Demonstrate scenarios
    demonstrate_budget_enforcement()

    # Real API integration
    demonstrate_real_openai_integration()

    # Show telemetry
    show_telemetry_data()

    print("\n🎯 KEY TAKEAWAYS")
    print("=" * 60)
    print("✅ Automatic budget enforcement prevents cost overruns")
    print("✅ Real-time policy evaluation before operations execute")
    print("✅ Comprehensive telemetry for cost attribution and monitoring")
    print("✅ Seamless integration with existing OpenAI workflows")
    print("✅ OpenTelemetry-native data exports to any observability platform")

    print("\n📚 NEXT STEPS")
    print("=" * 60)
    print("1. Review the governance policies and adjust limits for your use case")
    print("2. Set up OpenTelemetry integration with your observability platform")
    print("3. Configure alerting based on budget warnings and violations")
    print("4. Implement custom policies for your specific governance requirements")
    print("5. Scale across your organization's AI operations")

    print("\n🔗 Learn more: https://github.com/KoshiHQ/GenOps-AI/tree/main/docs")


if __name__ == "__main__":
    main()
