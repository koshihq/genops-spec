#!/usr/bin/env python3
"""
Example: Cost Optimization and Budget Management

Complexity: ⭐⭐ Intermediate

This example demonstrates comprehensive cost tracking, optimization analysis,
and budget management for Flowise workflows. Includes cost estimation,
provider comparison, and optimization recommendations.

Prerequisites:
- Flowise instance running
- At least one chatflow created
- GenOps package installed

Usage:
    python 04_cost_optimization.py

Environment Variables:
    FLOWISE_BASE_URL: Flowise instance URL
    FLOWISE_API_KEY: API key (optional for local dev)
    GENOPS_TEAM: Team name for governance
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from genops.providers.flowise import instrument_flowise
from genops.providers.flowise_pricing import (
    FLOWISE_PRICING_TIERS,
    FlowiseCostCalculator,
    analyze_cost_optimization_opportunities,
)
from genops.providers.flowise_validation import validate_flowise_setup

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CostTrackingSession:
    """Track costs for a session of Flowise executions."""

    session_id: str
    team: str
    project: str
    customer_id: Optional[str] = None
    pricing_tier: str = "self_hosted"
    executions: list[dict] = field(default_factory=list)
    total_cost: Decimal = Decimal("0.0")
    total_tokens: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    def add_execution(
        self,
        chatflow_id: str,
        chatflow_name: str,
        question: str,
        response: Any,
        execution_time_ms: int,
    ):
        """Add execution data for cost tracking."""

        # Estimate tokens (rough approximation)
        response_text = self._extract_response_text(response)
        input_tokens = len(question.split()) * 1.3
        output_tokens = len(response_text.split()) * 1.3
        total_tokens = int(input_tokens + output_tokens)

        # Simulate provider costs (in real scenario, these would come from telemetry)
        provider_calls = self._simulate_provider_costs(total_tokens)

        # Calculate cost
        cost_calc = FlowiseCostCalculator(pricing_tier=self.pricing_tier)
        cost = cost_calc.calculate_execution_cost(
            chatflow_id,
            chatflow_name,
            provider_calls,
            execution_duration_ms=execution_time_ms,
        )

        execution_data = {
            "timestamp": datetime.now(),
            "chatflow_id": chatflow_id,
            "chatflow_name": chatflow_name,
            "question": question,
            "response": response,
            "execution_time_ms": execution_time_ms,
            "cost_data": cost,
            "tokens_input": int(input_tokens),
            "tokens_output": int(output_tokens),
            "tokens_total": total_tokens,
        }

        self.executions.append(execution_data)
        self.total_cost += cost.total_cost
        self.total_tokens += total_tokens

        return execution_data

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from response object."""
        if isinstance(response, dict):
            return (
                response.get("text")
                or response.get("answer")
                or response.get("content")
                or str(response)
            )
        return str(response)

    def _simulate_provider_costs(self, total_tokens: int) -> list[dict]:
        """Simulate underlying provider costs for demonstration."""
        # In real scenarios, this data would come from actual provider usage

        # Simulate different provider distributions
        if self.team == "budget-conscious":
            # Prefer cheaper providers
            return [
                {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "input_tokens": int(total_tokens * 0.6),
                    "output_tokens": int(total_tokens * 0.4),
                    "cost": total_tokens * 0.000001,  # Cheaper model
                }
            ]
        elif self.team == "performance-focused":
            # Prefer high-quality providers
            return [
                {
                    "provider": "anthropic",
                    "model": "claude-3-opus",
                    "input_tokens": int(total_tokens * 0.6),
                    "output_tokens": int(total_tokens * 0.4),
                    "cost": total_tokens * 0.000025,  # Premium model
                }
            ]
        else:
            # Balanced approach - mix of providers
            return [
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    "input_tokens": int(total_tokens * 0.3),
                    "output_tokens": int(total_tokens * 0.2),
                    "cost": total_tokens * 0.000015 * 0.5,
                },
                {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet",
                    "input_tokens": int(total_tokens * 0.3),
                    "output_tokens": int(total_tokens * 0.2),
                    "cost": total_tokens * 0.000008 * 0.5,
                },
            ]

    def get_cost_summary(self) -> dict[str, Any]:
        """Get comprehensive cost summary."""
        if not self.executions:
            return {"error": "No executions recorded"}

        # Calculate metrics
        duration_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        avg_cost_per_execution = self.total_cost / len(self.executions)
        avg_tokens_per_execution = self.total_tokens / len(self.executions)

        # Group by chatflow
        costs_by_flow = {}
        tokens_by_flow = {}
        executions_by_flow = {}

        for exec_data in self.executions:
            flow_name = exec_data["chatflow_name"]
            cost = exec_data["cost_data"].total_cost
            tokens = exec_data["tokens_total"]

            costs_by_flow[flow_name] = (
                costs_by_flow.get(flow_name, Decimal("0.0")) + cost
            )
            tokens_by_flow[flow_name] = tokens_by_flow.get(flow_name, 0) + tokens
            executions_by_flow[flow_name] = executions_by_flow.get(flow_name, 0) + 1

        return {
            "session_summary": {
                "session_id": self.session_id,
                "team": self.team,
                "project": self.project,
                "customer_id": self.customer_id,
                "pricing_tier": self.pricing_tier,
                "duration_hours": round(duration_hours, 2),
                "total_executions": len(self.executions),
                "total_cost": float(self.total_cost),
                "total_tokens": self.total_tokens,
                "avg_cost_per_execution": float(avg_cost_per_execution),
                "avg_tokens_per_execution": int(avg_tokens_per_execution),
            },
            "costs_by_flow": {k: float(v) for k, v in costs_by_flow.items()},
            "tokens_by_flow": tokens_by_flow,
            "executions_by_flow": executions_by_flow,
        }


def demonstrate_cost_tracking():
    """Demonstrate comprehensive cost tracking for Flowise executions."""

    print("💰 Cost Tracking Demonstration")
    print("=" * 50)

    # Configuration
    base_url = os.getenv("FLOWISE_BASE_URL", "http://localhost:3000")
    api_key = os.getenv("FLOWISE_API_KEY")
    team = os.getenv("GENOPS_TEAM", "cost-optimization-demo")

    # Step 1: Setup and validation
    print("📋 Step 1: Setting up cost tracking...")

    try:
        result = validate_flowise_setup(base_url, api_key)
        if not result.is_valid:
            print("❌ Setup validation failed.")
            return False

        flowise = instrument_flowise(
            base_url=base_url,
            api_key=api_key,
            team=team,
            project="cost-optimization",
            environment="development",
        )

        chatflows = flowise.get_chatflows()
        if not chatflows:
            print("❌ No chatflows available.")
            return False

        chatflow_id = chatflows[0].get("id")
        chatflow_name = chatflows[0].get("name", "Unnamed")
        print(f"✅ Using chatflow: {chatflow_name}")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

    # Step 2: Create cost tracking sessions for different scenarios
    scenarios = [
        {
            "name": "Budget-Conscious Team",
            "team": "budget-conscious",
            "pricing_tier": "self_hosted",
            "questions": [
                "What is machine learning?",
                "Explain neural networks simply.",
                "How does AI help businesses?",
            ],
        },
        {
            "name": "Performance-Focused Team",
            "team": "performance-focused",
            "pricing_tier": "cloud_pro",
            "questions": [
                "Conduct a detailed analysis of market trends in artificial intelligence adoption across enterprise sectors.",
                "Generate a comprehensive technical specification for implementing a distributed machine learning pipeline with real-time inference capabilities.",
                "Develop a strategic roadmap for AI transformation including risk assessment, resource planning, and ROI projections.",
            ],
        },
        {
            "name": "Balanced Approach Team",
            "team": "balanced-team",
            "pricing_tier": "cloud_starter",
            "questions": [
                "How can we optimize our customer service with AI?",
                "What are best practices for AI model deployment?",
                "Explain the ROI of implementing chatbots.",
            ],
        },
    ]

    sessions = []

    for scenario in scenarios:
        print(f"\n🔄 Step 2: Running '{scenario['name']}' scenario...")

        session = CostTrackingSession(
            session_id=f"cost-demo-{uuid.uuid4().hex[:8]}",
            team=scenario["team"],
            project="cost-optimization",
            customer_id=f"customer-{scenario['team']}",
            pricing_tier=scenario["pricing_tier"],
        )

        for i, question in enumerate(scenario["questions"], 1):
            print(f"   Executing question {i}/{len(scenario['questions'])}...")

            try:
                start_time = time.time()

                response = flowise.predict_flow(
                    chatflow_id=chatflow_id,
                    question=question,
                    team=scenario["team"],
                    customer_id=session.customer_id,
                )

                execution_time = int((time.time() - start_time) * 1000)

                session.add_execution(
                    chatflow_id, chatflow_name, question, response, execution_time
                )

            except Exception as e:
                logger.error(f"Execution failed: {e}")
                continue

        sessions.append(session)
        print(f"   ✅ Completed {len(session.executions)} executions")

    # Step 3: Analyze costs across scenarios
    print("\n📊 Step 3: Cost Analysis Across Scenarios")
    print("=" * 50)

    for session in sessions:
        summary = session.get_cost_summary()
        session_info = summary["session_summary"]

        print(f"\n📋 {session.team.replace('-', ' ').title()} Results:")
        print(f"   Pricing Tier: {session_info['pricing_tier']}")
        print(f"   Total Executions: {session_info['total_executions']}")
        print(f"   Total Cost: ${session_info['total_cost']:.6f}")
        print(f"   Total Tokens: {session_info['total_tokens']:,}")
        print(f"   Avg Cost/Execution: ${session_info['avg_cost_per_execution']:.6f}")
        print(f"   Avg Tokens/Execution: {session_info['avg_tokens_per_execution']}")

        # Show cost breakdown by flow if multiple flows were used
        if len(summary["costs_by_flow"]) > 1:
            print("   Cost by Flow:")
            for flow, cost in summary["costs_by_flow"].items():
                print(f"     {flow}: ${cost:.6f}")

    return len(sessions) > 0 and all(len(s.executions) > 0 for s in sessions)


def demonstrate_pricing_tiers():
    """Demonstrate different Flowise pricing tiers and their impact."""

    print("\n💳 Pricing Tiers Comparison")
    print("=" * 50)

    # Simulate monthly usage scenarios
    usage_scenarios = [
        {"name": "Small Team", "monthly_executions": 1000, "avg_tokens": 500},
        {"name": "Growing Startup", "monthly_executions": 15000, "avg_tokens": 800},
        {"name": "Enterprise Team", "monthly_executions": 100000, "avg_tokens": 1200},
    ]

    for scenario in usage_scenarios:
        print(f"\n📊 {scenario['name']} Scenario:")
        print(f"   Monthly Executions: {scenario['monthly_executions']:,}")
        print(f"   Average Tokens: {scenario['avg_tokens']}")

        print("   \n   Cost Estimates by Pricing Tier:")

        for tier_name, tier_info in FLOWISE_PRICING_TIERS.items():
            calculator = FlowiseCostCalculator(pricing_tier=tier_name)

            estimate = calculator.estimate_monthly_spend(
                expected_executions_per_month=scenario["monthly_executions"],
                average_tokens_per_execution=scenario["avg_tokens"],
                provider_distribution={"openai": 0.7, "anthropic": 0.3},
            )

            print(f"     {tier_info.name}:")
            print(f"       Total Cost: ${estimate['total_estimated_cost']:.2f}")
            print(f"       Platform Cost: ${estimate['flowise_platform_cost']:.2f}")
            print(f"       Provider Costs: ${estimate['total_provider_costs']:.2f}")

        # Find most cost-effective tier
        tier_costs = {}
        for tier_name in FLOWISE_PRICING_TIERS.keys():
            calculator = FlowiseCostCalculator(pricing_tier=tier_name)
            estimate = calculator.estimate_monthly_spend(
                scenario["monthly_executions"], scenario["avg_tokens"]
            )
            tier_costs[tier_name] = estimate["total_estimated_cost"]

        best_tier = min(tier_costs.keys(), key=lambda k: tier_costs[k])
        savings = (
            tier_costs[max(tier_costs.keys(), key=lambda k: tier_costs[k])]
            - tier_costs[best_tier]
        )

        print(f"   \n   💡 Recommendation: {FLOWISE_PRICING_TIERS[best_tier].name}")
        print(f"      Potential Monthly Savings: ${savings:.2f}")


def demonstrate_cost_optimization():
    """Demonstrate cost optimization analysis and recommendations."""

    print("\n🔍 Cost Optimization Analysis")
    print("=" * 50)

    # Simulate execution cost data for optimization analysis
    from genops.providers.flowise_pricing import FlowiseExecutionCost

    # Create sample execution costs representing different usage patterns
    execution_costs = []

    # High-cost executions (premium models)
    for _i in range(20):
        cost = FlowiseExecutionCost(
            flow_id="premium-analysis-v1",
            flow_name="Premium Document Analysis",
            base_execution_cost=Decimal("0.001"),
            execution_duration_ms=5000,
        )
        cost.add_provider_cost("anthropic", Decimal("0.025"))  # Expensive
        cost.add_token_cost("anthropic-claude-3-opus", 800, 400, Decimal("0.025"))
        execution_costs.append(cost)

    # Medium-cost executions (balanced models)
    for _i in range(50):
        cost = FlowiseExecutionCost(
            flow_id="balanced-chatbot-v1",
            flow_name="Balanced Customer Chatbot",
            base_execution_cost=Decimal("0.0008"),
            execution_duration_ms=2500,
        )
        cost.add_provider_cost("openai", Decimal("0.008"))
        cost.add_token_cost("openai-gpt-4", 400, 200, Decimal("0.008"))
        execution_costs.append(cost)

    # Low-cost executions (efficient models)
    for _i in range(100):
        cost = FlowiseExecutionCost(
            flow_id="efficient-support-v1",
            flow_name="Efficient Support Assistant",
            base_execution_cost=Decimal("0.0005"),
            execution_duration_ms=1500,
        )
        cost.add_provider_cost("openai", Decimal("0.002"))
        cost.add_token_cost("openai-gpt-3.5-turbo", 200, 100, Decimal("0.002"))
        execution_costs.append(cost)

    # Analyze optimization opportunities
    optimization = analyze_cost_optimization_opportunities(execution_costs)

    print("📊 Analysis Results:")
    print(f"   Total Analyzed Cost: ${optimization['total_analyzed_cost']:.2f}")
    print(
        f"   Analysis Period: {optimization['analysis_period_executions']} executions"
    )
    print(f"   Potential Savings: ${optimization['total_potential_savings']:.2f}")
    print(
        f"   Savings Percentage: {(optimization['total_potential_savings'] / optimization['total_analyzed_cost'] * 100):.1f}%"
    )

    print("\n💡 Optimization Recommendations:")
    for i, rec in enumerate(optimization["recommendations"], 1):
        print(f"   {i}. {rec['suggestion']}")
        print(f"      Potential Savings: {rec['potential_savings_percent']}%")
        if "current_cost" in rec:
            print(f"      Current Cost: ${rec['current_cost']:.2f}")

    print("\n📈 Cost Breakdown:")
    print("   By Provider:")
    for provider, cost in optimization["cost_breakdown"]["by_provider"].items():
        print(f"     {provider}: ${cost:.2f}")

    print("   By Flow:")
    for flow, cost in optimization["cost_breakdown"]["by_flow"].items():
        print(f"     {flow}: ${cost:.2f}")


def demonstrate_budget_monitoring():
    """Demonstrate budget monitoring and alerting patterns."""

    print("\n🎯 Budget Monitoring and Alerting")
    print("=" * 50)

    # Simulate different budget scenarios
    budget_scenarios = [
        {
            "name": "Daily Budget Limit",
            "daily_budget": 10.00,
            "monthly_budget": 300.00,
            "current_daily_spend": 8.50,
            "current_monthly_spend": 245.00,
        },
        {
            "name": "Monthly Budget Approaching",
            "daily_budget": 50.00,
            "monthly_budget": 1000.00,
            "current_daily_spend": 25.00,
            "current_monthly_spend": 850.00,
        },
        {
            "name": "Budget Exceeded",
            "daily_budget": 20.00,
            "monthly_budget": 500.00,
            "current_daily_spend": 22.50,
            "current_monthly_spend": 520.00,
        },
    ]

    for scenario in budget_scenarios:
        print(f"\n📊 {scenario['name']}:")

        daily_usage = scenario["current_daily_spend"] / scenario["daily_budget"] * 100
        monthly_usage = (
            scenario["current_monthly_spend"] / scenario["monthly_budget"] * 100
        )

        print(f"   Daily Budget: ${scenario['daily_budget']:.2f}")
        print(
            f"   Daily Spend: ${scenario['current_daily_spend']:.2f} ({daily_usage:.1f}%)"
        )
        print(f"   Monthly Budget: ${scenario['monthly_budget']:.2f}")
        print(
            f"   Monthly Spend: ${scenario['current_monthly_spend']:.2f} ({monthly_usage:.1f}%)"
        )

        # Generate alerts based on usage
        alerts = []

        if daily_usage >= 100:
            alerts.append("🚨 CRITICAL: Daily budget exceeded!")
        elif daily_usage >= 90:
            alerts.append("⚠️  WARNING: Daily budget 90% used")
        elif daily_usage >= 80:
            alerts.append("💡 INFO: Daily budget 80% used")

        if monthly_usage >= 100:
            alerts.append("🚨 CRITICAL: Monthly budget exceeded!")
        elif monthly_usage >= 90:
            alerts.append("⚠️  WARNING: Monthly budget 90% used")
        elif monthly_usage >= 80:
            alerts.append("💡 INFO: Monthly budget 80% used")

        if alerts:
            print("   Alerts:")
            for alert in alerts:
                print(f"     {alert}")
        else:
            print("   ✅ Budget usage within normal limits")

        # Suggest actions
        if monthly_usage >= 85:
            print("   Suggested Actions:")
            print("     • Review high-cost flows for optimization opportunities")
            print("     • Consider switching to more cost-effective models")
            print("     • Implement request throttling or quotas")
            print("     • Analyze usage patterns for anomalies")


def main():
    """Main example function."""

    try:
        print("🚀 Cost Optimization and Budget Management Example")
        print("=" * 60)

        # Run all demonstrations
        success = True

        # 1. Cost tracking
        if not demonstrate_cost_tracking():
            success = False

        # 2. Pricing tiers comparison
        demonstrate_pricing_tiers()

        # 3. Cost optimization analysis
        demonstrate_cost_optimization()

        # 4. Budget monitoring
        demonstrate_budget_monitoring()

        if success:
            print("\n🎉 Cost Optimization Example Complete!")
            print("=" * 50)
            print("✅ You've learned how to:")
            print("   • Track costs across different execution scenarios")
            print("   • Compare Flowise pricing tiers for cost optimization")
            print("   • Analyze usage patterns for optimization opportunities")
            print("   • Set up budget monitoring and alerting")
            print("   • Generate cost optimization recommendations")

            print("\n💡 Key Takeaways:")
            print("   • Different usage patterns have dramatically different costs")
            print("   • Choosing the right pricing tier can save significant money")
            print(
                "   • Regular cost analysis helps identify optimization opportunities"
            )
            print("   • Budget monitoring prevents unexpected overages")
            print("   • Provider and model selection significantly impact costs")

            print("\n📚 Next Steps:")
            print("   • Implement cost tracking in your production applications")
            print("   • Set up automated budget alerts and monitoring")
            print("   • Explore multi-tenant cost isolation (05_multi_tenant_saas.py)")
            print(
                "   • Try enterprise governance patterns (06_enterprise_governance.py)"
            )

        return success

    except Exception as e:
        logger.error(f"Example failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)
