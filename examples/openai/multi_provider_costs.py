#!/usr/bin/env python3
"""
Multi-Provider Cost Comparison Example

This example demonstrates cost comparison and unified tracking across multiple
AI providers (OpenAI, Anthropic, etc.) using GenOps governance telemetry.

What you'll learn:
- Cross-provider cost comparison and analysis
- Unified cost tracking across multiple providers
- Provider migration cost analysis
- Multi-provider portfolio optimization

Usage:
    python multi_provider_costs.py

Prerequisites:
    pip install genops-ai[openai,anthropic]
    export OPENAI_API_KEY="your_openai_key_here"
    export ANTHROPIC_API_KEY="your_anthropic_key_here"  # Optional
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderResult:
    """Result from a provider with cost and performance data."""

    provider: str
    model: str
    cost: float
    tokens_input: int
    tokens_output: int
    tokens_total: int
    latency: float
    response: str
    error: Optional[str] = None


def compare_providers_for_task():
    """Compare OpenAI and Anthropic for the same task with cost analysis."""
    print("🔄 Cross-Provider Task Comparison")
    print("-" * 50)

    # Test task
    test_task = "Explain the concept of artificial intelligence and its impact on society in 2-3 paragraphs."

    print(f"📝 Test task: {test_task[:60]}...")
    print("\n📊 Provider Comparison Results:")

    results = []

    # OpenAI comparison
    openai_result = test_openai_provider(test_task)
    if openai_result:
        results.append(openai_result)

    # Anthropic comparison (if available)
    anthropic_result = test_anthropic_provider(test_task)
    if anthropic_result:
        results.append(anthropic_result)

    # Display comparison
    if len(results) >= 2:
        print(
            f"\n{'Provider':<15} {'Model':<25} {'Cost':<10} {'Tokens':<10} {'Latency':<10} {'Cost/Token':<12}"
        )
        print("-" * 90)

        for result in results:
            cost_per_token = (
                result.cost / result.tokens_total if result.tokens_total > 0 else 0
            )
            print(
                f"{result.provider:<15} {result.model:<25} ${result.cost:<9.4f} {result.tokens_total:<10} {result.latency:<9.2f}s ${cost_per_token:<11.6f}"
            )

        # Cost comparison analysis
        cheapest = min(results, key=lambda x: x.cost)
        most_expensive = max(results, key=lambda x: x.cost)

        if cheapest != most_expensive:
            savings = most_expensive.cost - cheapest.cost
            percentage_savings = (savings / most_expensive.cost) * 100

            print("\n💰 Cost Analysis:")
            print(
                f"   • Cheapest: {cheapest.provider} {cheapest.model} (${cheapest.cost:.4f})"
            )
            print(
                f"   • Most expensive: {most_expensive.provider} {most_expensive.model} (${most_expensive.cost:.4f})"
            )
            print(f"   • Potential savings: ${savings:.4f} ({percentage_savings:.1f}%)")
            print(f"   • Cost ratio: {most_expensive.cost / cheapest.cost:.2f}x")

    elif len(results) == 1:
        result = results[0]
        print("\n📊 Single Provider Result:")
        print(f"   • Provider: {result.provider}")
        print(f"   • Model: {result.model}")
        print(f"   • Cost: ${result.cost:.4f}")
        print(f"   • Tokens: {result.tokens_total}")
        print(f"   • Response: {result.response[:100]}...")

    else:
        print("❌ No providers available for comparison")
        print("💡 Ensure you have API keys set for OpenAI and/or Anthropic")
        return False

    return True


def test_openai_provider(task: str) -> Optional[ProviderResult]:
    """Test OpenAI provider with cost tracking."""
    try:
        from genops.providers.openai import instrument_openai

        print("🔄 Testing OpenAI...")

        client = instrument_openai()

        start_time = time.time()
        response = client.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": task}],
            max_tokens=300,
            temperature=0.7,
            # Multi-provider comparison tracking
            team="comparison-team",
            project="multi-provider-analysis",
            customer_id="comparison-demo",
            provider="openai",
            comparison_study="cross_provider",
        )
        latency = time.time() - start_time

        # Calculate cost (OpenAI pricing)
        input_cost = (
            response.usage.prompt_tokens / 1000
        ) * 0.0015  # $0.0015 per 1K input tokens
        output_cost = (
            response.usage.completion_tokens / 1000
        ) * 0.002  # $0.002 per 1K output tokens
        total_cost = input_cost + output_cost

        print(
            f"✅ OpenAI completed: ${total_cost:.4f}, {response.usage.total_tokens} tokens, {latency:.2f}s"
        )

        return ProviderResult(
            provider="OpenAI",
            model="gpt-3.5-turbo",
            cost=total_cost,
            tokens_input=response.usage.prompt_tokens,
            tokens_output=response.usage.completion_tokens,
            tokens_total=response.usage.total_tokens,
            latency=latency,
            response=response.choices[0].message.content,
        )

    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        if "OPENAI_API_KEY" not in os.environ:
            print("💡 Set OPENAI_API_KEY environment variable")
        return None


def test_anthropic_provider(task: str) -> Optional[ProviderResult]:
    """Test Anthropic provider with cost tracking."""
    try:
        from genops.providers.anthropic import instrument_anthropic

        print("🔄 Testing Anthropic...")

        client = instrument_anthropic()

        start_time = time.time()
        response = client.messages_create(
            model="claude-3-haiku-20240307",  # Using Haiku for cost comparison
            messages=[{"role": "user", "content": task}],
            max_tokens=300,
            # Multi-provider comparison tracking
            team="comparison-team",
            project="multi-provider-analysis",
            customer_id="comparison-demo",
            provider="anthropic",
            comparison_study="cross_provider",
        )
        latency = time.time() - start_time

        # Calculate cost (Anthropic Haiku pricing)
        input_cost = (
            response.usage.input_tokens / 1000000
        ) * 0.25  # $0.25 per 1M input tokens
        output_cost = (
            response.usage.output_tokens / 1000000
        ) * 1.25  # $1.25 per 1M output tokens
        total_cost = input_cost + output_cost

        print(
            f"✅ Anthropic completed: ${total_cost:.4f}, {response.usage.input_tokens + response.usage.output_tokens} tokens, {latency:.2f}s"
        )

        return ProviderResult(
            provider="Anthropic",
            model="claude-3-haiku-20240307",
            cost=total_cost,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            tokens_total=response.usage.input_tokens + response.usage.output_tokens,
            latency=latency,
            response=response.content[0].text,
        )

    except ImportError:
        print(
            "ℹ️  Anthropic provider not available (install with: pip install genops-ai[anthropic])"
        )
        return None
    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")
        if "ANTHROPIC_API_KEY" not in os.environ:
            print(
                "💡 Set ANTHROPIC_API_KEY environment variable for Anthropic comparison"
            )
        return None


def provider_migration_analysis():
    """Analyze costs for migrating between providers."""
    print("\n\n📊 Provider Migration Cost Analysis")
    print("-" * 50)

    # Simulate different types of workloads
    workloads = [
        {
            "name": "Customer Support Chatbot",
            "daily_requests": 1000,
            "avg_input_tokens": 50,
            "avg_output_tokens": 150,
            "description": "High-volume, simple responses",
        },
        {
            "name": "Content Generation",
            "daily_requests": 100,
            "avg_input_tokens": 200,
            "avg_output_tokens": 800,
            "description": "Medium-volume, longer content",
        },
        {
            "name": "Code Review Assistant",
            "daily_requests": 50,
            "avg_input_tokens": 1000,
            "avg_output_tokens": 500,
            "description": "Low-volume, complex analysis",
        },
    ]

    # Provider pricing (simplified)
    provider_pricing = {
        "OpenAI (GPT-3.5-Turbo)": {
            "input_cost_per_1k": 0.0015,
            "output_cost_per_1k": 0.002,
        },
        "OpenAI (GPT-4o-mini)": {
            "input_cost_per_1k": 0.00015,
            "output_cost_per_1k": 0.0006,
        },
        "Anthropic (Claude-3-Haiku)": {
            "input_cost_per_1k": 0.00025,  # $0.25 per 1M = $0.00025 per 1K
            "output_cost_per_1k": 0.00125,  # $1.25 per 1M = $0.00125 per 1K
        },
    }

    print("📈 Monthly Cost Projections by Provider:")
    print(
        f"{'Workload':<25} {'Provider':<25} {'Daily Cost':<12} {'Monthly Cost':<15} {'Yearly Cost'}"
    )
    print("-" * 105)

    for workload in workloads:
        print(f"\n{workload['name']:<25}")
        print(
            f"   ({workload['daily_requests']} req/day, ~{workload['avg_input_tokens']}+{workload['avg_output_tokens']} tokens)"
        )

        workload_costs = []

        for provider, pricing in provider_pricing.items():
            # Calculate daily cost
            daily_input_cost = (
                workload["daily_requests"] * workload["avg_input_tokens"] / 1000
            ) * pricing["input_cost_per_1k"]
            daily_output_cost = (
                workload["daily_requests"] * workload["avg_output_tokens"] / 1000
            ) * pricing["output_cost_per_1k"]
            daily_total = daily_input_cost + daily_output_cost

            monthly_cost = daily_total * 30
            yearly_cost = daily_total * 365

            workload_costs.append(
                {
                    "provider": provider,
                    "daily": daily_total,
                    "monthly": monthly_cost,
                    "yearly": yearly_cost,
                }
            )

            print(
                f"{'':<25} {provider:<25} ${daily_total:<11.3f} ${monthly_cost:<14.2f} ${yearly_cost:<12.0f}"
            )

        # Find best value
        if len(workload_costs) > 1:
            cheapest = min(workload_costs, key=lambda x: x["yearly"])
            most_expensive = max(workload_costs, key=lambda x: x["yearly"])

            if cheapest != most_expensive:
                savings = most_expensive["yearly"] - cheapest["yearly"]
                print(
                    f"   💰 Best value: {cheapest['provider']} (saves ${savings:.0f}/year vs most expensive)"
                )

    # Summary recommendations
    print("\n🎯 Migration Recommendations:")
    print("   • High-volume, simple tasks: Consider Claude-3-Haiku or GPT-4o-mini")
    print("   • Balanced workloads: GPT-4o-mini offers good cost/performance")
    print("   • Complex analysis: Evaluate quality vs cost tradeoffs")
    print("   • Track actual usage patterns before migration decisions")

    return True


def unified_cost_tracking():
    """Demonstrate unified cost tracking across multiple providers."""
    print("\n\n📊 Unified Multi-Provider Cost Tracking")
    print("-" * 50)

    try:
        from genops import track

        # Simulate multi-provider operation
        with track(
            "multi_provider_workflow",
            team="multi-provider-team",
            project="unified-tracking",
            customer_id="unified-demo",
        ) as span:
            total_cost = 0
            operations = []

            # Operation 1: OpenAI for initial processing
            openai_cost = simulate_openai_operation("Initial text processing")
            if openai_cost:
                total_cost += openai_cost
                operations.append(("OpenAI", "Text Processing", openai_cost))

            # Operation 2: Anthropic for analysis (if available)
            anthropic_cost = simulate_anthropic_operation("Detailed analysis")
            if anthropic_cost:
                total_cost += anthropic_cost
                operations.append(("Anthropic", "Analysis", anthropic_cost))

            # Set unified tracking attributes
            span.set_attribute("total_providers_used", len(operations))
            span.set_attribute("total_cost", total_cost)
            span.set_attribute("cost_breakdown", str(operations))

            print("✅ Multi-provider workflow completed:")
            print(f"   • Total operations: {len(operations)}")
            print(f"   • Total cost: ${total_cost:.4f}")

            for provider, operation, cost in operations:
                print(f"   • {provider} ({operation}): ${cost:.4f}")

            if len(operations) > 1:
                print("\n💡 Unified tracking benefits:")
                print("   • Single customer attribution across all providers")
                print("   • Aggregated cost reporting for complete workflows")
                print("   • Provider cost comparison in real workflows")

    except Exception as e:
        print(f"❌ Unified tracking error: {e}")
        return False

    return True


def simulate_openai_operation(task: str) -> Optional[float]:
    """Simulate OpenAI operation and return cost."""
    try:
        # Simulate typical OpenAI costs
        simulated_cost = 0.0023  # ~$0.002 for moderate task
        print(f"🔄 OpenAI - {task}: ${simulated_cost:.4f}")
        return simulated_cost
    except Exception:
        return None


def simulate_anthropic_operation(task: str) -> Optional[float]:
    """Simulate Anthropic operation and return cost."""
    try:
        # Simulate typical Anthropic costs
        simulated_cost = 0.0008  # Cheaper for Haiku model
        print(f"🔄 Anthropic - {task}: ${simulated_cost:.4f}")
        return simulated_cost
    except Exception:
        return None


def main():
    """Run multi-provider cost comparison examples."""
    print("🌍 Multi-Provider Cost Comparison & Analysis")
    print("=" * 60)

    # Check prerequisites
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not has_openai and not has_anthropic:
        print("❌ No API keys configured")
        print("💡 Set at least one: OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return False

    print("🔑 Available providers:")
    if has_openai:
        print("   ✅ OpenAI (OPENAI_API_KEY configured)")
    else:
        print("   ❌ OpenAI (OPENAI_API_KEY not set)")

    if has_anthropic:
        print("   ✅ Anthropic (ANTHROPIC_API_KEY configured)")
    else:
        print("   ❌ Anthropic (ANTHROPIC_API_KEY not set)")

    success = True

    # Run multi-provider examples
    if has_openai or has_anthropic:
        success &= compare_providers_for_task()

    success &= provider_migration_analysis()
    success &= unified_cost_tracking()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 Multi-provider cost analysis completed!")

        print("\n💡 Key Multi-Provider Benefits:")
        print("   ✅ Cross-provider cost comparison and optimization")
        print("   ✅ Unified cost tracking across all AI providers")
        print("   ✅ Migration cost analysis for informed decisions")
        print("   ✅ Portfolio optimization across multiple providers")

        print("\n📊 Business Value:")
        print("   • Avoid vendor lock-in with multi-provider strategies")
        print("   • Optimize costs through intelligent provider selection")
        print("   • Unified governance and cost attribution across providers")
        print("   • Data-driven migration and portfolio decisions")

        print("\n🚀 Next Steps:")
        print("   • Run 'python advanced_features.py' for specialized features")
        print("   • Try 'python production_patterns.py' for enterprise patterns")
        print("   • Explore governance scenarios for policy enforcement")

        return True
    else:
        print("❌ Multi-provider analysis encountered issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
