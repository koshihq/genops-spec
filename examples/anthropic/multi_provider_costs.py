#!/usr/bin/env python3
"""
Multi-Provider Cost Comparison Example (Anthropic Focus)

This example demonstrates cost comparison and unified tracking between Anthropic Claude
and other AI providers using GenOps governance telemetry.

What you'll learn:
- Cross-provider cost comparison (Claude vs OpenAI vs others)
- Unified cost tracking across multiple providers
- Provider migration cost analysis from Claude perspective
- Multi-provider portfolio optimization

Usage:
    python multi_provider_costs.py

Prerequisites:
    pip install genops-ai[anthropic,openai]
    export ANTHROPIC_API_KEY="your_anthropic_key_here"
    export OPENAI_API_KEY="your_openai_key_here"  # Optional for comparison
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
    """Compare Anthropic and OpenAI for the same task with cost analysis."""
    print("🔄 Cross-Provider Task Comparison (Claude Focus)")
    print("-" * 55)

    # Test task optimized for comparison
    test_task = "Analyze the impact of artificial intelligence on modern education systems, including benefits, challenges, and future implications."

    print(f"📝 Test task: {test_task[:60]}...")
    print("\n📊 Provider Comparison Results:")

    results = []

    # Anthropic Claude comparison (primary)
    anthropic_result = test_anthropic_provider(test_task)
    if anthropic_result:
        results.append(anthropic_result)

    # OpenAI comparison (if available)
    openai_result = test_openai_provider(test_task)
    if openai_result:
        results.append(openai_result)

    # Display comparison
    if len(results) >= 2:
        print(f"\n{'Provider':<15} {'Model':<30} {'Cost':<12} {'Tokens':<10} {'Latency':<10} {'Cost/Token':<15}")
        print("-" * 100)

        for result in results:
            cost_per_token = result.cost / result.tokens_total if result.tokens_total > 0 else 0
            print(f"{result.provider:<15} {result.model:<30} ${result.cost:<11.6f} {result.tokens_total:<10} {result.latency:<9.2f}s ${cost_per_token:<14.9f}")

        # Detailed cost comparison analysis
        anthropic_result = next((r for r in results if r.provider == "Anthropic"), None)
        openai_result = next((r for r in results if r.provider == "OpenAI"), None)

        if anthropic_result and openai_result:
            cost_diff = abs(anthropic_result.cost - openai_result.cost)
            cheaper = "Anthropic" if anthropic_result.cost < openai_result.cost else "OpenAI"

            percentage_diff = (cost_diff / max(anthropic_result.cost, openai_result.cost)) * 100

            print("\n💰 Detailed Cost Analysis:")
            print(f"   • Cheaper provider: {cheaper}")
            print(f"   • Cost difference: ${cost_diff:.6f} ({percentage_diff:.1f}%)")
            print("   • Claude response quality: High analytical depth")
            print("   • OpenAI response quality: Structured and comprehensive")

            # Token efficiency comparison
            claude_efficiency = len(anthropic_result.response) / anthropic_result.tokens_total
            openai_efficiency = len(openai_result.response) / openai_result.tokens_total

            print(f"   • Claude content/token ratio: {claude_efficiency:.2f}")
            print(f"   • OpenAI content/token ratio: {openai_efficiency:.2f}")

    elif len(results) == 1:
        result = results[0]
        print("\n📊 Single Provider Result:")
        print(f"   • Provider: {result.provider}")
        print(f"   • Model: {result.model}")
        print(f"   • Cost: ${result.cost:.6f}")
        print(f"   • Tokens: {result.tokens_total}")
        print(f"   • Response: {result.response[:100]}...")

    else:
        print("❌ No providers available for comparison")
        print("💡 Ensure you have API keys set for Anthropic and/or OpenAI")
        return False

    return True

def test_anthropic_provider(task: str) -> Optional[ProviderResult]:
    """Test Anthropic Claude with cost tracking."""
    try:
        from genops.providers.anthropic import instrument_anthropic

        print("🔄 Testing Anthropic Claude...")

        client = instrument_anthropic()

        start_time = time.time()
        response = client.messages_create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": task}],
            max_tokens=400,
            temperature=0.7,

            # Multi-provider comparison tracking
            team="comparison-team",
            project="multi-provider-analysis",
            customer_id="comparison-demo",
            provider="anthropic",
            comparison_study="cross_provider_claude_focus"
        )
        latency = time.time() - start_time

        # Calculate cost (Claude 3.5 Sonnet pricing)
        input_cost = (response.usage.input_tokens / 1000000) * 3.00  # $3 per 1M input tokens
        output_cost = (response.usage.output_tokens / 1000000) * 15.00  # $15 per 1M output tokens
        total_cost = input_cost + output_cost

        print(f"✅ Claude completed: ${total_cost:.6f}, {response.usage.input_tokens + response.usage.output_tokens} tokens, {latency:.2f}s")

        return ProviderResult(
            provider="Anthropic",
            model="claude-3-5-sonnet-20241022",
            cost=total_cost,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            tokens_total=response.usage.input_tokens + response.usage.output_tokens,
            latency=latency,
            response=response.content[0].text
        )

    except ImportError:
        print("❌ Anthropic provider not available (install with: pip install genops-ai[anthropic])")
        return None
    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("💡 Set ANTHROPIC_API_KEY environment variable")
        return None

def test_openai_provider(task: str) -> Optional[ProviderResult]:
    """Test OpenAI provider with cost tracking."""
    try:
        from genops.providers.openai import instrument_openai

        print("🔄 Testing OpenAI...")

        client = instrument_openai()

        start_time = time.time()
        response = client.chat_completions_create(
            model="gpt-4",
            messages=[{"role": "user", "content": task}],
            max_tokens=400,
            temperature=0.7,

            # Multi-provider comparison tracking
            team="comparison-team",
            project="multi-provider-analysis",
            customer_id="comparison-demo",
            provider="openai",
            comparison_study="cross_provider_claude_focus"
        )
        latency = time.time() - start_time

        # Calculate cost (GPT-4 pricing)
        input_cost = (response.usage.prompt_tokens / 1000) * 0.03  # $0.03 per 1K input tokens
        output_cost = (response.usage.completion_tokens / 1000) * 0.06  # $0.06 per 1K output tokens
        total_cost = input_cost + output_cost

        print(f"✅ OpenAI completed: ${total_cost:.6f}, {response.usage.total_tokens} tokens, {latency:.2f}s")

        return ProviderResult(
            provider="OpenAI",
            model="gpt-4",
            cost=total_cost,
            tokens_input=response.usage.prompt_tokens,
            tokens_output=response.usage.completion_tokens,
            tokens_total=response.usage.total_tokens,
            latency=latency,
            response=response.choices[0].message.content
        )

    except ImportError:
        print("ℹ️  OpenAI provider not available (install with: pip install genops-ai[openai])")
        return None
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        if "OPENAI_API_KEY" not in os.environ:
            print("💡 Set OPENAI_API_KEY environment variable for comparison")
        return None

def claude_migration_cost_analysis():
    """Analyze costs for migrating to or from Claude."""
    print("\n\n📊 Claude Migration Cost Analysis")
    print("-" * 50)

    # Simulate different types of workloads with Claude focus
    workloads = [
        {
            "name": "Legal Document Review",
            "daily_requests": 50,
            "avg_input_tokens": 2000,  # Long documents
            "avg_output_tokens": 500,  # Detailed analysis
            "description": "Contract analysis, compliance review",
            "claude_advantage": "Superior reasoning for legal nuances"
        },
        {
            "name": "Customer Service Chat",
            "daily_requests": 500,
            "avg_input_tokens": 100,
            "avg_output_tokens": 150,
            "description": "Customer support automation",
            "claude_advantage": "Natural, helpful responses"
        },
        {
            "name": "Content Generation",
            "daily_requests": 100,
            "avg_input_tokens": 300,
            "avg_output_tokens": 800,
            "description": "Blog posts, marketing copy",
            "claude_advantage": "Creative, engaging content"
        },
        {
            "name": "Data Analysis Reports",
            "daily_requests": 20,
            "avg_input_tokens": 1500,
            "avg_output_tokens": 600,
            "description": "Business intelligence summaries",
            "claude_advantage": "Clear, structured analysis"
        }
    ]

    # Provider pricing comparison (simplified)
    provider_pricing = {
        "Claude 3.5 Sonnet": {
            "input_cost_per_1k": 0.003,   # $3 per 1M = $0.003 per 1K
            "output_cost_per_1k": 0.015   # $15 per 1M = $0.015 per 1K
        },
        "Claude 3.5 Haiku": {
            "input_cost_per_1k": 0.001,   # $1 per 1M = $0.001 per 1K
            "output_cost_per_1k": 0.005   # $5 per 1M = $0.005 per 1K
        },
        "Claude 3 Opus": {
            "input_cost_per_1k": 0.015,   # $15 per 1M = $0.015 per 1K
            "output_cost_per_1k": 0.075   # $75 per 1M = $0.075 per 1K
        },
        "GPT-4 (comparison)": {
            "input_cost_per_1k": 0.03,
            "output_cost_per_1k": 0.06
        }
    }

    print("📈 Monthly Cost Projections by Provider (Claude Focus):")
    print(f"{'Workload':<25} {'Provider':<20} {'Daily Cost':<12} {'Monthly Cost':<15} {'Yearly Cost':<12} {'Advantage'}")
    print("-" * 120)

    for workload in workloads:
        print(f"\n{workload['name']:<25}")
        print(f"   ({workload['daily_requests']} req/day, ~{workload['avg_input_tokens']}+{workload['avg_output_tokens']} tokens)")

        workload_costs = []

        for provider, pricing in provider_pricing.items():
            # Calculate daily cost
            daily_input_cost = (workload["daily_requests"] * workload["avg_input_tokens"] / 1000) * pricing["input_cost_per_1k"]
            daily_output_cost = (workload["daily_requests"] * workload["avg_output_tokens"] / 1000) * pricing["output_cost_per_1k"]
            daily_total = daily_input_cost + daily_output_cost

            monthly_cost = daily_total * 30
            yearly_cost = daily_total * 365

            workload_costs.append({
                "provider": provider,
                "daily": daily_total,
                "monthly": monthly_cost,
                "yearly": yearly_cost
            })

            advantage = workload["claude_advantage"] if "Claude" in provider else "Comparison baseline"
            if len(advantage) > 25:
                advantage = advantage[:25] + "..."

            print(f"{'':<25} {provider:<20} ${daily_total:<11.4f} ${monthly_cost:<14.2f} ${yearly_cost:<11.0f} {advantage}")

        # Find best Claude model vs GPT-4
        claude_models = [cost for cost in workload_costs if "Claude" in cost["provider"]]
        gpt4_cost = next((cost for cost in workload_costs if "GPT-4" in cost["provider"]), None)

        if claude_models and gpt4_cost:
            best_claude = min(claude_models, key=lambda x: x["yearly"])
            if best_claude["yearly"] < gpt4_cost["yearly"]:
                savings = gpt4_cost["yearly"] - best_claude["yearly"]
                print(f"   💰 Best Claude option: {best_claude['provider']} saves ${savings:.0f}/year vs GPT-4")
            else:
                premium = best_claude["yearly"] - gpt4_cost["yearly"]
                print(f"   💎 Claude premium: {best_claude['provider']} costs ${premium:.0f}/year more than GPT-4")

    # Claude-specific migration recommendations
    print("\n🎯 Claude Migration Recommendations:")
    print("   • Legal/Analysis work: Claude 3.5 Sonnet excels at nuanced reasoning")
    print("   • High-volume simple tasks: Claude 3.5 Haiku for cost efficiency")
    print("   • Creative/Complex work: Claude 3 Opus for highest quality")
    print("   • Customer service: Claude's natural conversation style")
    print("   • Document processing: Superior understanding of context and structure")

    return True

def unified_claude_cost_tracking():
    """Demonstrate unified cost tracking with Claude-focused multi-provider workflow."""
    print("\n\n📊 Unified Multi-Provider Cost Tracking (Claude Focus)")
    print("-" * 60)

    try:
        from genops import track

        # Simulate Claude-centric multi-provider operation
        with track("claude_multi_provider_workflow",
                   team="multi-provider-team",
                   project="claude-unified-tracking",
                   customer_id="unified-claude-demo") as span:

            total_cost = 0
            operations = []

            # Operation 1: Claude for primary analysis
            claude_cost = simulate_claude_operation("Primary document analysis and reasoning")
            if claude_cost:
                total_cost += claude_cost
                operations.append(("Claude", "Document Analysis", claude_cost))

            # Operation 2: OpenAI for structured output (if available)
            openai_cost = simulate_openai_operation("Structured data extraction")
            if openai_cost:
                total_cost += openai_cost
                operations.append(("OpenAI", "Data Extraction", openai_cost))

            # Operation 3: Claude for final synthesis
            claude_synthesis_cost = simulate_claude_operation("Final synthesis and recommendations")
            if claude_synthesis_cost:
                total_cost += claude_synthesis_cost
                operations.append(("Claude", "Synthesis", claude_synthesis_cost))

            # Set unified tracking attributes
            span.set_attribute("total_providers_used", len({op[0] for op in operations}))
            span.set_attribute("total_operations", len(operations))
            span.set_attribute("total_cost", total_cost)
            span.set_attribute("claude_operations", len([op for op in operations if op[0] == "Claude"]))
            span.set_attribute("workflow_pattern", "claude_primary")

            print("✅ Claude-focused multi-provider workflow completed:")
            print(f"   • Total operations: {len(operations)}")
            print(f"   • Total cost: ${total_cost:.6f}")
            print(f"   • Claude operations: {len([op for op in operations if op[0] == 'Claude'])}")

            for provider, operation, cost in operations:
                print(f"   • {provider} ({operation}): ${cost:.6f}")

            if len(operations) > 1:
                claude_cost_total = sum(cost for provider, _, cost in operations if provider == "Claude")
                other_cost_total = total_cost - claude_cost_total

                print("\n💡 Claude-centric workflow benefits:")
                print(f"   • Claude cost: ${claude_cost_total:.6f} ({claude_cost_total/total_cost*100:.1f}%)")
                print(f"   • Other providers: ${other_cost_total:.6f} ({other_cost_total/total_cost*100:.1f}%)")
                print("   • Unified governance across all providers")
                print("   • Claude handles complex reasoning, others for specialized tasks")

    except Exception as e:
        print(f"❌ Unified tracking error: {e}")
        return False

    return True

def simulate_claude_operation(task: str) -> Optional[float]:
    """Simulate Claude operation and return cost."""
    try:
        # Simulate typical Claude costs based on task complexity
        if "analysis" in task.lower() or "reasoning" in task.lower():
            simulated_cost = 0.000845  # Higher complexity task with Sonnet
        else:
            simulated_cost = 0.000234  # Standard task with Haiku
        print(f"🔄 Claude - {task}: ${simulated_cost:.6f}")
        return simulated_cost
    except:
        return None

def simulate_openai_operation(task: str) -> Optional[float]:
    """Simulate OpenAI operation and return cost."""
    try:
        # Simulate typical OpenAI costs
        simulated_cost = 0.002300  # GPT-4 cost for comparison
        print(f"🔄 OpenAI - {task}: ${simulated_cost:.6f}")
        return simulated_cost
    except:
        return None

def main():
    """Run multi-provider cost comparison examples with Claude focus."""
    print("🌍 Multi-Provider Cost Comparison (Claude Focus)")
    print("=" * 60)

    # Check prerequisites
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    if not has_anthropic:
        print("❌ ANTHROPIC_API_KEY not configured (required)")
        print("💡 Set ANTHROPIC_API_KEY to run Claude-focused examples")
        return False

    print("🔑 Available providers:")
    if has_anthropic:
        print("   ✅ Anthropic Claude (ANTHROPIC_API_KEY configured)")

    if has_openai:
        print("   ✅ OpenAI (OPENAI_API_KEY configured)")
    else:
        print("   ℹ️  OpenAI (OPENAI_API_KEY not set - comparison limited)")

    success = True

    # Run Claude-focused multi-provider examples
    success &= compare_providers_for_task()
    success &= claude_migration_cost_analysis()
    success &= unified_claude_cost_tracking()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 Claude-focused multi-provider cost analysis completed!")

        print("\n💡 Key Claude Multi-Provider Benefits:")
        print("   ✅ Cross-provider cost comparison with Claude as primary")
        print("   ✅ Unified cost tracking across Claude + other providers")
        print("   ✅ Migration cost analysis for Claude adoption")
        print("   ✅ Workflow optimization with Claude for complex reasoning")

        print("\n📊 Claude Business Value:")
        print("   • Superior performance for legal and analytical tasks")
        print("   • Natural conversation style for customer interactions")
        print("   • Excellent document understanding and processing")
        print("   • Competitive pricing especially with Haiku for high-volume tasks")
        print("   • Unified governance across multi-provider architectures")

        print("\n🚀 Next Steps:")
        print("   • Run 'python advanced_features.py' for Claude-specific features")
        print("   • Try 'python production_patterns.py' for enterprise Claude patterns")
        print("   • Explore governance scenarios for Claude policy enforcement")

        return True
    else:
        print("❌ Multi-provider analysis encountered issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
