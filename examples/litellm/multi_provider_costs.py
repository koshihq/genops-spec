#!/usr/bin/env python3
"""
LiteLLM Multi-Provider Cost Optimization with GenOps

Demonstrates cost optimization across 100+ LLM providers using LiteLLM's
unified interface with GenOps governance. This example shows how to:

- Compare costs across equivalent models from different providers
- Implement cost-aware model selection strategies
- Track spending attribution across teams and projects
- Optimize for cost vs performance trade-offs

Usage:
    export OPENAI_API_KEY="your_key"
    export ANTHROPIC_API_KEY="your_key"  # Optional but recommended
    export GOOGLE_API_KEY="your_key"     # Optional but recommended
    python multi_provider_costs.py

Features:
    - Real-time cost comparison across providers
    - Intelligent model selection based on cost/performance
    - Team-based budget allocation and tracking
    - Provider switching strategies for cost optimization
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ModelBenchmark:
    """Benchmark data for a specific model."""

    provider: str
    model: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: float
    quality_score: float  # 1-10 scale
    use_case: str


# Model equivalency matrix for cost comparison
MODEL_EQUIVALENTS = {
    "fast_chat": [
        ModelBenchmark(
            "openai", "gpt-3.5-turbo", 0.0015, 0.002, 800, 8.0, "Fast general chat"
        ),
        ModelBenchmark(
            "anthropic",
            "claude-3-haiku",
            0.00025,
            0.00125,
            900,
            8.2,
            "Fast thoughtful responses",
        ),
        ModelBenchmark(
            "google", "gemini-pro", 0.0005, 0.0015, 1200, 7.8, "Fast multimodal"
        ),
        ModelBenchmark(
            "cohere", "command-light", 0.0003, 0.0006, 700, 7.5, "Fast enterprise"
        ),
    ],
    "powerful_reasoning": [
        ModelBenchmark("openai", "gpt-4", 0.03, 0.06, 2000, 9.2, "Advanced reasoning"),
        ModelBenchmark(
            "anthropic",
            "claude-3-sonnet",
            0.003,
            0.015,
            2200,
            9.0,
            "Balanced power/speed",
        ),
        ModelBenchmark(
            "google", "gemini-1.5-pro", 0.0035, 0.01, 2500, 8.8, "Advanced multimodal"
        ),
        ModelBenchmark(
            "anthropic", "claude-3-opus", 0.015, 0.075, 3000, 9.5, "Maximum capability"
        ),
    ],
    "coding": [
        ModelBenchmark("openai", "gpt-4", 0.03, 0.06, 2000, 9.0, "Code generation"),
        ModelBenchmark(
            "anthropic", "claude-3-sonnet", 0.003, 0.015, 2200, 8.8, "Code analysis"
        ),
        ModelBenchmark(
            "google", "gemini-1.5-pro", 0.0035, 0.01, 2500, 8.5, "Multimodal coding"
        ),
    ],
}


def check_setup():
    """Check if required packages and API keys are available."""
    print("🔍 Checking setup for multi-provider cost optimization...")

    # Check imports
    try:
        import litellm  # noqa: F401

        from genops.providers.litellm import (  # noqa: F401
            auto_instrument,
            get_cost_summary,
        )

        print("✅ LiteLLM and GenOps available")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install: pip install litellm genops[litellm]")
        return False

    # Check API keys
    providers_available = []
    api_checks = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "Cohere": "COHERE_API_KEY",
    }

    for provider, env_var in api_checks.items():
        if os.getenv(env_var):
            providers_available.append(provider)
            print(f"✅ {provider} API key configured")

    if len(providers_available) < 2:
        print(f"⚠️  Only {len(providers_available)} provider(s) configured")
        print("💡 For best cost optimization, configure multiple providers:")
        for provider, env_var in api_checks.items():
            if provider not in providers_available:
                print(f"   export {env_var}=your_key")
        print("\n🎯 Proceeding with available providers...")

    return len(providers_available) > 0


class CostOptimizationEngine:
    """Engine for multi-provider cost optimization."""

    def __init__(self):
        self.benchmarks = MODEL_EQUIVALENTS
        self.usage_history = []

    def get_available_models(self, use_case: str) -> list[ModelBenchmark]:
        """Get available models for a use case based on API keys."""
        if use_case not in self.benchmarks:
            return []

        available_models = []
        for model in self.benchmarks[use_case]:
            if self._is_provider_available(model.provider):
                available_models.append(model)

        return available_models

    def _is_provider_available(self, provider: str) -> bool:
        """Check if provider API key is available."""
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "cohere": "COHERE_API_KEY",
        }
        return bool(os.getenv(key_mapping.get(provider)))

    def select_optimal_model(
        self, use_case: str, budget_priority: float = 0.7
    ) -> Optional[ModelBenchmark]:
        """
        Select optimal model based on cost/performance trade-off.

        Args:
            use_case: The use case category
            budget_priority: 0-1 scale, higher = more cost-sensitive
        """
        available_models = self.get_available_models(use_case)
        if not available_models:
            return None

        # Calculate optimization score (lower is better)
        best_model = None
        best_score = float("inf")

        for model in available_models:
            # Estimate cost per request (assuming 150 input + 50 output tokens)
            estimated_cost = (model.cost_per_1k_input * 0.15) + (
                model.cost_per_1k_output * 0.05
            )

            # Normalize metrics (0-1 scale)
            cost_score = estimated_cost / 0.1  # Normalize against $0.10 baseline
            latency_score = model.avg_latency_ms / 3000  # Normalize against 3s baseline
            quality_score = (
                10 - model.quality_score
            ) / 10  # Invert quality (lower is better)

            # Weighted optimization score
            optimization_score = budget_priority * cost_score + (
                1 - budget_priority
            ) * 0.5 * (latency_score + quality_score)

            if optimization_score < best_score:
                best_score = optimization_score
                best_model = model

        return best_model

    def compare_costs(
        self, use_case: str, input_tokens: int = 150, output_tokens: int = 50
    ):
        """Compare costs across all available models for a use case."""
        available_models = self.get_available_models(use_case)

        comparisons = []
        for model in available_models:
            cost = (
                model.cost_per_1k_input * input_tokens / 1000
                + model.cost_per_1k_output * output_tokens / 1000
            )

            comparisons.append(
                {
                    "provider": model.provider,
                    "model": model.model,
                    "cost": cost,
                    "cost_per_1k_tokens": (cost / (input_tokens + output_tokens))
                    * 1000,
                    "quality_score": model.quality_score,
                    "latency_ms": model.avg_latency_ms,
                    "use_case": model.use_case,
                }
            )

        # Sort by cost
        comparisons.sort(key=lambda x: x["cost"])
        return comparisons


def demo_cost_comparison():
    """Demonstrate cost comparison across providers."""
    print("\n" + "=" * 60)
    print("💰 Demo: Multi-Provider Cost Comparison")
    print("=" * 60)

    optimizer = CostOptimizationEngine()

    print("Comparing costs for common use cases across all available providers:\n")

    use_cases = ["fast_chat", "powerful_reasoning", "coding"]

    for use_case in use_cases:
        print(f"📊 {use_case.replace('_', ' ').title()} Models:")

        # Standard request: 150 input tokens, 50 output tokens
        comparisons = optimizer.compare_costs(use_case, 150, 50)

        if not comparisons:
            print("   ⚠️  No providers available for this use case")
            continue

        for i, comp in enumerate(comparisons):
            rank_emoji = (
                "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
            )

            print(f"   {rank_emoji} {comp['provider'].title()} ({comp['model']})")
            print(f"      Cost: ${comp['cost']:.6f} per request")
            print(
                f"      Quality: {comp['quality_score']}/10, Latency: {comp['latency_ms']}ms"
            )

            if i == 0 and len(comparisons) > 1:
                savings = (
                    (comparisons[-1]["cost"] - comp["cost"]) / comparisons[-1]["cost"]
                ) * 100
                print(f"      💡 {savings:.1f}% cheaper than most expensive option")

        print()


def demo_smart_model_selection():
    """Demonstrate intelligent model selection based on preferences."""
    print("\n" + "=" * 60)
    print("🧠 Demo: Intelligent Model Selection")
    print("=" * 60)

    import litellm

    from genops.providers.litellm import auto_instrument

    # Enable GenOps instrumentation for cost tracking
    auto_instrument(
        team="cost-optimization-demo",
        project="multi-provider-comparison",
        daily_budget_limit=5.0,  # Low limit for demo
        governance_policy="advisory",
    )

    optimizer = CostOptimizationEngine()

    scenarios = [
        ("fast_chat", 0.8, "Budget-focused: Need quick responses, cost is key"),
        (
            "powerful_reasoning",
            0.3,
            "Quality-focused: Complex reasoning, quality over cost",
        ),
        ("coding", 0.5, "Balanced: Good code quality at reasonable cost"),
    ]

    print("Testing intelligent model selection for different scenarios:\n")

    for use_case, budget_priority, description in scenarios:
        print(f"🎯 Scenario: {description}")

        # Select optimal model
        optimal_model = optimizer.select_optimal_model(use_case, budget_priority)

        if not optimal_model:
            print("   ⚠️  No suitable models available")
            continue

        print(
            f"   📍 Selected: {optimal_model.provider.title()} ({optimal_model.model})"
        )
        print(f"   💡 Reason: {optimal_model.use_case}")

        # Test with actual LiteLLM call
        try:
            print("   🔄 Testing with real API call...")

            start_time = time.time()

            response = litellm.completion(
                model=optimal_model.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Test message for {use_case.replace('_', ' ')}. Respond briefly.",
                    }
                ],
                max_tokens=20,
                timeout=10,
            )

            end_time = time.time()
            actual_latency = (end_time - start_time) * 1000

            print(f"   ✅ Success! Actual latency: {actual_latency:.0f}ms")

            # Show token usage if available
            if hasattr(response, "usage") and response.usage:
                tokens_used = getattr(response.usage, "total_tokens", "unknown")
                print(f"   📊 Tokens used: {tokens_used}")

        except Exception:
            print("   ⚠️  API call failed: [Error details redacted for security]")

        print()


def demo_budget_allocation():
    """Demonstrate budget-based provider allocation."""
    print("\n" + "=" * 60)
    print("📈 Demo: Budget-Based Provider Allocation")
    print("=" * 60)

    from genops.providers.litellm import get_cost_summary, reset_usage_stats

    # Reset stats for clean demo
    reset_usage_stats()

    print("Simulating budget allocation across different teams and projects:\n")

    # Team budget scenarios
    teams = [
        {"name": "research-team", "budget": 100.0, "use_case": "powerful_reasoning"},
        {"name": "customer-support", "budget": 50.0, "use_case": "fast_chat"},
        {"name": "dev-team", "budget": 75.0, "use_case": "coding"},
    ]

    optimizer = CostOptimizationEngine()

    for team in teams:
        print(f"👥 Team: {team['name']}")
        print(f"   Budget: ${team['budget']}")
        print(f"   Use case: {team['use_case']}")

        # Get cost comparison for their use case
        comparisons = optimizer.compare_costs(team["use_case"])

        if comparisons:
            cheapest = comparisons[0]
            most_expensive = comparisons[-1] if len(comparisons) > 1 else comparisons[0]

            # Calculate how many requests they can afford
            requests_cheapest = team["budget"] / cheapest["cost"]
            requests_expensive = team["budget"] / most_expensive["cost"]

            print(f"   💰 With cheapest option ({cheapest['provider']}):")
            print(f"      {requests_cheapest:.0f} requests possible")

            if len(comparisons) > 1:
                print(f"   💸 With most expensive ({most_expensive['provider']}):")
                print(f"      {requests_expensive:.0f} requests possible")
                print(
                    f"   🎯 Potential savings: {requests_cheapest - requests_expensive:.0f} extra requests"
                )

        print()

    # Show overall cost summary
    cost_summary = get_cost_summary(group_by="provider")
    if cost_summary["total_cost"] > 0:
        print("📊 Current Usage Summary:")
        print(f"   Total cost: ${cost_summary['total_cost']:.6f}")
        if cost_summary.get("cost_by_provider"):
            for provider, cost in cost_summary["cost_by_provider"].items():
                percentage = (cost / cost_summary["total_cost"]) * 100
                print(f"   {provider}: ${cost:.6f} ({percentage:.1f}%)")


def main():
    """Run the complete multi-provider cost optimization demonstration."""

    print("🌟 LiteLLM + GenOps: Multi-Provider Cost Optimization")
    print("=" * 60)
    print("Maximize value across 100+ LLM providers with unified governance")

    # Check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please resolve issues above.")
        return 1

    try:
        # Run demonstrations
        demo_cost_comparison()
        demo_smart_model_selection()
        demo_budget_allocation()

        print("\n" + "=" * 60)
        print("🎉 Multi-Provider Cost Optimization Complete!")

        print("\n🚀 Key Insights:")
        print("   ✅ Cost differences up to 90% between equivalent models")
        print("   ✅ Intelligent selection saves money while maintaining quality")
        print("   ✅ Budget allocation optimizes team spending")
        print("   ✅ Single GenOps integration tracks ALL providers")

        print("\n💡 Production Recommendations:")
        print("   • Configure 3+ providers for maximum optimization opportunities")
        print("   • Use budget_priority parameter to balance cost vs quality")
        print("   • Monitor usage patterns to refine model selection")
        print("   • Implement automated fallbacks for high-availability")

        print("\n📖 Next Steps:")
        print("   • Try production_patterns.py for scaling strategies")
        print("   • Explore compliance_monitoring.py for governance")
        print("   • Integrate cost optimization into your applications!")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 1

    except Exception:
        print("\n❌ Demo failed: [Error details redacted for security]")
        print("💡 For debugging, check your API key configuration")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
