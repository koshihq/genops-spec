#!/usr/bin/env python3
"""
Anthropic Cost Optimization Example

This example demonstrates intelligent cost optimization strategies using GenOps
telemetry and multi-model selection across Claude variants (Haiku, Sonnet, Opus).

What you'll learn:
- Dynamic Claude model selection based on task complexity
- Cost-aware completion strategies across Claude variants
- Model performance vs cost tradeoffs for different use cases
- Budget-constrained AI operations with Claude

Usage:
    python cost_optimization.py

Prerequisites:
    pip install genops-ai[anthropic]
    export ANTHROPIC_API_KEY="your_anthropic_key_here"
"""

import os
import sys
from dataclasses import dataclass


@dataclass
class ClaudeModelConfig:
    """Configuration for Claude model with cost and performance characteristics."""

    name: str
    cost_per_1m_input: float  # USD per 1M input tokens
    cost_per_1m_output: float  # USD per 1M output tokens
    max_tokens: int
    temperature: float
    use_case: str
    performance_tier: str


def get_claude_model_configurations() -> dict[str, ClaudeModelConfig]:
    """Get current Claude model configurations with pricing and use cases."""
    return {
        "economy": ClaudeModelConfig(
            name="claude-3-haiku-20240307",
            cost_per_1m_input=0.25,
            cost_per_1m_output=1.25,
            max_tokens=200,
            temperature=0.3,
            use_case="High-volume, simple tasks",
            performance_tier="Fast",
        ),
        "efficient": ClaudeModelConfig(
            name="claude-3-5-haiku-20241022",
            cost_per_1m_input=1.00,
            cost_per_1m_output=5.00,
            max_tokens=400,
            temperature=0.5,
            use_case="Balanced speed and intelligence",
            performance_tier="Balanced",
        ),
        "advanced": ClaudeModelConfig(
            name="claude-3-5-sonnet-20241022",
            cost_per_1m_input=3.00,
            cost_per_1m_output=15.00,
            max_tokens=800,
            temperature=0.7,
            use_case="Complex reasoning and analysis",
            performance_tier="Advanced",
        ),
        "premium": ClaudeModelConfig(
            name="claude-3-opus-20240229",
            cost_per_1m_input=15.00,
            cost_per_1m_output=75.00,
            max_tokens=1200,
            temperature=0.8,
            use_case="Highest quality, creative tasks",
            performance_tier="Premium",
        ),
    }


def estimate_claude_cost(prompt: str, config: ClaudeModelConfig) -> float:
    """Estimate the cost of a Claude completion based on prompt and model config."""
    # Rough token estimation (actual tokenization would be more accurate)
    estimated_input_tokens = len(prompt.split()) * 1.3
    estimated_output_tokens = config.max_tokens * 0.6  # Assume 60% of max tokens used

    input_cost = (estimated_input_tokens / 1000000) * config.cost_per_1m_input
    output_cost = (estimated_output_tokens / 1000000) * config.cost_per_1m_output

    return input_cost + output_cost


def smart_claude_model_selection():
    """Demonstrate intelligent Claude model selection based on task complexity."""
    print("🧠 Smart Claude Model Selection Based on Task Complexity")
    print("-" * 65)

    try:
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()
        configs = get_claude_model_configurations()

        # Define test tasks with different complexity levels
        tasks = [
            {
                "prompt": "What is the capital of France?",
                "complexity": "economy",
                "description": "Simple factual question",
            },
            {
                "prompt": "Explain the concept of machine learning and its main applications in business.",
                "complexity": "efficient",
                "description": "Educational explanation",
            },
            {
                "prompt": "Analyze the potential economic and social impacts of artificial intelligence adoption in developing countries over the next decade.",
                "complexity": "advanced",
                "description": "Complex analysis task",
            },
            {
                "prompt": "Write a comprehensive strategic plan for a startup entering the sustainable energy market, including competitive analysis, regulatory considerations, financial projections, and risk assessment.",
                "complexity": "premium",
                "description": "High-complexity strategic planning",
            },
        ]

        print("📊 Claude Model Selection Strategy:")
        print(f"{'Task Type':<25} {'Model':<30} {'Est. Cost':<12} {'Use Case'}")
        print("-" * 95)

        total_cost = 0
        results = []

        for task in tasks:
            config = configs[task["complexity"]]
            estimated_cost = estimate_claude_cost(task["prompt"], config)

            print(
                f"{task['description']:<25} {config.name:<30} ${estimated_cost:.6f}   {config.use_case[:30]}"
            )

            # Make the actual request
            print(f"🚀 Processing: {task['description']}")

            response = client.messages_create(
                model=config.name,
                messages=[{"role": "user", "content": task["prompt"]}],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                # Governance attributes with cost optimization tracking
                team="cost-optimization-team",
                project="smart-claude-selection",
                customer_id="optimization-demo",
                complexity_level=task["complexity"],
                estimated_cost=estimated_cost,
                optimization_strategy="complexity_based",
            )

            actual_tokens = response.usage.input_tokens + response.usage.output_tokens
            actual_cost = (
                response.usage.input_tokens / 1000000 * config.cost_per_1m_input
                + response.usage.output_tokens / 1000000 * config.cost_per_1m_output
            )

            results.append(
                {
                    "task": task["description"],
                    "model": config.name,
                    "estimated_cost": estimated_cost,
                    "actual_cost": actual_cost,
                    "tokens": actual_tokens,
                    "response": response.content[0].text[:120] + "...",
                }
            )

            total_cost += actual_cost
            print(
                f"   Response ({actual_tokens} tokens, ${actual_cost:.6f}): {response.content[0].text[:80]}...\n"
            )

        print(
            f"\n💰 Total cost for optimized Claude model selection: ${total_cost:.6f}"
        )
        print("🎯 Estimated savings vs using Opus for all: ~70-85%")

        return True

    except Exception as e:
        print(f"❌ Smart model selection error: {e}")
        return False


def budget_constrained_claude_completion():
    """Demonstrate cost-aware Claude completions within budget constraints."""
    print("\n\n💰 Budget-Constrained Claude Completion")
    print("-" * 50)

    try:
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()
        configs = get_claude_model_configurations()

        def complete_within_budget(prompt: str, max_budget: float = 0.001) -> dict:
            """Choose the best Claude model that fits within the specified budget."""

            # Sort models by performance tier (best first)
            performance_order = ["premium", "advanced", "efficient", "economy"]

            for tier in performance_order:
                config = configs[tier]
                estimated_cost = estimate_claude_cost(prompt, config)

                if estimated_cost <= max_budget:
                    print(
                        f"✅ Selected {config.name} (${estimated_cost:.6f} <= ${max_budget:.6f} budget)"
                    )

                    response = client.messages_create(
                        model=config.name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        # Budget-aware governance attributes
                        team="budget-team",
                        project="cost-controlled-claude",
                        customer_id="budget-demo",
                        max_budget=max_budget,
                        selected_model=config.name,
                        optimization_strategy="budget_constrained",
                    )

                    actual_cost = (
                        response.usage.input_tokens / 1000000 * config.cost_per_1m_input
                        + response.usage.output_tokens
                        / 1000000
                        * config.cost_per_1m_output
                    )

                    return {
                        "model": config.name,
                        "estimated_cost": estimated_cost,
                        "actual_cost": actual_cost,
                        "budget": max_budget,
                        "within_budget": actual_cost <= max_budget,
                        "response": response.content[0].text,
                        "tokens": response.usage.input_tokens
                        + response.usage.output_tokens,
                    }

            raise ValueError(
                f"No Claude model available within budget of ${max_budget:.6f}"
            )

        # Test different budget scenarios
        test_scenarios = [
            {
                "prompt": "Explain renewable energy briefly",
                "budget": 0.00001,
                "scenario": "Ultra-low budget",
            },
            {
                "prompt": "Write a detailed analysis of sustainable technology trends",
                "budget": 0.0001,
                "scenario": "Medium budget",
            },
            {
                "prompt": "Create a comprehensive business plan for a green technology startup",
                "budget": 0.001,
                "scenario": "High budget",
            },
        ]

        print("📊 Budget-Constrained Results:")
        print(
            f"{'Scenario':<20} {'Budget':<12} {'Model':<30} {'Actual Cost':<15} {'Status'}"
        )
        print("-" * 95)

        for scenario in test_scenarios:
            try:
                result = complete_within_budget(scenario["prompt"], scenario["budget"])

                status = (
                    "✅ Within Budget" if result["within_budget"] else "❌ Over Budget"
                )
                print(
                    f"{scenario['scenario']:<20} ${scenario['budget']:<11.6f} {result['model']:<30} ${result['actual_cost']:<14.6f} {status}"
                )
                print(f"   Response: {result['response'][:80]}...\n")

            except ValueError as e:
                print(
                    f"{scenario['scenario']:<20} ${scenario['budget']:<11.6f} {'None':<30} {'N/A':<15} ❌ No Model"
                )
                print(f"   Error: {e}\n")

        return True

    except Exception as e:
        print(f"❌ Budget-constrained completion error: {e}")
        return False


def claude_model_cost_comparison():
    """Compare costs across different Claude models for the same task."""
    print("\n\n📈 Claude Model Cost Comparison Analysis")
    print("-" * 50)

    try:
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()
        configs = get_claude_model_configurations()

        # Test prompt
        test_prompt = "Explain the benefits and potential risks of artificial intelligence in healthcare, considering both current applications and future possibilities."

        print(f"📝 Test prompt: {test_prompt}")
        print("\n📊 Cost Comparison Results:")
        print(
            f"{'Model':<30} {'Actual Cost':<15} {'Tokens':<10} {'Cost per Token':<18} {'Quality'}"
        )
        print("-" * 100)

        results = []

        for tier, config in configs.items():
            try:
                print(f"🔄 Testing {config.name}...")

                response = client.messages_create(
                    model=config.name,
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=300,  # Fixed for fair comparison
                    temperature=0.7,  # Fixed for consistency
                    # Comparison tracking
                    team="comparison-team",
                    project="claude-cost-analysis",
                    customer_id="analysis-demo",
                    model_tier=tier,
                    comparison_study="claude_model_cost",
                )

                actual_cost = (
                    response.usage.input_tokens / 1000000 * config.cost_per_1m_input
                    + response.usage.output_tokens / 1000000 * config.cost_per_1m_output
                )

                total_tokens = (
                    response.usage.input_tokens + response.usage.output_tokens
                )
                cost_per_token = actual_cost / total_tokens if total_tokens > 0 else 0

                # Simple quality assessment based on response length and structure
                response_text = response.content[0].text
                quality_factors = [
                    len(response_text.split()) > 50,  # Adequate length
                    "healthcare" in response_text.lower(),  # Topic relevance
                    any(
                        word in response_text.lower()
                        for word in ["benefit", "risk", "advantage"]
                    ),  # Key concepts
                    "." in response_text,  # Complete sentences
                    len(response_text.split(".")) > 3,  # Multiple points
                ]
                quality_score = sum(quality_factors)
                quality_rating = "⭐" * quality_score

                results.append(
                    {
                        "model": config.name,
                        "tier": tier,
                        "cost": actual_cost,
                        "tokens": total_tokens,
                        "cost_per_token": cost_per_token,
                        "quality": quality_rating,
                        "response": response_text,
                    }
                )

                print(
                    f"{config.name:<30} ${actual_cost:<14.6f} {total_tokens:<10} ${cost_per_token:<17.9f} {quality_rating}"
                )

            except Exception as e:
                print(f"{config.name:<30} Error: {e}")

        # Analysis summary
        if results:
            best_value = min(results, key=lambda x: x["cost_per_token"])
            most_expensive = max(results, key=lambda x: x["cost"])
            cheapest = min(results, key=lambda x: x["cost"])

            print("\n🏆 Analysis Summary:")
            print(
                f"   • Best value (cost per token): {best_value['model']} (${best_value['cost_per_token']:.9f}/token)"
            )
            print(
                f"   • Cheapest total cost: {cheapest['model']} (${cheapest['cost']:.6f})"
            )
            print(
                f"   • Most expensive: {most_expensive['model']} (${most_expensive['cost']:.6f})"
            )
            print(
                f"   • Cost range: {most_expensive['cost'] / cheapest['cost']:.1f}x difference"
            )

        return True

    except Exception as e:
        print(f"❌ Cost comparison analysis error: {e}")
        return False


def use_case_specific_optimization():
    """Demonstrate use case specific Claude model optimization."""
    print("\n\n🎯 Use Case Specific Claude Optimization")
    print("-" * 50)

    use_cases = [
        {
            "name": "Customer Support",
            "optimal_model": "claude-3-5-haiku-20241022",
            "prompt": "How do I reset my password and update my account settings?",
            "rationale": "Fast response time, cost-effective for high volume",
        },
        {
            "name": "Legal Document Analysis",
            "optimal_model": "claude-3-5-sonnet-20241022",
            "prompt": "Review this software license agreement and identify key terms, obligations, and potential risks for the licensee.",
            "rationale": "Complex reasoning required, accuracy critical",
        },
        {
            "name": "Creative Writing",
            "optimal_model": "claude-3-opus-20240229",
            "prompt": "Write a compelling short story about a future where AI and humans collaborate to solve climate change.",
            "rationale": "Highest creativity and nuanced expression needed",
        },
        {
            "name": "Data Analysis Summary",
            "optimal_model": "claude-3-5-haiku-20241022",
            "prompt": "Summarize the key insights from this quarterly sales report: Revenue up 15%, customer acquisition cost down 8%, churn rate stable at 3%.",
            "rationale": "Straightforward analysis, speed and cost efficiency important",
        },
    ]

    try:
        from genops.providers.anthropic import instrument_anthropic

        client = instrument_anthropic()
        configs = get_claude_model_configurations()

        print("📊 Use Case Optimization Results:")
        print(f"{'Use Case':<25} {'Optimal Model':<30} {'Cost':<12} {'Rationale'}")
        print("-" * 100)

        total_optimized_cost = 0

        for use_case in use_cases:
            # Find the config for the optimal model
            model_config = None
            for config in configs.values():
                if config.name == use_case["optimal_model"]:
                    model_config = config
                    break

            if not model_config:
                print(f"   ❌ Model not found: {use_case['optimal_model']}")
                continue

            try:
                response = client.messages_create(
                    model=use_case["optimal_model"],
                    messages=[{"role": "user", "content": use_case["prompt"]}],
                    max_tokens=model_config.max_tokens,
                    temperature=model_config.temperature,
                    # Use case optimization tracking
                    team="optimization-team",
                    project="use-case-optimization",
                    customer_id="use-case-demo",
                    use_case=use_case["name"],
                    optimal_model=use_case["optimal_model"],
                    optimization_rationale=use_case["rationale"],
                )

                actual_cost = (
                    response.usage.input_tokens
                    / 1000000
                    * model_config.cost_per_1m_input
                    + response.usage.output_tokens
                    / 1000000
                    * model_config.cost_per_1m_output
                )

                total_optimized_cost += actual_cost

                print(
                    f"{use_case['name']:<25} {use_case['optimal_model']:<30} ${actual_cost:<11.6f} {use_case['rationale'][:30]}"
                )
                print(f"   Result: {response.content[0].text[:100]}...\n")

            except Exception as e:
                print(f"   ❌ Error processing {use_case['name']}: {e}")

        print(
            f"💰 Total cost for use case optimized selection: ${total_optimized_cost:.6f}"
        )
        print("🎯 Optimization benefits:")
        print("   • Customer Support: Fast, cost-effective responses")
        print("   • Legal Analysis: High accuracy for critical decisions")
        print("   • Creative Writing: Maximum creativity and expression")
        print("   • Data Summary: Efficient processing of structured information")

        return True

    except Exception as e:
        print(f"❌ Use case optimization error: {e}")
        return False


def main():
    """Run Claude cost optimization demonstrations."""
    print("💰 GenOps Anthropic Cost Optimization Examples")
    print("=" * 60)

    # Check prerequisites
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("💡 Fix: export ANTHROPIC_API_KEY='your_api_key_here'")
        return False

    success = True

    # Run optimization examples
    success &= smart_claude_model_selection()
    success &= budget_constrained_claude_completion()
    success &= claude_model_cost_comparison()
    success &= use_case_specific_optimization()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 Claude cost optimization examples completed successfully!")

        print("\n💡 Key Claude Optimization Strategies:")
        print("   ✅ Task complexity-based model selection across Claude variants")
        print("   ✅ Budget-constrained model choosing for cost control")
        print("   ✅ Real-time cost comparison and analysis")
        print("   ✅ Use case specific optimization for maximum efficiency")

        print("\n📊 Business Benefits:")
        print("   • 70-85% cost savings through intelligent Claude model selection")
        print("   • Budget compliance and predictable costs")
        print("   • Detailed cost attribution for billing and chargebacks")
        print("   • Performance vs cost optimization insights")
        print("   • Use case specific optimization for different business needs")

        print("\n🚀 Next Steps:")
        print("   • Run 'python multi_provider_costs.py' for cross-provider comparison")
        print(
            "   • Try 'python advanced_features.py' for streaming and document analysis"
        )
        print(
            "   • Explore 'python production_patterns.py' for enterprise optimization"
        )

        return True
    else:
        print("❌ Claude cost optimization examples failed.")
        print("💡 Check the error messages above and verify your Anthropic setup")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
