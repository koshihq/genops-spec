#!/usr/bin/env python3
"""
OpenRouter Cost Optimization Example

Demonstrates intelligent cost optimization strategies using GenOps with OpenRouter.
Shows how to automatically select the most cost-effective models and providers
based on task requirements and budget constraints.

Usage:
    export OPENROUTER_API_KEY="your-key"
    python cost_optimization.py

Key features demonstrated:
- Cost-aware model selection
- Budget-constrained operations
- Task complexity-based routing
- Real-time cost optimization
"""

import os
import time
from dataclasses import dataclass


@dataclass
class TaskProfile:
    """Profile for different types of AI tasks."""
    name: str
    description: str
    complexity: str  # "simple", "medium", "complex"
    max_tokens: int
    quality_threshold: float  # 0.0 to 1.0
    latency_requirement: str  # "fast", "medium", "slow"
    cost_priority: str  # "low", "medium", "high"

def cost_optimization_demo():
    """Demonstrate intelligent cost optimization with OpenRouter."""

    print("💰 OpenRouter Cost Optimization with GenOps")
    print("=" * 50)

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Missing API key. Set OPENROUTER_API_KEY environment variable.")
        return

    try:
        from genops.providers.openrouter import instrument_openrouter
        from genops.providers.openrouter_pricing import (
            calculate_openrouter_cost,
            get_pricing_engine,
        )

        print("🔧 Setting up cost-optimized OpenRouter client...")
        client = instrument_openrouter(openrouter_api_key=api_key)
        get_pricing_engine()
        print("   ✅ Client and pricing engine ready")

        # Define different task profiles
        task_profiles = [
            TaskProfile(
                name="Quick FAQ Response",
                description="Simple customer service questions",
                complexity="simple",
                max_tokens=50,
                quality_threshold=0.7,
                latency_requirement="fast",
                cost_priority="high"  # Very cost-sensitive
            ),
            TaskProfile(
                name="Content Summarization",
                description="Summarize articles or documents",
                complexity="medium",
                max_tokens=200,
                quality_threshold=0.8,
                latency_requirement="medium",
                cost_priority="medium"
            ),
            TaskProfile(
                name="Complex Analysis",
                description="Detailed reasoning and analysis tasks",
                complexity="complex",
                max_tokens=500,
                quality_threshold=0.9,
                latency_requirement="slow",
                cost_priority="low"  # Quality over cost
            ),
            TaskProfile(
                name="Code Generation",
                description="Generate and explain code",
                complexity="complex",
                max_tokens=300,
                quality_threshold=0.85,
                latency_requirement="medium",
                cost_priority="medium"
            )
        ]

        # Model tiers by cost and capability
        model_tiers = {
            "economy": [
                "meta-llama/llama-3.2-1b-instruct",
                "meta-llama/llama-3.2-3b-instruct",
                "google/gemma-2-9b-it"
            ],
            "balanced": [
                "openai/gpt-3.5-turbo",
                "meta-llama/llama-3.1-8b-instruct",
                "anthropic/claude-3-haiku",
                "mistralai/mistral-small"
            ],
            "premium": [
                "openai/gpt-4o",
                "anthropic/claude-3-5-sonnet",
                "google/gemini-1.5-pro",
                "mistralai/mistral-large"
            ],
            "flagship": [
                "openai/gpt-4o",
                "anthropic/claude-3-opus",
                "meta-llama/llama-3.1-405b-instruct"
            ]
        }

        print(f"\n🎯 Testing Cost Optimization for {len(task_profiles)} Task Types")
        print("=" * 55)

        total_cost = 0.0
        optimization_results = []

        for profile in task_profiles:
            print(f"\n📋 Task: {profile.name}")
            print(f"   Description: {profile.description}")
            print(f"   Complexity: {profile.complexity}, Cost priority: {profile.cost_priority}")

            # Select optimal tier based on task profile
            if profile.cost_priority == "high" and profile.complexity == "simple":
                selected_tier = "economy"
            elif profile.cost_priority == "low" and profile.complexity == "complex":
                selected_tier = "flagship"
            elif profile.complexity == "complex":
                selected_tier = "premium"
            else:
                selected_tier = "balanced"

            print(f"   🎯 Selected tier: {selected_tier}")

            # Find the most cost-effective model in the tier
            tier_models = model_tiers[selected_tier]
            best_model = None
            best_cost_per_token = float('inf')

            print(f"   🔍 Evaluating {len(tier_models)} models in tier...")

            for model in tier_models:
                # Estimate cost for this task
                estimated_cost = calculate_openrouter_cost(
                    model,
                    input_tokens=50,  # Estimated input
                    output_tokens=profile.max_tokens
                )
                cost_per_token = estimated_cost / profile.max_tokens

                if cost_per_token < best_cost_per_token:
                    best_cost_per_token = cost_per_token
                    best_model = model

            if best_model:
                print(f"   ✅ Optimal model: {best_model}")
                print(f"      Est. cost per token: ${best_cost_per_token:.8f}")

                # Test the selected model
                test_prompts = {
                    "Quick FAQ Response": "What is machine learning?",
                    "Content Summarization": "Summarize the key benefits of renewable energy sources including solar, wind, and hydroelectric power.",
                    "Complex Analysis": "Analyze the potential economic and social impacts of widespread AI adoption in the healthcare industry.",
                    "Code Generation": "Create a Python function that implements a binary search algorithm with error handling."
                }

                prompt = test_prompts.get(profile.name, "Hello, how can you help me?")

                try:
                    start_time = time.time()
                    response = client.chat_completions_create(
                        model=best_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=profile.max_tokens,
                        # Governance attributes for cost tracking
                        team="cost-optimization-team",
                        project="intelligent-routing",
                        task_profile=profile.name,
                        optimization_tier=selected_tier,
                        cost_priority=profile.cost_priority
                    )
                    response_time = time.time() - start_time

                    usage = response.usage
                    if usage:
                        actual_cost = calculate_openrouter_cost(
                            best_model,
                            input_tokens=usage.prompt_tokens,
                            output_tokens=usage.completion_tokens
                        )

                        print(f"      💰 Actual cost: ${actual_cost:.6f}")
                        print(f"      ⏱️  Response time: {response_time:.2f}s")
                        print(f"      📊 Tokens: {usage.total_tokens} total")
                        print(f"      📝 Response: {response.choices[0].message.content[:80]}...")

                        total_cost += actual_cost
                        optimization_results.append({
                            "task": profile.name,
                            "model": best_model,
                            "tier": selected_tier,
                            "cost": actual_cost,
                            "tokens": usage.total_tokens,
                            "cost_per_token": actual_cost / usage.total_tokens if usage.total_tokens > 0 else 0,
                            "response_time": response_time
                        })
                    else:
                        print("      ⚠️  No usage data available")

                except Exception as e:
                    print(f"      ❌ Error: {str(e)}")
            else:
                print("   ❌ No suitable model found in tier")

        # Cost optimization analysis
        print("\n" + "=" * 50)
        print("📊 Cost Optimization Analysis")
        print("=" * 50)

        if optimization_results:
            print(f"💰 Total Cost: ${total_cost:.6f}")
            print(f"📈 Tasks Completed: {len(optimization_results)}")

            # Cost efficiency analysis
            print("\n🎯 Cost Efficiency by Task:")
            for result in optimization_results:
                print(f"   • {result['task']}")
                print(f"     Model: {result['model']} ({result['tier']} tier)")
                print(f"     Cost: ${result['cost']:.6f} (${result['cost_per_token']:.8f}/token)")
                print(f"     Speed: {result['response_time']:.2f}s")

            # Tier effectiveness
            tier_costs = {}
            for result in optimization_results:
                tier = result['tier']
                if tier not in tier_costs:
                    tier_costs[tier] = []
                tier_costs[tier].append(result['cost'])

            print("\n📊 Cost by Tier:")
            for tier, costs in tier_costs.items():
                avg_cost = sum(costs) / len(costs)
                print(f"   • {tier.title()}: ${avg_cost:.6f} average")

            # Savings calculation (vs. using premium models for everything)
            premium_cost_estimate = len(optimization_results) * 0.002  # Rough premium cost estimate
            savings = premium_cost_estimate - total_cost
            savings_percentage = (savings / premium_cost_estimate) * 100 if premium_cost_estimate > 0 else 0

            print("\n💡 Optimization Impact:")
            print(f"   Estimated savings vs. premium-only: ${savings:.6f} ({savings_percentage:.1f}%)")
            print("   Cost optimization enabled by intelligent model selection")

        # Demonstrate budget-constrained operations
        print("\n💳 Budget-Constrained Operations Demo")
        print("=" * 40)

        budget_scenarios = [
            {"name": "Micro Budget", "budget": 0.001, "max_requests": 10},
            {"name": "Small Budget", "budget": 0.01, "max_requests": 20},
            {"name": "Medium Budget", "budget": 0.05, "max_requests": 50}
        ]

        for scenario in budget_scenarios:
            print(f"\n🎯 Scenario: {scenario['name']}")
            print(f"   Budget: ${scenario['budget']:.4f}")
            print(f"   Max requests: {scenario['max_requests']}")

            # Select most cost-effective models that fit budget
            remaining_budget = scenario['budget']
            requests_made = 0

            # Use economy tier for budget scenarios
            budget_model = "meta-llama/llama-3.2-3b-instruct"  # Very cost-effective

            estimated_cost_per_request = calculate_openrouter_cost(
                budget_model,
                input_tokens=20,
                output_tokens=40
            )

            max_affordable_requests = int(remaining_budget / estimated_cost_per_request)
            actual_requests = min(max_affordable_requests, 3, scenario['max_requests'])  # Limit demo to 3

            print(f"   💰 Est. cost per request: ${estimated_cost_per_request:.6f}")
            print(f"   📊 Affordable requests: {max_affordable_requests}")
            print(f"   🎯 Demo requests: {actual_requests}")

            for i in range(actual_requests):
                try:
                    response = client.chat_completions_create(
                        model=budget_model,
                        messages=[{"role": "user", "content": f"Quick question {i+1}: What is AI?"}],
                        max_tokens=40,
                        team="budget-optimization",
                        project="cost-constrained-ops",
                        budget_scenario=scenario["name"],
                        request_number=i+1
                    )

                    usage = response.usage
                    if usage:
                        actual_cost = calculate_openrouter_cost(
                            budget_model,
                            input_tokens=usage.prompt_tokens,
                            output_tokens=usage.completion_tokens
                        )
                        remaining_budget -= actual_cost
                        requests_made += 1

                        print(f"      Request {i+1}: ${actual_cost:.6f}, Budget left: ${remaining_budget:.6f}")

                except Exception as e:
                    print(f"      Request {i+1} failed: {str(e)}")

            print(f"   ✅ Completed {requests_made} requests within budget")

        print("\n" + "=" * 50)
        print("🎯 Cost Optimization Best Practices")
        print("=" * 50)

        print("🏗️ Model Selection Strategy:")
        print("   • Simple tasks → Economy tier (Llama 3.2 1B/3B, Gemma 2)")
        print("   • Balanced tasks → Mid-tier (GPT-3.5, Claude Haiku, Mistral Small)")
        print("   • Complex tasks → Premium tier (GPT-4o, Claude Sonnet, Gemini Pro)")
        print("   • Critical tasks → Flagship tier (Claude Opus, Llama 405B)")

        print("\n💰 Cost Control Techniques:")
        print("   • Dynamic model selection based on task complexity")
        print("   • Budget-constrained request batching")
        print("   • Real-time cost monitoring and alerting")
        print("   • Provider routing for cost optimization")

        print("\n📊 GenOps Cost Intelligence:")
        print("   • Automatic cost tracking across 400+ models")
        print("   • Real-time budget monitoring and enforcement")
        print("   • Cost-per-token analysis and optimization recommendations")
        print("   • Multi-dimensional cost attribution and reporting")

        print("\n✨ Next Steps:")
        print("   • Implement dynamic model selection in production")
        print("   • Set up cost-based alerting and budget controls")
        print("   • Try production_patterns.py for deployment guidance")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Install: pip install genops-ai openai")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    cost_optimization_demo()
