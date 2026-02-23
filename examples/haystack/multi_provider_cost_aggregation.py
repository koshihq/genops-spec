#!/usr/bin/env python3
"""
Multi-Provider Cost Aggregation with GenOps and Haystack

Demonstrates advanced cost tracking and optimization across multiple AI providers
in Haystack pipelines, including cross-provider cost analysis, optimization
recommendations, and intelligent provider selection.

Usage:
    python multi_provider_cost_aggregation.py

Features:
    - Multi-provider pipeline setup (OpenAI, Anthropic, Cohere, HuggingFace)
    - Cross-provider cost tracking and aggregation
    - Intelligent provider selection based on cost and performance
    - Real-time cost optimization recommendations
    - Provider failover and load balancing simulation
    - Comprehensive cost analysis and reporting
"""

import logging
import random
import sys
from dataclasses import dataclass
from typing import Optional

# Core Haystack imports
try:
    from haystack import Document, Pipeline  # noqa: F401
    from haystack.components.builders import PromptBuilder
    from haystack.components.embedders import OpenAITextEmbedder  # noqa: F401
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers import InMemoryBM25Retriever  # noqa: F401
    from haystack.document_stores.in_memory import InMemoryDocumentStore  # noqa: F401
except ImportError as e:
    print(f"❌ Haystack not installed: {e}")
    print("Please install Haystack: pip install haystack-ai")
    sys.exit(1)

# GenOps imports
try:
    from genops.providers.haystack import (
        GenOpsHaystackAdapter,
        analyze_pipeline_costs,
        print_validation_result,
        validate_haystack_setup,
    )
except ImportError as e:
    print(f"❌ GenOps not installed: {e}")
    print("Please install GenOps: pip install genops-ai[haystack]")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for an AI provider."""

    name: str
    model: str
    cost_per_1k_tokens: float
    avg_response_time: float
    reliability_score: float
    max_tokens: int = 150


class MultiProviderManager:
    """Manage multiple AI providers with cost optimization."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.providers = self.initialize_providers()
        self.usage_history = []
        self.performance_metrics = {}

    def initialize_providers(self) -> dict[str, ProviderConfig]:
        """Initialize available AI providers with their configurations."""
        return {
            "openai_gpt35": ProviderConfig(
                name="OpenAI GPT-3.5",
                model="gpt-3.5-turbo",
                cost_per_1k_tokens=0.002,
                avg_response_time=1.2,
                reliability_score=0.98,
                max_tokens=150,
            ),
            "openai_gpt4": ProviderConfig(
                name="OpenAI GPT-4",
                model="gpt-4",
                cost_per_1k_tokens=0.06,
                avg_response_time=2.8,
                reliability_score=0.99,
                max_tokens=150,
            ),
            # Simulated Anthropic (would require actual Anthropic components)
            "anthropic_claude": ProviderConfig(
                name="Anthropic Claude",
                model="claude-3-haiku",
                cost_per_1k_tokens=0.00025,
                avg_response_time=1.8,
                reliability_score=0.97,
                max_tokens=150,
            ),
            # Simulated Cohere (would require actual Cohere components)
            "cohere_command": ProviderConfig(
                name="Cohere Command",
                model="command",
                cost_per_1k_tokens=0.0015,
                avg_response_time=1.5,
                reliability_score=0.96,
                max_tokens=150,
            ),
        }

    def select_optimal_provider(
        self,
        task_type: str,
        budget_constraint: Optional[float] = None,
        performance_priority: str = "balanced",
    ) -> str:
        """Select optimal provider based on task requirements and constraints."""

        providers = list(self.providers.keys())

        if budget_constraint:
            # Filter providers within budget
            providers = [
                p
                for p in providers
                if self.providers[p].cost_per_1k_tokens <= budget_constraint
            ]

        if not providers:
            providers = ["openai_gpt35"]  # Fallback to cheapest option

        # Score providers based on priority
        provider_scores = {}

        for provider_id in providers:
            config = self.providers[provider_id]

            if performance_priority == "cost":
                # Prioritize cost (lower is better)
                score = 1.0 / (config.cost_per_1k_tokens + 0.0001)
            elif performance_priority == "speed":
                # Prioritize speed (lower response time is better)
                score = 1.0 / (config.avg_response_time + 0.1)
            elif performance_priority == "reliability":
                # Prioritize reliability
                score = config.reliability_score
            else:  # balanced
                # Balanced scoring
                cost_score = 1.0 / (config.cost_per_1k_tokens + 0.0001)
                speed_score = 1.0 / (config.avg_response_time + 0.1)
                reliability_score = config.reliability_score
                score = cost_score * 0.4 + speed_score * 0.3 + reliability_score * 0.3

            provider_scores[provider_id] = score

        # Select provider with highest score
        best_provider = max(provider_scores, key=provider_scores.get)

        logger.info(
            f"Selected provider: {self.providers[best_provider].name} "
            f"(priority: {performance_priority}, score: {provider_scores[best_provider]:.3f})"
        )

        return best_provider

    def create_provider_pipeline(
        self, provider_id: str, task_type: str = "general"
    ) -> Pipeline:
        """Create pipeline for specific provider."""
        config = self.providers[provider_id]

        pipeline = Pipeline()

        # Add prompt builder
        pipeline.add_component(
            "prompt_builder",
            PromptBuilder(
                template="""
            Task Type: {{task_type}}

            {{prompt}}

            Please provide a clear and concise response:
            """
            ),
        )

        # Add provider-specific generator
        if "openai" in provider_id:
            generator = OpenAIGenerator(
                model=config.model,
                generation_kwargs={
                    "max_tokens": config.max_tokens,
                    "temperature": 0.7 if task_type == "creative" else 0.3,
                },
            )
        else:
            # For demo purposes, use OpenAI as fallback for other providers
            # In real implementation, would use actual provider components
            generator = OpenAIGenerator(
                model="gpt-3.5-turbo",  # Fallback
                generation_kwargs={
                    "max_tokens": config.max_tokens,
                    "temperature": 0.7 if task_type == "creative" else 0.3,
                },
            )

        pipeline.add_component("llm", generator)
        pipeline.connect("prompt_builder", "llm")

        return pipeline

    def simulate_provider_costs(
        self, provider_id: str, prompt: str
    ) -> tuple[float, float]:
        """Simulate provider costs and response time."""
        config = self.providers[provider_id]

        # Estimate tokens (rough approximation)
        estimated_tokens = len(prompt.split()) * 1.3 + config.max_tokens

        # Calculate cost
        cost = (estimated_tokens / 1000) * config.cost_per_1k_tokens

        # Simulate response time with some randomness
        response_time = config.avg_response_time * random.uniform(0.8, 1.2)

        return cost, response_time


def create_multi_provider_comparison_pipeline() -> dict[str, Pipeline]:
    """Create pipelines for different providers to compare performance."""
    print("🏭 Creating Multi-Provider Comparison Pipelines")

    pipelines = {}

    # OpenAI GPT-3.5 Pipeline (cost-effective)
    gpt35_pipeline = Pipeline()
    gpt35_pipeline.add_component(
        "prompt_builder",
        PromptBuilder(template="Answer the following question concisely: {{question}}"),
    )
    gpt35_pipeline.add_component(
        "llm",
        OpenAIGenerator(
            model="gpt-3.5-turbo",
            generation_kwargs={"max_tokens": 150, "temperature": 0.5},
        ),
    )
    gpt35_pipeline.connect("prompt_builder", "llm")
    pipelines["openai_gpt35"] = gpt35_pipeline

    # OpenAI GPT-4 Pipeline (high-quality)
    gpt4_pipeline = Pipeline()
    gpt4_pipeline.add_component(
        "prompt_builder",
        PromptBuilder(template="Provide a detailed and accurate answer: {{question}}"),
    )
    gpt4_pipeline.add_component(
        "llm",
        OpenAIGenerator(
            model="gpt-4", generation_kwargs={"max_tokens": 200, "temperature": 0.3}
        ),
    )
    gpt4_pipeline.connect("prompt_builder", "llm")
    pipelines["openai_gpt4"] = gpt4_pipeline

    print(f"✅ Created {len(pipelines)} provider pipelines")
    return pipelines


def demo_multi_provider_cost_tracking():
    """Demonstrate multi-provider cost tracking and analysis."""
    print("\n" + "=" * 70)
    print("💰 Multi-Provider Cost Tracking")
    print("=" * 70)

    # Create main adapter for cost aggregation
    adapter = GenOpsHaystackAdapter(
        team="cost-optimization",
        project="multi-provider-analysis",
        daily_budget_limit=25.0,
        governance_policy="advisory",
    )

    print("✅ Multi-provider cost tracking adapter created")

    # Initialize provider manager
    provider_manager = MultiProviderManager(adapter)
    comparison_pipelines = create_multi_provider_comparison_pipeline()

    # Test questions for provider comparison
    test_questions = [
        {
            "question": "What are the main benefits of using AI in business?",
            "category": "business",
            "priority": "cost",
        },
        {
            "question": "Explain quantum computing in simple terms",
            "category": "technical",
            "priority": "balanced",
        },
        {
            "question": "Write a creative story about AI and humans working together",
            "category": "creative",
            "priority": "quality",
        },
        {
            "question": "What are the latest trends in machine learning?",
            "category": "research",
            "priority": "balanced",
        },
        {
            "question": "How can companies reduce their AI costs?",
            "category": "optimization",
            "priority": "cost",
        },
    ]

    # Track costs across multiple providers
    provider_results = []

    with adapter.track_session(
        "multi-provider-comparison", use_case="cost-analysis"
    ) as session:
        print(f"\n📋 Started multi-provider session: {session.session_name}")

        for i, test_case in enumerate(test_questions, 1):
            question = test_case["question"]
            category = test_case["category"]
            priority = test_case["priority"]

            print(
                f"\n🔍 Question {i}/{len(test_questions)}: {category} ({priority} priority)"
            )
            print(f"   Question: {question}")

            # Test with multiple providers
            question_results = {
                "question": question,
                "category": category,
                "providers": {},
            }

            for provider_id in ["openai_gpt35", "openai_gpt4"]:  # Available providers
                provider_name = provider_manager.providers[provider_id].name

                print(f"\n   🧠 Testing with {provider_name}...")

                with adapter.track_pipeline(
                    f"provider-{provider_id}",
                    provider=provider_name,
                    question_category=category,
                    optimization_priority=priority,
                ) as context:
                    # Execute with specific provider
                    if provider_id in comparison_pipelines:
                        result = comparison_pipelines[provider_id].run(
                            {"prompt_builder": {"question": question}}
                        )

                        answer = result["llm"]["replies"][0]

                        # Simulate realistic costs for different providers
                        actual_cost, response_time = (
                            provider_manager.simulate_provider_costs(
                                provider_id, question
                            )
                        )

                        context.add_custom_metric("provider_id", provider_id)
                        context.add_custom_metric("simulated_cost", actual_cost)
                        context.add_custom_metric(
                            "simulated_response_time", response_time
                        )

                        print(f"      📝 Answer: {answer[:100]}...")
                        print(f"      💰 Estimated cost: ${actual_cost:.6f}")
                        print(f"      ⏱️ Response time: {response_time:.2f}s")

                        question_results["providers"][provider_id] = {
                            "provider_name": provider_name,
                            "answer": answer,
                            "cost": actual_cost,
                            "response_time": response_time,
                            "metrics": context.get_metrics(),
                        }

                session.add_pipeline_result(context.get_metrics())

            provider_results.append(question_results)

        print("\n📊 Multi-Provider Session Summary:")
        print(f"   Total provider tests: {session.total_pipelines}")
        print(f"   Total cost: ${session.total_cost:.6f}")
        print(
            f"   Average cost per test: ${session.total_cost / session.total_pipelines:.6f}"
        )

    return adapter, provider_manager, provider_results


def analyze_cross_provider_performance(provider_results):
    """Analyze performance across different providers."""
    print("\n" + "=" * 70)
    print("📊 Cross-Provider Performance Analysis")
    print("=" * 70)

    # Aggregate performance metrics by provider
    provider_stats = {}

    for question_result in provider_results:
        for provider_id, result in question_result["providers"].items():
            if provider_id not in provider_stats:
                provider_stats[provider_id] = {
                    "costs": [],
                    "response_times": [],
                    "questions_processed": 0,
                }

            provider_stats[provider_id]["costs"].append(result["cost"])
            provider_stats[provider_id]["response_times"].append(
                result["response_time"]
            )
            provider_stats[provider_id]["questions_processed"] += 1

    # Calculate and display provider comparison
    print("🏆 Provider Performance Comparison:")

    for provider_id, stats in provider_stats.items():
        avg_cost = sum(stats["costs"]) / len(stats["costs"])
        avg_response_time = sum(stats["response_times"]) / len(stats["response_times"])
        total_cost = sum(stats["costs"])

        print(f"\n   {provider_id.upper()}:")
        print(f"     Questions processed: {stats['questions_processed']}")
        print(f"     Average cost: ${avg_cost:.6f}")
        print(f"     Total cost: ${total_cost:.6f}")
        print(f"     Average response time: {avg_response_time:.2f}s")
        print(f"     Cost efficiency: ${avg_cost / avg_response_time:.6f} per second")

    # Identify best provider by category
    category_analysis = {}
    for question_result in provider_results:
        category = question_result["category"]

        if category not in category_analysis:
            category_analysis[category] = {}

        # Find cheapest and fastest provider for this question
        cheapest_provider = min(
            question_result["providers"].items(), key=lambda x: x[1]["cost"]
        )

        fastest_provider = min(
            question_result["providers"].items(), key=lambda x: x[1]["response_time"]
        )

        category_analysis[category][question_result["question"]] = {
            "cheapest": cheapest_provider,
            "fastest": fastest_provider,
        }

    print("\n🎯 Optimization Recommendations by Category:")
    for category, questions in category_analysis.items():
        print(f"\n   {category.upper()} Questions:")

        # Count provider preferences
        cheapest_counts = {}
        fastest_counts = {}

        for question_data in questions.values():
            cheapest_id = question_data["cheapest"][0]
            fastest_id = question_data["fastest"][0]

            cheapest_counts[cheapest_id] = cheapest_counts.get(cheapest_id, 0) + 1
            fastest_counts[fastest_id] = fastest_counts.get(fastest_id, 0) + 1

        most_cost_effective = (
            max(cheapest_counts, key=cheapest_counts.get) if cheapest_counts else "None"
        )
        most_performant = (
            max(fastest_counts, key=fastest_counts.get) if fastest_counts else "None"
        )

        print(f"     Most cost-effective: {most_cost_effective}")
        print(f"     Most performant: {most_performant}")

    return provider_stats, category_analysis


def demo_intelligent_provider_selection():
    """Demonstrate intelligent provider selection based on optimization goals."""
    print("\n" + "=" * 70)
    print("🧠 Intelligent Provider Selection")
    print("=" * 70)

    adapter = GenOpsHaystackAdapter(
        team="intelligent-selection",
        project="provider-optimization",
        daily_budget_limit=15.0,
    )

    provider_manager = MultiProviderManager(adapter)

    # Scenarios with different optimization priorities
    optimization_scenarios = [
        {
            "name": "Budget-Conscious Batch Processing",
            "priority": "cost",
            "budget_constraint": 0.003,  # Max cost per 1K tokens
            "tasks": [
                "Summarize this document in 2 sentences",
                "Extract key points from the following text",
                "Classify this content as positive, negative, or neutral",
            ],
        },
        {
            "name": "Real-Time Customer Support",
            "priority": "speed",
            "budget_constraint": None,
            "tasks": [
                "Provide immediate help with this customer issue",
                "Generate a quick response to this inquiry",
                "Resolve this support ticket efficiently",
            ],
        },
        {
            "name": "High-Stakes Content Generation",
            "priority": "reliability",
            "budget_constraint": 0.08,  # Higher budget for quality
            "tasks": [
                "Write a comprehensive analysis of market trends",
                "Create detailed technical documentation",
                "Generate executive summary for board presentation",
            ],
        },
    ]

    scenario_results = []

    with adapter.track_session(
        "intelligent-selection", use_case="optimization-scenarios"
    ) as session:
        for scenario in optimization_scenarios:
            print(f"\n🎯 Scenario: {scenario['name']}")
            print(f"   Priority: {scenario['priority']}")
            print(f"   Budget constraint: {scenario['budget_constraint'] or 'None'}")

            scenario_cost = 0
            scenario_time = 0
            providers_used = []

            for task_num, task in enumerate(scenario["tasks"], 1):
                print(f"\n   📋 Task {task_num}: {task}")

                # Select optimal provider for this scenario
                selected_provider = provider_manager.select_optimal_provider(
                    task_type="general",
                    budget_constraint=scenario["budget_constraint"],
                    performance_priority=scenario["priority"],
                )

                providers_used.append(selected_provider)
                provider_name = provider_manager.providers[selected_provider].name

                print(f"      🎯 Selected: {provider_name}")

                with adapter.track_pipeline(
                    f"scenario-{scenario['name'].lower().replace(' ', '-')}",
                    scenario_name=scenario["name"],
                    optimization_priority=scenario["priority"],
                    selected_provider=provider_name,
                ) as context:
                    # Create and execute pipeline
                    pipeline = provider_manager.create_provider_pipeline(
                        selected_provider, "general"
                    )

                    result = pipeline.run(
                        {"prompt_builder": {"task_type": "general", "prompt": task}}
                    )

                    # Get simulated costs and timing
                    estimated_cost, response_time = (
                        provider_manager.simulate_provider_costs(
                            selected_provider, task
                        )
                    )

                    scenario_cost += estimated_cost
                    scenario_time += response_time

                    context.add_custom_metric("scenario_name", scenario["name"])
                    context.add_custom_metric(
                        "optimization_priority", scenario["priority"]
                    )
                    context.add_custom_metric("estimated_cost", estimated_cost)
                    context.add_custom_metric("response_time", response_time)

                    print(f"      💰 Cost: ${estimated_cost:.6f}")
                    print(f"      ⏱️ Time: {response_time:.2f}s")
                    print(f"      📝 Result: {result['llm']['replies'][0][:80]}...")

                session.add_pipeline_result(context.get_metrics())

            # Scenario summary
            unique_providers = list(set(providers_used))
            avg_cost_per_task = scenario_cost / len(scenario["tasks"])
            avg_time_per_task = scenario_time / len(scenario["tasks"])

            print("\n   📊 Scenario Summary:")
            print(f"      Total cost: ${scenario_cost:.6f}")
            print(f"      Total time: {scenario_time:.2f}s")
            print(f"      Average cost per task: ${avg_cost_per_task:.6f}")
            print(f"      Average time per task: {avg_time_per_task:.2f}s")
            print(f"      Providers used: {unique_providers}")

            scenario_results.append(
                {
                    "name": scenario["name"],
                    "priority": scenario["priority"],
                    "total_cost": scenario_cost,
                    "total_time": scenario_time,
                    "providers_used": providers_used,
                    "unique_providers": unique_providers,
                    "tasks_completed": len(scenario["tasks"]),
                }
            )

    # Compare scenarios
    print("\n🏆 Scenario Optimization Results:")
    for result in scenario_results:
        efficiency_score = result["tasks_completed"] / (
            result["total_cost"] * result["total_time"] + 0.01
        )

        print(f"\n   {result['name']}:")
        print(f"     Optimization priority: {result['priority']}")
        print(f"     Total cost: ${result['total_cost']:.6f}")
        print(f"     Total time: {result['total_time']:.2f}s")
        print(f"     Efficiency score: {efficiency_score:.2f}")
        print(
            f"     Provider diversity: {len(result['unique_providers'])}/{len(result['providers_used'])} unique"
        )

    return scenario_results


def demo_cost_optimization_recommendations(adapter):
    """Generate and demonstrate cost optimization recommendations."""
    print("\n" + "=" * 70)
    print("💡 Cost Optimization Recommendations")
    print("=" * 70)

    # Get comprehensive cost analysis
    cost_analysis = analyze_pipeline_costs(adapter, time_period_hours=1)

    if "error" in cost_analysis:
        print(f"❌ Could not generate cost analysis: {cost_analysis['error']}")
        return

    print("📈 Current Cost Analysis:")
    print(f"   Total cost (last hour): ${cost_analysis['total_cost']:.6f}")

    if cost_analysis["cost_by_provider"]:
        print("   Cost breakdown by provider:")
        for provider, cost in cost_analysis["cost_by_provider"].items():
            percentage = (
                (cost / cost_analysis["total_cost"]) * 100
                if cost_analysis["total_cost"] > 0
                else 0
            )
            print(f"     • {provider}: ${cost:.6f} ({percentage:.1f}%)")

    if cost_analysis["most_expensive_component"]:
        print(
            f"   Most expensive component: {cost_analysis['most_expensive_component']}"
        )

    # Generate optimization recommendations
    print("\n💡 Optimization Recommendations:")

    if cost_analysis.get("recommendations"):
        for i, rec in enumerate(cost_analysis["recommendations"], 1):
            print(f"\n   {i}. {rec['reasoning']}")
            print(f"      Current setup: {rec['current_provider']}")
            print(f"      Recommended: {rec['recommended_provider']}")
            print(
                f"      Potential savings: ${rec['potential_savings']:.6f} per operation"
            )

            # Calculate potential monthly savings
            monthly_savings = (
                rec["potential_savings"] * 1000
            )  # Assuming 1000 operations/month
            print(f"      Estimated monthly savings: ${monthly_savings:.2f}")
    else:
        print("   ✅ Your current setup is well-optimized!")
        print("   Consider these general best practices:")
        print("     • Use GPT-3.5-turbo for simple tasks")
        print("     • Reserve GPT-4 for complex reasoning tasks")
        print("     • Implement caching for repeated queries")
        print("     • Set appropriate max_tokens limits")

    # Additional optimization suggestions
    print("\n🚀 Advanced Optimization Strategies:")
    print("   1. Implement request caching for repeated queries")
    print("   2. Use batch processing to reduce per-request overhead")
    print("   3. Implement smart provider fallbacks for reliability")
    print("   4. Monitor and adjust token limits based on actual usage")
    print("   5. Consider fine-tuned models for specialized tasks")


def main():
    """Run the comprehensive multi-provider cost aggregation demonstration."""
    print("💰 Multi-Provider Cost Aggregation with Haystack + GenOps")
    print("=" * 70)

    # Validate environment setup
    print("🔍 Validating setup...")
    result = validate_haystack_setup()

    if not result.is_valid:
        print("❌ Setup validation failed!")
        print_validation_result(result)
        return 1
    else:
        print("✅ Environment validated and ready")

    try:
        # Multi-provider cost tracking demonstration
        adapter, provider_manager, provider_results = (
            demo_multi_provider_cost_tracking()
        )

        # Analyze cross-provider performance
        provider_stats, category_analysis = analyze_cross_provider_performance(
            provider_results
        )

        # Intelligent provider selection
        demo_intelligent_provider_selection()

        # Cost optimization recommendations
        demo_cost_optimization_recommendations(adapter)

        print("\n🎉 Multi-Provider Cost Aggregation demonstration completed!")
        print("\n🚀 Next Steps:")
        print("   • Try enterprise_governance_patterns.py for advanced governance")
        print("   • Run production_deployment_patterns.py for scaling strategies")
        print("   • Explore performance_optimization.py for speed improvements")
        print("   • Implement intelligent provider selection in your pipelines! 💰")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("Try running the setup validation to check your configuration")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
