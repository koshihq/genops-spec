#!/usr/bin/env python3
"""
GenOps Gemini Cost Optimization Example

This example demonstrates advanced cost intelligence and optimization
strategies for Google Gemini usage, including:
- Multi-model cost comparison and intelligent selection
- Budget-aware operation strategies with real-time alerts
- Cost optimization recommendations
- Performance vs cost trade-off analysis

Example usage:
    python cost_optimization.py
"""

import os
import time


def main():
    print("💡 GenOps Gemini Cost Optimization Example")
    print("=" * 44)
    print("Demonstrating intelligent cost optimization and budget management.\n")

    try:
        from genops.providers.gemini import GenOpsGeminiAdapter
        from genops.providers.gemini_cost_aggregator import create_gemini_cost_context
        from genops.providers.gemini_pricing import (
            compare_gemini_models,
            estimate_monthly_cost,
            get_cost_optimization_recommendations,
        )

        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable required")
            print("💡 Get your API key at: https://ai.google.dev/")
            return False

        # Initialize adapter
        adapter = GenOpsGeminiAdapter(api_key=api_key)
        print("✅ GenOps Gemini adapter initialized\n")

        # Example 1: Cost-Aware Model Selection
        print("🎯 Example 1: Intelligent Model Selection")
        print("-" * 38)

        # Test prompt for comparison
        analysis_prompt = """
        Analyze the impact of artificial intelligence on healthcare, focusing on:
        1. Diagnostic accuracy improvements
        2. Treatment personalization
        3. Cost reduction opportunities
        4. Patient outcome enhancements
        Provide specific examples and data where possible.
        """

        # Compare models before making decision
        estimated_input_tokens = len(analysis_prompt.split()) * 1.3
        estimated_output_tokens = 400  # Expected detailed analysis

        print("🔍 Comparing models for optimal cost/performance...")
        models_to_compare = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
        ]

        comparison = compare_gemini_models(
            models=models_to_compare,
            input_tokens=int(estimated_input_tokens),
            output_tokens=estimated_output_tokens,
            sort_by="total_cost",
        )

        print("Model Cost Comparison:")
        for i, model_data in enumerate(comparison):
            print(f"  {i + 1}. {model_data['display_name']}")
            print(f"     💰 Cost: ${model_data['total_cost']:.6f}")
            print(f"     📊 Per 1K tokens: ${model_data['cost_per_1k_tokens']:.6f}")
            print(f"     🎯 Best for: {model_data['description'][:40]}...")
        print()

        # Choose Flash for good balance of cost/performance
        print("📊 Selecting Gemini 2.5 Flash for optimal cost/performance balance...")

        result1 = adapter.text_generation(
            prompt=analysis_prompt,
            model="gemini-2.5-flash",
            team="healthcare-ai",
            project="ai-impact-analysis",
            customer_id="hospital-network",
        )

        print("✅ Analysis completed:")
        print(f"   💰 Actual cost: ${result1.cost_usd:.6f}")
        print(f"   ⚡ Latency: {result1.latency_ms:.0f}ms")
        print(f"   🔢 Tokens: {result1.input_tokens} → {result1.output_tokens}")
        print()

        # Example 2: Budget-Constrained Operations
        print("💰 Example 2: Budget-Constrained AI Operations")
        print("-" * 42)

        # Use cost context with budget limit
        with create_gemini_cost_context(
            context_id="budget_analysis_session",
            budget_limit=0.05,  # $0.05 budget limit
            enable_optimization=True,
            enable_alerts=True,
            team="marketing-analytics",
            project="campaign-optimization",
        ) as context:
            print("💳 Set budget limit: $0.05 for this analysis session")
            print()

            # Multiple operations within budget
            operations = [
                (
                    "Social media sentiment analysis",
                    "Analyze social media sentiment for our latest product launch.",
                ),
                (
                    "Competitor analysis",
                    "Compare our marketing strategy with top 3 competitors.",
                ),
                (
                    "Customer feedback summary",
                    "Summarize key themes from customer feedback data.",
                ),
                (
                    "Campaign optimization",
                    "Suggest improvements for our current ad campaign.",
                ),
            ]

            for i, (operation_name, prompt) in enumerate(operations, 1):
                print(f"🔄 Operation {i}: {operation_name}")

                # Check current budget utilization before operation
                current_summary = context.get_current_summary()
                remaining_budget = 0.05 - current_summary.total_cost

                if remaining_budget <= 0.001:  # Less than $0.001 remaining
                    print("⚠️  Budget exhausted! Skipping remaining operations.")
                    break

                # Perform operation
                start_time = time.time()
                result = adapter.text_generation(
                    prompt=prompt,
                    model="gemini-2.5-flash-lite",  # Most cost-efficient
                    max_tokens=150,  # Limit output to control costs
                )
                time.time() - start_time

                # Add to cost context
                context.add_operation(
                    operation_id=f"marketing_op_{i}",
                    model_id="gemini-2.5-flash-lite",
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    operation_type="marketing_analysis",
                )

                print(
                    f"   💰 Cost: ${result.cost_usd:.6f} | Remaining budget: ${remaining_budget - result.cost_usd:.6f}"
                )

            # Get final summary with optimization recommendations
            final_summary = context.get_current_summary()

        print("\n📈 Budget Analysis Summary:")
        print(f"   💰 Total spent: ${final_summary.total_cost:.6f} of $0.05 budget")
        print(
            f"   📊 Budget utilization: {(final_summary.total_cost / 0.05) * 100:.1f}%"
        )
        print(f"   🔢 Operations completed: {final_summary.total_operations}")
        print()

        # Show optimization recommendations
        if final_summary.optimization_recommendations:
            print("💡 Optimization Recommendations:")
            for i, rec in enumerate(final_summary.optimization_recommendations, 1):
                print(f"   {i}. {rec}")
            print()

        # Example 3: Task-Specific Cost Optimization
        print("🎯 Example 3: Task-Specific Cost Optimization")
        print("-" * 43)

        # Different types of tasks with different optimization strategies
        tasks = [
            (
                "code",
                "Write a Python function to process JSON data",
                "gemini-2.5-flash",
            ),
            (
                "creative",
                "Write a short marketing tagline for eco-friendly shoes",
                "gemini-2.5-flash-lite",
            ),
            (
                "analysis",
                "Analyze quarterly sales trends and predict next quarter",
                "gemini-2.5-pro",
            ),
        ]

        total_optimized_cost = 0.0

        for task_type, task_prompt, suggested_model in tasks:
            print(f"📋 Task: {task_type.title()} Generation")

            # Get optimization recommendations for this task
            recommendations = get_cost_optimization_recommendations(
                model_id="gemini-2.5-pro",  # Start with most expensive
                input_tokens=len(task_prompt.split()) * 1.3,
                output_tokens=200,
                use_case=task_type,
                budget_constraint=0.01,  # $0.01 per operation limit
            )

            if recommendations:
                best_model = recommendations[0]["model_id"]
                savings = recommendations[0]["savings"]
                print(f"   💡 Recommended model: {best_model}")
                print(f"   💰 Potential savings: ${savings:.6f}")
            else:
                best_model = suggested_model
                print(f"   💡 Using suggested model: {best_model}")

            # Execute with optimized model
            result = adapter.text_generation(
                prompt=task_prompt,
                model=best_model,
                team=f"{task_type}-team",
                project="cost-optimization-demo",
            )

            total_optimized_cost += result.cost_usd

            print(
                f"   ✅ Cost: ${result.cost_usd:.6f} | Latency: {result.latency_ms:.0f}ms"
            )
            print()

        print(f"📊 Task-Optimized Total Cost: ${total_optimized_cost:.6f}")
        print()

        # Example 4: Monthly Cost Estimation
        print("📅 Example 4: Monthly Cost Estimation")
        print("-" * 34)

        # Estimate costs based on usage patterns
        usage_scenarios = [
            (
                "Development Team",
                "gemini-2.5-flash",
                50,
                150,
                300,
            ),  # 50 ops/day, 150 in, 300 out tokens
            (
                "Content Team",
                "gemini-2.5-flash-lite",
                30,
                100,
                500,
            ),  # 30 ops/day, 100 in, 500 out tokens
            (
                "Research Team",
                "gemini-2.5-pro",
                10,
                300,
                800,
            ),  # 10 ops/day, 300 in, 800 out tokens
        ]

        total_monthly_estimate = 0.0

        print("Monthly Cost Projections:")
        for team, model, daily_ops, avg_input, avg_output in usage_scenarios:
            estimate = estimate_monthly_cost(
                model_id=model,
                daily_operations=daily_ops,
                avg_input_tokens=avg_input,
                avg_output_tokens=avg_output,
            )

            total_monthly_estimate += estimate["monthly_cost"]

            print(f"  {team}:")
            print(f"    Model: {model}")
            print(
                f"    Daily ops: {daily_ops} | Monthly cost: ${estimate['monthly_cost']:.2f}"
            )
            print(f"    Cost per operation: ${estimate['cost_per_operation']:.6f}")
            print()

        print(f"💰 Total Estimated Monthly Cost: ${total_monthly_estimate:.2f}")
        print()

        # Cost optimization summary
        print("🎉 Cost Optimization Summary")
        print("=" * 28)
        print("✅ Demonstrated intelligent model selection based on task complexity")
        print("✅ Implemented budget-constrained operations with real-time monitoring")
        print("✅ Provided task-specific optimization recommendations")
        print("✅ Generated monthly cost projections for planning")
        print()
        print("💡 Key Optimization Strategies:")
        print("   🎯 Use Flash-Lite for simple tasks (up to 90% cost savings)")
        print("   ⚖️  Use Flash for balanced performance/cost")
        print("   🚀 Reserve Pro for complex analysis requiring highest accuracy")
        print("   💳 Set budget limits to prevent cost overruns")
        print("   📊 Monitor usage patterns for continuous optimization")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Install required packages:")
        print("   pip install genops-ai[gemini] google-generativeai")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Verify GEMINI_API_KEY environment variable")
        print("   2. Check API quota and rate limits")
        print(
            '   3. Run validation: python -c "from genops.providers.gemini import validate_setup, print_validation_result; print_validation_result(validate_setup())"'
        )
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎯 Next Steps:")
        print("   → Explore cost aggregation: python cost_tracking.py")
        print("   → See production patterns: python production_patterns.py")
        print("   → Validate setup: python validation_example.py")

    exit(0 if success else 1)
