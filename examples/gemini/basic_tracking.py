#!/usr/bin/env python3
"""
GenOps Gemini Basic Tracking Example

This example demonstrates how to use GenOps with Google Gemini for:
- Team cost attribution and project tracking
- Multiple model comparison and cost optimization
- Governance attributes for enterprise compliance

Example usage:
    python basic_tracking.py
"""

import os
import time


def main():
    print("🎯 GenOps Gemini Basic Tracking Example")
    print("=" * 45)
    print("Demonstrating team attribution, cost tracking, and model comparison.\n")

    try:
        from genops.providers.gemini import GenOpsGeminiAdapter
        from genops.providers.gemini_pricing import compare_gemini_models

        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable required")
            print("💡 Get your API key at: https://ai.google.dev/")
            return False

        # Initialize GenOps adapter
        adapter = GenOpsGeminiAdapter(api_key=api_key)
        print("✅ GenOps Gemini adapter initialized\n")

        # Example 1: Team Attribution
        print("📊 Example 1: Team Cost Attribution")
        print("-" * 35)

        result1 = adapter.text_generation(
            prompt="Explain the concept of machine learning in simple terms.",
            model="gemini-2.5-flash",
            # Governance attributes - automatic cost attribution!
            team="ai-research",
            project="ml-education",
            customer_id="university-client-456",
            environment="production"
        )

        print("✅ AI Research Team Operation:")
        print(f"   💰 Cost: ${result1.cost_usd:.6f}")
        print(f"   ⚡ Latency: {result1.latency_ms:.0f}ms")
        print(f"   🔢 Tokens: {result1.input_tokens} in → {result1.output_tokens} out")
        print("   🏷️  Team: ai-research | Project: ml-education")
        print()

        # Example 2: Different Team and Project
        print("📊 Example 2: Different Team Attribution")
        print("-" * 38)

        result2 = adapter.text_generation(
            prompt="Write a professional summary of quarterly sales performance.",
            model="gemini-2.5-flash",
            team="sales-analytics",
            project="quarterly-reports",
            customer_id="enterprise-client-789",
            cost_center="Sales-Operations"
        )

        print("✅ Sales Analytics Team Operation:")
        print(f"   💰 Cost: ${result2.cost_usd:.6f}")
        print(f"   ⚡ Latency: {result2.latency_ms:.0f}ms")
        print(f"   🔢 Tokens: {result2.input_tokens} in → {result2.output_tokens} out")
        print("   🏷️  Team: sales-analytics | Project: quarterly-reports")
        print()

        # Example 3: Model Comparison
        print("🔬 Example 3: Multi-Model Cost Comparison")
        print("-" * 40)

        # Test prompt for comparison
        comparison_prompt = "Analyze the benefits and challenges of remote work in modern organizations."

        # Use Flash model
        start_time = time.time()
        flash_result = adapter.text_generation(
            prompt=comparison_prompt,
            model="gemini-2.5-flash",
            team="hr-analytics",
            project="workforce-analysis"
        )
        flash_duration = time.time() - start_time

        # Use Pro model for comparison
        start_time = time.time()
        pro_result = adapter.text_generation(
            prompt=comparison_prompt,
            model="gemini-2.5-pro",
            team="hr-analytics",
            project="workforce-analysis"
        )
        pro_duration = time.time() - start_time

        print("Model Performance Comparison:")
        print()
        print("📱 Gemini 2.5 Flash:")
        print(f"   💰 Cost: ${flash_result.cost_usd:.6f}")
        print(f"   ⚡ Latency: {flash_result.latency_ms:.0f}ms")
        print(f"   🔢 Tokens: {flash_result.input_tokens} → {flash_result.output_tokens}")
        print()
        print("🚀 Gemini 2.5 Pro:")
        print(f"   💰 Cost: ${pro_result.cost_usd:.6f}")
        print(f"   ⚡ Latency: {pro_result.latency_ms:.0f}ms")
        print(f"   🔢 Tokens: {pro_result.input_tokens} → {pro_result.output_tokens}")
        print()

        # Calculate cost difference
        cost_difference = pro_result.cost_usd - flash_result.cost_usd
        cost_ratio = pro_result.cost_usd / flash_result.cost_usd if flash_result.cost_usd > 0 else 0

        print("💡 Cost Analysis:")
        print(f"   📈 Pro costs ${cost_difference:.6f} more than Flash")
        print(f"   📊 Pro is {cost_ratio:.1f}x more expensive than Flash")
        print()

        # Example 4: Cost Comparison via API
        print("📋 Example 4: API-Based Model Comparison")
        print("-" * 42)

        models_to_compare = ["gemini-2.5-flash", "gemini-2.5-pro"]
        input_tokens = len(comparison_prompt.split()) * 1.3  # Rough estimate
        output_tokens = 200  # Estimated output

        comparison = compare_gemini_models(
            models=models_to_compare,
            input_tokens=int(input_tokens),
            output_tokens=output_tokens,
            sort_by="total_cost"
        )

        print("Cost Comparison for Similar Operations:")
        for i, model_data in enumerate(comparison):
            print(f"{i+1}. {model_data['display_name']}")
            print(f"   💰 Total Cost: ${model_data['total_cost']:.6f}")
            print(f"   📊 Cost per 1K tokens: ${model_data['cost_per_1k_tokens']:.6f}")
            print(f"   🎯 Best for: {model_data['description'][:50]}...")
            print()

        # Example 5: Cost Attribution Summary
        print("📈 Example 5: Cost Attribution Summary")
        print("-" * 37)

        total_cost = result1.cost_usd + result2.cost_usd + flash_result.cost_usd + pro_result.cost_usd

        print("Session Cost Breakdown:")
        print(f"   🔬 AI Research Team: ${result1.cost_usd:.6f}")
        print(f"   📊 Sales Analytics: ${result2.cost_usd:.6f}")
        print(f"   👥 HR Analytics (Flash): ${flash_result.cost_usd:.6f}")
        print(f"   👥 HR Analytics (Pro): ${pro_result.cost_usd:.6f}")
        print(f"   {'─' * 30}")
        print(f"   💰 Total Session Cost: ${total_cost:.6f}")
        print()

        # Teams summary
        team_costs = {
            "ai-research": result1.cost_usd,
            "sales-analytics": result2.cost_usd,
            "hr-analytics": flash_result.cost_usd + pro_result.cost_usd
        }

        print("Team Cost Attribution:")
        for team, cost in team_costs.items():
            percentage = (cost / total_cost) * 100 if total_cost > 0 else 0
            print(f"   {team}: ${cost:.6f} ({percentage:.1f}%)")
        print()

        print("🎉 Success! All operations tracked with GenOps governance:")
        print("   ✅ Automatic cost calculation and attribution")
        print("   ✅ Team and project tracking for billing")
        print("   ✅ Model performance comparison")
        print("   ✅ Real-time cost optimization insights")
        print("   ✅ OpenTelemetry export for observability platforms")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Install required packages:")
        print("   pip install genops-ai[gemini] google-generativeai")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Verify GEMINI_API_KEY is set correctly")
        print("   2. Check internet connectivity")
        print("   3. Run validation: python -c \"from genops.providers.gemini import validate_setup, print_validation_result; print_validation_result(validate_setup())\"")
        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎯 Next Steps:")
        print("   → Try cost optimization: python cost_optimization.py")
        print("   → Explore auto-instrumentation: python auto_instrumentation.py")
        print("   → Check production patterns: python production_patterns.py")

    exit(0 if success else 1)
