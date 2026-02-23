#!/usr/bin/env python3
"""
🚀 GenOps + Ollama: Local Model Optimization (Phase 2)

GOAL: Optimize local model costs and performance with GenOps intelligence
TIME: 15-30 minutes
WHAT YOU'LL LEARN: Cost comparison, resource optimization, model selection strategies

This example shows how to use GenOps to optimize your local Ollama deployment:
- Compare costs across different models
- Get resource utilization recommendations
- Optimize for different use cases (speed vs quality vs cost)
- Monitor resource efficiency over time

Prerequisites:
- Completed hello_ollama_minimal.py (Phase 1)
- Multiple models available (we'll help you pull them)
- Ollama server running
"""

import time
from dataclasses import dataclass


@dataclass
class OptimizationTest:
    """Test case for optimization analysis."""

    name: str
    prompt: str
    priority: str  # "speed", "quality", "cost"
    expected_complexity: str  # "simple", "medium", "complex"


def main():
    print("🚀 GenOps + Ollama: Local Model Optimization")
    print("=" * 55)

    # Step 1: Setup and validation
    print("\n📋 Step 1: Setting up optimization environment...")

    try:
        import ollama

        from genops.providers.ollama import (
            auto_instrument,
            get_model_manager,
            get_resource_monitor,
        )
        from genops.providers.ollama.validation import validate_setup  # noqa: F401

        # Enable comprehensive tracking
        auto_instrument(
            team="optimization-team",
            project="model-efficiency-analysis",
            resource_monitoring=True,
            model_management=True,
        )
        print("✅ GenOps optimization tracking enabled")

    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install: pip install genops-ai[ollama] ollama")
        return False
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

    # Step 2: Ensure we have multiple models for comparison
    print("\n🤖 Step 2: Checking available models...")

    try:
        available_models = ollama.list()["models"]
        model_names = [model["name"] for model in available_models]

        print(f"✅ Found {len(available_models)} models:")
        for model in available_models[:5]:  # Show first 5
            size_gb = model.get("size", 0) / (1024**3)
            print(f"   • {model['name']} ({size_gb:.1f}GB)")

        if len(available_models) > 5:
            print(f"   ... and {len(available_models) - 5} more")

        # Recommend additional models if needed
        if len(available_models) < 3:
            print(
                "\n💡 For better optimization analysis, consider pulling additional models:"
            )
            recommended = [
                ("llama3.2:1b", "Fast, lightweight model"),
                ("llama3.2:3b", "Balanced performance/quality"),
                ("mistral:7b", "Alternative architecture"),
            ]

            for model, desc in recommended:
                if model not in model_names:
                    print(f"   ollama pull {model}  # {desc}")

    except Exception as e:
        print(f"❌ Cannot list models: {e}")
        return False

    # Step 3: Run optimization tests
    print("\n⚡ Step 3: Running optimization test suite...")

    # Define test cases for different optimization scenarios
    test_cases = [
        OptimizationTest(
            name="Speed Priority",
            prompt="Hello world",
            priority="speed",
            expected_complexity="simple",
        ),
        OptimizationTest(
            name="Quality Priority",
            prompt="Explain quantum computing and its potential applications in cryptography",
            priority="quality",
            expected_complexity="complex",
        ),
        OptimizationTest(
            name="Cost Priority",
            prompt="What is 2+2?",
            priority="cost",
            expected_complexity="simple",
        ),
        OptimizationTest(
            name="Balanced Workload",
            prompt="Write a Python function to reverse a string",
            priority="balanced",
            expected_complexity="medium",
        ),
    ]

    # Select models to test (use available models, prefer variety)
    test_models = []
    for model_name in model_names[:4]:  # Test up to 4 models
        test_models.append(model_name)

    if not test_models:
        print("❌ No models available for testing")
        return False

    print(f"🧪 Testing with {len(test_models)} models: {', '.join(test_models)}")

    # Run the optimization tests
    results = {}

    for test_case in test_cases:
        print(f"\n   Running test: {test_case.name}")
        results[test_case.name] = {}

        for model in test_models:
            try:
                print(f"      Testing {model}...", end=" ")

                start_time = time.time()
                response = ollama.generate(
                    model=model,
                    prompt=test_case.prompt,
                    options={"num_predict": 100},  # Limit tokens for consistency
                )
                duration = time.time() - start_time

                results[test_case.name][model] = {
                    "duration_ms": duration * 1000,
                    "response_length": len(response.get("response", "")),
                    "success": True,
                }

                print(f"✅ {duration:.1f}s")

            except Exception as e:
                print(f"❌ {str(e)[:50]}...")
                results[test_case.name][model] = {
                    "duration_ms": 0,
                    "response_length": 0,
                    "success": False,
                    "error": str(e),
                }

        # Small delay between test cases
        time.sleep(1)

    # Step 4: Analyze optimization opportunities
    print("\n📊 Step 4: Analyzing optimization opportunities...")

    try:
        # Get comprehensive performance data
        manager = get_model_manager()
        monitor = get_resource_monitor()

        # Get model performance summary
        performance_summary = manager.get_model_performance_summary()

        # Get current system metrics
        current_metrics = monitor.get_current_metrics()
        monitor.get_hardware_summary(duration_minutes=10)

        print("\n   📈 Performance Analysis:")
        for model, stats in performance_summary.items():
            if stats.get("total_inferences", 0) > 0:
                print(f"      {model}:")
                print(
                    f"         Avg Latency: {stats.get('avg_inference_latency_ms', 0):.0f}ms"
                )
                print(
                    f"         Throughput: {stats.get('avg_tokens_per_second', 0):.1f} tokens/sec"
                )
                if stats.get("cost_per_inference", 0) > 0:
                    print(
                        f"         Cost/Inference: ${stats.get('cost_per_inference', 0):.6f}"
                    )

        print("\n   🖥️ Current System Utilization:")
        if current_metrics:
            print(f"      CPU: {current_metrics.cpu_usage_percent:.1f}%")
            print(f"      Memory: {current_metrics.memory_usage_mb:.0f}MB")
            if current_metrics.gpu_usage_percent > 0:
                print(f"      GPU: {current_metrics.gpu_usage_percent:.1f}%")

        # Get optimization recommendations
        print("\n   💡 System Optimization Recommendations:")
        recommendations = monitor.get_optimization_recommendations()
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"      {i}. {rec}")

        # Model-specific recommendations
        model_recommendations = manager.get_optimization_recommendations()
        if model_recommendations:
            print("\n   🤖 Model-Specific Recommendations:")
            for model, optimizer in model_recommendations.items():
                if (
                    hasattr(optimizer, "optimization_opportunities")
                    and optimizer.optimization_opportunities
                ):
                    print(f"      {model}:")
                    for opp in optimizer.optimization_opportunities[:2]:
                        print(f"         • {opp}")

    except Exception as e:
        print(f"⚠️ Analysis error: {e}")

    # Step 5: Generate optimization strategy
    print("\n🎯 Step 5: Generating optimization strategy...")

    # Analyze test results to provide recommendations
    if results:
        print("\n   📋 Optimization Strategy Based on Test Results:")

        # Find fastest model overall
        speed_scores = {}
        quality_scores = {}

        for _test_name, test_results in results.items():
            for model, result in test_results.items():
                if result.get("success", False):
                    # Speed score (lower duration is better)
                    duration = result.get("duration_ms", float("inf"))
                    if model not in speed_scores:
                        speed_scores[model] = []
                    speed_scores[model].append(
                        1000 / max(duration, 1)
                    )  # Inverse of duration

                    # Quality proxy (longer responses might be more detailed)
                    response_length = result.get("response_length", 0)
                    if model not in quality_scores:
                        quality_scores[model] = []
                    quality_scores[model].append(response_length)

        # Calculate averages
        avg_speed = {
            model: sum(scores) / len(scores)
            for model, scores in speed_scores.items()
            if scores
        }
        avg_quality = {
            model: sum(scores) / len(scores)
            for model, scores in quality_scores.items()
            if scores
        }

        if avg_speed:
            fastest_model = max(avg_speed.keys(), key=lambda m: avg_speed[m])
            print(f"      🏃 Speed Champion: {fastest_model}")
            print(
                "         Use for: Simple queries, real-time applications, high-volume processing"
            )

        if avg_quality and len(avg_quality) > 1:
            # Find model with best balance of speed and quality
            balanced_scores = {}
            for model in avg_speed.keys():
                if model in avg_quality:
                    # Normalize both scores and combine
                    speed_norm = avg_speed[model] / max(avg_speed.values())
                    quality_norm = avg_quality[model] / max(avg_quality.values())
                    balanced_scores[model] = (speed_norm + quality_norm) / 2

            if balanced_scores:
                balanced_model = max(
                    balanced_scores.keys(), key=lambda m: balanced_scores[m]
                )
                print(f"      ⚖️ Balanced Choice: {balanced_model}")
                print(
                    "         Use for: General purpose, mixed workloads, production defaults"
                )

    # Step 6: Cost optimization insights
    print("\n💰 Step 6: Cost optimization insights...")

    try:
        # Get usage analytics
        usage_analytics = manager.get_model_usage_analytics(days=1)  # Last day

        if usage_analytics.get("total_cost", 0) > 0:
            print("   📊 Today's Infrastructure Costs:")
            print(f"      Total Cost: ${usage_analytics['total_cost']:.6f}")
            print(f"      Total Inferences: {usage_analytics['total_inferences']}")
            print(f"      Active Models: {usage_analytics['active_models']}")

            # Show top cost contributors
            models_by_cost = usage_analytics.get("models_by_cost", [])
            if models_by_cost:
                print("   💸 Top Cost Contributors:")
                for i, model_cost in enumerate(models_by_cost[:3], 1):
                    print(
                        f"      {i}. {model_cost['model']}: ${model_cost['total_cost']:.6f}"
                    )

        # Cost optimization suggestions
        print("\n   💡 Cost Optimization Strategies:")
        print("      • Use smaller models (1B-3B params) for simple tasks")
        print("      • Cache frequently requested completions")
        print("      • Batch similar requests together")
        print("      • Set inference limits for development/testing")
        print("      • Monitor GPU utilization - scale hardware as needed")

    except Exception as e:
        print(f"⚠️ Cost analysis error: {e}")

    # Success summary
    print("\n" + "=" * 55)
    print("🎉 SUCCESS! Local Model Optimization Complete")
    print("=" * 55)

    print("\n✅ What you accomplished:")
    print("   • Compared performance across multiple local models")
    print("   • Identified optimization opportunities for different use cases")
    print("   • Analyzed infrastructure cost patterns")
    print("   • Generated data-driven optimization strategies")
    print("   • Got system-level performance recommendations")

    print("\n🚀 Next steps:")
    print("   • Apply recommendations to your production workloads")
    print("   • Set up monitoring dashboards with your preferred observability tool")
    print("   • Try ollama_production_deployment.py for enterprise patterns")
    print("   • Explore advanced cost controls and budget enforcement")

    print("\n📊 Export your data:")
    print("   • Performance data is automatically tracked in GenOps telemetry")
    print("   • Export model data: manager.export_model_data('json')")
    print("   • View in your observability platform via OpenTelemetry export")

    return True


if __name__ == "__main__":
    import sys

    try:
        success = main()
        if success:
            print("\n🎓 Ready for Phase 3? Try: python ollama_production_deployment.py")
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Optimization interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("\n🆘 If this persists:")
        print("   1. Ensure you have multiple models: ollama list")
        print("   2. Check system resources: free -h, nvidia-smi")
        print(
            "   3. Report issue with details: https://github.com/KoshiHQ/GenOps-AI/issues"
        )
        sys.exit(1)
