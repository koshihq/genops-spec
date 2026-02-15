#!/usr/bin/env python3
"""
🎯 GenOps + Ollama: 30-Second Confidence Builder

GOAL: Prove GenOps tracks your local Ollama models with zero code changes
TIME: 30 seconds
WHAT YOU'LL LEARN: GenOps automatically tracks local model costs and performance

This is your "hello world" for GenOps + Ollama integration.
Just run it and see GenOps tracking in action!

Prerequisites:
- Ollama installed and running: `ollama serve`
- At least one model: `ollama pull llama3.2:1b`
"""

import sys
import time


def main():
    print("🚀 GenOps + Ollama: 30-Second Confidence Builder")
    print("=" * 55)

    # Step 1: Validate setup
    print("\n📋 Step 1: Validating Ollama setup...")

    try:
        from genops.providers.ollama.validation import quick_validate

        if quick_validate():
            print("✅ Ollama server is running and accessible")
        else:
            print("❌ Ollama validation failed")
            print("\n🔧 Quick fixes:")
            print("   1. Start Ollama: ollama serve")
            print("   2. Pull a model: ollama pull llama3.2:1b")
            print("   3. Check connection: curl http://localhost:11434/api/version")
            return False

    except Exception as e:
        print(f"❌ Setup validation error: {e}")
        print("\n💡 Install GenOps: pip install genops-ai[ollama]")
        return False

    # Step 2: Enable GenOps tracking
    print("\n⚡ Step 2: Enabling GenOps tracking...")

    try:
        from genops.providers.ollama import auto_instrument

        # Enable automatic tracking with team attribution
        auto_instrument(team="quickstart-demo", project="30-second-test")
        print("✅ GenOps auto-instrumentation enabled")

    except Exception as e:
        print(f"❌ Auto-instrumentation error: {e}")
        return False

    # Step 3: Test with existing Ollama code
    print("\n🤖 Step 3: Testing with your existing Ollama code...")

    try:
        import ollama

        # Your existing Ollama code - NO CHANGES NEEDED!
        # GenOps will automatically track this
        print("   Generating text with local model...")

        start_time = time.time()
        response = ollama.generate(
            model="llama3.2:1b",  # Change to your available model
            prompt="What is GenOps in one sentence?",
        )
        duration = time.time() - start_time

        print("✅ Generation successful!")
        print(f"   📝 Response: {response['response'][:100]}...")
        print(f"   ⏱️  Duration: {duration:.1f}s")

    except Exception as e:
        error_str = str(e).lower()
        if "not found" in error_str or "model" in error_str:
            print("❌ Model not found")
            print("\n🔧 Available models:")
            try:
                models = ollama.list()
                if models.get("models"):
                    for model in models["models"][:3]:
                        print(f"   - {model['name']}")
                    print("\n💡 Update the model name in line 67 to one of the above")
                else:
                    print("   No models found. Pull one: ollama pull llama3.2:1b")
            except Exception:
                print("   Cannot list models. Check Ollama connection.")
            return False
        else:
            print(f"❌ Generation error: {e}")
            return False

    # Step 4: Show GenOps tracking results
    print("\n📊 Step 4: GenOps tracking results...")

    try:
        from genops.providers.ollama import get_model_manager, get_resource_monitor

        # Get resource monitoring data
        monitor = get_resource_monitor()
        current_metrics = monitor.get_current_metrics()

        if current_metrics:
            print("   🖥️ System Resources:")
            print(f"      CPU Usage: {current_metrics.cpu_usage_percent:.1f}%")
            print(f"      Memory: {current_metrics.memory_usage_mb:.0f}MB")
            if current_metrics.gpu_usage_percent > 0:
                print(f"      GPU Usage: {current_metrics.gpu_usage_percent:.1f}%")

        # Get model performance data
        manager = get_model_manager()
        performance = manager.get_model_performance_summary()

        if performance:
            for model, stats in performance.items():
                if stats.get("total_inferences", 0) > 0:
                    print(f"   🤖 Model Performance ({model}):")
                    print(f"      Inferences: {stats.get('total_inferences', 0)}")
                    print(
                        f"      Avg Latency: {stats.get('avg_inference_latency_ms', 0):.0f}ms"
                    )
                    if stats.get("cost_per_inference", 0) > 0:
                        print(
                            f"      Infrastructure Cost: ${stats.get('cost_per_inference', 0):.6f}/inference"
                        )

    except Exception as e:
        print(f"⚠️ Cannot display metrics: {e}")

    # Success!
    print("\n" + "=" * 55)
    print("🎉 SUCCESS! GenOps is now tracking your Ollama usage")
    print("=" * 55)

    print("\n✅ What you just accomplished:")
    print("   • GenOps automatically tracked your local model usage")
    print("   • Infrastructure costs calculated (GPU/CPU time, electricity)")
    print("   • Performance metrics captured (latency, throughput)")
    print("   • Team attribution applied (quickstart-demo team)")
    print("   • Zero changes to your existing Ollama code!")

    print("\n🚀 Next steps (choose your path):")
    print("   • 15 min: Run local_model_optimization.py for cost optimization")
    print("   • 30 min: Try ollama_production_deployment.py for enterprise patterns")
    print("   • 5 min: Check out the Ollama integration guide")

    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("\n🆘 If this persists:")
        print("   1. Check Ollama is running: ollama serve")
        print("   2. Reinstall GenOps: pip install --upgrade genops-ai[ollama]")
        print("   3. Report issue: https://github.com/KoshiHQ/GenOps-AI/issues")
        sys.exit(1)
