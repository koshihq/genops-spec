#!/usr/bin/env python3
"""
🎯 GenOps + Cohere: 30-Second Confidence Builder

GOAL: Prove GenOps tracks your Cohere operations with zero complexity
TIME: 30 seconds
WHAT YOU'LL LEARN: GenOps automatically tracks all Cohere costs and performance

This is your "hello world" for GenOps + Cohere integration.
Just run it and see GenOps tracking in action!

Prerequisites:
- Cohere API key: export CO_API_KEY="your-key"
- Cohere client: pip install cohere
"""

import os
import sys
import time


def main():
    print("🚀 GenOps + Cohere: 30-Second Confidence Builder")
    print("="*55)

    # Step 1: Validate setup
    print("\n📋 Step 1: Validating Cohere setup...")

    try:
        from genops.providers.cohere_validation import quick_validate

        if quick_validate():
            print("✅ Cohere API is accessible and authenticated")
        else:
            print("❌ Cohere validation failed")
            print("\n🔧 Quick fixes:")
            print("   1. Set API key: export CO_API_KEY=your-cohere-key")
            print("   2. Install client: pip install cohere")
            print("   3. Test connection: python -c \"import cohere; client = cohere.ClientV2(); print('OK')\"")
            return False

    except Exception as e:
        print(f"❌ Setup validation error: {e}")
        print("\n💡 Install GenOps: pip install genops-ai")
        return False

    # Step 2: Enable GenOps tracking
    print("\n⚡ Step 2: Enabling GenOps tracking...")

    try:
        from genops.providers.cohere import instrument_cohere

        # Create adapter with team attribution
        adapter = instrument_cohere(
            team="quickstart-demo",
            project="30-second-test"
        )
        print("✅ GenOps Cohere adapter initialized")

    except Exception as e:
        print(f"❌ Adapter initialization error: {e}")
        return False

    # Step 3: Test with Cohere operation
    print("\n🤖 Step 3: Testing Cohere operation with GenOps tracking...")

    try:
        print("   Generating text with Cohere...")

        start_time = time.time()
        response = adapter.chat(
            message="What is GenOps in one sentence?",
            model="command-light"  # Fast, cost-effective model
        )
        duration = time.time() - start_time

        print("✅ Generation successful!")
        print(f"   📝 Response: {response.content[:100]}...")
        print(f"   ⏱️  Duration: {duration:.1f}s")

        if response.usage:
            print(f"   🔢 Tokens: {response.usage.input_tokens} in + {response.usage.output_tokens} out = {response.usage.total_tokens} total")
            print(f"   💰 Cost: ${response.usage.total_cost:.6f}")
            if response.usage.tokens_per_second > 0:
                print(f"   ⚡ Speed: {response.usage.tokens_per_second:.1f} tokens/second")

    except Exception as e:
        error_str = str(e).lower()
        if "unauthorized" in error_str or "invalid" in error_str:
            print("❌ API authentication failed")
            print("\n🔧 Fix your API key:")
            print(f"   Current: {os.getenv('CO_API_KEY', 'NOT SET')[:10]}...")
            print("   Get key: https://dashboard.cohere.ai/")
            return False
        elif "not found" in error_str or "model" in error_str:
            print("❌ Model not found")
            print("\n🔧 Available models to try:")
            print("   - command-light (cheapest, fastest)")
            print("   - command-r-08-2024 (balanced)")
            print("   - command-r-plus-08-2024 (most capable)")
            return False
        elif "rate limit" in error_str:
            print("❌ Rate limit exceeded")
            print("\n💡 Try again in a few minutes or upgrade your Cohere plan")
            return False
        else:
            print(f"❌ Generation error: {e}")
            return False

    # Step 4: Show additional operation types
    print("\n🔄 Step 4: Testing multi-operation tracking...")

    try:
        # Test embedding operation
        print("   Creating embeddings...")
        embed_response = adapter.embed(
            texts=["GenOps tracks AI costs", "Cohere provides enterprise AI"],
            model="embed-english-v4.0"
        )

        if embed_response.usage:
            print(f"✅ Embedding successful: ${embed_response.usage.total_cost:.6f} cost")

        # Test reranking operation
        print("   Testing rerank operation...")
        rerank_response = adapter.rerank(
            query="AI cost tracking",
            documents=[
                "GenOps helps track AI costs and usage",
                "Machine learning models are expensive",
                "Cost optimization for AI workloads"
            ],
            model="rerank-english-v3.0"
        )

        if rerank_response.usage:
            print(f"✅ Rerank successful: ${rerank_response.usage.total_cost:.6f} cost")

    except Exception as e:
        print(f"⚠️ Additional operations test: {str(e)[:100]}...")
        print("   (This is normal - some operations may need specific API access)")

    # Step 5: Show usage summary
    print("\n📊 Step 5: GenOps usage summary...")

    try:
        summary = adapter.get_usage_summary()

        if summary:
            print("   💰 Cost Summary:")
            print(f"      Total Operations: {summary.get('total_operations', 0)}")
            print(f"      Total Cost: ${summary.get('total_cost', 0):.6f}")
            print(f"      Average Cost/Operation: ${summary.get('average_cost_per_operation', 0):.6f}")

            if summary.get('budget_limit'):
                utilization = (summary.get('total_cost', 0) / summary['budget_limit']) * 100
                print(f"      Budget Utilization: {utilization:.1f}%")

    except Exception as e:
        print(f"⚠️ Cannot display summary: {e}")

    # Success!
    print("\n" + "="*55)
    print("🎉 SUCCESS! GenOps is now tracking your Cohere usage")
    print("="*55)

    print("\n✅ What you just accomplished:")
    print("   • GenOps automatically tracked all your Cohere operations")
    print("   • Multi-operation cost tracking (chat, embed, rerank)")
    print("   • Performance metrics captured (latency, tokens/second)")
    print("   • Team attribution applied (quickstart-demo team)")
    print("   • Zero changes to standard Cohere workflow!")

    print("\n🚀 Next steps (choose your path):")
    print("   • 15 min: Run multi_operation_tracking.py for unified workflows")
    print("   • 30 min: Try cost_optimization.py for model comparison")
    print("   • 45 min: Check out auto_instrumentation.py for zero-code integration")
    print("   • 5 min: Review the Cohere integration guide")

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
        print("   1. Check API key: echo $CO_API_KEY")
        print("   2. Reinstall: pip install --upgrade genops-ai cohere")
        print("   3. Run diagnostics: python -c \"from genops.providers.cohere_validation import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)\"")
        print("   4. Report issue: https://github.com/KoshiHQ/GenOps-AI/issues")
        sys.exit(1)
