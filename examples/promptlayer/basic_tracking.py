#!/usr/bin/env python3
"""
Basic PromptLayer + GenOps Tracking Example

This example demonstrates basic PromptLayer prompt management enhanced with GenOps governance,
providing cost attribution, team tracking, and policy enforcement for your prompt engineering workflows.

About PromptLayer:
PromptLayer is a comprehensive prompt management platform that enables teams to version, evaluate,
and collaborate on AI prompts. GenOps enhances this with governance intelligence.

Usage:
    python basic_tracking.py

Prerequisites:
    pip install genops[promptlayer]  # Includes PromptLayer SDK
    export PROMPTLAYER_API_KEY="pl-your-api-key"
    export OPENAI_API_KEY="your-openai-api-key"  # For LLM operations
    
    # Optional: For governance attribution
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
"""

import asyncio
import os
from datetime import datetime


def basic_promptlayer_with_genops():
    """
    Demonstrates basic PromptLayer prompt management enhanced with GenOps governance.
    
    This example shows how GenOps adds cost attribution, team tracking, and 
    governance context to PromptLayer prompt operations.
    """
    print("🔍 Basic PromptLayer + GenOps Tracking Example")
    print("=" * 50)

    try:
        # Import GenOps PromptLayer adapter
        from genops.providers.promptlayer import instrument_promptlayer
        print("✅ GenOps PromptLayer adapter loaded successfully")

        # Initialize with governance context
        adapter = instrument_promptlayer(
            promptlayer_api_key=os.getenv('PROMPTLAYER_API_KEY'),
            team=os.getenv('GENOPS_TEAM', 'engineering'),
            project=os.getenv('GENOPS_PROJECT', 'prompt-optimization'),
            customer_id="demo-customer",
            environment="development",
            cost_center="rd-department",
            enable_cost_alerts=True
        )
        print("✅ GenOps governance context configured")

    except ImportError as e:
        print(f"❌ Failed to import GenOps PromptLayer adapter: {e}")
        print("💡 Fix: Run 'pip install genops[promptlayer]'")
        return False

    # Check if PromptLayer is available
    try:
        import promptlayer
        print("✅ PromptLayer SDK available")
    except ImportError:
        print("❌ PromptLayer SDK not found")
        print("💡 Fix: Run 'pip install promptlayer'")
        return False

    print("\n🚀 Running Enhanced Prompt Operations...")
    print("-" * 40)

    # Example 1: Basic prompt execution with governance tracking
    print("\n1️⃣ Basic Prompt Execution with Cost Attribution")
    try:
        with adapter.track_prompt_operation(
            prompt_name="customer_support_v1",
            prompt_version="1.0",
            operation_type="prompt_run",
            operation_name="basic_support_query",
            tags={"use_case": "customer_support", "priority": "high"}
        ) as span:

            # Run prompt with governance (mock for demonstration)
            result = adapter.run_prompt_with_governance(
                prompt_name="customer_support_v1",
                input_variables={
                    "customer_query": "How do I reset my password?",
                    "customer_tier": "premium"
                },
                tags=["customer_support", "password_reset"]
            )

            # Simulate response and cost
            span.update_cost(0.0023)  # Estimated cost
            span.update_token_usage(
                input_tokens=45,
                output_tokens=78,
                model="gpt-3.5-turbo"
            )

            print("✅ Prompt executed with governance tracking")
            print(f"📝 Response: {result.get('response', {}).get('mock', 'Demo response generated')}")

            # Access governance-enhanced metrics
            metrics = span.get_metrics()
            print(f"💰 Estimated cost: ${metrics.get('estimated_cost', 0):.6f}")
            print(f"🏷️ Team attribution: {metrics.get('team', 'N/A')}")
            print(f"📊 Tokens used: {metrics.get('total_tokens', 'N/A')}")

    except Exception as e:
        print(f"❌ Prompt execution failed: {e}")
        print("🔧 Troubleshooting:")
        print("   • Check API keys: echo $PROMPTLAYER_API_KEY $OPENAI_API_KEY")
        print("   • Verify network connectivity")
        print("   • Ensure PromptLayer account has valid prompts")
        if "api key" in str(e).lower():
            print("   💡 API Key Issue: Set PROMPTLAYER_API_KEY environment variable")
        elif "not found" in str(e).lower():
            print("   💡 Prompt Issue: Create prompts in PromptLayer dashboard first")
        return False

    # Example 2: Prompt A/B testing with governance
    print("\n2️⃣ A/B Testing with Governance Attribution")
    try:
        test_variants = ["v1_formal", "v2_casual", "v3_concise"]

        with adapter.track_prompt_operation(
            prompt_name="product_description",
            operation_type="ab_test",
            operation_name="description_optimization",
            tags={"experiment": "description_test_q4", "team": "marketing"}
        ) as batch_span:

            variant_costs = []
            for i, variant in enumerate(test_variants):
                with adapter.track_prompt_operation(
                    prompt_name=f"product_description_{variant}",
                    prompt_version=variant,
                    operation_type="prompt_run",
                    operation_name=f"variant_{variant}",
                    tags={"variant": variant, "test_group": "description_optimization"}
                ) as variant_span:

                    result = adapter.run_prompt_with_governance(
                        prompt_name=f"product_description_{variant}",
                        input_variables={
                            "product_name": "Smart Home Assistant",
                            "key_features": ["Voice control", "Smart scheduling", "Energy monitoring"]
                        },
                        tags=[f"variant_{variant}", "ab_test"]
                    )

                    # Simulate different costs for different variants
                    cost = 0.0015 + (i * 0.0005)  # Varying complexity
                    variant_span.update_cost(cost)
                    variant_span.update_token_usage(
                        input_tokens=35 + (i * 10),
                        output_tokens=120 + (i * 15),
                        model="gpt-3.5-turbo"
                    )

                    variant_costs.append(cost)
                    print(f"   ✅ Variant {variant}: ${cost:.6f}")

            total_cost = sum(variant_costs)
            print(f"💰 Total A/B test cost: ${total_cost:.6f}")
            print("🏷️ Cost attributed to team: marketing")

    except Exception as e:
        print(f"❌ A/B testing failed: {e}")
        print("🔧 A/B Testing Troubleshooting:")
        print("   • Ensure multiple prompt versions exist in PromptLayer")
        print("   • Check variant naming conventions")
        print("   • Consider starting with fewer variants")
        return False

    # Example 3: Prompt evaluation with governance
    print("\n3️⃣ Prompt Evaluation with Cost Intelligence")
    try:
        evaluation_prompts = [
            {"name": "email_writer_v1", "category": "formal"},
            {"name": "email_writer_v2", "category": "friendly"},
            {"name": "email_writer_v3", "category": "concise"}
        ]

        with adapter.track_prompt_operation(
            prompt_name="email_writer_evaluation",
            operation_type="evaluation",
            operation_name="performance_comparison",
            tags={"evaluation_type": "cost_performance", "team": "product"}
        ) as eval_span:

            evaluation_results = []
            for prompt_info in evaluation_prompts:
                # Simulate evaluation run
                prompt_name = prompt_info["name"]
                category = prompt_info["category"]

                result = adapter.run_prompt_with_governance(
                    prompt_name=prompt_name,
                    input_variables={
                        "recipient": "valued customer",
                        "subject": "Product update notification",
                        "key_points": ["New features", "Improved performance", "Thank you"]
                    },
                    tags=["evaluation", f"category_{category}"]
                )

                # Simulate evaluation metrics
                performance_score = 0.85 + (hash(prompt_name) % 100) / 1000  # Simulated
                cost_efficiency = 0.002 + (len(prompt_name) % 10) / 10000

                evaluation_results.append({
                    "prompt": prompt_name,
                    "category": category,
                    "performance_score": performance_score,
                    "cost_efficiency": cost_efficiency,
                    "cost_per_quality_point": cost_efficiency / performance_score
                })

                print(f"   📊 {prompt_name}: Performance {performance_score:.3f}, Cost ${cost_efficiency:.6f}")

            # Find best performing prompt
            best_prompt = min(evaluation_results, key=lambda x: x["cost_per_quality_point"])

            print(f"🏆 Best prompt: {best_prompt['prompt']}")
            print(f"   💰 Cost per quality point: ${best_prompt['cost_per_quality_point']:.6f}")
            print("   🏷️ Evaluation attributed to team: product")

            eval_span.add_attributes({
                "evaluation.best_prompt": best_prompt["prompt"],
                "evaluation.prompts_tested": len(evaluation_prompts),
                "evaluation.cost_per_quality": best_prompt["cost_per_quality_point"]
            })

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

    return True

def demonstrate_governance_features():
    """Demonstrate specific GenOps governance features with PromptLayer."""
    print("\n🛡️ GenOps Governance Features Demo")
    print("-" * 35)

    try:
        from genops.providers.promptlayer import (
            GovernancePolicy,
            instrument_promptlayer,
        )

        # Initialize with strict governance policies
        adapter = instrument_promptlayer(
            promptlayer_api_key=os.getenv('PROMPTLAYER_API_KEY'),
            team="compliance-team",
            project="prompt-governance",
            environment="production",
            enable_cost_alerts=True,
            max_operation_cost=0.05,  # $0.05 limit per operation
            daily_budget_limit=1.0,   # $1.00 daily limit
            governance_policy=GovernancePolicy.ADVISORY
        )

        print("✅ Governance policies configured:")
        print("   • Cost alerts: Enabled")
        print("   • Max operation cost: $0.05")
        print("   • Daily budget limit: $1.00")

        # Test governance enforcement
        with adapter.track_prompt_operation(
            prompt_name="governance_test",
            operation_type="policy_validation",
            operation_name="budget_compliance_demo"
        ) as span:

            # Simulate a prompt operation
            result = adapter.run_prompt_with_governance(
                prompt_name="governance_test",
                input_variables={"query": "Test governance policies"},
                tags=["governance", "compliance_test"]
            )

            # Simulate cost that might trigger policies
            span.update_cost(0.03)  # Within limits

            metrics = span.get_metrics()
            cost = metrics.get('estimated_cost', 0.0)

            if cost <= 0.05:
                print(f"✅ Operation within cost limits: ${cost:.6f}")
            else:
                print(f"⚠️ Cost threshold would be exceeded: ${cost:.6f}")

            print("📊 Governance context captured:")
            print(f"   • Team: {metrics.get('team')}")
            print(f"   • Project: {metrics.get('project')}")
            print(f"   • Environment: {metrics.get('environment', 'N/A')}")

    except Exception as e:
        print(f"❌ Governance demo failed: {e}")
        return False

    return True

def show_promptlayer_integration():
    """Show how GenOps integrates with PromptLayer features."""
    print("\n🔗 PromptLayer Integration Details")
    print("-" * 35)

    try:
        from genops.providers.promptlayer import GenOpsPromptLayerAdapter

        print("✅ PromptLayer integration features:")
        print("   • Prompt versioning with cost tracking")
        print("   • A/B testing with governance attribution")
        print("   • Evaluation workflows with budget enforcement")
        print("   • Team collaboration with cost intelligence")

        # Show adapter configuration
        adapter = GenOpsPromptLayerAdapter(
            team="integration-demo",
            project="feature-showcase"
        )

        print("✅ GenOps enhancements:")
        print("   • Automatic cost calculation per prompt execution")
        print("   • Team and project attribution for all operations")
        print("   • Policy enforcement and budget monitoring")
        print("   • Integration with existing observability platforms")

        # Show metrics
        metrics = adapter.get_metrics()
        print("✅ Available metrics:")
        print(f"   • Team: {metrics.get('team')}")
        print(f"   • Project: {metrics.get('project')}")
        print(f"   • Environment: {metrics.get('environment')}")
        print(f"   • Governance enabled: {metrics.get('governance_enabled')}")

    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        return False

    return True

async def main():
    """Main execution function."""
    print("🚀 Starting PromptLayer + GenOps Basic Tracking Demo")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check prerequisites
    if not os.getenv('PROMPTLAYER_API_KEY'):
        print("❌ PROMPTLAYER_API_KEY not found")
        print("💡 Set your PromptLayer API key: export PROMPTLAYER_API_KEY='pl-your-key'")
        print("📖 Get your API key from: https://promptlayer.com/")
        return False

    # Run examples
    success = True

    # Basic tracking examples
    if not basic_promptlayer_with_genops():
        success = False

    # Governance features
    if success and not demonstrate_governance_features():
        success = False

    # PromptLayer integration details
    if success and not show_promptlayer_integration():
        success = False

    if success:
        print("\n" + "🌟" * 50)
        print("🎉 PromptLayer + GenOps Basic Tracking Demo Complete!")
        print("\n📊 What You've Accomplished:")
        print("   ✅ Enhanced PromptLayer prompts with governance intelligence")
        print("   ✅ Automatic cost attribution and team tracking")
        print("   ✅ A/B testing with cost intelligence")
        print("   ✅ Prompt evaluation with governance oversight")

        print("\n🔍 Your Enhanced Prompt Management Stack:")
        print("   • PromptLayer: Prompt versioning and collaboration platform")
        print("   • GenOps: Governance, cost intelligence, and policy enforcement")
        print("   • OpenTelemetry: Industry-standard observability integration")
        print("   • Multi-provider: Works with OpenAI, Anthropic, and other LLM providers")

        print("\n📚 Next Steps:")
        print("   • Run 'python auto_instrumentation.py' for zero-code integration")
        print("   • Try 'python prompt_management.py' for advanced prompt governance")
        print("   • Explore 'python evaluation_integration.py' for evaluation workflows")

        print("\n💡 Quick Integration:")
        print("   Add this to your existing PromptLayer code:")
        print("   ```python")
        print("   from genops.providers.promptlayer import instrument_promptlayer")
        print("   adapter = instrument_promptlayer(team='your-team', project='your-project')")
        print("   # Your existing PromptLayer code now includes governance!")
        print("   ```")

        print("🌟" * 50)
    else:
        print("\n❌ Demo encountered errors. Please check the output above.")

    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
