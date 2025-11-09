#!/usr/bin/env python3
"""
Together AI Basic Tracking with GenOps Governance

Demonstrates basic Together AI operations with automatic cost tracking and governance.
Perfect starting point for integrating Together AI with GenOps governance controls.

Usage:
    python basic_tracking.py

Features:
    - Simple chat completions with cost tracking
    - Automatic governance attribute collection
    - Budget awareness and cost alerts
    - Multiple model comparisons
    - Session-based operation tracking
"""

import os
import sys
from decimal import Decimal

try:
    from genops.providers.together import GenOpsTogetherAdapter, TogetherModel
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install GenOps: pip install genops-ai[together]")
    sys.exit(1)


def main():
    """Demonstrate basic Together AI tracking with GenOps."""
    print("🤖 Together AI Basic Tracking with GenOps")
    print("=" * 50)
    
    # Initialize adapter with governance configuration
    adapter = GenOpsTogetherAdapter(
        team=os.getenv('GENOPS_TEAM', 'demo-team'),
        project=os.getenv('GENOPS_PROJECT', 'basic-tracking'),
        environment=os.getenv('GENOPS_ENVIRONMENT', 'development'),
        daily_budget_limit=50.0,  # $50 daily budget
        monthly_budget_limit=1000.0,  # $1000 monthly budget
        enable_governance=True,
        enable_cost_alerts=True,
        governance_policy='advisory',  # Won't block operations, just warns
        default_model=TogetherModel.LLAMA_3_1_8B_INSTRUCT  # Cost-effective default
    )
    
    print("✅ GenOps Together adapter initialized")
    print(f"   Team: {adapter.team}")
    print(f"   Project: {adapter.project}")
    print(f"   Daily budget: ${adapter.daily_budget_limit}")
    
    # Example 1: Simple chat completion with basic governance
    print("\n" + "=" * 50)
    print("📝 Example 1: Basic Chat Completion")
    print("=" * 50)
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain what makes Together AI unique in 2-3 sentences."}
        ]
        
        result = adapter.chat_with_governance(
            messages=messages,
            model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=150,
            temperature=0.7,
            # Governance attributes
            feature="basic-demo",
            use_case="model-explanation"
        )
        
        print("🎯 Response:")
        print(f"   {result.response}")
        print(f"\n📊 Metrics:")
        print(f"   Model: {result.model_used}")
        print(f"   Tokens: {result.tokens_used}")
        print(f"   Cost: ${result.cost:.6f}")
        print(f"   Time: {result.execution_time_seconds:.2f}s")
        
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return 1
    
    # Example 2: Compare multiple models
    print("\n" + "=" * 50)
    print("🔬 Example 2: Model Comparison")
    print("=" * 50)
    
    models_to_test = [
        TogetherModel.LLAMA_3_1_8B_INSTRUCT,      # Ultra-fast, cost-effective
        TogetherModel.DEEPSEEK_R1_DISTILL,        # Reasoning optimized
        TogetherModel.MIXTRAL_8X7B                # Balanced performance
    ]
    
    question = "What are the main benefits of open-source AI models?"
    messages = [{"role": "user", "content": question}]
    
    model_results = []
    
    for model in models_to_test:
        try:
            print(f"\n🧠 Testing {model.value}...")
            
            result = adapter.chat_with_governance(
                messages=messages,
                model=model,
                max_tokens=100,
                temperature=0.5,
                # Track which model comparison this is
                comparison_batch="model-comparison",
                model_name=model.value
            )
            
            model_results.append(result)
            print(f"   ✅ Response length: {len(result.response)} chars")
            print(f"   💰 Cost: ${result.cost:.6f}")
            print(f"   ⏱️  Time: {result.execution_time_seconds:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    # Compare results
    if model_results:
        print(f"\n📊 Model Comparison Summary:")
        total_cost = sum(r.cost for r in model_results)
        avg_time = sum(r.execution_time_seconds for r in model_results) / len(model_results)
        
        print(f"   Models tested: {len(model_results)}")
        print(f"   Total cost: ${total_cost:.6f}")
        print(f"   Average time: {avg_time:.2f}s")
        
        # Find most cost-effective
        cheapest = min(model_results, key=lambda x: x.cost)
        print(f"   Most cost-effective: {cheapest.model_used} (${cheapest.cost:.6f})")
        
        # Find fastest
        fastest = min(model_results, key=lambda x: x.execution_time_seconds)
        print(f"   Fastest: {fastest.model_used} ({fastest.execution_time_seconds:.2f}s)")
    
    # Example 3: Session-based tracking
    print("\n" + "=" * 50)
    print("🎯 Example 3: Session-Based Tracking")
    print("=" * 50)
    
    try:
        # Use session context manager for related operations
        with adapter.track_session(
            "creative-writing",
            customer_id="demo-customer",
            use_case="content-generation"
        ) as session:
            
            print(f"📋 Started session: {session.session_name}")
            print(f"   Session ID: {session.session_id}")
            
            # Multiple related operations in the same session
            creative_prompts = [
                "Write a haiku about artificial intelligence",
                "Create a short story opening line about robots and humans",
                "Suggest three creative names for an AI assistant"
            ]
            
            session_results = []
            for i, prompt in enumerate(creative_prompts, 1):
                print(f"\n   📝 Operation {i}/{len(creative_prompts)}")
                
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": prompt}],
                    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=80,
                    session_id=session.session_id,
                    operation_index=i
                )
                
                session_results.append(result)
                print(f"      Response: {result.response[:60]}...")
                print(f"      Cost: ${result.cost:.6f}")
            
            print(f"\n📊 Session Summary:")
            print(f"   Total operations: {session.total_operations}")
            print(f"   Total cost: ${session.total_cost:.6f}")
            print(f"   Average cost/operation: ${session.total_cost / len(session_results):.6f}")
    
    except Exception as e:
        print(f"❌ Session tracking failed: {e}")
        return 1
    
    # Show overall cost summary
    print("\n" + "=" * 50)
    print("💰 Cost Summary")
    print("=" * 50)
    
    cost_summary = adapter.get_cost_summary()
    print(f"Daily spending: ${cost_summary['daily_costs']:.6f}")
    print(f"Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
    print(f"Operations completed: {len(model_results) + len(session_results) + 1}")
    
    if cost_summary['daily_budget_utilization'] > 50:
        print("⚠️  High budget utilization - consider cost optimization")
    else:
        print("✅ Spending within comfortable limits")
    
    print("\n🎉 Basic tracking demonstration completed!")
    print("\n🚀 Next Steps:")
    print("   • Try cost_optimization.py for cost-aware model selection")
    print("   • Run advanced_features.py for multimodal and streaming")
    print("   • Explore production_patterns.py for enterprise patterns")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Try running setup_validation.py to check your configuration")
        sys.exit(1)