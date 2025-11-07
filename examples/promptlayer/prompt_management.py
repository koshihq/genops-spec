#!/usr/bin/env python3
"""
PromptLayer Advanced Prompt Management with GenOps Governance

This example demonstrates advanced prompt management features with GenOps governance,
including prompt versioning, cost optimization, and policy-driven prompt selection.

This is the Level 2 (30-minute) example - Advanced prompt governance with versioning.

Usage:
    python prompt_management.py

Prerequisites:
    pip install genops[promptlayer]  # Includes PromptLayer SDK
    export PROMPTLAYER_API_KEY="pl-your-api-key"
    export OPENAI_API_KEY="your-openai-key"  # For actual LLM calls
    
    # Required for governance features
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
"""

import os
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class PromptVersionMetrics:
    """Metrics for comparing prompt versions."""
    version: str
    avg_cost: float
    avg_latency_ms: float
    success_rate: float
    quality_score: float
    total_executions: int
    cost_per_quality_point: float

def advanced_prompt_versioning():
    """
    Demonstrates advanced prompt versioning with governance-driven selection.
    
    Shows how GenOps helps manage prompt versions based on cost, performance,
    and quality metrics while maintaining governance oversight.
    """
    print("📊 Advanced Prompt Versioning with Governance")
    print("=" * 50)
    
    try:
        from genops.providers.promptlayer import instrument_promptlayer
        print("✅ GenOps PromptLayer adapter loaded successfully")
        
        # Initialize with advanced governance policies
        adapter = instrument_promptlayer(
            promptlayer_api_key=os.getenv('PROMPTLAYER_API_KEY'),
            team=os.getenv('GENOPS_TEAM', 'ai-engineering'),
            project=os.getenv('GENOPS_PROJECT', 'prompt-optimization'),
            environment="production",
            enable_cost_alerts=True,
            max_operation_cost=0.50,  # $0.50 limit per operation
            daily_budget_limit=25.0,  # $25 daily limit
        )
        print("✅ Advanced governance policies configured")
        
    except ImportError as e:
        print(f"❌ Failed to import GenOps PromptLayer adapter: {e}")
        print("💡 Fix: Run 'pip install genops[promptlayer]'")
        return False
    
    print("\n🚀 Running Advanced Prompt Management Operations...")
    print("-" * 50)
    
    # Example 1: Intelligent prompt version selection
    print("\n1️⃣ Intelligent Prompt Version Selection")
    try:
        # Define multiple prompt versions with different characteristics
        prompt_versions = [
            {"version": "v1.0_concise", "complexity": "low", "expected_cost": 0.008},
            {"version": "v2.1_detailed", "complexity": "medium", "expected_cost": 0.015},
            {"version": "v3.0_premium", "complexity": "high", "expected_cost": 0.035}
        ]
        
        with adapter.track_prompt_operation(
            prompt_name="customer_support_suite",
            operation_type="version_selection",
            operation_name="intelligent_version_selection",
            tags={
                "feature": "smart_routing",
                "optimization_goal": "cost_quality_balance"
            }
        ) as span:
            
            # Simulate version selection based on governance policies
            selected_metrics = []
            
            for version_info in prompt_versions:
                version = version_info["version"]
                expected_cost = version_info["expected_cost"]
                
                # Check if version meets cost policies
                if expected_cost <= adapter.max_operation_cost:
                    # Simulate version execution with cost tracking
                    with adapter.track_prompt_operation(
                        prompt_name=f"customer_support_{version}",
                        prompt_version=version,
                        operation_type="prompt_run",
                        operation_name=f"version_test_{version}",
                        max_cost=expected_cost * 1.1  # 10% buffer
                    ) as version_span:
                        
                        # Simulate prompt execution
                        start_time = time.time()
                        
                        result = adapter.run_prompt_with_governance(
                            prompt_name=f"customer_support_{version}",
                            input_variables={
                                "customer_query": "I need help with billing",
                                "customer_tier": "premium",
                                "urgency": "medium"
                            },
                            tags=[f"version_{version}", "cost_optimization"]
                        )
                        
                        execution_time = (time.time() - start_time) * 1000
                        
                        # Update span with realistic metrics
                        version_span.update_cost(expected_cost)
                        version_span.update_token_usage(
                            input_tokens=45 + (len(version) * 5),
                            output_tokens=120 + (expected_cost * 1000),
                            model="gpt-3.5-turbo"
                        )
                        
                        # Calculate quality score (simulated)
                        quality_score = min(0.95, 0.7 + (expected_cost * 5))
                        
                        metrics = PromptVersionMetrics(
                            version=version,
                            avg_cost=expected_cost,
                            avg_latency_ms=execution_time,
                            success_rate=0.98,
                            quality_score=quality_score,
                            total_executions=1,
                            cost_per_quality_point=expected_cost / quality_score
                        )
                        
                        selected_metrics.append(metrics)
                        
                        print(f"   ✅ Version {version}: Cost ${expected_cost:.6f}, Quality {quality_score:.3f}, CPQ {metrics.cost_per_quality_point:.6f}")
                        
                else:
                    print(f"   ⚠️ Version {version}: Exceeds cost limit ${expected_cost:.6f} > ${adapter.max_operation_cost:.6f}")
            
            # Select optimal version based on cost-per-quality
            if selected_metrics:
                optimal_version = min(selected_metrics, key=lambda x: x.cost_per_quality_point)
                print(f"🏆 Optimal version selected: {optimal_version.version}")
                print(f"   💰 Cost efficiency: ${optimal_version.cost_per_quality_point:.6f} per quality point")
                
                span.add_attributes({
                    "selected_version": optimal_version.version,
                    "optimization_metric": "cost_per_quality_point",
                    "versions_evaluated": len(selected_metrics)
                })
            
    except Exception as e:
        print(f"❌ Version selection failed: {e}")
        return False
    
    # Example 2: Governance-driven A/B testing
    print("\n2️⃣ Governance-Driven A/B Testing")
    try:
        test_configurations = [
            {"variant": "control", "model": "gpt-3.5-turbo", "temperature": 0.7},
            {"variant": "experimental", "model": "gpt-4", "temperature": 0.5},
            {"variant": "cost_optimized", "model": "gpt-3.5-turbo", "temperature": 0.3}
        ]
        
        with adapter.track_prompt_operation(
            prompt_name="ab_test_suite",
            operation_type="ab_test",
            operation_name="governance_driven_testing",
            tags={
                "experiment": "model_comparison_q4",
                "optimization_goal": "cost_vs_quality"
            }
        ) as batch_span:
            
            test_results = []
            
            for config in test_configurations:
                variant = config["variant"]
                model = config["model"]
                
                with adapter.track_prompt_operation(
                    prompt_name=f"email_writer_{variant}",
                    prompt_version=f"ab_test_{variant}",
                    operation_type="prompt_run",
                    operation_name=f"ab_variant_{variant}",
                    tags={
                        "ab_variant": variant,
                        "model": model,
                        "test_group": "governance_comparison"
                    }
                ) as variant_span:
                    
                    # Simulate different costs based on model
                    if model == "gpt-4":
                        base_cost = 0.045
                        input_tokens = 50
                        output_tokens = 150
                    else:
                        base_cost = 0.012
                        input_tokens = 55
                        output_tokens = 140
                    
                    # Execute prompt with governance tracking
                    result = adapter.run_prompt_with_governance(
                        prompt_name=f"email_writer_{variant}",
                        input_variables={
                            "recipient": "valued customer",
                            "subject": "Important account update",
                            "key_points": ["Security enhancement", "New features", "Thank you"]
                        },
                        tags=[f"ab_test_{variant}", "model_comparison"]
                    )
                    
                    variant_span.update_cost(base_cost)
                    variant_span.update_token_usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=model
                    )
                    
                    # Simulate quality metrics
                    quality_score = 0.85 + (hash(variant) % 100) / 1000
                    user_satisfaction = 0.8 + (hash(model) % 150) / 1000
                    
                    test_results.append({
                        "variant": variant,
                        "model": model,
                        "cost": base_cost,
                        "quality_score": quality_score,
                        "user_satisfaction": user_satisfaction,
                        "cost_per_quality": base_cost / quality_score,
                        "governance_score": quality_score * user_satisfaction
                    })
                    
                    print(f"   📊 {variant}: Cost ${base_cost:.6f}, Quality {quality_score:.3f}, Satisfaction {user_satisfaction:.3f}")
            
            # Analyze results with governance lens
            best_overall = max(test_results, key=lambda x: x["governance_score"])
            most_cost_effective = min(test_results, key=lambda x: x["cost_per_quality"])
            
            print(f"\n🏆 A/B Test Results:")
            print(f"   Best Overall: {best_overall['variant']} (Governance Score: {best_overall['governance_score']:.3f})")
            print(f"   Most Cost-Effective: {most_cost_effective['variant']} (CPQ: ${most_cost_effective['cost_per_quality']:.6f})")
            
            # Governance recommendation
            if best_overall["variant"] == most_cost_effective["variant"]:
                print(f"   ✅ Recommendation: Deploy {best_overall['variant']} (optimal on both metrics)")
            else:
                cost_diff = abs(best_overall["cost"] - most_cost_effective["cost"])
                if cost_diff < 0.01:  # Less than 1 cent difference
                    print(f"   ✅ Recommendation: Deploy {best_overall['variant']} (minimal cost difference)")
                else:
                    print(f"   ⚖️ Trade-off Decision: {best_overall['variant']} (quality) vs {most_cost_effective['variant']} (cost)")
                    print(f"      Cost difference: ${cost_diff:.6f} per operation")
            
            batch_span.add_attributes({
                "best_variant": best_overall["variant"],
                "cost_effective_variant": most_cost_effective["variant"],
                "variants_tested": len(test_configurations),
                "total_test_cost": sum(r["cost"] for r in test_results)
            })
            
    except Exception as e:
        print(f"❌ A/B testing failed: {e}")
        return False
    
    # Example 3: Budget-constrained prompt selection
    print("\n3️⃣ Budget-Constrained Prompt Selection")
    try:
        # Scenario: Near daily budget limit, need to select cheaper prompts
        remaining_budget = 2.50  # $2.50 remaining for the day
        
        urgent_prompts = [
            {"name": "critical_alert", "estimated_cost": 0.08, "priority": "high"},
            {"name": "customer_escalation", "estimated_cost": 0.15, "priority": "critical"},
            {"name": "routine_notification", "estimated_cost": 0.02, "priority": "low"},
            {"name": "quality_summary", "estimated_cost": 0.12, "priority": "medium"}
        ]
        
        with adapter.track_prompt_operation(
            prompt_name="budget_optimization_suite",
            operation_type="budget_planning",
            operation_name="constrained_selection",
            tags={
                "budget_remaining": str(remaining_budget),
                "optimization_mode": "priority_cost_balance"
            }
        ) as planning_span:
            
            # Sort by priority and cost efficiency
            priority_scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            
            for prompt in urgent_prompts:
                prompt["priority_score"] = priority_scores[prompt["priority"]]
                prompt["value_per_dollar"] = prompt["priority_score"] / prompt["estimated_cost"]
            
            # Select prompts that fit within budget, prioritizing value
            selected_prompts = []
            current_cost = 0.0
            
            # Sort by value per dollar (descending)
            sorted_prompts = sorted(urgent_prompts, key=lambda x: x["value_per_dollar"], reverse=True)
            
            print(f"   💰 Budget constraint: ${remaining_budget:.2f} remaining")
            print(f"   🎯 Selecting prompts by value per dollar:")
            
            for prompt in sorted_prompts:
                if current_cost + prompt["estimated_cost"] <= remaining_budget:
                    selected_prompts.append(prompt)
                    current_cost += prompt["estimated_cost"]
                    
                    # Execute the selected prompt
                    with adapter.track_prompt_operation(
                        prompt_name=prompt["name"],
                        operation_type="budget_constrained_execution",
                        operation_name=f"execute_{prompt['name']}",
                        max_cost=prompt["estimated_cost"] * 1.05  # 5% buffer
                    ) as exec_span:
                        
                        result = adapter.run_prompt_with_governance(
                            prompt_name=prompt["name"],
                            input_variables={"urgency": prompt["priority"]},
                            tags=["budget_constrained", f"priority_{prompt['priority']}"]
                        )
                        
                        exec_span.update_cost(prompt["estimated_cost"])
                        
                        print(f"      ✅ {prompt['name']}: ${prompt['estimated_cost']:.6f} ({prompt['priority']} priority)")
                
                else:
                    print(f"      ⏭️ {prompt['name']}: Skipped (would exceed budget)")
            
            print(f"\n   📊 Budget Optimization Results:")
            print(f"      Selected: {len(selected_prompts)} prompts")
            print(f"      Total cost: ${current_cost:.6f}")
            print(f"      Budget utilization: {(current_cost/remaining_budget)*100:.1f}%")
            print(f"      Remaining budget: ${remaining_budget - current_cost:.6f}")
            
            planning_span.add_attributes({
                "prompts_considered": len(urgent_prompts),
                "prompts_selected": len(selected_prompts),
                "budget_utilization": current_cost / remaining_budget,
                "total_value_score": sum(p["priority_score"] for p in selected_prompts)
            })
            
    except Exception as e:
        print(f"❌ Budget optimization failed: {e}")
        return False
    
    return True

def demonstrate_prompt_lifecycle_management():
    """Demonstrate complete prompt lifecycle management with governance."""
    print("\n🔄 Prompt Lifecycle Management with Governance")
    print("-" * 45)
    
    try:
        from genops.providers.promptlayer import instrument_promptlayer
        
        adapter = instrument_promptlayer(
            team="prompt-engineering",
            project="lifecycle-management",
            environment="development"
        )
        
        # Simulate prompt lifecycle stages
        lifecycle_stages = [
            {"stage": "development", "cost_limit": 0.05, "governance": "advisory"},
            {"stage": "testing", "cost_limit": 0.10, "governance": "warning"},
            {"stage": "staging", "cost_limit": 0.20, "governance": "enforced"},
            {"stage": "production", "cost_limit": 0.15, "governance": "enforced"}
        ]
        
        print("🔄 Prompt Development Lifecycle:")
        
        for stage_info in lifecycle_stages:
            stage = stage_info["stage"]
            cost_limit = stage_info["cost_limit"]
            
            print(f"\n   📍 Stage: {stage.upper()}")
            print(f"      Cost limit: ${cost_limit:.6f}")
            print(f"      Governance: {stage_info['governance']}")
            
            with adapter.track_prompt_operation(
                prompt_name=f"email_assistant_{stage}",
                operation_type="lifecycle_stage",
                operation_name=f"stage_{stage}",
                tags={
                    "lifecycle_stage": stage,
                    "governance_mode": stage_info["governance"]
                },
                max_cost=cost_limit
            ) as stage_span:
                
                # Simulate stage-appropriate testing
                if stage == "development":
                    # Quick, cheap tests
                    test_cost = 0.02
                    print(f"      ✅ Quick validation tests: ${test_cost:.6f}")
                elif stage == "testing":
                    # More comprehensive testing
                    test_cost = 0.08
                    print(f"      ✅ Comprehensive testing suite: ${test_cost:.6f}")
                elif stage == "staging":
                    # Full integration testing
                    test_cost = 0.18
                    print(f"      ✅ Full integration tests: ${test_cost:.6f}")
                else:  # production
                    # Production validation
                    test_cost = 0.12
                    print(f"      ✅ Production validation: ${test_cost:.6f}")
                
                stage_span.update_cost(test_cost)
                
                if test_cost <= cost_limit:
                    print(f"      ✅ Stage passed governance policies")
                else:
                    print(f"      ❌ Stage exceeds cost limit (${test_cost:.6f} > ${cost_limit:.6f})")
        
        print(f"\n   🎯 Lifecycle Management Benefits:")
        print(f"      • Cost control at every development stage")
        print(f"      • Progressive governance enforcement")
        print(f"      • Automatic budget allocation by stage")
        print(f"      • Team accountability and attribution")
        
    except Exception as e:
        print(f"❌ Lifecycle management demo failed: {e}")

async def main():
    """Main execution function."""
    print("🚀 Starting PromptLayer Advanced Prompt Management Demo")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check prerequisites
    if not os.getenv('PROMPTLAYER_API_KEY'):
        print("❌ PROMPTLAYER_API_KEY not found")
        print("💡 Set your PromptLayer API key: export PROMPTLAYER_API_KEY='pl-your-key'")
        print("📖 Get your API key from: https://promptlayer.com/")
        return False
    
    # Run demonstrations
    success = True
    
    # Advanced prompt versioning and management
    if not advanced_prompt_versioning():
        success = False
    
    # Prompt lifecycle management
    if success:
        demonstrate_prompt_lifecycle_management()
    
    if success:
        print("\n" + "🌟" * 55)
        print("🎉 PromptLayer Advanced Prompt Management Demo Complete!")
        print("\n📊 What You've Mastered:")
        print("   ✅ Intelligent prompt version selection with cost optimization")
        print("   ✅ Governance-driven A/B testing with quality metrics")
        print("   ✅ Budget-constrained prompt selection and prioritization")
        print("   ✅ Complete prompt lifecycle management with stage-appropriate policies")
        
        print("\n🔍 Your Advanced Prompt Management Stack:")
        print("   • PromptLayer: Prompt versioning and management platform")
        print("   • GenOps: Advanced governance and cost optimization intelligence")
        print("   • OpenTelemetry: Comprehensive observability and metrics export")
        print("   • Multi-Model: Intelligent model selection and cost comparison")
        
        print("\n📚 Next Steps:")
        print("   • Explore evaluation workflows: python evaluation_integration.py")
        print("   • Advanced observability: python advanced_observability.py")
        print("   • Production deployment: python production_patterns.py")
        print("   • Run all examples: ./run_all_examples.sh")
        
        print("\n💡 Advanced Integration Patterns:")
        print("   ```python")
        print("   # Cost-optimized prompt selection")
        print("   with adapter.track_prompt_operation(max_cost=0.10) as span:")
        print("       best_version = select_optimal_prompt_version()")
        print("       result = adapter.run_prompt_with_governance(best_version)")
        print("   ```")
        
        print("🌟" * 55)
    else:
        print("\n❌ Demo encountered errors. Please check the output above.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)