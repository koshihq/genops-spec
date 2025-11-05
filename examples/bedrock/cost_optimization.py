#!/usr/bin/env python3
"""
Bedrock Advanced Cost Optimization Example

This example demonstrates advanced cost optimization strategies for AWS Bedrock
using GenOps cost intelligence, multi-model comparison, and budget-aware operations.

Example usage:
    python cost_optimization.py

Features demonstrated:
- Multi-model cost comparison and optimization
- Budget-aware operation strategies
- Regional cost optimization
- On-demand vs provisioned throughput analysis
- Advanced cost context management
- Real-time cost monitoring and alerts
"""

import sys
import os
import time

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def demonstrate_multi_model_optimization():
    """Demonstrate intelligent model selection for cost optimization."""
    
    print("🧠 Multi-Model Cost Optimization")
    print("=" * 40)
    print("GenOps automatically analyzes costs across models to find the best option:")
    print()
    
    try:
        from genops.providers.bedrock import GenOpsBedrockAdapter
        from genops.providers.bedrock_pricing import (
            compare_bedrock_models,
            get_cheapest_model_for_task,
            get_premium_model_for_task
        )
        
        adapter = GenOpsBedrockAdapter()
        
        # Test different task types for optimization
        task_scenarios = [
            {
                "task": "simple content generation",
                "input_tokens": 200,
                "output_tokens": 100,
                "description": "Blog post summarization"
            },
            {
                "task": "complex reasoning",
                "input_tokens": 1500,
                "output_tokens": 800,
                "description": "Technical analysis and recommendations"
            },
            {
                "task": "high volume processing",
                "input_tokens": 300,
                "output_tokens": 150,
                "description": "Customer inquiry responses (1000/day)"
            }
        ]
        
        for scenario in task_scenarios:
            print(f"📋 Scenario: {scenario['description']}")
            print(f"   Task type: {scenario['task']}")
            print(f"   Volume: {scenario['input_tokens']} → {scenario['output_tokens']} tokens")
            
            # Find optimal models for this task
            cheapest_model, cheapest_cost = get_cheapest_model_for_task(
                task_type=scenario['task'],
                input_tokens=scenario['input_tokens'],
                output_tokens=scenario['output_tokens']
            )
            
            premium_model, premium_cost = get_premium_model_for_task(
                task_type=scenario['task'],
                input_tokens=scenario['input_tokens'],
                output_tokens=scenario['output_tokens']
            )
            
            print(f"   💚 Most cost-effective: {cheapest_model} (${cheapest_cost:.6f})")
            if premium_model:
                print(f"   🏆 Premium option: {premium_model} (${premium_cost:.6f})")
                
                if cheapest_cost < premium_cost:
                    savings_per_op = premium_cost - cheapest_cost
                    print(f"   💰 Savings per operation: ${savings_per_op:.6f}")
                    
                    # Calculate volume savings
                    if "1000/day" in scenario['description']:
                        daily_savings = savings_per_op * 1000
                        monthly_savings = daily_savings * 30
                        print(f"   📊 Potential monthly savings: ${monthly_savings:.2f}")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization demo failed: {e}")
        return False


def demonstrate_budget_aware_operations():
    """Demonstrate budget-aware operation strategies."""
    
    print("💳 Budget-Aware Operations")
    print("=" * 30)
    print("GenOps can automatically enforce budget constraints and optimize costs:")
    print()
    
    try:
        from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context
        from genops.providers.bedrock import GenOpsBedrockAdapter
        
        # Example: Content generation with budget constraint
        print("📝 Content Generation with $0.05 Budget Limit:")
        
        with create_bedrock_cost_context(
            "budget_aware_content_generation",
            budget_limit=0.05,  # $0.05 budget
            alert_threshold=0.8  # Alert at 80% budget
        ) as cost_context:
            
            adapter = GenOpsBedrockAdapter()
            
            content_requests = [
                "Write a product description for a smart watch",
                "Create a social media post about sustainable technology",
                "Generate a brief company newsletter intro",
                "Write a customer service email template",
                "Create a technical blog post outline"
            ]
            
            total_operations = 0
            successful_operations = 0
            
            for i, request in enumerate(content_requests, 1):
                current_summary = cost_context.get_current_summary()
                remaining_budget = 0.05 - current_summary.total_cost
                
                print(f"   📝 Request {i}: Budget remaining ${remaining_budget:.4f}")
                
                if remaining_budget <= 0.001:  # Less than $0.001 remaining
                    print(f"   ⚠️  Budget exhausted, switching to cheapest model")
                    model = "amazon.titan-text-lite-v1"  # Cheapest available
                else:
                    model = "anthropic.claude-3-haiku-20240307-v1:0"  # Balanced option
                
                try:
                    # Simulate operation (we'll track costs manually here)
                    cost_context.add_operation(
                        operation_id=f"content_gen_{i}",
                        model_id=model,
                        provider="anthropic" if "claude" in model else "amazon",
                        region="us-east-1",
                        input_tokens=len(request) * 4,  # Rough estimate
                        output_tokens=80,
                        latency_ms=1500.0,
                        governance_attributes={
                            "team": "content-team",
                            "project": "marketing-automation",
                            "request_type": "content_generation"
                        }
                    )
                    
                    successful_operations += 1
                    print(f"      ✅ Generated content using {model}")
                    
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                
                total_operations += 1
            
            # Final budget analysis
            final_summary = cost_context.get_current_summary()
            print(f"\n   📊 Final Budget Analysis:")
            print(f"      Budget limit: $0.05")
            print(f"      Actual spend: ${final_summary.total_cost:.6f}")
            print(f"      Budget utilization: {(final_summary.total_cost/0.05)*100:.1f}%")
            print(f"      Operations completed: {successful_operations}/{total_operations}")
            print(f"      Average cost per operation: ${final_summary.get_average_cost_per_operation():.6f}")
            
            if final_summary.optimization_recommendations:
                print("   💡 Optimization recommendations:")
                for rec in final_summary.optimization_recommendations:
                    print(f"      • {rec}")
        
        print()
        
    except Exception as e:
        print(f"❌ Budget-aware demo failed: {e}")


def demonstrate_regional_optimization():
    """Demonstrate regional cost optimization."""
    
    print("🌍 Regional Cost Optimization")
    print("=" * 35)
    print("GenOps compares costs across AWS regions to find savings:")
    print()
    
    try:
        from genops.providers.bedrock_pricing import (
            calculate_bedrock_cost,
            REGIONAL_MULTIPLIERS
        )
        
        # Test model across different regions
        test_model = "anthropic.claude-3-haiku-20240307-v1:0"
        test_tokens_in = 1000
        test_tokens_out = 500
        
        print(f"💰 Cost comparison for {test_model}:")
        print(f"   Input: {test_tokens_in} tokens, Output: {test_tokens_out} tokens")
        print()
        
        regional_costs = []
        
        for region, multiplier in REGIONAL_MULTIPLIERS.items():
            cost = calculate_bedrock_cost(
                model_id=test_model,
                input_tokens=test_tokens_in,
                output_tokens=test_tokens_out,
                region=region
            )
            regional_costs.append((region, cost, multiplier))
        
        # Sort by cost
        regional_costs.sort(key=lambda x: x[1])
        
        cheapest_region, cheapest_cost, _ = regional_costs[0]
        most_expensive_region, most_expensive_cost, _ = regional_costs[-1]
        
        print("   🏆 Regional cost ranking:")
        for i, (region, cost, multiplier) in enumerate(regional_costs, 1):
            emoji = "💚" if i == 1 else "💛" if i <= 3 else "💰"
            print(f"      {emoji} {i}. {region}: ${cost:.6f} (multiplier: {multiplier:.2f})")
        
        savings = most_expensive_cost - cheapest_cost
        percentage_savings = (savings / most_expensive_cost) * 100
        
        print(f"\n   📈 Optimization opportunity:")
        print(f"      Best region: {cheapest_region} (${cheapest_cost:.6f})")
        print(f"      Most expensive: {most_expensive_region} (${most_expensive_cost:.6f})")
        print(f"      Potential savings: ${savings:.6f} per operation ({percentage_savings:.1f}%)")
        
        # High-volume impact
        monthly_operations = 10000
        monthly_savings = savings * monthly_operations
        print(f"      Monthly savings (10K ops): ${monthly_savings:.2f}")
        
        print()
        
    except Exception as e:
        print(f"❌ Regional optimization demo failed: {e}")


def demonstrate_provisioned_vs_ondemand():
    """Demonstrate on-demand vs provisioned throughput analysis."""
    
    print("⚡ On-Demand vs Provisioned Throughput Analysis")
    print("=" * 50)
    print("GenOps analyzes when provisioned throughput becomes cost-effective:")
    print()
    
    try:
        from genops.providers.bedrock_pricing import calculate_provisioned_vs_ondemand
        
        # Test scenarios with different usage levels
        usage_scenarios = [
            {"operations": 1000, "description": "Low usage (1K ops/month)"},
            {"operations": 10000, "description": "Medium usage (10K ops/month)"},
            {"operations": 100000, "description": "High usage (100K ops/month)"},
            {"operations": 1000000, "description": "Enterprise usage (1M ops/month)"}
        ]
        
        test_model = "anthropic.claude-3-haiku-20240307-v1:0"
        avg_input_tokens = 500
        avg_output_tokens = 200
        
        print(f"📊 Analysis for {test_model}:")
        print(f"   Average: {avg_input_tokens} input → {avg_output_tokens} output tokens")
        print()
        
        for scenario in usage_scenarios:
            ops = scenario["operations"]
            desc = scenario["description"]
            
            analysis = calculate_provisioned_vs_ondemand(
                model_id=test_model,
                monthly_operations=ops,
                avg_input_tokens=avg_input_tokens,
                avg_output_tokens=avg_output_tokens
            )
            
            print(f"   💼 {desc}:")
            print(f"      On-demand cost: ${analysis['ondemand_monthly']:.2f}/month")
            
            if analysis['provisioned_available']:
                print(f"      Provisioned cost: ${analysis['provisioned_monthly']:.2f}/month")
                savings = analysis['monthly_savings']
                if savings > 0:
                    print(f"      💚 Savings with provisioned: ${savings:.2f}/month")
                else:
                    print(f"      💛 On-demand cheaper by: ${abs(savings):.2f}/month")
                print(f"      Break-even point: {analysis['breakeven_operations']:,.0f} ops/month")
            else:
                print(f"      ⚠️  Provisioned throughput not available for this model")
            
            print(f"      Recommendation: {analysis['recommendation']}")
            print()
        
    except Exception as e:
        print(f"❌ Provisioned throughput analysis failed: {e}")


def demonstrate_real_time_cost_monitoring():
    """Demonstrate real-time cost monitoring during operations."""
    
    print("📊 Real-Time Cost Monitoring")
    print("=" * 35)
    print("GenOps provides real-time cost tracking with alerts:")
    print()
    
    try:
        from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context
        from genops.providers.bedrock import GenOpsBedrockAdapter
        
        print("🔄 Simulating batch processing with cost monitoring...")
        
        with create_bedrock_cost_context(
            "real_time_monitoring_demo",
            budget_limit=0.10,  # $0.10 budget
            alert_threshold=0.5,  # Alert at 50%
            enable_optimization_recommendations=True
        ) as cost_context:
            
            batch_tasks = [
                {"task": "Email classification", "model": "anthropic.claude-3-haiku-20240307-v1:0"},
                {"task": "Sentiment analysis", "model": "amazon.titan-text-express-v1"},
                {"task": "Content moderation", "model": "anthropic.claude-3-haiku-20240307-v1:0"},
                {"task": "Text summarization", "model": "ai21.j2-mid-v1"},
                {"task": "Language translation", "model": "anthropic.claude-3-haiku-20240307-v1:0"}
            ]
            
            for i, task in enumerate(batch_tasks, 1):
                # Simulate processing
                cost_context.add_operation(
                    operation_id=f"batch_op_{i}",
                    model_id=task["model"],
                    provider="anthropic" if "claude" in task["model"] else "amazon" if "titan" in task["model"] else "ai21",
                    region="us-east-1",
                    input_tokens=300 + (i * 50),  # Varying input sizes
                    output_tokens=150 + (i * 25),  # Varying output sizes
                    latency_ms=1200 + (i * 200),
                    governance_attributes={
                        "team": "batch-processing",
                        "task_type": task["task"]
                    }
                )
                
                current_summary = cost_context.get_current_summary()
                budget_used = (current_summary.total_cost / 0.10) * 100
                
                print(f"   📝 Task {i}: {task['task']}")
                print(f"      Model: {task['model']}")
                print(f"      Running cost: ${current_summary.total_cost:.6f}")
                print(f"      Budget used: {budget_used:.1f}%")
                
                # Show real-time recommendations
                if current_summary.optimization_recommendations:
                    print(f"      💡 Recommendation: {current_summary.optimization_recommendations[0]}")
                
                print()
                
                # Simulate processing time
                time.sleep(0.5)
            
            # Final analysis
            final_summary = cost_context.get_current_summary()
            print("🎯 Final Monitoring Results:")
            print(f"   Total operations: {final_summary.total_operations}")
            print(f"   Total cost: ${final_summary.total_cost:.6f}")
            print(f"   Models used: {len(final_summary.unique_models)}")
            print(f"   Average cost per operation: ${final_summary.get_average_cost_per_operation():.6f}")
            
            # Cost breakdown
            print("\n   📊 Cost breakdown by model:")
            for model, cost in final_summary.cost_by_model.items():
                percentage = (cost / final_summary.total_cost) * 100
                print(f"      {model}: ${cost:.6f} ({percentage:.1f}%)")
            
            # Export detailed report
            report = cost_context.export_cost_report(format="summary")
            print(f"\n   📋 Detailed Report Available:")
            print(f"      Export formats: JSON, CSV, Summary")
            print(f"      Report length: {len(report.split())} words")
        
        print()
        
    except Exception as e:
        print(f"❌ Real-time monitoring demo failed: {e}")


def main():
    """Main demonstration function."""
    
    print("Welcome to GenOps Bedrock Advanced Cost Optimization!")
    print()
    print("This example demonstrates intelligent cost optimization strategies")
    print("for AWS Bedrock using GenOps cost intelligence and analytics.")
    print()
    
    demos = [
        ("Multi-Model Optimization", demonstrate_multi_model_optimization),
        ("Budget-Aware Operations", demonstrate_budget_aware_operations),
        ("Regional Optimization", demonstrate_regional_optimization),
        ("Provisioned vs On-Demand", demonstrate_provisioned_vs_ondemand),
        ("Real-Time Monitoring", demonstrate_real_time_cost_monitoring)
    ]
    
    success_count = 0
    
    for demo_name, demo_func in demos:
        print(f"🚀 {demo_name} Demo")
        print("=" * (len(demo_name) + 7))
        
        try:
            result = demo_func()
            if result is not False:
                success_count += 1
                print(f"✅ {demo_name} completed successfully\n")
            else:
                print(f"⚠️ {demo_name} had issues\n")
        except Exception as e:
            print(f"❌ {demo_name} failed: {e}\n")
    
    # Summary
    print("🎉 Cost Optimization Demo Summary")
    print("=" * 40)
    print(f"Completed: {success_count}/{len(demos)} demonstrations")
    print()
    
    if success_count >= 3:
        print("🏆 Key Cost Optimization Features Demonstrated:")
        print("   💰 Multi-model cost comparison and selection")
        print("   📊 Budget-aware operation strategies")
        print("   🌍 Regional cost optimization analysis")
        print("   ⚡ On-demand vs provisioned throughput comparison")
        print("   📈 Real-time cost monitoring with alerts")
        print()
        print("💡 Next Steps:")
        print("   → Production: python production_patterns.py")
        print("   → Enterprise: python lambda_integration.py")
        print("   → Monitoring: Set up dashboards with exported cost data")
        print("   → Budgeting: Implement budget alerts in your workflows")
    
    return success_count >= len(demos) // 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)