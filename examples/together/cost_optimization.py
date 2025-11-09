#!/usr/bin/env python3
"""
Together AI Cost Optimization with GenOps

Demonstrates intelligent cost optimization across Together AI's 200+ models.
Shows how to minimize costs while maintaining quality through smart model selection.

Usage:
    python cost_optimization.py

Features:
    - Multi-model cost comparison and analysis
    - Task-complexity based model recommendations
    - Budget-constrained operations with automatic fallbacks
    - Cost projection and savings analysis
    - Real-time cost optimization strategies
"""

import os
import sys
from decimal import Decimal
from typing import List, Dict, Any

try:
    from genops.providers.together import GenOpsTogetherAdapter, TogetherModel
    from genops.providers.together_pricing import TogetherPricingCalculator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install genops-ai[together]")
    print("Then run: python setup_validation.py")
    sys.exit(1)


class CostOptimizer:
    """Intelligent cost optimization for Together AI operations."""
    
    def __init__(self, adapter: GenOpsTogetherAdapter):
        self.adapter = adapter
        self.pricing_calc = TogetherPricingCalculator()
    
    def find_cheapest_model_for_task(
        self,
        task_type: str,
        max_budget: float = 0.001,
        min_context_length: int = 8192
    ) -> Dict[str, Any]:
        """Find the most cost-effective model for a specific task type."""
        recommendation = self.pricing_calc.recommend_model(
            task_complexity=task_type,
            budget_per_operation=max_budget,
            min_context_length=min_context_length
        )
        
        return recommendation
    
    def compare_model_performance_costs(
        self,
        models: List[str],
        test_prompt: str,
        max_tokens: int = 100
    ) -> List[Dict[str, Any]]:
        """Compare actual performance vs costs across models."""
        results = []
        
        for model in models:
            try:
                with self.adapter.track_session(f"cost-comparison-{model}") as session:
                    result = self.adapter.chat_with_governance(
                        messages=[{"role": "user", "content": test_prompt}],
                        model=model,
                        max_tokens=max_tokens,
                        temperature=0.5,
                        session_id=session.session_id,
                        comparison_type="cost-optimization"
                    )
                    
                    results.append({
                        'model': model,
                        'cost': float(result.cost),
                        'tokens_used': result.tokens_used,
                        'execution_time': result.execution_time_seconds,
                        'cost_per_token': float(result.cost) / result.tokens_used if result.tokens_used > 0 else 0,
                        'tokens_per_second': result.tokens_used / result.execution_time_seconds if result.execution_time_seconds > 0 else 0,
                        'response_length': len(result.response),
                        'response': result.response
                    })
            
            except Exception as e:
                print(f"   ❌ Failed to test {model}: {e}")
                continue
        
        # Sort by cost-effectiveness (cost per token)
        return sorted(results, key=lambda x: x['cost_per_token'])


def demonstrate_cost_comparison():
    """Compare costs across different model tiers."""
    print("💰 Multi-Model Cost Comparison")
    print("=" * 50)
    
    adapter = GenOpsTogetherAdapter(
        team="cost-optimization",
        project="model-comparison",
        environment="development",
        daily_budget_limit=10.0,
        governance_policy="advisory"
    )
    
    # Models across different price tiers
    models_to_compare = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",    # Lite tier
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",       # Lite tier, reasoning
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",   # Standard tier
        "mistralai/Mixtral-8x7B-Instruct-v0.1",           # Standard tier
    ]
    
    print(f"🧪 Testing {len(models_to_compare)} models for cost-effectiveness...")
    
    optimizer = CostOptimizer(adapter)
    test_prompt = "Explain the concept of machine learning in simple terms suitable for beginners."
    
    results = optimizer.compare_model_performance_costs(
        models=models_to_compare,
        test_prompt=test_prompt,
        max_tokens=120
    )
    
    if results:
        print(f"\n📊 Cost Comparison Results (sorted by cost-effectiveness):")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['model']}")
            print(f"   Cost: ${result['cost']:.6f}")
            print(f"   Tokens: {result['tokens_used']}")
            print(f"   Time: {result['execution_time']:.2f}s")
            print(f"   Cost/token: ${result['cost_per_token']:.8f}")
            print(f"   Speed: {result['tokens_per_second']:.1f} tokens/s")
            print(f"   Response quality: {result['response_length']} chars")
            print()
        
        # Calculate savings potential
        cheapest = results[0]
        most_expensive = results[-1]
        savings_per_operation = most_expensive['cost'] - cheapest['cost']
        
        print(f"💡 Optimization Insights:")
        print(f"   Most cost-effective: {cheapest['model']}")
        print(f"   Potential savings: ${savings_per_operation:.6f} per operation")
        print(f"   For 1000 operations: ${savings_per_operation * 1000:.2f} savings")
        
        return adapter, results[0]['model']  # Return cheapest model
    
    return adapter, None


def demonstrate_task_based_optimization():
    """Show how different tasks require different optimization strategies."""
    print("\n🎯 Task-Based Model Optimization")
    print("=" * 50)
    
    adapter = GenOpsTogetherAdapter(
        team="task-optimization",
        project="smart-selection",
        environment="development",
        daily_budget_limit=15.0,
        governance_policy="advisory"
    )
    
    pricing_calc = TogetherPricingCalculator()
    
    # Define different task complexities with different requirements
    tasks = {
        "simple": {
            "description": "Simple Q&A, basic assistance",
            "example": "What is the capital of France?",
            "max_budget": 0.0005,  # Very low budget
            "requirements": {"min_context_length": 4096}
        },
        "moderate": {
            "description": "Analysis, explanation, code review",
            "example": "Analyze the pros and cons of microservices architecture",
            "max_budget": 0.002,   # Medium budget
            "requirements": {"min_context_length": 16384}
        },
        "complex": {
            "description": "Complex reasoning, advanced coding",
            "example": "Design a distributed system for real-time data processing with fault tolerance",
            "max_budget": 0.01,    # Higher budget for complex tasks
            "requirements": {"min_context_length": 32768}
        }
    }
    
    print("🧠 Finding optimal models for different task complexities:")
    
    task_results = {}
    
    for task_type, task_info in tasks.items():
        print(f"\n📋 {task_type.upper()} Task: {task_info['description']}")
        
        # Get model recommendation
        recommendation = pricing_calc.recommend_model(
            task_complexity=task_type,
            budget_per_operation=task_info['max_budget'],
            **task_info['requirements']
        )
        
        if recommendation['recommended_model']:
            print(f"   🎯 Recommended: {recommendation['recommended_model']}")
            print(f"   💰 Estimated cost: ${recommendation['estimated_cost']:.6f}")
            print(f"   📏 Context length: {recommendation['context_length']:,} tokens")
            print(f"   ✅ Budget compliant: {recommendation['budget_compliant']}")
            
            # Test the recommendation
            try:
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": task_info['example']}],
                    model=recommendation['recommended_model'],
                    max_tokens=150,
                    temperature=0.7,
                    task_complexity=task_type,
                    optimization_target="cost-effectiveness"
                )
                
                task_results[task_type] = {
                    'model': result.model_used,
                    'actual_cost': float(result.cost),
                    'estimated_cost': recommendation['estimated_cost'],
                    'cost_accuracy': abs(float(result.cost) - recommendation['estimated_cost']),
                    'response_quality': len(result.response)
                }
                
                print(f"   ✅ Actual cost: ${result.cost:.6f}")
                print(f"   📊 Cost estimation accuracy: ±${abs(float(result.cost) - recommendation['estimated_cost']):.6f}")
                
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
        else:
            print(f"   ❌ No suitable model found within budget")
    
    # Summary of task-based optimization
    if task_results:
        print(f"\n📊 Task Optimization Summary:")
        total_cost = sum(tr['actual_cost'] for tr in task_results.values())
        avg_accuracy = sum(tr['cost_accuracy'] for tr in task_results.values()) / len(task_results)
        
        print(f"   Total cost for all task types: ${total_cost:.6f}")
        print(f"   Average cost estimation accuracy: ±${avg_accuracy:.6f}")
        print(f"   Models used: {len(set(tr['model'] for tr in task_results.values()))}")


def demonstrate_budget_constrained_operations():
    """Show how to operate within strict budget constraints."""
    print("\n💸 Budget-Constrained Operations")
    print("=" * 50)
    
    # Create adapter with very tight budget
    adapter = GenOpsTogetherAdapter(
        team="budget-conscious",
        project="cost-control-demo",
        environment="development",
        daily_budget_limit=2.0,  # Only $2 per day
        governance_policy="enforced",  # Strict budget enforcement
        enable_cost_alerts=True
    )
    
    print(f"💰 Operating with strict ${adapter.daily_budget_limit} daily budget")
    
    pricing_calc = TogetherPricingCalculator()
    
    # Find the absolute cheapest models
    all_models = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-Coder-V2-Instruct"
    ]
    
    comparisons = pricing_calc.compare_models(all_models, estimated_tokens=500)
    print(f"\n📊 Cheapest models (500 tokens):")
    
    for i, comp in enumerate(comparisons[:3], 1):
        print(f"   {i}. {comp['model']}")
        print(f"      Cost: ${comp['estimated_cost']:.6f}")
        print(f"      Tier: {comp['tier']}")
    
    # Use the cheapest model for maximum operations within budget
    cheapest_model = comparisons[0]['model']
    operations_possible = int(adapter.daily_budget_limit / comparisons[0]['estimated_cost'])
    
    print(f"\n🎯 Budget Strategy:")
    print(f"   Using cheapest model: {cheapest_model}")
    print(f"   Estimated operations possible: {operations_possible}")
    
    # Simulate several operations
    print(f"\n🚀 Executing budget-optimized operations:")
    
    operations_completed = 0
    total_actual_cost = Decimal('0')
    
    test_queries = [
        "What is AI?",
        "Explain neural networks briefly",
        "Benefits of open source software",
        "How does cloud computing work?",
        "What is machine learning?"
    ]
    
    with adapter.track_session("budget-optimization") as session:
        for i, query in enumerate(test_queries[:operations_possible], 1):
            try:
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": query}],
                    model=cheapest_model,
                    max_tokens=50,  # Keep tokens low for cost control
                    session_id=session.session_id,
                    budget_optimization=True,
                    operation_index=i
                )
                
                operations_completed += 1
                total_actual_cost += result.cost
                
                print(f"   ✅ Operation {i}: ${result.cost:.6f}")
                
                # Check if we're approaching budget limits
                cost_summary = adapter.get_cost_summary()
                if cost_summary['daily_budget_utilization'] > 80:
                    print(f"   ⚠️  Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Operation {i} failed: {e}")
                break
    
    # Final budget analysis
    cost_summary = adapter.get_cost_summary()
    
    print(f"\n📊 Budget Performance:")
    print(f"   Operations completed: {operations_completed}")
    print(f"   Total cost: ${cost_summary['daily_costs']:.6f}")
    print(f"   Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
    print(f"   Average cost/operation: ${total_actual_cost / operations_completed:.6f}")
    print(f"   Remaining budget: ${adapter.daily_budget_limit - cost_summary['daily_costs']:.6f}")


def demonstrate_cost_projection_analysis():
    """Show cost projection and analysis for planning purposes."""
    print("\n📈 Cost Projection & Analysis")
    print("=" * 50)
    
    pricing_calc = TogetherPricingCalculator()
    
    # Analyze different usage patterns
    usage_scenarios = [
        {
            "name": "Light Usage",
            "operations_per_day": 100,
            "avg_tokens": 300,
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        },
        {
            "name": "Medium Usage",
            "operations_per_day": 1000,
            "avg_tokens": 500,
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        },
        {
            "name": "Heavy Usage",
            "operations_per_day": 5000,
            "avg_tokens": 800,
            "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        }
    ]
    
    print("🔮 Cost projections for different usage patterns:")
    print("-" * 80)
    
    for scenario in usage_scenarios:
        analysis = pricing_calc.analyze_costs(
            operations_per_day=scenario['operations_per_day'],
            avg_tokens_per_operation=scenario['avg_tokens'],
            model=scenario['model'],
            days_to_analyze=30
        )
        
        print(f"📋 {scenario['name']}:")
        print(f"   Model: {scenario['model']}")
        print(f"   Daily operations: {scenario['operations_per_day']:,}")
        print(f"   Daily cost: ${analysis['daily_cost']:.2f}")
        print(f"   Monthly cost: ${analysis['monthly_cost']:.2f}")
        print(f"   Yearly cost: ${analysis['yearly_cost']:.2f}")
        
        # Show potential savings
        if analysis['potential_savings']['best_alternative']:
            alt = analysis['potential_savings']['best_alternative']
            print(f"   💡 Alternative: {alt['model']}")
            print(f"   Monthly savings: ${analysis['potential_savings']['potential_monthly_savings']:.2f}")
        
        print()


def main():
    """Run comprehensive cost optimization demonstrations."""
    print("💰 Together AI Cost Optimization with GenOps")
    print("=" * 60)
    
    try:
        # Run all optimization demonstrations
        adapter, cheapest_model = demonstrate_cost_comparison()
        demonstrate_task_based_optimization()
        demonstrate_budget_constrained_operations()
        demonstrate_cost_projection_analysis()
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎯 Cost Optimization Summary")
        print("=" * 60)
        
        if adapter:
            cost_summary = adapter.get_cost_summary()
            print("✅ Optimization strategies demonstrated:")
            print(f"   • Multi-model comparison completed")
            print(f"   • Task-based optimization configured")
            print(f"   • Budget constraints successfully managed")
            print(f"   • Cost projections analyzed")
            
            print(f"\n💰 Session Totals:")
            print(f"   Total spending: ${cost_summary['daily_costs']:.6f}")
            print(f"   Models tested: Multiple across all price tiers")
            print(f"   Optimization focus: Cost-effectiveness and budget control")
        
        print("\n🚀 Key Takeaways:")
        print("   ✅ Lite tier models (8B) offer excellent cost-performance ratio")
        print("   ✅ Task complexity should drive model selection")
        print("   ✅ Budget constraints can be strictly enforced")
        print("   ✅ Cost projections help with planning and budgeting")
        print("   ✅ Automatic model recommendations save time and money")
        
        print("\n📚 Next Steps:")
        print("   • Set up budget alerts for your production workloads")
        print("   • Use task-complexity based model selection")
        print("   • Monitor cost-per-operation metrics")
        print("   • Consider lite tier models for high-volume operations")
        
        return 0
        
    except Exception as e:
        print(f"❌ Cost optimization demo failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(1)