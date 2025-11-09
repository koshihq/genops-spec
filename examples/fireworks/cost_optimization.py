#!/usr/bin/env python3
"""
Fireworks AI Cost Optimization with GenOps

Demonstrates intelligent cost optimization across Fireworks AI's 100+ models.
Shows how to minimize costs while maintaining quality through smart model selection,
batch processing, and performance optimization.

Usage:
    python cost_optimization.py

Features:
    - Multi-model cost comparison and analysis across pricing tiers
    - Task-complexity based model recommendations
    - Budget-constrained operations with automatic fallbacks
    - Batch processing optimization with 50% savings
    - Cost projection and savings analysis
    - Real-time cost optimization strategies
"""

import os
import sys
from decimal import Decimal
from typing import List, Dict, Any

try:
    from genops.providers.fireworks import GenOpsFireworksAdapter, FireworksModel
    from genops.providers.fireworks_pricing import FireworksPricingCalculator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install genops-ai[fireworks]")
    print("Then run: python setup_validation.py")
    sys.exit(1)


class FireworksCostOptimizer:
    """Intelligent cost optimization for Fireworks AI operations."""
    
    def __init__(self, adapter: GenOpsFireworksAdapter):
        self.adapter = adapter
        self.pricing_calc = FireworksPricingCalculator()
    
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
        
        return {
            "recommended_model": recommendation.recommended_model,
            "estimated_cost": float(recommendation.estimated_cost),
            "reasoning": recommendation.reasoning,
            "alternatives": recommendation.alternatives[:3]  # Top 3 alternatives
        }
    
    def compare_batch_vs_standard_pricing(
        self,
        model: FireworksModel,
        operations_count: int,
        avg_tokens: int = 500
    ) -> Dict[str, Any]:
        """Compare standard vs batch pricing for a workload."""
        standard_cost_per_op = self.pricing_calc.estimate_chat_cost(
            model.value, tokens=avg_tokens, is_batch=False
        )
        
        batch_cost_per_op = self.pricing_calc.estimate_chat_cost(
            model.value, tokens=avg_tokens, is_batch=True
        )
        
        standard_total = float(standard_cost_per_op) * operations_count
        batch_total = float(batch_cost_per_op) * operations_count
        savings = standard_total - batch_total
        
        return {
            "model": model.value.split('/')[-1],
            "operations": operations_count,
            "standard_cost_per_op": float(standard_cost_per_op),
            "batch_cost_per_op": float(batch_cost_per_op),
            "standard_total": standard_total,
            "batch_total": batch_total,
            "savings": savings,
            "savings_percentage": (savings / standard_total) * 100 if standard_total > 0 else 0
        }
    
    def optimize_model_selection_for_budget(
        self,
        prompts: List[str],
        total_budget: float,
        prefer_quality: bool = False
    ) -> List[Dict[str, Any]]:
        """Optimize model selection to fit within budget."""
        results = []
        remaining_budget = total_budget
        
        for i, prompt in enumerate(prompts):
            # Estimate complexity based on prompt length and content
            complexity = self._estimate_prompt_complexity(prompt)
            
            # Find best model within remaining budget
            budget_per_op = remaining_budget / (len(prompts) - i)
            
            recommendation = self.pricing_calc.recommend_model(
                task_complexity=complexity,
                budget_per_operation=budget_per_op,
                prefer_batch=True  # Always consider batch savings
            )
            
            if recommendation.recommended_model:
                model = recommendation.recommended_model
                estimated_cost = float(recommendation.estimated_cost)
                
                results.append({
                    "prompt_index": i,
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "complexity": complexity,
                    "model": model.split('/')[-1],
                    "estimated_cost": estimated_cost,
                    "remaining_budget": remaining_budget
                })
                
                remaining_budget -= estimated_cost
            else:
                results.append({
                    "prompt_index": i,
                    "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                    "complexity": complexity,
                    "model": "BUDGET_EXCEEDED",
                    "estimated_cost": 0,
                    "remaining_budget": remaining_budget
                })
        
        return results
    
    def _estimate_prompt_complexity(self, prompt: str) -> str:
        """Estimate task complexity based on prompt characteristics."""
        prompt_lower = prompt.lower()
        
        # Complex indicators
        complex_indicators = [
            "analyze", "compare", "detailed", "comprehensive", "explain in depth",
            "reasoning", "complex", "sophisticated", "nuanced"
        ]
        
        # Simple indicators  
        simple_indicators = [
            "summarize", "list", "what is", "define", "yes/no", "true/false",
            "quick", "brief", "simple"
        ]
        
        if any(indicator in prompt_lower for indicator in complex_indicators):
            return "complex"
        elif any(indicator in prompt_lower for indicator in simple_indicators):
            return "simple"
        else:
            return "moderate"


def main():
    """Demonstrate comprehensive cost optimization strategies."""
    print("💰 Fireworks AI Cost Optimization with GenOps")
    print("=" * 60)
    
    # Initialize cost-optimized adapter
    adapter = GenOpsFireworksAdapter(
        team=os.getenv('GENOPS_TEAM', 'cost-optimization-team'),
        project=os.getenv('GENOPS_PROJECT', 'cost-optimization'),
        environment=os.getenv('GENOPS_ENVIRONMENT', 'development'),
        daily_budget_limit=25.0,  # Conservative budget for optimization demo
        governance_policy='advisory',
        enable_cost_alerts=True,
        auto_optimize_costs=True
    )
    
    optimizer = FireworksCostOptimizer(adapter)
    
    print("✅ Cost optimizer initialized")
    print(f"   Daily budget: ${adapter.daily_budget_limit}")
    print("   Focus: Maximizing value while minimizing cost")
    
    # Example 1: Model comparison across pricing tiers
    print("\n" + "=" * 60)
    print("🔬 Example 1: Cross-Tier Model Cost Analysis")
    print("=" * 60)
    
    models_to_compare = [
        FireworksModel.LLAMA_3_2_1B_INSTRUCT,    # $0.10/M (Tiny)
        FireworksModel.LLAMA_3_1_8B_INSTRUCT,    # $0.20/M (Small)
        FireworksModel.LLAMA_3_1_70B_INSTRUCT,   # $0.90/M (Large)
        FireworksModel.MIXTRAL_8X7B,             # $0.50/M (MoE)
    ]
    
    test_prompt = "Explain the benefits of cost-optimized AI inference in business applications."
    
    print("Testing prompt:", test_prompt[:60] + "...")
    print("\n💰 Cost Analysis by Model:")
    
    model_results = []
    for model in models_to_compare:
        try:
            result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": test_prompt}],
                model=model,
                max_tokens=120,
                temperature=0.5,
                feature="cost-comparison",
                comparison_batch="tier-analysis"
            )
            
            model_results.append(result)
            
            # Calculate cost efficiency (response quality vs cost)
            quality_score = len(result.response.split()) / 120  # Words per max token
            efficiency = quality_score / float(result.cost) if result.cost > 0 else 0
            
            print(f"\n   🧠 {model.value.split('/')[-1]}:")
            print(f"      Cost: ${result.cost:.6f}")
            print(f"      Speed: {result.execution_time_seconds:.2f}s")
            print(f"      Words: {len(result.response.split())}")
            print(f"      Efficiency: {efficiency:.0f} words/$")
            
        except Exception as e:
            print(f"   ❌ {model.value.split('/')[-1]} failed: {e}")
    
    # Find most cost-effective
    if model_results:
        best_value = max(model_results, key=lambda x: len(x.response.split()) / float(x.cost) if x.cost > 0 else 0)
        print(f"\n🏆 Best value: {best_value.model_used.split('/')[-1]} (${best_value.cost:.6f})")
    
    # Example 2: Batch processing optimization
    print("\n" + "=" * 60)
    print("📦 Example 2: Batch Processing Cost Savings")
    print("=" * 60)
    
    # Test batch savings across different models
    batch_test_models = [
        FireworksModel.LLAMA_3_1_8B_INSTRUCT,
        FireworksModel.LLAMA_3_1_70B_INSTRUCT,
        FireworksModel.MIXTRAL_8X7B
    ]
    
    operations_count = 100
    print(f"Analyzing batch savings for {operations_count} operations:")
    
    for model in batch_test_models:
        batch_analysis = optimizer.compare_batch_vs_standard_pricing(
            model, operations_count, avg_tokens=500
        )
        
        print(f"\n   🔥 {batch_analysis['model']}:")
        print(f"      Standard: ${batch_analysis['standard_total']:.2f}")
        print(f"      Batch: ${batch_analysis['batch_total']:.2f}")
        print(f"      Savings: ${batch_analysis['savings']:.2f} ({batch_analysis['savings_percentage']:.0f}%)")
    
    # Example 3: Task-complexity based optimization
    print("\n" + "=" * 60)
    print("🎯 Example 3: Task-Complexity Based Optimization")
    print("=" * 60)
    
    task_examples = [
        ("What is 2+2?", "simple"),
        ("Explain machine learning concepts for beginners", "moderate"),
        ("Conduct a detailed competitive analysis of AI inference providers", "complex")
    ]
    
    print("Finding optimal models for different task complexities:")
    
    for prompt, expected_complexity in task_examples:
        recommendation = optimizer.find_cheapest_model_for_task(
            expected_complexity,
            max_budget=0.01,  # $0.01 budget per task
            min_context_length=4096
        )
        
        print(f"\n   📝 {expected_complexity.title()} Task:")
        print(f"      Prompt: {prompt[:50]}...")
        
        if recommendation["recommended_model"]:
            print(f"      Model: {recommendation['recommended_model'].split('/')[-1]}")
            print(f"      Cost: ${recommendation['estimated_cost']:.6f}")
            print(f"      Reason: {recommendation['reasoning'][:60]}...")
            
            # Test the actual recommendation
            try:
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": prompt}],
                    model=recommendation["recommended_model"],
                    max_tokens=80,
                    feature="task-optimization",
                    task_complexity=expected_complexity
                )
                print(f"      Actual cost: ${result.cost:.6f}")
                print(f"      Speed: {result.execution_time_seconds:.2f}s")
                
            except Exception as e:
                print(f"      ❌ Test failed: {e}")
        else:
            print("      ❌ No suitable model within budget")
    
    # Example 4: Budget-constrained workflow optimization
    print("\n" + "=" * 60)
    print("💸 Example 4: Budget-Constrained Workflow")
    print("=" * 60)
    
    workflow_prompts = [
        "Summarize this quarterly report",
        "Generate creative marketing copy",
        "Analyze customer feedback sentiment",
        "Create technical documentation",
        "Write code comments and explanations"
    ]
    
    total_budget = 0.025  # $0.025 total budget
    print(f"Optimizing workflow within ${total_budget:.3f} budget:")
    
    optimization_results = optimizer.optimize_model_selection_for_budget(
        workflow_prompts, total_budget, prefer_quality=False
    )
    
    total_estimated_cost = 0
    successful_operations = 0
    
    for result in optimization_results:
        print(f"\n   📝 Task {result['prompt_index'] + 1}: {result['prompt']}")
        print(f"      Complexity: {result['complexity']}")
        
        if result["model"] != "BUDGET_EXCEEDED":
            print(f"      Model: {result['model']}")
            print(f"      Cost: ${result['estimated_cost']:.6f}")
            total_estimated_cost += result["estimated_cost"]
            successful_operations += 1
        else:
            print(f"      ❌ Budget exceeded")
    
    print(f"\n   📊 Workflow Summary:")
    print(f"      Operations: {successful_operations}/{len(workflow_prompts)}")
    print(f"      Total cost: ${total_estimated_cost:.6f}")
    print(f"      Budget used: {(total_estimated_cost / total_budget) * 100:.1f}%")
    
    # Example 5: Real-world cost projection
    print("\n" + "=" * 60)
    print("📈 Example 5: Real-World Cost Projections")
    print("=" * 60)
    
    # Analyze costs for different usage patterns
    usage_scenarios = [
        ("High-volume simple tasks", 10000, 200, "simple"),
        ("Medium-volume analysis", 1000, 800, "moderate"),
        ("Low-volume complex reasoning", 100, 2000, "complex")
    ]
    
    print("Cost projections for different usage patterns:")
    
    for scenario, ops_per_day, avg_tokens, complexity in usage_scenarios:
        # Get recommended model for this scenario
        rec = optimizer.pricing_calc.recommend_model(
            task_complexity=complexity,
            budget_per_operation=0.01,
            prefer_batch=True
        )
        
        if rec.recommended_model:
            # Analyze costs for this scenario
            analysis = optimizer.pricing_calc.analyze_costs(
                operations_per_day=ops_per_day,
                avg_tokens_per_operation=avg_tokens,
                model=rec.recommended_model,
                days_to_analyze=30,
                batch_percentage=0.5  # 50% batch processing
            )
            
            print(f"\n   🏢 {scenario}:")
            print(f"      Model: {analysis['current_model'].split('/')[-1]}")
            print(f"      Daily: ${analysis['cost_analysis']['daily_cost']:.2f}")
            print(f"      Monthly: ${analysis['cost_analysis']['monthly_cost']:.2f}")
            print(f"      Batch savings: ${analysis['optimization']['batch_optimization_potential']:.2f}/month")
            
            if analysis['optimization']['best_alternative']:
                alt = analysis['optimization']['best_alternative']
                print(f"      Alternative: {alt['model'].split('/')[-1]}")
                print(f"      Potential savings: ${analysis['optimization']['potential_monthly_savings']:.2f}/month")
    
    # Show overall cost summary
    print("\n" + "=" * 60)
    print("💰 Cost Optimization Summary")
    print("=" * 60)
    
    cost_summary = adapter.get_cost_summary()
    print(f"Demo spending: ${cost_summary['daily_costs']:.6f}")
    print(f"Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
    
    total_operations = len(model_results) + successful_operations + 3  # +3 for task examples
    average_cost = float(cost_summary['daily_costs']) / total_operations if total_operations > 0 else 0
    
    print(f"Operations completed: {total_operations}")
    print(f"Average cost per operation: ${average_cost:.6f}")
    
    # Cost optimization recommendations
    print("\n🎯 Optimization Recommendations:")
    
    if cost_summary['daily_budget_utilization'] < 20:
        print("   • Budget very conservatively used - consider higher-quality models")
    elif cost_summary['daily_budget_utilization'] < 50:
        print("   • Good cost efficiency - well within budget")
    else:
        print("   • Consider batch processing for 50% cost savings")
        print("   • Switch to smaller models for high-volume tasks")
    
    print("   • Use 8B models for simple tasks (4x cheaper than 70B)")
    print("   • Leverage batch processing for 50% savings on large workloads") 
    print("   • Take advantage of Fireworks' 4x speed for better throughput")
    print("   • Monitor cost per task and optimize model selection accordingly")
    
    print("\n🎉 Cost optimization demonstration completed!")
    print("\n🚀 Next Steps:")
    print("   • Implement batch processing in production for 50% savings")
    print("   • Use task complexity analysis for automatic model selection")
    print("   • Set up budget alerts and governance policies")
    print("   • Leverage Fireworks' speed advantage for cost-effective scale")
    
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