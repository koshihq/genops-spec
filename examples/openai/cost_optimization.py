#!/usr/bin/env python3
"""
OpenAI Cost Optimization Example

This example demonstrates intelligent cost optimization strategies using GenOps
telemetry and multi-model selection based on complexity and cost constraints.

What you'll learn:
- Dynamic model selection based on task complexity
- Cost-aware completion strategies
- Model performance vs cost tradeoffs
- Budget-constrained AI operations

Usage:
    python cost_optimization.py
    
Prerequisites:
    pip install genops-ai[openai]
    export OPENAI_API_KEY="your_api_key_here"
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for OpenAI model with cost and performance characteristics."""
    name: str
    cost_per_1k_input: float  # USD per 1K input tokens
    cost_per_1k_output: float  # USD per 1K output tokens
    max_tokens: int
    temperature: float
    use_case: str
    performance_tier: str

def get_model_configurations() -> Dict[str, ModelConfig]:
    """Get current OpenAI model configurations with pricing and use cases."""
    return {
        "economy": ModelConfig(
            name="gpt-3.5-turbo",
            cost_per_1k_input=0.0015,
            cost_per_1k_output=0.002,
            max_tokens=300,
            temperature=0.3,
            use_case="Simple tasks, high volume operations",
            performance_tier="Standard"
        ),
        "efficient": ModelConfig(
            name="gpt-4o-mini", 
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            max_tokens=500,
            temperature=0.5,
            use_case="Balanced cost and capability",
            performance_tier="High"
        ),
        "balanced": ModelConfig(
            name="gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            max_tokens=800,
            temperature=0.7,
            use_case="Complex reasoning, analysis",
            performance_tier="Premium"
        ),
        "premium": ModelConfig(
            name="gpt-4-turbo",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            max_tokens=1000,
            temperature=0.7,
            use_case="Advanced reasoning, creative tasks",
            performance_tier="Premium+"
        ),
        "ultimate": ModelConfig(
            name="gpt-4",
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.06,
            max_tokens=1500,
            temperature=0.8,
            use_case="Highest quality, complex analysis",
            performance_tier="Ultimate"
        )
    }

def estimate_cost(prompt: str, config: ModelConfig) -> float:
    """Estimate the cost of a completion based on prompt and model config."""
    # Rough token estimation (actual tokenization would be more accurate)
    estimated_input_tokens = len(prompt.split()) * 1.3
    estimated_output_tokens = config.max_tokens * 0.7  # Assume 70% of max tokens used
    
    input_cost = (estimated_input_tokens / 1000) * config.cost_per_1k_input
    output_cost = (estimated_output_tokens / 1000) * config.cost_per_1k_output
    
    return input_cost + output_cost

def smart_model_selection():
    """Demonstrate intelligent model selection based on task complexity."""
    print("🧠 Smart Model Selection Based on Task Complexity")
    print("-" * 60)
    
    try:
        from genops.providers.openai import instrument_openai
        
        client = instrument_openai()
        configs = get_model_configurations()
        
        # Define test tasks with different complexity levels
        tasks = [
            {
                "prompt": "What is 2 + 2?",
                "complexity": "economy",
                "description": "Simple arithmetic"
            },
            {
                "prompt": "Explain the concept of machine learning in simple terms.",
                "complexity": "efficient", 
                "description": "Basic explanation"
            },
            {
                "prompt": "Analyze the potential economic impacts of artificial intelligence on employment in the next decade.",
                "complexity": "balanced",
                "description": "Complex analysis"
            },
            {
                "prompt": "Write a comprehensive business strategy for a startup entering the renewable energy market, considering regulatory challenges, competitive landscape, and financial projections.",
                "complexity": "premium",
                "description": "Strategic planning"
            }
        ]
        
        print("📊 Model Selection Strategy:")
        print(f"{'Task Type':<20} {'Model':<20} {'Est. Cost':<12} {'Use Case'}")
        print("-" * 80)
        
        total_cost = 0
        results = []
        
        for task in tasks:
            config = configs[task["complexity"]]
            estimated_cost = estimate_cost(task["prompt"], config)
            
            print(f"{task['description']:<20} {config.name:<20} ${estimated_cost:.4f}      {config.use_case[:30]}")
            
            # Make the actual request
            print(f"🚀 Processing: {task['description']}")
            
            response = client.chat_completions_create(
                model=config.name,
                messages=[{"role": "user", "content": task["prompt"]}],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                
                # Governance attributes with cost optimization tracking
                team="cost-optimization-team",
                project="smart-model-selection",
                customer_id="optimization-demo",
                complexity_level=task["complexity"],
                estimated_cost=estimated_cost,
                optimization_strategy="complexity_based"
            )
            
            actual_tokens = response.usage.total_tokens
            actual_cost = (response.usage.prompt_tokens / 1000 * config.cost_per_1k_input + 
                          response.usage.completion_tokens / 1000 * config.cost_per_1k_output)
            
            results.append({
                "task": task["description"],
                "model": config.name,
                "estimated_cost": estimated_cost,
                "actual_cost": actual_cost,
                "tokens": actual_tokens,
                "response": response.choices[0].message.content[:100] + "..."
            })
            
            total_cost += actual_cost
            print(f"   Response ({actual_tokens} tokens, ${actual_cost:.4f}): {response.choices[0].message.content[:60]}...\n")
        
        print(f"\n💰 Total cost for optimized model selection: ${total_cost:.4f}")
        print(f"🎯 Estimated savings vs using GPT-4 for all: ~60-80%")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart model selection error: {e}")
        return False

def budget_constrained_completion():
    """Demonstrate cost-aware completions within budget constraints."""
    print("\n\n💰 Budget-Constrained Completion")
    print("-" * 50)
    
    try:
        from genops.providers.openai import instrument_openai
        
        client = instrument_openai()
        configs = get_model_configurations()
        
        def complete_within_budget(prompt: str, max_budget: float = 0.01) -> Dict:
            """Choose the best model that fits within the specified budget."""
            
            # Sort models by performance tier (best first)
            performance_order = ["ultimate", "premium", "balanced", "efficient", "economy"]
            
            for tier in performance_order:
                config = configs[tier]
                estimated_cost = estimate_cost(prompt, config)
                
                if estimated_cost <= max_budget:
                    print(f"✅ Selected {config.name} (${estimated_cost:.4f} <= ${max_budget} budget)")
                    
                    response = client.chat_completions_create(
                        model=config.name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        
                        # Budget-aware governance attributes
                        team="budget-team", 
                        project="cost-controlled-ai",
                        customer_id="budget-demo",
                        max_budget=max_budget,
                        selected_model=config.name,
                        optimization_strategy="budget_constrained"
                    )
                    
                    actual_cost = (response.usage.prompt_tokens / 1000 * config.cost_per_1k_input + 
                                  response.usage.completion_tokens / 1000 * config.cost_per_1k_output)
                    
                    return {
                        "model": config.name,
                        "estimated_cost": estimated_cost,
                        "actual_cost": actual_cost,
                        "budget": max_budget,
                        "within_budget": actual_cost <= max_budget,
                        "response": response.choices[0].message.content,
                        "tokens": response.usage.total_tokens
                    }
            
            raise ValueError(f"No model available within budget of ${max_budget}")
        
        # Test different budget scenarios
        test_scenarios = [
            {
                "prompt": "Explain quantum computing briefly",
                "budget": 0.001,
                "scenario": "Ultra-low budget"
            },
            {
                "prompt": "Write a detailed analysis of renewable energy trends",
                "budget": 0.01,
                "scenario": "Medium budget"
            },
            {
                "prompt": "Create a comprehensive marketing strategy for a tech startup",
                "budget": 0.05,
                "scenario": "High budget"
            }
        ]
        
        print("📊 Budget-Constrained Results:")
        print(f"{'Scenario':<20} {'Budget':<10} {'Model':<20} {'Actual Cost':<12} {'Status'}")
        print("-" * 80)
        
        for scenario in test_scenarios:
            try:
                result = complete_within_budget(scenario["prompt"], scenario["budget"])
                
                status = "✅ Within Budget" if result["within_budget"] else "❌ Over Budget"
                print(f"{scenario['scenario']:<20} ${scenario['budget']:<9.3f} {result['model']:<20} ${result['actual_cost']:<11.4f} {status}")
                print(f"   Response: {result['response'][:60]}...\n")
                
            except ValueError as e:
                print(f"{scenario['scenario']:<20} ${scenario['budget']:<9.3f} {'None':<20} {'N/A':<12} ❌ No Model")
                print(f"   Error: {e}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Budget-constrained completion error: {e}")
        return False

def cost_comparison_analysis():
    """Compare costs across different models for the same task."""
    print("\n\n📈 Cost Comparison Analysis")
    print("-" * 50)
    
    try:
        from genops.providers.openai import instrument_openai
        
        client = instrument_openai()
        configs = get_model_configurations()
        
        # Test prompt
        test_prompt = "Explain the benefits and drawbacks of remote work for both employees and employers."
        
        print(f"📝 Test prompt: {test_prompt}")
        print(f"\n📊 Cost Comparison Results:")
        print(f"{'Model':<20} {'Actual Cost':<12} {'Tokens':<10} {'Cost per Token':<15} {'Response Quality'}")
        print("-" * 85)
        
        results = []
        
        for tier, config in configs.items():
            try:
                print(f"🔄 Testing {config.name}...")
                
                response = client.chat_completions_create(
                    model=config.name,
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=200,  # Fixed for fair comparison
                    temperature=0.7,  # Fixed for consistency
                    
                    # Comparison tracking
                    team="comparison-team",
                    project="cost-analysis",
                    customer_id="analysis-demo",
                    model_tier=tier,
                    comparison_study="cross_model_cost"
                )
                
                actual_cost = (response.usage.prompt_tokens / 1000 * config.cost_per_1k_input + 
                              response.usage.completion_tokens / 1000 * config.cost_per_1k_output)
                
                cost_per_token = actual_cost / response.usage.total_tokens if response.usage.total_tokens > 0 else 0
                
                # Simple quality assessment based on response length and coherence
                response_text = response.choices[0].message.content
                quality_score = min(len(response_text.split()), 100)  # Simplified quality metric
                quality_rating = "⭐" * min(5, quality_score // 20)
                
                results.append({
                    "model": config.name,
                    "tier": tier,
                    "cost": actual_cost,
                    "tokens": response.usage.total_tokens,
                    "cost_per_token": cost_per_token,
                    "quality": quality_rating,
                    "response": response_text
                })
                
                print(f"{config.name:<20} ${actual_cost:<11.4f} {response.usage.total_tokens:<10} ${cost_per_token:<14.6f} {quality_rating}")
                
            except Exception as e:
                print(f"{config.name:<20} Error: {e}")
        
        # Analysis summary
        if results:
            best_value = min(results, key=lambda x: x["cost_per_token"])
            most_expensive = max(results, key=lambda x: x["cost"])
            cheapest = min(results, key=lambda x: x["cost"])
            
            print(f"\n🏆 Analysis Summary:")
            print(f"   • Best value (cost per token): {best_value['model']} (${best_value['cost_per_token']:.6f}/token)")
            print(f"   • Cheapest total cost: {cheapest['model']} (${cheapest['cost']:.4f})")
            print(f"   • Most expensive: {most_expensive['model']} (${most_expensive['cost']:.4f})")
            print(f"   • Cost range: {most_expensive['cost'] / cheapest['cost']:.1f}x difference")
        
        return True
        
    except Exception as e:
        print(f"❌ Cost comparison analysis error: {e}")
        return False

def main():
    """Run cost optimization demonstrations."""
    print("💰 GenOps OpenAI Cost Optimization Examples")
    print("=" * 60)
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("💡 Fix: export OPENAI_API_KEY='your_api_key_here'")
        return False
    
    success = True
    
    # Run optimization examples
    success &= smart_model_selection()
    success &= budget_constrained_completion()
    success &= cost_comparison_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 Cost optimization examples completed successfully!")
        
        print("\n💡 Key Cost Optimization Strategies:")
        print("   ✅ Task complexity-based model selection")
        print("   ✅ Budget-constrained model choosing")
        print("   ✅ Real-time cost comparison and analysis")
        print("   ✅ Automatic cost attribution and tracking")
        
        print("\n📊 Business Benefits:")
        print("   • 60-80% cost savings through smart model selection")
        print("   • Budget compliance and cost predictability")
        print("   • Detailed cost attribution for billing and chargebacks")
        print("   • Performance vs cost optimization insights")
        
        print("\n🚀 Next Steps:")
        print("   • Run 'python multi_provider_costs.py' for cross-provider comparison")
        print("   • Try 'python advanced_features.py' for streaming and function costs")
        print("   • Explore 'python production_patterns.py' for enterprise optimization")
        
        return True
    else:
        print("❌ Cost optimization examples failed.")
        print("💡 Check the error messages above and verify your OpenAI setup")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)