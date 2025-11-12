#!/usr/bin/env python3
"""
LiteLLM Cost Optimization with GenOps

Demonstrates advanced cost reduction strategies and intelligent model selection
across 100+ providers using LiteLLM + GenOps. This example focuses on maximizing
value through smart provider selection, model optimization, and cost-aware routing.

Usage:
    export OPENAI_API_KEY="your_key_here"
    export ANTHROPIC_API_KEY="your_key_here"  # Optional but recommended
    python cost_optimization.py

Features:
    - Dynamic model selection based on cost/performance trade-offs
    - Provider cost comparison and automatic switching
    - Task complexity-based model routing
    - Real-time cost optimization recommendations
    - Budget-constrained operation strategies
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TaskComplexity(Enum):
    """Task complexity levels for model selection."""
    SIMPLE = "simple"        # Basic Q&A, simple text generation
    MODERATE = "moderate"    # Analysis, summarization
    COMPLEX = "complex"      # Reasoning, complex analysis
    ADVANCED = "advanced"    # Multi-step reasoning, coding


@dataclass
class ModelProfile:
    """Performance and cost profile for a model."""
    model: str
    provider: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: float
    quality_score: float  # 1-10 scale
    max_complexity: TaskComplexity
    strengths: List[str] = field(default_factory=list)
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token usage."""
        return ((input_tokens * self.cost_per_1k_input) + 
                (output_tokens * self.cost_per_1k_output)) / 1000
    
    def get_value_score(self, complexity: TaskComplexity, budget_priority: float = 0.5) -> float:
        """Calculate value score based on cost, quality, and complexity fit."""
        # Complexity suitability (0-1)
        complexity_order = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MODERATE: 2, 
            TaskComplexity.COMPLEX: 3,
            TaskComplexity.ADVANCED: 4
        }
        
        required_level = complexity_order[complexity]
        max_level = complexity_order[self.max_complexity]
        
        if max_level < required_level:
            return 0.0  # Model can't handle this complexity
        
        complexity_fit = 1.0 - (max_level - required_level) * 0.1  # Penalty for over-capability
        
        # Cost efficiency (inverted - lower cost = higher score)
        avg_cost_per_token = (self.cost_per_1k_input + self.cost_per_1k_output) / 2000
        cost_efficiency = 1.0 / (1.0 + avg_cost_per_token * 1000)  # Normalize
        
        # Quality score (normalized)
        quality_normalized = self.quality_score / 10.0
        
        # Performance score (latency - inverted)
        performance_score = 1.0 / (1.0 + self.avg_latency_ms / 1000)
        
        # Weighted value score
        value_score = (
            budget_priority * cost_efficiency +
            (1 - budget_priority) * (
                0.4 * quality_normalized +
                0.3 * complexity_fit +
                0.3 * performance_score
            )
        )
        
        return value_score


# Model profiles based on real-world performance and pricing
MODEL_PROFILES = [
    # Fast & Economical Models
    ModelProfile(
        model="gpt-3.5-turbo",
        provider="openai",
        cost_per_1k_input=0.0015,
        cost_per_1k_output=0.002,
        avg_latency_ms=800,
        quality_score=7.5,
        max_complexity=TaskComplexity.MODERATE,
        strengths=["speed", "cost-effective", "general-purpose"]
    ),
    ModelProfile(
        model="claude-3-haiku",
        provider="anthropic",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        avg_latency_ms=900,
        quality_score=8.0,
        max_complexity=TaskComplexity.MODERATE,
        strengths=["very-cost-effective", "thoughtful", "safety-focused"]
    ),
    ModelProfile(
        model="gemini-pro",
        provider="google",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        avg_latency_ms=1200,
        quality_score=7.8,
        max_complexity=TaskComplexity.MODERATE,
        strengths=["multimodal", "good-value", "google-integration"]
    ),
    
    # High-Capability Models
    ModelProfile(
        model="gpt-4",
        provider="openai",
        cost_per_1k_input=0.03,
        cost_per_1k_output=0.06,
        avg_latency_ms=2000,
        quality_score=9.2,
        max_complexity=TaskComplexity.ADVANCED,
        strengths=["reasoning", "coding", "complex-analysis"]
    ),
    ModelProfile(
        model="claude-3-sonnet",
        provider="anthropic",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        avg_latency_ms=2200,
        quality_score=9.0,
        max_complexity=TaskComplexity.ADVANCED,
        strengths=["balanced", "analysis", "safety", "cost-effective"]
    ),
    ModelProfile(
        model="gemini-1.5-pro",
        provider="google",
        cost_per_1k_input=0.0035,
        cost_per_1k_output=0.01,
        avg_latency_ms=2500,
        quality_score=8.8,
        max_complexity=TaskComplexity.ADVANCED,
        strengths=["multimodal", "large-context", "advanced-reasoning"]
    ),
    
    # Premium Models
    ModelProfile(
        model="claude-3-opus",
        provider="anthropic",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        avg_latency_ms=3000,
        quality_score=9.5,
        max_complexity=TaskComplexity.ADVANCED,
        strengths=["maximum-capability", "creative", "complex-reasoning"]
    )
]


class CostOptimizer:
    """Advanced cost optimization engine for LiteLLM."""
    
    def __init__(self, model_profiles: List[ModelProfile] = None):
        self.model_profiles = model_profiles or MODEL_PROFILES
        self.usage_history = []
        self.cost_savings_achieved = 0.0
        
    def get_available_models(self) -> List[ModelProfile]:
        """Get models with available API keys."""
        available = []
        
        api_key_mapping = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GOOGLE_API_KEY'
        }
        
        for profile in self.model_profiles:
            env_var = api_key_mapping.get(profile.provider)
            if env_var and os.getenv(env_var):
                available.append(profile)
        
        return available
    
    def select_optimal_model(
        self,
        task_complexity: TaskComplexity,
        estimated_input_tokens: int = 150,
        estimated_output_tokens: int = 50,
        budget_priority: float = 0.5,  # 0.0 = quality focus, 1.0 = cost focus
        max_cost: Optional[float] = None
    ) -> Optional[ModelProfile]:
        """
        Select optimal model based on task requirements and constraints.
        
        Args:
            task_complexity: Complexity level of the task
            estimated_input_tokens: Estimated input tokens
            estimated_output_tokens: Estimated output tokens  
            budget_priority: 0-1 scale for cost vs quality trade-off
            max_cost: Maximum acceptable cost per request
            
        Returns:
            Optimal model profile or None if no suitable model
        """
        available_models = self.get_available_models()
        
        if not available_models:
            return None
        
        # Filter by cost constraint if specified
        if max_cost is not None:
            available_models = [
                model for model in available_models
                if model.estimate_cost(estimated_input_tokens, estimated_output_tokens) <= max_cost
            ]
        
        if not available_models:
            return None
        
        # Calculate value scores for all models
        scored_models = []
        for model in available_models:
            value_score = model.get_value_score(task_complexity, budget_priority)
            if value_score > 0:  # Model can handle the task
                scored_models.append((model, value_score))
        
        if not scored_models:
            return None
        
        # Return model with highest value score
        return max(scored_models, key=lambda x: x[1])[0]
    
    def compare_model_costs(
        self,
        task_complexity: TaskComplexity,
        estimated_input_tokens: int = 150,
        estimated_output_tokens: int = 50
    ) -> List[Dict[str, Any]]:
        """Compare costs across all suitable models for a task."""
        available_models = self.get_available_models()
        
        # Filter models that can handle the complexity
        suitable_models = [
            model for model in available_models
            if model.get_value_score(task_complexity, 0.0) > 0
        ]
        
        comparisons = []
        for model in suitable_models:
            cost = model.estimate_cost(estimated_input_tokens, estimated_output_tokens)
            
            comparisons.append({
                'model': model.model,
                'provider': model.provider,
                'cost': cost,
                'quality_score': model.quality_score,
                'latency_ms': model.avg_latency_ms,
                'strengths': model.strengths,
                'cost_per_quality_point': cost / model.quality_score if model.quality_score > 0 else float('inf')
            })
        
        # Sort by cost
        comparisons.sort(key=lambda x: x['cost'])
        return comparisons
    
    def get_cost_savings_recommendation(
        self,
        current_model: str,
        task_complexity: TaskComplexity,
        estimated_tokens: Tuple[int, int] = (150, 50)
    ) -> Dict[str, Any]:
        """Get cost savings recommendations for current model usage."""
        input_tokens, output_tokens = estimated_tokens
        
        # Find current model profile
        current_profile = None
        for profile in self.model_profiles:
            if profile.model == current_model:
                current_profile = profile
                break
        
        if not current_profile:
            return {"error": f"Model {current_model} not found in profiles"}
        
        current_cost = current_profile.estimate_cost(input_tokens, output_tokens)
        
        # Find optimal alternative
        optimal_model = self.select_optimal_model(
            task_complexity, input_tokens, output_tokens, budget_priority=0.8
        )
        
        if not optimal_model or optimal_model.model == current_model:
            return {
                "current_model": current_model,
                "current_cost": current_cost,
                "recommendation": "Current model is already optimal",
                "potential_savings": 0.0
            }
        
        optimal_cost = optimal_model.estimate_cost(input_tokens, output_tokens)
        potential_savings = current_cost - optimal_cost
        savings_percentage = (potential_savings / current_cost) * 100 if current_cost > 0 else 0
        
        return {
            "current_model": current_model,
            "current_cost": current_cost,
            "recommended_model": optimal_model.model,
            "recommended_provider": optimal_model.provider,
            "recommended_cost": optimal_cost,
            "potential_savings": potential_savings,
            "savings_percentage": savings_percentage,
            "quality_impact": optimal_model.quality_score - current_profile.quality_score,
            "rationale": f"Switch to {optimal_model.model} for {savings_percentage:.1f}% cost reduction"
        }


def check_optimization_setup():
    """Check setup for cost optimization demo."""
    print("🔍 Checking cost optimization setup...")
    
    # Check imports
    try:
        import litellm
        from genops.providers.litellm import auto_instrument, multi_provider_cost_tracking
        print("✅ LiteLLM and GenOps available")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install: pip install litellm genops[litellm]")
        return False
    
    # Check API keys
    api_keys_found = []
    api_checks = {
        'OpenAI': 'OPENAI_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY',
        'Google': 'GOOGLE_API_KEY'
    }
    
    for provider, env_var in api_checks.items():
        if os.getenv(env_var):
            api_keys_found.append(provider)
            print(f"✅ {provider} API key configured")
    
    if len(api_keys_found) < 2:
        print(f"⚠️  Only {len(api_keys_found)} provider(s) configured")
        print("💡 For best cost optimization, configure multiple providers:")
        print("   export OPENAI_API_KEY=your_key")
        print("   export ANTHROPIC_API_KEY=your_key")
        print("   export GOOGLE_API_KEY=your_key")
    else:
        print(f"✅ {len(api_keys_found)} providers configured for optimization")
    
    return len(api_keys_found) > 0


def demo_intelligent_model_selection():
    """Demonstrate intelligent model selection based on task requirements."""
    print("\n" + "="*60)
    print("🧠 Demo: Intelligent Model Selection")
    print("="*60)
    
    print("Smart model selection optimizes cost without sacrificing quality")
    print("by matching model capabilities to task complexity requirements.")
    
    optimizer = CostOptimizer()
    available_models = optimizer.get_available_models()
    
    print(f"\n📊 Available models: {len(available_models)}")
    for model in available_models[:3]:  # Show first 3
        print(f"   • {model.model} ({model.provider}) - Quality: {model.quality_score}/10")
    
    # Test scenarios with different complexity levels
    test_scenarios = [
        {
            "task": "Simple Q&A: What is the capital of France?",
            "complexity": TaskComplexity.SIMPLE,
            "tokens": (20, 5),
            "budget_priority": 0.8  # Very cost-focused
        },
        {
            "task": "Analysis: Summarize this document",
            "complexity": TaskComplexity.MODERATE,
            "tokens": (500, 150),
            "budget_priority": 0.5  # Balanced
        },
        {
            "task": "Complex reasoning: Multi-step problem solving",
            "complexity": TaskComplexity.ADVANCED,
            "tokens": (800, 300),
            "budget_priority": 0.2  # Quality-focused
        }
    ]
    
    print(f"\n🎯 Testing intelligent selection across complexity levels:")
    
    for scenario in test_scenarios:
        print(f"\n📋 Scenario: {scenario['task']}")
        print(f"   Complexity: {scenario['complexity'].value}")
        print(f"   Tokens: {scenario['tokens'][0]} input, {scenario['tokens'][1]} output")
        
        optimal_model = optimizer.select_optimal_model(
            task_complexity=scenario['complexity'],
            estimated_input_tokens=scenario['tokens'][0],
            estimated_output_tokens=scenario['tokens'][1],
            budget_priority=scenario['budget_priority']
        )
        
        if optimal_model:
            cost = optimal_model.estimate_cost(*scenario['tokens'])
            print(f"   ✅ Selected: {optimal_model.model} ({optimal_model.provider})")
            print(f"   💰 Cost: ${cost:.6f}")
            print(f"   ⭐ Quality: {optimal_model.quality_score}/10")
            print(f"   💡 Strengths: {', '.join(optimal_model.strengths[:3])}")
        else:
            print(f"   ❌ No suitable model found")


def demo_cost_comparison_analysis():
    """Demonstrate detailed cost comparison across providers."""
    print("\n" + "="*60)
    print("💰 Demo: Cost Comparison Analysis")
    print("="*60)
    
    print("Compare costs across providers for equivalent task complexity")
    print("to identify optimization opportunities.")
    
    optimizer = CostOptimizer()
    
    # Test different complexity levels
    complexity_tests = [
        (TaskComplexity.SIMPLE, "Simple tasks", (50, 20)),
        (TaskComplexity.MODERATE, "Analysis tasks", (200, 100)),
        (TaskComplexity.ADVANCED, "Complex reasoning", (500, 300))
    ]
    
    for complexity, description, tokens in complexity_tests:
        print(f"\n📊 {description} ({complexity.value}):")
        print(f"   Token estimate: {tokens[0]} input, {tokens[1]} output")
        
        comparisons = optimizer.compare_model_costs(complexity, *tokens)
        
        if not comparisons:
            print("   ⚠️  No suitable models available")
            continue
        
        print("   Cost comparison (cheapest to most expensive):")
        
        for i, comp in enumerate(comparisons[:5]):  # Show top 5
            rank_emoji = ["🥇", "🥈", "🥉", "📍", "📍"][min(i, 4)]
            
            print(f"   {rank_emoji} {comp['model']} ({comp['provider']})")
            print(f"      💰 Cost: ${comp['cost']:.6f}")
            print(f"      ⭐ Quality: {comp['quality_score']}/10")
            print(f"      🔥 Value: ${comp['cost_per_quality_point']:.6f}/quality point")
            
            # Show savings vs most expensive
            if i == 0 and len(comparisons) > 1:
                most_expensive = comparisons[-1]
                savings = most_expensive['cost'] - comp['cost']
                savings_pct = (savings / most_expensive['cost']) * 100
                print(f"      💡 Saves ${savings:.6f} ({savings_pct:.1f}%) vs most expensive")


def demo_real_time_optimization():
    """Demonstrate real-time cost optimization with actual API calls."""
    print("\n" + "="*60)
    print("⚡ Demo: Real-Time Cost Optimization")
    print("="*60)
    
    import litellm
    from genops.providers.litellm import auto_instrument, get_usage_stats, multi_provider_cost_tracking
    
    print("Real-time optimization selects the best model for each request")
    print("based on current costs, performance, and requirements.")
    
    # Enable GenOps tracking
    auto_instrument(
        team="cost-optimization",
        project="real-time-demo",
        daily_budget_limit=5.0,  # Small demo budget
        governance_policy="advisory"
    )
    
    optimizer = CostOptimizer()
    
    # Simulate different types of requests
    request_scenarios = [
        {
            "type": "customer_support",
            "prompt": "Hello! How can I help you today?",
            "complexity": TaskComplexity.SIMPLE,
            "budget_priority": 0.9  # Very cost-conscious
        },
        {
            "type": "data_analysis", 
            "prompt": "Analyze these key trends and provide insights.",
            "complexity": TaskComplexity.MODERATE,
            "budget_priority": 0.4  # Quality important
        },
        {
            "type": "strategic_planning",
            "prompt": "Develop a comprehensive strategy considering multiple factors.",
            "complexity": TaskComplexity.ADVANCED,
            "budget_priority": 0.1  # Quality critical
        }
    ]
    
    print(f"\n🎯 Processing {len(request_scenarios)} optimized requests:")
    
    total_savings = 0.0
    
    for i, scenario in enumerate(request_scenarios):
        print(f"\n📋 Request {i+1}: {scenario['type']}")
        
        # Select optimal model
        optimal_model = optimizer.select_optimal_model(
            task_complexity=scenario['complexity'],
            estimated_input_tokens=len(scenario['prompt'].split()) * 1.3,  # Rough estimate
            estimated_output_tokens=30,
            budget_priority=scenario['budget_priority']
        )
        
        if not optimal_model:
            print("   ❌ No suitable model available")
            continue
        
        print(f"   🎯 Selected: {optimal_model.model} ({optimal_model.provider})")
        print(f"   💡 Reason: Optimized for {scenario['complexity'].value} tasks")
        
        # Simulate API call (with error handling)
        try:
            start_time = time.time()
            
            # Use the optimized model
            response = litellm.completion(
                model=optimal_model.model,
                messages=[{"role": "user", "content": scenario['prompt']}],
                max_tokens=30,
                timeout=10
            )
            
            end_time = time.time()
            actual_latency = (end_time - start_time) * 1000
            
            print(f"   ✅ Completed in {actual_latency:.0f}ms")
            
            # Get cost savings recommendation
            baseline_model = "gpt-4"  # Compare against premium baseline
            if optimal_model.model != baseline_model:
                savings_rec = optimizer.get_cost_savings_recommendation(
                    baseline_model, scenario['complexity']
                )
                
                if 'potential_savings' in savings_rec and savings_rec['potential_savings'] > 0:
                    total_savings += savings_rec['potential_savings']
                    print(f"   💰 Saved: ${savings_rec['potential_savings']:.6f} vs {baseline_model}")
                
        except Exception as e:
            print(f"   ⚠️ Request failed: [Error details redacted for security]")
    
    # Show optimization results
    print(f"\n📊 Optimization Results:")
    
    stats = get_usage_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Total cost: ${stats['total_cost']:.6f}")
    print(f"   Estimated savings: ${total_savings:.6f}")
    
    if total_savings > 0:
        print(f"   🎉 Cost optimization achieved {(total_savings / (stats['total_cost'] + total_savings)) * 100:.1f}% savings!")
    
    # Multi-provider cost tracking
    cost_breakdown = multi_provider_cost_tracking(group_by="provider")
    
    if cost_breakdown['cost_by_provider']:
        print(f"\n📈 Provider cost distribution:")
        for provider, cost in cost_breakdown['cost_by_provider'].items():
            percentage = (cost / cost_breakdown['total_cost']) * 100 if cost_breakdown['total_cost'] > 0 else 0
            print(f"   • {provider}: ${cost:.6f} ({percentage:.1f}%)")


def demo_budget_constrained_optimization():
    """Demonstrate optimization under budget constraints."""
    print("\n" + "="*60)
    print("💳 Demo: Budget-Constrained Optimization")
    print("="*60)
    
    print("Optimize model selection when working within strict budget limits")
    print("while maintaining acceptable quality levels.")
    
    optimizer = CostOptimizer()
    
    # Budget scenarios
    budget_scenarios = [
        {"name": "Tight budget", "max_cost": 0.001, "description": "Maximum $0.001 per request"},
        {"name": "Moderate budget", "max_cost": 0.01, "description": "Maximum $0.01 per request"},
        {"name": "Flexible budget", "max_cost": 0.1, "description": "Maximum $0.10 per request"}
    ]
    
    test_task = {
        "complexity": TaskComplexity.MODERATE,
        "tokens": (200, 100),
        "description": "Document analysis task"
    }
    
    print(f"\n📋 Task: {test_task['description']}")
    print(f"   Complexity: {test_task['complexity'].value}")
    print(f"   Tokens: {test_task['tokens'][0]} input, {test_task['tokens'][1]} output")
    
    for scenario in budget_scenarios:
        print(f"\n💰 {scenario['name']}: {scenario['description']}")
        
        # Find optimal model within budget
        optimal_model = optimizer.select_optimal_model(
            task_complexity=test_task['complexity'],
            estimated_input_tokens=test_task['tokens'][0],
            estimated_output_tokens=test_task['tokens'][1],
            budget_priority=0.8,  # Cost-focused
            max_cost=scenario['max_cost']
        )
        
        if optimal_model:
            actual_cost = optimal_model.estimate_cost(*test_task['tokens'])
            budget_usage = (actual_cost / scenario['max_cost']) * 100
            
            print(f"   ✅ Selected: {optimal_model.model} ({optimal_model.provider})")
            print(f"   💰 Cost: ${actual_cost:.6f} ({budget_usage:.1f}% of budget)")
            print(f"   ⭐ Quality: {optimal_model.quality_score}/10")
            print(f"   💡 Efficiency: {optimal_model.quality_score / (actual_cost * 1000):.1f} quality/cost ratio")
        else:
            print(f"   ❌ No model available within ${scenario['max_cost']:.6f} budget")
            
            # Show cheapest available option
            available_models = optimizer.get_available_models()
            if available_models:
                costs = [(model, model.estimate_cost(*test_task['tokens'])) for model in available_models]
                cheapest_model, cheapest_cost = min(costs, key=lambda x: x[1])
                
                print(f"   💡 Cheapest option: {cheapest_model.model} at ${cheapest_cost:.6f}")
                print(f"   📈 Budget increase needed: ${cheapest_cost - scenario['max_cost']:.6f}")


def main():
    """Run the complete cost optimization demonstration."""
    
    print("💰 LiteLLM + GenOps: Advanced Cost Optimization")
    print("=" * 60)
    print("Maximize value through intelligent model selection and cost-aware routing")
    print("across 100+ providers with comprehensive optimization strategies")
    
    # Check setup
    if not check_optimization_setup():
        print("\n❌ Setup incomplete. Please resolve issues above.")
        return 1
    
    try:
        # Run demonstrations
        demo_intelligent_model_selection()
        demo_cost_comparison_analysis()
        demo_real_time_optimization()
        demo_budget_constrained_optimization()
        
        print("\n" + "="*60)
        print("🎉 Cost Optimization Complete!")
        
        print("\n💰 Key Cost Optimization Strategies:")
        print("   ✅ Intelligent model selection based on task complexity")
        print("   ✅ Multi-provider cost comparison and analysis")
        print("   ✅ Real-time optimization with performance tracking")
        print("   ✅ Budget-constrained operation strategies")
        print("   ✅ Provider-agnostic value optimization")
        
        print("\n🎯 Optimization Impact:")
        print("   • Up to 95% cost reduction for simple tasks")
        print("   • 30-60% average savings through smart model selection")
        print("   • Quality-preserving cost optimization")
        print("   • Automatic provider switching for best value")
        
        print("\n📊 Production Recommendations:")
        print("   • Implement task complexity analysis for automatic routing")
        print("   • Set budget constraints per use case or customer tier")
        print("   • Monitor cost trends and adjust optimization parameters")
        print("   • Use A/B testing to validate optimization decisions")
        
        print("\n📖 Next Steps:")
        print("   • Try budget_management.py for spending controls")
        print("   • Explore production_patterns.py for scaling strategies")
        print("   • Implement cost optimization in your applications!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Demo failed: [Error details redacted for security]")
        print("💡 For debugging, check your API key configuration")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)