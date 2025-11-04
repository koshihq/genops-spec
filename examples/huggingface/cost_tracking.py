#!/usr/bin/env python3
"""
Hugging Face Multi-Provider Cost Tracking Example

This example demonstrates unified cost tracking across multiple AI providers
accessible through Hugging Face, including OpenAI, Anthropic, and Hub models.

Example usage:
    python multi_provider_costs.py

Features demonstrated:
- Multi-provider cost aggregation
- Provider comparison and optimization
- Unified governance across providers
- Cost attribution and reporting
- Budget-aware operations
"""

import sys
import os
import logging
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime

# Add src to path for development  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OperationCost:
    """Track cost details for a single AI operation."""
    operation_id: str
    provider: str
    model: str
    task: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    governance_attrs: Dict[str, str] = field(default_factory=dict)

@dataclass  
class MultiProviderSession:
    """Track costs across multiple providers in a single session."""
    session_id: str
    operations: List[OperationCost] = field(default_factory=list)
    
    @property
    def total_cost(self) -> float:
        return sum(op.cost for op in self.operations)
    
    @property
    def cost_by_provider(self) -> Dict[str, float]:
        costs = {}
        for op in self.operations:
            costs[op.provider] = costs.get(op.provider, 0) + op.cost
        return costs
    
    @property
    def cost_by_model(self) -> Dict[str, float]:
        costs = {}
        for op in self.operations:
            costs[op.model] = costs.get(op.model, 0) + op.cost
        return costs
    
    def get_cost_breakdown(self) -> Dict[str, any]:
        return {
            "total_cost": self.total_cost,
            "cost_by_provider": self.cost_by_provider,
            "cost_by_model": self.cost_by_model,
            "operations_count": len(self.operations),
            "providers_used": list(set(op.provider for op in self.operations)),
            "models_used": list(set(op.model for op in self.operations))
        }


def demonstrate_multi_provider_operations():
    """Demonstrate operations across multiple providers with unified cost tracking."""
    
    print("🌐 Multi-Provider Operations Demo")
    print("="*50)
    print("Demonstrating unified cost tracking across OpenAI, Anthropic, and Hub models")
    print()
    
    try:
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        from genops.providers.huggingface_pricing import calculate_huggingface_cost
        
        adapter = GenOpsHuggingFaceAdapter()
        session = MultiProviderSession(session_id="multi-provider-demo-2024")
        
        # Define test operations across different providers
        operations_to_test = [
            {
                "name": "OpenAI Text Generation",
                "model": "gpt-3.5-turbo",
                "prompt": "Write a brief product description for an AI-powered analytics platform.",
                "task": "text-generation",
                "governance": {
                    "team": "product-team",
                    "project": "marketing-copy",
                    "customer_id": "saas-client-001"
                }
            },
            {
                "name": "Anthropic Chat Completion",
                "model": "claude-3-haiku",
                "prompt": "Provide customer support response for a billing inquiry.",
                "task": "chat-completion", 
                "governance": {
                    "team": "support-team",
                    "project": "customer-service-ai",
                    "customer_id": "support-internal"
                }
            },
            {
                "name": "Hub Model Text Generation",
                "model": "microsoft/DialoGPT-medium",
                "prompt": "Generate a casual conversation starter for a networking event.",
                "task": "text-generation",
                "governance": {
                    "team": "events-team",
                    "project": "networking-bot", 
                    "customer_id": "events-client-789"
                }
            },
            {
                "name": "Hub Model Embeddings",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "prompt": "Transform customer feedback into searchable embeddings",
                "task": "feature-extraction",
                "governance": {
                    "team": "analytics-team",
                    "project": "feedback-analysis",
                    "customer_id": "analytics-internal"
                }
            }
        ]
        
        print("📊 Running operations across multiple providers...")
        print()
        
        for i, operation in enumerate(operations_to_test, 1):
            print(f"   {i}. {operation['name']}:")
            print(f"      Model: {operation['model']}")
            
            # Detect provider for cost calculation
            provider = adapter.detect_provider_for_model(operation['model'])
            print(f"      Provider: {provider}")
            
            # Estimate tokens (in real usage, these would come from actual API calls)
            estimated_input_tokens = len(operation['prompt'].split()) * 4  # Rough estimate
            estimated_output_tokens = 100  # Typical response size
            
            # Calculate cost
            try:
                cost = calculate_huggingface_cost(
                    provider=provider,
                    model=operation['model'],
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    task=operation['task']
                )
                
                print(f"      Tokens: {estimated_input_tokens} in, {estimated_output_tokens} out")
                print(f"      Cost: ${cost:.6f}")
                
                # Record operation
                op_cost = OperationCost(
                    operation_id=f"op-{i:03d}",
                    provider=provider,
                    model=operation['model'],
                    task=operation['task'],
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    cost=cost,
                    governance_attrs=operation['governance']
                )
                session.operations.append(op_cost)
                
                print(f"      ✅ Cost tracked for {operation['governance']['team']}")
                
            except Exception as e:
                print(f"      ⚠️ Cost calculation failed: {e}")
            
            print()
        
        # Try actual API calls (may fail due to rate limits/connectivity)
        print("🚀 Attempting live API calls (may be limited by rate limits)...")
        live_successes = 0
        
        for operation in operations_to_test[:2]:  # Just try first 2 to avoid rate limits
            try:
                if operation['task'] == 'feature-extraction':
                    response = adapter.feature_extraction(
                        inputs=operation['prompt'],
                        model=operation['model'],
                        **operation['governance']
                    )
                    live_successes += 1
                    print(f"   ✅ {operation['name']} succeeded")
                    
                else:
                    response = adapter.text_generation(
                        prompt=operation['prompt'],
                        model=operation['model'], 
                        max_new_tokens=50,
                        **operation['governance']
                    )
                    live_successes += 1
                    print(f"   ✅ {operation['name']} succeeded")
                    print(f"      Response: {str(response)[:80]}...")
                    
            except Exception as e:
                print(f"   ⚠️ {operation['name']} failed: {str(e)[:60]}...")
        
        print(f"\n   Live API Success Rate: {live_successes}/{min(2, len(operations_to_test))}")
        print()
        
        return session
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return None


def analyze_cost_breakdown(session: MultiProviderSession):
    """Analyze and display cost breakdown across providers."""
    
    print("💰 Cost Analysis and Breakdown")
    print("="*40)
    
    breakdown = session.get_cost_breakdown()
    
    print(f"📊 Session Summary:")
    print(f"   Total Operations: {breakdown['operations_count']}")
    print(f"   Providers Used: {len(breakdown['providers_used'])}")
    print(f"   Models Used: {len(breakdown['models_used'])}")
    print(f"   Total Cost: ${breakdown['total_cost']:.6f}")
    print()
    
    # Cost by provider
    print("🏢 Cost by Provider:")
    for provider, cost in breakdown['cost_by_provider'].items():
        percentage = (cost / breakdown['total_cost']) * 100 if breakdown['total_cost'] > 0 else 0
        provider_icon = {
            'openai': '🤖',
            'anthropic': '🧠', 
            'huggingface_hub': '🤗',
            'cohere': '🔮',
            'mistral': '🌟'
        }.get(provider, '🔧')
        
        print(f"   {provider_icon} {provider:15} → ${cost:8.6f} ({percentage:5.1f}%)")
    print()
    
    # Cost by model
    print("🎯 Cost by Model:")
    model_costs = sorted(breakdown['cost_by_model'].items(), key=lambda x: x[1], reverse=True)
    for model, cost in model_costs:
        percentage = (cost / breakdown['total_cost']) * 100 if breakdown['total_cost'] > 0 else 0
        print(f"   📱 {model[:30]:30} → ${cost:8.6f} ({percentage:5.1f}%)")
    print()
    
    # Team attribution
    print("👥 Cost Attribution by Team:")
    team_costs = {}
    for op in session.operations:
        team = op.governance_attrs.get('team', 'unknown')
        team_costs[team] = team_costs.get(team, 0) + op.cost
        
    for team, cost in sorted(team_costs.items(), key=lambda x: x[1], reverse=True):
        percentage = (cost / breakdown['total_cost']) * 100 if breakdown['total_cost'] > 0 else 0
        print(f"   👥 {team:15} → ${cost:8.6f} ({percentage:5.1f}%)")
    print()
    
    # Customer billing
    print("🏢 Customer Billing Attribution:")
    customer_costs = {}
    for op in session.operations:
        customer = op.governance_attrs.get('customer_id', 'internal')
        customer_costs[customer] = customer_costs.get(customer, 0) + op.cost
        
    for customer, cost in sorted(customer_costs.items(), key=lambda x: x[1], reverse=True):
        percentage = (cost / breakdown['total_cost']) * 100 if breakdown['total_cost'] > 0 else 0
        print(f"   🏢 {customer[:20]:20} → ${cost:8.6f} ({percentage:5.1f}%)")
    print()


def demonstrate_cost_optimization():
    """Show cost optimization strategies across providers."""
    
    print("🎯 Cost Optimization Strategies")
    print("="*40)
    print("Demonstrating intelligent model selection for cost optimization:")
    print()
    
    try:
        from genops.providers.huggingface_pricing import (
            compare_model_costs,
            get_cost_optimization_suggestions
        )
        
        # Compare costs for similar tasks across providers
        print("💡 Model Cost Comparison for Similar Tasks:")
        print()
        
        # Text generation task comparison
        text_models = [
            "gpt-3.5-turbo",                    # OpenAI
            "claude-3-haiku",                   # Anthropic
            "microsoft/DialoGPT-medium",        # Hugging Face Hub
            "mistral-7b-instruct"               # Mistral
        ]
        
        print("   📝 Text Generation (1000 input, 500 output tokens):")
        text_comparison = compare_model_costs(text_models, input_tokens=1000, output_tokens=500)
        
        cheapest_cost = min(info['cost'] for info in text_comparison.values())
        
        for model, info in text_comparison.items():
            cost_tier = "💰" if info['cost'] > cheapest_cost * 3 else "💛" if info['cost'] > cheapest_cost * 1.5 else "💚"
            savings = ((info['cost'] - cheapest_cost) / cheapest_cost * 100) if cheapest_cost > 0 else 0
            
            print(f"      {cost_tier} {model[:35]:35} → ${info['cost']:8.6f} ({info['relative_cost']:4.1f}x)")
            if savings > 0:
                print(f"         💸 ${info['cost'] - cheapest_cost:8.6f} more expensive ({savings:+5.1f}%)")
        print()
        
        # Embedding task comparison  
        embedding_models = [
            "text-embedding-ada-002",                    # OpenAI
            "sentence-transformers/all-MiniLM-L6-v2",    # Hugging Face Hub
            "embed-english-v3.0"                        # Cohere
        ]
        
        print("   🔍 Embeddings/Feature Extraction (1000 input tokens):")
        embedding_comparison = compare_model_costs(
            embedding_models, 
            input_tokens=1000, 
            output_tokens=0, 
            task="feature-extraction"
        )
        
        cheapest_embedding = min(info['cost'] for info in embedding_comparison.values())
        
        for model, info in embedding_comparison.items():
            cost_tier = "💰" if info['cost'] > cheapest_embedding * 2 else "💚"
            print(f"      {cost_tier} {model[:35]:35} → ${info['cost']:8.6f} ({info['relative_cost']:4.1f}x)")
        print()
        
        # Cost optimization suggestions
        print("🧠 Intelligent Cost Optimization Suggestions:")
        
        expensive_model = "gpt-4"  # Example expensive model
        suggestions = get_cost_optimization_suggestions(expensive_model, "text-generation")
        
        print(f"   Current model: {suggestions['current_model']['model']}")
        print(f"   Current cost: ${suggestions['current_model']['cost_per_1k']['input']:.6f} per 1K input tokens")
        print()
        
        print("   💡 Optimization recommendations:")
        for tip in suggestions['optimization_tips']:
            print(f"      • {tip}")
        print()
        
        if suggestions['alternatives']:
            print("   🔄 Alternative models:")
            for alt in suggestions['alternatives'][:3]:  # Show top 3 alternatives
                savings = alt.get('savings', 0)
                print(f"      💚 {alt['model'][:30]:30} → {savings:5.1f}% cost savings")
        
        return True
        
    except ImportError as e:
        print(f"❌ Cost optimization unavailable: {e}")
        return False


def demonstrate_budget_aware_operations():
    """Show budget-aware operation strategies."""
    
    print("💳 Budget-Aware Operations")
    print("="*35)
    print("Demonstrating operations that respect budget constraints:")
    print()
    
    # Simulated budget constraints
    budgets = {
        "product-team": 10.00,      # $10 daily budget
        "support-team": 25.00,      # $25 daily budget  
        "analytics-team": 5.00,     # $5 daily budget
    }
    
    # Current usage (simulated)
    current_usage = {
        "product-team": 7.50,       # $7.50 used
        "support-team": 18.75,      # $18.75 used
        "analytics-team": 4.20,     # $4.20 used
    }
    
    print("📊 Budget Status:")
    for team in budgets:
        budget = budgets[team]
        used = current_usage[team]
        remaining = budget - used
        usage_pct = (used / budget) * 100
        
        status_icon = "🔴" if remaining < 1 else "🟡" if usage_pct > 75 else "🟢"
        
        print(f"   {status_icon} {team:15} → ${used:6.2f} / ${budget:6.2f} ({usage_pct:5.1f}%) - ${remaining:6.2f} remaining")
    print()
    
    # Budget-aware model selection
    print("🎯 Budget-Aware Model Selection:")
    
    tasks_to_consider = [
        {
            "team": "product-team",
            "task": "Generate product feature description (200 tokens expected)",
            "estimated_tokens": 200,
            "models_to_consider": ["gpt-4", "gpt-3.5-turbo", "microsoft/DialoGPT-medium"]
        },
        {
            "team": "support-team", 
            "task": "Customer support response (150 tokens expected)",
            "estimated_tokens": 150,
            "models_to_consider": ["claude-3-opus", "claude-3-haiku", "microsoft/DialoGPT-medium"]
        },
        {
            "team": "analytics-team",
            "task": "Text embeddings for analysis (500 tokens)",
            "estimated_tokens": 500,
            "models_to_consider": ["text-embedding-ada-002", "sentence-transformers/all-MiniLM-L6-v2"]
        }
    ]
    
    try:
        from genops.providers.huggingface_pricing import calculate_huggingface_cost
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        adapter = GenOpsHuggingFaceAdapter()
        
        for task in tasks_to_consider:
            team = task['team']
            remaining_budget = budgets[team] - current_usage[team]
            
            print(f"   👥 {team} (${remaining_budget:.2f} remaining):")
            print(f"      Task: {task['task']}")
            
            # Evaluate models within budget
            affordable_models = []
            
            for model in task['models_to_consider']:
                provider = adapter.detect_provider_for_model(model)
                estimated_cost = calculate_huggingface_cost(
                    provider=provider,
                    model=model,
                    input_tokens=task['estimated_tokens'],
                    output_tokens=task['estimated_tokens'] // 2,  # Estimate output
                    task="text-generation"
                )
                
                within_budget = estimated_cost <= remaining_budget
                status = "✅" if within_budget else "❌"
                budget_indicator = "WITHIN BUDGET" if within_budget else "OVER BUDGET"
                
                print(f"         {status} {model[:30]:30} → ${estimated_cost:.6f} ({budget_indicator})")
                
                if within_budget:
                    affordable_models.append((model, estimated_cost))
            
            if affordable_models:
                # Recommend cheapest available option
                cheapest = min(affordable_models, key=lambda x: x[1])
                print(f"         💡 Recommended: {cheapest[0]} (${cheapest[1]:.6f})")
            else:
                print(f"         ⚠️  All models over budget - consider cost optimization")
            
            print()
        
        print("✅ Budget-aware selection helps teams stay within cost constraints")
        print("✅ Real-time budget tracking enables proactive cost management")
        print()
        
        return True
        
    except ImportError:
        print("❌ Budget analysis unavailable - check installation")
        return False


def main():
    """Main demonstration function."""
    
    print("Welcome to the Multi-Provider Cost Tracking Demo!")
    print()
    print("This example demonstrates comprehensive cost tracking and optimization")
    print("across multiple AI providers accessible through Hugging Face.")
    print()
    
    success_count = 0
    
    # Run multi-provider operations demo
    print("🚀 Running Multi-Provider Operations Demo...")
    session = demonstrate_multi_provider_operations()
    if session and len(session.operations) > 0:
        success_count += 1
        print("✅ Multi-provider operations demo completed successfully")
        print()
        
        # Analyze the results
        analyze_cost_breakdown(session)
        print("-" * 60)
    else:
        print("⚠️ Multi-provider operations demo had issues")
        print()
    
    # Cost optimization demo
    print("🚀 Running Cost Optimization Demo...")
    if demonstrate_cost_optimization():
        success_count += 1
        print("✅ Cost optimization demo completed successfully")
    else:
        print("⚠️ Cost optimization demo had issues")
    print("-" * 60)
    
    # Budget-aware operations demo
    print("🚀 Running Budget-Aware Operations Demo...")
    if demonstrate_budget_aware_operations():
        success_count += 1
        print("✅ Budget-aware operations demo completed successfully")
    else:
        print("⚠️ Budget-aware operations demo had issues")
    print("-" * 60)
    print()
    
    # Summary
    if success_count >= 2:
        print("🎉 Multi-Provider Cost Tracking Demo Completed Successfully!")
        print()
        print("🚀 Key Takeaways:")
        print("   ✅ Unified cost tracking across OpenAI, Anthropic, and Hub models")
        print("   ✅ Real-time provider detection and cost calculation")
        print("   ✅ Team and customer cost attribution for billing")
        print("   ✅ Cost optimization recommendations")
        print("   ✅ Budget-aware operation strategies")
        print()
        print("🚀 Next Steps:")
        print("   1. Set up OpenTelemetry export for production cost tracking")
        print("   2. Implement budget alerts and enforcement policies")
        print("   3. Try ai_task_examples.py for comprehensive task coverage")
        print("   4. Explore production_patterns.py for enterprise deployment")
        
    else:
        print("⚠️ Multi-provider demo encountered multiple issues")
        print("   Check setup_validation.py and internet connectivity")
    
    return 0 if success_count >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())