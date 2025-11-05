#!/usr/bin/env python3
"""
📊 GenOps Replicate Basic Tracking - Phase 2 (10-15 minutes)

Manual adapter usage for team cost attribution and governance.
Learn how to explicitly track costs and attribute them to teams and projects.

This example demonstrates the core GenOps patterns for team-based cost 
attribution and project tracking across different Replicate model types.

Requirements:
- REPLICATE_API_TOKEN environment variable  
- pip install replicate genops-ai

Key Learnings:
- Manual GenOpsReplicateAdapter usage patterns
- Team/project/customer cost attribution
- Multi-model cost comparison and optimization
- Real-time cost monitoring and budgeting
"""

import os
import time
from typing import Dict, Any, List

def demonstrate_basic_tracking():
    """Show basic GenOps adapter usage for cost tracking."""
    
    print("📊 GenOps Replicate Basic Tracking Demo")
    print("=" * 50)
    
    # Step 1: Create GenOps adapter
    print("Step 1: Creating GenOps Replicate adapter...")
    from genops.providers.replicate import GenOpsReplicateAdapter
    
    adapter = GenOpsReplicateAdapter()
    print("✅ GenOpsReplicateAdapter initialized")
    print()
    
    # Step 2: Basic text generation with cost tracking
    print("Step 2: Text generation with cost tracking...")
    try:
        response = adapter.text_generation(
            model="meta/llama-2-7b-chat",
            prompt="Explain AI cost optimization in simple terms",
            max_tokens=100,
            temperature=0.7
        )
        
        print(f"   💬 Response: {response.content[:100]}...")
        print(f"   💰 Cost: ${response.cost_usd:.6f}")
        print(f"   ⏱️  Latency: {response.latency_ms:.0f}ms")
        print(f"   🏷️  Model: {response.model}")
        print()
        
    except Exception as e:
        print(f"❌ Error in text generation: {e}")
        return False
    
    # Step 3: Image generation with cost tracking
    print("Step 3: Image generation with cost tracking...")
    try:
        response = adapter.image_generation(
            model="black-forest-labs/flux-schnell",
            prompt="A simple chart showing cost optimization trends",
            num_images=1
        )
        
        print(f"   🎨 Images: {1} generated")
        print(f"   💰 Cost: ${response.cost_usd:.6f}")
        print(f"   ⏱️  Latency: {response.latency_ms:.0f}ms") 
        print(f"   🏷️  Model: {response.model}")
        print()
        
    except Exception as e:
        print(f"❌ Error in image generation: {e}")
        return False
    
    print("✅ Basic tracking demonstration complete!")
    return True

def demonstrate_team_attribution():
    """Show team-based cost attribution patterns."""
    
    print("\n🏛️ Team Attribution & Governance")
    print("=" * 50)
    
    from genops.providers.replicate import GenOpsReplicateAdapter
    adapter = GenOpsReplicateAdapter()
    
    # Simulate different teams using AI services
    teams_data = [
        {
            "name": "marketing-team",
            "project": "campaign-optimization", 
            "customer": "internal-marketing",
            "task": "Generate marketing copy for AI product launch",
            "model": "meta/llama-2-7b-chat"
        },
        {
            "name": "design-team",
            "project": "brand-assets",
            "customer": "internal-design", 
            "task": "Create product icons and marketing visuals",
            "model": "black-forest-labs/flux-schnell"
        },
        {
            "name": "research-team",
            "project": "market-analysis",
            "customer": "internal-research",
            "task": "Analyze AI market trends and opportunities", 
            "model": "meta/llama-2-7b-chat"
        }
    ]
    
    team_costs = {}
    
    print("Step 4: Tracking costs by team and project...")
    
    for team_data in teams_data:
        try:
            print(f"\n   👥 {team_data['name']} - {team_data['project']}")
            
            if "image" in team_data['task'].lower() or "visual" in team_data['task'].lower():
                # Image generation task
                response = adapter.image_generation(
                    model=team_data['model'],
                    prompt=team_data['task'],
                    num_images=1,
                    team=team_data['name'],
                    project=team_data['project'], 
                    customer_id=team_data['customer'],
                    environment="development"
                )
            else:
                # Text generation task
                response = adapter.text_generation(
                    model=team_data['model'],
                    prompt=team_data['task'],
                    max_tokens=80,
                    team=team_data['name'],
                    project=team_data['project'],
                    customer_id=team_data['customer'], 
                    environment="development"
                )
            
            team_costs[team_data['name']] = response.cost_usd
            
            print(f"      💰 Cost: ${response.cost_usd:.6f}")
            print(f"      ⏱️  Time: {response.latency_ms:.0f}ms")
            print(f"      📊 Attribution: {team_data['name']} → {team_data['project']}")
            
        except Exception as e:
            print(f"      ❌ Error for {team_data['name']}: {e}")
            continue
    
    # Cost summary by team
    if team_costs:
        print(f"\n📊 TEAM COST SUMMARY:")
        total_cost = sum(team_costs.values())
        print(f"   💰 Total Spend: ${total_cost:.6f}")
        
        for team, cost in sorted(team_costs.items(), key=lambda x: x[1], reverse=True):
            percentage = (cost / total_cost) * 100 if total_cost > 0 else 0
            print(f"   • {team}: ${cost:.6f} ({percentage:.1f}%)")
    
    return True

def demonstrate_model_comparison():
    """Compare costs across different model types and sizes."""
    
    print("\n🔬 Model Cost Comparison")
    print("=" * 50)
    
    from genops.providers.replicate import GenOpsReplicateAdapter
    adapter = GenOpsReplicateAdapter()
    
    # Test different models for same task type
    text_models = [
        "meta/llama-2-7b-chat",   # Smaller model
        "meta/llama-2-13b-chat",  # Medium model
        "meta/llama-2-70b-chat",  # Larger model  
    ]
    
    test_prompt = "Explain machine learning in one sentence"
    model_results = {}
    
    print("Step 5: Comparing text models for cost optimization...")
    
    for model in text_models:
        try:
            print(f"\n   🧠 Testing {model}...")
            start_time = time.time()
            
            response = adapter.text_generation(
                model=model,
                prompt=test_prompt,
                max_tokens=50,
                temperature=0.7,
                team="evaluation-team",
                project="model-comparison"
            )
            
            model_results[model] = {
                "cost": response.cost_usd,
                "latency": response.latency_ms,
                "content": response.content[:50] + "..." if response.content else "No response"
            }
            
            print(f"      💰 Cost: ${response.cost_usd:.6f}")
            print(f"      ⏱️  Latency: {response.latency_ms:.0f}ms")
            print(f"      💬 Quality: {response.content[:50]}...")
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            continue
    
    # Analysis and recommendations
    if len(model_results) > 1:
        print(f"\n📈 MODEL COMPARISON ANALYSIS:")
        
        # Find most cost-effective
        cheapest = min(model_results.items(), key=lambda x: x[1]['cost'])
        fastest = min(model_results.items(), key=lambda x: x[1]['latency'])
        
        print(f"   💰 Most Cost-Effective: {cheapest[0]} (${cheapest[1]['cost']:.6f})")
        print(f"   ⚡ Fastest Response: {fastest[0]} ({fastest[1]['latency']:.0f}ms)")
        
        # Cost efficiency recommendations
        costs = [result['cost'] for result in model_results.values()]
        if max(costs) > min(costs) * 2:  # Significant cost difference
            print(f"   💡 Optimization: {cheapest[0]} offers best value for simple tasks")
    
    return True

def demonstrate_budget_monitoring():
    """Show budget monitoring and cost aggregation."""
    
    print("\n💰 Budget Monitoring & Cost Aggregation")
    print("=" * 50)
    
    from genops.providers.replicate_cost_aggregator import create_replicate_cost_context
    
    try:
        # Create cost context with budget limit
        with create_replicate_cost_context("budget_demo", budget_limit=0.50) as context:
            print("Step 6: Budget-controlled operations...")
            
            from genops.providers.replicate import GenOpsReplicateAdapter
            adapter = GenOpsReplicateAdapter()
            
            # Simulate multiple operations within budget
            operations = [
                ("Content generation", "meta/llama-2-7b-chat", "Write a product description", 30),
                ("Logo creation", "black-forest-labs/flux-schnell", "Company logo design", None),
                ("FAQ creation", "meta/llama-2-7b-chat", "Generate 3 FAQ entries", 60)
            ]
            
            for i, (task_name, model, prompt, max_tokens) in enumerate(operations, 1):
                print(f"\n   📋 Operation {i}: {task_name}")
                
                try:
                    if max_tokens:  # Text task
                        response = adapter.text_generation(
                            model=model,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            team="content-team", 
                            project="budget-demo"
                        )
                        
                        # Add to cost aggregator
                        context.add_operation(
                            model=model,
                            category="text",
                            cost_usd=response.cost_usd,
                            input_tokens=len(prompt) // 4,  # Rough estimate
                            output_tokens=len(str(response.content)) // 4,
                            latency_ms=response.latency_ms
                        )
                    else:  # Image task
                        response = adapter.image_generation(
                            model=model,
                            prompt=prompt,
                            num_images=1,
                            team="design-team",
                            project="budget-demo"
                        )
                        
                        # Add to cost aggregator
                        context.add_operation(
                            model=model,
                            category="image", 
                            cost_usd=response.cost_usd,
                            output_units=1,
                            latency_ms=response.latency_ms
                        )
                    
                    print(f"      ✅ Completed - Cost: ${response.cost_usd:.6f}")
                    
                    # Check budget status
                    summary = context.get_current_summary()
                    budget_used = (summary.total_cost / 0.50) * 100
                    print(f"      📊 Budget used: {budget_used:.1f}% (${summary.total_cost:.6f}/$0.50)")
                    
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    continue
            
            # Final budget summary
            final_summary = context.get_current_summary()
            
            print(f"\n📊 FINAL BUDGET SUMMARY:")
            print(f"   💰 Total Spent: ${final_summary.total_cost:.6f}")
            print(f"   🎯 Budget Limit: $0.50")
            print(f"   📊 Utilization: {(final_summary.total_cost / 0.50) * 100:.1f}%")
            print(f"   🔄 Operations: {final_summary.operation_count}")
            print(f"   🏷️  Models: {len(final_summary.unique_models)}")
            
            if final_summary.optimization_recommendations:
                print(f"   💡 Recommendations:")
                for rec in final_summary.optimization_recommendations[:2]:
                    print(f"      • {rec}")
    
    except Exception as e:
        print(f"❌ Error in budget demo: {e}")
        return False
    
    return True

def main():
    """Main demonstration of basic Replicate tracking patterns."""
    
    print("🚀 GenOps Replicate Basic Tracking Demo")
    print("Learn team attribution, cost comparison, and budget monitoring")
    print()
    
    # Check prerequisites
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ REPLICATE_API_TOKEN not set")
        print("🔧 Setup:")
        print("   1. Get token: https://replicate.com/account/api-tokens")
        print("   2. export REPLICATE_API_TOKEN='r8_your_token_here'")
        return False
    
    success = True
    
    # Run all demonstrations
    success &= demonstrate_basic_tracking()
    success &= demonstrate_team_attribution()
    success &= demonstrate_model_comparison()
    success &= demonstrate_budget_monitoring()
    
    if success:
        print("\n🎉 BASIC TRACKING DEMO COMPLETE!")
        print("=" * 50) 
        print("✅ You now understand:")
        print("   • Manual GenOpsReplicateAdapter usage")
        print("   • Team/project/customer cost attribution")
        print("   • Multi-model cost comparison and optimization")  
        print("   • Budget monitoring and cost aggregation")
        print("   • Real-time optimization recommendations")
        print()
        print("🎯 PHASE 2 MASTERY - Ready for advanced patterns!")
        print()
        print("🚀 NEXT STEPS:")
        print("   → python cost_optimization.py    # Advanced cost intelligence")
        print("   → examples/replicate/README.md  # Complete documentation")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)