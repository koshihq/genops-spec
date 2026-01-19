#!/usr/bin/env python3
"""
SkyRouter Basic Multi-Model Routing with GenOps Governance

This example demonstrates fundamental multi-model routing capabilities with
SkyRouter and GenOps governance. Learn how to route requests across 150+
models with automatic cost tracking, team attribution, and optimization.

Features demonstrated:
- Basic multi-model routing with cost tracking
- Route strategy comparison and optimization
- Team and project cost attribution
- Budget monitoring and alerts
- Route efficiency analysis

Usage:
    export SKYROUTER_API_KEY="your-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    python basic_routing.py

Author: GenOps AI Contributors
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def demonstrate_basic_routing():
    """Demonstrate basic multi-model routing with governance."""
    
    print("🔀 SkyRouter Basic Multi-Model Routing")
    print("=" * 45)
    print()
    
    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter
    except ImportError as e:
        print(f"❌ Error importing GenOps SkyRouter: {e}")
        print("💡 Make sure you're in the project root directory and GenOps is properly installed")
        print("💡 Try: pip install genops[skyrouter]")
        return False
    
    # Configuration
    api_key = os.getenv("SKYROUTER_API_KEY")
    team = os.getenv("GENOPS_TEAM", "basic-routing-team")
    project = os.getenv("GENOPS_PROJECT", "multi-model-demo")
    
    if not api_key:
        print("❌ SKYROUTER_API_KEY environment variable not set")
        print("💡 Set your API key: export SKYROUTER_API_KEY='your-api-key'")
        return False
    
    print(f"🏗️ Configuration:")
    print(f"  🔑 API Key: {api_key[:8]}...")
    print(f"  👥 Team: {team}")
    print(f"  📊 Project: {project}")
    print()
    
    # Initialize adapter
    adapter = GenOpsSkyRouterAdapter(
        skyrouter_api_key=api_key,
        team=team,
        project=project,
        environment="development",
        daily_budget_limit=25.0,  # $25 daily budget for demo
        enable_cost_alerts=True,
        governance_policy="advisory"
    )
    
    print("✅ SkyRouter adapter initialized successfully")
    print()
    
    # Example 1: Single Model Routing
    print("📝 Example 1: Single Model Routing")
    print("-" * 40)
    
    with adapter.track_routing_session("single-model-demo") as session:
        # Route to a specific model with cost tracking
        result1 = session.track_model_call(
            model="gpt-3.5-turbo",
            input_data={
                "prompt": "Explain the benefits of multi-model AI routing in simple terms.",
                "max_tokens": 150
            },
            route_optimization="cost_optimized",
            complexity="simple"
        )
        
        print(f"✅ Single model routing completed:")
        print(f"   🤖 Model: {result1.model}")
        print(f"   🔀 Route: {result1.route}")
        print(f"   💰 Cost: ${result1.total_cost:.4f}")
        print(f"   📊 Tokens: {result1.input_tokens} in, {result1.output_tokens} out")
        print()
    
    # Example 2: Multi-Model Routing with Strategy Comparison
    print("🔀 Example 2: Multi-Model Routing Strategies")
    print("-" * 50)
    
    # Sample request for routing
    sample_request = {
        "prompt": "Write a technical explanation of how machine learning models are deployed in production environments.",
        "requirements": ["technical_depth", "practical_examples", "500_words"]
    }
    
    routing_strategies = ["cost_optimized", "balanced", "latency_optimized"]
    routing_results = {}
    
    for strategy in routing_strategies:
        print(f"🧪 Testing {strategy} routing strategy...")
        
        with adapter.track_routing_session(f"strategy-{strategy}") as session:
            result = session.track_multi_model_routing(
                models=["gpt-4", "claude-3-sonnet", "gemini-pro", "gpt-3.5-turbo"],
                input_data=sample_request,
                routing_strategy=strategy
            )
            
            routing_results[strategy] = result
            
            print(f"   🤖 Selected: {result.model}")
            print(f"   💰 Cost: ${result.total_cost:.4f}")
            print(f"   ⚡ Efficiency: {result.route_efficiency_score:.2f}")
            print(f"   💾 Savings: ${result.optimization_savings:.4f}")
            print()
    
    # Compare routing strategies
    print("📊 Strategy Comparison Summary:")
    print("-" * 35)
    
    for strategy, result in routing_results.items():
        print(f"🔹 {strategy}:")
        print(f"   Model: {result.model}")
        print(f"   Cost: ${result.total_cost:.4f}")
        print(f"   Efficiency: {result.route_efficiency_score:.2f}")
        print()
    
    # Find most cost-effective strategy
    cheapest = min(routing_results.items(), key=lambda x: x[1].total_cost)
    most_efficient = max(routing_results.items(), key=lambda x: x[1].route_efficiency_score)
    
    print(f"🏆 Most cost-effective: {cheapest[0]} (${cheapest[1].total_cost:.4f})")
    print(f"🏆 Most efficient: {most_efficient[0]} (score: {most_efficient[1].route_efficiency_score:.2f})")
    print()
    
    return True

def demonstrate_agent_workflow():
    """Demonstrate multi-agent workflow routing."""
    
    print("🤖 Example 3: Multi-Agent Workflow Routing")
    print("-" * 45)
    
    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter
        
        adapter = GenOpsSkyRouterAdapter(
            team=os.getenv("GENOPS_TEAM", "agent-workflow-team"),
            project=os.getenv("GENOPS_PROJECT", "multi-agent-demo"),
            daily_budget_limit=50.0
        )
        
        # Define a customer support workflow
        workflow_steps = [
            {
                "model": "gpt-3.5-turbo",
                "input": {
                    "task": "intent_classification",
                    "customer_message": "I'm having trouble with my subscription billing"
                },
                "complexity": "simple",
                "optimization": "cost_optimized"
            },
            {
                "model": "claude-3-sonnet", 
                "input": {
                    "task": "solution_generation",
                    "intent": "billing_support",
                    "customer_context": "subscription_issue"
                },
                "complexity": "moderate",
                "optimization": "balanced"
            },
            {
                "model": "gpt-4",
                "input": {
                    "task": "quality_review",
                    "solution": "proposed_billing_solution",
                    "quality_criteria": ["accuracy", "empathy", "completeness"]
                },
                "complexity": "complex",
                "optimization": "reliability_first"
            }
        ]
        
        with adapter.track_routing_session("customer-support-workflow") as session:
            workflow_result = session.track_agent_workflow(
                workflow_name="customer_support_pipeline",
                agent_steps=workflow_steps
            )
            
            print(f"✅ Multi-agent workflow completed:")
            print(f"   🔄 Workflow: {workflow_result.metadata['workflow_name']}")
            print(f"   📈 Steps: {workflow_result.metadata['step_count']}")
            print(f"   🤖 Models used: {', '.join(workflow_result.metadata['models_used'])}")
            print(f"   💰 Total cost: ${workflow_result.total_cost:.4f}")
            print(f"   📊 Cost per step: ${float(workflow_result.total_cost) / len(workflow_steps):.4f}")
            print()
            
            # Show step-by-step breakdown
            print("📋 Step-by-step breakdown:")
            for i, step_cost in enumerate(workflow_result.metadata['step_costs'], 1):
                print(f"   Step {i}: {step_cost['model']} - ${step_cost['cost']:.4f} ({step_cost['optimization']})")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Agent workflow demo failed: {e}")
        return False

def demonstrate_cost_tracking():
    """Demonstrate comprehensive cost tracking and analysis."""
    
    print("💰 Example 4: Cost Tracking and Analysis")
    print("-" * 42)
    
    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter
        
        adapter = GenOpsSkyRouterAdapter(
            team=os.getenv("GENOPS_TEAM", "cost-analysis-team"),
            project=os.getenv("GENOPS_PROJECT", "cost-tracking-demo"),
            daily_budget_limit=30.0
        )
        
        # Simulate various operations for cost analysis
        operations = [
            {
                "type": "content_generation",
                "models": ["gpt-4", "claude-3-sonnet"],
                "strategy": "balanced"
            },
            {
                "type": "code_review", 
                "models": ["gpt-4", "claude-3-opus"],
                "strategy": "reliability_first"
            },
            {
                "type": "data_analysis",
                "models": ["gpt-3.5-turbo", "gemini-pro", "llama-2"],
                "strategy": "cost_optimized"
            },
            {
                "type": "customer_support",
                "models": ["gpt-3.5-turbo", "claude-3-haiku"],
                "strategy": "latency_optimized"
            }
        ]
        
        total_operations = 0
        
        for operation in operations:
            print(f"🔄 Processing {operation['type']} operation...")
            
            with adapter.track_routing_session(f"{operation['type']}-session") as session:
                result = session.track_multi_model_routing(
                    models=operation["models"],
                    input_data={
                        "task": operation["type"],
                        "complexity": "varies"
                    },
                    routing_strategy=operation["strategy"]
                )
                
                print(f"   ✅ {result.model} selected, cost: ${result.total_cost:.4f}")
                total_operations += 1
        
        print()
        
        # Get cost summary
        summary = adapter.cost_aggregator.get_summary()
        
        print("📊 Cost Analysis Summary:")
        print(f"   💰 Total cost: ${summary.total_cost:.4f}")
        print(f"   📈 Operations: {summary.total_operations}")
        print(f"   📉 Avg cost/op: ${summary.average_cost_per_operation:.4f}")
        print(f"   💾 Total savings: ${summary.optimization_savings:.4f}")
        print()
        
        # Cost breakdown by model
        if summary.cost_by_model:
            print("🤖 Cost by Model:")
            for model, cost in sorted(summary.cost_by_model.items(), 
                                    key=lambda x: x[1], reverse=True):
                percentage = (cost / summary.total_cost) * 100 if summary.total_cost > 0 else 0
                print(f"   • {model}: ${cost:.4f} ({percentage:.1f}%)")
            print()
        
        # Cost breakdown by routing strategy
        if summary.cost_by_route:
            print("🔀 Cost by Routing Strategy:")
            for route, cost in sorted(summary.cost_by_route.items(), 
                                    key=lambda x: x[1], reverse=True):
                percentage = (cost / summary.total_cost) * 100 if summary.total_cost > 0 else 0
                print(f"   • {route}: ${cost:.4f} ({percentage:.1f}%)")
            print()
        
        # Budget status
        budget_status = adapter.cost_aggregator.check_budget_status()
        current_cost = budget_status["current_daily_cost"]
        budget_limit = budget_status["daily_budget_limit"]
        
        if budget_limit:
            utilization = (current_cost / budget_limit) * 100
            print(f"📊 Budget Utilization: {utilization:.1f}% (${current_cost:.4f}/${budget_limit:.2f})")
            
            if utilization > 80:
                print("⚠️  Warning: High budget utilization detected!")
            elif utilization > 50:
                print("💡 Info: Moderate budget utilization")
            else:
                print("✅ Good: Low budget utilization")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Cost tracking demo failed: {e}")
        return False

def show_optimization_recommendations():
    """Show cost optimization recommendations."""
    
    print("💡 Cost Optimization Recommendations")
    print("-" * 40)
    
    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter
        
        adapter = GenOpsSkyRouterAdapter(
            team=os.getenv("GENOPS_TEAM", "optimization-team"),
            project=os.getenv("GENOPS_PROJECT", "recommendations-demo")
        )
        
        # Get optimization recommendations
        recommendations = adapter.cost_aggregator.get_cost_optimization_recommendations()
        
        if recommendations:
            print("🚀 Personalized recommendations based on your usage:")
            print()
            
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"{i}. **{rec['title']}**")
                print(f"   💰 Potential savings: ${rec['potential_savings']:.2f}/month")
                print(f"   🛠️  Effort level: {rec['effort_level']}")
                print(f"   🎯 Priority: {rec['priority_score']:.0f}/100")
                print(f"   📝 Type: {rec['optimization_type']}")
                print()
        else:
            print("🎉 Great! No specific optimization recommendations at this time.")
            print("Your routing patterns appear to be well-optimized.")
            print()
        
        # General optimization tips
        print("💡 General Multi-Model Optimization Tips:")
        print("1. Use 'cost_optimized' strategy for batch processing")
        print("2. Use 'latency_optimized' strategy for real-time applications")
        print("3. Use 'balanced' strategy for general production workloads")
        print("4. Consider cheaper models (e.g., GPT-3.5) for simple tasks")
        print("5. Implement caching for frequently repeated requests")
        print("6. Monitor route efficiency scores to identify suboptimal routing")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization recommendations failed: {e}")
        return False

def main():
    """Main execution function."""
    
    print("🔀 SkyRouter + GenOps Basic Multi-Model Routing Demo")
    print("=" * 60)
    print()
    
    print("This example demonstrates fundamental multi-model routing capabilities")
    print("with automatic cost tracking, governance, and optimization across 150+ models.")
    print()
    
    # Check prerequisites
    api_key = os.getenv("SKYROUTER_API_KEY")
    if not api_key:
        print("❌ Missing required environment variables:")
        print("   SKYROUTER_API_KEY - Your SkyRouter API key")
        print()
        print("💡 Set up your environment:")
        print("   export SKYROUTER_API_KEY='your-api-key'")
        print("   export GENOPS_TEAM='your-team'")
        print("   export GENOPS_PROJECT='your-project'")
        print()
        print("🔗 Get your API key from: https://skyrouter.ai")
        return
    
    try:
        # Run demonstrations
        success = True
        
        # Basic routing demonstration
        if success:
            success = demonstrate_basic_routing()
        
        # Agent workflow demonstration
        if success:
            success = demonstrate_agent_workflow()
        
        # Cost tracking demonstration
        if success:
            success = demonstrate_cost_tracking()
        
        # Show optimization recommendations
        if success:
            show_optimization_recommendations()
        
        if success:
            print("🎉 All basic routing demonstrations completed successfully!")
            print()
            print("🚀 **Next Steps:**")
            print("1. Try auto_instrumentation.py for zero-code integration")
            print("2. Explore route_optimization.py for advanced optimization")
            print("3. Check out agent_workflows.py for complex multi-agent patterns")
            print("4. Review enterprise_patterns.py for production deployment")
            print()
            print("📖 **Learn More:**")
            print("• Quickstart Guide: docs/skyrouter-quickstart.md")
            print("• Complete Guide: docs/integrations/skyrouter.md")
            print("• Performance Guide: docs/skyrouter-performance-benchmarks.md")
        
    except KeyboardInterrupt:
        print()
        print("👋 Demo cancelled.")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print()
        print("🔧 Troubleshooting tips:")
        print("1. Verify your SKYROUTER_API_KEY is correct")
        print("2. Check your internet connection")
        print("3. Ensure GenOps is properly installed: pip install genops[skyrouter]")

if __name__ == "__main__":
    main()