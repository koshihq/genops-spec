#!/usr/bin/env python3
"""
SkyRouter Zero-Code Auto-Instrumentation Example

This example demonstrates how to add GenOps governance to existing SkyRouter
applications with zero code changes using auto-instrumentation. Perfect for
teams wanting to add cost tracking and governance to existing multi-model
routing without modifying their current codebase.

Features demonstrated:
- Zero-code auto-instrumentation setup
- Automatic governance for existing SkyRouter code
- Transparent cost tracking and attribution
- Budget monitoring without code changes
- Easy enable/disable of governance

Usage:
    export SKYROUTER_API_KEY="your-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    python auto_instrumentation.py

Author: GenOps AI Contributors
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def demonstrate_auto_instrumentation():
    """Demonstrate zero-code auto-instrumentation."""
    
    print("🚀 SkyRouter Zero-Code Auto-Instrumentation")
    print("=" * 50)
    print()
    
    print("This example shows how to add governance to existing SkyRouter code")
    print("without making ANY changes to your current application logic.")
    print()
    
    # Step 1: Show existing code (before governance)
    print("📝 Step 1: Your Existing SkyRouter Code")
    print("-" * 42)
    print()
    
    print("```python")
    print("# Your existing SkyRouter application code")
    print("import skyrouter")
    print("")
    print("client = skyrouter.Client(api_key='your-api-key')")
    print("")
    print("# Multi-model routing")
    print("response = client.route_to_best_model(")
    print("    candidates=['gpt-4', 'claude-3-sonnet', 'gemini-pro'],")
    print("    prompt='Your application prompt',")
    print("    routing_strategy='balanced'")
    print(")")
    print("")
    print("# Agent workflows")
    print("result = client.run_agent_workflow(")
    print("    workflow_name='customer_support',")
    print("    steps=[...]")
    print(")")
    print("```")
    print()
    
    # Step 2: Enable auto-instrumentation
    print("🔧 Step 2: Add Auto-Instrumentation (Just 2 Lines!)")
    print("-" * 55)
    print()
    
    try:
        from genops.providers.skyrouter import auto_instrument
        
        # Configuration
        api_key = os.getenv("SKYROUTER_API_KEY")
        team = os.getenv("GENOPS_TEAM", "auto-instrumentation-team")
        project = os.getenv("GENOPS_PROJECT", "zero-code-demo")
        
        print("```python")
        print("# Add these 2 lines at the top of your file:")
        print("from genops.providers.skyrouter import auto_instrument")
        print("auto_instrument(team='{}', project='{}')".format(team, project))
        print("")
        print("# Your existing code stays EXACTLY the same!")
        print("import skyrouter")
        print("# ... rest of your code unchanged ...")
        print("```")
        print()
        
        # Actually enable auto-instrumentation
        adapter = auto_instrument(
            skyrouter_api_key=api_key,
            team=team,
            project=project,
            daily_budget_limit=20.0,
            enable_cost_alerts=True,
            governance_policy="advisory"
        )
        
        print("✅ Auto-instrumentation enabled successfully!")
        print(f"   👥 Team: {team}")
        print(f"   📊 Project: {project}")
        print(f"   💰 Daily budget: $20.00")
        print(f"   🔧 Policy: advisory")
        print()
        
    except ImportError as e:
        print(f"❌ Error importing GenOps SkyRouter: {e}")
        print("💡 Make sure GenOps is installed: pip install genops[skyrouter]")
        return False
    
    # Step 3: Simulate existing application running
    print("🎯 Step 3: Your Application Runs With Automatic Governance")
    print("-" * 60)
    print()
    
    print("Now when your existing code runs, it automatically includes:")
    print("• 💰 Cost tracking for all multi-model routing operations")
    print("• 👥 Team and project attribution")
    print("• 📊 Budget monitoring and alerts")
    print("• 🔀 Route optimization insights")
    print("• 📈 Performance metrics across all models")
    print()
    
    # Simulate some existing application operations
    print("🧪 Simulating your existing application operations...")
    print()
    
    try:
        # Simulate existing SkyRouter operations
        operations = [
            {
                "operation": "Multi-model content generation",
                "models": ["gpt-4", "claude-3-sonnet"],
                "strategy": "balanced"
            },
            {
                "operation": "Customer support routing",
                "models": ["gpt-3.5-turbo", "claude-3-haiku"],
                "strategy": "latency_optimized"
            },
            {
                "operation": "Code review workflow",
                "models": ["gpt-4", "claude-3-opus"],
                "strategy": "reliability_first"
            }
        ]
        
        total_cost = 0
        
        for i, op in enumerate(operations, 1):
            print(f"📋 Operation {i}: {op['operation']}")
            
            # Simulate the operation with automatic tracking
            with adapter.track_routing_session(f"auto-instrumented-{i}") as session:
                result = session.track_multi_model_routing(
                    models=op["models"],
                    input_data={
                        "operation": op["operation"],
                        "auto_instrumented": True
                    },
                    routing_strategy=op["strategy"]
                )
                
                print(f"   🤖 Selected model: {result.model}")
                print(f"   🔀 Route: {result.route}")
                print(f"   💰 Cost: ${result.total_cost:.4f}")
                print(f"   ⚡ Efficiency: {result.route_efficiency_score:.2f}")
                print(f"   📊 Governance: ✅ Automatically applied")
                
                total_cost += float(result.total_cost)
                print()
        
        print(f"📊 **Session Summary:**")
        print(f"   💰 Total cost: ${total_cost:.4f}")
        print(f"   🔄 Operations: {len(operations)}")
        print(f"   📉 Avg cost/operation: ${total_cost / len(operations):.4f}")
        print(f"   🎯 All operations automatically governed!")
        print()
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False
    
    return True

def demonstrate_enable_disable():
    """Demonstrate enabling and disabling auto-instrumentation."""
    
    print("🔄 Step 4: Easy Enable/Disable Control")
    print("-" * 38)
    print()
    
    try:
        from genops.providers.skyrouter import auto_instrument, restore_skyrouter, get_current_adapter
        
        # Check current status
        current_adapter = get_current_adapter()
        if current_adapter:
            print("✅ Auto-instrumentation is currently ENABLED")
            print(f"   👥 Team: {current_adapter.governance_attrs.team}")
            print(f"   📊 Project: {current_adapter.governance_attrs.project}")
        else:
            print("❌ Auto-instrumentation is currently DISABLED")
        print()
        
        # Show how to disable
        print("🔧 To disable auto-instrumentation:")
        print("```python")
        print("from genops.providers.skyrouter import restore_skyrouter")
        print("restore_skyrouter()  # Disables governance, returns to normal SkyRouter")
        print("```")
        print()
        
        # Show how to re-enable
        print("🔧 To re-enable with different settings:")
        print("```python")
        print("from genops.providers.skyrouter import auto_instrument")
        print("auto_instrument(")
        print("    team='new-team',")
        print("    project='new-project',")
        print("    daily_budget_limit=100.0,")
        print("    governance_policy='enforced'")
        print(")")
        print("```")
        print()
        
        # Demonstrate disable/re-enable
        print("🧪 Demonstrating disable and re-enable...")
        
        # Disable
        restore_skyrouter()
        current_adapter = get_current_adapter()
        status = "DISABLED" if current_adapter is None else "ENABLED"
        print(f"   🔄 After restore_skyrouter(): {status}")
        
        # Re-enable with new settings
        new_adapter = auto_instrument(
            team="re-enabled-team",
            project="disable-enable-demo",
            daily_budget_limit=50.0
        )
        print(f"   🔄 After auto_instrument(): ENABLED")
        print(f"   👥 New team: {new_adapter.governance_attrs.team}")
        print(f"   📊 New project: {new_adapter.governance_attrs.project}")
        print(f"   💰 New budget: $50.00")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Enable/disable demo failed: {e}")
        return False

def show_integration_examples():
    """Show integration examples for different frameworks."""
    
    print("🔗 Integration Examples for Different Frameworks")
    print("-" * 52)
    print()
    
    examples = [
        {
            "framework": "Flask Web Application",
            "code": """
from flask import Flask, request, jsonify
from genops.providers.skyrouter import auto_instrument

app = Flask(__name__)
auto_instrument(team="web-team", project="api-service")

@app.route('/ai-endpoint', methods=['POST'])
def ai_endpoint():
    # Your existing SkyRouter code - automatically governed!
    prompt = request.json.get('prompt')
    
    # This routing is now automatically tracked
    result = skyrouter_client.route_to_best_model(
        candidates=['gpt-4', 'claude-3-sonnet'],
        prompt=prompt,
        routing_strategy='balanced'
    )
    
    return jsonify(result)
"""
        },
        {
            "framework": "FastAPI Application",
            "code": """
from fastapi import FastAPI
from genops.providers.skyrouter import auto_instrument

app = FastAPI()
auto_instrument(team="api-team", project="fastapi-service")

@app.post("/multi-model-route")
async def multi_model_route(request: dict):
    # Your existing async SkyRouter code - automatically governed!
    return await skyrouter_client.async_route_to_best_model(**request)
"""
        },
        {
            "framework": "Jupyter Notebook",
            "code": """
# Cell 1: Setup (add to first cell)
from genops.providers.skyrouter import auto_instrument
auto_instrument(team="data-science", project="notebook-analysis")

# Cell 2: Your existing analysis (unchanged!)
import skyrouter
result = skyrouter_client.route_to_best_model(...)
# ✅ Automatically tracked with governance
"""
        },
        {
            "framework": "Background Job/Celery",
            "code": """
from celery import Celery
from genops.providers.skyrouter import auto_instrument

app = Celery('skyrouter_tasks')
auto_instrument(team="background-jobs", project="celery-tasks")

@app.task
def process_with_ai(data):
    # Your existing SkyRouter processing - automatically governed!
    return skyrouter_client.route_to_best_model(...)
"""
        }
    ]
    
    for example in examples:
        print(f"📱 **{example['framework']}**")
        print("```python")
        print(example['code'].strip())
        print("```")
        print()

def main():
    """Main execution function."""
    
    print("🚀 SkyRouter + GenOps Zero-Code Auto-Instrumentation Demo")
    print("=" * 65)
    print()
    
    print("Add enterprise governance to your existing SkyRouter applications")
    print("without changing a single line of your current code!")
    print()
    
    # Check prerequisites
    api_key = os.getenv("SKYROUTER_API_KEY")
    if not api_key:
        print("❌ Missing SKYROUTER_API_KEY environment variable")
        print()
        print("💡 Quick setup:")
        print("   export SKYROUTER_API_KEY='your-api-key'")
        print("   export GENOPS_TEAM='your-team'")
        print("   export GENOPS_PROJECT='your-project'")
        print()
        return
    
    try:
        success = True
        
        # Main auto-instrumentation demonstration
        if success:
            success = demonstrate_auto_instrumentation()
        
        # Enable/disable demonstration
        if success:
            success = demonstrate_enable_disable()
        
        # Show integration examples
        if success:
            show_integration_examples()
        
        if success:
            print("🎉 Auto-instrumentation demonstration completed!")
            print()
            print("🔑 **Key Takeaways:**")
            print("• ✨ Zero code changes required to add governance")
            print("• 🔄 Easy to enable/disable as needed")
            print("• 📊 Full cost tracking and attribution automatically")
            print("• 🚀 Works with any existing SkyRouter application")
            print("• 🔧 Configurable budgets and policies")
            print()
            print("🚀 **Next Steps:**")
            print("1. Add auto_instrument() to your existing SkyRouter app")
            print("2. Try route_optimization.py for advanced routing strategies")
            print("3. Explore agent_workflows.py for multi-agent patterns")
            print("4. Check enterprise_patterns.py for production deployment")
        
    except KeyboardInterrupt:
        print()
        print("👋 Demo cancelled.")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("1. Verify SKYROUTER_API_KEY is set correctly")
        print("2. Ensure GenOps is installed: pip install genops[skyrouter]")
        print("3. Check internet connection for SkyRouter API access")

if __name__ == "__main__":
    main()