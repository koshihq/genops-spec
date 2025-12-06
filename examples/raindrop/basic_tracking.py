#!/usr/bin/env python3
"""
Raindrop AI + GenOps Basic Tracking Example

This example demonstrates basic agent monitoring with GenOps governance,
cost tracking, and team attribution using the Raindrop AI platform.

Features demonstrated:
- Agent interaction tracking with cost attribution
- Performance signal monitoring
- Alert creation and management
- Automatic governance telemetry export
- Budget monitoring and enforcement

Usage:
    export RAINDROP_API_KEY="your-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    python basic_tracking.py

Author: GenOps AI Contributors
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from genops.providers.raindrop import GenOpsRaindropAdapter, auto_instrument
    from genops.providers.raindrop_validation import validate_setup, print_validation_result
except ImportError as e:
    print(f"❌ Error importing GenOps Raindrop: {e}")
    print("💡 Make sure you're in the project root directory and GenOps is properly installed")
    sys.exit(1)

def main():
    """
    Demonstrate basic Raindrop AI + GenOps integration.
    
    This example shows how to:
    1. Validate your setup and configuration 
    2. Initialize the GenOps Raindrop adapter
    3. Track agent interactions with cost attribution
    4. Monitor performance signals
    5. Create alerts with governance
    6. View cost summaries and governance metrics
    
    Expected runtime: 2-3 minutes
    Expected cost: < $0.10 (simulated operations)
    """
    
    print("🚀 Raindrop AI + GenOps Basic Tracking Example")
    print("=" * 60)
    print("📚 This example demonstrates:")
    print("  • Agent interaction tracking with cost attribution")
    print("  • Performance signal monitoring")
    print("  • Alert creation and management")
    print("  • Governance telemetry export")
    print("  • Budget monitoring and enforcement")
    print()
    
    # Enhanced prerequisites check with better error handling
    print("📋 Prerequisites Check:")
    try:
        # Environment variable validation
        api_key = os.getenv("RAINDROP_API_KEY")
        team = os.getenv("GENOPS_TEAM", "basic-tracking-team")
        project = os.getenv("GENOPS_PROJECT", "agent-monitoring-demo")
        
        print(f"  {'✅' if api_key else '❌'} RAINDROP_API_KEY: {'configured' if api_key else 'MISSING'}")
        print(f"  ✅ GENOPS_TEAM: {team}")
        print(f"  ✅ GENOPS_PROJECT: {project}")
        
        if not api_key:
            print("\n🔧 Missing API key. To fix this:")
            print("  1. Get your API key from https://app.raindrop.ai → Settings → API Keys")
            print("  2. Set it: export RAINDROP_API_KEY='your-api-key'")
            print("  3. Re-run this example")
            return
        
        # Comprehensive validation
        print("\n🔍 Running comprehensive validation...")
        validation_result = validate_setup(api_key)
        
        if not validation_result.is_valid:
            print("❌ Setup validation failed. Issues detected:")
            for error in validation_result.errors[:3]:  # Show first 3 errors
                print(f"  • {error.message}")
                if error.fix_suggestion:
                    print(f"    💡 Fix: {error.fix_suggestion}")
            
            print("\n🔧 To resolve these issues:")
            print("  • Run interactive setup: python -c \"from genops.providers.raindrop_validation import validate_setup_interactive; validate_setup_interactive()\"")
            print("  • Check the troubleshooting guide in docs/raindrop-quickstart.md")
            return
            
        print("  ✅ All validation checks passed!")
        
    except Exception as setup_error:
        print(f"❌ Prerequisites check failed: {setup_error}")
        print("💡 This might be due to missing dependencies or configuration issues")
        print("🔧 Try:")
        print("  • pip install --upgrade genops[raindrop]")
        print("  • Verify your environment variables are set correctly")
        return
    
    print("  ✅ GenOps installed")
    print("  ✅ Raindrop AI integration available")
    print(f"  {'✅' if api_key else '❌'} RAINDROP_API_KEY {'configured' if api_key else 'required'}")
    print(f"  ✅ Team: {team}")
    print(f"  ✅ Project: {project}")
    
    # Initialize GenOps adapter
    print(f"\n🎯 Starting basic agent monitoring with governance...")
    
    adapter = GenOpsRaindropAdapter(
        raindrop_api_key=api_key,
        team=team,
        project=project,
        environment="development",
        daily_budget_limit=50.0,
        enable_cost_alerts=True,
        governance_policy="advisory"  # Use advisory mode for demo
    )
    
    # Example 1: Track an agent monitoring session
    print(f"\n🤖 Example 1: Agent Interaction Tracking")
    print("-" * 40)
    
    session_name = "customer-support-agents"
    with adapter.track_agent_monitoring_session(session_name) as session:
        print(f"✅ Agent monitoring session started: {session_name}")
        
        # Simulate agent interactions
        agents = ["support-bot-1", "support-bot-2", "escalation-agent"]
        
        for i, agent_id in enumerate(agents, 1):
            interaction_data = {
                "input": f"Customer inquiry #{i}",
                "output": f"Agent response #{i}",
                "performance_signals": {
                    "response_time_ms": 250 + i * 50,
                    "confidence_score": 0.92 - i * 0.02,
                    "customer_satisfaction": 4.5
                },
                "metadata": {
                    "conversation_length": 3 + i,
                    "resolution_status": "resolved" if i <= 2 else "escalated"
                }
            }
            
            # Track the interaction with cost attribution
            cost_result = session.track_agent_interaction(
                agent_id=agent_id,
                interaction_data=interaction_data
            )
            
            print(f"  💬 Agent interaction logged: {agent_id} - ${cost_result.total_cost:.3f}")
            time.sleep(0.1)  # Simulate processing time
        
        # Example 2: Track performance signals
        print(f"\n📊 Example 2: Performance Signal Monitoring")
        print("-" * 40)
        
        signals = [
            ("response_time_alert", {"threshold": 500, "current": 320}, "simple"),
            ("confidence_degradation", {"threshold": 0.85, "current": 0.89}, "moderate"),
            ("customer_satisfaction", {"threshold": 4.0, "current": 4.3}, "simple")
        ]
        
        for signal_name, signal_data, complexity in signals:
            cost_result = session.track_performance_signal(
                signal_name=signal_name,
                signal_data=signal_data
            )
            print(f"  📈 Performance signal tracked: {signal_name} - ${cost_result.total_cost:.3f}")
        
        # Example 3: Create alerts
        print(f"\n🚨 Example 3: Alert Creation")
        print("-" * 40)
        
        alert_config = {
            "conditions": [
                {"metric": "response_time", "operator": ">", "threshold": 500},
                {"metric": "confidence", "operator": "<", "threshold": 0.8}
            ],
            "notification_channels": ["email", "slack"],
            "severity": "warning"
        }
        
        cost_result = session.create_alert(
            alert_name="agent_performance_degradation",
            alert_config=alert_config
        )
        print(f"  🔔 Alert created: agent_performance_degradation - ${cost_result.total_cost:.3f}")
        
        print(f"\n💰 Session Cost Summary:")
        print(f"  Total: ${session.total_cost:.3f}")
        print(f"  Operations: {session.operation_count}")
        print(f"  Duration: {session.duration_seconds:.1f}s")
        print(f"  Efficiency: {session.operation_count/max(session.duration_seconds/3600, 1/3600):.1f} operations/hour")
    
    # Display governance metrics
    print(f"\n📊 Governance Metrics:")
    print(f"  Team: {team}")
    print(f"  Project: {project}")
    print(f"  Environment: development")
    print(f"  Daily Usage: ${session.total_cost:.3f}")
    print(f"  Budget Remaining: ${50.0 - float(session.total_cost):.2f}")
    
    # Example 4: Demonstrate cost aggregation
    print(f"\n💡 Cost Intelligence Preview:")
    
    # Get cost aggregator data
    cost_aggregator = adapter.cost_aggregator
    summary = cost_aggregator.get_summary()
    
    print(f"  📈 Total monitored sessions: {summary.session_count}")
    print(f"  💰 Total cost: ${summary.total_cost:.3f}")
    print(f"  ⚡ Average cost per operation: ${summary.average_cost_per_operation:.4f}")
    
    if summary.cost_by_operation_type:
        print(f"  📊 Cost breakdown:")
        for op_type, cost in summary.cost_by_operation_type.items():
            percentage = float(cost / summary.total_cost * 100) if summary.total_cost > 0 else 0
            print(f"    • {op_type}: ${cost:.3f} ({percentage:.1f}%)")
    
    print(f"\n✅ Basic tracking example completed successfully!")
    print(f"\n🚀 Next Steps:")
    print(f"  1. Try auto-instrumentation: python auto_instrumentation.py")
    print(f"  2. Explore advanced features: python advanced_features.py")
    print(f"  3. Check cost optimization: python cost_optimization.py")
    print(f"  4. Review production patterns: python production_patterns.py")

if __name__ == "__main__":
    main()