#!/usr/bin/env python3
"""
Raindrop AI + GenOps Advanced Features Demo

This example demonstrates advanced Raindrop AI monitoring capabilities with
comprehensive GenOps governance including multi-agent cost aggregation,
performance optimization, and enterprise-grade policy enforcement.

Features demonstrated:
- Multi-agent production monitoring with unified cost tracking
- Advanced performance signal analysis with cost optimization
- Complex alert strategies with cost intelligence
- Enterprise governance patterns and compliance integration
- Real-time cost optimization recommendations

Usage:
    export RAINDROP_API_KEY="your-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    python advanced_features.py

Author: GenOps AI Contributors
"""

import os
import sys
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from genops.providers.raindrop import GenOpsRaindropAdapter
    from genops.providers.raindrop_validation import validate_setup
    from genops.providers.raindrop_pricing import RaindropPricingConfig
except ImportError as e:
    print(f"❌ Error importing GenOps Raindrop: {e}")
    print("💡 Make sure you're in the project root directory and GenOps is properly installed")
    sys.exit(1)

def simulate_production_agent(agent_config: dict, adapter: GenOpsRaindropAdapter) -> dict:
    """Simulate a production agent with realistic monitoring scenarios."""
    agent_id = agent_config["id"]
    agent_type = agent_config["type"]
    interaction_count = agent_config["interactions"]
    complexity = agent_config["complexity"]
    
    session_name = f"{agent_type}-{agent_id}"
    total_cost = 0
    operations = 0
    alerts_created = 0
    
    with adapter.track_agent_monitoring_session(session_name) as session:
        # Simulate agent interactions
        for i in range(interaction_count):
            # Generate realistic interaction data based on agent type
            if agent_type == "customer-service":
                interaction_data = {
                    "input": f"Customer inquiry about {random.choice(['billing', 'support', 'product'])}",
                    "output": f"Automated response with {random.choice(['resolution', 'escalation', 'follow-up'])}",
                    "performance_signals": {
                        "response_time_ms": random.randint(100, 800),
                        "confidence_score": round(random.uniform(0.7, 0.98), 3),
                        "customer_satisfaction": round(random.uniform(3.5, 5.0), 2),
                        "resolution_rate": round(random.uniform(0.75, 0.95), 3)
                    }
                }
            elif agent_type == "recommendation":
                interaction_data = {
                    "input": f"User profile and behavior data",
                    "output": f"Personalized recommendations for {random.choice(['products', 'content', 'services'])}",
                    "performance_signals": {
                        "response_time_ms": random.randint(50, 300),
                        "relevance_score": round(random.uniform(0.8, 0.99), 3),
                        "click_through_rate": round(random.uniform(0.15, 0.35), 3),
                        "conversion_rate": round(random.uniform(0.05, 0.15), 3)
                    }
                }
            else:  # fraud-detection
                interaction_data = {
                    "input": f"Transaction data with risk indicators",
                    "output": f"Risk assessment: {random.choice(['low', 'medium', 'high'])} risk",
                    "performance_signals": {
                        "response_time_ms": random.randint(25, 150),
                        "accuracy": round(random.uniform(0.92, 0.998), 4),
                        "false_positive_rate": round(random.uniform(0.001, 0.02), 4),
                        "detection_rate": round(random.uniform(0.94, 0.999), 4)
                    }
                }
            
            # Track the interaction
            cost_result = session.track_agent_interaction(
                agent_id=agent_id,
                interaction_data=interaction_data,
                complexity=complexity
            )
            
            total_cost += float(cost_result.total_cost)
            operations += 1
            
            # Simulate performance signal monitoring
            if i % 10 == 0:  # Monitor every 10th interaction
                signal_data = {
                    "monitoring_frequency": "high",
                    "threshold_config": interaction_data["performance_signals"],
                    "alert_conditions": ["response_time > 500ms", "confidence < 0.8"]
                }
                
                signal_cost = session.track_performance_signal(
                    signal_name=f"{agent_type}_performance_monitoring",
                    signal_data=signal_data,
                    complexity=complexity
                )
                total_cost += float(signal_cost.total_cost)
                operations += 1
            
            # Create alerts for performance issues
            if i % 25 == 0 and random.random() > 0.7:  # 30% chance every 25 interactions
                alert_config = {
                    "conditions": [
                        {"metric": "response_time", "operator": ">", "threshold": 400},
                        {"metric": "confidence", "operator": "<", "threshold": 0.85}
                    ],
                    "notification_channels": ["slack", "email"] if complexity == "enterprise" else ["email"],
                    "severity": random.choice(["warning", "critical"]),
                    "auto_resolution": True if complexity in ["complex", "enterprise"] else False
                }
                
                alert_cost = session.create_alert(
                    alert_name=f"{agent_type}_performance_alert_{alerts_created + 1}",
                    alert_config=alert_config,
                    complexity=complexity
                )
                total_cost += float(alert_cost.total_cost)
                operations += 1
                alerts_created += 1
        
        return {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "total_cost": total_cost,
            "operations": operations,
            "alerts_created": alerts_created,
            "session_duration": session.duration_seconds,
            "efficiency": operations / max(session.duration_seconds / 3600, 1/3600)
        }

def main():
    """Demonstrate advanced Raindrop AI + GenOps integration features."""
    
    print("🚀 Raindrop AI + GenOps Advanced Features Demo")
    print("=" * 60)
    
    # Configuration
    api_key = os.getenv("RAINDROP_API_KEY")
    team = os.getenv("GENOPS_TEAM", "advanced-features-team")
    project = os.getenv("GENOPS_PROJECT", "production-monitoring")
    
    # Validate setup
    validation_result = validate_setup(api_key)
    if not validation_result.is_valid:
        print("❌ Setup validation failed. Please check your configuration.")
        return
    
    # Advanced pricing configuration
    custom_pricing = RaindropPricingConfig()
    custom_pricing.volume_tiers = {
        500: 0.05,    # 5% discount for 500+ interactions
        2000: 0.12,   # 12% discount for 2K+ interactions  
        10000: 0.20,  # 20% discount for 10K+ interactions
        50000: 0.30   # 30% discount for 50K+ interactions
    }
    
    # Initialize adapter with advanced configuration
    adapter = GenOpsRaindropAdapter(
        raindrop_api_key=api_key,
        team=team,
        project=project,
        environment="production",
        daily_budget_limit=200.0,
        enable_cost_alerts=True,
        governance_policy="enforced"
    )
    
    # Update pricing calculator with custom config
    adapter.pricing_calculator.config = custom_pricing
    adapter.pricing_calculator.update_monthly_volume(15000)  # Simulate high-volume usage
    
    print(f"\n📊 Multi-Agent Production Monitoring Demo")
    print("-" * 50)
    
    # Define production agent fleet
    agent_fleet = [
        {"id": "cs-bot-1", "type": "customer-service", "interactions": 150, "complexity": "moderate"},
        {"id": "cs-bot-2", "type": "customer-service", "interactions": 180, "complexity": "complex"},
        {"id": "rec-engine-1", "type": "recommendation", "interactions": 300, "complexity": "enterprise"},
        {"id": "rec-engine-2", "type": "recommendation", "interactions": 275, "complexity": "complex"},
        {"id": "fraud-det-1", "type": "fraud-detection", "interactions": 120, "complexity": "enterprise"},
        {"id": "fraud-det-2", "type": "fraud-detection", "interactions": 95, "complexity": "complex"}
    ]
    
    print(f"🔄 Starting concurrent monitoring for {len(agent_fleet)} production agents...")
    
    # Execute concurrent monitoring
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit monitoring tasks
        future_to_agent = {
            executor.submit(simulate_production_agent, agent_config, adapter): agent_config
            for agent_config in agent_fleet
        }
        
        # Collect results
        for future in as_completed(future_to_agent):
            agent_config = future_to_agent[future]
            try:
                result = future.result()
                results.append(result)
                print(f"  ✅ {result['agent_id']}: ${result['total_cost']:.3f} cost, {result['alerts_created']} alerts")
            except Exception as e:
                print(f"  ❌ {agent_config['id']} failed: {str(e)}")
    
    # Analyze results
    total_monitoring_cost = sum(r['total_cost'] for r in results)
    total_operations = sum(r['operations'] for r in results)
    total_alerts = sum(r['alerts_created'] for r in results)
    
    print(f"\n📊 Multi-Agent Monitoring Summary:")
    print(f"  💰 Total monitoring cost: ${total_monitoring_cost:.2f}")
    print(f"  📈 Total operations monitored: {total_operations:,}")
    print(f"  🚨 Total active alerts: {total_alerts}")
    print(f"  🏭 Agents monitored: {len(results)}")
    
    # Advanced cost intelligence analysis
    print(f"\n💡 Advanced Cost Intelligence Demo")
    print("-" * 40)
    
    # Get comprehensive cost summary
    summary = adapter.cost_aggregator.get_summary()
    
    print(f"\n🔍 Cost breakdown by agent type:")
    agent_type_costs = {}
    for result in results:
        agent_type = result['agent_type']
        if agent_type not in agent_type_costs:
            agent_type_costs[agent_type] = 0
        agent_type_costs[agent_type] += result['total_cost']
    
    for agent_type, cost in sorted(agent_type_costs.items(), key=lambda x: x[1], reverse=True):
        percentage = (cost / total_monitoring_cost) * 100
        print(f"  • {agent_type}: ${cost:.2f} ({percentage:.1f}%)")
    
    # Volume discount analysis
    volume_info = adapter.pricing_calculator.get_volume_discount_info()
    print(f"\n📊 Volume Discount Analysis:")
    print(f"  Current monthly interactions: {volume_info['current_monthly_interactions']:,}")
    print(f"  Current discount rate: {volume_info['current_discount_percentage']:.1f}%")
    if volume_info['next_tier_threshold']:
        print(f"  Next discount tier: {volume_info['next_tier_threshold']:,} interactions ({volume_info['next_tier_discount_percentage']:.1f}% discount)")
        savings_potential = total_monitoring_cost * (volume_info['next_tier_discount_rate'] - volume_info['current_discount_rate'])
        print(f"  Potential additional savings: ${savings_potential:.2f}")
    
    # Cost optimization recommendations
    print(f"\n🚀 Cost Optimization Recommendations:")
    recommendations = adapter.cost_aggregator.get_cost_optimization_recommendations()
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['title']}")
        print(f"     💰 Potential savings: ${rec['potential_savings']:.2f}")
        print(f"     ⚡ Effort level: {rec['effort_level']}")
        print(f"     📊 Priority score: {rec['priority_score']:.1f}/100")
        print(f"     🔧 Key actions:")
        for action in rec['actions'][:2]:  # Show top 2 actions
            print(f"       • {action}")
        if len(rec['actions']) > 2:
            print(f"       • ... and {len(rec['actions']) - 2} more")
        print()
    
    # Advanced monitoring efficiency analysis
    print(f"\n📈 Monitoring Efficiency Analysis:")
    avg_cost_per_op = total_monitoring_cost / max(total_operations, 1)
    avg_efficiency = sum(r['efficiency'] for r in results) / len(results)
    
    print(f"  📊 Cost per operation: ${avg_cost_per_op:.4f}")
    print(f"  🔍 Cost per alert: ${total_monitoring_cost / max(total_alerts, 1):.2f}")
    print(f"  💵 Operations per dollar: {1 / avg_cost_per_op:.0f}")
    print(f"  ⚡ Average efficiency: {avg_efficiency:.1f} operations/hour")
    
    # Enterprise governance demonstration
    print(f"\n🏛️ Enterprise Governance Features:")
    print(f"  ✅ Multi-agent cost attribution")
    print(f"  ✅ Real-time budget enforcement")
    print(f"  ✅ Volume-based pricing optimization")
    print(f"  ✅ Performance-based cost intelligence")
    print(f"  ✅ Automated policy compliance")
    print(f"  ✅ OpenTelemetry-native telemetry export")
    
    # Budget status check
    budget_status = adapter.cost_aggregator.check_budget_status()
    if budget_status['budget_alerts']:
        print(f"\n⚠️ Budget Alerts:")
        for alert in budget_status['budget_alerts'][:3]:  # Show first 3 alerts
            print(f"  🚨 {alert['message']}")
    else:
        print(f"\n✅ All budgets within limits")
    
    print(f"\n✅ Advanced features demo completed successfully!")
    print(f"\n🔗 Integration Opportunities:")
    print(f"  1. Connect to your observability dashboard (Grafana, Datadog)")
    print(f"  2. Set up automated cost alerts and budget enforcement")
    print(f"  3. Integrate with your FinOps and procurement workflows")
    print(f"  4. Deploy governance policies across development teams")

if __name__ == "__main__":
    main()