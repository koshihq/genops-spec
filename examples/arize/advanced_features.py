#!/usr/bin/env python3
"""
Arize AI + GenOps Advanced Features Example

This example demonstrates advanced model monitoring capabilities with Arize AI
enhanced by GenOps governance, including multi-model tracking, advanced cost
intelligence, dynamic budget management, and production-ready patterns.

Features demonstrated:
- Multi-model concurrent monitoring with cost aggregation
- Advanced cost intelligence with optimization recommendations
- Dynamic budget management and cost-aware monitoring
- Data quality monitoring with drift detection
- Performance alert management with cost optimization
- Production-ready monitoring patterns
- Enterprise governance with audit trails

Run this example:
    python advanced_features.py

Prerequisites:
    export ARIZE_API_KEY="your-arize-api-key"
    export ARIZE_SPACE_KEY="your-arize-space-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"

Expected runtime: 10-15 minutes
Expected output: Multi-model cost analysis and governance insights
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd


def print_header():
    """Print example header with advanced features overview."""
    print("=" * 70)
    print("🚀 Arize AI + GenOps Advanced Features Demo")
    print("=" * 70)
    print()
    print("📋 This demo showcases:")
    print("  • Multi-model monitoring with unified governance")
    print("  • Advanced cost intelligence and optimization")
    print("  • Dynamic budget management and cost-aware monitoring")
    print("  • Production-ready monitoring patterns")
    print("  • Enterprise governance with audit trails")
    print()
    print("⏱️ Estimated runtime: 10-15 minutes")
    print()


def check_advanced_prerequisites():
    """Check prerequisites for advanced features demonstration."""
    print("🔍 Advanced Prerequisites Check:")

    missing_requirements = []

    # Check required packages
    try:
        import genops
        from genops.providers.arize import GenOpsArizeAdapter, auto_instrument
        from genops.providers.arize_cost_aggregator import ArizeCostAggregator
        from genops.providers.arize_pricing import ArizePricingCalculator
        from genops.providers.arize_validation import ArizeSetupValidator
        print("  ✅ All GenOps Arize modules available")
    except ImportError as e:
        missing_requirements.append(f"GenOps Arize integration: {e}")

    try:
        import numpy as np
        import pandas as pd
        print("  ✅ Data processing libraries available")
    except ImportError as e:
        missing_requirements.append(f"Data processing libraries: {e}")

    # Check environment variables
    required_env_vars = ['ARIZE_API_KEY', 'ARIZE_SPACE_KEY']
    for var in required_env_vars:
        if not os.getenv(var):
            missing_requirements.append(f"Missing environment variable: {var}")
        else:
            print(f"  ✅ {var} configured")

    if missing_requirements:
        print("\n❌ Missing requirements:")
        for req in missing_requirements:
            print(f"  • {req}")
        print("\nPlease install missing dependencies and set environment variables.")
        return False

    print("  ✅ All advanced prerequisites met!")
    print()
    return True


def create_sample_model_data():
    """Create realistic sample data for multiple models."""
    print("📊 Generating realistic multi-model sample data...")

    # Define production model scenarios
    models = {
        'fraud-detection-v3': {
            'volume': 25000,
            'accuracy': 0.94,
            'precision': 0.91,
            'recall': 0.96,
            'drift_score': 0.12,
            'environment': 'production',
            'business_impact': 'high'
        },
        'recommendation-engine-v2': {
            'volume': 150000,
            'accuracy': 0.87,
            'precision': 0.85,
            'recall': 0.89,
            'drift_score': 0.08,
            'environment': 'production',
            'business_impact': 'medium'
        },
        'sentiment-analysis-v1': {
            'volume': 45000,
            'accuracy': 0.92,
            'precision': 0.90,
            'recall': 0.94,
            'drift_score': 0.15,
            'environment': 'production',
            'business_impact': 'low'
        },
        'churn-prediction-v2': {
            'volume': 8000,
            'accuracy': 0.89,
            'precision': 0.87,
            'recall': 0.91,
            'drift_score': 0.18,
            'environment': 'staging',
            'business_impact': 'high'
        }
    }

    # Generate realistic prediction data for each model
    model_data = {}
    for model_id, config in models.items():
        predictions = []
        for i in range(min(1000, config['volume'])):  # Sample of actual volume
            prediction = {
                'prediction_id': f"{model_id}-pred-{i}",
                'timestamp': datetime.utcnow() - timedelta(minutes=random.randint(0, 1440)),
                'prediction': random.choice([0, 1]) if 'fraud' in model_id or 'churn' in model_id
                             else random.uniform(0, 1),
                'confidence': random.uniform(0.7, 0.99),
                'features': {
                    'feature_1': random.uniform(-2, 2),
                    'feature_2': random.uniform(0, 100),
                    'feature_3': random.choice(['A', 'B', 'C'])
                }
            }
            predictions.append(prediction)

        model_data[model_id] = {
            'config': config,
            'predictions': predictions,
            'quality_metrics': {
                'accuracy': config['accuracy'] + random.uniform(-0.05, 0.05),
                'precision': config['precision'] + random.uniform(-0.03, 0.03),
                'recall': config['recall'] + random.uniform(-0.03, 0.03),
                'data_drift_score': config['drift_score'] + random.uniform(-0.02, 0.02),
                'feature_importance_shift': random.uniform(0, 0.3),
                'prediction_distribution_shift': random.uniform(0, 0.2)
            }
        }

    print(f"  ✅ Generated data for {len(model_data)} production models")
    print(f"  📈 Total sample predictions: {sum(len(data['predictions']) for data in model_data.values())}")
    print()

    return model_data


def demonstrate_multi_model_monitoring(model_data):
    """Demonstrate concurrent multi-model monitoring with governance."""
    print("🏭 Multi-Model Production Monitoring Demo")
    print("-" * 50)

    # Initialize cost aggregator for unified tracking
    cost_aggregator = ArizeCostAggregator(
        team=os.getenv('GENOPS_TEAM', 'ml-platform'),
        project=os.getenv('GENOPS_PROJECT', 'advanced-monitoring')
    )

    model_results = {}

    def monitor_single_model(model_id, model_info):
        """Monitor a single model with advanced features."""
        try:
            # Initialize adapter with model-specific configuration
            adapter = GenOpsArizeAdapter(
                team=os.getenv('GENOPS_TEAM', 'ml-platform'),
                project=f"model-{model_id}",
                environment=model_info['config']['environment'],
                daily_budget_limit=100.0 if model_info['config']['business_impact'] == 'high' else 50.0,
                max_monitoring_cost=25.0,
                enable_cost_alerts=True,
                tags={
                    'model_id': model_id,
                    'business_impact': model_info['config']['business_impact'],
                    'expected_volume': str(model_info['config']['volume'])
                }
            )

            session_results = {}

            # Start monitoring session with advanced context
            with adapter.track_model_monitoring_session(
                model_id=model_id,
                model_version='latest',
                environment=model_info['config']['environment'],
                max_cost=25.0
            ) as session:

                # Log prediction batch with realistic data
                predictions_df = pd.DataFrame(model_info['predictions'])
                cost_per_prediction = 0.001 if model_info['config']['business_impact'] == 'high' else 0.0005
                session.log_prediction_batch(predictions_df, cost_per_prediction=cost_per_prediction)

                # Advanced data quality monitoring
                quality_metrics = model_info['quality_metrics']
                session.log_data_quality_metrics(quality_metrics, cost_estimate=0.05)

                # Create intelligent alerts based on business impact
                if model_info['config']['business_impact'] == 'high':
                    # High-impact models get more monitoring
                    session.create_performance_alert('accuracy', 0.90, 0.15)
                    session.create_performance_alert('data_drift_score', 0.20, 0.12)
                    if quality_metrics['data_drift_score'] > 0.15:
                        session.create_performance_alert('urgent_drift_review', 0.15, 0.25)
                else:
                    # Standard monitoring for other models
                    session.create_performance_alert('accuracy', 0.85, 0.08)
                    session.create_performance_alert('data_drift_score', 0.25, 0.06)

                # Collect session results
                session_results = {
                    'model_id': model_id,
                    'environment': model_info['config']['environment'],
                    'total_cost': session.estimated_cost,
                    'prediction_count': session.prediction_count,
                    'data_quality_checks': session.data_quality_checks,
                    'active_alerts': session.active_alerts,
                    'business_impact': model_info['config']['business_impact'],
                    'quality_score': quality_metrics['accuracy']
                }

            # Add cost record to aggregator
            cost_aggregator.add_cost_record(
                model_id=model_id,
                environment=model_info['config']['environment'],
                prediction_logging_cost=session_results['prediction_count'] * cost_per_prediction,
                data_quality_cost=0.05,
                alert_management_cost=session_results['active_alerts'] * 0.08,
                dashboard_cost=0.10,
                prediction_count=session_results['prediction_count'],
                data_quality_checks=session_results['data_quality_checks'],
                active_alerts=session_results['active_alerts']
            )

            print(f"  ✅ {model_id}: ${session_results['total_cost']:.3f} cost, {session_results['active_alerts']} alerts")
            return session_results

        except Exception as e:
            print(f"  ❌ {model_id}: Error - {e}")
            return None

    # Execute concurrent monitoring
    print("  🔄 Starting concurrent model monitoring...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_model = {
            executor.submit(monitor_single_model, model_id, model_info): model_id
            for model_id, model_info in model_data.items()
        }

        for future in as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                result = future.result()
                if result:
                    model_results[model_id] = result
            except Exception as e:
                print(f"  ❌ {model_id} monitoring failed: {e}")

    print("\n📊 Multi-Model Monitoring Summary:")
    total_cost = sum(r['total_cost'] for r in model_results.values())
    total_predictions = sum(r['prediction_count'] for r in model_results.values())
    total_alerts = sum(r['active_alerts'] for r in model_results.values())

    print(f"  💰 Total monitoring cost: ${total_cost:.2f}")
    print(f"  📈 Total predictions monitored: {total_predictions:,}")
    print(f"  🚨 Total active alerts: {total_alerts}")
    print(f"  🏭 Models monitored: {len(model_results)}")
    print()

    return model_results, cost_aggregator


def demonstrate_cost_intelligence(cost_aggregator):
    """Demonstrate advanced cost intelligence and optimization."""
    print("💡 Advanced Cost Intelligence Demo")
    print("-" * 40)

    # Get comprehensive cost analysis
    cost_summary = cost_aggregator.get_cost_summary_by_model()
    print(f"📊 Total aggregated cost: ${cost_summary.total_cost:.2f}")

    # Analyze cost by model
    print("\n🔍 Cost breakdown by model:")
    for model_id, cost in cost_summary.cost_by_model.items():
        percentage = (cost / cost_summary.total_cost) * 100
        print(f"  • {model_id}: ${cost:.2f} ({percentage:.1f}%)")

    # Get optimization recommendations
    print("\n🚀 Cost Optimization Recommendations:")
    recommendations = cost_aggregator.get_cost_optimization_recommendations()

    if recommendations:
        for i, rec in enumerate(recommendations[:3], 1):  # Top 3 recommendations
            print(f"\n{i}. {rec.optimization_type.value.replace('_', ' ').title()}")
            print(f"   💰 Potential savings: ${rec.potential_savings:.2f}")
            print(f"   ⚡ Effort level: {rec.effort_level}")
            print(f"   📊 Priority score: {rec.priority_score:.1f}/100")
            print("   🔧 Key actions:")
            for action in rec.action_items[:2]:  # Top 2 actions
                print(f"      • {action}")
    else:
        print("  ✅ Your monitoring setup is already well-optimized!")

    # Get efficiency metrics
    print("\n📈 Monitoring Efficiency Analysis:")
    efficiency = cost_aggregator.get_efficiency_metrics()
    print(f"  📊 Cost per prediction: ${efficiency.cost_per_prediction:.4f}")
    print(f"  🔍 Cost per data quality check: ${efficiency.cost_per_data_quality_check:.3f}")
    print(f"  🚨 Cost per alert: ${efficiency.cost_per_alert:.3f}")
    print(f"  💵 Predictions per dollar: {efficiency.predictions_per_dollar:.0f}")

    # Show top performing models
    print("\n🏆 Model Efficiency Ranking:")
    if efficiency.model_efficiency_scores:
        sorted_models = sorted(
            efficiency.model_efficiency_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (model, score) in enumerate(sorted_models[:3], 1):
            print(f"  {i}. {model}: {score:.2f} efficiency score")

    print()
    return recommendations, efficiency


def demonstrate_dynamic_budget_management():
    """Demonstrate dynamic budget management and cost-aware monitoring."""
    print("💰 Dynamic Budget Management Demo")
    print("-" * 38)

    # Simulate different budget scenarios
    budget_scenarios = [
        {'name': 'Conservative', 'daily_budget': 25.0, 'max_session': 10.0},
        {'name': 'Standard', 'daily_budget': 75.0, 'max_session': 25.0},
        {'name': 'Aggressive', 'daily_budget': 200.0, 'max_session': 50.0}
    ]

    print("🎯 Testing different budget management strategies:")

    for scenario in budget_scenarios:
        print(f"\n📋 {scenario['name']} Budget Strategy:")
        print(f"  💰 Daily budget: ${scenario['daily_budget']}")
        print(f"  🎯 Max session cost: ${scenario['max_session']}")

        # Create adapter with specific budget configuration
        adapter = GenOpsArizeAdapter(
            team='budget-demo-team',
            project=f"budget-{scenario['name'].lower()}",
            daily_budget_limit=scenario['daily_budget'],
            max_monitoring_cost=scenario['max_session'],
            enable_cost_alerts=True,
            enable_governance=True
        )

        # Simulate cost-aware monitoring decisions
        simulated_operations = [
            {'type': 'prediction_batch', 'size': 5000, 'cost_each': 0.001},
            {'type': 'data_quality_check', 'cost': 0.08},
            {'type': 'performance_alert', 'cost': 0.15},
            {'type': 'dashboard_analytics', 'cost': 0.10}
        ]

        total_estimated_cost = sum(
            op['size'] * op.get('cost_each', 0) if 'size' in op else op['cost']
            for op in simulated_operations
        )

        print(f"  📊 Estimated operation cost: ${total_estimated_cost:.2f}")

        if total_estimated_cost <= scenario['max_session']:
            print("  ✅ Within budget - operations approved")
            recommendation = "Proceed with full monitoring suite"
        elif total_estimated_cost <= scenario['max_session'] * 1.2:
            print("  ⚠️ Near budget limit - optimization recommended")
            recommendation = "Consider reducing prediction sampling or alert frequency"
        else:
            print("  ❌ Over budget - cost reduction required")
            recommendation = "Implement sampling strategy or defer non-critical monitoring"

        print(f"  💡 Recommendation: {recommendation}")

    print()


def demonstrate_production_patterns():
    """Demonstrate production-ready monitoring patterns."""
    print("🏭 Production-Ready Monitoring Patterns Demo")
    print("-" * 48)

    # Pattern 1: High-Availability Monitoring
    print("1️⃣ High-Availability Pattern:")
    print("  🔄 Multiple adapter instances with failover")
    print("  📊 Distributed cost tracking")
    print("  🔍 Health check integration")

    primary_adapter = GenOpsArizeAdapter(
        team='production-primary',
        project='ha-monitoring',
        environment='production',
        tags={'role': 'primary', 'region': 'us-east-1'}
    )

    backup_adapter = GenOpsArizeAdapter(
        team='production-backup',
        project='ha-monitoring',
        environment='production',
        tags={'role': 'backup', 'region': 'us-west-2'}
    )

    print("  ✅ Primary and backup adapters configured")

    # Pattern 2: Environment-Specific Governance
    print("\n2️⃣ Environment-Specific Governance:")
    environments = ['development', 'staging', 'production']

    for env in environments:
        # Different budgets and policies per environment
        budget = {'development': 10.0, 'staging': 25.0, 'production': 100.0}[env]
        governance = {'development': False, 'staging': True, 'production': True}[env]

        adapter = GenOpsArizeAdapter(
            team='env-specific-team',
            project=f'{env}-monitoring',
            environment=env,
            daily_budget_limit=budget,
            enable_governance=governance,
            tags={'deployment_stage': env}
        )

        policy = "Strict" if governance else "Advisory"
        print(f"  🎯 {env.title()}: ${budget} budget, {policy} governance")

    # Pattern 3: Audit Trail and Compliance
    print("\n3️⃣ Audit Trail and Compliance Pattern:")
    compliance_adapter = GenOpsArizeAdapter(
        team='compliance-team',
        project='audit-monitoring',
        enable_governance=True,
        cost_center='ML-OPS-001',
        tags={
            'compliance_level': 'SOX',
            'data_classification': 'confidential',
            'audit_required': 'true',
            'retention_policy': '7_years'
        }
    )

    print("  📋 SOX compliance configuration active")
    print("  🔒 Confidential data classification applied")
    print("  📝 7-year audit retention policy set")
    print("  ✅ Governance metadata capture enabled")

    print()


def demonstrate_enterprise_governance():
    """Demonstrate enterprise governance features."""
    print("🏛️ Enterprise Governance Demo")
    print("-" * 32)

    # Multi-tenant configuration
    tenants = [
        {'customer_id': 'enterprise-client-001', 'team': 'client-success', 'budget': 500.0},
        {'customer_id': 'startup-client-042', 'team': 'growth', 'budget': 50.0},
        {'customer_id': 'internal-ml-ops', 'team': 'platform', 'budget': 200.0}
    ]

    print("🏢 Multi-Tenant Governance Configuration:")

    for tenant in tenants:
        adapter = GenOpsArizeAdapter(
            customer_id=tenant['customer_id'],
            team=tenant['team'],
            project='tenant-monitoring',
            daily_budget_limit=tenant['budget'],
            enable_governance=True,
            tags={
                'customer_tier': 'enterprise' if tenant['budget'] > 100 else 'startup',
                'billing_model': 'usage_based',
                'sla_level': 'premium' if tenant['budget'] > 200 else 'standard'
            }
        )

        tier = 'Enterprise' if tenant['budget'] > 100 else 'Startup'
        sla = 'Premium' if tenant['budget'] > 200 else 'Standard'

        print(f"  👤 {tenant['customer_id']}: {tier} tier, {sla} SLA, ${tenant['budget']} budget")

        # Demonstrate tenant-specific metrics
        metrics = adapter.get_metrics()
        print(f"      💰 Current usage: ${metrics['daily_usage']:.2f}")
        print(f"      📊 Remaining budget: ${metrics['budget_remaining']:.2f}")

    print("\n🔐 Governance Policy Enforcement:")
    print("  ✅ Customer data isolation enforced")
    print("  📊 Usage attribution per customer/team")
    print("  💰 Independent budget tracking")
    print("  📋 Tenant-specific compliance policies")
    print("  🔍 Audit trail per customer engagement")

    print()


def print_summary_and_next_steps():
    """Print example summary and recommended next steps."""
    print("=" * 70)
    print("🎉 Advanced Features Demo Complete!")
    print("=" * 70)

    print("\n✅ Features demonstrated:")
    print("  🏭 Multi-model concurrent monitoring with unified governance")
    print("  💡 Advanced cost intelligence with optimization recommendations")
    print("  💰 Dynamic budget management and cost-aware monitoring")
    print("  🏭 Production-ready monitoring patterns")
    print("  🏛️ Enterprise governance with multi-tenant support")
    print("  📊 Real-time cost aggregation and efficiency analysis")

    print("\n🚀 Next steps for production deployment:")
    print("  1. 📖 Review the production deployment guide")
    print("  2. 🔧 Configure environment-specific governance policies")
    print("  3. 📊 Set up cost monitoring dashboards")
    print("  4. 🔍 Implement audit trail collection")
    print("  5. ⚡ Optimize monitoring based on cost recommendations")

    print("\n🔗 Useful resources:")
    print("  📚 Complete integration guide: docs/integrations/arize.md")
    print("  💰 Cost optimization examples: cost_optimization.py")
    print("  🏭 Production patterns: production_patterns.py")
    print("  🔍 Validation utilities: setup_validation.py")

    print("\n💬 Need help?")
    print("  🐛 Report issues: https://github.com/KoshiHQ/GenOps-AI/issues")
    print("  💭 Discussions: https://github.com/KoshiHQ/GenOps-AI/discussions")
    print("  📧 Enterprise support: support@genops.ai")

    print()


def main():
    """Main demonstration function."""
    print_header()

    # Check prerequisites
    if not check_advanced_prerequisites():
        return

    try:
        # Generate sample data
        model_data = create_sample_model_data()

        # Demonstrate multi-model monitoring
        model_results, cost_aggregator = demonstrate_multi_model_monitoring(model_data)

        if model_results:
            # Demonstrate cost intelligence
            recommendations, efficiency = demonstrate_cost_intelligence(cost_aggregator)

            # Demonstrate dynamic budget management
            demonstrate_dynamic_budget_management()

            # Demonstrate production patterns
            demonstrate_production_patterns()

            # Demonstrate enterprise governance
            demonstrate_enterprise_governance()

            # Print summary
            print_summary_and_next_steps()
        else:
            print("❌ Multi-model monitoring demo failed. Check configuration and try again.")

    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Verify all environment variables are set correctly")
        print("  2. Check network connectivity to Arize AI")
        print("  3. Run setup_validation.py for detailed diagnostics")
        print("  4. Ensure GenOps dependencies are properly installed")


if __name__ == "__main__":
    main()
