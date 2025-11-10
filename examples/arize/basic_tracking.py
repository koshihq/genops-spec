#!/usr/bin/env python3
"""
Arize AI + GenOps Basic Tracking Example

This example demonstrates basic model monitoring operations with Arize AI
enhanced by GenOps governance, cost tracking, and team attribution.

Features demonstrated:
- Model monitoring session with cost tracking
- Prediction batch logging with governance metadata
- Data quality monitoring with cost attribution
- Performance alert creation with budget controls
- Real-time cost calculation and reporting

Run this example:
    python basic_tracking.py

Prerequisites:
    export ARIZE_API_KEY="your-arize-api-key"
    export ARIZE_SPACE_KEY="your-arize-space-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd


def print_header():
    """Print example header."""
    print("=" * 60)
    print("🚀 Arize AI + GenOps Basic Tracking Example")
    print("=" * 60)
    print()


def check_prerequisites():
    """Check if all required dependencies and configuration are available."""
    print("📋 Prerequisites Check:")

    missing_requirements = []

    # Check required packages
    try:
        import genops
        print("  ✅ GenOps installed")
    except ImportError:
        print("  ❌ GenOps not installed")
        missing_requirements.append("pip install genops")

    try:
        import arize
        print("  ✅ Arize SDK installed")
    except ImportError:
        print("  ❌ Arize SDK not installed")
        missing_requirements.append("pip install arize>=6.0.0")

    try:
        import pandas
        print("  ✅ Pandas installed")
    except ImportError:
        print("  ❌ Pandas not installed")
        missing_requirements.append("pip install pandas")

    # Check environment variables
    required_env_vars = ["ARIZE_API_KEY", "ARIZE_SPACE_KEY"]
    for var in required_env_vars:
        if os.getenv(var):
            print(f"  ✅ {var} configured")
        else:
            print(f"  ❌ {var} not set")
            missing_requirements.append(f"export {var}='your-{var.lower().replace('_', '-')}'")

    if missing_requirements:
        print("\n❌ Missing requirements found. Please fix:")
        for req in missing_requirements:
            print(f"   {req}")
        return False

    print("  ✅ All prerequisites met!")
    print()
    return True


def create_sample_prediction_data() -> pd.DataFrame:
    """Create sample prediction data for demonstration."""
    print("📊 Creating Sample Prediction Data...")

    import random

    # Generate sample fraud detection predictions
    sample_size = 1000
    predictions = []

    for i in range(sample_size):
        prediction_id = f"pred_{i:04d}_{int(time.time())}"

        # Simulate fraud detection model predictions
        features = {
            "transaction_amount": random.uniform(1.0, 5000.0),
            "merchant_category": random.choice(["online", "retail", "gas", "restaurant"]),
            "hour_of_day": random.randint(0, 23),
            "day_of_week": random.randint(0, 6),
            "user_age": random.randint(18, 80),
            "account_age_days": random.randint(1, 3650)
        }

        # Simulate model prediction (fraud probability)
        fraud_score = random.uniform(0.0, 1.0)
        prediction_label = "fraud" if fraud_score > 0.5 else "legitimate"

        # Simulate actual label (with some noise)
        actual_label = prediction_label
        if random.random() < 0.1:  # 10% chance of different actual
            actual_label = "legitimate" if prediction_label == "fraud" else "fraud"

        predictions.append({
            "prediction_id": prediction_id,
            "prediction_label": prediction_label,
            "actual_label": actual_label,
            "fraud_score": fraud_score,
            "timestamp": datetime.utcnow(),
            **features
        })

    df = pd.DataFrame(predictions)
    print(f"  ✅ Created {len(df)} sample predictions")
    print(f"  📈 Fraud rate: {(df['prediction_label'] == 'fraud').mean():.1%}")
    print(f"  🎯 Accuracy: {(df['prediction_label'] == df['actual_label']).mean():.1%}")
    print()

    return df


def demonstrate_basic_monitoring():
    """Demonstrate basic model monitoring with GenOps governance."""
    print("🔍 Demonstrating Basic Model Monitoring with Governance...")

    try:
        from genops.providers.arize import GenOpsArizeAdapter

        # Initialize adapter with governance configuration
        adapter = GenOpsArizeAdapter(
            team=os.getenv('GENOPS_TEAM', 'ml-platform'),
            project=os.getenv('GENOPS_PROJECT', 'fraud-detection'),
            environment=os.getenv('GENOPS_ENVIRONMENT', 'development'),
            daily_budget_limit=25.0,
            max_monitoring_cost=10.0,
            enable_cost_alerts=True,
            enable_governance=True
        )

        print("  ✅ Adapter initialized:")
        print(f"     • Team: {adapter.team}")
        print(f"     • Project: {adapter.project}")
        print(f"     • Environment: {adapter.environment}")
        print(f"     • Daily Budget: ${adapter.daily_budget_limit:.2f}")
        print()

        # Create sample data
        predictions_df = create_sample_prediction_data()

        # Demonstrate monitoring session with governance
        model_id = "fraud-detection-basic"
        model_version = "1.0"

        print(f"🎯 Starting monitoring session for {model_id}-{model_version}...")

        with adapter.track_model_monitoring_session(
            model_id=model_id,
            model_version=model_version,
            environment="development"
        ) as session:

            print(f"  ✅ Model monitoring session started: {session.session_name}")

            # Log prediction batch with cost tracking
            print("  📊 Logging prediction batch...")
            session.log_prediction_batch(
                predictions_df,
                cost_per_prediction=0.001
            )
            print(f"     • Logged {len(predictions_df)} predictions")
            print(f"     • Estimated cost: ${len(predictions_df) * 0.001:.2f}")

            # Log data quality metrics
            print("  🔍 Logging data quality metrics...")
            quality_metrics = {
                "missing_values_rate": predictions_df.isnull().sum().sum() / (len(predictions_df) * len(predictions_df.columns)),
                "duplicate_rate": predictions_df.duplicated().sum() / len(predictions_df),
                "fraud_rate": (predictions_df['prediction_label'] == 'fraud').mean(),
                "accuracy": (predictions_df['prediction_label'] == predictions_df['actual_label']).mean(),
                "average_fraud_score": predictions_df['fraud_score'].mean()
            }

            session.log_data_quality_metrics(
                quality_metrics,
                cost_estimate=0.05
            )

            print("     • Data Quality Metrics:")
            for metric, value in quality_metrics.items():
                if isinstance(value, float):
                    print(f"       - {metric}: {value:.3f}")
                else:
                    print(f"       - {metric}: {value}")

            # Create performance alert
            print("  🚨 Creating performance alert...")
            session.create_performance_alert(
                metric="accuracy",
                threshold=0.85,
                cost_per_alert=0.10
            )
            print("     • Alert created for accuracy threshold")

            # Update monitoring costs manually (simulate additional operations)
            additional_cost = 0.15
            session.update_monitoring_cost(additional_cost)
            print(f"     • Additional monitoring cost: ${additional_cost:.2f}")

        # Get session cost summary
        print("\n💰 Session Cost Summary:")
        session_cost = adapter.get_monitoring_cost_summary(session.session_id)

        if session_cost:
            print("  📊 Cost Breakdown:")
            print(f"     • Total Cost: ${session_cost.total_cost:.2f}")
            print(f"     • Prediction Logging: ${session_cost.prediction_logging_cost:.2f}")
            print(f"     • Data Quality: ${session_cost.data_quality_cost:.2f}")
            print(f"     • Alert Management: ${session_cost.alert_management_cost:.2f}")
            print(f"     • Dashboard Analytics: ${session_cost.dashboard_cost:.2f}")
            print(f"     • Duration: {session_cost.monitoring_duration:.1f} seconds")
            print(f"     • Efficiency: {session_cost.efficiency_score:.2f} predictions/hour")

        # Display adapter metrics
        print("\n📈 Adapter Metrics:")
        metrics = adapter.get_metrics()
        print(f"  • Daily Usage: ${metrics['daily_usage']:.2f}")
        print(f"  • Budget Remaining: ${metrics['budget_remaining']:.2f}")
        print(f"  • Operations Count: {metrics['operation_count']}")
        print(f"  • Active Sessions: {metrics['active_monitoring_sessions']}")
        print(f"  • Cost Alerts Enabled: {metrics['cost_alerts_enabled']}")

        return True

    except ImportError as e:
        print(f"❌ Required package not available: {e}")
        print("   Fix: pip install genops[arize]")
        return False
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
        return False


def demonstrate_manual_arize_integration():
    """Demonstrate manual integration with Arize SDK."""
    print("\n🔧 Demonstrating Manual Arize SDK Integration...")

    try:
        from arize.pandas.logger import Client

        from genops.providers.arize import GenOpsArizeAdapter

        # Create Arize client
        arize_client = Client(
            api_key=os.getenv('ARIZE_API_KEY'),
            space_key=os.getenv('ARIZE_SPACE_KEY')
        )

        # Create GenOps adapter for governance
        adapter = GenOpsArizeAdapter(
            team=os.getenv('GENOPS_TEAM', 'ml-platform'),
            project=os.getenv('GENOPS_PROJECT', 'manual-integration'),
            environment="development"
        )

        print("  ✅ Arize client and GenOps adapter initialized")

        # Create sample data for manual logging
        sample_predictions = [
            {
                "prediction_id": f"manual_pred_{i}",
                "prediction_label": "fraud" if i % 3 == 0 else "legitimate",
                "actual_label": "fraud" if i % 3 == 0 else "legitimate",
                "model_id": "fraud-model-manual",
                "model_version": "1.0",
                "features": {
                    "amount": 100.0 + i * 50,
                    "merchant": "online_store"
                }
            }
            for i in range(5)
        ]

        print(f"  📊 Prepared {len(sample_predictions)} sample predictions for manual logging")

        # Log each prediction individually (simulating real-time logging)
        with adapter.track_model_monitoring_session("manual-logging") as session:
            for pred in sample_predictions:
                # This would be your actual Arize logging call
                # response = arize_client.log(
                #     prediction_id=pred["prediction_id"],
                #     prediction_label=pred["prediction_label"],
                #     actual_label=pred["actual_label"],
                #     model_id=pred["model_id"],
                #     model_version=pred["model_version"],
                #     features=pred["features"]
                # )

                # Simulate cost tracking for manual operations
                session.update_monitoring_cost(0.001)  # Cost per prediction

                print(f"     • Logged prediction: {pred['prediction_id']} -> {pred['prediction_label']}")

        print("  ✅ Manual integration demonstration completed")
        return True

    except Exception as e:
        print(f"❌ Manual integration failed: {e}")
        return False


def print_usage_examples():
    """Print example usage patterns."""
    print("\n📖 Usage Examples:")
    print("  This example demonstrates several key patterns:")
    print()

    print("  🔧 1. Adapter Configuration:")
    print("     adapter = GenOpsArizeAdapter(")
    print("         team='your-team',")
    print("         project='your-project',")
    print("         daily_budget_limit=50.0,")
    print("         enable_cost_alerts=True")
    print("     )")
    print()

    print("  📊 2. Monitoring Session:")
    print("     with adapter.track_model_monitoring_session('model-id') as session:")
    print("         session.log_prediction_batch(df, cost_per_prediction=0.001)")
    print("         session.log_data_quality_metrics(metrics, cost_estimate=0.05)")
    print("         session.create_performance_alert('accuracy', 0.85, 0.10)")
    print()

    print("  💰 3. Cost Tracking:")
    print("     cost_summary = adapter.get_monitoring_cost_summary(session_id)")
    print("     metrics = adapter.get_metrics()")
    print()

    print("  📚 Next steps:")
    print("     • Try auto_instrumentation.py for zero-code integration")
    print("     • Try cost_optimization.py for cost intelligence features")
    print("     • Try production_patterns.py for advanced deployment patterns")


def main():
    """Main example workflow."""
    print_header()

    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please install dependencies and set environment variables.")
        return 1

    # Demonstrate basic monitoring
    monitoring_success = demonstrate_basic_monitoring()

    if monitoring_success:
        # Demonstrate manual integration
        manual_success = demonstrate_manual_arize_integration()

        print("\n" + "=" * 60)
        print("✅ Basic tracking example completed successfully!")
        print("=" * 60)

        # Print usage examples
        print_usage_examples()

        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ Basic tracking example failed!")
        print("=" * 60)
        print("   Check error messages above for troubleshooting guidance.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
