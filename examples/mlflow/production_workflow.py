"""MLflow Production Workflow Example with GenOps Governance.

Demonstrates:
- Production-ready deployment patterns
- Multi-environment tracking (dev/staging/prod)
- Customer-level cost attribution
- Budget monitoring and alerting
- Model lifecycle management
- Production best practices
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from genops.providers.mlflow import instrument_mlflow, create_mlflow_cost_context


class ProductionMLflowWorkflow:
    """Production MLflow workflow with comprehensive governance."""

    def __init__(self, environment='production'):
        """Initialize production workflow."""
        self.environment = environment
        self.tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:///tmp/mlruns')
        self.team = os.getenv('GENOPS_TEAM', 'ml-platform')
        self.project = os.getenv('GENOPS_PROJECT', 'production-models')

        # Initialize adapter
        self.adapter = instrument_mlflow(
            tracking_uri=self.tracking_uri,
            team=self.team,
            project=self.project,
            environment=environment
        )

        print(f"✓ Production workflow initialized")
        print(f"  Environment: {environment}")
        print(f"  Team: {self.team}")
        print(f"  Project: {self.project}")
        print()

    def validate_governance(self):
        """Validate governance configuration."""
        print("Validating governance configuration...")

        required_env_vars = ['GENOPS_TEAM', 'GENOPS_PROJECT']
        missing_vars = [var for var in required_env_vars
                       if not os.getenv(var)]

        if missing_vars:
            print(f"  ⚠️  Warning: Missing environment variables: {missing_vars}")
            print(f"  Using defaults, but production should set these explicitly")
        else:
            print(f"  ✓ All required environment variables set")

        print()

    def train_model_with_validation(self, customer_id=None):
        """Train model with production validation."""
        print(f"Training model for customer: {customer_id or 'default'}")

        # Generate dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        with self.adapter.track_mlflow_run(
            experiment_name=f"production-{self.environment}",
            run_name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            customer_id=customer_id
        ) as run:
            # Production tags
            mlflow.set_tag("environment", self.environment)
            mlflow.set_tag("deployment_ready", "false")
            mlflow.set_tag("validation_status", "in_progress")
            mlflow.set_tag("owner", self.team)
            if customer_id:
                mlflow.set_tag("customer_id", customer_id)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Comprehensive evaluation
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }

            # Production thresholds
            thresholds = {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.80,
                'f1_score': 0.80
            }

            # Validate against thresholds
            passed_validation = all(
                metrics[key] >= thresholds[key]
                for key in thresholds.keys()
            )

            # Log everything
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
                mlflow.log_metric(f"{key}_threshold", thresholds[key])

            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("validation_passed", passed_validation)

            # Log model if validation passed
            if passed_validation:
                mlflow.sklearn.log_model(model, "model")
                mlflow.set_tag("deployment_ready", "true")
                mlflow.set_tag("validation_status", "passed")
                print(f"  ✓ Model validated - Accuracy: {metrics['accuracy']:.4f}")
            else:
                mlflow.set_tag("validation_status", "failed")
                print(f"  ✗ Model failed validation")
                failed_metrics = [k for k in thresholds.keys()
                                 if metrics[k] < thresholds[k]]
                print(f"    Failed metrics: {failed_metrics}")

            print()
            return run.info.run_id, passed_validation

    def monitor_customer_usage(self, customer_ids):
        """Monitor per-customer usage and costs."""
        print("Monitoring customer usage...")
        print()

        customer_costs = {}

        for customer_id in customer_ids:
            print(f"  Customer: {customer_id}")

            # Simulate inference workload
            with self.adapter.track_mlflow_run(
                experiment_name=f"inference-{self.environment}",
                run_name=f"inference-{customer_id}",
                customer_id=customer_id
            ) as run:
                # Log inference metrics
                mlflow.log_metric("requests_count", 1000)
                mlflow.log_metric("avg_latency_ms", 45.2)
                mlflow.log_metric("p95_latency_ms", 120.5)
                mlflow.log_metric("error_rate", 0.001)

                # Log customer-specific tags
                mlflow.set_tag("workload_type", "inference")
                mlflow.set_tag("customer_id", customer_id)

            print(f"    ✓ Tracked 1000 inference requests")

        # Get cost summary
        metrics = self.adapter.get_metrics()
        print()
        print(f"  Total cost: ${metrics['daily_usage']:.6f}")
        print(f"  Operations: {metrics['operation_count']}")
        print()

    def deploy_with_governance(self, run_id, stage='Staging'):
        """Deploy model with governance tracking."""
        print(f"Deploying model to {stage}...")

        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=self.tracking_uri)

        # Get run details
        run = client.get_run(run_id)

        # Check validation status
        validation_status = run.data.tags.get("validation_status", "unknown")
        if validation_status != "passed":
            print(f"  ✗ Cannot deploy: validation status is '{validation_status}'")
            print(f"    Only models with 'passed' validation can be deployed")
            print()
            return False

        # Register model
        model_name = f"{self.project}-classifier"
        model_uri = f"runs:/{run_id}/model"

        print(f"  Registering model '{model_name}'...")
        mlflow.register_model(model_uri, model_name)

        # Get latest version
        latest_version = client.get_latest_versions(model_name)[0]

        # Transition to stage
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage=stage,
            archive_existing_versions=True
        )

        # Update metadata
        client.update_model_version(
            name=model_name,
            version=latest_version.version,
            description=f"Deployed to {stage} on {datetime.now().isoformat()}"
        )

        # Add governance tags
        client.set_model_version_tag(
            name=model_name,
            version=latest_version.version,
            key="deployed_by",
            value=self.team
        )

        client.set_model_version_tag(
            name=model_name,
            version=latest_version.version,
            key="environment",
            value=self.environment
        )

        print(f"  ✓ Model version {latest_version.version} deployed to {stage}")
        print()
        return True

    def generate_governance_report(self):
        """Generate comprehensive governance report."""
        print("=" * 70)
        print("Governance Report")
        print("=" * 70)
        print()

        metrics = self.adapter.get_metrics()

        print(f"Environment: {self.environment}")
        print(f"Team: {self.team}")
        print(f"Project: {self.project}")
        print()

        print(f"Cost Metrics:")
        print(f"  Daily Usage: ${metrics['daily_usage']:.6f}")
        print(f"  Operations: {metrics['operation_count']}")
        print()

        print(f"Compliance:")
        print(f"  ✓ All operations attributed to team '{self.team}'")
        print(f"  ✓ All operations attributed to project '{self.project}'")
        print(f"  ✓ Environment segregation: {self.environment}")
        print(f"  ✓ Customer-level tracking enabled")
        print(f"  ✓ Complete audit trail maintained")
        print()


def main():
    """Production workflow demonstration."""
    print("=" * 70)
    print("MLflow Production Workflow - GenOps Governance")
    print("=" * 70)
    print()

    # ========================================================================
    # Development Environment
    # ========================================================================
    print("Stage 1: Development Environment")
    print("-" * 70)

    dev_workflow = ProductionMLflowWorkflow(environment='development')
    dev_workflow.validate_governance()

    # Train and validate model
    run_id_dev, validated_dev = dev_workflow.train_model_with_validation()

    if validated_dev:
        print("✓ Development model ready for staging")
    print()

    # ========================================================================
    # Staging Environment
    # ========================================================================
    print("Stage 2: Staging Environment")
    print("-" * 70)

    staging_workflow = ProductionMLflowWorkflow(environment='staging')

    # Train and validate for staging
    run_id_staging, validated_staging = staging_workflow.train_model_with_validation()

    if validated_staging:
        # Deploy to staging
        deployed = staging_workflow.deploy_with_governance(
            run_id_staging,
            stage='Staging'
        )

        if deployed:
            print("✓ Model deployed to Staging")
    print()

    # ========================================================================
    # Production Environment
    # ========================================================================
    print("Stage 3: Production Environment")
    print("-" * 70)

    prod_workflow = ProductionMLflowWorkflow(environment='production')

    # Train production model
    run_id_prod, validated_prod = prod_workflow.train_model_with_validation()

    if validated_prod:
        # Deploy to production
        deployed = prod_workflow.deploy_with_governance(
            run_id_prod,
            stage='Production'
        )

        if deployed:
            print("✓ Model deployed to Production")
    print()

    # ========================================================================
    # Multi-Tenant Usage Monitoring
    # ========================================================================
    print("Stage 4: Multi-Tenant Usage Monitoring")
    print("-" * 70)

    customers = ['customer-001', 'customer-002', 'customer-003']
    prod_workflow.monitor_customer_usage(customers)

    # ========================================================================
    # Governance Reports
    # ========================================================================
    print("Stage 5: Governance Reporting")
    print("-" * 70)
    print()

    for env_name, workflow in [
        ('Development', dev_workflow),
        ('Staging', staging_workflow),
        ('Production', prod_workflow)
    ]:
        print(f"{env_name} Environment:")
        workflow.generate_governance_report()

    # ========================================================================
    # Best Practices Summary
    # ========================================================================
    print("=" * 70)
    print("Production Best Practices Demonstrated")
    print("=" * 70)
    print()

    print("1. Environment Segregation:")
    print("   ✓ Separate tracking for dev/staging/prod")
    print("   ✓ Environment-specific governance attributes")
    print()

    print("2. Validation Gates:")
    print("   ✓ Threshold-based model validation")
    print("   ✓ Deployment gating based on validation")
    print("   ✓ Clear validation status tracking")
    print()

    print("3. Model Lifecycle:")
    print("   ✓ Development → Staging → Production pipeline")
    print("   ✓ Model registry integration")
    print("   ✓ Stage transitions with governance")
    print()

    print("4. Multi-Tenant Tracking:")
    print("   ✓ Customer-level cost attribution")
    print("   ✓ Per-customer usage monitoring")
    print("   ✓ Isolated governance tracking")
    print()

    print("5. Compliance & Audit:")
    print("   ✓ Complete audit trail")
    print("   ✓ Team and project attribution")
    print("   ✓ Governance report generation")
    print("   ✓ OpenTelemetry integration")
    print()

    print("6. Cost Management:")
    print("   ✓ Real-time cost tracking")
    print("   ✓ Multi-level cost aggregation")
    print("   ✓ Customer-level cost allocation")
    print("   ✓ Budget monitoring ready")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError running example: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
