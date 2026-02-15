"""MLflow Model Registry Example with GenOps Governance.

Demonstrates:
- Model training with governance tracking
- Model logging and registration
- Model versioning and stage transitions
- Cost tracking for registry operations
- Production deployment patterns
"""

import os
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from genops.providers.mlflow import instrument_mlflow


def main():
    """Model registry workflow with governance."""
    print("=" * 70)
    print("MLflow Model Registry Example - GenOps Governance")
    print("=" * 70)
    print()

    # Configuration
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    team = os.getenv("GENOPS_TEAM", "ml-team")
    project = os.getenv("GENOPS_PROJECT", "model-registry-demo")

    print("Configuration:")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Team: {team}")
    print(f"  Project: {project}")
    print()

    # Create adapter
    adapter = instrument_mlflow(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,  # Use same URI for registry
        team=team,
        project=project,
        environment="development",
    )

    print("✓ MLflow adapter created")
    print()

    # Generate synthetic dataset
    print("Generating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print()

    # Model name for registry
    model_name = "demo-classifier"

    # ========================================================================
    # Train and Register Initial Model (Version 1)
    # ========================================================================
    print("Training initial model (Version 1)...")
    with adapter.track_mlflow_run(
        experiment_name="model-registry-demo", run_name="v1-training"
    ):
        # Train model
        model_v1 = RandomForestClassifier(n_estimators=50, random_state=42)
        model_v1.fit(X_train, y_train)

        # Evaluate
        y_pred = model_v1.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log parameters
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", None)
        mlflow.log_param("model_version", "1.0")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model_v1, "model", registered_model_name=model_name)

        print(f"  ✓ Model trained - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"  ✓ Model registered as '{model_name}' version 1")
        print()

    # ========================================================================
    # Train Improved Model (Version 2)
    # ========================================================================
    print("Training improved model (Version 2)...")
    with adapter.track_mlflow_run(
        experiment_name="model-registry-demo", run_name="v2-training"
    ):
        # Train improved model
        model_v2 = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        model_v2.fit(X_train, y_train)

        # Evaluate
        y_pred = model_v2.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("model_version", "2.0")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log and register model
        mlflow.sklearn.log_model(model_v2, "model", registered_model_name=model_name)

        print(f"  ✓ Model trained - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"  ✓ Model registered as '{model_name}' version 2")
        print()

    # ========================================================================
    # Model Version Management
    # ========================================================================
    print("Managing model versions...")

    # Get MLflow client
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)

    # List all versions
    print(f"\nModel versions for '{model_name}':")
    versions = client.search_model_versions(f"name='{model_name}'")
    for version in versions:
        print(f"  Version {version.version}: {version.current_stage}")

    # Transition version 2 to Staging
    print("\nTransitioning version 2 to 'Staging' stage...")
    client.transition_model_version_stage(name=model_name, version=2, stage="Staging")
    print("  ✓ Version 2 transitioned to Staging")

    # After validation, promote to Production
    print("\nTransitioning version 2 to 'Production' stage...")
    client.transition_model_version_stage(
        name=model_name,
        version=2,
        stage="Production",
        archive_existing_versions=True,  # Archive previous production versions
    )
    print("  ✓ Version 2 transitioned to Production")
    print("  ✓ Previous production versions archived")
    print()

    # ========================================================================
    # Load Model from Registry
    # ========================================================================
    print("Loading model from registry...")

    # Load latest production model
    model_uri = f"models:/{model_name}/Production"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    print(f"  ✓ Loaded model from: {model_uri}")

    # Test loaded model
    y_pred = loaded_model.predict(X_test[:5])
    print("  ✓ Model inference successful")
    print(f"  Sample predictions: {y_pred}")
    print()

    # ========================================================================
    # Model Metadata and Tags
    # ========================================================================
    print("Setting model metadata...")

    # Get latest version
    latest_version = client.get_latest_versions(model_name, stages=["Production"])[0]

    # Set description
    client.update_model_version(
        name=model_name,
        version=latest_version.version,
        description="Random Forest classifier for demo purposes. Trained on synthetic data.",
    )

    # Set tags for governance
    client.set_model_version_tag(
        name=model_name,
        version=latest_version.version,
        key="validation_status",
        value="approved",
    )

    client.set_model_version_tag(
        name=model_name, version=latest_version.version, key="deployed_by", value=team
    )

    print("  ✓ Model metadata updated")
    print()

    # ========================================================================
    # Cost Summary
    # ========================================================================
    print("=" * 70)
    print("Cost Summary")
    print("=" * 70)

    metrics = adapter.get_metrics()
    print("\nGovernance Metrics:")
    print(f"  Daily Usage: ${metrics['daily_usage']:.6f}")
    print(f"  Operations Tracked: {metrics['operation_count']}")
    print(f"  Runs: {metrics.get('run_count', 'N/A')}")
    print()

    print("Cost Breakdown:")
    print("  Model Training (2 runs):")
    print("    - Parameter logging: ~$0.0006 (6 params)")
    print("    - Metric logging: ~$0.0004 (4 metrics)")
    print("    - Model storage: Size-based (depends on storage backend)")
    print("  Model Registry Operations:")
    print("    - Model registration (2 versions): ~$0.0010")
    print("    - Stage transitions (2 ops): ~$0.0010")
    print("    - Metadata updates: ~$0.0002")
    print()

    # ========================================================================
    # Cleanup Instructions
    # ========================================================================
    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print()
    print("View your results:")
    print(f"  1. Start MLflow UI: mlflow ui --backend-store-uri {tracking_uri}")
    print("  2. Open browser: http://localhost:5000")
    print(f"  3. Navigate to Models tab to see '{model_name}'")
    print()
    print("Governance features enabled:")
    print(f"  ✓ All operations attributed to team '{team}'")
    print(f"  ✓ All operations attributed to project '{project}'")
    print("  ✓ Cost tracking for all registry operations")
    print("  ✓ OpenTelemetry traces exported")
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
