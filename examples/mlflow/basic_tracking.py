"""
Basic MLflow Tracking with GenOps Governance

This example demonstrates basic MLflow experiment tracking with GenOps
governance telemetry and cost tracking.

Features demonstrated:
- Experiment creation with governance attributes
- Parameter and metric logging with cost tracking
- Artifact logging with storage cost estimation
- Governance attribute propagation to MLflow tags
- Cost summary and reporting

Usage:
    # Set governance environment variables (optional)
    export GENOPS_TEAM="ml-team"
    export GENOPS_PROJECT="model-optimization"
    export GENOPS_ENVIRONMENT="development"

    # Run the example
    python examples/mlflow/basic_tracking.py

Expected output:
    - MLflow experiment created with governance tags
    - Run tracked with params, metrics, and artifacts
    - Cost summary showing total governance costs
    - Success confirmation with run details
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run basic MLflow tracking with GenOps governance."""
    print("\n" + "=" * 70)
    print("MLFLOW BASIC TRACKING WITH GENOPS GOVERNANCE")
    print("=" * 70)

    try:
        # Import MLflow
        import mlflow

        # Import GenOps MLflow adapter
        from genops.providers.mlflow import instrument_mlflow

        print("\n📊 Step 1: Initialize GenOps MLflow Adapter")
        print("-" * 70)

        # Use environment variable or tempfile for cross-platform compatibility
        tracking_uri = (
            os.getenv("MLFLOW_TRACKING_URI") or f"file://{tempfile.gettempdir()}/mlruns"
        )

        # Create adapter with governance attributes
        adapter = instrument_mlflow(
            tracking_uri=tracking_uri,  # Local tracking for demo
            team="ml-team",
            project="basic-tracking-demo",
            environment="development",
        )

        print("✅ Adapter initialized")
        print(f"   Team: {adapter.team}")
        print(f"   Project: {adapter.project}")
        print(f"   Tracking URI: {adapter.tracking_uri}")

        print("\n📊 Step 2: Track Experiment with Governance")
        print("-" * 70)

        # Use the adapter's context manager for governance tracking
        with adapter.track_mlflow_run(
            experiment_name="basic-tracking-demo",
            run_name="demo-run-001",
            customer_id="demo-customer",
            cost_center="ml-research",
        ) as run:
            print(f"✅ Started MLflow run: {run.info.run_id}")

            # Log parameters
            print("\n   Logging parameters...")
            mlflow.log_param("learning_rate", 0.01)
            mlflow.log_param("batch_size", 32)
            mlflow.log_param("epochs", 10)
            mlflow.log_param("optimizer", "adam")

            # Log metrics
            print("   Logging metrics...")
            mlflow.log_metric("train_accuracy", 0.92)
            mlflow.log_metric("val_accuracy", 0.89)
            mlflow.log_metric("train_loss", 0.25)
            mlflow.log_metric("val_loss", 0.31)

            # Log metric over time (simulating training epochs)
            print("   Simulating training epochs...")
            for epoch in range(1, 6):
                accuracy = 0.7 + (epoch * 0.04)
                loss = 0.5 - (epoch * 0.08)
                mlflow.log_metric("epoch_accuracy", accuracy, step=epoch)
                mlflow.log_metric("epoch_loss", loss, step=epoch)
                time.sleep(0.1)  # Simulate training time

            # Log artifact (create a simple text file)
            print("   Logging artifacts...")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("Model Summary\n")
                f.write("=" * 40 + "\n")
                f.write("Learning Rate: 0.01\n")
                f.write("Batch Size: 32\n")
                f.write("Final Accuracy: 0.92\n")
                artifact_path = f.name

            mlflow.log_artifact(artifact_path)
            os.unlink(artifact_path)  # Clean up temporary file

            print("✅ Completed logging operations")

        print("\n✅ Run completed successfully")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Experiment ID: {run.info.experiment_id}")

        print("\n📊 Step 3: Cost Summary")
        print("-" * 70)

        print("✅ Governance tracking enabled")
        print(f"   Total operations tracked: {adapter.operation_count}")
        print(f"   Daily usage: ${adapter.daily_usage:.6f}")

        print("\n📊 Step 4: View Your Results")
        print("-" * 70)
        print("Your MLflow run is now tracked with full governance telemetry!")
        print("\nTo view your experiment:")
        print("  1. Start MLflow UI:")
        print(f"     mlflow ui --backend-store-uri {tracking_uri}")
        print("  2. Open browser:")
        print("     http://localhost:5000")
        print("  3. Look for experiment: 'basic-tracking-demo'")
        print("\nGovernance tags visible in MLflow:")
        print("  • genops.team = ml-team")
        print("  • genops.project = basic-tracking-demo")
        print("  • genops.environment = development")
        print("  • genops.customer_id = demo-customer")
        print("  • genops.cost_center = ml-research")

        print("\n✅ SUCCESS!")
        print("=" * 70)
        print()

        return 0

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nPossible fixes:")
        print("  1. Install MLflow: pip install mlflow")
        print("  2. Install GenOps: pip install -e .")
        print()
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
