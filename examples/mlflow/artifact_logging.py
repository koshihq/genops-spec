"""MLflow Artifact Logging Example with GenOps Governance.

Demonstrates:
- Logging various artifact types
- Directory artifact management
- Cost tracking for artifact storage
- Storage backend configuration
- Artifact retrieval and management
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow

from genops.providers.mlflow import instrument_mlflow


def create_sample_artifacts(output_dir: Path):
    """Create sample artifacts for demonstration."""
    # 1. Text file
    summary_file = output_dir / "model_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Model Training Summary\n")
        f.write("=" * 40 + "\n")
        f.write("Model Type: Random Forest\n")
        f.write("Training Samples: 1000\n")
        f.write("Features: 20\n")
        f.write("Accuracy: 0.95\n")

    # 2. JSON configuration
    config_file = output_dir / "config.json"
    config = {
        "model": {"type": "RandomForest", "n_estimators": 100, "max_depth": 10},
        "training": {"batch_size": 32, "epochs": 50, "learning_rate": 0.01},
    }
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    # 3. Plot - Training curves
    plot_file = output_dir / "training_curves.png"
    epochs = np.arange(1, 51)
    train_loss = 2.0 * np.exp(-epochs / 10) + 0.1
    val_loss = 2.2 * np.exp(-epochs / 10) + 0.15

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Training Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_file, dpi=100, bbox_inches="tight")
    plt.close()

    # 4. CSV data
    data_file = output_dir / "predictions.csv"
    with open(data_file, "w") as f:
        f.write("sample_id,true_label,predicted_label,confidence\n")
        for i in range(20):
            true_label = np.random.randint(0, 2)
            pred_label = true_label if np.random.random() > 0.1 else 1 - true_label
            confidence = np.random.uniform(0.7, 0.99)
            f.write(f"{i},{true_label},{pred_label},{confidence:.4f}\n")

    # 5. Binary file (simulate model weights)
    weights_file = output_dir / "model_weights.npy"
    weights = np.random.randn(100, 50)
    np.save(weights_file, weights)

    return {
        "summary": summary_file,
        "config": config_file,
        "plot": plot_file,
        "data": data_file,
        "weights": weights_file,
    }


def main():
    """Artifact logging workflow with governance."""
    print("=" * 70)
    print("MLflow Artifact Logging Example - GenOps Governance")
    print("=" * 70)
    print()

    # Configuration
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    team = os.getenv("GENOPS_TEAM", "ml-team")
    project = os.getenv("GENOPS_PROJECT", "artifact-logging-demo")

    print("Configuration:")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Team: {team}")
    print(f"  Project: {project}")
    print()

    # Create adapter
    adapter = instrument_mlflow(
        tracking_uri=tracking_uri, team=team, project=project, environment="development"
    )

    print("✓ MLflow adapter created")
    print()

    # ========================================================================
    # Example 1: Log Individual Artifacts
    # ========================================================================
    print("Example 1: Logging Individual Artifacts")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        artifacts = create_sample_artifacts(output_dir)

        with adapter.track_mlflow_run(
            experiment_name="artifact-logging-demo", run_name="individual-artifacts"
        ) as run:
            # Log each artifact individually
            print("Logging artifacts:")

            print("  1. Text summary...")
            mlflow.log_artifact(str(artifacts["summary"]), artifact_path="reports")
            file_size = artifacts["summary"].stat().st_size / 1024  # KB
            print(f"     ✓ Logged (size: {file_size:.2f} KB)")

            print("  2. JSON configuration...")
            mlflow.log_artifact(str(artifacts["config"]), artifact_path="configs")
            file_size = artifacts["config"].stat().st_size / 1024
            print(f"     ✓ Logged (size: {file_size:.2f} KB)")

            print("  3. Training plot...")
            mlflow.log_artifact(str(artifacts["plot"]), artifact_path="plots")
            file_size = artifacts["plot"].stat().st_size / 1024
            print(f"     ✓ Logged (size: {file_size:.2f} KB)")

            print("  4. Predictions CSV...")
            mlflow.log_artifact(str(artifacts["data"]), artifact_path="data")
            file_size = artifacts["data"].stat().st_size / 1024
            print(f"     ✓ Logged (size: {file_size:.2f} KB)")

            print("  5. Model weights...")
            mlflow.log_artifact(str(artifacts["weights"]), artifact_path="weights")
            file_size = artifacts["weights"].stat().st_size / 1024
            print(f"     ✓ Logged (size: {file_size:.2f} KB)")

            print()
            print(f"✓ Run ID: {run.info.run_id}")
            print()

    # ========================================================================
    # Example 2: Log Entire Directory
    # ========================================================================
    print("Example 2: Logging Entire Directory")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create directory structure
        reports_dir = output_dir / "reports"
        reports_dir.mkdir()

        # Create multiple files
        for i in range(5):
            report_file = reports_dir / f"report_{i}.txt"
            with open(report_file, "w") as f:
                f.write(f"Report {i}\n")
                f.write(f"Timestamp: 2024-01-{i + 1:02d}\n")
                f.write("Status: Complete\n")

        with adapter.track_mlflow_run(
            experiment_name="artifact-logging-demo", run_name="directory-artifacts"
        ) as run:
            print(f"Logging directory: {reports_dir}")
            mlflow.log_artifacts(str(reports_dir), artifact_path="reports")

            # Calculate total size
            total_size = sum(f.stat().st_size for f in reports_dir.glob("*"))
            print(f"  ✓ Logged {len(list(reports_dir.glob('*')))} files")
            print(f"  ✓ Total size: {total_size / 1024:.2f} KB")
            print()

    # ========================================================================
    # Example 3: Log Dictionary as JSON
    # ========================================================================
    print("Example 3: Logging Dictionary as JSON")
    print("-" * 70)

    with adapter.track_mlflow_run(
        experiment_name="artifact-logging-demo", run_name="dict-artifacts"
    ) as run:
        # Create metrics dictionary
        metrics_dict = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.94,
            "f1_score": 0.935,
            "confusion_matrix": [[450, 50], [30, 470]],
        }

        print("Logging metrics dictionary as JSON...")
        mlflow.log_dict(metrics_dict, "metrics.json")
        print("  ✓ Logged metrics.json")
        print()

    # ========================================================================
    # Example 4: Log Large Artifacts with Cost Tracking
    # ========================================================================
    print("Example 4: Large Artifacts with Cost Tracking")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create large file (10 MB)
        large_file = output_dir / "large_dataset.npy"
        large_array = np.random.randn(1000, 1000)  # ~8 MB
        np.save(large_file, large_array)

        file_size_mb = large_file.stat().st_size / (1024 * 1024)

        with adapter.track_mlflow_run(
            experiment_name="artifact-logging-demo", run_name="large-artifacts"
        ) as run:
            print(f"Logging large file: {file_size_mb:.2f} MB")
            mlflow.log_artifact(str(large_file), artifact_path="datasets")
            print("  ✓ Logged successfully")

            # Estimate storage cost (S3 pricing)
            # S3: $0.023 per GB-month, prorated daily
            gb_size = file_size_mb / 1024
            daily_cost = gb_size * 0.023 / 30
            print(f"\n  Estimated daily storage cost (S3): ${daily_cost:.6f}")
            print()

    # ========================================================================
    # Example 5: Artifact Retrieval
    # ========================================================================
    print("Example 5: Artifact Retrieval")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        artifacts = create_sample_artifacts(output_dir)

        # Log artifacts
        with adapter.track_mlflow_run(
            experiment_name="artifact-logging-demo", run_name="artifact-retrieval"
        ) as run:
            mlflow.log_artifact(str(artifacts["config"]), artifact_path="configs")
            run_id = run.info.run_id

        print(f"Logged artifacts in run: {run_id}")
        print()

        # Retrieve artifacts
        print("Retrieving artifacts...")
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

        # List artifacts
        artifacts_list = client.list_artifacts(run_id, path="configs")
        print(f"  Found {len(artifacts_list)} artifacts:")
        for artifact in artifacts_list:
            print(f"    - {artifact.path} ({artifact.file_size} bytes)")

        # Download artifact
        print("\n  Downloading config.json...")
        artifact_path = client.download_artifacts(run_id, "configs/config.json")
        print(f"  ✓ Downloaded to: {artifact_path}")

        # Read and display
        with open(artifact_path) as f:
            config = json.load(f)
        print(f"  ✓ Config loaded: {json.dumps(config, indent=2)}")
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

    print("Cost Breakdown by Storage Backend:")
    print("  Local storage: Free")
    print("  S3: ~$0.023 per GB-month (~$0.00077/GB/day)")
    print("  Azure Blob: ~$0.020 per GB-month (~$0.00067/GB/day)")
    print("  GCS: ~$0.020 per GB-month (~$0.00067/GB/day)")
    print()

    print("Artifact Logging Costs:")
    print("  Small files (<1 MB): ~$0.0001 per operation")
    print("  Large files (10-100 MB): Storage cost dominates")
    print("  Directory logging: Cost per file + storage")
    print()

    # ========================================================================
    # Cleanup Instructions
    # ========================================================================
    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print()
    print("View your artifacts:")
    print(f"  1. Start MLflow UI: mlflow ui --backend-store-uri {tracking_uri}")
    print("  2. Open browser: http://localhost:5000")
    print("  3. Navigate to experiment 'artifact-logging-demo'")
    print("  4. Click on any run to view artifacts")
    print()
    print("Governance features enabled:")
    print(f"  ✓ All artifacts attributed to team '{team}'")
    print(f"  ✓ All artifacts attributed to project '{project}'")
    print("  ✓ Cost tracking for all artifact operations")
    print("  ✓ Storage backend detection and cost estimation")
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
